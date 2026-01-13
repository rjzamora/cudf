# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Dynamic filter pushdown optimization.

This module implements optimizations that detect semi-join and selective
inner join patterns, using filter predicates to prefilter data before
shuffles to reduce data movement.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import singledispatch
from typing import TYPE_CHECKING, Literal

from cudf_polars.dsl.ir import (
    Cache,
    DataFrameScan,
    Distinct,
    Filter,
    GroupBy,
    Join,
    Scan,
    Select,
    Slice,
    Sort,
    Union,
)
from cudf_polars.dsl.traversal import post_traversal, traversal
from cudf_polars.experimental.base import ColumnStat
from cudf_polars.experimental.statistics import collect_base_stats

if TYPE_CHECKING:
    from collections.abc import Iterator, MutableMapping

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import StatsCollector
    from cudf_polars.utils.config import ConfigOptions


# Default selectivity for filter operations (0.0 = filters everything, 1.0 = keeps all)
DEFAULT_FILTER_SELECTIVITY = 0.3

# Default selectivity for GroupBy/Distinct operations
DEFAULT_GROUPBY_SELECTIVITY = 0.1

# Threshold for considering a branch "selective" (output/input ratio)
DEFAULT_SELECTIVITY_THRESHOLD = 0.5


@dataclass
class FilterSource:
    """
    Information about a filter source from a join pattern.

    This dataclass captures the information needed to potentially
    push down the filter to earlier operations.
    """

    join_node: Join
    """The join node that produces the filter."""

    source_type: Literal["semi_join", "selective_inner_join"]
    """Type of filter source pattern detected."""

    filter_keys_provider: IR
    """The side that provides filter keys (smaller/selective side)."""

    filter_target: IR
    """The side that can be prefiltered (larger side)."""

    provider_on: tuple[NamedExpr, ...]
    """Key columns on the filter keys provider side."""

    target_on: tuple[NamedExpr, ...]
    """Key columns on the filter target side."""

    selectivity_ratio: float | None = None
    """Estimated selectivity ratio of the provider side (output/source rows)."""

    @property
    def provider_key_names(self) -> tuple[str, ...]:
        """Get the column names for the provider join keys."""
        return tuple(expr.name for expr in self.provider_on)

    @property
    def target_key_names(self) -> tuple[str, ...]:
        """Get the column names for the target join keys."""
        return tuple(expr.name for expr in self.target_on)


@dataclass
class FilterSourceCollection:
    """Collection of filter sources detected in an IR graph."""

    sources: list[FilterSource] = field(default_factory=list)
    """List of detected filter sources."""

    # Mapping from IR nodes to their filter sources for quick lookup
    _join_to_source: dict[Join, FilterSource] = field(default_factory=dict, repr=False)

    def add(self, source: FilterSource) -> None:
        """Add a filter source to the collection."""
        self.sources.append(source)
        self._join_to_source[source.join_node] = source

    def get_by_join(self, join_node: Join) -> FilterSource | None:
        """Get the filter source associated with a join node."""
        return self._join_to_source.get(join_node)

    def __len__(self) -> int:
        """Return the number of filter sources."""
        return len(self.sources)

    def __iter__(self) -> Iterator[FilterSource]:
        """Iterate over filter sources."""
        return iter(self.sources)

    def semi_joins(self) -> list[FilterSource]:
        """Return filter sources from semi-joins."""
        return [s for s in self.sources if s.source_type == "semi_join"]

    def selective_inner_joins(self) -> list[FilterSource]:
        """Return filter sources from selective inner joins."""
        return [s for s in self.sources if s.source_type == "selective_inner_join"]


@dataclass
class FilterTarget:
    """
    A target that can be prefiltered using a filter source.

    This represents an opportunity to apply a filter to reduce data
    movement before a shuffle or join operation.
    """

    source: FilterSource
    """The filter source that provides the filter keys."""

    target_scan: IR
    """The Scan/Cache node that can be prefiltered."""

    downstream_join: Join
    """The join where the filtered data would be used."""

    target_column: str
    """The column on the target scan that should be filtered."""

    source_column: str
    """The column from the filter source that provides filter keys."""


@dataclass
class FilterTargetCollection:
    """Collection of filter targets detected in an IR graph."""

    targets: list[FilterTarget] = field(default_factory=list)
    """List of detected filter targets."""

    # Mapping from target scans to their filter targets for quick lookup
    _scan_to_targets: dict[IR, list[FilterTarget]] = field(
        default_factory=lambda: defaultdict(list), repr=False
    )

    def add(self, target: FilterTarget) -> None:
        """Add a filter target to the collection."""
        self.targets.append(target)
        self._scan_to_targets[target.target_scan].append(target)

    def get_by_scan(self, scan_node: IR) -> list[FilterTarget]:
        """Get the filter targets associated with a scan node."""
        return self._scan_to_targets.get(scan_node, [])

    def __len__(self) -> int:
        """Return the number of filter targets."""
        return len(self.targets)

    def __iter__(self) -> Iterator[FilterTarget]:
        """Iterate over filter targets."""
        return iter(self.targets)


# -----------------------------------------------------------------------------
# Lightweight Row Count Propagation
# -----------------------------------------------------------------------------


@singledispatch
def propagate_row_count(
    ir: IR,
    stats: StatsCollector,
    config_options: ConfigOptions,
) -> None:
    """
    Propagate row count estimates through the IR graph.

    This is a lightweight version of update_column_stats that only
    propagates row counts without expensive unique-value sampling.

    Parameters
    ----------
    ir
        The IR node to propagate row counts for.
    stats
        The statistics collector to update.
    config_options
        GPUEngine configuration options.
    """
    # Default: inherit max row count from children
    child_counts = [
        stats.row_count.get(c, ColumnStat[int](None)).value for c in ir.children
    ]
    known_counts = [c for c in child_counts if c is not None]
    stats.row_count[ir] = ColumnStat[int](max(known_counts) if known_counts else None)


@propagate_row_count.register(Scan)
def _(ir: Scan, stats: StatsCollector, config_options: ConfigOptions) -> None:
    """Propagate row count for Scan nodes using parquet metadata."""
    if stats.column_stats.get(ir):
        stats.row_count[ir] = next(
            iter(stats.column_stats[ir].values())
        ).source_info.row_count
    else:
        stats.row_count[ir] = ColumnStat[int](None)

    # Account for the n_rows argument
    if ir.n_rows != -1:
        if (metadata_value := stats.row_count[ir].value) is not None:
            stats.row_count[ir] = ColumnStat[int](min(metadata_value, ir.n_rows))
        else:
            stats.row_count[ir] = ColumnStat[int](ir.n_rows)

    # Apply selectivity for scan predicates
    if (
        ir.predicate is not None
        and ir.n_rows == -1
        and (row_count := stats.row_count[ir].value) is not None
    ):
        stats.row_count[ir] = ColumnStat[int](
            max(1, int(row_count * DEFAULT_FILTER_SELECTIVITY))
        )


@propagate_row_count.register(DataFrameScan)
def _(ir: DataFrameScan, stats: StatsCollector, config_options: ConfigOptions) -> None:
    """Propagate row count for DataFrameScan nodes."""
    if stats.column_stats.get(ir):
        stats.row_count[ir] = next(
            iter(stats.column_stats[ir].values())
        ).source_info.row_count
    else:
        stats.row_count[ir] = ColumnStat[int](None)


@propagate_row_count.register(Filter)
def _(ir: Filter, stats: StatsCollector, config_options: ConfigOptions) -> None:
    """Propagate row count for Filter nodes with selectivity estimate."""
    (child,) = ir.children
    child_count = stats.row_count.get(child, ColumnStat[int](None)).value
    if child_count is not None:
        stats.row_count[ir] = ColumnStat[int](
            max(1, int(child_count * DEFAULT_FILTER_SELECTIVITY))
        )
    else:
        stats.row_count[ir] = ColumnStat[int](None)


@propagate_row_count.register(GroupBy)
def _(ir: GroupBy, stats: StatsCollector, config_options: ConfigOptions) -> None:
    """Propagate row count for GroupBy nodes."""
    (child,) = ir.children
    child_count = stats.row_count.get(child, ColumnStat[int](None)).value
    if child_count is not None:
        # GroupBy typically reduces to unique key combinations
        # Use a conservative estimate
        stats.row_count[ir] = ColumnStat[int](
            max(1, int(child_count * DEFAULT_GROUPBY_SELECTIVITY))
        )
    else:
        stats.row_count[ir] = ColumnStat[int](None)


@propagate_row_count.register(Distinct)
def _(ir: Distinct, stats: StatsCollector, config_options: ConfigOptions) -> None:
    """Propagate row count for Distinct nodes."""
    (child,) = ir.children
    child_count = stats.row_count.get(child, ColumnStat[int](None)).value
    if child_count is not None:
        # Distinct typically reduces significantly
        stats.row_count[ir] = ColumnStat[int](
            max(1, int(child_count * DEFAULT_GROUPBY_SELECTIVITY))
        )
    else:
        stats.row_count[ir] = ColumnStat[int](None)


@propagate_row_count.register(Select)
def _(ir: Select, stats: StatsCollector, config_options: ConfigOptions) -> None:
    """Propagate row count for Select nodes (usually preserves row count)."""
    (child,) = ir.children
    stats.row_count[ir] = stats.row_count.get(child, ColumnStat[int](None))


@propagate_row_count.register(Slice)
def _(ir: Slice, stats: StatsCollector, config_options: ConfigOptions) -> None:
    """Propagate row count for Slice nodes."""
    (child,) = ir.children
    child_count = stats.row_count.get(child, ColumnStat[int](None)).value
    if child_count is not None and ir.length is not None:
        # Slice limits the output
        stats.row_count[ir] = ColumnStat[int](min(child_count, ir.length))
    else:
        stats.row_count[ir] = stats.row_count.get(child, ColumnStat[int](None))


@propagate_row_count.register(Sort)
def _(ir: Sort, stats: StatsCollector, config_options: ConfigOptions) -> None:
    """Propagate row count for Sort nodes (may have a slice)."""
    (child,) = ir.children
    child_count = stats.row_count.get(child, ColumnStat[int](None)).value
    if child_count is not None and ir.zlice is not None:
        offset, length = ir.zlice
        if length is not None:
            stats.row_count[ir] = ColumnStat[int](min(child_count, length))
        else:
            stats.row_count[ir] = ColumnStat[int](child_count)
    else:
        stats.row_count[ir] = stats.row_count.get(child, ColumnStat[int](None))


@propagate_row_count.register(Cache)
def _(ir: Cache, stats: StatsCollector, config_options: ConfigOptions) -> None:
    """Propagate row count through Cache nodes."""
    (child,) = ir.children
    stats.row_count[ir] = stats.row_count.get(child, ColumnStat[int](None))


@propagate_row_count.register(Union)
def _(ir: Union, stats: StatsCollector, config_options: ConfigOptions) -> None:
    """Propagate row count for Union nodes (sum of children)."""
    child_counts = [
        stats.row_count.get(c, ColumnStat[int](None)).value for c in ir.children
    ]
    known_counts = [c for c in child_counts if c is not None]
    if known_counts:
        stats.row_count[ir] = ColumnStat[int](sum(known_counts))
    else:
        stats.row_count[ir] = ColumnStat[int](None)


@propagate_row_count.register(Join)
def _(ir: Join, stats: StatsCollector, config_options: ConfigOptions) -> None:
    """Propagate row count for Join nodes."""
    left, right = ir.children
    left_count = stats.row_count.get(left, ColumnStat[int](None)).value
    right_count = stats.row_count.get(right, ColumnStat[int](None)).value
    how = ir.options[0]

    if how == "Semi" or how == "Anti":
        # Semi/Anti join returns subset of left table
        stats.row_count[ir] = ColumnStat[int](
            max(1, int(left_count * DEFAULT_FILTER_SELECTIVITY))
            if left_count is not None
            else None
        )
    elif how == "Cross":
        # Cross join multiplies row counts
        if left_count is not None and right_count is not None:
            stats.row_count[ir] = ColumnStat[int](left_count * right_count)
        else:
            stats.row_count[ir] = ColumnStat[int](None)
    else:
        # Inner, Left, Right, Full joins - use max of inputs as conservative estimate
        if left_count is not None and right_count is not None:
            stats.row_count[ir] = ColumnStat[int](max(left_count, right_count))
        elif left_count is not None:
            stats.row_count[ir] = ColumnStat[int](left_count)
        elif right_count is not None:
            stats.row_count[ir] = ColumnStat[int](right_count)
        else:
            stats.row_count[ir] = ColumnStat[int](None)


# -----------------------------------------------------------------------------
# Statistics Collection
# -----------------------------------------------------------------------------


def collect_selectivity_stats(
    ir: IR,
    config_options: ConfigOptions,
) -> StatsCollector:
    """
    Collect lightweight statistics for selectivity estimation.

    This is faster than full collect_statistics() because it:
    - Does NOT sample unique values
    - Only propagates row counts and applies selectivity heuristics

    Parameters
    ----------
    ir
        Root of the IR graph.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    A StatsCollector with row count estimates for each node.
    """
    # Start with base stats (parquet metadata, source row counts)
    stats = collect_base_stats(ir, config_options)

    # Propagate row counts through the graph (post-order traversal)
    for node in post_traversal([ir]):
        propagate_row_count(node, stats, config_options)

    return stats


# -----------------------------------------------------------------------------
# Selectivity Analysis
# -----------------------------------------------------------------------------


def get_source_row_count(ir: IR, stats: StatsCollector) -> int | None:
    """
    Get the maximum raw source row count in an IR subtree.

    This returns the original row count from parquet/source metadata,
    before any predicates or filters are applied.

    Parameters
    ----------
    ir
        Root of the IR subtree.
    stats
        Statistics collector with row counts.

    Returns
    -------
    Maximum raw source row count, or None if unknown.
    """
    source_counts = []
    for node in traversal([ir]):
        if isinstance(node, (Scan, DataFrameScan)):
            # Get the raw metadata row count from column_stats.source_info
            # This is the count BEFORE any selectivity is applied
            column_stats = stats.column_stats.get(node, {})
            if column_stats:
                # Get row count from any column's source_info
                for col_stats in column_stats.values():
                    raw_count = col_stats.source_info.row_count.value
                    if raw_count is not None:
                        source_counts.append(raw_count)
                        break  # Only need one column's source row count
    return max(source_counts) if source_counts else None


def compute_selectivity_ratio(ir: IR, stats: StatsCollector) -> float | None:
    """
    Compute the selectivity ratio for an IR subtree.

    Parameters
    ----------
    ir
        Root of the IR subtree.
    stats
        Statistics collector with row counts.

    Returns
    -------
    Selectivity ratio (output_rows / source_rows), or None if unknown.
    Values closer to 0.0 indicate more selective (fewer output rows).
    """
    output_count = stats.row_count.get(ir, ColumnStat[int](None)).value
    source_count = get_source_row_count(ir, stats)

    if output_count is None or source_count is None or source_count == 0:
        return None

    return output_count / source_count


def is_selective(
    ir: IR,
    stats: StatsCollector,
    threshold: float = DEFAULT_SELECTIVITY_THRESHOLD,
) -> bool:
    """
    Check if an IR subtree is selective based on row count estimates.

    Parameters
    ----------
    ir
        Root of the IR subtree.
    stats
        Statistics collector with row counts.
    threshold
        Selectivity threshold (default 0.5). A subtree is considered
        selective if output/source ratio is below this threshold.

    Returns
    -------
    True if the subtree is selective, False otherwise.
    """
    ratio = compute_selectivity_ratio(ir, stats)
    if ratio is None:
        return False
    return ratio < threshold


# -----------------------------------------------------------------------------
# Filter Source Collection
# -----------------------------------------------------------------------------


def collect_filter_sources(
    ir: IR,
    stats: StatsCollector | None = None,
    selectivity_threshold: float = DEFAULT_SELECTIVITY_THRESHOLD,
) -> FilterSourceCollection:
    """
    Detect join patterns that can be used for dynamic filter pushdown.

    This function traverses the IR graph and identifies:
    1. Semi-joins: Explicit filter operations
    2. Selective inner joins: When one side is significantly smaller

    Parameters
    ----------
    ir
        Root of the IR graph to analyze.
    stats
        Optional statistics collector with row count estimates.
        If provided, enables detection of selective inner joins.
    selectivity_threshold
        Threshold for considering a branch selective (default 0.5).

    Returns
    -------
    A collection of FilterSource objects representing the detected patterns.

    Notes
    -----
    Semi-joins are always detected. Selective inner joins are only
    detected when statistics are provided.

    Examples
    --------
    Pattern 1 - Semi-join (Q18):
    ```python
    orders.join(q1, left_on="o_orderkey", right_on="l_orderkey", how="semi")
    ```

    Pattern 2 - Selective inner join (Q21):
    ```python
    # left side is highly filtered (nation = "SAUDI ARABIA")
    selective_left.join(large_orders, left_on="l_orderkey", right_on="o_orderkey")
    ```
    """
    collection = FilterSourceCollection()

    for node in traversal([ir]):
        # Handle Cache nodes by looking at their child
        actual_node = node
        while isinstance(actual_node, Cache):
            actual_node = actual_node.children[0]

        if isinstance(actual_node, Join):
            how = actual_node.options[0]
            left, right = actual_node.children

            if how == "Semi":
                # Pattern 1: Semi-join
                # Left side is filtered, right side provides filter keys
                source = FilterSource(
                    join_node=actual_node,
                    source_type="semi_join",
                    filter_keys_provider=right,
                    filter_target=left,
                    provider_on=actual_node.right_on,
                    target_on=actual_node.left_on,
                )
                collection.add(source)

            elif how == "Inner" and stats is not None:
                # Pattern 2: Selective inner join
                # Check if one side is significantly more selective
                left_ratio = compute_selectivity_ratio(left, stats)
                right_ratio = compute_selectivity_ratio(right, stats)

                # Get row counts for comparison
                left_count = stats.row_count.get(left, ColumnStat[int](None)).value
                right_count = stats.row_count.get(right, ColumnStat[int](None)).value

                detected = False

                # First, check selectivity ratios (output/source reduction)
                if left_ratio is not None and right_ratio is not None:
                    if (
                        left_ratio < selectivity_threshold
                        and left_ratio < right_ratio * 0.5
                    ):
                        # Left side is selective - can prefilter right
                        source = FilterSource(
                            join_node=actual_node,
                            source_type="selective_inner_join",
                            filter_keys_provider=left,
                            filter_target=right,
                            provider_on=actual_node.left_on,
                            target_on=actual_node.right_on,
                            selectivity_ratio=left_ratio,
                        )
                        collection.add(source)
                        detected = True
                    elif (
                        right_ratio < selectivity_threshold
                        and right_ratio < left_ratio * 0.5
                    ):
                        # Right side is selective - can prefilter left
                        source = FilterSource(
                            join_node=actual_node,
                            source_type="selective_inner_join",
                            filter_keys_provider=right,
                            filter_target=left,
                            provider_on=actual_node.right_on,
                            target_on=actual_node.left_on,
                            selectivity_ratio=right_ratio,
                        )
                        collection.add(source)
                        detected = True

                # Also check absolute row counts if not already detected
                # This catches cases where one side is simply much smaller
                if (
                    not detected
                    and left_count is not None
                    and right_count is not None
                    and max(left_count, right_count) > 0
                ):
                    count_ratio = min(left_count, right_count) / max(
                        left_count, right_count
                    )
                    if count_ratio < selectivity_threshold:
                        if left_count < right_count:
                            source = FilterSource(
                                join_node=actual_node,
                                source_type="selective_inner_join",
                                filter_keys_provider=left,
                                filter_target=right,
                                provider_on=actual_node.left_on,
                                target_on=actual_node.right_on,
                                selectivity_ratio=count_ratio,
                            )
                            collection.add(source)
                        else:
                            source = FilterSource(
                                join_node=actual_node,
                                source_type="selective_inner_join",
                                filter_keys_provider=right,
                                filter_target=left,
                                provider_on=actual_node.right_on,
                                target_on=actual_node.left_on,
                                selectivity_ratio=count_ratio,
                            )
                            collection.add(source)

    return collection


def _build_parent_map(ir: IR) -> dict[IR, list[tuple[IR, int]]]:
    """
    Build a map from each node to its parent nodes.

    Returns a dict where each key is a node and the value is a list of
    (parent_node, child_index) tuples indicating which parents use this
    node and at which child position.
    """
    parent_map: dict[IR, list[tuple[IR, int]]] = defaultdict(list)
    for node in traversal([ir]):
        for i, child in enumerate(node.children):
            parent_map[child].append((node, i))
    return parent_map


def _find_scans_in_subtree(node: IR) -> list[tuple[IR, str | None]]:
    """
    Find all Scan/DataFrameScan/Cache nodes in a subtree.

    Returns a list of (scan_node, None) tuples.
    The second element is reserved for future use (e.g., column tracking).
    """
    scans: list[tuple[IR, str | None]] = []
    for n in traversal([node]):
        # Skip through Cache nodes to find the actual Scan
        actual = n
        while isinstance(actual, Cache) and actual.children:
            actual = actual.children[0]
        if isinstance(actual, (Scan, DataFrameScan)):
            # Use the Cache node if present, otherwise the Scan
            scans.append((n if isinstance(n, Cache) else actual, None))
    return scans


def find_filter_targets(
    ir: IR,
    filter_sources: FilterSourceCollection,
) -> FilterTargetCollection:
    """
    Find filter targets for the detected filter sources.

    For each semi-join filter source, this function traces downstream
    to find joins that use the filtered keys, and identifies Scan nodes
    that could be prefiltered.

    Parameters
    ----------
    ir
        Root of the IR graph.
    filter_sources
        Collection of detected filter sources.

    Returns
    -------
    A collection of FilterTarget objects.

    Notes
    -----
    For Q18 pattern:
    - Semi-join: orders.join(q1, left_on="o_orderkey", right_on="l_orderkey", how="semi")
    - Downstream: semi_result.join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
    - Target: lineitem Scan can be prefiltered on l_orderkey using filtered o_orderkey values

    This function traces from the semi-join result to find such opportunities.
    """
    collection = FilterTargetCollection()

    # Build parent map to trace from children to parents
    parent_map = _build_parent_map(ir)

    for source in filter_sources.semi_joins():
        # The semi-join result flows to parent nodes
        # Find parents of the semi-join node
        semi_join = source.join_node
        parents = parent_map.get(semi_join, [])

        for parent, child_idx in parents:
            # Skip through non-join nodes to find downstream joins
            # (the semi-join result might flow through Select, Cache, etc.)
            current: IR | None = parent
            current_child_idx = child_idx

            while current is not None and not isinstance(current, Join):
                # Trace through the parent map
                next_parents = parent_map.get(current, [])
                if next_parents:
                    current, current_child_idx = next_parents[0]
                else:
                    current = None

            if current is None or not isinstance(current, Join):
                continue

            downstream_join = current
            if downstream_join.options[0] not in ("Inner", "Left", "Right"):
                continue

            # Determine which side of the downstream join has the semi-join result
            # and which side is the potential filter target
            left_child, right_child = downstream_join.children

            # Check if the semi-join result flows into the left or right side
            semi_is_left = _is_ancestor_of(semi_join, left_child)
            semi_is_right = _is_ancestor_of(semi_join, right_child)

            if semi_is_left and not semi_is_right:
                # Semi-join result is on the left, right side can be prefiltered
                target_side = right_child
                target_on = downstream_join.right_on
                # The filter key column on the semi-join side
                filter_on = downstream_join.left_on
            elif semi_is_right and not semi_is_left:
                # Semi-join result is on the right, left side can be prefiltered
                target_side = left_child
                target_on = downstream_join.left_on
                filter_on = downstream_join.right_on
            else:
                # Both or neither - can't determine, skip
                continue

            # Check if the join key matches the semi-join's filtered key
            # For Q18: semi-join filters on o_orderkey, downstream join uses o_orderkey
            semi_key_names = source.target_key_names  # e.g., ("o_orderkey",)
            filter_key_names = tuple(expr.name for expr in filter_on)

            # Find matching columns
            for i, filter_key in enumerate(filter_key_names):
                if filter_key in semi_key_names:
                    target_col = target_on[i].name

                    # Find Scan nodes in the target side
                    scans = _find_scans_in_subtree(target_side)
                    for scan_node, _ in scans:
                        target = FilterTarget(
                            source=source,
                            target_scan=scan_node,
                            downstream_join=downstream_join,
                            target_column=target_col,
                            source_column=filter_key,
                        )
                        collection.add(target)

    return collection


def _is_ancestor_of(ancestor: IR, node: IR) -> bool:
    """Check if `ancestor` is an ancestor of `node` (or is `node` itself)."""
    return any(n is ancestor for n in traversal([node]))


# -----------------------------------------------------------------------------
# IR Rewriting for Filter Insertion
# -----------------------------------------------------------------------------


def _create_prefilter_semi_join(
    target: FilterTarget,
) -> Join:
    """
    Create a semi-join node to prefilter a target scan.

    Parameters
    ----------
    target
        The filter target containing all necessary information.

    Returns
    -------
    A new Join node configured as a semi-join for prefiltering.
    """
    from cudf_polars.dsl.expr import Col, NamedExpr

    source = target.source
    target_scan = target.target_scan

    # Find the provider key that corresponds to source_column
    # source_column is in source.target_key_names, find corresponding provider key
    source_key_names = source.target_key_names
    provider_key_names = source.provider_key_names

    # Find the index of source_column in target_key_names
    try:
        key_idx = source_key_names.index(target.source_column)
    except ValueError:
        # source_column not found, use the first key as fallback
        key_idx = 0

    provider_key = (
        provider_key_names[key_idx]
        if key_idx < len(provider_key_names)
        else provider_key_names[0]
    )

    # Get the dtype for the target column from the target scan's schema
    target_dtype = target_scan.schema[target.target_column]

    # Get the dtype for the provider key from the filter_keys_provider's schema
    filter_keys_provider = source.filter_keys_provider
    # Handle Cache nodes
    provider_for_schema = filter_keys_provider
    while isinstance(provider_for_schema, Cache) and provider_for_schema.children:
        provider_for_schema = provider_for_schema.children[0]
    provider_dtype = provider_for_schema.schema[provider_key]

    # Create the join key expressions
    # Left side: target scan's column
    left_col = Col(target_dtype, target.target_column)
    left_on = (NamedExpr(target.target_column, left_col),)

    # Right side: filter keys provider's column
    right_col = Col(provider_dtype, provider_key)
    right_on = (NamedExpr(provider_key, right_col),)

    # Semi-join options: (how, nulls_equal, slice, suffix, coalesce, maintain_order)
    options = ("Semi", True, None, "_right", True, "none")

    # Schema for semi-join is same as left side (target scan)
    schema = target_scan.schema

    # Create the semi-join node
    return Join(
        schema,
        left_on,
        right_on,
        options,
        target_scan,  # left child: the scan to be filtered
        filter_keys_provider,  # right child: provides the filter keys
    )


def _rebuild_ir_with_replacements(
    ir: IR,
    replacements: MutableMapping[IR, IR],
) -> IR:
    """
    Rebuild an IR graph with node replacements.

    This traverses the IR in post-order and rebuilds nodes that have
    children that were replaced.

    Parameters
    ----------
    ir
        Root of the IR graph.
    replacements
        Mapping from old nodes to their replacements.

    Returns
    -------
    The rebuilt IR graph with replacements applied.
    """
    # Memoization to avoid rebuilding the same subtree multiple times
    rebuilt: dict[IR, IR] = {}

    def rebuild(node: IR) -> IR:
        if node in rebuilt:
            return rebuilt[node]

        # Check if this node should be replaced
        if node in replacements:
            result = replacements[node]
            rebuilt[node] = result
            return result

        # Recursively rebuild children
        new_children = tuple(rebuild(child) for child in node.children)

        # If no children changed, keep the original node
        if all(
            new is old for new, old in zip(new_children, node.children, strict=False)
        ):
            rebuilt[node] = node
            return node

        # Reconstruct the node with new children
        result = node.reconstruct(new_children)
        rebuilt[node] = result
        return result

    return rebuild(ir)


def _apply_semi_join_prefilters(
    ir: IR,
    filter_targets: FilterTargetCollection,
) -> IR:
    """
    Apply semi-join prefilters to the IR graph.

    For each filter target, this inserts a semi-join between the target
    scan and its downstream join to prefilter the data.

    Parameters
    ----------
    ir
        Root of the IR graph.
    filter_targets
        Collection of filter targets to apply.

    Returns
    -------
    The rewritten IR graph with prefilter semi-joins inserted.

    Notes
    -----
    This function groups targets by downstream join to avoid inserting
    multiple prefilters for the same target scan at the same join.
    """
    if len(filter_targets) == 0:
        return ir

    # Build replacements: map from (downstream_join, target_scan) to prefiltered version
    # We replace at the downstream_join level to ensure the prefilter is applied
    replacements: dict[IR, IR] = {}

    # Group targets by downstream join
    targets_by_join: dict[Join, list[FilterTarget]] = defaultdict(list)
    for target in filter_targets:
        targets_by_join[target.downstream_join].append(target)

    for downstream_join, targets in targets_by_join.items():
        # For now, apply the first target's prefilter
        # TODO: Handle multiple targets for the same join
        target = targets[0]

        # Create the prefilter semi-join
        prefilter = _create_prefilter_semi_join(target)

        # Determine which child of the downstream join to replace
        left_child, right_child = downstream_join.children

        # Check which child contains the target scan
        if _is_ancestor_of(target.target_scan, left_child):
            # Replace in left subtree
            # We need to replace target_scan with prefilter in the left subtree
            # Then rebuild the downstream join with the new left child
            left_replacements: dict[IR, IR] = {target.target_scan: prefilter}
            new_left = _rebuild_ir_with_replacements(left_child, left_replacements)
            new_join = downstream_join.reconstruct((new_left, right_child))
            replacements[downstream_join] = new_join
        elif _is_ancestor_of(target.target_scan, right_child):
            # Replace in right subtree
            right_replacements: dict[IR, IR] = {target.target_scan: prefilter}
            new_right = _rebuild_ir_with_replacements(right_child, right_replacements)
            new_join = downstream_join.reconstruct((left_child, new_right))
            replacements[downstream_join] = new_join

    if not replacements:
        return ir

    # Rebuild the entire IR with the join replacements
    return _rebuild_ir_with_replacements(ir, replacements)


def add_filters(
    ir: IR,
    config_options: ConfigOptions,
    stats: StatsCollector | None = None,
) -> IR:
    """
    Rewrite the IR graph to add dynamic filter pushdown optimizations.

    This is the main entry point for filter pushdown optimization.

    Parameters
    ----------
    ir
        Root of the IR graph.
    config_options
        GPUEngine configuration options.
    stats
        Optional statistics collector. If not provided, lightweight
        statistics will be collected automatically.

    Returns
    -------
    The potentially rewritten IR graph.
    """
    # Collect lightweight statistics if not provided
    if stats is None:
        stats = collect_selectivity_stats(ir, config_options)

    # Detect filter opportunities (semi-joins and selective inner joins)
    filter_sources = collect_filter_sources(ir, stats)

    # Only process semi-joins for now
    # (selective inner joins would need bloom filter support)
    if not filter_sources.semi_joins():
        return ir

    # Phase 2: Identify filter targets (downstream scans that can be prefiltered)
    filter_targets = find_filter_targets(ir, filter_sources)

    if len(filter_targets) == 0:
        return ir

    # Phase 3: Rewrite the IR graph to insert prefilter semi-joins
    # This inserts semi-join nodes between target scans and their downstream joins
    # to reduce data shuffled for the joins.
    #
    # NOTE: This does not reduce I/O if the scan is wrapped in a Cache node,
    # because the Cache will still buffer all data. I/O reduction requires
    # either bloom filter support in Scan, or making Scan nodes distinguishable.
    return _apply_semi_join_prefilters(ir, filter_targets)
