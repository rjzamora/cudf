# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Dynamic filter pushdown optimization.

This module implements optimizations that detect semi-join and selective
inner join patterns, using filter predicates to prefilter data before
shuffles to reduce data movement.
"""

from __future__ import annotations

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
    from collections.abc import Iterator

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

    # Detect filter opportunities
    filter_sources = collect_filter_sources(ir, stats)

    # TODO: Phase 2 - Identify filter targets (downstream scans/shuffles)
    # TODO: Phase 3 - Rewrite graph to insert filter nodes

    # For now, just return the IR unchanged
    # (the detected filter sources can be used for debugging/analysis)
    _ = filter_sources

    return ir
