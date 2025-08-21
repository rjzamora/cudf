# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracking column statistics."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import (
    IR,
    DataFrameScan,
    Distinct,
    GroupBy,
    HConcat,
    Join,
    Scan,
    Union,
)
from cudf_polars.dsl.traversal import post_traversal
from cudf_polars.experimental.base import (
    ColumnStat,
    ColumnStats,
    JoinKey,
    StatsCollector,
    UniqueStats,
)
from cudf_polars.experimental.dispatch import (
    initialize_column_stats,
    update_column_stats,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cudf_polars.utils.config import ConfigOptions


def collect_statistics(root: IR, config_options: ConfigOptions) -> StatsCollector:
    """
    Collect column statistics for a query.

    Parameters
    ----------
    root
        Root IR node for collecting column statistics.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    A StatsCollector object with populated column statistics.
    """
    # Start with base statistics.
    # Here we build an outline of the statistics that will be
    # collected before any real data is sampled.
    stats = collect_base_stats(root, config_options)

    # Apply PK-FK heuristics.
    # Here we use PK-FK heuristics to estimate the unique-count
    # for each join key (without needing to calculate unique-value
    # statistics with sampled data).
    apply_pkfk_heuristics(stats.join_keys)

    # Update statistics for each node.
    # Here we set local row-count and unique-value statistics
    # on each node in the IR graph.
    for node in post_traversal([root]):
        update_column_stats(node, stats, config_options)

    return stats


def collect_base_stats(root: IR, config_options: ConfigOptions) -> StatsCollector:
    """
    Collect base datasource statistics.

    Parameters
    ----------
    root
        Root IR node for collecting base datasource statistics.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    A new StatsCollector object with populated datasource statistics.

    Notes
    -----
    This function initializes the ``StatsCollector`` object
    with the base datasource statistics. The goal is to build an
    outline of the statistics that will be collected before any
    real data is sampled.
    """
    stats: StatsCollector = StatsCollector()
    for node in post_traversal([root]):
        # Initialize column statistics from datasource information
        stats.column_stats[node] = initialize_column_stats(node, stats, config_options)
        # Initialize Join-key information
        initialize_join_key_info(node, stats, config_options)
    return stats


def initialize_join_key_info(
    node: IR, stats: StatsCollector, config_options: ConfigOptions
) -> None:
    """
    Initialize join-key information for the given node.

    Parameters
    ----------
    node
        IR node to initialize join-key information for.
    stats
        StatsCollector object to update.
    config_options
        GPUEngine configuration options.

    Notes
    -----
    This function updates ``stats.joins`` and ``stats.join_keys``.
    """
    if isinstance(node, Join):
        # Only need to update join-key information for Join nodes.
        left, right = node.children
        lkey = JoinKey(*[stats.column_stats[left][n.name] for n in node.left_on])
        rkey = JoinKey(*[stats.column_stats[right][n.name] for n in node.right_on])
        stats.join_keys[lkey].add(rkey)
        stats.join_keys[rkey].add(lkey)
        stats.joins[node] = [lkey, rkey]


def find_equivalence_sets(joins: Mapping[JoinKey, set[JoinKey]]) -> list[set[JoinKey]]:
    """
    Find equivalence sets in a join-key mapping.

    Parameters
    ----------
    joins
        Join-key mapping to find equivalence sets in.

    Returns
    -------
    List of equivalence sets.

    Notes
    -----
    This function is used by ``apply_pkfk_heuristics``.
    """
    seen = set()
    components = []
    for v in joins:
        if v not in seen:
            cluster = {v}
            stack = [v]
            while stack:
                node = stack.pop()
                for n in joins[node]:
                    if n not in cluster:
                        cluster.add(n)
                        stack.append(n)
            components.append(cluster)
            seen.update(cluster)
    return components


def apply_pkfk_heuristics(joins: Mapping[JoinKey, set[JoinKey]]) -> None:
    """
    Apply PK-FK unique-count heuristics to join keys.

    Parameters
    ----------
    joins
        Join-key mapping to apply PK-FK heuristics to.

    Notes
    -----
    This function modifies the ``JoinKey`` objects being tracked
    in ``StatsCollector.joins`` and ``StatsCollector.join_keys``
    using PK-FK heuristics to estimate the local unique-value count.
    """
    # This applies the PK-FK matching scheme of
    # https://blobs.duckdb.org/papers/tom-ebergen-msc-thesis-join-order-optimization-with-almost-no-statistics.pdf
    # See section 3.2
    for keys in find_equivalence_sets(joins):
        unique_count_estimate = max(
            (
                c.unique_count_estimate
                for c in keys
                if c.unique_count_estimate is not None
            ),
            # Default unique-count estimate is the minimum source row count
            default=min(
                (c.source_row_count for c in keys if c.source_row_count is not None),
                default=None,
            ),
        )
        for key in keys:
            # Update unique-count estimate for each join key
            key.unique_count_estimate = unique_count_estimate


def _update_unique_stats_columns(
    child_column_stats: dict[str, ColumnStats],
    key_names: Sequence[str],
    config_options: ConfigOptions,
) -> None:
    """Update set of unique-stats columns in datasource."""
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'add_source_stats'"
    )
    unique_fraction = config_options.executor.unique_fraction
    for name in key_names:
        if (
            name not in unique_fraction
            and (column_stats := child_column_stats.get(name)) is not None
        ):
            column_stats.source_info.add_unique_stats_column()


@initialize_column_stats.register(IR)
def _default_initialize_column_stats(
    ir: IR, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    # Default `initialize_column_stats` implementation.
    if len(ir.children) == 1:
        (child,) = ir.children
        child_column_stats = stats.column_stats.get(child, {})
        return {
            name: child_column_stats.get(name, ColumnStats(name=name)).new_parent()
            for name in ir.schema
        }
    else:  # pragma: no cover
        # Multi-child nodes loose all information by default.
        return {name: ColumnStats(name=name) for name in ir.schema}


@initialize_column_stats.register(Distinct)
def _(
    ir: Distinct, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    # Use default initialize_column_stats after updating
    # the known unique-stats columns.
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})
    key_names = ir.subset or ir.schema
    _update_unique_stats_columns(child_column_stats, list(key_names), config_options)
    return _default_initialize_column_stats(ir, stats, config_options)


@initialize_column_stats.register(Join)
def _(
    ir: Join, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    # Copy column statistics from both the left and right children.
    # Special cases to consider:
    #   - If a column name appears in both sides of the join,
    #     we take it from the "primary" column (right for "Right"
    #     joins, left for all other joins).
    #   - If a column name doesn't appear in either child, it
    #     corresponds to a non-"primary" column with a suffix.

    children, on = ir.children, (ir.left_on, ir.right_on)
    how = ir.options[0]
    suffix = ir.options[3]
    if how == "Right":
        children, on = children[::-1], on[::-1]
    primary, other = children
    primary_child_stats = stats.column_stats.get(primary, {})
    other_child_stats = stats.column_stats.get(other, {})

    # Build output column statistics
    column_stats: dict[str, ColumnStats] = {}
    for name in ir.schema:
        if name in primary.schema:
            # "Primary" child stats take preference.
            column_stats[name] = primary_child_stats[name].new_parent()
        elif name in other.schema:
            # "Other" column stats apply to everything else.
            column_stats[name] = other_child_stats[name].new_parent()
        else:
            # If the column name was not in either child table,
            # a suffix was added to a column in "other".
            _name = name.removesuffix(suffix)
            column_stats[name] = other_child_stats[_name].new_parent(name=name)

    # Update children
    for p_key, o_key in zip(*on, strict=True):
        column_stats[p_key.name].children = (
            primary_child_stats[p_key.name],
            other_child_stats[o_key.name],
        )
        # Add key columns to set of unique-stats columns.
        primary_child_stats[p_key.name].source_info.add_unique_stats_column()
        other_child_stats[o_key.name].source_info.add_unique_stats_column()

    return column_stats


@initialize_column_stats.register(GroupBy)
def _(
    ir: GroupBy, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    (child,) = ir.children
    child_column_stats = stats.column_stats.get(child, {})

    # Update set of source columns we may lazily sample
    _update_unique_stats_columns(
        child_column_stats, [n.name for n in ir.keys], config_options
    )
    return _default_initialize_column_stats(ir, stats, config_options)


@initialize_column_stats.register(HConcat)
def _(
    ir: HConcat, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    child_column_stats = dict(
        itertools.chain.from_iterable(
            stats.column_stats.get(c, {}).items() for c in ir.children
        )
    )
    return {
        name: child_column_stats.get(name, ColumnStats(name=name)).new_parent()
        for name in ir.schema
    }


@initialize_column_stats.register(Union)
def _(
    ir: IR, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    # Union looses source information for now.
    return {
        name: ColumnStats(
            name=name,
            children=tuple(stats.column_stats[child][name] for child in ir.children),
        )
        for name in ir.schema
    }


@initialize_column_stats.register(Scan)
def _(
    ir: Scan, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    from cudf_polars.experimental.io import _extract_scan_stats

    return _extract_scan_stats(ir, config_options)


@initialize_column_stats.register(DataFrameScan)
def _(
    ir: DataFrameScan, stats: StatsCollector, config_options: ConfigOptions
) -> dict[str, ColumnStats]:
    from cudf_polars.experimental.io import _extract_dataframescan_stats

    return _extract_dataframescan_stats(ir)


def child_row_counts(
    ir: IR, stats: StatsCollector, *, strict: bool = True
) -> list[int]:
    """
    Get row-count estimates for all children of the given IR node.

    Parameters
    ----------
    ir
        IR node to get row-count estimates for.
    stats
        StatsCollector object to get row-count estimates from.
    strict
        If True, returns an empty list if any child has an unknown row-count estimate.

    Returns
    -------
    List of non-null row-count estimates for all children.
    """
    child_row_counts: list[int] = []
    for child in ir.children:
        if (value := stats.row_count[child].value) is None:
            if strict:
                # If strict, return an empty list if any
                # child has an unknown row-count estimate.
                return []
        else:
            child_row_counts.append(value)
    return child_row_counts


def copy_child_unique_count(ir: IR, stats: StatsCollector) -> None:
    """
    Copy unique-value count information from child statistics.

    Parameters
    ----------
    ir
        IR node to copy unique-stats information from.
    stats
        StatsCollector object to update.
    """
    for column_stats in stats.column_stats[ir].values():
        child_unique_counts = [
            child_column_stats.unique_stats.count.value
            for child_column_stats in column_stats.children
        ]

        if child_unique_counts and None not in child_unique_counts:
            # We only need to update the unique-count if there are
            # children with known unique-counts.
            if len(column_stats.children) == 1:
                # If there is only one child, we can use the child's unique-stats.
                column_stats.unique_stats = column_stats.children[0].unique_stats
            else:
                # Take the maximum unique-count for multiple children.
                column_stats.unique_stats = UniqueStats(
                    count=ColumnStat[int](
                        max(c for c in child_unique_counts if c is not None)
                    )
                )


@update_column_stats.register(IR)
def _(ir: IR, stats: StatsCollector, config_options: ConfigOptions) -> None:
    # Default `update_column_stats` implementation.
    # Propagate largest child row-count estimate.
    stats.row_count[ir] = ColumnStat[int](
        max(child_row_counts(ir, stats), default=None)
    )
    # Copy unique-value count from children.
    copy_child_unique_count(ir, stats)


@update_column_stats.register(DataFrameScan)
def _(ir: DataFrameScan, stats: StatsCollector, config_options: ConfigOptions) -> None:
    # Use datasource row-count estimate.
    stats.row_count[ir] = next(
        iter(stats.column_stats[ir].values())
    ).source_info.row_count

    # Use datasource unique-stats information.
    for column_stats in stats.column_stats[ir].values():
        # We use force=False to avoid sampling unnecessary unique-stats.
        column_stats.unique_stats = column_stats.source_info.unique_stats(force=False)


@update_column_stats.register(Scan)
def _(ir: Scan, stats: StatsCollector, config_options: ConfigOptions) -> None:
    # Use datasource row-count estimate.
    if ir.n_rows != -1:
        stats.row_count[ir] = ColumnStat[int](ir.n_rows)
    else:
        # TODO: Apply predicate selectivity
        stats.row_count[ir] = next(
            iter(stats.column_stats[ir].values())
        ).source_info.row_count

    # Use datasource unique-stats information.
    for column_stats in stats.column_stats[ir].values():
        # We use force=False to avoid sampling unnecessary unique-stats.
        column_stats.unique_stats = column_stats.source_info.unique_stats(force=False)


@update_column_stats.register(Join)
def _(ir: Join, stats: StatsCollector, config_options: ConfigOptions) -> None:
    # Copy unique-value count from children.
    copy_child_unique_count(ir, stats)

    # Apply basic join-cardinality estimation.
    try:
        left_rows, right_rows = child_row_counts(ir, stats)
    except ValueError:
        # One or more children have an unknown row-count estimate.
        stats.row_count[ir] = ColumnStat[int](None)
    else:
        # Both children have row-count estimates.
        # Account for sampled unique-count estimates.
        sampled_unique_count_estimates: list[int] = []
        for key in ir.left_on:
            value = stats.column_stats[ir.children[0]][
                key.name
            ].unique_stats.count.value
            if value is not None:
                sampled_unique_count_estimates.append(value)
        for key in ir.right_on:
            value = stats.column_stats[ir.children[1]][
                key.name
            ].unique_stats.count.value
            if value is not None:
                sampled_unique_count_estimates.append(value)

        unique_estimate = max(
            # Use PK-FK join unique-count estimates in case
            # directly-sampled statistics are missing.
            [
                u.unique_count_estimate
                for u in stats.joins[ir]
                if u.unique_count_estimate is not None
            ]
            + sampled_unique_count_estimates,
            default=None,
        )
        if unique_estimate is not None:
            stats.row_count[ir] = ColumnStat[int](
                max(1, (left_rows * right_rows) // unique_estimate)
            )
        else:
            stats.row_count[ir] = ColumnStat[int](max((1, left_rows, right_rows)))


@update_column_stats.register(Union)
def _(ir: Union, stats: StatsCollector, config_options: ConfigOptions) -> None:
    # Sum child row-count estimates.
    # Note: We cannot inherit unique-stats information from children.
    row_counts = child_row_counts(ir, stats)
    stats.row_count[ir] = ColumnStat[int](sum(row_counts) if row_counts else None)
