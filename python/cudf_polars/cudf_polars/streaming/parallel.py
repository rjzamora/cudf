# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition evaluation."""

from __future__ import annotations

import operator
from functools import partial, reduce
from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc

# Side-effect imports: each module registers ``@lower_ir_node.register(...)``
# handlers at import time so the dispatch table is populated before any query
# is lowered.
import cudf_polars.streaming.distinct
import cudf_polars.streaming.groupby
import cudf_polars.streaming.io
import cudf_polars.streaming.join
import cudf_polars.streaming.select
import cudf_polars.streaming.shuffle
import cudf_polars.streaming.sort  # noqa: F401
from cudf_polars.containers import DataType
from cudf_polars.dsl.expr import Col, Literal, NamedExpr
from cudf_polars.dsl.ir import (
    IR,
    Cache,
    Filter,
    HConcat,
    HStack,
    MapFunction,
    Projection,
    Scan,
    Select,
    Slice,
    Sort,
    Union,
)
from cudf_polars.dsl.traversal import CachingVisitor, traversal
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.streaming.base import IOPartitionFlavor, PartitionInfo
from cudf_polars.streaming.dispatch import lower_ir_node
from cudf_polars.streaming.io import _clear_source_info_cache
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.sort import HintSorted, ParquetFooterHint
from cudf_polars.streaming.utils import (
    _contains_over,
    _dynamic_planning_on,
    _lower_ir_fallback,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.streaming.base import StatsCollector
    from cudf_polars.streaming.dispatch import LowerIRTransformer, State
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


def _hint_matches_sort_prefix(ir: Sort, sorted_info: tuple) -> bool:
    if len(sorted_info) > len(ir.by):
        return False
    for (name, descending, nulls_last), by, order, null_order in zip(
        sorted_info, ir.by, ir.order, ir.null_order, strict=False
    ):
        by_name = by.name if isinstance(by, NamedExpr) else by
        if name != by_name:
            return False
        if descending != (order == plc.types.Order.DESCENDING):
            return False
        expected_null_order = (
            plc.types.NullOrder.AFTER
            if nulls_last != descending
            else plc.types.NullOrder.BEFORE
        )
        if null_order != expected_null_order:
            return False
    return True


def _can_use_parquet_footer_hint(
    scan: Scan,
    key_names: list[str],
    info: PartitionInfo,
    config_options: ConfigOptions[StreamingExecutor],
) -> bool:
    return (
        len(key_names) == 1
        and scan.typ == "parquet"
        and scan.skip_rows == 0
        and scan.n_rows == -1
        and scan.row_index is None
        and scan.include_file_paths is None
        and info.io_plan is not None
        and not config_options.parquet_options.use_rapidsmpf_native
        and info.io_plan.flavor
        in {
            IOPartitionFlavor.SINGLE_FILE,
            IOPartitionFlavor.SPLIT_FILES,
            IOPartitionFlavor.FUSED_FILES,
            IOPartitionFlavor.SINGLE_READ,
        }
    )


@lower_ir_node.register(IR)
def _(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:  # pragma: no cover
    # Default logic - Requires single partition
    return _lower_ir_fallback(
        ir, rec, msg=f"Class {type(ir)} does not support multiple partitions."
    )


def lower_ir_graph(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    stats: StatsCollector,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Rewrite an IR graph and extract partitioning information.

    Parameters
    ----------
    ir
        Root of the graph to rewrite.
    config_options
        GPUEngine configuration options.
    stats
        Pre-computed statistics collector.

    Returns
    -------
    new_ir, partition_info
        The rewritten graph and a mapping from unique nodes
        in the new graph to associated partitioning information.

    Notes
    -----
    This function traverses the unique nodes of the graph with
    root `ir`, and applies :func:`lower_ir_node` to each node.

    See Also
    --------
    lower_ir_node
    """
    state: State = {
        "config_options": config_options,
        "stats": stats,
    }
    mapper: LowerIRTransformer = CachingVisitor(lower_ir_node, state=state)
    return mapper(ir)


def evaluate_streaming(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
) -> pl.DataFrame:
    """
    Evaluate an IR graph with partitioning.

    Parameters
    ----------
    ir
        Logical plan to evaluate.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    A cudf-polars DataFrame object.
    """
    # Clear source info cache in case data was overwritten
    _clear_source_info_cache()

    from cudf_polars.streaming.actor_graph.core import evaluate_logical_plan

    result, _ = evaluate_logical_plan(ir, config_options, collect_metadata=False)
    return result


@lower_ir_node.register(Union)
def _(
    ir: Union, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Check zlice
    if ir.zlice is not None:
        return rec(
            Slice(
                ir.schema,
                *ir.zlice,
                Union(ir.schema, None, *ir.children),
            )
        )

    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    # Partition count is the sum of all child partitions
    count = sum(partition_info[c].count for c in children)

    # Return reconstructed node and partition-info dict
    new_node = ir.reconstruct(children)
    partition_info[new_node] = PartitionInfo(count=count)
    return new_node, partition_info


@lower_ir_node.register(MapFunction)
def _(
    ir: MapFunction, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Allow pointwise operations
    if ir.name in ("rename", "explode"):
        return _lower_ir_pwise(ir, rec)
    if ir.name == "hint_sorted" and _dynamic_planning_on(rec.state["config_options"]):
        if isinstance(ir.children[0], Sort) and _hint_matches_sort_prefix(
            ir.children[0], ir.options[0]
        ):
            return rec(ir.children[0])
        child, partition_info = rec(ir.children[0])
        key_names = [col_name for col_name, *_ in ir.options[0]]
        original = ir.children[0]
        full_cols = (
            original.with_columns
            if isinstance(original, Scan) and original.with_columns is not None
            else list(original.schema)
        )
        if (
            isinstance(original, Scan)
            and original.typ == "parquet"
            and len(key_names) < len(full_cols)
        ):
            data_info = partition_info[child]
            if _can_use_parquet_footer_hint(
                original, key_names, data_info, rec.state["config_options"]
            ):
                assert data_info.io_plan is not None
                hint_node = HintSorted(
                    ir.schema,
                    ir.options,
                    ParquetFooterHint(
                        tuple(original.paths),
                        key_names[0],
                        int(data_info.io_plan.flavor),
                        data_info.io_plan.factor,
                    ),
                    child,
                )
                partition_info[hint_node] = data_info
                return hint_node, partition_info
        hint_node = HintSorted(ir.schema, ir.options, None, child)
        partition_info[hint_node] = partition_info[child]
        return hint_node, partition_info

    # Fallback for everything else
    return _lower_ir_fallback(
        ir, rec, msg=f"{ir.name} is not supported for multiple partitions."
    )


def _lower_ir_pwise(
    ir: IR, rec: LowerIRTransformer, *, preserve_partitioning: bool = False
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Lower a partition-wise (i.e. embarrassingly-parallel) IR node

    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)
    counts = {partition_info[c].count for c in children}

    # Check that child partitioning is supported
    if len(counts) > 1:  # pragma: no cover
        return _lower_ir_fallback(
            ir,
            rec,
            msg=f"Class {type(ir)} does not support children with mismatched partition counts.",
        )

    # Preserve child partition_info if possible
    if preserve_partitioning and len(children) == 1:
        partition = partition_info[children[0]]
    else:
        partition = PartitionInfo(count=max(counts))

    # Return reconstructed node and partition-info dict
    new_node = ir.reconstruct(children)
    partition_info[new_node] = partition
    return new_node, partition_info


_lower_ir_pwise_preserve = partial(_lower_ir_pwise, preserve_partitioning=True)
lower_ir_node.register(Cache, _lower_ir_pwise_preserve)
lower_ir_node.register(HConcat, _lower_ir_pwise)


@lower_ir_node.register(Filter)
def _(
    ir: Filter, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    child, partition_info = rec(ir.children[0])

    if partition_info[child].count > 1 and _contains_over([ir.mask.value]):
        # mask contains .over(...), collapse to single partition
        return _lower_ir_fallback(
            ir.reconstruct([child]),
            rec,
            msg=(
                "over(...) inside filter is not supported for multiple partitions; "
                "falling back to in-memory evaluation."
            ),
        )

    if partition_info[child].count > 1 and not all(
        expr.is_pointwise for expr in traversal([ir.mask.value])
    ):
        # TODO: Use expression decomposition to lower Filter
        # See: https://github.com/rapidsai/cudf/issues/20076
        return _lower_ir_fallback(
            ir, rec, msg="This filter is not supported for multiple partitions."
        )

    new_node = ir.reconstruct([child])
    partition_info[new_node] = partition_info[child]
    return new_node, partition_info


@lower_ir_node.register(Slice)
def _(
    ir: Slice, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Check for dynamic planning - may have more partitions at runtime
    config_options = rec.state["config_options"]
    dynamic_planning = _dynamic_planning_on(config_options)

    if ir.offset == 0:
        # Taking the first N rows.
        # We don't know how large each partition is, so we reduce.
        new_node, partition_info = _lower_ir_pwise(ir, rec)
        if partition_info[new_node].count > 1 or dynamic_planning:
            # Collapse down to single partition
            inter = Repartition(new_node.schema, new_node)
            partition_info[inter] = PartitionInfo(count=1)
            # Slice reduced partition
            new_node = ir.reconstruct([inter])
            partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info

    # Fallback
    return _lower_ir_fallback(
        ir, rec, msg="This slice not supported for multiple partitions."
    )


def _add_anchor_column(ir: HStack) -> tuple[HStack, str, DataType]:
    """Add temporary anchor column to preserve row count."""
    anchor_name = next(unique_names((*ir.schema, *ir.children[0].schema)))
    anchor_dtype = DataType(pl.datatypes.Int8())
    anchor_named_expr = NamedExpr(anchor_name, Literal(anchor_dtype, 0))
    new_ir = HStack(
        ir.children[0].schema | {anchor_name: anchor_dtype},
        (anchor_named_expr,),
        True,  # noqa: FBT003
        ir.children[0],
    )
    return new_ir, anchor_name, anchor_dtype


@lower_ir_node.register(HStack)
def _(
    ir: HStack, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    if not all(e.is_pointwise for e in traversal([ne.value for ne in ir.columns])):
        # Redirect non-pointwise HStack to Select so the Select handler can
        # attempt decomposition (or fall back gracefully via decompose_select).
        child: IR = ir.children[0]
        anchor_name: str | None = None
        col_map = {ne.name: ne for ne in ir.columns}
        schema = ir.schema
        if ir.should_broadcast and all(name in col_map for name in ir.schema):
            # We need to add a temporary anchor column to preserve row count.
            child, anchor_name, anchor_dtype = _add_anchor_column(ir)

            schema = ir.schema | {anchor_name: anchor_dtype}
        exprs = tuple(
            col_map[name] if name in col_map else NamedExpr(name, Col(dtype, name))
            for name, dtype in schema.items()
        )
        new_ir: Select | Projection = Select(schema, exprs, ir.should_broadcast, child)
        if anchor_name is not None:
            # Need to drop the temporary anchor column.
            schema = {
                name: dtype
                for name, dtype in new_ir.schema.items()
                if name != anchor_name
            }
            new_ir = Projection(schema, new_ir)
        return lower_ir_node(new_ir, rec)

    child, partition_info = rec(ir.children[0])
    new_node = ir.reconstruct([child])
    partition_info[new_node] = partition_info[child]
    return new_node, partition_info
