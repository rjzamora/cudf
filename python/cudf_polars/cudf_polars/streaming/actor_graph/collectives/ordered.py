# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local preparation primitives for row-ordered streams."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pylibcudf as plc
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.memory_reservation import opaque_memory_usage

from cudf_polars.utils.cuda_stream import stream_ordered_after

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_streaming.channel_metadata import OrderKey
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream


def _copy_slice(
    context: Context,
    table: plc.Table,
    start: int,
    stop: int,
    stream: Stream,
) -> TableChunk | None:
    """Copy a nonempty table slice into an independently owned chunk."""
    if start == stop:
        return None
    (view,) = plc.copying.slice(table, [start, stop], stream=stream)
    return TableChunk.from_pylibcudf_table(
        view.copy(stream, context.br().device_mr),
        stream,
        exclusive_view=True,
        br=context.br(),
    )


async def stable_sort_chunk(
    context: Context,
    chunk: TableChunk,
    order_keys: Sequence[OrderKey],
) -> TableChunk:
    """Return a chunk stably sorted by ``order_keys``."""
    if chunk.shape[0] < 2:
        return chunk
    chunk, extra = await make_table_chunks_available_or_wait(
        context,
        chunk,
        reserve_extra=chunk.data_alloc_size(),
        net_memory_delta=0,
    )
    with opaque_memory_usage(extra):
        table = chunk.table_view()
        keys = plc.Table([table.columns()[key.column_index] for key in order_keys])
        table = plc.sorting.stable_sort_by_key(
            table,
            keys,
            [key.order for key in order_keys],
            [key.null_order for key in order_keys],
            stream=chunk.stream,
            mr=context.br().device_mr,
        )
    return TableChunk.from_pylibcudf_table(
        table,
        chunk.stream,
        exclusive_view=True,
        br=context.br(),
    )


async def split_trailing_run(
    context: Context,
    chunk: TableChunk,
    order_keys: Sequence[OrderKey],
) -> tuple[TableChunk, TableChunk | None]:
    """Split a sorted chunk before its final equal-key run."""
    nrows = chunk.shape[0]
    if nrows == 0:
        return chunk, None
    chunk, extra = await make_table_chunks_available_or_wait(
        context,
        chunk,
        reserve_extra=chunk.data_alloc_size(),
        net_memory_delta=0,
    )
    with opaque_memory_usage(extra):
        table = chunk.table_view()
        keys = plc.Table([table.columns()[key.column_index] for key in order_keys])
        (last_key,) = plc.copying.slice(
            keys,
            [nrows - 1, nrows],
            stream=chunk.stream,
        )
        split_column = plc.search.lower_bound(
            keys,
            last_key,
            [key.order for key in order_keys],
            [key.null_order for key in order_keys],
            stream=chunk.stream,
        )
        split = cast(
            "int",
            split_column.to_scalar(stream=chunk.stream).to_py(stream=chunk.stream),
        )
        complete_view, trailing_view = plc.copying.split(
            table,
            [split],
            stream=chunk.stream,
        )
        complete = complete_view.copy(chunk.stream, context.br().device_mr)
        trailing = trailing_view.copy(chunk.stream, context.br().device_mr)
    return (
        TableChunk.from_pylibcudf_table(
            complete,
            chunk.stream,
            exclusive_view=True,
            br=context.br(),
        ),
        TableChunk.from_pylibcudf_table(
            trailing,
            chunk.stream,
            exclusive_view=True,
            br=context.br(),
        ),
    )


async def select_demand_range(
    context: Context,
    supply: TableChunk,
    demand: TableChunk,
    supply_keys: Sequence[OrderKey],
    demand_keys: Sequence[OrderKey],
) -> tuple[TableChunk | None, TableChunk | None, TableChunk, bool]:
    """Select supply rows in one demand range and retain any later suffix."""
    if supply.shape[0] == 0:
        return None, None, demand, False
    if demand.shape[0] == 0:
        return None, supply, demand, True
    (supply, demand), extra = await make_table_chunks_available_or_wait(
        context,
        [supply, demand],
        reserve_extra=supply.data_alloc_size(),
        net_memory_delta=0,
    )
    with (
        opaque_memory_usage(extra),
        stream_ordered_after(
            context.br().stream_pool.get_stream,
            upstreams=(supply.stream, demand.stream),
        ) as stream,
    ):
        supply_table = supply.table_view()
        demand_table = demand.table_view()
        supply_key_table = plc.Table(
            [supply_table.columns()[key.column_index] for key in supply_keys]
        )
        demand_key_table = plc.Table(
            [demand_table.columns()[key.column_index] for key in demand_keys]
        )
        (first_key,) = plc.copying.slice(demand_key_table, [0, 1], stream=stream)
        (last_key,) = plc.copying.slice(
            demand_key_table,
            [demand.shape[0] - 1, demand.shape[0]],
            stream=stream,
        )
        start_column = plc.search.lower_bound(
            supply_key_table,
            first_key,
            [key.order for key in supply_keys],
            [key.null_order for key in supply_keys],
            stream=stream,
        )
        stop_column = plc.search.upper_bound(
            supply_key_table,
            last_key,
            [key.order for key in supply_keys],
            [key.null_order for key in supply_keys],
            stream=stream,
        )
        start = cast("int", start_column.to_scalar(stream=stream).to_py(stream=stream))
        stop = cast("int", stop_column.to_scalar(stream=stream).to_py(stream=stream))
        passed_demand = stop < supply.shape[0]
        selected = _copy_slice(context, supply_table, start, stop, stream)
        suffix = (
            _copy_slice(context, supply_table, stop, supply.shape[0], stream)
            if passed_demand
            else None
        )
    return selected, suffix, demand, passed_demand
