# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Asof-join logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pylibcudf as plc
from cudf_streaming.channel_metadata import ChannelMetadata
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.memory_reserve_or_wait import (
    missing_net_memory_delta,
    reserve_memory,
)

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import AsofJoin
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.join import (
    JoinCollectiveIds,
    _broadcast_chunks_to_frames,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    ChunkSampler,
    chunk_to_frame,
    empty_table_chunk,
    gather_in_task_group,
    maybe_remap_partitioning,
    process_children,
    recv_metadata,
    sample_inputs,
    send_metadata,
    shutdown_on_error,
)

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer
    from cudf_polars.streaming.actor_graph.utils import TableSizeStats
    from cudf_polars.utils.config import StreamingExecutor


def _asof_by_names(ir: AsofJoin) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the left and right equality grouping names for an asof join."""
    ((_, _, _, _, left_by, right_by, _, _), *_) = ir.options
    return left_by, right_by


def _sort_asof_frame(
    df: DataFrame,
    by_names: tuple[str, ...],
    on_exprs: tuple[NamedExpr, ...],
    *,
    context: IRExecutionContext,
    force: bool = False,
) -> DataFrame:
    """Sort one grouped asof input by ``by`` columns and its ordered key."""
    if not by_names:
        return df
    first_by = df.column_map[by_names[0]]
    if not force and (
        first_by.is_sorted == plc.types.Sorted.YES
        and first_by.order == plc.types.Order.ASCENDING
    ):
        return df

    on = DataFrame(
        broadcast(
            *(e.evaluate(df) for e in on_exprs),
            stream=df.stream,
        ),
        stream=df.stream,
    )
    by = df.select(by_names)
    with context.stream_ordered_after(df) as stream:
        (on_col,) = on.table.columns()
        gather_map = plc.sorting.stable_sorted_order(
            plc.Table([*by.table.columns(), on_col]),
            [plc.types.Order.ASCENDING] * (by.num_columns + 1),
            [plc.types.NullOrder.BEFORE] * (by.num_columns + 1),
            stream=stream,
        )
        sorted_df = DataFrame.from_table(
            plc.copying.gather(
                df.table,
                gather_map,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                stream=stream,
            ),
            df.column_names,
            df.dtypes,
            stream=stream,
        )
        sorted_df.column_map[by_names[0]].set_sorted(
            is_sorted=plc.types.Sorted.YES,
            order=plc.types.Order.ASCENDING,
            null_order=plc.types.NullOrder.BEFORE,
        )
        return sorted_df


def _sort_grouped_asof_right(
    ir: AsofJoin,
    right: DataFrame,
    *,
    context: IRExecutionContext,
) -> DataFrame:
    """Sort grouped right-side input once for chunk-wise asof joins."""
    _, right_by_names = _asof_by_names(ir)
    return _sort_asof_frame(
        right,
        right_by_names,
        ir.right_on,
        context=context,
    )


async def _collect_asof_right(
    context: Context,
    comm: Communicator,
    ir: AsofJoin,
    ir_context: IRExecutionContext,
    chunks: list[TableChunk],
    *,
    need_allgather: bool,
    collective_id: int,
) -> tuple[DataFrame, int]:
    """Construct the single right-side frame used by broadcast asof joins."""
    right = ir.children[1]
    dfs, size = await _broadcast_chunks_to_frames(
        context,
        comm,
        chunks,
        right,
        need_allgather=need_allgather,
        collective_id=collective_id,
        ir_context=ir_context,
        must_concatenate=True,
    )
    if dfs:
        (right_df,) = dfs
    else:
        stream = ir_context.get_cuda_stream()
        empty_right = empty_table_chunk(right, context, stream)
        right_df = chunk_to_frame(empty_right, right)
    return (
        await ir_context.to_thread(
            _sort_grouped_asof_right,
            ir,
            right_df,
            context=ir_context,
        ),
        size,
    )


async def _asof_join_left_chunk(
    context: Context,
    ir: AsofJoin,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    right_df: DataFrame,
    left_chunk: TableChunk,
    seq_num: int,
    right_size: int,
    *,
    tracer: ActorTracer | None,
) -> int:
    """Join one left-side chunk against the already-collected right frame."""
    left = ir.children[0]
    left_df = chunk_to_frame(left_chunk, left)
    input_bytes = left_chunk.data_alloc_size() + right_size
    with opaque_memory_usage(
        await reserve_memory(context, size=input_bytes, net_memory_delta=0)
    ):
        df = await ir_context.to_thread(
            ir.do_evaluate,
            *ir._non_child_args,
            left_df,
            right_df,
            context=ir_context,
        )
    output_chunk = TableChunk.from_pylibcudf_table(
        df.table, df.stream, exclusive_view=True, br=context.br()
    )
    output_rows = output_chunk.shape[0]
    await send_chunk(context, ch_out, output_chunk, seq_num, tracer=tracer)
    del df, left_df
    return output_rows


async def _stream_asof_left(
    context: Context,
    ir: AsofJoin,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    right_df: DataFrame,
    right_size: int,
    *,
    tracer: ActorTracer | None,
) -> None:
    """Stream left-side chunks and emit asof-join results."""
    input_rows = 0
    output_rows = 0
    while (msg := await ch_left.recv(context)) is not None:
        left_chunk, _ = await make_table_chunks_available_or_wait(
            context,
            TableChunk.from_message(msg, br=context.br()),
            reserve_extra=0,
            net_memory_delta=missing_net_memory_delta,
        )
        input_rows += left_chunk.shape[0]
        output_rows += await _asof_join_left_chunk(
            context,
            ir,
            ir_context,
            ch_out,
            right_df,
            left_chunk,
            msg.sequence_number,
            right_size,
            tracer=tracer,
        )
    if tracer is not None:
        tracer.set_extra("input_rows", input_rows)
        tracer.row_count = output_rows
    await ch_out.drain(context)


async def _collect_local_right_chunks(
    context: Context,
    ch_right: Channel[TableChunk],
    sample: TableSizeStats,
) -> list[TableChunk]:
    """Collect all rank-local right-side chunks for broadcast asof joins."""
    chunks = [
        TableChunk.from_message(msg, br=context.br())
        for msg in sample.local_sample.chunks
    ]
    if not sample.is_complete:
        while (msg := await ch_right.recv(context)) is not None:
            chunks.append(TableChunk.from_message(msg, br=context.br()))
    return chunks


@define_actor()
async def join_asof_actor(
    context: Context,
    comm: Communicator,
    ir: AsofJoin,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    executor: StreamingExecutor,
    collective_ids: JoinCollectiveIds,
) -> None:
    """
    Dynamic asof-join actor.

    The first implementation broadcasts the right side. This preserves the
    left-side row order that ``join_asof`` exposes to downstream operations.
    """
    async with shutdown_on_error(
        context,
        chs_in=(ch_left, ch_right),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as tracer:
        ir_context = replace(ir_context, tracer=tracer)
        dynamic_planning = executor.dynamic_planning
        if dynamic_planning is None:
            raise RuntimeError("Streaming AsofJoin requires dynamic planning")

        left_metadata, right_metadata = await gather_in_task_group(
            recv_metadata(ch_left, context),
            recv_metadata(ch_right, context),
        )
        (right_sample,) = await sample_inputs(
            context,
            comm,
            (
                ChunkSampler(
                    context=context,
                    ch_in=ch_right,
                    max_chunks=dynamic_planning.sample_chunk_count,
                    max_bytes=executor.target_partition_size,
                    ch_in_chunk_count=right_metadata.local_count,
                ),
            ),
            collective_ids.size_estimate,
        )

        if tracer is not None:
            tracer.decision = "broadcast_right"
        need_allgather = comm.nranks > 1 and not right_metadata.duplicated
        right_df, right_size = await _collect_asof_right(
            context,
            comm,
            ir,
            ir_context,
            await _collect_local_right_chunks(context, ch_right, right_sample),
            need_allgather=need_allgather,
            collective_id=collective_ids.broadcast,
        )
        metadata_out = ChannelMetadata(
            local_count=left_metadata.local_count,
            partitioning=maybe_remap_partitioning(
                ir,
                left_metadata.partitioning,
                child_ir=ir.children[0],
                context=context,
            ),
            duplicated=(right_metadata.duplicated or need_allgather)
            and left_metadata.duplicated,
        )
        if tracer is not None:
            tracer.set_duplicated(duplicated=metadata_out.duplicated)
        await send_metadata(ch_out, context, metadata_out)
        await _stream_asof_left(
            context,
            ir,
            ir_context,
            ch_out,
            ch_left,
            right_df,
            right_size,
            tracer=tracer,
        )


@generate_ir_sub_network.register(AsofJoin)
def _(
    ir: AsofJoin, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate a dynamic asof-join actor network."""
    left, right = ir.children
    executor = rec.state["config_options"].executor
    if executor.dynamic_planning is None:
        raise RuntimeError("Streaming AsofJoin requires dynamic planning")

    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    ir_context = ir_context_for_node(rec, ir)
    collective_ids = JoinCollectiveIds.from_reserved(
        rec.state["collective_id_map"].get(ir, [])
    )
    actors[ir] = [
        join_asof_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            ir_context,
            channels[ir].reserve_input_slot(),
            channels[left].reserve_output_slot(),
            channels[right].reserve_output_slot(),
            executor,
            collective_ids,
        )
    ]
    return actors, channels
