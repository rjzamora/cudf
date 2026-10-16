# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming actor for dynamic grouped aggregations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    OrderKey,
    OrderScheme,
    Ordering,
    Partitioning,
)
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.streaming.core.actor import define_actor

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import Col
from cudf_polars.dsl.expressions.dynamic import DynamicWindowStart
from cudf_polars.dsl.ir import GroupByDynamic
from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.collectives.ordering import (
    adjust_ordering,
    ensure_ordering_metadata,
)
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    NormalizedPartitioning,
    chunkwise_evaluate,
    empty_table_chunk,
    evaluate_chunk,
    gather_in_task_group,
    process_children,
    recv_metadata,
    send_chunk,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.utils import partition_range

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer


def _index_order_key(ir: GroupByDynamic) -> OrderKey:
    """Return the ascending ordering key required by a dynamic groupby."""
    start = ir.start.value
    if not isinstance(start, DynamicWindowStart):  # pragma: no cover
        raise TypeError("GroupByDynamic requires a DynamicWindowStart expression")
    index = start.children[0]
    if not isinstance(index, Col):  # pragma: no cover
        raise NotImplementedError("group_by_dynamic requires a column index")
    (index_position,) = names_to_indices((index.name,), ir.children[0].schema)
    return OrderKey(
        index_position,
        plc.types.Order.ASCENDING,
        plc.types.NullOrder.BEFORE,
    )


def _window_ordering(
    context: Context,
    ir: GroupByDynamic,
    input_ordering: Ordering,
) -> Ordering:
    """Transform index boundaries into unique window-aligned boundaries."""
    start = ir.start.value
    assert isinstance(start, DynamicWindowStart)
    boundaries = input_ordering.get_boundaries(context.br())
    stream = boundaries.stream
    (index,) = boundaries.table_view().columns()
    boundaries = start.evaluate_partition_boundaries(
        Column(index, dtype=start.children[0].dtype), stream
    )
    non_null_boundaries = plc.stream_compaction.drop_nulls(
        plc.Table([boundaries.obj]), [0], 1, stream=stream
    )
    unique_boundaries = plc.stream_compaction.unique(
        non_null_boundaries,
        [0],
        plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
        plc.types.NullEquality.EQUAL,
        stream=stream,
    )
    return Ordering(
        input_ordering.keys,
        TableChunk.from_pylibcudf_table(
            unique_boundaries,
            stream,
            exclusive_view=True,
            br=context.br(),
        ),
        strict_boundaries=True,
        locally_ordered=input_ordering.locally_ordered,
    )


def _adjusted_metadata(
    comm: Communicator,
    metadata: ChannelMetadata,
    ordering: Ordering,
) -> ChannelMetadata:
    """Describe a channel adjusted to strict window boundaries."""
    start, stop = partition_range(
        comm.rank,
        comm.nranks,
        ordering.num_boundaries + 1,
    )
    return ChannelMetadata(
        local_count=stop - start,
        partitioning=Partitioning(OrderScheme([ordering]), "inherit"),
        duplicated=metadata.duplicated,
    )


async def _evaluate_fallback(
    context: Context,
    comm: Communicator,
    ir: GroupByDynamic,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata: ChannelMetadata,
    collective_id: int,
    tracer: ActorTracer | None,
) -> None:
    """Gather input and evaluate dynamic grouping as one operation."""
    allgather = AllGatherManager(context, comm, collective_id)
    sequence_number = 0
    with allgather.inserting() as inserter:
        while (msg := await ch_in.recv(context)) is not None:
            if not metadata.duplicated or comm.rank == 0:
                await inserter.insert(
                    sequence_number,
                    TableChunk.from_message(msg, br=context.br()),
                )
                sequence_number += 1

    stream = ir_context.get_cuda_stream()
    table = await allgather.extract_concatenated(
        stream,
        ordered=True,
        ir_context=ir_context,
    )
    chunk = (
        empty_table_chunk(ir.children[0], context, stream)
        if table.num_columns() == 0
        else TableChunk.from_pylibcudf_table(
            table,
            stream,
            exclusive_view=True,
            br=context.br(),
        )
    )
    await send_metadata(
        ch_out,
        context,
        ChannelMetadata(local_count=1, duplicated=True),
    )
    if tracer is not None:
        tracer.set_duplicated()
    result = await evaluate_chunk(context, chunk, ir, ir_context=ir_context)
    await send_chunk(context, ch_out, result, 0, tracer=tracer)
    await ch_out.drain(context)


async def _evaluate_ordered_input(
    context: Context,
    comm: Communicator,
    ir: GroupByDynamic,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    order_key: OrderKey,
    collective_id: int,
    tracer: ActorTracer | None,
) -> None:
    """Choose ordered execution or the gathered fallback."""
    metadata = await recv_metadata(ch_in, context)
    input_ordering = NormalizedPartitioning.from_keys(
        metadata.partitioning,
        comm.nranks,
        keys=(order_key,),
        allow_subset=False,
    ).get_ordering()
    start = ir.start.value
    assert isinstance(start, DynamicWindowStart)
    if (
        input_ordering is None
        or metadata.duplicated
        or start.dtype != start.children[0].dtype
    ):
        if tracer is not None:
            tracer.decision = "allgather"
        await _evaluate_fallback(
            context,
            comm,
            ir,
            ir_context,
            ch_out,
            ch_in,
            metadata,
            collective_id,
            tracer,
        )
        return

    if tracer is not None:
        tracer.decision = "ordered"
    output_ordering = _window_ordering(context, ir, input_ordering)
    adjusted_metadata = _adjusted_metadata(comm, metadata, output_ordering)
    result_ordering = output_ordering.with_keys(
        (OrderKey(0, order_key.order, order_key.null_order),)
    )
    result_metadata = _adjusted_metadata(comm, metadata, result_ordering)
    ch_adjusted = context.create_channel()
    await gather_in_task_group(
        adjust_ordering(
            context,
            comm,
            ir.children[0],
            ir_context,
            ch_adjusted,
            ch_in,
            input_ordering,
            output_ordering,
            collective_id=collective_id,
        ),
        chunkwise_evaluate(
            context,
            ir,
            ir_context,
            ch_out,
            ch_adjusted,
            result_metadata,
            input_metadata=adjusted_metadata,
            tracer=tracer,
        ),
    )


@define_actor()
async def groupby_dynamic_actor(
    context: Context,
    comm: Communicator,
    ir: GroupByDynamic,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    collective_id: int,
) -> None:
    """Evaluate a dynamic groupby using index ordering when available."""
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        ir_context = actor_scope.require_ir_context()
        ch_ordered = context.create_channel()
        order_key = _index_order_key(ir)
        await gather_in_task_group(
            ensure_ordering_metadata(
                context,
                comm,
                ir.children[0],
                ir_context,
                ch_in,
                ch_ordered,
                (order_key,),
                collective_id,
                validate=True,
            ),
            _evaluate_ordered_input(
                context,
                comm,
                ir,
                ir_context,
                ch_out,
                ch_ordered,
                order_key,
                collective_id,
                actor_scope.tracer,
            ),
        )


@generate_ir_sub_network.register(GroupByDynamic)
def _(
    ir: GroupByDynamic, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate the dynamic-groupby actor network."""
    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    (collective_id,) = rec.state["collective_id_map"][ir]
    actors[ir] = [
        groupby_dynamic_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            ir_context_for_node(rec, ir),
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            collective_id,
        )
    ]
    return actors, channels
