# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GroupByDynamic logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

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

from cudf_polars.dsl.expressions.dynamic import label_dynamic_window
from cudf_polars.dsl.ir import IR, GroupByDynamic
from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.ordering import adjust_ordering
from cudf_polars.streaming.actor_graph.collectives.sort import (
    extract_orderscheme_partitioning,
)
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.groupby import _partition_count_for_rank
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    NormalizedPartitioning,
    chunkwise_evaluate,
    gather_in_task_group,
    process_children,
    recv_metadata,
    shutdown_on_error,
)
from cudf_polars.utils.cuda_stream import stream_ordered_after
from cudf_polars.utils.dtypes import make_empty_column

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.utils import ChunkStore
    from cudf_polars.utils.config import StreamingExecutor


def _index_order_key(ir: GroupByDynamic, schema: dict[str, Any]) -> OrderKey:
    (index,) = names_to_indices((ir.index_name,), schema)
    return OrderKey(index, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)


def _bucket_boundary_chunk(
    context: Context,
    ir: GroupByDynamic,
    input_ordering: Ordering,
    stream: Stream,
) -> TableChunk:
    """Return unique bucket-edge boundaries in the input index domain."""
    boundaries = input_ordering.get_boundaries(context.br())
    with stream_ordered_after(lambda: stream, upstreams=(boundaries.stream,)):
        boundary_table = boundaries.table_view()
        if boundary_table.num_rows() == 0:
            unique_boundaries = plc.Table(
                [
                    plc.Column.from_iterable_of_py(
                        [], ir.index_dtype.plc_type, stream=stream
                    )
                ]
            )
        else:
            (index_boundaries,) = boundary_table.columns()
            labels = label_dynamic_window(
                index_boundaries,
                ir.index_dtype,
                ir.every,
                ir.offset,
                stream,
            )
            unique_boundaries = plc.stream_compaction.unique(
                plc.Table([labels]),
                [0],
                plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
                plc.types.NullEquality.EQUAL,
                stream=stream,
            )
        return TableChunk.from_pylibcudf_table(
            unique_boundaries,
            stream,
            exclusive_view=True,
            br=context.br(),
        )


def _input_bucket_ordering(
    context: Context,
    ir: GroupByDynamic,
    input_ordering: Ordering,
    stream: Stream,
) -> Ordering:
    """Return strict bucket-edge ordering over the input index column."""
    return Ordering(
        input_ordering.keys,
        _bucket_boundary_chunk(context, ir, input_ordering, stream),
        strict_boundaries=True,
        locally_ordered=True,
    )


def _metadata_for_ordering(
    comm: Communicator,
    metadata_in: ChannelMetadata,
    ordering: Ordering,
    *,
    result_ordering: Ordering | None,
) -> ChannelMetadata:
    """Build metadata for data partitioned by ``ordering``."""
    return ChannelMetadata(
        local_count=_partition_count_for_rank(
            comm.rank, comm.nranks, ordering.num_boundaries + 1
        ),
        partitioning=(
            None
            if result_ordering is None
            else Partitioning(OrderScheme([result_ordering]), "inherit")
        ),
        duplicated=metadata_in.duplicated,
    )


def _single_partition_input_ordering(
    context: Context,
    ir: GroupByDynamic,
    order_key: OrderKey,
    stream: Stream,
) -> Ordering:
    """Return trivial ordering metadata for a one-partition input."""
    boundaries = TableChunk.from_pylibcudf_table(
        plc.Table([make_empty_column(ir.index_dtype, stream)]),
        stream,
        exclusive_view=False,
        br=context.br(),
    )
    return Ordering([order_key], boundaries, strict_boundaries=True)


def _result_ordering(ir: GroupByDynamic, ordering: Ordering) -> Ordering:
    """Return the output ordering over the emitted dynamic index column."""
    output_key = _index_order_key(ir, ir.schema)
    return ordering.with_keys([output_key]).with_locally_ordered(
        locally_ordered=ir.preserves_output_order
    )


async def _send_stored_chunks(
    context: Context,
    ch_out: Channel[TableChunk],
    chunks: ChunkStore,
) -> None:
    """Replay stored chunks into ``ch_out``."""
    try:
        for msg in chunks:
            await ch_out.send(context, msg)
        await ch_out.drain(context)
    finally:
        chunks.clear()


def _get_input_ordering(
    metadata: ChannelMetadata,
    comm: Communicator,
    order_key: OrderKey,
) -> Ordering | None:
    partitioning = NormalizedPartitioning.from_keys(
        metadata.partitioning,
        comm.nranks,
        keys=[order_key],
    )
    return partitioning.get_ordering(level="local" if metadata.duplicated else "flat")


async def _extract_input_ordering(
    context: Context,
    comm: Communicator,
    ir: GroupByDynamic,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    order_key: OrderKey,
    collective_id: int,
) -> tuple[ChannelMetadata, Ordering, ChunkStore]:
    """Extract missing input ordering metadata and retain consumed chunks."""
    result = await extract_orderscheme_partitioning(
        context,
        comm,
        ir.children[0],
        ir_context,
        ch_in,
        [order_key],
        collective_id,
    )
    if result.partitioning is None:
        if metadata_in.local_count <= 1 and comm.nranks == 1:
            stream = ir_context.get_cuda_stream()
            ordering = _single_partition_input_ordering(context, ir, order_key, stream)
            metadata = ChannelMetadata(
                local_count=metadata_in.local_count,
                partitioning=Partitioning(OrderScheme([ordering]), "inherit"),
                duplicated=metadata_in.duplicated,
            )
            return metadata, ordering, result.chunks
        raise pl.exceptions.InvalidOperationError(
            "group_by_dynamic requires input sorted by the dynamic index column"
        )
    metadata = ChannelMetadata(
        local_count=metadata_in.local_count,
        partitioning=result.partitioning,
        duplicated=metadata_in.duplicated,
    )
    ordering = _get_input_ordering(metadata, comm, order_key)
    if ordering is None:  # pragma: no cover
        raise RuntimeError("failed to recover extracted dynamic-groupby ordering")
    return metadata, ordering, result.chunks


@define_actor()
async def groupby_dynamic_actor(
    context: Context,
    comm: Communicator,
    ir: GroupByDynamic,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    executor: StreamingExecutor,
    collective_ids: list[int],
) -> None:
    """Streaming fixed-width dynamic groupby actor."""
    ch_replay = context.create_channel()
    ch_aligned = context.create_channel()
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        chs_aux=(ch_replay, ch_aligned),
        trace_ir=ir,
        ir_context=ir_context,
    ) as tracer:
        if executor.dynamic_planning is None:
            raise ValueError("GroupByDynamic requires dynamic planning")

        metadata_in = await recv_metadata(ch_in, context)
        order_key = _index_order_key(ir, ir.children[0].schema)
        input_ordering = _get_input_ordering(metadata_in, comm, order_key)
        extracted_chunks: ChunkStore | None = None
        if input_ordering is None:
            (
                metadata_in,
                input_ordering,
                extracted_chunks,
            ) = await _extract_input_ordering(
                context,
                comm,
                ir,
                ir_context,
                ch_in,
                metadata_in,
                order_key,
                collective_ids.pop(),
            )

        stream = ir_context.get_cuda_stream()
        aligned_input_ordering = _input_bucket_ordering(
            context, ir, input_ordering, stream
        )
        input_aligned = (
            input_ordering.locally_ordered
            and input_ordering.boundaries_aligned_with(
                aligned_input_ordering, context.br()
            )
        )
        result_ordering = _result_ordering(ir, aligned_input_ordering)
        metadata_out = _metadata_for_ordering(
            comm,
            metadata_in,
            aligned_input_ordering,
            result_ordering=result_ordering,
        )

        if input_aligned:
            if tracer is not None:
                tracer.decision = "already_aligned"
            if extracted_chunks is None:
                await chunkwise_evaluate(
                    context,
                    ir,
                    ir_context,
                    ch_out,
                    ch_in,
                    metadata_out,
                    input_metadata=metadata_in,
                    tracer=tracer,
                )
            else:
                await gather_in_task_group(
                    _send_stored_chunks(context, ch_replay, extracted_chunks),
                    chunkwise_evaluate(
                        context,
                        ir,
                        ir_context,
                        ch_out,
                        ch_replay,
                        metadata_out,
                        input_metadata=metadata_in,
                        tracer=tracer,
                    ),
                )
            return

        if tracer is not None:
            tracer.decision = "adjust_ordering"
        input_channel = ch_in if extracted_chunks is None else ch_replay
        aligned_metadata = _metadata_for_ordering(
            comm,
            metadata_in,
            aligned_input_ordering,
            result_ordering=aligned_input_ordering,
        )
        tasks = []
        if extracted_chunks is not None:
            tasks.append(_send_stored_chunks(context, ch_replay, extracted_chunks))
        tasks.extend(
            [
                adjust_ordering(
                    context,
                    comm,
                    ir.children[0],
                    ir_context,
                    ch_aligned,
                    input_channel,
                    input_ordering,
                    aligned_input_ordering,
                    collective_id=collective_ids.pop(),
                ),
                chunkwise_evaluate(
                    context,
                    ir,
                    ir_context,
                    ch_out,
                    ch_aligned,
                    metadata_out,
                    input_metadata=aligned_metadata,
                    tracer=tracer,
                ),
            ]
        )
        await gather_in_task_group(*tasks)


@generate_ir_sub_network.register(GroupByDynamic)
def _(
    ir: GroupByDynamic, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for fixed-width dynamic groupby."""
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming"

    if config_options.executor.dynamic_planning is None:
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))
    ir_context = ir_context_for_node(rec, ir)
    assert len(collective_ids) == 2, (
        f"{type(ir).__name__} requires 2 collective IDs, got {len(collective_ids)}"
    )
    actors[ir] = [
        groupby_dynamic_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            ir_context,
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            config_options.executor,
            collective_ids,
        )
    ]
    return actors, channels
