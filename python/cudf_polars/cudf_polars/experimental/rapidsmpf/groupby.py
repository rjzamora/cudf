# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GroupBy logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.dsl.expr import Col, NamedExpr
from cudf_polars.dsl.ir import IR, GroupBy, Select
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.experimental.groupby import combine, decompose
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.collectives.shuffle import ShuffleManager
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    allgather_reduce,
    chunkwise_evaluate,
    concat_batch,
    empty_table_chunk,
    evaluate_batch,
    evaluate_chunk,
    is_partitioned_on_keys,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.experimental.repartition import Repartition

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.tracing import ActorTracer


@dataclass
class DecomposedGroupBy:
    """Holds decomposed GroupBy operations for multi-phase aggregation."""

    ir: GroupBy
    piecewise_ir: GroupBy
    reduction_ir: GroupBy
    select_ir: Select
    need_preshuffle: bool
    grouped_keys: tuple[NamedExpr, ...]

    @classmethod
    def from_groupby(cls, ir: GroupBy) -> DecomposedGroupBy:
        """Decompose a GroupBy IR node into multi-phase operations."""
        name_generator = unique_names(ir.schema.keys())
        selection_exprs, piecewise_exprs, reduction_exprs, need_preshuffle = combine(
            *(
                decompose(agg.name, agg.value, names=name_generator)
                for agg in ir.agg_requests
            )
        )

        # Piecewise groupby schema and IR
        pwise_schema = {k.name: k.value.dtype for k in ir.keys} | {
            k.name: k.value.dtype for k in piecewise_exprs
        }
        piecewise_ir = GroupBy(
            pwise_schema,
            ir.keys,
            piecewise_exprs,
            ir.maintain_order,
            None,
            ir.children[0],
        )

        # Grouped keys for reduction and selection
        grouped_keys = tuple(
            NamedExpr(k.name, Col(k.value.dtype, k.name)) for k in ir.keys
        )

        # Reduction groupby schema and IR (must match pwise_schema for tree reduction)
        reduction_schema = {k.name: k.value.dtype for k in grouped_keys} | {
            k.name: k.value.dtype for k in reduction_exprs
        }
        assert pwise_schema == reduction_schema, (
            "piecewise and reduction schemas must match for tree reduction"
        )
        reduction_ir = GroupBy(
            reduction_schema,
            grouped_keys,
            reduction_exprs,
            ir.maintain_order,
            None,
            piecewise_ir,
        )

        # Selection IR (child is reduction_ir, not piecewise_ir)
        select_ir = Select(
            ir.schema,
            [
                *(NamedExpr(k.name, Col(k.value.dtype, k.name)) for k in grouped_keys),
                *selection_exprs,
            ],
            False,  # noqa: FBT003
            reduction_ir,
        )

        return cls(
            ir=ir,
            piecewise_ir=piecewise_ir,
            reduction_ir=reduction_ir,
            select_ir=select_ir,
            need_preshuffle=need_preshuffle,
            grouped_keys=grouped_keys,
        )


async def _tree_groupby(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    target_partition_size: int,
    *,
    evaluated_chunks: list[TableChunk] | None = None,
    collective_id: int | None = None,
    reduction_ran: bool = False,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Execute groupby using tree reduction to single output.

    Reads chunks, applies piecewise aggregation, and reduces incrementally.
    When collective_id is provided and data is not duplicated, uses allgather
    to collect partial results from all ranks before final reduction.

    Parameters
    ----------
    context
        The rapidsmpf streaming context.
    decomposed
        The decomposed groupby containing piecewise, reduction, and select IRs.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    metadata_in
        The input channel metadata.
    target_partition_size
        Target size in bytes for output partitions.
    evaluated_chunks
        Chunks that have already been evaluated (e.g., during sampling).
    collective_id
        Optional collective ID for allgather. If None, no allgather is performed.
    reduction_ran
        Whether evaluated_chunks have already been through reduction_ir.
    tracer
        Optional tracer for runtime metrics.
    """
    tree_reduction_ran = reduction_ran
    nranks = context.comm().nranks
    need_allgather = (
        collective_id is not None and not metadata_in.duplicated and nranks > 1
    )

    metadata_out = ChannelMetadata(
        local_count=1,
        partitioning=None,
        duplicated=True if need_allgather else metadata_in.duplicated,
    )
    await send_metadata(ch_out, context, metadata_out)

    pwise_chunks: list[TableChunk] = list(evaluated_chunks or [])
    total_size = sum(c.data_alloc_size() for c in pwise_chunks)

    receiving = True
    while receiving or len(pwise_chunks) > 1:
        if receiving:
            msg = await ch_in.recv(context)
            if msg is None:
                receiving = False
            else:
                chunk = await evaluate_chunk(
                    context,
                    TableChunk.from_message(msg),
                    decomposed.piecewise_ir,
                    ir_context,
                )
                del msg
                pwise_chunks.append(chunk)
                total_size += chunk.data_alloc_size()

        if len(pwise_chunks) > 1 and (
            not receiving or total_size > target_partition_size
        ):
            merged = await evaluate_batch(
                pwise_chunks, context, decomposed.reduction_ir, ir_context
            )
            pwise_chunks = [merged]
            total_size = merged.data_alloc_size()
            tree_reduction_ran = True

    chunk_schema = (
        decomposed.reduction_ir.schema
        if tree_reduction_ran
        else decomposed.piecewise_ir.schema
    )

    # Allgather partial results from all ranks if needed
    if need_allgather:
        assert collective_id is not None

        allgather = AllGatherManager(context, collective_id)
        stream = ir_context.get_cuda_stream()

        if pwise_chunks:
            if tree_reduction_ran:
                reduced_chunk = await concat_batch(
                    pwise_chunks, context, chunk_schema, ir_context
                )
            else:
                reduced_chunk = await evaluate_batch(
                    pwise_chunks, context, decomposed.reduction_ir, ir_context
                )
            del pwise_chunks
            allgather.insert(0, reduced_chunk)
        # else: No local data - don't insert anything into allgather
        # Empty table chunks can have schema mismatches (e.g., STRING columns
        # with 0 children vs 1 child), so we skip them entirely
        allgather.insert_finished()

        gathered_table = await allgather.extract_concatenated(stream)
        gathered_chunk = TableChunk.from_pylibcudf_table(
            gathered_table, stream, exclusive_view=True
        )
        pwise_chunks = [
            await evaluate_chunk(
                context, gathered_chunk, decomposed.reduction_ir, ir_context
            )
        ]
        del gathered_chunk

    # Apply final selection and send
    if pwise_chunks:
        chunk = await evaluate_chunk(
            context, pwise_chunks[0], decomposed.select_ir, ir_context
        )
        if tracer is not None:
            tracer.add_chunk(table=chunk.table_view())
        await ch_out.send(context, Message(0, chunk))
    else:
        # No data - send empty chunk
        stream = ir_context.get_cuda_stream()
        chunk = empty_table_chunk(decomposed.ir, context, stream)
        if tracer is not None:
            tracer.add_chunk()
        await ch_out.send(context, Message(0, chunk))

    await ch_out.drain(context)


async def _shuffle_groupby(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    output_count: int,
    collective_id: int,
    key_indices: tuple[int, ...],
    *,
    evaluated_chunks: list[TableChunk] | None = None,
    tracer: ActorTracer | None = None,
) -> None:
    """Execute groupby using shuffle-based redistribution."""
    nranks = context.comm().nranks
    local_output_count = max(1, output_count // nranks)

    pwise_schema_keys = list(decomposed.piecewise_ir.schema.keys())
    groupby_key_names = tuple(ne.name for ne in decomposed.grouped_keys)
    pwise_key_indices = tuple(pwise_schema_keys.index(k) for k in groupby_key_names)

    metadata_out = ChannelMetadata(
        local_count=local_output_count,
        partitioning=Partitioning(
            inter_rank=HashScheme(pwise_key_indices, output_count),
            local="inherit",
        ),
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)

    shuffle = ShuffleManager(context, output_count, pwise_key_indices, collective_id)

    for chunk in evaluated_chunks or []:
        shuffle.insert_chunk(chunk)

    while (msg := await ch_in.recv(context)) is not None:
        shuffle.insert_chunk(
            await evaluate_chunk(
                context,
                TableChunk.from_message(msg),
                decomposed.piecewise_ir,
                ir_context,
            )
        )
        del msg

    await shuffle.insert_finished()

    # Extract shuffled partitions and apply reduction + selection
    for partition_id in range(context.comm().rank, output_count, nranks):
        stream = ir_context.get_cuda_stream()
        partition_chunk = TableChunk.from_pylibcudf_table(
            await shuffle.extract_chunk(partition_id, stream),
            stream,
            exclusive_view=True,
        )
        chunk = await evaluate_chunk(
            context,
            partition_chunk,
            [decomposed.reduction_ir, decomposed.select_ir],
            ir_context,
        )
        del partition_chunk
        if tracer is not None:
            tracer.add_chunk(table=chunk.table_view())
        await ch_out.send(context, Message(partition_id, chunk))

    await ch_out.drain(context)


@define_py_node()
async def groupby_node(
    context: Context,
    ir: GroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    sample_chunk_count: int,
    target_partition_size: int,
    collective_ids: list[int],
) -> None:
    """
    Dynamic GroupBy node that selects the best strategy at runtime.

    Strategy selection based on sampled data:
    - Partition-wise: Data already partitioned on groupby keys
    - Tree reduction: Small estimated output (< target_partition_size)
    - Shuffle: Large estimated output requiring redistribution

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The IR node to evaluate.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    sample_chunk_count
        The number of chunks to sample.
    target_partition_size
        The target partition size.
    collective_ids
        The collective IDs.
    """
    async with shutdown_on_error(context, ch_in, ch_out, trace_ir=ir) as tracer:
        # Receive input metadata
        metadata_in = await recv_metadata(ch_in, context)

        # Get groupby key column indices
        input_schema_keys = list(ir.children[0].schema.keys())
        groupby_key_names = tuple(ne.name for ne in ir.keys)
        key_indices = tuple(
            input_schema_keys.index(k)
            for k in groupby_key_names
            if k in input_schema_keys
        )

        # Check if already partitioned on keys
        nranks = context.comm().nranks
        already_partitioned_inter_rank, already_partitioned_local = (
            is_partitioned_on_keys(metadata_in, key_indices, nranks)
        )
        fully_partitioned = already_partitioned_inter_rank and already_partitioned_local
        fallback_case = (
            metadata_in.local_count == 1
            and (metadata_in.duplicated or nranks == 1)
            and isinstance(ir.children[0], Repartition)
        )

        # If already partitioned or concatenated, just do a chunk-wise groupby
        if fully_partitioned or fallback_case:
            await chunkwise_evaluate(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                tracer=tracer,
            )
            return

        # Decompose for multi-phase execution
        # Note: Lowering guarantees decomposition succeeds and preshuffle is done
        decomposed = DecomposedGroupBy.from_groupby(ir)

        # Determine if we can skip global communication
        # Yes if: single rank, OR data is duplicated, OR already globally partitioned
        can_skip_global_comm = (
            nranks == 1 or metadata_in.duplicated or already_partitioned_inter_rank
        )

        # Check for ordering requirements (shuffle doesn't preserve order)
        require_tree = ir.maintain_order

        evaluated_chunks: list[TableChunk] = []
        total_size = 0
        merge_count = 0

        for _ in range(sample_chunk_count):
            msg = await ch_in.recv(context)
            if msg is None:
                break
            chunk = await evaluate_chunk(
                context,
                TableChunk.from_message(msg),
                decomposed.piecewise_ir,
                ir_context,
            )
            del msg
            total_size += chunk.data_alloc_size(MemoryType.DEVICE)
            evaluated_chunks.append(chunk)

            if total_size > target_partition_size and len(evaluated_chunks) > 1:
                merged = await evaluate_batch(
                    evaluated_chunks, context, decomposed.reduction_ir, ir_context
                )
                total_size = merged.data_alloc_size(MemoryType.DEVICE)
                evaluated_chunks = [merged]
                merge_count += 1
                if total_size > target_partition_size:
                    break

        local_count = metadata_in.local_count
        if collective_ids and nranks > 1:
            global_size, global_chunk_count = await allgather_reduce(
                context, collective_ids.pop(), total_size, local_count
            )
        else:
            global_size = total_size
            global_chunk_count = local_count

        use_tree = global_size < target_partition_size or require_tree

        if already_partitioned_inter_rank:
            if tracer is not None:
                tracer.decision = "tree_local"
            await _tree_groupby(
                context,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                target_partition_size,
                evaluated_chunks=evaluated_chunks,
                reduction_ran=merge_count > 0,
                tracer=tracer,
            )
        elif can_skip_global_comm:
            if use_tree:
                if tracer is not None:
                    tracer.decision = "tree_local"
                await _tree_groupby(
                    context,
                    decomposed,
                    ir_context,
                    ch_out,
                    ch_in,
                    metadata_in,
                    target_partition_size,
                    evaluated_chunks=evaluated_chunks,
                    reduction_ran=merge_count > 0,
                    tracer=tracer,
                )
            else:
                if tracer is not None:
                    tracer.decision = "shuffle_local"
                ideal_count = max(1, global_size // target_partition_size)
                output_count = max(1, min(ideal_count, local_count))
                await _shuffle_groupby(
                    context,
                    decomposed,
                    ir_context,
                    ch_out,
                    ch_in,
                    metadata_in,
                    output_count,
                    collective_ids.pop() if collective_ids else 0,
                    key_indices,
                    evaluated_chunks=evaluated_chunks,
                    tracer=tracer,
                )
        elif use_tree:
            if tracer is not None:
                tracer.decision = "tree_allgather"
            await _tree_groupby(
                context,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                target_partition_size,
                evaluated_chunks=evaluated_chunks,
                collective_id=collective_ids.pop() if collective_ids else None,
                reduction_ran=merge_count > 0,
                tracer=tracer,
            )
        elif not collective_ids:
            if tracer is not None:
                tracer.decision = "tree_fallback"
            await _tree_groupby(
                context,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                target_partition_size,
                evaluated_chunks=evaluated_chunks,
                reduction_ran=merge_count > 0,
                tracer=tracer,
            )
        else:
            if tracer is not None:
                tracer.decision = "shuffle"
            ideal_count = max(1, global_size // target_partition_size)
            output_count = max(nranks, min(ideal_count, global_chunk_count))
            await _shuffle_groupby(
                context,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                output_count,
                collective_ids.pop(),
                key_indices,
                evaluated_chunks=evaluated_chunks,
                tracer=tracer,
            )


@generate_ir_sub_network.register(GroupBy)
def _(
    ir: GroupBy, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for GroupBy operation."""
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming"

    # Only use the dynamic groupby node when dynamic planning is enabled
    if config_options.executor.dynamic_planning is None:
        # Fall back to the default IR handler (bypass GroupBy dispatch)
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    # For type narrowing after the early return
    dynamic_planning = config_options.executor.dynamic_planning

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Get collective IDs for this GroupBy (may be empty if not reserved)
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))

    # Create the dynamic groupby node
    nodes[ir] = [
        groupby_node(
            rec.state["context"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            dynamic_planning.sample_chunk_count_groupby,
            config_options.executor.target_partition_size,
            collective_ids,
        )
    ]

    return nodes, channels
