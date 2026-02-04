# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GroupBy logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
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

from cudf_polars.containers import DataFrame
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
    opaque_reservation,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.tracing import ActorTracer

# ============================================================================
# Helper Functions
# ============================================================================


def _is_partitioned_on_keys(
    metadata: ChannelMetadata,
    key_indices: tuple[int, ...],
) -> bool:
    """Check if data is already partitioned on the groupby keys."""
    if metadata.partitioning is None:
        return False
    inter_rank = metadata.partitioning.inter_rank
    if inter_rank is None or inter_rank == "inherit":
        return False
    return set(inter_rank.column_indices) == set(key_indices)


def _apply_do_evaluate(
    chunk: TableChunk,
    ir: GroupBy | Select,
    ir_context: IRExecutionContext,
    *,
    input_schema: dict[str, Any] | None = None,
) -> TableChunk:
    """Apply GroupBy or Select evaluation to a chunk."""
    if input_schema is None:
        input_schema = ir.children[0].schema
    names = list(input_schema.keys())
    dtypes = list(input_schema.values())
    df = ir.do_evaluate(
        *ir._non_child_args,
        DataFrame.from_table(chunk.table_view(), names, dtypes, chunk.stream),
        context=ir_context,
    )
    return TableChunk.from_pylibcudf_table(df.table, chunk.stream, exclusive_view=True)


# ============================================================================
# Decomposed GroupBy State
# ============================================================================


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

        # Reduction groupby schema and IR
        reduction_schema = {k.name: k.value.dtype for k in grouped_keys} | {
            k.name: k.value.dtype for k in reduction_exprs
        }
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


# ============================================================================
# GroupBy Strategies
# ============================================================================


async def _partitionwise_groupby(
    context: Context,
    ir: GroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    initial_chunks: list[TableChunk],
    tracer: ActorTracer | None = None,
) -> None:
    """Execute partition-wise groupby (data already partitioned on keys)."""
    # Send output metadata preserving partitioning
    await send_metadata(ch_out, context, metadata_in)

    # Process initial chunks
    for seq_num, chunk in enumerate(initial_chunks):
        with opaque_reservation(context, chunk.data_alloc_size(MemoryType.DEVICE)):
            result = await asyncio.to_thread(_apply_do_evaluate, chunk, ir, ir_context)
            if tracer is not None:
                tracer.add_chunk(table=result.table_view())
            await ch_out.send(context, Message(seq_num, result))
            del chunk, result

    # Process remaining chunks
    seq_num = len(initial_chunks)
    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        del msg
        with opaque_reservation(context, chunk.data_alloc_size(MemoryType.DEVICE)):
            result = await asyncio.to_thread(_apply_do_evaluate, chunk, ir, ir_context)
            if tracer is not None:
                tracer.add_chunk(table=result.table_view())
            await ch_out.send(context, Message(seq_num, result))
            del chunk, result
        seq_num += 1

    await ch_out.drain(context)


async def _concat_groupby(
    context: Context,
    ir: GroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    initial_chunks: list[TableChunk],
    collective_id: int | None = None,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Execute groupby by concatenating all data first.

    Used when outputting a single partition, maintaining order, or when
    aggregation cannot be decomposed for piecewise processing.

    When collective_id is provided and data is not duplicated, uses allgather
    to collect data from all ranks first, producing a single duplicated chunk.
    """
    nranks = context.comm().nranks

    # Determine if allgather is needed
    need_allgather = (
        collective_id is not None and not metadata_in.duplicated and nranks > 1
    )

    # Set output metadata
    if need_allgather:
        # After allgather, all workers have identical data
        metadata_out = ChannelMetadata(local_count=1, duplicated=True)
    else:
        # Local concat: single chunk per rank
        metadata_out = ChannelMetadata(local_count=1, duplicated=metadata_in.duplicated)
    await send_metadata(ch_out, context, metadata_out)

    # Collect chunks
    input_bytes = 0
    chunks: list[TableChunk] = list(initial_chunks)
    for chunk in chunks:
        input_bytes += chunk.data_alloc_size(MemoryType.DEVICE)

    if need_allgather:
        assert collective_id is not None
        allgather = AllGatherManager(context, collective_id)

        # Insert initial chunks
        for seq_num, chunk in enumerate(chunks):
            allgather.insert(seq_num, chunk)

        # Insert remaining chunks from channel
        seq_num = len(chunks)
        while (msg := await ch_in.recv(context)) is not None:
            allgather.insert(seq_num, TableChunk.from_message(msg))
            del msg
            seq_num += 1

        allgather.insert_finished()
        stream = ir_context.get_cuda_stream()
        chunks = [
            TableChunk.from_pylibcudf_table(
                await allgather.extract_concatenated(stream),
                stream,
                exclusive_view=True,
            )
        ]
        input_bytes = chunks[0].data_alloc_size(MemoryType.DEVICE)
    else:
        # Local collection only
        while (msg := await ch_in.recv(context)) is not None:
            chunk = TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )
            del msg
            chunks.append(chunk)
            input_bytes += chunk.data_alloc_size(MemoryType.DEVICE)

    if chunks:
        # Reserve extra for groupby working memory (input + output)
        with opaque_reservation(context, input_bytes * 2):
            multi_chunks = len(chunks) > 1
            input_schema = ir.children[0].schema

            # Concatenate chunks
            concatenated = await asyncio.to_thread(
                _concat,
                *[
                    DataFrame.from_table(
                        chunk.table_view(),
                        list(input_schema.keys()),
                        list(input_schema.values()),
                        chunk.stream,
                    )
                    for chunk in chunks
                ],
                context=ir_context,
            )
            if multi_chunks:
                del chunks

            # Apply full groupby
            df = await asyncio.to_thread(
                ir.do_evaluate, *ir._non_child_args, concatenated, context=ir_context
            )
            del concatenated
            if tracer is not None:
                tracer.add_chunk(table=df.table)
            await ch_out.send(
                context,
                Message(
                    0,
                    TableChunk.from_pylibcudf_table(
                        df.table, df.stream, exclusive_view=True
                    ),
                ),
            )
            del df
            if not multi_chunks:
                del chunks

    await ch_out.drain(context)


async def _tree_groupby(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    initial_chunks: list[TableChunk],
    groupby_n_ary: int,
    collective_id: int | None = None,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Execute groupby using N-ary tree reduction to single output.

    When collective_id is provided and data is not duplicated, uses allgather
    to collect partial results from all ranks before final reduction.
    """
    nranks = context.comm().nranks
    need_allgather = (
        collective_id is not None and not metadata_in.duplicated and nranks > 1
    )

    # Output: single chunk, duplicated if allgather is used
    metadata_out = ChannelMetadata(
        local_count=1,
        partitioning=None,
        duplicated=True if need_allgather else metadata_in.duplicated,
    )
    await send_metadata(ch_out, context, metadata_out)

    # Collect all chunks, applying piecewise groupby
    # Note: initial_chunks already have piecewise applied (done during sampling)
    pwise_chunks: list[TableChunk] = list(initial_chunks)

    # Apply piecewise to remaining chunks from input channel
    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        del msg
        with opaque_reservation(context, chunk.data_alloc_size(MemoryType.DEVICE)):
            pwise_chunk = await asyncio.to_thread(
                _apply_do_evaluate, chunk, decomposed.piecewise_ir, ir_context
            )
            pwise_chunks.append(pwise_chunk)
            del chunk

    # Tree reduction
    # After first reduction pass, chunks are in reduction_ir schema (not piecewise_ir)
    k = groupby_n_ary
    # Track current schema - starts as piecewise, becomes reduction after first pass
    chunk_schema = decomposed.piecewise_ir.schema
    tree_reduction_ran = False

    while len(pwise_chunks) > 1:
        new_chunks: list[TableChunk] = []
        for i in range(0, len(pwise_chunks), k):
            batch = pwise_chunks[i : i + k]
            # Concatenate and reduce (even single-chunk batches need reduction
            # to convert from piecewise to reduction schema on first pass)
            input_bytes = sum(c.data_alloc_size(MemoryType.DEVICE) for c in batch)
            # Reserve extra for groupby working memory (input + output)
            with opaque_reservation(context, input_bytes * 2):
                concatenated = await asyncio.to_thread(
                    _concat,
                    *[
                        DataFrame.from_table(
                            c.table_view(),
                            list(chunk_schema.keys()),
                            list(chunk_schema.values()),
                            c.stream,
                        )
                        for c in batch
                    ],
                    context=ir_context,
                )
                del batch
                df = await asyncio.to_thread(
                    decomposed.reduction_ir.do_evaluate,
                    *decomposed.reduction_ir._non_child_args,
                    concatenated,
                    context=ir_context,
                )
                del concatenated
                new_chunks.append(
                    TableChunk.from_pylibcudf_table(
                        df.table, df.stream, exclusive_view=True
                    )
                )
                del df
        pwise_chunks = new_chunks
        # After first reduction, output is in reduction_ir schema
        chunk_schema = decomposed.reduction_ir.schema
        tree_reduction_ran = True

    # Allgather partial results from all ranks if needed
    if need_allgather:
        assert collective_id is not None

        allgather = AllGatherManager(context, collective_id)
        stream = ir_context.get_cuda_stream()

        # Before allgather, ensure we have exactly one chunk in reduction_ir schema
        reduction_schema = decomposed.reduction_ir.schema
        if pwise_chunks:
            # Construct DataFrames with correct schema based on whether tree reduction ran
            input_bytes = sum(
                c.data_alloc_size(MemoryType.DEVICE) for c in pwise_chunks
            )
            # Reserve extra for groupby working memory (3x for safety)
            with opaque_reservation(context, input_bytes * 3):
                concatenated = await asyncio.to_thread(
                    _concat,
                    *[
                        DataFrame.from_table(
                            c.table_view(),
                            list(chunk_schema.keys()),
                            list(chunk_schema.values()),
                            c.stream,
                        )
                        for c in pwise_chunks
                    ],
                    context=ir_context,
                )
                del pwise_chunks
                if not tree_reduction_ran:
                    # Chunks still in piecewise schema - apply reduction
                    df = await asyncio.to_thread(
                        decomposed.reduction_ir.do_evaluate,
                        *decomposed.reduction_ir._non_child_args,
                        concatenated,
                        context=ir_context,
                    )
                    del concatenated
                else:
                    # Tree reduction already applied - chunks in reduction schema
                    df = concatenated
                reduced_chunk = TableChunk.from_pylibcudf_table(
                    df.table, df.stream, exclusive_view=True
                )
                del df
            allgather.insert(0, reduced_chunk)
        # else: No local data - don't insert anything into allgather
        # Empty table chunks can have schema mismatches (e.g., STRING columns
        # with 0 children vs 1 child), so we skip them entirely
        allgather.insert_finished()

        # Extract concatenated results from all ranks
        gathered_table = await allgather.extract_concatenated(stream)
        pwise_chunks = [
            TableChunk.from_pylibcudf_table(gathered_table, stream, exclusive_view=True)
        ]

        # One more reduction round to merge results from all ranks
        chunk = pwise_chunks[0]
        input_bytes = chunk.data_alloc_size(MemoryType.DEVICE)
        # Reserve extra for groupby working memory (input + output)
        with opaque_reservation(context, input_bytes * 2):
            reduction_schema = decomposed.reduction_ir.schema
            concatenated = DataFrame.from_table(
                chunk.table_view(),
                list(reduction_schema.keys()),
                list(reduction_schema.values()),
                chunk.stream,
            )
            del chunk
            df = await asyncio.to_thread(
                decomposed.reduction_ir.do_evaluate,
                *decomposed.reduction_ir._non_child_args,
                concatenated,
                context=ir_context,
            )
            del concatenated
            pwise_chunks = [
                TableChunk.from_pylibcudf_table(
                    df.table, df.stream, exclusive_view=True
                )
            ]
            del df

    # Apply final selection and send
    if pwise_chunks:
        chunk = pwise_chunks[0]
        with opaque_reservation(context, chunk.data_alloc_size(MemoryType.DEVICE)):
            chunk = await asyncio.to_thread(
                _apply_do_evaluate,
                chunk,
                decomposed.select_ir,
                ir_context,
                input_schema=decomposed.reduction_ir.schema,
            )
            if tracer is not None:
                tracer.add_chunk(table=chunk.table_view())
            await ch_out.send(context, Message(0, chunk))
            del chunk
    else:
        # No data - send empty chunk
        from cudf_polars.experimental.rapidsmpf.utils import empty_table_chunk

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
    initial_chunks: list[TableChunk],
    output_count: int,
    collective_id: int,
    key_indices: tuple[int, ...],
    tracer: ActorTracer | None = None,
) -> None:
    """Execute groupby using shuffle-based redistribution."""
    nranks = context.comm().nranks
    local_output_count = max(1, output_count // nranks)

    # Output metadata with hash partitioning
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

    # Create shuffle manager
    shuffle = ShuffleManager(context, output_count, pwise_key_indices, collective_id)

    # Insert initial chunks (already have piecewise applied from sampling)
    for chunk in initial_chunks:
        shuffle.insert_chunk(chunk)

    # Process remaining chunks (need piecewise applied)
    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        del msg
        with opaque_reservation(context, chunk.data_alloc_size(MemoryType.DEVICE)):
            pwise_chunk = await asyncio.to_thread(
                _apply_do_evaluate, chunk, decomposed.piecewise_ir, ir_context
            )
            shuffle.insert_chunk(pwise_chunk)
            del pwise_chunk, chunk

    await shuffle.insert_finished()

    # Extract shuffled partitions and apply reduction + selection
    for partition_id in range(context.comm().rank, output_count, nranks):
        stream = ir_context.get_cuda_stream()
        chunk = TableChunk.from_pylibcudf_table(
            await shuffle.extract_chunk(partition_id, stream),
            stream,
            exclusive_view=True,
        )

        # Reserve extra for groupby working memory (input + output)
        with opaque_reservation(context, chunk.data_alloc_size(MemoryType.DEVICE) * 2):
            # Apply reduction
            chunk = await asyncio.to_thread(
                _apply_do_evaluate,
                chunk,
                decomposed.reduction_ir,
                ir_context,
            )
            # Apply selection
            chunk = await asyncio.to_thread(
                _apply_do_evaluate,
                chunk,
                decomposed.select_ir,
                ir_context,
                input_schema=decomposed.reduction_ir.schema,
            )
            if tracer is not None:
                tracer.add_chunk(table=chunk.table_view())
            await ch_out.send(context, Message(partition_id, chunk))
            del chunk

    await ch_out.drain(context)


async def _shuffle_full_groupby(
    context: Context,
    ir: GroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    initial_chunks: list[TableChunk],
    output_count: int,
    collective_id: int,
    key_indices: tuple[int, ...],
    tracer: ActorTracer | None = None,
) -> None:
    """
    Execute non-decomposable groupby using shuffle-based redistribution.

    Unlike _shuffle_groupby, this doesn't use piecewise aggregation.
    It shuffles raw data by keys, then applies the full groupby.
    """
    nranks = context.comm().nranks
    local_output_count = max(1, output_count // nranks)

    # Output metadata with hash partitioning on groupby keys
    metadata_out = ChannelMetadata(
        local_count=local_output_count,
        partitioning=Partitioning(
            inter_rank=HashScheme(key_indices, output_count),
            local="inherit",
        ),
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)

    # Create shuffle manager using input schema key indices
    shuffle = ShuffleManager(context, output_count, key_indices, collective_id)

    # Insert initial chunks (raw, not piecewise processed)
    for chunk in initial_chunks:
        shuffle.insert_chunk(chunk)

    # Process remaining chunks
    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        del msg
        shuffle.insert_chunk(chunk)

    await shuffle.insert_finished()

    # Extract shuffled partitions and apply full groupby
    for partition_id in range(context.comm().rank, output_count, nranks):
        stream = ir_context.get_cuda_stream()
        chunk = TableChunk.from_pylibcudf_table(
            await shuffle.extract_chunk(partition_id, stream),
            stream,
            exclusive_view=True,
        )

        # Reserve extra for groupby working memory (input + output)
        with opaque_reservation(context, chunk.data_alloc_size(MemoryType.DEVICE) * 2):
            # Apply full groupby (including zlice if present)
            chunk = await asyncio.to_thread(
                _apply_do_evaluate,
                chunk,
                ir,
                ir_context,
            )
            if tracer is not None:
                tracer.add_chunk(table=chunk.table_view())
            await ch_out.send(context, Message(partition_id, chunk))
            del chunk

    await ch_out.drain(context)


# ============================================================================
# Main GroupBy Node
# ============================================================================


@define_py_node()
async def groupby_node(
    context: Context,
    ir: GroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    sample_chunk_count: int,
    target_partition_size: int,
    groupby_n_ary: int,
    collective_ids: list[int],
) -> None:
    """
    Dynamic GroupBy node that selects the best strategy at runtime.

    Strategy selection based on sampled data:
    - Partition-wise: Data already partitioned on groupby keys
    - Tree reduction: Small estimated output (< target_partition_size)
    - Shuffle: Large estimated output requiring redistribution
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
        already_partitioned = _is_partitioned_on_keys(metadata_in, key_indices)

        # Try to decompose for multi-phase execution
        # need_preconcat is True when:
        # - maintain_order is set (must preserve row ordering)
        # - aggregations cannot be decomposed
        # Note: zlice is handled at lowering time by extracting it to a Slice node
        decomposed: DecomposedGroupBy | None = None
        need_preconcat = ir.maintain_order
        need_preshuffle = False
        try:
            decomposed = DecomposedGroupBy.from_groupby(ir)
            need_preshuffle = decomposed.need_preshuffle
        except NotImplementedError:
            need_preconcat = True  # Cannot decompose

        nranks = context.comm().nranks

        # Determine if we can skip global communication
        # Yes if: single rank, OR data is duplicated, OR already globally partitioned
        can_skip_global_comm = (
            nranks == 1 or metadata_in.duplicated or already_partitioned
        )

        # =====================================================================
        # Handle need_preconcat case (maintain_order or non-decomposable)
        # =====================================================================
        if need_preconcat:
            # Collect all initial chunks without transformation
            initial_chunks: list[TableChunk] = []
            for _ in range(sample_chunk_count):
                msg = await ch_in.recv(context)
                if msg is None:
                    break
                chunk = TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
                del msg
                initial_chunks.append(chunk)

            # Use concat_groupby strategy
            if tracer is not None:
                tracer.decision = "concat"
            collective_id = collective_ids.pop() if collective_ids else None
            await _concat_groupby(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                initial_chunks,
                collective_id if not can_skip_global_comm else None,
                tracer,
            )
            return

        # From here on, decomposed is not None
        assert decomposed is not None

        # =====================================================================
        # Handle need_preshuffle case (e.g., n_unique)
        # =====================================================================
        # For n_unique etc., we need to shuffle by keys BEFORE piecewise
        # This ensures all instances of the same key are together
        # TODO: We can also do a pre-concat if the data is small enough
        if (
            need_preshuffle
            and not already_partitioned
            and collective_ids
            and not can_skip_global_comm
        ):
            # Collect initial chunks (no piecewise yet)
            initial_chunks = []
            local_size = 0
            for _ in range(sample_chunk_count):
                msg = await ch_in.recv(context)
                if msg is None:
                    break
                chunk = TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
                del msg
                initial_chunks.append(chunk)
                local_size += chunk.data_alloc_size(MemoryType.DEVICE)

            # Estimate total size and chunk count across all ranks
            local_chunk_count = metadata_in.local_count
            if initial_chunks:
                avg_chunk_size = local_size / len(initial_chunks)
                local_estimate = int(avg_chunk_size * local_chunk_count)
            else:
                local_estimate = 0

            if nranks > 1:
                estimated_total_size, global_chunk_count = await allgather_reduce(
                    context, collective_ids.pop(), local_estimate, local_chunk_count
                )
            else:
                estimated_total_size = local_estimate
                global_chunk_count = local_chunk_count

            # Use shuffle_full_groupby (shuffles raw data, then full groupby)
            if tracer is not None:
                tracer.decision = "shuffle_full"
            ideal_count = max(1, estimated_total_size // target_partition_size)
            # Cap at global chunk count, but use the rank count if it's larger
            output_count = max(nranks, min(ideal_count, global_chunk_count))
            await _shuffle_full_groupby(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                initial_chunks,
                output_count,
                collective_ids.pop(),
                key_indices,
                tracer,
            )
            return

        # =====================================================================
        # Standard decomposed path: sample, estimate size, select strategy
        # =====================================================================
        initial_chunks = []
        total_pwise_size = 0

        for _ in range(sample_chunk_count):
            msg = await ch_in.recv(context)
            if msg is None:
                break
            chunk = TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )
            del msg

            with opaque_reservation(context, chunk.data_alloc_size(MemoryType.DEVICE)):
                pwise_chunk = await asyncio.to_thread(
                    _apply_do_evaluate, chunk, decomposed.piecewise_ir, ir_context
                )
                total_pwise_size += pwise_chunk.data_alloc_size(MemoryType.DEVICE)
                initial_chunks.append(pwise_chunk)
                del chunk

        # Estimate total size: avg_sample_size * local_count, summed across ranks
        local_count = metadata_in.local_count
        if initial_chunks and total_pwise_size > 0:
            avg_sample_size = total_pwise_size / len(initial_chunks)
            local_estimate = int(avg_sample_size * local_count)
            # Adaptive n-ary: how many chunks can fit in target_partition_size?
            # Bounded between 2 (minimum progress) and 256 (reasonable upper limit)
            chunks_per_partition = max(1, target_partition_size // avg_sample_size)
            adaptive_n_ary = max(2, min(256, int(chunks_per_partition)))
        else:
            local_estimate = 0
            adaptive_n_ary = groupby_n_ary  # fallback to configured value

        if collective_ids and nranks > 1:
            estimated_total_size, global_chunk_count = await allgather_reduce(
                context, collective_ids.pop(), local_estimate, local_count
            )
        else:
            estimated_total_size = local_estimate
            global_chunk_count = local_count

        # =====================================================================
        # Strategy Selection
        # =====================================================================

        if already_partitioned:
            # Already partitioned on groupby keys - use tree reduction (no allgather)
            if tracer is not None:
                tracer.decision = "tree_local"
            await _tree_groupby(
                context,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                initial_chunks,
                adaptive_n_ary,
                tracer=tracer,
            )
        elif can_skip_global_comm and estimated_total_size < target_partition_size:
            # Single rank with small data - use tree reduction
            if tracer is not None:
                tracer.decision = "tree_local"
            await _tree_groupby(
                context,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                initial_chunks,
                adaptive_n_ary,
                tracer=tracer,
            )
        elif estimated_total_size < target_partition_size:
            # Small output - use tree reduction with allgather to merge across ranks
            if tracer is not None:
                tracer.decision = "tree_allgather"
            await _tree_groupby(
                context,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                initial_chunks,
                adaptive_n_ary,
                collective_ids.pop() if collective_ids else None,
                tracer,
            )
        elif not collective_ids:
            # No shuffle ID available - fall back to tree (no allgather)
            if tracer is not None:
                tracer.decision = "tree_fallback"
            await _tree_groupby(
                context,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                initial_chunks,
                adaptive_n_ary,
                tracer=tracer,
            )
        else:
            # Large output - use shuffle
            ideal_count = max(1, estimated_total_size // target_partition_size)
            # Cap at global chunk count, but use the rank count if it's larger
            output_count = max(nranks, min(ideal_count, global_chunk_count))
            if tracer is not None:
                tracer.decision = "shuffle"
            await _shuffle_groupby(
                context,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                initial_chunks,
                output_count,
                collective_ids.pop(),
                key_indices,
                tracer,
            )


# ============================================================================
# Network Generation
# ============================================================================


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
            dynamic_planning.sample_chunk_count,
            config_options.executor.target_partition_size,
            config_options.executor.groupby_n_ary,
            collective_ids,
        )
    ]

    return nodes, channels
