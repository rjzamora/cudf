# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GroupBy logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.message import Message
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
from cudf_polars.experimental.rapidsmpf.nodes import (
    define_py_node,
    shutdown_on_error,
)
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    Metadata,
    opaque_reservation,
    process_children,
)
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair


# ============================================================================
# Helper Functions
# ============================================================================


def apply_do_evaluate(
    chunk: TableChunk,
    ir: GroupBy | Select,
    ir_context: IRExecutionContext,
    *,
    input_schema: dict[str, Any] | None = None,
) -> TableChunk:
    """
    Apply GroupBy or Select evaluation to a chunk.

    Parameters
    ----------
    chunk
        The input TableChunk to evaluate.
    ir
        The GroupBy or Select IR node.
    ir_context
        The IR execution context.
    input_schema
        Schema for the input chunk. If None, uses ir.children[0].schema.

    Returns
    -------
    The evaluated TableChunk.
    """
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
    """
    Holds decomposed GroupBy operations for multi-phase aggregation.

    This class encapsulates the three-phase decomposition of a GroupBy:
    1. Piecewise aggregation (initial per-partition groupby)
    2. Reduction aggregation (combining partial results)
    3. Selection (final column selection/transformation)
    """

    ir: GroupBy
    """The original GroupBy IR node."""

    piecewise_ir: GroupBy
    """IR for the initial partition-wise groupby."""

    reduction_ir: GroupBy
    """IR for the reduction phase groupby."""

    select_ir: Select
    """IR for the final selection phase."""

    need_preshuffle: bool
    """Whether a pre-shuffle is needed (e.g., for n_unique)."""

    grouped_keys: tuple[NamedExpr, ...]
    """The groupby key expressions."""

    @classmethod
    def from_groupby(cls, ir: GroupBy) -> DecomposedGroupBy:
        """
        Decompose a GroupBy IR node into multi-phase operations.

        Parameters
        ----------
        ir
            The GroupBy IR node to decompose.

        Returns
        -------
        A DecomposedGroupBy instance with all phase operations defined.

        Raises
        ------
        NotImplementedError
            If the aggregation cannot be decomposed.
        """
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

        # Selection IR
        select_ir = Select(
            ir.schema,
            [
                *(NamedExpr(k.name, Col(k.value.dtype, k.name)) for k in grouped_keys),
                *selection_exprs,
            ],
            False,  # noqa: FBT003
            piecewise_ir,
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
    ch_out: ChannelPair,
    ch_in: ChannelPair,
    input_metadata: Metadata,
    groupby_keys: tuple[str, ...],
) -> None:
    """
    Execute partition-wise groupby when data is already partitioned on keys.

    This is the simplest and most efficient strategy - just apply the groupby
    to each partition independently since data is already partitioned by keys.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The GroupBy IR node.
    ir_context
        The IR execution context.
    ch_out
        The output channel pair.
    ch_in
        The input channel pair.
    input_metadata
        Metadata from the input channel.
    groupby_keys
        The groupby key column names.
    """
    # Output metadata preserves partitioning info
    output_metadata = Metadata(
        local_count=input_metadata.local_count,
        global_count=input_metadata.global_count,
        local_partitioned_on=groupby_keys,
        global_partitioned_on=input_metadata.global_partitioned_on,
        duplicated=input_metadata.duplicated,
    )
    await ch_out.send_metadata(context, output_metadata)

    # Apply groupby to each chunk
    while (msg := await ch_in.data.recv(context)) is not None:
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        seq_num = msg.sequence_number
        del msg
        with opaque_reservation(context, chunk.data_alloc_size(MemoryType.DEVICE)):
            chunk = await asyncio.to_thread(apply_do_evaluate, chunk, ir, ir_context)
            await ch_out.data.send(context, Message(seq_num, chunk))
            del chunk

    await ch_out.data.drain(context)


async def _allgather_groupby(
    context: Context,
    ir: GroupBy,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_in: ChannelPair,
    input_metadata: Metadata,
    collective_id: int,
) -> None:
    """
    Execute groupby by allgathering all data to each rank.

    Used when outputting a single partition or when data must be pre-concatenated
    (e.g., maintain_order is True). Produces a single duplicated chunk.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The GroupBy IR node.
    ir_context
        The IR execution context.
    ch_out
        The output channel pair.
    ch_in
        The input channel pair.
    input_metadata
        Metadata from the input channel.
    collective_id
        The collective ID for allgather.
    """
    # Collect chunks (allgather if needed for multi-rank)
    input_bytes = 0
    chunks: list[TableChunk] = []
    need_allgather = not input_metadata.duplicated and context.comm().nranks > 1

    # After allgather, all workers have identical data, so output is duplicated
    output_metadata = Metadata(
        local_count=1,
        global_count=1,
        duplicated=need_allgather or input_metadata.duplicated,
    )
    await ch_out.send_metadata(context, output_metadata)

    if need_allgather:
        allgather = AllGatherManager(context, collective_id)
        stream = context.get_stream_from_pool()
        seq_num = 0
        while (msg := await ch_in.data.recv(context)) is not None:
            allgather.insert(seq_num, TableChunk.from_message(msg))
            del msg
            seq_num += 1
        allgather.insert_finished()
        chunks.append(
            TableChunk.from_pylibcudf_table(
                await allgather.extract_concatenated(stream),
                stream,
                exclusive_view=True,
            )
        )
        input_bytes += chunks[-1].data_alloc_size(MemoryType.DEVICE)
    else:
        while (msg := await ch_in.data.recv(context)) is not None:
            chunk = TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )
            del msg
            chunks.append(chunk)
            input_bytes += chunk.data_alloc_size(MemoryType.DEVICE)

    if chunks:
        with opaque_reservation(context, input_bytes):
            multi_chunks = len(chunks) > 1

            def _do_concat_and_evaluate() -> DataFrame:
                return ir.do_evaluate(
                    *ir._non_child_args,
                    _concat(
                        *[
                            DataFrame.from_table(
                                chunk.table_view(),
                                list(ir.children[0].schema.keys()),
                                list(ir.children[0].schema.values()),
                                chunk.stream,
                            )
                            for chunk in chunks
                        ],
                        context=ir_context,
                    ),
                    context=ir_context,
                )

            df = await asyncio.to_thread(_do_concat_and_evaluate)
            if multi_chunks:
                del chunks
            await ch_out.data.send(
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

    await ch_out.data.drain(context)


async def _local_concat_groupby(
    context: Context,
    ir: GroupBy,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_in: ChannelPair,
    input_metadata: Metadata,
) -> None:
    """
    Execute groupby by concatenating all local data first.

    Like _allgather_groupby but without allgather - used when data is already
    confined to the rank (duplicated, globally partitioned, or single rank)
    but we need to concatenate before groupby (e.g., maintain_order is True
    or non-decomposable aggregation).

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The GroupBy IR node.
    ir_context
        The IR execution context.
    ch_out
        The output channel pair.
    ch_in
        The input channel pair.
    input_metadata
        Metadata from the input channel.
    """
    nranks = context.comm().nranks

    # Output metadata: single chunk per rank
    output_metadata = Metadata(
        local_count=1,
        global_count=1 if input_metadata.duplicated else nranks,
        duplicated=input_metadata.duplicated,
    )
    await ch_out.send_metadata(context, output_metadata)

    # Collect all local chunks
    input_bytes = 0
    chunks: list[TableChunk] = []
    while (msg := await ch_in.data.recv(context)) is not None:
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        del msg
        chunks.append(chunk)
        input_bytes += chunk.data_alloc_size(MemoryType.DEVICE)

    if chunks:
        with opaque_reservation(context, input_bytes):
            multi_chunks = len(chunks) > 1

            def _do_concat_and_evaluate() -> DataFrame:
                return ir.do_evaluate(
                    *ir._non_child_args,
                    _concat(
                        *[
                            DataFrame.from_table(
                                chunk.table_view(),
                                list(ir.children[0].schema.keys()),
                                list(ir.children[0].schema.values()),
                                chunk.stream,
                            )
                            for chunk in chunks
                        ],
                        context=ir_context,
                    ),
                    context=ir_context,
                )

            df = await asyncio.to_thread(_do_concat_and_evaluate)
            if multi_chunks:
                del chunks
            await ch_out.data.send(
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

    await ch_out.data.drain(context)


async def _shuffle_groupby(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_in: ChannelPair,
    output_count: int,
    collective_id: int,
    groupby_keys: tuple[str, ...],
    initial_chunks: list[TableChunk],
) -> None:
    """
    Execute groupby using shuffle-based redistribution.

    Data is shuffled by groupby keys so that all rows with the same key
    end up on the same partition, then a local groupby is applied.

    Parameters
    ----------
    context
        The rapidsmpf context.
    decomposed
        The decomposed GroupBy operations.
    ir_context
        The IR execution context.
    ch_out
        The output channel pair.
    ch_in
        The input channel pair.
    output_count
        The global number of output partitions.
    collective_id
        The collective ID for shuffle.
    groupby_keys
        The groupby key column names.
    initial_chunks
        Pre-processed chunks to insert (list is cleared after consumption).
    """
    nranks = context.comm().nranks

    # Send output metadata - after shuffle, data is globally partitioned on keys
    output_metadata = Metadata(
        local_count=max(1, math.ceil(output_count / nranks)),
        global_count=output_count,
        local_partitioned_on=groupby_keys,
        global_partitioned_on=groupby_keys,
        duplicated=False,
    )
    await ch_out.send_metadata(context, output_metadata)

    # Set up shuffle manager
    pwise_schema_keys = list(decomposed.piecewise_ir.schema.keys())
    pwise_key_indices = tuple(pwise_schema_keys.index(key) for key in groupby_keys)
    shuffle = ShuffleManager(
        context,
        output_count,
        pwise_key_indices,
        collective_id,
    )

    # Insert initial chunks (already piecewise-grouped) and release references
    for chunk in initial_chunks:
        shuffle.insert_chunk(chunk)
    initial_chunks.clear()

    # Process remaining chunks: apply piecewise groupby and insert into shuffle
    while (msg := await ch_in.data.recv(context)) is not None:
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        del msg
        with opaque_reservation(context, chunk.data_alloc_size(MemoryType.DEVICE)):
            chunk = await asyncio.to_thread(
                apply_do_evaluate,
                chunk,
                decomposed.piecewise_ir,
                ir_context,
            )
            shuffle.insert_chunk(chunk)
            del chunk

    await shuffle.insert_finished()

    # Extract shuffled chunks and apply reduction + selection
    for partition_id in range(
        context.comm().rank,
        output_count,
        context.comm().nranks,
    ):
        stream = ir_context.get_cuda_stream()
        chunk = TableChunk.from_pylibcudf_table(
            await shuffle.extract_chunk(partition_id, stream),
            stream,
            exclusive_view=True,
        )

        with opaque_reservation(context, chunk.data_alloc_size(MemoryType.DEVICE)):
            # Apply reduction then selection
            chunk = await asyncio.to_thread(
                apply_do_evaluate,
                chunk,
                decomposed.reduction_ir,
                ir_context,
            )
            chunk = await asyncio.to_thread(
                apply_do_evaluate,
                chunk,
                decomposed.select_ir,
                ir_context,
                input_schema=decomposed.reduction_ir.schema,
            )
            await ch_out.data.send(context, Message(partition_id, chunk))
            del chunk

    await ch_out.data.drain(context)


async def _local_shuffle_groupby(
    context: Context,
    ir: GroupBy,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_in: ChannelPair,
    input_metadata: Metadata,
    output_count: int,
    collective_id: int,
    groupby_keys: tuple[str, ...],
) -> None:
    """
    Execute groupby using local shuffle-based redistribution.

    Like _shuffle_groupby but uses a local (single-rank) communicator.
    Used when data is already globally partitioned (or duplicated) but needs
    local repartitioning by groupby keys to produce multiple output chunks.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The GroupBy IR node.
    ir_context
        The IR execution context.
    ch_out
        The output channel pair.
    ch_in
        The input channel pair.
    input_metadata
        Metadata from the input channel.
    output_count
        The global number of output partitions.
    collective_id
        The collective ID for local shuffle.
    groupby_keys
        The groupby key column names.
    """
    nranks = context.comm().nranks

    # Calculate local output count
    local_output_count = max(1, math.ceil(output_count / nranks))

    # Send output metadata - local shuffle establishes local partitioning
    output_metadata = Metadata(
        local_count=local_output_count,
        global_count=output_count
        if input_metadata.duplicated
        else local_output_count * nranks,
        local_partitioned_on=groupby_keys,
        global_partitioned_on=input_metadata.global_partitioned_on,
        duplicated=input_metadata.duplicated,
    )
    await ch_out.send_metadata(context, output_metadata)

    # Set up local shuffle manager (single-rank communicator)
    # Use INPUT schema (ir.children[0].schema) since we're shuffling input data
    # Only include keys that exist in the input schema (exclude literal expressions)
    input_schema_keys = list(ir.children[0].schema.keys())
    shuffleable_keys = tuple(k for k in groupby_keys if k in input_schema_keys)
    key_indices = tuple(input_schema_keys.index(key) for key in shuffleable_keys)
    shuffle = ShuffleManager(
        context,
        local_output_count,
        key_indices,
        collective_id,
        local=True,  # Use single-rank communicator
    )

    # Insert all input chunks into local shuffle
    while (msg := await ch_in.data.recv(context)) is not None:
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        del msg
        shuffle.insert_chunk(chunk)
        del chunk

    await shuffle.insert_finished()

    # Extract shuffled chunks and apply groupby
    for partition_id in range(local_output_count):
        stream = ir_context.get_cuda_stream()
        chunk = TableChunk.from_pylibcudf_table(
            await shuffle.extract_chunk(partition_id, stream),
            stream,
            exclusive_view=True,
        )

        with opaque_reservation(context, chunk.data_alloc_size(MemoryType.DEVICE)):
            chunk = await asyncio.to_thread(
                apply_do_evaluate,
                chunk,
                ir,
                ir_context,
            )
            await ch_out.data.send(context, Message(partition_id, chunk))
            del chunk

    await ch_out.data.drain(context)


async def _tree_groupby(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_in: ChannelPair,
    input_metadata: Metadata,
    global_output_count: int,
    collective_ids: list[int],
    groupby_n_ary: int,
    target_partition_size: int,
    initial_chunks: list[TableChunk],
    sample_pwise_size: int | None,
    shuffled: bool,  # noqa: FBT001
    pre_shuffle: ShuffleManager | None,
    my_preshuffle_ids: list[int],
) -> None:
    """
    Execute groupby using N-ary tree reduction.

    Reduces the number of partitions by repeatedly combining groups of k
    partitions via local groupby operations, forming a tree structure.

    Parameters
    ----------
    context
        The rapidsmpf context.
    decomposed
        The decomposed GroupBy operations.
    ir_context
        The IR execution context.
    ch_out
        The output channel pair.
    ch_in
        The input channel pair.
    input_metadata
        Metadata from the input channel.
    global_output_count
        The number of output chunks specified at the global planning stage.
    collective_ids
        The collective IDs for optional post-allgather.
    groupby_n_ary
        The N-ary factor for tree reduction.
    target_partition_size
        Target partition size in bytes.
    initial_chunks
        Pre-processed chunks to start with (list is cleared after consumption).
    sample_pwise_size
        Average size of sampled chunks after piecewise groupby (for k estimation).
    shuffled
        Whether data was pre-shuffled.
    pre_shuffle
        Optional pre-shuffle manager if data was pre-shuffled.
    my_preshuffle_ids
        Partition IDs owned by this rank after pre-shuffle.
    """
    nranks = context.comm().nranks

    # Determine if we need post-allgather and calculate local output count
    # Post-allgather is needed when:
    # - global_output_count == 1 (must produce single duplicated chunk)
    # - Data is NOT duplicated (each rank has different data)
    # - Data is NOT shuffled on groupby keys (groups may span ranks)
    # - Multiple ranks exist
    post_allgather: AllGatherManager | None = None
    need_post_allgather = (
        global_output_count == 1
        and not input_metadata.duplicated
        and not shuffled
        and nranks > 1
    )

    if need_post_allgather:
        post_allgather = AllGatherManager(context, collective_ids.pop())
        local_output_count = 1
        output_duplicated = True
        output_global_count = 1
    elif input_metadata.duplicated:
        # Duplicated input: each rank reduces to global_output_count chunks
        local_output_count = global_output_count
        output_duplicated = True
        output_global_count = global_output_count
    else:
        # Shuffled or globally partitioned: groups confined to ranks
        local_output_count = max(1, math.ceil(global_output_count / nranks))
        output_duplicated = False
        output_global_count = local_output_count * nranks

    # Send output metadata
    output_metadata = Metadata(
        local_count=local_output_count,
        global_count=output_global_count,
        duplicated=output_duplicated,
    )
    await ch_out.send_metadata(context, output_metadata)

    # Calculate tree parameters
    n = input_metadata.local_count
    if sample_pwise_size is not None and sample_pwise_size > 0:
        k = min(max(2, target_partition_size // sample_pwise_size), 1024)
    else:  # pragma: no cover
        k = groupby_n_ary
    level_count = int(math.ceil(math.log(n * (k - 1) + 1) / math.log(k)))

    # Tree reduction state: store TableChunks directly (they own their memory)
    # Level 0 chunks use piecewise_ir.schema, level 1+ use reduction_ir.schema
    done_receiving = False
    levels: defaultdict[int, list[TableChunk]] = defaultdict(list)

    # Process initial chunks (already piecewise-grouped) and release references
    input_partition_idx = len(initial_chunks)
    levels[0].extend(initial_chunks)
    initial_chunks.clear()

    # Main tree reduction loop
    sequence_num: int = 0
    while not done_receiving:
        chunk: TableChunk | None = None

        if pre_shuffle is not None:
            # Extract from pre-shuffle
            if input_partition_idx < len(my_preshuffle_ids):
                stream = ir_context.get_cuda_stream()
                chunk = TableChunk.from_pylibcudf_table(
                    await pre_shuffle.extract_chunk(
                        my_preshuffle_ids[input_partition_idx], stream
                    ),
                    stream,
                    exclusive_view=True,
                )
                chunk = await asyncio.to_thread(
                    apply_do_evaluate,
                    chunk,
                    decomposed.piecewise_ir,
                    ir_context,
                )
                input_partition_idx += 1
            else:
                done_receiving = True
        else:
            # Read from input channel
            msg = await ch_in.data.recv(context)
            if msg is None:
                done_receiving = True
            else:
                chunk = TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
                del msg
                chunk = await asyncio.to_thread(
                    apply_do_evaluate,
                    chunk,
                    decomposed.piecewise_ir,
                    ir_context,
                )
                input_partition_idx += 1

        if chunk is not None:
            levels[0].append(chunk)

        # Push chunks through tree levels
        for level in range(level_count):
            if levels[level]:
                count = len(levels[level])
                if count >= k or done_receiving:
                    next_level = min(level + 1, level_count - 1)
                    # Get schema for this level's chunks
                    level_schema = (
                        decomposed.piecewise_ir.schema
                        if level == 0
                        else decomposed.reduction_ir.schema
                    )
                    # Pop chunks and convert to DataFrames for reduction
                    chunks_to_reduce = [levels[level].pop() for _ in range(count)]

                    def _do_reduction(
                        _schema: dict[str, Any] = level_schema,
                        _chunks: list[TableChunk] = chunks_to_reduce,
                    ) -> DataFrame:
                        return decomposed.reduction_ir.do_evaluate(
                            *decomposed.reduction_ir._non_child_args,
                            _concat(
                                *[
                                    DataFrame.from_table(
                                        c.table_view(),
                                        list(_schema.keys()),
                                        list(_schema.values()),
                                        c.stream,
                                    )
                                    for c in _chunks
                                ],
                                context=ir_context,
                            ),
                            context=ir_context,
                        )

                    df = await asyncio.to_thread(_do_reduction)
                    del chunks_to_reduce
                    # Convert result back to TableChunk for storage
                    levels[next_level].append(
                        TableChunk.from_pylibcudf_table(
                            df.table, df.stream, exclusive_view=True
                        )
                    )
                    del df

                if level == level_count - 1 and (
                    local_output_count > 1 or done_receiving
                ):
                    assert len(levels[level]) == 1, "Expected 1 chunk at the last level"
                    chunk = levels[level].pop()

                    if post_allgather is not None:
                        post_allgather.insert(sequence_num, chunk)
                        del chunk
                    else:
                        # Apply selection and send
                        def _do_selection(
                            _chunk: TableChunk = chunk,
                        ) -> DataFrame:
                            return decomposed.select_ir.do_evaluate(
                                *decomposed.select_ir._non_child_args,
                                DataFrame.from_table(
                                    _chunk.table_view(),
                                    list(decomposed.reduction_ir.schema.keys()),
                                    list(decomposed.reduction_ir.schema.values()),
                                    _chunk.stream,
                                ),
                                context=ir_context,
                            )

                        df = await asyncio.to_thread(_do_selection)
                        del chunk
                        await ch_out.data.send(
                            context,
                            Message(
                                sequence_num,
                                TableChunk.from_pylibcudf_table(
                                    df.table, df.stream, exclusive_view=True
                                ),
                            ),
                        )
                        del df
                    sequence_num += 1

    # Handle post-allgather if needed
    if post_allgather is not None:
        post_allgather.insert_finished()
        stream = ir_context.get_cuda_stream()
        allgather_result = await post_allgather.extract_concatenated(stream)

        def _do_final_reduction_and_selection() -> DataFrame:
            return decomposed.select_ir.do_evaluate(
                *decomposed.select_ir._non_child_args,
                decomposed.reduction_ir.do_evaluate(
                    *decomposed.reduction_ir._non_child_args,
                    DataFrame.from_table(
                        allgather_result,
                        list(decomposed.reduction_ir.schema.keys()),
                        list(decomposed.reduction_ir.schema.values()),
                        stream,
                    ),
                    context=ir_context,
                ),
                context=ir_context,
            )

        df = await asyncio.to_thread(_do_final_reduction_and_selection)
        await ch_out.data.send(
            context,
            Message(
                0,
                TableChunk.from_pylibcudf_table(
                    df.table, df.stream, exclusive_view=True
                ),
            ),
        )
        del df

    await ch_out.data.drain(context)


# ============================================================================
# Main GroupBy Node
# ============================================================================


@define_py_node()
async def groupby_node(
    context: Context,
    ir: GroupBy,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_in: ChannelPair,
    groupby_n_ary: int,
    target_partition_size: int,
    output_count: int,
    collective_id: int,
) -> None:
    """
    Dynamic GroupBy node that selects the best strategy at runtime.

    Chooses between four strategies based on input metadata:
    1. Partition-wise: Data already shuffled on groupby keys
    2. Allgather: Single output partition or maintain_order
    3. Shuffle: Multiple output partitions, not already shuffled
    4. Tree: N-ary tree reduction

    Parameters
    ----------
    context
        The context of the node.
    ir
        The GroupBy IR node.
    ir_context
        The IR execution context.
    ch_out
        The output channel pair.
    ch_in
        The input channel pair.
    groupby_n_ary
        The groupby n-ary factor for tree reduction.
    target_partition_size
        The target partition size in bytes.
    output_count
        The output partition count.
    collective_id
        The collective ID for shuffle/allgather operations.
    """
    collective_ids = [collective_id]

    async with shutdown_on_error(
        context,
        ch_in.metadata,
        ch_in.data,
        ch_out.metadata,
        ch_out.data,
    ):
        # Get groupby key column names and indices
        groupby_keys = tuple(ne.name for ne in ir.keys)
        nranks = context.comm().nranks

        # Receive input metadata
        input_metadata = await ch_in.recv_metadata(context)

        # Determine partitioning status
        globally_partitioned = input_metadata.global_partitioned_on == groupby_keys
        locally_partitioned = input_metadata.local_partitioned_on == groupby_keys

        # Can we skip global communication?
        # Yes if: single rank, OR data is duplicated, OR already globally partitioned
        can_skip_global_comm = (
            nranks == 1 or input_metadata.duplicated or globally_partitioned
        )

        # Try to decompose the aggregation for multi-phase execution
        need_preconcat = ir.maintain_order
        try:
            decomposed = DecomposedGroupBy.from_groupby(ir)
        except NotImplementedError:
            need_preconcat = True
            decomposed = None

        # =====================================================================
        # Algorithm Selection
        # =====================================================================

        if can_skip_global_comm:
            # No global communication needed
            if output_count == 1:
                # Reduce to single chunk per rank
                if need_preconcat:
                    # B: Local concat + single groupby
                    await _local_concat_groupby(
                        context, ir, ir_context, ch_out, ch_in, input_metadata
                    )
                    return
                else:
                    # A: Tree reduction to single chunk
                    # Fall through to tree groupby below
                    pass
            else:  # output_count > 1
                if locally_partitioned:
                    # C: Already locally partitioned - partition-wise
                    await _partitionwise_groupby(
                        context,
                        ir,
                        ir_context,
                        ch_out,
                        ch_in,
                        input_metadata,
                        groupby_keys,
                    )
                    return
                else:
                    # D: Need local shuffle first
                    await _local_shuffle_groupby(
                        context,
                        ir,
                        ir_context,
                        ch_out,
                        ch_in,
                        input_metadata,
                        output_count,
                        collective_ids.pop(),
                        groupby_keys,
                    )
                    return
        else:
            # Need global communication
            if output_count == 1:
                if need_preconcat:
                    # E: Allgather + local groupby
                    await _allgather_groupby(
                        context,
                        ir,
                        ir_context,
                        ch_out,
                        ch_in,
                        input_metadata,
                        collective_ids.pop(),
                    )
                    return
                else:
                    # F: Tree + post-allgather (fall through to tree groupby)
                    pass
            else:  # output_count > 1
                # G: Global shuffle required
                assert decomposed is not None, "Need decomposed for shuffle groupby"

                # Sample chunks for size estimation before shuffle
                max_sample_chunks = 1  # TODO: Make configurable
                initial_chunks: list[TableChunk] = []
                sample_count = min(max_sample_chunks, input_metadata.local_count)
                for _ in range(sample_count):
                    msg = await ch_in.data.recv(context)
                    if msg is None:
                        break
                    initial_chunks.append(TableChunk.from_message(msg))
                    del msg

                # Apply piecewise groupby to sampled chunks
                for i in range(len(initial_chunks)):
                    initial_chunks[i] = initial_chunks[i].make_available_and_spill(
                        context.br(), allow_overbooking=True
                    )
                    with opaque_reservation(
                        context, initial_chunks[i].data_alloc_size(MemoryType.DEVICE)
                    ):
                        initial_chunks[i] = await asyncio.to_thread(
                            apply_do_evaluate,
                            initial_chunks[i],
                            decomposed.piecewise_ir,
                            ir_context,
                        )

                await _shuffle_groupby(
                    context,
                    decomposed,
                    ir_context,
                    ch_out,
                    ch_in,
                    output_count,
                    collective_ids.pop(),
                    groupby_keys,
                    initial_chunks,
                )
                return

        # =====================================================================
        # Tree Groupby (cases A and F)
        # =====================================================================
        assert decomposed is not None, "Need decomposed for tree groupby"

        # Pre-shuffle if needed (e.g., for n_unique aggregation)
        # Use INPUT schema since pre-shuffle happens before any groupby transformation
        # Only include keys that exist in the input schema (exclude literal expressions)
        input_schema_keys = list(ir.children[0].schema.keys())
        shuffleable_keys = tuple(k for k in groupby_keys if k in input_schema_keys)
        groupby_key_indices = tuple(
            input_schema_keys.index(k) for k in shuffleable_keys
        )
        pre_shuffle: ShuffleManager | None = None
        shuffled = globally_partitioned
        my_preshuffle_ids = list(
            range(
                context.comm().rank,
                input_metadata.local_count,
                nranks,
            )
        )
        max_sample_chunks = 1  # TODO: Make configurable

        # Can only pre-shuffle if we have keys that exist in the input schema
        if decomposed.need_preshuffle and groupby_key_indices:
            pre_shuffle = ShuffleManager(
                context,
                input_metadata.local_count,
                groupby_key_indices,
                collective_ids.pop(),
            )
            while (msg := await ch_in.data.recv(context)) is not None:
                pre_shuffle.insert_chunk(
                    TableChunk.from_message(msg).make_available_and_spill(
                        context.br(), allow_overbooking=True
                    )
                )
                del msg
            shuffled = True
            await pre_shuffle.insert_finished()

        # Sample chunks for size estimation and pre-processing
        initial_chunks = []
        sample_count = min(
            max_sample_chunks,
            len(my_preshuffle_ids)
            if pre_shuffle is not None
            else input_metadata.local_count,
        )

        if pre_shuffle is not None:
            # Sample chunks from pre-shuffle
            for i in range(sample_count):
                stream = ir_context.get_cuda_stream()
                chunk = TableChunk.from_pylibcudf_table(
                    await pre_shuffle.extract_chunk(my_preshuffle_ids[i], stream),
                    stream,
                    exclusive_view=True,
                )
                initial_chunks.append(chunk)
        else:
            # Sample chunks from input channel
            for _ in range(sample_count):
                msg = await ch_in.data.recv(context)
                if msg is None:
                    break
                initial_chunks.append(TableChunk.from_message(msg))
                del msg

        # Apply piecewise groupby to sampled chunks for size estimation
        sample_pwise_size: int | None = None
        total_pwise_size = 0
        for i in range(len(initial_chunks)):
            initial_chunks[i] = initial_chunks[i].make_available_and_spill(
                context.br(), allow_overbooking=True
            )
            with opaque_reservation(
                context, initial_chunks[i].data_alloc_size(MemoryType.DEVICE)
            ):
                initial_chunks[i] = await asyncio.to_thread(
                    apply_do_evaluate,
                    initial_chunks[i],
                    decomposed.piecewise_ir,
                    ir_context,
                )
            total_pwise_size += initial_chunks[i].data_alloc_size(MemoryType.DEVICE)
        if initial_chunks:
            sample_pwise_size = total_pwise_size // len(initial_chunks)

        await _tree_groupby(
            context,
            decomposed,
            ir_context,
            ch_out,
            ch_in,
            input_metadata,
            output_count,
            collective_ids,
            groupby_n_ary,
            target_partition_size,
            initial_chunks,
            sample_pwise_size,
            shuffled,
            pre_shuffle,
            my_preshuffle_ids,
        )


# ============================================================================
# Network Generation
# ============================================================================


@generate_ir_sub_network.register(GroupBy)
def _(
    ir: GroupBy, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for GroupBy operation."""
    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Get executor configuration
    config_options = rec.state["config_options"]
    executor = config_options.executor
    assert executor.name == "streaming", "GroupBy node requires streaming executor"

    # Create the groupby node
    nodes[ir] = [
        groupby_node(
            rec.state["context"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            executor.groupby_n_ary,
            executor.target_partition_size,
            rec.state["partition_info"][ir].count,
            rec.state["collective_id_map"][ir],
        )
    ]

    return nodes, channels
