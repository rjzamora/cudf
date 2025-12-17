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
) -> None:
    """
    Execute partition-wise groupby when data is already shuffled on keys.

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
    """
    # Forward metadata directly
    await ch_out.send_metadata(context, input_metadata)

    # Apply groupby to each chunk
    while (msg := await ch_in.data.recv(context)) is not None:
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        with opaque_reservation(context, chunk.data_alloc_size(MemoryType.DEVICE)):
            result_chunk = await asyncio.to_thread(
                apply_do_evaluate, chunk, ir, ir_context
            )
            await ch_out.data.send(context, Message(msg.sequence_number, result_chunk))
            del chunk, msg

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
    (e.g., maintain_order is True).

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
    output_metadata = Metadata(1, duplicated=input_metadata.duplicated)
    await ch_out.send_metadata(context, output_metadata)

    # Collect chunks (allgather if needed for multi-rank)
    input_bytes = 0
    chunks: list[TableChunk] = []
    need_allgather = not input_metadata.duplicated and context.comm().nranks > 1

    if need_allgather:
        allgather = AllGatherManager(context, collective_id)
        stream = context.get_stream_from_pool()
        seq_num = 0
        while (msg := await ch_in.data.recv(context)) is not None:
            allgather.insert(seq_num, TableChunk.from_message(msg))
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
            chunks.append(chunk)
            input_bytes += chunk.data_alloc_size(MemoryType.DEVICE)

    if chunks:
        with opaque_reservation(context, input_bytes):
            multi_chunks = len(chunks) > 1
            df = ir.do_evaluate(
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
    groupby_key_columns: list[str],
    first_chunk: TableChunk | None = None,
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
        The number of output partitions.
    collective_id
        The collective ID for shuffle.
    groupby_key_columns
        The names of the groupby key columns.
    first_chunk
        Optional first chunk that was already read and processed.
    """
    # Send output metadata
    output_metadata = Metadata(
        output_count,
        partitioned_on=tuple(groupby_key_columns),
    )
    await ch_out.send_metadata(context, output_metadata)

    # Set up shuffle manager
    pwise_schema_keys = list(decomposed.piecewise_ir.schema.keys())
    pwise_key_indices = tuple(
        pwise_schema_keys.index(key) for key in groupby_key_columns
    )
    shuffle = ShuffleManager(
        context,
        output_count,
        pwise_key_indices,
        collective_id,
    )

    # Insert first chunk if available (already piecewise-grouped)
    if first_chunk is not None:
        shuffle.insert_chunk(first_chunk)

    # Process remaining chunks: apply piecewise groupby and insert into shuffle
    while (msg := await ch_in.data.recv(context)) is not None:
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        del msg
        with opaque_reservation(context, chunk.data_alloc_size(MemoryType.DEVICE)):
            pwise_chunk = await asyncio.to_thread(
                apply_do_evaluate,
                chunk,
                decomposed.piecewise_ir,
                ir_context,
            )
            shuffle.insert_chunk(pwise_chunk)
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
            reduced_chunk = await asyncio.to_thread(
                apply_do_evaluate,
                chunk,
                decomposed.reduction_ir,
                ir_context,
            )
            final_chunk = await asyncio.to_thread(
                apply_do_evaluate,
                reduced_chunk,
                decomposed.select_ir,
                ir_context,
                input_schema=decomposed.reduction_ir.schema,
            )
            await ch_out.data.send(context, Message(partition_id, final_chunk))
            del chunk

    await ch_out.data.drain(context)


async def _tree_groupby(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_in: ChannelPair,
    input_metadata: Metadata,
    output_count: int,
    collective_id: int | None,
    groupby_n_ary: int,
    target_partition_size: int,
    first_chunk: TableChunk | None,
    first_pwise_size: int | None,
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
    output_count
        The number of output partitions.
    collective_id
        The collective ID for optional post-allgather.
    groupby_n_ary
        The N-ary factor for tree reduction.
    target_partition_size
        Target partition size in bytes.
    first_chunk
        First chunk (already piecewise-grouped) for sampling.
    first_pwise_size
        Size of first chunk after piecewise groupby (for k estimation).
    shuffled
        Whether data was pre-shuffled.
    pre_shuffle
        Optional pre-shuffle manager if data was pre-shuffled.
    my_preshuffle_ids
        Partition IDs owned by this rank after pre-shuffle.
    """
    # Send output metadata
    output_metadata = Metadata(output_count, duplicated=not shuffled)
    await ch_out.send_metadata(context, output_metadata)

    # Prepare optional post-allgather
    post_allgather: AllGatherManager | None = None
    if (
        output_count == 1
        and not input_metadata.duplicated
        and not shuffled
        and context.comm().nranks > 1
        and collective_id is not None
    ):
        post_allgather = AllGatherManager(context, collective_id)

    # Calculate tree parameters
    n = input_metadata.count
    if first_pwise_size is not None:
        k = min(max(2, target_partition_size // first_pwise_size), 1024)
    else:  # pragma: no cover
        k = groupby_n_ary
    level_count = int(math.ceil(math.log(n * (k - 1) + 1) / math.log(k)))

    # Tree reduction state
    done_receiving = False
    levels: defaultdict[int, list[DataFrame]] = defaultdict(list)
    input_partition_idx = 0

    # Process first chunk if available
    if first_chunk is not None:
        levels[0].append(
            DataFrame.from_table(
                first_chunk.table_view(),
                list(decomposed.piecewise_ir.schema.keys()),
                list(decomposed.piecewise_ir.schema.values()),
                first_chunk.stream,
            )
        )
        input_partition_idx = 1

    # Main tree reduction loop
    sequence_num: int = 0
    while not done_receiving:
        chunk: TableChunk | None = None

        if pre_shuffle is not None:
            # Extract from pre-shuffle
            if input_partition_idx < len(my_preshuffle_ids):
                stream = ir_context.get_cuda_stream()
                input_chunk = TableChunk.from_pylibcudf_table(
                    await pre_shuffle.extract_chunk(
                        my_preshuffle_ids[input_partition_idx], stream
                    ),
                    stream,
                    exclusive_view=True,
                )
                chunk = await asyncio.to_thread(
                    apply_do_evaluate,
                    input_chunk,
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
                input_chunk = TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
                chunk = await asyncio.to_thread(
                    apply_do_evaluate,
                    input_chunk,
                    decomposed.piecewise_ir,
                    ir_context,
                )
                input_partition_idx += 1

        if chunk is not None:
            levels[0].append(
                DataFrame.from_table(
                    chunk.table_view(),
                    list(decomposed.piecewise_ir.schema.keys()),
                    list(decomposed.piecewise_ir.schema.values()),
                    chunk.stream,
                )
            )

        # Push chunks through tree levels
        for level in range(level_count):
            if levels[level]:
                count = len(levels[level])
                if count >= k or done_receiving:
                    next_level = min(level + 1, level_count - 1)
                    df = decomposed.reduction_ir.do_evaluate(
                        *decomposed.reduction_ir._non_child_args,
                        _concat(
                            *[levels[level].pop() for _ in range(count)],
                            context=ir_context,
                        ),
                        context=ir_context,
                    )
                    levels[next_level].append(df)

                if level == level_count - 1 and (output_count > 1 or done_receiving):
                    assert len(levels[level]) == 1, "Expected 1 chunk at the last level"
                    df = levels[level].pop()

                    if post_allgather is not None:
                        table_chunk = TableChunk.from_pylibcudf_table(
                            df.table, df.stream, exclusive_view=True
                        )
                        post_allgather.insert(sequence_num, table_chunk)
                    else:
                        df = decomposed.select_ir.do_evaluate(
                            *decomposed.select_ir._non_child_args,
                            df,
                            context=ir_context,
                        )
                        table_chunk = TableChunk.from_pylibcudf_table(
                            df.table, df.stream, exclusive_view=True
                        )
                        await ch_out.data.send(
                            context, Message(sequence_num, table_chunk)
                        )
                    sequence_num += 1

    # Handle post-allgather if needed
    if post_allgather is not None:
        post_allgather.insert_finished()
        stream = ir_context.get_cuda_stream()
        df = decomposed.select_ir.do_evaluate(
            *decomposed.select_ir._non_child_args,
            decomposed.reduction_ir.do_evaluate(
                *decomposed.reduction_ir._non_child_args,
                DataFrame.from_table(
                    await post_allgather.extract_concatenated(stream),
                    list(decomposed.reduction_ir.schema.keys()),
                    list(decomposed.reduction_ir.schema.values()),
                    stream,
                ),
                context=ir_context,
            ),
            context=ir_context,
        )
        await ch_out.data.send(
            context,
            Message(
                0,
                TableChunk.from_pylibcudf_table(
                    df.table, df.stream, exclusive_view=True
                ),
            ),
        )

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
        # Get groupby key column names
        groupby_key_columns = [ne.name for ne in ir.keys]

        # Receive input metadata
        input_metadata = await ch_in.recv_metadata(context)
        shuffled = input_metadata.partitioned_on == tuple(groupby_key_columns)

        # Strategy 1: Partition-wise groupby (data already shuffled on keys)
        if shuffled:
            await _partitionwise_groupby(
                context, ir, ir_context, ch_out, ch_in, input_metadata
            )
            return

        # Try to decompose the aggregation for multi-phase execution
        need_preconcat = ir.maintain_order
        try:
            decomposed = DecomposedGroupBy.from_groupby(ir)
        except NotImplementedError:
            need_preconcat = True
            decomposed = None

        # Strategy 2: Allgather + local groupby
        if need_preconcat or input_metadata.count == output_count == 1:
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

        # From here on, we need decomposed operations
        assert decomposed is not None

        # Pre-shuffle if needed (e.g., for n_unique)
        schema_keys = list(ir.schema.keys())
        groupby_key_indices = tuple(schema_keys.index(k.name) for k in ir.keys)
        first_chunk: TableChunk | None = None
        pre_shuffle: ShuffleManager | None = None
        my_preshuffle_ids = list(
            range(
                context.comm().rank,
                input_metadata.count,
                context.comm().nranks,
            )
        )
        sample_first_chunk = True

        if decomposed.need_preshuffle:
            pre_shuffle = ShuffleManager(
                context,
                input_metadata.count,
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

            # Sample first chunk from pre-shuffle
            if sample_first_chunk and my_preshuffle_ids:
                stream = ir_context.get_cuda_stream()
                first_chunk = TableChunk.from_pylibcudf_table(
                    await pre_shuffle.extract_chunk(my_preshuffle_ids[0], stream),
                    stream,
                    exclusive_view=True,
                )
        elif sample_first_chunk:
            # Sample first chunk from input channel
            msg = await ch_in.data.recv(context)
            if msg is not None:
                first_chunk = TableChunk.from_message(msg)
                del msg

        # Apply piecewise groupby to first chunk for size estimation
        first_pwise_size: int | None = None
        if first_chunk is not None:
            first_chunk = first_chunk.make_available_and_spill(
                context.br(), allow_overbooking=True
            )
            with opaque_reservation(
                context, first_chunk.data_alloc_size(MemoryType.DEVICE)
            ):
                first_chunk = await asyncio.to_thread(
                    apply_do_evaluate,
                    first_chunk,
                    decomposed.piecewise_ir,
                    ir_context,
                )
            first_pwise_size = first_chunk.data_alloc_size(MemoryType.DEVICE)

        # Strategy 3: Shuffle-based groupby
        if output_count > 1 and not shuffled:
            await _shuffle_groupby(
                context,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                output_count,
                collective_ids.pop(),
                groupby_key_columns,
                first_chunk,
            )
            return

        # Strategy 4: Tree-based groupby
        await _tree_groupby(
            context,
            decomposed,
            ir_context,
            ch_out,
            ch_in,
            input_metadata,
            output_count,
            collective_ids.pop() if collective_ids else None,
            groupby_n_ary,
            target_partition_size,
            first_chunk,
            first_pwise_size,
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
