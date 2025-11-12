# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Join logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Join
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import (
    define_py_node,
    shutdown_on_error,
)
from cudf_polars.experimental.rapidsmpf.shuffle import LocalShuffle
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    Metadata,
    process_children,
)
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair


async def _partition_wise_join(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_left: ChannelPair,
    ch_right: ChannelPair,
    left_metadata: Metadata,
    right_metadata: Metadata,
    num_partitions: int,
    sampled_chunks: dict[str, TableChunk],
) -> None:
    """
    Perform a partition-wise join (both sides already correctly partitioned).

    Parameters
    ----------
    context
        The streaming context.
    ir
        The Join IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    ch_left
        The left input ChannelPair.
    ch_right
        The right input ChannelPair.
    left_metadata
        The left input metadata.
    right_metadata
        The right input metadata.
    num_partitions
        The number of partitions.
    sampled_chunks
        Dictionary with keys "left" and "right" containing sampled chunks.
    """
    # Send output metadata with partitioning preservation
    join_type = ir.options[0]
    if join_type == "Right":
        partitioned_on = right_metadata.partitioned_on
    else:
        partitioned_on = left_metadata.partitioned_on

    output_metadata = Metadata(num_partitions, partitioned_on=partitioned_on)
    await ch_out.send_metadata(context, output_metadata)

    left_child, right_child = ir.children

    # Process aligned chunks - Lineariser ensures in-order delivery
    for seq_num in range(num_partitions):
        # Get left partition (use sampled chunk on first iteration)
        if seq_num == 0:
            left_chunk = sampled_chunks.pop("left").make_available_and_spill(
                context.br(), allow_overbooking=True
            )
        else:
            left_msg = await ch_left.data.recv(context)
            assert left_msg is not None, f"Missing left partition {seq_num}"
            left_chunk = TableChunk.from_message(left_msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )

        # Get right partition (use sampled chunk on first iteration)
        if seq_num == 0:
            right_chunk = sampled_chunks.pop("right").make_available_and_spill(
                context.br(), allow_overbooking=True
            )
        else:
            right_msg = await ch_right.data.recv(context)
            assert right_msg is not None, f"Missing right partition {seq_num}"
            right_chunk = TableChunk.from_message(right_msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )

        # Convert to DataFrames and join
        left_df = DataFrame.from_table(
            left_chunk.table_view(),
            list(left_child.schema.keys()),
            list(left_child.schema.values()),
            left_chunk.stream,
        )
        right_df = DataFrame.from_table(
            right_chunk.table_view(),
            list(right_child.schema.keys()),
            list(right_child.schema.values()),
            right_chunk.stream,
        )

        result = await asyncio.to_thread(
            ir.do_evaluate,
            *ir._non_child_args,
            left_df,
            right_df,
            context=ir_context,
        )

        # Send output chunk
        await ch_out.data.send(
            context,
            Message(
                seq_num,
                TableChunk.from_pylibcudf_table(
                    result.table,
                    result.stream,
                    exclusive_view=True,
                ),
            ),
        )

    await ch_out.data.drain(context)


async def _broadcast_join(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_left: ChannelPair,
    ch_right: ChannelPair,
    broadcast_side: Literal["left", "right"],
    left_metadata: Metadata,
    right_metadata: Metadata,
    sampled_chunks: dict[str, TableChunk],
) -> None:
    """
    Perform a broadcast join (one side is broadcast to all partitions).

    Parameters
    ----------
    context
        The streaming context.
    ir
        The Join IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    ch_left
        The left input ChannelPair.
    ch_right
        The right input ChannelPair.
    broadcast_side
        The side to broadcast.
    left_metadata
        The left input metadata.
    right_metadata
        The right input metadata.
    sampled_chunks
        Dictionary with keys "left" and "right" containing sampled chunks.
    """
    # Determine which side to broadcast and stream
    if broadcast_side == "right":
        small_ch = ch_right
        large_ch = ch_left
        small_child = ir.children[1]
        large_child = ir.children[0]
        chunk_count = left_metadata.count
        partitioned_on = left_metadata.partitioned_on
        small_key, large_key = "right", "left"
    else:
        small_ch = ch_left
        large_ch = ch_right
        small_child = ir.children[0]
        large_child = ir.children[1]
        chunk_count = right_metadata.count
        partitioned_on = (
            right_metadata.partitioned_on if ir.options[0] == "Right" else ()
        )
        small_key, large_key = "left", "right"

    # Send output metadata
    output_metadata = Metadata(chunk_count, partitioned_on=partitioned_on)
    await ch_out.send_metadata(context, output_metadata)

    # Collect small-side chunks (including sampled chunk)
    small_chunks = [
        sampled_chunks.pop(small_key).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
    ]
    while (msg := await small_ch.data.recv(context)) is not None:
        small_chunks.append(
            TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            )
        )

    small_dfs = [
        DataFrame.from_table(
            small_chunk.table_view(),
            list(small_child.schema.keys()),
            list(small_child.schema.values()),
            small_chunk.stream,
        )
        for small_chunk in small_chunks
    ]

    if ir.options[0] != "Inner":
        # TODO: Use local repartitioning for non-inner joins
        small_dfs = [_concat(*small_dfs, context=ir_context)]

    # Helper to join one large chunk with all small chunks
    async def join_and_send(seq_num: int, large_chunk: TableChunk) -> None:
        large_df = DataFrame.from_table(
            large_chunk.table_view(),
            list(large_child.schema.keys()),
            list(large_child.schema.values()),
            large_chunk.stream,
        )

        # Perform the join
        df = _concat(
            *[
                (
                    await asyncio.to_thread(
                        ir.do_evaluate,
                        *ir._non_child_args,
                        *(
                            [large_df, small_df]
                            if broadcast_side == "right"
                            else [small_df, large_df]
                        ),
                        context=ir_context,
                    )
                )
                for small_df in small_dfs
            ],
            context=ir_context,
        )

        # Send output chunk
        await ch_out.data.send(
            context,
            Message(
                seq_num,
                TableChunk.from_pylibcudf_table(
                    df.table,
                    df.stream,
                    exclusive_view=True,
                ),
            ),
        )

    # Process sampled large chunk first, then stream remaining chunks
    await join_and_send(
        0,
        sampled_chunks.pop(large_key).make_available_and_spill(
            context.br(), allow_overbooking=True
        ),
    )

    while (msg := await large_ch.data.recv(context)) is not None:
        await join_and_send(
            msg.sequence_number,
            TableChunk.from_message(msg).make_available_and_spill(
                context.br(), allow_overbooking=True
            ),
        )

    await ch_out.data.drain(context)


async def _shuffle_join(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_left: ChannelPair,
    ch_right: ChannelPair,
    num_partitions: int,
    left_join_keys: tuple[str, ...],
    right_join_keys: tuple[str, ...],
    left_schema_keys: list[str],
    right_schema_keys: list[str],
    sampled_chunks: dict[str, TableChunk],
    *,
    left_partitioned: bool,
    right_partitioned: bool,
) -> None:
    """
    Perform a shuffle-based join (one or both sides are shuffled on join keys).

    Parameters
    ----------
    context
        The streaming context.
    ir
        The Join IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    ch_left
        The left input ChannelPair.
    ch_right
        The right input ChannelPair.
    num_partitions
        The number of partitions to shuffle into.
    left_join_keys
        The left join key column names.
    right_join_keys
        The right join key column names.
    left_schema_keys
        The left table schema column names.
    right_schema_keys
        The right table schema column names.
    sampled_chunks
        Dictionary with keys "left" and "right" containing sampled chunks.
    left_partitioned
        Whether the left side is already correctly partitioned.
    right_partitioned
        Whether the right side is already correctly partitioned.
    """
    # Send output metadata with join-type-specific partitioning preservation
    join_type = ir.options[0]

    # Preserve partitioning from the side that dictates output cardinality/order
    if join_type == "Right":
        partitioned_on = right_join_keys
    else:
        partitioned_on = left_join_keys

    output_metadata = Metadata(num_partitions, partitioned_on=partitioned_on)
    await ch_out.send_metadata(context, output_metadata)

    left_child, right_child = ir.children

    # Create shuffle contexts for sides that need it (None for pre-partitioned sides)
    left_shuffle = (
        None
        if left_partitioned
        else LocalShuffle(
            context,
            num_partitions,
            tuple(left_schema_keys.index(key) for key in left_join_keys),
        )
    )
    right_shuffle = (
        None
        if right_partitioned
        else LocalShuffle(
            context,
            num_partitions,
            tuple(right_schema_keys.index(key) for key in right_join_keys),
        )
    )

    # Helper to get context manager (shuffle or nullcontext)
    left_ctx = left_shuffle if left_shuffle else nullcontext()
    right_ctx = right_shuffle if right_shuffle else nullcontext()

    with left_ctx as left_input, right_ctx as right_input:
        # Phase 1: Insert chunks into shuffles that need it
        if left_shuffle:
            assert left_input is not None  # Type narrowing
            # Insert the sampled chunk we already received
            left_input.insert_chunk(sampled_chunks.pop("left"))
            # Insert remaining chunks
            while (msg := await ch_left.data.recv(context)) is not None:
                left_chunk = TableChunk.from_message(msg)
                left_input.insert_chunk(left_chunk)

        if right_shuffle:
            assert right_input is not None  # Type narrowing
            # Insert the sampled chunk we already received
            right_input.insert_chunk(sampled_chunks.pop("right"))
            # Insert remaining chunks
            while (msg := await ch_right.data.recv(context)) is not None:
                right_chunk = TableChunk.from_message(msg)
                right_input.insert_chunk(right_chunk)

        # Phase 2: Join partitions
        # Lineariser ensures in-order delivery, so we can receive chunks directly
        for partition_id in range(num_partitions):
            # Get left partition
            if left_shuffle:
                assert left_input is not None  # Type narrowing
                stream = ir_context.get_cuda_stream()
                left_table = left_input.extract_chunk(partition_id, stream)
                left_stream = stream
            else:
                # Pre-partitioned left - receive in order (use sampled chunk on first iteration)
                if partition_id == 0:
                    left_chunk = sampled_chunks.pop("left")
                else:
                    left_msg = await ch_left.data.recv(context)
                    assert left_msg is not None, (
                        f"Missing left partition {partition_id}"
                    )
                    left_chunk = TableChunk.from_message(
                        left_msg
                    ).make_available_and_spill(context.br(), allow_overbooking=True)
                left_table = left_chunk.table_view()
                left_stream = left_chunk.stream

            # Get right partition
            if right_shuffle:
                assert right_input is not None  # Type narrowing
                stream = ir_context.get_cuda_stream()
                right_table = right_input.extract_chunk(partition_id, stream)
                right_stream = stream
            else:
                # Pre-partitioned right - receive in order (use sampled chunk on first iteration)
                if partition_id == 0:
                    right_chunk = sampled_chunks.pop("right")
                else:
                    right_msg = await ch_right.data.recv(context)
                    assert right_msg is not None, (
                        f"Missing right partition {partition_id}"
                    )
                    right_chunk = TableChunk.from_message(
                        right_msg
                    ).make_available_and_spill(context.br(), allow_overbooking=True)
                right_table = right_chunk.table_view()
                right_stream = right_chunk.stream

            # Convert to DataFrames and join
            left_df = DataFrame.from_table(
                left_table,
                list(left_child.schema.keys()),
                list(left_child.schema.values()),
                left_stream,
            )
            right_df = DataFrame.from_table(
                right_table,
                list(right_child.schema.keys()),
                list(right_child.schema.values()),
                right_stream,
            )

            result = await asyncio.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                left_df,
                right_df,
                context=ir_context,
            )

            # Send output chunk
            await ch_out.data.send(
                context,
                Message(
                    partition_id,
                    TableChunk.from_pylibcudf_table(
                        result.table,
                        result.stream,
                        exclusive_view=True,
                    ),
                ),
            )

        await ch_out.data.drain(context)


def make_join_plan(
    ir: Join,
    left_metadata: Metadata,
    right_metadata: Metadata,
    sampled_chunks: dict[str, TableChunk],
    broadcast_join_limit: int,
    target_partition_size: int,
) -> tuple[
    Literal["partition-wise", "broadcast", "shuffle"],
    Literal["left", "right", None],
    tuple[bool, bool],
    int,
]:
    """Make a join plan for the join operation."""
    left_join_keys = tuple(col.name for col in ir.left_on)
    right_join_keys = tuple(col.name for col in ir.right_on)

    # Calculate output partition count from child metadata
    num_partitions = max(left_metadata.count, right_metadata.count)

    # Check if sides are correctly partitioned (right keys AND matching count)
    left_partitioned = (
        left_metadata.partitioned_on == left_join_keys
        and left_metadata.count == num_partitions
    )
    right_partitioned = (
        right_metadata.partitioned_on == right_join_keys
        and right_metadata.count == num_partitions
    )

    # Check if this is a simple partition-wise join
    if num_partitions == 1 or (left_partitioned and right_partitioned):
        return (
            "partition-wise",
            None,
            (left_partitioned, right_partitioned),
            num_partitions,
        )

    # Not partition-wise join - check for broadcast eligibility
    # Inner: can broadcast either side
    # Left: preserve left (child 0), can broadcast right (child 1)
    # Right: preserve right (child 1), can broadcast left (child 0)
    # Semi/Anti: preserve left (child 0), can broadcast right (child 1) only
    can_broadcast_left = ir.options[0] in ("Inner", "Right")
    can_broadcast_right = ir.options[0] in ("Inner", "Left", "Semi", "Anti")

    # Estimate total size of each side using sampled chunk
    size_left_first = sampled_chunks["left"].data_alloc_size(MemoryType.DEVICE)
    size_right_first = sampled_chunks["right"].data_alloc_size(MemoryType.DEVICE)
    size_left_estimate = size_left_first * left_metadata.count
    size_right_estimate = size_right_first * right_metadata.count

    # Check if either side is small enough to broadcast
    # We broadcast if:
    #   1. Join type allows broadcasting that side
    #   2. Estimated table size is small enough
    #   3. Partition count is small enough
    left_small_enough = (
        can_broadcast_left
        # The estimated size must be small enough
        and size_left_estimate <= 4 * target_partition_size
        # And the partition count must be small enough
        and left_metadata.count <= broadcast_join_limit
    )
    right_small_enough = (
        can_broadcast_right
        # The estimated size must be small enough
        and size_right_estimate <= 4 * target_partition_size
        # And the partition count must be small enough
        and right_metadata.count <= broadcast_join_limit
    )

    if left_small_enough and not right_small_enough:
        # Broadcast left
        return (
            "broadcast",
            "left",
            (False, False),
            left_metadata.count,
        )
    elif right_small_enough and not left_small_enough:
        # Broadcast right
        return (
            "broadcast",
            "right",
            (False, False),
            right_metadata.count,
        )
    elif left_small_enough and right_small_enough:
        # Both sides small enough - broadcast the smaller one (by estimated size)
        broadcast_side: Literal["left", "right"]
        if size_left_estimate <= size_right_estimate:
            broadcast_side = "left"
            num_partitions = left_metadata.count
        else:
            broadcast_side = "right"
            num_partitions = right_metadata.count
        return (
            "broadcast",
            broadcast_side,
            (False, False),
            num_partitions,
        )
    else:
        # Catch-all: shuffle join
        return (
            "shuffle",
            None,
            (left_partitioned, right_partitioned),
            num_partitions,
        )


@define_py_node()
async def join_node(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_left: ChannelPair,
    ch_right: ChannelPair,
    broadcast_join_limit: int,
    target_partition_size: int,
) -> None:
    """
    Unified join node that handles broadcast, shuffle, and partition-wise joins.

    This node inspects the metadata from both input channels at runtime and decides
    whether to perform:
    - A partition-wise join (when both sides are already correctly partitioned)
    - A broadcast join (when one side is small enough)
    - A shuffle-based join (when one or both sides need partitioning on join keys)

    Parameters
    ----------
    context
        The streaming context.
    ir
        The Join IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    ch_left
        The left input ChannelPair.
    ch_right
        The right input ChannelPair.
    broadcast_join_limit
        Maximum partition count for broadcasting a side in a join.
    target_partition_size
        Target partition size.
    """
    async with shutdown_on_error(
        context,
        ch_left.metadata,
        ch_left.data,
        ch_right.metadata,
        ch_right.data,
        ch_out.metadata,
        ch_out.data,
    ):
        # Get join key column names
        left_child, right_child = ir.children
        left_schema_keys = list(left_child.schema.keys())
        right_schema_keys = list(right_child.schema.keys())
        left_join_keys = tuple(col.name for col in ir.left_on)
        right_join_keys = tuple(col.name for col in ir.right_on)

        # Receive metadata to inspect it (concurrently to avoid deadlock)
        left_metadata, right_metadata = await asyncio.gather(
            ch_left.recv_metadata(context),
            ch_right.recv_metadata(context),
        )
        assert isinstance(left_metadata, Metadata), (
            f"Expected Metadata, got {type(left_metadata)}."
        )
        assert isinstance(right_metadata, Metadata), (
            f"Expected Metadata, got {type(right_metadata)}."
        )

        # Receive first data chunk from both sides to sample and make smart decisions
        # Store in dict for easy memory management (references cleared on pop)
        # Receive concurrently to avoid deadlock in nested joins
        first_left_msg, first_right_msg = await asyncio.gather(
            ch_left.data.recv(context),
            ch_right.data.recv(context),
        )
        assert first_left_msg is not None, "Missing first left chunk"
        assert first_right_msg is not None, "Missing first right chunk"
        sampled_chunks = {
            "left": TableChunk.from_message(first_left_msg),
            "right": TableChunk.from_message(first_right_msg),
        }

        (
            join_type,
            broadcast_side,
            partitioned,
            num_partitions,
        ) = make_join_plan(
            ir,
            left_metadata,
            right_metadata,
            sampled_chunks,
            broadcast_join_limit,
            target_partition_size,
        )

        # Decision log
        if join_type == "partition-wise":
            await _partition_wise_join(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left,
                ch_right,
                left_metadata,
                right_metadata,
                num_partitions,
                sampled_chunks,
            )
        elif join_type == "broadcast":
            assert broadcast_side is not None
            await _broadcast_join(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left,
                ch_right,
                broadcast_side,
                left_metadata,
                right_metadata,
                sampled_chunks,
            )
        elif join_type == "shuffle":
            await _shuffle_join(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left,
                ch_right,
                num_partitions,
                left_join_keys,
                right_join_keys,
                left_schema_keys,
                right_schema_keys,
                sampled_chunks,
                left_partitioned=partitioned[0],
                right_partitioned=partitioned[1],
            )


@generate_ir_sub_network.register(Join)
def _(
    ir: Join, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """
    Generate sub-network for Join operation using unified join_node.

    The join_node will inspect metadata at runtime to determine whether to:
    - Perform a partition-wise join (both sides already partitioned)
    - Perform a broadcast join (one side has small partition count and join type allows)
    - Perform a shuffle join (one or both sides need shuffling)
    """
    left, right = ir.children

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Get broadcast_join_limit from config options
    config_options = rec.state["config_options"]
    executor = config_options.executor
    assert executor.name == "streaming", "Join node requires streaming executor"
    broadcast_join_limit = executor.broadcast_join_limit
    target_partition_size = executor.target_partition_size

    # Use unified join_node that decides strategy based on metadata
    nodes[ir] = [
        join_node(
            rec.state["context"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[left].reserve_output_slot(),
            channels[right].reserve_output_slot(),
            broadcast_join_limit,
            target_partition_size,
        )
    ]

    return nodes, channels
