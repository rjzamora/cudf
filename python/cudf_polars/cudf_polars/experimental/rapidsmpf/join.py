# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Join logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Literal

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


async def get_small_table(
    context: Context,
    small_child: IR,
    ch_small: ChannelPair,
) -> list[DataFrame]:
    """
    Get the small-table DataFrame partitions from the small-table ChannelPair.

    Parameters
    ----------
    context
        The rapidsmpf context.
    small_child
        The small-table child IR node.
    ch_small
        The small-table ChannelPair.

    Returns
    -------
    list[DataFrame]
        The small-table DataFrame partitions.
    """
    small_chunks = []
    while (msg := await ch_small.data.recv(context)) is not None:
        small_chunks.append(TableChunk.from_message(msg))
    assert small_chunks, "Empty small side"

    return [
        DataFrame.from_table(
            small_chunk.table_view(),
            list(small_child.schema.keys()),
            list(small_child.schema.values()),
            small_chunk.stream,
        )
        for small_chunk in small_chunks
    ]


# @define_py_node()
# async def broadcast_join_node(
#     context: Context,
#     ir: Join,
#     ir_context: IRExecutionContext,
#     ch_out: ChannelPair,
#     ch_left: ChannelPair,
#     ch_right: ChannelPair,
#     broadcast_side: Literal["left", "right"],
# ) -> None:
#     """
#     Join node for rapidsmpf.

#     Parameters
#     ----------
#     context
#         The rapidsmpf context.
#     ir
#         The Join IR node.
#     ir_context
#         The execution context for the IR node.
#     ch_out
#         The output ChannelPair.
#     ch_left
#         The left input ChannelPair.
#     ch_right
#         The right input ChannelPair.
#     broadcast_side
#         The side to broadcast.
#     """
#     async with shutdown_on_error(
#         context,
#         ch_left.metadata,
#         ch_left.data,
#         ch_right.metadata,
#         ch_right.data,
#         ch_out.metadata,
#         ch_out.data,
#     ):
#         metadata_left = await ch_left.recv_metadata(context)
#         metadata_right = await ch_right.recv_metadata(context)
#         assert isinstance(metadata_left, Metadata), (
#             f"Expected Metadata, got {type(metadata_left)}."
#         )
#         assert isinstance(metadata_right, Metadata), (
#             f"Expected Metadata, got {type(metadata_right)}."
#         )
#         partitioned_on: tuple[str, ...] = ()

#         if broadcast_side == "right":
#             # Broadcast right, stream left
#             small_ch = ch_right
#             large_ch = ch_left
#             small_child = ir.children[1]
#             large_child = ir.children[0]
#             chunk_count = metadata_left.count
#             partitioned_on = metadata_left.partitioned_on
#         else:
#             # Broadcast left, stream right
#             small_ch = ch_left
#             large_ch = ch_right
#             small_child = ir.children[0]
#             large_child = ir.children[1]
#             chunk_count = metadata_right.count
#             if ir.options[0] == "Right":
#                 partitioned_on = metadata_right.partitioned_on

#         # Output count is determined by the large side (streaming side)
#         new_metadata = Metadata(chunk_count, partitioned_on=partitioned_on)
#         await ch_out.send_metadata(context, new_metadata)

#         # Collect small-side chunks
#         small_dfs = await get_small_table(context, small_child, small_ch)
#         if ir.options[0] != "Inner":
#             # TODO: Use local repartitioning for non-inner joins
#             small_dfs = [_concat(*small_dfs, context=ir_context)]

#         # Stream through large side, joining with the small-side
#         while (msg := await large_ch.data.recv(context)) is not None:
#             large_chunk = TableChunk.from_message(msg)
#             seq_num = msg.sequence_number
#             large_df = DataFrame.from_table(
#                 large_chunk.table_view(),
#                 list(large_child.schema.keys()),
#                 list(large_child.schema.values()),
#                 large_chunk.stream,
#             )

#             # Perform the join
#             df = _concat(
#                 *[
#                     (
#                         await asyncio.to_thread(
#                             ir.do_evaluate,
#                             *ir._non_child_args,
#                             *(
#                                 [large_df, small_df]
#                                 if broadcast_side == "right"
#                                 else [small_df, large_df]
#                             ),
#                             context=ir_context,
#                         )
#                     )
#                     for small_df in small_dfs
#                 ],
#                 context=ir_context,
#             )

#             # Send output chunk
#             await ch_out.data.send(
#                 context,
#                 Message(
#                     seq_num,
#                     TableChunk.from_pylibcudf_table(
#                         df.table, df.stream, exclusive_view=True
#                     ),
#                 ),
#             )

#         await ch_out.data.drain(context)


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

    # Process aligned chunks
    staged_left: dict[int, TableChunk] = {}
    staged_right: dict[int, TableChunk] = {}
    left_finished = False
    right_finished = False

    for seq_num, partition_id in enumerate(range(num_partitions)):
        # Get left partition
        while partition_id not in staged_left and not left_finished:
            left_msg = await ch_left.data.recv(context)
            if left_msg is None:
                left_finished = True
                break
            left_chunk = TableChunk.from_message(left_msg)
            staged_left[left_msg.sequence_number] = left_chunk

        if partition_id not in staged_left:
            raise ValueError(
                f"Missing left partition {partition_id}, "
                f"available: {list(staged_left.keys())}"
            )
        left_chunk = staged_left.pop(partition_id)

        # Get right partition
        while partition_id not in staged_right and not right_finished:
            right_msg = await ch_right.data.recv(context)
            if right_msg is None:
                right_finished = True
                break
            right_chunk = TableChunk.from_message(right_msg)
            staged_right[right_msg.sequence_number] = right_chunk

        if partition_id not in staged_right:
            raise ValueError(
                f"Missing right partition {partition_id}, "
                f"available: {list(staged_right.keys())}"
            )
        right_chunk = staged_right.pop(partition_id)

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

    # Check for leftover staged chunks
    if staged_left or staged_right:
        raise RuntimeError(
            f"Leftover staged chunks after join: "
            f"left={list(staged_left.keys())}, right={list(staged_right.keys())}"
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
    """
    # Determine which side to broadcast and stream
    if broadcast_side == "right":
        small_ch = ch_right
        large_ch = ch_left
        small_child = ir.children[1]
        large_child = ir.children[0]
        chunk_count = left_metadata.count
        partitioned_on = left_metadata.partitioned_on
    else:
        small_ch = ch_left
        large_ch = ch_right
        small_child = ir.children[0]
        large_child = ir.children[1]
        chunk_count = right_metadata.count
        partitioned_on = (
            right_metadata.partitioned_on if ir.options[0] == "Right" else ()
        )

    # Send output metadata
    output_metadata = Metadata(chunk_count, partitioned_on=partitioned_on)
    await ch_out.send_metadata(context, output_metadata)

    # Collect small-side chunks
    small_dfs = await get_small_table(context, small_child, small_ch)
    if ir.options[0] != "Inner":
        # TODO: Use local repartitioning for non-inner joins
        small_dfs = [_concat(*small_dfs, context=ir_context)]

    # Stream through large side, joining with the small-side
    while (msg := await large_ch.data.recv(context)) is not None:
        large_chunk = TableChunk.from_message(msg)
        seq_num = msg.sequence_number
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

    await ch_out.data.drain(context)


async def _shuffle_join(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_left: ChannelPair,
    ch_right: ChannelPair,
    left_metadata: Metadata,
    right_metadata: Metadata,
    num_partitions: int,
    left_join_keys: tuple[str, ...],
    right_join_keys: tuple[str, ...],
    left_schema_keys: list[str],
    right_schema_keys: list[str],
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
    left_metadata
        The left input metadata.
    right_metadata
        The right input metadata.
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
            while (msg := await ch_left.data.recv(context)) is not None:
                left_chunk = TableChunk.from_message(msg)
                left_input.insert_chunk(left_chunk)

        if right_shuffle:
            assert right_input is not None  # Type narrowing
            while (msg := await ch_right.data.recv(context)) is not None:
                right_chunk = TableChunk.from_message(msg)
                right_input.insert_chunk(right_chunk)

        # Phase 2: Join partitions
        # Only stage pre-partitioned chunks if they arrive out-of-order
        staged_left: dict[int, TableChunk] = {}
        staged_right: dict[int, TableChunk] = {}
        left_finished = False
        right_finished = False

        for partition_id in range(num_partitions):
            # Get left partition
            if left_shuffle:
                assert left_input is not None  # Type narrowing
                stream = ir_context.get_cuda_stream()
                left_table = left_input.extract_chunk(partition_id, stream)
                left_stream = stream
            else:
                # Pre-partitioned left - receive on-demand, stage if out-of-order
                while partition_id not in staged_left and not left_finished:
                    left_msg = await ch_left.data.recv(context)
                    if left_msg is None:
                        left_finished = True
                        break
                    left_chunk = TableChunk.from_message(left_msg)
                    staged_left[left_msg.sequence_number] = left_chunk

                if partition_id not in staged_left:
                    raise ValueError(
                        f"Missing left partition {partition_id}, "
                        f"available: {list(staged_left.keys())}"
                    )
                left_chunk = staged_left.pop(partition_id)
                left_table = left_chunk.table_view()
                left_stream = left_chunk.stream

            # Get right partition
            if right_shuffle:
                assert right_input is not None  # Type narrowing
                stream = ir_context.get_cuda_stream()
                right_table = right_input.extract_chunk(partition_id, stream)
                right_stream = stream
            else:
                # Pre-partitioned right - receive on-demand, stage if out-of-order
                while partition_id not in staged_right and not right_finished:
                    right_msg = await ch_right.data.recv(context)
                    if right_msg is None:
                        right_finished = True
                        break
                    right_chunk = TableChunk.from_message(right_msg)
                    staged_right[right_msg.sequence_number] = right_chunk

                if partition_id not in staged_right:
                    raise ValueError(
                        f"Missing right partition {partition_id}, "
                        f"available: {list(staged_right.keys())}"
                    )
                right_chunk = staged_right.pop(partition_id)
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

        # Check for leftover staged chunks
        if staged_left or staged_right:
            raise RuntimeError(
                f"Leftover staged chunks after join: "
                f"left={list(staged_left.keys())}, right={list(staged_right.keys())}"
            )

    await ch_out.data.drain(context)


@define_py_node()
async def join_node(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_left: ChannelPair,
    ch_right: ChannelPair,
    broadcast_join_limit: int,
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

        # Receive metadata to inspect it
        left_metadata = await ch_left.recv_metadata(context)
        right_metadata = await ch_right.recv_metadata(context)
        assert isinstance(left_metadata, Metadata), (
            f"Expected Metadata, got {type(left_metadata)}."
        )
        assert isinstance(right_metadata, Metadata), (
            f"Expected Metadata, got {type(right_metadata)}."
        )

        # Get join type to determine valid broadcast sides
        join_type = ir.options[0]

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

        # Decision logic
        if num_partitions == 1:
            # Single partition - use partition-wise join
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
            )
        elif left_partitioned and right_partitioned:
            # Both sides correctly partitioned - use partition-wise join
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
            )
        else:
            # Not partition-wise join - check for broadcast eligibility
            # Inner: can broadcast either side
            # Left: preserve left (child 0), can broadcast right (child 1)
            # Right: preserve right (child 1), can broadcast left (child 0)
            # Semi/Anti: preserve left (child 0), can broadcast right (child 1) only
            can_broadcast_left = join_type in ("Inner", "Right")
            can_broadcast_right = join_type in ("Inner", "Left", "Semi", "Anti")

            # Check if either side is small enough to broadcast
            left_small_enough = (
                can_broadcast_left and left_metadata.count <= broadcast_join_limit
            )
            right_small_enough = (
                can_broadcast_right and right_metadata.count <= broadcast_join_limit
            )

            if left_small_enough and not right_small_enough:
                # Broadcast left
                await _broadcast_join(
                    context,
                    ir,
                    ir_context,
                    ch_out,
                    ch_left,
                    ch_right,
                    "left",
                    left_metadata,
                    right_metadata,
                )
            elif right_small_enough and not left_small_enough:
                # Broadcast right
                await _broadcast_join(
                    context,
                    ir,
                    ir_context,
                    ch_out,
                    ch_left,
                    ch_right,
                    "right",
                    left_metadata,
                    right_metadata,
                )
            elif left_small_enough and right_small_enough:
                # Both sides small enough - broadcast the smaller one
                if left_metadata.count <= right_metadata.count:
                    await _broadcast_join(
                        context,
                        ir,
                        ir_context,
                        ch_out,
                        ch_left,
                        ch_right,
                        "left",
                        left_metadata,
                        right_metadata,
                    )
                else:
                    await _broadcast_join(
                        context,
                        ir,
                        ir_context,
                        ch_out,
                        ch_left,
                        ch_right,
                        "right",
                        left_metadata,
                        right_metadata,
                    )
            else:
                # Neither side is small enough or correctly partitioned - shuffle join
                await _shuffle_join(
                    context,
                    ir,
                    ir_context,
                    ch_out,
                    ch_left,
                    ch_right,
                    left_metadata,
                    right_metadata,
                    num_partitions,
                    left_join_keys,
                    right_join_keys,
                    left_schema_keys,
                    right_schema_keys,
                    left_partitioned=left_partitioned,
                    right_partitioned=right_partitioned,
                )


@generate_ir_sub_network.register(Join)
def _(ir: Join, rec: SubNetGenerator) -> tuple[list[Any], dict[IR, ChannelManager]]:
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

    # Use unified join_node that decides strategy based on metadata
    nodes.append(
        join_node(
            rec.state["context"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[left].reserve_output_slot(),
            channels[right].reserve_output_slot(),
            broadcast_join_limit,
        )
    )

    return nodes, channels
