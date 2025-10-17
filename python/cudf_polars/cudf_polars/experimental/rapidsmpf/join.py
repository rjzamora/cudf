# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Join logic for the RapidsMPF streaming engine."""

from __future__ import annotations

import asyncio
import operator
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.streaming.core.channel import Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import pylibcudf as plc
from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Join
from cudf_polars.experimental.base import ChunkMetadata
from cudf_polars.experimental.rapidsmpf.channel_pair import ChannelPair
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import (
    _aligned_multi_input,
    define_py_node,
    shutdown_on_error,
)
from cudf_polars.experimental.rapidsmpf.shuffle import LocalShuffle
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator


async def get_small_table(
    ctx: Context,
    small_child: IR,
    ch_small: ChannelPair,
) -> list[DataFrame]:
    """
    Get the small-table DataFrame partitions from the small-table ChannelPair.

    Parameters
    ----------
    ctx
        The context.
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
    while (msg := await ch_small.data.recv(ctx)) is not None:
        small_chunks.append(TableChunk.from_message(msg))

    if len(small_chunks) == 0:
        raise ValueError("Empty small side")

    return [
        DataFrame.from_table(
            small_chunk.table_view(),
            list(small_child.schema.keys()),
            list(small_child.schema.values()),
            small_chunk.stream,
        )
        for small_chunk in small_chunks
    ]


@define_py_node()
async def broadcast_join_node(
    ctx: Context,
    ir: Join,
    ch_out: ChannelPair,
    ch_left: ChannelPair,
    ch_right: ChannelPair,
    broadcast_side: Literal["left", "right"],
) -> None:
    """
    Join node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ir
        The Join IR node.
    ch_out
        The output ChannelPair.
    ch_left
        The left input ChannelPair.
    ch_right
        The right input ChannelPair.
    broadcast_side
        The side to broadcast.
    """
    all_channels = [
        ch_left.metadata,
        ch_left.data,
        ch_right.metadata,
        ch_right.data,
        ch_out.metadata,
        ch_out.data,
    ]
    async with shutdown_on_error(ctx, *all_channels):
        # Receive metadata from both sides, merge them
        left_metadata = await ch_left.recv_metadata(ctx)
        right_metadata = await ch_right.recv_metadata(ctx)
        assert isinstance(left_metadata, ChunkMetadata), (
            f"Expected ChunkMetadata, got {type(left_metadata)}."
        )
        assert isinstance(right_metadata, ChunkMetadata), (
            f"Expected ChunkMetadata, got {type(right_metadata)}."
        )
        local_partitioned_on: tuple[str, ...] = ()
        global_partitioned_on: tuple[str, ...] = ()

        if broadcast_side == "right":
            # Broadcast right, stream left
            small_ch = ch_right
            large_ch = ch_left
            small_child = ir.children[1]
            large_child = ir.children[0]
            local_count = left_metadata.local_count
            local_partitioned_on = left_metadata.local_partitioned_on
            global_partitioned_on = left_metadata.global_partitioned_on
        else:
            # Broadcast left, stream right
            small_ch = ch_left
            large_ch = ch_right
            small_child = ir.children[0]
            large_child = ir.children[1]
            local_count = right_metadata.local_count
            if ir.options[0] == "Right":
                local_partitioned_on = right_metadata.local_partitioned_on
                global_partitioned_on = right_metadata.global_partitioned_on

        # Send output metadata
        metadata = ChunkMetadata(
            local_count,
            local_partitioned_on=local_partitioned_on,
            global_partitioned_on=global_partitioned_on,
        )
        await ch_out.send_metadata(ctx, metadata)

        # TODO: Build output partition incrementally?
        small_dfs: list[DataFrame] = []
        get_small_table_fut = asyncio.create_task(
            get_small_table(ctx, small_child, small_ch)
        )

        # Stream through large side, joining with broadcast data
        while (msg := await large_ch.data.recv(ctx)) is not None:
            large_chunk = TableChunk.from_message(msg)
            large_df = DataFrame.from_table(
                large_chunk.table_view(),
                list(large_child.schema.keys()),
                list(large_child.schema.values()),
                large_chunk.stream,
            )

            if not small_dfs:
                small_dfs = list(
                    chain.from_iterable(await asyncio.gather(get_small_table_fut))
                )
                if ir.options[0] != "Inner":
                    # TODO: Use local repartitioning for non-inner joins
                    small_dfs = [_concat(*small_dfs)]

            # Perform the join
            results = [
                (
                    await asyncio.to_thread(
                        ir.do_evaluate,
                        *ir._non_child_args,
                        *(
                            [large_df, small_df]
                            if broadcast_side == "right"
                            else [small_df, large_df]
                        ),
                    )
                ).table
                for small_df in small_dfs
            ]

            # Send output chunk
            build_stream = DEFAULT_STREAM
            await ch_out.data.send(
                ctx,
                Message(
                    TableChunk.from_pylibcudf_table(
                        large_chunk.sequence_number,
                        (
                            results[0]
                            if len(results) == 1
                            else plc.concatenate.concatenate(results, build_stream)
                        ),
                        build_stream,
                        exclusive_view=True,
                    )
                ),
            )

        # Drain the output data channel
        await ch_out.data.drain(ctx)


@define_py_node()
async def join_node(
    ctx: Context,
    ir: Join,
    ch_out: ChannelPair,
    ch_left: ChannelPair,
    ch_right: ChannelPair,
    target_partition_size: int,
    broadcast_join_limit: int,
) -> None:
    """
    Unified join node that handles both shuffle-based and broadcast joins.

    This node inspects the metadata from both input channels and decides
    whether to perform a shuffle-based join (when both sides need partitioning)
    or a broadcast join (when one side is already suitable).

    Parameters
    ----------
    ctx
        The streaming context.
    ir
        The Join IR node.
    ch_out
        The output ChannelPair.
    ch_left
        The left input ChannelPair.
    ch_right
        The right input ChannelPair.
    target_partition_size
        Target partition size for determining output partition count.
    broadcast_join_limit
        Maximum size (in bytes) for broadcasting a side in a join.
    """
    all_channels = [
        ch_left.metadata,
        ch_left.data,
        ch_right.metadata,
        ch_right.data,
        ch_out.metadata,
        ch_out.data,
    ]
    async with shutdown_on_error(ctx, *all_channels):
        # Peek at metadata to decide join strategy (without consuming from channels)
        # For partition-wise join, let _aligned_multi_input handle metadata
        # For other joins, we need to receive it here to make decisions

        # Get join key column names
        left_child, right_child = ir.children
        left_schema_keys = list(left_child.schema.keys())
        right_schema_keys = list(right_child.schema.keys())
        left_join_keys = tuple(col.name for col in ir.left_on)
        right_join_keys = tuple(col.name for col in ir.right_on)

        # Receive metadata to inspect it
        left_metadata = await ch_left.recv_metadata(ctx)
        right_metadata = await ch_right.recv_metadata(ctx)
        assert isinstance(left_metadata, ChunkMetadata), (
            f"Expected ChunkMetadata, got {type(left_metadata)}."
        )
        assert isinstance(right_metadata, ChunkMetadata), (
            f"Expected ChunkMetadata, got {type(right_metadata)}."
        )

        # Get join type to determine valid broadcast sides
        join_type = ir.options[0]  # "Inner", "Left", "Right", "Outer", etc.

        # Calculate output partition count from child metadata
        num_partitions = max(left_metadata.local_count, right_metadata.local_count)

        # Check if sides are correctly partitioned (right keys AND matching count)
        left_partitioned = (
            left_metadata.local_partitioned_on == left_join_keys
            and left_metadata.local_count == num_partitions
        )
        right_partitioned = (
            right_metadata.local_partitioned_on == right_join_keys
            and right_metadata.local_count == num_partitions
        )

        # Determine join strategy based on metadata
        if left_partitioned and right_partitioned:
            # Both sides already partitioned - send output metadata and do aligned join
            # Preserve partitioning from the side that dictates output cardinality/order
            # Determine which side's metadata to preserve
            if join_type == "Right":
                # Right join or maintain_order from right side
                local_partitioned_on = right_metadata.local_partitioned_on
                global_partitioned_on = right_metadata.global_partitioned_on
            else:
                # Left/Inner/Outer join or maintain_order from left side (or none)
                local_partitioned_on = left_metadata.local_partitioned_on
                global_partitioned_on = left_metadata.global_partitioned_on

            output_metadata = ChunkMetadata(
                num_partitions,
                local_partitioned_on=local_partitioned_on,
                global_partitioned_on=global_partitioned_on,
            )
            await ch_out.send_metadata(ctx, output_metadata)

            # Do aligned chunk-wise join using shared helper (metadata already handled)
            await _aligned_multi_input(
                ctx,
                ir,
                ch_out,
                (ch_left, ch_right),
                bcast_indices=[],
            )
        else:
            # Check if we can/should do a broadcast join
            # For Left join: keep all left rows → broadcast right
            # For Right join: keep all right rows → broadcast left
            # For Inner join: can broadcast either
            can_broadcast_left = join_type in ("Right", "Inner")
            can_broadcast_right = join_type in ("Left", "Inner")

            # Decide whether to broadcast based on partition counts and join type
            should_broadcast_left = (
                can_broadcast_left and left_metadata.local_count <= broadcast_join_limit
            )
            should_broadcast_right = (
                can_broadcast_right
                and right_metadata.local_count <= broadcast_join_limit
            )

            if should_broadcast_left or should_broadcast_right:
                # Choose which side to broadcast
                if should_broadcast_left and should_broadcast_right:
                    # Both are single partition - choose smaller one
                    # For now, just use left (could be enhanced with size heuristics)
                    broadcast_side: Literal["left", "right"] = "left"
                elif should_broadcast_left:
                    broadcast_side = "left"
                else:
                    broadcast_side = "right"

                await _broadcast_join(
                    ctx,
                    ir,
                    ch_out,
                    ch_left,
                    ch_right,
                    left_metadata,
                    right_metadata,
                    broadcast_side,
                )
            else:
                # At least one side needs shuffling - perform shuffle join
                await _shuffle_join(
                    ctx,
                    ir,
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


async def _broadcast_join(
    ctx: Context,
    ir: Join,
    ch_out: ChannelPair,
    ch_left: ChannelPair,
    ch_right: ChannelPair,
    left_metadata: ChunkMetadata,
    right_metadata: ChunkMetadata,
    broadcast_side: Literal["left", "right"],
) -> None:
    """
    Perform a broadcast join (one side is broadcast to all partitions).

    This is essentially the same logic as broadcast_join_node, extracted as a helper.

    Parameters
    ----------
    ctx
        The streaming context.
    ir
        The Join IR node.
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
    broadcast_side
        The side to broadcast.
    """
    local_partitioned_on: tuple[str, ...] = ()
    global_partitioned_on: tuple[str, ...] = ()

    if broadcast_side == "right":
        # Broadcast right, stream left
        small_ch = ch_right
        large_ch = ch_left
        small_child = ir.children[1]
        large_child = ir.children[0]
        local_count = left_metadata.local_count
        local_partitioned_on = left_metadata.local_partitioned_on
        global_partitioned_on = left_metadata.global_partitioned_on
    else:
        # Broadcast left, stream right
        small_ch = ch_left
        large_ch = ch_right
        small_child = ir.children[0]
        large_child = ir.children[1]
        local_count = right_metadata.local_count
        if ir.options[0] == "Right":
            local_partitioned_on = right_metadata.local_partitioned_on
            global_partitioned_on = right_metadata.global_partitioned_on

    # Send output metadata
    metadata = ChunkMetadata(
        local_count,
        local_partitioned_on=local_partitioned_on,
        global_partitioned_on=global_partitioned_on,
    )
    await ch_out.send_metadata(ctx, metadata)

    # Get small table
    small_dfs: list[DataFrame] = []
    get_small_table_fut = asyncio.create_task(
        get_small_table(ctx, small_child, small_ch)
    )

    # Stream through large side, joining with broadcast data
    while (msg := await large_ch.data.recv(ctx)) is not None:
        large_chunk = TableChunk.from_message(msg)
        large_df = DataFrame.from_table(
            large_chunk.table_view(),
            list(large_child.schema.keys()),
            list(large_child.schema.values()),
            large_chunk.stream,
        )

        if not small_dfs:
            small_dfs = list(
                chain.from_iterable(await asyncio.gather(get_small_table_fut))
            )
            if ir.options[0] != "Inner":
                # TODO: Use local repartitioning for non-inner joins
                small_dfs = [_concat(*small_dfs)]

        # Perform the join
        results = [
            (
                await asyncio.to_thread(
                    ir.do_evaluate,
                    *ir._non_child_args,
                    *(
                        [large_df, small_df]
                        if broadcast_side == "right"
                        else [small_df, large_df]
                    ),
                )
            ).table
            for small_df in small_dfs
        ]

        # Send output chunk
        build_stream = DEFAULT_STREAM
        await ch_out.data.send(
            ctx,
            Message(
                TableChunk.from_pylibcudf_table(
                    large_chunk.sequence_number,
                    (
                        results[0]
                        if len(results) == 1
                        else plc.concatenate.concatenate(results, build_stream)
                    ),
                    build_stream,
                    exclusive_view=True,
                )
            ),
        )

    # Drain the output data channel
    await ch_out.data.drain(ctx)


async def _shuffle_join(
    ctx: Context,
    ir: Join,
    ch_out: ChannelPair,
    ch_left: ChannelPair,
    ch_right: ChannelPair,
    left_metadata: ChunkMetadata,
    right_metadata: ChunkMetadata,
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
    ctx
        The streaming context.
    ir
        The Join IR node.
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
    join_type = ir.options[0]  # "Inner", "Left", "Right", "Outer", etc.

    # Preserve partitioning from the side that dictates output cardinality/order
    if join_type == "Right":
        # Right join preserves right side
        local_partitioned_on = right_join_keys
        global_partitioned_on = right_metadata.global_partitioned_on
    else:
        # Left/Inner/Outer join preserves left side
        local_partitioned_on = left_join_keys
        global_partitioned_on = left_metadata.global_partitioned_on

    output_metadata = ChunkMetadata(
        num_partitions,
        local_partitioned_on=local_partitioned_on,
        global_partitioned_on=global_partitioned_on,
    )
    await ch_out.send_metadata(ctx, output_metadata)

    left_child, right_child = ir.children

    # Create shuffle contexts for sides that need it (None for pre-partitioned sides)
    left_shuffle = (
        None
        if left_partitioned
        else LocalShuffle(
            ctx,
            num_partitions,
            tuple(left_schema_keys.index(key) for key in left_join_keys),
        )
    )
    right_shuffle = (
        None
        if right_partitioned
        else LocalShuffle(
            ctx,
            num_partitions,
            tuple(right_schema_keys.index(key) for key in right_join_keys),
        )
    )

    # Helper to get context manager (shuffle or nullcontext)
    from contextlib import nullcontext

    left_ctx = left_shuffle if left_shuffle else nullcontext()
    right_ctx = right_shuffle if right_shuffle else nullcontext()

    with left_ctx as left_input, right_ctx as right_input:
        # Phase 1: Insert chunks into shuffles that need it
        if left_shuffle:
            # Shuffle left - insert all chunks
            assert left_input is not None  # Type narrowing for mypy
            while (msg := await ch_left.data.recv(ctx)) is not None:
                left_chunk = TableChunk.from_message(msg)
                left_input.insert_chunk(left_chunk.table_view())

        if right_shuffle:
            # Shuffle right - insert all chunks
            assert right_input is not None  # Type narrowing for mypy
            while (msg := await ch_right.data.recv(ctx)) is not None:
                right_chunk = TableChunk.from_message(msg)
                right_input.insert_chunk(right_chunk.table_view())

        # Phase 2: Join partitions
        # Only stage pre-partitioned chunks if they arrive out-of-order
        staged_left: dict[int, TableChunk] = {}
        staged_right: dict[int, TableChunk] = {}
        left_finished = False
        right_finished = False

        for partition_id in range(num_partitions):
            # Get left partition
            if left_shuffle:
                assert left_input is not None  # Type narrowing for mypy
                left_table = left_input.extract_chunk(partition_id)
                left_stream = left_input.stream
            else:
                # Pre-partitioned left - receive on-demand, stage if out-of-order
                while partition_id not in staged_left and not left_finished:
                    left_msg = await ch_left.data.recv(ctx)
                    if left_msg is None:
                        left_finished = True
                        break
                    left_chunk = TableChunk.from_message(left_msg)
                    staged_left[left_chunk.sequence_number] = left_chunk

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
                assert right_input is not None  # Type narrowing for mypy
                right_table = right_input.extract_chunk(partition_id)
                right_stream = right_input.stream
            else:
                # Pre-partitioned right - receive on-demand, stage if out-of-order
                while partition_id not in staged_right and not right_finished:
                    right_msg = await ch_right.data.recv(ctx)
                    if right_msg is None:
                        right_finished = True
                        break
                    right_chunk = TableChunk.from_message(right_msg)
                    staged_right[right_chunk.sequence_number] = right_chunk

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
            )

            # Send output chunk
            await ch_out.data.send(
                ctx,
                Message(
                    TableChunk.from_pylibcudf_table(
                        partition_id,
                        result.table,
                        left_stream,
                        exclusive_view=True,
                    )
                ),
            )

        # Check for leftover staged chunks
        if staged_left or staged_right:
            raise RuntimeError(
                f"Leftover staged chunks after join: "
                f"left={list(staged_left.keys())}, right={list(staged_right.keys())}"
            )

    # Drain the output data channel
    await ch_out.data.drain(ctx)


@generate_ir_sub_network.register(Join)
def _(
    ir: Join, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, list[Any]]]:
    """
    Generate sub-network for Join operation using unified join_node.

    The join_node will inspect metadata at runtime to determine whether to:
    - Perform a partition-wise join (both sides already partitioned)
    - Perform a broadcast join (one side has single partition and join type allows)
    - Perform a shuffle join (one or both sides need shuffling)
    """
    left, right = ir.children

    # Process children
    nodes: dict[IR, list[Any]] = {}
    channels: dict[IR, list[Any]] = {}
    if ir.children:
        _nodes, _channels = zip(*(rec(c) for c in ir.children), strict=True)
        nodes = reduce(operator.or_, _nodes)
        channels = reduce(operator.or_, _channels)

    # Create output ChannelPair
    channels[ir] = [ChannelPair.create()]

    # Get parameters from config options
    config_options = rec.state["config_options"]
    executor = config_options.executor
    # These attributes only exist on StreamingExecutor
    target_partition_size = getattr(executor, "target_partition_size", None)
    broadcast_join_limit = getattr(executor, "broadcast_join_limit", 1)

    # Use unified join_node that decides strategy based on metadata
    nodes[ir] = [
        join_node(
            rec.state["ctx"],
            ir,
            channels[ir][0],
            channels[left].pop(),
            channels[right].pop(),
            target_partition_size,
            broadcast_join_limit,
        )
    ]

    return nodes, channels
