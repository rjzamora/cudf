# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Join logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Join
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import (
    default_node_multi,
    define_py_node,
    shutdown_on_error,
)
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    allgather_reduce,
    chunk_to_frame,
    empty_table_chunk,
    opaque_reservation,
    process_children,
    recv_metadata,
    remap_partitioning,
    send_metadata,
)
from cudf_polars.experimental.utils import _concat
from cudf_polars.utils.config import StreamingExecutor

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.channel_metadata import Partitioning

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.base import Profiler
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator


@define_py_node()
async def broadcast_join_node(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    broadcast_side: Literal["left", "right"],
    collective_id: int,
    target_partition_size: int,
) -> None:
    """
    Join node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The Join IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output Channel[TableChunk].
    ch_left
        The left input Channel[TableChunk].
    ch_right
        The right input Channel[TableChunk].
    broadcast_side
        The side to broadcast.
    collective_id
        Pre-allocated collective ID for this operation.
    target_partition_size
        The target partition size in bytes.
    """
    async with shutdown_on_error(context, ch_left, ch_right, ch_out):
        # Receive metadata.
        left_metadata, right_metadata = await asyncio.gather(
            recv_metadata(ch_left, context),
            recv_metadata(ch_right, context),
        )

        partitioning: Partitioning | None = None
        if broadcast_side == "right":
            # Broadcast right, stream left
            small_ch = ch_right
            large_ch = ch_left
            small_child = ir.children[1]
            large_child = ir.children[0]
            # Preserve left-side partitioning metadata
            local_count = left_metadata.local_count
            # Remap partitioning from child schema to output schema
            partitioning = remap_partitioning(
                left_metadata.partitioning, large_child.schema, ir.schema
            )
            # Check if the right-side is already broadcasted
            small_duplicated = right_metadata.duplicated
        else:
            # Broadcast left, stream right
            small_ch = ch_left
            large_ch = ch_right
            small_child = ir.children[0]
            large_child = ir.children[1]
            # Preserve right-side partitioning metadata
            local_count = right_metadata.local_count
            if ir.options[0] == "Right":
                # Remap partitioning from child schema to output schema
                partitioning = remap_partitioning(
                    right_metadata.partitioning, large_child.schema, ir.schema
                )
            # Check if the right-side is already broadcasted
            small_duplicated = left_metadata.duplicated

        # Determine which metadata belongs to the large side
        large_metadata = left_metadata if broadcast_side == "right" else right_metadata

        # Allgather is a collective - all ranks must participate even with no local data
        need_allgather = context.comm().nranks > 1 and not small_duplicated

        # The result is duplicated if:
        # - The small side is/will be duplicated (already duplicated OR will be AllGathered)
        # - AND the large side is already duplicated
        output_duplicated = (
            small_duplicated or need_allgather
        ) and large_metadata.duplicated

        # Send metadata.
        output_metadata = ChannelMetadata(
            local_count=local_count,
            partitioning=partitioning,
            duplicated=output_duplicated,
        )
        await send_metadata(ch_out, context, output_metadata)

        # Collect small-side (may be empty if no data received)
        small_chunks: list[TableChunk] = []
        small_size = 0
        while (msg := await small_ch.recv(context)) is not None:
            small_chunks.append(
                TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
            )
            del msg
            small_size += small_chunks[-1].data_alloc_size(MemoryType.DEVICE)

        if need_allgather:
            allgather = AllGatherManager(context, collective_id)
            for s_id in range(len(small_chunks)):
                allgather.insert(s_id, small_chunks.pop(0))
            allgather.insert_finished()
            stream = ir_context.get_cuda_stream()
            # extract_concatenated returns a plc.Table, not a TableChunk
            small_dfs = [
                DataFrame.from_table(
                    await allgather.extract_concatenated(stream),
                    list(small_child.schema.keys()),
                    list(small_child.schema.values()),
                    stream,
                )
            ]
        elif len(small_chunks) > 1 and (
            ir.options[0] != "Inner" or small_size < target_partition_size
        ):
            # Pre-concat for non-inner joins, otherwise
            # we need a local shuffle, and face additional
            # memory pressure anyway.
            small_dfs = [
                _concat(
                    *[chunk_to_frame(chunk, small_child) for chunk in small_chunks],
                    context=ir_context,
                )
            ]
            small_chunks.clear()  # small_dfs is not a view of small_chunks anymore
        else:
            small_dfs = [
                chunk_to_frame(small_chunk, small_child) for small_chunk in small_chunks
            ]

        # Stream through large side, joining with the small-side
        seq_num = 0
        large_chunk_processed = False
        receiving_large_chunks = True
        while receiving_large_chunks:
            msg = await large_ch.recv(context)
            if msg is None:
                receiving_large_chunks = False
                if large_chunk_processed:
                    # Normal exit - We've processed all large-table data
                    break
                elif small_dfs:
                    # We received small-table data, but no large-table data.
                    # This may never happen, but we can handle it by generating
                    # an empty large-table chunk
                    stream = ir_context.get_cuda_stream()
                    large_chunk = empty_table_chunk(large_child, context, stream)
                else:
                    # We received no data for either the small or large table.
                    # Drain the output channel and return
                    await ch_out.drain(context)
                    return
            else:
                large_chunk_processed = True
                large_chunk = TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
                seq_num = msg.sequence_number
                del msg

            large_df = DataFrame.from_table(
                large_chunk.table_view(),
                list(large_child.schema.keys()),
                list(large_child.schema.values()),
                large_chunk.stream,
            )

            # Lazily create empty small table if small_dfs is empty
            if not small_dfs:
                stream = ir_context.get_cuda_stream()
                empty_small_chunk = empty_table_chunk(small_child, context, stream)
                small_dfs = [chunk_to_frame(empty_small_chunk, small_child)]

            large_chunk_size = large_chunk.data_alloc_size(MemoryType.DEVICE)
            input_bytes = large_chunk_size + small_size
            with opaque_reservation(context, input_bytes):
                df = _concat(
                    *[
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
                        for small_df in small_dfs
                    ],
                    context=ir_context,
                )

                # Send output chunk
                await ch_out.send(
                    context,
                    Message(
                        seq_num,
                        TableChunk.from_pylibcudf_table(
                            df.table, df.stream, exclusive_view=True
                        ),
                    ),
                )
                del df, large_df, large_chunk

        del small_dfs, small_chunks
        await ch_out.drain(context)


async def _broadcast_join(
    context: Context,
    ir: Join,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    left_initial_chunks: list[TableChunk],
    right_initial_chunks: list[TableChunk],
    broadcast_side: Literal["left", "right"],
    collective_id: int,
    target_partition_size: int,
    profiler: Profiler | None = None,
) -> None:
    """
    Execute a broadcast join after initial sampling.

    The small side is gathered (if not already duplicated) and joined
    with each chunk from the large side.
    """
    n_rows_out = 0
    left, right = ir.children

    if broadcast_side == "right":
        small_ch, large_ch = ch_right, ch_left
        small_child, large_child = right, left
        small_metadata, large_metadata = right_metadata, left_metadata
        small_initial_chunks = right_initial_chunks
        large_initial_chunks = left_initial_chunks
        local_count = left_metadata.local_count
        partitioning: Partitioning | None = remap_partitioning(
            left_metadata.partitioning, large_child.schema, ir.schema
        )
    else:
        small_ch, large_ch = ch_left, ch_right
        small_child, large_child = left, right
        small_metadata, large_metadata = left_metadata, right_metadata
        small_initial_chunks = left_initial_chunks
        large_initial_chunks = right_initial_chunks
        local_count = right_metadata.local_count
        partitioning = (
            remap_partitioning(
                right_metadata.partitioning, large_child.schema, ir.schema
            )
            if ir.options[0] == "Right"
            else None
        )

    small_duplicated = small_metadata.duplicated
    need_allgather = context.comm().nranks > 1 and not small_duplicated
    output_duplicated = (
        small_duplicated or need_allgather
    ) and large_metadata.duplicated

    metadata_out = ChannelMetadata(
        local_count=local_count,
        partitioning=partitioning,
        duplicated=output_duplicated,
    )
    await send_metadata(ch_out, context, metadata_out)

    # Collect remaining small-side chunks
    small_chunks: list[TableChunk] = list(small_initial_chunks)
    small_size = sum(c.data_alloc_size(MemoryType.DEVICE) for c in small_chunks)
    while (msg := await small_ch.recv(context)) is not None:
        chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        small_chunks.append(chunk)
        small_size += chunk.data_alloc_size(MemoryType.DEVICE)
        del msg

    # AllGather small side if needed
    if need_allgather:
        allgather = AllGatherManager(context, collective_id)
        for s_id in range(len(small_chunks)):
            allgather.insert(s_id, small_chunks.pop(0))
        allgather.insert_finished()
        stream = ir_context.get_cuda_stream()
        small_dfs = [
            DataFrame.from_table(
                await allgather.extract_concatenated(stream),
                list(small_child.schema.keys()),
                list(small_child.schema.values()),
                stream,
            )
        ]
    elif len(small_chunks) > 1 and (
        ir.options[0] != "Inner" or small_size < target_partition_size
    ):
        small_dfs = [
            _concat(
                *[chunk_to_frame(chunk, small_child) for chunk in small_chunks],
                context=ir_context,
            )
        ]
        small_chunks.clear()
    else:
        small_dfs = [chunk_to_frame(c, small_child) for c in small_chunks]

    # Stream through large side with initial chunks first
    large_chunk_processed = False

    # Process initial large chunks first
    for seq_num, chunk in enumerate(large_initial_chunks):
        large_chunk_processed = True
        if not small_dfs:
            stream = ir_context.get_cuda_stream()
            empty_small = empty_table_chunk(small_child, context, stream)
            small_dfs = [chunk_to_frame(empty_small, small_child)]

        large_df = DataFrame.from_table(
            chunk.table_view(),
            list(large_child.schema.keys()),
            list(large_child.schema.values()),
            chunk.stream,
        )
        large_chunk_size = chunk.data_alloc_size(MemoryType.DEVICE)
        input_bytes = large_chunk_size + small_size
        with opaque_reservation(context, input_bytes):
            df = _concat(
                *[
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
                    for small_df in small_dfs
                ],
                context=ir_context,
            )
            n_rows_out += df.num_rows
            await ch_out.send(
                context,
                Message(
                    seq_num,
                    TableChunk.from_pylibcudf_table(
                        df.table, df.stream, exclusive_view=True
                    ),
                ),
            )
            del df, large_df

    # Process remaining large chunks from channel
    while (msg := await large_ch.recv(context)) is not None:
        large_chunk_processed = True
        large_chunk = TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        msg_seq = msg.sequence_number
        del msg

        if not small_dfs:
            stream = ir_context.get_cuda_stream()
            empty_small = empty_table_chunk(small_child, context, stream)
            small_dfs = [chunk_to_frame(empty_small, small_child)]

        large_df = DataFrame.from_table(
            large_chunk.table_view(),
            list(large_child.schema.keys()),
            list(large_child.schema.values()),
            large_chunk.stream,
        )
        large_chunk_size = large_chunk.data_alloc_size(MemoryType.DEVICE)
        input_bytes = large_chunk_size + small_size
        with opaque_reservation(context, input_bytes):
            df = _concat(
                *[
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
                    for small_df in small_dfs
                ],
                context=ir_context,
            )
            n_rows_out += df.num_rows
            await ch_out.send(
                context,
                Message(
                    msg_seq,
                    TableChunk.from_pylibcudf_table(
                        df.table, df.stream, exclusive_view=True
                    ),
                ),
            )
            del df, large_df, large_chunk

    # Handle edge case: no large-side data received
    if not large_chunk_processed and small_dfs:
        stream = ir_context.get_cuda_stream()
        large_chunk = empty_table_chunk(large_child, context, stream)
        large_df = chunk_to_frame(large_chunk, large_child)
        with opaque_reservation(context, small_size):
            df = _concat(
                *[
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
                    for small_df in small_dfs
                ],
                context=ir_context,
            )
            n_rows_out += df.num_rows
            await ch_out.send(
                context,
                Message(
                    0,
                    TableChunk.from_pylibcudf_table(
                        df.table, df.stream, exclusive_view=True
                    ),
                ),
            )
            del df, large_df

    del small_dfs, small_chunks
    if profiler is not None:
        profiler.row_count[ir] += n_rows_out
    await ch_out.drain(context)


async def _shuffle_join(
    context: Context,
    ir: Join,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    left_initial_chunks: list[TableChunk],
    right_initial_chunks: list[TableChunk],
    output_count: int,
    left_collective_id: int,
    right_collective_id: int,
    profiler: Profiler | None = None,
) -> None:
    """
    Execute a shuffle (hash) join after initial sampling.

    Both sides are shuffled by their join keys, then partition-wise joins
    are performed.
    """
    from rapidsmpf.streaming.cudf.channel_metadata import HashScheme, Partitioning

    from cudf_polars.experimental.rapidsmpf.collectives.shuffle import ShuffleManager

    n_rows_out = 0
    left, right = ir.children
    nranks = context.comm().nranks
    modulus = nranks * output_count

    # Get key column indices for both sides
    left_schema_keys = list(left.schema.keys())
    right_schema_keys = list(right.schema.keys())
    left_key_indices = tuple(left_schema_keys.index(expr.name) for expr in ir.left_on)
    right_key_indices = tuple(
        right_schema_keys.index(expr.name) for expr in ir.right_on
    )

    # Send output metadata
    # Output partitioning depends on join type
    output_key_indices: tuple[int, ...] = ()
    output_schema_keys = list(ir.schema.keys())
    if ir.options[0] in ("Inner", "Left", "Semi", "Anti"):
        # Use left keys for output partitioning
        output_key_indices = tuple(
            output_schema_keys.index(expr.name)
            for expr in ir.left_on
            if expr.name in output_schema_keys
        )
    elif ir.options[0] == "Right":
        output_key_indices = tuple(
            output_schema_keys.index(expr.name)
            for expr in ir.right_on
            if expr.name in output_schema_keys
        )

    metadata_out = ChannelMetadata(
        local_count=output_count,
        partitioning=Partitioning(
            HashScheme(column_indices=output_key_indices, modulus=modulus),
            local="aligned",
        )
        if output_key_indices
        else None,
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)

    # Create shuffle managers for both sides
    left_shuffle = ShuffleManager(
        context, output_count, left_key_indices, left_collective_id
    )
    right_shuffle = ShuffleManager(
        context, output_count, right_key_indices, right_collective_id
    )

    # Insert initial chunks
    for chunk in left_initial_chunks:
        left_shuffle.insert_chunk(
            chunk.make_available_and_spill(context.br(), allow_overbooking=True)
        )
    for chunk in right_initial_chunks:
        right_shuffle.insert_chunk(
            chunk.make_available_and_spill(context.br(), allow_overbooking=True)
        )

    # Drain remaining chunks from both channels concurrently
    async def drain_left() -> None:
        while (msg := await ch_left.recv(context)) is not None:
            left_shuffle.insert_chunk(
                TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
            )
            del msg
        await left_shuffle.insert_finished()

    async def drain_right() -> None:
        while (msg := await ch_right.recv(context)) is not None:
            right_shuffle.insert_chunk(
                TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
            )
            del msg
        await right_shuffle.insert_finished()

    await asyncio.gather(drain_left(), drain_right())

    # Extract shuffled partitions and perform partition-wise joins
    stream = ir_context.get_cuda_stream()
    for seq_num, partition_id in enumerate(
        range(context.comm().rank, output_count, nranks)
    ):
        left_table = await left_shuffle.extract_chunk(partition_id, stream)
        right_table = await right_shuffle.extract_chunk(partition_id, stream)

        left_df = DataFrame.from_table(
            left_table, list(left.schema.keys()), list(left.schema.values()), stream
        )
        right_df = DataFrame.from_table(
            right_table, list(right.schema.keys()), list(right.schema.values()), stream
        )

        input_bytes = sum(
            col.device_buffer_size()
            for col in (*left_df.table.columns(), *right_df.table.columns())
        )
        with opaque_reservation(context, input_bytes):
            df = await asyncio.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                left_df,
                right_df,
                context=ir_context,
            )
            n_rows_out += df.num_rows
            await ch_out.send(
                context,
                Message(
                    seq_num,
                    TableChunk.from_pylibcudf_table(
                        df.table, df.stream, exclusive_view=True
                    ),
                ),
            )
            del df, left_df, right_df, left_table, right_table

    if profiler is not None:
        profiler.row_count[ir] += n_rows_out
    await ch_out.drain(context)


@define_py_node()
async def join_node(
    context: Context,
    ir: Join,
    ir_context: Any,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    sample_chunk_count: int,
    broadcast_threshold: int,
    target_partition_size: int,
    collective_ids: list[int],
    profiler: Profiler | None = None,
) -> None:
    """
    Dynamic Join node that selects the best strategy at runtime.

    Strategy selection based on sampled data:
    - Broadcast right: If right side is small (< broadcast_threshold)
    - Broadcast left: If left side is small (< broadcast_threshold)
    - Shuffle: Both sides are large, shuffle by join keys
    """
    async with shutdown_on_error(context, ch_left, ch_right, ch_out):
        # Receive metadata from both sides
        left_metadata, right_metadata = await asyncio.gather(
            recv_metadata(ch_left, context),
            recv_metadata(ch_right, context),
        )

        nranks = context.comm().nranks

        # Sample chunks from both sides concurrently
        left_initial_chunks: list[TableChunk] = []
        right_initial_chunks: list[TableChunk] = []
        left_sample_size = 0
        right_sample_size = 0

        async def sample_left() -> None:
            nonlocal left_sample_size
            for _ in range(sample_chunk_count):
                msg = await ch_left.recv(context)
                if msg is None:
                    break
                chunk = TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
                left_initial_chunks.append(chunk)
                left_sample_size += chunk.data_alloc_size(MemoryType.DEVICE)
                del msg

        async def sample_right() -> None:
            nonlocal right_sample_size
            for _ in range(sample_chunk_count):
                msg = await ch_right.recv(context)
                if msg is None:
                    break
                chunk = TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
                right_initial_chunks.append(chunk)
                right_sample_size += chunk.data_alloc_size(MemoryType.DEVICE)
                del msg

        await asyncio.gather(sample_left(), sample_right())

        # Estimate total sizes
        left_local_count = left_metadata.local_count
        right_local_count = right_metadata.local_count

        if left_initial_chunks:
            left_avg = left_sample_size / len(left_initial_chunks)
            left_estimate = int(left_avg * left_local_count)
        else:
            left_estimate = 0

        if right_initial_chunks:
            right_avg = right_sample_size / len(right_initial_chunks)
            right_estimate = int(right_avg * right_local_count)
        else:
            right_estimate = 0

        # AllGather size estimates across ranks
        if collective_ids and nranks > 1:
            left_total, right_total = await allgather_reduce(
                context, collective_ids.pop(), left_estimate, right_estimate
            )
        else:
            left_total, right_total = left_estimate, right_estimate

        # =====================================================================
        # Strategy Selection
        # =====================================================================
        # Note: Dynamic join planning only handles Inner/Left/Semi/Anti joins.
        # - Inner: can broadcast either side
        # - Left/Semi/Anti: must broadcast right (stream left to preserve all left rows)

        join_type = ir.options[0]
        can_broadcast_left = join_type == "Inner"
        # All supported types (Inner/Left/Semi/Anti) can broadcast right

        # Check if one side is already duplicated
        left_duplicated = left_metadata.duplicated
        right_duplicated = right_metadata.duplicated

        # Determine strategy
        broadcast_side: Literal["left", "right"] | None = None

        if nranks == 1:
            # Single rank - just do a local join (broadcast right as default)
            broadcast_side = "right"
        elif right_duplicated:
            # Right already duplicated - broadcast right (no allgather needed)
            broadcast_side = "right"
        elif left_duplicated and can_broadcast_left:
            # Left already duplicated - broadcast left (only for Inner)
            broadcast_side = "left"
        elif right_total < broadcast_threshold:
            # Right is small enough to broadcast
            broadcast_side = "right"
        elif left_total < broadcast_threshold and can_broadcast_left:
            # Left is small enough to broadcast (only for Inner)
            broadcast_side = "left"
        # else: shuffle both sides

        if broadcast_side is not None:
            # Broadcast join
            if profiler is not None:
                profiler.decisions[ir] = f"broadcast_{broadcast_side}"
            bcast_collective_id = collective_ids.pop() if collective_ids else 0
            await _broadcast_join(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left,
                ch_right,
                left_metadata,
                right_metadata,
                left_initial_chunks,
                right_initial_chunks,
                broadcast_side,
                bcast_collective_id,
                target_partition_size,
                profiler,
            )
        else:
            # Shuffle join - need 2 collective IDs (left and right shuffles)
            if len(collective_ids) >= 2:
                left_collective_id = collective_ids.pop()
                right_collective_id = collective_ids.pop()
            else:
                # Fallback: not enough IDs, use broadcast instead
                # For Left/Semi/Anti must broadcast right; for Inner prefer smaller
                fallback_side: Literal["left", "right"] = (
                    "left"
                    if can_broadcast_left and left_total < right_total
                    else "right"
                )
                if profiler is not None:
                    profiler.decisions[ir] = f"broadcast_{fallback_side}_fallback"
                await _broadcast_join(
                    context,
                    ir,
                    ir_context,
                    ch_out,
                    ch_left,
                    ch_right,
                    left_metadata,
                    right_metadata,
                    left_initial_chunks,
                    right_initial_chunks,
                    fallback_side,
                    collective_ids.pop() if collective_ids else 0,
                    target_partition_size,
                    profiler,
                )
                return

            # Calculate output partition count
            total_size = left_total + right_total
            output_count = max(1, total_size // target_partition_size)

            if profiler is not None:
                profiler.decisions[ir] = "shuffle"

            await _shuffle_join(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left,
                ch_right,
                left_metadata,
                right_metadata,
                left_initial_chunks,
                right_initial_chunks,
                output_count,
                left_collective_id,
                right_collective_id,
                profiler,
            )


@generate_ir_sub_network.register(Join)
def _(
    ir: Join, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Join operation.
    left, right = ir.children
    partition_info = rec.state["partition_info"]
    output_count = partition_info[ir].count
    config_options = rec.state["config_options"]
    executor = config_options.executor

    # Check for dynamic planning (only for inner/left/semi/anti joins)
    join_type = ir.options[0]
    use_dynamic = (
        isinstance(executor, StreamingExecutor)
        and executor.dynamic_planning.enabled
        and join_type in ("Inner", "Left", "Semi", "Anti")
    )

    left_count = partition_info[left].count
    right_count = partition_info[right].count
    left_partitioned = (
        partition_info[left].partitioned_on == ir.left_on and left_count == output_count
    )
    right_partitioned = (
        partition_info[right].partitioned_on == ir.right_on
        and right_count == output_count
    )

    pwise_join = output_count == 1 or (left_partitioned and right_partitioned)

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    if pwise_join:
        # Partition-wise join (use default_node_multi)
        partitioning_index = 1 if ir.options[0] == "Right" else 0
        nodes[ir] = [
            default_node_multi(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                (
                    channels[left].reserve_output_slot(),
                    channels[right].reserve_output_slot(),
                ),
                partitioning_index=partitioning_index,
            )
        ]
        return nodes, channels

    elif use_dynamic:
        # Dynamic join - decide strategy at runtime
        assert isinstance(executor, StreamingExecutor)
        collective_ids = list(rec.state["collective_id_map"].get(ir, []))
        broadcast_threshold = (
            executor.target_partition_size * executor.broadcast_join_limit
        )
        nodes[ir] = [
            join_node(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                channels[left].reserve_output_slot(),
                channels[right].reserve_output_slot(),
                executor.dynamic_planning.sample_chunk_count,
                broadcast_threshold,
                executor.target_partition_size,
                collective_ids,
                rec.state["profiler"],
            )
        ]
        return nodes, channels

    else:
        # Broadcast join (use broadcast_join_node)
        broadcast_side: Literal["left", "right"]
        if left_count >= right_count:
            # Broadcast right, stream left
            broadcast_side = "right"
        else:
            broadcast_side = "left"

        # Get target partition size
        assert isinstance(executor, StreamingExecutor), (
            "Join node requires streaming executor"
        )
        target_partition_size = executor.target_partition_size

        nodes[ir] = [
            broadcast_join_node(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                channels[left].reserve_output_slot(),
                channels[right].reserve_output_slot(),
                broadcast_side=broadcast_side,
                collective_id=rec.state["collective_id_map"][ir][0],
                target_partition_size=target_partition_size,
            )
        ]
        return nodes, channels
