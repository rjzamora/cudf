# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core node definitions for the RapidsMPF streaming engine."""

from __future__ import annotations

import asyncio
import operator
from contextlib import asynccontextmanager
from functools import reduce
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.channel import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Empty
from cudf_polars.experimental.rapidsmpf.channel_pair import ChannelPair
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    import pylibcudf as plc

    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator


@asynccontextmanager
async def shutdown_on_error(
    ctx: Context, *channels: Channel[Any]
) -> AsyncIterator[None]:
    """
    Shutdown on error for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    channels
        The channels to shutdown.
    """
    # TODO: This probably belongs in rapidsmpf.
    try:
        yield
    except Exception:
        await asyncio.gather(*(ch.shutdown(ctx) for ch in channels))
        raise


@define_py_node()
async def default_node_single(
    ctx: Context,
    ir: IR,
    ch_out: ChannelPair,
    ch_in: ChannelPair,
) -> None:
    """
    Single-channel default node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ir
        The IR node.
    ch_out
        The output ChannelPair.
    ch_in
        The input ChannelPair.

    Notes
    -----
    Chunks are processed in the order they are received.
    """
    async with shutdown_on_error(
        ctx, ch_in.metadata, ch_in.data, ch_out.metadata, ch_out.data
    ):
        # Pass through metadata
        metadata = await ch_in.recv_metadata(ctx)
        await ch_out.send_metadata(ctx, metadata)

        # Process data chunks
        while (msg := await ch_in.data.recv(ctx)) is not None:
            chunk = TableChunk.from_message(msg)
            seq_num = chunk.sequence_number
            df = await asyncio.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                DataFrame.from_table(
                    chunk.table_view(),
                    list(ir.children[0].schema.keys()),
                    list(ir.children[0].schema.values()),
                ),
            )
            chunk = TableChunk.from_pylibcudf_table(
                seq_num, df.table, chunk.stream, exclusive_view=True
            )
            await ch_out.data.send(ctx, Message(chunk))

        await ch_out.data.drain(ctx)


@define_py_node()
async def default_node_multi(
    ctx: Context,
    ir: IR,
    ch_out: ChannelPair,
    *chs_in: ChannelPair,
    bcast_indices: list[int],
) -> None:
    """
    Pointwise node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ir
        The IR node.
    ch_out
        The output ChannelPair.
    chs_in
        The input ChannelPairs.
    bcast_indices
        The indices of the broadcasted children.

    Notes
    -----
    Input chunks are aligned for evaluation.
    """
    # TODO: Use multiple streams
    all_channels = [ch for pair in chs_in for ch in (pair.metadata, pair.data)]
    async with shutdown_on_error(ctx, *all_channels, ch_out.metadata, ch_out.data):
        # Receive metadata from all inputs and merge
        # For now, just take the first non-None metadata
        # TODO: May need a more sophisticated merge strategy
        metadata = None
        for ch_in in chs_in:
            md = await ch_in.recv_metadata(ctx)
            if md is not None and metadata is None:
                metadata = md
        await ch_out.send_metadata(ctx, metadata)

        seq_num = 0
        n_children = len(chs_in)
        accepting_data = True
        finished_channels: set[int] = set()
        staged_chunks: dict[int, dict[int, DataFrame]] = {
            c: {} for c in range(n_children)
        }

        while True:
            if accepting_data:
                for ch_idx, (ch_in, child) in enumerate(
                    zip(chs_in, ir.children, strict=True)
                ):
                    if (ch_not_finished := ch_idx not in finished_channels) and (
                        msg := await ch_in.data.recv(ctx)
                    ) is not None:
                        table_chunk = TableChunk.from_message(msg)
                        if ch_idx in bcast_indices and staged_chunks[ch_idx]:
                            raise RuntimeError(
                                f"Broadcasted chunk already staged for channel {ch_idx}."
                            )
                        staged_chunks[ch_idx][table_chunk.sequence_number] = (
                            DataFrame.from_table(
                                table_chunk.table_view(),
                                list(child.schema.keys()),
                                list(child.schema.values()),
                            )
                        )
                    elif ch_not_finished:
                        finished_channels.add(ch_idx)
                        if all(
                            ch_idx in finished_channels for ch_idx in range(n_children)
                        ):
                            accepting_data = False

            if all(
                (
                    (seq_num in staged_chunks[ch_idx])
                    or (ch_idx in bcast_indices and 0 in staged_chunks[ch_idx])
                )
                for ch_idx in range(n_children)
            ):
                # Ready to produce the output chunk for seq_num.
                # Evaluate and send.
                df = await asyncio.to_thread(
                    ir.do_evaluate,
                    *ir._non_child_args,
                    *[
                        (
                            staged_chunks[ch_idx][0]
                            if ch_idx in bcast_indices
                            else staged_chunks[ch_idx].pop(seq_num)
                        )
                        for ch_idx in range(n_children)
                    ],
                )
                await ch_out.data.send(
                    ctx,
                    Message(
                        TableChunk.from_pylibcudf_table(
                            seq_num,
                            df.table,
                            DEFAULT_STREAM,
                            exclusive_view=True,
                        )
                    ),
                )
                seq_num += 1
            elif not accepting_data:
                if any(
                    staged_chunks[ch_idx]
                    for ch_idx in range(n_children)
                    if ch_idx not in bcast_indices
                ):
                    raise RuntimeError(
                        f"Leftover data in staged chunks: {staged_chunks}."
                    )
                break  # All channels have finished

        # Drain the data channel
        await ch_out.data.drain(ctx)


@define_py_node()
async def multicast_node(
    ctx: Context,
    ch_in: ChannelPair,
    *chs_out: ChannelPair,
) -> None:
    """
    Multicast node for rapidsmpf - broadcasts both metadata and data.

    Parameters
    ----------
    ctx
        The context.
    ch_in
        The input ChannelPair.
    chs_out
        The output ChannelPairs.
    """
    # TODO: Use multiple streams
    all_out_channels = [ch for pair in chs_out for ch in (pair.metadata, pair.data)]
    async with shutdown_on_error(ctx, ch_in.metadata, ch_in.data, *all_out_channels):
        # Receive metadata
        metadata = await ch_in.recv_metadata(ctx)

        # Collect all data chunks from input channel
        chunks: list[TableChunk] = []
        while (msg := await ch_in.data.recv(ctx)) is not None:
            chunks.append(TableChunk.from_message(msg).table_view())

        # Send metadata and data to all output channels using atomic per-channel processing
        # This ensures that channels consuming at different rates don't block each other
        # (e.g., streaming operations vs fallback repartition operations)
        await asyncio.gather(
            *(
                _multicast_to_channel_pair(ctx, metadata, chunks, ch_out)
                for ch_out in chs_out
            )
        )


async def _multicast_to_channel_pair(
    ctx: Context,
    metadata: dict[str, Any] | None,
    chunks: list[plc.Table],
    ch_out: ChannelPair,
) -> None:
    """
    Send metadata and data chunks to a single output ChannelPair, then drain it.

    Parameters
    ----------
    ctx
        The context.
    metadata
        The metadata to send (or None).
    chunks
        The data chunks to send.
    ch_out
        The output ChannelPair.
    """
    # Send metadata first
    await ch_out.send_metadata(ctx, metadata)

    # Send data chunks
    for seq_num, chunk in enumerate(chunks):
        await ch_out.data.send(
            ctx,
            Message(
                TableChunk.from_pylibcudf_table(
                    seq_num, chunk, DEFAULT_STREAM, exclusive_view=False
                )
            ),
        )
    await ch_out.data.drain(ctx)


@generate_ir_sub_network.register(IR)
def _(ir: IR, rec: SubNetGenerator) -> tuple[dict[IR, list[Any]], dict[IR, Any]]:
    # Default generate_ir_sub_network logic.
    # Use simple pointwise node.

    # Process children
    nodes: dict[IR, list[Any]] = {}
    channels: dict[IR, list[Any]] = {}
    if ir.children:
        _nodes, _channels = zip(*(rec(c) for c in ir.children), strict=True)
        nodes = reduce(operator.or_, _nodes)
        channels = reduce(operator.or_, _channels)

    # Create output ChannelPair
    channels[ir] = [ChannelPair.create()]

    if len(ir.children) == 1:
        # Single-channel default node
        nodes[ir] = [
            default_node_single(
                rec.state["ctx"],
                ir,
                channels[ir][0],
                channels[ir.children[0]].pop(),
            )
        ]
    else:
        # Multi-channel default node
        counts = [rec.state["partition_info"][c].count for c in ir.children]
        bcast_indices = (
            []
            if all(c == 1 for c in counts)
            else [i for i, c in enumerate(counts) if c == 1]
        )
        nodes[ir] = [
            default_node_multi(
                rec.state["ctx"],
                ir,
                channels[ir][0],
                *[channels[c].pop() for c in ir.children],
                bcast_indices=bcast_indices,
            )
        ]

    return nodes, channels


@define_py_node()
async def empty_node(
    ctx: Context,
    ir: Empty,
    ch_out: ChannelPair,
) -> None:
    """
    Empty node for rapidsmpf - produces a single empty chunk.

    Parameters
    ----------
    ctx
        The context.
    ir
        The Empty node.
    ch_out
        The output ChannelPair.
    """
    async with shutdown_on_error(ctx, ch_out.metadata, ch_out.data):
        # No metadata for empty node
        await ch_out.send_metadata(ctx, None)

        # Evaluate the IR node to create an empty DataFrame
        df: DataFrame = ir.do_evaluate(*ir._non_child_args)

        # Return the output chunk (empty but with correct schema)
        chunk = TableChunk.from_pylibcudf_table(
            0, df.table, DEFAULT_STREAM, exclusive_view=True
        )
        await ch_out.data.send(ctx, Message(chunk))

        await ch_out.data.drain(ctx)


@generate_ir_sub_network.register(Empty)
def _(ir: Empty, rec: SubNetGenerator) -> tuple[dict[IR, list[Any]], dict[IR, Any]]:
    """Generate network for Empty node - produces one empty chunk."""
    ctx = rec.state["ctx"]
    ch_out = ChannelPair.create()
    nodes: dict[IR, list[Any]] = {ir: [empty_node(ctx, ir, ch_out)]}
    channels: dict[IR, list[Any]] = {ir: [ch_out]}
    return nodes, channels


def generate_ir_sub_network_wrapper(
    ir: IR, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, list[Any]]]:
    """
    Generate a sub-network for the RapidsMPF streaming engine.

    Parameters
    ----------
    ir
        The IR node.
    rec
        Recursive SubNetGenerator callable.

    Returns
    -------
    nodes
        Dictionary mapping between each IR node and its
        corresponding streaming-network node(s).
    channels
        Dictionary mapping between each IR node and its
        corresponding streaming-network output channels.
    """
    nodes, channels = generate_ir_sub_network(ir, rec)
    if (count := rec.state["output_ch_count"][ir]) > 1:
        output_chs = [ChannelPair.create() for _ in range(count)]
        nodes[ir].append(multicast_node(rec.state["ctx"], channels[ir][0], *output_chs))
        channels[ir] = output_chs
    return nodes, channels
