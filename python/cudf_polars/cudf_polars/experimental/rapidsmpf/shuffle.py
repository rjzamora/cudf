# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Union logic for the RapidsMPF streaming engine."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from rapidsmpf.communicator.single import new_communicator
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.integrations.cudf.partition import (
    partition_and_pack as py_partition_and_pack,
    unpack_and_concat as py_unpack_and_concat,
)
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.shuffler import Shuffler
from rapidsmpf.streaming.core.channel import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.dsl.expr import Col
from cudf_polars.experimental.base import ChunkMetadata
from cudf_polars.experimental.rapidsmpf.channel_pair import ChannelPair
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import shutdown_on_error
from cudf_polars.experimental.shuffle import Shuffle

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator


# Set of available shuffle IDs
_shuffle_id_vacancy: set[int] = set(range(Shuffler.max_concurrent_shuffles))
_shuffle_id_vacancy_lock: threading.Lock = threading.Lock()


def _get_new_shuffle_id() -> int:
    with _shuffle_id_vacancy_lock:
        if not _shuffle_id_vacancy:
            raise ValueError(
                f"Cannot shuffle more than {Shuffler.max_concurrent_shuffles} "
                "times in a single query."
            )

        return _shuffle_id_vacancy.pop()


def _release_shuffle_id(op_id: int) -> None:
    """Release a shuffle ID back to the vacancy set."""
    with _shuffle_id_vacancy_lock:
        _shuffle_id_vacancy.add(op_id)


@define_py_node()
async def local_shuffle_node(
    ctx: Context,
    ir: Shuffle,
    ch_in: ChannelPair,
    ch_out: ChannelPair,
    columns_to_hash: tuple[int, ...],
    num_partitions: int,
    op_id: int,
) -> None:
    """
    Execute a local shuffle pipeline in a single node with metadata passthrough.

    This node combines partition_and_pack, shuffler, and unpack_and_concat
    into a single Python node using rapidsmpf.shuffler.Shuffler and utilities
    from rapidsmpf.integrations.cudf.partition. It also handles metadata
    passthrough from input to output.

    Parameters
    ----------
    ctx
        The streaming context.
    ir
        The Shuffle IR node.
    ch_in
        Input ChannelPair with metadata and data channels.
    ch_out
        Output ChannelPair with metadata and data channels.
    columns_to_hash
        Tuple of column indices to use for hashing.
    num_partitions
        Number of partitions to shuffle into.
    op_id
        Unique shuffle operation ID.
    """
    async with shutdown_on_error(
        ctx, ch_in.metadata, ch_in.data, ch_out.metadata, ch_out.data
    ):
        # Always use a single-process communicator for local shuffle
        comm = new_communicator(Options(get_environment_variables()))

        # Get resources from context
        br = ctx.br()
        statistics = ctx.statistics()

        # Create a progress thread for this shuffle
        # Note: The C++ context has a progress_thread() method, but Python
        # binding might not expose it, so we create a new one
        progress_thread = ProgressThread(comm, statistics)

        # Use the default CUDA stream for operations
        stream = DEFAULT_STREAM

        # Create the Shuffler instance
        shuffler = Shuffler(
            comm=comm,
            progress_thread=progress_thread,
            op_id=op_id,
            total_num_partitions=num_partitions,
            br=br,
            statistics=statistics,
        )

        # Update metadata
        names = list(ir.schema.keys())
        metadata_in = await ch_in.recv_metadata(ctx)
        assert isinstance(metadata_in, ChunkMetadata), (
            f"Expected ChunkMetadata, got {type(metadata_in)}."
        )
        metadata_out = ChunkMetadata(
            num_partitions,
            local_partitioned_on=tuple(names[i] for i in columns_to_hash),
            global_partitioned_on=metadata_in.global_partitioned_on,
            duplicated=metadata_in.duplicated,
        )
        await ch_out.send_metadata(ctx, metadata_out)

        # # Handle metadata passthrough concurrently with data processing
        # async def metadata_passthrough() -> None:
        #     metadata_msg = await ch_in.metadata.recv(ctx)
        #     if metadata_msg is not None:
        #         await ch_out.metadata.send(ctx, metadata_msg)
        #     await ch_out.metadata.drain(ctx)

        # # Start metadata passthrough as a background task
        # metadata_task = asyncio.create_task(metadata_passthrough())

        # try:
        # Process input chunks
        while True:
            msg = await ch_in.data.recv(ctx)
            if msg is None:
                break

            # Extract TableChunk from message
            chunk = TableChunk.from_message(msg)

            # Get the table view
            table = chunk.table_view()

            # Partition and pack using the Python function
            partitioned_chunks = py_partition_and_pack(
                table=table,
                columns_to_hash=columns_to_hash,
                num_partitions=num_partitions,
                stream=stream,
                br=br,
            )

            # Insert into shuffler
            shuffler.insert_chunks(partitioned_chunks)

        # Mark all partitions as finished for insertion
        shuffler.insert_finished(list(range(num_partitions)))

        # Extract shuffled partitions and send them out
        # We need to maintain sequence ordering for output chunks
        output_seq = 0

        while not shuffler.finished():
            # Wait for any partition to be ready
            pid = shuffler.wait_any()

            # Extract the partition
            partition_chunks = shuffler.extract(pid)

            if partition_chunks:
                # Unpack and concatenate the partition chunks
                result_table = py_unpack_and_concat(
                    partitions=partition_chunks,
                    stream=stream,
                    br=br,
                )

                # Create a new TableChunk with the result
                output_chunk = TableChunk.from_pylibcudf_table(
                    sequence_number=output_seq,
                    table=result_table,
                    stream=stream,
                    exclusive_view=True,
                )
                output_seq += 1

                # Send the output chunk
                await ch_out.data.send(ctx, Message(output_chunk))

        # Shutdown the shuffler
        shuffler.shutdown()

        # Release the shuffle ID
        _release_shuffle_id(op_id)

        # Drain data output channel
        await ch_out.data.drain(ctx)

        # finally:
        #     # Wait for metadata passthrough to complete
        #     await metadata_task


@generate_ir_sub_network.register(Shuffle)
def _(
    ir: Shuffle, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, list[Any]]]:
    # Local shuffle operation.
    # TODO: How to distinguish between local and global shuffle?
    # May need to track two different contexts?

    # Process children
    (child,) = ir.children
    nodes, channels = rec(child)

    keys: list[Col] = [ne.value for ne in ir.keys if isinstance(ne.value, Col)]
    if len(keys) != len(ir.keys):  # pragma: no cover
        raise NotImplementedError("Shuffle requires simple keys.")
    column_names = list(ir.schema.keys())

    context = rec.state["ctx"]
    columns_to_hash = tuple(column_names.index(k.name) for k in keys)
    num_partitions = rec.state["partition_info"][ir].count
    op_id = _get_new_shuffle_id()

    # Get input and create output ChannelPairs
    ch_in = channels[child].pop()
    ch_out = ChannelPair.create()

    # Complete shuffle pipeline with metadata passthrough in a single node
    nodes[ir] = [
        local_shuffle_node(
            context,
            ir,
            ch_in=ch_in,
            ch_out=ch_out,
            columns_to_hash=columns_to_hash,
            num_partitions=num_partitions,
            op_id=op_id,
        )
    ]

    # Return output ChannelPair
    channels[ir] = [ch_out]

    return nodes, channels
