# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffle logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.communicator.single import new_communicator as single_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.integrations.cudf.partition import (
    partition_and_pack as py_partition_and_pack,
    unpack_and_concat as py_unpack_and_concat,
)
from rapidsmpf.streaming.coll.shuffler import ShufflerAsync
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.dsl.expr import Col
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import shutdown_on_error
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    NormalizedPartitioning,
    recv_metadata,
    send_metadata,
)
from cudf_polars.experimental.shuffle import Shuffle

ShuffleMode = Literal["flat", "stratified"]
"""
Shuffle routing strategy.

flat:       hash % num_partitions → route by partition owner (current behaviour)
stratified: stage-1 hash % num_ranks, stage-2 local hash % (num_partitions // num_ranks)
"""

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel

    import pylibcudf as plc
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator


class ShuffleManager:
    """
    ShufflerAsync manager.

    Parameters
    ----------
    context: Context
        The streaming context.
    comm: Communicator
        The communicator.
    num_partitions: int
        The number of partitions to shuffle into.
    columns_to_hash: tuple[int, ...]
        The columns to hash.
    collective_id: int
        The collective ID.
    shuffle_mode: ShuffleMode
        Routing strategy: ``"flat"`` (single-phase, hash % num_partitions) or
        ``"stratified"`` (two-stage: hash % num_ranks in stage 1, local hash %
        local_count in stage 2).
    """

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        num_partitions: int,
        columns_to_hash: tuple[int, ...],
        collective_id: int,
        *,
        shuffle_mode: ShuffleMode = "flat",
    ):
        self.context = context
        self.comm = comm
        self.num_partitions = num_partitions
        self.columns_to_hash = columns_to_hash
        self.collective_id = collective_id
        self.shuffle_mode = shuffle_mode
        self._local_count = max(1, num_partitions // comm.nranks)
        self._stage1_partitions = (
            comm.nranks if shuffle_mode == "stratified" else num_partitions
        )
        self.shuffler = ShufflerAsync(
            context,
            comm,
            collective_id,
            self._stage1_partitions,
        )
        # _local_shuffle is the ShufflerAsync to extract final partitions from.
        # For flat it stays as self.shuffler; for stratified it is replaced with
        # the single-rank stage-2 shuffler inside insert_finished.
        self._local_shuffle: ShufflerAsync = self.shuffler
        self._local_pid_offset: int = 0

    def local_partitions(self) -> list[int]:
        """Get the local partition IDs for this rank."""
        return [
            pid + self._local_pid_offset
            for pid in self._local_shuffle.local_partitions()
        ]

    def insert_chunk(self, chunk: TableChunk) -> None:
        """
        Insert a chunk into the ShuffleContext.

        Parameters
        ----------
        chunk: TableChunk
            The table chunk to insert.
        """
        partitioned_chunks = py_partition_and_pack(
            table=chunk.table_view(),
            columns_to_hash=self.columns_to_hash,
            num_partitions=self._stage1_partitions,
            stream=chunk.stream,
            br=self.context.br(),
        )
        self.shuffler.insert(partitioned_chunks)

    async def insert_finished(self) -> None:
        """Insert finished into the ShuffleManager."""
        await self.shuffler.insert_finished(self.context)
        if self.shuffle_mode == "stratified":
            rank_pid = self.shuffler.local_partitions()[0]
            raw_chunks = self.shuffler.extract(rank_pid)
            options = Options(get_environment_variables())
            local_comm = single_comm(options, self.comm.progress_thread)
            local_ctx = Context(local_comm.logger, self.context.br(), options)
            local_mgr = ShuffleManager(
                local_ctx,
                local_comm,
                self._local_count,
                self.columns_to_hash,
                self.collective_id,
            )
            for pd in raw_chunks:
                local_mgr.insert_chunk(
                    TableChunk.from_packed_data(pd).make_available_and_spill(
                        self.context.br(), allow_overbooking=True
                    )
                )
            await local_mgr.insert_finished()
            self._local_shuffle = local_mgr.shuffler
            self._local_pid_offset = self.comm.rank * self._local_count

    def extract_chunk(self, sequence_number: int, stream: Stream) -> plc.Table:
        """
        Extract a chunk from the ShuffleManager.

        Parameters
        ----------
        sequence_number: int
            The sequence number of the chunk to extract.
        stream: Stream
            The stream to use for chunk extraction.

        Returns
        -------
        The extracted table.

        Raises
        ------
        KeyError
            If the requested sequence number has already been extracted.
        """
        local_pid = sequence_number - self._local_pid_offset
        return py_unpack_and_concat(
            partitions=self._local_shuffle.extract(local_pid),
            stream=stream,
            br=self.context.br(),
        )


def _is_already_partitioned(
    metadata: ChannelMetadata,
    columns_to_hash: tuple[int, ...],
    num_partitions: int,
    nranks: int,
    *,
    shuffle_mode: ShuffleMode = "flat",
) -> bool:
    """Check if data is already partitioned on the required keys."""
    local_count = max(1, num_partitions // nranks)
    if shuffle_mode == "flat":
        partitioning_desired = NormalizedPartitioning(
            inter_rank_modulus=num_partitions,
            inter_rank_indices=columns_to_hash,
            local_modulus=None,
            local_indices=(),
        )
    else:  # stratified
        partitioning_desired = NormalizedPartitioning(
            inter_rank_modulus=nranks,
            inter_rank_indices=columns_to_hash,
            local_modulus=local_count,
            local_indices=columns_to_hash,
        )
    partitioning = NormalizedPartitioning.from_indices(
        metadata.partitioning,
        nranks,
        indices=columns_to_hash,
        allow_subset=False,
    )
    return bool(partitioning and partitioning == partitioning_desired)


async def _global_shuffle(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    columns_to_hash: tuple[int, ...],
    num_partitions: int,
    collective_id: int,
    *,
    shuffle_mode: ShuffleMode = "flat",
) -> None:
    """
    Global shuffle implementation.

    Parameters
    ----------
    context
        The streaming context.
    comm
        The communicator.
    ir_context
        The execution context for the IR node.
    ch_out
        Output Channel[TableChunk] with metadata and data channels.
    ch_in
        Input Channel[TableChunk] with metadata and data channels.
    columns_to_hash
        Tuple of column indices to use for hashing.
    num_partitions
        Number of partitions to shuffle into.
    collective_id
        The collective ID.
    shuffle_mode
        Routing strategy: ``"flat"`` (current single-phase) or ``"stratified"``
        (two-stage: hash % num_ranks in stage 1, hash % local_count locally in
        stage 2).  Both sides of a join must use the same mode.
    """
    metadata_in = await recv_metadata(ch_in, context)
    local_count = max(1, num_partitions // comm.nranks)

    # Check if we can skip the shuffle (already partitioned correctly)
    if _is_already_partitioned(
        metadata_in,
        columns_to_hash,
        num_partitions,
        comm.nranks,
        shuffle_mode=shuffle_mode,
    ):
        # Forward metadata and data unchanged
        await send_metadata(ch_out, context, metadata_in)
        while (msg := await ch_in.recv(context)) is not None:
            await ch_out.send(context, msg)
        await ch_out.drain(context)
        return

    if shuffle_mode == "flat":
        output_metadata = ChannelMetadata(
            local_count=local_count,
            partitioning=Partitioning(
                inter_rank=HashScheme(columns_to_hash, num_partitions),
                local="inherit",
            ),
        )
    else:  # stratified
        output_metadata = ChannelMetadata(
            local_count=local_count,
            partitioning=Partitioning(
                inter_rank=HashScheme(columns_to_hash, comm.nranks),
                local=HashScheme(columns_to_hash, local_count),
            ),
        )
    await send_metadata(ch_out, context, output_metadata)

    # When input is duplicated, only rank 0 should contribute data.
    # Other ranks still participate in the shuffle protocol.
    skip_insert = metadata_in.duplicated and comm.rank != 0

    shuffle = ShuffleManager(
        context,
        comm,
        num_partitions,
        columns_to_hash,
        collective_id,
        shuffle_mode=shuffle_mode,
    )
    while (msg := await ch_in.recv(context)) is not None:
        if not skip_insert:
            shuffle.insert_chunk(
                TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
            )
    await shuffle.insert_finished()
    for partition_id in shuffle.local_partitions():
        stream = ir_context.get_cuda_stream()
        await ch_out.send(
            context,
            Message(
                partition_id,
                TableChunk.from_pylibcudf_table(
                    table=shuffle.extract_chunk(partition_id, stream),
                    stream=stream,
                    exclusive_view=True,
                ),
            ),
        )

    await ch_out.drain(context)


@define_actor()
async def shuffle_actor(
    context: Context,
    comm: Communicator,
    ir: Shuffle,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    columns_to_hash: tuple[int, ...],
    num_partitions: int,
    collective_id: int,
) -> None:
    """
    Execute a global shuffle pipeline within a single node.

    This node combines partition_and_pack, shuffler, and unpack_and_concat
    into a single Python node using rapidsmpf.shuffler.Shuffler and utilities
    from rapidsmpf.integrations.cudf.partition.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The Shuffle IR node.
    ir_context
        The execution context for the IR node.
    ch_in
        Input Channel[TableChunk] with metadata and data channels.
    ch_out
        Output Channel[TableChunk] with metadata and data channels.
    columns_to_hash
        Tuple of column indices to use for hashing.
    num_partitions
        Number of partitions to shuffle into.
    collective_id
        The collective ID.
    """
    async with shutdown_on_error(
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
    ):
        await _global_shuffle(
            context,
            comm,
            ir_context,
            ch_out,
            ch_in,
            columns_to_hash,
            num_partitions,
            collective_id,
        )


@generate_ir_sub_network.register(Shuffle)
def _(
    ir: Shuffle, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Local shuffle operation.

    # Process children
    (child,) = ir.children
    nodes, channels = rec(child)

    keys: list[Col] = [ne.value for ne in ir.keys if isinstance(ne.value, Col)]
    if len(keys) != len(ir.keys):  # pragma: no cover
        raise NotImplementedError("Shuffle requires simple keys.")
    column_names = list(ir.schema.keys())

    context = rec.state["context"]
    columns_to_hash = tuple(column_names.index(k.name) for k in keys)
    num_partitions = rec.state["partition_info"][ir].count

    # Look up the reserved collective ID for this operation
    collective_id = rec.state["collective_id_map"][ir][0]

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Complete shuffle node
    nodes[ir] = [
        shuffle_actor(
            context,
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            ch_in=channels[child].reserve_output_slot(),
            ch_out=channels[ir].reserve_input_slot(),
            columns_to_hash=columns_to_hash,
            num_partitions=num_partitions,
            collective_id=collective_id,
        )
    ]

    return nodes, channels
