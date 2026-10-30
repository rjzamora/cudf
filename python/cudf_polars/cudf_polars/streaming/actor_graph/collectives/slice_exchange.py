# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route owned rows and required context between ordered input streams."""

from __future__ import annotations

import asyncio
import struct
from contextlib import aclosing, suppress
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import pylibcudf as plc
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.streaming.core.message import Message

from cudf_polars.streaming.actor_graph.utils import ChunkStore

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        Awaitable,
        Callable,
        Sequence,
    )

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context


type RowRange = tuple[int, int]
_OWNERSHIP_HEADER = struct.Struct("<4sQ")
_OWNERSHIP_MAGIC = b"OWN1"


@dataclass(frozen=True)
class InputDemand[OwnedT, NeededT]:
    """Data owned by one destination and the context it needs."""

    owned: OwnedT | None
    needed: NeededT


@dataclass(frozen=True)
class RoutedSlice:
    """One half-open input slice routed to a destination rank."""

    destination: int
    start: int
    stop: int
    owned: bool


type _ChunkResolver[SummaryT, OwnedT, NeededT] = Callable[
    [TableChunk, SummaryT | None, Sequence[tuple[int, InputDemand[OwnedT, NeededT]]]],
    Awaitable[list[RoutedSlice]],
]
type _SummaryResolver[SummaryT, OwnedT, NeededT] = Callable[
    [int, SummaryT | None, Sequence[tuple[int, InputDemand[OwnedT, NeededT]]]],
    list[RoutedSlice] | None,
]


@dataclass(frozen=True)
class SliceResolver[SummaryT, OwnedT, NeededT]:
    """
    Resolve routed slices from a summary or, when necessary, a chunk.

    The summary resolver returns slices when it can resolve a chunk without
    materializing it, an empty list to skip it, or ``None`` to defer to the
    chunk resolver.
    """

    resolve_chunk: _ChunkResolver[SummaryT, OwnedT, NeededT]
    resolve_summary: _SummaryResolver[SummaryT, OwnedT, NeededT] | None = None


def use_precomputed_slices[OwnedT, NeededT](
    _size: int,
    slices: list[RoutedSlice] | None,
    _requests: Sequence[tuple[int, InputDemand[OwnedT, NeededT]]],
) -> list[RoutedSlice] | None:
    """Return routing decisions computed before source replay."""
    return slices


@dataclass(frozen=True)
class ExchangePlan[OwnedT, NeededT]:
    """Validated outgoing requests and remote suppliers for one rank."""

    outgoing: tuple[tuple[int, InputDemand[OwnedT, NeededT]], ...]
    sources: tuple[int, ...]

    @classmethod
    def from_routes(
        cls,
        rank: int,
        routes: Sequence[Sequence[int]],
        outgoing: Sequence[tuple[int, InputDemand[OwnedT, NeededT]]],
    ) -> ExchangePlan[OwnedT, NeededT]:
        """Build one rank's plan from a route graph derived on every rank."""
        nranks = len(routes)
        if not 0 <= rank < nranks:
            raise ValueError(f"Invalid source rank {rank}")
        normalized = tuple(
            tuple(dict.fromkeys(destinations)) for destinations in routes
        )
        if any(
            destination < 0 or destination >= nranks
            for destinations in normalized
            for destination in destinations
        ):
            raise ValueError("Exchange route contains an invalid destination rank")
        outgoing = tuple(outgoing)
        if tuple(destination for destination, _ in outgoing) != normalized[rank]:
            raise ValueError("Outgoing requests do not match the global routing graph")
        return cls(
            outgoing,
            tuple(
                source
                for source, destinations in enumerate(normalized)
                if source != rank and rank in destinations
            ),
        )


def intersection(left: RowRange, right: RowRange) -> RowRange | None:
    """Return the intersection of two half-open row ranges."""
    start, stop = max(left[0], right[0]), min(left[1], right[1])
    return (start, stop) if start < stop else None


def merge_row_ranges(*ranges: RowRange | None) -> tuple[RowRange, ...]:
    """Return the sorted union of half-open row ranges."""
    result: list[RowRange] = []
    for start, stop in sorted(
        item for item in ranges if item is not None and item[0] < item[1]
    ):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(stop, result[-1][1]))
        else:
            result.append((start, stop))
    return tuple(result)


def tag_row_slice(
    destination: int,
    chunk_span: RowRange,
    piece: RowRange,
    owned: RowRange | None,
) -> list[RoutedSlice]:
    """Split one row slice where its ownership changes."""
    boundaries = {piece[0], piece[1]}
    if owned is not None and (owned_piece := intersection(piece, owned)) is not None:
        boundaries.update(owned_piece)
    return [
        RoutedSlice(
            destination,
            start - chunk_span[0],
            stop - chunk_span[0],
            owned=bool(owned is not None and owned[0] <= start and stop <= owned[1]),
        )
        for start, stop in pairwise(sorted(boundaries))
    ]


async def row_range_slices(
    chunk: TableChunk,
    span: RowRange | None,
    requests: Sequence[tuple[int, InputDemand[RowRange, RowRange]]],
) -> list[RoutedSlice]:
    """Resolve global row-range requests within one chunk."""
    return row_range_summary_slices(chunk.shape[0], span, requests) or []


def row_range_summary_slices(
    _size: int,
    span: RowRange | None,
    requests: Sequence[tuple[int, InputDemand[RowRange, RowRange]]],
) -> list[RoutedSlice]:
    """Resolve row-range requests without making a chunk available."""
    if span is None:
        return []
    result = []
    for destination, demand in requests:
        needed = intersection(span, demand.needed)
        owned = intersection(span, demand.owned) if demand.owned is not None else None
        for piece in merge_row_ranges(needed, owned):
            result.extend(tag_row_slice(destination, span, piece, demand.owned))
    return result


def _decode_ownership_descriptor(
    packed_items: Sequence[PackedData],
    source: int,
) -> tuple[Sequence[PackedData], tuple[bool, ...]]:
    """Separate payloads from one source's trailing ownership descriptor."""
    if not packed_items:
        raise RuntimeError(f"Missing ownership descriptor from source rank {source}")
    descriptor = packed_items[-1].to_host_bytes()
    if len(descriptor) < _OWNERSHIP_HEADER.size:
        raise RuntimeError(f"Invalid ownership descriptor from source rank {source}")
    magic, count = _OWNERSHIP_HEADER.unpack_from(descriptor)
    flags = descriptor[_OWNERSHIP_HEADER.size :]
    if magic != _OWNERSHIP_MAGIC or count != len(flags):
        raise RuntimeError(f"Invalid ownership descriptor from source rank {source}")
    if len(packed_items) - 1 != count or any(flag not in (0, 1) for flag in flags):
        raise RuntimeError(f"Ownership metadata does not match source rank {source}")
    return packed_items[:-1], tuple(map(bool, flags))


def _encode_ownership_descriptor(owned: Sequence[bool]) -> bytes:
    """Encode ownership flags for one source-destination payload stream."""
    return _OWNERSHIP_HEADER.pack(_OWNERSHIP_MAGIC, len(owned)) + bytes(owned)


def _discard_remote_sources(
    exchange: SparseAlltoall,
    sources: Sequence[int],
) -> None:
    """Best-effort release of unconsumed collective output."""
    for source in sources:
        with suppress(Exception):
            exchange.extract(source)


def _validate_routed_slices(
    pieces: Sequence[RoutedSlice],
    size: int,
    destinations: set[int],
) -> None:
    """Validate one resolver's ordered, non-overlapping chunk slices."""
    stops: dict[int, int] = {}
    owned: list[RowRange] = []
    for piece in pieces:
        if piece.destination not in destinations:
            raise ValueError(f"Unexpected slice destination {piece.destination}")
        if not 0 <= piece.start < piece.stop <= size:
            raise ValueError(f"Invalid routed slice [{piece.start}, {piece.stop})")
        if piece.start < stops.get(piece.destination, 0):
            raise ValueError("Routed slices must be ordered and non-overlapping")
        stops[piece.destination] = piece.stop
        if piece.owned:
            owned.append((piece.start, piece.stop))
    if any(left[1] > right[0] for left, right in pairwise(sorted(owned))):
        raise ValueError("Owned source rows may have only one destination")


@dataclass
class PreparedExchange:
    """Spillable prepared slices that must be consumed or cleared."""

    local_chunks: ChunkStore
    local_owned: tuple[bool, ...]
    exchange: SparseAlltoall | None
    remote_sources: tuple[int, ...]

    def _take_remote_source(
        self, context: Context, source: int
    ) -> tuple[ChunkStore, tuple[bool, ...]]:
        """Transfer one remote source from the collective to spillable storage."""
        chunks = ChunkStore(context)
        try:
            packed_items = (
                tuple(self.exchange.extract(source))
                if self.exchange is not None
                else ()
            )
            self.remote_sources = tuple(
                candidate for candidate in self.remote_sources if candidate != source
            )
            payloads, owned = _decode_ownership_descriptor(packed_items, source)
            for packed in payloads:
                chunks.insert(
                    Message(0, TableChunk.from_packed_data(packed, context.br()))
                )
        except BaseException:
            chunks.clear()
            raise
        return chunks, owned

    async def iter_tagged_source_order(
        self,
        context: Context,
        rank: int,
    ) -> AsyncGenerator[tuple[Message[TableChunk], bool]]:
        """Yield local and remote slices with ownership in source-rank order."""
        sources = {*self.remote_sources, rank}
        try:
            for source in sorted(sources):
                if source == rank:
                    chunks, owned = self.local_chunks, self.local_owned
                else:
                    chunks, owned = self._take_remote_source(context, source)
                try:
                    if len(chunks) != len(owned):
                        raise RuntimeError("Ownership metadata does not match chunks")
                    for message, is_owned in zip(chunks, owned, strict=True):
                        yield message, is_owned
                finally:
                    chunks.clear()
            self.exchange = None
            self.remote_sources = ()
        finally:
            self.clear()

    async def iter_source_order(
        self,
        context: Context,
        rank: int,
    ) -> AsyncGenerator[Message[TableChunk]]:
        """Yield local and remote slices in source-rank order."""
        tagged = self.iter_tagged_source_order(context, rank)
        async with aclosing(tagged):
            async for message, _ in tagged:
                yield message

    async def take_source_order(
        self,
        context: Context,
        rank: int,
    ) -> ChunkStore:
        """Transfer all prepared chunks to one ordered spillable store."""
        chunks = ChunkStore(context)
        try:
            async for message in self.iter_source_order(context, rank):
                chunks.insert(message)
        except BaseException:
            chunks.clear()
            raise
        return chunks

    def clear(self) -> None:
        """Release retained local and remote messages."""
        self.local_chunks.clear()
        self.local_owned = ()
        exchange, self.exchange = self.exchange, None
        sources, self.remote_sources = self.remote_sources, ()
        if exchange is not None:
            _discard_remote_sources(exchange, sources)


async def replay_buffer[SummaryT](
    buffer: ChunkStore,
    summaries: Sequence[SummaryT | None] = (),
) -> AsyncGenerator[tuple[Message[TableChunk], SummaryT | None]]:
    """Yield buffered messages and optional summaries, then clear the store."""
    try:
        if summaries and len(summaries) != len(buffer):
            raise ValueError("Summaries must match the buffered chunks")
        for position, message in enumerate(buffer):
            yield message, summaries[position] if summaries else None
    finally:
        buffer.clear()


class SliceExchange:
    """Route supplied slices to destination demands."""

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        collective_id: int,
    ) -> None:
        self.context = context
        self.comm = comm
        self.collective_id = collective_id

    async def _finish(self, exchange: SparseAlltoall) -> None:
        """Finish collective insertion before allowing task cancellation."""
        task = asyncio.create_task(exchange.insert_finished(self.context))
        try:
            await asyncio.shield(task)
        except BaseException:
            await task
            raise

    async def exchange[SummaryT, OwnedT, NeededT](
        self,
        source: AsyncGenerator[tuple[Message[TableChunk], SummaryT | None]],
        plan: ExchangePlan[OwnedT, NeededT],
        *,
        resolver: SliceResolver[SummaryT, OwnedT, NeededT],
    ) -> PreparedExchange:
        """Consume a source and return prepared slices owned by the caller."""
        destinations = tuple(
            dict.fromkeys(
                destination
                for destination, _ in plan.outgoing
                if destination != self.comm.rank
            )
        )
        exchange = (
            SparseAlltoall(
                self.context,
                self.comm,
                self.collective_id,
                srcs=plan.sources,
                dsts=destinations,
            )
            if plan.sources or destinations
            else None
        )
        output = ChunkStore(self.context)
        output_owned: list[bool] = []
        remote_owned: dict[int, list[bool]] = {
            destination: [] for destination in destinations
        }
        try:
            async with aclosing(source):
                async for message, summary in source:
                    chunk = TableChunk.from_message(message, br=self.context.br())
                    size = chunk.shape[0]
                    if size == 0:
                        local_demand = next(
                            (
                                demand
                                for destination, demand in plan.outgoing
                                if destination == self.comm.rank
                            ),
                            None,
                        )
                        if local_demand is not None:
                            output.insert(Message(0, chunk))
                            output_owned.append(local_demand.owned is not None)
                        continue
                    requests = plan.outgoing
                    if not requests:
                        continue
                    pieces = (
                        resolver.resolve_summary(size, summary, requests)
                        if resolver.resolve_summary is not None
                        else None
                    )
                    valid_destinations = {destination for destination, _ in requests}
                    if pieces is not None:
                        _validate_routed_slices(pieces, size, valid_destinations)
                    if (
                        pieces is not None
                        and len(pieces) == 1
                        and pieces[0].start == 0
                        and pieces[0].stop == size
                    ):
                        piece = pieces[0]
                        if piece.destination == self.comm.rank:
                            output.insert(Message(0, chunk))
                            output_owned.append(piece.owned)
                        else:
                            assert exchange is not None
                            await self._send_whole(exchange, piece.destination, chunk)
                            remote_owned[piece.destination].append(piece.owned)
                        continue
                    if pieces == []:
                        continue
                    available, extra = await make_table_chunks_available_or_wait(
                        self.context,
                        chunk,
                        reserve_extra=0,
                        net_memory_delta=0,
                    )
                    with opaque_memory_usage(extra):
                        if pieces is None:
                            pieces = await resolver.resolve_chunk(
                                available, summary, requests
                            )
                            _validate_routed_slices(pieces, size, valid_destinations)
                        local_pieces = []
                        for piece in pieces:
                            if piece.destination == self.comm.rank:
                                local_pieces.append(piece)
                            else:
                                assert exchange is not None
                                await self._send(
                                    exchange,
                                    piece.destination,
                                    available,
                                    piece.start,
                                    piece.stop,
                                )
                                remote_owned[piece.destination].append(piece.owned)
                        if (
                            len(local_pieces) == 1
                            and local_pieces[0].start == 0
                            and local_pieces[0].stop == size
                        ):
                            output.insert(Message(0, available))
                            output_owned.append(local_pieces[0].owned)
                        else:
                            for piece in local_pieces:
                                await self._store_local(
                                    output,
                                    available,
                                    piece.start,
                                    piece.stop,
                                )
                                output_owned.append(piece.owned)
            if exchange is not None:
                for destination in destinations:
                    exchange.insert(
                        destination,
                        PackedData.from_host_bytes(
                            _encode_ownership_descriptor(remote_owned[destination]),
                            self.context.br(),
                        ),
                    )
        except BaseException:
            output.clear()
            if exchange is not None:
                with suppress(BaseException):
                    await self._finish(exchange)
                _discard_remote_sources(exchange, plan.sources)
            raise
        if exchange is not None:
            try:
                await self._finish(exchange)
            except BaseException:
                output.clear()
                _discard_remote_sources(exchange, plan.sources)
                raise
        return PreparedExchange(
            output,
            tuple(output_owned),
            exchange,
            plan.sources,
        )

    async def exchange_whole(
        self,
        source: AsyncGenerator[Message[TableChunk]],
        routes: Sequence[Sequence[int]],
    ) -> PreparedExchange:
        """Move complete source messages along a globally agreed route graph."""
        if any(len(set(destinations)) > 1 for destinations in routes):
            raise ValueError("A complete owned message may move to only one rank")
        destinations = tuple(dict.fromkeys(routes[self.comm.rank]))
        plan: ExchangePlan[None, None] = ExchangePlan.from_routes(
            self.comm.rank,
            routes,
            tuple(
                (destination, InputDemand(owned=None, needed=None))
                for destination in destinations
            ),
        )

        async def summarized_source() -> AsyncGenerator[
            tuple[Message[TableChunk], None]
        ]:
            async with aclosing(source):
                async for message in source:
                    yield message, None

        async def resolve(
            chunk: TableChunk,
            _summary: None,
            requests: Sequence[tuple[int, InputDemand[None, None]]],
        ) -> list[RoutedSlice]:
            return [
                RoutedSlice(destination, 0, chunk.shape[0], owned=True)
                for destination, _ in requests
                if chunk.shape[0]
            ]

        def resolve_summary(
            size: int,
            _summary: None,
            requests: Sequence[tuple[int, InputDemand[None, None]]],
        ) -> list[RoutedSlice]:
            return [
                RoutedSlice(destination, 0, size, owned=True)
                for destination, _ in requests
                if size
            ]

        return await self.exchange(
            summarized_source(),
            plan,
            resolver=SliceResolver(
                resolve_chunk=resolve,
                resolve_summary=resolve_summary,
            ),
        )

    async def _store_local(
        self,
        output: ChunkStore,
        chunk: TableChunk,
        start: int,
        stop: int,
    ) -> None:
        """Copy one local slice into spillable storage."""
        piece = self._slice_chunk(chunk, start, stop)
        reservation = await self.context.memory(MemoryType.DEVICE).reserve_or_wait(
            piece.into_packed_data_cost(),
            net_memory_delta=0,
        )
        with opaque_memory_usage(reservation):
            table = piece.table_view().copy(
                piece.stream,
                self.context.br().device_mr,
            )
        output.insert(
            Message(
                0,
                TableChunk.from_pylibcudf_table(
                    table,
                    piece.stream,
                    exclusive_view=True,
                    br=self.context.br(),
                ),
            )
        )

    def _slice_chunk(
        self,
        chunk: TableChunk,
        start: int,
        stop: int,
    ) -> TableChunk:
        """Return a non-owning view of one half-open chunk slice."""
        table = (
            plc.Table(chunk.table_view().columns())
            if start == 0 and stop == chunk.shape[0]
            else plc.copying.slice(
                chunk.table_view(), [start, stop], stream=chunk.stream
            )[0]
        )
        return TableChunk.from_pylibcudf_table(
            table,
            chunk.stream,
            exclusive_view=False,
            br=self.context.br(),
        )

    async def _send(
        self,
        exchange: SparseAlltoall,
        destination: int,
        chunk: TableChunk,
        start: int,
        stop: int,
    ) -> None:
        """Pack and send one half-open chunk slice to a destination rank."""
        piece = self._slice_chunk(chunk, start, stop)
        piece, extra = await make_table_chunks_available_or_wait(
            self.context,
            piece,
            reserve_extra=piece.into_packed_data_cost(),
            net_memory_delta=0,
        )
        exchange.insert(destination, piece.into_packed_data(extra))

    async def _send_whole(
        self,
        exchange: SparseAlltoall,
        destination: int,
        chunk: TableChunk,
    ) -> None:
        """Move one complete chunk to a destination without slicing it."""
        pack_cost = chunk.into_packed_data_cost()
        if not chunk.is_available() and pack_cost:
            chunk, reservation = await make_table_chunks_available_or_wait(
                self.context,
                chunk,
                reserve_extra=pack_cost,
                net_memory_delta=0,
            )
        else:
            reservation = await self.context.memory(MemoryType.DEVICE).reserve_or_wait(
                pack_cost, net_memory_delta=0
            )
        exchange.insert(destination, chunk.into_packed_data(reservation))
