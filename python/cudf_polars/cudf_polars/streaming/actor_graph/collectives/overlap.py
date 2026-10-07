# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Overlap-context exchange helpers for ordered streaming operators."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc
from cudf_streaming.partition_utils import unpack_and_concat, unpack_and_concat_cost
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.utils.cuda_stream import stream_ordered_after

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.packed_data import PackedData
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext


RowRange = tuple[int, int]
_INT64_DTYPE = DataType(pl.Int64())


@dataclass
class BufferedChunk:
    """
    A table chunk with its global row span.

    The span is half-open: ``[row_start, row_stop)``.
    """

    sequence_number: int
    chunk: TableChunk
    row_start: int
    num_rows: int

    @property
    def row_stop(self) -> int:
        """Global row offset immediately after this chunk."""
        return self.row_start + self.num_rows


@dataclass
class RowExchangeResult:
    """Rows owned by this rank plus temporary ghost context rows."""

    local_source: BufferedChunkSource
    received_owned_source: BufferedChunkSource
    ghost_source: BufferedChunkSource
    local_owned_intervals: tuple[RowRange, ...]
    local_empty_span: RowRange | None

    def clear_received(self) -> None:
        """Discard rows received from remote ranks."""
        self.received_owned_source.clear()
        self.ghost_source.clear()

    def has_remote_owned(self) -> bool:
        """Return whether this rank received remotely owned output rows."""
        return self.received_owned_source.has_chunks()

    async def iter_local_owned(
        self,
        *,
        ir_context: IRExecutionContext,
        include_empty_chunks: bool = False,
    ) -> AsyncIterator[BufferedChunk]:
        """Yield locally owned chunks directly from the spillable source."""
        for start, stop in self.local_owned_intervals:
            async for chunk in self.local_source.iter_region_chunks(
                start,
                stop,
                ir_context=ir_context,
                include_empty_in_span=(start, stop) if include_empty_chunks else None,
            ):
                yield chunk
        if self.local_empty_span is not None and not self.local_owned_intervals:
            start, stop = self.local_empty_span
            async for chunk in self.local_source.iter_region_chunks(
                start,
                stop,
                ir_context=ir_context,
                include_empty_in_span=self.local_empty_span
                if include_empty_chunks
                else None,
            ):
                yield chunk

    async def iter_remote_owned(
        self,
        *,
        ir_context: IRExecutionContext,
    ) -> AsyncIterator[BufferedChunk]:
        """Yield owned chunks received from remote ranks."""
        async for chunk in self.received_owned_source.iter_chunks(
            ir_context=ir_context
        ):
            yield chunk

    async def iter_owned(
        self,
        *,
        ir_context: IRExecutionContext,
        include_empty_chunks: bool = False,
    ) -> AsyncIterator[BufferedChunk]:
        """Yield local and received owned chunks in global row order."""
        local_chunks = self.iter_local_owned(
            ir_context=ir_context,
            include_empty_chunks=include_empty_chunks,
        )
        remote_chunks = self.iter_remote_owned(ir_context=ir_context)
        local = await _anext_or_none(local_chunks)
        remote = await _anext_or_none(remote_chunks)
        while local is not None or remote is not None:
            if remote is None or (
                local is not None
                and (local.row_start, local.sequence_number)
                <= (remote.row_start, remote.sequence_number)
            ):
                assert local is not None
                yield local
                local = await _anext_or_none(local_chunks)
            else:
                yield remote
                remote = await _anext_or_none(remote_chunks)

    async def iter_ghosts(
        self,
        *,
        ir_context: IRExecutionContext,
    ) -> AsyncIterator[BufferedChunk]:
        """Yield ghost chunks received from remote ranks."""
        async for chunk in self.ghost_source.iter_chunks(ir_context=ir_context):
            yield chunk


@dataclass(frozen=True)
class BufferedChunkIndex:
    """Searchable view over buffered chunks sorted by global row span."""

    chunks: tuple[BufferedChunk, ...]
    row_stops: tuple[int, ...]

    @classmethod
    def from_chunks(cls, chunks: Sequence[BufferedChunk]) -> BufferedChunkIndex:
        """Build an index over globally sorted buffered chunks."""
        ordered = tuple(
            sorted(chunks, key=lambda chunk: (chunk.row_start, chunk.sequence_number))
        )
        _validate_buffered_chunk_order(ordered)
        return cls(ordered, tuple(chunk.row_stop for chunk in ordered))

    def first_intersecting(self, row_start: int) -> int:
        """Return the first chunk that may intersect ``row_start``."""
        return bisect_right(self.row_stops, row_start)


@dataclass(frozen=True)
class StoredBufferedChunk:
    """Stored message ID and row-span metadata for one buffered chunk."""

    mid: int
    sequence_number: int
    row_start: int
    num_rows: int

    @property
    def row_stop(self) -> int:
        """Row offset immediately after this chunk, before any source offset."""
        return self.row_start + self.num_rows


class BufferedChunkSource:
    """
    Spillable, row-addressable source of buffered table chunks.

    The payload lives in the context's spillable message store. The small
    row-span index stays in Python so callers can resolve row-slice requests
    without first materializing every chunk.
    """

    def __init__(self, context: Context) -> None:
        self._context = context
        self._store = context.spillable_messages()
        self._records: list[StoredBufferedChunk] = []
        self._cache: dict[int, BufferedChunk] = {}
        self._extracted: set[int] = set()
        self._release_index = 0
        self._row_start_offset = 0

    def insert(self, chunk: BufferedChunk) -> None:
        """Insert a chunk into the spillable source."""
        if self._cache or self._extracted:
            raise RuntimeError("Cannot insert after materializing buffered chunks")
        if self._records and chunk.row_start < self._records[-1].row_stop:
            raise RuntimeError("Buffered chunks must be inserted in row-span order")
        mid = self._store.insert(Message(chunk.sequence_number, chunk.chunk))
        self._records.append(
            StoredBufferedChunk(
                mid,
                chunk.sequence_number,
                chunk.row_start,
                chunk.num_rows,
            )
        )

    def set_row_start_offset(self, offset: int) -> None:
        """Shift all stored row spans by a global row-start offset."""
        if self._cache or self._extracted:
            raise RuntimeError("Cannot shift buffered chunks after materialization")
        self._row_start_offset = offset

    def clear(self) -> None:
        """Discard all stored and materialized chunks."""
        for record in self._records:
            if record.mid not in self._extracted:
                self._store.extract(mid=record.mid)
        self._records.clear()
        self._cache.clear()
        self._extracted.clear()
        self._release_index = 0
        self._row_start_offset = 0

    def has_chunks(self) -> bool:
        """Return whether this source has stored chunks."""
        return bool(self._records)

    def release_cached_before(self, row_stop: int) -> None:
        """Release stored or materialized chunks ending at or before ``row_stop``."""
        while self._release_index < len(self._records):
            record = self._records[self._release_index]
            if self._row_stop(record) > row_stop:
                break
            if self._cache.pop(record.mid, None) is None:
                self._store.extract(mid=record.mid)
                self._extracted.add(record.mid)
            self._release_index += 1

    def _row_start(self, record: StoredBufferedChunk) -> int:
        return record.row_start + self._row_start_offset

    def _row_stop(self, record: StoredBufferedChunk) -> int:
        return record.row_stop + self._row_start_offset

    async def _chunk_for(self, record: StoredBufferedChunk) -> BufferedChunk:
        """Return an available buffered chunk for ``record``."""
        if (cached := self._cache.get(record.mid)) is not None:
            return cached
        if record.mid in self._extracted:
            raise RuntimeError(
                "Requested buffered chunk after it was released from cache"
            )
        msg = self._store.extract(mid=record.mid)
        if msg.sequence_number != record.sequence_number:
            raise RuntimeError(
                "Buffered chunk metadata/message sequence mismatch: "
                f"{record.sequence_number} != {msg.sequence_number}"
            )
        self._extracted.add(record.mid)
        chunk = TableChunk.from_message(msg, br=self._context.br())
        nrows, _ = chunk.shape
        if nrows != record.num_rows:
            raise RuntimeError(
                "Buffered chunk metadata/message row-count mismatch: "
                f"{record.num_rows} != {nrows}"
            )
        chunk, extra = await make_table_chunks_available_or_wait(
            self._context,
            chunk,
            reserve_extra=0,
            net_memory_delta=0,
        )
        with opaque_memory_usage(extra):
            buffered = BufferedChunk(
                record.sequence_number,
                chunk,
                self._row_start(record),
                record.num_rows,
            )
        self._cache[record.mid] = buffered
        return buffered

    async def iter_region_chunks(
        self,
        row_start: int,
        row_stop: int,
        *,
        ir_context: IRExecutionContext,
        copy_result: bool = False,
        include_empty_in_span: RowRange | None = None,
    ) -> AsyncIterator[BufferedChunk]:
        """Yield chunks intersecting a complete global row range."""
        del ir_context
        if row_start > row_stop:
            raise ValueError(f"Invalid row range [{row_start}, {row_stop})")
        expected_start = row_start
        for record in self._records:
            record_start = self._row_start(record)
            record_stop = self._row_stop(record)
            if record.num_rows == 0:
                if include_empty_in_span is not None and _empty_chunk_in_span(
                    record_start, *include_empty_in_span
                ):
                    yield await self._chunk_for(record)
                continue
            if record_stop <= row_start:
                continue
            if record_start >= row_stop:
                break
            start = max(row_start, record_start)
            stop = min(row_stop, record_stop)
            if start < stop:
                if start != expected_start:
                    raise RuntimeError(
                        "Buffered chunks do not cover requested row range "
                        f"[{row_start}, {row_stop})"
                    )
                yield await self.extract_chunk_slice(
                    start,
                    stop,
                    record=record,
                    copy_result=copy_result,
                )
                expected_start = stop
        if expected_start != row_stop:
            raise RuntimeError(
                "Buffered chunks do not cover requested row range "
                f"[{row_start}, {row_stop})"
            )

    async def extract_chunk_slice(
        self,
        row_start: int,
        row_stop: int,
        *,
        record: StoredBufferedChunk,
        copy_result: bool = False,
    ) -> BufferedChunk:
        """Return one stored chunk slice."""
        _validate_row_range(row_start, row_stop)
        buffered = await self._chunk_for(record)
        full_chunk = row_start == buffered.row_start and row_stop == buffered.row_stop
        if full_chunk:
            chunk = buffered.chunk
        else:
            chunk = TableChunk.from_pylibcudf_table(
                plc.copying.slice(
                    buffered.chunk.table_view(),
                    [row_start - buffered.row_start, row_stop - buffered.row_start],
                    stream=buffered.chunk.stream,
                )[0],
                buffered.chunk.stream,
                exclusive_view=False,
                br=self._context.br(),
            )
        if copy_result:
            reservation = await self._context.memory(MemoryType.DEVICE).reserve_or_wait(
                chunk.data_alloc_size(), net_memory_delta=0
            )
            with opaque_memory_usage(reservation):
                chunk = TableChunk.from_pylibcudf_table(
                    chunk.table_view().copy(chunk.stream, self._context.br().device_mr),
                    chunk.stream,
                    exclusive_view=True,
                    br=self._context.br(),
                )
        if full_chunk and not copy_result:
            # Local replay shares the original chunk. Callers may release source
            # cache only after all windows that need this chunk are complete.
            return replace(buffered, sequence_number=-1)
        return BufferedChunk(-1, chunk, row_start, chunk.shape[0])

    async def iter_chunks(
        self,
        *,
        ir_context: IRExecutionContext,
        copy_result: bool = False,
    ) -> AsyncIterator[BufferedChunk]:
        """Yield all stored chunks in row-span order."""
        del ir_context
        for record in self._records:
            start = self._row_start(record)
            stop = self._row_stop(record)
            if record.num_rows == 0:
                yield await self._chunk_for(record)
            else:
                yield await self.extract_chunk_slice(
                    start,
                    stop,
                    record=record,
                    copy_result=copy_result,
                )

    async def extract_region_chunks(
        self,
        row_start: int,
        row_stop: int,
        *,
        ir_context: IRExecutionContext,
        copy_result: bool = False,
        include_empty_in_span: RowRange | None = None,
    ) -> list[BufferedChunk]:
        """Return chunks intersecting a complete global row range."""
        return [
            chunk
            async for chunk in self.iter_region_chunks(
                row_start,
                row_stop,
                ir_context=ir_context,
                copy_result=copy_result,
                include_empty_in_span=include_empty_in_span,
            )
        ]

    async def extract_region(
        self,
        row_start: int,
        row_stop: int,
        *,
        ir_context: IRExecutionContext,
        copy_result: bool = False,
    ) -> TableChunk:
        """Return one chunk containing a complete global row range."""
        chunks = [
            chunk.chunk
            async for chunk in self.iter_region_chunks(
                row_start,
                row_stop,
                ir_context=ir_context,
                copy_result=copy_result,
            )
        ]
        if len(chunks) == 1:
            return chunks[0]
        reservation = await self._context.memory(MemoryType.DEVICE).reserve_or_wait(
            sum(chunk.data_alloc_size() for chunk in chunks),
            net_memory_delta=0,
        )
        chunk_streams = [chunk.stream for chunk in chunks]
        with (
            opaque_memory_usage(reservation),
            stream_ordered_after(ir_context.get_cuda_stream, chunk_streams) as stream,
        ):
            table = plc.concatenate.concatenate(
                [chunk.table_view() for chunk in chunks],
                stream=stream,
                mr=self._context.br().device_mr,
            )
            return TableChunk.from_pylibcudf_table(
                table,
                stream=stream,
                exclusive_view=True,
                br=self._context.br(),
            )


async def _anext_or_none(
    chunks: AsyncIterator[BufferedChunk],
) -> BufferedChunk | None:
    """Return the next chunk from an async iterator, or ``None``."""
    try:
        return await anext(chunks)
    except StopAsyncIteration:
        return None


class RowExchangeKind(IntEnum):
    """Kind of row payload carried by a row-slice exchange."""

    OWNED = 0
    GHOST = 1


@dataclass(frozen=True)
class RowExchangePlan:
    """
    Row-offset routing plan for owned and ghost slice exchange.

    ``source_spans`` describe the rows currently owned by each source rank.
    ``output_spans`` describe rows owned by each output rank after any boundary
    adjustment. ``ghost_requests`` describe extra source rows needed by each
    output rank for local evaluation.

    This plan is intentionally limited to global row offsets. Operators that
    route by value boundaries, output partition IDs, or grouped state should
    resolve those semantics to row slices before using this transport, or use a
    richer exchange metadata layer.

    Empty chunks do not own rows and are not represented in remote row-slice
    movement. Operators that need to preserve empty chunk markers across
    ownership changes must carry that policy outside this row-span transport.
    """

    source_spans: tuple[RowRange, ...]
    output_spans: tuple[RowRange, ...]
    ghost_requests: tuple[tuple[RowRange, ...], ...]

    @classmethod
    def from_spans(
        cls,
        source_spans: Sequence[RowRange],
        output_spans: Sequence[RowRange],
        ghost_requests: Sequence[Sequence[RowRange]],
    ) -> RowExchangePlan:
        """Build a row exchange plan from explicit row spans."""
        plan = cls(
            tuple(source_spans),
            tuple(output_spans),
            tuple(tuple(_merge_intervals(requests)) for requests in ghost_requests),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        """Validate row-span shape and monotonicity."""
        if len(self.source_spans) != len(self.output_spans):
            raise ValueError(
                "RowExchangePlan source and output span counts must match: "
                f"{len(self.source_spans)} != {len(self.output_spans)}"
            )
        if len(self.ghost_requests) != len(self.output_spans):
            raise ValueError(
                "RowExchangePlan ghost request count must match output span count: "
                f"{len(self.ghost_requests)} != {len(self.output_spans)}"
            )
        _validate_plan_spans("source", self.source_spans, contiguous=True)
        _validate_plan_spans("output", self.output_spans, contiguous=True)
        total_rows = self.total_rows
        if self.output_spans and self.output_spans[-1][1] != total_rows:
            raise ValueError(
                "RowExchangePlan output spans must cover the source row domain: "
                f"output stops at {self.output_spans[-1][1]}, source stops at "
                f"{total_rows}"
            )
        for rank, (start, stop) in enumerate(self.output_spans):
            if stop > total_rows:
                raise ValueError(
                    "Invalid RowExchangePlan output span for rank "
                    f"{rank}: [{start}, {stop}) outside [0, {total_rows})"
                )
        for rank, requests in enumerate(self.ghost_requests):
            for start, stop in requests:
                if start < 0 or stop < start or stop > total_rows:
                    raise ValueError(
                        "Invalid RowExchangePlan ghost request for rank "
                        f"{rank}: [{start}, {stop}) outside [0, {total_rows})"
                    )

    @property
    def total_rows(self) -> int:
        """Total rows in the source stream."""
        if not self.source_spans:
            return 0
        return self.source_spans[-1][1]

    def source_span(self, rank: int) -> RowRange:
        """Return the source rows owned by ``rank``."""
        return self.source_spans[rank]

    def output_span(self, rank: int) -> RowRange:
        """Return the output rows owned by ``rank``."""
        return self.output_spans[rank]

    def ghost_intervals_from_source(
        self, source_rank: int, destination_rank: int
    ) -> tuple[RowRange, ...]:
        """Return source-owned row intervals needed as destination ghosts."""
        return tuple(
            _intersections(
                self.source_spans[source_rank],
                self.ghost_requests[destination_rank],
            )
        )

    def owned_intervals_from_source(
        self, source_rank: int, destination_rank: int
    ) -> tuple[RowRange, ...]:
        """Return source-owned row intervals owned by destination output."""
        interval = _range_intersection(
            self.source_spans[source_rank],
            self.output_spans[destination_rank],
        )
        return () if interval is None else (interval,)

    def remote_ghost_sources(self, rank: int) -> tuple[int, ...]:
        """Return remote ranks that provide ghosts to ``rank``."""
        return tuple(
            source_rank
            for source_rank in range(len(self.source_spans))
            if source_rank != rank
            and self.ghost_intervals_from_source(source_rank, rank)
        )

    def remote_ghost_destinations(self, rank: int) -> tuple[int, ...]:
        """Return remote ranks that need ghosts from ``rank``."""
        return tuple(
            destination_rank
            for destination_rank in range(len(self.output_spans))
            if destination_rank != rank
            and self.ghost_intervals_from_source(rank, destination_rank)
        )

    def remote_owned_sources(self, rank: int) -> tuple[int, ...]:
        """Return remote ranks that own rows this rank must output."""
        return tuple(
            source_rank
            for source_rank in range(len(self.source_spans))
            if source_rank != rank
            and self.owned_intervals_from_source(source_rank, rank)
        )

    def remote_owned_destinations(self, rank: int) -> tuple[int, ...]:
        """Return remote ranks that must output rows owned by this rank."""
        return tuple(
            destination_rank
            for destination_rank in range(len(self.output_spans))
            if destination_rank != rank
            and self.owned_intervals_from_source(rank, destination_rank)
        )

    def remote_sources(self, rank: int) -> tuple[int, ...]:
        """Return remote ranks that send owned or ghost rows to ``rank``."""
        return tuple(
            sorted(
                {
                    *self.remote_owned_sources(rank),
                    *self.remote_ghost_sources(rank),
                }
            )
        )

    def remote_destinations(self, rank: int) -> tuple[int, ...]:
        """Return remote ranks that receive owned or ghost rows from ``rank``."""
        return tuple(
            sorted(
                {
                    *self.remote_owned_destinations(rank),
                    *self.remote_ghost_destinations(rank),
                }
            )
        )


@dataclass(frozen=True)
class ResolvedRowSend:
    """One resolved row slice owed to another rank."""

    destination: int
    start: int
    stop: int
    kind: RowExchangeKind


@dataclass(frozen=True)
class ResolvedRowExchangePlan:
    """Resolved row-slice sends plus the sparse rendezvous shape."""

    sends: tuple[ResolvedRowSend, ...]
    expected_sources: tuple[int, ...]
    candidate_destinations: tuple[int, ...]

    @classmethod
    def from_sends(
        cls,
        sends: Sequence[ResolvedRowSend],
        *,
        expected_sources: Sequence[int],
        candidate_destinations: Sequence[int] | None = None,
    ) -> ResolvedRowExchangePlan:
        """Build a validated sparse-exchange plan from resolved sends."""
        send_tuple = tuple(
            sorted(
                sends,
                key=lambda send: (
                    send.destination,
                    send.start,
                    send.stop,
                    int(send.kind),
                ),
            )
        )
        destination_tuple = tuple(
            sorted(
                {send.destination for send in send_tuple}
                if candidate_destinations is None
                else set(candidate_destinations)
            )
        )
        source_tuple = tuple(sorted(set(expected_sources)))
        destination_set = set(destination_tuple)
        missing = sorted(
            {send.destination for send in send_tuple}.difference(destination_set)
        )
        if missing:
            raise ValueError(
                "Resolved row sends target destinations that are not in the "
                f"exchange plan: {missing}"
            )
        return cls(send_tuple, source_tuple, destination_tuple)


class RowExchange:
    """Sparse exchange of owned and ghost row slices for one local rank."""

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        ir_context: IRExecutionContext,
        plan: RowExchangePlan,
        collective_id: int,
    ) -> None:
        self.context = context
        self.comm = comm
        self.ir_context = ir_context
        self.plan = plan
        self.plan.validate()
        if len(self.plan.source_spans) != comm.nranks:
            raise ValueError(
                "RowExchangePlan rank count must match communicator size: "
                f"{len(self.plan.source_spans)} != {comm.nranks}"
            )
        self.collective_id = collective_id

    async def exchange(self, local_source: BufferedChunkSource) -> RowExchangeResult:
        """Exchange owned and ghost rows and return local output context."""
        sends: list[ResolvedRowSend] = []
        for dst in self.plan.remote_destinations(self.comm.rank):
            sends.extend(
                ResolvedRowSend(dst, start, stop, RowExchangeKind.OWNED)
                for start, stop in self.plan.owned_intervals_from_source(
                    self.comm.rank, dst
                )
            )
            sends.extend(
                ResolvedRowSend(dst, start, stop, RowExchangeKind.GHOST)
                for start, stop in self.plan.ghost_intervals_from_source(
                    self.comm.rank, dst
                )
            )
        return await exchange_resolved_slices(
            self.context,
            self.comm,
            self.ir_context,
            local_source,
            local_owned_intervals=self.plan.owned_intervals_from_source(
                self.comm.rank, self.comm.rank
            ),
            local_empty_span=self.plan.output_span(self.comm.rank),
            plan=ResolvedRowExchangePlan.from_sends(
                sends,
                expected_sources=self.plan.remote_sources(self.comm.rank),
            ),
            collective_id=self.collective_id,
        )


async def exchange_resolved_slices(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    local_source: BufferedChunkSource,
    *,
    local_owned_intervals: Sequence[RowRange],
    local_empty_span: RowRange | None = None,
    plan: ResolvedRowExchangePlan,
    collective_id: int,
) -> RowExchangeResult:
    """Exchange resolved row slices with row-span metadata."""
    if not plan.expected_sources and not plan.candidate_destinations:
        return RowExchangeResult(
            local_source,
            BufferedChunkSource(context),
            BufferedChunkSource(context),
            tuple(local_owned_intervals),
            local_empty_span,
        )

    exchange = SparseAlltoall(
        context,
        comm,
        collective_id,
        srcs=plan.expected_sources,
        dsts=plan.candidate_destinations,
    )
    for send in plan.sends:
        await _send_resolved_slice(
            context,
            ir_context,
            exchange,
            local_source,
            send,
        )

    await exchange.insert_finished(context)

    received_owned_source = BufferedChunkSource(context)
    ghost_source = BufferedChunkSource(context)
    received_owned_sequence_number = 0
    ghost_sequence_number = 0
    for src in plan.expected_sources:
        pieces = exchange.extract(src)
        if len(pieces) % 2 != 0:
            raise RuntimeError(
                "RowExchange received an odd number of metadata/data payloads "
                f"from rank {src}: got {len(pieces)}"
            )
        # SparseAlltoall preserves insertion order for payloads from one source
        # to one destination. Each resolved slice is inserted as
        # metadata followed by data.
        for metadata, data in zip(pieces[::2], pieces[1::2], strict=True):
            start, stop, kind = await _unpack_slice_metadata(
                context, ir_context, metadata
            )
            chunk = await _unpack_sparse_chunk(context, ir_context, [data])
            buffered = BufferedChunk(-1, chunk, start, stop - start)
            if kind == RowExchangeKind.GHOST:
                ghost_source.insert(
                    replace(buffered, sequence_number=ghost_sequence_number)
                )
                ghost_sequence_number += 1
            else:
                received_owned_source.insert(
                    replace(
                        buffered,
                        sequence_number=received_owned_sequence_number,
                    )
                )
                received_owned_sequence_number += 1
    return RowExchangeResult(
        local_source,
        received_owned_source,
        ghost_source,
        tuple(local_owned_intervals),
        local_empty_span,
    )


def _empty_chunk_in_span(row_start: int, start: int, stop: int) -> bool:
    """Return whether an empty chunk belongs to a possibly-empty span."""
    if start == stop:
        return row_start == start
    # Empty chunks carry position but no rows. Include both boundaries so a
    # trailing empty chunk is preserved for the owner of the preceding rows.
    return start <= row_start <= stop


async def _send_resolved_slice(
    context: Context,
    ir_context: IRExecutionContext,
    exchange: SparseAlltoall,
    local_source: BufferedChunkSource,
    send: ResolvedRowSend,
) -> None:
    """Send one resolved slice and its row-span metadata."""
    if send.start >= send.stop:
        return
    async for chunk in local_source.iter_region_chunks(
        send.start,
        send.stop,
        ir_context=ir_context,
        copy_result=True,
    ):
        metadata = _slice_metadata_chunk(
            context,
            chunk.row_start,
            chunk.row_stop,
            kind=send.kind,
        )
        await _insert_sparse_chunk(context, exchange, send.destination, metadata)
        await _insert_sparse_chunk(context, exchange, send.destination, chunk.chunk)


def _slice_metadata_chunk(
    context: Context,
    start: int,
    stop: int,
    *,
    kind: RowExchangeKind,
) -> TableChunk:
    """Return a one-row metadata chunk for a resolved slice."""
    stream = context.br().stream_pool.get_stream()
    dtype = plc.DataType(plc.TypeId.INT64)
    table = plc.Table(
        [
            plc.Column.from_scalar(
                plc.Scalar.from_py(value, dtype, stream=stream),
                1,
                stream=stream,
            )
            for value in (start, stop, int(kind))
        ]
    )
    return TableChunk.from_pylibcudf_table(
        table,
        stream,
        exclusive_view=True,
        br=context.br(),
    )


async def _unpack_slice_metadata(
    context: Context,
    ir_context: IRExecutionContext,
    piece: PackedData,
) -> tuple[int, int, RowExchangeKind]:
    """Unpack one resolved-slice metadata payload."""
    chunk = await _unpack_sparse_chunk(context, ir_context, [piece])
    metadata = (
        DataFrame.from_table(
            chunk.table_view(),
            ["start", "stop", "kind"],
            [_INT64_DTYPE, _INT64_DTYPE, _INT64_DTYPE],
            chunk.stream,
        )
        .to_polars()
        .row(0)
    )
    return metadata[0], metadata[1], RowExchangeKind(metadata[2])


def _range_intersection(left: RowRange, right: RowRange) -> RowRange | None:
    """Return the non-empty intersection of two row ranges."""
    start = max(left[0], right[0])
    stop = min(left[1], right[1])
    if start < stop:
        return start, stop
    return None


def _validate_plan_spans(
    name: str,
    spans: Sequence[RowRange],
    *,
    contiguous: bool,
) -> None:
    """Validate monotone non-overlapping plan spans."""
    previous_stop = 0
    for rank, (start, stop) in enumerate(spans):
        if start < 0 or stop < start:
            raise ValueError(
                f"Invalid RowExchangePlan {name} span for rank {rank}: "
                f"[{start}, {stop})"
            )
        if contiguous and start != previous_stop:
            raise ValueError(
                f"RowExchangePlan {name} spans must be contiguous: rank {rank} "
                f"starts at {start}, expected {previous_stop}"
            )
        if not contiguous and start < previous_stop:
            raise ValueError(
                f"RowExchangePlan {name} spans must be non-overlapping: rank "
                f"{rank} starts at {start}, previous stop was {previous_stop}"
            )
        previous_stop = stop


def _intersections(span: RowRange, requests: Sequence[RowRange]) -> list[RowRange]:
    """Return sorted intersections between one source span and many requests."""
    intervals = [
        interval
        for request in requests
        if (interval := _range_intersection(span, request)) is not None
    ]
    return _merge_intervals(intervals)


def _merge_intervals(intervals: Sequence[RowRange]) -> list[RowRange]:
    """Return sorted non-empty intervals with overlaps merged."""
    merged: list[RowRange] = []
    for start, stop in sorted(intervals):
        if start >= stop:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, stop))
        else:
            previous_start, previous_stop = merged[-1]
            merged[-1] = (previous_start, max(previous_stop, stop))
    return merged


def _validate_buffered_chunk_order(input_chunks: Sequence[BufferedChunk]) -> None:
    """Validate that buffered chunks are sorted by row span."""
    previous_stop = None
    for chunk in input_chunks:
        if previous_stop is not None and chunk.row_start < previous_stop:
            raise RuntimeError("Buffered chunks are not sorted by row span")
        previous_stop = chunk.row_stop


def _validate_row_range(row_start: int, row_stop: int) -> None:
    """Validate one half-open row range."""
    if row_start >= row_stop:
        raise ValueError(f"Invalid row range [{row_start}, {row_stop})")


async def extract_region_chunks(
    context: Context,
    input_chunks: Sequence[BufferedChunk],
    row_start: int,
    row_stop: int,
    *,
    ir_context: IRExecutionContext,
    copy_result: bool = False,
) -> list[BufferedChunk]:
    """Slice buffered chunks intersecting a complete global row range."""
    return await extract_region_chunks_from_index(
        context,
        BufferedChunkIndex.from_chunks(input_chunks),
        row_start,
        row_stop,
        ir_context=ir_context,
        copy_result=copy_result,
    )


async def extract_region_chunks_from_index(
    context: Context,
    chunk_index: BufferedChunkIndex,
    row_start: int,
    row_stop: int,
    *,
    ir_context: IRExecutionContext,
    copy_result: bool = False,
) -> list[BufferedChunk]:
    """Slice indexed buffered chunks intersecting a complete global row range."""
    del ir_context
    _validate_row_range(row_start, row_stop)
    result: list[BufferedChunk] = []
    expected_start = row_start
    for buf in chunk_index.chunks[chunk_index.first_intersecting(row_start) :]:
        if buf.row_start >= row_stop:
            break
        start = max(row_start, buf.row_start)
        stop = min(row_stop, buf.row_stop)
        if start < stop:
            if start != expected_start:
                raise RuntimeError(
                    "Buffered chunks do not cover requested row range "
                    f"[{row_start}, {row_stop})"
                )
            full_chunk = start == buf.row_start and stop == buf.row_stop
            if start == buf.row_start and stop == buf.row_stop:
                chunk = buf.chunk
            else:
                chunk = TableChunk.from_pylibcudf_table(
                    plc.copying.slice(
                        buf.chunk.table_view(),
                        [start - buf.row_start, stop - buf.row_start],
                        stream=buf.chunk.stream,
                    )[0],
                    buf.chunk.stream,
                    exclusive_view=False,
                    br=context.br(),
                )
            if copy_result:
                reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
                    chunk.data_alloc_size(), net_memory_delta=0
                )
                with opaque_memory_usage(reservation):
                    chunk = TableChunk.from_pylibcudf_table(
                        chunk.table_view().copy(chunk.stream, context.br().device_mr),
                        chunk.stream,
                        exclusive_view=True,
                        br=context.br(),
                    )
            if full_chunk and not copy_result:
                result.append(replace(buf, sequence_number=-1))
            else:
                result.append(BufferedChunk(-1, chunk, start, chunk.shape[0]))
            expected_start = stop
    if expected_start != row_stop:
        raise RuntimeError(
            "Buffered chunks do not cover requested row range "
            f"[{row_start}, {row_stop})"
        )
    return result


async def extract_region(
    context: Context,
    input_chunks: Sequence[BufferedChunk],
    row_start: int,
    row_stop: int,
    *,
    ir_context: IRExecutionContext,
    copy_result: bool = False,
) -> TableChunk:
    """Return one chunk containing a complete global row range."""
    return await extract_region_from_index(
        context,
        BufferedChunkIndex.from_chunks(input_chunks),
        row_start,
        row_stop,
        ir_context=ir_context,
        copy_result=copy_result,
    )


async def extract_region_from_index(
    context: Context,
    chunk_index: BufferedChunkIndex,
    row_start: int,
    row_stop: int,
    *,
    ir_context: IRExecutionContext,
    copy_result: bool = False,
) -> TableChunk:
    """Return one chunk from indexed chunks containing a complete row range."""
    buffered = await extract_region_chunks_from_index(
        context,
        chunk_index,
        row_start,
        row_stop,
        ir_context=ir_context,
        copy_result=copy_result,
    )
    chunks = [chunk.chunk for chunk in buffered]
    if len(chunks) == 1:
        return chunks[0]
    reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
        sum(chunk.data_alloc_size() for chunk in chunks), net_memory_delta=0
    )
    chunk_streams = [chunk.stream for chunk in chunks]
    with (
        opaque_memory_usage(reservation),
        stream_ordered_after(ir_context.get_cuda_stream, chunk_streams) as stream,
    ):
        table = plc.concatenate.concatenate(
            [chunk.table_view() for chunk in chunks],
            stream=stream,
            mr=context.br().device_mr,
        )
        return TableChunk.from_pylibcudf_table(
            table, stream=stream, exclusive_view=True, br=context.br()
        )


def _row_count_chunk(context: Context, row_count: int, stream: Stream) -> TableChunk:
    """Return a single-row chunk containing one int64 row count."""
    col = plc.Column.from_scalar(
        plc.Scalar.from_py(
            row_count,
            plc.DataType(plc.TypeId.INT64),
            stream=stream,
        ),
        1,
        stream=stream,
    )
    return TableChunk.from_pylibcudf_table(
        plc.Table([col]),
        stream,
        exclusive_view=True,
        br=context.br(),
    )


async def gather_row_counts(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    *,
    local_rows: int,
    collective_id: int,
) -> list[int]:
    """Collect one local row count from every rank."""
    stream = context.br().stream_pool.get_stream()
    ag = AllGatherManager(context, comm, collective_id)
    with ag.inserting() as inserter:
        await inserter.insert(0, _row_count_chunk(context, local_rows, stream))
    table = await ag.extract_concatenated(stream, ordered=True, ir_context=ir_context)
    counts = (
        DataFrame.from_table(table, ["row_count"], [_INT64_DTYPE], stream)
        .to_polars()["row_count"]
        .to_list()
    )
    if len(counts) != comm.nranks:
        raise RuntimeError(
            "Row-count allgather returned an unexpected number of counts: "
            f"expected {comm.nranks}, got {len(counts)}"
        )
    return counts


async def _insert_sparse_chunk(
    context: Context,
    exchange: SparseAlltoall,
    dst: int,
    chunk: TableChunk,
) -> None:
    """Insert one table chunk into a sparse all-to-all exchange."""
    chunk, extra = await make_table_chunks_available_or_wait(
        context,
        chunk,
        reserve_extra=chunk.into_packed_data_cost(),
        net_memory_delta=0,
    )
    exchange.insert(dst, chunk.into_packed_data(extra))
    del chunk


async def _unpack_sparse_chunk(
    context: Context,
    ir_context: IRExecutionContext,
    pieces: Sequence[PackedData],
) -> TableChunk:
    """Unpack one expected sparse all-to-all payload into a table chunk."""
    stream = context.br().stream_pool.get_stream()
    reservation = await reserve_memory(
        context,
        unpack_and_concat_cost(pieces),
        net_memory_delta=0,
    )
    table = await ir_context.to_thread(
        unpack_and_concat,
        partitions=pieces,
        stream=stream,
        br=context.br(),
        reservation=reservation,
    )
    return TableChunk.from_pylibcudf_table(
        table,
        stream,
        exclusive_view=True,
        br=context.br(),
    )
