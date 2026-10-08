# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Overlap-context helpers for ordered streaming operators."""

from __future__ import annotations

from dataclasses import dataclass, replace
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

    The payload lives in the context's spillable message store. Row-span
    metadata stays in Python so callers can resolve row-slice requests without
    first materializing every chunk.
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
            return replace(buffered, sequence_number=-1)
        return BufferedChunk(-1, chunk, row_start, chunk.shape[0])


@dataclass
class RowExchangeResult:
    """Local rows plus temporary ghost context rows."""

    local_source: BufferedChunkSource
    ghost_source: BufferedChunkSource
    local_owned_intervals: tuple[RowRange, ...]
    local_empty_span: RowRange | None

    def clear_received(self) -> None:
        """Discard rows received from remote ranks."""
        self.ghost_source.clear()

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
            async for chunk in self.local_source.iter_region_chunks(
                *self.local_empty_span,
                ir_context=ir_context,
                include_empty_in_span=self.local_empty_span
                if include_empty_chunks
                else None,
            ):
                yield chunk

    async def iter_ghosts(
        self,
        *,
        ir_context: IRExecutionContext,
    ) -> AsyncIterator[BufferedChunk]:
        """Yield ghost chunks received from remote ranks."""
        async for chunk in self.ghost_source.iter_chunks(ir_context=ir_context):
            yield chunk


@dataclass(frozen=True)
class RowExchangePlan:
    """
    Row-offset routing plan for ghost slice exchange.

    ``source_spans`` describe the rows owned by each source rank.
    ``ghost_requests`` describe extra source rows needed by each rank for local
    evaluation. Output ownership is unchanged by this primitive.
    """

    source_spans: tuple[RowRange, ...]
    ghost_requests: tuple[tuple[RowRange, ...], ...]

    @classmethod
    def from_spans(
        cls,
        source_spans: Sequence[RowRange],
        ghost_requests: Sequence[Sequence[RowRange]],
    ) -> RowExchangePlan:
        """Build a row exchange plan from explicit row spans."""
        plan = cls(
            tuple(source_spans),
            tuple(tuple(_merge_intervals(requests)) for requests in ghost_requests),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        """Validate row-span shape and monotonicity."""
        if len(self.ghost_requests) != len(self.source_spans):
            raise ValueError(
                "RowExchangePlan ghost request count must match source span count: "
                f"{len(self.ghost_requests)} != {len(self.source_spans)}"
            )
        _validate_plan_spans("source", self.source_spans)
        total_rows = self.total_rows
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
            for destination_rank in range(len(self.source_spans))
            if destination_rank != rank
            and self.ghost_intervals_from_source(rank, destination_rank)
        )


@dataclass(frozen=True)
class ResolvedGhostSend:
    """One resolved ghost row slice owed to another rank."""

    destination: int
    start: int
    stop: int


class RowExchange:
    """Sparse exchange of ghost row slices for one local rank."""

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
        """Exchange ghost rows and return local output context."""
        sends: list[ResolvedGhostSend] = []
        for dst in self.plan.remote_ghost_destinations(self.comm.rank):
            sends.extend(
                ResolvedGhostSend(dst, start, stop)
                for start, stop in self.plan.ghost_intervals_from_source(
                    self.comm.rank, dst
                )
            )
        local_span = self.plan.source_span(self.comm.rank)
        local_owned = (local_span,) if local_span[0] < local_span[1] else ()
        return await exchange_resolved_ghost_slices(
            self.context,
            self.comm,
            self.ir_context,
            local_source,
            local_owned_intervals=local_owned,
            local_empty_span=local_span,
            sends=sends,
            expected_sources=self.plan.remote_ghost_sources(self.comm.rank),
            candidate_destinations=self.plan.remote_ghost_destinations(self.comm.rank),
            collective_id=self.collective_id,
        )


def _check_metadata_data_payload_pairs(src: int, pieces: Sequence[PackedData]) -> None:
    """Validate that exchanged ghost payloads arrive as metadata/data pairs."""
    if len(pieces) % 2 != 0:
        raise RuntimeError(
            "RowExchange received an odd number of metadata/data payloads "
            f"from rank {src}: got {len(pieces)}"
        )


async def exchange_resolved_ghost_slices(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    local_source: BufferedChunkSource,
    *,
    local_owned_intervals: Sequence[RowRange],
    local_empty_span: RowRange | None,
    sends: Sequence[ResolvedGhostSend],
    expected_sources: Sequence[int],
    candidate_destinations: Sequence[int],
    collective_id: int,
) -> RowExchangeResult:
    """Exchange resolved ghost row slices with row-span metadata."""
    if not expected_sources and not candidate_destinations:
        return RowExchangeResult(
            local_source,
            BufferedChunkSource(context),
            tuple(local_owned_intervals),
            local_empty_span,
        )

    exchange = SparseAlltoall(
        context,
        comm,
        collective_id,
        srcs=tuple(sorted(set(expected_sources))),
        dsts=tuple(sorted(set(candidate_destinations))),
    )
    insert_finished = False
    try:
        for send in sorted(sends, key=lambda item: (item.destination, item.start)):
            await _send_resolved_slice(
                context, ir_context, exchange, local_source, send
            )
    finally:
        await exchange.insert_finished(context)
        insert_finished = True

    ghost_source = BufferedChunkSource(context)
    sequence_number = 0
    try:
        for src in sorted(set(expected_sources)):
            pieces = exchange.extract(src)
            _check_metadata_data_payload_pairs(src, pieces)
            for metadata, data in zip(pieces[::2], pieces[1::2], strict=True):
                start, stop = await _unpack_slice_metadata(
                    context, ir_context, metadata
                )
                chunk = await _unpack_sparse_chunk(context, ir_context, [data])
                ghost_source.insert(
                    BufferedChunk(sequence_number, chunk, start, stop - start)
                )
                sequence_number += 1
        return RowExchangeResult(
            local_source,
            ghost_source,
            tuple(local_owned_intervals),
            local_empty_span,
        )
    except BaseException:
        if insert_finished:
            ghost_source.clear()
        raise


async def _send_resolved_slice(
    context: Context,
    ir_context: IRExecutionContext,
    exchange: SparseAlltoall,
    local_source: BufferedChunkSource,
    send: ResolvedGhostSend,
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
        metadata = _slice_metadata_chunk(context, chunk.row_start, chunk.row_stop)
        await _insert_sparse_chunk(context, exchange, send.destination, metadata)
        await _insert_sparse_chunk(context, exchange, send.destination, chunk.chunk)


def _slice_metadata_chunk(context: Context, start: int, stop: int) -> TableChunk:
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
            for value in (start, stop)
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
) -> RowRange:
    """Unpack one resolved-slice metadata payload."""
    chunk = await _unpack_sparse_chunk(context, ir_context, [piece])
    start, stop = (
        DataFrame.from_table(
            chunk.table_view(),
            ["start", "stop"],
            [_INT64_DTYPE, _INT64_DTYPE],
            chunk.stream,
        )
        .to_polars()
        .row(0)
    )
    return start, stop


def _empty_chunk_in_span(row_start: int, start: int, stop: int) -> bool:
    """Return whether an empty chunk belongs to a possibly-empty span."""
    if start == stop:
        return row_start == start
    return start <= row_start <= stop


def _range_intersection(left: RowRange, right: RowRange) -> RowRange | None:
    """Return the non-empty intersection of two row ranges."""
    start = max(left[0], right[0])
    stop = min(left[1], right[1])
    if start < stop:
        return start, stop
    return None


def _validate_plan_spans(name: str, spans: Sequence[RowRange]) -> None:
    """Validate monotone contiguous plan spans."""
    previous_stop = 0
    for rank, (start, stop) in enumerate(spans):
        if start < 0 or stop < start:
            raise ValueError(
                f"Invalid RowExchangePlan {name} span for rank {rank}: "
                f"[{start}, {stop})"
            )
        if start != previous_stop:
            raise ValueError(
                f"RowExchangePlan {name} spans must be contiguous: rank {rank} "
                f"starts at {start}, expected {previous_stop}"
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
    del ir_context
    _validate_row_range(row_start, row_stop)
    result: list[BufferedChunk] = []
    expected_start = row_start
    for buf in input_chunks:
        if buf.row_stop <= row_start:
            continue
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
            if full_chunk:
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
            result.append(
                replace(buf, sequence_number=-1)
                if full_chunk and not copy_result
                else BufferedChunk(-1, chunk, start, chunk.shape[0])
            )
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
    buffered = await extract_region_chunks(
        context,
        input_chunks,
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


def _row_count_chunk(context: Context, row_count: int) -> TableChunk:
    """Return a single-row chunk containing one int64 row count."""
    stream = context.br().stream_pool.get_stream()
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
        await inserter.insert(0, _row_count_chunk(context, local_rows))
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
