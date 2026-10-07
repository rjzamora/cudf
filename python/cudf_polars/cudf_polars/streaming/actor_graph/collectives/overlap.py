# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Overlap-context exchange helpers for ordered streaming operators."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import TYPE_CHECKING, Any

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

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.utils.cuda_stream import stream_ordered_after

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.packed_data import PackedData
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext


RowRange = tuple[int, int]
ValueRange = tuple[Any, Any]
_INT64_DTYPE = DataType(pl.Int64())


def value_ranges_overlap(
    source: ValueRange | None,
    request: ValueRange | None,
) -> bool:
    """Return whether a source value range may satisfy a request range."""
    if source is None or request is None:
        return False
    source_lower, source_upper = source
    request_lower, request_upper = request
    return source_lower <= request_upper and source_upper >= request_lower


@dataclass(frozen=True)
class RankValueRangeRouter:
    """
    Route logical value-range requests with rank-level endpoint stats.

    ``source_ranges`` describe values each source rank may own. ``request_ranges``
    describe values each destination rank may need. The router is conservative:
    overlapping ranges mean a request may be needed, not that rows must be sent.
    Source ranks still resolve exact payloads locally.

    This is intentionally coarser than an Ordering-boundary router. It only
    knows one value range per rank.
    """

    source_ranges: tuple[ValueRange | None, ...]
    request_ranges: tuple[ValueRange | None, ...]

    def request_sources(self, rank: int) -> tuple[int, ...]:
        """Return remote ranks that may satisfy this rank's request range."""
        request = self.request_ranges[rank]
        return tuple(
            source
            for source, source_range in enumerate(self.source_ranges)
            if source != rank and value_ranges_overlap(source_range, request)
        )

    def request_destinations(self, rank: int) -> tuple[int, ...]:
        """Return remote ranks that may request values from this rank."""
        source = self.source_ranges[rank]
        return tuple(
            destination
            for destination, request_range in enumerate(self.request_ranges)
            if destination != rank and value_ranges_overlap(source, request_range)
        )


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

    owned: list[BufferedChunk]
    ghosts: list[BufferedChunk]


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
    """

    source_spans: tuple[RowRange, ...]
    output_spans: tuple[RowRange, ...]
    ghost_requests: tuple[tuple[RowRange, ...], ...]

    @classmethod
    def from_row_counts(
        cls,
        row_counts: Sequence[int],
        *,
        preceding: int,
        following: int,
    ) -> RowExchangePlan:
        """Build a row-count overlap plan from contiguous rank-owned rows."""
        offsets = _row_starts(row_counts)
        spans = tuple((offsets[i], offsets[i + 1]) for i in range(len(row_counts)))
        total_rows = offsets[-1]
        ghost_requests = tuple(
            _request_spans(
                span,
                preceding=preceding,
                following=following,
                total_rows=total_rows,
            )
            for span in spans
        )
        return cls(spans, spans, ghost_requests)

    @classmethod
    def from_spans(
        cls,
        source_spans: Sequence[RowRange],
        output_spans: Sequence[RowRange],
        ghost_requests: Sequence[Sequence[RowRange]],
    ) -> RowExchangePlan:
        """Build a row exchange plan from explicit row spans."""
        return cls(
            tuple(source_spans),
            tuple(output_spans),
            tuple(tuple(requests) for requests in ghost_requests),
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
        self.collective_id = collective_id

    async def exchange(
        self, local_chunks: Sequence[BufferedChunk]
    ) -> RowExchangeResult:
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
            local_chunks,
            local_owned_intervals=self.plan.owned_intervals_from_source(
                self.comm.rank, self.comm.rank
            ),
            local_empty_span=self.plan.output_span(self.comm.rank),
            sends=sends,
            sources=self.plan.remote_sources(self.comm.rank),
            collective_id=self.collective_id,
        )


async def exchange_resolved_slices(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    local_chunks: Sequence[BufferedChunk],
    *,
    local_owned_intervals: Sequence[RowRange],
    local_empty_span: RowRange | None = None,
    sends: Sequence[ResolvedRowSend],
    sources: Sequence[int],
    collective_id: int,
) -> RowExchangeResult:
    """Exchange resolved row slices with row-span metadata."""
    owned = await _local_owned_chunks(
        context,
        ir_context,
        local_chunks,
        local_owned_intervals,
        local_empty_span=local_empty_span,
    )
    destinations = tuple(sorted({send.destination for send in sends}))
    if not sources and not destinations:
        if not owned:
            owned.extend(chunk for chunk in local_chunks if chunk.num_rows == 0)
        return RowExchangeResult(_sorted_chunks_with_sequence_numbers(owned), [])

    exchange = SparseAlltoall(
        context,
        comm,
        collective_id,
        srcs=tuple(sorted(sources)),
        dsts=destinations,
    )
    for send in sends:
        await _send_resolved_slice(
            context,
            ir_context,
            exchange,
            local_chunks,
            send,
        )

    await exchange.insert_finished(context)

    ghosts: list[BufferedChunk] = []
    for src in sources:
        pieces = exchange.extract(src)
        if len(pieces) % 2 != 0:
            raise RuntimeError(
                "RowExchange received an odd number of metadata/data payloads "
                f"from rank {src}: got {len(pieces)}"
            )
        for metadata, data in zip(pieces[::2], pieces[1::2], strict=True):
            start, stop, kind = await _unpack_slice_metadata(
                context, ir_context, metadata
            )
            chunk = await _unpack_sparse_chunk(context, ir_context, [data])
            buffered = BufferedChunk(-1, chunk, start, stop - start)
            if kind == RowExchangeKind.GHOST:
                ghosts.append(buffered)
            else:
                owned.append(buffered)
    return RowExchangeResult(
        _sorted_chunks_with_sequence_numbers(owned),
        sorted(ghosts, key=lambda chunk: chunk.row_start),
    )


async def exchange_control_chunks(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    *,
    chunks_by_destination: Mapping[int, TableChunk],
    sources: Sequence[int],
    collective_id: int,
) -> dict[int, list[TableChunk]]:
    """Exchange compact overlap request/state chunks by source rank."""
    if not sources and not chunks_by_destination:
        return {}
    exchange = SparseAlltoall(
        context,
        comm,
        collective_id,
        srcs=tuple(sorted(sources)),
        dsts=tuple(sorted(chunks_by_destination)),
    )
    for destination, chunk in chunks_by_destination.items():
        await _insert_sparse_chunk(context, exchange, destination, chunk)
    await exchange.insert_finished(context)
    return {
        source: [
            await _unpack_sparse_chunk(context, ir_context, [piece])
            for piece in exchange.extract(source)
        ]
        for source in sources
    }


async def _local_owned_chunks(
    context: Context,
    ir_context: IRExecutionContext,
    local_chunks: Sequence[BufferedChunk],
    intervals: Sequence[RowRange],
    *,
    local_empty_span: RowRange | None = None,
) -> list[BufferedChunk]:
    """Return rows this rank already owns after boundary routing."""
    owned: list[BufferedChunk] = []
    for start, stop in intervals:
        if start >= stop:
            continue
        owned.extend(
            await extract_region_chunks(
                context,
                local_chunks,
                start,
                stop,
                ir_context=ir_context,
                copy_result=False,
            )
        )
    if local_empty_span is not None:
        start, stop = local_empty_span
        owned.extend(
            replace(chunk, sequence_number=-1)
            for chunk in local_chunks
            if chunk.num_rows == 0
            and _empty_chunk_in_span(chunk.row_start, start, stop)
        )
    return owned


def _empty_chunk_in_span(row_start: int, start: int, stop: int) -> bool:
    """Return whether an empty chunk belongs to a possibly-empty span."""
    if start == stop:
        return row_start == start
    return start <= row_start < stop


async def _send_resolved_slice(
    context: Context,
    ir_context: IRExecutionContext,
    exchange: SparseAlltoall,
    local_chunks: Sequence[BufferedChunk],
    send: ResolvedRowSend,
) -> None:
    """Send one resolved slice and its row-span metadata."""
    if send.start >= send.stop:
        return
    metadata = _slice_metadata_chunk(
        context,
        send.start,
        send.stop,
        kind=send.kind,
    )
    chunk = await extract_region(
        context,
        local_chunks,
        send.start,
        send.stop,
        ir_context=ir_context,
        copy_result=True,
    )
    await _insert_sparse_chunk(context, exchange, send.destination, metadata)
    await _insert_sparse_chunk(context, exchange, send.destination, chunk)


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


def _intersections(span: RowRange, requests: Sequence[RowRange]) -> list[RowRange]:
    """Return sorted intersections between one source span and many requests."""
    intervals = [
        interval
        for request in requests
        if (interval := _range_intersection(span, request)) is not None
    ]
    return sorted(intervals)


def _row_starts(row_counts: Sequence[int]) -> list[int]:
    """Return global row starts for a sequence of per-rank row counts."""
    starts = [0]
    for count in row_counts:
        starts.append(starts[-1] + count)
    return starts


def _request_spans(
    owned: RowRange,
    *,
    preceding: int,
    following: int,
    total_rows: int,
) -> tuple[RowRange, ...]:
    """Return ghost spans needed to evaluate a rank-owned row range."""
    start, stop = owned
    if start == stop:
        return ()
    spans: list[RowRange] = []
    if preceding > 0:
        spans.append((max(0, start - preceding), start))
    if following > 0:
        spans.append((stop, min(total_rows, stop + following)))
    return tuple(span for span in spans if span[0] < span[1])


def _sorted_chunks_with_sequence_numbers(
    chunks: Sequence[BufferedChunk],
) -> list[BufferedChunk]:
    """Return chunks sorted by row position and numbered in output order."""
    result = sorted(chunks, key=lambda chunk: (chunk.row_start, chunk.sequence_number))
    return [
        replace(chunk, sequence_number=sequence_number)
        for sequence_number, chunk in enumerate(result)
    ]


def _validate_chunk_coverage(
    input_chunks: Sequence[BufferedChunk],
    row_start: int,
    row_stop: int,
) -> None:
    """Validate that chunks are sorted and can cover the requested range."""
    previous_stop = None
    for chunk in input_chunks:
        if previous_stop is not None and chunk.row_start < previous_stop:
            raise RuntimeError("Buffered chunks are not sorted by row span")
        previous_stop = chunk.row_stop
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
    _validate_chunk_coverage(input_chunks, row_start, row_stop)
    result: list[BufferedChunk] = []
    expected_start = row_start
    for buf in input_chunks:
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
