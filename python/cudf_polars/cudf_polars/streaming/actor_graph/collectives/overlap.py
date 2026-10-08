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
from cudf_polars.streaming.actor_graph.utils import RandomAccessChunkStore
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
        self._store = RandomAccessChunkStore(context)
        self._records: list[StoredBufferedChunk] = []
        self._release_index = 0
        self._row_start_offset = 0

    def insert(self, chunk: BufferedChunk) -> None:
        """Insert a chunk into the spillable source."""
        if self._release_index != 0:
            raise RuntimeError("Cannot insert after releasing buffered chunks")
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
        if self._release_index != 0:
            raise RuntimeError("Cannot shift buffered chunks after releasing them")
        self._row_start_offset = offset

    def clear(self) -> None:
        """Discard all stored and materialized chunks."""
        self._store.clear()
        self._records.clear()
        self._release_index = 0
        self._row_start_offset = 0

    def release_cached_before(self, row_stop: int) -> None:
        """Release stored or materialized chunks ending at or before ``row_stop``."""
        while self._release_index < len(self._records):
            record = self._records[self._release_index]
            if self._row_stop(record) > row_stop:
                break
            self._store.release(record.mid)
            self._release_index += 1

    def _row_start(self, record: StoredBufferedChunk) -> int:
        return record.row_start + self._row_start_offset

    def _row_stop(self, record: StoredBufferedChunk) -> int:
        return record.row_stop + self._row_start_offset

    async def _chunk_for(self, record: StoredBufferedChunk) -> BufferedChunk:
        """Return an available buffered chunk for ``record``."""
        msg = self._store.extract(record.mid)
        if msg.sequence_number != record.sequence_number:
            raise RuntimeError(
                "Buffered chunk metadata/message sequence mismatch: "
                f"{record.sequence_number} != {msg.sequence_number}"
            )
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
            return BufferedChunk(
                record.sequence_number,
                chunk,
                self._row_start(record),
                record.num_rows,
            )

    async def iter_region_chunks(
        self,
        row_start: int,
        row_stop: int,
        *,
        copy_result: bool = False,
        include_empty_in_span: RowRange | None = None,
    ) -> AsyncIterator[BufferedChunk]:
        """Yield chunks intersecting a complete global row range."""
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

    async def iter_chunks(self) -> AsyncIterator[BufferedChunk]:
        """Yield all stored chunks in row-span order."""
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
        record_start = self._row_start(record)
        if copy_result:
            msg = await self._store.copy(
                record.mid,
                start=row_start - record_start,
                stop=row_stop - record_start,
            )
            return BufferedChunk(
                -1,
                TableChunk.from_message(msg, br=self._context.br()),
                row_start,
                row_stop - row_start,
            )
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
        if full_chunk:
            return replace(buffered, sequence_number=-1)
        return BufferedChunk(-1, chunk, row_start, chunk.shape[0])


@dataclass
class RowExchangeResult:
    """Local rows plus temporary ghost context rows."""

    local_source: BufferedChunkSource
    ghost_source: BufferedChunkSource
    local_owned_intervals: tuple[RowRange, ...]
    local_empty_span: RowRange | None

    async def iter_local_owned(
        self,
        *,
        include_empty_chunks: bool = False,
    ) -> AsyncIterator[BufferedChunk]:
        """Yield locally owned chunks directly from the spillable source."""
        for start, stop in self.local_owned_intervals:
            async for chunk in self.local_source.iter_region_chunks(
                start,
                stop,
                include_empty_in_span=(start, stop) if include_empty_chunks else None,
            ):
                yield chunk
        if self.local_empty_span is not None and not self.local_owned_intervals:
            async for chunk in self.local_source.iter_region_chunks(
                *self.local_empty_span,
                include_empty_in_span=self.local_empty_span
                if include_empty_chunks
                else None,
            ):
                yield chunk


@dataclass(frozen=True)
class ResolvedGhostSend:
    """One resolved ghost row slice owed to another rank."""

    destination: int
    start: int
    stop: int


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
            await _send_resolved_slice(context, exchange, local_source, send)
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


def _validate_row_range(row_start: int, row_stop: int) -> None:
    """Validate one half-open row range."""
    if row_start >= row_stop:
        raise ValueError(f"Invalid row range [{row_start}, {row_stop})")


async def extract_region_chunks(
    context: Context,
    input_chunks: Sequence[BufferedChunk],
    row_start: int,
    row_stop: int,
) -> list[BufferedChunk]:
    """Slice buffered chunks intersecting a complete global row range."""
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
            result.append(
                replace(buf, sequence_number=-1)
                if full_chunk
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
) -> TableChunk:
    """Return one chunk containing a complete global row range."""
    buffered = await extract_region_chunks(
        context,
        input_chunks,
        row_start,
        row_stop,
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
