# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-overlap planning and exchange for streaming operators."""

from __future__ import annotations

import operator
import struct
from bisect import bisect_left
from contextlib import aclosing, nullcontext
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING, cast

import pylibcudf as plc
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.allgather import AllGather
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.streaming.core.message import Message

from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.dsl.utils.windows import duration_to_scalar
from cudf_polars.streaming.actor_graph.utils import ChunkStore
from cudf_polars.utils.cuda_stream import join_cuda_streams

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Callable,
        Iterable,
        Sequence,
    )

    from pylibcudf.utils import CudaStreamLike
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.memory_resource import DeviceMemoryResource
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import Rolling


type RowRange = tuple[int, int]


type _BoundSearch = Callable[
    [
        plc.Table,
        plc.Table,
        list[plc.types.Order],
        list[plc.types.NullOrder],
        CudaStreamLike | None,
        DeviceMemoryResource | None,
    ],
    plc.Column,
]


def _chrono_storage_dtype(dtype: plc.DataType) -> plc.DataType:
    """Return the integer storage type for a chrono dtype."""
    if dtype.id() in (plc.TypeId.TIMESTAMP_DAYS, plc.TypeId.DURATION_DAYS):
        return plc.DataType(plc.TypeId.INT32)
    return plc.DataType(plc.TypeId.INT64)


async def _column_insertion_point(
    context: Context,
    index_column: plc.Column,
    needle: plc.Column,
    find: _BoundSearch,
    dtype: plc.DataType,
    stream: Stream,
) -> int:
    """Find a needle's insertion point in one available index column."""
    needs_cast = index_column.type() != dtype
    memory_scope = nullcontext()
    if needs_cast:
        reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
            index_column.size() * 8, net_memory_delta=0
        )
        memory_scope = opaque_memory_usage(reservation)
    with memory_scope:
        if needs_cast:
            index_column = plc.unary.cast(index_column, dtype, stream=stream)
        value = (
            find(
                plc.Table([index_column]),
                plc.Table([needle]),
                [plc.types.Order.ASCENDING],
                [plc.types.NullOrder.AFTER],
                stream,
                context.br().device_mr,
            )
            .to_scalar(stream=stream)
            .to_py(stream=stream)
        )
    assert isinstance(value, int)
    return value


async def _value_insertion_point(
    context: Context,
    index_column: plc.Column,
    value: int,
    find: _BoundSearch,
    dtype: plc.DataType,
    stream: Stream,
) -> int:
    """Find a host value's insertion point in an index column."""
    scalar = plc.Scalar.from_py(
        value,
        _chrono_storage_dtype(dtype) if plc.traits.is_chrono(dtype) else dtype,
        stream=stream,
    )
    needle = plc.Column.from_scalar(scalar, 1, stream=stream)
    if plc.traits.is_chrono(dtype):
        needle = plc.unary.bit_cast(
            needle,
            dtype,
            stream=stream,
            mr=context.br().device_mr,
        )
    return await _column_insertion_point(
        context,
        index_column,
        needle,
        find,
        dtype,
        stream,
    )


def _host_ordering_value(
    value: plc.Column,
    *,
    stream: Stream,
    br: BufferResource,
) -> int:
    """Copy a single-row ordering value to host."""
    if plc.traits.is_chrono(value.type()):
        value = plc.unary.bit_cast(
            value,
            _chrono_storage_dtype(value.type()),
            stream=stream,
            mr=br.device_mr,
        )
    result = value.to_scalar(stream=stream).to_py(stream=stream)
    assert isinstance(result, int)
    return result


def _host_ordering_value_at(
    column: plc.Column,
    row: int,
    *,
    stream: Stream,
    br: BufferResource,
) -> int:
    """Copy one ordering value from a device column to host."""
    (value,) = plc.copying.slice(column, [row, row + 1], stream=stream)
    return _host_ordering_value(value, stream=stream, br=br)


class RangeValidation(IntEnum):
    """Validation status for one rank's range index."""

    VALID = 0
    NULL = 1
    UNSORTED = 2
    OUT_OF_BOUNDS = 3


@dataclass(frozen=True)
class _RangeBounds:
    """First and last values in an ordered range."""

    first: int | None
    last: int | None


@dataclass(frozen=True)
class _RangeSliceSpec:
    """Ordered column and endpoint semantics used for range slicing."""

    index_position: int
    index_dtype: plc.DataType
    start_closed: bool
    end_closed: bool


@dataclass(frozen=True)
class _OverlapPlan[RequestT]:
    """Overlap requests and remote suppliers for one rank."""

    outgoing: tuple[tuple[int, RequestT], ...]
    sources: tuple[int, ...]


@dataclass
class BufferedChunk:
    """An available chunk with its span in the prepared input stream."""

    chunk: TableChunk
    row_start: int
    num_rows: int

    @property
    def row_stop(self) -> int:
        """Return the first row after this chunk."""
        return self.row_start + self.num_rows


@dataclass
class StagedChunk[BufferedChunkT: BufferedChunk]:
    """A prepared chunk marked as owned or ghost."""

    chunk: BufferedChunkT
    """Chunk and policy metadata used by local execution."""
    owned: bool
    """Whether this rank owns the chunk's output rows."""


@dataclass(frozen=True)
class RangeValues:
    """
    Host values describing one nonempty index range.

    The policy determines whether each needed boundary is included.
    """

    first_owned: int
    """First index value in the source-owned range."""
    last_owned: int
    """Last index value in the source-owned range."""
    first_needed: int
    """First boundary required to evaluate the chunk's rows."""
    last_needed: int
    """Last boundary required to evaluate the chunk's rows."""


@dataclass
class RangeOverlap:
    """
    Device and host range metadata for one buffered chunk.

    Device columns preserve the index type and support exact libcudf searches.
    Host values support validation and overlap planning with ordinary Python
    comparisons.
    """

    index_column: plc.Column
    """Index values stored in the buffered chunk."""
    first_needed_boundary: plc.Column
    """Single-row device column containing the first needed boundary."""
    last_needed_boundary: plc.Column
    """Single-row device column containing the last needed boundary."""
    host_values: RangeValues | None
    """Host values for a nonempty chunk, or ``None`` for an empty chunk."""


@dataclass
class RangeBufferedChunk(BufferedChunk):
    """A buffered chunk with range-overlap metadata."""

    overlap: RangeOverlap


class _OverlapPolicy[BufferedChunkT: BufferedChunk]:
    """Base policy for finding the rows needed to evaluate a cursor."""

    def observe(self, chunk: BufferedChunkT) -> None:
        """Record resources that must remain valid while using a chunk."""

    def close(self) -> None:
        """Release resources owned by this policy."""

    def validate_chunk(self, chunk: BufferedChunkT) -> None:
        """Validate a chunk against chunks previously seen by this policy."""

    def evict_history(
        self,
        history: list[BufferedChunkT],
        cursor: BufferedChunkT,
    ) -> list[BufferedChunkT]:
        """Return history chunks that may overlap the cursor or a later chunk."""
        raise NotImplementedError

    def has_complete_future(
        self,
        latest: BufferedChunkT,
        current: BufferedChunkT,
    ) -> bool:
        """Return whether the buffered future covers the current cursor."""
        raise NotImplementedError

    async def history_chunk(
        self,
        context: Context,
        cursor: BufferedChunkT,
    ) -> BufferedChunkT | None:
        """Return the cursor rows that future cursors may need as history."""
        del context
        return cursor

    async def overlap_bounds(
        self,
        context: Context,
        cursor: BufferedChunkT,
        chunks: Sequence[BufferedChunkT],
    ) -> tuple[int, int]:
        """Return the needed span in prepared-stream coordinates."""
        raise NotImplementedError

    async def prepare_chunk(
        self,
        context: Context,
        chunk: TableChunk,
        *,
        row_offset: int,
        host_values: RangeValues | None = None,
    ) -> BufferedChunkT:
        """
        Attach policy metadata to a chunk at a prepared-stream row offset.

        ``row_offset`` is the chunk's first row in the prepared input stream.
        """
        raise NotImplementedError


@dataclass
class RangeOverlapPolicy(_OverlapPolicy[RangeBufferedChunk]):
    """
    Policy and local execution state for range-based overlap.

    The mutable fields track ordering across chunks and the streams using the
    policy's device scalars. Each consumer creates and closes one instance.
    """

    first_needed_offset: plc.Scalar
    """Offset from the first owned value to the first needed boundary."""
    last_needed_offset: plc.Scalar
    """Offset from the last owned value to the last needed boundary."""
    index_position: int
    """Position of the ordered index column in each input chunk."""
    index_dtype: plc.DataType
    find_start: _BoundSearch
    find_end: _BoundSearch
    start_closed: bool
    end_closed: bool
    stream: Stream
    index_name: str
    observed_streams: set[Stream] = field(default_factory=set, init=False, repr=False)
    previous_last_owned: int | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_ir(cls, ir: Rolling, stream: Stream) -> RangeOverlapPolicy:
        """Create actor-local range-bound state for a rolling node."""
        (index_position,) = names_to_indices([ir.index.name], ir.children[0].schema)
        side = ir.closed_window
        start_closed = side in ("both", "left")
        end_closed = side in ("both", "right")
        dtype = ir.index_dtype
        return cls(
            # Note: not using windows_to_offsets because that flips the sign of
            # preceding_ordinal.
            duration_to_scalar(dtype, ir.preceding_ordinal, stream=stream),
            duration_to_scalar(
                dtype,
                ir.preceding_ordinal + ir.following_ordinal,
                stream=stream,
            ),
            index_position,
            dtype,
            plc.search.lower_bound if start_closed else plc.search.upper_bound,
            plc.search.upper_bound if end_closed else plc.search.lower_bound,
            start_closed,
            end_closed,
            stream,
            ir.index.name,
        )

    def observe(self, chunk: RangeBufferedChunk) -> None:
        """Track streams used by range-overlap metadata."""
        self.observed_streams.add(chunk.chunk.stream)

    def close(self) -> None:
        """Keep offset scalars alive until all observed work is ordered."""
        if self.observed_streams:
            join_cuda_streams(
                downstreams=(self.stream,),
                upstreams=tuple(self.observed_streams),
            )

    @staticmethod
    def _index_with_offset(
        index: plc.Column,
        row: int,
        offset: plc.Scalar,
        dtype: plc.DataType,
        stream: Stream,
        br: BufferResource,
    ) -> plc.Column:
        """Return ``index[row] + offset`` as a single-row device column."""
        (endpoint,) = plc.copying.slice(index, [row, row + 1], stream=stream)
        if endpoint.type() != dtype:
            endpoint = plc.unary.cast(endpoint, dtype, stream=stream)
        return plc.binaryop.binary_operation(
            endpoint,
            offset,
            plc.binaryop.BinaryOperator.ADD,
            dtype,
            stream=stream,
            mr=br.device_mr,
        )

    @staticmethod
    async def _insertion_row(
        chunks: Sequence[RangeBufferedChunk],
        needle: plc.Column,
        needle_value: int,
        find: _BoundSearch,
        *,
        upper_bound: bool,
        dtype: plc.DataType,
        context: Context,
        needle_stream: Stream,
    ) -> int:
        """Return a needle's insertion row in prepared-stream coordinates."""
        assert len(chunks) > 0
        for chunk in chunks:
            if chunk.num_rows == 0:
                continue
            assert chunk.overlap.host_values is not None
            if (
                chunk.overlap.host_values.last_owned <= needle_value
                if upper_bound
                else chunk.overlap.host_values.last_owned < needle_value
            ):
                continue
            stream = chunk.chunk.stream
            join_cuda_streams(downstreams=[stream], upstreams=[needle_stream])
            insertion_point = await _column_insertion_point(
                context,
                chunk.overlap.index_column,
                needle,
                find,
                dtype,
                stream,
            )
            if insertion_point < chunk.num_rows:
                return chunk.row_start + insertion_point
        # Needle is later than all the chunks we know about.
        return chunks[-1].row_stop

    @staticmethod
    def index_validation(index_column: plc.Column, stream: Stream) -> RangeValidation:
        """Return the validation status for one index column."""
        if index_column.null_count() != 0:
            return RangeValidation.NULL
        if index_column.size() > 1 and not plc.sorting.is_sorted(
            plc.Table([index_column]),
            [plc.types.Order.ASCENDING],
            [plc.types.NullOrder.BEFORE],
            stream=stream,
        ):
            return RangeValidation.UNSORTED
        return RangeValidation.VALID

    def validate_index_column(self, index_column: plc.Column, stream: Stream) -> None:
        """Validate chunk-local ordering before using search primitives."""
        self.raise_validation(self.index_validation(index_column, stream))

    def raise_validation(self, validation: RangeValidation) -> None:
        """Raise the corresponding range-index validation error."""
        if validation is RangeValidation.NULL:
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling may not contain nulls"
            )
        if validation is RangeValidation.UNSORTED:
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling is not sorted, "
                "please sort first"
            )

    def validate_chunk(self, chunk: RangeBufferedChunk) -> None:
        """Check that the index remains sorted across chunk boundaries."""
        if chunk.num_rows == 0:
            return
        assert chunk.overlap.host_values is not None
        if (
            self.previous_last_owned is not None
            and chunk.overlap.host_values.first_owned < self.previous_last_owned
        ):
            self.raise_validation(RangeValidation.UNSORTED)
        self.previous_last_owned = chunk.overlap.host_values.last_owned

    async def prepare_chunk(
        self,
        context: Context,
        chunk: TableChunk,
        *,
        row_offset: int,
        host_values: RangeValues | None = None,
        validate_index: bool = True,
    ) -> RangeBufferedChunk:
        """Extract range metadata from a table chunk."""
        nrows, _ = chunk.shape
        chunk, extra = await make_table_chunks_available_or_wait(
            context,
            chunk,
            reserve_extra=32,
            net_memory_delta=0,
        )
        with opaque_memory_usage(extra):
            index_column = chunk.table_view().columns()[self.index_position]
            if host_values is None and validate_index:
                self.validate_index_column(index_column, chunk.stream)
            if nrows == 0:
                overlap = RangeOverlap(index_column, index_column, index_column, None)
            else:
                join_cuda_streams(downstreams=(chunk.stream,), upstreams=(self.stream,))
                first_needed_boundary = self._index_with_offset(
                    index_column,
                    0,
                    self.first_needed_offset,
                    self.index_dtype,
                    chunk.stream,
                    context.br(),
                )
                last_needed_boundary = self._index_with_offset(
                    index_column,
                    nrows - 1,
                    self.last_needed_offset,
                    self.index_dtype,
                    chunk.stream,
                    context.br(),
                )
                if host_values is None and (
                    validate_index or index_column.null_count() == 0
                ):
                    host_values = RangeValues(
                        _host_ordering_value_at(
                            index_column, 0, stream=chunk.stream, br=context.br()
                        ),
                        _host_ordering_value_at(
                            index_column,
                            nrows - 1,
                            stream=chunk.stream,
                            br=context.br(),
                        ),
                        _host_ordering_value(
                            first_needed_boundary,
                            stream=chunk.stream,
                            br=context.br(),
                        ),
                        _host_ordering_value(
                            last_needed_boundary,
                            stream=chunk.stream,
                            br=context.br(),
                        ),
                    )
                overlap = RangeOverlap(
                    index_column,
                    first_needed_boundary,
                    last_needed_boundary,
                    host_values,
                )
        return RangeBufferedChunk(chunk, row_offset, nrows, overlap)

    def evict_history(
        self,
        history: list[RangeBufferedChunk],
        cursor: RangeBufferedChunk,
    ) -> list[RangeBufferedChunk]:
        """Drop history chunks that cannot contribute to the cursor chunk."""
        assert cursor.overlap.host_values is not None
        keep = operator.ge if self.start_closed else operator.gt
        return [
            chunk
            for chunk in history
            if chunk.overlap.host_values is not None
            and keep(
                chunk.overlap.host_values.last_owned,
                cursor.overlap.host_values.first_needed,
            )
        ]

    def has_complete_future(
        self,
        latest: RangeBufferedChunk,
        current: RangeBufferedChunk,
    ) -> bool:
        """Return whether buffered input reaches the last needed boundary."""
        if latest.overlap.host_values is None:
            return False
        assert current.overlap.host_values is not None
        complete = operator.gt if self.end_closed else operator.ge
        return complete(
            latest.overlap.host_values.last_owned,
            current.overlap.host_values.last_needed,
        )

    async def overlap_bounds(
        self,
        context: Context,
        cursor: RangeBufferedChunk,
        chunks: Sequence[RangeBufferedChunk],
    ) -> tuple[int, int]:
        """Return the rows needed to evaluate a range cursor."""
        assert cursor.overlap.host_values is not None
        overlap_start = await self._insertion_row(
            chunks,
            cursor.overlap.first_needed_boundary,
            cursor.overlap.host_values.first_needed,
            self.find_start,
            upper_bound=not self.start_closed,
            dtype=self.index_dtype,
            context=context,
            needle_stream=cursor.chunk.stream,
        )
        overlap_stop = await self._insertion_row(
            chunks,
            cursor.overlap.last_needed_boundary,
            cursor.overlap.host_values.last_needed,
            self.find_end,
            upper_bound=self.end_closed,
            dtype=self.index_dtype,
            context=context,
            needle_stream=cursor.chunk.stream,
        )
        # We must extract at least the whole of the current cursor chunk.
        overlap_start = min(cursor.row_start, overlap_start)
        overlap_stop = max(cursor.row_stop, overlap_stop)
        return overlap_start, overlap_stop


@dataclass
class RowCountOverlapPolicy(_OverlapPolicy[BufferedChunk]):
    """Overlap policy for fixed-size row windows."""

    preceding: int
    following: int

    async def prepare_chunk(
        self,
        context: Context,
        chunk: TableChunk,
        *,
        row_offset: int,
        host_values: RangeValues | None = None,
    ) -> BufferedChunk:
        """Attach row-span metadata to an available chunk."""
        del host_values
        nrows, _ = chunk.shape
        chunk, _ = await make_table_chunks_available_or_wait(
            context, chunk, reserve_extra=0, net_memory_delta=0
        )
        return BufferedChunk(chunk, row_offset, nrows)

    def evict_history(
        self,
        history: list[BufferedChunk],
        cursor: BufferedChunk,
    ) -> list[BufferedChunk]:
        """Drop history chunks that cannot contribute to the cursor chunk."""
        overlap_start = max(0, cursor.row_start - self.preceding)
        return [chunk for chunk in history if chunk.row_stop > overlap_start]

    def has_complete_future(
        self,
        latest: BufferedChunk,
        current: BufferedChunk,
    ) -> bool:
        """Return whether buffered input reaches the needed stop row."""
        return latest.row_stop >= current.row_stop + self.following

    async def history_chunk(
        self,
        context: Context,
        cursor: BufferedChunk,
    ) -> BufferedChunk | None:
        """Keep only cursor rows that a future fixed-size window can use."""
        if self.preceding == 0 or cursor.num_rows == 0:
            return None
        start = max(cursor.row_start, cursor.row_stop - self.preceding)
        if start == cursor.row_start:
            return cursor

        stream = cursor.chunk.stream
        (table,) = plc.copying.slice(
            cursor.chunk.table_view(),
            [start - cursor.row_start, cursor.num_rows],
            stream=stream,
        )
        reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
            cursor.chunk.data_alloc_size(), net_memory_delta=0
        )
        with opaque_memory_usage(reservation):
            table = table.copy(stream=stream, mr=context.br().device_mr)
        return BufferedChunk(
            TableChunk.from_pylibcudf_table(
                table,
                stream,
                exclusive_view=True,
                br=context.br(),
            ),
            start,
            cursor.row_stop - start,
        )

    async def overlap_bounds(
        self,
        context: Context,
        cursor: BufferedChunk,
        chunks: Sequence[BufferedChunk],
    ) -> tuple[int, int]:
        """Return the rows needed to evaluate a fixed-size cursor."""
        del context
        overlap_start = max(0, cursor.row_start - self.preceding)
        overlap_stop = min(cursor.row_stop + self.following, chunks[-1].row_stop)
        return overlap_start, overlap_stop


type OverlapPolicy[BufferedChunkT: BufferedChunk] = _OverlapPolicy[BufferedChunkT]


def _intersection(left: RowRange, right: RowRange) -> RowRange | None:
    """Return the intersection of two half-open row spans."""
    start, stop = max(left[0], right[0]), min(left[1], right[1])
    return (start, stop) if start < stop else None


def _count_demands(
    spans: Sequence[RowRange],
    preceding: int,
    following: int,
) -> tuple[tuple[RowRange, ...], ...]:
    """Return the row-count overlap needed by each rank."""
    total = spans[-1][1] if spans else 0
    return tuple(
        ((max(0, start - preceding), min(total, stop + following)),)
        if start != stop
        else ()
        for start, stop in spans
    )


def _plan_count_overlap(
    rank: int,
    supplies: Sequence[RowRange],
    demands: Sequence[Sequence[RowRange]],
) -> _OverlapPlan[RowRange]:
    """Plan row-count overlap from independent supplies and demands."""
    requests = tuple(
        (destination, piece)
        for destination, destination_demands in enumerate(demands)
        for demand in destination_demands
        if (piece := _intersection(supplies[rank], demand)) is not None
    )
    sources = tuple(
        source
        for source, supply in enumerate(supplies)
        if source != rank
        and any(_intersection(supply, demand) for demand in demands[rank])
    )
    return _OverlapPlan(requests, sources)


def _ranges_may_overlap(supply: _RangeBounds, demand: _RangeBounds) -> bool:
    """Return whether two ranges may overlap."""
    return not (
        (
            supply.last is not None
            and demand.first is not None
            and supply.last < demand.first
        )
        or (
            supply.first is not None
            and demand.last is not None
            and supply.first > demand.last
        )
    )


def _plan_range_overlap(
    rank: int,
    supplies: Sequence[_RangeBounds | None],
    demands: Sequence[_RangeBounds | None],
) -> _OverlapPlan[_RangeBounds]:
    """Plan key-range overlap from independent supplies and demands."""
    local_supply = supplies[rank]
    local_demand = demands[rank]
    requests = (
        tuple(
            (destination, demand)
            for destination, demand in enumerate(demands)
            if demand is not None and _ranges_may_overlap(local_supply, demand)
        )
        if local_supply is not None
        else ()
    )
    sources = (
        tuple(
            source
            for source, supply in enumerate(supplies)
            if source != rank
            and supply is not None
            and _ranges_may_overlap(supply, local_demand)
        )
        if local_demand is not None
        else ()
    )
    return _OverlapPlan(requests, sources)


@dataclass
class _PreparedInput:
    """Retained local input and completed context exchange."""

    local_chunks: ChunkStore
    """Spillable messages retained from the local source."""
    exchange: SparseAlltoall | None
    """Exchange used to send or receive remote context."""
    remote_sources: tuple[int, ...]
    """Candidate context sources in input order."""

    def clear(self) -> None:
        """Release retained local and remote messages."""
        self.local_chunks.clear()
        if self.exchange is not None:
            for source in self.remote_sources:
                self.exchange.extract(source)


async def _replay_buffer(
    buffer: ChunkStore,
    bounds: Sequence[_RangeBounds | None] = (),
) -> AsyncGenerator[tuple[Message[TableChunk], _RangeBounds | None]]:
    """Yield buffered messages and optional range bounds, then clear the store."""
    try:
        if bounds and len(bounds) != len(buffer):
            raise ValueError("Range bounds must match the buffered chunks")
        for position, msg in enumerate(buffer):
            yield msg, bounds[position] if bounds else None
    finally:
        buffer.clear()


class _OverlapCollective:
    """Gather overlap metadata and route supplied rows to demand ranges."""

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        collective_id: int,
    ) -> None:
        self.context = context
        self.comm = comm
        self.collective_id = collective_id

    async def _allgather_ints(
        self, payload: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...]:
        """All-gather fixed-width signed integer metadata in rank order."""
        fmt = f"<{len(payload)}q"
        gather = AllGather(self.context, self.comm, self.collective_id)
        try:
            gather.insert(
                0,
                PackedData.from_host_bytes(
                    struct.pack(fmt, *payload), self.context.br()
                ),
            )
        finally:
            gather.insert_finished()
        return tuple(
            struct.unpack(fmt, item.to_host_bytes())
            for item in await gather.extract_all(self.context, ordered=True)
        )

    async def gather_counts(self, rows: int) -> tuple[int, ...]:
        """Gather the row count supplied by every rank."""
        return tuple(item[0] for item in await self._allgather_ints((rows,)))

    async def gather_ranges(
        self,
        supply: _RangeBounds | None,
        demand: _RangeBounds | None,
        validation: RangeValidation,
    ) -> tuple[
        tuple[_RangeBounds | None, ...],
        tuple[_RangeBounds | None, ...],
        tuple[RangeValidation, ...],
    ]:
        """Gather independent range supplies and demands."""
        bounds = (supply, demand)
        if any(
            value is not None and not -(2**63) <= value < 2**63
            for item in bounds
            if item is not None
            for value in (item.first, item.last)
        ):
            validation = RangeValidation.OUT_OF_BOUNDS

        def pack(item: _RangeBounds | None) -> tuple[int, ...]:
            return (
                int(item is not None),
                int(item is not None and item.first is not None),
                item.first
                if item is not None
                and item.first is not None
                and -(2**63) <= item.first < 2**63
                else 0,
                int(item is not None and item.last is not None),
                item.last
                if item is not None
                and item.last is not None
                and -(2**63) <= item.last < 2**63
                else 0,
            )

        gathered = await self._allgather_ints(
            (int(validation), *pack(supply), *pack(demand))
        )

        def unpack(item: tuple[int, ...]) -> _RangeBounds | None:
            if not item[0]:
                return None
            return _RangeBounds(
                item[2] if item[1] else None,
                item[4] if item[3] else None,
            )

        return (
            tuple(
                unpack(item[1:6])
                if RangeValidation(item[0]) is RangeValidation.VALID
                else None
                for item in gathered
            ),
            tuple(
                unpack(item[6:])
                if RangeValidation(item[0]) is RangeValidation.VALID
                else None
                for item in gathered
            ),
            tuple(RangeValidation(item[0]) for item in gathered),
        )

    async def exchange[RequestT: (RowRange, _RangeBounds)](
        self,
        source: AsyncGenerator[tuple[Message[TableChunk], _RangeBounds | None]],
        plan: _OverlapPlan[RequestT],
        *,
        retain_source: bool,
        global_row_offset: int = 0,
        slice_spec: _RangeSliceSpec | None = None,
    ) -> _PreparedInput:
        """
        Exchange requested context from a live or replayed source.

        Range bounds may be omitted or conservatively enclose the chunk.
        Omitted bounds require searches for every candidate range demand.
        """
        count_requests: tuple[tuple[int, int, int], ...]
        range_requests: tuple[tuple[int, _RangeBounds], ...]
        if slice_spec is None:
            count_requests = tuple(
                (destination, piece[0], piece[1])
                for destination, piece in cast(
                    "Sequence[tuple[int, RowRange]]", plan.outgoing
                )
                if not retain_source or destination != self.comm.rank
            )
            range_requests = ()
        else:
            count_requests = ()
            range_requests = tuple(
                (destination, demand)
                for destination, demand in cast(
                    "Sequence[tuple[int, _RangeBounds]]", plan.outgoing
                )
                if not retain_source or destination != self.comm.rank
            )
        sources = plan.sources
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
                srcs=sources,
                dsts=destinations,
            )
            if sources or destinations
            else None
        )
        output = ChunkStore(self.context)
        start = 0
        try:
            async with aclosing(source):
                async for msg, input_bounds in source:
                    chunk = TableChunk.from_message(msg, br=self.context.br())
                    size = chunk.shape[0]
                    chunk_bounds = input_bounds
                    pieces: list[tuple[int, int, int]] = []
                    chunk_requests = (
                        (
                            tuple(
                                (destination, demand)
                                for destination, demand in range_requests
                                if _ranges_may_overlap(chunk_bounds, demand)
                            )
                            if chunk_bounds is not None
                            else range_requests
                        )
                        if size
                        else ()
                    )
                    if slice_spec is None:
                        span = (
                            start + global_row_offset,
                            start + global_row_offset + size,
                        )
                        pieces = [
                            (destination, piece[0] - span[0], piece[1] - span[0])
                            for destination, request_start, request_stop in count_requests
                            if (
                                piece := _intersection(
                                    span, (request_start, request_stop)
                                )
                            )
                            is not None
                        ]
                    if not chunk_requests and not pieces:
                        if retain_source:
                            output.insert(Message(0, chunk))
                        start += size
                        continue
                    available, extra = await make_table_chunks_available_or_wait(
                        self.context,
                        chunk,
                        reserve_extra=0,
                        net_memory_delta=0,
                    )
                    with opaque_memory_usage(extra):
                        if slice_spec is not None:
                            pieces = await self._range_pieces(
                                slice_spec,
                                available,
                                size,
                                chunk_bounds,
                                chunk_requests,
                            )
                        local_pieces = []
                        for destination, begin, end in pieces:
                            if destination == self.comm.rank:
                                local_pieces.append((begin, end))
                                continue
                            assert exchange is not None
                            await self._send(
                                exchange, destination, available, begin, end
                            )
                        if local_pieces == [(0, size)]:
                            output.insert(Message(0, available))
                        else:
                            for begin, end in local_pieces:
                                await self._store_local(output, available, begin, end)
                        if retain_source:
                            output.insert(Message(0, available))
                    start += size
        except BaseException:
            output.clear()
            raise
        finally:
            if exchange is not None:
                try:
                    await exchange.insert_finished(self.context)
                except BaseException:
                    output.clear()
                    raise
        return _PreparedInput(output, exchange, sources)

    async def _range_pieces(
        self,
        spec: _RangeSliceSpec,
        chunk: TableChunk,
        size: int,
        chunk_bounds: _RangeBounds | None,
        requests: Sequence[tuple[int, _RangeBounds]],
    ) -> list[tuple[int, int, int]]:
        """Resolve range requests to half-open slices within one chunk."""
        pieces = []
        index = chunk.table_view().columns()[spec.index_position]
        first_supplied = chunk_bounds.first if chunk_bounds is not None else None
        last_supplied = chunk_bounds.last if chunk_bounds is not None else None
        for destination, demand in requests:
            first_needed = demand.first
            last_needed = demand.last
            begin = (
                0
                if first_needed is None
                or (
                    first_supplied is not None
                    and (
                        first_needed < first_supplied
                        or (spec.start_closed and first_needed == first_supplied)
                    )
                )
                else await _value_insertion_point(
                    self.context,
                    index,
                    first_needed,
                    plc.search.lower_bound
                    if spec.start_closed
                    else plc.search.upper_bound,
                    spec.index_dtype,
                    chunk.stream,
                )
            )
            end = (
                size
                if last_needed is None
                or (
                    last_supplied is not None
                    and (
                        last_needed > last_supplied
                        or (spec.end_closed and last_needed == last_supplied)
                    )
                )
                else await _value_insertion_point(
                    self.context,
                    index,
                    last_needed,
                    plc.search.upper_bound
                    if spec.end_closed
                    else plc.search.lower_bound,
                    spec.index_dtype,
                    chunk.stream,
                )
            )
            if begin < end:
                pieces.append((destination, begin, end))
        return pieces

    async def _store_local(
        self,
        output: ChunkStore,
        chunk: TableChunk,
        start: int,
        stop: int,
    ) -> None:
        """Copy one local context slice into spillable storage."""
        msg = Message(0, self._slice_chunk(chunk, start, stop))
        reservation = await self.context.memory(MemoryType.DEVICE).reserve_or_wait(
            msg.copy_cost(),
            net_memory_delta=0,
        )
        output.insert(msg.copy(reservation))

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


class OverlapInput[BufferedChunkT: BufferedChunk]:
    """
    Yield owned input with the ghost chunks required by the overlap policy.

    A window near a rank boundary may need rows owned by another rank.
    For non-duplicated multi-rank input, entering this context buffers the full
    local input in spillable messages, all-gathers a compact count or range
    summary, and exchanges only the required slices. This preparation is a
    barrier, so no chunks are yielded until every rank has summarized its
    local input.

    Iteration yields left ghost chunks, owned chunks, and right ghost chunks in
    row order. Ghost chunks provide context but do not produce output rows.
    Single-rank and duplicated input streams directly from the channel.
    """

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        ch_in: Channel[TableChunk],
        policy: OverlapPolicy[BufferedChunkT],
        *,
        duplicated: bool,
        collective_id: int,
    ) -> None:
        """Create an input stream for one overlap policy and input channel."""
        self.context = context
        self.comm = comm
        self.ch_in = ch_in
        self.policy = policy
        self.multirank = comm.nranks != 1 and not duplicated
        self.collective_id = collective_id
        self._prepared: _PreparedInput | None = None
        self._host_values: tuple[RangeValues | None, ...] = ()

    async def __aenter__(self) -> OverlapInput[BufferedChunkT]:
        """Prepare non-duplicated multi-rank input and wait for all ranks."""
        if self.multirank:
            self._prepared = await self._prepare()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Release messages still owned by this input."""
        prepared, self._prepared = self._prepared, None
        self._host_values = ()
        if prepared is not None:
            prepared.clear()

    def __aiter__(self) -> AsyncIterator[StagedChunk[BufferedChunkT]]:
        """Yield prepared chunks in row order."""
        return self._chunks()

    async def _raw_owned_chunks(
        self,
    ) -> AsyncIterator[tuple[TableChunk, RangeValues | None]]:
        """Yield locally owned chunks and any cached host range values."""
        if not self.multirank:
            while (msg := await self.ch_in.recv(self.context)) is not None:
                yield TableChunk.from_message(msg, br=self.context.br()), None
        else:
            assert self._prepared is not None
            for position, msg in enumerate(self._prepared.local_chunks):
                yield (
                    TableChunk.from_message(msg, br=self.context.br()),
                    self._host_values[position] if self._host_values else None,
                )

    async def _raw_ghost_chunks(
        self, sources: Iterable[int]
    ) -> AsyncIterator[tuple[TableChunk, None]]:
        """Yield exchanged chunks from each source rank in row order."""
        prepared = self._prepared
        for source in sources:
            assert prepared is not None
            assert prepared.exchange is not None
            for packed in prepared.exchange.extract(source):
                yield TableChunk.from_packed_data(packed, self.context.br()), None

    async def _chunks(self) -> AsyncIterator[StagedChunk[BufferedChunkT]]:
        """Prepare chunks in one row coordinate system for this rank."""
        if self.multirank and self._prepared is None:
            raise RuntimeError(
                "Multi-rank overlap input must be used as an async context manager"
            )
        row_offset = 0
        sources = self._prepared.remote_sources if self._prepared else ()
        split = bisect_left(sources, self.comm.rank)
        inputs = (
            (False, self._raw_ghost_chunks(sources[:split])),
            (True, self._raw_owned_chunks()),
            (False, self._raw_ghost_chunks(sources[split:])),
        )
        for owned, values in inputs:
            async for raw_chunk, host_values in values:
                chunk = await self.policy.prepare_chunk(
                    self.context,
                    raw_chunk,
                    row_offset=row_offset,
                    host_values=host_values,
                )
                self.policy.observe(chunk)
                row_offset = chunk.row_stop
                yield StagedChunk(chunk, owned=owned)

    async def _prepare(self) -> _PreparedInput:
        """Buffer local input, agree on a plan, and exchange ghost chunks."""
        policy = self.policy
        if not isinstance(policy, (RangeOverlapPolicy, RowCountOverlapPolicy)):
            raise TypeError(f"Unsupported overlap policy: {type(policy).__name__}")
        buffer = ChunkStore(self.context)
        collective = _OverlapCollective(
            self.context,
            self.comm,
            self.collective_id,
        )
        try:
            if isinstance(policy, RangeOverlapPolicy):
                (
                    supply,
                    demand,
                    validation,
                    host_values,
                ) = await self._buffer_range_input(buffer, policy)
                self._host_values = host_values
                supplies, demands, validations = await collective.gather_ranges(
                    supply,
                    demand,
                    validation,
                )
                self._validate_ranges(policy, supplies, validations)
                range_plan = _plan_range_overlap(
                    self.comm.rank,
                    supplies,
                    demands,
                )
                return await collective.exchange(
                    _replay_buffer(
                        buffer,
                        tuple(
                            _RangeBounds(value.first_owned, value.last_owned)
                            if value is not None
                            else None
                            for value in host_values
                        ),
                    ),
                    range_plan,
                    slice_spec=_RangeSliceSpec(
                        policy.index_position,
                        policy.index_dtype,
                        policy.start_closed,
                        policy.end_closed,
                    ),
                    retain_source=True,
                )

            local_rows = await self._buffer_count_input(buffer)
            counts = await collective.gather_counts(local_rows)
            starts = tuple(accumulate(counts, initial=0))
            spans = tuple(pairwise(starts))
            count_plan = _plan_count_overlap(
                self.comm.rank,
                spans,
                _count_demands(spans, policy.preceding, policy.following),
            )
            return await collective.exchange(
                _replay_buffer(buffer),
                count_plan,
                global_row_offset=starts[self.comm.rank],
                retain_source=True,
            )
        except BaseException:
            buffer.clear()
            raise

    async def _buffer_count_input(self, buffer: ChunkStore) -> int:
        """Buffer local chunks and return their total row count."""
        rows = 0
        while (msg := await self.ch_in.recv(self.context)) is not None:
            chunk = TableChunk.from_message(msg, br=self.context.br())
            rows += chunk.shape[0]
            buffer.insert(Message(0, chunk))
        return rows

    async def _buffer_range_input(
        self,
        buffer: ChunkStore,
        policy: RangeOverlapPolicy,
    ) -> tuple[
        _RangeBounds | None,
        _RangeBounds | None,
        RangeValidation,
        tuple[RangeValues | None, ...],
    ]:
        """Buffer local chunks and collect host range values."""
        row_offset = 0
        validation = RangeValidation.VALID
        summary_values = None
        host_values: list[RangeValues | None] = []
        while (msg := await self.ch_in.recv(self.context)) is not None:
            table_chunk = TableChunk.from_message(msg, br=self.context.br())
            nrows, _ = table_chunk.shape
            chunk = await policy.prepare_chunk(
                self.context,
                table_chunk,
                row_offset=row_offset,
                validate_index=False,
            )
            policy.observe(chunk)
            table_chunk = chunk.chunk
            values = chunk.overlap.host_values
            if nrows:
                status = policy.index_validation(
                    chunk.overlap.index_column, table_chunk.stream
                )
                if status is RangeValidation.VALID:
                    assert values is not None
                    if (
                        summary_values is not None
                        and values.first_owned < summary_values.last_owned
                    ):
                        status = RangeValidation.UNSORTED
                if validation is RangeValidation.VALID:
                    validation = status
                if status is RangeValidation.VALID:
                    assert values is not None
                    summary_values = RangeValues(
                        summary_values.first_owned
                        if summary_values is not None
                        else values.first_owned,
                        values.last_owned,
                        summary_values.first_needed
                        if summary_values is not None
                        else values.first_needed,
                        values.last_needed,
                    )
            host_values.append(values)
            buffer.insert(Message(0, table_chunk))
            row_offset += nrows
        return (
            _RangeBounds(summary_values.first_owned, summary_values.last_owned)
            if summary_values is not None
            else None,
            _RangeBounds(summary_values.first_needed, summary_values.last_needed)
            if summary_values is not None
            else None,
            validation,
            tuple(host_values),
        )

    @staticmethod
    def _validate_ranges(
        policy: RangeOverlapPolicy,
        supplies: Sequence[_RangeBounds | None],
        validations: Sequence[RangeValidation],
    ) -> None:
        """Raise the same range-validation error on every rank."""
        if RangeValidation.OUT_OF_BOUNDS in validations:
            raise NotImplementedError(
                "Multi-rank rolling does not support index values outside the "
                "signed 64-bit range"
            )
        if RangeValidation.NULL in validations:
            policy.raise_validation(RangeValidation.NULL)
        previous = None
        for supply, validation in zip(supplies, validations, strict=True):
            if validation is RangeValidation.UNSORTED:
                break
            if supply is None:
                continue
            values = supply
            if (
                values.first is None
                or values.last is None
                or values.first > values.last
                or (previous is not None and values.first < previous)
            ):
                break
            previous = values.last
        else:
            return
        policy.raise_validation(RangeValidation.UNSORTED)
