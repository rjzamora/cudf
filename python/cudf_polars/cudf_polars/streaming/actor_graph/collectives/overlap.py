# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-overlap planning and exchange for streaming operators."""

from __future__ import annotations

import operator
import struct
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntEnum
from functools import partial
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING

import pylibcudf as plc
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.allgather import AllGather
from rapidsmpf.streaming.core.message import Message

from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.dsl.utils.windows import duration_to_scalar
from cudf_polars.streaming.actor_graph.collectives.slice_exchange import (
    ExchangePlan,
    InputDemand,
    SliceExchange,
    SliceResolver,
    intersection as _intersection,
    merge_row_ranges as _merge_row_ranges,
    replay_buffer as _replay_buffer,
    row_range_slices as _row_range_pieces,
    row_range_summary_slices as _row_range_summary_pieces,
    tag_row_slice as _tag_row_piece,
)
from cudf_polars.streaming.actor_graph.utils import ChunkStore
from cudf_polars.utils.cuda_stream import join_cuda_streams

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterator,
        Callable,
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
    from cudf_polars.streaming.actor_graph.collectives.slice_exchange import (
        PreparedExchange,
        RoutedSlice,
        RowRange,
    )


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
class _RangeSupply:
    """Global row span and key bounds supplied by one chunk."""

    rows: RowRange
    keys: _RangeBounds


@dataclass(frozen=True)
class _RangeSliceSpec:
    """Ordered column and endpoint semantics used for range slicing."""

    index_position: int
    index_dtype: plc.DataType
    start_closed: bool
    end_closed: bool


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


def _count_demands(
    spans: Sequence[RowRange],
    preceding: int,
    following: int,
) -> tuple[InputDemand[RowRange, RowRange] | None, ...]:
    """Return the row-count overlap needed by each rank."""
    total = spans[-1][1] if spans else 0
    return tuple(
        InputDemand(
            owned=(start, stop),
            needed=(max(0, start - preceding), min(total, stop + following)),
        )
        if start != stop
        else None
        for start, stop in spans
    )


def _plan_count_overlap(
    rank: int,
    supplies: Sequence[RowRange],
    demands: Sequence[InputDemand[RowRange, RowRange] | None],
) -> ExchangePlan[RowRange, RowRange]:
    """Plan row-count overlap from independent supplies and demands."""
    routes = tuple(
        tuple(
            destination
            for destination, demand in enumerate(demands)
            if demand is not None
            and (
                _intersection(supply, demand.needed) is not None
                or (
                    demand.owned is not None
                    and _intersection(supply, demand.owned) is not None
                )
            )
        )
        for supply in supplies
    )
    requests = tuple(
        (destination, demand)
        for destination in routes[rank]
        if (demand := demands[destination]) is not None
    )
    return ExchangePlan.from_routes(rank, routes, requests)


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
    supplies: Sequence[_RangeSupply | None],
    demands: Sequence[InputDemand[RowRange, _RangeBounds] | None],
) -> ExchangePlan[RowRange, _RangeBounds]:
    """Plan key-range overlap from independent supplies and demands."""
    routes = tuple(
        tuple(
            (destination, demand)
            for destination, demand in enumerate(demands)
            if demand is not None
            and supply is not None
            and (
                _ranges_may_overlap(supply.keys, demand.needed)
                or (
                    demand.owned is not None
                    and _intersection(supply.rows, demand.owned) is not None
                )
            )
        )
        for supply in supplies
    )
    requests = tuple((destination, demand) for destination, demand in routes[rank])
    destinations = tuple(
        tuple(destination for destination, _ in source_routes)
        for source_routes in routes
    )
    return ExchangePlan.from_routes(rank, destinations, requests)


async def _allgather_ints(
    context: Context,
    comm: Communicator,
    collective_id: int,
    payload: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """All-gather fixed-width signed integer metadata in rank order."""
    fmt = f"<{len(payload)}q"
    gather = AllGather(context, comm, collective_id)
    try:
        gather.insert(
            0,
            PackedData.from_host_bytes(struct.pack(fmt, *payload), context.br()),
        )
    finally:
        gather.insert_finished()
    return tuple(
        struct.unpack(fmt, item.to_host_bytes())
        for item in await gather.extract_all(context, ordered=True)
    )


async def _gather_counts(
    context: Context,
    comm: Communicator,
    collective_id: int,
    rows: int,
) -> tuple[int, ...]:
    """Gather the row count owned by every rank."""
    return tuple(
        item[0]
        for item in await _allgather_ints(
            context,
            comm,
            collective_id,
            (rows,),
        )
    )


async def _gather_ranges(
    context: Context,
    comm: Communicator,
    collective_id: int,
    rows: int,
    supply: _RangeBounds | None,
    demand: _RangeBounds | None,
    validation: RangeValidation,
) -> tuple[
    tuple[int, ...],
    tuple[_RangeBounds | None, ...],
    tuple[_RangeBounds | None, ...],
    tuple[RangeValidation, ...],
]:
    """Gather row counts, range bounds, and validation from every rank."""
    supply_values = (supply.first, supply.last) if supply is not None else (0, 0)
    demand_values = (demand.first, demand.last) if demand is not None else (0, 0)
    values = (*supply_values, *demand_values)
    if any(value is None or not -(2**63) <= value < 2**63 for value in values):
        validation = RangeValidation.OUT_OF_BOUNDS
    packed_values = tuple(
        value if isinstance(value, int) and -(2**63) <= value < 2**63 else 0
        for value in values
    )
    packed = await _allgather_ints(
        context,
        comm,
        collective_id,
        (
            rows,
            int(validation),
            int(supply is not None),
            *packed_values[:2],
            int(demand is not None),
            *packed_values[2:],
        ),
    )
    return (
        tuple(item[0] for item in packed),
        tuple(_RangeBounds(item[3], item[4]) if item[2] else None for item in packed),
        tuple(_RangeBounds(item[6], item[7]) if item[5] else None for item in packed),
        tuple(RangeValidation(item[1]) for item in packed),
    )


def _range_summary_pieces(
    spec: _RangeSliceSpec,
    size: int,
    supply: _RangeSupply | None,
    requests: Sequence[tuple[int, InputDemand[RowRange, _RangeBounds]]],
) -> list[RoutedSlice] | None:
    """Resolve requests that cover or exclude a complete ordered chunk."""
    if size == 0:
        return []
    if supply is None:
        return None
    supply_first, supply_last = supply.keys.first, supply.keys.last
    if supply_first is None or supply_last is None:
        return None
    result: list[RoutedSlice] = []
    for destination, demand in requests:
        first, last = demand.needed.first, demand.needed.last
        lower_full = (
            first is None
            or first < supply_first
            or (first == supply_first and spec.start_closed)
        )
        upper_full = (
            last is None
            or last > supply_last
            or (last == supply_last and spec.end_closed)
        )
        below = last is not None and (
            last < supply_first or (last == supply_first and not spec.end_closed)
        )
        above = first is not None and (
            first > supply_last or (first == supply_last and not spec.start_closed)
        )
        if lower_full and upper_full:
            needed: RowRange | None = supply.rows
        elif below or above:
            needed = None
        else:
            return None
        owned = (
            _intersection(supply.rows, demand.owned)
            if demand.owned is not None
            else None
        )
        for piece in _merge_row_ranges(needed, owned):
            result.extend(_tag_row_piece(destination, supply.rows, piece, demand.owned))
    return result


async def _range_pieces(
    context: Context,
    spec: _RangeSliceSpec,
    chunk: TableChunk,
    supply: _RangeSupply | None,
    requests: Sequence[tuple[int, InputDemand[RowRange, _RangeBounds]]],
) -> list[RoutedSlice]:
    """Resolve range requests to exact half-open slices within one chunk."""
    size = chunk.shape[0]
    if size == 0:
        return []
    if supply is None:
        raise ValueError("Nonempty range input requires a supply summary")
    supply_first, supply_last = supply.keys.first, supply.keys.last
    if supply_first is None or supply_last is None:
        raise ValueError("Nonempty range input requires finite key bounds")
    index = chunk.table_view().columns()[spec.index_position]
    result: list[RoutedSlice] = []
    for destination, demand in requests:
        first, last = demand.needed.first, demand.needed.last
        begin = (
            0
            if first is None
            or first < supply_first
            or (first == supply_first and spec.start_closed)
            else await _value_insertion_point(
                context,
                index,
                first,
                plc.search.lower_bound if spec.start_closed else plc.search.upper_bound,
                spec.index_dtype,
                chunk.stream,
            )
        )
        stop = (
            size
            if last is None
            or last > supply_last
            or (last == supply_last and spec.end_closed)
            else await _value_insertion_point(
                context,
                index,
                last,
                plc.search.upper_bound if spec.end_closed else plc.search.lower_bound,
                spec.index_dtype,
                chunk.stream,
            )
        )
        needed = (
            (supply.rows[0] + begin, supply.rows[0] + stop) if begin < stop else None
        )
        owned = (
            _intersection(supply.rows, demand.owned)
            if demand.owned is not None
            else None
        )
        for piece in _merge_row_ranges(needed, owned):
            result.extend(_tag_row_piece(destination, supply.rows, piece, demand.owned))
    return result


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
        self._prepared: PreparedExchange | None = None

    async def __aenter__(self) -> OverlapInput[BufferedChunkT]:
        """Prepare non-duplicated multi-rank input and wait for all ranks."""
        if self.multirank:
            self._prepared = await self._prepare()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Release messages still owned by this input."""
        prepared, self._prepared = self._prepared, None
        if prepared is not None:
            prepared.clear()

    def __aiter__(self) -> AsyncIterator[StagedChunk[BufferedChunkT]]:
        """Yield prepared chunks in row order."""
        return self._chunks()

    async def _chunks(self) -> AsyncIterator[StagedChunk[BufferedChunkT]]:
        """Prepare chunks in one row coordinate system for this rank."""
        if self.multirank and self._prepared is None:
            raise RuntimeError(
                "Multi-rank overlap input must be used as an async context manager"
            )
        if self._prepared is None:

            async def direct_input() -> AsyncIterator[tuple[Message[TableChunk], bool]]:
                while (msg := await self.ch_in.recv(self.context)) is not None:
                    yield msg, True

            messages = direct_input()
        else:
            messages = self._prepared.iter_tagged_source_order(
                self.context,
                self.comm.rank,
            )

        row_offset = 0
        async for message, owned in messages:
            raw_chunk = TableChunk.from_message(message, br=self.context.br())
            chunk = await self.policy.prepare_chunk(
                self.context,
                raw_chunk,
                row_offset=row_offset,
            )
            self.policy.observe(chunk)
            row_offset = chunk.row_stop
            yield StagedChunk(chunk, owned=owned)

    async def _prepare(self) -> PreparedExchange:
        """Buffer local input, agree on a plan, and exchange ghost chunks."""
        policy = self.policy
        if not isinstance(policy, (RangeOverlapPolicy, RowCountOverlapPolicy)):
            raise TypeError(f"Unsupported overlap policy: {type(policy).__name__}")
        buffer = ChunkStore(self.context)
        collective = SliceExchange(
            self.context,
            self.comm,
            self.collective_id,
        )
        try:
            if isinstance(policy, RangeOverlapPolicy):
                (
                    local_rows,
                    local_spans,
                    supply,
                    demand,
                    validation,
                    host_values,
                ) = await self._buffer_range_input(buffer, policy)
                counts, supplies, needed, validations = await _gather_ranges(
                    self.context,
                    self.comm,
                    self.collective_id,
                    local_rows,
                    supply,
                    demand,
                    validation,
                )
                self._validate_ranges(policy, supplies, validations)
                starts = tuple(accumulate(counts, initial=0))
                spans = tuple(pairwise(starts))
                demands = tuple(
                    InputDemand(
                        owned=span if span[0] != span[1] else None,
                        needed=rank_needed,
                    )
                    if rank_needed is not None
                    else None
                    for span, rank_needed in zip(spans, needed, strict=True)
                )
                rank_supplies = tuple(
                    _RangeSupply(span, keys) if keys is not None else None
                    for span, keys in zip(spans, supplies, strict=True)
                )
                range_plan = _plan_range_overlap(
                    self.comm.rank,
                    rank_supplies,
                    demands,
                )
                slice_spec = _RangeSliceSpec(
                    policy.index_position,
                    policy.index_dtype,
                    policy.start_closed,
                    policy.end_closed,
                )

                rank_start = starts[self.comm.rank]
                chunk_supplies = tuple(
                    _RangeSupply(
                        (rank_start + span[0], rank_start + span[1]),
                        _RangeBounds(value.first_owned, value.last_owned),
                    )
                    if value is not None
                    else None
                    for span, value in zip(local_spans, host_values, strict=True)
                )
                return await collective.exchange(
                    _replay_buffer(buffer, chunk_supplies),
                    range_plan,
                    resolver=SliceResolver(
                        resolve_chunk=partial(
                            _range_pieces,
                            self.context,
                            slice_spec,
                        ),
                        resolve_summary=partial(_range_summary_pieces, slice_spec),
                    ),
                )

            local_rows, local_spans = await self._buffer_count_input(buffer)
            counts = await _gather_counts(
                self.context,
                self.comm,
                self.collective_id,
                local_rows,
            )
            starts = tuple(accumulate(counts, initial=0))
            spans = tuple(pairwise(starts))
            rank_start = starts[self.comm.rank]
            chunk_spans = tuple(
                (rank_start + start, rank_start + stop) for start, stop in local_spans
            )
            count_plan = _plan_count_overlap(
                self.comm.rank,
                spans,
                _count_demands(spans, policy.preceding, policy.following),
            )
            return await collective.exchange(
                _replay_buffer(buffer, chunk_spans),
                count_plan,
                resolver=SliceResolver(
                    resolve_chunk=_row_range_pieces,
                    resolve_summary=_row_range_summary_pieces,
                ),
            )
        except BaseException:
            buffer.clear()
            raise

    async def _buffer_count_input(
        self, buffer: ChunkStore
    ) -> tuple[int, tuple[RowRange, ...]]:
        """Buffer local chunks and return their count and rank-local spans."""
        rows = 0
        spans = []
        while (msg := await self.ch_in.recv(self.context)) is not None:
            chunk = TableChunk.from_message(msg, br=self.context.br())
            start = rows
            rows += chunk.shape[0]
            spans.append((start, rows))
            buffer.insert(Message(0, chunk))
        return rows, tuple(spans)

    async def _buffer_range_input(
        self,
        buffer: ChunkStore,
        policy: RangeOverlapPolicy,
    ) -> tuple[
        int,
        tuple[RowRange, ...],
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
        spans = []
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
            spans.append((row_offset, row_offset + nrows))
            row_offset += nrows
        return (
            row_offset,
            tuple(spans),
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
