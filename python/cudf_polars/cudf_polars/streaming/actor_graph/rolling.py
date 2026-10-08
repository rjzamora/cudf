# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    OrderKey,
    OrderScheme,
    Ordering,
    Partitioning,
)
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.ir import IR, Rolling
from cudf_polars.dsl.utils.windows import duration_to_scalar
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.collectives.overlap import (
    BufferedChunk,
    BufferedChunkSource,
    ResolvedGhostSend,
    RowExchange,
    RowExchangePlan,
    exchange_resolved_ghost_slices,
    extract_region,
    gather_row_counts,
)
from cudf_polars.streaming.actor_graph.collectives.sort import (
    _extract_boundaries_from_endpoint_rows,
)
from cudf_polars.streaming.actor_graph.dispatch import generate_ir_sub_network
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    _evaluate_chunk_sync,
    maybe_remap_partitioning,
    names_to_indices,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.rolling import FixedSizeRolling
from cudf_polars.utils.cuda_stream import join_cuda_streams

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.collectives.overlap import (
        RowExchangeResult,
        RowRange,
    )
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer


@dataclass
class RangeOverlap:
    """Range-window metadata for one buffered input chunk."""

    index_column: plc.Column
    lower_bound_column: plc.Column
    upper_bound_column: plc.Column
    first: Any | None
    last: Any | None


@dataclass
class RangeBufferedChunk(BufferedChunk):
    """A buffered chunk with range-window overlap metadata."""

    overlap: RangeOverlap


BufferedChunkT = TypeVar("BufferedChunkT", bound=BufferedChunk)
_INT64_DTYPE = DataType(pl.Int64())
_INT8_DTYPE = DataType(pl.Int8())


class RollingPolicy(Protocol[BufferedChunkT]):
    """Protocol for evaluating ghost-expanded cursor chunks."""

    def observe(self, chunk: BufferedChunkT) -> None:
        """Record resources that must outlive chunk processing."""
        ...

    def close(self) -> None:
        """Finalize any policy-owned resources."""
        ...

    def validate_cursor(self, context: Context, cursor: BufferedChunkT) -> None:
        """Validate ordering assumptions before evaluating cursor."""
        ...

    def evict_history(
        self,
        history: list[BufferedChunkT],
        cursor: BufferedChunkT,
        context: Context,
    ) -> list[BufferedChunkT]:
        """Drop chunks that cannot contribute to the cursor chunk."""
        ...

    def has_complete_future(
        self,
        latest: BufferedChunkT,
        current: BufferedChunkT,
        context: Context,
    ) -> bool:
        """Return whether latest has enough leading context for current."""
        ...

    def release_cached_before(
        self,
        history: list[BufferedChunkT],
        cursor: BufferedChunkT,
        context: Context,
    ) -> int:
        """Return the row boundary before which local source chunks can be released."""
        ...

    async def evaluate_cursor(
        self,
        context: Context,
        ir: IR,
        ir_context: IRExecutionContext,
        cursor: BufferedChunkT,
        *,
        history: list[BufferedChunkT],
        future: list[BufferedChunkT],
    ) -> TableChunk:
        """Evaluate the cursor chunk with any required overlap."""
        ...

    async def prepare_chunk(
        self,
        context: Context,
        msg: Message,
        *,
        row_offset: int,
    ) -> BufferedChunkT:
        """Convert an input message to a chunk with overlap metadata."""
        ...


@dataclass
class RangeOverlapPolicy(RollingPolicy[RangeBufferedChunk]):
    """Overlap policy for range-based rolling windows."""

    lower: plc.Scalar
    upper: plc.Scalar
    index: int
    index_dtype: plc.DataType
    find_start: Callable[..., plc.Column]
    find_end: Callable[..., plc.Column]
    start_inclusive: bool
    end_inclusive: bool
    stream: Stream
    index_name: str
    observed_streams: set[Stream] = field(default_factory=set)
    previous_index_value: Any | None = None

    @classmethod
    def from_ir(cls, ir: Rolling, stream: Stream) -> RangeOverlapPolicy:
        """Create reusable range-bound state for the rolling actor."""
        (index,) = names_to_indices([ir.index.name], ir.children[0].schema)
        side = ir.closed_window
        start_inclusive = side in ("both", "left")
        end_inclusive = side not in ("both", "right")
        find_start = (
            plc.search.lower_bound if start_inclusive else plc.search.upper_bound
        )
        find_end = plc.search.lower_bound if end_inclusive else plc.search.upper_bound
        dtype = ir.index_dtype
        policy = cls(
            # Note: not using windows_to_offsets because that flips the sign of
            # preceding_ordinal.
            duration_to_scalar(dtype, ir.preceding_ordinal, stream=stream),
            duration_to_scalar(
                dtype,
                ir.preceding_ordinal + ir.following_ordinal,
                stream=stream,
            ),
            index,
            dtype,
            find_start,
            find_end,
            start_inclusive,
            end_inclusive,
            stream,
            ir.index.name,
        )
        # Simpler and probably more efficient that inducing cross-stream deps
        # to read the windows every time we add them to a chunk.
        stream.synchronize()
        return policy

    def observe(self, chunk: RangeBufferedChunk) -> None:
        """Track streams that own range-bound columns."""
        self.observed_streams.add(chunk.chunk.stream)

    def close(self) -> None:
        """Keep offset scalars alive until all observed work is ordered."""
        if self.observed_streams:
            join_cuda_streams(
                downstreams=(self.stream,),
                upstreams=tuple(self.observed_streams),
            )

    def validate_index_column(self, index_column: plc.Column, stream: Stream) -> None:
        """Validate chunk-local ordering before using search primitives."""
        if index_column.null_count() != 0:
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling may not contain nulls"
            )
        if index_column.size() > 1 and not plc.sorting.is_sorted(
            plc.Table([index_column]),
            [plc.types.Order.ASCENDING],
            [plc.types.NullOrder.BEFORE],
            stream=stream,
        ):
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling is not sorted, "
                "please sort first"
            )

    def validate_cursor(self, context: Context, cursor: RangeBufferedChunk) -> None:
        """Check that the index remains sorted across chunk boundaries."""
        if cursor.num_rows == 0:
            return
        del context
        assert cursor.overlap.first is not None
        assert cursor.overlap.last is not None
        if (
            self.previous_index_value is not None
            and cursor.overlap.first < self.previous_index_value
        ):
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling is not sorted, "
                "please sort first"
            )
        self.previous_index_value = cursor.overlap.last

    async def prepare_table_chunk(
        self,
        context: Context,
        sequence_number: int,
        chunk: TableChunk,
        *,
        row_offset: int,
    ) -> RangeBufferedChunk:
        """Stage a chunk and extract its physical index bounds."""
        nrows, _ = chunk.shape
        chunk, extra = await make_table_chunks_available_or_wait(
            context,
            chunk,
            # TODO: Only reserve if needing to cast index column
            reserve_extra=nrows * 8,
            net_memory_delta=0,
        )
        with opaque_memory_usage(extra):
            index_column = chunk.table_view().columns()[self.index]
            if index_column.type() != self.index_dtype:
                index_column = plc.unary.cast(
                    index_column, self.index_dtype, stream=chunk.stream
                )
            self.validate_index_column(index_column, chunk.stream)
        if nrows == 0:
            lower_bound = upper_bound = index_column
            first = last = None
        else:
            lower_bound = index_with_offset(
                index_column, 0, self.lower, chunk.stream, context.br()
            )
            upper_bound = index_with_offset(
                index_column, nrows - 1, self.upper, chunk.stream, context.br()
            )
            first = column_value(index_column, 0, stream=chunk.stream, br=context.br())
            last = column_value(
                index_column, nrows - 1, stream=chunk.stream, br=context.br()
            )
        return RangeBufferedChunk(
            sequence_number,
            chunk,
            row_offset,
            nrows,
            RangeOverlap(
                index_column,
                lower_bound,
                upper_bound,
                first,
                last,
            ),
        )

    async def prepare_chunk(
        self,
        context: Context,
        msg: Message,
        *,
        row_offset: int,
    ) -> RangeBufferedChunk:
        """Convert a message to a staged chunk and extract its physical index."""
        return await self.prepare_table_chunk(
            context,
            msg.sequence_number,
            TableChunk.from_message(msg, br=context.br()),
            row_offset=row_offset,
        )

    def evict_history(
        self,
        history: list[RangeBufferedChunk],
        cursor: RangeBufferedChunk,
        context: Context,
    ) -> list[RangeBufferedChunk]:
        """Drop history chunks that cannot contribute to the cursor chunk."""
        if not history:
            return []
        insertion_point = global_insertion_row(
            history,
            cursor.overlap.lower_bound_column,
            self.find_start,
            needle_stream=cursor.chunk.stream,
            br=context.br(),
        )
        return [chunk for chunk in history if chunk.row_stop > insertion_point]

    def has_complete_future(
        self,
        chunk: RangeBufferedChunk,
        current: RangeBufferedChunk,
        context: Context,
    ) -> bool:
        """Return whether chunk contains current's upper insertion point."""
        insertion_point = global_insertion_row(
            [chunk],
            current.overlap.upper_bound_column,
            self.find_end,
            needle_stream=current.chunk.stream,
            br=context.br(),
        )
        return insertion_point < chunk.row_stop

    def release_cached_before(
        self,
        history: list[RangeBufferedChunk],
        cursor: RangeBufferedChunk,
        context: Context,
    ) -> int:
        """Return the earliest retained row needed by future range windows."""
        del context
        return history[0].row_start if history else cursor.row_start

    async def evaluate_cursor(
        self,
        context: Context,
        ir: IR,
        ir_context: IRExecutionContext,
        cursor: RangeBufferedChunk,
        *,
        history: list[RangeBufferedChunk],
        future: list[RangeBufferedChunk],
    ) -> TableChunk:
        """Evaluate the rolling aggregation for the cursor chunk."""
        chunks = [*history, cursor, *future]
        return await self.evaluate_cursor_with_chunks(
            context,
            ir,
            ir_context,
            cursor,
            chunks=chunks,
        )

    async def evaluate_cursor_with_chunks(
        self,
        context: Context,
        ir: IR,
        ir_context: IRExecutionContext,
        cursor: RangeBufferedChunk,
        *,
        chunks: Sequence[RangeBufferedChunk],
    ) -> TableChunk:
        """Evaluate the cursor against an already assembled context."""
        ghost_start = global_insertion_row(
            chunks,
            cursor.overlap.lower_bound_column,
            self.find_start,
            needle_stream=cursor.chunk.stream,
            br=context.br(),
        )
        ghost_stop = global_insertion_row(
            chunks,
            cursor.overlap.upper_bound_column,
            self.find_end,
            needle_stream=cursor.chunk.stream,
            br=context.br(),
        )
        # We must extract at least the whole of the current cursor chunk.
        ghost_start = min(cursor.row_start, ghost_start)
        ghost_stop = max(cursor.row_stop, ghost_stop)
        return await evaluate_ghosted_cursor(
            context,
            ir,
            ir_context,
            cursor,
            chunks=chunks,
            ghost_start=ghost_start,
            ghost_stop=ghost_stop,
        )


@dataclass
class RowCountOverlapPolicy(RollingPolicy[BufferedChunk]):
    """Overlap policy for fixed-size rolling expressions."""

    preceding: int
    following: int

    def observe(self, chunk: BufferedChunk) -> None:
        """Row-count overlap has no policy-owned chunk resources."""
        del chunk

    def close(self) -> None:
        """Row-count overlap owns no resources that need finalization."""

    def validate_cursor(self, context: Context, cursor: BufferedChunk) -> None:
        """Row-count rolling has no ordering requirement."""
        del context, cursor

    async def prepare_chunk(
        self,
        context: Context,
        msg: Message,
        *,
        row_offset: int,
    ) -> BufferedChunk:
        """Convert a message to an available chunk with row-span metadata."""
        chunk = TableChunk.from_message(msg, br=context.br())
        nrows, _ = chunk.shape
        chunk, extra = await make_table_chunks_available_or_wait(
            context, chunk, reserve_extra=0, net_memory_delta=0
        )
        with opaque_memory_usage(extra):
            pass
        return BufferedChunk(msg.sequence_number, chunk, row_offset, nrows)

    def evict_history(
        self,
        history: list[BufferedChunk],
        cursor: BufferedChunk,
        context: Context,
    ) -> list[BufferedChunk]:
        """Drop history chunks that cannot contribute to the cursor chunk."""
        del context
        ghost_start = max(0, cursor.row_start - self.preceding)
        return [chunk for chunk in history if chunk.row_stop > ghost_start]

    def has_complete_future(
        self,
        latest: BufferedChunk,
        current: BufferedChunk,
        context: Context,
    ) -> bool:
        """Return whether latest contains enough fixed-size leading rows."""
        del context
        return latest.row_stop >= current.row_stop + self.following

    def release_cached_before(
        self,
        history: list[BufferedChunk],
        cursor: BufferedChunk,
        context: Context,
    ) -> int:
        """Return the earliest retained row needed by fixed-size windows."""
        del context, history
        return max(0, cursor.row_start - self.preceding)

    async def evaluate_cursor(
        self,
        context: Context,
        ir: IR,
        ir_context: IRExecutionContext,
        cursor: BufferedChunk,
        *,
        history: list[BufferedChunk],
        future: list[BufferedChunk],
    ) -> TableChunk:
        """Evaluate fixed-size rolling expressions for the cursor chunk."""
        ghost_start = max(0, cursor.row_start - self.preceding)
        chunks = [*history, cursor, *future]
        ghost_stop = min(cursor.row_stop + self.following, chunks[-1].row_stop)
        return await evaluate_ghosted_cursor(
            context,
            ir,
            ir_context,
            cursor,
            chunks=chunks,
            ghost_start=ghost_start,
            ghost_stop=ghost_stop,
        )


@dataclass(frozen=True)
class RangeChunkBounds:
    """Host-side value bounds for one local range-rolling chunk."""

    row_start: int
    row_stop: int
    first: Any
    last: Any


@dataclass(frozen=True)
class RangeInputSummary:
    """Local rows and request bounds for one rank."""

    local_rows: int
    has_rows: bool
    lower_bound_column: plc.Column | None
    upper_bound_column: plc.Column | None
    first_value_column: plc.Column | None
    last_value_column: plc.Column | None
    chunk_bounds: tuple[RangeChunkBounds, ...]
    endpoint_rows: plc.Table | None


@dataclass(frozen=True)
class RangeRequests:
    """All-gathered range requests, one row per rank."""

    row_counts: tuple[int, ...]
    has_request: tuple[bool, ...]
    lower_values: tuple[Any | None, ...]
    upper_values: tuple[Any | None, ...]
    first_values: tuple[Any | None, ...]
    last_values: tuple[Any | None, ...]
    lower_column: plc.Column
    upper_column: plc.Column
    first_column: plc.Column
    last_column: plc.Column
    stream: Stream
    table: plc.Table


@dataclass(frozen=True)
class StagedChunk(Generic[BufferedChunkT]):
    """A chunk staged for evaluation, with output-ownership state."""

    chunk: BufferedChunkT
    owned: bool
    output_sequence_number: int | None = None


def _noop_cleanup() -> None:
    """Default cleanup hook for prepared inputs without buffered state."""


@dataclass
class RollingInput(Generic[BufferedChunkT]):
    """A prepared rolling input stream plus optional metadata learned from it."""

    chunks: AsyncIterator[StagedChunk[BufferedChunkT]]
    partitioning: Partitioning | None = None
    release_sources: tuple[BufferedChunkSource, ...] = ()
    cleanup: Callable[[], None] = _noop_cleanup
    _closed: bool = field(default=False, init=False)

    def close(self) -> None:
        """Release input-owned buffered state."""
        if not self._closed:
            self.cleanup()
            self._closed = True

    def release_cached_before(self, row_stop: int) -> None:
        """Release buffered chunks that can no longer affect future cursors."""
        for source in self.release_sources:
            source.release_cached_before(row_stop)


@dataclass
class RollingManager(Generic[BufferedChunkT]):
    """Prepared rolling policy and input, with shared cleanup."""

    policy: RollingPolicy[BufferedChunkT]
    rolling_input: RollingInput[BufferedChunkT]

    async def __aenter__(self) -> RollingManager[BufferedChunkT]:
        """Return the prepared rolling context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Release prepared input and policy-owned resources."""
        self.rolling_input.close()
        self.policy.close()


def index_with_offset(
    index: plc.Column,
    row: int,
    offset: plc.Scalar,
    stream: Stream,
    br: BufferResource,
) -> plc.Column:
    """Return ``index[row] + offset`` as a single-row device column."""
    (endpoint,) = plc.copying.slice(index, [row, row + 1], stream=stream)
    return plc.binaryop.binary_operation(
        endpoint,
        offset,
        plc.binaryop.BinaryOperator.ADD,
        index.type(),
        stream=stream,
        mr=br.device_mr,
    )


def _chrono_storage_dtype(dtype: plc.DataType) -> plc.DataType:
    """Return the integer storage type for a chrono dtype."""
    if dtype.id() in (plc.TypeId.TIMESTAMP_DAYS, plc.TypeId.DURATION_DAYS):
        return plc.DataType(plc.TypeId.INT32)
    return plc.DataType(plc.TypeId.INT64)


def _host_ordering_value(
    value: plc.Column,
    *,
    stream: Stream,
    br: BufferResource,
) -> Any:
    """Copy a single-row ordering value to host."""
    if plc.traits.is_chrono(value.type()):
        value = plc.unary.bit_cast(
            value, _chrono_storage_dtype(value.type()), stream=stream, mr=br.device_mr
        )
    return value.to_scalar(stream=stream).to_py(stream=stream)


def column_value(
    column: plc.Column, row: int, *, stream: Stream, br: BufferResource
) -> Any:
    """Return one value from a non-empty column."""
    (value,) = plc.copying.slice(column, [row, row + 1], stream=stream)
    return _host_ordering_value(value, stream=stream, br=br)


def global_insertion_row(
    chunks: Sequence[RangeBufferedChunk],
    needle: plc.Column,
    find: Callable[..., plc.Column],
    *,
    needle_stream: Stream,
    br: BufferResource,
) -> int:
    """Return the globally indexed insertion row of a needle in some chunks."""
    assert len(chunks) > 0
    for chunk in chunks:
        if chunk.num_rows == 0:
            continue
        stream = chunk.chunk.stream
        join_cuda_streams(downstreams=[stream], upstreams=[needle_stream])
        # Since this returns a python integer, the work queued on search stream
        # is complete, so we don't need to join back to the search and needle
        # streams.
        insertion_point = _search_in_chunk(
            chunk.overlap.index_column, needle, find, stream, br
        )
        if insertion_point < chunk.num_rows:
            return chunk.row_start + insertion_point
    # Needle is later than all the chunks we know about.
    return chunks[-1].row_stop


def _search_in_chunk(
    index_column: plc.Column,
    needle: plc.Column,
    find: Callable[..., plc.Column],
    stream: Stream,
    br: BufferResource,
) -> int:
    """Return the chunk-local insertion row of a single-row needle."""
    value = (
        find(
            plc.Table([index_column]),
            plc.Table([needle]),
            [plc.types.Order.ASCENDING],
            [plc.types.NullOrder.AFTER],
            stream=stream,
            mr=br.device_mr,
        )
        .to_scalar(stream=stream)
        .to_py(stream=stream)
    )
    if not isinstance(value, int):
        raise TypeError(f"Expected integer insertion point, got {type(value).__name__}")
    return value


def latest_nonempty_chunk(
    current: BufferedChunkT, future: list[BufferedChunkT]
) -> BufferedChunkT:
    """Return the last non-empty chunk staged at or after current."""
    for chunk in reversed(future):
        if chunk.num_rows != 0:
            return chunk
    return current


async def evaluate_available_chunk(
    context: Context,
    chunk: TableChunk,
    ir: IR,
    *,
    ir_context: IRExecutionContext,
) -> TableChunk:
    """Evaluate an already available chunk."""
    reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
        chunk.data_alloc_size(), net_memory_delta=0
    )
    with opaque_memory_usage(reservation):
        return await ir_context.to_thread(
            _evaluate_chunk_sync, chunk, ir, ir_context, context.br()
        )


async def evaluate_ghosted_cursor(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    cursor: BufferedChunk,
    *,
    chunks: Sequence[BufferedChunk],
    ghost_start: int,
    ghost_stop: int,
) -> TableChunk:
    """Evaluate a ghost-expanded region and slice back to cursor rows."""
    ghosted_chunk = await extract_region(
        context,
        chunks,
        ghost_start,
        ghost_stop,
        ir_context=ir_context,
    )
    result = await evaluate_available_chunk(
        context,
        ghosted_chunk,
        ir,
        ir_context=ir_context,
    )
    (table,) = plc.copying.slice(
        result.table_view(),
        [cursor.row_start - ghost_start, cursor.row_stop - ghost_start],
        stream=result.stream,
    )
    result_bytes = sum(column.device_buffer_size() for column in table.columns())
    reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
        result_bytes, net_memory_delta=result_bytes
    )
    with opaque_memory_usage(reservation):
        table = table.copy(result.stream, context.br().device_mr)
    return TableChunk.from_pylibcudf_table(
        table,
        result.stream,
        exclusive_view=True,
        br=context.br(),
    )


def _row_starts(row_counts: Sequence[int]) -> list[int]:
    """Return global row starts for per-rank fixed-size rolling input."""
    starts = [0]
    for count in row_counts:
        starts.append(starts[-1] + count)
    return starts


def _row_count_ghost_spans(
    owned: RowRange,
    *,
    preceding: int,
    following: int,
    total_rows: int,
) -> tuple[RowRange, ...]:
    """Return fixed-size ghost spans needed by one rank-owned row span."""
    start, stop = owned
    if start == stop:
        return ()
    spans: list[RowRange] = []
    if preceding > 0:
        spans.append((max(0, start - preceding), start))
    if following > 0:
        spans.append((stop, min(total_rows, stop + following)))
    return tuple(span for span in spans if span[0] < span[1])


def _fixed_size_exchange_plan(
    row_counts: Sequence[int],
    policy: RowCountOverlapPolicy,
) -> RowExchangePlan:
    """Build the row-slice exchange plan for fixed-size rolling."""
    offsets = _row_starts(row_counts)
    spans = tuple((offsets[i], offsets[i + 1]) for i in range(len(row_counts)))
    total_rows = offsets[-1]
    ghost_requests = tuple(
        _row_count_ghost_spans(
            span,
            preceding=policy.preceding,
            following=policy.following,
            total_rows=total_rows,
        )
        for span in spans
    )
    return RowExchangePlan.from_spans(spans, ghost_requests)


async def _recv_all_fixed_size_chunks(
    context: Context,
    ch_in: Channel[TableChunk],
    source: BufferedChunkSource,
) -> int:
    """Drain a fixed-size rolling input channel into a spillable chunk source."""
    row_offset = 0
    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg, br=context.br())
        nrows, _ = chunk.shape
        source.insert(
            BufferedChunk(
                msg.sequence_number,
                chunk,
                row_offset,
                nrows,
            )
        )
        row_offset += nrows
    return row_offset


async def _recv_all_range_chunks(
    context: Context,
    ch_in: Channel[TableChunk],
    source: BufferedChunkSource,
    policy: RangeOverlapPolicy,
) -> RangeInputSummary:
    """Drain a range rolling input channel into a spillable chunk source."""
    row_offset = 0
    has_rows = False
    lower_bound_column: plc.Column | None = None
    upper_bound_column: plc.Column | None = None
    first_value_column: plc.Column | None = None
    last_value_column: plc.Column | None = None
    chunk_bounds: list[RangeChunkBounds] = []
    endpoint_rows: list[plc.Table] = []
    previous_last: Any | None = None
    while (msg := await ch_in.recv(context)) is not None:
        buffered = await policy.prepare_chunk(context, msg, row_offset=row_offset)
        policy.observe(buffered)
        row_offset = buffered.row_stop
        if buffered.num_rows == 0:
            source.insert(buffered)
            continue
        assert buffered.overlap.first is not None
        assert buffered.overlap.last is not None
        if previous_last is not None and buffered.overlap.first < previous_last:
            raise RuntimeError(
                f"Index column '{policy.index_name}' in rolling is not sorted, "
                "please sort first"
            )
        previous_last = buffered.overlap.last
        chunk_bounds.append(
            RangeChunkBounds(
                buffered.row_start,
                buffered.row_stop,
                buffered.overlap.first,
                buffered.overlap.last,
            )
        )
        has_rows = True
        if lower_bound_column is None:
            lower_bound_column = buffered.overlap.lower_bound_column
            first_value_column = _single_ordering_value_column(
                buffered.overlap.first,
                policy.index_dtype,
                stream=buffered.chunk.stream,
                br=context.br(),
            )
        upper_bound_column = buffered.overlap.upper_bound_column
        last_value_column = _single_ordering_value_column(
            buffered.overlap.last,
            policy.index_dtype,
            stream=buffered.chunk.stream,
            br=context.br(),
        )
        first_endpoint = _single_ordering_value_column(
            buffered.overlap.first,
            policy.index_dtype,
            stream=buffered.chunk.stream,
            br=context.br(),
        )
        last_endpoint = _single_ordering_value_column(
            buffered.overlap.last,
            policy.index_dtype,
            stream=buffered.chunk.stream,
            br=context.br(),
        )
        endpoint_rows.append(
            plc.Table(
                [
                    plc.concatenate.concatenate(
                        [first_endpoint, last_endpoint],
                        stream=buffered.chunk.stream,
                    )
                ]
            )
        )
        source.insert(buffered)
    if endpoint_rows and policy.observed_streams:
        join_cuda_streams(
            downstreams=(policy.stream,),
            upstreams=tuple(policy.observed_streams),
        )
    endpoint_table = (
        plc.concatenate.concatenate(endpoint_rows, stream=policy.stream)
        if endpoint_rows
        else None
    )
    return RangeInputSummary(
        local_rows=row_offset,
        has_rows=has_rows,
        lower_bound_column=lower_bound_column,
        upper_bound_column=upper_bound_column,
        first_value_column=first_value_column,
        last_value_column=last_value_column,
        chunk_bounds=tuple(chunk_bounds),
        endpoint_rows=endpoint_table,
    )


def _single_value_column(
    value: Any,
    dtype: plc.DataType,
    stream: Stream,
) -> plc.Column:
    """Return a one-row column containing ``value``."""
    return plc.Column.from_scalar(
        plc.Scalar.from_py(value, dtype, stream=stream),
        1,
        stream=stream,
    )


def _single_ordering_value_column(
    value: Any,
    dtype: plc.DataType,
    stream: Stream,
    br: BufferResource,
) -> plc.Column:
    """Return a one-row ordering column from a host-comparable value."""
    if plc.traits.is_chrono(dtype):
        storage = _single_value_column(value, _chrono_storage_dtype(dtype), stream)
        return plc.unary.bit_cast(storage, dtype, stream=stream, mr=br.device_mr)
    return _single_value_column(value, dtype, stream)


def _empty_column(dtype: plc.DataType, stream: Stream) -> plc.Column:
    """Return an empty column for a pylibcudf dtype."""
    return plc.column_factories.make_empty_column(dtype, stream=stream)


def _single_key_ordering_partitioning(
    context: Context,
    policy: RangeOverlapPolicy,
    endpoint_rows: plc.Table,
    num_partitions: int,
    stream: Stream,
) -> Partitioning | None:
    """Build ordering metadata from all-gathered chunk endpoint rows."""
    if num_partitions < 2:
        return None
    boundaries, strict = _extract_boundaries_from_endpoint_rows(
        endpoint_rows,
        num_partitions,
        stream,
    )
    return Partitioning(
        inter_rank=OrderScheme(
            [
                Ordering(
                    [
                        OrderKey(
                            policy.index,
                            plc.types.Order.ASCENDING,
                            plc.types.NullOrder.BEFORE,
                        )
                    ],
                    TableChunk.from_pylibcudf_table(
                        boundaries,
                        stream,
                        exclusive_view=True,
                        br=context.br(),
                    ),
                    strict_boundaries=strict,
                )
            ]
        ),
        local="inherit",
    )


def _partitioning_has_inter_rank(partitioning: Partitioning | None) -> bool:
    """Return whether partitioning advertises inter-rank ownership."""
    return partitioning is not None and partitioning.inter_rank is not None


def _merge_learned_inter_rank(
    existing: Partitioning | None,
    learned: Partitioning | None,
) -> Partitioning | None:
    """Fill a missing inter-rank scheme without discarding local metadata."""
    if learned is None or _partitioning_has_inter_rank(existing):
        return existing
    if existing is None:
        return learned
    return Partitioning(inter_rank=learned.inter_rank, local=existing.local)


def _range_endpoint_chunk(
    context: Context,
    summary: RangeInputSummary,
    policy: RangeOverlapPolicy,
    stream: Stream,
) -> TableChunk:
    """Return local endpoint rows for ordering-metadata extraction."""
    table = (
        summary.endpoint_rows
        if summary.endpoint_rows is not None
        else plc.Table([_empty_column(policy.index_dtype, stream)])
    )
    return TableChunk.from_pylibcudf_table(
        table,
        stream,
        exclusive_view=True,
        br=context.br(),
    )


async def _gather_range_partitioning(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    summary: RangeInputSummary,
    policy: RangeOverlapPolicy,
    *,
    collective_id: int,
) -> Partitioning | None:
    """Gather chunk endpoints and derive output ordering metadata."""
    stream = context.br().stream_pool.get_stream()
    if policy.observed_streams:
        join_cuda_streams(
            downstreams=(stream,),
            upstreams=tuple(policy.observed_streams),
        )
    ag = AllGatherManager(context, comm, collective_id)
    with ag.inserting() as inserter:
        await inserter.insert(
            comm.rank,
            _range_endpoint_chunk(context, summary, policy, stream),
        )
    endpoint_rows = await ag.extract_concatenated(
        stream,
        ordered=True,
        ir_context=ir_context,
    )
    num_partitions = endpoint_rows.num_rows() // 2
    if endpoint_rows.num_rows() != num_partitions * 2:
        raise RuntimeError(
            "Range rolling gathered an invalid number of endpoint rows: "
            f"{endpoint_rows.num_rows()}"
        )
    if num_partitions == 0:
        return None
    return _single_key_ordering_partitioning(
        context,
        policy,
        endpoint_rows,
        num_partitions,
        stream,
    )


def _range_request_chunk(
    context: Context,
    summary: RangeInputSummary,
    policy: RangeOverlapPolicy,
) -> TableChunk:
    """Build this rank's compact range-overlap request."""
    stream = context.br().stream_pool.get_stream()
    if policy.observed_streams:
        join_cuda_streams(
            downstreams=(stream,),
            upstreams=tuple(policy.observed_streams),
        )
    row_count = _single_value_column(
        summary.local_rows,
        plc.DataType(plc.TypeId.INT64),
        stream,
    )
    has_request = _single_value_column(
        int(summary.has_rows),
        plc.DataType(plc.TypeId.INT8),
        stream,
    )
    lower_bound = (
        summary.lower_bound_column
        if summary.lower_bound_column is not None
        else _single_value_column(None, policy.index_dtype, stream)
    )
    upper_bound = (
        summary.upper_bound_column
        if summary.upper_bound_column is not None
        else _single_value_column(None, policy.index_dtype, stream)
    )
    first_value = (
        summary.first_value_column
        if summary.first_value_column is not None
        else _single_value_column(None, policy.index_dtype, stream)
    )
    last_value = (
        summary.last_value_column
        if summary.last_value_column is not None
        else _single_value_column(None, policy.index_dtype, stream)
    )
    return TableChunk.from_pylibcudf_table(
        plc.Table(
            [row_count, has_request, lower_bound, upper_bound, first_value, last_value]
        ),
        stream,
        exclusive_view=False,
        br=context.br(),
    )


async def _gather_range_requests(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    summary: RangeInputSummary,
    policy: RangeOverlapPolicy,
    *,
    collective_id: int,
) -> RangeRequests:
    """Collect one compact range request from every rank."""
    ag = AllGatherManager(context, comm, collective_id)
    with ag.inserting() as inserter:
        await inserter.insert(0, _range_request_chunk(context, summary, policy))
    stream = context.br().stream_pool.get_stream()
    table = await ag.extract_concatenated(stream, ordered=True, ir_context=ir_context)
    request_df = DataFrame.from_table(
        plc.Table(table.columns()[:2]),
        ["row_count", "has_request"],
        [_INT64_DTYPE, _INT8_DTYPE],
        stream,
    ).to_polars()
    row_counts = tuple(int(value) for value in request_df["row_count"].to_list())
    has_request = tuple(bool(value) for value in request_df["has_request"].to_list())
    if len(row_counts) != comm.nranks:
        raise RuntimeError(
            "Range-request allgather returned an unexpected number of rows: "
            f"expected {comm.nranks}, got {len(row_counts)}"
        )
    lower_column = table.columns()[2]
    upper_column = table.columns()[3]
    first_column = table.columns()[4]
    last_column = table.columns()[5]
    raw_lower_values = tuple(
        _request_host_value(lower_column, rank, stream=stream, br=context.br())
        if has_request[rank]
        else None
        for rank in range(comm.nranks)
    )
    raw_upper_values = tuple(
        _request_host_value(upper_column, rank, stream=stream, br=context.br())
        if has_request[rank]
        else None
        for rank in range(comm.nranks)
    )
    first_values = tuple(
        _request_host_value(first_column, rank, stream=stream, br=context.br())
        if has_request[rank]
        else None
        for rank in range(comm.nranks)
    )
    last_values = tuple(
        _request_host_value(last_column, rank, stream=stream, br=context.br())
        if has_request[rank]
        else None
        for rank in range(comm.nranks)
    )
    lower_values = tuple(
        _minimum_requested_value(
            first_values[rank],
            raw_lower_values[rank],
            has_request=has_request[rank],
        )
        for rank in range(comm.nranks)
    )
    upper_values = tuple(
        _maximum_requested_value(
            last_values[rank],
            raw_upper_values[rank],
            has_request=has_request[rank],
        )
        for rank in range(comm.nranks)
    )
    return RangeRequests(
        row_counts=row_counts,
        has_request=has_request,
        lower_values=lower_values,
        upper_values=upper_values,
        first_values=first_values,
        last_values=last_values,
        lower_column=lower_column,
        upper_column=upper_column,
        first_column=first_column,
        last_column=last_column,
        stream=stream,
        table=table,
    )


def _request_host_value(
    column: plc.Column,
    rank: int,
    *,
    stream: Stream,
    br: BufferResource,
) -> Any:
    """Return one all-gathered request value on the host."""
    return _host_ordering_value(
        plc.copying.slice(column, [rank, rank + 1], stream=stream)[0],
        stream=stream,
        br=br,
    )


def _minimum_requested_value(
    left: Any | None, right: Any | None, *, has_request: bool
) -> Any | None:
    """Return the lower of two requested values, or null for an empty request."""
    if not has_request or left is None or right is None:
        return None
    return left if left <= right else right


def _maximum_requested_value(
    left: Any | None, right: Any | None, *, has_request: bool
) -> Any | None:
    """Return the upper of two requested values, or null for an empty request."""
    if not has_request or left is None or right is None:
        return None
    return left if left >= right else right


def _validate_global_range_order(
    requests: RangeRequests,
    policy: RangeOverlapPolicy,
) -> None:
    """Validate that rank-local sorted ranges are globally ordered."""
    previous_last: Any | None = None
    for rank, has_request in enumerate(requests.has_request):
        if not has_request:
            continue
        first = requests.first_values[rank]
        last = requests.last_values[rank]
        if first is None or last is None:
            raise RuntimeError(
                "Range rolling gathered incomplete rank bounds for non-empty "
                f"rank {rank}"
            )
        if first > last or (previous_last is not None and first < previous_last):
            raise RuntimeError(
                f"Index column '{policy.index_name}' in rolling is not sorted, "
                "please sort first"
            )
        previous_last = last


def _rank_may_satisfy_range_request(
    summary: RangeInputSummary,
    requests: RangeRequests,
    *,
    destination: int,
) -> bool:
    """Return whether this rank may contain rows requested by a destination."""
    if not summary.has_rows or not requests.has_request[destination]:
        return False
    request_lower = requests.lower_values[destination]
    request_upper = requests.upper_values[destination]
    if request_lower is None or request_upper is None:
        return False
    rank_first = summary.chunk_bounds[0].first
    rank_last = summary.chunk_bounds[-1].last
    return rank_last >= request_lower and rank_first <= request_upper


def _range_remote_sources(
    comm: Communicator,
    requests: RangeRequests,
) -> tuple[int, ...]:
    """Return remote ranks that may send range ghosts to this rank."""
    if not requests.has_request[comm.rank]:
        return ()
    request_lower = requests.lower_values[comm.rank]
    request_upper = requests.upper_values[comm.rank]
    if request_lower is None or request_upper is None:
        return ()
    return tuple(
        source
        for source in range(comm.nranks)
        if source != comm.rank
        and requests.has_request[source]
        and requests.first_values[source] is not None
        and requests.last_values[source] is not None
        and requests.last_values[source] >= request_lower
        and requests.first_values[source] <= request_upper
    )


def _range_remote_destinations(
    comm: Communicator,
    summary: RangeInputSummary,
    requests: RangeRequests,
) -> tuple[int, ...]:
    """Return remote ranks that may need range ghosts from this rank."""
    return tuple(
        destination
        for destination in range(comm.nranks)
        if destination != comm.rank
        and _rank_may_satisfy_range_request(
            summary,
            requests,
            destination=destination,
        )
    )


async def _range_request_sends(
    context: Context,
    source: BufferedChunkSource,
    summary: RangeInputSummary,
    requests: RangeRequests,
    destinations: Sequence[int],
    policy: RangeOverlapPolicy,
    *,
    row_offset: int,
) -> list[ResolvedGhostSend]:
    """Resolve remote range requests to exact local row slices."""
    sends: list[ResolvedGhostSend] = []
    for destination in destinations:
        request_lower = requests.lower_values[destination]
        request_upper = requests.upper_values[destination]
        if request_lower is None or request_upper is None:
            continue
        for chunk in summary.chunk_bounds:
            if chunk.last < request_lower or chunk.first > request_upper:
                continue
            async for buffered in source.iter_region_chunks(
                chunk.row_start + row_offset,
                chunk.row_stop + row_offset,
            ):
                index_column = buffered.chunk.table_view().columns()[policy.index]
                if index_column.type() != policy.index_dtype:
                    index_column = plc.unary.cast(
                        index_column,
                        policy.index_dtype,
                        stream=buffered.chunk.stream,
                    )
                lower = _single_ordering_value_column(
                    request_lower,
                    policy.index_dtype,
                    buffered.chunk.stream,
                    context.br(),
                )
                upper = _single_ordering_value_column(
                    request_upper,
                    policy.index_dtype,
                    buffered.chunk.stream,
                    context.br(),
                )
                start = _search_in_chunk(
                    index_column,
                    lower,
                    plc.search.lower_bound,
                    buffered.chunk.stream,
                    context.br(),
                )
                stop = _search_in_chunk(
                    index_column,
                    upper,
                    plc.search.upper_bound,
                    buffered.chunk.stream,
                    context.br(),
                )
                if start < stop:
                    sends.append(
                        ResolvedGhostSend(
                            destination,
                            buffered.row_start + start,
                            buffered.row_start + stop,
                        )
                    )
    return sends


async def _next_staged_chunk(
    chunks: AsyncIterator[StagedChunk[BufferedChunkT]],
) -> StagedChunk[BufferedChunkT] | None:
    """Return the next staged chunk, or ``None`` when exhausted."""
    try:
        return await anext(chunks)
    except StopAsyncIteration:
        return None


def _latest_nonempty_staged_chunk(
    current: BufferedChunkT,
    future: list[StagedChunk[BufferedChunkT]],
) -> BufferedChunkT:
    """Return the latest non-empty chunk among current and staged future."""
    for staged in reversed(future):
        if staged.chunk.num_rows != 0:
            return staged.chunk
    return current


async def _fill_staged_future(
    context: Context,
    chunks: AsyncIterator[StagedChunk[BufferedChunkT]],
    current: BufferedChunkT,
    future: list[StagedChunk[BufferedChunkT]],
    policy: RollingPolicy[BufferedChunkT],
) -> bool:
    """Read staged chunks until current has enough leading context."""
    while not policy.has_complete_future(
        _latest_nonempty_staged_chunk(current, future),
        current,
        context,
    ):
        staged = await _next_staged_chunk(chunks)
        if staged is None:
            return True
        future.append(staged)
    return False


async def _local_staged_chunks(
    context: Context,
    ch_in: Channel[TableChunk],
    policy: RollingPolicy[BufferedChunkT],
) -> AsyncIterator[StagedChunk[BufferedChunkT]]:
    """Yield a single-rank input channel as owned staged chunks."""
    row_offset = 0
    while (msg := await ch_in.recv(context)) is not None:
        chunk = await policy.prepare_chunk(
            context,
            msg,
            row_offset=row_offset,
        )
        policy.observe(chunk)
        row_offset = chunk.row_stop
        yield StagedChunk(chunk, owned=True, output_sequence_number=msg.sequence_number)


async def execute_rolling_policy(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    rolling_input: RollingInput[BufferedChunkT],
    policy: RollingPolicy[BufferedChunkT],
    tracer: ActorTracer | None,
) -> None:
    """Evaluate owned staged chunks with any required ghost context."""
    chunks = rolling_input.chunks
    history: list[BufferedChunkT] = []
    future: list[StagedChunk[BufferedChunkT]] = []
    input_exhausted = False
    next_output_sequence_number = 0
    staged = await _next_staged_chunk(chunks)
    while staged is not None:
        cursor, owned = staged.chunk, staged.owned
        policy.validate_cursor(context, cursor)
        if not owned:
            history.append(cursor)
        elif cursor.num_rows == 0:
            result = await evaluate_available_chunk(
                context,
                cursor.chunk,
                ir,
                ir_context=ir_context,
            )
            if tracer is not None:
                tracer.add_chunk(chunk=result)
            output_sequence_number = (
                staged.output_sequence_number
                if staged.output_sequence_number is not None
                else next_output_sequence_number
            )
            await ch_out.send(context, Message(output_sequence_number, result))
            next_output_sequence_number = output_sequence_number + 1
        else:
            history = policy.evict_history(history, cursor, context)
            release_before = policy.release_cached_before(history, cursor, context)
            rolling_input.release_cached_before(release_before)
            if not input_exhausted:
                input_exhausted = await _fill_staged_future(
                    context,
                    chunks,
                    cursor,
                    future,
                    policy,
                )
            result = await policy.evaluate_cursor(
                context,
                ir,
                ir_context,
                cursor,
                history=history,
                future=[item.chunk for item in future],
            )
            if tracer is not None:
                tracer.add_chunk(chunk=result)
            output_sequence_number = (
                staged.output_sequence_number
                if staged.output_sequence_number is not None
                else next_output_sequence_number
            )
            await ch_out.send(context, Message(output_sequence_number, result))
            next_output_sequence_number = output_sequence_number + 1
            history.append(cursor)

        if future:
            staged = future.pop(0)
        else:
            staged = await _next_staged_chunk(chunks)

    await ch_out.drain(context)


async def _fixed_size_context_chunks(
    result: RowExchangeResult,
    output_span: RowRange,
) -> AsyncIterator[StagedChunk[BufferedChunk]]:
    """Yield ghost and owned chunks in global row order for fixed-size rolling."""
    output_start, output_stop = output_span
    ghost_iter = result.ghost_source.iter_chunks()
    first_right_ghost: BufferedChunk | None = None
    async for chunk in ghost_iter:
        if chunk.row_stop <= output_start:
            yield StagedChunk(chunk, owned=False)
        elif chunk.row_start >= output_stop:
            first_right_ghost = chunk
            break
        else:
            raise RuntimeError(
                "Fixed-size rolling received a ghost chunk that overlaps owned rows"
            )
    async for chunk in result.iter_local_owned(
        include_empty_chunks=True,
    ):
        yield StagedChunk(chunk, owned=True)
    if first_right_ghost is not None:
        yield StagedChunk(first_right_ghost, owned=False)
    async for chunk in ghost_iter:
        if chunk.row_start < output_stop:
            raise RuntimeError(
                "Fixed-size rolling received a ghost chunk that overlaps owned rows"
            )
        yield StagedChunk(chunk, owned=False)


async def prepare_fixed_size_multirank_input(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    policy: RowCountOverlapPolicy,
    *,
    collective_id: int,
) -> RollingInput[BufferedChunk]:
    """
    Prepare fixed-size rolling input with sparse inter-rank ghost slices.

    The row-count all-gather completes before the row-slice exchange starts, so
    they intentionally share one collective ID. This matches the sequential
    reuse pattern used by non-scalar ``over``.
    """
    local_source = BufferedChunkSource(context)
    try:
        local_rows = await _recv_all_fixed_size_chunks(context, ch_in, local_source)
        row_counts = await gather_row_counts(
            context,
            comm,
            ir_context,
            local_rows=local_rows,
            collective_id=collective_id,
        )
        plan = _fixed_size_exchange_plan(row_counts, policy)

        local_start, _ = plan.source_span(comm.rank)
        local_source.set_row_start_offset(local_start)

        exchange_result = await RowExchange(
            context,
            comm,
            ir_context,
            plan,
            collective_id,
        ).exchange(local_source)

        def cleanup() -> None:
            exchange_result.ghost_source.clear()
            local_source.clear()

        return RollingInput(
            _fixed_size_context_chunks(
                exchange_result,
                plan.source_span(comm.rank),
            ),
            release_sources=(
                exchange_result.local_source,
                exchange_result.ghost_source,
            ),
            cleanup=cleanup,
        )
    except BaseException:
        local_source.clear()
        raise


async def _range_context_chunks(
    context: Context,
    result: RowExchangeResult,
    output_span: RowRange,
    policy: RangeOverlapPolicy,
) -> AsyncIterator[StagedChunk[RangeBufferedChunk]]:
    """Yield range ghosts and owned chunks in global row order."""
    output_start, output_stop = output_span
    ghost_iter = result.ghost_source.iter_chunks()
    first_right_ghost: RangeBufferedChunk | None = None
    async for chunk in ghost_iter:
        prepared = await policy.prepare_table_chunk(
            context,
            chunk.sequence_number,
            chunk.chunk,
            row_offset=chunk.row_start,
        )
        policy.observe(prepared)
        if prepared.row_stop <= output_start:
            yield StagedChunk(prepared, owned=False)
        elif prepared.row_start >= output_stop:
            first_right_ghost = prepared
            break
        else:
            raise RuntimeError(
                "Range rolling received a ghost chunk that overlaps owned rows"
            )
    async for chunk in result.iter_local_owned(
        include_empty_chunks=True,
    ):
        prepared = await policy.prepare_table_chunk(
            context,
            chunk.sequence_number,
            chunk.chunk,
            row_offset=chunk.row_start,
        )
        policy.observe(prepared)
        yield StagedChunk(prepared, owned=True)
    if first_right_ghost is not None:
        yield StagedChunk(first_right_ghost, owned=False)
    async for chunk in ghost_iter:
        prepared = await policy.prepare_table_chunk(
            context,
            chunk.sequence_number,
            chunk.chunk,
            row_offset=chunk.row_start,
        )
        policy.observe(prepared)
        if prepared.row_start < output_stop:
            raise RuntimeError(
                "Range rolling received a ghost chunk that overlaps owned rows"
            )
        yield StagedChunk(prepared, owned=False)


async def prepare_range_multirank_input(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    policy: RangeOverlapPolicy,
    *,
    collective_id: int,
    derive_partitioning: bool,
) -> RollingInput[RangeBufferedChunk]:
    """
    Prepare range rolling input with sparse inter-rank ghost chunks.

    Each rank gathers one compact request and its data value bounds. Source
    ranks use host-side chunk bounds to resolve which local chunks may satisfy
    remote requests, and exchange only those chunks as ghost context.
    """
    local_source = BufferedChunkSource(context)
    try:
        summary = await _recv_all_range_chunks(context, ch_in, local_source, policy)
        requests = await _gather_range_requests(
            context,
            comm,
            ir_context,
            summary,
            policy,
            collective_id=collective_id,
        )
        _validate_global_range_order(requests, policy)
        # The request all-gather, optional endpoint all-gather, and sparse
        # exchange are sequential, so they can share one collective ID.
        partitioning = (
            await _gather_range_partitioning(
                context,
                comm,
                ir_context,
                summary,
                policy,
                collective_id=collective_id,
            )
            if derive_partitioning
            else None
        )
        row_starts = _row_starts(requests.row_counts)
        output_span = (row_starts[comm.rank], row_starts[comm.rank + 1])
        local_source.set_row_start_offset(output_span[0])

        remote_sources = _range_remote_sources(comm, requests)
        remote_destinations = _range_remote_destinations(comm, summary, requests)
        sends = await _range_request_sends(
            context,
            local_source,
            summary,
            requests,
            remote_destinations,
            policy,
            row_offset=output_span[0],
        )
        exchange_result = await exchange_resolved_ghost_slices(
            context,
            comm,
            ir_context,
            local_source,
            local_owned_intervals=(output_span,)
            if output_span[0] < output_span[1]
            else (),
            local_empty_span=output_span,
            sends=sends,
            expected_sources=remote_sources,
            candidate_destinations=remote_destinations,
            collective_id=collective_id,
        )

        def cleanup() -> None:
            exchange_result.ghost_source.clear()
            local_source.clear()

        return RollingInput(
            _range_context_chunks(
                context,
                exchange_result,
                output_span,
                policy,
            ),
            release_sources=(
                exchange_result.local_source,
                exchange_result.ghost_source,
            ),
            partitioning=partitioning,
            cleanup=cleanup,
        )
    except BaseException:
        local_source.clear()
        raise


async def prepare_rolling_input(
    context: Context,
    comm: Communicator,
    ir: Rolling | FixedSizeRolling,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    policy: RollingPolicy[Any],
    metadata_in: ChannelMetadata,
    *,
    collective_id: int,
) -> RollingInput[Any]:
    """Prepare rank-agnostic input chunks for a rolling operation."""
    if comm.nranks == 1 or metadata_in.duplicated:
        return RollingInput(_local_staged_chunks(context, ch_in, policy))
    if isinstance(ir, Rolling):
        assert isinstance(policy, RangeOverlapPolicy)
        return await prepare_range_multirank_input(
            context,
            comm,
            ir_context,
            ch_in,
            policy,
            collective_id=collective_id,
            derive_partitioning=not _partitioning_has_inter_rank(
                metadata_in.partitioning
            ),
        )
    assert isinstance(policy, RowCountOverlapPolicy)
    return await prepare_fixed_size_multirank_input(
        context,
        comm,
        ir_context,
        ch_in,
        policy,
        collective_id=collective_id,
    )


async def prepare_rolling_manager(
    context: Context,
    comm: Communicator,
    ir: Rolling | FixedSizeRolling,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    *,
    collective_id: int,
) -> RollingManager[Any]:
    """Prepare the rolling policy and input with one cleanup owner."""
    if isinstance(ir, Rolling):
        policy: RollingPolicy[Any] = RangeOverlapPolicy.from_ir(
            ir, context.br().stream_pool.get_stream()
        )
    else:
        policy = RowCountOverlapPolicy(ir.preceding_overlap, ir.following_overlap)

    try:
        rolling_input = await prepare_rolling_input(
            context,
            comm,
            ir,
            ir_context,
            ch_in,
            policy,
            metadata_in,
            collective_id=collective_id,
        )
    except BaseException:
        policy.close()
        raise
    return RollingManager(policy, rolling_input)


@define_actor()
async def rolling_actor(
    context: Context,
    comm: Communicator,
    ir: Rolling | FixedSizeRolling,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    *,
    collective_id: int,
) -> None:
    """Streaming actor for rolling operations."""
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as tracer:
        metadata_in = await recv_metadata(ch_in, context)
        manager = await prepare_rolling_manager(
            context,
            comm,
            ir,
            ir_context,
            ch_in,
            metadata_in,
            collective_id=collective_id,
        )
        async with manager:
            input_partitioning = _merge_learned_inter_rank(
                metadata_in.partitioning,
                manager.rolling_input.partitioning,
            )
            partitioning = maybe_remap_partitioning(
                ir, input_partitioning, context=context
            )
            await send_metadata(
                ch_out,
                context,
                ChannelMetadata(
                    local_count=metadata_in.local_count,
                    partitioning=partitioning,
                    duplicated=metadata_in.duplicated,
                ),
            )
            if tracer is not None and metadata_in.duplicated:
                tracer.set_duplicated()

            await execute_rolling_policy(
                context,
                ir,
                ir_context,
                ch_out,
                manager.rolling_input,
                manager.policy,
                tracer,
            )


def generate_rolling_sub_network(
    ir: Rolling | FixedSizeRolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for rolling operations."""
    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    (collective_id,) = rec.state["collective_id_map"][ir]
    actors[ir] = [
        rolling_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            collective_id=collective_id,
        )
    ]
    return actors, channels


@generate_ir_sub_network.register(FixedSizeRolling)
def _(
    ir: FixedSizeRolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for fixed-size rolling expressions."""
    return generate_rolling_sub_network(ir, rec)


@generate_ir_sub_network.register(Rolling)
def _(
    ir: Rolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for a Rolling operation."""
    if len(ir.keys) > 0 or ir.zlice is not None:
        # Bypass this Rolling registration and use the generic IR actor.
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    return generate_rolling_sub_network(ir, rec)
