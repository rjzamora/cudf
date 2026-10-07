# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import ChannelMetadata, OrderScheme
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
    ResolvedRowExchangePlan,
    ResolvedRowSend,
    RowExchange,
    RowExchangeKind,
    RowExchangePlan,
    exchange_resolved_slices,
    extract_region,
    extract_region_from_index,
    gather_row_counts,
)
from cudf_polars.streaming.actor_graph.dispatch import generate_ir_sub_network
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    _evaluate_chunk_sync,
    empty_table_chunk,
    maybe_remap_partitioning,
    names_to_indices,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.rolling import FixedSizeRolling
from cudf_polars.streaming.utils import _fallback_inform
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
        BufferedChunkIndex,
        RowExchangeResult,
        RowRange,
    )
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


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


class StreamingRollingPolicy(RollingPolicy[BufferedChunkT], Protocol[BufferedChunkT]):
    """Protocol for single-stream local rolling evaluation."""

    async def recv_chunk(
        self,
        context: Context,
        ch_in: Channel[TableChunk],
        *,
        row_offset: int,
    ) -> tuple[BufferedChunkT | None, int]:
        """Receive and prepare one input chunk."""
        ...

    async def fill_future(
        self,
        context: Context,
        ch_in: Channel[TableChunk],
        current: BufferedChunkT,
        future: list[BufferedChunkT],
        row_offset: int,
    ) -> tuple[bool, int]:
        """Read leading chunks needed to evaluate current."""
        ...


@dataclass
class RangeOverlapPolicy(StreamingRollingPolicy[RangeBufferedChunk]):
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
        buffered = await self.prepare_table_chunk(
            context,
            msg.sequence_number,
            TableChunk.from_message(msg, br=context.br()),
            row_offset=row_offset,
        )
        self.observe(buffered)
        return buffered

    async def recv_chunk(
        self,
        context: Context,
        ch_in: Channel[TableChunk],
        *,
        row_offset: int,
    ) -> tuple[RangeBufferedChunk | None, int]:
        """Receive and prepare one input chunk."""
        if (msg := await ch_in.recv(context)) is None:
            return None, row_offset
        chunk = await self.prepare_chunk(
            context,
            msg,
            row_offset=row_offset,
        )
        return chunk, chunk.row_stop

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

    async def fill_future(
        self,
        context: Context,
        ch_in: Channel[TableChunk],
        current: RangeBufferedChunk,
        future: list[RangeBufferedChunk],
        row_offset: int,
    ) -> tuple[bool, int]:
        """Read leading chunks until current has a complete range window."""
        while not self.has_complete_future(
            latest_nonempty_chunk(current, future), current, context
        ):
            chunk, row_offset = await self.recv_chunk(
                context, ch_in, row_offset=row_offset
            )
            if chunk is None:
                return True, row_offset
            future.append(chunk)
        return False, row_offset

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
        chunk_index: BufferedChunkIndex | None = None,
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
            chunk_index=chunk_index,
        )


@dataclass
class RowCountOverlapPolicy(StreamingRollingPolicy[BufferedChunk]):
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

    async def recv_chunk(
        self,
        context: Context,
        ch_in: Channel[TableChunk],
        *,
        row_offset: int,
    ) -> tuple[BufferedChunk | None, int]:
        """Receive and prepare one fixed-size rolling input chunk."""
        if (msg := await ch_in.recv(context)) is None:
            return None, row_offset
        chunk = await self.prepare_chunk(context, msg, row_offset=row_offset)
        return chunk, chunk.row_stop

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

    async def fill_future(
        self,
        context: Context,
        ch_in: Channel[TableChunk],
        current: BufferedChunk,
        future: list[BufferedChunk],
        row_offset: int,
    ) -> tuple[bool, int]:
        """Read leading chunks needed for fixed-size rolling over current."""
        while not self.has_complete_future(
            latest_nonempty_chunk(current, future),
            current,
            context,
        ):
            chunk, row_offset = await self.recv_chunk(
                context, ch_in, row_offset=row_offset
            )
            if chunk is None:
                return True, row_offset
            future.append(chunk)
        return False, row_offset

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
    """Host-side value bounds for one non-empty range-rolling chunk."""

    row_start: int
    row_stop: int
    first: Any
    last: Any


@dataclass(frozen=True)
class RangeInputSummary:
    """Local rows and value-window request bounds for one rank."""

    local_rows: int
    has_rows: bool
    lower_bound_column: plc.Column | None
    upper_bound_column: plc.Column | None
    chunk_bounds: tuple[RangeChunkBounds, ...]


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
    # Keep the parent table alive for the device columns above.
    table: plc.Table


@dataclass(frozen=True)
class RangeExchangeRouting:
    """Range request metadata and sparse-exchange rendezvous shape."""

    requests: RangeRequests
    output_span: RowRange
    remote_sources: tuple[int, ...]
    remote_destinations: tuple[int, ...]


@dataclass
class RangeOverlapPlanner:
    """Plan and resolve multi-rank range-overlap exchange."""

    context: Context
    comm: Communicator
    ir_context: IRExecutionContext
    policy: RangeOverlapPolicy
    collective_id: int

    async def plan(self, summary: RangeInputSummary) -> RangeExchangeRouting:
        """Gather compact range requests and derive sparse rendezvous ranks."""
        requests = await _gather_range_requests(
            self.context,
            self.comm,
            self.ir_context,
            summary,
            self.policy,
            collective_id=self.collective_id,
        )
        row_starts = _row_starts(requests.row_counts)
        output_span = (
            row_starts[self.comm.rank],
            row_starts[self.comm.rank + 1],
        )
        return RangeExchangeRouting(
            requests,
            output_span,
            _range_remote_sources(self.comm, requests),
            _range_remote_destinations(self.comm, requests),
        )

    async def resolve_sends(
        self,
        source: BufferedChunkSource,
        bounds: Sequence[RangeChunkBounds],
        routing: RangeExchangeRouting,
    ) -> list[ResolvedRowSend]:
        """Resolve remote range requests to concrete local row slices."""
        return await _range_request_sends(
            self.context,
            source,
            bounds,
            routing.requests,
            self.policy,
            routing.remote_destinations,
            ir_context=self.ir_context,
            row_offset=routing.output_span[0],
            br=self.context.br(),
        )


@dataclass(frozen=True)
class StagedChunk(Generic[BufferedChunkT]):
    """A chunk staged for evaluation, with output-ownership state."""

    chunk: BufferedChunkT
    owned: bool


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
        insertion_point: int = (
            find(  # type: ignore[assignment]
                plc.Table([chunk.overlap.index_column]),
                plc.Table([needle]),
                [plc.types.Order.ASCENDING],
                [plc.types.NullOrder.AFTER],
                stream=stream,
                mr=br.device_mr,
            )
            .to_scalar(stream=stream)
            .to_py(stream=stream)
        )
        if insertion_point < chunk.num_rows:
            return chunk.row_start + insertion_point
    # Needle is later than all the chunks we know about.
    return chunks[-1].row_stop


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
    chunk_index: BufferedChunkIndex | None = None,
) -> TableChunk:
    """Evaluate a ghost-expanded region and slice back to cursor rows."""
    if chunk_index is None:
        ghosted_chunk = await extract_region(
            context,
            chunks,
            ghost_start,
            ghost_stop,
            ir_context=ir_context,
        )
    else:
        ghosted_chunk = await extract_region_from_index(
            context,
            chunk_index,
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
    return RowExchangePlan.from_spans(spans, spans, ghost_requests)


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
    chunk_bounds: list[RangeChunkBounds] = []
    while (msg := await ch_in.recv(context)) is not None:
        buffered = await policy.prepare_chunk(context, msg, row_offset=row_offset)
        source.insert(buffered)
        row_offset = buffered.row_stop
        if buffered.num_rows == 0:
            continue
        assert buffered.overlap.first is not None
        assert buffered.overlap.last is not None
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
        upper_bound_column = buffered.overlap.upper_bound_column
    return RangeInputSummary(
        local_rows=row_offset,
        has_rows=has_rows,
        lower_bound_column=lower_bound_column,
        upper_bound_column=upper_bound_column,
        chunk_bounds=tuple(chunk_bounds),
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
    first_value = _single_value_column(
        summary.chunk_bounds[0].first if summary.has_rows else None,
        policy.index_dtype,
        stream,
    )
    last_value = _single_value_column(
        summary.chunk_bounds[-1].last if summary.has_rows else None,
        policy.index_dtype,
        stream,
    )
    return TableChunk.from_pylibcudf_table(
        plc.Table(
            [
                row_count,
                has_request,
                lower_bound,
                upper_bound,
                first_value,
                last_value,
            ]
        ),
        stream,
        exclusive_view=True,
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
    lower_values = tuple(
        _request_host_value(lower_column, rank, stream=stream, br=context.br())
        if has_request[rank]
        else None
        for rank in range(comm.nranks)
    )
    upper_values = tuple(
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


def _slice_request_column(
    column: plc.Column,
    rank: int,
    stream: Stream,
) -> plc.Column:
    """Return one rank's request value as a single-row column."""
    return plc.copying.slice(column, [rank, rank + 1], stream=stream)[0]


def _request_host_value(
    column: plc.Column,
    rank: int,
    *,
    stream: Stream,
    br: BufferResource,
) -> Any:
    """Return one all-gathered request value on the host."""
    return _host_ordering_value(
        _slice_request_column(column, rank, stream),
        stream=stream,
        br=br,
    )


def _candidate_range_span(
    bounds: Sequence[RangeChunkBounds],
    lower: Any,
    upper: Any,
    *,
    row_offset: int,
) -> RowRange | None:
    """Return a conservative global row span that may overlap a value range."""
    start: int | None = None
    stop: int | None = None
    for chunk in bounds:
        chunk_start = chunk.row_start + row_offset
        chunk_stop = chunk.row_stop + row_offset
        if start is None:
            if chunk.last < lower:
                continue
            start = chunk_start
        stop = chunk_stop
        if chunk.first > upper:
            break
    if start is None or stop is None:
        return None
    return start, stop


def _rank_may_satisfy_range_request(
    requests: RangeRequests,
    *,
    source: int,
    destination: int,
) -> bool:
    """Return whether a source rank may contain rows requested by a destination."""
    if not requests.has_request[source] or not requests.has_request[destination]:
        return False
    source_first = requests.first_values[source]
    source_last = requests.last_values[source]
    request_lower = requests.lower_values[destination]
    request_upper = requests.upper_values[destination]
    if (
        source_first is None
        or source_last is None
        or request_lower is None
        or request_upper is None
    ):
        return False
    return not (source_last < request_lower or source_first > request_upper)


def _range_remote_sources(
    comm: Communicator,
    requests: RangeRequests,
) -> tuple[int, ...]:
    """Return remote ranks that may send range ghosts to this rank."""
    return tuple(
        source
        for source in range(comm.nranks)
        if source != comm.rank
        and _rank_may_satisfy_range_request(
            requests,
            source=source,
            destination=comm.rank,
        )
    )


def _range_remote_destinations(
    comm: Communicator,
    requests: RangeRequests,
) -> tuple[int, ...]:
    """Return remote ranks that may need range ghosts from this rank."""
    return tuple(
        destination
        for destination in range(comm.nranks)
        if destination != comm.rank
        and _rank_may_satisfy_range_request(
            requests,
            source=comm.rank,
            destination=destination,
        )
    )


async def _prepare_range_chunks_from_span(
    context: Context,
    source: BufferedChunkSource,
    policy: RangeOverlapPolicy,
    span: RowRange,
    *,
    ir_context: IRExecutionContext,
) -> list[RangeBufferedChunk]:
    """Prepare source chunks from one global row span for range search."""
    chunks: list[RangeBufferedChunk] = []
    async for chunk in source.iter_region_chunks(
        *span,
        ir_context=ir_context,
    ):
        prepared = await policy.prepare_table_chunk(
            context,
            chunk.sequence_number,
            chunk.chunk,
            row_offset=chunk.row_start,
        )
        chunks.append(prepared)
    return chunks


async def _range_request_sends(
    context: Context,
    source: BufferedChunkSource,
    bounds: Sequence[RangeChunkBounds],
    requests: RangeRequests,
    policy: RangeOverlapPolicy,
    destinations: Sequence[int],
    *,
    ir_context: IRExecutionContext,
    row_offset: int,
    br: BufferResource,
) -> list[ResolvedRowSend]:
    """Resolve all remote range requests against this rank's local rows."""
    if not bounds:
        return []
    sends: list[ResolvedRowSend] = []
    for destination in destinations:
        lower = _slice_request_column(
            requests.lower_column,
            destination,
            requests.stream,
        )
        upper = _slice_request_column(
            requests.upper_column,
            destination,
            requests.stream,
        )
        candidate_span = _candidate_range_span(
            bounds,
            _request_host_value(
                requests.lower_column,
                destination,
                stream=requests.stream,
                br=br,
            ),
            _request_host_value(
                requests.upper_column,
                destination,
                stream=requests.stream,
                br=br,
            ),
            row_offset=row_offset,
        )
        if candidate_span is None:
            continue
        range_chunks = await _prepare_range_chunks_from_span(
            context,
            source,
            policy,
            candidate_span,
            ir_context=ir_context,
        )
        if not range_chunks:
            continue
        start = global_insertion_row(
            range_chunks,
            lower,
            policy.find_start,
            needle_stream=requests.stream,
            br=br,
        )
        stop = global_insertion_row(
            range_chunks,
            upper,
            policy.find_end,
            needle_stream=requests.stream,
            br=br,
        )
        if start < stop:
            sends.append(
                ResolvedRowSend(destination, start, stop, RowExchangeKind.GHOST)
            )
    return sends


def _has_range_input_ordering(
    metadata: ChannelMetadata,
    policy: RangeOverlapPolicy,
) -> bool:
    """Return whether metadata proves range rolling input ordering."""
    partitioning = metadata.partitioning
    if partitioning is None:
        return False
    scheme = partitioning.inter_rank
    if not isinstance(scheme, OrderScheme) or partitioning.local != "inherit":
        return False
    for ordering in scheme.orderings:
        if not ordering.keys or not ordering.locally_ordered:
            continue
        key = ordering.keys[0]
        if key.column_index == policy.index and key.order == plc.types.Order.ASCENDING:
            return True
    return False


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


async def _range_context_chunks(
    context: Context,
    result: RowExchangeResult,
    output_span: RowRange,
    policy: RangeOverlapPolicy,
    *,
    ir_context: IRExecutionContext,
) -> AsyncIterator[StagedChunk[RangeBufferedChunk]]:
    """Yield ghost and owned range chunks in global row order."""
    if result.has_remote_owned():
        raise RuntimeError(
            "Range rolling does not expect RowExchange to move owned rows"
        )
    output_start, output_stop = output_span
    ghost_iter = result.iter_ghosts(ir_context=ir_context)
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
        ir_context=ir_context,
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


async def _evaluate_staged_rolling(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    chunks: AsyncIterator[StagedChunk[BufferedChunkT]],
    local_source: BufferedChunkSource,
    policy: RollingPolicy[BufferedChunkT],
    tracer: ActorTracer | None,
) -> None:
    """Evaluate owned staged chunks with any required ghost context."""
    history: list[BufferedChunkT] = []
    future: list[StagedChunk[BufferedChunkT]] = []
    input_exhausted = False
    output_sequence_number = 0
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
            await ch_out.send(context, Message(output_sequence_number, result))
            output_sequence_number += 1
        else:
            history = policy.evict_history(history, cursor, context)
            local_source.release_cached_before(
                policy.release_cached_before(history, cursor, context)
            )
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
            await ch_out.send(context, Message(output_sequence_number, result))
            output_sequence_number += 1
            history.append(cursor)

        if future:
            staged = future.pop(0)
        else:
            staged = await _next_staged_chunk(chunks)

    await ch_out.drain(context)


async def _evaluate_range_exchange_result(
    context: Context,
    ir: Rolling,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    exchange_result: RowExchangeResult,
    output_span: RowRange,
    policy: RangeOverlapPolicy,
    tracer: ActorTracer | None,
) -> None:
    """Evaluate range rolling over a replayed owned stream plus ghosts."""
    chunks = _range_context_chunks(
        context,
        exchange_result,
        output_span,
        policy,
        ir_context=ir_context,
    )
    await _evaluate_staged_rolling(
        context,
        ir,
        ir_context,
        ch_out,
        chunks,
        exchange_result.local_source,
        policy,
        tracer,
    )


async def execute_range_multirank_policy(
    context: Context,
    comm: Communicator,
    ir: Rolling,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    policy: RangeOverlapPolicy,
    tracer: ActorTracer | None,
    *,
    collective_id: int,
) -> None:
    """
    Evaluate range rolling with sparse inter-rank ghost slices.

    Each rank publishes one compact value request covering the rows it owns.
    Source ranks resolve those value requests to exact local row slices before
    the row transport moves data.

    The range-request all-gather completes before the row-slice exchange starts,
    so they intentionally share one collective ID. This matches the sequential
    reuse pattern used by fixed-size rolling and non-scalar ``over``.
    """
    local_source = BufferedChunkSource(context)
    try:
        summary = await _recv_all_range_chunks(context, ch_in, local_source, policy)
        planner = RangeOverlapPlanner(
            context,
            comm,
            ir_context,
            policy,
            collective_id,
        )
        routing = await planner.plan(summary)
        output_span = routing.output_span
        local_source.set_row_start_offset(output_span[0])

        sends = await planner.resolve_sends(
            local_source,
            summary.chunk_bounds,
            routing,
        )
        exchange_result = await exchange_resolved_slices(
            context,
            comm,
            ir_context,
            local_source,
            local_owned_intervals=(output_span,)
            if output_span[0] < output_span[1]
            else (),
            local_empty_span=output_span,
            plan=ResolvedRowExchangePlan.from_sends(
                sends,
                expected_sources=routing.remote_sources,
                candidate_destinations=routing.remote_destinations,
            ),
            collective_id=collective_id,
        )
        try:
            await _evaluate_range_exchange_result(
                context,
                ir,
                ir_context,
                ch_out,
                exchange_result,
                output_span,
                policy,
                tracer,
            )
        finally:
            exchange_result.clear_received()
    finally:
        local_source.clear()
        policy.close()


async def execute_local_rolling_policy(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    policy: StreamingRollingPolicy[Any],
    tracer: ActorTracer | None,
) -> None:
    """Evaluate ordered chunks with contiguous ghost regions."""
    history: list[Any] = []
    future: list[Any] = []
    input_exhausted = False
    try:
        cursor, row_offset = await policy.recv_chunk(context, ch_in, row_offset=0)
        if cursor is None:
            await ch_out.drain(context)
            return

        while cursor is not None:
            policy.observe(cursor)
            policy.validate_cursor(context, cursor)
            if cursor.num_rows == 0:
                result = await evaluate_available_chunk(
                    context,
                    cursor.chunk,
                    ir,
                    ir_context=ir_context,
                )
            else:
                history = policy.evict_history(history, cursor, context)
                if not input_exhausted:
                    input_exhausted, row_offset = await policy.fill_future(
                        context,
                        ch_in,
                        cursor,
                        future,
                        row_offset,
                    )
                result = await policy.evaluate_cursor(
                    context,
                    ir,
                    ir_context,
                    cursor,
                    history=history,
                    future=future,
                )
                history.append(cursor)

            if tracer is not None:
                tracer.add_chunk(chunk=result)
            await ch_out.send(context, Message(cursor.sequence_number, result))

            if future:
                cursor, *future = future
            else:
                cursor, row_offset = await policy.recv_chunk(
                    context, ch_in, row_offset=row_offset
                )

        await ch_out.drain(context)
    finally:
        policy.close()


async def _fixed_size_context_chunks(
    result: RowExchangeResult,
    output_span: RowRange,
    *,
    ir_context: IRExecutionContext,
) -> AsyncIterator[StagedChunk[BufferedChunk]]:
    """Yield ghost and owned chunks in global row order for fixed-size rolling."""
    if result.has_remote_owned():
        raise RuntimeError(
            "Fixed-size rolling does not expect RowExchange to move owned rows"
        )
    output_start, output_stop = output_span
    ghosts = [chunk async for chunk in result.iter_ghosts(ir_context=ir_context)]
    left_ghosts = [chunk for chunk in ghosts if chunk.row_stop <= output_start]
    right_ghosts = [chunk for chunk in ghosts if chunk.row_start >= output_stop]
    if len(left_ghosts) + len(right_ghosts) != len(ghosts):
        raise RuntimeError(
            "Fixed-size rolling received a ghost chunk that overlaps owned rows"
        )
    for chunk in sorted(left_ghosts, key=lambda chunk: chunk.row_start):
        yield StagedChunk(chunk, owned=False)
    async for chunk in result.iter_local_owned(
        ir_context=ir_context,
        include_empty_chunks=True,
    ):
        yield StagedChunk(chunk, owned=True)
    for chunk in sorted(right_ghosts, key=lambda chunk: chunk.row_start):
        yield StagedChunk(chunk, owned=False)


async def _evaluate_fixed_size_exchange_result(
    context: Context,
    ir: FixedSizeRolling,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    exchange_result: RowExchangeResult,
    output_span: RowRange,
    policy: RowCountOverlapPolicy,
    tracer: ActorTracer | None,
) -> None:
    """Evaluate fixed-size rolling over a replayed owned stream plus ghosts."""
    chunks = _fixed_size_context_chunks(
        exchange_result,
        output_span,
        ir_context=ir_context,
    )
    await _evaluate_staged_rolling(
        context,
        ir,
        ir_context,
        ch_out,
        chunks,
        exchange_result.local_source,
        policy,
        tracer,
    )


async def execute_fixed_size_multirank_policy(
    context: Context,
    comm: Communicator,
    ir: FixedSizeRolling,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    policy: RowCountOverlapPolicy,
    tracer: ActorTracer | None,
    *,
    collective_id: int,
) -> None:
    """
    Evaluate fixed-size rolling with sparse inter-rank ghost slices.

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
        try:
            await _evaluate_fixed_size_exchange_result(
                context,
                ir,
                ir_context,
                ch_out,
                exchange_result,
                plan.output_span(comm.rank),
                policy,
                tracer,
            )
        finally:
            exchange_result.clear_received()
    finally:
        local_source.clear()


async def execute_range_allgather_fallback(
    context: Context,
    comm: Communicator,
    ir: Rolling,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    policy: RangeOverlapPolicy,
    tracer: ActorTracer | None,
    *,
    collective_id: int,
    config_options: ConfigOptions[StreamingExecutor],
) -> None:
    """Evaluate multi-rank range rolling with the existing all-gather fallback."""
    del policy
    _fallback_inform(
        "Rolling does not support multi-rank inputs. "
        "Falling back to all-gather evaluation.",
        config_options,
    )
    await send_metadata(
        ch_out,
        context,
        ChannelMetadata(local_count=1, partitioning=None, duplicated=True),
    )
    if tracer is not None:
        tracer.set_duplicated()

    stream = ir_context.get_cuda_stream()
    ag = AllGatherManager(context, comm, collective_id)
    with ag.inserting() as inserter:
        while (msg := await ch_in.recv(context)) is not None:
            chunk = TableChunk.from_message(msg, context.br())
            await inserter.insert(msg.sequence_number, chunk)
    table = await ag.extract_concatenated(stream, ordered=True, ir_context=ir_context)
    if table.num_columns() == 0 and len(ir.children[0].schema) > 0:
        chunk = empty_table_chunk(ir.children[0], context, stream)
    else:
        chunk = TableChunk.from_pylibcudf_table(
            table, stream, exclusive_view=True, br=context.br()
        )
    result = await evaluate_available_chunk(
        context,
        chunk,
        ir,
        ir_context=ir_context,
    )
    if tracer is not None:
        tracer.add_chunk(chunk=result)
    await ch_out.send(context, Message(0, result))
    await ch_out.drain(context)


@define_actor()
async def overlap_actor(
    context: Context,
    comm: Communicator,
    ir: Rolling | FixedSizeRolling,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    *,
    collective_id: int,
    config_options: ConfigOptions[StreamingExecutor],
) -> None:
    """Streaming actor for rolling operations requiring chunk overlap."""
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as tracer:
        metadata_in = await recv_metadata(ch_in, context)
        policy: StreamingRollingPolicy[Any]
        has_range_ordering = False

        if isinstance(ir, Rolling):
            policy = RangeOverlapPolicy.from_ir(
                ir, context.br().stream_pool.get_stream()
            )
            if comm.nranks != 1 and not metadata_in.duplicated:
                has_range_ordering = _has_range_input_ordering(metadata_in, policy)
                if not has_range_ordering:
                    await execute_range_allgather_fallback(
                        context,
                        comm,
                        ir,
                        ir_context,
                        ch_out,
                        ch_in,
                        policy,
                        tracer,
                        collective_id=collective_id,
                        config_options=config_options,
                    )
                    return
        else:
            policy = RowCountOverlapPolicy(ir.preceding_overlap, ir.following_overlap)

        partitioning = maybe_remap_partitioning(
            ir, metadata_in.partitioning, context=context
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

        if (
            isinstance(ir, FixedSizeRolling)
            and comm.nranks != 1
            and not metadata_in.duplicated
        ):
            assert isinstance(policy, RowCountOverlapPolicy)
            await execute_fixed_size_multirank_policy(
                context,
                comm,
                ir,
                ir_context,
                ch_out,
                ch_in,
                policy,
                tracer,
                collective_id=collective_id,
            )
            return
        if isinstance(ir, Rolling) and comm.nranks != 1 and not metadata_in.duplicated:
            assert isinstance(policy, RangeOverlapPolicy)
            assert has_range_ordering
            await execute_range_multirank_policy(
                context,
                comm,
                ir,
                ir_context,
                ch_out,
                ch_in,
                policy,
                tracer,
                collective_id=collective_id,
            )
            return
        await execute_local_rolling_policy(
            context,
            ir,
            ir_context,
            ch_out,
            ch_in,
            policy,
            tracer,
        )


def generate_overlap_sub_network(
    ir: Rolling | FixedSizeRolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for rolling operations requiring overlap."""
    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    (collective_id,) = rec.state["collective_id_map"][ir]
    actors[ir] = [
        overlap_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            collective_id=collective_id,
            config_options=rec.state["config_options"],
        )
    ]
    return actors, channels


@generate_ir_sub_network.register(FixedSizeRolling)
def _(
    ir: FixedSizeRolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for fixed-size rolling expressions."""
    return generate_overlap_sub_network(ir, rec)


@generate_ir_sub_network.register(Rolling)
def _(
    ir: Rolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for a Rolling operation."""
    if len(ir.keys) > 0 or ir.zlice is not None:
        # Bypass this Rolling registration and use the generic IR actor.
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    return generate_overlap_sub_network(ir, rec)
