# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import ChannelMetadata
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
    ResolvedRowSend,
    RowExchange,
    RowExchangePlan,
    ValueRangeRouter,
    exchange_payload_chunks,
    exchange_resolved_slices,
    extract_region,
    gather_row_counts,
    value_ranges_overlap,
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
    from collections.abc import Callable, Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.collectives.overlap import (
        RowRange,
        ValueRange,
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
    lower_bound: Any | None
    upper_bound: Any | None


@dataclass
class RangeBufferedChunk(BufferedChunk):
    """A buffered chunk with range-window overlap metadata."""

    overlap: RangeOverlap


@dataclass(frozen=True)
class RangeRankStats:
    """Per-rank row counts and index bounds for range-window exchange."""

    row_count: int
    first: Any | None
    last: Any | None
    lower_bound: Any | None
    upper_bound: Any | None


BufferedChunkT = TypeVar("BufferedChunkT", bound=BufferedChunk)
_INT64_DTYPE = DataType(pl.Int64())


class OverlapPolicy(Protocol[BufferedChunkT]):
    """Protocol for staging overlap around a cursor chunk."""

    def observe(self, chunk: BufferedChunkT) -> None:
        """Record resources that must outlive chunk processing."""
        ...

    def close(self) -> None:
        """Finalize any policy-owned resources."""
        ...

    def validate_cursor(self, context: Context, cursor: BufferedChunkT) -> None:
        """Validate ordering assumptions before evaluating cursor."""
        ...

    async def recv_chunk(
        self,
        context: Context,
        ch_in: Channel[TableChunk],
        *,
        row_offset: int,
    ) -> tuple[BufferedChunkT | None, int]:
        """Receive and prepare one input chunk."""
        ...

    def evict_history(
        self,
        history: list[BufferedChunkT],
        cursor: BufferedChunkT,
        context: Context,
    ) -> list[BufferedChunkT]:
        """Drop chunks that cannot contribute to the cursor chunk."""
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


@dataclass
class RangeOverlapPolicy(OverlapPolicy[RangeBufferedChunk]):
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
        if nrows == 0:
            lower_bound = upper_bound = index_column
            first = last = lower_bound_value = upper_bound_value = None
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
            lower_bound_value = _host_ordering_value(
                lower_bound, stream=chunk.stream, br=context.br()
            )
            upper_bound_value = _host_ordering_value(
                upper_bound, stream=chunk.stream, br=context.br()
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
                lower_bound_value,
                upper_bound_value,
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
        chunk = await self.prepare_chunk(context, msg, row_offset=row_offset)
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
        del context
        return [
            chunk
            for chunk in history
            if chunk_may_contain_insertion_point(
                chunk,
                cursor.overlap.lower_bound,
                inclusive=self.start_inclusive,
            )
        ]

    def has_complete_future(
        self, chunk: RangeBufferedChunk, current: RangeBufferedChunk
    ) -> bool:
        """Return whether chunk contains current's upper insertion point."""
        return chunk_may_contain_insertion_point(
            chunk,
            current.overlap.upper_bound,
            inclusive=self.end_inclusive,
        )

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
            latest_nonempty_chunk(current, future), current
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
        ghost_start = global_insertion_row(
            chunks,
            cursor.overlap.lower_bound_column,
            cursor.overlap.lower_bound,
            self.find_start,
            inclusive=self.start_inclusive,
            needle_stream=cursor.chunk.stream,
            br=context.br(),
        )
        ghost_stop = global_insertion_row(
            chunks,
            cursor.overlap.upper_bound_column,
            cursor.overlap.upper_bound,
            self.find_end,
            inclusive=self.end_inclusive,
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
class RowCountOverlapPolicy(OverlapPolicy[BufferedChunk]):
    """Overlap policy for fixed-size rolling expressions."""

    preceding: int
    following: int

    def row_slice_exchange_plan(self, row_counts: Sequence[int]) -> RowExchangePlan:
        """Return the inter-rank row-count slice exchange plan."""
        return RowExchangePlan.from_row_counts(
            row_counts,
            preceding=self.preceding,
            following=self.following,
        )

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

    async def fill_future(
        self,
        context: Context,
        ch_in: Channel[TableChunk],
        current: BufferedChunk,
        future: list[BufferedChunk],
        row_offset: int,
    ) -> tuple[bool, int]:
        """Read leading chunks needed for fixed-size rolling over current."""
        required_stop = current.row_stop + self.following
        while latest_nonempty_chunk(current, future).row_stop < required_stop:
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


def chunk_may_contain_insertion_point(
    chunk: RangeBufferedChunk,
    needle_value: Any | None,
    *,
    inclusive: bool,
) -> bool:
    """Return whether chunk may contain a lower/upper-bound insertion point."""
    if chunk.num_rows == 0:
        return False
    assert needle_value is not None
    assert chunk.overlap.last is not None
    if inclusive:
        return chunk.overlap.last >= needle_value
    return chunk.overlap.last > needle_value


def global_insertion_row(
    chunks: Sequence[RangeBufferedChunk],
    needle: plc.Column,
    needle_value: Any | None,
    find: Callable[..., plc.Column],
    *,
    inclusive: bool,
    needle_stream: Stream,
    br: BufferResource,
) -> int:
    """Return the globally indexed insertion row of a needle in some chunks."""
    assert len(chunks) > 0
    for chunk in chunks:
        if not chunk_may_contain_insertion_point(
            chunk, needle_value, inclusive=inclusive
        ):
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
        chunk.data_alloc_size(), net_memory_delta=-chunk.data_alloc_size()
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
    return TableChunk.from_pylibcudf_table(
        table.copy(result.stream, context.br().device_mr),
        result.stream,
        exclusive_view=True,
        br=context.br(),
    )


async def _recv_all_fixed_size_chunks(
    context: Context,
    ch_in: Channel[TableChunk],
    policy: RowCountOverlapPolicy,
) -> tuple[list[BufferedChunk], int]:
    """Drain a fixed-size rolling input channel into available chunks."""
    chunks: list[BufferedChunk] = []
    row_offset = 0
    while True:
        chunk, row_offset = await policy.recv_chunk(
            context, ch_in, row_offset=row_offset
        )
        if chunk is None:
            return chunks, row_offset
        chunks.append(chunk)


async def _recv_all_range_chunks(
    context: Context,
    ch_in: Channel[TableChunk],
    policy: RangeOverlapPolicy,
) -> tuple[list[RangeBufferedChunk], int]:
    """Drain a range rolling input channel into available chunks."""
    chunks: list[RangeBufferedChunk] = []
    row_offset = 0
    while True:
        chunk, row_offset = await policy.recv_chunk(
            context, ch_in, row_offset=row_offset
        )
        if chunk is None:
            return chunks, row_offset
        chunks.append(chunk)


def _nonempty_range_chunks(
    chunks: Sequence[RangeBufferedChunk],
) -> list[RangeBufferedChunk]:
    """Return non-empty buffered range chunks."""
    return [chunk for chunk in chunks if chunk.num_rows > 0]


def _local_range_stats(
    chunks: Sequence[RangeBufferedChunk],
    local_rows: int,
) -> RangeRankStats:
    """Return range stats for one rank."""
    nonempty = _nonempty_range_chunks(chunks)
    if not nonempty:
        return RangeRankStats(local_rows, None, None, None, None)
    first = nonempty[0].overlap
    last = nonempty[-1].overlap
    return RangeRankStats(
        local_rows,
        first.first,
        last.last,
        first.lower_bound,
        last.upper_bound,
    )


def _range_stats_chunk(
    context: Context,
    stats: RangeRankStats,
    stream: Stream,
) -> TableChunk:
    """Return one row of int64 range stats for all-gather."""
    dtype = plc.DataType(plc.TypeId.INT64)
    table = plc.Table(
        [
            plc.Column.from_scalar(
                plc.Scalar.from_py(value, dtype, stream=stream),
                1,
                stream=stream,
            )
            for value in (
                stats.row_count,
                stats.first,
                stats.last,
                stats.lower_bound,
                stats.upper_bound,
            )
        ]
    )
    return TableChunk.from_pylibcudf_table(
        table,
        stream,
        exclusive_view=True,
        br=context.br(),
    )


async def gather_range_stats(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    local_stats: RangeRankStats,
    *,
    collective_id: int,
) -> list[RangeRankStats]:
    """Collect range-window stats from every rank."""
    stream = context.br().stream_pool.get_stream()
    ag = AllGatherManager(context, comm, collective_id)
    with ag.inserting() as inserter:
        await inserter.insert(0, _range_stats_chunk(context, local_stats, stream))
    table = await ag.extract_concatenated(stream, ordered=True, ir_context=ir_context)
    stats = DataFrame.from_table(
        table,
        ["row_count", "first", "last", "lower_bound", "upper_bound"],
        [_INT64_DTYPE] * 5,
        stream,
    ).to_polars()
    if len(stats) != comm.nranks:
        raise RuntimeError(
            "Range-stat allgather returned an unexpected number of rows: "
            f"expected {comm.nranks}, got {len(stats)}"
        )
    return [
        RangeRankStats(
            row["row_count"],
            row["first"],
            row["last"],
            row["lower_bound"],
            row["upper_bound"],
        )
        for row in stats.iter_rows(named=True)
    ]


def _validate_global_range_ordering(stats: Sequence[RangeRankStats]) -> None:
    """Raise if rank endpoints prove the range index is not globally sorted."""
    previous_last = None
    for rank, stat in enumerate(stats):
        if stat.row_count == 0:
            continue
        if stat.first is None or stat.last is None:
            raise RuntimeError(f"Rank {rank} has rows but missing range endpoints")
        if previous_last is not None and stat.first < previous_last:
            raise RuntimeError(
                "Index column in rolling is not globally sorted across ranks"
            )
        previous_last = stat.last


def _source_spans_from_rank_stats(
    stats: Sequence[RangeRankStats],
) -> tuple[RowRange, ...]:
    """Return global row spans implied by per-rank stats."""
    offsets = [0]
    for stat in stats:
        offsets.append(offsets[-1] + stat.row_count)
    return tuple((offsets[i], offsets[i + 1]) for i in range(len(stats)))


def _rank_source_range(stats: RangeRankStats) -> ValueRange | None:
    """Return the inclusive source value range owned by one rank."""
    if stats.row_count == 0 or stats.first is None or stats.last is None:
        return None
    return stats.first, stats.last


def _range_request_bounds(chunk: RangeBufferedChunk) -> ValueRange | None:
    """Return an inclusive value range covering cursor rows and their windows."""
    if chunk.num_rows == 0:
        return None
    values = (
        chunk.overlap.first,
        chunk.overlap.last,
        chunk.overlap.lower_bound,
        chunk.overlap.upper_bound,
    )
    if any(value is None for value in values):
        return None
    value_range = [value for value in values if value is not None]
    return min(value_range), max(value_range)


def _rank_request_bounds(stats: RangeRankStats) -> ValueRange | None:
    """Return an inclusive value range that may be requested by one rank."""
    if stats.row_count == 0:
        return None
    values = (stats.first, stats.last, stats.lower_bound, stats.upper_bound)
    if any(value is None for value in values):
        return None
    value_range = [value for value in values if value is not None]
    return min(value_range), max(value_range)


def _range_router(rank_stats: Sequence[RangeRankStats]) -> ValueRangeRouter:
    """Return a value-range router for inter-rank range overlap."""
    return ValueRangeRouter(
        tuple(_rank_source_range(stats) for stats in rank_stats),
        tuple(_rank_request_bounds(stats) for stats in rank_stats),
    )


def _merge_value_ranges(ranges: Sequence[ValueRange]) -> list[ValueRange]:
    """Merge overlapping value ranges."""
    if not ranges:
        return []
    merged: list[ValueRange] = []
    for lower, upper in sorted(ranges):
        if not merged or lower > merged[-1][1]:
            merged.append((lower, upper))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
    return merged


def _merge_row_ranges(ranges: Sequence[RowRange]) -> list[RowRange]:
    """Merge overlapping or adjacent row ranges."""
    if not ranges:
        return []
    merged: list[RowRange] = []
    for start, stop in sorted(ranges):
        if start >= stop:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, stop))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
    return merged


def _range_requests_by_source(
    rank: int,
    rank_stats: Sequence[RangeRankStats],
    local_chunks: Sequence[RangeBufferedChunk],
    candidate_sources: Sequence[int],
) -> dict[int, list[ValueRange]]:
    """Return value-range requests this rank needs from remote source ranks."""
    requests: dict[int, list[ValueRange]] = {}
    for chunk in local_chunks:
        bounds = _range_request_bounds(chunk)
        if bounds is None:
            continue
        for source in candidate_sources:
            if value_ranges_overlap(_rank_source_range(rank_stats[source]), bounds):
                requests.setdefault(source, []).append(bounds)
    return {
        source: _merge_value_ranges(source_requests)
        for source, source_requests in requests.items()
    }


def _range_request_chunk(
    context: Context,
    requests: Sequence[ValueRange],
) -> TableChunk:
    """Return a table chunk containing value-range requests."""
    stream = context.br().stream_pool.get_stream()
    dtype = plc.DataType(plc.TypeId.INT64)
    lower = plc.Column.from_iterable_of_py(
        [request[0] for request in requests],
        dtype,
        stream=stream,
    )
    upper = plc.Column.from_iterable_of_py(
        [request[1] for request in requests],
        dtype,
        stream=stream,
    )
    return TableChunk.from_pylibcudf_table(
        plc.Table([lower, upper]),
        stream,
        exclusive_view=True,
        br=context.br(),
    )


def _range_requests_from_chunk(chunk: TableChunk) -> list[ValueRange]:
    """Read value-range requests from an internal request chunk."""
    requests = DataFrame.from_table(
        chunk.table_view(),
        ["lower", "upper"],
        [_INT64_DTYPE, _INT64_DTYPE],
        chunk.stream,
    ).to_polars()
    return list(requests.iter_rows())


def _host_value_to_index_column(
    value: Any,
    dtype: plc.DataType,
    stream: Stream,
    br: BufferResource,
) -> plc.Column:
    """Return a single-row index-typed column from a host ordering value."""
    if plc.traits.is_chrono(dtype):
        column = plc.Column.from_scalar(
            plc.Scalar.from_py(value, _chrono_storage_dtype(dtype), stream=stream),
            1,
            stream=stream,
        )
        return plc.unary.bit_cast(column, dtype, stream=stream, mr=br.device_mr)
    return plc.Column.from_scalar(
        plc.Scalar.from_py(value, dtype, stream=stream),
        1,
        stream=stream,
    )


def _search_range_value(
    chunk: RangeBufferedChunk,
    value: Any,
    find: Callable[..., plc.Column],
    dtype: plc.DataType,
    br: BufferResource,
) -> int:
    """Return the global insertion row for one host value in one chunk."""
    stream = chunk.chunk.stream
    needle = _host_value_to_index_column(value, dtype, stream, br)
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
    return chunk.row_start + insertion_point


def _resolve_range_requests(
    context: Context,
    policy: RangeOverlapPolicy,
    local_chunks: Sequence[RangeBufferedChunk],
    requests: Sequence[ValueRange],
) -> list[RowRange]:
    """Resolve value-range requests to local global row ranges."""
    intervals: list[RowRange] = []
    for lower, upper in requests:
        for chunk in local_chunks:
            if (
                chunk.num_rows == 0
                or chunk.overlap.first is None
                or chunk.overlap.last is None
                or chunk.overlap.first > upper
                or chunk.overlap.last < lower
            ):
                continue
            start = _search_range_value(
                chunk,
                lower,
                plc.search.lower_bound,
                policy.index_dtype,
                context.br(),
            )
            stop = _search_range_value(
                chunk,
                upper,
                plc.search.upper_bound,
                policy.index_dtype,
                context.br(),
            )
            if start < stop:
                intervals.append((start, stop))
    return _merge_row_ranges(intervals)


async def _prepare_exchanged_range_chunks(
    context: Context,
    policy: RangeOverlapPolicy,
    chunks: Sequence[BufferedChunk],
) -> list[RangeBufferedChunk]:
    """Attach range-overlap metadata to exchanged chunks."""
    result: list[RangeBufferedChunk] = []
    for chunk in chunks:
        range_chunk = await policy.prepare_table_chunk(
            context,
            chunk.sequence_number,
            chunk.chunk,
            row_offset=chunk.row_start,
        )
        policy.observe(range_chunk)
        result.append(range_chunk)
    return result


async def execute_rolling_policy(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    policy: OverlapPolicy[Any],
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
    """Evaluate fixed-size rolling with sparse inter-rank ghost slices."""
    local_chunks, local_rows = await _recv_all_fixed_size_chunks(
        context,
        ch_in,
        policy,
    )
    row_counts = await gather_row_counts(
        context,
        comm,
        ir_context,
        local_rows=local_rows,
        collective_id=collective_id,
    )
    plan = policy.row_slice_exchange_plan(row_counts)

    local_start, _ = plan.source_span(comm.rank)
    for chunk in local_chunks:
        chunk.row_start += local_start

    exchange_result = await RowExchange(
        context,
        comm,
        ir_context,
        plan,
        collective_id,
    ).exchange(local_chunks)
    owned_chunks = exchange_result.owned

    all_chunks = sorted(
        [*exchange_result.ghosts, *owned_chunks],
        key=lambda chunk: (chunk.row_start, chunk.sequence_number),
    )
    for cursor in owned_chunks:
        if cursor.num_rows == 0:
            result = await evaluate_available_chunk(
                context,
                cursor.chunk,
                ir,
                ir_context=ir_context,
            )
        else:
            result = await evaluate_ghosted_cursor(
                context,
                ir,
                ir_context,
                cursor,
                chunks=all_chunks,
                ghost_start=max(0, cursor.row_start - policy.preceding),
                ghost_stop=min(plan.total_rows, cursor.row_stop + policy.following),
            )
        if tracer is not None:
            tracer.add_chunk(chunk=result)
        await ch_out.send(context, Message(cursor.sequence_number, result))

    await ch_out.drain(context)


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
    """Evaluate range rolling with sparse inter-rank ghost slices."""
    local_chunks, local_rows = await _recv_all_range_chunks(context, ch_in, policy)
    local_stats = _local_range_stats(local_chunks, local_rows)
    rank_stats = await gather_range_stats(
        context,
        comm,
        ir_context,
        local_stats,
        collective_id=collective_id,
    )
    _validate_global_range_ordering(rank_stats)
    source_spans = _source_spans_from_rank_stats(rank_stats)

    local_start, local_stop = source_spans[comm.rank]
    for chunk in local_chunks:
        chunk.row_start += local_start

    router = _range_router(rank_stats)
    request_destinations = router.request_sources(comm.rank)
    request_sources = router.request_destinations(comm.rank)
    range_requests = _range_requests_by_source(
        comm.rank,
        rank_stats,
        local_chunks,
        request_destinations,
    )
    request_chunks = {
        destination: _range_request_chunk(context, range_requests.get(destination, ()))
        for destination in request_destinations
    }
    remote_request_chunks = await exchange_payload_chunks(
        context,
        comm,
        ir_context,
        chunks_by_destination=request_chunks,
        sources=request_sources,
        collective_id=collective_id,
    )
    sends: list[ResolvedRowSend] = []
    for destination, chunks in remote_request_chunks.items():
        requests = [
            request for chunk in chunks for request in _range_requests_from_chunk(chunk)
        ]
        sends.extend(
            ResolvedRowSend(destination, start, stop, is_ghost=True)
            for start, stop in _resolve_range_requests(
                context,
                policy,
                local_chunks,
                _merge_value_ranges(requests),
            )
        )
    exchange_result = await exchange_resolved_slices(
        context,
        comm,
        ir_context,
        local_chunks,
        local_owned_intervals=((local_start, local_stop),),
        sends=sends,
        sources=tuple(sorted(range_requests)),
        collective_id=collective_id,
    )
    owned_chunks = await _prepare_exchanged_range_chunks(
        context, policy, exchange_result.owned
    )
    ghost_chunks = await _prepare_exchanged_range_chunks(
        context, policy, exchange_result.ghosts
    )
    all_chunks = sorted(
        [*ghost_chunks, *owned_chunks],
        key=lambda chunk: (chunk.row_start, chunk.sequence_number),
    )
    for cursor in owned_chunks:
        policy.validate_cursor(context, cursor)
        if cursor.num_rows == 0:
            result = await evaluate_available_chunk(
                context,
                cursor.chunk,
                ir,
                ir_context=ir_context,
            )
        else:
            cursor_index = all_chunks.index(cursor)
            result = await policy.evaluate_cursor(
                context,
                ir,
                ir_context,
                cursor,
                history=all_chunks[:cursor_index],
                future=all_chunks[cursor_index + 1 :],
            )
        if tracer is not None:
            tracer.add_chunk(chunk=result)
        await ch_out.send(context, Message(cursor.sequence_number, result))

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
        del config_options
        metadata_in = await recv_metadata(ch_in, context)
        partitioning = (
            maybe_remap_partitioning(ir, metadata_in.partitioning, context=context)
            if isinstance(ir, Rolling)
            else None
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

        policy: OverlapPolicy[Any]
        if isinstance(ir, Rolling):
            policy = RangeOverlapPolicy.from_ir(
                ir, context.br().stream_pool.get_stream()
            )
            if comm.nranks != 1 and not metadata_in.duplicated:
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
        else:
            policy = RowCountOverlapPolicy(ir.preceding_overlap, ir.following_overlap)
            if comm.nranks != 1 and not metadata_in.duplicated:
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
        await execute_rolling_policy(
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
