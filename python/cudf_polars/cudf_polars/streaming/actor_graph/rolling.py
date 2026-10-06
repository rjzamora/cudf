# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

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

from cudf_polars.dsl.ir import IR, Rolling
from cudf_polars.dsl.utils.windows import duration_to_scalar
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
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
from cudf_polars.utils.cuda_stream import join_cuda_streams, stream_ordered_after

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext
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
class BufferedChunk:
    """
    An input chunk, with additional metadata about location in the global frame.

    Consider a stream of chunks representing a frame with N total rows.
    Each BufferedChunk represents a non-overlapping slice of that frame,
    corresponding to the rows [global_start, global_stop).
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
class RangeBufferedChunk(BufferedChunk):
    """A buffered chunk with range-window overlap metadata."""

    overlap: RangeOverlap


BufferedChunkT = TypeVar("BufferedChunkT", bound=BufferedChunk)


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

    async def prepare_chunk(
        self,
        context: Context,
        msg: Message,
        *,
        row_offset: int,
    ) -> RangeBufferedChunk:
        """Convert a message to a staged chunk and extract its physical index."""
        chunk = TableChunk.from_message(msg, br=context.br())
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
        buffered = RangeBufferedChunk(
            msg.sequence_number,
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
        ghost_stop = cursor.row_stop + self.following
        return await evaluate_ghosted_cursor(
            context,
            ir,
            ir_context,
            cursor,
            chunks=[*history, cursor, *future],
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


async def extract_region(
    context: Context,
    input_chunks: Sequence[BufferedChunk],
    row_start: int,
    row_stop: int,
    *,
    ir_context: IRExecutionContext,
) -> TableChunk:
    """Slice all buffered chunks intersecting a global row range."""
    chunks: list[TableChunk] = []
    for buf in input_chunks:
        if buf.row_start >= row_stop:
            break
        start = max(row_start, buf.row_start)
        stop = min(row_stop, buf.row_stop)
        if start < stop:
            if start == buf.row_start and stop == buf.row_stop:
                chunks.append(buf.chunk)
            else:
                chunks.append(
                    TableChunk.from_pylibcudf_table(
                        plc.copying.slice(
                            buf.chunk.table_view(),
                            # Translate to chunk-local row coordinates
                            [
                                start - buf.row_start,
                                stop - buf.row_start,
                            ],
                            stream=buf.chunk.stream,
                        )[0],
                        buf.chunk.stream,
                        exclusive_view=False,
                        br=context.br(),
                    )
                )
    assert chunks, "Should have found at least one chunk to extract from"
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


async def evaluate_allgathered(
    context: Context,
    comm: Communicator,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    *,
    collective_id: int,
) -> TableChunk:
    """Gather rank-local chunks, concatenate globally, and evaluate ir."""
    stream = context.br().stream_pool.get_stream()
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
    return await evaluate_available_chunk(
        context,
        chunk,
        ir,
        ir_context=ir_context,
    )


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
        if comm.nranks != 1 and not metadata_in.duplicated:
            _fallback_inform(
                "Rolling does not support multi-rank inputs. "
                "Falling back to all-gather evaluation.",
                config_options,
            )
            metadata = ChannelMetadata(
                local_count=1, partitioning=None, duplicated=True
            )
            await send_metadata(ch_out, context, metadata)
            if tracer is not None:
                tracer.set_duplicated()

            result = await evaluate_allgathered(
                context,
                comm,
                ir,
                ir_context,
                ch_in,
                collective_id=collective_id,
            )
            if tracer is not None:
                tracer.add_chunk(chunk=result)
            await ch_out.send(context, Message(0, result))
            await ch_out.drain(context)
            return

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
        else:
            policy = RowCountOverlapPolicy(ir.preceding_overlap, ir.following_overlap)
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
