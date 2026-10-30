# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect and route row-ordered streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from itertools import pairwise
from typing import TYPE_CHECKING, cast

import pylibcudf as plc
from cudf_streaming.channel_metadata import Ordering
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.message import Message

from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.collectives.slice_exchange import (
    ExchangePlan,
    InputDemand,
    RoutedSlice,
    SliceExchange,
    SliceResolver,
    merge_row_ranges,
    replay_buffer,
    tag_row_slice,
    use_precomputed_slices,
)
from cudf_polars.streaming.actor_graph.utils import ChunkStore
from cudf_polars.utils.cuda_stream import join_cuda_streams, stream_ordered_after
from cudf_polars.utils.dtypes import make_empty_column

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from cudf_streaming.channel_metadata import OrderKey
    from pylibcudf.utils import CudaStreamLike
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.collectives.slice_exchange import (
        PreparedExchange,
    )
    from cudf_polars.typing import Schema


@dataclass(frozen=True)
class RankEndpoints:
    """First and last key rows for each nonempty rank."""

    keys: plc.Table
    """Endpoint key rows ordered as ``first0, last0, first1, last1, ...``."""
    ranks: tuple[int, ...]
    """Rank corresponding to each pair of rows in ``keys``."""
    stream: CudaStreamLike | None = field(default=None, repr=False, compare=False)
    """CUDA stream producing ``keys``."""


@dataclass(frozen=True)
class KeyBounds:
    """Lower and upper bounds of one ordered key range."""

    lower: plc.Table | None
    """Lower bound, or ``None`` when unbounded."""
    upper: plc.Table | None
    """Upper bound, or ``None`` when unbounded."""
    lower_inclusive: bool = True
    """Whether ``lower`` belongs to the range."""
    upper_inclusive: bool = True
    """Whether ``upper`` belongs to the interval."""
    stream: CudaStreamLike | None = field(default=None, repr=False, compare=False)
    """CUDA stream producing the bound tables."""


def _order_rank_endpoints_after(
    stream: CudaStreamLike,
    endpoints: RankEndpoints,
) -> None:
    """Order a consumer stream after the endpoint producer stream."""
    if endpoints.stream is not None and endpoints.stream is not stream:
        join_cuda_streams(downstreams=(stream,), upstreams=(endpoints.stream,))


@dataclass
class BufferedOrderedInput:
    """Spillable input chunks and their rank-local key endpoints."""

    chunks: ChunkStore
    endpoints: TableChunk
    summaries: ChunkStore | None = None
    summary_sizes: tuple[int, ...] = ()

    def clear(self) -> None:
        """Release buffered chunks."""
        self.chunks.clear()
        if self.summaries is not None:
            self.summaries.clear()
            self.summaries = None
        self.summary_sizes = ()

    async def iter_summaries(
        self,
        context: Context,
    ) -> AsyncGenerator[tuple[int, KeyBounds | None]]:
        """Yield compact chunk summaries before replaying the buffered data."""
        if self.summaries is None:
            raise ValueError("This ordered input has no chunk summaries")
        summaries = iter(self.summaries)
        try:
            for size in self.summary_sizes:
                if size == 0:
                    yield size, None
                    continue
                message = next(summaries, None)
                if message is None:
                    raise ValueError("Chunk key ranges must match buffered chunks")
                summary = TableChunk.from_message(
                    message,
                    br=context.br(),
                )
                summary, _ = await make_table_chunks_available_or_wait(
                    context,
                    summary,
                    reserve_extra=0,
                    net_memory_delta=0,
                )
                first, last = plc.copying.slice(
                    summary.table_view(),
                    [0, 1, 1, 2],
                    stream=summary.stream,
                )
                yield (
                    size,
                    KeyBounds(
                        first,
                        last,
                        upper_inclusive=True,
                        stream=summary.stream,
                    ),
                )
            if next(summaries, None) is not None:
                raise ValueError("Chunk key ranges must match buffered chunks")
        finally:
            self.summaries.clear()
            self.summaries = None
            self.summary_sizes = ()

    async def replay[SummaryT](
        self,
        _context: Context,
        summaries: Sequence[SummaryT | None] = (),
    ) -> AsyncGenerator[tuple[Message[TableChunk], SummaryT | None]]:
        """Replay buffered chunks with optional precomputed routing summaries."""
        if self.summaries is not None:
            raise ValueError("Chunk summaries must be consumed before replay")
        async for item in replay_buffer(self.chunks, summaries):
            yield item


def _scalar_int(column: plc.Column, row: int, stream: Stream) -> int:
    """Copy one integer value to the host."""
    (value,) = plc.copying.slice(column, [row, row + 1], stream=stream)
    result = value.to_scalar(stream=stream).to_py(stream=stream)
    assert isinstance(result, int)
    return result


async def gather_rank_endpoints(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    collective_id: int,
    local_keys: TableChunk,
    stream: Stream,
) -> RankEndpoints:
    """All-gather zero or two endpoint key rows from every rank."""
    return (
        await gather_rank_endpoint_sets(
            context,
            comm,
            ir_context,
            collective_id,
            (local_keys,),
            stream,
        )
    )[0]


async def gather_rank_endpoint_sets(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    collective_id: int,
    local_key_sets: Sequence[TableChunk],
    stream: Stream,
) -> tuple[RankEndpoints, ...]:
    """Consume and all-gather zero-or-two-row endpoint tables."""
    if not local_key_sets:
        return ()
    if any(keys.shape[0] not in (0, 2) for keys in local_key_sets):
        raise ValueError("Rank endpoint tables must contain zero or two rows")
    available, extra = await make_table_chunks_available_or_wait(
        context,
        local_key_sets,
        reserve_extra=sum(keys.data_alloc_size() for keys in local_key_sets),
        net_memory_delta=0,
    )
    with (
        opaque_memory_usage(extra),
        stream_ordered_after(
            context.br().stream_pool.get_stream,
            upstreams=tuple(keys.stream for keys in available),
        ) as concat_stream,
    ):
        tagged_tables = []
        for set_index, local_keys in enumerate(available):
            size = local_keys.shape[0]
            set_column = plc.Column.from_iterable_of_py(
                [set_index] * size,
                plc.DataType(plc.TypeId.INT32),
                stream=concat_stream,
            )
            rank_column = plc.Column.from_iterable_of_py(
                [comm.rank] * size,
                plc.DataType(plc.TypeId.INT32),
                stream=concat_stream,
            )
            tagged_tables.append(
                plc.Table(
                    [
                        set_column,
                        rank_column,
                        *local_keys.table_view().columns(),
                    ]
                )
            )
        tagged_table = plc.concatenate.concatenate(
            tagged_tables,
            stream=concat_stream,
            mr=context.br().device_mr,
        )
    tagged = TableChunk.from_pylibcudf_table(
        tagged_table,
        concat_stream,
        exclusive_view=True,
        br=context.br(),
    )
    gather = AllGatherManager(context, comm, collective_id)
    with gather.inserting() as inserter:
        await inserter.insert(comm.rank, tagged)
    table = await gather.extract_concatenated(
        stream,
        ordered=True,
        ir_context=ir_context,
    )
    set_column, rank_column, *key_columns = table.columns()
    key_table = plc.Table(key_columns)
    set_indices = set_column.to_arrow(stream=stream).to_pylist()
    rank_indices = rank_column.to_arrow(stream=stream).to_pylist()
    result = []
    for set_index in range(len(local_key_sets)):
        positions = tuple(
            row for row, value in enumerate(set_indices) if value == set_index
        )
        if len(positions) % 2:
            raise RuntimeError("Invalid gathered rank endpoints")
        ranks = tuple(rank_indices[row] for row in positions[::2])
        if any(
            rank_indices[right] != rank
            for right, rank in zip(positions[1::2], ranks, strict=True)
        ):
            raise RuntimeError("Invalid gathered rank endpoint pairs")
        gather_map = plc.Column.from_iterable_of_py(
            list(positions),
            plc.DataType(plc.TypeId.INT32),
            stream=stream,
        )
        keys = plc.copying.gather(
            key_table,
            gather_map,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            stream=stream,
        )
        result.append(RankEndpoints(keys, ranks, stream))
    return tuple(result)


def _ordering_from_boundary_rows(
    br: BufferResource,
    endpoints: RankEndpoints,
    output_keys: Sequence[OrderKey],
    boundary_rows: Sequence[int],
    stream: Stream,
) -> Ordering:
    """Build strict ordering metadata from selected endpoint rows."""
    _order_rank_endpoints_after(stream, endpoints)
    gather_map = plc.Column.from_iterable_of_py(
        list(boundary_rows),
        plc.DataType(plc.TypeId.INT32),
        stream=stream,
    )
    boundaries = plc.copying.gather(
        endpoints.keys,
        gather_map,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )
    boundary_chunk = TableChunk.from_pylibcudf_table(
        boundaries,
        stream,
        exclusive_view=True,
        br=br,
    )
    return Ordering(output_keys, boundary_chunk, strict_boundaries=True)


def strict_inter_rank_ordering(
    br: BufferResource,
    endpoints: RankEndpoints,
    input_keys: Sequence[OrderKey],
    output_keys: Sequence[OrderKey],
    nranks: int,
    stream: Stream,
) -> Ordering | None:
    """Return strict rank-range metadata when every rank boundary is disjoint."""
    if endpoints.ranks != tuple(range(nranks)):
        return None
    if any(
        endpoint_rows_equal(endpoints, 2 * rank + 1, 2 * rank + 2, input_keys, stream)
        for rank in range(nranks - 1)
    ):
        return None
    return _ordering_from_boundary_rows(
        br,
        endpoints,
        output_keys,
        tuple(2 * rank for rank in range(1, nranks)),
        stream,
    )


def repaired_inter_rank_ordering(
    br: BufferResource,
    endpoints: RankEndpoints,
    output_keys: Sequence[OrderKey],
    nranks: int,
    stream: Stream,
) -> Ordering | None:
    """Return strict metadata after shared boundary keys move to the right."""
    boundary_rows = _ownership_boundary_rows(endpoints, nranks)
    if boundary_rows is None:
        return None
    return _ordering_from_boundary_rows(
        br,
        endpoints,
        output_keys,
        boundary_rows,
        stream,
    )


async def buffer_ordered_input(
    context: Context,
    channel: Channel[TableChunk],
    schema: Schema,
    order_keys: Sequence[OrderKey],
    stream: Stream,
    *,
    summarize_chunks: bool = False,
) -> BufferedOrderedInput:
    """Buffer an ordered stream and copy its first and last key rows."""
    chunks = ChunkStore(context)
    summaries = ChunkStore(context) if summarize_chunks else None
    summary_sizes: list[int] = []
    first: TableChunk | None = None
    last: TableChunk | None = None
    try:
        while (message := await channel.recv(context)) is not None:
            sequence_number = message.sequence_number
            chunk = TableChunk.from_message(message, br=context.br())
            if summarize_chunks:
                summary_sizes.append(chunk.shape[0])
            if chunk.shape[0]:
                (chunk,), extra = await make_table_chunks_available_or_wait(
                    context,
                    [chunk],
                    reserve_extra=0,
                    net_memory_delta=0,
                )
                with opaque_memory_usage(extra):
                    if first is None:
                        first = await copy_key_row(context, chunk, order_keys, 0)
                    last = await copy_key_row(
                        context,
                        chunk,
                        order_keys,
                        chunk.shape[0] - 1,
                    )
                    if summarize_chunks:
                        assert summaries is not None
                        summaries.insert(
                            Message(
                                sequence_number,
                                await concatenate_key_rows(
                                    context,
                                    await copy_key_row(context, chunk, order_keys, 0),
                                    await copy_key_row(
                                        context,
                                        chunk,
                                        order_keys,
                                        chunk.shape[0] - 1,
                                    ),
                                ),
                            )
                        )
            chunks.insert(Message(sequence_number, chunk))
        if first is None:
            endpoints = TableChunk.from_pylibcudf_table(
                plc.Table(
                    [
                        make_empty_column(
                            tuple(schema.values())[key.column_index], stream
                        )
                        for key in order_keys
                    ]
                ),
                stream,
                exclusive_view=True,
                br=context.br(),
            )
        else:
            assert last is not None
            endpoints = await concatenate_key_rows(context, first, last)
        return BufferedOrderedInput(
            chunks,
            endpoints,
            summaries,
            tuple(summary_sizes),
        )
    except BaseException:
        chunks.clear()
        if summaries is not None:
            summaries.clear()
        raise


def endpoint_rows_equal(
    endpoints: RankEndpoints,
    left: int,
    right: int,
    order_keys: Sequence[OrderKey],
    stream: Stream,
) -> bool:
    """Return whether two endpoint rows belong to the same ordered key group."""
    _order_rank_endpoints_after(stream, endpoints)
    if endpoints.keys.num_columns() == 0:
        raise ValueError("Endpoint equality requires at least one key")
    left_row = plc.copying.slice(endpoints.keys, [left, left + 1], stream=stream)[0]
    right_row = plc.copying.slice(endpoints.keys, [right, right + 1], stream=stream)[0]
    return not key_row_less(
        left_row,
        right_row,
        order_keys,
        stream,
    ) and not key_row_less(
        right_row,
        left_row,
        order_keys,
        stream,
    )


def key_row_less(
    left: plc.Table,
    right: plc.Table,
    order_keys: Sequence[OrderKey],
    stream: Stream,
) -> bool:
    """Return whether one key row precedes another in the specified order."""
    if left.num_rows() != 1 or right.num_rows() != 1:
        raise ValueError("Key comparison requires one row per table")
    insertion = plc.search.lower_bound(
        left,
        right,
        [key.order for key in order_keys],
        [key.null_order for key in order_keys],
        stream=stream,
    )
    return _scalar_int(insertion, 0, stream) == 1


def key_rows_equal(
    left: plc.Table,
    right: plc.Table,
    order_keys: Sequence[OrderKey],
    stream: Stream,
) -> bool:
    """Return whether two single-row keys compare equal."""
    return not key_row_less(left, right, order_keys, stream) and not key_row_less(
        right, left, order_keys, stream
    )


def key_bounds_view(chunk: TableChunk) -> KeyBounds:
    """Return inclusive endpoint bounds from an available two-row chunk."""
    if chunk.shape[0] != 2:
        raise ValueError("A key-bounds chunk must contain exactly two rows")
    lower, upper = plc.copying.slice(
        chunk.table_view(),
        [0, 1, 1, 2],
        stream=chunk.stream,
    )
    return KeyBounds(lower, upper, stream=chunk.stream)


def _order_key_bounds_after(
    stream: CudaStreamLike,
    *bounds: KeyBounds,
) -> None:
    """Order a consumer stream after streams producing bound tables."""
    upstreams = tuple(
        bound.stream
        for bound in bounds
        if bound.stream is not None and bound.stream is not stream
    )
    if upstreams:
        join_cuda_streams(downstreams=(stream,), upstreams=upstreams)


def _key_bounds_are_empty(
    bounds: KeyBounds,
    order_keys: Sequence[OrderKey],
    stream: Stream,
) -> bool:
    """Return whether an ordered key range contains no values."""
    _order_key_bounds_after(stream, bounds)
    if bounds.lower is None or bounds.upper is None:
        return False
    if key_row_less(bounds.upper, bounds.lower, order_keys, stream):
        raise ValueError("Key bounds are not ordered")
    return (
        not bounds.lower_inclusive or not bounds.upper_inclusive
    ) and key_rows_equal(bounds.lower, bounds.upper, order_keys, stream)


def _ownership_boundary_rows(
    endpoints: RankEndpoints,
    nranks: int,
) -> tuple[int, ...] | None:
    """Return endpoint rows defining one ownership interval per rank."""
    if not endpoints.ranks:
        return None
    rows = []
    position = 0
    for cut in range(nranks - 1):
        while position < len(endpoints.ranks) and endpoints.ranks[position] <= cut:
            position += 1
        rows.append(
            2 * position
            if position < len(endpoints.ranks)
            else endpoints.keys.num_rows() - 1
        )
    return tuple(rows)


def rank_key_bounds(
    endpoints: RankEndpoints,
    nranks: int,
    stream: Stream,
) -> tuple[KeyBounds | None, ...]:
    """Return each physical rank's inclusive endpoint range."""
    _order_rank_endpoints_after(stream, endpoints)
    result: list[KeyBounds | None] = [None] * nranks
    for position, rank in enumerate(endpoints.ranks):
        first, last = plc.copying.slice(
            endpoints.keys,
            [2 * position, 2 * position + 1, 2 * position + 1, 2 * position + 2],
            stream=stream,
        )
        result[rank] = KeyBounds(first, last, stream=stream)
    return tuple(result)


def rank_ownership_intervals(
    endpoints: RankEndpoints,
    nranks: int,
    order_keys: Sequence[OrderKey],
    stream: Stream,
    *,
    clip_to_domain: bool,
) -> tuple[KeyBounds | None, ...]:
    """
    Assign one disjoint ordered key interval to every physical rank.

    Adjacent intervals meet at the first key of the right rank, so a key run
    crossing a rank boundary belongs to the rightmost rank in that run.
    """
    _order_rank_endpoints_after(stream, endpoints)
    boundary_rows = _ownership_boundary_rows(endpoints, nranks)
    if boundary_rows is None:
        return (None,) * nranks
    boundaries = tuple(
        plc.copying.slice(
            endpoints.keys,
            [row, row + 1],
            stream=stream,
        )[0]
        for row in boundary_rows
    )
    global_first = plc.copying.slice(
        endpoints.keys,
        [0, 1],
        stream=stream,
    )[0]
    global_last = plc.copying.slice(
        endpoints.keys,
        [endpoints.keys.num_rows() - 1, endpoints.keys.num_rows()],
        stream=stream,
    )[0]
    result = []
    for rank in range(nranks):
        lower = boundaries[rank - 1] if rank else None
        upper = boundaries[rank] if rank < nranks - 1 else None
        upper_inclusive = False
        if clip_to_domain:
            lower = global_first if rank == 0 else lower
            if rank == nranks - 1:
                upper = global_last
                upper_inclusive = True
        interval = KeyBounds(
            lower,
            upper,
            upper_inclusive=upper_inclusive,
            stream=stream,
        )
        result.append(
            None if _key_bounds_are_empty(interval, order_keys, stream) else interval
        )
    return tuple(result)


def key_bounds_overlap(
    left: KeyBounds,
    right: KeyBounds,
    order_keys: Sequence[OrderKey],
    stream: Stream,
) -> bool:
    """Return whether two ordered key ranges intersect."""
    return not (
        _key_bounds_precede(left, right, order_keys, stream)
        or _key_bounds_precede(right, left, order_keys, stream)
    )


def _key_bounds_precede(
    left: KeyBounds,
    right: KeyBounds,
    order_keys: Sequence[OrderKey],
    stream: Stream,
) -> bool:
    """Return whether one range ends before another begins."""
    _order_key_bounds_after(stream, left, right)
    if left.upper is None or right.lower is None:
        return False
    if key_row_less(left.upper, right.lower, order_keys, stream):
        return True
    return key_rows_equal(left.upper, right.lower, order_keys, stream) and not (
        left.upper_inclusive and right.lower_inclusive
    )


def _key_bounds_contain(
    outer: KeyBounds,
    inner: KeyBounds,
    order_keys: Sequence[OrderKey],
    stream: Stream,
) -> bool:
    """Return whether one ordered key range contains another."""
    _order_key_bounds_after(stream, outer, inner)
    if inner.lower is None:
        if outer.lower is not None:
            return False
    elif outer.lower is not None:
        if key_row_less(inner.lower, outer.lower, order_keys, stream):
            return False
        if (
            inner.lower_inclusive
            and not outer.lower_inclusive
            and key_rows_equal(inner.lower, outer.lower, order_keys, stream)
        ):
            return False
    if inner.upper is None:
        return outer.upper is None
    if outer.upper is None or key_row_less(
        inner.upper, outer.upper, order_keys, stream
    ):
        return True
    return (not inner.upper_inclusive or outer.upper_inclusive) and key_rows_equal(
        inner.upper, outer.upper, order_keys, stream
    )


def key_bounds_summary_slices(
    size: int,
    supply: KeyBounds | None,
    requests: Sequence[tuple[int, InputDemand[KeyBounds, KeyBounds]]],
    order_keys: Sequence[OrderKey],
    stream: Stream,
) -> list[RoutedSlice] | None:
    """Resolve skipped or whole chunks from endpoint summaries."""
    if supply is None:
        return []
    pieces = []
    for destination, demand in requests:
        needed_overlap = key_bounds_overlap(supply, demand.needed, order_keys, stream)
        owned_overlap = demand.owned is not None and key_bounds_overlap(
            supply, demand.owned, order_keys, stream
        )
        if not needed_overlap and not owned_overlap:
            continue
        needed_full = _key_bounds_contain(demand.needed, supply, order_keys, stream)
        owned_full = demand.owned is not None and _key_bounds_contain(
            demand.owned, supply, order_keys, stream
        )
        if (needed_overlap and not needed_full) or (owned_overlap and not owned_full):
            return None
        pieces.append(RoutedSlice(destination, 0, size, owned=owned_full))
    return pieces


def _key_bounds_span(
    keys: plc.Table,
    bounds: KeyBounds,
    order_keys: Sequence[OrderKey],
    stream: Stream,
) -> tuple[int, int]:
    """Return the half-open row span selected by ordered key bounds."""
    _order_key_bounds_after(stream, bounds)
    orders = [key.order for key in order_keys]
    null_orders = [key.null_order for key in order_keys]
    start = 0
    if bounds.lower is not None:
        search = (
            plc.search.lower_bound if bounds.lower_inclusive else plc.search.upper_bound
        )
        start = _scalar_int(
            search(keys, bounds.lower, orders, null_orders, stream=stream),
            0,
            stream,
        )
    stop = keys.num_rows()
    if bounds.upper is not None:
        search = (
            plc.search.upper_bound if bounds.upper_inclusive else plc.search.lower_bound
        )
        stop = _scalar_int(
            search(keys, bounds.upper, orders, null_orders, stream=stream),
            0,
            stream,
        )
    return start, stop


async def key_bounds_slices(
    chunk: TableChunk,
    _summary: object | None,
    requests: Sequence[tuple[int, InputDemand[KeyBounds, KeyBounds]]],
    order_keys: Sequence[OrderKey],
) -> list[RoutedSlice]:
    """Resolve destination key ranges within one ordered chunk."""
    table = chunk.table_view()
    keys = plc.Table([table.columns()[key.column_index] for key in order_keys])
    pieces = []
    for destination, demand in requests:
        needed_span = _key_bounds_span(keys, demand.needed, order_keys, chunk.stream)
        owned_span = (
            _key_bounds_span(keys, owned, order_keys, chunk.stream)
            if (owned := demand.owned) is not None
            else None
        )
        for start, stop in merge_row_ranges(needed_span, owned_span):
            pieces.extend(
                tag_row_slice(
                    destination,
                    (0, chunk.shape[0]),
                    (start, stop),
                    owned_span,
                )
            )
    return pieces


def plan_key_bounds_exchange(
    rank: int,
    nranks: int,
    supply_endpoints: RankEndpoints,
    demands: Sequence[KeyBounds | None],
    order_keys: Sequence[OrderKey],
    stream: Stream,
    *,
    transfer_ownership: bool,
) -> ExchangePlan[KeyBounds, KeyBounds]:
    """Plan routes between ordered rank supplies and disjoint demands."""
    if len(demands) != nranks:
        raise ValueError("Key demands must match the rank count")
    active_demands = tuple(
        (destination, demand)
        for destination, demand in enumerate(demands)
        if demand is not None
    )
    if any(
        not _key_bounds_precede(left[1], right[1], order_keys, stream)
        for left, right in pairwise(active_demands)
    ):
        raise ValueError("Key demands must be ordered and non-overlapping")

    routes: list[tuple[int, ...]] = []
    first_demand = 0
    supplies = rank_key_bounds(supply_endpoints, nranks, stream)
    for supply in supplies:
        if supply is None:
            routes.append(())
            continue
        while first_demand < len(active_demands) and _key_bounds_precede(
            active_demands[first_demand][1], supply, order_keys, stream
        ):
            first_demand += 1
        destinations: list[int] = []
        for destination, demand in active_demands[first_demand:]:
            if _key_bounds_precede(supply, demand, order_keys, stream):
                break
            destinations.append(destination)
        routes.append(tuple(destinations))
    outgoing = []
    for destination in routes[rank]:
        selected_demand = demands[destination]
        assert selected_demand is not None
        outgoing.append(
            (
                destination,
                InputDemand(
                    owned=selected_demand if transfer_ownership else None,
                    needed=selected_demand,
                ),
            )
        )
    return ExchangePlan.from_routes(rank, routes, outgoing)


async def plan_buffered_key_slices(
    context: Context,
    input_: BufferedOrderedInput,
    plan: ExchangePlan[KeyBounds, KeyBounds],
    order_keys: Sequence[OrderKey],
    stream: Stream,
) -> list[list[RoutedSlice] | None]:
    """Resolve key routes from compact buffered-chunk summaries."""
    result = []
    async for size, summary in input_.iter_summaries(context):
        result.append(
            key_bounds_summary_slices(
                size,
                summary,
                plan.outgoing,
                order_keys,
                stream,
            )
        )
    return result


async def exchange_buffered_key_input(
    context: Context,
    comm: Communicator,
    collective_id: int,
    input_: BufferedOrderedInput,
    plan: ExchangePlan[KeyBounds, KeyBounds],
    order_keys: Sequence[OrderKey],
    stream: Stream,
) -> PreparedExchange:
    """Exchange buffered ordered input according to key-bound demands."""
    planned = await plan_buffered_key_slices(
        context,
        input_,
        plan,
        order_keys,
        stream,
    )
    return await SliceExchange(context, comm, collective_id).exchange(
        input_.replay(context, planned),
        plan,
        resolver=SliceResolver(
            resolve_chunk=partial(key_bounds_slices, order_keys=order_keys),
            resolve_summary=use_precomputed_slices,
        ),
    )


async def copy_key_row(
    context: Context,
    chunk: TableChunk,
    order_keys: Sequence[OrderKey],
    row: int,
) -> TableChunk:
    """Copy one ordered-key row from an available chunk."""
    table = chunk.table_view()
    keys = plc.Table([table.columns()[key.column_index] for key in order_keys])
    (row_view,) = plc.copying.slice(keys, [row, row + 1], stream=chunk.stream)
    view = TableChunk.from_pylibcudf_table(
        row_view,
        chunk.stream,
        exclusive_view=False,
        br=context.br(),
    )
    reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
        view.into_packed_data_cost(),
        net_memory_delta=0,
    )
    with opaque_memory_usage(reservation):
        copied = row_view.copy(chunk.stream, context.br().device_mr)
    return TableChunk.from_pylibcudf_table(
        copied,
        chunk.stream,
        exclusive_view=True,
        br=context.br(),
    )


async def concatenate_key_rows(
    context: Context,
    first: TableChunk,
    last: TableChunk,
) -> TableChunk:
    """Concatenate independently owned first and last key rows."""
    (first, last), extra = await make_table_chunks_available_or_wait(
        context,
        [first, last],
        reserve_extra=first.data_alloc_size() + last.data_alloc_size(),
        net_memory_delta=0,
    )
    with (
        opaque_memory_usage(extra),
        stream_ordered_after(
            context.br().stream_pool.get_stream,
            upstreams=(first.stream, last.stream),
        ) as stream,
    ):
        return TableChunk.from_pylibcudf_table(
            plc.concatenate.concatenate(
                [first.table_view(), last.table_view()],
                stream=stream,
                mr=context.br().device_mr,
            ),
            stream,
            exclusive_view=True,
            br=context.br(),
        )


async def copy_key_range(
    context: Context,
    chunk: TableChunk,
    order_keys: Sequence[OrderKey],
) -> tuple[TableChunk, TableChunk]:
    """Copy the first and last ordered-key rows from one nonempty chunk."""
    if chunk.shape[0] == 0:
        raise ValueError("Cannot copy a key range from an empty chunk")
    chunk, extra = await make_table_chunks_available_or_wait(
        context,
        chunk,
        reserve_extra=0,
        net_memory_delta=0,
    )
    with opaque_memory_usage(extra):
        result = await concatenate_key_rows(
            context,
            await copy_key_row(context, chunk, order_keys, 0),
            await copy_key_row(context, chunk, order_keys, chunk.shape[0] - 1),
        )
    return chunk, result


def _copy_slice(
    context: Context,
    table: plc.Table,
    start: int,
    stop: int,
    stream: Stream,
) -> TableChunk | None:
    """Copy a nonempty table slice into an independently owned chunk."""
    if start == stop:
        return None
    (view,) = plc.copying.slice(table, [start, stop], stream=stream)
    return TableChunk.from_pylibcudf_table(
        view.copy(stream, context.br().device_mr),
        stream,
        exclusive_view=True,
        br=context.br(),
    )


async def stable_sort_chunk(
    context: Context,
    chunk: TableChunk,
    order_keys: Sequence[OrderKey],
) -> TableChunk:
    """Return a chunk stably sorted by ``order_keys``."""
    if chunk.shape[0] < 2:
        return chunk
    chunk, extra = await make_table_chunks_available_or_wait(
        context,
        chunk,
        reserve_extra=chunk.data_alloc_size(),
        net_memory_delta=0,
    )
    with opaque_memory_usage(extra):
        table = chunk.table_view()
        keys = plc.Table([table.columns()[key.column_index] for key in order_keys])
        table = plc.sorting.stable_sort_by_key(
            table,
            keys,
            [key.order for key in order_keys],
            [key.null_order for key in order_keys],
            stream=chunk.stream,
            mr=context.br().device_mr,
        )
    return TableChunk.from_pylibcudf_table(
        table,
        chunk.stream,
        exclusive_view=True,
        br=context.br(),
    )


async def _split_equal_run(
    context: Context,
    chunk: TableChunk,
    order_keys: Sequence[OrderKey],
    *,
    leading: bool,
) -> tuple[TableChunk, TableChunk | None]:
    """Split a sorted chunk beside its first or final equal-key run."""
    nrows = chunk.shape[0]
    if nrows == 0:
        return chunk, None
    chunk, extra = await make_table_chunks_available_or_wait(
        context,
        chunk,
        reserve_extra=chunk.data_alloc_size(),
        net_memory_delta=0,
    )
    with opaque_memory_usage(extra):
        table = chunk.table_view()
        keys = plc.Table([table.columns()[key.column_index] for key in order_keys])
        row = 0 if leading else nrows - 1
        (boundary_key,) = plc.copying.slice(
            keys,
            [row, row + 1],
            stream=chunk.stream,
        )
        find = plc.search.upper_bound if leading else plc.search.lower_bound
        split = cast(
            "int",
            find(
                keys,
                boundary_key,
                [key.order for key in order_keys],
                [key.null_order for key in order_keys],
                stream=chunk.stream,
            )
            .to_scalar(stream=chunk.stream)
            .to_py(stream=chunk.stream),
        )
        if split == nrows:
            return chunk, None
        first_view, second_view = plc.copying.split(table, [split], stream=chunk.stream)
        first = first_view.copy(chunk.stream, context.br().device_mr)
        if split == 0:
            return (
                TableChunk.from_pylibcudf_table(
                    first,
                    chunk.stream,
                    exclusive_view=True,
                    br=context.br(),
                ),
                chunk,
            )
        second = second_view.copy(chunk.stream, context.br().device_mr)
    return (
        TableChunk.from_pylibcudf_table(
            first,
            chunk.stream,
            exclusive_view=True,
            br=context.br(),
        ),
        TableChunk.from_pylibcudf_table(
            second,
            chunk.stream,
            exclusive_view=True,
            br=context.br(),
        ),
    )


async def split_trailing_run(
    context: Context,
    chunk: TableChunk,
    order_keys: Sequence[OrderKey],
) -> tuple[TableChunk, TableChunk | None]:
    """Split a sorted chunk before its final equal-key run."""
    return await _split_equal_run(
        context,
        chunk,
        order_keys,
        leading=False,
    )


async def split_leading_run(
    context: Context,
    chunk: TableChunk,
    order_keys: Sequence[OrderKey],
) -> tuple[TableChunk, TableChunk | None]:
    """Split a sorted chunk after its first equal-key run."""
    return await _split_equal_run(
        context,
        chunk,
        order_keys,
        leading=True,
    )


async def select_demand_range(
    context: Context,
    supply: TableChunk,
    demand: KeyBounds,
    demand_stream: Stream,
    supply_keys: Sequence[OrderKey],
) -> tuple[TableChunk | None, TableChunk | None, bool]:
    """Select supply rows in one demand range and retain any later suffix."""
    if supply.shape[0] == 0:
        return None, None, False
    supply, extra = await make_table_chunks_available_or_wait(
        context,
        supply,
        reserve_extra=supply.data_alloc_size(),
        net_memory_delta=0,
    )
    with (
        opaque_memory_usage(extra),
        stream_ordered_after(
            context.br().stream_pool.get_stream,
            upstreams=(supply.stream, demand_stream),
        ) as stream,
    ):
        supply_table = supply.table_view()
        supply_key_table = plc.Table(
            [supply_table.columns()[key.column_index] for key in supply_keys]
        )
        start, stop = _key_bounds_span(
            supply_key_table,
            demand,
            supply_keys,
            stream,
        )
        passed_demand = stop < supply.shape[0]
        if start == 0 and stop == supply.shape[0]:
            return supply, None, False
        if stop == 0:
            return None, supply, True
        selected = _copy_slice(context, supply_table, start, stop, stream)
        suffix = _copy_slice(context, supply_table, stop, supply.shape[0], stream)
    return selected, suffix, passed_demand
