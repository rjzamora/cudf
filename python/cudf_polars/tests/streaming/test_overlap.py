# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING

import pytest

import polars as pl

import pylibcudf as plc
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.collectives.overlap import (
    RangeValidation,
    _RangeBounds,
    _RangeSliceSpec,
    _RangeSupply,
    _count_demands,
    _gather_ranges,
    _plan_count_overlap,
    _plan_range_overlap,
    _range_pieces,
)
from cudf_polars.streaming.actor_graph.collectives.slice_exchange import (
    ExchangePlan,
    InputDemand,
    SliceExchange,
    SliceResolver,
    row_range_slices,
    row_range_summary_slices,
)

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream


_RANGE_SPEC = _RangeSliceSpec(
    index_position=0,
    index_dtype=plc.DataType(plc.TypeId.INT64),
    start_closed=True,
    end_closed=True,
)


def _message(
    values: list[int], context: Context, stream: Stream
) -> Message[TableChunk]:
    table = plc.Table(
        [
            plc.Column.from_arrow(
                pl.Series("value", values, dtype=pl.Int64()),
                stream=stream,
            )
        ]
    )
    return Message(
        0,
        TableChunk.from_pylibcudf_table(
            table,
            stream,
            exclusive_view=True,
            br=context.br(),
        ),
    )


def _values(msg: Message[TableChunk], context: Context) -> list[int]:
    chunk = TableChunk.from_message(msg, br=context.br())
    result = DataFrame.from_table(
        chunk.table_view(),
        ["value"],
        [DataType(pl.Int64())],
        chunk.stream,
    ).to_polars()
    return result["value"].to_list()


def test_count_overlap_plans_independent_supply_and_demand() -> None:
    supplies = ((0, 2), (2, 2), (2, 4))
    demands: tuple[InputDemand[tuple[int, int], tuple[int, int]] | None, ...] = (
        None,
        InputDemand(owned=None, needed=(1, 3)),
        None,
    )

    plan = _plan_count_overlap(1, supplies, demands)

    assert plan.outgoing == ()
    assert plan.sources == (0, 2)
    assert _plan_count_overlap(0, supplies, demands).outgoing == ((1, demands[1]),)
    assert _plan_count_overlap(2, supplies, demands).outgoing == ((1, demands[1]),)


def test_range_overlap_plans_independent_supply_and_demand() -> None:
    supplies = (
        _RangeSupply((0, 5), _RangeBounds(0, 4)),
        None,
        _RangeSupply((5, 10), _RangeBounds(10, 14)),
    )
    demands: tuple[InputDemand[tuple[int, int], _RangeBounds] | None, ...] = (
        None,
        InputDemand(owned=None, needed=_RangeBounds(3, 11)),
        None,
    )

    plan = _plan_range_overlap(1, supplies, demands)

    assert plan.outgoing == ()
    assert plan.sources == (0, 2)

    left = _plan_range_overlap(0, supplies, demands)
    assert left.outgoing == ((1, demands[1]),)
    assert left.sources == ()
    assert _plan_range_overlap(2, supplies, demands).outgoing == ((1, demands[1]),)


def test_range_overlap_conservatively_includes_equal_endpoints() -> None:
    supply = _RangeSupply((0, 3), _RangeBounds(0, 2))
    demand: InputDemand[tuple[int, int], _RangeBounds] = InputDemand(
        owned=None, needed=_RangeBounds(2, None)
    )

    plan = _plan_range_overlap(0, (supply,), (demand,))

    assert plan.outgoing == ((0, demand),)


def test_count_demands_skip_empty_ownership() -> None:
    assert _count_demands(((0, 2), (2, 2), (2, 4)), 1, 1) == (
        InputDemand(owned=(0, 2), needed=(0, 3)),
        None,
        InputDemand(owned=(2, 4), needed=(1, 4)),
    )


def test_row_routes_include_owned_rows_outside_needed_context() -> None:
    pieces = row_range_summary_slices(
        4,
        (0, 4),
        ((1, InputDemand(owned=(2, 3), needed=(0, 1))),),
    )

    assert [(piece.start, piece.stop, piece.owned) for piece in pieces] == [
        (0, 1, False),
        (2, 3, True),
    ]


def _range_resolver(context, spec):
    async def resolve(chunk, bounds, requests):
        return await _range_pieces(
            context,
            spec,
            chunk,
            bounds,
            requests,
        )

    return resolve


@pytest.mark.parametrize(
    "needed,owned,expected",
    [
        (_RangeBounds(1, 2), (1, 3), [1, 2]),
        (_RangeBounds(None, None), (0, 4), [0, 1, 2, 3]),
    ],
)
def test_range_exchange_routes_local_context(
    spmd_engine, needed, owned, expected
) -> None:
    async def run() -> None:
        context = spmd_engine.context
        stream = context.br().stream_pool.get_stream()
        closed = False

        async def source():
            nonlocal closed
            try:
                yield (
                    _message([0, 1, 2, 3], context, stream),
                    _RangeSupply((0, 4), _RangeBounds(0, 3)),
                )
            finally:
                closed = True

        collective = SliceExchange(
            context,
            spmd_engine.comm,
            0,
        )
        request = InputDemand(owned=owned, needed=needed)
        routes = tuple((source,) for source in range(spmd_engine.comm.nranks))
        prepared = await collective.exchange(
            source(),
            ExchangePlan.from_routes(
                spmd_engine.comm.rank,
                routes,
                ((spmd_engine.comm.rank, request),),
            ),
            resolver=SliceResolver(resolve_chunk=_range_resolver(context, _RANGE_SPEC)),
        )
        try:
            (msg,) = tuple(prepared.local_chunks)
            assert _values(msg, context) == expected
            assert prepared.local_owned == (True,)
            assert closed
        finally:
            prepared.clear()

    asyncio.run(run())


def test_range_exchange_closes_source_on_error(spmd_engine) -> None:
    async def run() -> None:
        context = spmd_engine.context
        stream = context.br().stream_pool.get_stream()
        closed = False

        async def source():
            nonlocal closed
            try:
                yield (
                    _message([0, 1], context, stream),
                    _RangeSupply((0, 2), _RangeBounds(0, 1)),
                )
                yield (
                    _message([2, 3], context, stream),
                    _RangeSupply((2, 4), _RangeBounds(2, 3)),
                )
            finally:
                closed = True

        demand = _RangeBounds(0, 3)
        collective = SliceExchange(context, spmd_engine.comm, 0)
        spec = _RangeSliceSpec(
            index_position=1,
            index_dtype=plc.DataType(plc.TypeId.INT64),
            start_closed=True,
            end_closed=True,
        )
        request = InputDemand(owned=(0, 4), needed=demand)
        routes = tuple((source,) for source in range(spmd_engine.comm.nranks))
        with pytest.raises(IndexError):
            await collective.exchange(
                source(),
                ExchangePlan.from_routes(
                    spmd_engine.comm.rank,
                    routes,
                    ((spmd_engine.comm.rank, request),),
                ),
                resolver=SliceResolver(resolve_chunk=_range_resolver(context, spec)),
            )
        assert closed

    asyncio.run(run())


@pytest.mark.spmd
def test_range_exchange_routes_remote_context(spmd_engine) -> None:
    if spmd_engine.comm.nranks != 2:
        pytest.skip("This test expects exactly two ranks.")

    async def run() -> None:
        context = spmd_engine.context
        rank = spmd_engine.comm.rank
        stream = context.br().stream_pool.get_stream()
        supply = _RangeBounds(0, 4) if rank == 0 else None
        demand = _RangeBounds(3, 4) if rank == 1 else None

        async def source():
            if rank == 0:
                yield (
                    _message([0, 1, 2, 3, 4], context, stream),
                    _RangeSupply((0, 5), _RangeBounds(0, 4)),
                )

        with reserve_op_id() as op_id:
            collective = SliceExchange(context, spmd_engine.comm, op_id)
            counts, supplies, needed, validations = await _gather_ranges(
                context,
                spmd_engine.comm,
                op_id,
                5 if rank == 0 else 0,
                supply,
                demand,
                RangeValidation.VALID,
            )
            assert counts == (5, 0)
            assert all(status is RangeValidation.VALID for status in validations)
            demands: tuple[InputDemand[tuple[int, int], _RangeBounds] | None, ...] = (
                tuple(
                    InputDemand(owned=None, needed=rank_needed)
                    if rank_needed is not None
                    else None
                    for rank_needed in needed
                )
            )
            spans = tuple(pairwise(accumulate(counts, initial=0)))
            rank_supplies = tuple(
                _RangeSupply(span, keys) if keys is not None else None
                for span, keys in zip(spans, supplies, strict=True)
            )
            prepared = await collective.exchange(
                source(),
                _plan_range_overlap(rank, rank_supplies, demands),
                resolver=SliceResolver(
                    resolve_chunk=_range_resolver(context, _RANGE_SPEC)
                ),
            )
            try:
                tagged = [
                    (_values(message, context), owned)
                    async for message, owned in prepared.iter_tagged_source_order(
                        context, rank
                    )
                ]
                values = [value for chunk, _ in tagged for value in chunk]
                assert values == ([3, 4] if rank == 1 else [])
                assert [owned for _, owned in tagged] == ([False] if rank == 1 else [])
            finally:
                prepared.clear()

    asyncio.run(run())


@pytest.mark.spmd
def test_exchange_preserves_remote_ownership_tags(spmd_engine) -> None:
    if spmd_engine.comm.nranks != 2:
        pytest.skip("This test expects exactly two ranks.")

    async def run() -> None:
        context = spmd_engine.context
        rank = spmd_engine.comm.rank
        stream = context.br().stream_pool.get_stream()

        async def source():
            if rank == 0:
                yield _message([0, 1], context, stream), (0, 2)
                yield _message([2, 3], context, stream), (2, 4)

        demand = InputDemand(owned=(2, 3), needed=(0, 4))
        plan = ExchangePlan.from_routes(
            rank,
            ((1,), ()),
            ((1, demand),) if rank == 0 else (),
        )
        with reserve_op_id() as op_id:
            prepared = await SliceExchange(context, spmd_engine.comm, op_id).exchange(
                source(),
                plan,
                resolver=SliceResolver(resolve_chunk=row_range_slices),
            )
            tagged = [
                (_values(message, context), owned)
                async for message, owned in prepared.iter_tagged_source_order(
                    context, rank
                )
            ]
        assert tagged == (
            [([0, 1], False), ([2], True), ([3], False)] if rank == 1 else []
        )

    asyncio.run(run())
