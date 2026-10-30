# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import OrderKey
from cudf_streaming.table_chunk import TableChunk

from cudf_polars.streaming.actor_graph.collectives.ordered import (
    KeyBounds,
    RankEndpoints,
    key_bounds_overlap,
    key_bounds_slices,
    key_bounds_summary_slices,
    plan_key_bounds_exchange,
    rank_ownership_intervals,
)
from cudf_polars.streaming.actor_graph.collectives.slice_exchange import (
    InputDemand,
    RoutedSlice,
)

_KEYS = (OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),)


def _table(values: list[int], stream) -> plc.Table:
    return plc.Table(
        [
            plc.Column.from_arrow(
                pl.Series("key", values, dtype=pl.Int64()),
                stream=stream,
            )
        ]
    )


def _bounds(
    stream,
    lower: int | None,
    upper: int | None,
    *,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> KeyBounds:
    return KeyBounds(
        _table([lower], stream) if lower is not None else None,
        _table([upper], stream) if upper is not None else None,
        lower_inclusive,
        upper_inclusive,
        stream,
    )


def _value(table: plc.Table | None, stream) -> int | None:
    if table is None:
        return None
    value = table.columns()[0].to_scalar(stream=stream).to_py(stream=stream)
    assert isinstance(value, int)
    return value


def test_key_bounds_overlap_respects_open_and_unbounded_ends(spmd_engine) -> None:
    stream = spmd_engine.context.br().stream_pool.get_stream()

    assert key_bounds_overlap(
        _bounds(stream, 0, 2),
        _bounds(stream, 2, 4),
        _KEYS,
        stream,
    )
    assert not key_bounds_overlap(
        _bounds(stream, 0, 2, upper_inclusive=False),
        _bounds(stream, 2, 4),
        _KEYS,
        stream,
    )
    assert key_bounds_overlap(
        _bounds(stream, None, 0),
        _bounds(stream, -1, None),
        _KEYS,
        stream,
    )


def test_rank_ownership_and_exchange_plan_handle_sparse_sources(spmd_engine) -> None:
    stream = spmd_engine.context.br().stream_pool.get_stream()
    sparse = RankEndpoints(_table([0, 2, 4, 6], stream), (0, 2), stream)
    intervals = rank_ownership_intervals(
        sparse,
        4,
        _KEYS,
        stream,
        clip_to_domain=False,
    )
    assert intervals[0] is not None
    assert (_value(intervals[0].lower, stream), _value(intervals[0].upper, stream)) == (
        None,
        4,
    )
    assert intervals[1] is None
    assert intervals[2] is not None
    assert (_value(intervals[2].lower, stream), _value(intervals[2].upper, stream)) == (
        4,
        6,
    )
    assert intervals[3] is not None
    assert (_value(intervals[3].lower, stream), _value(intervals[3].upper, stream)) == (
        6,
        None,
    )

    endpoints = RankEndpoints(
        _table([0, 3, 3, 6, 7, 9], stream),
        (0, 1, 2),
        stream,
    )
    demands = (
        _bounds(stream, 0, 4, upper_inclusive=False),
        None,
        _bounds(stream, 4, 8, upper_inclusive=False),
        _bounds(stream, 8, 10, upper_inclusive=False),
    )
    plan = plan_key_bounds_exchange(
        1,
        4,
        endpoints,
        demands,
        _KEYS,
        stream,
        transfer_ownership=True,
    )
    assert tuple(destination for destination, _ in plan.outgoing) == (0, 2)
    assert plan.sources == ()
    assert plan_key_bounds_exchange(
        2,
        4,
        endpoints,
        demands,
        _KEYS,
        stream,
        transfer_ownership=True,
    ).sources == (1,)

    overlapping = (_bounds(stream, 0, 5), _bounds(stream, 4, 8), None, None)
    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        plan_key_bounds_exchange(
            0,
            4,
            endpoints,
            overlapping,
            _KEYS,
            stream,
            transfer_ownership=True,
        )


def test_key_bounds_plan_orders_consumer_after_producer_stream(spmd_engine) -> None:
    pool = spmd_engine.context.br().stream_pool
    producer = pool.get_stream()
    consumer = pool.get_stream()
    endpoints = RankEndpoints(_table([0, 4, 5, 9], producer), (0, 1), producer)
    demands = (
        _bounds(producer, 0, 5, upper_inclusive=False),
        _bounds(producer, 5, 10, upper_inclusive=False),
    )

    plan = plan_key_bounds_exchange(
        0,
        2,
        endpoints,
        demands,
        _KEYS,
        consumer,
        transfer_ownership=True,
    )

    assert tuple(destination for destination, _ in plan.outgoing) == (0,)
    assert plan.sources == ()


def test_key_bounds_summary_matches_materialized_resolution(spmd_engine) -> None:
    context = spmd_engine.context
    stream = context.br().stream_pool.get_stream()
    supply = _bounds(stream, 0, 9)
    demand = InputDemand(
        owned=_bounds(stream, 3, 5, upper_inclusive=False),
        needed=_bounds(stream, 2, 7),
    )
    requests = ((0, demand),)

    assert key_bounds_summary_slices(10, supply, requests, _KEYS, stream) is None

    async def resolve() -> list[RoutedSlice]:
        chunk = TableChunk.from_pylibcudf_table(
            _table(list(range(10)), stream),
            stream,
            exclusive_view=True,
            br=context.br(),
        )
        return await key_bounds_slices(chunk, None, requests, _KEYS)

    assert asyncio.run(resolve()) == [
        RoutedSlice(0, 2, 3, owned=False),
        RoutedSlice(0, 3, 5, owned=True),
        RoutedSlice(0, 5, 8, owned=False),
    ]
    full = InputDemand(owned=supply, needed=supply)
    assert key_bounds_summary_slices(10, supply, ((0, full),), _KEYS, stream) == [
        RoutedSlice(0, 0, 10, owned=True)
    ]
    assert (
        key_bounds_summary_slices(
            10,
            supply,
            ((0, InputDemand(None, _bounds(stream, 10, 20))),),
            _KEYS,
            stream,
        )
        == []
    )
