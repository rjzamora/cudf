# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

import polars as pl

import pylibcudf as plc
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.collectives.slice_exchange import (
    ExchangePlan,
    InputDemand,
    PreparedExchange,
    RoutedSlice,
    SliceExchange,
    SliceResolver,
    row_range_slices,
)
from cudf_polars.streaming.actor_graph.utils import ChunkStore

if TYPE_CHECKING:
    from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream


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


def _values(message: Message[TableChunk], context: Context) -> list[int]:
    chunk = TableChunk.from_message(message, br=context.br())
    return (
        DataFrame.from_table(
            chunk.table_view(),
            ["value"],
            [DataType(pl.Int64())],
            chunk.stream,
        )
        .to_polars()["value"]
        .to_list()
    )


@pytest.mark.spmd
def test_slice_exchange_splits_one_chunk_across_ranks(spmd_engine) -> None:
    """Route local and multiple remote slices from one source chunk."""
    if spmd_engine.comm.nranks != 3:
        pytest.skip("This test expects exactly three ranks.")

    async def run() -> None:
        context = spmd_engine.context
        rank = spmd_engine.comm.rank
        stream = context.br().stream_pool.get_stream()
        demands = tuple(
            (destination, InputDemand(owned=span, needed=span))
            for destination, span in enumerate(((0, 2), (2, 4), (4, 6)))
        )
        plan = ExchangePlan.from_routes(
            rank,
            ((0, 1, 2), (), ()),
            demands if rank == 0 else (),
        )

        async def source():
            if rank == 0:
                yield _message(list(range(6)), context, stream), (0, 6)

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
        assert tagged == [([2 * rank, 2 * rank + 1], True)]

    asyncio.run(run())


def test_slice_exchange_clear_after_partial_iteration(spmd_engine) -> None:
    """Release retained chunks when a consumer stops early."""
    if spmd_engine.comm.nranks != 1:
        pytest.skip("This test expects exactly one rank.")

    async def run() -> None:
        context = spmd_engine.context
        rank = spmd_engine.comm.rank
        stream = context.br().stream_pool.get_stream()
        demand = InputDemand(owned=(0, 4), needed=(0, 4))

        async def source():
            yield _message([0, 1], context, stream), (0, 2)
            yield _message([2, 3], context, stream), (2, 4)

        plan = ExchangePlan.from_routes(rank, ((rank,),), ((rank, demand),))
        prepared = await SliceExchange(context, spmd_engine.comm, 0).exchange(
            source(),
            plan,
            resolver=SliceResolver(resolve_chunk=row_range_slices),
        )
        iterator = prepared.iter_source_order(context, rank)
        async with aclosing(iterator):
            first = await anext(iterator)
            assert _values(first, context) == [0, 1]
        assert len(prepared.local_chunks) == 0
        assert prepared.exchange is None
        assert prepared.remote_sources == ()

    asyncio.run(run())


def test_prepared_exchange_clear_attempts_every_remote_source(spmd_engine) -> None:
    """Release later sources even if extracting an earlier source fails."""
    context = spmd_engine.context
    extracted: list[int] = []

    class FailingExchange:
        def extract(self, source: int):
            extracted.append(source)
            if source == 0:
                raise RuntimeError("failed extraction")
            return ()

    prepared = PreparedExchange(
        ChunkStore(context),
        (),
        cast("SparseAlltoall", FailingExchange()),
        (0, 1),
    )
    prepared.clear()

    assert extracted == [0, 1]
    assert prepared.exchange is None
    assert prepared.remote_sources == ()


@pytest.mark.spmd
def test_slice_exchange_cleans_remote_payload_after_resolver_failure(
    spmd_engine,
) -> None:
    """Finish and drain an exchange after failure follows a remote send."""
    if spmd_engine.comm.nranks != 2:
        pytest.skip("This test expects exactly two ranks.")

    async def run() -> None:
        context = spmd_engine.context
        rank = spmd_engine.comm.rank
        stream = context.br().stream_pool.get_stream()
        demand = InputDemand(owned=(0, 4), needed=(0, 4))
        plan = ExchangePlan.from_routes(
            rank,
            ((1,), ()),
            ((1, demand),) if rank == 0 else (),
        )
        calls = 0

        async def source():
            if rank == 0:
                yield _message([0, 1], context, stream), (0, 2)
                yield _message([2, 3], context, stream), (2, 4)

        async def resolve(chunk, _summary, requests):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("resolver failed")
            return [
                RoutedSlice(destination, 0, chunk.shape[0], owned=True)
                for destination, _ in requests
            ]

        with reserve_op_id() as op_id:
            exchange = SliceExchange(context, spmd_engine.comm, op_id)
            if rank == 0:
                with pytest.raises(RuntimeError, match="resolver failed"):
                    await exchange.exchange(
                        source(),
                        plan,
                        resolver=SliceResolver(resolve_chunk=resolve),
                    )
            else:
                prepared = await exchange.exchange(
                    source(),
                    plan,
                    resolver=SliceResolver(resolve_chunk=resolve),
                )
                prepared.clear()

    asyncio.run(run())


@pytest.mark.spmd
def test_slice_exchange_cleans_remote_payload_after_finish_failure(
    spmd_engine,
) -> None:
    """Drain collective output when completion itself reports failure."""
    if spmd_engine.comm.nranks != 2:
        pytest.skip("This test expects exactly two ranks.")

    async def run() -> None:
        context = spmd_engine.context
        rank = spmd_engine.comm.rank
        stream = context.br().stream_pool.get_stream()
        demand = InputDemand(owned=(0, 2), needed=(0, 2))
        plan = ExchangePlan.from_routes(
            rank,
            ((1,), ()),
            ((1, demand),) if rank == 0 else (),
        )

        async def source():
            if rank == 0:
                yield _message([0, 1], context, stream), (0, 2)

        with reserve_op_id() as op_id:
            exchange = SliceExchange(context, spmd_engine.comm, op_id)
            finish = exchange._finish

            async def fail_after_finish(collective) -> None:
                await finish(collective)
                raise RuntimeError("finish failed")

            with (
                patch.object(exchange, "_finish", side_effect=fail_after_finish),
                pytest.raises(RuntimeError, match="finish failed"),
            ):
                await exchange.exchange(
                    source(),
                    plan,
                    resolver=SliceResolver(resolve_chunk=row_range_slices),
                )

    asyncio.run(run())
