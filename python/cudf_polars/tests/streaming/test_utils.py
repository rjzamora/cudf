# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest

import polars as pl

import pylibcudf as plc
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl import expr
from cudf_polars.streaming.actor_graph.utils import RandomAccessChunkStore
from cudf_polars.streaming.utils import _leaf_column_names


def test_leaf_column_names():
    dt = DataType(pl.datatypes.Int32())
    a = expr.Col(dt, "a")
    b = expr.Literal(dt, 1)
    c = expr.Col(dt, "c")
    d = expr.BinOp(dt, plc.binaryop.BinaryOperator.ADD, a, b)
    e = expr.BinOp(dt, plc.binaryop.BinaryOperator.ADD, d, c)
    assert set(_leaf_column_names(e)) == {"a", "c"}


def _chunk_from_polars(spmd_engine, df: pl.DataFrame) -> TableChunk:
    context = spmd_engine.context
    stream = context.br().stream_pool.get_stream()
    cudf_df = DataFrame.from_polars(df, stream)
    return TableChunk.from_pylibcudf_table(
        cudf_df.table,
        stream,
        exclusive_view=True,
        br=context.br(),
    )


def _message_to_polars(spmd_engine, msg: Message) -> pl.DataFrame:
    context = spmd_engine.context
    chunk = TableChunk.from_message(msg, br=context.br())
    return DataFrame.from_table(
        chunk.table_view(),
        ["x"],
        [DataType(pl.Int64())],
        chunk.stream,
    ).to_polars()


@pytest.mark.spmd
def test_random_access_chunk_store_copy_does_not_consume(spmd_engine) -> None:
    context = spmd_engine.context

    async def run() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        store = RandomAccessChunkStore(context)
        mid = store.insert(
            Message(
                7,
                _chunk_from_polars(spmd_engine, pl.DataFrame({"x": [10, 20, 30, 40]})),
            )
        )
        first = await store.copy(mid, start=1, stop=3)
        second = await store.copy(mid, start=0, stop=1)
        original = store.extract(mid)
        return (
            _message_to_polars(spmd_engine, first),
            _message_to_polars(spmd_engine, second),
            _message_to_polars(spmd_engine, original),
        )

    first, second, original = asyncio.run(run())

    assert first["x"].to_list() == [20, 30]
    assert second["x"].to_list() == [10]
    assert original["x"].to_list() == [10, 20, 30, 40]


@pytest.mark.spmd
def test_random_access_chunk_store_release(spmd_engine) -> None:
    context = spmd_engine.context
    store = RandomAccessChunkStore(context)
    mid = store.insert(
        Message(
            7,
            _chunk_from_polars(spmd_engine, pl.DataFrame({"x": [10, 20, 30, 40]})),
        )
    )
    copied = asyncio.run(store.copy(mid, start=1, stop=2))

    assert _message_to_polars(spmd_engine, copied)["x"].to_list() == [20]
    store.release(mid)

    with pytest.raises(KeyError):
        store.extract(mid)
