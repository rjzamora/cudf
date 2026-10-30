# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.streaming.actor_graph.utils import gather_in_task_group
from cudf_polars.streaming.utils import _leaf_column_names


def test_leaf_column_names():
    dt = DataType(pl.datatypes.Int32())
    a = expr.Col(dt, "a")
    b = expr.Literal(dt, 1)
    c = expr.Col(dt, "c")
    d = expr.BinOp(dt, plc.binaryop.BinaryOperator.ADD, a, b)
    e = expr.BinOp(dt, plc.binaryop.BinaryOperator.ADD, d, c)
    assert set(_leaf_column_names(e)) == {"a", "c"}


def test_gather_in_task_group_cleans_completed_result_on_failure() -> None:
    """Release an owning result when a sibling task fails."""

    async def run() -> None:
        ready = asyncio.Event()
        resource = object()
        cleaned: list[object] = []

        async def succeed():
            ready.set()
            return resource

        async def fail():
            await ready.wait()
            raise RuntimeError("failed sibling")

        with pytest.raises(ExceptionGroup):
            await gather_in_task_group(
                succeed(),
                fail(),
                cleanup=cleaned.append,
            )
        assert cleaned == [resource]

    asyncio.run(run())


def test_gather_in_task_group_preserves_task_failure_when_cleanup_fails() -> None:
    """Do not replace the task failure with a secondary cleanup error."""

    async def run() -> None:
        ready = asyncio.Event()

        async def succeed():
            ready.set()
            return object()

        async def fail():
            await ready.wait()
            raise RuntimeError("task failed")

        def fail_cleanup(_result: object) -> None:
            raise RuntimeError("cleanup failed")

        with pytest.raises(ExceptionGroup) as exc_info:
            await gather_in_task_group(
                succeed(),
                fail(),
                cleanup=fail_cleanup,
            )
        (failure,) = exc_info.value.exceptions
        assert isinstance(failure, RuntimeError)
        assert str(failure) == "task failed"

    asyncio.run(run())
