# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import polars as pl

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize(
    "values,every",
    [
        pytest.param(list(range(10)), "3i", id="integer"),
        pytest.param(
            [datetime(2025, 1, 1) + timedelta(minutes=i) for i in range(10)],
            "3m",
            id="datetime",
        ),
    ],
)
def test_groupby_dynamic_fixed_width(streaming_engine_factory, values, every):
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=3, dynamic_planning={})
    )
    df = pl.LazyFrame(
        {
            "ts": values,
            "value": list(range(10)),
        }
    )

    q = df.group_by_dynamic("ts", every=every).agg(
        pl.col("value").sum().alias("value_sum")
    )

    assert_gpu_result_equal(q, engine=engine)


def test_groupby_dynamic_fixed_width_grouped(streaming_engine_factory):
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=3, dynamic_planning={})
    )
    df = pl.LazyFrame(
        {
            "ts": list(range(10)),
            "sym": ["a", "a", "b", "a", "b", "b", "a", "b", "a", "b"],
            "value": list(range(10)),
        }
    )

    q = (
        df.group_by_dynamic("ts", every="3i", group_by="sym")
        .agg(pl.col("value").sum().alias("value_sum"))
        .sort(["ts", "sym"])
    )

    assert_gpu_result_equal(q, engine=engine)


def test_groupby_dynamic_raises(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "dt": [
                datetime(2021, 12, 31, 0, 0, 0),
                datetime(2022, 1, 1, 0, 0, 1),
                datetime(2022, 3, 31, 0, 0, 1),
                datetime(2022, 4, 1, 0, 0, 1),
            ]
        }
    )

    q = (
        df.sort("dt")
        .group_by_dynamic("dt", every="1q")
        .agg(pl.col("dt").count().alias("num_values"))
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)
