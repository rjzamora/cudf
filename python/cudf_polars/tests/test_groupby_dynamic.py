# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import datetime

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


def test_groupby_dynamic_ms_buckets(engine: pl.GPUEngine):
    lf = pl.LazyFrame(
        {
            "sym": ["A", "A", "A", "B", "B"],
            "ts": [
                datetime(2025, 1, 2, 9, 30, 0, 100),
                datetime(2025, 1, 2, 9, 30, 0, 200),
                datetime(2025, 1, 2, 9, 30, 0, 1000),
                datetime(2025, 1, 2, 9, 30, 0, 50),
                datetime(2025, 1, 2, 9, 30, 0, 150),
            ],
            "px": [10.0, 11.0, 12.0, 20.0, 21.0],
        }
    ).sort("sym", "ts")

    q = (
        lf.group_by_dynamic(
            "ts",
            every="1ms",
            period="1ms",
            group_by="sym",
        )
        .agg(
            hi=pl.col("px").max(),
            n=pl.len(),
        )
        .sort("sym", "ts")
    )
    assert_gpu_result_equal(q, engine=engine)


def test_groupby_dynamic_integer_buckets(engine: pl.GPUEngine):
    lf = pl.LazyFrame(
        {
            "idx": [0, 1, 4, 5, 8, 10, 11],
            "x": [1, 2, 3, 4, 5, 6, 7],
        }
    )

    q = (
        lf.group_by_dynamic("idx", every="5i", period="5i")
        .agg(pl.col("x").sum().alias("x_sum"), pl.len().alias("n"))
        .sort("idx")
    )
    assert_gpu_result_equal(q, engine=engine)


def test_groupby_dynamic_interleaved_groups_preserve_output_order(
    engine: pl.GPUEngine,
):
    lf = pl.LazyFrame(
        {
            "sym": ["B", "A", "B", "A"],
            "idx": [0, 0, 10, 10],
            "x": [1, 2, 3, 4],
        }
    )

    q = lf.group_by_dynamic("idx", every="10i", period="10i", group_by="sym").agg(
        pl.col("x").sum()
    )
    assert_gpu_result_equal(q, engine=engine)


def test_groupby_dynamic_aggregates_original_index(engine: pl.GPUEngine):
    lf = pl.LazyFrame(
        {
            "ts": [
                datetime(2025, 1, 2, 9, 30, 0, 100),
                datetime(2025, 1, 2, 9, 30, 0, 200),
                datetime(2025, 1, 2, 9, 30, 0, 1000),
            ],
            "x": [1, 2, 3],
        }
    ).sort("ts")

    q = (
        lf.group_by_dynamic("ts", every="1ms", period="1ms")
        .agg(
            pl.col("ts").min().alias("min_ts"),
            pl.col("x").sum().alias("x_sum"),
        )
        .sort("ts")
    )
    assert_gpu_result_equal(q, engine=engine)


def test_groupby_dynamic_calendar_duration_raises(engine: pl.GPUEngine):
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


def test_groupby_dynamic_overlapping_window_raises(engine: pl.GPUEngine):
    q = (
        pl.LazyFrame({"idx": [0, 1, 2], "x": [1, 2, 3]})
        .group_by_dynamic("idx", every="1i", period="2i")
        .agg(pl.col("x").sum())
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)
