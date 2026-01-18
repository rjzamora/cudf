# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.dsl.ir import Cache
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.experimental.shuffle import Shuffle
from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
    assert_gpu_result_equal,
)
from cudf_polars.utils.config import ConfigOptions
from cudf_polars.utils.versions import POLARS_VERSION_LT_130


@pytest.fixture(scope="module")
def engine():
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 4,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "x": range(150),
            "xx": list(range(75)) * 2,
            "y": [1, 2, 3] * 50,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 30,
        }
    )


@pytest.mark.parametrize("op", ["sum", "mean", "len"])
@pytest.mark.parametrize("keys", [("y",), ("y", "z")])
def test_groupby(df, engine, op, keys):
    q = getattr(df.group_by(*keys), op)()
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("op", ["sum", "mean", "len"])
@pytest.mark.parametrize("keys", [("y",), ("y", "z")])
def test_groupby_single_partitions(df, op, keys):
    q = getattr(df.group_by(*keys), op)()
    assert_gpu_result_equal(
        q,
        engine=pl.GPUEngine(
            raise_on_fail=True,
            executor="streaming",
            executor_options={
                "max_rows_per_partition": int(1e9),
                "cluster": DEFAULT_CLUSTER,
                "runtime": DEFAULT_RUNTIME,
            },
        ),
        check_row_order=False,
    )


@pytest.mark.parametrize(
    "op", ["sum", "mean", "len", "count", "min", "max", "n_unique"]
)
@pytest.mark.parametrize("keys", [("y",), ("y", "z")])
def test_groupby_agg(df, engine, op, keys):
    agg = getattr(pl.col("x"), op)()
    q = df.group_by(*keys).agg(agg)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("op", ["sum", "mean", "len", "count"])
@pytest.mark.parametrize("keys", [("y",), ("y", "z")])
def test_groupby_agg_config_options(df, op, keys):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 4,
            # Trigger shuffle-based groupby
            "unique_fraction": {"z": 0.5},
            # Check that we can change the n-ary factor
            "groupby_n_ary": 8,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "shuffle_method": DEFAULT_RUNTIME,  # Names coincide
        },
    )
    agg = getattr(pl.col("x"), op)()
    if op in ("sum", "mean"):
        if POLARS_VERSION_LT_130:
            agg = agg.cast(pl.Float64)
        agg = agg.round(2)  # Unary test coverage
    q = df.group_by(*keys).agg(agg)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("fallback_mode", ["silent", "raise", "warn", "foo"])
def test_groupby_fallback(df, engine, fallback_mode):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "fallback_mode": fallback_mode,
            "max_rows_per_partition": 4,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )
    match = "Failed to decompose groupby aggs"

    q = df.group_by("y").median()

    if fallback_mode == "silent":
        ctx = contextlib.nullcontext()
    elif fallback_mode == "raise":
        ctx = pytest.raises(
            pl.exceptions.ComputeError
            if POLARS_VERSION_LT_130
            else NotImplementedError,
            match=match,
        )
    elif fallback_mode == "foo":
        ctx = pytest.raises(
            pl.exceptions.ComputeError,
            match="'foo' is not a valid StreamingFallbackMode",
        )
    else:
        ctx = pytest.warns(UserWarning, match=match)
    with ctx:
        assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_groupby_agg_literal(df, engine):
    q = df.group_by("y").agg(1)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "op",
    [
        pl.max("x") - pl.min("x"),
        pl.mean("x") * pl.sum("x"),
        pl.max("x") + pl.max("z"),
        pl.max("x") + 1,
    ],
)
def test_groupby_agg_binop(df: pl.LazyFrame, engine: pl.GPUEngine, op: pl.Expr) -> None:
    q = df.group_by("y").agg(op)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "op, column_name",
    [
        (pl.max("x") - pl.min("x"), "x__max_min"),
        (pl.mean("x"), "x__mean_sum"),
    ],
)
def test_groupby_agg_duplicate(
    engine: pl.GPUEngine, op: pl.Expr, column_name: str
) -> None:
    # Ensure that the column names we create internally don't collide with
    # the user's column names.
    df = pl.LazyFrame(
        {
            "x": [0, 1, 2, 3] * 2,
            column_name: [4, 5, 6, 7] * 2,
            "y": [1, 2, 1, 2] * 2,
        }
    )
    q = df.group_by("y").agg(op, pl.min(column_name))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_groupby_agg_empty(df: pl.LazyFrame, engine: pl.GPUEngine) -> None:
    q = df.group_by("y").agg()
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("zlice", [(0, 2), (2, 2), (-2, None)])
def test_groupby_then_slice(df: pl.LazyFrame, zlice: tuple[int, int]) -> None:
    # Use a custom engine with selectivity hint to reduce output partitions.
    # maintain_order=True requires single output partition.
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 4,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "selectivity_hints": {"GROUPBY ('y',)": 0.0001},
        },
    )
    df = pl.LazyFrame(
        {
            "x": [0, 1, 2, 3] * 2,
            "y": [1, 2, 1, 2] * 2,
        }
    )
    q = df.group_by("y", maintain_order=True).max().slice(*zlice)
    assert_gpu_result_equal(q, engine=engine)


def test_groupby_tree_reduction_with_hint() -> None:
    # Test N-ary tree reduction with selectivity hints.
    # With 16 partitions and n_ary=4, we need multiple reduction levels:
    # 16 -> 4 -> 1, which exercises the intermediate GroupBy creation.
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 2,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "groupby_n_ary": 4,
            "selectivity_hints": {"GROUPBY ('y',)": 0.0001},
        },
    )
    df = pl.LazyFrame(
        {
            "x": list(range(32)),
            "y": [1, 2, 3, 4] * 8,
        }
    )
    q = df.group_by("y").sum()
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_groupby_on_equality(df: pl.LazyFrame, engine: pl.GPUEngine) -> None:
    # See: https://github.com/rapidsai/cudf/issues/19152
    df = pl.LazyFrame(
        {
            "key1": [1, 1, 1, 2, 3, 1, 4, 6, 7],
            "key2": [2, 2, 2, 2, 6, 1, 4, 6, 8],
            "int32": pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=pl.Int32()),
        }
    )
    q = df.group_by(pl.col("key1") == pl.col("key2")).agg(pl.col("int32").sum())
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "values",
    [
        [1, None, 2, None],
        [1, None, None, None],
    ],
)
def test_mean_partitioned(values: list[int | None]) -> None:
    df = pl.LazyFrame(
        {
            "key1": [1, 1, 2, 2],
            "uint16_with_null": pl.Series(values, dtype=pl.UInt16()),
        }
    )

    q = df.group_by("key1").agg(pl.col("uint16_with_null").mean())
    assert_gpu_result_equal(
        q,
        engine=pl.GPUEngine(
            executor="streaming",
            executor_options={
                "max_rows_per_partition": 2,
                "cluster": DEFAULT_CLUSTER,
                "runtime": DEFAULT_RUNTIME,
            },
        ),
        check_row_order=False,
    )


def test_groupby_literal_with_stats_planning(df):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 4,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "stats_planning": {"use_reduction_planning": True},
        },
    )

    q = (
        df.group_by(
            pl.lit(True).alias("key"),  # noqa: FBT003
            maintain_order=False,
        )
        .agg(pl.col("x").sum())
        .drop("key")
    )

    assert_gpu_result_equal(q, engine=engine)


def test_cache_preserves_partitioning_groupby():
    """
    Test that Cache preserves partitioning from a groupby shuffle.

    After a groupby shuffles data by 'key', the Cache node should have
    partitioned_on=['key']. When the cached result is used in subsequent
    groupbys on the same key, no additional shuffles should be needed.

    Regression test for Cache nodes losing partitioning information,
    which caused unnecessary shuffles and significant performance degradation.
    """
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )

    df = pl.LazyFrame(
        {
            "key": [1, 2, 1, 2, 1, 2, 3, 3, 3],
            "value": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        }
    )

    # Create a grouped subexpression that will be cached
    grouped = df.group_by("key").agg(pl.col("value").sum().alias("total"))

    # Use the grouped result twice to trigger CSE/Cache
    q = pl.concat(
        [
            grouped.group_by("key").agg(pl.col("total").max()),
            grouped.group_by("key").agg(pl.col("total").min()),
        ]
    )

    config_options = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    lowered_ir, partition_info, _ = lower_ir_graph(ir, config_options)

    # Verify Cache exists
    num_cache = sum(1 for node in traversal([lowered_ir]) if isinstance(node, Cache))
    assert num_cache > 0, "Expected Cache nodes from CSE"

    # Cache should preserve partitioning on 'key'
    cache_partitioning = []
    for node in traversal([lowered_ir]):
        if isinstance(node, Cache):
            pi = partition_info.get(node)
            if pi:
                cache_partitioning.append([ne.name for ne in pi.partitioned_on])
    assert cache_partitioning == [["key"]], (
        f"Expected Cache to preserve partitioning on ['key'], "
        f"but got {cache_partitioning}"
    )

    # With preserved partitioning: only 1 shuffle needed (initial groupby)
    num_shuffles = sum(
        1 for node in traversal([lowered_ir]) if isinstance(node, Shuffle)
    )
    assert num_shuffles == 1, (
        f"Expected 1 shuffle (initial groupby only), but got {num_shuffles}. "
        f"Cache is not preserving partitioning information."
    )
