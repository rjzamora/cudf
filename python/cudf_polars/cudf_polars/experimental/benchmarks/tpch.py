# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Experimental TPC-H benchmarks."""

from __future__ import annotations

import argparse
import time
from datetime import date

import polars as pl

from cudf_polars.dsl.translate import Translator
from cudf_polars.experimental.parallel import evaluate_dask

parser = argparse.ArgumentParser(
    prog="Cudf-Polars TPC-H Benchmarks",
    description="Experimental Dask-Executor benchmarks.",
)
parser.add_argument(
    "query",
    type=int,
    choices=[1, 5, 6, 9, 10, 18],
    help="Query number.",
)
parser.add_argument(
    "--path",
    type=str,
    default="/datasets/rzamora/data/tpch-data/scale-30.0",
    help="Root directory path.",
)
parser.add_argument(
    "--suffix",
    type=str,
    default=".parquet",
    help="Table file suffix.",
)
parser.add_argument(
    "-e",
    "--executor",
    default="dask",
    type=str,
    choices=["dask", "dask-cuda", "dask-experimental", "pylibcudf", "polars"],
    help="Executor.",
)
parser.add_argument(
    "--n-workers",
    default=1,
    type=int,
    help="Number of Dask-CUDA workers (requires dask-cuda executor).",
)
parser.add_argument(
    "--blocksize",
    default=1000**3,
    type=int,
    help="Approx. partition size.",
)
parser.add_argument(
    "--debug",
    default=False,
    action="store_true",
    help="Debug run.",
)
parser.add_argument(
    "--shuffle",
    default="rapidsmp",
    type=str,
    choices=["rapidsmp", "tasks"],
    help="Shuffle method to use for distributed execution.",
)
args = parser.parse_args()


def get_data(path, table_name, suffix=""):
    """Get table from dataset."""
    return pl.scan_parquet(f"{path}/{table_name}{suffix}")


def q1(args):
    """Query 1."""
    lineitem = get_data(args.path, "lineitem", args.suffix).cast(
        {
            "l_extendedprice": pl.Float64,
            "l_discount": pl.Float64,
            "l_tax": pl.Float64,
            "l_quantity": pl.Float64,
        }
    )

    var1 = date(1998, 9, 2)

    return (
        lineitem.filter(pl.col("l_shipdate") <= var1)
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            pl.sum("l_quantity").alias("sum_qty"),
            pl.sum("l_extendedprice").alias("sum_base_price"),
            (pl.col("l_extendedprice") * (1.0 - pl.col("l_discount")))
            .sum()
            .alias("sum_disc_price"),
            (
                pl.col("l_extendedprice")
                * (1.0 - pl.col("l_discount"))
                * (1.0 + pl.col("l_tax"))
            )
            .sum()
            .alias("sum_charge"),
            pl.mean("l_quantity").alias("avg_qty"),
            pl.mean("l_extendedprice").alias("avg_price"),
            pl.mean("l_discount").alias("avg_disc"),
            pl.len().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
    )


def q5(args):
    """Query 5."""
    path = args.path
    suffix = args.suffix
    customer = get_data(path, "customer", suffix)
    lineitem = get_data(path, "lineitem", suffix).cast(
        {
            "l_extendedprice": pl.Float64,
            "l_discount": pl.Float64,
        }
    )
    nation = get_data(path, "nation", suffix)
    orders = get_data(path, "orders", suffix)
    region = get_data(path, "region", suffix)
    supplier = get_data(path, "supplier", suffix)

    var1 = "ASIA"
    var2 = date(1994, 1, 1)
    var3 = date(1995, 1, 1)

    return (
        region.join(nation, left_on="r_regionkey", right_on="n_regionkey")
        .join(customer, left_on="n_nationkey", right_on="c_nationkey")
        .join(orders, left_on="c_custkey", right_on="o_custkey")
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .join(
            supplier,
            left_on=["l_suppkey", "n_nationkey"],
            right_on=["s_suppkey", "s_nationkey"],
        )
        .filter(pl.col("r_name") == var1)
        .filter(pl.col("o_orderdate").is_between(var2, var3, closed="left"))
        .with_columns(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).alias("revenue")
        )
        .group_by("n_name")
        .agg(pl.sum("revenue"))
        .sort(by="revenue", descending=True)
    )


def q6(args):
    """Query 6."""
    path = args.path
    suffix = args.suffix
    lineitem = get_data(path, "lineitem", suffix).cast(
        {
            "l_extendedprice": pl.Float64,
            "l_discount": pl.Float64,
            "l_quantity": pl.Float64,
        }
    )

    var1 = date(1994, 1, 1)
    var2 = date(1995, 1, 1)
    var3 = 0.05
    var4 = 0.07
    var5 = 24

    return (
        lineitem.filter(pl.col("l_shipdate").is_between(var1, var2, closed="left"))
        .filter(pl.col("l_discount").is_between(var3, var4))
        .filter(pl.col("l_quantity") < var5)
        .with_columns(
            (pl.col("l_extendedprice") * pl.col("l_discount")).alias("revenue")
        )
        .select(pl.sum("revenue"))
    )


def q9(args):
    """Query 9."""
    path = args.path
    suffix = args.suffix
    lineitem = get_data(path, "lineitem", suffix).cast(
        {
            "l_extendedprice": pl.Float64,
            "l_discount": pl.Float64,
            "l_quantity": pl.Float64,
        }
    )
    nation = get_data(path, "nation", suffix)
    orders = get_data(path, "orders", suffix)
    part = get_data(path, "part", suffix)
    partsupp = get_data(path, "partsupp", suffix).cast({"ps_supplycost": pl.Float64})
    supplier = get_data(path, "supplier", suffix)

    return (
        part.join(partsupp, left_on="p_partkey", right_on="ps_partkey")
        .join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .join(
            lineitem,
            left_on=["p_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
        )
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(nation, left_on="s_nationkey", right_on="n_nationkey")
        .filter(pl.col("p_name").str.contains("green"))
        .select(
            pl.col("n_name").alias("nation"),
            pl.col("o_orderdate").dt.year().alias("o_year"),
            (
                pl.col("l_extendedprice") * (1 - pl.col("l_discount"))
                - pl.col("ps_supplycost") * pl.col("l_quantity")
            ).alias("amount"),
        )
        .group_by("nation", "o_year")
        .agg(pl.sum("amount").round(2).alias("sum_profit"))
        .sort(by=["nation", "o_year"], descending=[False, True])
    )


def q10(args):
    """Query 10."""
    path = args.path
    suffix = args.suffix
    customer = get_data(path, "customer", suffix).cast({"c_acctbal": pl.Float64})
    lineitem = get_data(path, "lineitem", suffix).cast(
        {
            "l_extendedprice": pl.Float64,
            "l_discount": pl.Float64,
        }
    )
    nation = get_data(path, "nation", suffix)
    orders = get_data(path, "orders", suffix)

    var1 = date(1993, 10, 1)
    var2 = date(1994, 1, 1)

    return (
        customer.join(orders, left_on="c_custkey", right_on="o_custkey")
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .join(nation, left_on="c_nationkey", right_on="n_nationkey")
        .filter(pl.col("o_orderdate").is_between(var1, var2, closed="left"))
        .filter(pl.col("l_returnflag") == "R")
        .group_by(
            "c_custkey",
            "c_name",
            "c_acctbal",
            "c_phone",
            "n_name",
            "c_address",
            "c_comment",
        )
        .agg(
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
            .sum()
            .round(2)
            .alias("revenue")
        )
        .select(
            "c_custkey",
            "c_name",
            "revenue",
            "c_acctbal",
            "n_name",
            "c_address",
            "c_phone",
            "c_comment",
        )
        .sort(by="revenue", descending=True)
        .head(20)
    )


def q18(args):
    """Query 18."""
    path = args.path
    suffix = args.suffix
    customer = get_data(path, "customer", suffix)
    lineitem = get_data(path, "lineitem", suffix)
    orders = get_data(path, "orders", suffix)

    var1 = 300

    q1 = (
        lineitem.group_by("l_orderkey")
        .agg(pl.col("l_quantity").sum().alias("sum_quantity"))
        .filter(pl.col("sum_quantity") > var1)
    )

    return (
        orders.join(q1, left_on="o_orderkey", right_on="l_orderkey", how="semi")
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .join(customer, left_on="o_custkey", right_on="c_custkey")
        .group_by("c_name", "o_custkey", "o_orderkey", "o_orderdate", "o_totalprice")
        .agg(pl.col("l_quantity").sum().alias("col6"))
        .select(
            pl.col("c_name"),
            pl.col("o_custkey").alias("c_custkey"),
            pl.col("o_orderkey"),
            pl.col("o_orderdate").alias("o_orderdat"),
            pl.col("o_totalprice"),
            pl.col("col6"),
        )
        .sort(by=["o_totalprice", "o_orderdat"], descending=[True, False])
        .head(100)
    )


def run(args):
    """Run the benchmark once."""
    executor = args.executor

    if executor == "dask-cuda":
        try:
            from rapidsmp.integrations.dask import LocalRMPCluster as LocalCUDACluster

            kwargs = {
                "protocol": "ucx",
            }
        except ImportError:
            from dask_cuda import LocalCUDACluster

            kwargs = {
                "protocol": "ucx",
            }

        from distributed import Client

        client = Client(
            LocalCUDACluster(
                n_workers=args.n_workers,
                dashboard_address=":8585",
                **kwargs,
            )
        )
    else:
        client = None

    t0 = time.time()

    q_id = args.query
    if q_id == 1:
        q = q1(args)
    elif q_id == 5:
        q = q5(args)
    elif q_id == 6:
        q = q6(args)
    elif q_id == 9:
        q = q9(args)
    elif q_id == 10:
        q = q10(args)
    elif q_id == 18:
        q = q18(args)
    else:
        raise NotImplementedError(f"Query {q_id} not implemented.")

    if executor == "polars":
        result = q.collect()
    else:
        if executor == "pylibcudf":
            executor_options = {}
            engine_options = {}
        else:
            executor_options = {
                "parquet_blocksize": args.blocksize,
                "shuffle_method": args.shuffle,
                "bcast_join_limit": 2 if executor == "dask-cuda" else 32,
            }
            # Avoid chunked reading (Decimal hack/work-around)
            engine_options = {"parquet_options": {"chunked": False}}
        engine = pl.GPUEngine(
            raise_on_fail=True,
            executor="dask-experimental" if executor.startswith("dask") else executor,
            executor_options=executor_options,
            **engine_options,
        )
        if args.debug:
            ir = Translator(q._ldf.visit(), engine).translate_ir()
            if executor == "pylibcudf":
                result = ir.evaluate(cache={}).to_polars()
            elif executor.startswith("dask"):
                result = evaluate_dask(ir).to_polars()
        else:
            result = q.collect(engine=engine)

    t1 = time.time()
    print(result)
    print(f"time is {t1 - t0}")

    if client is not None:
        client.close(timeout=60)


if __name__ == "__main__":
    run(args)
