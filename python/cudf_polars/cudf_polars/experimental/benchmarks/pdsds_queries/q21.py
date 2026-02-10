# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 21."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 21."""
    return """
    SELECT
             *
    FROM    (
                      SELECT   w_warehouse_name ,
                               i_item_id ,
                               Sum(
                               CASE
                                        WHEN (Cast(d_date AS DATE) < Cast('2000-03-11' AS DATE)) THEN inv_quantity_on_hand
                                        ELSE 0
                               END) AS inv_before ,
                               Sum(
                               CASE
                                        WHEN (Cast(d_date AS DATE) >= Cast('2000-03-11' AS DATE)) THEN inv_quantity_on_hand
                                        ELSE 0
                               END) AS inv_after
                      FROM     inventory ,
                               warehouse ,
                               item ,
                               date_dim
                      WHERE    i_current_price BETWEEN 0.99 AND 1.49
                      AND      i_item_sk = inv_item_sk
                      AND      inv_warehouse_sk = w_warehouse_sk
                      AND      inv_date_sk = d_date_sk
                      AND      d_date BETWEEN Cast('2000-02-10' AS DATE) AND Cast('2000-04-10' AS DATE)
                      GROUP BY w_warehouse_name,
                               i_item_id) x
    WHERE    (CASE
                    WHEN inv_before > 0 THEN (inv_after*1.000) / inv_before
                    ELSE NULL
             END) BETWEEN 2.000/3.000 AND 3.000/2.000
    ORDER BY w_warehouse_name NULLS FIRST,
             i_item_id NULLS FIRST
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 21."""
    # Load tables
    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    pivot = date(2000, 3, 11)
    start_date = date(2000, 2, 10)
    end_date = date(2000, 4, 10)
    return (
        inventory.join(item, left_on="inv_item_sk", right_on="i_item_sk")
        .join(warehouse, left_on="inv_warehouse_sk", right_on="w_warehouse_sk")
        .join(date_dim, left_on="inv_date_sk", right_on="d_date_sk")
        .filter(
            (pl.col("i_current_price").is_between(0.99, 1.49))
            & (pl.col("d_date").is_between(start_date, end_date, closed="both"))
        )
        .with_columns(
            [
                pl.when(pl.col("d_date") < pivot)
                .then(pl.col("inv_quantity_on_hand"))
                .otherwise(0)
                .alias("inv_before_amount"),
                pl.when(pl.col("d_date") >= pivot)
                .then(pl.col("inv_quantity_on_hand"))
                .otherwise(0)
                .alias("inv_after_amount"),
            ]
        )
        .group_by(["w_warehouse_name", "i_item_id"])
        .agg(
            [
                # Cast -> Decimal to match DuckDB
                pl.col("inv_before_amount").sum().alias("inv_before"),
                pl.col("inv_after_amount").sum().alias("inv_after"),
            ]
        )
        .filter(
            pl.when(pl.col("inv_before") > 0)
            .then(pl.col("inv_after") / pl.col("inv_before"))
            .otherwise(None)
            .is_between(2.0 / 3.0, 3.0 / 2.0)
        )
        .sort(["w_warehouse_name", "i_item_id"], nulls_last=False)
        .limit(100)
    )
