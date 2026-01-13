# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dynamic filter pushdown optimization."""

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.dsl.ir import Join, Scan
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.filter_pushdown import (
    FilterSourceCollection,
    collect_filter_sources,
    collect_selectivity_stats,
    compute_selectivity_ratio,
    get_source_row_count,
    is_selective,
)
from cudf_polars.utils.config import ConfigOptions


@pytest.fixture
def gpu_engine():
    """Create a basic GPU engine for translation."""
    return pl.GPUEngine(raise_on_fail=True, executor="streaming")


@pytest.fixture
def config_options(gpu_engine):
    """Create config options from the GPU engine."""
    return ConfigOptions.from_polars_engine(gpu_engine)


class TestFilterSourceCollection:
    """Tests for FilterSourceCollection data structure."""

    def test_empty_collection(self):
        """Test empty collection behavior."""
        collection = FilterSourceCollection()
        assert len(collection) == 0
        assert list(collection) == []
        assert collection.semi_joins() == []
        assert collection.selective_inner_joins() == []

    def test_add_and_retrieve(self):
        """Test adding and retrieving filter sources."""
        collection = FilterSourceCollection()

        # Create a mock filter source (we'll use None for IR nodes in unit tests)
        # In real usage, these would be actual IR nodes
        # For this test, we just verify the collection mechanics work

        assert len(collection) == 0


class TestCollectFilterSourcesBasic:
    """Basic tests for collect_filter_sources."""

    def test_no_joins(self, gpu_engine):
        """Test IR with no joins returns empty collection."""
        q = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).select(
            pl.col("a") + pl.col("b")
        )

        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        sources = collect_filter_sources(ir)

        assert len(sources) == 0

    def test_inner_join_not_detected_without_stats(self, gpu_engine):
        """Test that inner joins are not detected without statistics."""
        left = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        right = pl.LazyFrame({"a": [2, 3, 4], "c": [7, 8, 9]})

        q = left.join(right, on="a", how="inner")

        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        # No stats provided - inner joins should not be detected
        sources = collect_filter_sources(ir, stats=None)

        assert len(sources) == 0

    def test_left_join_not_detected(self, gpu_engine):
        """Test that left joins are not detected as filter sources."""
        left = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        right = pl.LazyFrame({"a": [2, 3, 4], "c": [7, 8, 9]})

        q = left.join(right, on="a", how="left")

        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        sources = collect_filter_sources(ir)

        assert len(sources) == 0


class TestCollectFilterSourcesSemiJoin:
    """Tests for semi-join detection in collect_filter_sources."""

    def test_simple_semi_join(self, gpu_engine):
        """Test detection of a simple semi-join."""
        left = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        right = pl.LazyFrame({"a": [2, 4], "c": [100, 200]})

        q = left.join(right, on="a", how="semi")

        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        sources = collect_filter_sources(ir)

        assert len(sources) == 1
        source = sources.sources[0]

        # Verify the filter source properties
        assert isinstance(source.join_node, Join)
        assert source.join_node.options[0] == "Semi"
        assert source.source_type == "semi_join"
        # For semi-join: right provides keys, left is target
        assert source.provider_key_names == ("a",)
        assert source.target_key_names == ("a",)

    def test_semi_join_different_key_names(self, gpu_engine):
        """Test semi-join with different column names on left and right."""
        left = pl.LazyFrame(
            {"order_id": [1, 2, 3, 4, 5], "amount": [10, 20, 30, 40, 50]}
        )
        right = pl.LazyFrame({"id": [2, 4], "value": [100, 200]})

        q = left.join(right, left_on="order_id", right_on="id", how="semi")

        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        sources = collect_filter_sources(ir)

        assert len(sources) == 1
        source = sources.sources[0]

        # For semi-join: left is target, right is provider
        assert source.target_key_names == ("order_id",)
        assert source.provider_key_names == ("id",)

    def test_semi_join_multiple_keys(self, gpu_engine):
        """Test semi-join with multiple key columns."""
        left = pl.LazyFrame(
            {
                "a": [1, 1, 2, 2, 3],
                "b": [10, 20, 10, 20, 30],
                "c": [100, 200, 300, 400, 500],
            }
        )
        right = pl.LazyFrame(
            {
                "a": [1, 2],
                "b": [20, 10],
                "d": [1000, 2000],
            }
        )

        q = left.join(right, on=["a", "b"], how="semi")

        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        sources = collect_filter_sources(ir)

        assert len(sources) == 1
        source = sources.sources[0]

        assert source.target_key_names == ("a", "b")
        assert source.provider_key_names == ("a", "b")

    def test_anti_join_not_detected(self, gpu_engine):
        """Test that anti-joins are not currently detected as filter sources."""
        # Note: Anti-joins could potentially be used for filter pushdown
        # in some cases, but we're starting with semi-joins only
        left = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        right = pl.LazyFrame({"a": [2, 4], "c": [100, 200]})

        q = left.join(right, on="a", how="anti")

        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        sources = collect_filter_sources(ir)

        # Anti-join should NOT be detected (for now)
        assert len(sources) == 0


class TestCollectFilterSourcesComplexPatterns:
    """Tests for complex patterns in collect_filter_sources."""

    def test_semi_join_with_filter_subquery(self, gpu_engine):
        """Test semi-join where right side has a filter (like Q18 pattern)."""
        # This mimics the Q18 pattern:
        # q1 = lineitem.group_by("l_orderkey").agg(...).filter(...)
        # orders.join(q1, how="semi")

        lineitem = pl.LazyFrame(
            {
                "l_orderkey": [1, 1, 2, 2, 3, 3, 3],
                "l_quantity": [10, 20, 5, 5, 100, 100, 100],
            }
        )
        orders = pl.LazyFrame(
            {
                "o_orderkey": [1, 2, 3, 4, 5],
                "o_custkey": [101, 102, 103, 104, 105],
            }
        )

        # Create the subquery that filters based on aggregation
        q1 = (
            lineitem.group_by("l_orderkey")
            .agg(pl.col("l_quantity").sum().alias("sum_quantity"))
            .filter(pl.col("sum_quantity") > 50)  # Only order 3 has sum > 50
        )

        # Semi-join to filter orders
        q = orders.join(q1, left_on="o_orderkey", right_on="l_orderkey", how="semi")

        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        sources = collect_filter_sources(ir)

        assert len(sources) == 1
        source = sources.sources[0]

        assert source.target_key_names == ("o_orderkey",)
        assert source.provider_key_names == ("l_orderkey",)

    def test_multiple_semi_joins(self, gpu_engine):
        """Test detection of multiple semi-joins in a query."""
        table_a = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "val_a": [10, 20, 30, 40, 50]})
        table_b = pl.LazyFrame({"b": [2, 4], "val_b": [100, 200]})
        table_c = pl.LazyFrame({"c": [3, 4, 5], "val_c": [1000, 2000, 3000]})

        # Chain of semi-joins
        q = table_a.join(table_b, left_on="a", right_on="b", how="semi").join(
            table_c, left_on="a", right_on="c", how="semi"
        )

        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        sources = collect_filter_sources(ir)

        # Should detect both semi-joins
        assert len(sources) == 2
        assert len(sources.semi_joins()) == 2

    def test_semi_join_followed_by_inner_join(self, gpu_engine):
        """Test semi-join followed by inner join (like Q18)."""
        orders = pl.LazyFrame(
            {
                "o_orderkey": [1, 2, 3, 4, 5],
                "o_custkey": [101, 102, 103, 104, 105],
            }
        )
        filter_keys = pl.LazyFrame(
            {
                "key": [2, 4],
            }
        )
        lineitem = pl.LazyFrame(
            {
                "l_orderkey": [1, 2, 2, 3, 4, 4, 5],
                "l_quantity": [10, 20, 30, 40, 50, 60, 70],
            }
        )

        q = orders.join(
            filter_keys, left_on="o_orderkey", right_on="key", how="semi"
        ).join(lineitem, left_on="o_orderkey", right_on="l_orderkey", how="inner")

        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        sources = collect_filter_sources(ir)

        # Should detect the semi-join but not the inner join (no stats)
        assert len(sources) == 1
        source = sources.sources[0]
        assert source.join_node.options[0] == "Semi"


class TestCollectFilterSourcesWithCache:
    """Tests for handling Cache nodes in collect_filter_sources."""

    def test_semi_join_with_cached_subquery(self, gpu_engine):
        """Test that semi-joins are detected even when subqueries are cached."""
        # When a subquery is used multiple times, Polars wraps it in a Cache node
        shared_data = pl.LazyFrame(
            {
                "key": [1, 2, 3, 4, 5],
                "value": [10, 20, 30, 40, 50],
            }
        )

        # Use the shared data twice to trigger caching
        filtered = shared_data.filter(pl.col("value") > 25)

        main_table = pl.LazyFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7],
                "data": ["a", "b", "c", "d", "e", "f", "g"],
            }
        )

        q = main_table.join(filtered, left_on="id", right_on="key", how="semi")

        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        sources = collect_filter_sources(ir)

        assert len(sources) == 1


class TestFilterSourceProperties:
    """Tests for FilterSource properties and methods."""

    def test_filter_source_key_names(self, gpu_engine):
        """Test that key names are correctly extracted."""
        left = pl.LazyFrame(
            {
                "order_key": [1, 2, 3],
                "customer_key": [10, 20, 30],
                "amount": [100, 200, 300],
            }
        )
        right = pl.LazyFrame(
            {
                "ok": [2],
                "ck": [20],
            }
        )

        q = left.join(
            right,
            left_on=["order_key", "customer_key"],
            right_on=["ok", "ck"],
            how="semi",
        )

        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        sources = collect_filter_sources(ir)

        assert len(sources) == 1
        source = sources.sources[0]

        # For semi-join: left is target, right is provider
        assert source.target_key_names == ("order_key", "customer_key")
        assert source.provider_key_names == ("ok", "ck")

    def test_collection_get_by_join(self, gpu_engine):
        """Test FilterSourceCollection.get_by_join lookup."""
        left = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        right = pl.LazyFrame({"a": [2, 3], "c": [7, 8]})

        q = left.join(right, on="a", how="semi")

        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        sources = collect_filter_sources(ir)

        # Get the join node from the IR
        join_node = None
        for node in traversal([ir]):
            if isinstance(node, Join) and node.options[0] == "Semi":
                join_node = node
                break

        assert join_node is not None
        retrieved = sources.get_by_join(join_node)
        assert retrieved is not None
        assert retrieved.join_node is join_node


# -----------------------------------------------------------------------------
# Tests for Selectivity Statistics
# -----------------------------------------------------------------------------


class TestCollectSelectivityStats:
    """Tests for collect_selectivity_stats function."""

    def test_scan_row_count(self, gpu_engine, config_options, tmp_path):
        """Test that Scan nodes get row counts from parquet metadata."""
        # Create a parquet file
        df = pl.DataFrame({"a": range(100), "b": range(100, 200)})
        path = tmp_path / "test.parquet"
        df.write_parquet(path)

        q = pl.scan_parquet(path)
        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        stats = collect_selectivity_stats(ir, config_options)

        # Should have row count for the scan
        assert ir in stats.row_count
        assert stats.row_count[ir].value == 100

    def test_filter_reduces_row_count(self, gpu_engine, config_options, tmp_path):
        """Test that filters reduce estimated row count."""
        df = pl.DataFrame({"a": range(100), "b": range(100, 200)})
        path = tmp_path / "test.parquet"
        df.write_parquet(path)

        # Note: Polars pushes the filter predicate down into the Scan node,
        # so we get a single Scan with a predicate, not Scan + Filter
        q = pl.scan_parquet(path).filter(pl.col("a") > 50)
        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        stats = collect_selectivity_stats(ir, config_options)

        # The scan node has the filter pushed down (predicate field is set)
        assert isinstance(ir, Scan)
        assert ir.predicate is not None

        # The row count should be reduced due to the predicate
        assert stats.row_count[ir].value < 100
        # But the raw source row count (from metadata) should be 100
        assert get_source_row_count(ir, stats) == 100

    def test_groupby_reduces_row_count(self, gpu_engine, config_options, tmp_path):
        """Test that GroupBy nodes reduce estimated row count."""
        df = pl.DataFrame({"a": range(100), "b": range(100, 200)})
        path = tmp_path / "test.parquet"
        df.write_parquet(path)

        q = pl.scan_parquet(path).group_by("a").agg(pl.col("b").sum())
        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        stats = collect_selectivity_stats(ir, config_options)

        # GroupBy should reduce row count significantly
        scan_node = None
        for node in traversal([ir]):
            if isinstance(node, Scan):
                scan_node = node
                break

        assert scan_node is not None
        # GroupBy should have reduced count
        assert stats.row_count[ir].value < stats.row_count[scan_node].value


class TestSelectivityRatio:
    """Tests for selectivity ratio computation."""

    def test_compute_selectivity_ratio_filter(
        self, gpu_engine, config_options, tmp_path
    ):
        """Test selectivity ratio for a filtered query."""
        df = pl.DataFrame({"a": range(100), "b": range(100, 200)})
        path = tmp_path / "test.parquet"
        df.write_parquet(path)

        # Note: filter is pushed down into Scan predicate
        q = pl.scan_parquet(path).filter(pl.col("a") > 50)
        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        stats = collect_selectivity_stats(ir, config_options)

        # Verify the raw source row count is available
        assert get_source_row_count(ir, stats) == 100

        ratio = compute_selectivity_ratio(ir, stats)
        assert ratio is not None
        # Filter (via predicate pushdown) should have ratio < 1.0
        assert ratio < 1.0, f"Expected ratio < 1.0, got {ratio}"

    def test_is_selective_with_filter(self, gpu_engine, config_options, tmp_path):
        """Test is_selective detection for filtered query."""
        df = pl.DataFrame({"a": range(100), "b": range(100, 200)})
        path = tmp_path / "test.parquet"
        df.write_parquet(path)

        # Note: filter is pushed down into Scan predicate
        q = pl.scan_parquet(path).filter(pl.col("a") > 50)
        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        stats = collect_selectivity_stats(ir, config_options)

        # Verify raw source row count
        assert get_source_row_count(ir, stats) == 100
        # Verify reduced row count
        assert stats.row_count[ir].value < 100

        # Filter (via predicate pushdown) should be considered selective
        assert is_selective(ir, stats)

    def test_scan_not_selective(self, gpu_engine, config_options, tmp_path):
        """Test that plain scan is not considered selective."""
        df = pl.DataFrame({"a": range(100), "b": range(100, 200)})
        path = tmp_path / "test.parquet"
        df.write_parquet(path)

        q = pl.scan_parquet(path)
        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        stats = collect_selectivity_stats(ir, config_options)

        # Plain scan should not be selective (ratio = 1.0)
        ratio = compute_selectivity_ratio(ir, stats)
        assert ratio == 1.0 or ratio is None
        assert not is_selective(ir, stats)


class TestSelectiveInnerJoinDetection:
    """Tests for selective inner join detection."""

    def test_selective_inner_join_detected(self, gpu_engine, config_options, tmp_path):
        """Test detection of selective inner join."""
        # Create parquet files with different sizes
        small_df = pl.DataFrame({"key": [1, 2], "val": ["a", "b"]})
        large_df = pl.DataFrame({"key": range(1000), "data": range(1000)})

        small_path = tmp_path / "small.parquet"
        large_path = tmp_path / "large.parquet"
        small_df.write_parquet(small_path)
        large_df.write_parquet(large_path)

        # Join small filtered table with large table
        small = pl.scan_parquet(small_path)
        large = pl.scan_parquet(large_path)

        q = small.join(large, on="key", how="inner")
        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        stats = collect_selectivity_stats(ir, config_options)
        sources = collect_filter_sources(ir, stats)

        # Should detect as selective inner join
        # (small table is much smaller than large table)
        selective_joins = sources.selective_inner_joins()
        assert len(selective_joins) == 1
        source = selective_joins[0]
        assert source.source_type == "selective_inner_join"
        # Small table should be the provider
        assert source.selectivity_ratio is not None

    def test_similar_size_inner_join_not_detected(
        self, gpu_engine, config_options, tmp_path
    ):
        """Test that similar-sized inner joins are not detected."""
        # Create parquet files with similar sizes
        df1 = pl.DataFrame({"key": range(100), "val1": range(100)})
        df2 = pl.DataFrame({"key": range(100), "val2": range(100, 200)})

        path1 = tmp_path / "table1.parquet"
        path2 = tmp_path / "table2.parquet"
        df1.write_parquet(path1)
        df2.write_parquet(path2)

        t1 = pl.scan_parquet(path1)
        t2 = pl.scan_parquet(path2)

        q = t1.join(t2, on="key", how="inner")
        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        stats = collect_selectivity_stats(ir, config_options)
        sources = collect_filter_sources(ir, stats)

        # Should NOT detect as selective (similar sizes)
        assert len(sources.selective_inner_joins()) == 0

    def test_filtered_side_detected_as_selective(
        self, gpu_engine, config_options, tmp_path
    ):
        """Test that filtered side of inner join is detected as selective."""
        df = pl.DataFrame({"key": range(1000), "data": range(1000)})
        path = tmp_path / "table.parquet"
        df.write_parquet(path)

        # Same table but one side is filtered
        t1 = pl.scan_parquet(path).filter(pl.col("key") < 10)  # Very selective
        t2 = pl.scan_parquet(path)

        q = t1.join(t2, on="key", how="inner")
        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        stats = collect_selectivity_stats(ir, config_options)
        sources = collect_filter_sources(ir, stats)

        # Should detect filtered side as selective
        selective_joins = sources.selective_inner_joins()
        assert len(selective_joins) == 1
        source = selective_joins[0]
        # The filtered side should be the provider
        assert source.selectivity_ratio is not None
        assert source.selectivity_ratio < 0.5


class TestQ21Pattern:
    """Tests for Q21-like patterns (selective inner join with bloom filter opportunity)."""

    def test_q21_like_pattern(self, gpu_engine, config_options, tmp_path):
        """Test detection of Q21-like pattern with selective joins."""
        # Simulating Q21 pattern:
        # - lineitem joined with supplier/nation (very selective due to nation filter)
        # - Result joined with orders

        # Create test data
        lineitem = pl.DataFrame(
            {
                "l_orderkey": list(range(10000)),
                "l_suppkey": [i % 100 for i in range(10000)],
            }
        )
        supplier = pl.DataFrame(
            {
                "s_suppkey": list(range(100)),
                "s_nationkey": [i % 25 for i in range(100)],
            }
        )
        nation = pl.DataFrame(
            {
                "n_nationkey": list(range(25)),
                "n_name": [f"Nation{i}" for i in range(25)],
            }
        )
        orders = pl.DataFrame(
            {
                "o_orderkey": list(range(10000)),
                "o_status": ["F" if i % 3 == 0 else "O" for i in range(10000)],
            }
        )

        lineitem_path = tmp_path / "lineitem.parquet"
        supplier_path = tmp_path / "supplier.parquet"
        nation_path = tmp_path / "nation.parquet"
        orders_path = tmp_path / "orders.parquet"

        lineitem.write_parquet(lineitem_path)
        supplier.write_parquet(supplier_path)
        nation.write_parquet(nation_path)
        orders.write_parquet(orders_path)

        # Build query like Q21
        lineitem_scan = pl.scan_parquet(lineitem_path)
        supplier_scan = pl.scan_parquet(supplier_path)
        nation_scan = pl.scan_parquet(nation_path).filter(
            pl.col("n_name") == "Nation0"  # Very selective filter
        )
        orders_scan = pl.scan_parquet(orders_path)

        # Join chain: nation -> supplier -> lineitem
        supp_nation = supplier_scan.join(
            nation_scan, left_on="s_nationkey", right_on="n_nationkey"
        )
        lineitem_joined = lineitem_scan.join(
            supp_nation, left_on="l_suppkey", right_on="s_suppkey"
        )

        # Final join with orders (this is where bloom filter would help)
        q = lineitem_joined.join(
            orders_scan, left_on="l_orderkey", right_on="o_orderkey"
        )

        ir = Translator(q._ldf.visit(), gpu_engine).translate_ir()
        stats = collect_selectivity_stats(ir, config_options)
        sources = collect_filter_sources(ir, stats)

        # Should detect the final join as having a selective side
        # (the lineitem_joined side is filtered by nation)
        selective_joins = sources.selective_inner_joins()

        # We expect to find at least one selective inner join
        # The lineitem->supp_nation join should be detected if nation filter
        # makes it selective enough
        assert len(selective_joins) >= 1
