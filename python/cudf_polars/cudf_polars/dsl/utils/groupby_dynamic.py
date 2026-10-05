# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for dynamic grouped aggregations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl import expr, ir
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.expressions.dynamic import duration_to_window_ticks
from cudf_polars.dsl.utils.aggregations import apply_pre_evaluation
from cudf_polars.dsl.utils.naming import unique_names

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from cudf_polars.typing import Schema

__all__ = ["rewrite_dynamic_groupby"]


def _validate_supported_options(options: Any) -> None:
    dynamic = options.dynamic
    if dynamic.period != dynamic.every:
        raise NotImplementedError("group_by_dynamic with period != every")
    if dynamic.closed_window != "left":
        raise NotImplementedError("group_by_dynamic with closed != 'left'")
    if dynamic.label != "left":
        raise NotImplementedError("group_by_dynamic with label != 'left'")
    if dynamic.include_boundaries:
        raise NotImplementedError("group_by_dynamic include_boundaries")
    if dynamic.start_by not in {"window", "window_bound"}:
        raise NotImplementedError("group_by_dynamic with start_by != 'window'")


def rewrite_dynamic_groupby(
    node: Any,
    schema: Schema,
    keys: Sequence[expr.NamedExpr],
    aggs: Sequence[expr.NamedExpr],
    inp: ir.IR,
) -> ir.IR:
    """
    Rewrite supported dynamic groupby nodes to a dedicated IR node.

    This supports the fixed-width, non-overlapping case where each row belongs
    to exactly one dynamic window.
    """
    _validate_supported_options(node.options)
    dynamic = node.options.dynamic
    index_name = dynamic.index_column
    index_dtype = inp.schema[index_name]
    every = duration_to_window_ticks(index_dtype, dynamic.every)
    period = duration_to_window_ticks(index_dtype, dynamic.period)
    offset = duration_to_window_ticks(index_dtype, dynamic.offset)
    if period != every:  # pragma: no cover - validated above
        raise NotImplementedError("group_by_dynamic with period != every")

    dynamic_key = expr.NamedExpr(
        index_name,
        expr.Col(index_dtype, index_name),
    )
    aggs, group_schema, apply_post_evaluation = apply_pre_evaluation(
        schema,
        (*keys, dynamic_key),
        aggs,
        unique_names(schema.keys()),
        ExecutionContext.GROUPBY,
    )
    grouped = ir.GroupByDynamic(
        group_schema,
        index_name,
        index_dtype,
        every,
        offset,
        keys,
        aggs,
        node.options.slice,
        inp,
    )
    return apply_post_evaluation(grouped)
