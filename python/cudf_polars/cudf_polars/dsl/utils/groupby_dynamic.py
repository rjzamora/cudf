# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for dynamic groupby aggregations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.dsl import expr, ir
from cudf_polars.dsl.expressions.dynamic import (
    DynamicWindowLabel,
    duration_to_window_ticks,
)
from cudf_polars.dsl.utils.groupby import rewrite_groupby
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
    Rewrite supported dynamic groupby nodes to bucketed ordinary groupbys.

    This handles the fixed-width, non-overlapping case where each row belongs
    to exactly one dynamic window. The rewrite materializes a hidden bucket
    label column and then reuses normal grouped aggregation machinery.
    """
    _validate_supported_options(node.options)
    dynamic = node.options.dynamic
    index_name = dynamic.index_column
    index_dtype = inp.schema[index_name]
    every = duration_to_window_ticks(index_dtype, dynamic.every)
    period = duration_to_window_ticks(index_dtype, dynamic.period)
    offset = duration_to_window_ticks(index_dtype, dynamic.offset)
    if period != every:
        raise NotImplementedError("group_by_dynamic with period != every")

    if (n := len(keys)) > 0:
        # Grouped dynamic windows are emitted grouped by the equality keys.
        # Stable sorting by those keys preserves the per-group index order
        # required by Polars.
        inp = ir.Sort(
            inp.schema,
            keys,
            [plc.types.Order.ASCENDING] * n,
            [plc.types.NullOrder.BEFORE] * n,
            True,  # noqa: FBT003
            None,
            inp,
        )

    hidden_name = next(unique_names((*inp.schema.keys(), *schema.keys())))
    select_exprs = [
        expr.NamedExpr(name, expr.Col(dtype, name))
        for name, dtype in inp.schema.items()
    ]
    select_exprs.append(
        expr.NamedExpr(
            hidden_name,
            DynamicWindowLabel(
                index_dtype,
                every,
                offset,
                # Grouped dynamic windows require groupwise sortedness. This
                # expression can only validate the plain single-index case.
                not keys,
                expr.Col(index_dtype, index_name),
            ),
        )
    )
    select_schema = {**inp.schema, hidden_name: index_dtype}
    bucketed = ir.Select(select_schema, select_exprs, True, inp)  # noqa: FBT003
    dynamic_key = expr.NamedExpr(index_name, expr.Col(index_dtype, hidden_name))
    return rewrite_groupby(node, schema, [*keys, dynamic_key], aggs, bucketed)
