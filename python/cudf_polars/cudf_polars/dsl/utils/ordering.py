# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for recognizing expressions that derive ordering metadata."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.dsl.expr import Cast, Col, TemporalFunction
from cudf_polars.dsl.expressions.dynamic import DynamicWindowLabel

if TYPE_CHECKING:
    from cudf_polars.dsl.expressions.base import Expr

__all__ = ["OrderingDerivation", "ordering_derivation"]


@dataclass(frozen=True)
class OrderingDerivation:
    """A monotone output expression derived from a single input column."""

    source_name: str
    strict_boundaries: bool


def _is_order_transparent_cast(expr: Cast) -> bool:
    src_id = expr.children[0].dtype.id()
    dst_id = expr.dtype.id()
    if src_id == dst_id:
        return True
    return (
        src_id == plc.TypeId.INT64 and dst_id == plc.TypeId.TIMESTAMP_NANOSECONDS
    ) or (src_id == plc.TypeId.TIMESTAMP_NANOSECONDS and dst_id == plc.TypeId.INT64)


def _source_column_name(expr: Expr) -> str | None:
    while isinstance(expr, Cast) and _is_order_transparent_cast(expr):
        (expr,) = expr.children
    return expr.name if isinstance(expr, Col) else None


def _bucketed_derivation(source: Expr) -> OrderingDerivation | None:
    if (source_name := _source_column_name(source)) is None:
        return None
    # Adjacent input partitions can map to the same bucket label.
    return OrderingDerivation(source_name, strict_boundaries=False)


@singledispatch
def ordering_derivation(expr: Expr) -> OrderingDerivation | None:
    """Return ordering metadata derivable from an expression, if supported."""
    return None


@ordering_derivation.register
def _(expr: Cast) -> OrderingDerivation | None:
    if _is_order_transparent_cast(expr):
        return ordering_derivation(expr.children[0])
    return None


@ordering_derivation.register
def _(expr: TemporalFunction) -> OrderingDerivation | None:
    if expr.name is TemporalFunction.Name.Truncate:
        return _bucketed_derivation(expr.children[0])
    return None


@ordering_derivation.register
def _(expr: DynamicWindowLabel) -> OrderingDerivation | None:
    return _bucketed_derivation(expr.children[0])
