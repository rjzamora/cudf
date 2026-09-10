# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expression nodes used by dynamic-window rewrites."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from cudf_polars.containers import DataFrame

__all__ = ["DynamicWindowLabel", "duration_to_window_ticks"]

_TIME_UNIT_NS = {
    plc.TypeId.TIMESTAMP_NANOSECONDS: 1,
    plc.TypeId.TIMESTAMP_MICROSECONDS: 1_000,
    plc.TypeId.TIMESTAMP_MILLISECONDS: 1_000_000,
}


def _is_zero_duration(duration: tuple[int, int, int, int, bool, bool]) -> bool:
    months, weeks, days, nanoseconds, _, _ = duration
    return months == weeks == days == nanoseconds == 0


def duration_to_window_ticks(
    dtype: DataType,
    duration: tuple[int, int, int, int, bool, bool],
) -> int:
    """
    Convert a polars dynamic-window duration into physical column ticks.

    Fixed-width dynamic windows are currently supported for integer indices
    using parsed integer durations (``"Ni"``) and timestamp indices using
    sub-day fixed durations. Calendar durations are deliberately rejected here.
    """
    months, weeks, days, nanoseconds, parsed_int, negative = duration
    if _is_zero_duration(duration):
        return 0
    if months != 0 or weeks != 0 or days != 0:
        raise NotImplementedError("calendar durations in group_by_dynamic")

    tid = dtype.id()
    if parsed_int:
        if tid not in {plc.TypeId.INT32, plc.TypeId.INT64}:
            raise NotImplementedError("integer dynamic windows require integer index")
        value = nanoseconds
    else:
        if tid in {plc.TypeId.INT32, plc.TypeId.INT64}:
            raise pl.exceptions.InvalidOperationError(
                "integer dynamic windows require parsed integer durations"
            )
        try:
            unit_ns = _TIME_UNIT_NS[tid]
        except KeyError:
            raise NotImplementedError(
                "unsupported dynamic-window index dtype"
            ) from None
        if nanoseconds % unit_ns != 0:
            raise NotImplementedError(
                "dynamic-window duration is smaller than index time unit"
            )
        value = nanoseconds // unit_ns

    return -value if negative else value


class DynamicWindowLabel(Expr):
    """Compute the left label for a fixed-width dynamic window."""

    __slots__ = ("check_sorted", "every", "offset")
    _non_child = ("dtype", "every", "offset", "check_sorted")
    check_sorted: bool
    every: int
    offset: int

    def __init__(
        self,
        dtype: DataType,
        every: int,
        offset: int,
        check_sorted: bool,  # noqa: FBT001
        index: Expr,
    ) -> None:
        self.dtype = dtype
        self.every = every
        self.offset = offset
        self.check_sorted = check_sorted
        self.children = (index,)
        self.is_pointwise = True
        if every <= 0:
            raise NotImplementedError("non-positive group_by_dynamic window")

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (index_expr,) = self.children
        index = index_expr.evaluate(df, context=context)
        if index.obj.null_count() != 0:
            raise pl.exceptions.ComputeError(
                "null values in dynamic group_by not supported, fill nulls"
            )
        if self.check_sorted and not index.check_sorted(
            order=plc.types.Order.ASCENDING,
            null_order=plc.types.NullOrder.BEFORE,
            stream=df.stream,
        ):
            raise pl.exceptions.InvalidOperationError(
                "argument in operation 'group_by_dynamic' is not sorted, "
                "please sort the 'expr/series/column' first"
            )

        int64_dtype = DataType(pl.Int64())
        int64 = int64_dtype.plc_type
        work = index.obj
        if self.dtype.id() != plc.TypeId.INT64:
            work = index.astype(int64_dtype, stream=df.stream).obj

        every = plc.Scalar.from_py(self.every, int64, stream=df.stream)
        if self.offset != 0:
            offset = plc.Scalar.from_py(self.offset, int64, stream=df.stream)
            work = plc.binaryop.binary_operation(
                work,
                offset,
                plc.binaryop.BinaryOperator.SUB,
                int64,
                stream=df.stream,
            )

        labels = plc.binaryop.binary_operation(
            work,
            every,
            plc.binaryop.BinaryOperator.FLOOR_DIV,
            int64,
            stream=df.stream,
        )
        labels = plc.binaryop.binary_operation(
            labels,
            every,
            plc.binaryop.BinaryOperator.MUL,
            int64,
            stream=df.stream,
        )
        if self.offset != 0:
            labels = plc.binaryop.binary_operation(
                labels,
                offset,
                plc.binaryop.BinaryOperator.ADD,
                int64,
                stream=df.stream,
            )

        out = Column(labels, dtype=int64_dtype).sorted_like(index)
        if self.dtype.id() != plc.TypeId.INT64:
            out = out.astype(self.dtype, stream=df.stream)
        return out
