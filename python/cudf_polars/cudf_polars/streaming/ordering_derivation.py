# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive ordering metadata through supported expressions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING

import pylibcudf as plc
from cudf_streaming.channel_metadata import OrderKey, Ordering
from cudf_streaming.table_chunk import TableChunk

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Cast, Col, TemporalFunction
from cudf_polars.dsl.expressions.datetime import (
    _TruncatePlan,
    _truncate_ordering_lower_bound,
)
from cudf_polars.dsl.traversal import traversal
from cudf_polars.dsl.utils.naming import indices_to_names, names_to_indices

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.expr import Expr, NamedExpr
    from cudf_polars.typing import Schema


@dataclass(frozen=True)
class OrderingDerivation:
    """Derive output ordering metadata through one expression."""

    named_expr: NamedExpr
    ordering: Ordering
    column_targets: Mapping[str, Mapping[str, None]]
    input_schema: Schema
    output_schema: Schema
    context: Context

    def _transform_values(
        self,
        values: TableChunk,
        old_key_names: tuple[str, ...],
        source_position: int,
    ) -> TableChunk:
        br = self.context.br()
        stream = values.stream
        frame = DataFrame.from_table(
            values.table_view(),
            old_key_names,
            [self.input_schema[name] for name in old_key_names],
            stream,
        )
        column = self.named_expr.evaluate(frame)
        table = plc.concatenate.concatenate(
            [
                plc.Table(
                    [
                        *values.table_view().columns()[:source_position],
                        column.obj,
                    ]
                )
            ],
            stream=stream,
            mr=br.device_mr,
        )
        return TableChunk.from_pylibcudf_table(
            table,
            stream,
            exclusive_view=True,
            br=br,
        )

    def derive_from_source(
        self,
        source_name: str,
        *,
        preserves_strictness: bool,
    ) -> Ordering | None:
        """Derive metadata for a proven nondecreasing source transformation."""
        old_key_names = indices_to_names(
            self.ordering.column_indices, self.input_schema
        )
        if any(
            isinstance(node, Col) and node.name != source_name
            for node in traversal([self.named_expr.value])
        ):
            return None
        try:
            source_position = old_key_names.index(source_name)
        except ValueError:
            return None

        prefix_names = old_key_names[:source_position]
        if not set(prefix_names).issubset(self.column_targets):
            return None
        target_prefix = tuple(
            name
            if name in self.column_targets[name]
            else next(iter(self.column_targets[name]))
            for name in prefix_names
        )
        target_key_names = (*target_prefix, self.named_expr.name)
        new_indices = names_to_indices(target_key_names, self.output_schema)
        boundaries = self._transform_values(
            self.ordering.get_boundaries(self.context.br()),
            old_key_names,
            source_position,
        )
        source_bounds = self.ordering.get_domain_bounds(self.context.br())
        domain_bounds = (
            None
            if source_bounds is None
            else self._transform_values(
                source_bounds,
                old_key_names,
                source_position,
            )
        )
        source_keys = self.ordering.keys[: source_position + 1]
        keys = tuple(
            OrderKey(index, key.order, key.null_order)
            for index, key in zip(new_indices, source_keys, strict=True)
        )
        return Ordering(
            keys,
            boundaries,
            domain_bounds=domain_bounds,
            strict_boundaries=(
                preserves_strictness and self.ordering.strict_boundaries
            ),
            locally_ordered=self.ordering.locally_ordered,
        )

    def derive(self) -> Ordering | None:
        """Return ordering metadata derived through this expression."""
        return _derive_ordering(self.named_expr.value, self)


def _is_order_transparent_cast(expression: Cast) -> bool:
    source_dtype = expression.children[0].dtype
    target_dtype = expression.dtype
    source_id = source_dtype.id()
    target_id = target_dtype.id()
    return (
        source_dtype == target_dtype
        or (
            source_id == plc.TypeId.INT64
            and target_id == plc.TypeId.TIMESTAMP_NANOSECONDS
        )
        or (
            source_id == plc.TypeId.TIMESTAMP_NANOSECONDS
            and target_id == plc.TypeId.INT64
        )
    )


def _source_column_name(expression: Expr) -> str | None:
    while isinstance(expression, Cast) and _is_order_transparent_cast(expression):
        (expression,) = expression.children
    return expression.name if isinstance(expression, Col) else None


@singledispatch
def _derive_ordering(
    expression: Expr,
    derivation: OrderingDerivation,
) -> Ordering | None:
    """Return ordering metadata derivable from an expression."""
    return None


@_derive_ordering.register
def _(expression: Cast, derivation: OrderingDerivation) -> Ordering | None:
    if not _is_order_transparent_cast(expression):
        return None
    source_name = _source_column_name(expression)
    if source_name is not None:
        return derivation.derive_from_source(
            source_name,
            preserves_strictness=True,
        )
    return _derive_ordering(expression.children[0], derivation)


@_derive_ordering.register
def _(expression: TemporalFunction, derivation: OrderingDerivation) -> Ordering | None:
    if expression.name is not TemporalFunction.Name.Truncate:
        return None
    source_name = _source_column_name(expression.children[0])
    if source_name is None:
        return None
    old_key_names = indices_to_names(
        derivation.ordering.column_indices, derivation.input_schema
    )
    if not old_key_names or old_key_names[0] != source_name:
        return None
    source_bounds = derivation.ordering.get_domain_bounds(derivation.context.br())
    if source_bounds is None:
        return None
    key = derivation.ordering.keys[0]
    lower_row = 0 if key.order is plc.types.Order.ASCENDING else 1
    source_column = source_bounds.table_view().columns()[0]
    source_type = source_column.type()
    if plc.traits.is_timestamp(source_type):
        integer_type = plc.DataType(
            plc.TypeId.INT32
            if source_type.id() == plc.TypeId.TIMESTAMP_DAYS
            else plc.TypeId.INT64
        )
        source_column = plc.unary.bit_cast(
            source_column, integer_type, stream=source_bounds.stream
        )
    lower = plc.copying.get_element(source_column, lower_row).to_py(
        stream=source_bounds.stream
    )
    plan = _TruncatePlan(*expression.options)
    if not isinstance(lower, int) or lower < _truncate_ordering_lower_bound(
        plan, expression.children[0].dtype
    ):
        return None
    derived = derivation.derive_from_source(source_name, preserves_strictness=False)
    if derived is None:
        return None
    bounds = derived.get_domain_bounds(derivation.context.br())
    assert bounds is not None
    key = derived.keys[0]
    if not plc.sorting.is_sorted(
        bounds.table_view(),
        [key.order],
        [key.null_order],
        stream=bounds.stream,
    ):
        return None

    return derived
