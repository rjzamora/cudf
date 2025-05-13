# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Select Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import HConcat, Select
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.dispatch import lower_ir_node
from cudf_polars.experimental.expressions import decompose_expr_graph
from cudf_polars.experimental.utils import _lower_ir_fallback

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer
    from cudf_polars.utils.config import ConfigOptions


def decompose_select(
    select_ir: Select,
    input_ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Decompose a multi-partition Select operation.

    Parameters
    ----------
    select_ir
        The original Select operation to decompose.
        This object has not been reconstructed with
        ``input_ir`` as its child yet.
    input_ir
        The lowered child of ``select_ir``. This object
        will be decomposed into a "partial" selection
        for each element of  ``select_ir.exprs``.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    new_ir, partition_info
        The rewritten Select node, and a mapping from
        unique nodes in the new graph to associated
        partitioning information.

    Notes
    -----
    This function uses ``decompose_expr_graph`` to further
    decompose each element of  ``select_ir.exprs``.

    See Also
    --------
    decompose_expr_graph
    """
    # Collect partial selections
    selections = []
    for ne in select_ir.exprs:
        # Decompose this partial expression
        new_ne, partial_input_ir, _partition_info = decompose_expr_graph(
            ne, input_ir, partition_info, config_options
        )
        pi = _partition_info[partial_input_ir]
        partial_input_ir = Select(
            {ne.name: ne.value.dtype},
            [new_ne],
            True,  # noqa: FBT003
            partial_input_ir,
        )
        _partition_info[partial_input_ir] = pi
        partition_info.update(_partition_info)
        selections.append(partial_input_ir)

    # Concatenate partial selections
    new_ir: HConcat | Select
    if len(selections) > 1:
        new_ir = HConcat(
            select_ir.schema,
            True,  # noqa: FBT003
            *selections,
        )
        partition_info[new_ir] = PartitionInfo(
            count=max(partition_info[c].count for c in selections)
        )
    else:
        new_ir = selections[0]

    return new_ir, partition_info


from collections.abc import Sequence

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import NamedExpr
from cudf_polars.dsl.ir import IR, broadcast
from cudf_polars.typing import Schema


class FusedSelect(IR):
    """Fused selection."""

    __slots__ = ("selections", "should_broadcast")
    _non_child = ("schema", "selections", "should_broadcast")
    selections: Sequence[Sequence[NamedExpr]]
    """List of expressions to evaluate to form the new dataframe."""
    should_broadcast: Sequence[bool]
    """Should columns be broadcast?"""

    def __init__(
        self,
        schema: Schema,
        selections: Sequence[Sequence[NamedExpr]],
        should_broadcast: Sequence[bool],
        df: IR,
    ):
        self.schema = schema
        self.selections = tuple(tuple(exprs) for exprs in selections)
        self.should_broadcast = tuple(should_broadcast)
        self.children = (df,)
        self._non_child_args = (self.selections, should_broadcast)

    @classmethod
    def do_evaluate(
        cls,
        selections: tuple[tuple[NamedExpr, ...], ...],
        should_broadcast: tuple[bool, ...],
        df: DataFrame,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        # Handle any broadcasting
        result = df
        for exprs, bcast in zip(selections, should_broadcast, strict=False):
            columns = [e.evaluate(result) for e in exprs]
            if bcast:
                columns = broadcast(*columns)
            result = DataFrame(columns)
        return result


@lower_ir_node.register(Select)
def _(
    ir: Select, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    child, partition_info = rec(ir.children[0])
    pi = partition_info[child]
    if pi.count > 1 and not all(
        expr.is_pointwise for expr in traversal([e.value for e in ir.exprs])
    ):
        try:
            # Try decomposing the underlying expressions
            return decompose_select(
                ir, child, partition_info, rec.state["config_options"]
            )
        except NotImplementedError:
            return _lower_ir_fallback(
                ir, rec, msg="This selection is not supported for multiple partitions."
            )

    if False:  # isinstance(child, Select):
        selections = (child.exprs, ir.exprs)
        should_broadcast = (child.should_broadcast, ir.should_broadcast)
        while True:
            (grandchild,) = child.children
            selections = (child.exprs, *selections)
            should_broadcast = (child.should_broadcast, *should_broadcast)
            if isinstance(grandchild, Select):
                child = grandchild
            else:
                break
        new_node = FusedSelect(
            ir.schema,
            selections,
            should_broadcast,
            grandchild,
        )
    elif False:  # isinstance(child, FusedSelect):
        (grandchild,) = child.children
        new_node = FusedSelect(
            ir.schema,
            (*child.selections, ir.exprs),
            (*child.should_broadcast, ir.should_broadcast),
            grandchild,
        )
    else:
        # new_node = FusedSelect(
        #     ir.schema,
        #     (ir.exprs,),
        #     (ir.should_broadcast,),
        #     child,
        # )
        new_node = ir.reconstruct([child])
    partition_info[new_node] = pi
    return new_node, partition_info
