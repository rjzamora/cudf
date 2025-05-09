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
    # from cudf_polars.typing import Schema
    # from cudf_polars.experimental.utils import _leaf_column_names
    # from cudf_polars.dsl.expr import NamedExpr, Col

    pwise_named_exprs = []
    # single_agg_named_exprs = []

    # Collect partial selections
    selections = []
    for ne in select_ir.exprs:
        nonpwise = [expr for expr in traversal([ne.value]) if not expr.is_pointwise]
        if not nonpwise:
            # Everything is pointwise
            pwise_named_exprs.append(ne)
        # elif len(nonpwise) == 1 and isinstance(nonpwise[0], Agg):
        #     # Expr contains a single aggregation.
        #     # We should try to fuse this with other single-aggregations.
        #     # TODO: What about BinOp(Agg, Agg)?
        #     single_agg_named_exprs.append(ne)
        else:
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

    # Deal with pointwise selections
    if pwise_named_exprs:
        pwise = Select(
            {ne.name: select_ir.schema[ne.name] for ne in pwise_named_exprs},
            pwise_named_exprs,
            True,  # noqa: FBT003
            input_ir,
        )
        partition_info[pwise] = partition_info[input_ir]
        selections = [pwise, *selections]

    # if single_agg_named_exprs:
    #     pass
    #     # pwise = Select(
    #     #     pwise_schema,
    #     #     [NamedExpr(k, Col(v, k)) for k, v in pwise_schema.items()],
    #     #     True,
    #     #     input_ir,
    #     # )
    #     # partition_info[pwise] = partition_info[input_ir]
    #     # selections = [pwise] + selections

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

    new_node = ir.reconstruct([child])
    partition_info[new_node] = pi
    return new_node, partition_info
