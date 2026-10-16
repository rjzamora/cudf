# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition dynamic-groupby lowering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import GroupByDynamic
from cudf_polars.streaming.base import PartitionInfo
from cudf_polars.streaming.dispatch import lower_ir_node

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.dispatch import LowerIRTransformer


@lower_ir_node.register(GroupByDynamic)
def _(
    ir: GroupByDynamic, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """Lower a dynamic groupby without changing its input partition count."""
    child, partition_info = rec(ir.children[0])
    node = ir.reconstruct([child])
    partition_info[node] = PartitionInfo(count=partition_info[child].count)
    return node, partition_info
