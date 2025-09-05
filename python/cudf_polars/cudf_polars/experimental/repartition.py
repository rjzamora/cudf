# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Repartitioning Logic."""

from __future__ import annotations

import itertools
from functools import partial
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl.ir import IR
from cudf_polars.experimental.base import get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.experimental.parallel import PartitionInfo
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import ConfigOptions


class Repartition(IR):
    """
    Repartition a DataFrame.

    Notes
    -----
    Repartitioning means that we are not modifying any
    data, nor are we reordering or shuffling rows. We
    are only changing the overall partition count. For
    now, we only support an N -> [1...N] repartitioning
    (inclusive). The output partition count is tracked
    separately using PartitionInfo.
    """

    __slots__ = ()
    _non_child = ("schema",)

    def __init__(self, schema: Schema, df: IR):
        self.schema = schema
        self._non_child_args = ()
        self.children = (df,)


@generate_ir_tasks.register(Repartition)
def _(
    ir: Repartition,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
) -> MutableMapping[Any, Any]:
    # Repartition an IR node.
    # Only supports rapartitioning to fewer (for now).

    (child,) = ir.children
    count_in = partition_info[child].count
    count_out = partition_info[ir].count

    assert config_options.executor.name == "streaming", (
        "in-memory executor not supported in generate_ir_tasks"
    )
    spillable_output = config_options.executor.rapidsmpf_spill

    if count_out > count_in:  # pragma: no cover
        raise NotImplementedError(
            f"Repartition {count_in} -> {count_out} not supported."
        )

    key_name = get_key_name(ir)
    n, remainder = divmod(count_in, count_out)
    # Spread remainder evenly over the partitions.
    offsets = [0, *itertools.accumulate(n + (i < remainder) for i in range(count_out))]
    child_keys = tuple(partition_info[child].keys(child))
    # If rapidsmpf_spill is enabled, wrap the output of _concat in a SpillableWrapper.
    # The output of _concat may be large, so we want it to be spillable.
    func = partial(_concat, spillable_output=True) if spillable_output else _concat
    return {
        (key_name, i): (
            func,
            *child_keys[offsets[i] : offsets[i + 1]],
        )
        for i in range(count_out)
    }
