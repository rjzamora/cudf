# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Spilling with RAPIDSMPF."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from cudf_polars.containers import DataFrame

try:
    # For now, we can only handle SpillableWrapper objects.
    from rapidsmpf.integrations.dask.spilling import SpillableWrapper as Spillable

except ImportError:

    class Spillable:  # type: ignore[no-redef]
        """Placeholder for Spillable."""


def unspill(obj: DataFrame | Spillable) -> DataFrame:
    """Unspill a wrapped DataFrame object."""
    return obj.unspill() if isinstance(obj, Spillable) else obj


def unspill_and_evaluate(
    do_evaluate: Callable[..., DataFrame],
    spillable_output: bool,  # noqa: FBT001
    non_child_args: tuple[Any, ...],
    *args: DataFrame | Spillable,
) -> DataFrame:
    """Call an IR.do_evaluate function with unspilled child inputs."""
    output = do_evaluate(*non_child_args, *(unspill(arg) for arg in args))
    if spillable_output:
        return Spillable(on_device=output)
    return output
