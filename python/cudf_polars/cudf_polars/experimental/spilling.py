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
    from rapidsmpf.buffer.buffer import MemoryType
    from rapidsmpf.integrations.dask.spilling import SpillableWrapper as Spillable

except ImportError:
    MemoryType = None  # type: ignore[no-redef]

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
    # Make headroom before executing the task (if possible)
    io_task = not args
    n_spilled = (
        len(
            [
                isinstance(obj, Spillable) and obj.mem_type() == MemoryType.HOST
                for obj in args
            ]
        )
        if MemoryType is not None
        else 0
    )
    if io_task or n_spilled:
        try:
            from rapidsmpf.integrations.dask.core import get_worker_context

            ctx = get_worker_context()
            with ctx.lock:
                headroom = 1_000_000_000 * (n_spilled + 1)
                available = ctx.br.memory_available(MemoryType.DEVICE)
                if available < headroom:
                    spilled = ctx.br.spill_manager.spill_to_make_headroom(
                        headroom=headroom
                    )
                    if io_task and (available + spilled) < headroom:
                        # Don't execute an IO task yet if we can't make headroom.
                        # We can't do this for non-IO tasks, because we don't want
                        # to move the input data.
                        from distributed.exceptions import Reschedule

                        raise Reschedule()

        except (AttributeError, ImportError, ValueError):
            pass

    # Execute the task
    output = do_evaluate(*non_child_args, *(unspill(arg) for arg in args))
    if spillable_output:
        return Spillable(on_device=output)
    return output
