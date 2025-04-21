# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Spilling in multi-partition Dask execution using RAPIDSMPF."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, overload

from rapidsmpf.integrations.dask.spilling import SpillableWrapper

from cudf_polars.containers import DataFrame

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from typing import Any


T = TypeVar("T")


class _Callable[T](Protocol):
    def __call__(self, *args: Any) -> T: ...


@overload
def wrap_arg(obj: DataFrame) -> SpillableWrapper[DataFrame]: ...


@overload
def wrap_arg(obj: T) -> T: ...


def wrap_arg(obj: DataFrame | T) -> SpillableWrapper[DataFrame] | T:
    """
    Make `obj` spillable if it is a DataFrame.

    Parameters
    ----------
    obj
        The object to be wrapped (if it is a DataFrame).

    Returns
    -------
    A SpillableWrapper if obj is a DataFrame, otherwise the original object.
    """
    if isinstance(obj, DataFrame):
        return SpillableWrapper(on_device=obj)
    return obj


def unwrap_arg(obj: SpillableWrapper[T] | T) -> T:
    """
    Unwraps a SpillableWrapper to retrieve the original object.

    Parameters
    ----------
    obj
        The object to be unwrapped.

    Returns
    -------
    The unwrapped obj is a SpillableWrapper, otherwise the original object.
    """
    if isinstance(obj, SpillableWrapper):
        # Add headroom and unspill
        try:
            import dask.sizeof
            from distributed import get_worker

            buffer_resource = get_worker()._rmp_buffer_resource
            headroom = dask.sizeof.sizeof(obj._on_host)
            buffer_resource.spill_manager.spill_to_make_headroom(headroom=headroom)
        except ValueError:
            pass
        return obj.unspill()
    return obj


@overload
def wrap_func_spillable(
    func: _Callable[DataFrame],
    *,
    make_func_output_spillable: Literal[True],
    leaf: bool = False,
) -> _Callable[SpillableWrapper[DataFrame]]: ...


@overload
def wrap_func_spillable(
    func: _Callable[T],
    *,
    make_func_output_spillable: bool,
    leaf: bool = False,
) -> _Callable[T]: ...


def wrap_func_spillable(
    func: _Callable[T] | _Callable[DataFrame],
    *,
    make_func_output_spillable: bool,
    leaf: bool = False,
) -> _Callable[T] | _Callable[SpillableWrapper[DataFrame]]:
    """
    Wraps a function to handle spillable DataFrames.

    Parameters
    ----------
    func
        The function to be wrapped.
    make_func_output_spillable
        Whether to wrap the function's output in a SpillableWrapper.

    Returns
    -------
    A wrapped function that processes spillable DataFrames.
    """

    def wrapper(*args: Any) -> T:
        from distributed import get_worker

        worker = get_worker()
        spill_manager = worker._rmp_buffer_resource.spill_manager
        if not hasattr(worker, "_cudf_polars_retries"):
            worker._cudf_polars_retries = 0

        while True:
            try:
                ret: Any = func(*(unwrap_arg(arg) for arg in args))
                worker._cudf_polars_retries = 0
            except MemoryError as err:
                # OOM Error - Try to spill as much as possible
                worker._cudf_polars_retries += 1
                spilled = spill_manager.spill_to_make_headroom(headroom=10_000_000_000)
                if spilled < 1_000_000 and leaf:
                    # Couldn't spill much.
                    # Reschedule the task if it's a leaf task (usually IO)
                    from distributed.exceptions import Reschedule

                    raise Reschedule()
                elif spilled < 1_000_000 or worker._cudf_polars_retries > 10:
                    raise err
            else:
                worker._cudf_polars_retries = 0
                break
        if make_func_output_spillable:
            ret = wrap_arg(ret)
        return ret

    return wrapper


def wrap_dataframe_in_spillable(
    graph: MutableMapping[Any, Any], ignore_key: str | tuple[str, int]
) -> MutableMapping[Any, Any]:
    """
    Wraps functions within a task graph to handle spillable DataFrames.

    Only supports flat task graphs where each DataFrame can be found in the
    outermost level. Currently, this is true for all cudf-polars task graphs.

    Parameters
    ----------
    graph
        Dask graph.
    ignore_key
        The key to ignore when wrapping function, typically the key of the
        output node.

    Returns
    -------
    A new task graph with wrapped functions.
    """
    ret = {}
    for key, task in graph.items():
        if isinstance(task, tuple) and task and callable(task[0]):
            leaf = not any(isinstance(arg, tuple) and arg in graph for arg in task[1:])
            ret[key] = (
                wrap_func_spillable(
                    task[0],
                    make_func_output_spillable=key != ignore_key,
                    leaf=leaf,
                ),
                *task[1:],
            )
        else:
            ret[key] = task
    return ret
