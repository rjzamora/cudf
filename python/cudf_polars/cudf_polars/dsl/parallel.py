# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Parallel LogicalPlan nodes.
"""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, List, Protocol, runtime_checkable

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Scan

if TYPE_CHECKING:
    from collections.abc import MutableMapping


class PartitionInfo:
    """Partitioning information."""
    __slots__ = ("npartitions",)

    def __init__(self, npartitions: int):
        self.npartitions = npartitions


@runtime_checkable
class PartitionedIR(Protocol):
    _key: str
    _parts: PartitionInfo
    def _tasks(self) -> MutableMapping:
        raise NotImplementedError()


@singledispatch
def _make_partitioned(ir: IR) -> PartitionedIR:
    cls = type(ir)
    children: List[PartitionedIR] = []
    npartitions = 1
    for child in ir.children:
        _child: PartitionedIR = child if isinstance(child, PartitionedIR) else _make_partitioned(child)
        children.append(_child)
        npartitions = max(npartitions, _child._parts.npartitions)

    # We need a mechanism to check if the IR in question
    # is "partitionwise". Otherwise, we can only allow
    # single partitions for now.
    assert npartitions == 1

    class _Partitionwise(cls):
        __slots__ = cls.__slots__ + ("_key", "_parts",)
        _non_child = cls._non_child + ("_key", "_parts",)
        _key: str
        _parts: PartitionInfo

        def __init__(self, *args: Any):
            super().__init__(*args)
            self._key = f"{cls.__name__.lower()}-{hash(ir)}"
            self._parts = PartitionInfo(npartitions=npartitions)

        def _tasks(self):
            return {
                (self._key, i): (
                    self.do_evaluate,
                    *self._non_child_args,
                    *((child._key, i) for child in children),
                )
                for i in range(self._parts.npartitions)
            }

    _Partitionwise.__name__ = f"Par{cls.__name__}"
    return _Partitionwise(*ir._ctor_arguments(children))


def task_graph(_ir: IR) -> tuple[MutableMapping[str, Any], str]:
    """Construct a task graph."""
    from cudf_polars.dsl.traversal import traversal

    # Rewrite IR graph into a ParIR graph
    ir: PartitionedIR = _make_partitioned(_ir)

    dsk = {
        k: v
        for layer in [n._tasks() for n in traversal(ir)]
        for k, v in layer.items()
    }

    # Add task to reduce output partitions
    npartitions = ir._parts.npartitions
    if npartitions > 1:
        dsk[ir._key] = (
            DataFrame.concat,
            [(ir._key, i) for i in range(npartitions)],
        )
    else:
        dsk[ir._key] = (ir._key, 0)

    return dsk, ir._key


class ParScan(Scan):

    __slots__ = Scan.__slots__ + ("_key", "_parts",)
    _non_child = Scan._non_child + ("_key", "_parts",)
    _key: str
    _parts: PartitionInfo

    def __init__(self, *args: Any):
        super().__init__(*args)
        self._key = f"scan-{hash(self)}"
        if self.typ in ("parquet", "csv"):
            npartitions = len(self.paths)
        else:
            npartitions = 1
        self._parts = PartitionInfo(npartitions=npartitions)

    def _tasks(self) -> MutableMapping:
        name = self._key
        return {
            (name, i): (
                self.do_evaluate,
                self.schema,
                self.typ,
                self.reader_options,
                path,
                self.with_columns,
                self.skip_rows,
                self.n_rows,
                self.row_index,
                self.predicate,
            )
            for i, path in enumerate(self.paths)
        }

# @_make_partitioned.register(Scan)
# def _make_partitioned(scan: Scan) -> ParScan:
#     return ParScan(*scan._ctor_arguments([]))
