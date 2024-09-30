from __future__ import annotations

import dataclasses
from functools import cached_property
from typing import TYPE_CHECKING, Any

from cudf_polars.containers import Column, DataFrame, NamedColumn
from cudf_polars.dsl.ir import broadcast, IR, Scan
from cudf_polars.dsl.expr import Expr

import pylibcudf as plc

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence


__all__ = [
    "DaskNode",
    "ReadParquet",
    "Select",
    "SumAgg",
    "Col",
]


class DaskNode:
    """
    Dask-specific version of an IR or Expr node.

    A DaskNode object always stores a reference to an IR object.
    If the DaskNode will also store a reference to an Expr object
    if it implements the parallel logic for that Expr.
    """

    __slots__ = ("_ir", "_expr", "_name")
    _ir: IR
    """IR object liked to this node."""
    _expr: Expr | None
    """Expr object liked to this node (Optional)."""
    _name: str | None
    """Name of the column produced by this node (Optional)."""

    def __init__(
        self,
        ir: IR,
        expr: Expr | None = None,
        name: str | None = None,
    ):
        self._ir = ir
        self._expr = expr
        self._name = name

    @cached_property
    def _key(self) -> str:
        from dask.tokenize import tokenize

        name = type(self).__name__.lower()
        expr = self._expr
        token = tokenize(
            dataclasses.fields(self._ir),
            expr if expr is None else expr.get_hash(),
            ensure_deterministic=True,
        )
        return f"{name}-{token}"

    def _ir_dependencies(self):
        # Return IR dependencies of self._ir
        ir = self._ir
        return [
            getattr(ir, val.name) for val in dataclasses.fields(ir) if val.type == "IR"
        ]

    @property
    def _npartitions(self) -> int:
        if self._expr is None:
            raise NotImplementedError(f"Partition count for {type(self).__name__}")
        # A DaskNode linked only to an IR object must implement _npartitions
        return self._ir._dask_node()._npartitions

    def _tasks(self) -> MutableMapping[Any, Any]:
        # A DaskNode must implement _tasks
        raise NotImplementedError(f"Generate tasks for {type(self).__name__}")

    def _task_graph(self) -> MutableMapping[Any, Any]:
        # Build a task graph
        import toolz

        ir = self._ir
        stack = [ir]
        seen = set()
        layers = []
        while stack:
            node = stack.pop()
            dask_node = node._dask_node()
            if dask_node._key in seen:
                continue
            seen.add(dask_node._key)
            layers.append(dask_node._tasks())
            stack.extend(dask_node._ir_dependencies())
        dsk = toolz.merge(layers)

        # Add task to reduce output partitions
        key = self._key
        if self._npartitions > 1:
            dsk[key] = (
                DataFrame.concatenate,
                [(key, i) for i in range(self._npartitions)],
            )
        else:
            dsk[key] = (key, 0)

        return dsk


class ReadParquet(DaskNode):
    _ir: Scan

    @property
    def _npartitions(self):
        return len(self._ir.paths)

    @staticmethod
    def read_parquet(path, columns, nrows, skip_rows, predicate, schema):
        """Read parquet data."""
        tbl_w_meta = plc.io.parquet.read_parquet(
            plc.io.SourceInfo([path]),
            columns=columns,
            nrows=nrows,
            skip_rows=skip_rows,
        )
        df = DataFrame.from_table(
            tbl_w_meta.tbl,
            # TODO: consider nested column names?
            tbl_w_meta.column_names(include_children=False),
        )
        assert all(c.obj.type() == schema[c.name] for c in df.columns)
        if predicate is None:
            return df
        else:
            (mask,) = broadcast(predicate.evaluate(df), target_length=df.num_rows)
            return df.filter(mask)

    def _tasks(self) -> MutableMapping[Any, Any]:
        key = self._key
        with_columns = self._ir.with_columns
        n_rows = self._ir.n_rows
        skip_rows = self._ir.skip_rows
        predicate = self._ir.predicate
        schema = self._ir.schema
        return {
            (key, i): (
                self.read_parquet,
                path,
                with_columns,
                n_rows,
                skip_rows,
                predicate,
                schema,
            )
            for i, path in enumerate(self._ir.paths)
        }


class Select(DaskNode):
    @staticmethod
    def _op(columns, should_broadcast):
        if should_broadcast:
            columns = broadcast(*columns)
        return DataFrame(columns)

    @cached_property
    def _npartitions(self):
        # TODO: Convert this logic into a convenience function
        ir = self._ir
        df = self._ir.df
        col_dask_nodes = [e._dask_node(df) for e in ir.expr]
        col_npartitions = [c._npartitions for c in col_dask_nodes]
        npartitions = col_npartitions[0]
        # TODO: How to deal with mismatched partition counts?
        assert set(col_npartitions) == {npartitions}, "mismatched partitions"
        return npartitions

    def _tasks(self):
        import toolz

        ir = self._ir
        df = self._ir.df
        col_dask_nodes = [e._dask_node(df) for e in ir.expr]
        col_keys = [c._key for c in col_dask_nodes]
        col_graphs = [c._tasks() for c in col_dask_nodes]
        key = self._key
        dsk = {
            (key, i): (
                self._op,
                [(c_key, i) for c_key in col_keys],
                ir.should_broadcast,
            )
            for i in range(self._npartitions)
        }

        return toolz.merge([dsk, *col_graphs])


class Col(DaskNode):
    @staticmethod
    def _op(df, name):
        return df._column_map[name]

    def _tasks(self):
        this_key = self._key
        this_name = self._expr.name
        ir_node = self._ir._dask_node()
        ir_key = ir_node._key
        return {
            (this_key, i): (self._op, (ir_key, i), this_name)
            for i in range(self._npartitions)
        }


class SumAgg(DaskNode):
    @property
    def _npartitions(self):
        return 1  # Assume reduction

    @staticmethod
    def _chunk(
        column: Column, request: plc.aggregation.Aggregation, dtype: plc.DataType
    ) -> Column:
        # TODO: This logic should be different than `request` in many cases
        return Column(
            plc.Column.from_scalar(
                plc.reduce.reduce(column.obj, request, dtype),
                1,
            )
        )

    @staticmethod
    def _concat(columns: Sequence[Column]) -> Column:
        return Column(plc.concatenate.concatenate([col.obj for col in columns]))

    @staticmethod
    def _finalize(
        column: Column,
        request: plc.aggregation.Aggregation,
        dtype: plc.DataType,
        name: str,
    ) -> Column:
        # TODO: This logic should be different than `request` in many cases
        obj = Column(
            plc.Column.from_scalar(
                plc.reduce.reduce(column.obj, request, dtype),
                1,
            )
        )
        return NamedColumn(
            obj.obj,
            name,
            is_sorted=obj.is_sorted,
            order=obj.order,
            null_order=obj.null_order,
        )

    def _tasks(self) -> MutableMapping[tuple[str, int], Any]:
        try:
            expr = self._expr
            npartitions = self._ir._dask_node()._npartitions
            child = expr.children[0]
            child_dask_node = child._dask_node(self._ir, name=expr.name)
            child_dsk = child_dask_node._tasks()
            key = self._key
            child_key = child_dask_node._key

            # Simple all-to-one reduction
            chunk_key = f"chunk-{key}"
            concat_key = f"concat-{key}"
            dsk: MutableMapping[tuple[str, int], Any] = {
                (chunk_key, i): (
                    self._chunk,
                    child_dsk[(child_key, i)],
                    expr.request,
                    expr.dtype,
                )
                for i in range(npartitions)
            }
            dsk[(concat_key, 0)] = (self._concat, list(dsk.keys()))
            dsk[(key, 0)] = (
                self._finalize,
                (concat_key, 0),
                expr.request,
                expr.dtype,
                self._name,
            )
        except Exception as err:
            import pdb

            pdb.set_trace()
            pass
        return dsk
