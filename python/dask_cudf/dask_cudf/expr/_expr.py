# Copyright (c) 2024, NVIDIA CORPORATION.
import functools
import math

import dask_expr._shuffle as _shuffle_module
import pandas as pd
from dask_expr import new_collection
from dask_expr._cumulative import CumulativeBlockwise
from dask_expr._expr import Elemwise, Expr, RenameAxis, VarColumns
from dask_expr._groupby import (
    DecomposableGroupbyAggregation,
    GroupbyAggregation,
)
from dask_expr._reductions import Reduction, Var
from dask_expr.io.io import FusedParquetIO
from dask_expr.io.parquet import FragmentWrapper, ReadParquetPyarrowFS

from dask.dataframe.core import (
    _concat,
    is_dataframe_like,
    make_meta,
    meta_nonempty,
)
from dask.dataframe.dispatch import is_categorical_dtype
from dask.typing import no_default

import cudf

##
## Custom expressions
##


def _get_spec_info(gb):
    if isinstance(gb.arg, (dict, list)):
        aggs = gb.arg.copy()
    else:
        aggs = gb.arg

    if gb._slice and not isinstance(aggs, dict):
        aggs = {gb._slice: aggs}

    gb_cols = gb._by_columns
    if isinstance(gb_cols, str):
        gb_cols = [gb_cols]
    columns = [c for c in gb.frame.columns if c not in gb_cols]
    if not isinstance(aggs, dict):
        aggs = {col: aggs for col in columns}

    # Assert if our output will have a MultiIndex; this will be the case if
    # any value in the `aggs` dict is not a string (i.e. multiple/named
    # aggregations per column)
    str_cols_out = True
    aggs_renames = {}
    for col in aggs:
        if isinstance(aggs[col], str) or callable(aggs[col]):
            aggs[col] = [aggs[col]]
        elif isinstance(aggs[col], dict):
            str_cols_out = False
            col_aggs = []
            for k, v in aggs[col].items():
                aggs_renames[col, v] = k
                col_aggs.append(v)
            aggs[col] = col_aggs
        else:
            str_cols_out = False
        if col in gb_cols:
            columns.append(col)

    return {
        "aggs": aggs,
        "columns": columns,
        "str_cols_out": str_cols_out,
        "aggs_renames": aggs_renames,
    }


def _get_meta(gb):
    spec_info = gb.spec_info
    gb_cols = gb._by_columns
    aggs = spec_info["aggs"].copy()
    aggs_renames = spec_info["aggs_renames"]
    if spec_info["str_cols_out"]:
        # Metadata should use `str` for dict values if that is
        # what the user originally specified (column names will
        # be str, rather than tuples).
        for col in aggs:
            aggs[col] = aggs[col][0]
    _meta = gb.frame._meta.groupby(gb_cols).agg(aggs)
    if aggs_renames:
        col_array = []
        agg_array = []
        for col, agg in _meta.columns:
            col_array.append(col)
            agg_array.append(aggs_renames.get((col, agg), agg))
        _meta.columns = pd.MultiIndex.from_arrays([col_array, agg_array])
    return _meta


class DecomposableCudfGroupbyAgg(DecomposableGroupbyAggregation):
    sep = "___"

    @functools.cached_property
    def spec_info(self):
        return _get_spec_info(self)

    @functools.cached_property
    def _meta(self):
        return _get_meta(self)

    @property
    def shuffle_by_index(self):
        return False  # We always group by column(s)

    @classmethod
    def chunk(cls, df, *by, **kwargs):
        from dask_cudf.groupby import _groupby_partition_agg

        return _groupby_partition_agg(df, **kwargs)

    @classmethod
    def combine(cls, inputs, **kwargs):
        from dask_cudf.groupby import _tree_node_agg

        return _tree_node_agg(_concat(inputs), **kwargs)

    @classmethod
    def aggregate(cls, inputs, **kwargs):
        from dask_cudf.groupby import _finalize_gb_agg

        return _finalize_gb_agg(_concat(inputs), **kwargs)

    @property
    def chunk_kwargs(self) -> dict:
        dropna = True if self.dropna is None else self.dropna
        return {
            "gb_cols": self._by_columns,
            "aggs": self.spec_info["aggs"],
            "columns": self.spec_info["columns"],
            "dropna": dropna,
            "sort": self.sort,
            "sep": self.sep,
        }

    @property
    def combine_kwargs(self) -> dict:
        dropna = True if self.dropna is None else self.dropna
        return {
            "gb_cols": self._by_columns,
            "dropna": dropna,
            "sort": self.sort,
            "sep": self.sep,
        }

    @property
    def aggregate_kwargs(self) -> dict:
        dropna = True if self.dropna is None else self.dropna
        final_columns = self._slice or self._meta.columns
        return {
            "gb_cols": self._by_columns,
            "aggs": self.spec_info["aggs"],
            "columns": self.spec_info["columns"],
            "final_columns": final_columns,
            "as_index": True,
            "dropna": dropna,
            "sort": self.sort,
            "sep": self.sep,
            "str_cols_out": self.spec_info["str_cols_out"],
            "aggs_renames": self.spec_info["aggs_renames"],
        }


class CudfGroupbyAgg(GroupbyAggregation):
    @functools.cached_property
    def spec_info(self):
        return _get_spec_info(self)

    @functools.cached_property
    def _meta(self):
        return _get_meta(self)

    def _lower(self):
        return DecomposableCudfGroupbyAgg(
            self.frame,
            self.arg,
            self.observed,
            self.dropna,
            self.split_every,
            self.split_out,
            self.sort,
            self.shuffle_method,
            self._slice,
            *self.by,
        )


def _maybe_get_custom_expr(
    gb,
    aggs,
    split_every=None,
    split_out=None,
    shuffle_method=None,
    **kwargs,
):
    from dask_cudf.groupby import (
        OPTIMIZED_AGGS,
        _aggs_optimized,
        _redirect_aggs,
    )

    if kwargs:
        # Unsupported key-word arguments
        return None

    if not hasattr(gb.obj._meta, "to_pandas"):
        # Not cuDF-backed data
        return None

    _aggs = _redirect_aggs(aggs)
    if not _aggs_optimized(_aggs, OPTIMIZED_AGGS):
        # One or more aggregations are unsupported
        return None

    return CudfGroupbyAgg(
        gb.obj.expr,
        _aggs,
        gb.observed,
        gb.dropna,
        split_every,
        split_out,
        gb.sort,
        shuffle_method,
        gb._slice,
        *gb.by,
    )


class CudfReadParquetPyarrowFS(ReadParquetPyarrowFS):
    @functools.cached_property
    def _dataset_info(self):
        from dask_cudf.io.parquet import set_object_dtypes_from_pa_schema

        dataset_info = super()._dataset_info
        meta_pd = dataset_info["base_meta"]
        if isinstance(meta_pd, cudf.DataFrame):
            return dataset_info

        # Convert to cudf
        # (drop unsupported timezone information)
        for k, v in meta_pd.dtypes.items():
            if isinstance(v, pd.DatetimeTZDtype) and v.tz is not None:
                meta_pd[k] = meta_pd[k].dt.tz_localize(None)
        meta_cudf = cudf.from_pandas(meta_pd)

        # Re-set "object" dtypes to align with pa schema
        kwargs = dataset_info.get("kwargs", {})
        set_object_dtypes_from_pa_schema(
            meta_cudf,
            kwargs.get("schema", None),
        )

        dataset_info["base_meta"] = meta_cudf
        self.operands[type(self)._parameters.index("_dataset_info_cache")] = (
            dataset_info
        )
        return dataset_info

    @staticmethod
    def _table_to_pandas(table, index_name):
        if isinstance(table, cudf.DataFrame):
            df = table
        else:
            df = cudf.DataFrame.from_arrow(table)
        if index_name is not None:
            return df.set_index(index_name)
        return df

    @staticmethod
    def _fragments_to_cudf_dataframe(
        fragment_wrappers,
        filters,
        columns,
        schema,
    ):
        from dask.dataframe.io.utils import _is_local_fs

        from cudf.io.parquet import _apply_post_filters, _normalize_filters

        if not isinstance(fragment_wrappers, list):
            fragment_wrappers = [fragment_wrappers]

        filesystem = None
        paths, row_groups = [], []
        for fw in fragment_wrappers:
            frag = fw.fragment if isinstance(fw, FragmentWrapper) else fw
            paths.append(frag.path)
            row_groups.append(
                [rg.id for rg in frag.row_groups] if frag.row_groups else None
            )
            if filesystem is None:
                filesystem = frag.filesystem

        if _is_local_fs(filesystem):
            filesystem = None
        else:
            from fsspec.implementations.arrow import ArrowFSWrapper

            filesystem = ArrowFSWrapper(filesystem)
            protocol = filesystem.protocol
            paths = [f"{protocol}://{path}" for path in paths]

        filters = _normalize_filters(filters)
        if row_groups == [None for path in paths]:
            row_groups = None

        df = cudf.read_parquet(
            paths,
            columns=columns,
            filters=filters,
            row_groups=row_groups,
            dataset_kwargs={"schema": schema},
        )

        # Apply filters (if any are defined)
        df = _apply_post_filters(df, filters)

        # TODO: Deal with hive partitioning
        return df

    @functools.cached_property
    def _use_device_io(self):
        from dask.dataframe.io.utils import _is_local_fs

        # Use host for remote filesystem only, or
        # if KvikIO-S3 is enabled
        return _is_local_fs(self.fs) or (
            self.fs.type_name
            == "s3"  # TODO: and cudf.get_option("kvikio_remote_io")
        )

    def _filtered_task(self, index: int):
        columns = self.columns.copy()
        index_name = self.index.name
        if self.index is not None:
            index_name = self.index.name
        schema = self._dataset_info["schema"].remove_metadata()
        if index_name:
            if columns is None:
                columns = list(schema.names)
            columns.append(index_name)

        frag_to_table = self._fragment_to_table
        if self._use_device_io:
            frag_to_table = self._fragments_to_cudf_dataframe
        return (
            self._table_to_pandas,
            (
                frag_to_table,
                FragmentWrapper(self.fragments[index], filesystem=self.fs),
                self.filters,
                columns,
                schema,
            ),
            index_name,
        )

    def _tune_up(self, parent):
        if self._fusion_compression_factor >= 1:
            return
        fused_cls = (
            CudfFusedParquetIO
            if self._use_device_io
            else CudfFusedParquetIOHost
        )
        if isinstance(parent, fused_cls):
            return
        return parent.substitute(self, fused_cls(self))


class CudfFusedParquetIO(FusedParquetIO):
    @functools.cached_property
    def _fusion_buckets(self):
        partitions = self.operand("_expr")._partitions
        npartitions = len(partitions)

        step = math.ceil(1 / self.operand("_expr")._fusion_compression_factor)
        heuristic = max(math.ceil(math.sqrt(npartitions)), 10)
        step = min(step, heuristic, 100)

        buckets = [
            partitions[i : i + step] for i in range(0, npartitions, step)
        ]
        return buckets

    @classmethod
    def _load_multiple_files(
        cls,
        frag_filters,
        columns,
        schema,
        *to_pandas_args,
    ):
        frag_to_table = CudfReadParquetPyarrowFS._fragments_to_cudf_dataframe
        return CudfReadParquetPyarrowFS._table_to_pandas(
            frag_to_table(
                [frag[0] for frag in frag_filters],
                frag_filters[0][1],  # TODO: Check for consistent filters?
                columns,
                schema,
            ),
            *to_pandas_args,
        )


class CudfFusedParquetIOHost(CudfFusedParquetIO):
    @classmethod
    def _load_multiple_files(
        cls,
        frag_filters,
        columns,
        schema,
        *to_pandas_args,
    ):
        import pyarrow as pa

        from dask.base import apply, tokenize
        from dask.threaded import get

        token = tokenize(frag_filters, columns, schema)
        name = f"pq-file-{token}"
        dsk = {
            (name, i): (
                CudfReadParquetPyarrowFS._fragment_to_table,
                frag,
                filter,
                columns,
                schema,
            )
            for i, (frag, filter) in enumerate(frag_filters)
        }
        dsk[name] = (
            apply,
            pa.concat_tables,
            [list(dsk.keys())],
            {"promote_options": "permissive"},
        )

        return CudfReadParquetPyarrowFS._table_to_pandas(
            get(dsk, name),
            *to_pandas_args,
        )


class RenameAxisCudf(RenameAxis):
    # TODO: Remove this after rename_axis is supported in cudf
    # (See: https://github.com/rapidsai/cudf/issues/16895)
    @staticmethod
    def operation(df, index=no_default, **kwargs):
        if index != no_default:
            df.index.name = index
            return df
        raise NotImplementedError(
            "Only `index` is supported for the cudf backend"
        )


class ToCudfBackend(Elemwise):
    # TODO: Inherit from ToBackend when rapids-dask-dependency
    # is pinned to dask>=2024.8.1
    _parameters = ["frame", "options"]
    _projection_passthrough = True
    _filter_passthrough = True
    _preserves_partitioning_information = True

    @staticmethod
    def operation(df, options):
        from dask_cudf.backends import to_cudf_dispatch

        return to_cudf_dispatch(df, **options)

    def _simplify_down(self):
        if isinstance(
            self.frame._meta, (cudf.DataFrame, cudf.Series, cudf.Index)
        ):
            # We already have cudf data
            return self.frame


##
## Custom expression patching
##


# This can be removed after cudf#15176 is addressed.
# See: https://github.com/rapidsai/cudf/issues/15176
class PatchCumulativeBlockwise(CumulativeBlockwise):
    @property
    def _args(self) -> list:
        return self.operands[:1]

    @property
    def _kwargs(self) -> dict:
        # Must pass axis and skipna as kwargs in cudf
        return {"axis": self.axis, "skipna": self.skipna}


CumulativeBlockwise._args = PatchCumulativeBlockwise._args
CumulativeBlockwise._kwargs = PatchCumulativeBlockwise._kwargs


# The upstream Var code uses `Series.values`, and relies on numpy
# for most of the logic. Unfortunately, cudf -> cupy conversion
# is not supported for data containing null values. Therefore,
# we must implement our own version of Var for now. This logic
# is mostly copied from dask-cudf.


class VarCudf(Reduction):
    # Uses the parallel version of Welford's online algorithm (Chan '79)
    # (http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf)
    _parameters = ["frame", "skipna", "ddof", "numeric_only", "split_every"]
    _defaults = {
        "skipna": True,
        "ddof": 1,
        "numeric_only": False,
        "split_every": False,
    }

    @functools.cached_property
    def _meta(self):
        return make_meta(
            meta_nonempty(self.frame._meta).var(
                skipna=self.skipna, numeric_only=self.numeric_only
            )
        )

    @property
    def chunk_kwargs(self):
        return dict(skipna=self.skipna, numeric_only=self.numeric_only)

    @property
    def combine_kwargs(self):
        return {}

    @property
    def aggregate_kwargs(self):
        return dict(ddof=self.ddof)

    @classmethod
    def reduction_chunk(cls, x, skipna=True, numeric_only=False):
        kwargs = {"numeric_only": numeric_only} if is_dataframe_like(x) else {}
        if skipna or numeric_only:
            n = x.count(**kwargs)
            kwargs["skipna"] = skipna
            avg = x.mean(**kwargs)
        else:
            # Not skipping nulls, so might as well
            # avoid the full `count` operation
            n = len(x)
            kwargs["skipna"] = skipna
            avg = x.sum(**kwargs) / n
        if numeric_only:
            # Workaround for cudf bug
            # (see: https://github.com/rapidsai/cudf/issues/13731)
            x = x[n.index]
        m2 = ((x - avg) ** 2).sum(**kwargs)
        return n, avg, m2

    @classmethod
    def reduction_combine(cls, parts):
        n, avg, m2 = parts[0]
        for i in range(1, len(parts)):
            n_a, avg_a, m2_a = n, avg, m2
            n_b, avg_b, m2_b = parts[i]
            n = n_a + n_b
            avg = (n_a * avg_a + n_b * avg_b) / n
            delta = avg_b - avg_a
            m2 = m2_a + m2_b + delta**2 * n_a * n_b / n
        return n, avg, m2

    @classmethod
    def reduction_aggregate(cls, vals, ddof=1):
        vals = cls.reduction_combine(vals)
        n, _, m2 = vals
        return m2 / (n - ddof)


def _patched_var(
    self, axis=0, skipna=True, ddof=1, numeric_only=False, split_every=False
):
    if axis == 0:
        if hasattr(self._meta, "to_pandas"):
            return VarCudf(self, skipna, ddof, numeric_only, split_every)
        else:
            return Var(self, skipna, ddof, numeric_only, split_every)
    elif axis == 1:
        return VarColumns(self, skipna, ddof, numeric_only)
    else:
        raise ValueError(f"axis={axis} not supported. Please specify 0 or 1")


Expr.var = _patched_var


# Temporary work-around for missing cudf + categorical support
# See: https://github.com/rapidsai/cudf/issues/11795
# TODO: Fix RepartitionQuantiles and remove this in cudf>24.06

_original_get_divisions = _shuffle_module._get_divisions


def _patched_get_divisions(frame, other, *args, **kwargs):
    # NOTE: The following two lines contains the "patch"
    # (we simply convert the partitioning column to pandas)
    if is_categorical_dtype(other._meta.dtype) and hasattr(
        other.frame._meta, "to_pandas"
    ):
        other = new_collection(other).to_backend("pandas")._expr

    # Call "original" function
    return _original_get_divisions(frame, other, *args, **kwargs)


_shuffle_module._get_divisions = _patched_get_divisions
