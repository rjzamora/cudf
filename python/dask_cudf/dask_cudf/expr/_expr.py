# Copyright (c) 2024, NVIDIA CORPORATION.

import numpy as np

from dask_expr._cumulative import CumulativeBlockwise, TakeLast
from dask_expr._groupby import NUnique
from dask_expr._shuffle import DiskShuffle

##
## Custom expression patching
##


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


def _takelast(a, skipna=True):
    if not len(a):
        return a
    if skipna:
        a = a.bfill()
    # Cannot use `squeeze` with cudf
    return a.tail(n=1).iloc[0]


TakeLast.operation = staticmethod(_takelast)


def _shuffle_group(df, col, _filter, p):
    from dask.dataframe.shuffle import ensure_cleanup_on_exception

    with ensure_cleanup_on_exception(p):
        g = df.groupby(col)
        if hasattr(g, "_grouped"):
            # Avoid `get_group` for cudf data.
            # See: https://github.com/rapidsai/cudf/issues/14955
            keys, part_offsets, _, grouped_df = df.groupby(col)._grouped()
            d = {
                k: grouped_df.iloc[part_offsets[i] : part_offsets[i + 1]]
                for i, k in enumerate(keys.to_pandas())
                if k in _filter
            }
        else:
            d = {i: g.get_group(i) for i in g.groups if i in _filter}
        p.append(d, fsync=True)


DiskShuffle._shuffle_group = staticmethod(_shuffle_group)



from dask.dataframe.dispatch import concat
from dask.dataframe.groupby import _determine_levels, _groupby_raise_unaligned


def _drop_duplicates_reindex(df):
    if hasattr(df, "to_pandas"):
        # Don't reindex for cudf
        return df.drop_duplicates()
    result = df.drop_duplicates()
    result.index = np.zeros(len(result), dtype=int)
    return result


def _nunique_df_chunk(df, *by, **kwargs):
    name = kwargs.pop("name")
    group_keys = {}
    group_keys["group_keys"] = True

    g = _groupby_raise_unaligned(df, by=by, **group_keys)
    if len(df) > 0:
        grouped = (
            g[[name]].apply(_drop_duplicates_reindex).reset_index(level=-1, drop=True)
        )
    else:
        # Manually create empty version, since groupby-apply for empty frame
        # results in df with no columns
        grouped = g[[name]].nunique()
        grouped = grouped.astype(df.dtypes[grouped.columns].to_dict())

    return grouped


def _nunique_df_combine(df, levels, sort=False):
    result = (
        df.groupby(level=levels, sort=sort, observed=True)
        .apply(_drop_duplicates_reindex)
        .reset_index(level=-1, drop=True)
    )
    return result


def nunique_df_chunk(df, *by, **kwargs):
    if df.ndim == 1:
        df = df.to_frame()
        kwargs = dict(name=df.columns[0], levels=_determine_levels(by))
    return _nunique_df_chunk(df, *by, **kwargs)


def nunique_df_combine(dfs, *args, **kwargs):
    return _nunique_df_combine(concat(dfs), *args, **kwargs)


NUnique.chunk = staticmethod(nunique_df_chunk)
NUnique.combine = staticmethod(nunique_df_combine)
