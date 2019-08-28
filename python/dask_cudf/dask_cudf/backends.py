import numpy as np
from pandas._libs.algos import groupsort_indexer

from dask.dataframe.core import get_parallel_type, make_meta, meta_nonempty
from dask.dataframe.methods import (
    concat_dispatch,
    group_split,
    group_split_2,
    hash_df,
)

import cudf

from .core import DataFrame, Index, Series

get_parallel_type.register(cudf.DataFrame, lambda _: DataFrame)
get_parallel_type.register(cudf.Series, lambda _: Series)
get_parallel_type.register(cudf.Index, lambda _: Index)


@meta_nonempty.register((cudf.DataFrame, cudf.Series, cudf.Index))
def meta_nonempty_cudf(x, index=None):
    y = meta_nonempty(x.to_pandas())  # TODO: add iloc[:5]
    return cudf.from_pandas(y)


@make_meta.register((cudf.Series, cudf.DataFrame))
def make_meta_cudf(x, index=None):
    return x.head(0)


@make_meta.register(cudf.Index)
def make_meta_cudf_index(x, index=None):
    return x[:0]


@concat_dispatch.register((cudf.DataFrame, cudf.Series, cudf.Index))
def concat_cudf(
    dfs, axis=0, join="outer", uniform=False, filter_warning=True, sort=None
):
    assert axis == 0
    assert join == "outer"
    return cudf.concat(dfs)


@hash_df.register((cudf.DataFrame, cudf.Series, cudf.Index))
def hash_df_cudf(dfs):
    return dfs.hash_columns()


@group_split.register((cudf.DataFrame, cudf.Series, cudf.Index))
def group_split_cudf(df, c, k):
    indexer, locations = groupsort_indexer(c.get().astype(np.int64), k)
    df2 = df.take(indexer)
    locations = locations.cumsum()
    parts = [df2.iloc[a:b] for a, b in zip(locations[:-1], locations[1:])]
    return dict(zip(range(k), parts))


@group_split_2.register((cudf.DataFrame, cudf.Series, cudf.Index))
def group_split_2_cudf(df, col):
    if not len(df):
        return {}, df
    ind = df[col].values_host.astype(np.int64)
    n = ind.max() + 1
    indexer, locations = groupsort_indexer(ind.view(np.int64), n)
    df2 = df.take(indexer)
    locations = locations.cumsum()
    parts = [df2.iloc[a:b] for a, b in zip(locations[:-1], locations[1:])]
    result2 = dict(zip(range(n), parts))
    return result2, df.iloc[:0]
