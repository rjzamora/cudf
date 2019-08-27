import cupy
from pandas._libs.algos import groupsort_indexer

from dask.dataframe.core import get_parallel_type, make_meta, meta_nonempty
from dask.dataframe.methods import (
    concat_dispatch,
    df_index_split_dispatch,
    hash_df_dispatch,
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


@hash_df_dispatch.register((cudf.DataFrame, cudf.Series, cudf.Index))
def hash_df_cudf(dfs):
    return dfs.hash_columns()


@df_index_split_dispatch.register((cudf.DataFrame, cudf.Series, cudf.Index))
def df_index_split_cudf(df, c, k):
    if type(k) == cupy.core.core.ndarray:
        k = k.get().tolist()
    indexer, locations = groupsort_indexer(c.get(), k)
    df2 = df.take(indexer)
    locations = locations.cumsum()
    parts = [df2.iloc[a:b] for a, b in zip(locations[:-1], locations[1:])]
    return dict(zip(range(k), parts))
