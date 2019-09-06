# from functools import wraps

# import cupy
import numpy as np
from pandas._libs.algos import groupsort_indexer

# from dask.compatibility import Iterator
from dask.dataframe.core import get_parallel_type, make_meta, meta_nonempty
from dask.dataframe.methods import (
    concat_dispatch,
    group_split,
    group_split_2,
    hash_df,
    percentiles_summary,
)
from dask.dataframe.utils import is_categorical_dtype

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


@percentiles_summary.register(cudf.Series)
def percentiles_summary_cudf(df, num_old, num_new, upsample, state):
    from dask.dataframe.partitionquantiles import (
        sample_percentiles,
        percentiles_to_weights,
    )
    from dask.array.percentile import _percentile

    length = len(df)
    if length == 0:
        return ()
    random_state = np.random.RandomState(state)
    qs = sample_percentiles(num_old, num_new, length, upsample, random_state)
    data = df.values_host
    interpolation = "linear"
    if is_categorical_dtype(data):
        data = data.codes
        interpolation = "nearest"
    vals, _ = _percentile(data.get(), qs, interpolation=interpolation)
    if interpolation == "linear" and np.issubdtype(data.dtype, np.integer):
        vals = np.around(vals).astype(data.dtype)
    vals_and_weights = percentiles_to_weights(qs, vals, length)
    print(vals_and_weights)
    return vals_and_weights
