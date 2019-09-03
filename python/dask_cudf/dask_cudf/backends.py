from functools import wraps

import cupy
import numpy as np
from pandas._libs.algos import groupsort_indexer

from dask.compatibility import Iterator
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
    def _sample_percentiles(
        num_old, num_new, chunk_length, upsample=1.0, random_state=None
    ):
        # *waves hands*
        random_percentage = 1 / (1 + (4 * num_new / num_old) ** 0.5)
        num_percentiles = upsample * num_new * (num_old + 22) ** 0.55 / num_old
        num_fixed = int(num_percentiles * (1 - random_percentage)) + 2
        num_random = int(num_percentiles * random_percentage) + 2

        if num_fixed + num_random + 5 >= chunk_length:
            return cupy.linspace(0, 100, chunk_length + 1)

        if not isinstance(random_state, cupy.random.RandomState):
            random_state = cupy.random.RandomState(random_state)

        q_fixed = cupy.linspace(0, 100, num_fixed)
        q_random = random_state.rand(num_random) * 100
        q_edges = cupy.asarray(
            [60 / (num_fixed - 1), 100 - 60 / (num_fixed - 1)]
        )
        qs = cupy.concatenate(
            [q_fixed, q_random, q_edges, cupy.asarray([0, 100])]
        )
        qs.sort()
        # Make the divisions between percentiles a little more even
        qs = 0.5 * (qs[:-1] + qs[1:])
        return qs

    @wraps(cupy.percentile)
    def _percentile(a, q, interpolation="linear"):
        n = len(a)
        if not len(a):
            return None, n
        if isinstance(q, Iterator):
            q = list(q)
        if a.dtype.name == "category":
            result = cupy.percentile(a.codes, q, interpolation=interpolation)
            import pandas as pd

            return (
                pd.Categorical.from_codes(result, a.categories, a.ordered),
                n,
            )
        if cupy.issubdtype(a.dtype, np.datetime64):
            a2 = a.astype("i8")
            result = cupy.percentile(a2, q, interpolation=interpolation)
            return result.astype(a.dtype), n
        if not cupy.issubdtype(a.dtype, cupy.number):
            interpolation = "nearest"
        return cupy.percentile(a, q, interpolation=interpolation), n

    def _percentiles_to_weights(qs, vals, length):
        if length == 0:
            return ()
        diff = np.ediff1d(qs, 0.0, 0.0)
        weights = 0.5 * length * (diff[1:] + diff[:-1])
        return vals.tolist(), weights.tolist()

    length = len(df)
    if length == 0:
        return ()
    random_state = cupy.random.RandomState(state)
    qs = _sample_percentiles(num_old, num_new, length, upsample, random_state)
    data = df.values
    interpolation = "linear"
    if is_categorical_dtype(data):
        data = data.codes
        interpolation = "nearest"
    vals, _ = _percentile(data, qs, interpolation=interpolation)
    if interpolation == "linear" and cupy.issubdtype(data.dtype, cupy.integer):
        vals = cupy.around(vals).astype(data.dtype)
    vals_and_weights = _percentiles_to_weights(qs.get(), vals, length)
    return vals_and_weights
