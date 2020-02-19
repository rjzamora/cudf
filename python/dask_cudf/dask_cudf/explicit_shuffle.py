import asyncio
import gc
import sys
import warnings

import cupy
import pandas as pd

from dask.distributed import wait
from dask_cuda.explicit_comms import comms
from distributed.protocol import to_serialize

import cudf
import rmm

cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)


def _cleanup_parts(df_parts):
    for part in df_parts:
        if part:
            del part


class DFWrapper:
    def __init__(self, df):
        if df is None:
            self.df = None
        else:
            self.df = df.copy()


async def send_df(ep, dfw):
    if dfw.df is None:
        return await ep.write("empty")
    else:
        ret = ep.write([to_serialize(dfw.df)])
        await ret
        assert sys.getrefcount(dfw.df) <= 2
        del dfw.df
        return ret


async def recv_df(ep):
    ret = await ep.read()
    if ret == "empty":
        return None
    else:
        return ret[0]


async def send_parts(eps, parts):
    futures = []
    for rank, ep in eps.items():
        assert sys.getrefcount(parts[rank]) <= 2
        dfw = DFWrapper(parts[rank].copy())
        del parts[rank]
        futures.append(send_df(ep, dfw))
    await asyncio.gather(*futures)


async def recv_parts(eps, parts):
    futures = []
    for ep in eps.values():
        futures.append(recv_df(ep))
    parts.extend(await asyncio.gather(*futures))


def concat(df_list, sort_by=None):
    if len(df_list) == 0:
        return None
    if isinstance(df_list[0], cudf.DataFrame):
        if sort_by:
            new_df = cudf.merge_sorted(df_list, keys=sort_by)
        else:
            new_df = cudf.concat(df_list)
    else:
        new_df = pd.concat(df_list)
    del df_list
    return new_df


async def exchange_and_concat_parts(rank, eps, parts, sort_by):
    ret = [parts[rank].copy()]
    del parts[rank]
    parts[rank] = None
    await asyncio.gather(recv_parts(eps, ret), send_parts(eps, parts))
    del parts
    new_df = concat(
        [df for df in ret if df is not None and len(df)], sort_by=sort_by
    )
    del ret
    return new_df


def partition_table(dfw, partitions, n_chunks, sort_by=None):
    if dfw is None:
        result = [None] * n_chunks
    elif sort_by:
        result = {
            i: dfw.df.iloc[partitions[i] : partitions[i + 1]].copy()
            for i in range(0, len(partitions) - 1)
        }
    else:
        result = dict(
            zip(
                range(n_chunks),
                dfw.df.copy().scatter_by_map(partitions, map_size=n_chunks),
            )
        )

    assert sys.getrefcount(dfw.df) <= 2
    del dfw.df
    return result


async def distributed_shuffle(
    n_chunks, rank, eps, table_wrapper, partitions, index, sort_by
):
    parts = partition_table(
        table_wrapper, partitions, n_chunks, sort_by=sort_by
    )
    del table_wrapper
    return await exchange_and_concat_parts(rank, eps, parts, sort_by)


async def _explicit_shuffle(
    s, df_nparts, df_parts, index, sort_by, divisions, to_cpu
):
    def df_concat(df_parts, sort_by=None):
        """Making sure df_parts is a single dataframe or None"""
        if len(df_parts) == 0:
            return None
        elif len(df_parts) == 1:
            return df_parts[0]
        else:
            with open(
                "/home/nfs/rzamora/workspace/ucx_cudf++_12/debugging.out", "a"
            ) as f:
                f.write(
                    "Concat Ref Count: %d\n" % (sys.getrefcount(df_parts[0]))
                )
                f.write(
                    "Referrers: %s\n" % (str(gc.get_referrers(df_parts[0])))
                )
            df_new = df_parts[0].copy()
            del df_parts[0]
            for i in range(1, len(df_parts)):
                df_new = concat([df_new, df_parts[0].copy()], sort_by=sort_by)
                del df_parts[0]
            return df_new

    # Concatenate all parts owned by this worker into
    # a single cudf DataFrame
    if to_cpu:
        df = df_concat([dfp.to_pandas() for dfp in df_parts[0]])
    else:
        # NOTE: Returning here to debug memory...
        return df_concat(df_parts[0], sort_by=sort_by)
    del df_parts

    with open(
        "/home/nfs/rzamora/workspace/ucx_cudf++_12/debugging.out", "a"
    ) as f:
        f.write("A Ref Count: %d\n" % (sys.getrefcount(df)))
        # f.write(
        #     "A memory_usage: %d\n" % (
        #         df.memory_usage(deep=True, index=True)/1e9
        #     ).sum()
        # )

    # Calculate new partition mapping
    if df is not None:
        divisions = df._constructor_sliced(divisions, dtype=df[index].dtype)
        if sort_by:
            splits = df[index].copy().searchsorted(divisions, side="left")
            splits[-1] = len(df[index])
            partitions = splits.tolist()
            del splits
        else:
            partitions = divisions.searchsorted(df[index], side="right") - 1
            partitions[(df[index] >= divisions.iloc[-1]).values] = (
                len(divisions) - 2
            )
        del divisions
    else:
        partitions = None

    with open(
        "/home/nfs/rzamora/workspace/ucx_cudf++_12/debugging.out", "a"
    ) as f:
        f.write("B Ref Count: %d\n" % (sys.getrefcount(df)))

    # Wrap dataframe object so we can delete memory later
    assert sys.getrefcount(df) <= 2
    dfw = DFWrapper(df.copy())
    del df

    # Run distributed shuffle and set_index algorithm
    new_df = await distributed_shuffle(
        s["nworkers"], s["rank"], s["eps"], dfw, partitions, index, sort_by
    )

    if to_cpu:
        return cudf.from_pandas(new_df)
    return new_df


def explicit_sorted_shuffle(df, index, divisions, sort_by, client, **kwargs):
    # Explict-comms shuffle
    wait(df.persist())
    to_cpu = kwargs.get("to_cpu", False)
    if to_cpu:
        warnings.warn("Using CPU for shuffling. Performance will suffer!")
    with open(
        "/home/nfs/rzamora/workspace/ucx_cudf++_12/debugging.out", "a"
    ) as f:
        f.write(
            "df memory_usage: %f\n"
            % (df.memory_usage(deep=True, index=True).sum().compute() / 1e9)
        )
    return comms.default_comms().dataframe_operation(
        _explicit_shuffle,
        df_list=(df,),
        extra_args=(index, sort_by, divisions, to_cpu),
    )
