# Copyright (c) 2019-2020, NVIDIA CORPORATION.
import warnings
from functools import partial
from io import BufferedWriter, BytesIO, IOBase

import numpy as np
from pyarrow import parquet as pq

from dask import dataframe as dd
from dask.dataframe.io.parquet.arrow import ArrowDatasetEngine as ArrowEngine

try:
    from dask.dataframe.io.parquet import (
        create_metadata_file as create_metadata_file_dd,
    )
except ImportError:
    create_metadata_file_dd = None

import cudf
from cudf.core.column import as_column
from cudf.io import write_to_dataset


class CudfEngine(ArrowEngine):
    @staticmethod
    def read_metadata(*args, **kwargs):
        meta, stats, parts, index = ArrowEngine.read_metadata(*args, **kwargs)

        # If `strings_to_categorical==True`, convert objects to int32
        strings_to_cats = kwargs.get("strings_to_categorical", False)

        new_meta = cudf.DataFrame(index=meta.index)
        for col in meta.columns:
            if meta[col].dtype == "O":
                new_meta[col] = as_column(
                    meta[col], dtype="int32" if strings_to_cats else "object"
                )
            else:
                new_meta[col] = as_column(meta[col])

        return (new_meta, stats, parts, index)

    @classmethod
    def multi_support(cls):
        # Assert that this class is CudfEngine
        # and that multi-part reading is supported
        return cls == CudfEngine

    @staticmethod
    def read_partition(
        fs, pieces, columns, index, categories=(), partitions=(), **kwargs
    ):
        if columns is not None:
            columns = [c for c in columns]
        if isinstance(index, list):
            columns += index

        if not isinstance(pieces, list):
            pieces = [pieces]

        strings_to_cats = kwargs.get("strings_to_categorical", False)
        if len(pieces) > 1:

            paths = []
            rgs = []
            partition_keys = []

            for piece in pieces:
                if isinstance(piece, str):
                    paths.append(piece)
                    rgs.append(None)
                else:
                    (path, row_group, partition_keys) = piece

                    row_group = None if row_group == [None] else row_group

                    paths.append(path)
                    rgs.append(
                        [row_group]
                        if not isinstance(row_group, list)
                        else row_group
                    )

            df = cudf.read_parquet(
                paths,
                engine="cudf",
                columns=columns,
                row_groups=rgs if rgs else None,
                strings_to_categorical=strings_to_cats,
                **kwargs.get("read", {}),
            )

        else:
            # Single-piece read
            if isinstance(pieces[0], str):
                path = pieces[0]
                row_group = None
                partition_keys = []
            else:
                (path, row_group, partition_keys) = pieces[0]
                row_group = None if row_group == [None] else row_group

            if cudf.utils.ioutils._is_local_filesystem(fs):
                df = cudf.read_parquet(
                    path,
                    engine="cudf",
                    columns=columns,
                    row_groups=row_group,
                    strings_to_categorical=strings_to_cats,
                    **kwargs.get("read", {}),
                )
            else:
                with fs.open(path, mode="rb") as f:
                    df = cudf.read_parquet(
                        f,
                        engine="cudf",
                        columns=columns,
                        row_groups=row_group,
                        strings_to_categorical=strings_to_cats,
                        **kwargs.get("read", {}),
                    )

        if index and (index[0] in df.columns):
            df = df.set_index(index[0])
        elif index is False and set(df.index.names).issubset(columns):
            # If index=False, we need to make sure all of the
            # names in `columns` are actually in `df.columns`
            df.reset_index(inplace=True)

        if partition_keys:
            if partitions is None:
                raise ValueError("Must pass partition sets")

            for i, (name, index2) in enumerate(partition_keys):

                categories = partitions[i].keys

                col = as_column(index2).as_frame().repeat(len(df))._data[None]

                df[name] = col.as_categorical_column(
                    cudf.CategoricalDtype(
                        categories=categories, ordered=False,
                    )
                )

        return df

    @staticmethod
    def write_partition(
        df,
        path,
        fs,
        filename,
        partition_on,
        return_metadata,
        fmd=None,
        compression="snappy",
        index_cols=None,
        **kwargs,
    ):
        preserve_index = False
        if set(index_cols).issubset(set(df.columns)):
            df.index = df[index_cols].copy(deep=False)
            df.drop(columns=index_cols, inplace=True)
            preserve_index = True
        if partition_on:
            md = write_to_dataset(
                df,
                path,
                filename=filename,
                partition_cols=partition_on,
                fs=fs,
                preserve_index=preserve_index,
                return_metadata=return_metadata,
                **kwargs,
            )
        else:
            with fs.open(fs.sep.join([path, filename]), mode="wb") as out_file:
                if not isinstance(out_file, IOBase):
                    out_file = BufferedWriter(out_file)
                md = df.to_parquet(
                    out_file,
                    compression=compression,
                    metadata_file_path=filename if return_metadata else None,
                    **kwargs,
                )
        # Return the schema needed to write the metadata
        if return_metadata:
            return [{"meta": md}]
        else:
            return []

    @staticmethod
    def write_metadata(parts, fmd, fs, path, append=False, **kwargs):
        if parts:
            # Aggregate metadata and write to _metadata file
            metadata_path = fs.sep.join([path, "_metadata"])
            _meta = []
            if append and fmd is not None:
                _meta = [fmd]
            _meta.extend([parts[i][0]["meta"] for i in range(len(parts))])
            _meta = (
                cudf.io.merge_parquet_filemetadata(_meta)
                if len(_meta) > 1
                else _meta[0]
            )
            with fs.open(metadata_path, "wb") as fil:
                fil.write(memoryview(_meta))

    @classmethod
    def collect_file_metadata(cls, path, fs, file_path):
        with fs.open(path, "rb") as f:
            meta = pq.ParquetFile(f).metadata
        if file_path:
            meta.set_file_path(file_path)
        with BytesIO() as myio:
            meta.write_metadata_file(myio)
            myio.seek(0)
            meta = np.frombuffer(myio.read(), dtype="uint8")
        return meta

    @classmethod
    def aggregate_metadata(cls, meta_list, fs, out_path):
        meta = (
            cudf.io.merge_parquet_filemetadata(meta_list)
            if len(meta_list) > 1
            else meta_list[0]
        )
        if out_path:
            metadata_path = fs.sep.join([out_path, "_metadata"])
            with fs.open(metadata_path, "wb") as fil:
                fil.write(memoryview(meta))
            return None
        else:
            return meta


def read_parquet(
    path,
    columns=None,
    split_row_groups=None,
    row_groups_per_part=None,
    **kwargs,
):
    """ Read parquet files into a Dask DataFrame

    Calls ``dask.dataframe.read_parquet`` to cordinate the execution of
    ``cudf.read_parquet``, and ultimately read multiple partitions into a
    single Dask dataframe. The Dask version must supply an ``ArrowEngine``
    class to support full functionality.
    See ``cudf.read_parquet`` and Dask documentation for further details.

    Examples
    --------
    >>> import dask_cudf
    >>> df = dask_cudf.read_parquet("/path/to/dataset/")  # doctest: +SKIP

    See Also
    --------
    cudf.read_parquet
    """
    if isinstance(columns, str):
        columns = [columns]

    if row_groups_per_part:
        warnings.warn(
            "row_groups_per_part is deprecated. "
            "Pass an integer value to split_row_groups instead."
        )
        if split_row_groups is None:
            split_row_groups = row_groups_per_part

    return dd.read_parquet(
        path,
        columns=columns,
        split_row_groups=split_row_groups,
        engine=CudfEngine,
        **kwargs,
    )


to_parquet = partial(dd.to_parquet, engine=CudfEngine)

if create_metadata_file_dd is None:
    create_metadata_file = create_metadata_file_dd
else:
    create_metadata_file = partial(create_metadata_file_dd, engine=CudfEngine)
