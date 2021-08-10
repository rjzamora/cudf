# Copyright (c) 2020-2021, NVIDIA CORPORATION.
from contextlib import ExitStack

from io import BufferedWriter, IOBase

from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path
from pyarrow import orc as orc

from dask import dataframe as dd
from dask.base import tokenize
from dask.dataframe.io.utils import _get_pyarrow_dtypes

import cudf
from cudf.core.column import as_column, build_categorical_column

try:

    from dask.dataframe.io.orc.arrow import ArrowORCEngine

    class CudfORCEngine(ArrowORCEngine):

        @classmethod
        def _create_meta(cls, fs, part0, columns, *args, **kwargs):
            pd_meta = ArrowORCEngine._create_meta(fs, part0, columns, *args, **kwargs)
            if part0:
                with fs.open(part0[0], "rb") as f:
                    cudf_meta = cudf.read_orc(
                        f,
                        stripes=[0] if part0[1] else None,
                        columns=columns,
                        **kwargs,
                    )
            else:
                cudf_meta = cudf.DataFrame(index=pd_meta.index)

            for col in pd_meta.columns:
                if col not in cudf_meta.columns:
                    cudf_meta[col] = as_column(pd_meta[col])

            return cudf_meta

        @classmethod
        def read_partition(
            cls,
            fs,
            parts,
            columns,
            filters=None,
            schema=None,
            partition_uniques=None,
            **kwargs,
        ):
            # Create a seperate dataframe for each directory partition.
            # We are only creating a single cudf dataframe if there
            # are no partition columns.
            tables = []
            partitions = []
            partition_uniques = partition_uniques or {}
            if columns:
                # Separate file columns and partition columns
                file_columns = [c for c in columns if c in set(schema)]
                partition_columns = [c for c in columns if c not in set(schema)]
            else:
                file_columns, partition_columns = None, list(partition_uniques)

            dfs = []
            path, stripes, hive_parts = parts[0]
            path_list = [path]
            stripe_list = [stripes]
            for path, stripes, next_hive_parts in parts[1:]:
                if hive_parts == next_hive_parts:
                    path_list.append(path)
                    stripe_list.append(stripes)
                else:
                    dfs.append(
                        cls._read_partition(
                            fs,
                            path_list,
                            file_columns,
                            filters,
                            stripe_list,
                            **kwargs,
                        )
                    )
                    partitions.append(hive_parts)
                    path_list = [path]
                    stripe_list = [stripes]
                    hive_parts = next_hive_parts
            dfs.append(
                cls._read_partition(
                    fs,
                    path_list,
                    file_columns,
                    filters,
                    stripe_list,
                    **kwargs,
                )
            )
            partitions.append(hive_parts)

            # Add partition columns to each partition dataframe
            for i, hive_parts in enumerate(partitions):
                for (part_name, cat) in hive_parts:
                    if part_name in partition_columns:
                        # We read from file paths, so the partition
                        # columns are NOT in our table yet.
                        categories = partition_uniques[part_name]

                        col = as_column(categories.index(cat)).as_frame().repeat(len(df))._data[None]
                        dfs[i][part_name] = build_categorical_column(
                            categories=categories,
                            codes=as_column(col.base_data, dtype=col.dtype),
                            size=col.size,
                            offset=col.offset,
                            ordered=False,
                        )
            return cudf.concat(dfs)


        @classmethod
        def _read_partition(cls, fs, path_list, columns, filters, stripe_list, **kwargs):

            with ExitStack() as stack:
                if cudf.utils.ioutils._is_local_filesystem(fs):
                    # Let cudf open the files if this is
                    # a local file system
                    _source_list = path_list
                else:
                    # Use fs.open to pass file handles to cudf
                    _source_list = [
                        stack.enter_context(fs.open(path, "rb"))
                        for path in path_list
                    ]
                df = cudf.io.read_orc(
                    _source_list,
                    columns=columns,
                    filters=filters,
                    stripes=stripe_list,
                    **kwargs,
                )
            return df

except ImportError:
    CudfORCEngine = None


def _read_orc_stripe(fs, path, stripe, columns, kwargs=None):
    """Pull out specific columns from specific stripe"""
    if kwargs is None:
        kwargs = {}
    with fs.open(path, "rb") as f:
        df_stripe = cudf.read_orc(
            f, stripes=[stripe], columns=columns, **kwargs
        )
    return df_stripe


def read_orc(*args, legacy=False, **kwargs):
    if CudfORCEngine is None or legacy is True:
        return read_orc_legacy(*args, **kwargs)

    engine = kwargs.pop("engine", CudfORCEngine)
    if engine == "cudf":
        engine = CudfORCEngine
    return dd.read_orc(*args, engine=engine, **kwargs)


def read_orc_legacy(path, columns=None, filters=None, storage_options=None, **kwargs):
    """Read cudf dataframe from ORC file(s).

    Note that this function is mostly borrowed from upstream Dask.

    Parameters
    ----------
    path: str or list(str)
        Location of file(s), which can be a full URL with protocol specifier,
        and may include glob character if a single string.
    columns: None or list(str)
        Columns to load. If None, loads all.
    filters : None or list of tuple or list of lists of tuples
        If not None, specifies a filter predicate used to filter out row groups
        using statistics stored for each row group as Parquet metadata. Row
        groups that do not match the given filter predicate are not read. The
        predicate is expressed in disjunctive normal form (DNF) like
        `[[('x', '=', 0), ...], ...]`. DNF allows arbitrary boolean logical
        combinations of single column predicates. The innermost tuples each
        describe a single column predicate. The list of inner predicates is
        interpreted as a conjunction (AND), forming a more selective and
        multiple column predicate. Finally, the outermost list combines
        these filters as a disjunction (OR). Predicates may also be passed
        as a list of tuples. This form is interpreted as a single conjunction.
        To express OR in predicates, one must use the (preferred) notation of
        list of lists of tuples.
    storage_options: None or dict
        Further parameters to pass to the bytes backend.

    Returns
    -------
    cudf.DataFrame
    """

    storage_options = storage_options or {}
    fs, fs_token, paths = get_fs_token_paths(
        path, mode="rb", storage_options=storage_options
    )
    schema = None
    nstripes_per_file = []
    for path in paths:
        with fs.open(path, "rb") as f:
            o = orc.ORCFile(f)
            if schema is None:
                schema = o.schema
            elif schema != o.schema:
                raise ValueError(
                    "Incompatible schemas while parsing ORC files"
                )
            nstripes_per_file.append(o.nstripes)
    schema = _get_pyarrow_dtypes(schema, categories=None)
    if columns is not None:
        ex = set(columns) - set(schema)
        if ex:
            raise ValueError(
                "Requested columns (%s) not in schema (%s)" % (ex, set(schema))
            )
    else:
        columns = list(schema)

    with fs.open(paths[0], "rb") as f:
        meta = cudf.read_orc(
            f,
            stripes=[0] if nstripes_per_file[0] else None,
            columns=columns,
            **kwargs,
        )

    name = "read-orc-" + tokenize(fs_token, path, columns, **kwargs)
    dsk = {}
    N = 0
    for path, n in zip(paths, nstripes_per_file):
        for stripe in (
            range(n)
            if filters is None
            else cudf.io.orc._filter_stripes(filters, path)
        ):
            dsk[(name, N)] = (
                _read_orc_stripe,
                fs,
                path,
                stripe,
                columns,
                kwargs,
            )
            N += 1

    divisions = [None] * (len(dsk) + 1)
    return dd.core.new_dd_object(dsk, name, meta, divisions)


def write_orc_partition(df, path, fs, filename, compression=None):
    full_path = fs.sep.join([path, filename])
    with fs.open(full_path, mode="wb") as out_file:
        if not isinstance(out_file, IOBase):
            out_file = BufferedWriter(out_file)
        cudf.io.to_orc(df, out_file, compression=compression)
    return full_path


def to_orc(
    df,
    path,
    write_index=True,
    storage_options=None,
    compression=None,
    compute=True,
    **kwargs,
):
    """Write a dask_cudf dataframe to ORC file(s) (one file per partition).

    Parameters
    ----------
    df : dask_cudf.DataFrame
    path: string or pathlib.Path
        Destination directory for data.  Prepend with protocol like ``s3://``
        or ``hdfs://`` for remote data.
    write_index : boolean, optional
        Whether or not to write the index. Defaults to True.
    storage_options: None or dict
        Further parameters to pass to the bytes backend.
    compression : string or dict, optional
    compute : bool, optional
        If True (default) then the result is computed immediately. If False
        then a ``dask.delayed`` object is returned for future computation.
    """

    from dask import compute as dask_compute, delayed

    # TODO: Use upstream dask implementation once available
    #       (see: Dask Issue#5596)

    if hasattr(path, "name"):
        path = stringify_path(path)
    fs, _, _ = get_fs_token_paths(
        path, mode="wb", storage_options=storage_options
    )
    # Trim any protocol information from the path before forwarding
    path = fs._strip_protocol(path)

    if write_index:
        df = df.reset_index()
    else:
        # Not writing index - might as well drop it
        df = df.reset_index(drop=True)

    fs.mkdirs(path, exist_ok=True)

    # Use i_offset and df.npartitions to define file-name list
    filenames = ["part.%i.orc" % i for i in range(df.npartitions)]

    # write parts
    dwrite = delayed(write_orc_partition)
    parts = [
        dwrite(d, path, fs, filename, compression=compression)
        for d, filename in zip(df.to_delayed(), filenames)
    ]

    if compute:
        return dask_compute(*parts)

    return delayed(list)(parts)
