# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parquet metadata helpers."""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field

import kvikio

import pylibcudf as plc

from cudf_polars.dsl.tracing import nvtx_annotate_cudf_polars


@dataclass(frozen=True)
class CachedParquetInfo:
    """
    Metadata for a parquet file.

    File metadata is only cached when the setting
    ``ParquetOptions.prefetch_file_metadata`` is ``True``. Metadata is cached
    for the duration of the query.

    Parameters
    ----------
    path
        The path of an individual parquet file. This is one element of a
        ``paths`` tuple in a ``Scan`` node.
    size
        The size of the parquet file, in bytes. This is typically only set
        for remote URLs, since it allows skipping subsequent HTTP HEAD requests
        made by kvikio on operations involving that file.
    file_metadata
        The ``FileMetaData`` object for the parquet file returned from
        ``read_parquet_footers``.
    parse_hybrid_metadata
        Whether to eagerly parse ``HybridScanMetadata`` for this file.
        Otherwise it's parsed lazily, on first use.
    """

    path: str
    size: int | None
    file_metadata: plc.io.parquet_metadata.FileMetaData
    parse_hybrid_metadata: bool = field(default=False, compare=False, repr=False)
    # For splits of the same file, the metadata is parsed once and shared.
    _hybrid_scan_metadata: plc.io.experimental.HybridScanMetadata | None = field(
        default=None, init=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:  # noqa: D105
        if self.parse_hybrid_metadata:
            object.__setattr__(
                self,
                "_hybrid_scan_metadata",
                plc.io.experimental.HybridScanMetadata.from_parquet_metadata(
                    self.file_metadata, self.default_reader_options()
                ),
            )

    def hybrid_scan_reader(
        self,
        options: plc.io.parquet.ParquetReaderOptions,
    ) -> plc.io.experimental.HybridScanReader:
        """Return a fresh HybridScanReader backed by shared pre-parsed file metadata."""
        metadata = self._hybrid_scan_metadata
        if metadata is None:
            metadata = plc.io.experimental.HybridScanMetadata.from_parquet_metadata(
                self.file_metadata, options
            )
            object.__setattr__(self, "_hybrid_scan_metadata", metadata)
        return plc.io.experimental.HybridScanReader.from_metadata(metadata)

    def default_reader_options(self) -> plc.io.parquet.ParquetReaderOptions:
        """Return baseline ``ParquetReaderOptions`` for this cached parquet file."""
        return (
            plc.io.parquet.ParquetReaderOptions.builder(
                plc.io.SourceInfo([plc.io.types.FilepathSource(self.path, self.size)])
            )
            .decimal_width(plc.TypeId.DECIMAL128)
            .build()
        )


@nvtx_annotate_cudf_polars(message="fetch_parquet_footers_for_paths")
def _prefetch_parquet_footers_for_paths(
    paths: list[str], *, parse_hybrid_metadata: bool = False
) -> list[CachedParquetInfo]:
    """
    Prefetch parquet footers for a list of paths.

    This is typically executed concurrently with prefetch operations for other
    path groups for other parquet scan nodes.

    Parameters
    ----------
    paths
        The paths to prefetch.
    parse_hybrid_metadata
        Whether to eagerly parse ``HybridScanMetadata`` for each path.

    Returns
    -------
    list[CachedParquetInfo]
        Cached parquet footer metadata aligned with ``paths``.
    """
    # TODO: https://github.com/NVIDIA/cudf/issues/22734, use object metadata from polars
    # For now, we'll just use kvikio to explicitly get the size.
    sizes: list[int | None] = []

    for path in paths:
        if paths and plc.io.SourceInfo._is_remote_uri(path):
            # We're OK to use `kvikio.RemoteFile.open` here. It does make an HTTP HEAD
            # request for S3/HTTP endpoints, but that's the entire reason we're running
            # this code. So long as it makes just *one* HTTP request, there's no advantage
            # to inferring the endpoint type.
            with kvikio.RemoteFile.open(path) as remote_file:  # pragma: no cover
                sizes.append(remote_file.nbytes())
        else:
            sizes.append(None)

    metadata = plc.io.parquet_metadata.read_parquet_footers(
        plc.io.types.SourceInfo(
            [
                plc.io.types.FilepathSource(path, size)
                for path, size in zip(paths, sizes, strict=True)
            ]
        )
    )

    return [
        CachedParquetInfo(
            path, size, file_metadata, parse_hybrid_metadata=parse_hybrid_metadata
        )
        for path, size, file_metadata in zip(paths, sizes, metadata, strict=True)
    ]


def _columnchunk_metadata_from_footers(
    footers: list[plc.io.parquet_metadata.FileMetaData],
) -> dict[str, list[int]]:
    columnchunk_metadata: dict[str, list[int]] = {}
    for fmd in footers:
        for name, uncompressed_sizes in fmd.columnchunk_metadata.items():
            columnchunk_metadata.setdefault(name, []).extend(uncompressed_sizes)
    return columnchunk_metadata


class ParquetMetadata:
    """
    Parquet metadata container.

    Parameters
    ----------
    paths
        Parquet-dataset paths.
    max_footer_samples
        Maximum number of file footers to sample metadata from.
    parse_hybrid_metadata
        Whether to eagerly parse ``HybridScanMetadata`` for sampled paths.
        Only useful when ``ParquetOptions.use_hybrid_scan`` is enabled.
    """

    __slots__ = (
        "cached_parquet_info",
        "column_names",
        "max_footer_samples",
        "mean_size_per_file",
        "num_row_groups_per_file",
        "paths",
        "row_count",
        "sample_paths",
        "sampled_file_count",
        "total_file_count",
    )

    paths: tuple[str, ...]
    """Parquet-dataset paths."""
    max_footer_samples: int
    """Maximum number of file footers to sample metadata from."""
    row_count: int | None
    """Total row-count estimate."""
    num_row_groups_per_file: tuple[int, ...]
    """Number of row groups in each sampled file."""
    mean_size_per_file: dict[str, int]
    """Average column storage size in a single file."""
    column_names: tuple[str, ...]
    """All column names found in the dataset."""
    sample_paths: tuple[str, ...]
    """Sampled file paths."""
    cached_parquet_info: list[CachedParquetInfo] | None
    """Cached parquet info for the sampled paths."""

    @nvtx_annotate_cudf_polars(message="ParquetMetadata")
    def __init__(
        self,
        paths: tuple[str, ...],
        max_footer_samples: int,
        *,
        parse_hybrid_metadata: bool = False,
    ):
        self.paths = paths
        self.max_footer_samples = max_footer_samples
        self.row_count = None
        self.num_row_groups_per_file = ()
        self.mean_size_per_file = {}
        self.column_names = ()
        self.cached_parquet_info = None
        self.total_file_count = len(self.paths)
        self.sampled_file_count = 0
        if max_footer_samples <= 0:
            self.sample_paths = ()
            return

        stride = max(1, int(len(paths) / max_footer_samples))
        self.sample_paths = paths[: stride * max_footer_samples : stride]

        if not self.sample_paths:
            # No paths to sample from
            # TODO: This requires row_count to be nullable. Why do we allow empty paths?
            return

        sampled_file_count = len(self.sample_paths)

        sample_parquet_info = _prefetch_parquet_footers_for_paths(
            list(self.sample_paths), parse_hybrid_metadata=parse_hybrid_metadata
        )
        sample_footers = [info.file_metadata for info in sample_parquet_info]

        self.cached_parquet_info = sample_parquet_info
        sampled_row_count = sum(fmd.num_rows for fmd in sample_footers)
        if self.total_file_count == sampled_file_count:
            row_count = sampled_row_count
        else:
            num_rows_per_sampled_file = int(sampled_row_count / sampled_file_count)
            row_count = num_rows_per_sampled_file * self.total_file_count

        num_row_groups_per_sampled_file = [
            len(fmd.row_group_num_rows) for fmd in sample_footers
        ]
        rowgroup_offsets_per_file = list(
            itertools.accumulate(num_row_groups_per_sampled_file, initial=0)
        )

        column_sizes_per_file = {
            name: [
                sum(uncompressed_sizes[start:end])
                for (start, end) in itertools.pairwise(rowgroup_offsets_per_file)
            ]
            for name, uncompressed_sizes in _columnchunk_metadata_from_footers(
                sample_footers
            ).items()
        }

        self.column_names = tuple(column_sizes_per_file)
        self.mean_size_per_file = {
            name: int(statistics.mean(sizes))
            for name, sizes in column_sizes_per_file.items()
        }
        self.num_row_groups_per_file = tuple(num_row_groups_per_sampled_file)
        self.row_count = row_count
        self.sampled_file_count = sampled_file_count
