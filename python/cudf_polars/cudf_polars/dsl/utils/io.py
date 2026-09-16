# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for IR nodes."""

from __future__ import annotations

import concurrent.futures
import contextlib
from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.dsl.tracing import nvtx_annotate_cudf_polars
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.io import (
    ParquetSourceInfo,
    Scan,
    StreamingScan,
)
from cudf_polars.utils.parquet_metadata import _prefetch_parquet_footers_for_paths

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.base import StatsCollector
    from cudf_polars.utils.parquet_metadata import CachedParquetInfo


@nvtx_annotate_cudf_polars(message="prefetch_parquet_file_metadata_for_ir")
def prefetch_parquet_file_metadata_for_ir(
    root: IR,
    py_executor: concurrent.futures.Executor | None,
    stats: StatsCollector | None = None,
    *,
    remote_only: bool = False,
    parse_hybrid_metadata: bool = False,
) -> dict[str, CachedParquetInfo]:
    """
    Prefetch parquet metadata for all parquet scans in an IR graph.

    Parameters
    ----------
    root
        The root of the IR graph, which will be traversed.
    py_executor
        The thread pool executor to use for fetching parquet metadata concurrently.
    stats
        The stats collector. The file metadata might have already been
        prefetched during statistics collection, when the number of files
        sampled equals the total number of files. Providing ``stats`` here will
        skip rereading metadata for those files.
    remote_only
        If ``True``, only prefetch metadata for remote URIs (e.g. ``s3://``),
        skipping local paths.
    parse_hybrid_metadata
        Whether to eagerly parse ``HybridScanMetadata`` for newly-prefetched
        paths. Only useful when ``ParquetOptions.use_hybrid_scan`` is enabled.

    Returns
    -------
    A dictionary mapping each individual path to its cached parquet metadata.
    """
    all_paths: set[str] = set()

    for node in traversal([root]):
        if isinstance(node, StreamingScan) and node.base_scan.typ == "parquet":
            for task in node.tasks:
                for path in task.paths:
                    all_paths.add(path)
        elif isinstance(node, Scan) and node.typ == "parquet":  # pragma: no cover
            raise RuntimeError("Unexpected parquet 'Scan' node in lowered IR graph.")

    cached_parquet_info: dict[str, CachedParquetInfo] = {}
    if stats is not None:
        for node, datasource_info in stats.scan_stats.items():
            if (
                isinstance(node, Scan)
                and node.typ == "parquet"
                and isinstance(datasource_info, ParquetSourceInfo)
                and datasource_info.cached_parquet_info is not None
            ):
                for info in datasource_info.cached_parquet_info:
                    cached_parquet_info[info.path] = info

    missing_paths = all_paths - set(cached_parquet_info.keys())
    if remote_only:
        missing_paths = {
            p for p in missing_paths if plc.io.SourceInfo._is_remote_uri(p)
        }
    cm: contextlib.AbstractContextManager[concurrent.futures.Executor | None]

    if py_executor is None:
        cm = py_executor = concurrent.futures.ThreadPoolExecutor(
            thread_name_prefix="cudf-polars-io"
        )
    else:
        # We didn't create the executor, so we don't close it.
        cm = contextlib.nullcontext()

    with cm:
        futures = [
            py_executor.submit(
                _prefetch_parquet_footers_for_paths,
                [path],
                parse_hybrid_metadata=parse_hybrid_metadata,
            )
            for path in missing_paths
        ]

        for future in concurrent.futures.as_completed(futures):
            for info in future.result():
                cached_parquet_info[info.path] = info
    return cached_parquet_info


def attach_cached_parquet_metadata(
    root: IR,
    cached_parquet_info_map: dict[str, CachedParquetInfo],
) -> None:
    """
    Attach prefetched metadata to parquet scan tasks.

    This is an optimization only and does not affect IR identity.

    Parameters
    ----------
    root
        Root of the IR graph to update.
    cached_parquet_info_map
        Mapping from file paths to cached parquet metadata.
    """
    for node in traversal([root]):
        if isinstance(node, StreamingScan) and node.base_scan.typ == "parquet":
            base_scan = node.base_scan
            task_paths = {path for task in node.tasks for path in task.paths}
            cached_paths = [
                path
                for path in base_scan.paths
                if path in task_paths and path in cached_parquet_info_map
            ]
            cached = [cached_parquet_info_map[path] for path in cached_paths]
            if not cached:
                continue
            Scan._validate_cached_parquet_info(cached_paths, cached)
            base_scan.cached_parquet_info = cached
