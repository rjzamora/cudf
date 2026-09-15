# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract ordering metadata from parquet scan tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    OrderKey,
    OrderScheme,
    Ordering,
    Partitioning,
)
from cudf_streaming.table_chunk import TableChunk

from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.collectives.sort import (
    _extract_boundaries_from_endpoint_rows as _extract_ordering_boundaries,
)
from cudf_polars.streaming.io import ParquetScanTask
from cudf_polars.streaming.partitioning_requests import OrderPartitioningRequest
from cudf_polars.utils.dtypes import make_empty_column

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import CachedParquetInfo, IRExecutionContext, Scan
    from cudf_polars.streaming.io import StreamingScan
    from cudf_polars.streaming.partitioning_requests import PartitioningRequest


def _null_column(ir: StreamingScan, name: str, size: int, stream: Stream) -> plc.Column:
    empty = make_empty_column(ir.schema[name], stream)
    if size == 0:
        return empty
    return plc.Column.all_null_like(empty, size, stream=stream)


def _gather_rows(table: plc.Table, rows: list[int], stream: Stream) -> plc.Table:
    return plc.copying.gather(
        table,
        plc.Column.from_iterable_of_py(
            rows, plc.DataType(plc.TypeId.INT32), stream=stream
        ),
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )


def _get_ordering_candidates(
    ir: StreamingScan,
    requests: tuple[PartitioningRequest, ...],
) -> list[tuple[str, OrderKey]]:
    candidates: list[tuple[str, OrderKey]] = []
    for request in requests:
        if not isinstance(request, OrderPartitioningRequest):
            continue
        assert request.keys, "Order partitioning requests must have at least one key."
        key = request.keys[0]
        assert key.name in ir.schema, (
            f"Ordering request key {key.name!r} must be present in scan schema."
        )
        (column_index,) = names_to_indices((key.name,), ir.schema)
        candidate = (key.name, OrderKey(column_index, key.order, key.null_order))
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


async def _get_rank_parquet_info_map(
    base_scan: Scan,
    paths: list[str],
    ir_context: IRExecutionContext,
) -> dict[str, CachedParquetInfo]:
    cached_parquet_info_map = {
        info.path: info
        for info in (base_scan.cached_parquet_info or ())
        if info.path in paths
    }
    if set(cached_parquet_info_map) == set(paths):
        return cached_parquet_info_map

    from cudf_polars.dsl.utils.io import _prefetch_parquet_footers_for_paths

    fetched = await ir_context.to_thread(_prefetch_parquet_footers_for_paths, paths)
    rank_parquet_info_map = {info.path: info for info in fetched}
    assert all(path in rank_parquet_info_map for path in paths), (
        "Ordering footer metadata must contain all rank-local scan paths."
    )
    return rank_parquet_info_map


def _stats_are_safe(
    rank_row_groups: Sequence[tuple[CachedParquetInfo, int]],
    name: str,
    indices: list[int],
) -> bool:
    for index in indices:
        info, group = rank_row_groups[index]
        chunk = next(
            (
                chunk
                for chunk in info.file_metadata.row_groups[group].columns
                if ".".join(chunk.meta_data.path_in_schema) == name
            ),
            None,
        )
        stats = None if chunk is None else chunk.meta_data.statistics
        if (
            stats is None
            or stats.null_count is None
            or stats.null_count != 0
            or getattr(stats, "is_min_value_exact", True) is False
            or getattr(stats, "is_max_value_exact", True) is False
        ):
            return False
    return True


def _candidate_bounds(
    file_metadata: list[plc.io.parquet_metadata.FileMetaData],
    rank_row_groups: Sequence[tuple[CachedParquetInfo, int]],
    task_row_groups: Sequence[list[int] | None],
    name: str,
    key: OrderKey,
    stream: Stream,
) -> plc.Column | None:
    try:
        bounds = plc.io.parquet_metadata.read_parquet_column_chunk_bounds(
            file_metadata, columns=[name], stream=stream
        )
    except (TypeError, ValueError, RuntimeError):
        return None

    columns = bounds.columns()[2:]
    assert len(columns) == 2, "Single-column parquet bounds must have min/max columns."
    min_col, max_col = columns
    assert min_col.size() == max_col.size(), "Parquet min/max columns must align."
    assert len(rank_row_groups) == min_col.size(), (
        "Decoded parquet bounds must match footer row-group metadata."
    )

    def invalidate() -> plc.Column:
        return plc.Column.all_null_like(
            min_col, 2 * len(task_row_groups), stream=stream
        )

    if min_col.null_count() or max_col.null_count():
        return invalidate()

    start, end = (
        (max_col, min_col)
        if key.order == plc.types.Order.DESCENDING
        else (
            min_col,
            max_col,
        )
    )
    row_group_bounds = plc.concatenate.concatenate(
        [plc.Table([start]), plc.Table([end])], stream=stream
    )

    chunk_bounds: list[plc.Table] = []
    for indices in task_row_groups:
        if indices is None or not _stats_are_safe(rank_row_groups, name, indices):
            return invalidate()

        selected = _gather_rows(
            row_group_bounds,
            [i for group in indices for i in (group, len(rank_row_groups) + group)],
            stream,
        )
        if not plc.sorting.is_sorted(
            selected, [key.order], [key.null_order], stream=stream
        ):
            return invalidate()
        chunk_bounds.append(
            _gather_rows(selected, [0, selected.num_rows() - 1], stream)
        )

    return plc.concatenate.concatenate(chunk_bounds, stream=stream).columns()[0]


async def parquet_metadata_ordering(
    context: Context,
    comm: Communicator,
    ir: StreamingScan,
    global_chunk_count: int,
    requests: tuple[PartitioningRequest, ...],
    ir_context: IRExecutionContext,
    collective_id: int,
) -> Partitioning | None:
    """
    Return ordering partitioning inferred from parquet footer metadata.

    Only columns identified by order partitioning requests are inspected. This
    succeeds when footer min/max statistics prove that rank-local scan tasks
    have globally ordered, non-overlapping bounds for one requested key.
    """
    assert ir.base_scan.typ == "parquet", (
        f"Expected parquet Scan, got {ir.base_scan.typ}."
    )
    assert global_chunk_count > 0, "Scan partition count must be positive."

    if not (candidates := _get_ordering_candidates(ir, requests)):
        return None

    tasks: list[ParquetScanTask] = []
    for task in ir.tasks:
        assert isinstance(task, ParquetScanTask)
        tasks.append(task)

    paths = list(dict.fromkeys(path for task in tasks for path in task.paths))
    rank_parquet_info_map = await _get_rank_parquet_info_map(
        ir.base_scan, paths, ir_context
    )
    rank_row_group_offset_map: dict[str, int] = {}
    rank_row_groups: list[tuple[CachedParquetInfo, int]] = []
    for path in paths:
        info = rank_parquet_info_map[path]
        rank_row_group_offset_map[path] = len(rank_row_groups)
        rank_row_groups.extend(
            (info, i) for i in range(len(info.file_metadata.row_group_num_rows))
        )
    task_row_groups = [
        task.absolute_row_group_indices(
            rank_parquet_info_map, rank_row_group_offset_map
        )
        for task in tasks
    ]

    stream = ir_context.get_cuda_stream()
    file_metadata = [rank_parquet_info_map[path].file_metadata for path in paths]
    bound_count = 2 * len(task_row_groups)
    columns: list[plc.Column] = []
    for name, key in candidates:
        column = (
            _candidate_bounds(
                file_metadata,
                rank_row_groups,
                task_row_groups,
                name,
                key,
                stream,
            )
            if task_row_groups
            else None
        )
        if column is None:
            column = _null_column(ir, name, bound_count, stream)
        columns.append(column)
    bounds = plc.Table(columns)
    if comm.nranks > 1:
        local_chunk = TableChunk.from_pylibcudf_table(
            bounds, stream, exclusive_view=True, br=context.br()
        )
        allgather = AllGatherManager(context, comm, collective_id)
        with allgather.inserting() as inserter:
            await inserter.insert(comm.rank, local_chunk)
        bounds = await allgather.extract_concatenated(
            stream, ordered=True, ir_context=ir_context
        )

    assert bounds.num_rows() == 2 * global_chunk_count, (
        "Ordering chunk bounds must contain first/last rows for every scan chunk."
    )

    for i, (_, key) in enumerate(candidates):
        column = bounds.columns()[i]
        if column.null_count():
            continue

        candidate_bounds = plc.Table([column])
        if not plc.sorting.is_sorted(
            candidate_bounds, [key.order], [key.null_order], stream=stream
        ):
            continue

        if global_chunk_count < 2:
            ordering_boundaries = plc.Table(
                [plc.Column.from_iterable_of_py([], column.type(), stream=stream)]
            )
            strict = True
        else:
            ordering_boundaries, strict = _extract_ordering_boundaries(
                candidate_bounds, global_chunk_count, stream
            )
        return Partitioning(
            inter_rank=OrderScheme(
                [
                    Ordering(
                        [key],
                        TableChunk.from_pylibcudf_table(
                            ordering_boundaries,
                            stream,
                            exclusive_view=True,
                            br=context.br(),
                        ),
                        strict_boundaries=strict,
                        locally_ordered=False,
                    )
                ]
            ),
            local="inherit",
        )
    return None
