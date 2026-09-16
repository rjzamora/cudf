# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract ordering metadata from scan tasks."""

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
from cudf_polars.utils.parquet_metadata import _prefetch_parquet_footers_for_paths

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext, Scan
    from cudf_polars.streaming.io import StreamingScan
    from cudf_polars.streaming.partitioning_requests import PartitioningRequest
    from cudf_polars.utils.parquet_metadata import CachedParquetInfo


def _null_bounds_column(
    ir: StreamingScan,
    column_name: str,
    size: int,
    stream: Stream,
) -> plc.Column:
    """Return null chunk bounds to invalidate one candidate."""
    empty = make_empty_column(ir.schema[column_name], stream)
    if size == 0:
        return empty
    return plc.Column.all_null_like(empty, size, stream=stream)


async def _get_task_parquet_info(
    base_scan: Scan,
    tasks: Sequence[ParquetScanTask],
    ir_context: IRExecutionContext,
) -> list[list[CachedParquetInfo]]:
    """Return cached or freshly fetched footer metadata for each task."""
    paths = list(dict.fromkeys(path for task in tasks for path in task.paths))
    info_by_path = {
        info.path: info
        for info in (base_scan.cached_parquet_info or ())
        if info.path in paths
    }
    if set(info_by_path) == set(paths):
        return [[info_by_path[path] for path in task.paths] for task in tasks]

    fetched = await ir_context.to_thread(_prefetch_parquet_footers_for_paths, paths)
    info_by_path = {info.path: info for info in fetched}
    assert all(path in info_by_path for path in paths), (
        "Ordering footer metadata must contain all rank-local scan paths."
    )
    return [[info_by_path[path] for path in task.paths] for task in tasks]


def _get_ordering_candidates(
    ir: StreamingScan,
    requests: tuple[PartitioningRequest, ...],
) -> list[tuple[str, OrderKey]]:
    """Return first-key ordering candidates requested by downstream nodes."""
    candidates: list[tuple[str, OrderKey]] = []
    for request in requests:
        if isinstance(request, OrderPartitioningRequest):
            assert request.keys, (
                "Order partitioning requests must have at least one key."
            )
            key = request.keys[0]
            assert key.name in ir.schema, (
                f"Ordering request key {key.name!r} must be present in scan schema."
            )
            (column_index,) = names_to_indices((key.name,), ir.schema)
            candidate = (key.name, OrderKey(column_index, key.order, key.null_order))
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _selected_row_group_indices(
    cached_info: list[CachedParquetInfo],
    column: str,
    row_groups: list[list[int]] | None,
) -> list[int] | None:
    """Return selected row-group indices if footer stats are usable."""
    if row_groups is None:
        return None
    assert len(row_groups) == len(cached_info), (
        "Task row-group bounds must match task parquet metadata."
    )

    indices: list[int] = []
    offset = 0
    for groups, info in zip(row_groups, cached_info, strict=True):
        row_group_count = len(info.file_metadata.row_group_num_rows)
        for group in groups:
            assert 0 <= group < row_group_count, (
                f"Invalid row-group index {group} for file with "
                f"{row_group_count} row groups."
            )
            chunk = next(
                (
                    chunk
                    for chunk in info.file_metadata.row_groups[group].columns
                    if ".".join(chunk.meta_data.path_in_schema) == column
                ),
                None,
            )
            if chunk is None:
                return None
            statistics = chunk.meta_data.statistics
            if (
                statistics is None
                or statistics.null_count is None
                or statistics.null_count != 0
            ):
                return None
            indices.append(offset + group)
        offset += row_group_count
    return indices or None


def _task_chunk_bounds(
    task: ParquetScanTask,
    cached_info: list[CachedParquetInfo],
    column: str,
    order: plc.types.Order,
    null_order: plc.types.NullOrder,
    stream: Stream,
) -> plc.Column | None:
    """Return first/last task bounds for a provably ordered column."""
    row_group_indices = _selected_row_group_indices(
        cached_info, column, task._get_task_bounds(cached_info).row_groups
    )
    if row_group_indices is None:
        return None

    try:
        column_bounds = plc.io.parquet_metadata.read_parquet_column_chunk_bounds(
            [info.file_metadata for info in cached_info],
            columns=[column],
            stream=stream,
        )
    except (TypeError, ValueError, RuntimeError):
        return None

    min_max_columns = column_bounds.columns()[2:]
    assert len(min_max_columns) == 2, (
        "Single-column parquet bounds must contain min and max columns."
    )
    min_column, max_column = min_max_columns
    assert min_column.size() == max_column.size(), (
        "Parquet min/max bound columns must have matching row counts."
    )
    if min_column.null_count() or max_column.null_count():
        return None

    start_column, end_column = (
        (max_column, min_column)
        if order == plc.types.Order.DESCENDING
        else (min_column, max_column)
    )
    row_group_count = min_column.size()
    assert all(0 <= index < row_group_count for index in row_group_indices), (
        "Selected row-group indices must reference decoded parquet bounds."
    )
    row_group_bounds = plc.copying.gather(
        plc.concatenate.concatenate(
            [plc.Table([start_column]), plc.Table([end_column])],
            stream=stream,
        ),
        plc.Column.from_iterable_of_py(
            [
                index
                for group in row_group_indices
                for index in (group, row_group_count + group)
            ],
            plc.DataType(plc.TypeId.INT32),
            stream=stream,
        ),
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )
    if not plc.sorting.is_sorted(
        row_group_bounds, [order], [null_order], stream=stream
    ):
        return None

    return plc.copying.gather(
        row_group_bounds,
        plc.Column.from_iterable_of_py(
            [0, row_group_bounds.num_rows() - 1],
            plc.DataType(plc.TypeId.INT32),
            stream=stream,
        ),
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    ).columns()[0]


def _get_local_chunk_bounds(
    ir: StreamingScan,
    parquet_tasks: Sequence[ParquetScanTask],
    candidates: list[tuple[str, OrderKey]],
    task_infos: Sequence[list[CachedParquetInfo]],
    stream: Stream,
) -> plc.Table:
    """Build rank-local first/last chunk bounds for each candidate."""
    if not parquet_tasks:
        return plc.Table(
            [
                _null_bounds_column(ir, column_name, 0, stream)
                for column_name, _ in candidates
            ]
        )

    local_bound_count = 2 * len(parquet_tasks)
    bound_columns: list[plc.Column] = []
    for column_name, order_key in candidates:
        task_bounds: list[plc.Table] = []
        for task, task_info in zip(parquet_tasks, task_infos, strict=True):
            bounds_column = _task_chunk_bounds(
                task,
                task_info,
                column_name,
                order_key.order,
                order_key.null_order,
                stream,
            )
            if bounds_column is None:
                break
            task_bounds.append(plc.Table([bounds_column]))

        if len(task_bounds) == len(parquet_tasks):
            bound_columns.append(
                plc.concatenate.concatenate(task_bounds, stream=stream).columns()[0]
            )
        else:
            bound_columns.append(
                _null_bounds_column(ir, column_name, local_bound_count, stream)
            )

    return plc.Table(bound_columns)


async def parquet_ordering_partitioning(
    context: Context,
    comm: Communicator,
    ir: StreamingScan,
    global_chunk_count: int,
    requests: tuple[PartitioningRequest, ...],
    ir_context: IRExecutionContext,
    collective_id: int,
) -> Partitioning | None:
    """Extract parquet scan ordering from footer metadata, when safe."""
    if ir.base_scan.typ != "parquet" or global_chunk_count == 0:
        return None

    if not (candidates := _get_ordering_candidates(ir, requests)):
        return None

    parquet_tasks: list[ParquetScanTask] = []
    for task in ir.tasks:
        assert isinstance(task, ParquetScanTask)
        parquet_tasks.append(task)
    task_infos = await _get_task_parquet_info(ir.base_scan, parquet_tasks, ir_context)

    stream = ir_context.get_cuda_stream()
    chunk_bounds = _get_local_chunk_bounds(
        ir, parquet_tasks, candidates, task_infos, stream
    )
    if comm.nranks > 1:
        local_chunk = TableChunk.from_pylibcudf_table(
            chunk_bounds, stream, exclusive_view=True, br=context.br()
        )
        allgather = AllGatherManager(context, comm, collective_id)
        with allgather.inserting() as inserter:
            await inserter.insert(comm.rank, local_chunk)
        chunk_bounds = await allgather.extract_concatenated(
            stream, ordered=True, ir_context=ir_context
        )

    assert chunk_bounds.num_rows() == 2 * global_chunk_count, (
        "Ordering chunk bounds must contain first/last rows for every scan chunk."
    )

    for i, (_, order_key) in enumerate(candidates):
        bounds_column = chunk_bounds.columns()[i]
        if bounds_column.null_count():
            continue
        candidate_bounds = plc.Table([bounds_column])
        if not plc.sorting.is_sorted(
            candidate_bounds,
            [order_key.order],
            [order_key.null_order],
            stream=stream,
        ):
            continue

        if global_chunk_count < 2:
            ordering_boundaries = plc.Table(
                [
                    plc.Column.from_iterable_of_py([], column.type(), stream=stream)
                    for column in candidate_bounds.columns()
                ]
            )
            strict = True
        else:
            ordering_boundaries, strict = _extract_ordering_boundaries(
                candidate_bounds, global_chunk_count, stream
            )
        boundaries_chunk = TableChunk.from_pylibcudf_table(
            ordering_boundaries,
            stream,
            exclusive_view=True,
            br=context.br(),
        )
        return Partitioning(
            inter_rank=OrderScheme(
                [
                    Ordering(
                        [order_key],
                        boundaries_chunk,
                        strict_boundaries=strict,
                        locally_ordered=False,
                    )
                ]
            ),
            local="inherit",
        )
    return None
