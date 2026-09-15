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
    _extract_boundaries_from_endpoint_rows,
)
from cudf_polars.streaming.io import ParquetScanTask
from cudf_polars.streaming.partitioning_requests import OrderPartitioningRequest
from cudf_polars.utils.dtypes import make_empty_column

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import CachedParquetInfo, IRExecutionContext
    from cudf_polars.streaming.io import StreamingScan
    from cudf_polars.streaming.partitioning_requests import PartitioningRequest


def _empty_endpoint_column(
    ir: StreamingScan, request: OrderPartitioningRequest, stream: Stream
) -> plc.Column:
    """Return an empty endpoint column for a single-column ordering request."""
    return make_empty_column(ir.schema[request.keys[0].name], stream)


def _null_endpoint_column(
    ir: StreamingScan,
    request: OrderPartitioningRequest,
    size: int,
    stream: Stream,
) -> plc.Column:
    """Return a null endpoint column to invalidate one ordering candidate."""
    empty = _empty_endpoint_column(ir, request, stream)
    if size == 0:
        return empty
    return plc.Column.all_null_like(empty, size, stream=stream)


async def _ensure_cached_parquet_info(
    tasks: Sequence[ParquetScanTask],
    ir_context: IRExecutionContext,
) -> bool:
    """Ensure rank-local scan task paths have cached parquet metadata."""
    if not tasks:
        return True

    base_scan = tasks[0].base_scan
    paths = list(dict.fromkeys(path for task in tasks for path in task.paths))
    cached_by_path = {
        info.path: info
        for info in (base_scan.cached_parquet_info or ())
        if info.path in paths
    }
    missing_paths = [path for path in paths if path not in cached_by_path]
    if missing_paths:
        from cudf_polars.dsl.utils.io import _prefetch_parquet_footers_for_paths

        fetched = await ir_context.to_thread(
            _prefetch_parquet_footers_for_paths, missing_paths
        )
        cached_by_path.update({info.path: info for info in fetched})

    if not all(path in cached_by_path for path in paths):
        return False
    base_scan.cached_parquet_info = [cached_by_path[path] for path in paths]
    return True


def _flat_row_group_indices(
    row_groups: list[list[int]], cached_info: Sequence[CachedParquetInfo]
) -> list[int] | None:
    """Translate per-path row-group indices into flat bounds-table row indices."""
    if len(row_groups) != len(cached_info):
        return None

    flat_indices: list[int] = []
    offset = 0
    for groups, info in zip(row_groups, cached_info, strict=True):
        row_group_count = len(info.file_metadata.row_group_num_rows)
        if not all(0 <= group < row_group_count for group in groups):
            return None
        flat_indices.extend(offset + group for group in groups)
        offset += row_group_count
    return flat_indices


def _parquet_task_column_bounds(
    task: ParquetScanTask,
    columns: Sequence[str],
    stream: Stream,
) -> tuple[plc.Table, list[int]] | None:
    """Read requested column bounds for one row-group-aligned parquet task."""
    bounds = task.get_task_bounds()
    if bounds.row_groups is None or bounds.skip_rows != 0 or bounds.n_rows != -1:
        return None

    cached_info = task._get_cached_parquet_info()
    if cached_info is None:
        return None

    flat_indices = _flat_row_group_indices(bounds.row_groups, cached_info)
    if not flat_indices:
        return None

    column_bounds = plc.io.parquet_metadata.read_parquet_column_chunk_bounds(
        [info.file_metadata for info in cached_info],
        columns=columns,
        stream=stream,
    )
    if len(column_bounds.columns()[2:]) != 2 * len(columns):
        return None
    return column_bounds, flat_indices


def _candidate_endpoint_rows(
    column_bounds: plc.Table,
    flat_indices: list[int],
    request: OrderPartitioningRequest,
    order_keys: Sequence[OrderKey],
    column_positions: dict[str, int],
    stream: Stream,
) -> plc.Table | None:
    """Extract one start/end endpoint pair for a candidate ordering."""
    bound_columns = column_bounds.columns()[2:]

    start_columns = []
    end_columns = []
    for key in request.keys:
        position = column_positions[key.name]
        min_column = bound_columns[2 * position]
        max_column = bound_columns[2 * position + 1]
        if min_column.null_count() or max_column.null_count():
            return None
        if key.order == plc.types.Order.DESCENDING:
            start_column, end_column = max_column, min_column
        else:
            start_column, end_column = min_column, max_column
        start_columns.append(start_column)
        end_columns.append(end_column)
    endpoint_source = plc.concatenate.concatenate(
        [plc.Table(start_columns), plc.Table(end_columns)],
        stream=stream,
    )
    row_group_count = column_bounds.num_rows()
    endpoint_indices = [
        index
        for row_group_index in flat_indices
        for index in (row_group_index, row_group_count + row_group_index)
    ]
    row_group_endpoints = plc.copying.gather(
        endpoint_source,
        plc.Column.from_iterable_of_py(
            endpoint_indices,
            plc.DataType(plc.TypeId.INT32),
            stream=stream,
        ),
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )
    order = [key.order for key in order_keys]
    null_order = [key.null_order for key in order_keys]
    if not plc.sorting.is_sorted(row_group_endpoints, order, null_order, stream=stream):
        return None

    return plc.copying.gather(
        row_group_endpoints,
        plc.Column.from_iterable_of_py(
            [0, row_group_endpoints.num_rows() - 1],
            plc.DataType(plc.TypeId.INT32),
            stream=stream,
        ),
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )


async def _local_parquet_endpoint_rows_from_requests(
    ir: StreamingScan,
    requests: tuple[PartitioningRequest, ...],
    ir_context: IRExecutionContext,
    stream: Stream,
) -> tuple[list[list[OrderKey]], plc.Table] | None:
    """Build rank-local endpoint rows for candidate scan orderings."""
    if ir.base_scan.typ != "parquet":
        return None

    parquet_tasks = [task for task in ir.tasks if isinstance(task, ParquetScanTask)]
    if len(parquet_tasks) != len(ir.tasks):
        return None

    candidates: list[tuple[OrderPartitioningRequest, list[OrderKey]]] = []
    columns: list[str] = []
    for request in requests:
        # Parquet footer min/max statistics can prove single-column ordering,
        # but not arbitrary lexicographic multi-column ordering.
        if not isinstance(request, OrderPartitioningRequest) or len(request.keys) != 1:
            continue
        try:
            column_indices = names_to_indices(
                tuple(key.name for key in request.keys), ir.schema
            )
        except ValueError:
            continue
        candidates.append(
            (
                request,
                [
                    OrderKey(column_index, key.order, key.null_order)
                    for column_index, key in zip(
                        column_indices, request.keys, strict=True
                    )
                ],
            )
        )
        columns.extend(key.name for key in request.keys)

    if not candidates:
        return None

    candidate_order_keys = [order_keys for _, order_keys in candidates]
    local_endpoint_count = 2 * len(parquet_tasks)

    if not await _ensure_cached_parquet_info(parquet_tasks, ir_context):
        # Return null endpoints instead of None so every rank still participates
        # in the allgather. Candidate evaluation rejects any column containing
        # nulls after the allgather.
        return candidate_order_keys, plc.Table(
            [
                _null_endpoint_column(ir, request, local_endpoint_count, stream)
                for request, _ in candidates
            ]
        )

    columns = list(dict.fromkeys(columns))
    column_positions = {name: i for i, name in enumerate(columns)}
    task_bounds: list[tuple[plc.Table, list[int]]] = []
    for task in parquet_tasks:
        bounds = _parquet_task_column_bounds(task, columns, stream)
        if bounds is None:
            return candidate_order_keys, plc.Table(
                [
                    _null_endpoint_column(ir, request, local_endpoint_count, stream)
                    for request, _ in candidates
                ]
            )
        task_bounds.append(bounds)

    endpoint_columns: list[plc.Column] = []
    for request, order_keys in candidates:
        endpoint_rows: list[plc.Table] = []
        failed = False
        for column_bounds, flat_indices in task_bounds:
            endpoints = _candidate_endpoint_rows(
                column_bounds,
                flat_indices,
                request,
                order_keys,
                column_positions,
                stream,
            )
            if endpoints is None:
                failed = True
                break
            endpoint_rows.append(endpoints)
        if failed:
            endpoint_columns.append(
                _null_endpoint_column(ir, request, local_endpoint_count, stream)
            )
        elif endpoint_rows:
            endpoint_columns.append(
                plc.concatenate.concatenate(endpoint_rows, stream=stream).columns()[0]
            )
        else:
            endpoint_columns.append(_empty_endpoint_column(ir, request, stream))

    return candidate_order_keys, plc.Table(endpoint_columns)


async def parquet_ordering_partitioning(
    context: Context,
    comm: Communicator,
    ir: StreamingScan,
    global_chunk_count: int,
    requests: tuple[PartitioningRequest, ...],
    ir_context: IRExecutionContext,
    collective_id: int | None,
) -> Partitioning | None:
    """Extract parquet scan ordering from footer metadata, when safe."""
    if global_chunk_count == 0:
        return None

    stream = ir_context.get_cuda_stream()
    local_results = await _local_parquet_endpoint_rows_from_requests(
        ir,
        requests,
        ir_context,
        stream,
    )
    if local_results is None:
        return None

    candidate_order_keys, endpoint_rows = local_results
    if comm.nranks > 1:
        if collective_id is None:
            return None
        local_chunk = TableChunk.from_pylibcudf_table(
            endpoint_rows, stream, exclusive_view=True, br=context.br()
        )
        allgather = AllGatherManager(context, comm, collective_id)
        with allgather.inserting() as inserter:
            await inserter.insert(comm.rank, local_chunk)
        endpoint_rows = await allgather.extract_concatenated(
            stream, ordered=True, ir_context=ir_context
        )

    if endpoint_rows.num_rows() != 2 * global_chunk_count:
        return None

    for i, order_keys in enumerate(candidate_order_keys):
        endpoint_column = endpoint_rows.columns()[i]
        if endpoint_column.null_count():
            continue
        endpoint_column_rows = plc.Table([endpoint_column])
        column_order = [key.order for key in order_keys]
        null_order = [key.null_order for key in order_keys]
        if not plc.sorting.is_sorted(
            endpoint_column_rows, column_order, null_order, stream=stream
        ):
            continue

        if global_chunk_count < 2:
            boundaries = plc.Table(
                [
                    plc.Column.from_iterable_of_py([], column.type(), stream=stream)
                    for column in endpoint_column_rows.columns()
                ]
            )
            strict = True
        else:
            boundaries, strict = _extract_boundaries_from_endpoint_rows(
                endpoint_column_rows, global_chunk_count, stream
            )
        boundaries_chunk = TableChunk.from_pylibcudf_table(
            boundaries,
            stream,
            exclusive_view=True,
            br=context.br(),
        )
        return Partitioning(
            inter_rank=OrderScheme(
                [
                    Ordering(
                        order_keys,
                        boundaries_chunk,
                        strict_boundaries=strict,
                        locally_ordered=False,
                    )
                ]
            ),
            local="inherit",
        )
    return None
