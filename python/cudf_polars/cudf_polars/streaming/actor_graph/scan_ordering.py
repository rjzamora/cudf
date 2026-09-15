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
    from cudf_polars.streaming.base import PartitionInfo
    from cudf_polars.streaming.io import StreamingScan
    from cudf_polars.streaming.partitioning_requests import PartitioningRequest


def _scan_order_request(
    requests: tuple[PartitioningRequest, ...],
) -> OrderPartitioningRequest | None:
    """Return the first ordering request, if one exists."""
    for request in requests:
        if isinstance(request, OrderPartitioningRequest):
            return request
    return None


def _empty_endpoint_table(
    ir: StreamingScan, request: OrderPartitioningRequest, stream: Stream
) -> plc.Table:
    """Return an empty endpoint table for the requested ordering keys."""
    return plc.Table(
        [make_empty_column(ir.schema[key.name], stream) for key in request.keys]
    )


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


def _parquet_task_endpoint_rows(
    task: ParquetScanTask,
    request: OrderPartitioningRequest,
    order_keys: Sequence[OrderKey],
    stream: Stream,
) -> plc.Table | None:
    """Extract one start/end endpoint pair for a parquet scan task."""
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
        columns=[key.name for key in request.keys],
        stream=stream,
    )
    bound_columns = column_bounds.columns()[2:]
    if len(bound_columns) != 2 * len(request.keys):
        return None

    start_columns = []
    end_columns = []
    for key, min_column, max_column in zip(
        request.keys, bound_columns[::2], bound_columns[1::2], strict=True
    ):
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


async def _local_parquet_endpoint_rows_from_request(
    ir: StreamingScan,
    request: OrderPartitioningRequest,
    ir_context: IRExecutionContext,
    stream: Stream,
) -> tuple[list[OrderKey], plc.Table] | None:
    """Build rank-local ordered endpoint rows from parquet row-group statistics."""
    if not request.keys or ir.base_scan.typ != "parquet":
        return None

    tasks = list(ir.tasks)
    if not all(isinstance(task, ParquetScanTask) for task in tasks):
        return None
    parquet_tasks = [task for task in tasks if isinstance(task, ParquetScanTask)]

    try:
        column_indices = names_to_indices(
            tuple(key.name for key in request.keys), ir.schema
        )
    except ValueError:
        return None
    order_keys = [
        OrderKey(column_index, key.order, key.null_order)
        for column_index, key in zip(column_indices, request.keys, strict=True)
    ]

    if not await _ensure_cached_parquet_info(parquet_tasks, ir_context):
        return order_keys, _empty_endpoint_table(ir, request, stream)

    endpoint_rows: list[plc.Table] = []
    for task in parquet_tasks:
        endpoints = _parquet_task_endpoint_rows(task, request, order_keys, stream)
        if endpoints is None:
            return order_keys, _empty_endpoint_table(ir, request, stream)
        endpoint_rows.append(endpoints)

    return (
        order_keys,
        plc.concatenate.concatenate(endpoint_rows, stream=stream)
        if endpoint_rows
        else _empty_endpoint_table(ir, request, stream),
    )


async def parquet_ordering_partitioning(
    context: Context,
    comm: Communicator,
    ir: StreamingScan,
    partition_info: PartitionInfo,
    requests: tuple[PartitioningRequest, ...],
    ir_context: IRExecutionContext,
    collective_id: int | None,
) -> Partitioning | None:
    """Extract parquet scan ordering from footer metadata, when safe."""
    request = _scan_order_request(requests)
    if request is None or partition_info.io_plan is None:
        return None

    stream = ir_context.get_cuda_stream()
    result = await _local_parquet_endpoint_rows_from_request(
        ir,
        request,
        ir_context,
        stream,
    )
    if result is None:
        return None

    order_keys, endpoint_rows = result
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

    if endpoint_rows.num_rows() != 2 * partition_info.count:
        return None

    column_order = [key.order for key in order_keys]
    null_order = [key.null_order for key in order_keys]
    if not plc.sorting.is_sorted(
        endpoint_rows, column_order, null_order, stream=stream
    ):
        return None

    num_partitions = partition_info.count
    if num_partitions == 0:
        return None
    if num_partitions < 2:
        boundaries = plc.Table(
            [
                plc.Column.from_iterable_of_py([], column.type(), stream=stream)
                for column in endpoint_rows.columns()
            ]
        )
        strict = True
    else:
        boundaries, strict = _extract_boundaries_from_endpoint_rows(
            endpoint_rows, num_partitions, stream
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
