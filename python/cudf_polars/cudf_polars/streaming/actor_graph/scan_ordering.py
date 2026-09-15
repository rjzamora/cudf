# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract ordering metadata from scan tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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
    _extract_boundaries_from_endpoint_rows as _make_ordering_boundaries,
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


def _invalid_chunk_boundary_column(
    ir: StreamingScan,
    request: OrderPartitioningRequest,
    size: int,
    stream: Stream,
) -> plc.Column:
    """Return a null chunk-boundary column to invalidate one candidate."""
    empty = make_empty_column(ir.schema[request.keys[0].name], stream)
    if size == 0:
        return empty
    return plc.Column.all_null_like(empty, size, stream=stream)


async def _parquet_info_for_ordering(
    base_scan: Scan,
    tasks: Sequence[ParquetScanTask],
    ir_context: IRExecutionContext,
) -> dict[str, CachedParquetInfo] | None:
    """Return footer metadata for rank-local ordering extraction."""
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
        return None
    return cached_by_path


async def _local_parquet_chunk_boundaries_from_requests(
    ir: StreamingScan,
    parquet_tasks: Sequence[ParquetScanTask],
    requests: tuple[PartitioningRequest, ...],
    ir_context: IRExecutionContext,
    stream: Stream,
) -> tuple[list[OrderKey], plc.Table] | None:
    """Build rank-local chunk boundary rows for candidate scan orderings."""
    candidates: list[tuple[OrderPartitioningRequest, OrderKey]] = []
    for request in requests:
        # Parquet footer min/max statistics can prove single-column ordering,
        # but not arbitrary lexicographic multi-column ordering.
        if not isinstance(request, OrderPartitioningRequest) or len(request.keys) != 1:
            continue
        (key,) = request.keys
        try:
            (column_index,) = names_to_indices((key.name,), ir.schema)
        except ValueError:
            continue
        candidates.append((request, OrderKey(column_index, key.order, key.null_order)))

    if not candidates:
        return None

    candidate_order_keys = [order_key for _, order_key in candidates]
    local_chunk_boundary_count = 2 * len(parquet_tasks)

    parquet_info = await _parquet_info_for_ordering(
        ir.base_scan, parquet_tasks, ir_context
    )
    if parquet_info is None:
        # Return null boundaries instead of None so every rank still participates
        # in the allgather. Candidate evaluation rejects any column containing
        # nulls after the allgather.
        return candidate_order_keys, plc.Table(
            [
                _invalid_chunk_boundary_column(
                    ir, request, local_chunk_boundary_count, stream
                )
                for request, _ in candidates
            ]
        )

    chunk_boundary_columns: list[plc.Column] = []
    for request, _order_key in candidates:
        (key,) = request.keys
        chunk_boundary_rows: list[plc.Table] = []
        for task in parquet_tasks:
            task_info = [parquet_info[path] for path in task.paths]
            chunk_boundary_column = task.get_ordered_boundaries(
                task_info,
                key.name,
                key.order,
                key.null_order,
                stream,
            )
            if chunk_boundary_column is None:
                break
            chunk_boundary_rows.append(plc.Table([chunk_boundary_column]))
        if len(chunk_boundary_rows) != len(parquet_tasks):
            chunk_boundary_columns.append(
                _invalid_chunk_boundary_column(
                    ir, request, local_chunk_boundary_count, stream
                )
            )
        elif chunk_boundary_rows:
            chunk_boundary_columns.append(
                plc.concatenate.concatenate(
                    chunk_boundary_rows, stream=stream
                ).columns()[0]
            )
        else:
            chunk_boundary_columns.append(
                _invalid_chunk_boundary_column(ir, request, 0, stream)
            )

    return candidate_order_keys, plc.Table(chunk_boundary_columns)


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
    if ir.base_scan.typ != "parquet" or global_chunk_count == 0:
        return None
    assert all(isinstance(task, ParquetScanTask) for task in ir.tasks)
    parquet_tasks = cast("Sequence[ParquetScanTask]", ir.tasks)

    stream = ir_context.get_cuda_stream()
    local_results = await _local_parquet_chunk_boundaries_from_requests(
        ir,
        parquet_tasks,
        requests,
        ir_context,
        stream,
    )
    if local_results is None:
        return None

    candidate_order_keys, chunk_boundary_rows = local_results
    if comm.nranks > 1:
        if collective_id is None:
            return None
        local_chunk = TableChunk.from_pylibcudf_table(
            chunk_boundary_rows, stream, exclusive_view=True, br=context.br()
        )
        allgather = AllGatherManager(context, comm, collective_id)
        with allgather.inserting() as inserter:
            await inserter.insert(comm.rank, local_chunk)
        chunk_boundary_rows = await allgather.extract_concatenated(
            stream, ordered=True, ir_context=ir_context
        )

    if chunk_boundary_rows.num_rows() != 2 * global_chunk_count:
        return None

    for i, order_key in enumerate(candidate_order_keys):
        chunk_boundary_column = chunk_boundary_rows.columns()[i]
        if chunk_boundary_column.null_count():
            continue
        chunk_boundary_column_rows = plc.Table([chunk_boundary_column])
        if not plc.sorting.is_sorted(
            chunk_boundary_column_rows,
            [order_key.order],
            [order_key.null_order],
            stream=stream,
        ):
            continue

        if global_chunk_count < 2:
            ordering_boundaries = plc.Table(
                [
                    plc.Column.from_iterable_of_py([], column.type(), stream=stream)
                    for column in chunk_boundary_column_rows.columns()
                ]
            )
            strict = True
        else:
            ordering_boundaries, strict = _make_ordering_boundaries(
                chunk_boundary_column_rows, global_chunk_count, stream
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
