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


def _null_boundary_column(
    ir: StreamingScan,
    column_name: str,
    size: int,
    stream: Stream,
) -> plc.Column:
    """Return a null boundary column to invalidate one candidate."""
    empty = make_empty_column(ir.schema[column_name], stream)
    if size == 0:
        return empty
    return plc.Column.all_null_like(empty, size, stream=stream)


async def _get_parquet_info(
    base_scan: Scan,
    tasks: Sequence[ParquetScanTask],
    ir_context: IRExecutionContext,
) -> dict[str, CachedParquetInfo]:
    """Return cached or freshly fetched footer metadata for rank-local paths."""
    paths = list(dict.fromkeys(path for task in tasks for path in task.paths))
    cached_by_path = {
        info.path: info
        for info in (base_scan.cached_parquet_info or ())
        if info.path in paths
    }
    if set(cached_by_path) != set(paths):
        from cudf_polars.dsl.utils.io import _prefetch_parquet_footers_for_paths

        fetched = await ir_context.to_thread(_prefetch_parquet_footers_for_paths, paths)
        cached_by_path = {info.path: info for info in fetched}

    assert all(path in cached_by_path for path in paths), (
        "Ordering footer metadata must contain all rank-local scan paths."
    )
    return cached_by_path


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


def _get_local_boundaries(
    ir: StreamingScan,
    parquet_tasks: Sequence[ParquetScanTask],
    candidates: list[tuple[str, OrderKey]],
    parquet_info: dict[str, CachedParquetInfo],
    stream: Stream,
) -> plc.Table:
    """Build rank-local chunk boundaries for each candidate."""
    if not parquet_tasks:
        return plc.Table(
            [
                _null_boundary_column(ir, column_name, 0, stream)
                for column_name, _ in candidates
            ]
        )

    local_boundary_count = 2 * len(parquet_tasks)
    boundary_columns: list[plc.Column] = []
    for column_name, order_key in candidates:
        task_boundaries: list[plc.Table] = []
        for task in parquet_tasks:
            task_info = [parquet_info[path] for path in task.paths]
            boundary_column = task.get_ordered_boundaries(
                task_info,
                column_name,
                order_key.order,
                order_key.null_order,
                stream,
            )
            if boundary_column is None:
                break
            task_boundaries.append(plc.Table([boundary_column]))

        if len(task_boundaries) == len(parquet_tasks):
            boundary_columns.append(
                plc.concatenate.concatenate(task_boundaries, stream=stream).columns()[0]
            )
        else:
            boundary_columns.append(
                _null_boundary_column(ir, column_name, local_boundary_count, stream)
            )

    return plc.Table(boundary_columns)


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

    assert all(isinstance(task, ParquetScanTask) for task in ir.tasks)
    parquet_tasks = cast("Sequence[ParquetScanTask]", ir.tasks)
    parquet_info = await _get_parquet_info(ir.base_scan, parquet_tasks, ir_context)

    stream = ir_context.get_cuda_stream()
    boundary_rows = _get_local_boundaries(
        ir, parquet_tasks, candidates, parquet_info, stream
    )
    if comm.nranks > 1:
        local_chunk = TableChunk.from_pylibcudf_table(
            boundary_rows, stream, exclusive_view=True, br=context.br()
        )
        allgather = AllGatherManager(context, comm, collective_id)
        with allgather.inserting() as inserter:
            await inserter.insert(comm.rank, local_chunk)
        boundary_rows = await allgather.extract_concatenated(
            stream, ordered=True, ir_context=ir_context
        )

    assert boundary_rows.num_rows() == 2 * global_chunk_count, (
        "Ordering boundary rows must contain first/last rows for every scan chunk."
    )

    for i, (_, order_key) in enumerate(candidates):
        boundary_column = boundary_rows.columns()[i]
        if boundary_column.null_count():
            continue
        candidate_boundaries = plc.Table([boundary_column])
        if not plc.sorting.is_sorted(
            candidate_boundaries,
            [order_key.order],
            [order_key.null_order],
            stream=stream,
        ):
            continue

        if global_chunk_count < 2:
            ordering_boundaries = plc.Table(
                [
                    plc.Column.from_iterable_of_py([], column.type(), stream=stream)
                    for column in candidate_boundaries.columns()
                ]
            )
            strict = True
        else:
            ordering_boundaries, strict = _extract_ordering_boundaries(
                candidate_boundaries, global_chunk_count, stream
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
