# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Infer scan ordering metadata from Parquet row-group statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    OrderKey,
    OrderScheme,
    Ordering,
    Partitioning,
)
from cudf_streaming.table_chunk import TableChunk

from cudf_polars.dsl.utils.io import _prefetch_parquet_footers_for_paths
from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.collectives.sort import (
    _extract_boundaries_from_endpoint_rows,
)
from cudf_polars.streaming.base import IOPartitionFlavor
from cudf_polars.streaming.partitioning_requests import OrderPartitioningRequest
from cudf_polars.utils.dtypes import make_empty_column

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.dsl.utils.io import CachedParquetInfo
    from cudf_polars.streaming.base import IOPartitionPlan, PartitionInfo
    from cudf_polars.streaming.io import FusedScan, SplitScan, StreamingScan
    from cudf_polars.streaming.partitioning_requests import (
        NamedOrderKey,
        PartitioningRequest,
    )


@dataclass(frozen=True)
class _ScanOrderCandidate:
    """Single-column ordering request that a parquet scan may satisfy."""

    key: NamedOrderKey
    order_key: OrderKey


def _scan_order_candidates(
    ir: StreamingScan,
    requests: tuple[PartitioningRequest, ...],
) -> tuple[_ScanOrderCandidate, ...]:
    """Return unique single-column prefixes of downstream ordering requests."""
    candidates: list[_ScanOrderCandidate] = []
    seen: set[tuple[str, plc.types.Order, plc.types.NullOrder]] = set()
    for request in requests:
        if not isinstance(request, OrderPartitioningRequest) or not request.keys:
            continue
        key = request.keys[0]
        token = (key.name, key.order, key.null_order)
        if token in seen:
            continue
        try:
            (column_index,) = names_to_indices((key.name,), ir.schema)
        except ValueError:
            continue
        seen.add(token)
        candidates.append(
            _ScanOrderCandidate(
                key,
                OrderKey(column_index, key.order, key.null_order),
            )
        )
    return tuple(candidates)


def _empty_endpoint_table(
    ir: StreamingScan, candidates: Sequence[_ScanOrderCandidate], stream: Stream
) -> plc.Table:
    """Return an empty endpoint table for the requested ordering keys."""
    return plc.Table(
        [
            make_empty_column(ir.schema[candidate.key.name], stream)
            for candidate in candidates
        ]
        + [
            plc.column_factories.make_empty_column(
                plc.DataType(plc.TypeId.BOOL8), stream=stream
            )
            for _ in candidates
        ]
    )


def _cached_parquet_info_matches_paths(
    cached_info: Sequence[CachedParquetInfo], paths: Sequence[str]
) -> bool:
    """Return whether cached parquet metadata exactly covers ``paths``."""
    return [info.path for info in cached_info] == list(paths)


async def _scan_cached_parquet_info(
    scan: FusedScan | SplitScan,
    ir_context: IRExecutionContext,
    cache_by_paths: dict[tuple[str, ...], list[CachedParquetInfo]],
) -> list[CachedParquetInfo]:
    """Return cached parquet metadata for one local scan task."""
    if scan.cached_parquet_info is not None:
        existing = list(scan.cached_parquet_info)
        if _cached_parquet_info_matches_paths(existing, scan.paths):
            return existing
        scan.cached_parquet_info = None
        scan._non_child_args = (*scan._non_child_args[:-1], None)

    paths_key = tuple(scan.paths)
    cached = cache_by_paths.get(paths_key)
    if cached is None:
        fetched = await ir_context.to_thread(
            _prefetch_parquet_footers_for_paths, list(scan.paths)
        )
        if not _cached_parquet_info_matches_paths(fetched, scan.paths):
            return []
        cache_by_paths[paths_key] = fetched
        cached = fetched

    scan.cached_parquet_info = cached
    scan._non_child_args = (*scan._non_child_args[:-1], cached)
    return cached


def _split_scan_row_group_range(
    scan: SplitScan, cached_info: Sequence[CachedParquetInfo]
) -> tuple[int, int] | None:
    """Return the row-group range read by a row-group-aligned split scan."""
    if len(cached_info) != 1:
        return None
    total_row_groups = len(cached_info[0].file_metadata.row_group_num_rows)
    if scan.total_splits > total_row_groups:
        return None

    row_group_stride = total_row_groups // scan.total_splits
    start = row_group_stride * scan.split_index
    stop = (
        total_row_groups
        if scan.split_index == scan.total_splits - 1
        else start + row_group_stride
    )
    return (start, stop) if start < stop else None


def _column_chunk_stats_for(
    cached_info: Sequence[CachedParquetInfo],
    column_name: str,
    row_group_range: tuple[int, int] | None,
) -> list[tuple[int, Any]] | None:
    """Return ``(num_rows, statistics)`` pairs for selected row groups."""
    if row_group_range is not None and len(cached_info) != 1:
        return None

    result: list[tuple[int, Any]] = []
    for info in cached_info:
        row_groups = info.file_metadata.row_groups
        start, stop = (
            row_group_range if row_group_range is not None else (0, len(row_groups))
        )
        for row_group in row_groups[start:stop]:
            matches = [
                column.meta_data.statistics
                for column in row_group.columns
                if ".".join(column.meta_data.path_in_schema) == column_name
            ]
            if len(matches) != 1:
                return None
            result.append((row_group.num_rows, matches[0]))
    return result


def _allows_nulls_for_partition(
    key: NamedOrderKey, partition_index: int, partition_count: int
) -> bool:
    """Return whether null keys may appear in this ordered partition."""
    if key.null_order == plc.types.NullOrder.BEFORE:
        return partition_index == 0
    return partition_index == partition_count - 1


def _statistics_support_ordering(
    cached_info: Sequence[CachedParquetInfo],
    key: NamedOrderKey,
    partition_index: int,
    partition_count: int,
    row_group_range: tuple[int, int] | None,
) -> bool:
    """Return whether parquet statistics can prove scan range partitioning."""
    stats = _column_chunk_stats_for(cached_info, key.name, row_group_range)
    if not stats:
        return False

    allow_nulls = _allows_nulls_for_partition(key, partition_index, partition_count)
    for num_rows, statistics in stats:
        if statistics is None or num_rows == 0:
            return False
        null_count = statistics.null_count
        if null_count is not None and null_count > num_rows:
            return False
        if null_count and not allow_nulls:
            return False
        if null_count == num_rows and allow_nulls:
            continue
        if null_count is None and not allow_nulls:
            return False
        if (
            statistics.is_min_value_exact is False
            or statistics.is_max_value_exact is False
        ):
            return False
        if not statistics.has_min_max:
            return False
    return True


def _null_endpoint_column(like: plc.Column, stream: Stream) -> plc.Column:
    """Return two null endpoint values for an invalid local candidate."""
    return plc.Column.all_null_like(like, 2, stream=stream)


def _parquet_scan_endpoint_column_from_bounds(
    bounds: plc.Table,
    candidate: _ScanOrderCandidate,
    bounds_column_index: int,
    stream: Stream,
    row_group_range: tuple[int, int] | None = None,
) -> plc.Column | None:
    """Extract a scan endpoint pair from row-group min/max statistics."""
    row_group_count = bounds.num_rows()
    start, stop = (
        row_group_range if row_group_range is not None else (0, row_group_count)
    )
    if row_group_count == 0 or not 0 <= start < stop <= row_group_count:
        return None

    columns = bounds.columns()
    min_column = columns[bounds_column_index]
    max_column = columns[bounds_column_index + 1]
    endpoint_source = plc.concatenate.concatenate(
        [plc.Table([min_column]), plc.Table([max_column])],
        stream=stream,
    )
    endpoint_indices = [
        index
        for row_group in range(start, stop)
        for index in (row_group, row_group_count + row_group)
    ]
    endpoints = plc.copying.gather(
        endpoint_source,
        plc.Column.from_iterable_of_py(
            endpoint_indices,
            plc.DataType(plc.TypeId.INT32),
            stream=stream,
        ),
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )
    endpoints = plc.sorting.sort(
        endpoints,
        [candidate.key.order],
        [candidate.key.null_order],
        stream=stream,
    )

    return plc.copying.gather(
        endpoints,
        plc.Column.from_iterable_of_py(
            [0, endpoints.num_rows() - 1],
            plc.DataType(plc.TypeId.INT32),
            stream=stream,
        ),
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    ).columns()[0]


def _parquet_scan_endpoint_table_from_bounds(
    cached_info: Sequence[CachedParquetInfo],
    bounds: plc.Table,
    candidates: Sequence[_ScanOrderCandidate],
    column_indices: dict[str, int],
    partition_index: int,
    partition_count: int,
    stream: Stream,
    row_group_range: tuple[int, int] | None = None,
) -> plc.Table:
    """Return scan endpoint rows and per-candidate validity columns."""
    endpoint_columns: list[plc.Column] = []
    valid_columns: list[plc.Column] = []
    for candidate in candidates:
        endpoint_column_index = 2 + 2 * column_indices[candidate.key.name]
        valid = _statistics_support_ordering(
            cached_info,
            candidate.key,
            partition_index,
            partition_count,
            row_group_range,
        )
        endpoint_column = (
            _parquet_scan_endpoint_column_from_bounds(
                bounds,
                candidate,
                endpoint_column_index,
                stream,
                row_group_range,
            )
            if valid
            else None
        )
        if endpoint_column is None:
            valid = False
            endpoint_column = _null_endpoint_column(
                bounds.columns()[endpoint_column_index], stream
            )
        endpoint_columns.append(endpoint_column)
        valid_columns.append(
            plc.Column.from_iterable_of_py(
                [valid, valid],
                plc.DataType(plc.TypeId.BOOL8),
                stream=stream,
            )
        )
    return plc.Table(endpoint_columns + valid_columns)


async def _local_parquet_endpoint_rows_from_requests(
    ir: StreamingScan,
    plan: IOPartitionPlan,
    candidates: Sequence[_ScanOrderCandidate],
    partition_count: int,
    local_partition_start: int,
    ir_context: IRExecutionContext,
    stream: Stream,
) -> plc.Table | None:
    """Build rank-local ordered endpoint rows from parquet row-group statistics."""
    if not candidates or ir.base_scan.typ != "parquet":
        return None

    scans = [cast("FusedScan | SplitScan", scan) for scan in ir.scans]
    cache_by_paths: dict[tuple[str, ...], list[CachedParquetInfo]] = {}
    cached_info_by_scan = [
        await _scan_cached_parquet_info(scan, ir_context, cache_by_paths)
        for scan in scans
    ]
    if any(
        not _cached_parquet_info_matches_paths(cached_info, scan.paths)
        for scan, cached_info in zip(scans, cached_info_by_scan, strict=True)
    ):
        return _empty_endpoint_table(ir, candidates, stream)

    column_names = tuple(dict.fromkeys(candidate.key.name for candidate in candidates))
    column_indices = {name: index for index, name in enumerate(column_names)}
    endpoint_rows: list[plc.Table] = []
    for scan_idx, (scan, cached_info) in enumerate(
        zip(scans, cached_info_by_scan, strict=True)
    ):
        row_group_range = (
            _split_scan_row_group_range(cast("SplitScan", scan), cached_info)
            if plan.flavor == IOPartitionFlavor.SPLIT_FILES
            else None
        )
        if plan.flavor == IOPartitionFlavor.SPLIT_FILES and row_group_range is None:
            return _empty_endpoint_table(ir, candidates, stream)

        try:
            bounds = plc.io.parquet_metadata.read_parquet_column_chunk_bounds(
                [info.file_metadata for info in cached_info],
                columns=column_names,
                stream=stream,
            )
        except (RuntimeError, ValueError):
            return _empty_endpoint_table(ir, candidates, stream)
        endpoints = _parquet_scan_endpoint_table_from_bounds(
            cached_info,
            bounds,
            candidates,
            column_indices,
            local_partition_start + scan_idx,
            partition_count,
            stream,
            row_group_range,
        )
        endpoint_rows.append(endpoints)

    if endpoint_rows:
        return plc.concatenate.concatenate(endpoint_rows, stream=stream)
    return _empty_endpoint_table(ir, candidates, stream)


def _all_true(column: plc.Column, stream: Stream) -> bool:
    """Return whether every row in a boolean column is true."""
    return bool(
        plc.reduce.reduce(
            column,
            plc.aggregation.all(),
            plc.DataType(plc.TypeId.BOOL8),
            stream=stream,
        ).to_py(stream=stream)
    )


async def extract_parquet_ordering_partitioning(
    context: Context,
    comm: Communicator,
    ir: StreamingScan,
    partition_info: PartitionInfo,
    requests: tuple[PartitioningRequest, ...],
    ir_context: IRExecutionContext,
    collective_id: int,
) -> Partitioning | None:
    """Extract inter-rank scan ordering from parquet metadata, when safe."""
    candidates = _scan_order_candidates(ir, requests)
    if not candidates or partition_info.io_plan is None:
        return None

    stream = ir_context.get_cuda_stream()
    local_partition_start = math.ceil(partition_info.count / comm.nranks) * comm.rank
    endpoint_rows = await _local_parquet_endpoint_rows_from_requests(
        ir,
        partition_info.io_plan,
        candidates,
        partition_info.count,
        local_partition_start,
        ir_context,
        stream,
    )
    if endpoint_rows is None:
        return None

    if comm.nranks > 1:
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

    num_partitions = partition_info.count
    if num_partitions == 0:
        return None

    endpoint_columns = endpoint_rows.columns()
    orderings: list[Ordering] = []
    for index, candidate in enumerate(candidates):
        if not _all_true(endpoint_columns[len(candidates) + index], stream):
            continue
        candidate_endpoint_rows = plc.Table([endpoint_columns[index]])
        if not plc.sorting.is_sorted(
            candidate_endpoint_rows,
            [candidate.key.order],
            [candidate.key.null_order],
            stream=stream,
        ):
            continue
        if num_partitions == 1:
            boundaries = plc.Table(
                [
                    plc.Column.from_iterable_of_py(
                        [], candidate_endpoint_rows.columns()[0].type(), stream=stream
                    )
                ]
            )
            strict = True
        else:
            boundaries, strict = _extract_boundaries_from_endpoint_rows(
                candidate_endpoint_rows, num_partitions, stream
            )
        boundaries_chunk = TableChunk.from_pylibcudf_table(
            boundaries,
            stream,
            exclusive_view=True,
            br=context.br(),
        )
        orderings.append(
            Ordering(
                [candidate.order_key],
                boundaries_chunk,
                strict_boundaries=strict,
                locally_ordered=False,
            )
        )
    if not orderings:
        return None
    return Partitioning(
        inter_rank=OrderScheme(orderings),
        local="inherit",
    )


def parquet_ordering_decision(partitioning: Partitioning | None) -> str | None:
    """Return a trace decision for scan ordering extracted from parquet stats."""
    if partitioning is None or not isinstance(partitioning.inter_rank, OrderScheme):
        return None
    strict_values = {
        ordering.strict_boundaries for ordering in partitioning.inter_rank.orderings
    }
    if strict_values == {True}:
        return "parquet_ordering_strict"
    if strict_values == {False}:
        return "parquet_ordering_non_strict"
    return "parquet_ordering_mixed"
