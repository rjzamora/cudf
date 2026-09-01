# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IO logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
import functools
import io
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    OrderKey,
    OrderScheme,
    Ordering,
    Partitioning,
)
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, DataFrameScan, PythonScan, Sink
from cudf_polars.dsl.tracing import Scope, log
from cudf_polars.dsl.utils.io import _prefetch_parquet_footers_for_paths
from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.collectives.sort import (
    _extract_boundaries_from_endpoint_rows,
)
from cudf_polars.streaming.actor_graph.dispatch import generate_ir_sub_network
from cudf_polars.streaming.actor_graph.nodes import define_actor, shutdown_on_error
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    chunk_to_frame,
    empty_table_chunk,
    gather_in_task_group,
    process_children,
    recv_metadata,
    send_metadata,
)
from cudf_polars.streaming.base import IOPartitionFlavor
from cudf_polars.streaming.io import (
    StreamingScan,
    StreamingSink,
    _prepare_sink_directory,
    _sink_to_file,
)
from cudf_polars.streaming.partitioning_requests import OrderPartitioningRequest
from cudf_polars.streaming.rank_aware_source import RankAwareSource
from cudf_polars.utils.dtypes import make_empty_column

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.dsl.utils.io import CachedParquetInfo
    from cudf_polars.streaming.actor_graph.core import SubNetGenerator
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer
    from cudf_polars.streaming.base import (
        IOPartitionPlan,
        PartitionInfo,
    )
    from cudf_polars.streaming.io import FusedScan, SplitScan
    from cudf_polars.streaming.partitioning_requests import (
        NamedOrderKey,
        PartitioningRequest,
    )
    from cudf_polars.utils.config import MaxConcurrentIOTasks


def resolve_max_concurrent_io_tasks(
    max_concurrent_io_tasks: MaxConcurrentIOTasks,
    paths: Iterable[str],
) -> int:
    """Resolve the scan-local IO producer count."""
    if any(plc.io.SourceInfo._is_remote_uri(path) for path in paths):
        return max_concurrent_io_tasks.remote
    return max_concurrent_io_tasks.local


class Lineariser:
    """
    Linearizer that ensures ordered delivery from multiple concurrent producers.

    Creates one input channel per producer and streams messages to output
    in sequence-number order, buffering only out-of-order arrivals.
    """

    def __init__(
        self, context: Context, ch_out: Channel[TableChunk], num_producers: int
    ):
        self.context = context
        self.ch_out = ch_out
        self.num_producers = num_producers
        self.input_channels = [context.create_channel() for _ in range(num_producers)]

    async def drain(self) -> None:
        """
        Drain producer channels and forward messages in sequence-number order.

        Streams messages to output as soon as they arrive in order, buffering
        only out-of-order messages to minimize memory pressure.
        """
        next_seq = 0
        buffer = {}

        pending_tasks = {
            asyncio.create_task(ch.recv(self.context)): ch for ch in self.input_channels
        }

        while pending_tasks:
            done, _ = await asyncio.wait(
                pending_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                ch = pending_tasks.pop(task)
                msg = await task

                if msg is not None:
                    buffer[msg.sequence_number] = msg
                    new_task = asyncio.create_task(ch.recv(self.context))
                    pending_tasks[new_task] = ch

            # Forward consecutive messages
            while next_seq in buffer:
                await self.ch_out.send(self.context, buffer.pop(next_seq))
                next_seq += 1

        # Forward any remaining buffered messages
        for seq in sorted(buffer.keys()):
            await self.ch_out.send(self.context, buffer.pop(seq))

        await self.ch_out.drain(self.context)


@define_actor()
async def dataframescan_node(
    context: Context,
    comm: Communicator,
    ir: DataFrameScan,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    *,
    num_producers: int,
    rows_per_partition: int,
    estimated_chunk_bytes: int,
    distributed_scan: bool,
) -> None:
    """
    DataFrameScan node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The DataFrameScan node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output Channel[TableChunk].
    num_producers
        The number of producers to use for the DataFrameScan node.
    rows_per_partition
        The number of rows per partition.
    estimated_chunk_bytes
        Estimated size of each chunk in bytes. Used for memory reservation
        with block spilling to avoid thrashing.
    distributed_scan
        If ``True``, the DataFrame is treated as a shared object and divided
        across workers so each rank reads a disjoint subset. This is normally
        used in ``Cluster.RAY`` and ``Cluster.DASK`` modes.

        If ``False``, the DataFrame is treated as rank-local and each rank
        scans its local DataFrame in full. This is normally used in
        ``Cluster.SPMD`` mode.
    """
    async with shutdown_on_error(
        context, ch_out, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        # Find local partition count.
        nrows = ir.df.shape()[0]
        global_count = math.ceil(nrows / rows_per_partition) if nrows > 0 else 0

        # For single rank or when scanning the full local DataFrame, each rank
        # uses all partitions with no offset.
        if not distributed_scan or comm.nranks == 1:
            local_count = global_count
            local_offset = 0
        else:
            local_count = math.ceil(global_count / comm.nranks)
            local_offset = local_count * comm.rank

        # Send basic metadata
        await send_metadata(
            ch_out,
            context,
            ChannelMetadata(local_count=local_count),
        )

        # Build list of IR slices to read
        ir_slices = []
        # Partial workaround for
        # https://github.com/pola-rs/polars/issues/23214 If a struct column
        # has nulls and is sliced then polars exports invalid validity
        # buffers. We can't detect this exact state because we can't know
        # when the column is sliced.
        copy_slice = any(
            isinstance(dt, pl.Struct)
            for dt in pl.datatypes.unpack_dtypes(ir.df.dtypes(), include_compound=True)
        )

        for seq_num in range(local_count):
            offset = local_offset * rows_per_partition + seq_num * rows_per_partition
            if offset >= nrows:
                break
            sliced = ir.df.slice(offset, rows_per_partition)
            if copy_slice:
                # OK, we have structs that might have nulls, and we're
                # slicing. So let's copy to contiguous storage. This is
                # hacky and doesn't handle the case where we didn't slice
                # but the user sliced the input.
                f = io.BytesIO()
                sliced.serialize_binary(f)
                f.seek(0)
                sliced = pl._plr.PyDataFrame.deserialize_binary(f)
            ir_slices.append(
                DataFrameScan(
                    ir.schema,
                    sliced,
                    ir.projection,
                )
            )

        # If there are no slices, drain the channel and return
        if len(ir_slices) == 0:
            await ch_out.drain(context)
            return

        # If there is only one ir_slices or one producer, we can
        # skip the lineariser and read the chunks directly
        if len(ir_slices) == 1 or num_producers == 1:
            for seq_num, ir_slice in enumerate(ir_slices):
                await read_chunk(
                    context,
                    ir_slice,
                    seq_num,
                    ch_out,
                    ir_context,
                    estimated_chunk_bytes,
                    tracer=tracer,
                )
            await ch_out.drain(context)
            return

        # Use Lineariser to ensure ordered delivery
        num_producers = min(num_producers, len(ir_slices))
        lineariser = Lineariser(context, ch_out, num_producers)

        # Assign tasks to producers using round-robin
        producer_tasks: list[list[tuple[int, DataFrameScan]]] = [
            [] for _ in range(num_producers)
        ]
        for task_idx, ir_slice in enumerate(ir_slices):
            producer_id = task_idx % num_producers
            producer_tasks[producer_id].append((task_idx, ir_slice))

        async def _producer(producer_id: int, ch_out: Channel) -> None:
            for task_idx, ir_slice in producer_tasks[producer_id]:
                await read_chunk(
                    context,
                    ir_slice,
                    task_idx,
                    ch_out,
                    ir_context,
                    estimated_chunk_bytes,
                    tracer=tracer,
                )
            await ch_out.drain(context)

        async with (
            shutdown_on_error(context, *lineariser.input_channels, trace_ir=ir),
        ):
            await gather_in_task_group(
                lineariser.drain(),
                *(
                    _producer(i, ch_in)
                    for i, ch_in in enumerate(lineariser.input_channels)
                ),
            )


@generate_ir_sub_network.register(DataFrameScan)
def _(
    ir: DataFrameScan, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    config_options = rec.state["config_options"]
    rows_per_partition = config_options.executor.max_rows_per_partition
    num_producers = resolve_max_concurrent_io_tasks(
        rec.state["max_concurrent_io_tasks"], ()
    )
    # Use target_partition_size as the estimated chunk size
    estimated_chunk_bytes = config_options.executor.target_partition_size

    context = rec.state["context"]
    ir_context = rec.state["ir_context"]
    channels: dict[IR, ChannelManager] = {ir: ChannelManager(rec.state["context"])}
    nodes: dict[IR, list[Any]] = {
        ir: [
            dataframescan_node(
                context,
                rec.state["comm"],
                ir,
                ir_context,
                channels[ir].reserve_input_slot(),
                num_producers=num_producers,
                rows_per_partition=rows_per_partition,
                estimated_chunk_bytes=estimated_chunk_bytes,
                distributed_scan=config_options.executor.cluster != "spmd",
            )
        ]
    }
    return nodes, channels


def _find_rank_aware_source(scan_fn: Callable[..., Any]) -> RankAwareSource | None:
    """
    Return the :class:`RankAwareSource` captured by a registered IO source function.

    Parameters
    ----------
    scan_fn
        Python scan function exported by Polars for a ``PythonScan`` node. For
        sources created with :func:`polars.io.plugins.register_io_source`, this
        is the wrapper function that captures the original user-provided source.

    Returns
    -------
    The captured `RankAwareSource`, or ``None`` if the IO source function does not
    capture one directly (a plain or wrapped source, treated as rank-unaware).

    Notes
    -----
    This reaches into Polars' ``register_io_source`` closure layout (the captured
    source object). It is the only available hook today. When Polars exposes a
    supported way to thread state into a source this should move to it. See
    https://github.com/NVIDIA/cudf/issues/22917.
    """
    for cell in getattr(scan_fn, "__closure__", ()):
        source = cell.cell_contents
        if isinstance(source, RankAwareSource):
            return source
    return None


async def _process_and_send_chunk(
    context: Context,
    ch_out: Channel[TableChunk],
    ir: PythonScan,
    ir_context: IRExecutionContext,
    tracer: ActorTracer | None,
    chunk: pl.DataFrame | DataFrame,
    seq_num: int,
) -> None:
    """Move a raw chunk to the device, validate and filter it, then send it."""
    process = functools.partial(
        ir.process_chunk, chunk, ir.schema, ir.predicate, context=ir_context
    )

    # Reserve memory for allocations introduced by this step:
    #
    #   host input, no predicate  -> 1x input size (host->device)
    #   host input, predicate     -> 2x input size (host->device + filter output)
    #   GPU input,  no predicate  -> 0
    #   GPU input,  predicate     -> 1x input size (filter output)
    #
    # The net memory increase is the retained host->device copy for host inputs,
    # and 0 for GPU-resident inputs.
    if isinstance(chunk, DataFrame):
        input_bytes = sum(col.device_buffer_size() for col in chunk.table.columns())
        net_memory_delta = 0
        reservation = input_bytes * (ir.predicate is not None)
    else:  # pl.DataFrame
        input_bytes = int(chunk.estimated_size())
        net_memory_delta = input_bytes
        reservation = input_bytes * (1 + (ir.predicate is not None))
    with opaque_memory_usage(
        await reserve_memory(
            context, size=reservation, net_memory_delta=net_memory_delta
        )
    ):
        df = await ir_context.to_thread(process)
    chunk_out = TableChunk.from_pylibcudf_table(
        df.table, df.stream, exclusive_view=True, br=context.br()
    )
    await send_chunk(context, ch_out, chunk_out, seq_num, tracer=tracer)


@define_actor()
async def python_scan_node(
    context: Context,
    comm: Communicator,
    ir: PythonScan,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
) -> None:
    """
    PythonScan node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The PythonScan node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output Channel[TableChunk].
    """
    async with shutdown_on_error(
        context, ch_out, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        rank_aware_source = _find_rank_aware_source(ir.options[0])
        if rank_aware_source is None and comm.nranks > 1 and comm.rank != 0:
            # A plain (rank-unaware) source runs on rank 0 only; other ranks
            # contribute nothing to avoid duplicating the data.
            await send_metadata(ch_out, context, ChannelMetadata(local_count=0))
            await ch_out.drain(context)
            return

        count, raw_chunks = await ir_context.to_thread(
            lambda: ir.run_source_function(
                ir.options,
                ir.schema,
                rank_aware_source=rank_aware_source,
                rank=comm.rank,
                nranks=comm.nranks,
                context=ir_context,
            )
        )
        # A rank-aware source may emit a duplicated output (an identical copy on
        # every rank, e.g. a persisted global sort/limit). Re-advertise that as
        # the channel's ``duplicated`` flag so downstream collectives treat the
        # copies as duplicates rather than distinct partitions.
        duplicated = (
            rank_aware_source is not None
            and rank_aware_source.output_duplicated(comm.rank, comm.nranks)
        )
        if count is not None:
            # The chunk count is available so we can stream one chunk at a time.
            announced = max(count, 1)
            await send_metadata(
                ch_out,
                context,
                ChannelMetadata(local_count=announced, duplicated=duplicated),
            )
            sentinel = object()
            seq_num = 0
            while True:
                chunk = await ir_context.to_thread(next, raw_chunks, sentinel)
                if chunk is sentinel:
                    break
                await _process_and_send_chunk(
                    context,
                    ch_out,
                    ir,
                    ir_context,
                    tracer,
                    cast("pl.DataFrame | DataFrame", chunk),
                    seq_num,
                )
                seq_num += 1
            if seq_num != announced:
                raise RuntimeError(
                    f"PythonScan source reported {announced} chunk(s) but "
                    f"produced {seq_num}"
                )
        else:
            # A plain generator hides its count, so we must drain it to learn the
            # count before announcing it.
            chunks = await ir_context.to_thread(lambda: list(raw_chunks))
            await send_metadata(
                ch_out,
                context,
                ChannelMetadata(local_count=len(chunks), duplicated=duplicated),
            )
            for seq_num, chunk in enumerate(chunks):
                await _process_and_send_chunk(
                    context, ch_out, ir, ir_context, tracer, chunk, seq_num
                )
        await ch_out.drain(context)


@generate_ir_sub_network.register(PythonScan)
def _(
    ir: PythonScan, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    context = rec.state["context"]
    ir_context = rec.state["ir_context"]
    channels: dict[IR, ChannelManager] = {ir: ChannelManager(context)}
    nodes: dict[IR, list[Any]] = {
        ir: [
            python_scan_node(
                context,
                rec.state["comm"],
                ir,
                ir_context,
                channels[ir].reserve_input_slot(),
            )
        ]
    }
    return nodes, channels


async def read_chunk(
    context: Context,
    scan: IR,
    seq_num: int,
    ch_out: Channel[TableChunk],
    ir_context: IRExecutionContext,
    estimated_chunk_bytes: int,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Read a chunk from disk and send it to the output channel.

    Parameters
    ----------
    context
        The rapidsmpf context.
    scan
        The Scan or DataFrameScan node.
    seq_num
        The sequence number.
    ch_out
        The output channel.
    ir_context
        The execution context for the IR node.
    estimated_chunk_bytes
        Estimated retained output size in bytes. Used to estimate peak memory
        for admission before launching the read.
    tracer
        The actor tracer for collecting runtime statistics.
    """
    reservation_bytes = (
        estimated_chunk_bytes
        if isinstance(scan, DataFrameScan)
        else 2 * estimated_chunk_bytes
    )
    start = time.monotonic_ns()
    reservation = await reserve_memory(
        context,
        size=reservation_bytes,
        net_memory_delta=estimated_chunk_bytes,
    )
    admitted = time.monotonic_ns()
    with opaque_memory_usage(reservation):
        df = await ir_context.to_thread(
            scan.do_evaluate,
            *scan._non_child_args,
            context=ir_context,
        )
        chunk = TableChunk.from_pylibcudf_table(
            df.table,
            df.stream,
            exclusive_view=True,
            br=context.br(),
        )
    stop = time.monotonic_ns()
    log(
        "IO Task",
        scope=Scope.IO_TASK.value,
        start=start,
        admitted=admitted,
        stop=stop,
        ir_id=scan.get_stable_id(),
        ir_type=type(scan).__name__,
        sequence_number=seq_num,
        estimated_output_bytes=estimated_chunk_bytes,
        reservation_bytes=reservation_bytes,
    )
    await send_chunk(context, ch_out, chunk, seq_num, tracer=tracer)


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


async def _parquet_ordering_partitioning(
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


def _parquet_ordering_decision(partitioning: Partitioning | None) -> str | None:
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


def _parquet_ordering_trace_info(
    partitioning: Partitioning | None,
) -> dict[str, Any] | None:
    if partitioning is None or not isinstance(partitioning.inter_rank, OrderScheme):
        return None
    return {
        "ordering_count": len(partitioning.inter_rank.orderings),
        "partition_counts": [
            ordering.num_boundaries + 1
            for ordering in partitioning.inter_rank.orderings
        ],
        "strict_boundaries": [
            ordering.strict_boundaries for ordering in partitioning.inter_rank.orderings
        ],
        "locally_ordered": [
            ordering.locally_ordered for ordering in partitioning.inter_rank.orderings
        ],
    }


@define_actor()
async def scan_node(
    context: Context,
    comm: Communicator,
    ir: StreamingScan,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    *,
    partition_info: PartitionInfo,
    partitioning_requests: tuple[PartitioningRequest, ...],
    collective_id: int,
    num_producers: int,
    estimated_chunk_bytes: int,
) -> None:
    """
    Scan node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The Scan node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output Channel[TableChunk].
    partition_info
        Partition information for this scan node.
    partitioning_requests
        Downstream partitioning requests for this scan node.
    collective_id
        Collective ID used to allgather local parquet statistics.
    num_producers
        The number of producers to use for the scan node.
    estimated_chunk_bytes
        Estimated retained output size of each chunk in bytes. Used to estimate
        peak memory for admission before launching each read.
    """
    scans: Sequence[SplitScan] | Sequence[FusedScan] = ir.scans

    async with shutdown_on_error(
        context, ch_out, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        partitioning = await _parquet_ordering_partitioning(
            context,
            comm,
            ir,
            partition_info,
            partitioning_requests,
            ir_context,
            collective_id,
        )
        if tracer is not None:
            tracer.decision = _parquet_ordering_decision(partitioning)
            if (extra := _parquet_ordering_trace_info(partitioning)) is not None:
                tracer.set_extra("parquet_ordering", extra)
        await send_metadata(
            ch_out,
            context,
            ChannelMetadata(local_count=len(scans), partitioning=partitioning),
        )

        # If there is nothing to scan, drain the channel and return
        if len(scans) == 0:
            await ch_out.drain(context)
            return

        # If there is only one scan or one producer, we can
        # skip the lineariser and read the chunks directly
        if len(scans) == 1 or num_producers == 1:
            for seq_num, scan in enumerate(scans):
                await read_chunk(
                    context,
                    scan,
                    seq_num,
                    ch_out,
                    ir_context,
                    estimated_chunk_bytes,
                    tracer=tracer,
                )
            await ch_out.drain(context)
            return

        # Use Lineariser to ensure ordered delivery
        num_producers = min(num_producers, len(scans))
        lineariser = Lineariser(context, ch_out, num_producers)

        # Assign tasks to producers using round-robin
        producer_tasks: list[list[tuple[int, SplitScan | FusedScan]]] = [
            [] for _ in range(num_producers)
        ]
        for task_idx, scan in enumerate(scans):
            producer_id = task_idx % num_producers
            # mypy resolves __iter__ on union-of-sequences to the common base (IR)
            producer_tasks[producer_id].append((task_idx, scan))  # type: ignore[arg-type]

        async def _producer(producer_id: int, ch_out: Channel) -> None:
            for task_idx, scan in producer_tasks[producer_id]:
                await read_chunk(
                    context,
                    scan,
                    task_idx,
                    ch_out,
                    ir_context,
                    estimated_chunk_bytes,
                    tracer=tracer,
                )
            await ch_out.drain(context)

        async with (
            shutdown_on_error(context, *lineariser.input_channels, trace_ir=ir),
        ):
            await gather_in_task_group(
                lineariser.drain(),
                *(
                    _producer(i, ch_in)
                    for i, ch_in in enumerate(lineariser.input_channels)
                ),
            )


@generate_ir_sub_network.register(StreamingScan)
def _(
    ir: StreamingScan, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    config_options = rec.state["config_options"]
    executor = config_options.executor
    partition_info = rec.state["partition_info"][ir]
    num_producers = resolve_max_concurrent_io_tasks(
        rec.state["max_concurrent_io_tasks"],
        ir.base_scan.paths,
    )
    channels: dict[IR, ChannelManager] = {ir: ChannelManager(rec.state["context"])}

    assert partition_info.io_plan is not None, "Scan node must have a partition plan"
    plan: IOPartitionPlan = partition_info.io_plan

    ch_out = channels[ir].reserve_input_slot()
    nodes: dict[IR, list[Any]] = {}

    nodes[ir] = [
        scan_node(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            ch_out,
            partition_info=partition_info,
            partitioning_requests=rec.state["partitioning_requests"].get(ir, ()),
            collective_id=rec.state["collective_id_map"][ir][0],
            num_producers=num_producers,
            estimated_chunk_bytes=(
                plan.estimated_chunk_bytes or executor.target_partition_size
            ),
        )
    ]
    return nodes, channels


@define_actor()
async def sink_node(
    context: Context,
    comm: Communicator,
    ir: StreamingSink,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    partition_info: PartitionInfo,
    collective_id: int,
) -> None:
    """
    Sink node for rapidsmpf - writes data chunks to a file.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The StreamingSink node.
    ir_context
        The execution context for the IR node.
    ch_in
        The input ChannelPair.
    ch_out
        The output ChannelPair for returning an empty result DataFrame.
    partition_info
        The partition information.
    collective_id
        The collective ID for this operation, used for AllGather
        reduction of the chunk count.
    """
    child_ir = ir.children[0]

    suffix = ir.sink.kind.lower()
    # safety-net, if count is too low, we might get conflicts
    # with other files.

    async with shutdown_on_error(
        context, ch_in, ch_out, ir_context=ir_context, trace_ir=ir
    ):
        metadata = await recv_metadata(ch_in, context)
        await send_metadata(
            ch_out, context, ChannelMetadata(local_count=1, duplicated=True)
        )
        skip_write = metadata.duplicated and comm.rank != 0

        if skip_write:
            while await ch_in.recv(context) is not None:
                pass
        else:
            path_root = f"{ir.sink.path}/part"
            if comm.nranks > 1:
                rank_width = math.ceil(math.log10(comm.nranks))
                rank_str = str(comm.rank).zfill(rank_width)
                path_root = f"{path_root}.{rank_str}"
            # local_count may be 0 when a rank receives no partitions
            # (e.g. more ranks than input files); log10(0) is undefined.
            count_width = math.ceil(math.log10(max(metadata.local_count, 1)))
            count_width = max(count_width, 6)

            if ir.sink_to_directory:
                _prepare_sink_directory(ir.sink.path)
                i = 0
                while (msg := await ch_in.recv(context)) is not None:
                    chunk = TableChunk.from_message(msg, br=context.br())
                    # Terminal: the chunk is dropped after the write, so its
                    # whole footprint leaves the system.
                    chunk, _ = await make_table_chunks_available_or_wait(
                        context,
                        chunk,
                        reserve_extra=0,
                        net_memory_delta=-chunk.data_alloc_size(),
                    )
                    df = chunk_to_frame(chunk, child_ir)
                    part_path = f"{path_root}.{str(i).zfill(count_width)}.{suffix}"
                    await ir_context.to_thread(
                        Sink.do_evaluate,
                        ir.sink.schema,
                        ir.sink.kind,
                        part_path,
                        ir.sink.parquet_options,
                        ir.sink.options,
                        df,
                        context=ir_context,
                    )
                    i += 1
            else:
                # Write chunks to a single file
                writer_state = None
                while (msg := await ch_in.recv(context)) is not None:
                    chunk = TableChunk.from_message(msg, br=context.br())
                    # Terminal: the chunk is dropped after the write, so its
                    # whole footprint leaves the system.
                    chunk, _ = await make_table_chunks_available_or_wait(
                        context,
                        chunk,
                        reserve_extra=0,
                        net_memory_delta=-chunk.data_alloc_size(),
                    )
                    # Multiple chunks - use chunked writer
                    df = chunk_to_frame(chunk, child_ir)
                    writer_state = await ir_context.to_thread(
                        _sink_to_file,  # type: ignore[arg-type]  # (to_thread accepts this keyword-only sink helper)
                        ir.sink.kind,
                        ir.sink.path,
                        ir.sink.options,
                        writer_state=writer_state,
                        df=df,
                    )

                # Finalize the writer after all chunks are processed
                if writer_state and ir.sink.kind == "Parquet":
                    # We know that with ir.sink.kind == "Parquet", writer_state being truthy
                    # means that it's a ChunkedParquetWriter.
                    await ir_context.to_thread(writer_state.close, [])  # type: ignore[attr-defined]

        # Signal completion on the metadata and data channels with empty results
        stream = ir_context.get_cuda_stream()
        empty_chunk = empty_table_chunk(ir, context, stream)
        await ch_out.send(context, Message(0, empty_chunk))
        await ch_out.drain(context)


@generate_ir_sub_network.register(StreamingSink)
def _(
    ir: StreamingSink, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate network for StreamingSink node."""
    nodes, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    nodes[ir] = [
        sink_node(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir.children[0]].reserve_output_slot(),
            channels[ir].reserve_input_slot(),
            rec.state["partition_info"][ir],
            rec.state["collective_id_map"][ir][0],
        )
    ]

    return nodes, channels
