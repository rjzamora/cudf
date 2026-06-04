# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""OrderScheme adjustment utilities for the RapidsMPF streaming runtime."""

from __future__ import annotations

import os
from itertools import pairwise
from math import ceil, floor
from statistics import fmean, median
from typing import TYPE_CHECKING, Literal

from rapidsmpf.integrations.cudf.partition import unpack_and_concat
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import OrderScheme
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

import pylibcudf as plc
from pylibcudf.contiguous_split import pack

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.streaming.actor_graph.utils import (
    ChunkStore,
    concat_batch,
    empty_table_chunk,
    gather_in_task_group,
)
from cudf_polars.utils.cuda_stream import stream_ordered_after

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext


_PID_DTYPE = DataType(pl.Int32())
_PID_PLC_DTYPE = plc.DataType(plc.TypeId.INT32)
_BOUNDARY_DTYPES = {
    plc.TypeId.INT8: pl.Int8(),
    plc.TypeId.INT16: pl.Int16(),
    plc.TypeId.INT32: pl.Int32(),
    plc.TypeId.INT64: pl.Int64(),
    plc.TypeId.UINT8: pl.UInt8(),
    plc.TypeId.UINT16: pl.UInt16(),
    plc.TypeId.UINT32: pl.UInt32(),
    plc.TypeId.UINT64: pl.UInt64(),
    plc.TypeId.FLOAT32: pl.Float32(),
    plc.TypeId.FLOAT64: pl.Float64(),
    plc.TypeId.TIMESTAMP_DAYS: pl.Date(),
    plc.TypeId.TIMESTAMP_MILLISECONDS: pl.Datetime("ms"),
    plc.TypeId.TIMESTAMP_MICROSECONDS: pl.Datetime("us"),
    plc.TypeId.TIMESTAMP_NANOSECONDS: pl.Datetime("ns"),
    plc.TypeId.DURATION_MILLISECONDS: pl.Duration("ms"),
    plc.TypeId.DURATION_MICROSECONDS: pl.Duration("us"),
    plc.TypeId.DURATION_NANOSECONDS: pl.Duration("ns"),
}
_UNSIGNED_BOUNDARY_IDS = {
    plc.TypeId.UINT8,
    plc.TypeId.UINT16,
    plc.TypeId.UINT32,
    plc.TypeId.UINT64,
}
_FLOAT_BOUNDARY_IDS = {plc.TypeId.FLOAT32, plc.TypeId.FLOAT64}
_ADJUST_ALGORITHM_ENV = "CUDF_POLARS_ADJUST_ORDERSCHEME_ALGORITHM"


def _contiguous_owner(pid: int, nranks: int, npartitions: int) -> int:
    """Return the rank owning *pid* under contiguous partition assignment."""
    return pid * nranks // npartitions


def _partition_range(rank: int, nranks: int, npartitions: int) -> tuple[int, int]:
    """Return the half-open partition ID range owned by *rank*."""
    return (
        (rank * npartitions + nranks - 1) // nranks,
        ((rank + 1) * npartitions + nranks - 1) // nranks,
    )


def _local_partitions(rank: int, nranks: int, npartitions: int) -> list[int]:
    """Return partition IDs owned by *rank* under contiguous assignment."""
    start, stop = _partition_range(rank, nranks, npartitions)
    return list(range(start, stop))


def _contiguous_owners(
    start: int,
    stop: int,
    nranks: int,
    npartitions: int,
) -> list[int]:
    """Return ranks owning any partition in the half-open range [start, stop)."""
    if start >= stop:
        return []
    first_rank = _contiguous_owner(start, nranks, npartitions)
    last_rank = _contiguous_owner(stop - 1, nranks, npartitions)
    owners = []
    for rank in range(first_rank, last_rank + 1):
        rank_start, rank_stop = _partition_range(rank, nranks, npartitions)
        if max(start, rank_start) < min(stop, rank_stop):
            owners.append(rank)
    return owners


def orderscheme_local_partitions(
    rank: int, nranks: int, scheme: OrderScheme
) -> list[int]:
    """Return local partition IDs for a flat OrderScheme."""
    return _local_partitions(rank, nranks, scheme.num_boundaries + 1)


def orderscheme_local_count(rank: int, nranks: int, scheme: OrderScheme) -> int:
    """Return local partition count for a flat OrderScheme."""
    return len(orderscheme_local_partitions(rank, nranks, scheme))


def make_strict_orderscheme(scheme: OrderScheme, context: Context) -> OrderScheme:
    """Return an equivalent OrderScheme with strict boundary metadata."""
    return OrderScheme(
        scheme.keys,
        scheme.get_boundaries(context.br()),
        strict_boundaries=True,
    )


def _numeric_boundary_series(
    boundaries: pl.Series,
    dtype: pl.DataType,
) -> pl.Series:
    if isinstance(dtype, pl.Date):
        return boundaries.cast(pl.Int32())
    if isinstance(dtype, pl.Datetime | pl.Duration):
        return boundaries.cast(pl.Int64())
    return boundaries


def _boundary_series(
    values: list[float | int],
    dtype: pl.DataType,
    dtype_id: plc.TypeId,
) -> pl.Series:
    if dtype_id in _FLOAT_BOUNDARY_IDS:
        return pl.Series("boundary", values, dtype=dtype)
    int_values = [floor(value) for value in values]
    if isinstance(dtype, pl.Date):
        return pl.Series("boundary", int_values, dtype=pl.Int32()).cast(dtype)
    if isinstance(dtype, pl.Datetime | pl.Duration):
        return pl.Series("boundary", int_values, dtype=pl.Int64()).cast(dtype)
    return pl.Series("boundary", int_values, dtype=dtype)


def _interpolate_boundary_values(
    values: list[float | int],
    dtype_id: plc.TypeId,
    target_npartitions: int,
    method: Literal["median", "mean"],
) -> list[float | int] | None:
    diffs = [stop - start for start, stop in pairwise(values) if stop > start]
    if not diffs:
        return None

    gap = median(diffs) if method == "median" else fmean(diffs)
    start = values[0] - gap
    stop = values[-1] + gap
    if stop <= start:
        return None

    subdivisions = max(2, ceil(target_npartitions / (len(values) + 1)))
    edges = [start, *values, stop]
    raw_boundaries = [
        start + (stop - start) * index / subdivisions
        for partition, (start, stop) in enumerate(pairwise(edges))
        for index in range(1, subdivisions + 1)
        if partition < len(values) or index < subdivisions
    ]
    if dtype_id in _FLOAT_BOUNDARY_IDS:
        boundaries = raw_boundaries
    else:
        boundaries = [floor(value) for value in raw_boundaries]
        if dtype_id in _UNSIGNED_BOUNDARY_IDS and boundaries[0] < 0:
            return None

    unique_boundaries: list[float | int] = []
    for value in boundaries:
        if not unique_boundaries or value > unique_boundaries[-1]:
            unique_boundaries.append(value)
    return unique_boundaries


def interpolate_orderscheme(
    context: Context,
    scheme: OrderScheme,
    target_npartitions: int,
    *,
    method: Literal["median", "mean"] = "median",
) -> OrderScheme | None:
    """Return a refined single-key numeric OrderScheme using interpolated boundaries."""
    if target_npartitions <= scheme.num_boundaries + 1:
        return scheme
    if len(scheme.keys) != 1 or scheme.keys[0].order != plc.types.Order.ASCENDING:
        return None
    if method not in {"median", "mean"}:
        raise ValueError("method must be 'median' or 'mean'.")

    boundary_chunk = scheme.get_boundaries(context.br())
    boundary_table = boundary_chunk.table_view()
    if boundary_table.num_rows() < 2:
        return None
    dtype_id = boundary_table.columns()[0].type().id()
    dtype = _BOUNDARY_DTYPES.get(dtype_id)
    if dtype is None:
        return None

    with stream_ordered_after(
        context.get_stream_from_pool, upstreams=(boundary_chunk.stream,)
    ) as stream:
        boundary_series = (
            DataFrame.from_table(
                boundary_table,
                ["boundary"],
                [DataType(dtype)],
                stream,
            )
            .to_polars()["boundary"]
            .rechunk()
        )
        numeric_values = _numeric_boundary_series(boundary_series, dtype).to_list()
        new_values = _interpolate_boundary_values(
            numeric_values, dtype_id, target_npartitions, method
        )
        if new_values is None or len(new_values) <= scheme.num_boundaries:
            return None
        boundaries = DataFrame.from_polars(
            pl.DataFrame({"boundary": _boundary_series(new_values, dtype, dtype_id)}),
            stream,
        )
        return OrderScheme(
            scheme.keys,
            TableChunk.from_pylibcudf_table(
                boundaries.table,
                stream,
                exclusive_view=True,
                br=context.br(),
            ),
            strict_boundaries=True,
        )


def _validate_schemes(input_scheme: OrderScheme, output_scheme: OrderScheme) -> None:
    """Validate the first-pass flat OrderScheme adjustment contract."""
    if not output_scheme.strict_boundaries:
        raise ValueError("adjust_orderscheme requires a strict output OrderScheme.")
    prefix_len = len(output_scheme.keys)
    if input_scheme.keys[:prefix_len] != output_scheme.keys:
        raise NotImplementedError(
            "adjust_orderscheme currently requires the output OrderScheme keys "
            "to be a prefix of the input OrderScheme keys."
        )


def _split_points(
    table: plc.Table,
    boundary_table: plc.Table,
    scheme: OrderScheme,
    stream: Stream,
) -> list[int]:
    """Return row split points that partition *table* by *scheme* boundaries."""
    if boundary_table.num_rows() == 0:
        return []
    key_table = plc.Table([table.columns()[key.column_index] for key in scheme.keys])
    split_col = plc.search.lower_bound(
        key_table,
        boundary_table,
        [key.order for key in scheme.keys],
        [key.null_order for key in scheme.keys],
        stream=stream,
    )
    return (
        DataFrame.from_table(
            plc.Table([split_col]),
            ["split"],
            [_PID_DTYPE],
            stream,
        )
        .to_polars()["split"]
        .to_list()
    )


def _append_partition_id(table: plc.Table, pid: int, stream: Stream) -> plc.Table:
    """Append a hidden target-partition-id column to *table*."""
    pid_col = plc.Column.from_scalar(
        plc.Scalar.from_py(pid, _PID_PLC_DTYPE, stream=stream),
        table.num_rows(),
        stream=stream,
    )
    return plc.Table([*table.columns(), pid_col])


def _boundary_search_positions(
    input_boundary_table: plc.Table,
    output_boundary_table: plc.Table,
    output_scheme: OrderScheme,
    stream: Stream,
) -> tuple[list[int], list[int]]:
    """Search output boundary positions for projected input boundary rows."""
    if input_boundary_table.num_rows() == 0:
        return [], []
    prefix_len = len(output_scheme.keys)
    input_prefix_boundaries = plc.Table(input_boundary_table.columns()[:prefix_len])
    orders = [key.order for key in output_scheme.keys]
    null_orders = [key.null_order for key in output_scheme.keys]
    lower_col = plc.search.lower_bound(
        output_boundary_table,
        input_prefix_boundaries,
        orders,
        null_orders,
        stream=stream,
    )
    upper_col = plc.search.upper_bound(
        output_boundary_table,
        input_prefix_boundaries,
        orders,
        null_orders,
        stream=stream,
    )
    positions = DataFrame.from_table(
        plc.Table([lower_col, upper_col]),
        ["lower", "upper"],
        [_PID_DTYPE, _PID_DTYPE],
        stream,
    ).to_polars()
    return positions["lower"].to_list(), positions["upper"].to_list()


def _peer_ranks(
    rank: int,
    nranks: int,
    input_scheme: OrderScheme,
    output_scheme: OrderScheme,
    lower_positions: list[int],
    upper_positions: list[int],
) -> tuple[list[int], list[int]]:
    """Return source and destination ranks needed for OrderScheme adjustment."""
    output_npartitions = output_scheme.num_boundaries + 1

    def dsts_for_source(source_rank: int) -> list[int]:
        output_start, output_stop = _source_output_range(
            source_rank,
            nranks,
            input_scheme,
            output_scheme,
            lower_positions,
            upper_positions,
        )
        if output_start == output_stop:
            return []
        return [
            dst
            for dst in _contiguous_owners(
                output_start, output_stop, nranks, output_npartitions
            )
            if dst != source_rank
        ]

    dsts = dsts_for_source(rank)
    srcs = [
        source_rank
        for source_rank in range(nranks)
        if source_rank != rank and rank in dsts_for_source(source_rank)
    ]
    return srcs, dsts


def _source_output_range(
    source_rank: int,
    nranks: int,
    input_scheme: OrderScheme,
    output_scheme: OrderScheme,
    lower_positions: list[int],
    upper_positions: list[int],
) -> tuple[int, int]:
    input_npartitions = input_scheme.num_boundaries + 1
    output_npartitions = output_scheme.num_boundaries + 1
    output_prefix_only = len(output_scheme.keys) < len(input_scheme.keys)
    include_upper_boundary = output_prefix_only or not input_scheme.strict_boundaries
    input_start, input_stop = _partition_range(source_rank, nranks, input_npartitions)
    if input_start == input_stop:
        return 0, 0
    output_start = 0 if input_start == 0 else upper_positions[input_start - 1]
    output_stop = (
        output_npartitions
        if input_stop == input_npartitions
        else (
            upper_positions[input_stop - 1] + 1
            if include_upper_boundary
            else lower_positions[input_stop - 1] + 1
        )
    )
    return output_start, output_stop


def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _adjust_algorithm() -> Literal["windowed", "batch"]:
    algorithm = os.environ.get(_ADJUST_ALGORITHM_ENV, "windowed").lower()
    if algorithm == "windowed":
        return "windowed"
    if algorithm == "batch":
        return "batch"
    raise ValueError(f"{_ADJUST_ALGORITHM_ENV} must be 'windowed' or 'batch'.")


def _unpack_remote_piece(
    packed: PackedData,
    stream: Stream,
    br: BufferResource,
) -> tuple[int, TableChunk] | None:
    """Unpack one remote piece and recover its hidden target partition ID."""
    table = unpack_and_concat([packed], stream=stream, br=br)
    if table.num_rows() == 0:
        return None
    *payload_cols, pid_col = table.columns()
    pid = int(
        DataFrame.from_table(
            plc.Table([pid_col]),
            ["pid"],
            [_PID_DTYPE],
            stream,
        )
        .to_polars()
        .item(0, 0)
    )
    payload = plc.concatenate.concatenate(
        [plc.Table(payload_cols)], stream=stream, mr=br.device_mr
    )
    return pid, TableChunk.from_pylibcudf_table(
        payload,
        stream,
        exclusive_view=True,
        br=br,
    )


def _copy_to_owned_chunk(
    table: plc.Table,
    stream: Stream,
    br: BufferResource,
) -> TableChunk:
    """Copy a table view into a uniquely-owned chunk."""
    table = plc.concatenate.concatenate([table], stream=stream, mr=br.device_mr)
    return TableChunk.from_pylibcudf_table(
        table,
        stream,
        exclusive_view=True,
        br=br,
    )


class _OutputPieceReader:
    def __init__(
        self,
        context: Context,
        ch_in: Channel[TableChunk],
        boundary_chunk: TableChunk,
        output_scheme: OrderScheme,
    ) -> None:
        self.context = context
        self.ch_in = ch_in
        self.boundary_chunk = boundary_chunk
        self.boundary_table = boundary_chunk.table_view()
        self.output_scheme = output_scheme
        self.pending: dict[int, ChunkStore] = {}
        self.input_done = False

    def _store(self, pid: int, chunk: TableChunk) -> None:
        if pid not in self.pending:
            self.pending[pid] = ChunkStore(self.context)
        self.pending[pid].insert(Message(pid, chunk))

    def _has_reached(self, stop: int) -> bool:
        return any(pid >= stop for pid in self.pending)

    async def collect_window(self, start: int, stop: int) -> dict[int, ChunkStore]:
        while not self.input_done and not self._has_reached(stop):
            msg = await self.ch_in.recv(self.context)
            if msg is None:
                self.input_done = True
                break
            chunk = TableChunk.from_message(
                msg, br=self.context.br()
            ).make_available_and_spill(self.context.br(), allow_overbooking=True)
            if chunk.table_view().num_rows() == 0:
                continue
            with stream_ordered_after(
                self.context.get_stream_from_pool,
                upstreams=(chunk.stream, self.boundary_chunk.stream),
            ) as stream:
                table = chunk.table_view()
                splits = _split_points(
                    table, self.boundary_table, self.output_scheme, stream
                )
                for pid, piece in enumerate(
                    plc.copying.split(table, splits, stream=stream)
                ):
                    if piece.num_rows() == 0:
                        continue
                    self._store(
                        pid, _copy_to_owned_chunk(piece, stream, self.context.br())
                    )

        out = {
            pid: self.pending.pop(pid)
            for pid in list(self.pending)
            if start <= pid < stop
        }
        return dict(sorted(out.items()))


async def _adjust_orderscheme_local(
    context: Context,
    ref_ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    output_scheme: OrderScheme,
) -> None:
    npartitions = output_scheme.num_boundaries + 1
    boundary_chunk = output_scheme.get_boundaries(context.br())
    boundary_table = boundary_chunk.table_view()
    pending_pid: int | None = None
    pending_chunks: ChunkStore | None = None
    next_pid = 0

    async def emit_pending(pid: int) -> None:
        nonlocal pending_pid, pending_chunks
        if pending_pid == pid and pending_chunks is not None:
            chunks = [
                TableChunk.from_message(msg, br=context.br()) for msg in pending_chunks
            ]
            chunk = await concat_batch(chunks, context, ref_ir.schema, ir_context)
            pending_pid = None
            pending_chunks = None
        else:
            chunk = empty_table_chunk(ref_ir, context, ir_context.get_cuda_stream())
        await ch_out.send(context, Message(pid, chunk))

    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg, br=context.br()).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        if chunk.table_view().num_rows() == 0:
            continue
        with stream_ordered_after(
            context.get_stream_from_pool,
            upstreams=(chunk.stream, boundary_chunk.stream),
        ) as stream:
            table = chunk.table_view()
            splits = _split_points(table, boundary_table, output_scheme, stream)
            for pid, piece in enumerate(
                plc.copying.split(table, splits, stream=stream)
            ):
                if piece.num_rows() == 0:
                    continue
                while next_pid < pid:
                    await emit_pending(next_pid)
                    next_pid += 1
                if pending_pid is None:
                    pending_pid = pid
                    pending_chunks = ChunkStore(context)
                elif pending_pid != pid:
                    await emit_pending(pending_pid)
                    next_pid = pending_pid + 1
                    pending_pid = pid
                    pending_chunks = ChunkStore(context)
                assert pending_chunks is not None
                pending_chunks.insert(
                    Message(
                        pid,
                        _copy_to_owned_chunk(piece, stream, context.br()),
                    )
                )

    while next_pid < npartitions:
        await emit_pending(next_pid)
        next_pid += 1
    await ch_out.drain(context)


async def _adjust_orderscheme_rank_windows(
    context: Context,
    comm: Communicator,
    ref_ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    input_scheme: OrderScheme,
    output_scheme: OrderScheme,
    collective_id: int,
    lower_positions: list[int],
    upper_positions: list[int],
) -> None:
    npartitions = output_scheme.num_boundaries + 1
    boundary_chunk = output_scheme.get_boundaries(context.br())
    reader = _OutputPieceReader(context, ch_in, boundary_chunk, output_scheme)
    source_ranges = [
        _source_output_range(
            source_rank,
            comm.nranks,
            input_scheme,
            output_scheme,
            lower_positions,
            upper_positions,
        )
        for source_rank in range(comm.nranks)
    ]

    for output_rank in range(comm.nranks):
        window = _partition_range(output_rank, comm.nranks, npartitions)
        if window[0] == window[1]:
            continue

        contributing_sources = [
            source_rank
            for source_rank, source_range in enumerate(source_ranges)
            if _ranges_overlap(source_range, window)
        ]
        srcs = (
            [source for source in contributing_sources if source != comm.rank]
            if output_rank == comm.rank
            else []
        )
        dsts = (
            [output_rank]
            if output_rank != comm.rank
            and _ranges_overlap(source_ranges[comm.rank], window)
            else []
        )
        exchange = SparseAlltoall(
            context,
            comm,
            collective_id,
            srcs=srcs,
            dsts=dsts,
        )
        local_pieces = (
            await reader.collect_window(*window)
            if _ranges_overlap(source_ranges[comm.rank], window)
            else {}
        )

        if output_rank != comm.rank:
            for pid, store in local_pieces.items():
                for msg in store:
                    chunk = TableChunk.from_message(msg, br=context.br())
                    with stream_ordered_after(
                        context.get_stream_from_pool, upstreams=(chunk.stream,)
                    ) as stream:
                        exchange.insert(
                            output_rank,
                            PackedData.from_cudf_packed_columns(
                                pack(
                                    _append_partition_id(
                                        chunk.table_view(), pid, stream
                                    ),
                                    stream,
                                    mr=context.br().device_mr,
                                ),
                                stream,
                                context.br(),
                            ),
                        )
        await exchange.insert_finished(context)

        if output_rank == comm.rank:
            pieces_by_source: dict[int, dict[int, ChunkStore]] = {}
            if local_pieces:
                pieces_by_source[comm.rank] = local_pieces
            for source_rank in srcs:
                remote_pieces: dict[int, ChunkStore] = {}
                stream = context.get_stream_from_pool()
                for packed in exchange.extract(source_rank):
                    remote_piece = _unpack_remote_piece(packed, stream, context.br())
                    if remote_piece is None:
                        continue
                    pid, chunk = remote_piece
                    if pid not in remote_pieces:
                        remote_pieces[pid] = ChunkStore(context)
                    remote_pieces[pid].insert(Message(pid, chunk))
                pieces_by_source[source_rank] = remote_pieces

            for pid in range(*window):
                chunks: list[TableChunk] = []
                for source_rank in contributing_sources:
                    stores = pieces_by_source.get(source_rank)
                    if stores is None:
                        continue
                    pid_store = stores.get(pid)
                    if pid_store is None:
                        continue
                    chunks.extend(
                        TableChunk.from_message(msg, br=context.br())
                        for msg in pid_store
                    )
                chunk = (
                    await concat_batch(chunks, context, ref_ir.schema, ir_context)
                    if chunks
                    else empty_table_chunk(
                        ref_ir, context, ir_context.get_cuda_stream()
                    )
                )
                await ch_out.send(context, Message(pid, chunk))
        del exchange

    await ch_out.drain(context)


async def adjust_orderscheme(
    context: Context,
    comm: Communicator,
    ref_ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    input_scheme: OrderScheme,
    output_scheme: OrderScheme,
    *,
    collective_id: int | None = None,
) -> None:
    """
    Adjust flat OrderScheme boundaries using contiguous partition ownership.

    Parameters
    ----------
    context
        The streaming context.
    comm
        The communicator.
    ref_ir
        An IR node describing the payload schema.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    input_scheme
        The input OrderScheme.
    output_scheme
        The output OrderScheme.
    collective_id
        The collective ID to use for SparseAlltoall.

    Notes
    -----
    This utility is intentionally narrow and only adjusts data messages. The
    caller is responsible for receiving input metadata and sending output
    metadata. Input rows are assumed to be globally ordered by ``input_scheme``;
    sortedness is not checked here.
    """
    _validate_schemes(input_scheme, output_scheme)
    npartitions = output_scheme.num_boundaries + 1
    local_pids = _local_partitions(comm.rank, comm.nranks, npartitions)

    if comm.nranks > 1 and collective_id is None:
        raise ValueError("collective_id is required when comm.nranks > 1.")

    try:
        if comm.nranks == 1:
            await _adjust_orderscheme_local(
                context,
                ref_ir,
                ir_context,
                ch_out,
                ch_in,
                output_scheme,
            )
            return

        input_boundary_chunk = input_scheme.get_boundaries(context.br())
        boundary_chunk = output_scheme.get_boundaries(context.br())
        boundary_table = boundary_chunk.table_view()
        srcs: list[int] = []
        dsts: list[int] = []
        if comm.nranks > 1:
            with stream_ordered_after(
                context.get_stream_from_pool,
                upstreams=(input_boundary_chunk.stream, boundary_chunk.stream),
            ) as stream:
                lower_positions, upper_positions = _boundary_search_positions(
                    input_boundary_chunk.table_view(),
                    boundary_table,
                    output_scheme,
                    stream,
                )
            srcs, dsts = _peer_ranks(
                comm.rank,
                comm.nranks,
                input_scheme,
                output_scheme,
                lower_positions,
                upper_positions,
            )
            assert collective_id is not None
            if _adjust_algorithm() == "windowed":
                await _adjust_orderscheme_rank_windows(
                    context,
                    comm,
                    ref_ir,
                    ir_context,
                    ch_out,
                    ch_in,
                    input_scheme,
                    output_scheme,
                    collective_id,
                    lower_positions,
                    upper_positions,
                )
                return
            srcs = [rank for rank in range(comm.nranks) if rank != comm.rank]
            dsts = srcs
        exchange = (
            SparseAlltoall(context, comm, collective_id, srcs=srcs, dsts=dsts)
            if comm.nranks > 1
            else None
        )
        local_chunks: dict[int, ChunkStore] = {}

        def store_chunk(
            stores: dict[int, ChunkStore], pid: int, chunk: TableChunk
        ) -> None:
            if pid not in stores:
                stores[pid] = ChunkStore(context)
            stores[pid].insert(Message(pid, chunk))

        try:
            while (msg := await ch_in.recv(context)) is not None:
                chunk = TableChunk.from_message(
                    msg, br=context.br()
                ).make_available_and_spill(context.br(), allow_overbooking=True)
                if chunk.table_view().num_rows() == 0:
                    continue
                with stream_ordered_after(
                    context.get_stream_from_pool,
                    upstreams=(chunk.stream, boundary_chunk.stream),
                ) as stream:
                    table = chunk.table_view()
                    splits = _split_points(table, boundary_table, output_scheme, stream)
                    for pid, piece in enumerate(
                        plc.copying.split(table, splits, stream=stream)
                    ):
                        if piece.num_rows() == 0:
                            continue
                        owner = _contiguous_owner(pid, comm.nranks, npartitions)
                        if owner == comm.rank:
                            store_chunk(
                                local_chunks,
                                pid,
                                _copy_to_owned_chunk(piece, stream, context.br()),
                            )
                        else:
                            assert exchange is not None
                            exchange.insert(
                                owner,
                                PackedData.from_cudf_packed_columns(
                                    pack(
                                        _append_partition_id(piece, pid, stream),
                                        stream,
                                        mr=context.br().device_mr,
                                    ),
                                    stream,
                                    context.br(),
                                ),
                            )
        finally:
            if exchange is not None:
                await exchange.insert_finished(context)

        source_order = (
            *[src for src in srcs if src < comm.rank],
            comm.rank,
            *[src for src in srcs if src > comm.rank],
        )
        chunks_by_source: dict[int, dict[int, ChunkStore]] = {comm.rank: local_chunks}
        for source_rank in source_order:
            if source_rank == comm.rank:
                continue
            else:
                assert exchange is not None
                remote_chunks: dict[int, ChunkStore] = {}
                stream = context.get_stream_from_pool()
                for packed in exchange.extract(source_rank):
                    remote_piece = _unpack_remote_piece(packed, stream, context.br())
                    if remote_piece is None:
                        continue
                    pid, chunk = remote_piece
                    store_chunk(remote_chunks, pid, chunk)
                chunks_by_source[source_rank] = remote_chunks

        for pid in local_pids:
            chunks: list[TableChunk] = []
            for source_rank in source_order:
                if store := chunks_by_source.get(source_rank, {}).get(pid):
                    chunks.extend(
                        TableChunk.from_message(msg, br=context.br()) for msg in store
                    )
            chunk = (
                await concat_batch(chunks, context, ref_ir.schema, ir_context)
                if chunks
                else empty_table_chunk(ref_ir, context, ir_context.get_cuda_stream())
            )
            await ch_out.send(context, Message(pid, chunk))
        await ch_out.drain(context)
    except BaseException:
        await gather_in_task_group(
            ch_in.shutdown(context),
            ch_in.shutdown_metadata(context),
            ch_out.shutdown(context),
            ch_out.shutdown_metadata(context),
        )
        raise
