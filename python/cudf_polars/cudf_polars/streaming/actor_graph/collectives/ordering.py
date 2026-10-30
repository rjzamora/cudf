# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordering metadata and boundary extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    OrderScheme,
    Ordering,
    Partitioning,
    RowOrdering,
    RowOrderingScope,
)
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataType
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.partitioning import add_row_ordering
from cudf_polars.streaming.actor_graph.utils import (
    ChunkStore,
    recv_metadata,
    send_metadata,
)
from cudf_polars.utils.cuda_stream import stream_ordered_after
from cudf_polars.utils.dtypes import make_empty_column

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_streaming.channel_metadata import OrderKey
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext


_PID_DTYPE = DataType(pl.Int32())


@dataclass(frozen=True)
class OrderSchemePartitioningResult:
    """OrderScheme partitioning and buffered chunks consumed to derive it."""

    partitioning: Partitioning | None
    """The extracted partitioning, or ``None`` if extraction was not possible."""
    chunks: ChunkStore
    """The consumed chunks, stored in replay order."""


def _extract_boundaries_from_endpoint_rows(
    endpoint_key_rows: plc.Table,
    num_partitions: int,
    stream: Stream,
) -> tuple[plc.Table, bool]:
    """
    Extract boundaries from alternating partition endpoint rows.

    Parameters
    ----------
    endpoint_key_rows
        The table containing key-column rows ordered as
        ``[start0, end0, start1, end1, ...]``.
    num_partitions
        The number of partitions.
    stream
        The CUDA stream to use for the operation.

    Returns
    -------
    The boundaries table and whether they are strict.
    """
    # Boundaries are the first endpoint of all partitions except the first.
    # Strictness compares each partition end with the following partition start.
    previous_partition_ends = plc.concatenate.concatenate(
        plc.copying.slice(
            endpoint_key_rows, list(range(1, 2 * num_partitions - 1)), stream=stream
        ),
        stream=stream,
    )
    next_partition_starts = plc.concatenate.concatenate(
        plc.copying.slice(
            endpoint_key_rows, list(range(2, 2 * num_partitions)), stream=stream
        ),
        stream=stream,
    )

    # Single-kernel row-equality check across all key columns using AST
    num_cols = previous_partition_ends.num_columns()
    combined = plc.Table(
        list(previous_partition_ends.columns()) + list(next_partition_starts.columns())
    )
    eq_exprs = [
        plc.expressions.Operation(
            plc.expressions.ASTOperator.NULL_EQUAL,
            plc.expressions.ColumnReference(j),
            plc.expressions.ColumnReference(num_cols + j),
        )
        for j in range(num_cols)
    ]
    row_eq_expr = eq_exprs[0]
    for expr in eq_exprs[1:]:
        row_eq_expr = plc.expressions.Operation(
            plc.expressions.ASTOperator.NULL_LOGICAL_AND,
            row_eq_expr,
            expr,
        )
    row_eq_col = plc.transform.compute_column(combined, row_eq_expr, stream=stream)
    strict = not plc.reduce.reduce(
        row_eq_col,
        plc.aggregation.any(),
        plc.DataType(plc.TypeId.BOOL8),
        stream=stream,
    ).to_py(stream=stream)
    return next_partition_starts, strict


def _metadata_has_ordering(metadata: ChannelMetadata, keys: Sequence[OrderKey]) -> bool:
    """Return whether metadata already has compatible ordering."""
    if metadata.partitioning is None:
        return False
    scheme = metadata.partitioning.inter_rank
    keys = tuple(keys)
    # Exact metadata is always useful. Prefix metadata is only enough for a
    # longer ordering when its boundaries are strict.
    return isinstance(scheme, OrderScheme) and any(
        (
            len(ordering.keys) == len(keys)
            or (ordering.strict_boundaries and len(ordering.keys) < len(keys))
        )
        and tuple(ordering.keys) == keys[: len(ordering.keys)]
        for ordering in scheme.orderings
    )


def _with_extracted_ordering(
    metadata: ChannelMetadata, partitioning: Partitioning | None
) -> ChannelMetadata:
    """Return metadata with extracted ordering when extraction succeeded."""
    if partitioning is None:
        return metadata

    existing = metadata.partitioning
    assert existing is not None
    orderings: list[Ordering] = []
    if isinstance(existing.inter_rank, OrderScheme):
        orderings.extend(existing.inter_rank.orderings)

    scheme = partitioning.inter_rank
    if isinstance(scheme, OrderScheme):
        orderings.extend(scheme.orderings)

    return ChannelMetadata(
        local_count=metadata.local_count,
        partitioning=Partitioning(
            OrderScheme(orderings),
            partitioning.local,
            row_orderings=existing.row_orderings,
        ),
        duplicated=metadata.duplicated,
    )


def _trivial_ordering_metadata(
    context: Context,
    comm: Communicator,
    schema_ir: IR,
    metadata: ChannelMetadata,
    keys: Sequence[OrderKey],
) -> ChannelMetadata | None:
    """Attach ordering without scanning data when the boundaries are trivial."""
    if comm.nranks != 1 or metadata.local_count > 1:
        return None

    partitioning = metadata.partitioning
    assert partitioning is not None
    existing_orderings: list[Ordering] = []
    if isinstance(partitioning.inter_rank, OrderScheme):
        existing_orderings = list(partitioning.inter_rank.orderings)

    dtypes = tuple(schema_ir.schema.values())
    stream = context.br().stream_pool.get_stream()
    boundaries = TableChunk.from_pylibcudf_table(
        plc.Table(
            [make_empty_column(dtypes[key.column_index], stream) for key in keys]
        ),
        stream,
        exclusive_view=False,
        br=context.br(),
    )
    ordering = Ordering(keys, boundaries, strict_boundaries=True)
    return ChannelMetadata(
        local_count=metadata.local_count,
        partitioning=Partitioning(
            OrderScheme([*existing_orderings, ordering]),
            "inherit",
            row_orderings=partitioning.row_orderings,
        ),
        duplicated=metadata.duplicated,
    )


async def extract_orderscheme_partitioning(
    context: Context,
    comm: Communicator,
    schema_ir: IR,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    order_keys: Sequence[OrderKey],
    collective_id: int,
) -> OrderSchemePartitioningResult:
    """
    Extract the partitioning metadata for a sorted channel.

    Parameters
    ----------
    context
        The RapidsMPF context.
    comm
        The RapidsMPF communicator.
    schema_ir
        The IR reference to use for the boundary table schema.
    ir_context
        The IR execution context.
    ch_in
        The channel to collect boundaries from.
    order_keys
        The order keys associated with the boundaries.
    collective_id
        The collective ID for the allgather.

    Returns
    -------
    A result containing a ``Partitioning`` whose ``inter_rank`` is an
    ``OrderScheme`` built from the observed boundaries and ``local`` is
    ``"inherit"``, or ``None`` if the channel contains insufficient data
    or the data is not globally sorted. The result also contains the
    consumed chunks in replay order.

    Notes
    -----
    This utility does not collect channel metadata, nor does
    it push any messages into an output channel. All data
    messages from the input channel are consumed.

    Boundaries are not collected for empty chunks.
    """
    # Collect the first and last row from each non-empty sorted chunk.
    key_indices = [key.column_index for key in order_keys]
    endpoint_rows: list[plc.Table] = []
    chunks = ChunkStore(context)
    stream = ir_context.get_cuda_stream()
    row_indices = plc.Column.from_iterable_of_py(
        [0, -1],
        plc.DataType(plc.TypeId.INT32),
        stream=stream,
    )
    while (msg := await ch_in.recv(context)) is not None:
        chunk, _ = await make_table_chunks_available_or_wait(
            context,
            TableChunk.from_message(msg, br=context.br()),
            reserve_extra=0,
            net_memory_delta=0,
        )
        tbl = chunk.table_view()
        key_tbl = plc.Table([tbl.columns()[index] for index in key_indices])
        if key_tbl.num_rows() == 0:
            chunks.insert(Message(msg.sequence_number, chunk))
            continue
        with stream_ordered_after(lambda: stream, upstreams=(chunk.stream,)):
            endpoint_rows.append(
                plc.copying.gather(
                    key_tbl,
                    row_indices,
                    plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                    stream=stream,
                )
            )
        chunks.insert(Message(msg.sequence_number, chunk))
    endpoint_table: plc.Table | None = (
        plc.concatenate.concatenate(endpoint_rows, stream=stream)
        if endpoint_rows
        else None
    )
    del endpoint_rows

    # Allgather endpoint rows across all ranks.
    if comm.nranks > 1:
        dtypes = tuple(schema_ir.schema.values())
        local_chunk = (
            TableChunk.from_pylibcudf_table(
                endpoint_table, stream, exclusive_view=True, br=context.br()
            )
            if endpoint_table is not None
            else TableChunk.from_pylibcudf_table(
                plc.Table(
                    [make_empty_column(dtypes[index], stream) for index in key_indices]
                ),
                stream,
                exclusive_view=False,
                br=context.br(),
            )
        )
        allgather = AllGatherManager(context, comm, collective_id)
        with allgather.inserting() as inserter:
            await inserter.insert(comm.rank, local_chunk)
        endpoint_table = await allgather.extract_concatenated(
            stream, ordered=True, ir_context=ir_context
        )

    # Return None if there are insufficient endpoints to process.
    if endpoint_table is None or (num_partitions := endpoint_table.num_rows() // 2) < 2:
        return OrderSchemePartitioningResult(None, chunks)

    # Return None if chunk endpoints are not globally sorted.
    column_order = [key.order for key in order_keys]
    null_order = [key.null_order for key in order_keys]
    if not plc.sorting.is_sorted(
        endpoint_table, column_order, null_order, stream=stream
    ):
        return OrderSchemePartitioningResult(None, chunks)

    # Extract boundaries and construct the Partitioning
    boundaries, strict = _extract_boundaries_from_endpoint_rows(
        endpoint_table, num_partitions, stream
    )
    del endpoint_table
    boundaries_chunk = TableChunk.from_pylibcudf_table(
        boundaries, stream, exclusive_view=True, br=context.br()
    )
    return OrderSchemePartitioningResult(
        Partitioning(
            inter_rank=OrderScheme(
                [Ordering(order_keys, boundaries_chunk, strict_boundaries=strict)]
            ),
            local="inherit",
        ),
        chunks,
    )


async def ensure_ordering_metadata(
    context: Context,
    comm: Communicator,
    schema_ir: IR,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    order_keys: Sequence[OrderKey],
    collective_id: int,
) -> None:
    """Forward a channel, extracting ordering boundaries when metadata lacks them."""
    metadata = await recv_metadata(ch_in, context)
    if order_keys:
        partitioning = metadata.partitioning or Partitioning()
        metadata = ChannelMetadata(
            local_count=metadata.local_count,
            partitioning=add_row_ordering(
                partitioning,
                RowOrdering(order_keys, RowOrderingScope.GLOBAL),
            ),
            duplicated=metadata.duplicated,
        )
    buffered_chunks: ChunkStore | None = None

    if order_keys and not _metadata_has_ordering(metadata, order_keys):
        if (
            extracted_metadata := _trivial_ordering_metadata(
                context, comm, schema_ir, metadata, order_keys
            )
        ) is not None:
            metadata = extracted_metadata
        else:
            result = await extract_orderscheme_partitioning(
                context,
                comm,
                schema_ir,
                ir_context,
                ch_in,
                order_keys,
                collective_id,
            )
            metadata = _with_extracted_ordering(metadata, result.partitioning)
            buffered_chunks = result.chunks

    await send_metadata(ch_out, context, metadata)
    if buffered_chunks is None:
        while (msg := await ch_in.recv(context)) is not None:
            await ch_out.send(context, msg)
    else:
        for msg in buffered_chunks:
            await ch_out.send(context, msg)
    await ch_out.drain(context)
