# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Sort logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from rapidsmpf.shuffler import PartitionAssignment
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    OrderKey,
    OrderScheme,
    Partitioning,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.expr import Col, NamedExpr
from cudf_polars.dsl.ir import Empty, MapFunction, Sort
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.experimental.rapidsmpf.collectives.allgather import AllGatherManager
from cudf_polars.experimental.rapidsmpf.collectives.shuffle import ShuffleManager
from cudf_polars.experimental.rapidsmpf.dispatch import generate_ir_sub_network
from cudf_polars.experimental.rapidsmpf.nodes import (
    default_node_single,
    shutdown_on_error,
)
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    allgather_reduce,
    chunk_to_frame,
    chunkwise_evaluate,
    concat_batch,
    drop_empty_chunks,
    empty_table_chunk,
    evaluate_batch,
    evaluate_chunk,
    gather_in_task_group,
    names_to_indices,
    process_children,
    recv_metadata,
    replay_buffered_channel,
    send_metadata,
)
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.experimental.sort import (
    HintSorted,
    _get_final_sort_boundaries,
    _has_simple_zlice,
    _select_local_split_candidates,
    find_sort_splits,
)
from cudf_polars.utils.cuda_stream import get_joined_cuda_stream

if TYPE_CHECKING:
    from collections.abc import Generator

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.dispatch import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.tracing import ActorTracer
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import StreamingExecutor


class ChunkStore:
    """Ordered spillable buffer for TableChunk messages."""

    def __init__(self, ctx: Context) -> None:
        self._mids: deque[int] = deque()
        self._store = ctx.spillable_messages()

    def insert(self, msg: Message) -> None:
        """Insert a message into the store."""
        self._mids.append(self._store.insert(msg))

    def __iter__(self) -> Generator[Message, None, None]:
        """Yield messages in insertion order, draining the store."""
        while self._mids:
            yield self._store.extract(mid=self._mids.popleft())


async def _simple_top_or_bottom_k(
    context: Context,
    comm: Communicator,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    ir: Sort,
    ir_context: IRExecutionContext,
    metadata_in: ChannelMetadata,
    collective_ids: list[int],
    tracer: ActorTracer | None,
) -> None:
    """Sort + simple head/tail slice."""
    # TODO: We may need to gate this optimization on the slice size.
    await send_metadata(
        ch_out,
        context,
        ChannelMetadata(local_count=1, partitioning=None, duplicated=True),
    )

    chunks: list[TableChunk] = []
    while (msg := await ch_in.recv(context)) is not None:
        chunks.append(
            await evaluate_chunk(
                context,
                TableChunk.from_message(msg, br=context.br()),
                ir,
                ir_context=ir_context,
            )
        )
    chunk: TableChunk = await evaluate_batch(chunks, context, ir, ir_context=ir_context)
    chunks.clear()

    if comm.nranks > 1 and not metadata_in.duplicated:
        allgather = AllGatherManager(context, comm, collective_ids.pop())
        with allgather.inserting() as inserter:
            inserter.insert(comm.rank, chunk)

        stream = ir_context.get_cuda_stream()
        chunk = await evaluate_chunk(
            context,
            TableChunk.from_pylibcudf_table(
                await allgather.extract_concatenated(stream, ordered=True),
                stream,
                exclusive_view=True,
                br=context.br(),
            ),
            ir,
            ir_context=ir_context,
        )

    if tracer is not None:
        tracer.add_chunk(table=chunk.table_view())
    await ch_out.send(context, Message(comm.rank, chunk))

    await ch_out.drain(context)


def _boundary_schema(by: list[str], by_dtypes: list[DataType]) -> Schema:
    """Schema of boundaries table."""
    name_gen = unique_names(by)
    part_id_dtype = DataType(pl.UInt32())
    return dict(
        zip(
            [*by, next(name_gen), next(name_gen)],
            [*by_dtypes, part_id_dtype, part_id_dtype],
            strict=True,
        )
    )


async def _compute_sort_boundaries(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    local_candidates_list: list[TableChunk],
    ir: Sort,
    by: list[str],
    num_partitions: int,
    allgather_id: int | None,
) -> DataFrame:
    """Compute global sort boundaries."""
    column_order = list(ir.order)
    null_order = list(ir.null_order)
    by_dtypes = [ir.schema[b] for b in by]
    boundary_ir = Empty(_boundary_schema(by, by_dtypes))
    local_boundaries_df = _get_final_sort_boundaries(
        chunk_to_frame(
            await concat_batch(
                local_candidates_list,
                context,
                boundary_ir.schema,
                ir_context,
            )
            if local_candidates_list
            else empty_table_chunk(
                boundary_ir,
                context,
                ir_context.get_cuda_stream(),
            ),
            boundary_ir,
        ),
        column_order,
        null_order,
        num_partitions,
    )
    stream = local_boundaries_df.stream

    if allgather_id is not None:
        chunk = TableChunk.from_pylibcudf_table(
            local_boundaries_df.table,
            stream,
            exclusive_view=True,
            br=context.br(),
        )
        allgather = AllGatherManager(context, comm, allgather_id)
        with allgather.inserting() as inserter:
            inserter.insert(comm.rank, chunk)
        concat_table = await allgather.extract_concatenated(stream, ordered=True)
        return _get_final_sort_boundaries(
            DataFrame.from_table(
                concat_table,
                list(boundary_ir.schema.keys()),
                list(boundary_ir.schema.values()),
                stream=stream,
            ),
            column_order,
            null_order,
            num_partitions,
        )
    else:
        return local_boundaries_df


async def _sample_chunks_for_size_estimate(
    context: Context,
    comm: Communicator,
    ch_in: Channel[TableChunk],
    num_partitions: int,
    metadata_in: ChannelMetadata,
    executor: StreamingExecutor,
    collective_ids: list[int],
) -> tuple[dict[int, TableChunk], int]:
    """
    Sample chunks and estimate total data size to derive num_partitions dynamically.

    The sampled chunks are returned keyed by sequence number. The caller is
    responsible for replaying them into a channel via replay_buffered_channel.
    """
    if executor.dynamic_planning is None:
        return {}, num_partitions

    size_estimate_id = collective_ids.pop()
    target_partition_size = executor.target_partition_size
    sample_chunk_count = executor.dynamic_planning.sample_chunk_count

    sampled_chunks: dict[int, TableChunk] = {}
    sampled_bytes = 0
    for _ in range(sample_chunk_count):
        msg = await ch_in.recv(context)
        if msg is None:
            break
        chunk = TableChunk.from_message(msg, br=context.br()).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        sampled_bytes += chunk.data_alloc_size()
        sampled_chunks[msg.sequence_number] = chunk
        if sampled_bytes >= target_partition_size:
            break

    # Extrapolate local size estimate from samples
    local_count = metadata_in.local_count
    local_size = (
        int(sampled_bytes / len(sampled_chunks) * local_count) if sampled_chunks else 0
    )

    # Allgather to get global size estimate across all ranks
    if comm.nranks > 1 and not metadata_in.duplicated:
        (global_size,) = await allgather_reduce(
            context, comm, size_estimate_id, local_size
        )
    else:
        global_size = local_size

    num_partitions = max(1, global_size // target_partition_size)
    return sampled_chunks, num_partitions


async def _receive_and_buffer_chunks(
    context: Context,
    ch_in: Channel[TableChunk],
    chunk_store: ChunkStore,
    ir: Sort,
    by: list[str],
    num_partitions: int,
    comm: Communicator,
    ir_context: IRExecutionContext,
) -> list[TableChunk]:
    """Receive input chunks, collect local split candidates, and buffer chunks for later insert."""
    await recv_metadata(ch_in, context)
    local_candidates_list: list[TableChunk] = []
    local_row_offset = 0

    while (msg := await ch_in.recv(context)) is not None:
        seq_num = msg.sequence_number
        df = chunk_to_frame(
            # Make sure chunks are pre-sorted
            await evaluate_chunk(
                context,
                TableChunk.from_message(msg, br=context.br()),
                ir,
                ir_context=ir_context,
            ),
            ir,
        )
        local_candidates_list.append(
            TableChunk.from_pylibcudf_table(
                _select_local_split_candidates(
                    df.select(by), by, num_partitions, seq_num
                ).table,
                df.stream,
                exclusive_view=True,
                br=context.br(),
            )
        )
        if ir.stable:
            nrows = df.table.num_rows()
            start = (comm.rank * (1 << 48)) + local_row_offset
            seq_id_col = plc.filling.sequence(
                nrows,
                plc.Scalar.from_py(
                    start, plc.DataType(plc.TypeId.UINT64), stream=df.stream
                ),
                plc.Scalar.from_py(
                    1, plc.DataType(plc.TypeId.UINT64), stream=df.stream
                ),
                stream=df.stream,
            )
            local_row_offset += nrows
            tbl = plc.Table([*df.table.columns(), seq_id_col])
        else:
            tbl = df.table
        chunk_store.insert(
            Message(
                seq_num,
                TableChunk.from_pylibcudf_table(
                    tbl, df.stream, exclusive_view=True, br=context.br()
                ),
            )
        )
        del df

    return local_candidates_list


async def _insert_chunks_into_shuffle(
    context: Context,
    comm: Communicator,
    num_partitions: int,
    collective_ids: list[int],
    metadata_in: ChannelMetadata,
    chunk_store: ChunkStore,
    sort_boundaries_df: DataFrame,
    ir: Sort,
    ir_context: IRExecutionContext,
    by: list[str],
) -> tuple[ShuffleManager, Sort]:
    """Create shuffle manager and insert each buffered chunk with sort-based splits."""
    column_order = list(ir.order)
    null_order = list(ir.null_order)
    by_indices = names_to_indices(tuple(by), ir.schema)

    skip_insert = metadata_in.duplicated and comm.rank != 0

    shuffle = ShuffleManager(
        context,
        comm,
        num_partitions,
        collective_ids.pop(),
        partition_assignment=PartitionAssignment.CONTIGUOUS,
    )
    async with shuffle.inserting() as inserter:
        for msg in chunk_store:
            if skip_insert:
                continue
            seq_num = msg.sequence_number
            available_chunk = TableChunk.from_message(
                msg, br=context.br()
            ).make_available_and_spill(context.br(), allow_overbooking=True)
            tbl = available_chunk.table_view()
            sort_cols_tbl = plc.Table([tbl.columns()[i] for i in by_indices])

            stream = get_joined_cuda_stream(
                ir_context.get_cuda_stream,
                upstreams=(available_chunk.stream, sort_boundaries_df.stream),
            )

            splits = find_sort_splits(
                sort_cols_tbl,
                sort_boundaries_df.table,
                seq_num,
                column_order,
                null_order,
                stream=stream,
                chunk_relative=True,
            )
            inserter.insert_split(available_chunk, splits)

    post_sort_ir = ir
    if ir.stable:
        assert ir.zlice is None
        seq_id_name = next(unique_names(ir.schema.keys()))
        post_sort_ir = Sort(
            ir.schema | {seq_id_name: DataType(pl.UInt64())},
            (
                *ir.by,
                NamedExpr(seq_id_name, Col(DataType(pl.UInt64()), seq_id_name)),
            ),
            (*ir.order, plc.types.Order.ASCENDING),
            (*ir.null_order, plc.types.NullOrder.AFTER),
            ir.stable,
            None,
            ir.children[0],
        )

    return shuffle, post_sort_ir


async def _extract_partitions_and_send(
    context: Context,
    ch_out: Channel[TableChunk],
    shuffle: ShuffleManager,
    post_sort_ir: Sort,
    ir_context: IRExecutionContext,
    output_schema: Schema,
    *,
    tracer: ActorTracer | None,
) -> None:
    """Extract each local partition from the shuffle, sort if needed, and send."""
    ncols_out = len(output_schema)
    for partition_id in shuffle.local_partitions():
        stream = ir_context.get_cuda_stream()
        table = shuffle.extract_chunk(partition_id, stream)
        if table.num_rows() > 0:
            table = post_sort_ir.do_evaluate(
                *post_sort_ir._non_child_args,
                DataFrame.from_table(
                    table,
                    list(post_sort_ir.schema.keys()),
                    list(post_sort_ir.schema.values()),
                    stream,
                ),
                context=ir_context,
            ).table
            if table.num_columns() > ncols_out:
                table = plc.Table(table.columns()[:ncols_out])
            if tracer is not None:
                tracer.add_chunk(table=table)
            await ch_out.send(
                context,
                Message(
                    partition_id,
                    TableChunk.from_pylibcudf_table(
                        table, stream, exclusive_view=True, br=context.br()
                    ),
                ),
            )

    await ch_out.drain(context)


@define_actor()
async def sort_actor(
    context: Context,
    comm: Communicator,
    ir: Sort,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    by: list[str],
    num_partitions: int,
    executor: StreamingExecutor,
    collective_ids: list[int],
) -> None:
    """Streaming sort actor."""
    ch_replay = context.create_channel()
    async with shutdown_on_error(
        context, ch_in, ch_out, ch_replay, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        metadata_in = await recv_metadata(ch_in, context)

        if ir.zlice is not None:
            assert _has_simple_zlice(ir.zlice), (
                f"This slice not supported in `sort_actor`: {ir.zlice}."
            )
            await _simple_top_or_bottom_k(
                context,
                comm,
                ch_in,
                ch_out,
                ir,
                ir_context,
                metadata_in,
                collective_ids,
                tracer,
            )
            return

        sampled_chunks, num_partitions = await _sample_chunks_for_size_estimate(
            context, comm, ch_in, num_partitions, metadata_in, executor, collective_ids
        )

        output_metadata = ChannelMetadata(
            local_count=max(1, num_partitions // comm.nranks),
            partitioning=Partitioning(inter_rank=None, local="inherit"),
        )
        await send_metadata(ch_out, context, output_metadata)

        chunk_store = ChunkStore(context)
        _, local_candidates_list = await gather_in_task_group(
            replay_buffered_channel(
                context, ch_replay, ch_in, sampled_chunks, metadata_in, trace_ir=ir
            ),
            _receive_and_buffer_chunks(
                context,
                ch_replay,
                chunk_store,
                ir,
                by,
                num_partitions,
                comm,
                ir_context,
            ),
        )

        need_allgather = comm.nranks > 1 and not metadata_in.duplicated
        sort_boundaries_df = await _compute_sort_boundaries(
            context,
            comm,
            ir_context,
            local_candidates_list,
            ir,
            by,
            num_partitions,
            collective_ids.pop() if need_allgather else None,
        )

        shuffle, post_sort_ir = await _insert_chunks_into_shuffle(
            context,
            comm,
            num_partitions,
            collective_ids,
            metadata_in,
            chunk_store,
            sort_boundaries_df,
            ir,
            ir_context,
            by,
        )

        await _extract_partitions_and_send(
            context,
            ch_out,
            shuffle,
            post_sort_ir,
            ir_context,
            ir.schema,
            tracer=tracer,
        )


@generate_ir_sub_network.register(Sort)
def _sort_rapidsmpf_network(ir: Sort, rec: SubNetGenerator) -> tuple[dict, dict]:
    """Wire multi-partition ``Sort`` to ``sort_actor``; single-partition uses ``default_node_single``."""
    executor = rec.state["config_options"].executor
    partition_info = rec.state["partition_info"]
    dynamic = executor.dynamic_planning is not None

    if partition_info[ir].count == 1 and (
        not dynamic or isinstance(ir.children[0], Repartition)
    ):
        nodes, channels = process_children(ir, rec)
        channels[ir] = ChannelManager(rec.state["context"])
        nodes[ir] = [
            default_node_single(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                channels[ir.children[0]].reserve_output_slot(),
            )
        ]
        return nodes, channels

    (child,) = ir.children
    nodes, channels = rec(child)
    by = [ne.value.name for ne in ir.by if isinstance(ne.value, Col)]
    if len(by) != len(ir.by):
        raise NotImplementedError("Sorting columns must be column names.")

    collective_ids = list(rec.state["collective_id_map"][ir])
    expected_id_count = 3 if dynamic else 2
    assert len(collective_ids) == expected_id_count, (
        f"Sort must have {expected_id_count} collective IDs, got {len(collective_ids)}."
    )

    channels[ir] = ChannelManager(rec.state["context"])
    nodes[ir] = [
        sort_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            ch_in=channels[child].reserve_output_slot(),
            ch_out=channels[ir].reserve_input_slot(),
            by=by,
            num_partitions=partition_info[ir].count,
            executor=executor,
            collective_ids=collective_ids,
        )
    ]
    return nodes, channels


async def _collect_boundaries(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    sorted_ch_in: Channel[TableChunk],
    hint_sorted_info: tuple[tuple[str, bool, bool], ...],
    collective_ids: list[int],
    ir: HintSorted,
) -> tuple[TableChunk | None, bool]:
    """Collect boundaries from a sorted-data channel."""
    stream = ir_context.get_cuda_stream()
    column_order = [
        plc.types.Order.DESCENDING if desc else plc.types.Order.ASCENDING
        for _, desc, _ in hint_sorted_info
    ]
    null_order = [
        plc.types.NullOrder.AFTER if nl else plc.types.NullOrder.BEFORE
        for _, _, nl in hint_sorted_info
    ]

    local_parts: list[tuple[int, plc.Table]] = []
    while (msg := await sorted_ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg, br=context.br()).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        tbl = chunk.table_view()
        if (n := tbl.num_rows()) == 0:
            continue  # drop_empty_chunks will drop this chunk
        slices = plc.copying.slice(
            tbl, [0, 1] if n == 1 else [0, 1, n - 1, n], stream=stream
        )
        two_rows = plc.concatenate.concatenate(
            [slices[0], slices[0 if n == 1 else 1]], stream=stream
        )
        local_parts.append((msg.sequence_number, two_rows))

    local_parts.sort(key=lambda x: x[0])
    tables = [t for _, t in local_parts]
    local_table: plc.Table | None = (
        plc.concatenate.concatenate(tables, stream=stream) if tables else None
    )

    if comm.nranks > 1:
        local_chunk = (
            TableChunk.from_pylibcudf_table(
                local_table, stream, exclusive_view=True, br=context.br()
            )
            if local_table is not None
            else empty_table_chunk(ir.children[1], context, stream)
        )
        allgather = AllGatherManager(context, comm, collective_ids.pop())
        with allgather.inserting() as inserter:
            inserter.insert(comm.rank, local_chunk)
        full_table = await allgather.extract_concatenated(stream, ordered=True)
    else:
        collective_ids.pop()
        if local_table is None:
            return None, False
        full_table = local_table

    N = full_table.num_rows() // 2

    if N == 0:
        return None, False

    if N == 1:
        empty = plc.copying.slice(full_table, [0, 0], stream=stream)[0]
        return (
            TableChunk.from_pylibcudf_table(
                empty, stream, exclusive_view=True, br=context.br()
            ),
            True,
        )

    if not plc.sorting.is_sorted(full_table, column_order, null_order, stream=stream):
        return None, False

    def _slice_rows(positions: list[int]) -> plc.Table:
        bounds = [v for k in positions for v in (k, k + 1)]
        return plc.concatenate.concatenate(
            plc.copying.slice(full_table, bounds, stream=stream), stream=stream
        )

    boundaries_tbl = _slice_rows(list(range(1, 2 * N - 1, 2)))
    next_first_tbl = _slice_rows(list(range(2, 2 * N, 2)))

    bool_type = plc.DataType(plc.TypeId.BOOL8)
    row_eq = plc.binaryop.binary_operation(
        boundaries_tbl.columns()[0],
        next_first_tbl.columns()[0],
        plc.binaryop.BinaryOperator.NULL_EQUALS,
        bool_type,
        stream=stream,
    )
    for j in range(1, boundaries_tbl.num_columns()):
        row_eq = plc.binaryop.binary_operation(
            row_eq,
            plc.binaryop.binary_operation(
                boundaries_tbl.columns()[j],
                next_first_tbl.columns()[j],
                plc.binaryop.BinaryOperator.NULL_EQUALS,
                bool_type,
                stream=stream,
            ),
            plc.binaryop.BinaryOperator.LOGICAL_AND,
            bool_type,
            stream=stream,
        )
    strict = (
        plc.stream_compaction.apply_boolean_mask(
            plc.Table([row_eq]), row_eq, stream=stream
        ).num_rows()
        == 0
    )
    return (
        TableChunk.from_pylibcudf_table(
            boundaries_tbl, stream, exclusive_view=True, br=context.br()
        ),
        strict,
    )


def _construct_ordered_metadata(
    ir: HintSorted,
    data_metadata: ChannelMetadata,
    hint_sorted_info: tuple[tuple[str, bool, bool], ...],
    boundaries: TableChunk,
    *,
    strict_boundaries: bool,
) -> ChannelMetadata:
    """Construct ChannelMetadata with an appropriate OrderScheme."""
    col_names = list(ir.schema.keys())
    keys = [
        OrderKey(
            col_names.index(name),
            plc.types.Order.DESCENDING if desc else plc.types.Order.ASCENDING,
            plc.types.NullOrder.AFTER if nl else plc.types.NullOrder.BEFORE,
        )
        for name, desc, nl in hint_sorted_info
    ]
    return ChannelMetadata(
        local_count=data_metadata.local_count,
        partitioning=Partitioning(
            OrderScheme(keys, boundaries, strict_boundaries=strict_boundaries),
            "inherit",
        ),
        duplicated=data_metadata.duplicated,
    )


@define_actor()
async def hint_sorted(
    context: Context,
    comm: Communicator,
    ir: HintSorted,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    data_ch_in: Channel[TableChunk],
    sorted_ch_in: Channel[TableChunk],
    hint_sorted_info: tuple[tuple[str, bool, bool], ...],
    collective_ids: list[int],
) -> None:
    """Hint-sorted actor."""
    nonempty_data_ch = context.create_channel()
    async with shutdown_on_error(
        context,
        data_ch_in,
        sorted_ch_in,
        nonempty_data_ch,
        ch_out,
        trace_ir=ir,
        ir_context=ir_context,
    ) as tracer:
        data_metadata, _ = await gather_in_task_group(
            recv_metadata(data_ch_in, context),
            recv_metadata(sorted_ch_in, context),
        )

        boundaries, strict_boundaries = await _collect_boundaries(
            context,
            comm,
            ir_context,
            sorted_ch_in,
            hint_sorted_info,
            collective_ids,
            ir,
        )

        output_metadata = (
            _construct_ordered_metadata(
                ir,
                data_metadata,
                hint_sorted_info,
                boundaries,
                strict_boundaries=strict_boundaries,
            )
            if boundaries is not None
            else data_metadata
        )

        data_ir = ir.children[0]
        await gather_in_task_group(
            drop_empty_chunks(context, data_ch_in, nonempty_data_ch),
            chunkwise_evaluate(
                context,
                MapFunction(
                    data_ir.schema, "hint_sorted", (hint_sorted_info,), data_ir
                ),
                ir_context,
                ch_out,
                nonempty_data_ch,
                output_metadata,
                tracer=tracer,
            ),
        )


@generate_ir_sub_network.register(HintSorted)
def _hint_sorted_rapidsmpf_network(
    ir: HintSorted, rec: SubNetGenerator
) -> tuple[dict, dict]:
    df, keys = ir.children
    nodes, channels = process_children(ir, rec)
    collective_ids = list(rec.state["collective_id_map"][ir])
    assert len(collective_ids) == 1
    channels[ir] = ChannelManager(rec.state["context"])
    nodes[ir] = [
        hint_sorted(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            ch_out=channels[ir].reserve_input_slot(),
            data_ch_in=channels[df].reserve_output_slot(),
            sorted_ch_in=channels[keys].reserve_output_slot(),
            hint_sorted_info=ir.sorted_info,
            collective_ids=collective_ids,
        )
    ]
    return nodes, channels
