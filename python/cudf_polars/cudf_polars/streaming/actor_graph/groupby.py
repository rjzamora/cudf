# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GroupBy and Distinct logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    OrderKey,
    OrderScheme,
    Partitioning,
    RowOrdering,
    RowOrderingScope,
)
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.communicator.single import new_communicator as single_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataType
from cudf_polars.dsl.expr import Col, NamedExpr
from cudf_polars.dsl.ir import IR, Distinct, GroupBy, Select
from cudf_polars.dsl.utils.naming import names_to_indices, unique_names
from cudf_polars.streaming.actor_graph.collectives.ordering import (
    endpoint_rows_equal,
    gather_rank_endpoints,
)
from cudf_polars.streaming.actor_graph.collectives.shuffle import ShuffleManager
from cudf_polars.streaming.actor_graph.collectives.slice_exchange import SparseExchange
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.partitioning import (
    NormalizedPartitioning,
    is_globally_row_ordered_by,
    maybe_remap_partitioning,
    partitioning_or_none,
    replace_row_orderings,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    MAX_ROWS_PER_PARTITION,
    ChannelManager,
    ChunkStore,
    _make_hash_shuffle_metadata,
    allgather_and_reduce,
    allgather_reduce,
    chunkwise_evaluate,
    empty_table_chunk,
    evaluate_batch,
    evaluate_chunk,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.groupby import (
    _has_order_sensitive_agg,
    _has_stable_sorted_agg,
    combine,
    decompose,
)
from cudf_polars.streaming.partitioning_requests import OrderPartitioningRequest
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.utils.cuda_stream import stream_ordered_after

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.collectives.ordering import RankEndpoints
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.partitioning import PartitioningLevel
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer
    from cudf_polars.streaming.partitioning_requests import PartitioningRequest
    from cudf_polars.typing import Schema


@dataclass
class DecomposedGroupBy:
    """Holds decomposed GroupBy or Distinct operations for multi-phase aggregation."""

    ir: GroupBy | Distinct
    """The original IR node."""
    piecewise_ir: GroupBy | Distinct
    """The decomposed piecewise IR node."""
    reduction_ir: GroupBy | Distinct
    """The decomposed reduction IR node. Same as piecewise_ir for Distinct."""
    select_ir: Select | None
    """The decomposed select IR node. Always None for Distinct."""
    need_preshuffle: bool
    """Whether the operation requires preshuffling."""
    output_indices: tuple[int, ...]
    """The output indices of the GroupBy keys."""
    shuffle_indices: tuple[int, ...]
    """The column indices to shuffle on."""

    @classmethod
    def from_ir(cls, ir: GroupBy | Distinct) -> DecomposedGroupBy:
        """Decompose a GroupBy IR node into multi-phase operations."""
        piecewise_ir: GroupBy | Distinct
        reduction_ir: GroupBy | Distinct
        select_ir: Select | None
        need_preshuffle: bool
        output_indices: tuple[int, ...]
        shuffle_indices: tuple[int, ...]

        if isinstance(ir, Distinct):
            piecewise_ir = reduction_ir = ir
            select_ir = None
            need_preshuffle = (
                ir.keep == plc.stream_compaction.DuplicateKeepOption.KEEP_NONE
            )
        elif isinstance(ir, GroupBy):
            name_generator = unique_names(ir.schema.keys())
            selection_exprs, piecewise_exprs, reduction_exprs, need_preshuffle = (
                combine(
                    *(
                        decompose(agg.name, agg.value, names=name_generator)
                        for agg in ir.agg_requests
                    )
                )
            )

            # Piecewise groupby schema and IR
            pwise_schema = {k.name: k.value.dtype for k in ir.keys} | {
                k.name: k.value.dtype for k in piecewise_exprs
            }
            piecewise_ir = GroupBy(
                pwise_schema,
                ir.keys,
                piecewise_exprs,
                ir.maintain_order,
                None,
                ir.children[0],
            )

            # Grouped keys for reduction and selection
            groupby_keys = tuple(
                NamedExpr(k.name, Col(k.value.dtype, k.name)) for k in ir.keys
            )

            # Reduction groupby schema and IR
            reduction_schema = {k.name: k.value.dtype for k in groupby_keys} | {
                k.name: k.value.dtype for k in reduction_exprs
            }
            reduction_ir = GroupBy(
                reduction_schema,
                groupby_keys,
                reduction_exprs,
                ir.maintain_order,
                None,
                piecewise_ir,
            )

            select_ir = Select(
                ir.schema,
                [
                    *(
                        NamedExpr(k.name, Col(k.value.dtype, k.name))
                        for k in groupby_keys
                    ),
                    *selection_exprs,
                ],
                False,  # noqa: FBT003
                reduction_ir,
            )
        else:  # pragma: no cover
            raise TypeError(f"Unsupported IR type: {type(ir)}")

        output_indices = _key_indices(ir, ir.schema)
        if isinstance(ir, Distinct):
            shuffle_indices = output_indices
        else:
            shuffle_indices = _key_indices(piecewise_ir, piecewise_ir.schema)

        return cls(
            ir=ir,
            piecewise_ir=piecewise_ir,
            reduction_ir=reduction_ir,
            select_ir=select_ir,
            need_preshuffle=need_preshuffle,
            output_indices=output_indices,
            shuffle_indices=shuffle_indices,
        )


async def _local_aggregation(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    target_partition_size: int,
    *,
    allow_early_exit: bool = True,
) -> tuple[TableChunk, bool, int]:
    """
    Local groupby or distinct aggregation.

    Parameters
    ----------
    context
        The rapidsmpf context.
    decomposed
        The decomposed groupby or distinct.
    ir_context
        The IR execution context.
    ch_in
        The input channel.
    target_partition_size
        The target partition size.
    allow_early_exit
        Whether to allow early exit from the loop when
        the total size exceeds the target partition size.

    Returns
    -------
    aggregated
        The aggregated result.
    input_drained
        Whether the input channel is drained.
    chunks_received
        The number of chunks received from the input channel.
    """
    total_size = 0
    chunks_received = 0
    input_drained = False
    evaluated_chunks: list[TableChunk] = []
    while True:
        msg = await ch_in.recv(context)
        if msg is None:
            input_drained = True
            break

        chunks_received += 1
        chunk = await evaluate_chunk(
            context,
            TableChunk.from_message(msg, br=context.br()),
            decomposed.piecewise_ir,
            ir_context=ir_context,
        )
        chunk = _enforce_schema(chunk, decomposed.piecewise_ir.schema, context.br())

        total_size += chunk.data_alloc_size()
        evaluated_chunks.append(chunk)
        if total_size > target_partition_size and len(evaluated_chunks) > 1:
            evaluated_chunks = [
                await evaluate_batch(
                    evaluated_chunks,
                    context,
                    decomposed.reduction_ir,
                    ir_context=ir_context,
                )
            ]
            total_size = evaluated_chunks[0].data_alloc_size()
        if total_size > target_partition_size and allow_early_exit:
            break

    aggregated: TableChunk
    if len(evaluated_chunks) > 1:
        aggregated = await evaluate_batch(
            evaluated_chunks,
            context,
            decomposed.reduction_ir,
            ir_context=ir_context,
        )
    elif evaluated_chunks:
        aggregated = evaluated_chunks[0]
    else:
        aggregated = empty_table_chunk(
            decomposed.reduction_ir,
            context,
            ir_context.get_cuda_stream(),
        )

    return aggregated, input_drained, chunks_received


async def _tree_reduce(
    context: Context,
    comm: Communicator,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    collective_id: int,
    *,
    local: bool,
    aggregated: TableChunk,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Execute groupby or distinct using tree reduction to a single output.

    Parameters
    ----------
    context
        The rapidsmpf streaming context.
    comm
        The communicator.
    decomposed
        The decomposed groupby containing piecewise, reduction, and select IRs.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    metadata_in
        The input channel metadata.
    aggregated
        The aggregated result for already-evaluated chunks.
    collective_id
        Collective ID for allgather (used when local is False).
    local
        Whether to use a local aggregation.
    tracer
        Optional tracer for runtime metrics.
    """
    need_allgather = not local and not metadata_in.duplicated and comm.nranks > 1

    metadata_out = ChannelMetadata(
        local_count=1,
        partitioning=None,
        duplicated=True if need_allgather else metadata_in.duplicated,
    )
    await send_metadata(ch_out, context, metadata_out)

    if need_allgather:
        aggregated = await allgather_and_reduce(
            context,
            comm,
            collective_id,
            _enforce_schema(aggregated, decomposed.reduction_ir.schema, context.br()),
            decomposed.reduction_ir,
            ir_context,
        )

    if decomposed.select_ir is not None:
        aggregated = await evaluate_chunk(
            context, aggregated, decomposed.select_ir, ir_context=ir_context
        )
    await send_chunk(context, ch_out, aggregated, 0, tracer=tracer)

    await ch_out.drain(context)


async def _shuffle_reduce(
    context: Context,
    comm: Communicator,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    modulus: int,
    collective_id: int,
    target_partition_size: int,
    *,
    local: bool,
    aggregated: TableChunk,
    input_drained: bool = False,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Shuffle-based groupby or distinct.

    Shuffles data by keys, then applies local groupby or distinct to
    each partition. Use when the expected output is large.

    Parameters
    ----------
    context
        The rapidsmpf streaming context for channel operations.
    comm
        The communicator.
    decomposed
        The decomposed groupby or distinct containing piecewise, reduction, and select IRs.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    metadata_in
        The input channel metadata.
    modulus
        The modulus for the shuffle operation.
    collective_id
        The collective ID for the shuffle operation.
    target_partition_size
        Target partition size for local aggregation batches.
    local
        Whether to use a local shuffle operation.
    aggregated
        The aggregated result for already-evaluated chunks.
    input_drained
        Whether the input channel is drained.
    tracer
        Optional tracer for runtime metrics.
    """
    shuffle_context = context
    shuffle_comm = comm
    if local and comm.nranks > 1:
        options = Options(get_environment_variables())
        shuffle_comm = single_comm(options, comm.progress_thread)
        shuffle_context = Context(shuffle_comm.logger, context.br(), options)
    modulus = max(shuffle_comm.nranks, modulus)
    metadata_out = _make_hash_shuffle_metadata(
        shuffle_comm, decomposed.output_indices, modulus, metadata_in
    )
    await send_metadata(ch_out, context, metadata_out)

    shuffle = ShuffleManager(
        shuffle_context,
        shuffle_comm,
        modulus,
        collective_id,
    )
    async with shuffle.inserting() as inserter:
        await inserter.insert_hash(
            _enforce_schema(
                aggregated,
                decomposed.reduction_ir.schema,
                context.br(),
            ),
            decomposed.shuffle_indices,
        )
        del aggregated

        while not input_drained:
            aggregated, input_drained, _ = await _local_aggregation(
                context,
                decomposed,
                ir_context,
                ch_in,
                target_partition_size,
            )
            await inserter.insert_hash(
                _enforce_schema(
                    aggregated, decomposed.reduction_ir.schema, context.br()
                ),
                decomposed.shuffle_indices,
            )
            del aggregated
    extract_irs = [decomposed.reduction_ir] + (
        [decomposed.select_ir] if decomposed.select_ir else []
    )
    for partition_id in shuffle.local_partitions():
        stream = ir_context.get_cuda_stream()
        partition_chunk = TableChunk.from_pylibcudf_table(
            await shuffle.extract_chunk(partition_id, stream),
            stream,
            exclusive_view=True,
            br=context.br(),
        )
        partition_chunk = await evaluate_chunk(
            context,
            partition_chunk,
            *extract_irs,
            ir_context=ir_context,
        )
        await send_chunk(context, ch_out, partition_chunk, partition_id, tracer=tracer)

    await ch_out.drain(context)


def _table_chunk(
    context: Context, table: plc.Table, stream: Any, *, exclusive: bool = True
) -> TableChunk:
    return TableChunk.from_pylibcudf_table(
        table, stream, exclusive_view=exclusive, br=context.br()
    )


async def _copy_key_endpoints(
    context: Context,
    first: TableChunk,
    last: TableChunk,
    order_keys: Sequence[OrderKey],
) -> tuple[TableChunk, TableChunk, TableChunk]:
    """Copy the first and last key rows into independent storage."""
    inputs = [first] if first is last else [first, last]
    chunks, extra = await make_table_chunks_available_or_wait(
        context, inputs, reserve_extra=0, net_memory_delta=0
    )
    first, last = chunks[0], chunks[-1]
    with (
        opaque_memory_usage(extra),
        stream_ordered_after(
            context.br().stream_pool.get_stream,
            upstreams=(first.stream, last.stream),
        ) as stream,
    ):
        views = []
        for chunk, row in ((first, 0), (last, last.shape[0] - 1)):
            table = chunk.table_view()
            keys = plc.Table([table.columns()[key.column_index] for key in order_keys])
            views.append(plc.copying.slice(keys, [row, row + 1], stream=stream)[0])
        cost = sum(
            _table_chunk(context, view, stream, exclusive=False).into_packed_data_cost()
            for view in views
        )
        reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
            cost, net_memory_delta=0
        )
        with opaque_memory_usage(reservation):
            result = plc.concatenate.concatenate(
                views, stream=stream, mr=context.br().device_mr
            )
    return _table_chunk(context, result, stream), first, last


async def _sort_chunk(
    context: Context, chunk: TableChunk, keys: Sequence[OrderKey]
) -> TableChunk:
    """Stably sort one partial reduction."""
    if chunk.shape[0] < 2:
        return chunk
    chunk, extra = await make_table_chunks_available_or_wait(
        context, chunk, reserve_extra=chunk.data_alloc_size(), net_memory_delta=0
    )
    with opaque_memory_usage(extra):
        table = chunk.table_view()
        result = plc.sorting.stable_sort_by_key(
            table,
            plc.Table([table.columns()[key.column_index] for key in keys]),
            [key.order for key in keys],
            [key.null_order for key in keys],
            stream=chunk.stream,
            mr=context.br().device_mr,
        )
    return _table_chunk(context, result, chunk.stream)


async def _split_key_run(
    context: Context,
    chunk: TableChunk,
    keys: Sequence[OrderKey],
    *,
    leading: bool,
) -> tuple[TableChunk, TableChunk | None]:
    """Split a sorted chunk beside its leading or trailing equal-key run."""
    if not (nrows := chunk.shape[0]):
        return chunk, None
    chunk, extra = await make_table_chunks_available_or_wait(
        context, chunk, reserve_extra=chunk.data_alloc_size(), net_memory_delta=0
    )
    with opaque_memory_usage(extra):
        table = chunk.table_view()
        key_table = plc.Table([table.columns()[key.column_index] for key in keys])
        row = 0 if leading else nrows - 1
        boundary = plc.copying.slice(key_table, [row, row + 1], stream=chunk.stream)[0]
        find = plc.search.upper_bound if leading else plc.search.lower_bound
        split = cast(
            "int",
            find(
                key_table,
                boundary,
                [key.order for key in keys],
                [key.null_order for key in keys],
                stream=chunk.stream,
            )
            .to_scalar(stream=chunk.stream)
            .to_py(stream=chunk.stream),
        )
        if split == nrows:
            return chunk, None
        views = plc.copying.split(table, [split], stream=chunk.stream)
        pieces = tuple(
            chunk
            if view.num_rows() == nrows
            else _table_chunk(
                context,
                view.copy(chunk.stream, context.br().device_mr),
                chunk.stream,
            )
            for view in views
        )
    first, second = pieces
    return first, second


async def _ordered_reductions(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    prefix_keys: tuple[OrderKey, ...],
    output_keys: tuple[OrderKey, ...],
) -> AsyncIterator[tuple[TableChunk, bool]]:
    """
    Yield complete reductions followed by one unresolved trailing run.

    Reduced state for one ordered-prefix run must fit in a local reduction.
    Full-key ordering limits that state to one group.
    """
    carry = None
    while (message := await ch_in.recv(context)) is not None:
        current = await evaluate_chunk(
            context,
            TableChunk.from_message(message, br=context.br()),
            decomposed.piecewise_ir,
            ir_context=ir_context,
        )
        current = _enforce_schema(
            current,
            decomposed.reduction_ir.schema,
            context.br(),
        )
        if carry is not None:
            current = await evaluate_batch(
                [carry, current],
                context,
                decomposed.reduction_ir,
                ir_context=ir_context,
            )
        current = await _sort_chunk(context, current, output_keys)
        complete, carry = await _split_key_run(
            context, current, prefix_keys, leading=False
        )
        if complete.shape[0]:
            yield complete, False
    if carry is not None:
        yield carry, True


def _boundary_owners(
    endpoints: RankEndpoints,
    keys: tuple[OrderKey, ...],
    stream: Any,
) -> dict[int, int]:
    """Assign each trailing key run to its rightmost nonempty rank."""
    nranks = len(endpoints.ranks)
    if not nranks:
        return {}
    full_rank = tuple(
        endpoint_rows_equal(endpoints, 2 * i, 2 * i + 1, keys, stream)
        for i in range(nranks)
    )
    shared_boundary = tuple(
        endpoint_rows_equal(endpoints, 2 * i + 1, 2 * i + 2, keys, stream)
        for i in range(nranks - 1)
    )
    owners = {endpoints.ranks[-1]: endpoints.ranks[-1]}
    for position in range(nranks - 2, -1, -1):
        rank = endpoints.ranks[position]
        next_rank = endpoints.ranks[position + 1]
        owners[rank] = (
            owners[next_rank]
            if shared_boundary[position] and full_rank[position + 1]
            else next_rank
            if shared_boundary[position]
            else rank
        )
    return owners


async def _ordered_reduce(
    context: Context,
    comm: Communicator,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    collective_ids: list[int],
    input_keys: tuple[OrderKey, ...],
    output_keys: tuple[OrderKey, ...],
    *,
    distributed: bool,
    tracer: ActorTracer | None,
) -> None:
    """Stream ordered reductions, reconciling rank-boundary state if needed."""
    prefix_keys = output_keys[: len(input_keys)]
    reductions = _ordered_reductions(
        context, decomposed, ir_context, ch_in, prefix_keys, output_keys
    )

    async def emit(chunk: TableChunk | None, sequence_number: int) -> int:
        if chunk is None or chunk.shape[0] == 0:
            return sequence_number
        if decomposed.select_ir is not None:
            chunk = await evaluate_chunk(
                context, chunk, decomposed.select_ir, ir_context=ir_context
            )
        await send_chunk(context, ch_out, chunk, sequence_number, tracer=tracer)
        return sequence_number + 1

    async def send_output_metadata(local_count: int) -> None:
        partitioning = Partitioning()
        if not distributed:
            remapped = maybe_remap_partitioning(
                decomposed.ir,
                metadata_in.partitioning,
                context=context,
                preserves_row_order=True,
            )
            if remapped is not None:
                partitioning = (
                    partitioning_or_none(remapped.inter_rank, None, ()) or partitioning
                )
        result_keys = tuple(
            OrderKey(index, key.order, key.null_order)
            for index, key in zip(decomposed.output_indices, output_keys, strict=False)
        )
        await send_metadata(
            ch_out,
            context,
            ChannelMetadata(
                local_count=local_count,
                partitioning=replace_row_orderings(
                    partitioning,
                    (RowOrdering(result_keys, RowOrderingScope.GLOBAL),),
                ),
                duplicated=False,
            ),
        )

    if not distributed:
        await send_output_metadata(
            max(1, metadata_in.local_count),
        )
        if tracer is not None:
            tracer.decision = "ordered"
        sequence_number = 0
        async for chunk, _ in reductions:
            sequence_number = await emit(chunk, sequence_number)
        if sequence_number == 0:
            await emit(
                empty_table_chunk(
                    decomposed.reduction_ir, context, ir_context.get_cuda_stream()
                ),
                0,
            )
        await ch_out.drain(context)
        return

    complete = ChunkStore(context)
    trailing = None
    try:
        async for chunk, is_trailing in reductions:
            if is_trailing:
                trailing = chunk
            else:
                complete.insert(Message(0, chunk))

        stream = ir_context.get_cuda_stream()
        messages = iter(complete)
        first_message = next(messages, None)
        first = (
            TableChunk.from_message(first_message, br=context.br())
            if first_message is not None
            else trailing
        )
        if first is None:
            empty = empty_table_chunk(decomposed.reduction_ir, context, stream)
            table = empty.table_view()
            local_endpoints = _table_chunk(
                context,
                plc.Table([table.columns()[key.column_index] for key in prefix_keys]),
                stream,
                exclusive=False,
            )
        else:
            assert trailing is not None
            local_endpoints, first, trailing = await _copy_key_endpoints(
                context, first, trailing, prefix_keys
            )
        (endpoints,) = await gather_rank_endpoints(
            context,
            comm,
            ir_context,
            collective_ids.pop(),
            (local_endpoints,),
            stream,
        )
        owners = _boundary_owners(endpoints, prefix_keys, stream)
        sources = tuple(
            sorted(
                source
                for source, owner in owners.items()
                if owner == comm.rank and source != comm.rank
            )
        )
        owner = owners.get(comm.rank)
        destinations = (owner,) if owner is not None and owner != comm.rank else ()
        local_count_estimate = (
            1
            if not endpoints.ranks and comm.rank == 0
            else max(0, metadata_in.local_count)
        )
        await send_output_metadata(local_count_estimate)
        if tracer is not None:
            tracer.decision = "ordered_distributed"

        async with SparseExchange(
            context,
            comm,
            collective_ids.pop(),
            sources=sources,
            destinations=destinations,
        ) as exchange:
            first_is_trailing = first is trailing
            if destinations:
                assert trailing is not None
                await exchange.insert_chunk(destinations[0], trailing)
                trailing = None
            await exchange.finish()

            local_first = None
            remainder = None
            if sources:
                if not first_is_trailing:
                    local_first, remainder = await _split_key_run(
                        context, first, prefix_keys, leading=True
                    )
                else:
                    local_first, trailing = trailing, None
                states: list[TableChunk] = []
                for source in sources:
                    remote = exchange.extract(source)
                    try:
                        states.extend(
                            TableChunk.from_message(message, br=context.br())
                            for message in remote
                        )
                    finally:
                        remote.clear()
                if local_first is not None:
                    states.append(local_first)
                boundary = (
                    await evaluate_batch(
                        states,
                        context,
                        decomposed.reduction_ir,
                        ir_context=ir_context,
                    )
                    if len(states) > 1
                    else states[0]
                )
                boundary = await _sort_chunk(context, boundary, output_keys)
            else:
                boundary = first if first_message is not None else None

            sequence_number = 0
            chunks = [boundary, remainder]
            chunks.extend(
                TableChunk.from_message(message, br=context.br())
                for message in messages
            )
            chunks.append(trailing)
            for chunk in chunks:
                sequence_number = await emit(chunk, sequence_number)
            if not endpoints.ranks and comm.rank == 0:
                await emit(
                    empty_table_chunk(
                        decomposed.reduction_ir,
                        context,
                        stream,
                    ),
                    sequence_number,
                )
    finally:
        complete.clear()
    await ch_out.drain(context)


def _enforce_schema(
    chunk: TableChunk,
    canonical_schema: dict[str, Any],
    br: BufferResource,
) -> TableChunk:
    """Enforce the canonical schema of a TableChunk."""
    tbl = chunk.table_view()
    cols = tbl.columns()
    names = list(canonical_schema.keys())
    if len(cols) != len(names):  # pragma: no cover
        raise ValueError(
            f"Column count ({len(cols)}) does not match schema ({len(names)})"
        )

    target_plcs = []
    needs_cast = False
    for col, name in zip(cols, names, strict=True):
        dt = canonical_schema[name]
        target_plc = (dt if isinstance(dt, DataType) else DataType(dt)).plc_type
        target_plcs.append(target_plc)
        if col.type().id() != target_plc.id():
            needs_cast = True
    if not needs_cast:
        return chunk
    new_columns = [
        plc.unary.cast(col, target_plc, stream=chunk.stream)
        if col.type().id() != target_plc.id()
        else col
        for col, target_plc in zip(cols, target_plcs, strict=True)
    ]
    return TableChunk.from_pylibcudf_table(
        plc.Table(new_columns), chunk.stream, exclusive_view=True, br=br
    )


def _key_indices(
    ir: GroupBy | Distinct, schema: Schema, *, concrete_prefix: bool = False
) -> tuple[int, ...]:
    schema_keys = {n: i for i, n in enumerate(schema.keys())}
    if isinstance(ir, GroupBy):
        if concrete_prefix:
            return names_to_indices(ir.keys, schema, concrete_prefix=True)
        groupby_key_names = tuple(ne.name for ne in ir.keys)
        if not all(k in schema_keys for k in groupby_key_names):
            return ()
        return tuple(schema_keys[k] for k in groupby_key_names)
    else:
        subset = ir.subset or frozenset(ir.schema)
        return tuple(schema_keys[k] for k in schema if k in subset)


def _ordered_groupby_keys(
    ir: GroupBy | Distinct,
    decomposed: DecomposedGroupBy,
    requests: tuple[PartitioningRequest, ...],
    partitioning: Partitioning | None,
    group_keys: tuple[int, ...],
) -> tuple[tuple[OrderKey, ...], tuple[OrderKey, ...]]:
    """Return compatible input and output group-key orderings."""
    if not isinstance(ir, GroupBy) or partitioning is None:
        return (), ()

    def matching_prefix(ordering: RowOrdering) -> tuple[OrderKey, ...]:
        result = []
        for key, index in zip(ordering.order_keys, group_keys, strict=False):
            if key.column_index != index:
                break
            result.append(key)
        return tuple(result)

    candidates = (
        prefix
        for ordering in partitioning.row_orderings
        if not ordering.group_keys
        and (prefix := matching_prefix(ordering))
        and is_globally_row_ordered_by(partitioning, prefix)
    )
    input_keys = max(candidates, key=len, default=())
    if not input_keys:
        return (), ()
    key_names = tuple(key.name for key in ir.keys)
    prefix = tuple(
        (name, key.order, key.null_order)
        for name, key in zip(key_names, input_keys, strict=False)
    )
    if ir.maintain_order:
        if len(input_keys) != len(key_names):
            return input_keys, ()
        directions = input_keys
    else:
        request = next(
            (
                request
                for request in requests
                if isinstance(request, OrderPartitioningRequest)
                and tuple(key.name for key in request.keys) == key_names
                and tuple(
                    (key.name, key.order, key.null_order)
                    for key in request.keys[: len(input_keys)]
                )
                == prefix
            ),
            None,
        )
        if request is None:
            return input_keys, ()
        directions = request.keys
    return input_keys, tuple(
        OrderKey(index, key.order, key.null_order)
        for index, key in zip(decomposed.shuffle_indices, directions, strict=True)
    )


def _maintain_order(ir: GroupBy | Distinct) -> bool:
    if isinstance(ir, GroupBy):
        return (
            ir.maintain_order
            or _has_stable_sorted_agg(ir.agg_requests)
            or _has_order_sensitive_agg(ir.agg_requests)
        )
    else:
        return ir.stable or ir.keep in (
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
        )


async def _choose_strategy(
    context: Context,
    comm: Communicator,
    local_count: int,
    aggregated: TableChunk,
    chunks_received: int,
    input_drained: bool,  # noqa: FBT001
    collective_ids: list[int],
    target_partition_size: int,
    skip_global_comm: bool,  # noqa: FBT001
    maintain_order: bool,  # noqa: FBT001
    tracer: ActorTracer | None,
) -> int:
    """
    Select the best algorithm for the given context and metadata.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    local_count
        The local count of the input channel.
    aggregated
        The aggregated result for already-evaluated chunks.
    chunks_received
        The number of chunks received from the input channel.
    input_drained
        Whether the input channel is drained.
    collective_ids
        The collective IDs.
    target_partition_size
        The target partition size.
    skip_global_comm
        Whether to skip the global communication.
    maintain_order
        Whether the operation should maintain input ordering semantics.
    tracer
        Optional tracer for runtime metrics.

    Returns
    -------
    The output count.
    """
    aggregated_size = aggregated.data_alloc_size()
    sample_count = max(1, chunks_received)
    local_estimated_size = (aggregated_size // sample_count) * local_count
    if input_drained:
        local_estimated_rows = aggregated.shape[0]
    else:
        local_estimated_rows = (
            aggregated.shape[0] * local_count + sample_count - 1
        ) // sample_count

    if skip_global_comm:
        total_estimated_size = local_estimated_size
        total_estimated_rows = local_estimated_rows
        total_chunk_count = local_count
        total_need_shuffle = int(not input_drained)
    else:
        (
            total_estimated_size,
            total_estimated_rows,
            total_chunk_count,
            total_need_shuffle,
        ) = await allgather_reduce(
            context,
            comm,
            collective_ids.pop(),
            local_estimated_size,
            local_estimated_rows,
            local_count,
            int(not input_drained),
        )

    min_row_limit_count = 1
    if total_estimated_rows > 0:
        min_row_limit_count = (
            total_estimated_rows + MAX_ROWS_PER_PARTITION - 1
        ) // MAX_ROWS_PER_PARTITION

    ideal_count = 1
    use_tree = maintain_order or (
        total_need_shuffle == 0
        and (skip_global_comm or total_estimated_rows < MAX_ROWS_PER_PARTITION)
    )
    if not use_tree:
        ideal_count = max(
            2, total_estimated_size // target_partition_size, min_row_limit_count
        )

    output_count_limit = local_count if skip_global_comm else total_chunk_count
    output_count = min(ideal_count, output_count_limit)
    if not use_tree:
        output_count = max(2, output_count, min_row_limit_count)
    if tracer is not None:
        tracer.decision = (
            "tree_local"
            if skip_global_comm and use_tree
            else "shuffle_local"
            if skip_global_comm
            else "tree_allgather"
            if use_tree
            else "shuffle"
        )

    return output_count


@define_actor()
async def groupby_actor(
    context: Context,
    comm: Communicator,
    ir: GroupBy | Distinct,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    target_partition_size: int,
    collective_ids: list[int],
    partitioning_requests: tuple[PartitioningRequest, ...],
) -> None:
    """
    Dynamic GroupBy or Distinct actor that selects the best strategy at runtime.

    Strategy selection based on observed data:
    - Chunk-wise: Data already partitioned on the necessary keys
    - Tree reduction: Small estimated output (< target_partition_size)
    - Shuffle: Large estimated output requiring redistribution

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The IR node to evaluate.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    target_partition_size
        The target partition size.
    collective_ids
        The collective IDs.
    partitioning_requests
        Partitioning requested by downstream consumers.
    """
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        tracer = actor_scope.tracer
        ir_context = actor_scope.require_ir_context()
        metadata_in = await recv_metadata(ch_in, context)

        nranks = comm.nranks
        group_keys = _key_indices(ir, ir.children[0].schema, concrete_prefix=True)
        partitioning = NormalizedPartitioning.from_keys(
            metadata_in.partitioning,
            nranks,
            keys=group_keys,
        )
        partitioning_level: PartitioningLevel = (
            "local" if metadata_in.duplicated else "flat"
        )
        maintain_order = _maintain_order(ir)
        preserves_output_order = ir.preserves_output_order
        fully_partitioned = partitioning.is_strictly_partitioned(
            level=partitioning_level,
        )
        rank_ordered = isinstance(
            partitioning.inter_rank_scheme,
            OrderScheme,
        ) and partitioning.is_strictly_partitioned(level="inter_rank")
        fallback_case = (
            # NOTE: This criteria means that we fell back
            # to one partition at lowering time.
            metadata_in.local_count == 1
            and (metadata_in.duplicated or nranks == 1)
            and isinstance(ir.children[0], Repartition)
        )

        if fully_partitioned or fallback_case:
            if tracer is not None:
                tracer.decision = "chunkwise"
            output_partitioning = maybe_remap_partitioning(
                ir,
                metadata_in.partitioning,
                context=context,
                preserves_row_order=preserves_output_order,
            )
            metadata_out = ChannelMetadata(
                local_count=metadata_in.local_count,
                partitioning=output_partitioning,
                duplicated=metadata_in.duplicated,
            )
            await chunkwise_evaluate(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_out,
                input_metadata=metadata_in,
                tracer=tracer,
            )
            return

        decomposed = DecomposedGroupBy.from_ir(ir)
        assert not decomposed.need_preshuffle, "Should already be shuffled."
        input_row_order_keys, requested_order_keys = _ordered_groupby_keys(
            ir,
            decomposed,
            partitioning_requests,
            metadata_in.partitioning,
            group_keys,
        )
        if not metadata_in.duplicated and requested_order_keys:
            await _ordered_reduce(
                context,
                comm,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                collective_ids,
                input_row_order_keys,
                requested_order_keys,
                distributed=nranks > 1 and not rank_ordered,
                tracer=tracer,
            )
            return

        aggregated, input_drained, chunks_received = await _local_aggregation(
            context,
            decomposed,
            ir_context,
            ch_in,
            target_partition_size,
            allow_early_exit=not maintain_order,
        )

        skip_global_comm = metadata_in.duplicated or isinstance(
            partitioning.inter_rank_scheme, HashScheme
        )
        output_count = await _choose_strategy(
            context,
            comm,
            metadata_in.local_count,
            aggregated,
            chunks_received,
            input_drained,
            collective_ids,
            target_partition_size,
            skip_global_comm,
            maintain_order,
            tracer,
        )

        if output_count == 1:
            await _tree_reduce(
                context,
                comm,
                decomposed,
                ir_context,
                ch_out,
                metadata_in,
                collective_ids.pop(),
                local=skip_global_comm,
                aggregated=aggregated,
                tracer=tracer,
            )
        else:
            await _shuffle_reduce(
                context,
                comm,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                output_count,
                collective_ids.pop(),
                target_partition_size,
                local=skip_global_comm,
                aggregated=aggregated,
                input_drained=input_drained,
                tracer=tracer,
            )


@generate_ir_sub_network.register(GroupBy)
@generate_ir_sub_network.register(Distinct)
def _(
    ir: GroupBy | Distinct, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for GroupBy or Distinct operation."""
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming"

    if config_options.executor.dynamic_planning is None:
        # Fall back to the default IR handler (bypass GroupBy/Distinct dispatch)
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))
    ir_context = ir_context_for_node(rec, ir)
    assert len(collective_ids) == 2, (
        f"{type(ir).__name__} requires 2 collective IDs, got {len(collective_ids)}"
    )
    actors[ir] = [
        groupby_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            ir_context,
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            config_options.executor.target_partition_size,
            collective_ids,
            rec.state["partitioning_requests"].get(ir, ()),
        )
    ]

    return actors, channels
