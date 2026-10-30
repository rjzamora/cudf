# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import dataclass, replace
from functools import partial
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal, assert_never

from cudf_streaming import CardinalityEstimator
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
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.memory_reserve_or_wait import (
    missing_net_memory_delta,
    reserve_memory,
)
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Join, Projection
from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.allgather import (
    AllGatherManager,
)
from cudf_polars.streaming.actor_graph.collectives.ordered import (
    buffer_ordered_input,
    copy_key_range,
    exchange_buffered_key_input,
    gather_rank_endpoint_sets,
    key_bounds_view,
    key_row_less,
    plan_key_bounds_exchange,
    rank_key_bounds,
    rank_ownership_intervals,
    repaired_inter_rank_ordering,
    select_demand_range,
    strict_inter_rank_ordering,
)
from cudf_polars.streaming.actor_graph.collectives.shuffle import (
    _global_shuffle,
    _key_column_indices,
)
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.join_planning import JoinPlanningState
from cudf_polars.streaming.actor_graph.nodes import default_node_multi
from cudf_polars.streaming.actor_graph.partitioning import (
    NormalizedPartitioning,
    is_globally_row_ordered_by,
    is_row_ordered_by,
    maybe_remap_partitioning,
    remap_ordering_keys,
)
from cudf_polars.streaming.actor_graph.prefilter import (
    JoinPrefilterExecution,
    RuntimePrefilterStatistics,
    add_bloom_prefilter,
    choose_prefilter,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    CUDF_ROW_LIMIT,
    MAX_ROWS_PER_PARTITION,
    ChannelManager,
    ChunkSampler,
    ChunkStore,
    TableSample,
    TableSizeStats,
    allgather_reduce,
    chunk_to_frame,
    concat_batch,
    empty_table_chunk,
    gather_in_task_group,
    process_children,
    recv_metadata,
    replay_buffered_channel,
    sample_inputs,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.filter_hint import (
    ExternalDomain,
    JoinInputDomain,
    JoinWithPrefilter,
)
from cudf_polars.streaming.join import supports_ordered_cursor_join
from cudf_polars.streaming.partitioning_requests import OrderPartitioningRequest
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.utils import _concat, partition_range

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, MutableMapping, Sequence

    from cudf_streaming.channel_metadata import Ordering
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.collectives.ordered import (
        RankEndpoints,
    )
    from cudf_polars.streaming.actor_graph.collectives.slice_exchange import (
        PreparedExchange,
    )
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.join_planning import JoinInput
    from cudf_polars.streaming.actor_graph.prefilter import (
        PrefilterDecision,
        PrefilterExecution,
    )
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer
    from cudf_polars.streaming.base import PartitionInfo
    from cudf_polars.streaming.filter_hint import (
        JoinSide,
        Prefilter,
        PushdownFilterHint,
    )
    from cudf_polars.streaming.partitioning_requests import PartitioningRequest
    from cudf_polars.utils.config import StreamingExecutor


def _join_preserves_side_order(
    how: Literal["Inner", "Left", "Right", "Full", "Semi", "Anti", "Cross"],
    maintain_order: Literal["none", "left", "right", "left_right", "right_left"],
    side: Literal["left", "right"],
) -> bool:
    """Return whether a join preserves one input's row order."""
    return maintain_order.startswith(side) and (
        how == "Inner"
        or (side == "left" and how in ("Left", "Cross"))
        or (side == "right" and how == "Right")
    )


def _preserved_join_side(ir: Join) -> Literal["left", "right"] | None:
    """Return the input side whose row order the join preserves."""
    how, maintain_order = ir.options[0], ir.options[5]
    for side in ("left", "right"):
        if _join_preserves_side_order(how, maintain_order, side):
            return side
    return None


def _remap_join_partitioning(
    ir: Join,
    metadata: ChannelMetadata,
    child_index: int,
    context: Context,
) -> Partitioning | None:
    """Remap one join input's metadata to the output schema."""
    side: Literal["left", "right"] = "left" if child_index == 0 else "right"
    return maybe_remap_partitioning(
        ir,
        metadata.partitioning,
        child_index=child_index,
        context=context,
        preserves_row_order=_preserved_join_side(ir) == side,
    )


@dataclass(frozen=True)
class BroadcastJoinStrategy:
    """Broadcast one side to all ranks."""

    side: Literal["left", "right"]


@dataclass(frozen=True)
class ShuffleJoinStrategy:
    """Hash-shuffle both sides before joining."""

    shuffle_modulus: int = 0
    """The shuffle modulus."""
    output_indices: tuple[int, ...] = ()
    """Output key-column indices for output metadata."""
    left_indices: tuple[int, ...] = ()
    """Left input key-column indices."""
    right_indices: tuple[int, ...] = ()
    """Right input key-column indices."""
    left_keys: tuple[NamedExpr, ...] = ()
    """Left key expressions."""
    right_keys: tuple[NamedExpr, ...] = ()
    """Right key expressions."""


@dataclass(frozen=True)
class OrderedJoinStrategy:
    """Join inputs that already share strict ordering boundaries."""

    left_input_ordering: Ordering
    """Input ordering for the left side."""
    right_input_ordering: Ordering
    """Input ordering for the right side."""
    output_ordering: Ordering
    """Join-output ordering metadata."""


@dataclass(frozen=True)
class IncrementalOrderedJoinStrategy:
    """Incrementally join globally row-ordered inputs."""

    output_order_keys: tuple[OrderKey, ...]
    left_order_keys: tuple[OrderKey, ...]
    right_order_keys: tuple[OrderKey, ...]
    inter_rank_ordering: Ordering | None = None
    """Existing strict rank ownership, when no exchange is required."""


type JoinStrategy = (
    BroadcastJoinStrategy
    | ShuffleJoinStrategy
    | OrderedJoinStrategy
    | IncrementalOrderedJoinStrategy
)


@dataclass(frozen=True, slots=True)
class JoinCollectiveIds:
    """Named collective-ID slots reserved for a dynamic join."""

    size_estimate: int
    left_redistribution: int
    right_redistribution: int

    @classmethod
    def from_reserved(cls, collective_ids: list[int]) -> JoinCollectiveIds:
        """Construct the named slots from IDs reserved for a dynamic join."""
        if len(collective_ids) < 3:
            raise ValueError(
                "Dynamic join requires 3 reserved collective IDs "
                "(allgather + left shuffle + right shuffle); got "
                f"{len(collective_ids)} for this Join. "
                "Ensure ReserveOpIDs is run with dynamic_planning enabled."
            )
        return cls(*collective_ids[:3])

    @property
    def cardinality_tags(self) -> tuple[int, int]:
        """Tags available for concurrent prefilter cardinality estimates."""
        return (self.size_estimate, self.left_redistribution)

    @property
    def broadcast(self) -> int:
        """ID used by a broadcast join after size estimation completes."""
        return self.left_redistribution

    def shuffle(self, side: JoinSide) -> int:
        """Return the collective ID for one shuffle input."""
        if side == "left":
            return self.left_redistribution
        return self.right_redistribution

    def prefilter(self, strategy: JoinStrategy, target_side: JoinSide) -> int:
        """Return the subsequent join collective reused by a prefilter."""
        if isinstance(strategy, BroadcastJoinStrategy):
            if target_side != strategy.side:
                raise ValueError(
                    "Only the broadcast input can have an active prefilter"
                )
            return self.broadcast
        return self.shuffle(target_side)


@define_actor()
async def broadcast_join_actor(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    broadcast_side: Literal["left", "right"],
    collective_id: int,
    target_partition_size: int,
) -> None:
    """
    Broadcast-join actor for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The Join IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output Channel[TableChunk].
    ch_left
        The left input Channel[TableChunk].
    ch_right
        The right input Channel[TableChunk].
    broadcast_side
        The side to broadcast.
    collective_id
        Pre-allocated collective ID for this operation.
    target_partition_size
        The target partition size in bytes.
    """
    async with shutdown_on_error(
        context,
        chs_in=(ch_left, ch_right),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        tracer = actor_scope.tracer
        ir_context = actor_scope.require_ir_context()
        await broadcast_join(
            context,
            comm,
            ir,
            ir_context,
            ch_out,
            ch_left,
            ch_right,
            BroadcastJoinStrategy(side=broadcast_side),
            collective_id,
            target_partition_size,
            tracer=tracer,
        )


async def _collect_small_side_for_broadcast(
    context: Context,
    comm: Communicator,
    ch: Channel[TableChunk],
    ir: IR,
    *,
    need_allgather: bool,
    collective_id: int,
    ir_context: IRExecutionContext,
    must_concatenate: bool,
) -> tuple[list[DataFrame], int]:
    """
    Drain small-side channel into chunks, then build DataFrame(s) for broadcast.

    Returns (list of DataFrames to join against, total byte size of small side).
    """
    size = 0
    chunks: list[TableChunk] = []
    while (msg := await ch.recv(context)) is not None:
        chunks.append(TableChunk.from_message(msg, br=context.br()))
        size += chunks[-1].data_alloc_size()
    row_count = sum(c.shape[0] for c in chunks)

    dfs: list[DataFrame] = []
    if need_allgather:
        allgather = AllGatherManager(context, comm, collective_id)
        with allgather.inserting() as inserter:
            for s_id in range(len(chunks)):
                await inserter.insert(s_id, chunks.pop(0))
        stream = ir_context.get_cuda_stream()
        # TODO: if we can't concatenate and we needn't then this spuriously fails.
        gathered = await allgather.extract_concatenated(stream, ir_context=ir_context)
        # When every rank inserted zero chunks, the AllGather has no schema
        # to infer and returns a 0 column table. Substitute a properly typed
        # empty table for the small side so downstream joins still match the
        # expected schema.
        table = (
            empty_table_chunk(ir, context, stream).table_view()
            if gathered.num_columns() == 0 and len(ir.schema) > 0
            else gathered
        )
        dfs = [
            DataFrame.from_table(
                table,
                list(ir.schema.keys()),
                list(ir.schema.values()),
                stream,
            )
        ]
    elif chunks:
        can_concatenate = row_count <= CUDF_ROW_LIMIT
        if must_concatenate and not can_concatenate:
            raise RuntimeError(
                "Broadcast join selected but broadcast side cannot be constructed"
            )
        if can_concatenate:
            chunks, extra = await make_table_chunks_available_or_wait(
                context,
                chunks,
                reserve_extra=0 if len(chunks) == 1 else size,
                net_memory_delta=0,
            )
            with opaque_memory_usage(extra):
                dfs = [
                    _concat(
                        *[chunk_to_frame(chunk, ir) for chunk in chunks],
                        context=ir_context,
                    )
                ]
        else:
            # Group greedily ensuring that only single-chunk groups can be
            # above MAX_ROWS_PER_PARTITION.
            groups = []
            group: list[TableChunk] = []
            rows = 0
            for chunk in chunks:
                chunk_rows = chunk.shape[0]
                if group and chunk_rows + rows > MAX_ROWS_PER_PARTITION:
                    groups.append(group)
                    rows = 0
                    group = []
                group.append(chunk)
                rows += chunk_rows
            if group:
                groups.append(group)
            del chunks
            for group in groups:
                group_size = sum(chunk.data_alloc_size() for chunk in group)
                group, extra = await make_table_chunks_available_or_wait(  # noqa: PLW2901
                    context,
                    group,
                    reserve_extra=0 if len(group) == 1 else group_size,
                    net_memory_delta=0,
                )
                with opaque_memory_usage(extra):
                    dfs.append(
                        _concat(
                            *[chunk_to_frame(chunk, ir) for chunk in group],
                            context=ir_context,
                        )
                    )

    return dfs, size


async def _broadcast_join_large_chunk(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    small_dfs: list[DataFrame],
    small_child: IR,
    large_chunk: TableChunk,
    large_child: IR,
    seq_num: int,
    small_size: int,
    broadcast_side: Literal["left", "right"],
    *,
    tracer: ActorTracer | None,
) -> int:
    """Join one large-side chunk with the small DataFrame(s) and send the result."""
    large_df = chunk_to_frame(large_chunk, large_child)
    large_chunk_size = large_chunk.data_alloc_size()

    dfs_to_join = small_dfs
    if not dfs_to_join:
        stream = ir_context.get_cuda_stream()
        empty_small = empty_table_chunk(small_child, context, stream)
        dfs_to_join = [chunk_to_frame(empty_small, small_child)]

    join_results: list[DataFrame] = []
    input_bytes = large_chunk_size + small_size
    with opaque_memory_usage(
        await reserve_memory(context, size=input_bytes, net_memory_delta=0)
    ):
        for sdf in dfs_to_join:
            result = await ir_context.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                *([large_df, sdf] if broadcast_side == "right" else [sdf, large_df]),
                context=ir_context,
            )
            join_results.append(result)

        df = _concat(*join_results, context=ir_context)
        del join_results

    output_chunk = TableChunk.from_pylibcudf_table(
        df.table, df.stream, exclusive_view=True, br=context.br()
    )
    output_rows = output_chunk.shape[0]
    await send_chunk(context, ch_out, output_chunk, seq_num, tracer=tracer)
    del df, large_df
    return output_rows


async def broadcast_join(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    strategy: BroadcastJoinStrategy,
    collective_id: int,
    target_partition_size: int | None,
    *,
    tracer: ActorTracer | None,
    prefilter_statistics: RuntimePrefilterStatistics | None = None,
) -> None:
    """
    Execute a broadcast join after initial sampling.

    The small side is gathered (if not already duplicated) and concatenated
    into a single DataFrame, then joined with each chunk from the large side.
    Uses ``collective_id`` for the allgather when needed.
    """
    left_metadata, right_metadata = await gather_in_task_group(
        recv_metadata(ch_left, context),
        recv_metadata(ch_right, context),
    )

    broadcast_side = strategy.side
    left, right = ir.children[:2]
    if tracer is not None:
        tracer.decision = f"broadcast_{broadcast_side}"

    if broadcast_side == "right":
        small_ch, large_ch = ch_right, ch_left
        small_child, large_child = right, left
        small_metadata, large_metadata = right_metadata, left_metadata
        local_count = left_metadata.local_count
        partitioning = _remap_join_partitioning(ir, left_metadata, 0, context)
    else:
        small_ch, large_ch = ch_left, ch_right
        small_child, large_child = left, right
        small_metadata, large_metadata = left_metadata, right_metadata
        local_count = right_metadata.local_count
        partitioning = _remap_join_partitioning(ir, right_metadata, 1, context)

    small_duplicated = small_metadata.duplicated
    need_allgather = comm.nranks > 1 and not small_duplicated
    can_join_build_chunks_separately = _can_join_build_chunks_separately(
        comm, ir, small_metadata
    )
    output_duplicated = (
        small_duplicated or need_allgather
    ) and large_metadata.duplicated

    metadata_out = ChannelMetadata(
        local_count=local_count,
        partitioning=partitioning,
        duplicated=output_duplicated,
    )
    small_dfs, small_size = await _collect_small_side_for_broadcast(
        context,
        comm,
        small_ch,
        small_child,
        need_allgather=need_allgather,
        collective_id=collective_id,
        ir_context=ir_context,
        must_concatenate=not can_join_build_chunks_separately,
    )

    # Publish output metadata only once the broadcast-side collective has
    # completed. Besides making the data channel ready when advertised, this
    # permits a consumer to reuse the collective ID after receiving metadata.
    await send_metadata(ch_out, context, metadata_out)

    input_rows = 0
    output_rows = 0
    while (msg := await large_ch.recv(context)) is not None:
        # Unknown: the large chunk is freed but the join output replaces
        # it, and its size depends on selectivity we cannot estimate here.
        large_chunk, _ = await make_table_chunks_available_or_wait(
            context,
            TableChunk.from_message(msg, br=context.br()),
            reserve_extra=0,
            net_memory_delta=missing_net_memory_delta,
        )
        input_rows += large_chunk.shape[0]
        output_rows += await _broadcast_join_large_chunk(
            context,
            ir,
            ir_context,
            ch_out,
            small_dfs,
            small_child,
            large_chunk,
            large_child,
            msg.sequence_number,
            small_size,
            broadcast_side,
            tracer=tracer,
        )

    if prefilter_statistics is not None:
        prefilter_statistics.input_rows = input_rows
        prefilter_statistics.output_rows = output_rows
    await ch_out.drain(context)


def add_prefilter(
    execution: PrefilterExecution,
    comm: Communicator,
    *,
    spec: Prefilter | PushdownFilterHint,
    decision: PrefilterDecision,
    target: IR,
    domain: IR,
    ch_target: Channel[TableChunk],
    ch_domain_keys: Channel[TableChunk],
    ch_filtered: Channel[TableChunk],
    collective_id: int,
    ir_context: IRExecutionContext,
    prefilter_statistics: RuntimePrefilterStatistics | None,
) -> None:
    """Add the actors and channels that apply one selected prefilter."""
    context = execution.context
    if decision.method == "bloom":
        if decision.bloom_bytes is None:
            raise ValueError("Bloom prefilter decision has no filter size")
        add_bloom_prefilter(
            context,
            comm,
            decision.bloom_bytes,
            execution,
            names_to_indices(spec.target_on, target.schema),
            ch_domain_keys,
            ch_target,
            ch_filtered,
            collective_id,
            prefilter_statistics,
        )
    elif decision.method == "broadcast_semi_join":
        domain_schema = {key.name: key.value.dtype for key in spec.domain_on}
        if len(domain_schema) != len(spec.domain_on):
            raise ValueError("Broadcast semi-join keys must have unique names")
        semi_join = Join(
            target.schema,
            spec.target_on,
            spec.domain_on,
            ("Semi", spec.nulls_equal, None, "", False, "none"),
            target,
            Projection(domain_schema, domain),
        )
        execution.add_task(
            broadcast_join(
                context,
                comm,
                semi_join,
                ir_context,
                ch_filtered,
                ch_target,
                ch_domain_keys,
                BroadcastJoinStrategy(side="right"),
                collective_id,
                target_partition_size=None,
                tracer=None,
                prefilter_statistics=prefilter_statistics,
            )
        )
    else:
        raise ValueError(f"Cannot apply prefilter method {decision.method!r}")


def make_prefilter_execution(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    strategy: JoinStrategy,
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    join_state: JoinPlanningState,
    collective_ids: JoinCollectiveIds,
) -> JoinPrefilterExecution:
    """Create the actors and channels that realize selected prefilters."""
    execution = JoinPrefilterExecution(context, ch_left, ch_right)

    # Prepare every required domain before connecting target-side filters. This
    # is important for opposing direct filters: each filter must consume the
    # replay produced while the same input's keys are copied for the other one.
    for candidate in join_state.candidates:
        decision = candidate.decision
        if decision is None:
            raise ValueError("Join prefilter has no runtime decision")
        spec = candidate.spec
        if decision.method == "skip":
            continue

        if isinstance(spec.domain, JoinInputDomain):
            indices = names_to_indices(spec.domain_on, candidate.domain.node.schema)
            candidate.key_channel = execution.buffer_domain(spec.domain.side, indices)
        else:
            sample = candidate.domain.sample
            if sample is None:
                raise ValueError("Active external prefilter has no domain sample")
            indices = names_to_indices(spec.domain_on, candidate.domain.node.schema)
            if indices != tuple(range(len(candidate.domain.node.schema))):
                raise ValueError("External prefilter domains must contain only keys")
            candidate.key_channel = context.create_channel()
            execution.add_channel(candidate.key_channel)
            execution.add_task(
                replay_buffered_channel(
                    context,
                    candidate.key_channel,
                    candidate.domain.channel,
                    sample.local_sample.chunks,
                    candidate.domain.metadata,
                    trace_ir=ir,
                )
            )

    for candidate in join_state.candidates:
        decision = candidate.decision
        assert decision is not None
        if decision.method == "skip":
            continue
        spec = candidate.spec
        ch_domain_keys = candidate.key_channel
        assert ch_domain_keys is not None
        target_side = spec.target_side
        target = candidate.target.node
        ch_target = execution.join_inputs[target_side]
        ch_filtered: Channel[TableChunk] = context.create_channel()
        prefilter_statistics = candidate.statistics

        add_prefilter(
            execution,
            comm,
            spec=spec,
            decision=decision,
            target=target,
            domain=candidate.domain.node,
            ch_target=ch_target,
            ch_domain_keys=ch_domain_keys,
            ch_filtered=ch_filtered,
            collective_id=collective_ids.prefilter(strategy, target_side),
            ir_context=ir_context,
            prefilter_statistics=prefilter_statistics,
        )
        execution.replace_join_input(target_side, ch_filtered)

    return execution


def _get_key_indices(
    ir: Join,
    n_partitioned_keys: int | None,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[NamedExpr, ...],
    tuple[NamedExpr, ...],
]:
    left, right = ir.children[:2]
    n_keys = n_partitioned_keys if n_partitioned_keys is not None else len(ir.left_on)
    left_keys = ir.left_on[:n_keys]
    right_keys = ir.right_on[:n_keys]
    left_key_indices = _key_column_indices(left_keys, left.schema) or ()
    right_key_indices = _key_column_indices(right_keys, right.schema) or ()
    if ir.options[0] == "Right":
        output_keys = right_keys
    else:
        output_keys = left_keys
    output_key_indices = (
        _key_column_indices(output_keys, ir.schema)
        if left_key_indices and right_key_indices
        else None
    )
    return (
        left_key_indices,
        right_key_indices,
        output_key_indices or (),
        left_keys,
        right_keys,
    )


def _requested_join_order_keys(
    ir: Join,
    requests: tuple[PartitioningRequest, ...],
) -> tuple[OrderKey, ...]:
    """Return a downstream ordering request that exactly matches the join keys."""
    _, _, output_indices, _, _ = _get_key_indices(ir, None)
    columns = tuple(ir.schema)
    output_names = tuple(columns[index] for index in output_indices)
    for request in requests:
        if not isinstance(request, OrderPartitioningRequest):
            continue
        if tuple(key.name for key in request.keys) != output_names:
            continue
        return tuple(
            OrderKey(index, key.order, key.null_order)
            for index, key in zip(output_indices, request.keys, strict=True)
        )
    return ()


def _ordering_prefix_matches(
    ordering: Ordering,
    reference: Ordering,
    column_indices: tuple[int, ...],
) -> bool:
    """True when ordering has the same leading order semantics as reference."""
    if len(ordering.keys) < len(column_indices):
        return False
    if len(reference.keys) < len(column_indices):
        return False
    expected = tuple(
        OrderKey(index, key.order, key.null_order)
        for index, key in zip(
            column_indices, reference.keys[: len(column_indices)], strict=True
        )
    )
    return ordering.keys[: len(expected)] == expected


def _make_ordered_strategy(
    ir: Join,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
) -> OrderedJoinStrategy | None:
    """Make a strategy for inputs with compatible strict boundaries."""
    if ir.options[0] not in ("Inner", "Left", "Semi", "Anti"):
        return None

    left_scheme = left_partitioning.inter_rank_scheme
    right_scheme = right_partitioning.inter_rank_scheme
    if not isinstance(left_scheme, OrderScheme) or not isinstance(
        right_scheme, OrderScheme
    ):
        return None

    left_ordering = left_scheme.orderings[0]
    right_ordering = right_scheme.orderings[0]
    if not (left_ordering.strict_boundaries and right_ordering.strict_boundaries):
        return None

    reference = left_ordering
    reference_key_count = len(reference.keys)
    (
        left_key_indices,
        right_key_indices,
        output_key_indices,
        _,
        _,
    ) = _get_key_indices(ir, reference_key_count)
    if not (
        len(left_key_indices)
        == len(right_key_indices)
        == len(output_key_indices)
        == reference_key_count
    ):
        return None
    if not _ordering_prefix_matches(left_ordering, reference, left_key_indices):
        return None
    if not _ordering_prefix_matches(right_ordering, reference, right_key_indices):
        return None

    output_ordering = remap_ordering_keys(reference, output_key_indices)
    return OrderedJoinStrategy(
        left_input_ordering=left_ordering,
        right_input_ordering=right_ordering,
        output_ordering=output_ordering,
    )


def _make_incremental_ordered_strategy(
    ir: Join,
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    requested_order_keys: tuple[OrderKey, ...],
) -> IncrementalOrderedJoinStrategy | None:
    """Return a cursor strategy for globally row-ordered join keys."""
    if not supports_ordered_cursor_join(ir):
        return None
    (
        left_indices,
        right_indices,
        output_indices,
        _,
        _,
    ) = _get_key_indices(ir, None)
    if not (
        left_indices and len(left_indices) == len(right_indices) == len(output_indices)
    ):
        return None

    left_partitioning = left_metadata.partitioning
    right_partitioning = right_metadata.partitioning
    if left_partitioning is None or right_partitioning is None:
        return None
    for row_ordering in left_partitioning.row_orderings:
        if row_ordering.group_keys:
            continue
        left_keys = row_ordering.order_keys[: len(left_indices)]
        if tuple(key.column_index for key in left_keys) != left_indices:
            continue
        if not is_globally_row_ordered_by(left_partitioning, left_keys):
            continue
        right_keys = tuple(
            OrderKey(index, key.order, key.null_order)
            for index, key in zip(right_indices, left_keys, strict=True)
        )
        if not is_globally_row_ordered_by(right_partitioning, right_keys):
            continue
        output_keys = tuple(
            OrderKey(index, key.order, key.null_order)
            for index, key in zip(output_indices, left_keys, strict=True)
        )
        if output_keys != requested_order_keys:
            continue
        return IncrementalOrderedJoinStrategy(
            output_order_keys=output_keys,
            left_order_keys=left_keys,
            right_order_keys=right_keys,
        )
    return None


async def _join_chunks(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    tracer: ActorTracer | None,
) -> None:
    # Consume metadata from both shuffle outputs before reading data
    await gather_in_task_group(
        recv_metadata(ch_left, context),
        recv_metadata(ch_right, context),
    )

    left, right = ir.children[:2]
    while True:
        left_msg, right_msg = await gather_in_task_group(
            ch_left.recv(context), ch_right.recv(context)
        )
        if left_msg is None or right_msg is None:
            assert left_msg is None, (
                "Mismatched chunk counts in shuffle join: left has unmatched chunk. "
                f"Seq num: {left_msg.sequence_number}"
            )
            assert right_msg is None, (
                "Mismatched chunk counts in shuffle join: right has unmatched chunk. "
                f"Seq num: {right_msg.sequence_number}"
            )
            break
        assert left_msg.sequence_number == right_msg.sequence_number, (
            "Mismatched chunk sequence numbers in shuffle join. "
            f"Left: {left_msg.sequence_number}, Right: {right_msg.sequence_number}"
        )

        # Unknown: both chunks are freed but the join output replaces them,
        # and its size depends on selectivity we cannot estimate here.
        (left_chunk, right_chunk), _ = await make_table_chunks_available_or_wait(
            context,
            [
                TableChunk.from_message(left_msg, br=context.br()),
                TableChunk.from_message(right_msg, br=context.br()),
            ],
            reserve_extra=0,
            net_memory_delta=missing_net_memory_delta,
        )

        input_bytes = sum(
            col.device_buffer_size()
            for col in (
                *left_chunk.table_view().columns(),
                *right_chunk.table_view().columns(),
            )
        )
        with opaque_memory_usage(
            await reserve_memory(context, size=input_bytes, net_memory_delta=0)
        ):
            df = await ir_context.to_thread(
                partial(
                    ir.do_evaluate,
                    *ir._non_child_args,
                    chunk_to_frame(left_chunk, left),
                    chunk_to_frame(right_chunk, right),
                    context=ir_context,
                )
            )
            del left_chunk, right_chunk

        output_chunk = TableChunk.from_pylibcudf_table(
            df.table, df.stream, exclusive_view=True, br=context.br()
        )
        await send_chunk(
            context,
            ch_out,
            output_chunk,
            left_msg.sequence_number,
            tracer=tracer,
        )
        del df

    await ch_out.drain(context)


def _shuffle_strategy_decision(
    strategy: ShuffleJoinStrategy,
    partitioning_left: NormalizedPartitioning,
    partitioning_right: NormalizedPartitioning,
) -> Literal["chunkwise", "shuffle_left", "shuffle_right", "shuffle"]:
    left_scheme_desired = HashScheme(strategy.left_indices, strategy.shuffle_modulus)
    right_scheme_desired = HashScheme(strategy.right_indices, strategy.shuffle_modulus)
    left_partitioned = (
        partitioning_left.inter_rank_scheme == left_scheme_desired
        and partitioning_left.local_scheme == "inherit"
    )
    right_partitioned = (
        partitioning_right.inter_rank_scheme == right_scheme_desired
        and partitioning_right.local_scheme == "inherit"
    )
    if left_partitioned and right_partitioned:
        return "chunkwise"
    elif left_partitioned:
        return "shuffle_right"
    elif right_partitioned:
        return "shuffle_left"
    else:
        return "shuffle"


async def _shuffle_join(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    strategy: ShuffleJoinStrategy,
    left_collective_id: int,
    right_collective_id: int,
    *,
    tracer: ActorTracer | None,
) -> None:
    """Execute a shuffle (hash) join."""
    # Send output metadata
    shuffle_modulus = strategy.shuffle_modulus
    output_indices = strategy.output_indices
    nranks = comm.nranks
    metadata_out = ChannelMetadata(
        local_count=max(1, shuffle_modulus // nranks),
        partitioning=Partitioning(
            HashScheme(column_indices=output_indices, modulus=shuffle_modulus),
            local="inherit",
        )
        if output_indices
        else None,
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)
    # Construct a shuffle-shuffle-join pipeline.
    # The shuffle operations will pass chunks through unchanged
    # if the data is already partitioned correctly.
    ch_left_shuffle = context.create_channel()
    ch_right_shuffle = context.create_channel()
    # note: this is an actor inside of an actor. How should we log that in our traces?
    async with shutdown_on_error(
        context,
        chs_aux=(ch_left_shuffle, ch_right_shuffle),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        ir_context = actor_scope.require_ir_context()
        actor_tasks = [
            _global_shuffle(
                context,
                comm,
                ir_context,
                ch_left_shuffle,
                ch_left,
                strategy.left_keys,
                ir.children[0].schema,
                strategy.shuffle_modulus,
                left_collective_id,
            ),
            _global_shuffle(
                context,
                comm,
                ir_context,
                ch_right_shuffle,
                ch_right,
                strategy.right_keys,
                ir.children[1].schema,
                strategy.shuffle_modulus,
                right_collective_id,
            ),
            _join_chunks(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left_shuffle,
                ch_right_shuffle,
                tracer=tracer,
            ),
        ]
        await gather_in_task_group(*actor_tasks)


def _local_count_for_ordering(comm: Communicator, ordering: Ordering) -> int:
    """Return this rank's local partition count for a contiguous Ordering."""
    npartitions = ordering.num_boundaries + 1
    start, stop = partition_range(comm.rank, comm.nranks, npartitions)
    return stop - start


def _ordered_join_is_aligned(
    context: Context,
    comm: Communicator,
    strategy: OrderedJoinStrategy,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
) -> bool:
    """Return whether strict ordered inputs already share local partitions."""
    return (
        left_partitioning.local_scheme == "inherit"
        and right_partitioning.local_scheme == "inherit"
        and left_partitioning.is_aligned_with(right_partitioning, context.br())
        and left_metadata.local_count
        == _local_count_for_ordering(comm, strategy.left_input_ordering)
        and right_metadata.local_count
        == _local_count_for_ordering(comm, strategy.right_input_ordering)
    )


def _ordered_join_execution(
    context: Context,
    comm: Communicator,
    ir: Join,
    strategy: OrderedJoinStrategy,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
) -> Literal["aligned", "cursor"] | None:
    """Choose local execution for inputs with aligned strict rank ownership."""
    if not left_partitioning.is_aligned_with(
        right_partitioning,
        context.br(),
        level="inter_rank",
    ):
        return None
    if _ordered_join_is_aligned(
        context,
        comm,
        strategy,
        left_partitioning,
        right_partitioning,
        left_metadata,
        right_metadata,
    ):
        return "aligned"
    if not supports_ordered_cursor_join(ir):
        return None
    if is_row_ordered_by(
        left_metadata.partitioning,
        strategy.left_input_ordering.keys,
        scope=RowOrderingScope.RANK_LOCAL,
    ) and is_row_ordered_by(
        right_metadata.partitioning,
        strategy.right_input_ordering.keys,
        scope=RowOrderingScope.RANK_LOCAL,
    ):
        return "cursor"
    return None


async def _ordered_join(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    strategy: OrderedJoinStrategy,
    *,
    tracer: ActorTracer | None,
) -> None:
    """Join inputs that already share strict ordering boundaries."""
    metadata_out = ChannelMetadata(
        local_count=_local_count_for_ordering(comm, strategy.output_ordering),
        partitioning=Partitioning(
            OrderScheme([strategy.output_ordering]),
            local="inherit",
        ),
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)
    if tracer is not None:
        tracer.decision = "ordered_aligned"
    await _join_chunks(
        context,
        ir,
        ir_context,
        ch_out,
        ch_left,
        ch_right,
        tracer=tracer,
    )


async def _evaluate_incremental_join(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    left_chunk: TableChunk,
    right_chunk: TableChunk,
    *,
    right_available: bool = False,
) -> TableChunk:
    """Join one left key range while preserving its row order."""
    if right_available:
        if not right_chunk.is_available():
            raise ValueError("right_available=True requires an available TableChunk")
        left_chunk, _ = await make_table_chunks_available_or_wait(
            context,
            left_chunk,
            reserve_extra=0,
            net_memory_delta=missing_net_memory_delta,
        )
    else:
        (left_chunk, right_chunk), _ = await make_table_chunks_available_or_wait(
            context,
            [left_chunk, right_chunk],
            reserve_extra=0,
            net_memory_delta=missing_net_memory_delta,
        )
    input_bytes = left_chunk.data_alloc_size() + right_chunk.data_alloc_size()
    options = (*ir.options[:5], "left")
    with opaque_memory_usage(
        await reserve_memory(context, size=input_bytes, net_memory_delta=0)
    ):
        df = await ir_context.to_thread(
            partial(
                ir.do_evaluate,
                ir.left_on,
                ir.right_on,
                options,
                chunk_to_frame(left_chunk, ir.children[0]),
                chunk_to_frame(right_chunk, ir.children[1]),
                context=ir_context,
            )
        )
    return TableChunk.from_pylibcudf_table(
        df.table,
        df.stream,
        exclusive_view=True,
        br=context.br(),
    )


async def _incremental_ordered_join(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    strategy: IncrementalOrderedJoinStrategy,
    *,
    tracer: ActorTracer | None,
) -> None:
    """Incrementally join two globally row-ordered channel streams."""
    left_metadata, _ = await gather_in_task_group(
        recv_metadata(ch_left, context),
        recv_metadata(ch_right, context),
    )
    await send_metadata(
        ch_out,
        context,
        _incremental_join_metadata(
            left_metadata.local_count,
            strategy,
        ),
    )
    if tracer is not None:
        tracer.decision = "ordered_incremental"

    await _incremental_ordered_join_chunks(
        context,
        ir,
        ir_context,
        ch_out,
        _channel_chunks(context, ch_left),
        _channel_chunks(context, ch_right),
        strategy,
        tracer=tracer,
    )


def _incremental_join_metadata(
    local_count: int,
    strategy: IncrementalOrderedJoinStrategy,
) -> ChannelMetadata:
    """Return output metadata for an incremental ordered join."""
    return ChannelMetadata(
        local_count=local_count,
        partitioning=Partitioning(
            inter_rank=(
                OrderScheme([strategy.inter_rank_ordering])
                if strategy.inter_rank_ordering is not None
                else None
            ),
            row_orderings=(
                RowOrdering(strategy.output_order_keys, RowOrderingScope.GLOBAL),
            ),
        ),
        duplicated=False,
    )


async def _channel_chunks(
    context: Context,
    channel: Channel[TableChunk],
) -> AsyncGenerator[TableChunk]:
    """Yield table chunks from a channel."""
    while (message := await channel.recv(context)) is not None:
        yield TableChunk.from_message(message, br=context.br())


async def _stored_chunks(
    context: Context,
    chunks: ChunkStore,
) -> AsyncGenerator[TableChunk]:
    """Yield table chunks from spillable storage and release the store."""
    try:
        for message in chunks:
            yield TableChunk.from_message(message, br=context.br())
    finally:
        chunks.clear()


async def _prepared_chunks(
    context: Context,
    prepared: PreparedExchange,
    rank: int,
) -> AsyncGenerator[TableChunk]:
    """Yield exchanged table chunks in source-rank order."""
    try:
        async for message in prepared.iter_source_order(context, rank):
            yield TableChunk.from_message(message, br=context.br())
    finally:
        prepared.clear()


async def _incremental_ordered_join_chunks(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    left_chunks: AsyncIterator[TableChunk],
    right_chunks: AsyncIterator[TableChunk],
    strategy: IncrementalOrderedJoinStrategy,
    *,
    tracer: ActorTracer | None,
) -> None:
    """
    Incrementally join two globally row-ordered chunk streams.

    Candidate right chunks remain spillable while they accumulate. The rows
    matching one left chunk must fit on the device for concatenation and the
    local join. Avoiding that bound requires a more granular merge cursor.
    """
    right_replay = ChunkStore(context)
    right_pending = ChunkStore(context)
    right_iterator = aiter(right_chunks)
    sequence_number = 0
    try:

        async def evaluate_left_chunk(
            left: TableChunk,
            demand: TableChunk,
        ) -> None:
            """Join one left chunk against its ordered right key range."""
            nonlocal sequence_number
            candidates = ChunkStore(context)
            try:
                demand, _ = await make_table_chunks_available_or_wait(
                    context,
                    demand,
                    reserve_extra=0,
                    net_memory_delta=0,
                )
                demand_range = key_bounds_view(demand)
                while True:
                    replay_message = next(iter(right_replay), None)
                    if replay_message is not None:
                        supply = TableChunk.from_message(
                            replay_message,
                            br=context.br(),
                        )
                    else:
                        pending_message = next(iter(right_pending), None)
                        if pending_message is not None:
                            supply = TableChunk.from_message(
                                pending_message,
                                br=context.br(),
                            )
                        else:
                            supply = await anext(right_iterator, None)
                            if supply is None:
                                break
                    (
                        selected,
                        pending,
                        passed_demand,
                    ) = await select_demand_range(
                        context,
                        supply,
                        demand_range,
                        demand.stream,
                        strategy.right_order_keys,
                    )
                    if selected is not None:
                        candidates.insert(Message(0, selected))
                    if pending is not None:
                        right_pending.insert(Message(0, pending))
                    if passed_demand:
                        break

                candidate_chunks = [
                    TableChunk.from_message(message, br=context.br())
                    for message in candidates
                ]
                if not candidate_chunks:
                    right_chunk = empty_table_chunk(
                        ir.children[1],
                        context,
                        ir_context.get_cuda_stream(),
                    )
                elif len(candidate_chunks) == 1:
                    right_chunk = candidate_chunks[0]
                else:
                    right_chunk = await concat_batch(
                        candidate_chunks,
                        context,
                        ir.children[1].schema,
                        ir_context,
                    )
                right_chunk, extra = await make_table_chunks_available_or_wait(
                    context,
                    right_chunk,
                    reserve_extra=0,
                    net_memory_delta=0,
                )
                with opaque_memory_usage(extra):
                    result = await _evaluate_incremental_join(
                        context,
                        ir,
                        ir_context,
                        left,
                        right_chunk,
                        right_available=True,
                    )
                    await send_chunk(
                        context,
                        ch_out,
                        result,
                        sequence_number,
                        tracer=tracer,
                    )
                    sequence_number += 1
                if right_chunk.shape[0]:
                    right_replay.insert(Message(0, right_chunk))
            finally:
                candidates.clear()

        async for chunk in left_chunks:
            if chunk.shape[0] == 0:
                result = await _evaluate_incremental_join(
                    context,
                    ir,
                    ir_context,
                    chunk,
                    empty_table_chunk(
                        ir.children[1],
                        context,
                        ir_context.get_cuda_stream(),
                    ),
                )
                await send_chunk(
                    context,
                    ch_out,
                    result,
                    sequence_number,
                    tracer=tracer,
                )
                sequence_number += 1
                continue
            available, demand = await copy_key_range(
                context,
                chunk,
                strategy.left_order_keys,
            )
            await evaluate_left_chunk(available, demand)

        async for _ in right_iterator:
            pass
        await ch_out.drain(context)
    finally:
        right_replay.clear()
        right_pending.clear()


def _rank_ranges_are_disjoint(
    endpoints: RankEndpoints,
    nranks: int,
    order_keys: tuple[OrderKey, ...],
    stream: Any,
) -> bool:
    """Return whether adjacent nonempty ranks have disjoint key ranges."""
    ranges = tuple(
        bounds
        for bounds in rank_key_bounds(endpoints, nranks, stream)
        if bounds is not None
    )
    for left, right in pairwise(ranges):
        assert left.upper is not None
        assert right.lower is not None
        if not key_row_less(left.upper, right.lower, order_keys, stream):
            return False
    return True


async def _distributed_incremental_ordered_join(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    strategy: IncrementalOrderedJoinStrategy,
    collective_ids: JoinCollectiveIds,
    *,
    tracer: ActorTracer | None,
) -> None:
    """Align globally ordered inputs by rank, then join each rank locally."""
    stream = ir_context.get_cuda_stream()
    await gather_in_task_group(
        recv_metadata(ch_left, context),
        recv_metadata(ch_right, context),
    )
    left, right = await gather_in_task_group(
        buffer_ordered_input(
            context,
            ch_left,
            ir.children[0].schema,
            strategy.left_order_keys,
            stream,
            summarize_chunks=True,
        ),
        buffer_ordered_input(
            context,
            ch_right,
            ir.children[1].schema,
            strategy.right_order_keys,
            stream,
            summarize_chunks=True,
        ),
        cleanup=lambda input_: input_.clear(),
    )
    prepared_left: PreparedExchange | None = None
    prepared_right: PreparedExchange | None = None
    prepared_left_chunks: ChunkStore | None = None
    try:
        left_endpoints, right_endpoints = await gather_rank_endpoint_sets(
            context,
            comm,
            ir_context,
            collective_ids.size_estimate,
            (left.endpoints, right.endpoints),
            stream,
        )
        if _rank_ranges_are_disjoint(
            left_endpoints,
            comm.nranks,
            strategy.left_order_keys,
            stream,
        ):
            demands = rank_key_bounds(left_endpoints, comm.nranks, stream)
            plan = plan_key_bounds_exchange(
                comm.rank,
                comm.nranks,
                right_endpoints,
                demands,
                strategy.right_order_keys,
                stream,
                transfer_ownership=False,
            )
            left_sizes = tuple(left.summary_sizes)
            assert left.summaries is not None
            left.summaries.clear()
            left.summaries = None
            left.summary_sizes = ()
            prepared_right = await exchange_buffered_key_input(
                context,
                comm,
                collective_ids.right_redistribution,
                right,
                plan,
                strategy.right_order_keys,
                stream,
            )
            left_count = len(left_sizes)
            left_chunks = _stored_chunks(context, left.chunks)
            inter_rank_ordering = strict_inter_rank_ordering(
                context.br(),
                left_endpoints,
                strategy.left_order_keys,
                strategy.output_order_keys,
                comm.nranks,
                stream,
            )
        else:
            left_intervals = rank_ownership_intervals(
                left_endpoints,
                comm.nranks,
                strategy.left_order_keys,
                stream,
                clip_to_domain=False,
            )
            right_intervals = rank_ownership_intervals(
                left_endpoints,
                comm.nranks,
                strategy.left_order_keys,
                stream,
                clip_to_domain=True,
            )
            left_plan = plan_key_bounds_exchange(
                comm.rank,
                comm.nranks,
                left_endpoints,
                left_intervals,
                strategy.left_order_keys,
                stream,
                transfer_ownership=True,
            )
            right_plan = plan_key_bounds_exchange(
                comm.rank,
                comm.nranks,
                right_endpoints,
                right_intervals,
                strategy.right_order_keys,
                stream,
                transfer_ownership=True,
            )
            prepared_left, prepared_right = await gather_in_task_group(
                exchange_buffered_key_input(
                    context,
                    comm,
                    collective_ids.left_redistribution,
                    left,
                    left_plan,
                    strategy.left_order_keys,
                    stream,
                ),
                exchange_buffered_key_input(
                    context,
                    comm,
                    collective_ids.right_redistribution,
                    right,
                    right_plan,
                    strategy.right_order_keys,
                    stream,
                ),
                cleanup=lambda prepared: prepared.clear(),
            )
            prepared_left_chunks = await prepared_left.take_source_order(
                context, comm.rank
            )
            left_count = len(prepared_left_chunks)
            left_chunks = _stored_chunks(context, prepared_left_chunks)
            inter_rank_ordering = repaired_inter_rank_ordering(
                context.br(),
                left_endpoints,
                strategy.output_order_keys,
                comm.nranks,
                stream,
            )
        output_strategy = replace(
            strategy,
            inter_rank_ordering=inter_rank_ordering,
        )
        await send_metadata(
            ch_out,
            context,
            _incremental_join_metadata(
                left_count,
                output_strategy,
            ),
        )
        if tracer is not None:
            tracer.decision = "ordered_distributed"
        right_chunks = _prepared_chunks(context, prepared_right, comm.rank)
        async with aclosing(left_chunks), aclosing(right_chunks):
            await _incremental_ordered_join_chunks(
                context,
                ir,
                ir_context,
                ch_out,
                left_chunks,
                right_chunks,
                strategy,
                tracer=tracer,
            )
    finally:
        if prepared_left is not None:
            prepared_left.clear()
        if prepared_right is not None:
            prepared_right.clear()
        if prepared_left_chunks is not None:
            prepared_left_chunks.clear()
        left.clear()
        right.clear()


def _make_shuffle_strategy(
    ir: Join,
    shuffle_modulus: int,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
) -> ShuffleJoinStrategy:
    """Make a hash-partitioned join strategy."""

    # Use the coarsest prefix so we only shuffle on keys one side may already have
    def _num_indices(partitioning: NormalizedPartitioning) -> int:
        return (
            len(partitioning.inter_rank_scheme.column_indices)
            if isinstance(partitioning.inter_rank_scheme, HashScheme)
            else 0
        )

    n_left = _num_indices(left_partitioning)
    n_right = _num_indices(right_partitioning)
    if n_left and n_right:
        n_partitioned_keys = min(n_left, n_right)
    elif n_left or n_right:
        n_partitioned_keys = max(n_left, n_right)
    else:
        n_partitioned_keys = None  # both unpartitioned: shuffle on all join keys

    (
        left_key_indices,
        right_key_indices,
        output_key_indices,
        left_keys,
        right_keys,
    ) = _get_key_indices(ir, n_partitioned_keys)

    return ShuffleJoinStrategy(
        shuffle_modulus=shuffle_modulus,
        output_indices=output_key_indices,
        left_indices=left_key_indices,
        right_indices=right_key_indices,
        left_keys=left_keys,
        right_keys=right_keys,
    )


def _broadcast_input_fits(
    size: int,
    rows: int,
    broadcast_limit: int,
    *,
    can_join_build_chunks_separately: bool,
) -> bool:
    """Return whether an input is small enough to broadcast."""
    return size < broadcast_limit and (
        rows < MAX_ROWS_PER_PARTITION or can_join_build_chunks_separately
    )


def _can_join_build_chunks_separately(
    comm: Communicator,
    ir: Join,
    metadata: ChannelMetadata,
) -> bool:
    """Return whether broadcast execution may retain separate build chunks."""
    need_allgather = comm.nranks > 1 and not metadata.duplicated
    return not need_allgather and ir.options[0] == "Inner" and ir.options[5] == "none"


def _allow_decomposed_broadcast(
    comm: Communicator,
    ir: Join,
    metadata: ChannelMetadata,
) -> bool:
    """Return whether planning may keep a broadcast input as separate chunks."""
    return (
        comm.nranks > 1
        and metadata.duplicated
        and _can_join_build_chunks_separately(comm, ir, metadata)
    )


def _choose_strategy_from_samples(
    comm: Communicator,
    ir: Join,
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    executor: StreamingExecutor,
    *,
    left_sample: TableSizeStats,
    right_sample: TableSizeStats,
    chunkwise: bool,
    tracer: ActorTracer | None,
    allow_broadcast: bool = True,
) -> JoinStrategy:
    """
    Choose a tentative broadcast side or a committed non-broadcast plan.

    When ``allow_broadcast`` is false, choose a shuffle plan.
    """
    if chunkwise:
        if tracer is not None:
            tracer.decision = "chunkwise"
        # TODO: Ensure this emits a "dynamic planning" decision of "chunkwise"
        # Or push it up a level to the caller?
        assert isinstance(left_partitioning.inter_rank_scheme, HashScheme)
        return _make_shuffle_strategy(
            ir,
            left_partitioning.inter_rank_scheme.modulus,
            left_partitioning,
            right_partitioning,
        )

    left_total, right_total = left_sample.total_size, right_sample.total_size
    left_total_rows, right_total_rows = left_sample.total_rows, right_sample.total_rows
    how = ir.options[0]

    # =====================================================================
    # Broadcast-Join Strategy Selection
    # =====================================================================
    # - Inner: can broadcast either side
    # - Left/Semi/Anti: must broadcast right (stream left to preserve all left rows)
    # - Right: must broadcast left (stream right to preserve all right rows)
    # - Full: cannot broadcast (must shuffle both to preserve both sides)

    # Determine which sides may be broadcasted
    left_size_ok = _broadcast_input_fits(
        left_total,
        left_total_rows,
        executor.broadcast_limit,
        can_join_build_chunks_separately=_allow_decomposed_broadcast(
            comm, ir, left_metadata
        ),
    )
    right_size_ok = _broadcast_input_fits(
        right_total,
        right_total_rows,
        executor.broadcast_limit,
        can_join_build_chunks_separately=_allow_decomposed_broadcast(
            comm, ir, right_metadata
        ),
    )
    can_broadcast_left = allow_broadcast and left_size_ok and how in ("Inner", "Right")
    can_broadcast_right = (
        allow_broadcast
        and right_size_ok
        and how
        in (
            "Inner",
            "Left",
            "Semi",
            "Anti",
        )
    )

    broadcast_side: Literal["left", "right"] | None = None
    if can_broadcast_left and can_broadcast_right:
        # Choose side that is already duplicated.
        # If both or neither are duplicated, choose the side with fewer rows.
        if left_metadata.duplicated == right_metadata.duplicated:
            broadcast_side = "right" if right_total_rows <= left_total_rows else "left"
        elif left_metadata.duplicated:
            broadcast_side = "left"
        else:
            broadcast_side = "right"
    elif can_broadcast_left:
        broadcast_side = "left"
    elif can_broadcast_right:
        broadcast_side = "right"
    if broadcast_side is not None:
        return BroadcastJoinStrategy(side=broadcast_side)

    # Couldn't broadcast - Use a shuffle join instead.
    estimated_output_size = max(left_total, right_total)
    ideal_output_count = max(1, estimated_output_size // executor.target_partition_size)
    # Limit the output count to 10x the larger input side. This prevents an
    # oversized sample from blowing up the chunk count.
    max_output_chunks = 10 * max(left_sample.total_chunks, right_sample.total_chunks)
    min_shuffle_modulus = min(ideal_output_count, max_output_chunks)

    # Stay away from cuDF's row limit.
    estimated_rows_count = max(left_total_rows, right_total_rows)
    if estimated_rows_count > 0:
        min_partitions_for_row_limit = (
            estimated_rows_count + MAX_ROWS_PER_PARTITION - 1
        ) // MAX_ROWS_PER_PARTITION
        min_shuffle_modulus = max(min_shuffle_modulus, min_partitions_for_row_limit)

    shuffle_modulus = _choose_shuffle_modulus(
        comm,
        left_partitioning,
        right_partitioning,
        min_shuffle_modulus,
    )
    strategy = _make_shuffle_strategy(
        ir,
        shuffle_modulus,
        left_partitioning,
        right_partitioning,
    )
    if tracer is not None:
        tracer.decision = _shuffle_strategy_decision(
            strategy,
            left_partitioning,
            right_partitioning,
        )
    return strategy


def _choose_shuffle_modulus(
    comm: Communicator,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    min_shuffle_modulus: int,
) -> int:
    """Choose an appropriate modulus for a shuffle join."""

    def _modulus(partitioning: NormalizedPartitioning) -> int | None:
        return (
            partitioning.inter_rank_scheme.modulus
            if isinstance(partitioning.inter_rank_scheme, HashScheme)
            else None
        )

    left_modulus = _modulus(left_partitioning)
    right_modulus = _modulus(right_partitioning)
    default_modulus = max(comm.nranks, min_shuffle_modulus)
    small, large = sorted(
        [left_modulus or default_modulus, right_modulus or default_modulus]
    )
    if large % small == 0 and small >= min_shuffle_modulus:
        return small
    else:
        return max(large, min_shuffle_modulus)


def join_input_requires_redistribution(
    strategy: JoinStrategy,
    side: Literal["left", "right"],
    partitioning: NormalizedPartitioning,
    metadata: ChannelMetadata,
) -> bool:
    """Return whether the join strategy redistributes an input side."""
    if isinstance(strategy, BroadcastJoinStrategy):
        return side == strategy.side and not metadata.duplicated
    if isinstance(strategy, (OrderedJoinStrategy, IncrementalOrderedJoinStrategy)):
        # Ordered inputs already have a viable join strategy without adaptive
        # sampling. Keep that strategy and avoid introducing a sampled
        # prefilter pipeline solely to optimize boundary alignment.
        return False

    assert isinstance(strategy, ShuffleJoinStrategy)
    indices = strategy.left_indices if side == "left" else strategy.right_indices
    if not indices:
        return True
    desired = HashScheme(indices, strategy.shuffle_modulus)
    return not (
        partitioning.inter_rank_scheme == desired
        and partitioning.local_scheme == "inherit"
    )


def choose_prefilters(
    join_state: JoinPlanningState,
    strategy: JoinStrategy,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    broadcast_limit: int,
    bloom_filter_max_size: int,
) -> None:
    """Choose strategies for prefilters with sufficient available statistics."""
    partitionings = {
        "left": left_partitioning,
        "right": right_partitioning,
    }
    for candidate in join_state.candidates:
        if candidate.decision is not None:
            continue
        target = candidate.target.sample
        if target is None:
            raise ValueError("Join target has not been sampled")
        target_side = candidate.spec.target_side
        target_requires_redistribution = join_input_requires_redistribution(
            strategy,
            target_side,
            partitionings[target_side],
            candidate.target.metadata,
        )
        if (
            isinstance(candidate.spec.domain, ExternalDomain)
            and candidate.domain.sample is None
            and target_requires_redistribution
        ):
            continue
        candidate.decision = choose_prefilter(
            candidate.spec,
            target,
            candidate.domain.sample,
            target_requires_redistribution=target_requires_redistribution,
            broadcast_limit=broadcast_limit,
            bloom_filter_max_size=bloom_filter_max_size,
        )


async def collect_samples(
    context: Context,
    comm: Communicator,
    join_state: JoinPlanningState,
    inputs: tuple[JoinInput, ...],
    sample_chunk_count: int,
    target_partition_size: int,
    collective_id: int,
) -> None:
    """Sample inputs and attach aggregate estimates to their planning state."""
    if not inputs:
        return
    sampling_inputs = []
    for input_ in inputs:
        candidates = [
            candidate
            for candidate in join_state.candidates
            if candidate.domain is input_
        ]
        if len(candidates) > 1:
            raise ValueError("One join input cannot provide multiple prefilter domains")
        sampling_inputs.append((input_, candidates[0] if candidates else None))
    samplers = []
    for input_, candidate in sampling_inputs:
        if candidate is None:
            cardinality_estimator = None
            cardinality_columns: tuple[int, ...] = ()
        else:
            cardinality_estimator = CardinalityEstimator(
                context,
                comm,
                tag=candidate.cardinality_tag,
            )
            cardinality_columns = names_to_indices(
                candidate.spec.domain_on,
                input_.node.schema,
            )
            assert len(cardinality_columns) == len(candidate.spec.domain_on), (
                "Prefilter domain keys must be columns"
            )
        samplers.append(
            ChunkSampler(
                context=context,
                ch_in=input_.channel,
                max_chunks=sample_chunk_count,
                max_bytes=target_partition_size,
                ch_in_chunk_count=input_.metadata.local_count,
                cardinality_estimator=cardinality_estimator,
                cardinality_columns=cardinality_columns,
            )
        )
    samples = await sample_inputs(
        context,
        comm,
        samplers,
        collective_id,
    )
    for (input_, _), sample in zip(sampling_inputs, samples, strict=True):
        input_.sample = sample


async def release_skipped_external_domains(
    context: Context, join_state: JoinPlanningState
) -> None:
    """Release buffered data and stop external domains rejected by planning."""
    channels = []
    for candidate in join_state.candidates:
        if not isinstance(candidate.spec.domain, ExternalDomain):
            continue
        if candidate.decision is None:
            raise ValueError("Join prefilter has no runtime decision")
        if candidate.decision.method != "skip":
            continue
        if candidate.domain.sample is not None:
            candidate.domain.sample.local_sample.chunks.clear()
        channels.append(candidate.domain.channel)
    if channels:
        await gather_in_task_group(*(channel.shutdown(context) for channel in channels))


async def resolve_prefilters(
    context: Context,
    comm: Communicator,
    join_state: JoinPlanningState,
    strategy: JoinStrategy,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    executor: StreamingExecutor,
    collective_id: int,
) -> None:
    """Resolve optional prefilters after selecting the join strategy."""
    config = executor.join_filter_pushdown
    if config is None or not join_state.candidates:
        return

    choose_prefilters(
        join_state,
        strategy,
        left_partitioning,
        right_partitioning,
        executor.broadcast_limit,
        config.bloom_filter_max_size,
    )
    assert executor.dynamic_planning is not None
    await collect_samples(
        context,
        comm,
        join_state,
        tuple(
            candidate.domain
            for candidate in join_state.candidates
            if isinstance(candidate.spec.domain, ExternalDomain)
            and candidate.decision is None
        ),
        executor.dynamic_planning.sample_chunk_count,
        executor.target_partition_size,
        collective_id,
    )
    choose_prefilters(
        join_state,
        strategy,
        left_partitioning,
        right_partitioning,
        executor.broadcast_limit,
        config.bloom_filter_max_size,
    )
    await release_skipped_external_domains(context, join_state)


async def _validate_broadcast_candidate(
    context: Context,
    comm: Communicator,
    input_: JoinInput,
    broadcast_limit: int,
    collective_id: int,
    *,
    can_join_build_chunks_separately: bool,
) -> bool:
    """Buffer a candidate until exhausted or proven too large to broadcast."""
    sample = input_.sample
    if sample is None:
        raise ValueError("Broadcast candidate has not been sampled")
    if sample.is_complete:
        return True

    local_sample = sample.local_sample
    local_size = local_sample.size
    local_rows = local_sample.rows
    local_is_complete = local_sample.is_complete
    fits_broadcast = partial(
        _broadcast_input_fits,
        broadcast_limit=broadcast_limit,
        can_join_build_chunks_separately=can_join_build_chunks_separately,
    )

    while not local_is_complete and fits_broadcast(local_size, local_rows):
        msg = await input_.channel.recv(context)
        if msg is None:
            local_is_complete = True
            break
        chunk = TableChunk.from_message(msg, br=context.br())
        local_size += chunk.data_alloc_size()
        local_rows += chunk.shape[0]
        local_sample.chunks.insert(Message(msg.sequence_number, chunk))

    local_fits = fits_broadcast(local_size, local_rows)
    (
        total_size,
        total_rows,
        complete_rank_count,
        fitting_rank_count,
    ) = await allgather_reduce(
        context,
        comm,
        collective_id,
        local_size,
        local_rows,
        int(local_is_complete),
        int(local_fits),
    )
    is_complete = complete_rank_count == comm.nranks

    if input_.metadata.duplicated:
        # Avoid counting a duplicated input once per rank.
        is_safe = is_complete and fitting_rank_count == comm.nranks
    else:
        is_safe = is_complete and fits_broadcast(total_size, total_rows)

    # Preserve estimates when validation stops before exhausting the input.
    input_.sample = replace(
        sample,
        local_sample=replace(
            local_sample,
            size=local_size,
            rows=local_rows,
            is_complete=local_is_complete,
        ),
        total_size=total_size if is_complete else max(sample.total_size, total_size),
        total_rows=total_rows if is_complete else max(sample.total_rows, total_rows),
        is_complete=is_complete,
    )
    return is_safe


async def choose_strategy(
    context: Context,
    comm: Communicator,
    ir: Join,
    join_state: JoinPlanningState,
    executor: StreamingExecutor,
    collective_ids: JoinCollectiveIds,
    requested_order_keys: tuple[OrderKey, ...],
    *,
    tracer: ActorTracer | None,
) -> JoinStrategy:
    """Collect any required samples and choose broadcast vs shuffle."""
    left, right = ir.children[:2]
    left_metadata = join_state.left.metadata
    right_metadata = join_state.right.metadata
    nranks = comm.nranks
    left_partitioning = NormalizedPartitioning.from_keys(
        left_metadata.partitioning,
        nranks,
        keys=names_to_indices(ir.left_on, left.schema, concrete_prefix=True),
    )
    right_partitioning = NormalizedPartitioning.from_keys(
        right_metadata.partitioning,
        nranks,
        keys=names_to_indices(ir.right_on, right.schema, concrete_prefix=True),
    )
    hash_chunkwise = isinstance(
        left_partitioning.inter_rank_scheme, HashScheme
    ) and isinstance(right_partitioning.inter_rank_scheme, HashScheme)
    chunkwise = hash_chunkwise and left_partitioning.is_aligned_with(
        right_partitioning, context.br()
    )
    if chunkwise:
        join_state.left.sample = TableSizeStats(
            local_sample=TableSample(ChunkStore(context)),
            total_chunks=left_metadata.local_count,
        )
        join_state.right.sample = TableSizeStats(
            local_sample=TableSample(ChunkStore(context)),
            total_chunks=right_metadata.local_count,
        )
    elif (
        not left_metadata.duplicated
        and not right_metadata.duplicated
        and (
            ordered_strategy := _make_ordered_strategy(
                ir,
                left_partitioning,
                right_partitioning,
            )
        )
        is not None
        and (
            ordered_execution := _ordered_join_execution(
                context,
                comm,
                ir,
                ordered_strategy,
                left_partitioning,
                right_partitioning,
                left_metadata,
                right_metadata,
            )
        )
        is not None
    ):
        selected_strategy: JoinStrategy = (
            IncrementalOrderedJoinStrategy(
                output_order_keys=ordered_strategy.output_ordering.keys,
                left_order_keys=ordered_strategy.left_input_ordering.keys,
                right_order_keys=ordered_strategy.right_input_ordering.keys,
                inter_rank_ordering=ordered_strategy.output_ordering,
            )
            if ordered_execution == "cursor"
            else ordered_strategy
        )
        if tracer is not None:
            tracer.decision = "ordered"
        join_state.left.sample = TableSizeStats(
            local_sample=TableSample(ChunkStore(context)),
            total_chunks=left_metadata.local_count,
        )
        join_state.right.sample = TableSizeStats(
            local_sample=TableSample(ChunkStore(context)),
            total_chunks=right_metadata.local_count,
        )
        await resolve_prefilters(
            context,
            comm,
            join_state,
            selected_strategy,
            left_partitioning,
            right_partitioning,
            executor,
            collective_ids.size_estimate,
        )
        return selected_strategy
    elif (
        (nranks == 1 or not join_state.candidates)
        and not left_metadata.duplicated
        and not right_metadata.duplicated
        and (
            incremental_strategy := _make_incremental_ordered_strategy(
                ir,
                left_metadata,
                right_metadata,
                requested_order_keys,
            )
        )
        is not None
    ):
        join_state.left.sample = TableSizeStats(
            local_sample=TableSample(ChunkStore(context)),
            total_chunks=left_metadata.local_count,
        )
        join_state.right.sample = TableSizeStats(
            local_sample=TableSample(ChunkStore(context)),
            total_chunks=right_metadata.local_count,
        )
        await resolve_prefilters(
            context,
            comm,
            join_state,
            incremental_strategy,
            left_partitioning,
            right_partitioning,
            executor,
            collective_ids.size_estimate,
        )
        return incremental_strategy
    else:
        assert executor.dynamic_planning is not None
        await collect_samples(
            context,
            comm,
            join_state,
            (join_state.left, join_state.right),
            executor.dynamic_planning.sample_chunk_count,
            executor.target_partition_size,
            collective_ids.size_estimate,
        )

    left_sample = join_state.left.sample
    right_sample = join_state.right.sample
    if left_sample is None or right_sample is None:
        raise ValueError("Join inputs have not been sampled")
    proposed_strategy = _choose_strategy_from_samples(
        comm,
        ir,
        left_metadata,
        right_metadata,
        left_partitioning,
        right_partitioning,
        executor,
        left_sample=left_sample,
        right_sample=right_sample,
        chunkwise=chunkwise,
        tracer=tracer,
    )
    if isinstance(proposed_strategy, BroadcastJoinStrategy):
        broadcast_candidate = proposed_strategy
        candidate_input = (
            join_state.left if broadcast_candidate.side == "left" else join_state.right
        )
        if await _validate_broadcast_candidate(
            context,
            comm,
            candidate_input,
            executor.broadcast_limit,
            collective_ids.size_estimate,
            can_join_build_chunks_separately=_allow_decomposed_broadcast(
                comm, ir, candidate_input.metadata
            ),
        ):
            strategy: JoinStrategy = broadcast_candidate
        else:
            left_sample = join_state.left.sample
            right_sample = join_state.right.sample
            if left_sample is None or right_sample is None:
                raise ValueError("Join inputs have not been sampled")
            strategy = _choose_strategy_from_samples(
                comm,
                ir,
                left_metadata,
                right_metadata,
                left_partitioning,
                right_partitioning,
                executor,
                left_sample=left_sample,
                right_sample=right_sample,
                chunkwise=False,
                tracer=None,
                allow_broadcast=False,
            )
            assert isinstance(strategy, ShuffleJoinStrategy)
            if tracer is not None:
                shuffle_decision = _shuffle_strategy_decision(
                    strategy,
                    left_partitioning,
                    right_partitioning,
                )
                tracer.decision = (
                    f"broadcast_{broadcast_candidate.side}_rejected_{shuffle_decision}"
                )
    else:
        strategy = proposed_strategy
    await resolve_prefilters(
        context,
        comm,
        join_state,
        strategy,
        left_partitioning,
        right_partitioning,
        executor,
        collective_ids.size_estimate,
    )
    return strategy


@define_actor()
async def join_actor(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    ch_prefilter_domains: tuple[Channel[TableChunk], ...],
    executor: StreamingExecutor,
    collective_ids: JoinCollectiveIds,
    requested_order_keys: tuple[OrderKey, ...],
) -> None:
    """
    Dynamic Join actor that selects the best strategy at runtime.

    Receives metadata from the left and right channels, then executes a
    broadcast, hash-partitioned, or ordered join. Strategy is chosen at
    runtime from metadata and sampled chunks when partitioning is not aligned.

    Parameters
    ----------
    context
        RapidsMPF context (communicator, etc.).
    comm
        The communicator.
    ir
        The Join IR node.
    ir_context
        Execution context for the plan.
    ch_out
        Output channel for the join result.
    ch_left
        Input channel for the left side.
    ch_right
        Input channel for the right side.
    ch_prefilter_domains
        Input channels providing the prefilter key domains.
    executor
        Streaming executor configuration.
    collective_ids
        List of collective IDs for shuffle/broadcast; consumed as needed.
    requested_order_keys
        Output ordering requested by a downstream consumer, if supported.
    """
    async with shutdown_on_error(
        context,
        chs_in=(ch_left, ch_right),
        chs_out=(ch_out,),
        chs_aux=ch_prefilter_domains,
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        tracer = actor_scope.tracer
        ir_context = actor_scope.require_ir_context()
        (
            left_metadata,
            right_metadata,
            *prefilter_domain_metadata,
        ) = await gather_in_task_group(
            recv_metadata(ch_left, context),
            recv_metadata(ch_right, context),
            *(recv_metadata(ch, context) for ch in ch_prefilter_domains),
        )

        join_state = JoinPlanningState.create(
            ir,
            ch_left,
            ch_right,
            ch_prefilter_domains,
            left_metadata,
            right_metadata,
            tuple(prefilter_domain_metadata),
            collective_ids.cardinality_tags,
        )

        strategy = await choose_strategy(
            context,
            comm,
            ir,
            join_state,
            executor,
            collective_ids,
            requested_order_keys,
            tracer=tracer,
        )
        for candidate in join_state.candidates:
            if candidate.decision is None:
                raise ValueError("Join prefilter has no runtime decision")
            if tracer is not None:
                statistics = RuntimePrefilterStatistics.from_decision_prefilter(
                    candidate.decision,
                    candidate.spec,
                )
                candidate.statistics = statistics
                tracer.prefilters.append(statistics)
        left_sample = join_state.left.sample
        right_sample = join_state.right.sample
        if left_sample is None or right_sample is None:
            raise ValueError("Join inputs have not been sampled")
        ch_left_replay = context.create_channel()
        ch_right_replay = context.create_channel()
        prefilter_execution = make_prefilter_execution(
            context,
            comm,
            ir,
            ir_context,
            strategy,
            ch_left_replay,
            ch_right_replay,
            join_state,
            collective_ids,
        )
        async with shutdown_on_error(
            context,
            chs_aux=(ch_left_replay, ch_right_replay, *prefilter_execution.channels),
            trace_ir=ir,
            ir_context=ir_context,
        ) as inner_actor_scope:
            ir_context = inner_actor_scope.require_ir_context()
            actor_tasks = [
                replay_buffered_channel(
                    context,
                    ch_left_replay,
                    ch_left,
                    left_sample.local_sample.chunks,
                    left_metadata,
                    trace_ir=ir,
                ),
                replay_buffered_channel(
                    context,
                    ch_right_replay,
                    ch_right,
                    right_sample.local_sample.chunks,
                    right_metadata,
                    trace_ir=ir,
                ),
                *prefilter_execution.tasks,
            ]
            ch_left = prefilter_execution.left
            ch_right = prefilter_execution.right

            if isinstance(strategy, BroadcastJoinStrategy):
                actor_tasks.append(
                    broadcast_join(
                        context,
                        comm,
                        ir,
                        ir_context,
                        ch_out,
                        ch_left,
                        ch_right,
                        strategy,
                        collective_ids.broadcast,
                        executor.target_partition_size,
                        tracer=tracer,
                    )
                )
            elif isinstance(strategy, OrderedJoinStrategy):
                actor_tasks.append(
                    _ordered_join(
                        context,
                        comm,
                        ir,
                        ir_context,
                        ch_out,
                        ch_left,
                        ch_right,
                        strategy,
                        tracer=tracer,
                    )
                )
            elif isinstance(strategy, IncrementalOrderedJoinStrategy):
                ordered_join = (
                    _incremental_ordered_join(
                        context,
                        ir,
                        ir_context,
                        ch_out,
                        ch_left,
                        ch_right,
                        strategy,
                        tracer=tracer,
                    )
                    if comm.nranks == 1 or strategy.inter_rank_ordering is not None
                    else _distributed_incremental_ordered_join(
                        context,
                        comm,
                        ir,
                        ir_context,
                        ch_out,
                        ch_left,
                        ch_right,
                        strategy,
                        collective_ids,
                        tracer=tracer,
                    )
                )
                actor_tasks.append(ordered_join)
            elif isinstance(strategy, ShuffleJoinStrategy):
                actor_tasks.append(
                    _shuffle_join(
                        context,
                        comm,
                        ir,
                        ir_context,
                        ch_out,
                        ch_left,
                        ch_right,
                        strategy,
                        collective_ids.shuffle("left"),
                        collective_ids.shuffle("right"),
                        tracer=tracer,
                    )
                )
            else:
                assert_never(strategy)
            await gather_in_task_group(*actor_tasks)


def _use_pwise_join(
    executor: StreamingExecutor,
    partition_info: MutableMapping[IR, PartitionInfo],
    ir: Join,
) -> bool:
    """Whether to use a static-planning partition-wise join."""
    left, right = ir.children[:2]
    output_count = partition_info[ir].count
    if (
        output_count == 1
        and isinstance(left, Repartition)
        and isinstance(right, Repartition)
    ):
        # We fell back to single-partition behavior at lowering time
        return True

    if executor.name == "streaming" and executor.dynamic_planning is not None:
        return False

    left_count = partition_info[left].count
    right_count = partition_info[right].count
    left_partitioned = (
        partition_info[left].partitioned_on == ir.left_on and left_count == output_count
    )
    right_partitioned = (
        partition_info[right].partitioned_on == ir.right_on
        and right_count == output_count
    )
    return left_partitioned and right_partitioned


def _forward_join_partitioning(
    ir: Join,
    child_metadatas: Sequence[ChannelMetadata],
    context: Context,
) -> Partitioning | None:
    """Build partitioning metadata for a partition-wise join."""
    preserved_side = _preserved_join_side(ir)
    child_index = (
        1
        if preserved_side == "right"
        or (preserved_side is None and ir.options[0] == "Right")
        else 0
    )
    return _remap_join_partitioning(
        ir, child_metadatas[child_index], child_index, context
    )


@generate_ir_sub_network.register(Join)
@generate_ir_sub_network.register(JoinWithPrefilter)
def _(
    ir: Join | JoinWithPrefilter, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Join operation.
    left, right, *prefilter_domains = ir.children
    partition_info = rec.state["partition_info"]
    left_count = partition_info[left].count
    right_count = partition_info[right].count
    executor = rec.state["config_options"].executor
    pwise_join = _use_pwise_join(executor, partition_info, ir)

    if pwise_join and isinstance(ir, JoinWithPrefilter):
        raise AssertionError(
            "Partition-wise JoinWithPrefilter should have been simplified "
            "during IR lowering"
        )

    actors, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])
    ir_context = ir_context_for_node(rec, ir)

    if pwise_join:
        # Partition-wise join (use default_node_multi)
        actors[ir] = [
            default_node_multi(
                rec.state["context"],
                ir,
                ir_context,
                channels[ir].reserve_input_slot(),
                (
                    channels[left].reserve_output_slot(),
                    channels[right].reserve_output_slot(),
                ),
                partitioning_callback=partial(_forward_join_partitioning, ir),
            )
        ]
        return actors, channels

    elif (
        executor.name == "streaming"
        and executor.dynamic_planning is not None
        and ir.options[0] in ("Inner", "Left", "Right", "Full", "Semi", "Anti")
    ):
        # Dynamic join - decide strategy at runtime
        collective_ids = JoinCollectiveIds.from_reserved(
            rec.state["collective_id_map"].get(ir, [])
        )
        # Join uses up to 3 collective IDs. Cardinality allreduces complete
        # before the size allgather and join collectives. Runtime prefilters
        # reuse the collective ID of the target-side join redistribution, with
        # their filtered output channel providing the ordering barrier.
        actors[ir] = [
            join_actor(
                rec.state["context"],
                rec.state["comm"],
                ir,
                ir_context,
                channels[ir].reserve_input_slot(),
                channels[left].reserve_output_slot(),
                channels[right].reserve_output_slot(),
                tuple(
                    channels[domain].reserve_output_slot()
                    for domain in prefilter_domains
                ),
                executor,
                collective_ids,
                _requested_join_order_keys(
                    ir,
                    rec.state["partitioning_requests"].get(ir, ()),
                ),
            )
        ]
        return actors, channels

    else:
        # Broadcast join (use broadcast_join_actor)
        broadcast_side: Literal["left", "right"]
        if left_count >= right_count:
            # Broadcast right, stream left
            broadcast_side = "right"
        else:
            broadcast_side = "left"
        actors[ir] = [
            broadcast_join_actor(
                rec.state["context"],
                rec.state["comm"],
                ir,
                ir_context,
                channels[ir].reserve_input_slot(),
                channels[left].reserve_output_slot(),
                channels[right].reserve_output_slot(),
                broadcast_side=broadcast_side,
                collective_id=rec.state["collective_id_map"][ir][0],
                target_partition_size=executor.target_partition_size,
            )
        ]
        return actors, channels
