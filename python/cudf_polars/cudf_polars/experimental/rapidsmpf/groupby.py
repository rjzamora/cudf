# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GroupBy logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Col, NamedExpr
from cudf_polars.dsl.ir import IR, GroupBy, Select
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.experimental.groupby import combine, decompose
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import (
    define_py_node,
    shutdown_on_error,
)
from cudf_polars.experimental.rapidsmpf.shuffle import LocalShuffle
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    Metadata,
    process_children,
)
from cudf_polars.experimental.utils import _concat, _get_unique_fractions

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair


def apply_do_evaluate(
    chunk: TableChunk, ir: GroupBy, ir_context: IRExecutionContext
) -> TableChunk:
    """Apply GroupBy evaluation to a chunk."""
    df = ir.do_evaluate(
        *ir._non_child_args,
        DataFrame.from_table(
            chunk.table_view(),
            list(ir.children[0].schema.keys()),
            list(ir.children[0].schema.values()),
            chunk.stream,
        ),
        context=ir_context,
    )
    return TableChunk.from_pylibcudf_table(df.table, chunk.stream, exclusive_view=True)


@define_py_node()
async def groupby_node(
    context: Context,
    ir: GroupBy,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    ch_in: ChannelPair,
    user_unique_fraction: float | None,
    groupby_n_ary: int,
    target_partition_size: int,
    output_count: int,
) -> None:
    """Unified  GroupBy node."""
    async with shutdown_on_error(
        context,
        ch_in.metadata,
        ch_in.data,
        ch_out.metadata,
        ch_out.data,
    ):
        # Get groupby key column names
        groupby_key_columns = [ne.name for ne in ir.keys]

        # Receive metadata to inspect it (concurrently to avoid deadlock)
        metadata = await ch_in.recv_metadata(context)
        assert isinstance(metadata, Metadata), (
            f"Expected Metadata, got {type(metadata)}."
        )
        shuffled = metadata.partitioned_on == tuple(groupby_key_columns)

        need_preshuffle = False
        need_preconcat = ir.maintain_order and not shuffled
        sample_first_chunk = True  # TODO: Do we ever want to use False?
        name_generator = unique_names(ir.schema.keys())
        # Decompose the aggregation requests into three distinct phases
        try:
            selection_exprs, piecewise_exprs, reduction_exprs, need_preshuffle = (
                combine(
                    *(
                        decompose(agg.name, agg.value, names=name_generator)
                        for agg in ir.agg_requests
                    )
                )
            )
        except NotImplementedError:  # pragma: no cover
            need_preconcat = not shuffled

        # Simple single-partition or pre-concatenation case.
        if need_preconcat or metadata.count == output_count == 1:
            output_metadata = Metadata(output_count)
            await ch_out.send_metadata(context, output_metadata)

            chunks: list[TableChunk] = []
            while (msg := await ch_in.data.recv(context)) is not None:
                chunk = TableChunk.from_message(msg).make_available_and_spill(
                    context.br(), allow_overbooking=True
                )
                chunks.append(chunk)
            assert chunks, "Missing chunks"
            df = ir.do_evaluate(
                *ir._non_child_args,
                _concat(
                    *(
                        DataFrame.from_table(
                            chunk.table_view(),
                            list(ir.children[0].schema.keys()),
                            list(ir.children[0].schema.values()),
                            chunk.stream,
                        )
                        for chunk in chunks
                    ),
                    context=ir_context,
                ),
                context=ir_context,
            )
            await ch_out.data.send(
                context,
                Message(
                    0,
                    TableChunk.from_pylibcudf_table(
                        df.table, df.stream, exclusive_view=True
                    ),
                ),
            )
            await ch_out.data.drain(context)
            return

        # Define the partition-wise groupby operation
        pwise_schema = {k.name: k.value.dtype for k in ir.keys} | {
            k.name: k.value.dtype for k in piecewise_exprs
        }
        ir_pwise = GroupBy(
            pwise_schema,
            ir.keys,
            piecewise_exprs,
            ir.maintain_order,
            None,
            ir.children[0],
        )
        grouped_keys = tuple(
            NamedExpr(k.name, Col(k.value.dtype, k.name)) for k in ir.keys
        )

        # Define the select operation for the final output
        ir_select = Select(
            ir.schema,
            [
                *(NamedExpr(k.name, Col(k.value.dtype, k.name)) for k in grouped_keys),
                *selection_exprs,
            ],
            False,  # noqa: FBT003
            ir_pwise,
        )

        # Outer pre-shuffle context.
        # This will be a null context unless we need to preshuffle.
        schema_keys = list(ir.schema.keys())
        groupby_key_indices = tuple(schema_keys.index(k.name) for k in ir.keys)
        pre_shuffle_ctx = (
            LocalShuffle(
                context,
                metadata.count,
                groupby_key_indices,
            )
            if need_preshuffle
            else nullcontext()
        )
        with pre_shuffle_ctx as pre_shuffle:
            # Insert all chunks into the pre-shuffle (if needed)
            first_chunk: TableChunk | None = None
            if need_preshuffle:
                assert isinstance(pre_shuffle, LocalShuffle)
                while (msg := await ch_in.data.recv(context)) is not None:
                    chunk = TableChunk.from_message(msg).make_available_and_spill(
                        context.br(), allow_overbooking=True
                    )
                    pre_shuffle.insert_chunk(chunk)

                # Extract the first chunk from the pre-shuffle.
                if sample_first_chunk:
                    stream = ir_context.get_cuda_stream()
                    first_chunk = TableChunk.from_pylibcudf_table(
                        pre_shuffle.extract_chunk(0, stream),
                        stream,
                        exclusive_view=True,
                    )
            elif sample_first_chunk:
                # Receive the first chunk from the input channel.
                first_msg = await ch_in.data.recv(context)
                assert first_msg is not None, "Missing first chunk"
                first_chunk = TableChunk.from_message(
                    first_msg
                ).make_available_and_spill(context.br(), allow_overbooking=True)

            # Apply the piecewise groupby operation to the first chunk for sampling.
            first_pwise_chunk: TableChunk | None = None
            first_pwise_size: int | None = None
            if sample_first_chunk:
                assert first_chunk is not None, "Missing first chunk"
                first_pwise_chunk = await asyncio.to_thread(
                    apply_do_evaluate,
                    first_chunk,
                    ir_pwise,
                    ir_context,
                )
                assert first_pwise_chunk is not None
                first_pwise_size = first_pwise_chunk.data_alloc_size(MemoryType.DEVICE)
                assert first_pwise_size is not None
                # TODO: Update the output_count based on the first_pwise_size and metadata.count
                ideal_output_count = max(
                    1, (first_pwise_size * metadata.count) // target_partition_size
                )
                output_count = min(ideal_output_count, output_count)

            # Define the reduction operation (used in both shuffle and tree cases)
            reduction_schema = {k.name: k.value.dtype for k in grouped_keys} | {
                k.name: k.value.dtype for k in reduction_exprs
            }
            ir_reduction = GroupBy(
                reduction_schema,
                grouped_keys,
                reduction_exprs,
                ir.maintain_order,
                None,
                ir_pwise,
            )

            # Shuffle-based groupby case.
            if output_count > 1 and not shuffled:
                # Send output metadata
                output_metadata = Metadata(
                    output_count,
                    partitioned_on=tuple(groupby_key_columns),
                )
                await ch_out.send_metadata(context, output_metadata)

                # Inner shuffle context.
                pwise_schema_keys = list(ir_pwise.schema.keys())
                pwise_key_indices = tuple(
                    pwise_schema_keys.index(key) for key in groupby_key_columns
                )
                with LocalShuffle(
                    context,
                    output_count,
                    pwise_key_indices,
                ) as shuffle:
                    # Insert grouped data into shuffle
                    if first_pwise_chunk is not None:
                        shuffle.insert_chunk(first_pwise_chunk)

                    if need_preshuffle:
                        # Extract remaining chunks from pre-shuffle and process
                        assert isinstance(pre_shuffle, LocalShuffle)
                        start_idx = 1 if sample_first_chunk else 0
                        for partition_id in range(start_idx, metadata.count):
                            stream = ir_context.get_cuda_stream()
                            input_chunk = TableChunk.from_pylibcudf_table(
                                pre_shuffle.extract_chunk(partition_id, stream),
                                stream,
                                exclusive_view=True,
                            )
                            chunk = await asyncio.to_thread(
                                apply_do_evaluate,
                                input_chunk,
                                ir_pwise,
                                ir_context,
                            )
                            shuffle.insert_chunk(chunk)
                    else:
                        # Read remaining chunks from input channel
                        while (msg := await ch_in.data.recv(context)) is not None:
                            chunk = await asyncio.to_thread(
                                apply_do_evaluate,
                                TableChunk.from_message(msg).make_available_and_spill(
                                    context.br(), allow_overbooking=True
                                ),
                                ir_pwise,
                                ir_context,
                            )
                            shuffle.insert_chunk(chunk)

                    # Extract shuffled chunks and apply the final select operation
                    for partition_id in range(output_count):
                        stream = ir_context.get_cuda_stream()
                        reduction_chunk = TableChunk.from_pylibcudf_table(
                            shuffle.extract_chunk(partition_id, stream),
                            stream,
                            exclusive_view=True,
                        )
                        # Apply reduction groupby
                        reduced_chunk = await asyncio.to_thread(
                            apply_do_evaluate,
                            reduction_chunk,
                            ir_reduction,
                            ir_context,
                        )
                        # Apply final select
                        df = ir_select.do_evaluate(
                            *ir_select._non_child_args,
                            DataFrame.from_table(
                                reduced_chunk.table_view(),
                                list(ir_reduction.schema.keys()),
                                list(ir_reduction.schema.values()),
                                reduced_chunk.stream,
                            ),
                            context=ir_context,
                        )
                        final_chunk = TableChunk.from_pylibcudf_table(
                            df.table, df.stream, exclusive_view=True
                        )
                        await ch_out.data.send(
                            context, Message(partition_id, final_chunk)
                        )

                    await ch_out.data.drain(context)

            else:
                # Tree-based groupby case.
                # Send output metadata
                output_metadata = Metadata(output_count)
                await ch_out.send_metadata(context, output_metadata)

                # Prepare for the tree reduction
                n = metadata.count
                if first_pwise_size is not None:
                    # TODO: Using 1024 as an arbitrary limit.
                    k = min(max(2, target_partition_size // first_pwise_size), 1024)
                else:  # pragma: no cover
                    k = groupby_n_ary
                level_count = int(math.ceil(math.log(n * (k - 1) + 1) / math.log(k)))
                done_receiving = False
                levels: defaultdict[int, list[DataFrame]] = defaultdict(list)
                if first_pwise_chunk is not None:
                    # Process the first (sampled) chunk
                    levels[0].append(
                        DataFrame.from_table(
                            first_pwise_chunk.table_view(),
                            list(ir_pwise.schema.keys()),
                            list(ir_pwise.schema.values()),
                            first_pwise_chunk.stream,
                        )
                    )

                # Perform the tree reduction
                sequence_num: int = 0
                input_partition_idx = 1 if sample_first_chunk else 0
                while not done_receiving:
                    if need_preshuffle:
                        # Extract from pre-shuffle
                        if input_partition_idx < metadata.count:
                            assert isinstance(pre_shuffle, LocalShuffle)
                            stream = ir_context.get_cuda_stream()
                            input_chunk = TableChunk.from_pylibcudf_table(
                                pre_shuffle.extract_chunk(input_partition_idx, stream),
                                stream,
                                exclusive_view=True,
                            )
                            chunk = await asyncio.to_thread(
                                apply_do_evaluate,
                                input_chunk,
                                ir_pwise,
                                ir_context,
                            )
                            input_partition_idx += 1
                        else:
                            done_receiving = True
                            chunk = None
                    else:
                        # Read from input channel
                        msg = await ch_in.data.recv(context)
                        if msg is None:
                            done_receiving = True
                            chunk = None
                        else:
                            chunk = TableChunk.from_message(
                                msg
                            ).make_available_and_spill(
                                context.br(), allow_overbooking=True
                            )
                            chunk = await asyncio.to_thread(
                                apply_do_evaluate,
                                chunk,
                                ir_pwise,
                                ir_context,
                            )
                            input_partition_idx += 1

                    if chunk is not None:
                        levels[0].append(
                            DataFrame.from_table(
                                chunk.table_view(),
                                list(ir_pwise.schema.keys()),
                                list(ir_pwise.schema.values()),
                                chunk.stream,
                            )
                        )

                    # Loop through the levels to push chunks as far as possible
                    for level in range(level_count):
                        if levels[level]:
                            count = len(levels[level])
                            if count >= k or done_receiving:
                                next_level = min(level + 1, level_count - 1)
                                # NOTE: We must call do_evaluate to promote the
                                # chunk(s) at this level, even if there is only
                                # one chunk. Skipping that call can result in
                                # mismatched dtypes in the next concatenation.
                                df = ir_reduction.do_evaluate(
                                    *ir_reduction._non_child_args,
                                    _concat(
                                        *[levels[level].pop() for _ in range(count)],
                                        context=ir_context,
                                    ),
                                    context=ir_context,
                                )
                                levels[next_level].append(df)
                            if level == level_count - 1 and (
                                output_count > 1 or done_receiving
                            ):
                                assert len(levels[level]) == 1, (
                                    "Expected 1 chunk at the last level"
                                )
                                df = ir_select.do_evaluate(
                                    *ir_select._non_child_args,
                                    levels[level].pop(),
                                    context=ir_context,
                                )
                                await ch_out.data.send(
                                    context,
                                    Message(
                                        sequence_num,
                                        TableChunk.from_pylibcudf_table(
                                            df.table, df.stream, exclusive_view=True
                                        ),
                                    ),
                                )
                                sequence_num += 1

                await ch_out.data.drain(context)


@generate_ir_sub_network.register(GroupBy)
def _(
    ir: GroupBy, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for GroupBy operation."""
    # Process children
    nodes, channels = process_children(ir, rec)
    if ir in nodes:
        return nodes, channels

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Get user-specified unique fraction
    config_options = rec.state["config_options"]
    executor = config_options.executor
    assert executor.name == "streaming", "Join node requires streaming executor"
    user_unique_fraction: float | None = None
    groupby_key_columns = [ne.name for ne in ir.keys]
    if unique_fraction_dict := _get_unique_fractions(
        groupby_key_columns,
        executor.unique_fraction,
    ):
        # Use unique_fraction to determine output partitioning
        user_unique_fraction = max(unique_fraction_dict.values())

    # Use unified join_node that decides strategy based on metadata
    nodes[ir] = [
        groupby_node(
            rec.state["context"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            user_unique_fraction,
            executor.groupby_n_ary,
            executor.target_partition_size,
            rec.state["partition_info"][ir].count,
        )
    ]

    return nodes, channels
