# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Union logic for the RapidsMPF streaming engine."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from rapidsmpf.shuffler import Shuffler
from rapidsmpf.streaming.coll.shuffler import shuffler
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.partition import partition_and_pack, unpack_and_concat

from cudf_polars.dsl.expr import Col
from cudf_polars.experimental.rapidsmpf.channel_pair import ChannelPair
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import shutdown_on_error
from cudf_polars.experimental.shuffle import Shuffle

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator


# Set of available shuffle IDs
_shuffle_id_vacancy: set[int] = set(range(Shuffler.max_concurrent_shuffles))
_shuffle_id_vacancy_lock: threading.Lock = threading.Lock()


def _get_new_shuffle_id() -> int:
    with _shuffle_id_vacancy_lock:
        if not _shuffle_id_vacancy:
            raise ValueError(
                f"Cannot shuffle more than {Shuffler.max_concurrent_shuffles} "
                "times in a single query."
            )

        return _shuffle_id_vacancy.pop()


@define_py_node()
async def metadata_passthrough_node(
    ctx: Context,
    ch_in: Channel,
    ch_out: Channel,
) -> None:
    """
    Pass through metadata from input to output.

    This is used to route metadata around the shuffle operation,
    since shuffle only operates on data chunks.

    Parameters
    ----------
    ctx
        The streaming context.
    ch_in
        Input metadata channel.
    ch_out
        Output metadata channel.
    """
    async with shutdown_on_error(ctx, ch_in, ch_out):
        # Receive and forward metadata
        msg = await ch_in.recv(ctx)
        if msg is not None:
            await ch_out.send(ctx, msg)

        # Drain output
        await ch_out.drain(ctx)


@generate_ir_sub_network.register(Shuffle)
def _(
    ir: Shuffle, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, list[Any]]]:
    # Local shuffle operation.
    # TODO: How to distinguish between local and global shuffle?
    # May need to track two different contexts?

    # Process children
    (child,) = ir.children
    nodes, channels = rec(child)

    keys: list[Col] = [ne.value for ne in ir.keys if isinstance(ne.value, Col)]
    if len(keys) != len(ir.keys):  # pragma: no cover
        raise NotImplementedError("Shuffle requires simple keys.")
    column_names = list(ir.schema.keys())

    context = rec.state["ctx"]
    columns_to_hash = tuple(column_names.index(k.name) for k in keys)
    num_partitions = rec.state["partition_info"][ir].count
    op_id = _get_new_shuffle_id()

    # Get input ChannelPair
    ch_in_pair = channels[child].pop()

    # Metadata passthrough (around the shuffle)
    ch_metadata_out = Channel()
    nodes[ir] = []
    nodes[ir].append(
        metadata_passthrough_node(
            context,
            ch_in=ch_in_pair.metadata,
            ch_out=ch_metadata_out,
        )
    )

    # Data shuffle pipeline
    # Partition and pack
    ch2 = Channel()
    nodes[ir].append(
        partition_and_pack(
            context,
            ch_in=ch_in_pair.data,
            ch_out=ch2,
            columns_to_hash=columns_to_hash,
            num_partitions=num_partitions,
        )
    )

    # Shuffle
    ch3 = Channel()
    nodes[ir].append(
        shuffler(
            context,
            ch_in=ch2,
            ch_out=ch3,
            op_id=op_id,
            total_num_partitions=num_partitions,
        )
    )

    # Unpack and concat
    ch_data_out = Channel()
    nodes[ir].append(unpack_and_concat(context, ch_in=ch3, ch_out=ch_data_out))

    # Create output ChannelPair combining metadata and shuffled data
    channels[ir] = [ChannelPair(metadata=ch_metadata_out, data=ch_data_out)]

    return nodes, channels
