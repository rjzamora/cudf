# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Join Logic."""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal

import pylibcudf as plc
from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.ir import ConditionalJoin, Join, Slice
from cudf_polars.experimental.base import PartitionInfo, get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.experimental.shuffle import (
    RMPFIntegration,
    Shuffle,
    _hash_partition_dataframe,
)
from cudf_polars.experimental.utils import _concat, _fallback_inform, _lower_ir_fallback

try:
    from rapidsmpf.buffer.packed_data import PackedData

    if TYPE_CHECKING:
        from rapidsmpf.integrations.core import WorkerContext

except ImportError:
    PackedData = Any

    if TYPE_CHECKING:
        WorkerContext = Any


if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer
    from cudf_polars.utils.config import ShuffleMethod


def _use_rapidsmpf_join(ir: Join, shuffle_method: ShuffleMethod) -> bool:
    """Return whether RapidsMPF will be used for the join."""
    # Don't use rapidsmpf join if the shuffle method is not "rapidsmpf"
    # or if the keys are not "simple"
    return shuffle_method == "rapidsmpf" and all(
        isinstance(ne.value, Col) for ne in (ir.left_on + ir.right_on)
    )


class FusedJoin(Join):
    """Fused RapidsMPF join."""


class LeftBcastJoin(FusedJoin):
    """Fused RapidsMPF left bcast join."""


class RightBcastJoin(FusedJoin):
    """Fused RapidsMPF right bcast join."""


class RMPFJoinIntegration:
    """RapidsMPF join integration."""

    @staticmethod
    def get_shuffler_integration() -> RMPFIntegration:
        """Return the shuffler integration."""
        return RMPFIntegration()

    @staticmethod
    def pack_partition(ctx: WorkerContext, data: DataFrame, options: Any) -> PackedData:
        """Pack a partition for broadcasting."""
        packed_columns = plc.contiguous_split.pack(data.table)
        return PackedData.from_cudf_packed_columns(
            packed_columns, DEFAULT_STREAM, ctx.br
        )

    @staticmethod
    def unpack_partition(
        ctx: WorkerContext, data: PackedData, options: Any
    ) -> DataFrame:
        """Unpack a finished partition from the RMPF shuffler."""
        from rapidsmpf.integrations.cudf.partition import (
            unpack_and_concat,
            unspill_partitions,
        )

        plc_table = unpack_and_concat(
            unspill_partitions(
                [data],
                br=ctx.br,
                allow_overbooking=True,
                statistics=ctx.statistics,
            ),
            br=ctx.br,
            stream=DEFAULT_STREAM,
        )
        return DataFrame.from_table(
            plc_table,
            options["column_names"],
            options["dtypes"],
        )

    @staticmethod
    def local_repartition(
        data: DataFrame,
        partition_count: int,
        options: Any,
    ) -> dict[int, DataFrame]:
        """
        Break a single DataFrame partition into multiple local partitions.

        Parameters
        ----------
        data
            The local DataFrame partition.
        partition_count
            The number of local partitions to generate.
        options
            Additional options.

        Returns
        -------
        A dictionary of DataFrame partitions.
        The keys are the partition ids.
        The values are the DataFrame partitions.
        """
        return _hash_partition_dataframe(
            data,
            0,  # Used only by sorted shuffling
            partition_count,
            None,
            options["on"],
        )

    @staticmethod
    def join_partition(
        left_input: Callable[[int], DataFrame],
        right_input: Callable[[int], DataFrame],
        bcast_side: Literal["left", "right", "none"],
        bcast_count: int,
        options: Any,
    ) -> DataFrame:
        """
        Produce a joined DataFrame partition.

        Parameters
        ----------
        left_input
            A callable that produces the partition(s) needed for the left table.
        right_input
            A callable that produces the partition(s) needed for the right table.
        bcast_side
            The side of the join being broadcasted (if either).
        bcast_count
            The number of broadcasted chunks.
            Ignored unless ``bcast_side`` is "left" or "right".
        options
            Additional join options.

        Returns
        -------
        A joined DataFrame partition.

        Notes
        -----
        This method is used to produce a single joined table chunk.
        """
        if bcast_side not in ("left", "right", "none"):  # pragma: no cover
            raise ValueError(
                f"Expected one of 'left', 'right', or 'none'. Got {bcast_side}"
            )

        non_child_args = options.get("non_child_args", ())
        if bcast_side == "none" or bcast_count < 2:
            return Join.do_evaluate(*non_child_args, left_input(0), right_input(0))
        else:
            return _concat(
                *(
                    Join.do_evaluate(*non_child_args, left_input(i), right_input(i))
                    for i in range(bcast_count)
                )
            )


def _maybe_shuffle_frame(
    frame: IR,
    on: tuple[NamedExpr, ...],
    partition_info: MutableMapping[IR, PartitionInfo],
    shuffle_method: ShuffleMethod,
    output_count: int,
) -> IR:
    # Shuffle `frame` if it isn't already shuffled.
    if (
        partition_info[frame].partitioned_on == on
        and partition_info[frame].count == output_count
    ):
        # Already shuffled
        return frame
    else:
        # Insert new Shuffle node
        frame = Shuffle(
            frame.schema,
            on,
            shuffle_method,
            frame,
        )
        partition_info[frame] = PartitionInfo(
            count=output_count,
            partitioned_on=on,
        )
        return frame


def _make_hash_join(
    ir: Join,
    output_count: int,
    partition_info: MutableMapping[IR, PartitionInfo],
    left: IR,
    right: IR,
    shuffle_method: ShuffleMethod,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    if _use_rapidsmpf_join(ir, shuffle_method):
        # Convert ir to RMPFJoin.
        # We don't need to shuffle the children
        ir = FusedJoin(
            ir.schema,
            ir.left_on,
            ir.right_on,
            ir.options,
            left,
            right,
        )
    else:
        # Shuffle left and right dataframes (if necessary)
        new_left = _maybe_shuffle_frame(
            left,
            ir.left_on,
            partition_info,
            shuffle_method,
            output_count,
        )
        new_right = _maybe_shuffle_frame(
            right,
            ir.right_on,
            partition_info,
            shuffle_method,
            output_count,
        )
        if left != new_left or right != new_right:
            ir = ir.reconstruct([new_left, new_right])
        left = new_left
        right = new_right

    # Record new partitioning info
    partitioned_on: tuple[NamedExpr, ...] = ()
    if ir.left_on == ir.right_on or (ir.options[0] in ("Left", "Semi", "Anti")):
        partitioned_on = ir.left_on
    elif ir.options[0] == "Right":
        partitioned_on = ir.right_on
    partition_info[ir] = PartitionInfo(
        count=output_count,
        partitioned_on=partitioned_on,
    )

    return ir, partition_info


def _should_bcast_join(
    ir: Join,
    left: IR,
    right: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    output_count: int,
    broadcast_join_limit: int,
) -> bool:
    # Decide if a broadcast join is appropriate.
    if partition_info[left].count >= partition_info[right].count:
        small_count = partition_info[right].count
        large = left
        large_on = ir.left_on
    else:
        small_count = partition_info[left].count
        large = right
        large_on = ir.right_on

    # Avoid the broadcast if the "large" table is already shuffled
    large_shuffled = (
        partition_info[large].partitioned_on == large_on
        and partition_info[large].count == output_count
    )

    # Broadcast-Join Criteria:
    # 1. Large dataframe isn't already shuffled
    # 2. Small dataframe has 8 partitions (or fewer).
    #    TODO: Make this value/heuristic configurable).
    #    We may want to account for the number of workers.
    # 3. The "kind" of join is compatible with a broadcast join

    return (
        not large_shuffled
        and small_count <= broadcast_join_limit
        and (
            ir.options[0] == "Inner"
            or (ir.options[0] in ("Left", "Semi", "Anti") and large == left)
            or (ir.options[0] == "Right" and large == right)
        )
    )


def _make_bcast_join(
    ir: Join,
    output_count: int,
    partition_info: MutableMapping[IR, PartitionInfo],
    left: IR,
    right: IR,
    shuffle_method: ShuffleMethod,
) -> tuple[Join, MutableMapping[IR, PartitionInfo]]:
    left_count = partition_info[left].count
    right_count = partition_info[right].count

    new_node: Join
    rmp_bcast_enabled = False  # Spilling not currently working with RMPF bcast join yet
    if rmp_bcast_enabled and _use_rapidsmpf_join(
        ir, shuffle_method
    ):  # pragma: no cover
        # Convert ir to RMPFJoin.
        # We don't need to pre-shuffle the small table yet.
        join_type = RightBcastJoin if left_count >= right_count else LeftBcastJoin
        new_node = join_type(
            ir.schema,
            ir.left_on,
            ir.right_on,
            ir.options,
            left,
            right,
        )

    else:
        if ir.options[0] != "Inner":
            # Shuffle the smaller table (if necessary) - Notes:
            # - We need to shuffle the smaller table if
            #   (1) we are not doing an "inner" join,
            #   and (2) the small table contains multiple
            #   partitions.
            # - We cannot simply join a large-table partition
            #   to each small-table partition, and then
            #   concatenate the partial-join results, because
            #   a non-"inner" join does NOT commute with
            #   concatenation.
            # - In some cases, we can perform the partial joins
            #   sequentially. However, we are starting with a
            #   catch-all algorithm that works for all cases.
            if left_count >= right_count:
                right = _maybe_shuffle_frame(
                    right,
                    ir.right_on,
                    partition_info,
                    shuffle_method,
                    right_count,
                )
            else:
                left = _maybe_shuffle_frame(
                    left,
                    ir.left_on,
                    partition_info,
                    shuffle_method,
                    left_count,
                )

        new_node = ir.reconstruct([left, right])

    partition_info[new_node] = PartitionInfo(count=output_count)
    return new_node, partition_info


@lower_ir_node.register(ConditionalJoin)
def _(
    ir: ConditionalJoin, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    if ir.options[2]:  # pragma: no cover
        return _lower_ir_fallback(
            ir,
            rec,
            msg="Slice not supported in ConditionalJoin for multiple partitions.",
        )

    # Lower children
    left, right = ir.children
    left, pi_left = rec(left)
    right, pi_right = rec(right)

    # Fallback to single partition on the smaller table
    left_count = pi_left[left].count
    right_count = pi_right[right].count
    output_count = max(left_count, right_count)
    fallback_msg = "ConditionalJoin not supported for multiple partitions."
    if left_count < right_count:
        if left_count > 1:
            left = Repartition(left.schema, left)
            pi_left[left] = PartitionInfo(count=1)
            _fallback_inform(fallback_msg, rec.state["config_options"])
    elif right_count > 1:
        right = Repartition(left.schema, right)
        pi_right[right] = PartitionInfo(count=1)
        _fallback_inform(fallback_msg, rec.state["config_options"])

    # Reconstruct and return
    new_node = ir.reconstruct([left, right])
    partition_info = reduce(operator.or_, (pi_left, pi_right))
    partition_info[new_node] = PartitionInfo(count=output_count)
    return new_node, partition_info


@lower_ir_node.register(Join)
def _(
    ir: Join, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Pull slice operations out of the Join before lowering
    if (zlice := ir.options[2]) is not None:
        offset, length = zlice
        if length is None:  # pragma: no cover
            return _lower_ir_fallback(
                ir,
                rec,
                msg="This slice not supported for multiple partitions.",
            )
        new_join = Join(
            ir.schema,
            ir.left_on,
            ir.right_on,
            (*ir.options[:2], None, *ir.options[3:]),
            *ir.children,
        )
        return rec(Slice(ir.schema, offset, length, new_join))

    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    left, right = children
    output_count = max(partition_info[left].count, partition_info[right].count)
    if output_count == 1:
        new_node = ir.reconstruct(children)
        partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info
    elif ir.options[0] == "Cross":  # pragma: no cover
        return _lower_ir_fallback(
            ir, rec, msg="Cross join not support for multiple partitions."
        )

    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'lower_join'"
    )
    if _should_bcast_join(
        ir,
        left,
        right,
        partition_info,
        output_count,
        config_options.executor.broadcast_join_limit,
    ):
        # Create a broadcast join
        return _make_bcast_join(
            ir,
            output_count,
            partition_info,
            left,
            right,
            config_options.executor.shuffle_method,
        )
    else:
        # Create a hash join
        return _make_hash_join(
            ir,
            output_count,
            partition_info,
            left,
            right,
            config_options.executor.shuffle_method,
        )


@generate_ir_tasks.register(Join)
def _(
    ir: Join, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    left, right = ir.children
    output_count = partition_info[ir].count

    left_partitioned = (
        partition_info[left].partitioned_on == ir.left_on
        and partition_info[left].count == output_count
    )
    right_partitioned = (
        partition_info[right].partitioned_on == ir.right_on
        and partition_info[right].count == output_count
    )

    if isinstance(ir, FusedJoin):
        from rapidsmpf.integrations.dask.join import rapidsmpf_join_graph

        need_local_repartition = False
        bcast_side: Literal["left", "right", "none"] = "none"
        if isinstance(ir, (LeftBcastJoin, RightBcastJoin)):  # pragma: no cover
            # TODO: RMPF bcast join doesn't seem to work with spilling yet
            bcast_side = "left" if isinstance(ir, LeftBcastJoin) else "right"
            if ir.options[0] != "Inner":
                need_local_repartition = True
                # We need to adjust the pre-partitioned check,
                # because the small-table doesn't need to have the
                # final partition count for us to avoid the pre-shuffle.
                if bcast_side == "left":
                    left_partitioned = partition_info[left].partitioned_on == ir.left_on
                elif bcast_side == "right":
                    right_partitioned = (
                        partition_info[right].partitioned_on == ir.right_on
                    )

        return rapidsmpf_join_graph(
            get_key_name(left),
            get_key_name(right),
            get_key_name(ir),
            partition_info[left].count,
            partition_info[right].count,
            RMPFJoinIntegration(),
            {
                "on": [ne.name for ne in ir.left_on],
                "column_names": list(left.schema.keys()),
                "dtypes": list(left.schema.values()),
            },
            {
                "on": [ne.name for ne in ir.right_on],
                "column_names": list(right.schema.keys()),
                "dtypes": list(right.schema.values()),
            },
            {
                "non_child_args": ir._non_child_args,
            },
            bcast_side=bcast_side,
            need_local_repartition=need_local_repartition,
            left_pre_shuffled=left_partitioned,
            right_pre_shuffled=right_partitioned,
        )
    elif output_count == 1 or (left_partitioned and right_partitioned):
        # Partition-wise join
        left_name = get_key_name(left)
        right_name = get_key_name(right)
        return {
            key: (
                ir.do_evaluate,
                *ir._non_child_args,
                (left_name, i),
                (right_name, i),
            )
            for i, key in enumerate(partition_info[ir].keys(ir))
        }
    else:
        # Broadcast join
        left_parts = partition_info[left]
        right_parts = partition_info[right]
        if left_parts.count >= right_parts.count:
            small_side = "Right"
            small_name = get_key_name(right)
            small_size = partition_info[right].count
            large_name = get_key_name(left)
            large_on = ir.left_on
        else:
            small_side = "Left"
            small_name = get_key_name(left)
            small_size = partition_info[left].count
            large_name = get_key_name(right)
            large_on = ir.right_on

        graph: MutableMapping[Any, Any] = {}

        out_name = get_key_name(ir)
        out_size = partition_info[ir].count
        split_name = f"split-{out_name}"
        getit_name = f"getit-{out_name}"
        inter_name = f"inter-{out_name}"

        # Split each large partition if we have
        # multiple small partitions (unless this
        # is an inner join)
        split_large = ir.options[0] != "Inner" and small_size > 1

        for part_out in range(out_size):
            if split_large:
                graph[(split_name, part_out)] = (
                    _hash_partition_dataframe,
                    (large_name, part_out),
                    part_out,
                    small_size,
                    None,
                    large_on,
                )

            _concat_list = []
            for j in range(small_size):
                left_key: tuple[str, int] | tuple[str, int, int]
                if split_large:
                    left_key = (getit_name, part_out, j)
                    graph[left_key] = (operator.getitem, (split_name, part_out), j)
                else:
                    left_key = (large_name, part_out)
                join_children = [left_key, (small_name, j)]
                if small_side == "Left":
                    join_children.reverse()

                inter_key = (inter_name, part_out, j)
                graph[(inter_name, part_out, j)] = (
                    ir.do_evaluate,
                    ir.left_on,
                    ir.right_on,
                    ir.options,
                    *join_children,
                )
                _concat_list.append(inter_key)
            if len(_concat_list) == 1:
                graph[(out_name, part_out)] = graph.pop(_concat_list[0])
            else:
                graph[(out_name, part_out)] = (_concat, *_concat_list)

        return graph
