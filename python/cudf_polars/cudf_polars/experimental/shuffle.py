# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffle Logic."""

from __future__ import annotations

import operator
from collections import defaultdict
from typing import TYPE_CHECKING, Any, TypedDict

import pylibcudf as plc
import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.containers import Column, DataFrame
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.ir import IR
from cudf_polars.experimental.base import get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks, lower_ir_node
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.experimental.parallel import PartitionInfo
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import ConfigOptions


# Supported shuffle methods
_SHUFFLE_METHODS = ("rapidsmpf", "tasks")


_LOCAL_SHUFFLES: dict[int, defaultdict[int, list[DataFrame]]] = {}


class ShuffleOptions(TypedDict):
    """RapidsMPF shuffling options."""

    on: Sequence[str]
    column_names: Sequence[str]


# Experimental rapidsmpf shuffler integration
class RMPFIntegration:  # pragma: no cover
    """cuDF-Polars protocol for rapidsmpf shuffler."""

    @staticmethod
    def insert_partition(
        df: DataFrame,
        partition_id: int,  # Not currently used
        partition_count: int,
        shuffler: Any,
        options: ShuffleOptions,
        *other: Any,
    ) -> None:
        """Add cudf-polars DataFrame chunks to an RMP shuffler."""
        from rapidsmpf.shuffler import partition_and_pack

        on = options["on"]
        assert not other, f"Unexpected arguments: {other}"
        columns_to_hash = tuple(df.column_names.index(val) for val in on)
        packed_inputs = partition_and_pack(
            df.table,
            columns_to_hash=columns_to_hash,
            num_partitions=partition_count,
            stream=DEFAULT_STREAM,
            device_mr=rmm.mr.get_current_device_resource(),
        )
        shuffler.insert_chunks(packed_inputs)

    @staticmethod
    def extract_partition(
        partition_id: int,
        shuffler: Any,
        options: ShuffleOptions,
    ) -> DataFrame:
        """Extract a finished partition from the RMP shuffler."""
        from rapidsmpf.shuffler import unpack_and_concat

        shuffler.wait_on(partition_id)
        column_names = options["column_names"]
        return DataFrame.from_table(
            unpack_and_concat(
                shuffler.extract(partition_id),
                stream=DEFAULT_STREAM,
                device_mr=rmm.mr.get_current_device_resource(),
            ),
            column_names,
        )


class Shuffle(IR):
    """
    Shuffle multi-partition data.

    Notes
    -----
    Only hash-based partitioning is supported (for now).
    """

    __slots__ = ("config_options", "keys")
    _non_child = ("schema", "keys", "config_options")
    keys: tuple[NamedExpr, ...]
    """Keys to shuffle on."""
    config_options: ConfigOptions
    """Configuration options."""

    def __init__(
        self,
        schema: Schema,
        keys: tuple[NamedExpr, ...],
        config_options: ConfigOptions,
        df: IR,
    ):
        self.schema = schema
        self.keys = keys
        self.config_options = config_options
        self._non_child_args = (schema, keys, config_options)
        self.children = (df,)

    @classmethod
    def do_evaluate(
        cls,
        schema: Schema,
        keys: tuple[NamedExpr, ...],
        config_options: ConfigOptions,
        df: DataFrame,
    ) -> DataFrame:  # pragma: no cover
        """Evaluate and return a dataframe."""
        # Single-partition Shuffle evaluation is a no-op
        return df


def _insert_dataframe(
    df: DataFrame,
    keys: tuple[NamedExpr, ...],
    count: int,
    shuffle_id: int,
    first: bool,  # noqa: FBT001
) -> list[str] | None:
    # First task should return column names
    return_val = df.column_names if first else None

    if shuffle_id not in _LOCAL_SHUFFLES:
        _LOCAL_SHUFFLES[shuffle_id] = defaultdict(list)

    if df.num_rows == 0:
        # Fast path for empty DataFrame
        for i in range(count):
            _LOCAL_SHUFFLES[shuffle_id][i].append(df.table)
        return return_val

    # Hash the specified keys to calculate the output
    # partition for each row
    partition_map = plc.binaryop.binary_operation(
        plc.hashing.murmurhash3_x86_32(
            DataFrame([expr.evaluate(df) for expr in keys]).table
        ),
        plc.Scalar.from_py(count, plc.DataType(plc.TypeId.UINT32)),
        plc.binaryop.BinaryOperator.PYMOD,
        plc.types.DataType(plc.types.TypeId.UINT32),
    )

    # Apply partitioning
    t, offsets = plc.partitioning.partition(
        df.table,
        partition_map,
        count,
    )

    # Store arrow slices
    t_pa = plc.interop.to_arrow(t)
    for i in range(len(offsets) - 1):
        offset = offsets[i]
        size = offsets[i + 1] - offset
        _LOCAL_SHUFFLES[shuffle_id][i].append(t_pa.slice(offset, size))

    # # Store pylibcudf slices
    # # Split and return the partitioned result
    # for i, split in enumerate(plc.copying.split(t, offsets[1:-1])):
    #     _LOCAL_SHUFFLES[shuffle_id][i].append(split)

    return return_val


def _barrier(column_names: list[str], *args: Any) -> list[str]:
    return column_names


def _extract_dataframe(pid: int, shuffle_id: int, column_names: list[str]) -> DataFrame:
    import pyarrow as pa

    return DataFrame(
        Column(c, name=name)
        for c, name in zip(
            plc.Table.from_arrow(
                pa.concat_tables(_LOCAL_SHUFFLES[shuffle_id].pop(pid))
            ).columns(),
            column_names,
            strict=True,
        )
    )

    # # Storing plc tables
    # return DataFrame.from_table(
    #     plc.concatenate.concatenate(_LOCAL_SHUFFLES[shuffle_id].pop(pid)),
    #     column_names,
    # )


def _partition_dataframe(
    df: DataFrame,
    keys: tuple[NamedExpr, ...],
    count: int,
) -> dict[int, DataFrame]:
    """
    Partition an input DataFrame for shuffling.

    Notes
    -----
    This utility only supports hash partitioning (for now).

    Parameters
    ----------
    df
        DataFrame to partition.
    keys
        Shuffle key(s).
    count
        Total number of output partitions.

    Returns
    -------
    A dictionary mapping between int partition indices and
    DataFrame fragments.
    """
    if df.num_rows == 0:
        # Fast path for empty DataFrame
        return {i: df for i in range(count)}

    # Hash the specified keys to calculate the output
    # partition for each row
    partition_map = plc.binaryop.binary_operation(
        plc.hashing.murmurhash3_x86_32(
            DataFrame([expr.evaluate(df) for expr in keys]).table
        ),
        plc.Scalar.from_py(count, plc.DataType(plc.TypeId.UINT32)),
        plc.binaryop.BinaryOperator.PYMOD,
        plc.types.DataType(plc.types.TypeId.UINT32),
    )

    # Apply partitioning
    t, offsets = plc.partitioning.partition(
        df.table,
        partition_map,
        count,
    )

    # Split and return the partitioned result
    return {
        i: DataFrame.from_table(
            split,
            df.column_names,
        )
        for i, split in enumerate(plc.copying.split(t, offsets[1:-1]))
    }


def _simple_shuffle_graph(
    name_in: str,
    name_out: str,
    keys: tuple[NamedExpr, ...],
    count_in: int,
    count_out: int,
) -> MutableMapping[Any, Any]:
    """Make a simple all-to-all shuffle graph."""
    split_name = f"split-{name_out}"
    inter_name = f"inter-{name_out}"
    barrier_name = f"barrier-{name_out}"

    graph: MutableMapping[Any, Any] = {}
    if True and count_in > 2 and count_out > 2:
        shuffle_id = hash(name_in)
        first = True
        for part_in in range(count_in):
            graph[(split_name, part_in)] = (
                _insert_dataframe,
                (name_in, part_in),
                keys,
                count_out,
                shuffle_id,
                first,
            )
            first = False
        graph[barrier_name] = (_barrier, *graph.keys())
        for part_out in range(count_out):
            graph[(name_out, part_out)] = (
                _extract_dataframe,
                part_out,
                shuffle_id,
                barrier_name,
            )
    else:
        for part_out in range(count_out):
            _concat_list = []
            for part_in in range(count_in):
                graph[(split_name, part_in)] = (
                    _partition_dataframe,
                    (name_in, part_in),
                    keys,
                    count_out,
                )
                _concat_list.append((inter_name, part_out, part_in))
                graph[_concat_list[-1]] = (
                    operator.getitem,
                    (split_name, part_in),
                    part_out,
                )
            graph[(name_out, part_out)] = (_concat, *_concat_list)
    return graph


@lower_ir_node.register(Shuffle)
def _(
    ir: Shuffle, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Simple lower_ir_node handling for the default hash-based shuffle.
    # More-complex logic (e.g. joining and sorting) should
    # be handled separately.
    from cudf_polars.experimental.parallel import PartitionInfo

    (child,) = ir.children

    new_child, pi = rec(child)
    if pi[new_child].count == 1 or ir.keys == pi[new_child].partitioned_on:
        # Already shuffled
        return new_child, pi
    new_node = ir.reconstruct([new_child])
    pi[new_node] = PartitionInfo(
        # Default shuffle preserves partition count
        count=pi[new_child].count,
        # Add partitioned_on info
        partitioned_on=ir.keys,
    )
    return new_node, pi


@generate_ir_tasks.register(Shuffle)
def _(
    ir: Shuffle, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    # Extract "shuffle_method" configuration
    assert ir.config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'generate_ir_tasks'"
    )

    shuffle_method = ir.config_options.executor.shuffle_method

    # Try using rapidsmpf shuffler if we have "simple" shuffle
    # keys, and the "shuffle_method" config is set to "rapidsmpf"
    _keys: list[Col]
    if shuffle_method in (None, "rapidsmpf") and len(
        _keys := [ne.value for ne in ir.keys if isinstance(ne.value, Col)]
    ) == len(ir.keys):  # pragma: no cover
        shuffle_on = [k.name for k in _keys]
        try:
            from rapidsmpf.integrations.dask import rapidsmpf_shuffle_graph

            return rapidsmpf_shuffle_graph(
                get_key_name(ir.children[0]),
                get_key_name(ir),
                partition_info[ir.children[0]].count,
                partition_info[ir].count,
                RMPFIntegration,
                {"on": shuffle_on, "column_names": list(ir.schema.keys())},
            )
        except (ImportError, ValueError) as err:
            # ImportError: rapidsmpf is not installed
            # ValueError: rapidsmpf couldn't find a distributed client
            if shuffle_method == "rapidsmpf":
                # Only raise an error if the user specifically
                # set the shuffle method to "rapidsmpf"
                raise ValueError(
                    "Rapidsmp is not installed correctly or the current "
                    "Dask cluster does not support rapidsmpf shuffling."
                ) from err

    # Simple task-based fall-back
    return _simple_shuffle_graph(
        get_key_name(ir.children[0]),
        get_key_name(ir),
        ir.keys,
        partition_info[ir.children[0]].count,
        partition_info[ir].count,
    )
