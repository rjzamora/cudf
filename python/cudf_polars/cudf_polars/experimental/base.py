# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition base classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, Sequence

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR
    from cudf_polars.dsl.nodebase import Node


class PartitionInfo:
    """Partitioning information."""

    __slots__ = ("count", "partitioned_on")
    count: int
    """Partition count."""
    partitioned_on: tuple[NamedExpr, ...]
    """Columns the data is hash-partitioned on."""

    def __init__(
        self,
        count: int,
        partitioned_on: tuple[NamedExpr, ...] = (),
    ):
        self.count = count
        self.partitioned_on = partitioned_on

    def keys(self, node: Node) -> Iterator[tuple[str, int]]:
        """Return the partitioned keys for a given node."""
        name = get_key_name(node)
        yield from ((name, i) for i in range(self.count))

    def __rich_repr__(self) -> Generator[Any, None, None]:
        """Formatting for rich.pretty.pprint."""
        yield "count", self.count
        yield "partitioned_on", self.partitioned_on


def get_key_name(node: Node) -> str:
    """Generate the key name for a Node."""
    return f"{type(node).__name__.lower()}-{hash(node)}"


class UniqueStats:
    """
    Unique-value statistics.

    Parameters
    ----------
    count
        Unique-value count.
    fraction
        Unique-value fraction.
    """

    __slots__ = ("count", "fraction")

    def __init__(
        self,
        *,
        count: int | None = None,
        fraction: float | None = None,
    ):
        self.count = count
        self.fraction = fraction


class DataSourceStats:
    """Datasource statistics sampler."""

    @property
    def cardinality(self) -> int | None:
        """Datasource cardinality estimate."""
        return None

    @property
    def exact_cardinality(self) -> bool:
        """Whether the cardinality estimate is exact."""
        return False

    def mean_storage_size(self, column: str) -> int | None:
        """Return the average column size across all files."""
        return None

    def unique_stats(self, column: str) -> UniqueStats | None:
        """Return unique-value statistics for a column."""
        return None

    def add_keys(self, columns: Sequence[str]) -> None:
        """Specify column names needing unique-value statistics."""
        raise NotImplementedError()


class ColumnStats:
    """
    Column statistics.

    Parameters
    ----------
    name
        Column name.
    source
        Datasource statistics.
    source_name
        Source-column name.
    unique_count
        Unique-count estimate.
    """

    __slots__ = ("name", "source", "source_name", "unique_count")

    def __init__(
        self,
        *,
        name: str | None = None,
        source: DataSourceStats | None = None,
        source_name: str | None = None,
        unique_count: int | None = None,
    ) -> None:
        self.name = name
        self.source = source
        self.source_name = source_name
        self.unique_count = unique_count


class StatsCollector:
    """Column statistics collector."""

    __slots__ = ("cardinality", "column_stats")

    def __init__(self) -> None:
        self.cardinality: dict[IR, int] = {}
        self.column_stats: dict[IR, dict[str, ColumnStats]] = {}
