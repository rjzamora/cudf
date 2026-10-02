# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pytest
from utils import assert_table_eq

import pylibcudf as plc


@pytest.fixture
def left():
    return pa.Table.from_arrays(
        [[0, 1, 2, 100], [3, 4, 5, None]],
        schema=pa.schema({"a": pa.int32(), "b": pa.int32()}),
    )


@pytest.fixture
def right():
    return pa.Table.from_arrays(
        [[-1, -2, 0, 1, -3], [10, 3, 4, 5, None]],
        schema=pa.schema({"c": pa.int32(), "d": pa.int32()}),
    )


@pytest.fixture
def expr():
    return plc.expressions.Operation(
        plc.expressions.ASTOperator.LESS,
        plc.expressions.ColumnReference(
            0, plc.expressions.TableReference.LEFT
        ),
        plc.expressions.ColumnReference(
            0, plc.expressions.TableReference.RIGHT
        ),
    )


def test_cross_join(left, right):
    # Remove the nulls so the calculation of the expected result works
    left = left[:-1]
    right = right[:-1]
    pleft = plc.Table.from_arrow(left)
    pright = plc.Table.from_arrow(right)

    expect = pa.Table.from_arrays(
        [
            *(np.repeat(c.to_numpy(), len(right)) for c in left.columns),
            *(np.tile(c.to_numpy(), len(left)) for c in right.columns),
        ],
        names=["a", "b", "c", "d"],
    )

    got = plc.join.cross_join(pleft, pright)

    assert_table_eq(expect, got)


sentinel = np.iinfo(np.int32).min


def test_asof_join_backward():
    left = plc.Table.from_arrow(
        pa.table(
            {
                "by": ["a", "a", "a", "b", "b", "c"],
                "on": pa.array([0, 1, 2, 4, 6, 9], type=pa.int32()),
            }
        )
    )
    right = plc.Table.from_arrow(
        pa.table(
            {
                "by": ["a", "a", "b", "b"],
                "on": pa.array([1, 3, 2, 5], type=pa.int32()),
            }
        )
    )

    got = plc.join.AsofJoin(
        plc.Table(right.columns()[:1]), right.columns()[1]
    ).join(
        plc.Table(left.columns()[:1]),
        left.columns()[1],
        plc.join.AsofJoinStrategy.BACKWARD,
        True,
    )

    assert got.to_arrow().to_pylist() == [sentinel, 0, 0, 2, 3, sentinel]


@pytest.mark.parametrize(
    "join_type,expect_left,expect_right",
    [
        (plc.join.conditional_inner_join, {0}, {3}),
        (plc.join.conditional_left_join, {0, 1, 2, 3}, {3, sentinel}),
        (
            plc.join.conditional_full_join,
            {0, 1, 2, 3, sentinel},
            {0, 1, 2, 3, 4, sentinel},
        ),
    ],
    ids=["inner", "left", "full"],
)
def test_conditional_join(
    left, right, expr, join_type, expect_left, expect_right
):
    pleft = plc.Table.from_arrow(left)
    pright = plc.Table.from_arrow(right)

    g_left, g_right = map(
        lambda self: self.to_arrow(), join_type(pleft, pright, expr)
    )

    assert set(g_left.to_pylist()) == expect_left
    assert set(g_right.to_pylist()) == expect_right


@pytest.mark.parametrize(
    "join_type,expect",
    [
        (plc.join.conditional_left_semi_join, {0}),
        (plc.join.conditional_left_anti_join, {1, 2, 3}),
    ],
    ids=["semi", "anti"],
)
def test_conditional_semianti_join(left, right, expr, join_type, expect):
    pleft = plc.Table.from_arrow(left)
    pright = plc.Table.from_arrow(right)

    g_left = join_type(pleft, pright, expr).to_arrow()

    assert set(g_left.to_pylist()) == expect


@pytest.mark.parametrize(
    "join_type,expect_left,expect_right",
    [
        (plc.join.mixed_inner_join, set(), set()),
        (plc.join.mixed_left_join, {0, 1, 2, 3}, {sentinel}),
        (
            plc.join.mixed_full_join,
            {0, 1, 2, 3, sentinel},
            {0, 1, 2, 3, 4, sentinel},
        ),
    ],
    ids=["inner", "left", "full"],
)
@pytest.mark.parametrize(
    "null_equality",
    [plc.types.NullEquality.EQUAL, plc.types.NullEquality.UNEQUAL],
    ids=["nulls_equal", "nulls_not_equal"],
)
def test_mixed_join(
    left, right, expr, join_type, expect_left, expect_right, null_equality
):
    pleft = plc.Table.from_arrow(left)
    pright = plc.Table.from_arrow(right)

    g_left, g_right = map(
        lambda self: self.to_arrow(),
        join_type(
            plc.Table(pleft.columns()[1:]),
            plc.Table(pright.columns()[1:]),
            pleft,
            pright,
            expr,
            null_equality,
        ),
    )

    assert set(g_left.to_pylist()) == expect_left
    assert set(g_right.to_pylist()) == expect_right


@pytest.mark.parametrize(
    "join_type,expect",
    [
        (plc.join.mixed_left_semi_join, set()),
        (plc.join.mixed_left_anti_join, {0, 1, 2, 3}),
    ],
    ids=["semi", "anti"],
)
@pytest.mark.parametrize(
    "null_equality",
    [plc.types.NullEquality.EQUAL, plc.types.NullEquality.UNEQUAL],
    ids=["nulls_equal", "nulls_not_equal"],
)
def test_mixed_semianti_join(
    left, right, expr, join_type, expect, null_equality
):
    pleft = plc.Table.from_arrow(left)
    pright = plc.Table.from_arrow(right)

    g_left = join_type(
        plc.Table(pleft.columns()[1:]),
        plc.Table(pright.columns()[1:]),
        pleft,
        pright,
        expr,
        null_equality,
    ).to_arrow()

    assert set(g_left.to_pylist()) == expect
