# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf as gd
from cudf.tests.utils import assert_eq


def make_frames(index=None, nulls="none"):
    df = pd.DataFrame(
        {
            "x": range(10),
            "y": list(map(float, range(10))),
            "z": list("abcde") * 2,
        }
    )
    df.z = df.z.astype("category")
    df2 = pd.DataFrame(
        {
            "x": range(10, 20),
            "y": list(map(float, range(10, 20))),
            "z": list("edcba") * 2,
        }
    )
    df2.z = df2.z.astype("category")
    if nulls == "all":
        df.y = np.full_like(df.y, np.nan)
        df2.y = np.full_like(df2.y, np.nan)
    if nulls == "some":
        mask = np.arange(10)
        np.random.shuffle(mask)
        mask = mask[:5]
        df.y.loc[mask] = np.nan
        df2.y.loc[mask] = np.nan
    gdf = gd.DataFrame.from_pandas(df)
    gdf2 = gd.DataFrame.from_pandas(df2)
    if index:
        df = df.set_index(index)
        df2 = df2.set_index(index)
        gdf = gdf.set_index(index)
        gdf2 = gdf2.set_index(index)
    return df, df2, gdf, gdf2


@pytest.mark.parametrize("nulls", ["none", "some", "all"])
@pytest.mark.parametrize("index", [False, "z", "y"])
@pytest.mark.parametrize("axis", [0, "index"])
def test_concat_dataframe(index, nulls, axis):
    if index == "y" and nulls in ("some", "all"):
        pytest.skip("nulls in columns, dont index")
    df, df2, gdf, gdf2 = make_frames(index, nulls=nulls)
    # Make empty frame
    gdf_empty1 = gdf2[:0]
    assert len(gdf_empty1) == 0
    df_empty1 = gdf_empty1.to_pandas()

    # DataFrame
    res = gd.concat([gdf, gdf2, gdf, gdf_empty1], axis=axis).to_pandas()
    sol = pd.concat([df, df2, df, df_empty1], axis=axis)
    pd.util.testing.assert_frame_equal(
        res, sol, check_names=False, check_categorical=False
    )

    # Series
    for c in [i for i in ("x", "y", "z") if i != index]:
        res = gd.concat([gdf[c], gdf2[c], gdf[c]], axis=axis).to_pandas()
        sol = pd.concat([df[c], df2[c], df[c]], axis=axis)
        pd.util.testing.assert_series_equal(
            res, sol, check_names=False, check_categorical=False
        )

    # Index
    res = gd.concat([gdf.index, gdf2.index], axis=axis).to_pandas()
    sol = df.index.append(df2.index)
    pd.util.testing.assert_index_equal(
        res, sol, check_names=False, check_categorical=False
    )


@pytest.mark.parametrize(
    "values",
    [["foo", "bar"], [1.0, 2.0], pd.Series(["one", "two"], dtype="category")],
)
def test_concat_all_nulls(values):
    pa = pd.Series(values)
    pb = pd.Series([None])
    ps = pd.concat([pa, pb])

    ga = gd.Series(values)
    gb = gd.Series([None])
    gs = gd.concat([ga, gb])

    assert_eq(ps, gs, check_dtype=False, check_categorical=False)


def test_concat_errors():
    df, df2, gdf, gdf2 = make_frames()

    # No objs
    with pytest.raises(ValueError):
        gd.concat([])

    # Mismatched types
    with pytest.raises(ValueError):
        gd.concat([gdf, gdf.x])

    # Unknown type
    with pytest.raises(ValueError):
        gd.concat(["bar", "foo"])

    # Mismatched index dtypes
    gdf3 = gdf2.copy()
    del gdf3["z"]
    gdf4 = gdf2.set_index("z")
    with pytest.raises(ValueError):
        gd.concat([gdf3, gdf4])

    # Bad axis value
    with pytest.raises(ValueError):
        gd.concat([gdf, gdf2], axis="bad_value")


def test_concat_misordered_columns():
    df, df2, gdf, gdf2 = make_frames(False)
    gdf2 = gdf2[["z", "x", "y"]]
    df2 = df2[["z", "x", "y"]]

    res = gd.concat([gdf, gdf2]).to_pandas()
    sol = pd.concat([df, df2], sort=False)

    pd.util.testing.assert_frame_equal(
        res, sol, check_names=False, check_categorical=False
    )


@pytest.mark.parametrize("axis", [1, "columns"])
def test_concat_columns(axis):
    pdf1 = pd.DataFrame(np.random.randint(10, size=(5, 3)), columns=[1, 2, 3])
    pdf2 = pd.DataFrame(
        np.random.randint(10, size=(5, 4)), columns=[4, 5, 6, 7]
    )
    gdf1 = gd.from_pandas(pdf1)
    gdf2 = gd.from_pandas(pdf2)

    expect = pd.concat([pdf1, pdf2], axis=axis)
    got = gd.concat([gdf1, gdf2], axis=axis)

    assert_eq(expect, got)


def test_concat_multiindex_dataframe():
    gdf = gd.DataFrame(
        {
            "w": np.arange(4),
            "x": np.arange(4),
            "y": np.arange(4),
            "z": np.arange(4),
        }
    )
    gdg = gdf.groupby(["w", "x"]).min()
    pdg = gdg.to_pandas()
    pdg1 = pdg.iloc[:, :1]
    pdg2 = pdg.iloc[:, 1:]
    gdg1 = gd.from_pandas(pdg1)
    gdg2 = gd.from_pandas(pdg2)
    assert_eq(
        gd.concat([gdg1, gdg2]).astype("float64"), pd.concat([pdg1, pdg2])
    )
    assert_eq(gd.concat([gdg1, gdg2], axis=1), pd.concat([pdg1, pdg2], axis=1))


def test_concat_multiindex_series():
    gdf = gd.DataFrame(
        {
            "w": np.arange(4),
            "x": np.arange(4),
            "y": np.arange(4),
            "z": np.arange(4),
        }
    )
    gdg = gdf.groupby(["w", "x"]).min()
    pdg = gdg.to_pandas()
    pdg1 = pdg["y"]
    pdg2 = pdg["z"]
    gdg1 = gd.from_pandas(pdg1)
    gdg2 = gd.from_pandas(pdg2)
    assert_eq(gd.concat([gdg1, gdg2]), pd.concat([pdg1, pdg2]))
    assert_eq(gd.concat([gdg1, gdg2], axis=1), pd.concat([pdg1, pdg2], axis=1))


def test_concat_multiindex_dataframe_and_series():
    gdf = gd.DataFrame(
        {
            "w": np.arange(4),
            "x": np.arange(4),
            "y": np.arange(4),
            "z": np.arange(4),
        }
    )
    gdg = gdf.groupby(["w", "x"]).min()
    pdg = gdg.to_pandas()
    pdg1 = pdg[["y", "z"]]
    pdg2 = pdg["z"]
    pdg2.name = "a"
    gdg1 = gd.from_pandas(pdg1)
    gdg2 = gd.from_pandas(pdg2)
    assert_eq(gd.concat([gdg1, gdg2], axis=1), pd.concat([pdg1, pdg2], axis=1))


def test_concat_multiindex_series_and_dataframe():
    gdf = gd.DataFrame(
        {
            "w": np.arange(4),
            "x": np.arange(4),
            "y": np.arange(4),
            "z": np.arange(4),
        }
    )
    gdg = gdf.groupby(["w", "x"]).min()
    pdg = gdg.to_pandas()
    pdg1 = pdg["z"]
    pdg2 = pdg[["y", "z"]]
    pdg1.name = "a"
    gdg1 = gd.from_pandas(pdg1)
    gdg2 = gd.from_pandas(pdg2)
    assert_eq(gd.concat([gdg1, gdg2], axis=1), pd.concat([pdg1, pdg2], axis=1))


@pytest.mark.parametrize("myindex", ["a", "b"])
def test_concat_string_index_name(myindex):
    # GH-Issue #3420
    data = {"a": [123, 456], "b": ["s1", "s2"]}
    df1 = gd.DataFrame(data).set_index(myindex)
    df2 = df1.copy()
    df3 = gd.concat([df1, df2])

    assert df3.index.name == myindex


@pytest.mark.parametrize("overlap", [False, True])
def test_pandas_concat_compatibility_axis1(overlap):
    d1 = gd.datasets.randomdata(
        3, dtypes={"a": float, "ind": float}
    ).set_index("ind")
    d2 = gd.datasets.randomdata(
        3, dtypes={"b": float, "ind": float}
    ).set_index("ind")
    d3 = gd.datasets.randomdata(
        3, dtypes={"c": float, "ind": float}
    ).set_index("ind")
    d4 = gd.datasets.randomdata(
        3, dtypes={"d": float, "ind": float}
    ).set_index("ind")
    d5 = gd.datasets.randomdata(
        3, dtypes={"e": float, "ind": float}
    ).set_index("ind")

    pd1 = d1.to_pandas()
    pd2 = d2.to_pandas()
    pd3 = d3.to_pandas()
    pd4 = d4.to_pandas()
    pd5 = d5.to_pandas()

    if overlap:
        d6 = d5.rename(columns={"e": "f"})
        d7 = d5.rename(columns={"e": "f"})
        pd6 = d6.to_pandas()
        pd7 = d7.to_pandas()
        expect = pd.concat([pd1, pd2, pd3, pd4, pd5, pd6, pd7], axis=1)
        got = gd.concat([d1, d2, d3, d4, d5, d6, d7], axis=1)
        pytest.xfail(
            "concat does not collapes columns with overlapping indices."
        )
    else:
        expect = pd.concat([pd1, pd2, pd3, pd4, pd5], axis=1)
        got = gd.concat([d1, d2, d3, d4, d5], axis=1)
    print(got)
    print(expect)
    assert_eq(got, expect)
