from __future__ import annotations

import polars as pl
import polars_ds as pds
from polars.testing import assert_frame_equal


def test_replace_non_ascii():
    df = pl.DataFrame({"x": ["mercy", "xbĤ", "ĤŇƏ"]})

    assert_frame_equal(
        df.select(pds.replace_non_ascii("x")), pl.DataFrame({"x": ["mercy", "xb", ""]})
    )

    assert_frame_equal(
        df.select(pds.replace_non_ascii("x", "?")),
        pl.DataFrame({"x": ["mercy", "xb?", "???"]}),
    )

    assert_frame_equal(
        df.select(pds.replace_non_ascii("x", "??")),
        pl.DataFrame({"x": ["mercy", "xb??", "??????"]}),
    )


def test_remove_diacritics():
    df = pl.DataFrame({"x": ["mercy", "mèrcy", "françoise", "über"]})

    assert_frame_equal(
        df.select(pds.remove_diacritics("x")),
        pl.DataFrame({"x": ["mercy", "mercy", "francoise", "uber"]}),
    )


def test_normalize_string():
    df = pl.DataFrame({"x": ["\u0043\u0327"], "y": ["\u00c7"]}).with_columns(
        pl.col("x").eq(pl.col("y")).alias("is_equal"),
        pds.normalize_string("x", "NFC")
        .eq(pds.normalize_string("y", "NFC"))
        .alias("normalized_is_equal"),
    )

    assert df["is_equal"].sum() == 0
    assert df["normalized_is_equal"].sum() == df.height