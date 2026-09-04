"""Assembly, checked on generators whose output is known by construction.

The generators here are deliberately trivial - fire every Nth bar, fire at a
list of hours - because what is under test is the assembly, not the signal.
Two of them are not trivial: one peeks at the newest bar it is handed to prove
it can never see past its own signal, and one openly reads ten bars into the
future to prove `build` refuses it.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from falsify import events, schema

SYMBOL = "EURUSD"
TIMEFRAME = "M1"
ATR = 1.0
SOURCE = "range_breakouts_v1"
SOURCE_VERSION = "0.1.0"

# Fire on every 50th bar. 1000 bars is 20 signals, and the last bar is a
# firing bar, which matters for the isolation test below.
EVERY = 50
N_BARS = 1000

BAR_START = datetime(2024, 1, 1, tzinfo=UTC)

BAR_SCHEMA = {
    "ts": pl.Datetime("us", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
}

SIGNAL_SCHEMA = {
    "signal_ts": pl.Datetime("us", time_zone="UTC"),
    "direction": pl.Int8,
    "ref_px": pl.Float64,
    "atr_at_signal": pl.Float64,
}
FEATURE_SCHEMA = {**SIGNAL_SCHEMA, "f_rejects": pl.Int64}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_bars(n: int = N_BARS, *, step: timedelta = timedelta(minutes=1)) -> pl.DataFrame:
    """A drifting wave, so excursions run both ways and the sign conventions
    in `schema.validate` are actually exercised rather than trivially true."""
    closes = [100.0 + 2.0 * math.sin(i / 6.0) + 0.001 * i for i in range(n)]
    return pl.DataFrame(
        {
            "ts": [BAR_START + i * step for i in range(n)],
            "open": closes,
            "high": [c + 0.05 for c in closes],
            "low": [c - 0.05 for c in closes],
            "close": closes,
            "symbol": [SYMBOL] * n,
            "timeframe": [TIMEFRAME] * n,
        },
        schema=BAR_SCHEMA,
    )


def every_50th(bars: pl.DataFrame) -> pl.DataFrame:
    """Fire on the close of every 50th bar, alternating side, carrying one
    feature.

    Causal by construction: which bars fire, and the ordinal each one gets,
    depend only on position counted from the start of the frame. A prefix of
    the bars therefore yields exactly the matching prefix of the signals, which
    is what lets `build` reproduce any single row from truncated bars.
    """
    frame = bars.sort("ts").with_row_index("i")
    fired = frame.filter((pl.col("i") + 1) % EVERY == 0)
    ordinals = [int(i + 1) // EVERY for i in fired["i"].to_list()]
    return pl.DataFrame(
        {
            "signal_ts": fired["ts"].alias("signal_ts"),
            "direction": [1 if k % 2 else -1 for k in ordinals],
            "ref_px": fired["close"].alias("ref_px"),
            "atr_at_signal": [ATR] * fired.height,
            "f_rejects": [k % 4 for k in ordinals],
        },
        schema=FEATURE_SCHEMA,
    )


def expected_signal_rows(bars: pl.DataFrame) -> list[dict]:
    """The same signals worked out independently of the generator."""
    rows = []
    for k, i in enumerate(range(EVERY - 1, bars.height, EVERY), start=1):
        bar = bars.row(i, named=True)
        rows.append(
            {
                "signal_ts": bar["ts"],
                "direction": 1 if k % 2 else -1,
                "ref_px": bar["close"],
                "f_rejects": k % 4,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 1. A build from end to end
# ---------------------------------------------------------------------------
def test_basic_build():
    """Twenty alternating signals with one feature, assembled and validated."""
    bars = make_bars()
    out = events.build(bars, every_50th, SOURCE, SOURCE_VERSION)

    expected = expected_signal_rows(bars)
    assert out.height == N_BARS // EVERY == len(expected) == 20

    # Every base column is present, with the dtype schema.py declares.
    for col, dtype in schema.BASE_COLUMNS.items():
        assert col in out.columns, f"missing base column {col}"
        assert out.schema[col] == dtype, f"{col}: expected {dtype}"

    result = schema.validate(out)
    assert result.ok, result.problems

    # Identity: a sha256 hex digest per event, all distinct.
    ids = out["event_id"].to_list()
    assert len(set(ids)) == out.height
    assert all(len(i) == 64 and set(i) <= set("0123456789abcdef") for i in ids)

    assert out["source"].unique().to_list() == [SOURCE]
    assert out["source_version"].unique().to_list() == [SOURCE_VERSION]
    assert out["symbol"].unique().to_list() == [SYMBOL]
    assert out["timeframe"].unique().to_list() == [TIMEFRAME]

    # The signals themselves survive assembly unchanged.
    assert out["signal_ts"].to_list() == [r["signal_ts"] for r in expected]
    assert out["direction"].to_list() == [r["direction"] for r in expected]
    assert out["ref_px"].to_list() == pytest.approx([r["ref_px"] for r in expected])
    assert set(out["direction"].to_list()) == {-1, 1}

    # The feature rides through untouched, dtype included.
    assert out["f_rejects"].to_list() == [r["f_rejects"] for r in expected]
    assert out.schema["f_rejects"] == pl.Int64
    assert schema.feature_columns(out) == ["f_rejects"]

    # Reference columns.
    assert out["spread_at_signal"].to_list() == [0.0] * out.height
    assert out.schema["spread_at_signal"] == pl.Float64
    assert out["atr_at_signal"].to_list() == pytest.approx([ATR] * out.height)
    assert out["session"].null_count() == 0

    # Outcomes got attached. The last signal fires on the last bar, so it has
    # no forward window at all and reads null, per forward.py, rather than 0.
    assert out["fwd_1"].to_list()[-1] is None
    assert out["fwd_1"].head(out.height - 1).null_count() == 0
    assert out["mfe_20"].head(out.height - 1).null_count() == 0
    assert out["mae_20"].head(out.height - 1).null_count() == 0


def test_event_id_is_stable_across_builds():
    """A content hash, not a row number: the same event built twice, from a
    different slice of bars, keeps its identity. prereg pins a registration to
    a dataset by hashing these, so a drifting id would silently invalidate it."""
    bars = make_bars()
    first = events.build(bars, every_50th, SOURCE, SOURCE_VERSION)
    again = events.build(bars, every_50th, SOURCE, SOURCE_VERSION)
    assert first["event_id"].to_list() == again["event_id"].to_list()

    # The same generator over the first half produces the same ids for the
    # events it shares with the full run.
    half = events.build(bars.head(N_BARS // 2), every_50th, SOURCE, SOURCE_VERSION)
    assert half["event_id"].to_list() == first["event_id"].to_list()[: half.height]

    # A different source is a different event, on the same bar.
    other = events.build(bars, every_50th, "other_source", SOURCE_VERSION)
    assert set(other["event_id"]).isdisjoint(first["event_id"])


# ---------------------------------------------------------------------------
# 2. Session tagging
# ---------------------------------------------------------------------------
def fire_at_hours(hours):
    """A generator firing once on each bar whose UTC hour is in `hours`,
    restricted to the first day so each hour produces exactly one event."""
    wanted = sorted(hours)
    first_day_end = BAR_START + timedelta(days=1)

    def generator(bars: pl.DataFrame) -> pl.DataFrame:
        fired = bars.sort("ts").filter(
            pl.col("ts").dt.hour().is_in(wanted) & (pl.col("ts") < first_day_end)
        )
        return pl.DataFrame(
            {
                "signal_ts": fired["ts"].alias("signal_ts"),
                "direction": [1] * fired.height,
                "ref_px": fired["close"].alias("ref_px"),
                "atr_at_signal": [ATR] * fired.height,
            },
            schema=SIGNAL_SCHEMA,
        )

    return generator


def sessions_for(hours) -> dict[int, str]:
    """Hour of day to session label, straight off a built event table."""
    bars = make_bars(48, step=timedelta(hours=1))
    out = events.build(bars, fire_at_hours(hours), SOURCE, SOURCE_VERSION)
    assert out.height == len(hours)
    assert out.schema["session"] == pl.Utf8
    return dict(zip(out["signal_ts"].dt.hour().to_list(), out["session"].to_list()))


def test_session_tagging():
    """One signal in the middle of each session gets that session's label."""
    assert sessions_for([2, 9, 13, 18, 22]) == {
        2: "asia",
        9: "london",
        13: "overlap",
        18: "ny",
        22: "off",
    }


def test_session_boundaries_are_half_open():
    """Each session runs from its opening hour up to, but not including, the
    next one - so 08:00 is london rather than the last hour of asia."""
    assert sessions_for([0, 7, 8, 11, 12, 15, 16, 20, 21, 23]) == {
        0: "asia",
        7: "asia",
        8: "london",
        11: "london",
        12: "overlap",
        15: "overlap",
        16: "ny",
        20: "ny",
        21: "off",
        23: "off",
    }


# ---------------------------------------------------------------------------
# 3. Generator isolation
# ---------------------------------------------------------------------------
def test_generator_never_sees_a_bar_after_its_signal():
    """The generator peeks at the last row of every frame it is handed - the
    most future-looking thing a generator can do - and finds nothing past the
    signal it is currently deciding.

    Older signals in the same frame are re-emissions of decisions already made;
    the row `build` keeps from each call is the one on the frontier, and the
    assertion below is about that row.
    """
    bars = make_bars()
    frames: list[pl.DataFrame] = []

    def peeking(frame: pl.DataFrame) -> pl.DataFrame:
        frame = frame.sort("ts")
        frames.append(frame)
        signals = every_50th(frame)

        if signals.height:
            signal_ts = signals["signal_ts"].max()
            # The peek. The newest bar available is the signal bar itself.
            assert frame.row(frame.height - 1, named=True)["ts"] == signal_ts
            assert frame.filter(pl.col("ts") > signal_ts).height == 0

        return signals

    out = events.build(bars, peeking, SOURCE, SOURCE_VERSION)
    assert out.height == 20

    # The assertion above is worth nothing unless it ran on truncated frames,
    # so pin down what build actually passed: one enumeration call over
    # everything, then one confirming call per event, each ending on its own
    # signal bar.
    assert len(frames) == 1 + out.height
    assert frames[0].height == N_BARS

    confirming = frames[1:]
    assert [f.height for f in confirming] == [EVERY * k for k in range(1, 21)]
    for frame, signal_ts in zip(confirming, out["signal_ts"].to_list()):
        assert frame["ts"].max() == signal_ts
        assert frame.filter(pl.col("ts") > signal_ts).height == 0


def test_a_generator_that_reads_ahead_is_rejected():
    """The other half of the guarantee: a signal that only exists because the
    next ten bars were visible cannot be reproduced from bars ending at its own
    signal_ts, and build says so instead of quietly keeping it."""

    def reads_ahead(bars: pl.DataFrame) -> pl.DataFrame:
        frame = (
            bars.sort("ts")
            .with_row_index("i")
            .with_columns(pl.col("close").shift(-10).alias("close_ahead"))
        )
        # Fires only where the bar ten ahead closed higher - unknowable at the
        # signal bar, and null once the frame stops there.
        fired = frame.filter(
            ((pl.col("i") + 1) % EVERY == 0)
            & (pl.col("close_ahead") > pl.col("close"))
        )
        return pl.DataFrame(
            {
                "signal_ts": fired["ts"].alias("signal_ts"),
                "direction": [1] * fired.height,
                "ref_px": fired["close"].alias("ref_px"),
                "atr_at_signal": [ATR] * fired.height,
            },
            schema=SIGNAL_SCHEMA,
        )

    bars = make_bars()
    assert reads_ahead(bars).height > 0, "fixture should produce candidates"

    with pytest.raises(ValueError, match="had not happened yet"):
        events.build(bars, reads_ahead, SOURCE, SOURCE_VERSION)


# ---------------------------------------------------------------------------
# Contract checks
# ---------------------------------------------------------------------------
def test_a_generator_that_fires_nothing_builds_an_empty_table():
    """No signals is a legitimate answer, not an error. The frame still has to
    be schema-shaped, or downstream code discovers the empty case the hard
    way."""

    def never(bars: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame(schema=SIGNAL_SCHEMA)

    out = events.build(make_bars(), never, SOURCE, SOURCE_VERSION)

    assert out.height == 0
    assert schema.validate(out).ok
    for col, dtype in schema.BASE_COLUMNS.items():
        assert out.schema[col] == dtype


def test_generator_columns_that_are_neither_base_nor_features_are_rejected():
    """The f_ prefix is the whole defence against a column that looks like an
    outcome being read as one, so an unprefixed extra is a build failure."""

    def leaky_column(bars: pl.DataFrame) -> pl.DataFrame:
        return every_50th(bars).with_columns(pl.lit(1.0).alias("edge"))

    with pytest.raises(ValueError, match="neither base nor"):
        events.build(make_bars(), leaky_column, SOURCE, SOURCE_VERSION)


def test_missing_generator_columns_are_named():
    def no_atr(bars: pl.DataFrame) -> pl.DataFrame:
        return every_50th(bars).drop("atr_at_signal")

    with pytest.raises(ValueError, match="atr_at_signal"):
        events.build(make_bars(), no_atr, SOURCE, SOURCE_VERSION)
