"""Forward measurement, checked against paths whose answers are known by hand.

Every fixture here is a shape simple enough to compute in your head: a ramp, a
flat line, a V. If a measurement is wrong the arithmetic in the assertion is
the specification, not the code.

Bar 1 is the first bar strictly after signal_ts throughout.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from falsify.forward import measure_forward

SYMBOL = "EURUSD"
TIMEFRAME = "M1"
ATR = 1.0

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
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
}

HORIZONS_TESTED = (1, 3, 5, 10, 20, 50, 100)

BAR_START = datetime(2024, 1, 1, tzinfo=UTC)


def bar_ts(i: int) -> datetime:
    """Timestamp of bar `i`."""
    return BAR_START + timedelta(minutes=i)


def make_bars(closes, *, half_range: float = 0.0) -> pl.DataFrame:
    """Bars at one-minute spacing. `half_range` puts the high and low that far
    either side of the close, so intrabar extremes can be exercised."""
    return pl.DataFrame(
        {
            "ts": [bar_ts(i) for i in range(len(closes))],
            "open": [float(c) for c in closes],
            "high": [float(c) + half_range for c in closes],
            "low": [float(c) - half_range for c in closes],
            "close": [float(c) for c in closes],
            "symbol": [SYMBOL] * len(closes),
            "timeframe": [TIMEFRAME] * len(closes),
        },
        schema=BAR_SCHEMA,
    )


def make_signal(bars: pl.DataFrame, bar_index: int, direction: int) -> pl.DataFrame:
    """One signal fired on the close of `bar_index`, referenced to that close.

    The bar is read back out of the frame rather than reconstructed, so the
    signal timestamp is the same tz-aware UTC value the bars carry and the
    strictly-after matching is exercised on real datetimes.
    """
    bar = bars.row(bar_index, named=True)
    assert bar["ts"] == bar_ts(bar_index)
    assert bar["ts"].utcoffset() == timedelta(0)
    return pl.DataFrame(
        {
            "signal_ts": [bar["ts"]],
            "direction": [direction],
            "ref_px": [bar["close"]],
            "atr_at_signal": [ATR],
            "symbol": [SYMBOL],
            "timeframe": [TIMEFRAME],
        },
        schema=SIGNAL_SCHEMA,
    )


def measure_one(bars: pl.DataFrame, signal: pl.DataFrame) -> dict:
    out = measure_forward(bars, signal)
    assert out.height == 1
    row = out.row(0, named=True)
    # The signal timestamp survives measurement unchanged and tz-aware.
    assert row["signal_ts"] == signal["signal_ts"][0]
    assert row["signal_ts"].utcoffset() == timedelta(0)
    return row


# ---------------------------------------------------------------------------
# 1. Constant uptrend, +1 per bar
# ---------------------------------------------------------------------------
def test_constant_uptrend_long():
    """Price is 100 + i on every bar, so bar h sits exactly h ATR above the
    reference. Every horizon should read h, MFE should track it, and MAE
    should be flat zero because price never trades below where it started."""
    bars = make_bars([100 + i for i in range(200)])
    row = measure_one(bars, make_signal(bars, 0, direction=+1))

    for h in HORIZONS_TESTED:
        assert row[f"fwd_{h}"] == pytest.approx(float(h))
        assert row[f"fwd_{h}"] > 0
        assert row[f"mfe_{h}"] == pytest.approx(float(h))
        # Clamped at zero: the lowest low after the signal is above ref_px.
        assert row[f"mae_{h}"] == pytest.approx(0.0)

    # MFE grows strictly with the horizon.
    mfes = [row[f"mfe_{h}"] for h in HORIZONS_TESTED]
    assert mfes == sorted(mfes)
    assert all(b > a for a, b in itertools.pairwise(mfes))

    # Nothing ever goes against a monotone ramp.
    assert row["max_retrace_20"] == pytest.approx(0.0)
    assert row["bars_to_first_retrace"] == -1
    assert row["rebreak_bar"] == -1
    # No retrace, so there is no extension to measure from one.
    assert row["swing_ext_after_retrace"] is None
    # The lowest low in the window is bar 1, a full ATR *above* ref_px, so the
    # distance to it is negative: there is no swing to put a stop under.
    assert row["swing_stop_dist"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 2. Constant downtrend, -1 per bar
# ---------------------------------------------------------------------------
def test_constant_downtrend_short_positive_long_negative():
    """The same ramp inverted. Direction signing means the short reads exactly
    what the long read in the uptrend, and the long reads its negative."""
    bars = make_bars([100 - i for i in range(200)])

    short = measure_one(bars, make_signal(bars, 0, direction=-1))
    long = measure_one(bars, make_signal(bars, 0, direction=+1))

    for h in HORIZONS_TESTED:
        assert short[f"fwd_{h}"] == pytest.approx(float(h))
        assert short[f"fwd_{h}"] > 0
        assert long[f"fwd_{h}"] == pytest.approx(-float(h))
        assert long[f"fwd_{h}"] < 0

        # The short's favourable extreme is the low; the long's is the high.
        assert short[f"mfe_{h}"] == pytest.approx(float(h))
        assert short[f"mae_{h}"] == pytest.approx(0.0)
        assert long[f"mfe_{h}"] == pytest.approx(0.0)
        assert long[f"mae_{h}"] == pytest.approx(-float(h))

    # The long is wrong from bar 1 and never recovers.
    assert long["bars_to_first_retrace"] == 1
    assert long["rebreak_bar"] == -1
    assert long["max_retrace_20"] == pytest.approx(20.0)
    assert long["swing_stop_dist"] == pytest.approx(20.0)
    # Retraced, but nothing after it ever got back to ref_px: zero, not null
    # and not the negative distance it actually reached.
    assert long["swing_ext_after_retrace"] == pytest.approx(0.0)

    assert short["bars_to_first_retrace"] == -1
    assert short["max_retrace_20"] == pytest.approx(0.0)
    assert short["swing_ext_after_retrace"] is None


# ---------------------------------------------------------------------------
# 3. V-shape: falls, bottoms, rises through the reference
# ---------------------------------------------------------------------------
def test_v_shape_long_before_the_bottom():
    """Closes run 120 down to 100 over bars 0-20, then back up to 119 by bar
    39, with highs and lows half a point either side.

    The long fires on bar 18 at ref_px = 102, so it is two points early:
    price falls two more bars to the 100 bottom before turning. Bar h is bar
    18 + h, and every number below is that path read off by hand.
    """
    closes = [120 - i for i in range(20)] + [100 + (i - 20) for i in range(20, 40)]
    bars = make_bars(closes, half_range=0.5)
    assert closes[18] == 102 and closes[20] == 100 and closes[39] == 119

    row = measure_one(bars, make_signal(bars, 18, direction=+1))

    # bar 1 -> close 101, bar 3 -> close 101, bar 5 -> 103, bar 10 -> 108,
    # bar 20 -> 118. Bar 39 is the last, so 50 and 100 do not exist.
    assert row["fwd_1"] == pytest.approx(-1.0)
    assert row["fwd_3"] == pytest.approx(-1.0)
    assert row["fwd_5"] == pytest.approx(1.0)
    assert row["fwd_10"] == pytest.approx(6.0)
    assert row["fwd_20"] == pytest.approx(16.0)
    assert row["fwd_50"] is None
    assert row["fwd_100"] is None

    # MFE: highest high so far, minus 102, floored at zero. The first high
    # above the reference is bar 4 (close 102, high 102.5).
    assert row["mfe_1"] == pytest.approx(0.0)   # high 101.5, still under ref
    assert row["mfe_3"] == pytest.approx(0.0)   # high 101.5
    assert row["mfe_5"] == pytest.approx(1.5)   # high 103.5
    assert row["mfe_10"] == pytest.approx(6.5)  # high 108.5
    assert row["mfe_20"] == pytest.approx(16.5)  # high 118.5

    # MAE: lowest low so far, minus 102. The bottom is bar 2 (close 100,
    # low 99.5) and nothing goes lower afterwards.
    assert row["mae_1"] == pytest.approx(-1.5)   # low 100.5
    assert row["mae_3"] == pytest.approx(-2.5)   # low 99.5, set at bar 2
    assert row["mae_5"] == pytest.approx(-2.5)
    assert row["mae_10"] == pytest.approx(-2.5)
    assert row["mae_20"] == pytest.approx(-2.5)

    # Deepest give-back from the running peak. The peak is still the entry
    # through bar 3, so the worst reading is the bar-2 low: 0 - (-2.5).
    assert row["max_retrace_20"] == pytest.approx(2.5)
    # Distance from ref_px down to the lowest low of the window.
    assert row["swing_stop_dist"] == pytest.approx(2.5)
    # Bar 1 already trades 1.5 ATR against, well past the 0.25 threshold.
    assert row["bars_to_first_retrace"] == 1
    # Bar 4 is the first bar after that to trade back above 102.
    assert row["rebreak_bar"] == 4
    # Highest high from bar 2 onward is bar 20's 118.5, so the move extended
    # 16.5 ATR beyond ref_px after pulling back.
    assert row["swing_ext_after_retrace"] == pytest.approx(16.5)


# ---------------------------------------------------------------------------
# 4. Flat line
# ---------------------------------------------------------------------------
def test_flat_line_is_all_zero():
    """No movement anywhere, so every outcome is exactly zero and neither a
    retrace nor a re-break ever happens."""
    bars = make_bars([100.0] * 200)
    row = measure_one(bars, make_signal(bars, 0, direction=+1))

    for h in HORIZONS_TESTED:
        assert row[f"fwd_{h}"] == pytest.approx(0.0)
        assert row[f"mfe_{h}"] == pytest.approx(0.0)
        assert row[f"mae_{h}"] == pytest.approx(0.0)

    assert row["max_retrace_20"] == pytest.approx(0.0)
    assert row["swing_stop_dist"] == pytest.approx(0.0)
    assert row["bars_to_first_retrace"] == -1
    assert row["rebreak_bar"] == -1
    assert row["swing_ext_after_retrace"] is None


# ---------------------------------------------------------------------------
# 5. Truncated window at the end of the data
# ---------------------------------------------------------------------------
def test_signal_three_bars_from_the_end():
    """Ten bars, signal on bar 6, so bars 7, 8 and 9 remain. Horizons 1 and 3
    fit; 5 and beyond do not and must be null rather than measured short."""
    bars = make_bars([100 + i for i in range(10)])
    row = measure_one(bars, make_signal(bars, 6, direction=+1))

    assert row["fwd_1"] == pytest.approx(1.0)
    assert row["fwd_3"] == pytest.approx(3.0)
    assert row["mfe_1"] == pytest.approx(1.0)
    assert row["mfe_3"] == pytest.approx(3.0)
    assert row["mae_1"] == pytest.approx(0.0)
    assert row["mae_3"] == pytest.approx(0.0)

    for h in (5, 10, 20, 50, 100):
        assert row[f"fwd_{h}"] is None, f"fwd_{h} should be null"
        assert row[f"mfe_{h}"] is None, f"mfe_{h} should be null"
        assert row[f"mae_{h}"] is None, f"mae_{h} should be null"

    # The path columns still describe the three bars that do exist.
    assert row["max_retrace_20"] == pytest.approx(0.0)
    assert row["bars_to_first_retrace"] == -1
    assert row["swing_ext_after_retrace"] is None


# ---------------------------------------------------------------------------
# Contract checks that hold across all of the above
# ---------------------------------------------------------------------------
def test_output_dtypes_and_feature_passthrough():
    """Outcomes are Float64, bar-index columns are Int32 as schema.PATH
    declares, and f_ features ride along untouched."""
    bars = make_bars([100 + i for i in range(200)])
    signal = make_signal(bars, 0, direction=+1).with_columns(
        pl.lit(3).alias("f_rejects")
    )
    out = measure_forward(bars, signal)

    for h in HORIZONS_TESTED:
        for stat in ("fwd", "mfe", "mae"):
            assert out.schema[f"{stat}_{h}"] == pl.Float64
    assert out.schema["max_retrace_20"] == pl.Float64
    assert out.schema["swing_stop_dist"] == pl.Float64
    assert out.schema["swing_ext_after_retrace"] == pl.Float64
    assert out.schema["bars_to_first_retrace"] == pl.Int32
    assert out.schema["rebreak_bar"] == pl.Int32
    assert out.schema["signal_ts"] == pl.Datetime("us", time_zone="UTC")

    assert out["f_rejects"].to_list() == [3]
