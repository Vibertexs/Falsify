"""Synthetic event table for the viewer.

Produces `viewer/test_events.parquet`: two symbols, 2000 bars each, a signal
every 40 bars with alternating direction and three `f_` features. Nothing here
is a result. The bars are a driftless random walk, so any edge the viewer shows
on this file is sampling noise, and that is the point - it is the shape the
panels should render when there is nothing to find.

Run from anywhere:

    python scripts/generate_test_data.py

Where the bars come from
------------------------
`falsify/demo.py` does not exist in this repository, so the bar generator lives
here rather than being imported. It is deliberately small and has no dependency
on anything in `falsify`, so it can move to `falsify/demo.py` or into the
`random_walk` slot that `nulls.py` already documents without changing a line of
the signal generator or the summary below.

Why the features are hashed rather than drawn
---------------------------------------------
`events.build` re-runs the generator once per candidate on bars truncated to
that candidate's own signal_ts, and raises if a row fails to reproduce. A
generator drawing from a running RNG would produce different features on the
second call and be rejected as non-causal. Each feature here is therefore a
deterministic function of (symbol, signal_ts, feature name): random-looking,
identical on every call, and independent across rows.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from falsify import events, forward

OUT_PATH = "viewer/test_events.parquet"

SOURCE = "synthetic_probe_v1"
SOURCE_VERSION = "0.1.0"

N_BARS = 2000
TIMEFRAME = "M15"
BAR_STEP = timedelta(minutes=15)
START = datetime(2024, 1, 1, tzinfo=UTC)

# Two instruments an order of magnitude apart in price and tick size, because
# ATR normalisation is exactly what is supposed to make them comparable.
INSTRUMENTS = (
    # symbol,    start price, per-bar vol, wick vol, seed
    ("EURUSD", 1.0850, 0.00035, 0.00025, 11),
    ("XAUUSD", 2350.00, 0.00090, 0.00070, 22),
)

# A signal every 40 bars, skipping the first, so ~49 per symbol.
SIGNAL_EVERY = 40
ATR_WINDOW = 14

BAR_SCHEMA = {
    "ts": pl.Datetime("us", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
}


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------
def synthetic_bars(
    symbol: str,
    start_px: float,
    vol: float,
    wick_vol: float,
    seed: int,
    n_bars: int = N_BARS,
) -> pl.DataFrame:
    """A driftless lognormal random walk sampled as OHLC.

    Each bar opens at the previous close, and the wicks are drawn independently
    on each side so the high and low are never inside the body. `vol` is a
    per-bar log-return standard deviation, so the two instruments end up with
    comparable ATR-relative movement despite the price scale.
    """
    rng = np.random.default_rng(seed)

    close = start_px * np.exp(np.cumsum(rng.normal(0.0, vol, n_bars)))
    open_ = np.empty(n_bars, dtype=np.float64)
    open_[0] = start_px
    open_[1:] = close[:-1]

    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    high = body_hi + np.abs(rng.normal(0.0, wick_vol, n_bars)) * close
    low = body_lo - np.abs(rng.normal(0.0, wick_vol, n_bars)) * close

    return pl.DataFrame(
        {
            "ts": [START + i * BAR_STEP for i in range(n_bars)],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "symbol": [symbol] * n_bars,
            "timeframe": [TIMEFRAME] * n_bars,
        },
        schema=BAR_SCHEMA,
    )


def build_bars() -> pl.DataFrame:
    frames = [
        synthetic_bars(symbol, start_px, vol, wick_vol, seed)
        for symbol, start_px, vol, wick_vol, seed in INSTRUMENTS
    ]
    return pl.concat(frames, how="vertical").sort("ts")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _unit_draw(symbol: str, signal_ts: datetime, name: str) -> float:
    """A stable uniform(0, 1) for one (row, feature).

    Hashed rather than sampled so the confirming call in `events.build` gets
    the same number as the enumerating call. The hash is over the ISO timestamp
    rather than an epoch integer so it does not depend on the frame's dtype.
    """
    payload = f"{symbol}|{signal_ts.isoformat()}|{name}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return float(np.random.default_rng(seed).random())


def _features(symbol: str, signal_ts: datetime) -> dict[str, float | int]:
    return {
        # A count with a heavy mode at the low end, which is the case the
        # viewer's median split has a documented fallback for.
        "f_rejects": int(_unit_draw(symbol, signal_ts, "rejects") * 4),
        "f_quality": _unit_draw(symbol, signal_ts, "quality") * 100.0,
        "f_poc_location": _unit_draw(symbol, signal_ts, "poc_location"),
    }


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------
def periodic_probe(bars: pl.DataFrame) -> pl.DataFrame:
    """Fire every SIGNAL_EVERY bars per symbol, alternating direction.

    Causal by construction: the bar ordinal counts backwards from the start of
    each symbol and the ATR is a trailing mean, so truncating the frame at any
    signal_ts leaves every value at or before it untouched. Direction comes
    from the ordinal rather than from the position of the row in the emitted
    frame, which would shift as later signals are truncated away.
    """
    df = bars.sort(["symbol", "ts"])

    prev_close = pl.col("close").shift(1).over("symbol")
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )

    df = df.with_columns(
        pl.int_range(pl.len()).over("symbol").alias("bar_no"),
        true_range.alias("tr"),
    ).with_columns(
        pl.col("tr").rolling_mean(ATR_WINDOW).over("symbol").alias("atr"),
    )

    fired = df.filter(
        (pl.col("bar_no") >= SIGNAL_EVERY)
        & (pl.col("bar_no") % SIGNAL_EVERY == 0)
        & pl.col("atr").is_not_null()
        & (pl.col("atr") > 0)
    )
    if fired.height == 0:
        return _empty_signals()

    ordinal = fired["bar_no"].to_numpy() // SIGNAL_EVERY
    direction = np.where(ordinal % 2 == 0, 1, -1).astype(np.int8)

    symbols = fired["symbol"].to_list()
    stamps = fired["ts"].to_list()
    features = [_features(s, t) for s, t in zip(symbols, stamps)]

    return pl.DataFrame(
        {
            "signal_ts": stamps,
            "direction": direction,
            "ref_px": fired["close"].to_list(),
            "atr_at_signal": fired["atr"].to_list(),
            "symbol": symbols,
            "timeframe": fired["timeframe"].to_list(),
            "f_rejects": [f["f_rejects"] for f in features],
            "f_quality": [f["f_quality"] for f in features],
            "f_poc_location": [f["f_poc_location"] for f in features],
        },
        schema=_signal_schema(),
    )


def _signal_schema() -> dict[str, pl.DataType]:
    return {
        "signal_ts": pl.Datetime("us", time_zone="UTC"),
        "direction": pl.Int8,
        "ref_px": pl.Float64,
        "atr_at_signal": pl.Float64,
        "symbol": pl.Utf8,
        "timeframe": pl.Utf8,
        "f_rejects": pl.Int32,
        "f_quality": pl.Float64,
        "f_poc_location": pl.Float64,
    }


def _empty_signals() -> pl.DataFrame:
    return pl.DataFrame(schema=_signal_schema())


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarise(table: pl.DataFrame) -> None:
    print(f"\nwrote {OUT_PATH}")
    print(f"  events        {table.height}")

    by_symbol = (
        table.group_by("symbol").agg(pl.len().alias("n")).sort("symbol")
    )
    parts = [f"{r['symbol']} ({r['n']})" for r in by_symbol.iter_rows(named=True)]
    print(f"  symbols       {', '.join(parts)}")

    lo = table["signal_ts"].min()
    hi = table["signal_ts"].max()
    print(f"  date range    {lo:%Y-%m-%d %H:%M} to {hi:%Y-%m-%d %H:%M} UTC")

    fwd_10 = table["fwd_10"]
    measured = fwd_10.drop_nulls()
    mean = measured.mean()
    print(
        f"  mean fwd_10   {mean:+.4f} ATR "
        f"({measured.len()} measured, {fwd_10.null_count()} null)"
    )

    features = [c for c in table.columns if c.startswith("f_")]
    print(f"  features      {', '.join(features)}")


def main() -> None:
    bars = build_bars()
    print(f"generated {bars.height} bars across {bars['symbol'].n_unique()} symbols")

    table = events.build(bars, periodic_probe, SOURCE, SOURCE_VERSION)
    table.write_parquet(OUT_PATH)
    summarise(table)

    # The viewer reads these; a missing one is a silent empty panel.
    expected = {"signal_ts", "symbol", "fwd_10"} | {
        f"fwd_{h}" for h in forward.HORIZONS
    }
    missing = sorted(expected - set(table.columns))
    if missing:
        raise SystemExit(f"event table is missing viewer columns: {missing}")


if __name__ == "__main__":
    main()
