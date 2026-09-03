"""Forward outcome measurement.

Unstopped, untargeted, signed by direction, normalised by atr_at_signal.
Computes fwd/mfe/mae at each horizon plus the path-structure columns used to
define populations (retrace depth, re-break bar, swing stop distance).

This module is the only place allowed to look at bars after signal_ts.

Conventions, all of them load-bearing
-------------------------------------
Bar indexing is 1-based and strictly forward: bar 1 is the first bar whose
`ts` is strictly greater than `signal_ts`. A signal is never measured against
the bar it was generated on, so a generator that fires on the close of bar k
cannot be credited with bar k's range.

Every outcome is divided by `atr_at_signal` and multiplied by `direction`, so
a positive number always means "the trade went the way it was supposed to",
whichever side it was on. That is what makes longs and shorts poolable.

Excursions use the intrabar extremes, and which extreme is favourable depends
on the side: a long's favourable extreme is the high and its adverse extreme
is the low; a short's are the other way round. Both are then signed by
direction, so mfe is a max of favourable signed excursions and mae a min of
adverse ones. Both are clamped at zero - an event that never traded above its
reference price has mfe = 0, not a negative "favourable" excursion - which is
the sign convention `schema.validate` enforces.

`fwd_h`, `mfe_h` and `mae_h` are null unless a full h bars exist after the
signal. A truncated window is a missing measurement, not a short one; filling
it with whatever data happened to be there would bias every horizon towards
the end of the sample.

The path columns describe the shape of the first `PATH_WINDOW` (20) bars and
are computed over however many of those exist, because their purpose is to
define populations rather than to be averaged. With no bars at all they are
null, or -1 for the two bar-index columns.

`f_`-prefixed feature columns are never read. The computation is built from an
explicit allow-list of signal columns, checked at runtime; features are only
carried through to the output frame untouched.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from falsify.schema import FEATURE_PREFIX, HORIZONS

# The path columns describe the first 20 bars. Kept separate from HORIZONS
# because it is a description of shape, not a measurement horizon.
PATH_WINDOW = 20

# A retrace has to be big enough to be a retrace rather than noise. In ATR
# units, so it means the same thing on EURUSD and XAUUSD.
RETRACE_THRESHOLD = 0.25

# Columns this module is allowed to read off the signals frame. Anything not
# in here - and every f_ feature in particular - is carried through to the
# output but never touched by the computation.
SIGNAL_INPUTS = (
    "signal_ts",
    "direction",
    "ref_px",
    "atr_at_signal",
    "symbol",
    "timeframe",
)

BAR_INPUTS = ("ts", "open", "high", "low", "close", "symbol", "timeframe")

OUTCOME_COLUMNS = tuple(
    f"{stat}_{h}" for h in HORIZONS for stat in ("fwd", "mfe", "mae")
)
PATH_COLUMNS = (
    "max_retrace_20",
    "bars_to_first_retrace",
    "swing_ext_after_retrace",
    "swing_stop_dist",
    "rebreak_bar",
)

# Bar-index columns are Int32, matching schema.PATH, with -1 meaning "did not
# happen". Every other output is Float64.
_INT_COLUMNS = ("bars_to_first_retrace", "rebreak_bar")

_MAX_HORIZON = max(HORIZONS)
_WINDOW = max(_MAX_HORIZON, PATH_WINDOW)


def measure_forward(bars: pl.DataFrame, signals: pl.DataFrame) -> pl.DataFrame:
    """Attach forward outcomes and path structure to `signals`.

    `bars` needs [ts, open, high, low, close, symbol, timeframe]; `signals`
    needs [signal_ts, direction, ref_px, atr_at_signal, symbol, timeframe].
    Bars are matched to signals within a (symbol, timeframe) group; a signal
    whose group has no bars after it gets nulls, not an error.

    Returns `signals` with `OUTCOME_COLUMNS` and `PATH_COLUMNS` appended, in
    the original row order.
    """
    _require(bars, BAR_INPUTS, "bars")
    _require(signals, SIGNAL_INPUTS, "signals")

    n = signals.height
    out = _empty_outputs(n)
    if n == 0:
        return _attach(signals, out)

    direction = signals["direction"].to_numpy().astype(np.int64)
    if not np.isin(direction, (-1, 1)).all():
        raise ValueError("direction must be -1 or +1 for every signal")

    atr = signals["atr_at_signal"].to_numpy().astype(np.float64)
    if not np.all(np.isfinite(atr)) or np.any(atr <= 0):
        raise ValueError(
            "atr_at_signal must be finite and strictly positive; it is the "
            "normaliser every outcome is divided by"
        )

    ref_px = signals["ref_px"].to_numpy().astype(np.float64)
    signal_ts = signals["signal_ts"].to_numpy().astype("datetime64[us]")

    # Group both frames on (symbol, timeframe) and measure each group against
    # its own bars. Positions are kept so results land back in input order.
    keys = list(zip(signals["symbol"].to_list(), signals["timeframe"].to_list()))
    positions: dict[tuple[str, str], list[int]] = {}
    for i, key in enumerate(keys):
        positions.setdefault(key, []).append(i)

    bars = bars.sort("ts")
    for (symbol, timeframe), idx_list in positions.items():
        group = bars.filter(
            (pl.col("symbol") == symbol) & (pl.col("timeframe") == timeframe)
        )
        if group.height == 0:
            continue

        rows = np.asarray(idx_list, dtype=np.int64)
        _measure_group(
            bar_ts=group["ts"].to_numpy().astype("datetime64[us]"),
            high=group["high"].to_numpy().astype(np.float64),
            low=group["low"].to_numpy().astype(np.float64),
            close=group["close"].to_numpy().astype(np.float64),
            signal_ts=signal_ts[rows],
            direction=direction[rows],
            ref_px=ref_px[rows],
            atr=atr[rows],
            rows=rows,
            out=out,
        )

    return _attach(signals, out)


# ---------------------------------------------------------------------------
# The measurement itself
# ---------------------------------------------------------------------------
def _measure_group(
    *,
    bar_ts: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    signal_ts: np.ndarray,
    direction: np.ndarray,
    ref_px: np.ndarray,
    atr: np.ndarray,
    rows: np.ndarray,
    out: dict[str, np.ndarray],
) -> None:
    """Fill `out` at `rows` for one (symbol, timeframe) group.

    Everything is done on an (n_signals x _WINDOW) matrix of the bars that
    follow each signal, so the horizons are cumulative slices of one array
    rather than a loop per event.
    """
    n_bars = bar_ts.size

    # First bar strictly after the signal. side="right" is what makes it
    # strict, and is the entire lookahead guarantee of this module.
    start = np.searchsorted(bar_ts, signal_ts, side="right")

    offsets = start[:, None] + np.arange(_WINDOW)[None, :]
    exists = offsets < n_bars
    gather = np.where(exists, offsets, 0)

    # Which extreme is favourable depends on the side.
    is_long = (direction > 0)[:, None]
    fav_px = np.where(is_long, high[gather], low[gather])
    adv_px = np.where(is_long, low[gather], high[gather])

    scale = (direction / atr)[:, None]
    fav = (fav_px - ref_px[:, None]) * scale
    adv = (adv_px - ref_px[:, None]) * scale
    fwd = (close[gather] - ref_px[:, None]) * scale

    # Out-of-range cells are filled with the identity for the reduction that
    # will read them, so no reduction ever sees a bar that does not exist.
    fav_run = np.maximum.accumulate(np.where(exists, fav, -np.inf), axis=1)
    adv_run = np.minimum.accumulate(np.where(exists, adv, np.inf), axis=1)

    for h in HORIZONS:
        # A horizon is measured only if all h bars are there.
        complete = start + h <= n_bars
        col = h - 1
        out[f"fwd_{h}"][rows] = np.where(complete, fwd[:, col], np.nan)
        out[f"mfe_{h}"][rows] = np.where(
            complete, np.maximum(0.0, fav_run[:, col]), np.nan
        )
        out[f"mae_{h}"][rows] = np.where(
            complete, np.minimum(0.0, adv_run[:, col]), np.nan
        )

    _measure_path(
        exists=exists[:, :PATH_WINDOW],
        fav=fav[:, :PATH_WINDOW],
        adv=adv[:, :PATH_WINDOW],
        fav_run=fav_run[:, :PATH_WINDOW],
        adv_run=adv_run[:, :PATH_WINDOW],
        rows=rows,
        out=out,
    )


def _measure_path(
    *,
    exists: np.ndarray,
    fav: np.ndarray,
    adv: np.ndarray,
    fav_run: np.ndarray,
    adv_run: np.ndarray,
    rows: np.ndarray,
    out: dict[str, np.ndarray],
) -> None:
    """The four path-structure columns, over the first PATH_WINDOW bars.

    Unlike the horizons these use whatever part of the window exists, since
    they describe a shape rather than measure a fixed-length return. A signal
    with no bars after it at all still gets nulls.
    """
    any_bar = exists.any(axis=1)
    bar_no = np.arange(1, exists.shape[1] + 1)[None, :]

    # max_retrace_20: deepest give-back from the running favourable peak,
    # where the peak starts at the reference price. So an event that never
    # trades in its favour retraces by its adverse excursion, and one that
    # runs and hands it back is measured from the high-water mark rather than
    # from entry - which is what distinguishes this from mae_20.
    peak = np.maximum(0.0, fav_run)
    retrace = np.where(exists, peak - adv, -np.inf)
    out["max_retrace_20"][rows] = np.where(any_bar, retrace.max(axis=1), np.nan)

    # swing_stop_dist: reference price to the lowest low (long) or highest
    # high (short) in the window. A distance, so positive whenever price
    # traded against the signal at all, and the natural swing-stop candidate.
    out["swing_stop_dist"][rows] = np.where(any_bar, -adv_run[:, -1], np.nan)

    # bars_to_first_retrace: first bar trading more than RETRACE_THRESHOLD
    # against the signal, measured from the reference price - the same anchor
    # rebreak_bar uses, so the two describe one story.
    retraced = exists & (adv < -RETRACE_THRESHOLD)
    has_retrace = retraced.any(axis=1)
    first_retrace = np.where(has_retrace, retraced.argmax(axis=1) + 1, -1)
    out["bars_to_first_retrace"][rows] = first_retrace

    # Everything below describes what happened once the retrace was in. It is
    # strictly after the retrace bar, because intrabar order is unknowable: a
    # bar that both retraced and recovered cannot be shown to have done so in
    # that order.
    after = exists & (bar_no > first_retrace[:, None]) & has_retrace[:, None]

    # rebreak_bar: first of those bars to trade beyond the reference price
    # again. No retrace means nothing to re-break, hence -1.
    rebroke = after & (fav > 0.0)
    out["rebreak_bar"][rows] = np.where(
        rebroke.any(axis=1), rebroke.argmax(axis=1) + 1, -1
    )

    # swing_ext_after_retrace: how far the move eventually got once it had
    # pulled back, measured from the reference price rather than from the
    # retrace low - so it answers "was the pullback worth buying" on the same
    # scale as every other outcome. Floored at zero, so a retrace that never
    # recovered reads 0 rather than a negative "extension"; -inf, meaning the
    # retrace was the last bar of the window, floors to 0 the same way. Null
    # when there was no retrace to extend from.
    extension = np.maximum(0.0, np.where(after, fav, -np.inf).max(axis=1))
    out["swing_ext_after_retrace"][rows] = np.where(has_retrace, extension, np.nan)


# ---------------------------------------------------------------------------
# Frame plumbing
# ---------------------------------------------------------------------------
def _require(df: pl.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _empty_outputs(n: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for col in OUTCOME_COLUMNS + PATH_COLUMNS:
        if col in _INT_COLUMNS:
            out[col] = np.full(n, -1, dtype=np.int64)
        else:
            out[col] = np.full(n, np.nan, dtype=np.float64)
    return out


def _attach(signals: pl.DataFrame, out: dict[str, np.ndarray]) -> pl.DataFrame:
    """Append the measurements to the signals frame, in input row order.

    The assert is not defensive noise: it is the structural half of "the
    forward module never sees features". Nothing f_-prefixed can reach the
    computation, because the computation only ever builds these columns.
    """
    assert not any(c.startswith(FEATURE_PREFIX) for c in out), (
        "forward measurement produced an f_-prefixed column"
    )
    series = []
    for col, values in out.items():
        if col in _INT_COLUMNS:
            series.append(pl.Series(col, values, dtype=pl.Int32))
        else:
            series.append(pl.Series(col, values, dtype=pl.Float64).fill_nan(None))
    return signals.hstack(series)
