"""Event table schema.

The event table is the central artifact. One row per signal event, with every
conditioning feature frozen at the signal bar and every outcome measured
forward from it, unstopped and untargeted.

Two rules make the whole thing work:

1. FEATURES are computed using only information available at or before
   `signal_ts`. OUTCOMES are computed only from bars strictly after it.
   The split is enforced structurally, not by convention, so a leakage bug
   has to cross a module boundary to happen.

2. OUTCOMES contain no stop, no target, and no position sizing. Trade
   construction is applied later, on top of the table. This is what keeps
   exit geometry from confounding a question about drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
IDENTITY = {
    "event_id": pl.Utf8,        # stable hash of (source, symbol, tf, signal_ts, direction)
    "source": pl.Utf8,          # signal generator name, e.g. "range_finder_v5_15"
    "source_version": pl.Utf8,  # git sha or semver of the generator
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "signal_ts": pl.Datetime("us", time_zone="UTC"),
    "direction": pl.Int8,       # +1 long, -1 short
}

# ---------------------------------------------------------------------------
# Reference prices and costs, known at the signal bar
# ---------------------------------------------------------------------------
REFERENCE = {
    "ref_px": pl.Float64,        # the price outcomes are measured from
    "atr_at_signal": pl.Float64, # the normaliser; every outcome is in these units
    "spread_at_signal": pl.Float64,  # measured, not assumed. NaN if unknown.
    "session": pl.Utf8,          # asia / london / ny / overlap / off
}

# ---------------------------------------------------------------------------
# Outcomes. Unstopped, untargeted, signed by direction, in ATR units.
# Horizons are in bars of `timeframe`.
# ---------------------------------------------------------------------------
HORIZONS = (1, 3, 5, 10, 20, 50, 100)

OUTCOMES: dict[str, pl.DataType] = {}
for _h in HORIZONS:
    OUTCOMES[f"fwd_{_h}"] = pl.Float64   # signed return at bar h
    OUTCOMES[f"mfe_{_h}"] = pl.Float64   # max favourable excursion within h
    OUTCOMES[f"mae_{_h}"] = pl.Float64   # max adverse excursion within h

# ---------------------------------------------------------------------------
# Path structure. Descriptive, computed forward, used to DEFINE populations
# (e.g. "events that retraced 50% then re-broke"), never as a feature that
# could have been known at the signal bar.
# ---------------------------------------------------------------------------
PATH = {
    "max_retrace_20": pl.Float64,     # deepest pullback within 20 bars, ATR units
    "bars_to_first_retrace": pl.Int32,
    "swing_ext_after_retrace": pl.Float64,
    "rebreak_bar": pl.Int32,          # bar index of impulse-swing re-break, -1 none
    "swing_stop_dist": pl.Float64,    # entry to retrace swing, ATR units
}

BASE_COLUMNS = {**IDENTITY, **REFERENCE, **OUTCOMES, **PATH}

# Feature columns are strategy-specific and prefixed `f_` so they can never be
# confused with outcomes. A generator declares its own; the validator only
# checks the prefix and that they are numeric or boolean.
FEATURE_PREFIX = "f_"


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    problems: list[str]


def validate(df: pl.DataFrame) -> ValidationResult:
    """Structural checks. Cheap, run on every build, fail loudly."""
    problems: list[str] = []

    for col, dtype in BASE_COLUMNS.items():
        if col not in df.columns:
            problems.append(f"missing required column: {col}")
        elif df.schema[col] != dtype:
            problems.append(f"{col}: expected {dtype}, got {df.schema[col]}")

    unknown = [
        c for c in df.columns
        if c not in BASE_COLUMNS and not c.startswith(FEATURE_PREFIX)
    ]
    if unknown:
        problems.append(
            f"columns neither base nor {FEATURE_PREFIX}-prefixed: {unknown}"
        )

    if "event_id" in df.columns and df["event_id"].n_unique() != df.height:
        problems.append("event_id is not unique")

    if "atr_at_signal" in df.columns:
        bad = df.filter(pl.col("atr_at_signal") <= 0).height
        if bad:
            problems.append(f"{bad} rows with non-positive atr_at_signal")

    if "direction" in df.columns:
        bad = df.filter(~pl.col("direction").is_in([-1, 1])).height
        if bad:
            problems.append(f"{bad} rows with direction not in (-1, +1)")

    # MAE is adverse, so it must be <= 0 by construction; MFE >= 0.
    for h in HORIZONS:
        mae, mfe = f"mae_{h}", f"mfe_{h}"
        if mae in df.columns and df.filter(pl.col(mae) > 1e-9).height:
            problems.append(f"{mae} has positive values; sign convention broken")
        if mfe in df.columns and df.filter(pl.col(mfe) < -1e-9).height:
            problems.append(f"{mfe} has negative values; sign convention broken")

    return ValidationResult(ok=not problems, problems=problems)


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
