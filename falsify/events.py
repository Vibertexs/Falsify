"""Event table construction.

A signal generator is any callable that takes a bar frame and returns rows of
(signal_ts, direction, ref_px, plus f_-prefixed features). This module attaches
the outcomes and validates the result against schema.py.

The generator NEVER sees future bars. The forward module never sees features.

How that first guarantee is enforced
------------------------------------
A generator cannot be trusted to be causal just because its author meant it to
be, so `build` does not trust it. It runs the generator twice:

  1. ENUMERATION. One call on the whole frame, used for nothing except the set
     of timestamps at which a signal might exist. Every value it returns is
     thrown away, so nothing measured on that call can reach the event table.

  2. CONFIRMATION. One call per candidate timestamp on `bars` truncated to
     `ts <= signal_ts`, keeping only the row that lands exactly on that
     timestamp. The frontier of that frame is the signal bar itself: there is
     no later bar in it to read, so the values kept are the ones the generator
     would have produced live.

Every row in the returned table therefore comes from a call that could not see
past its own signal. A candidate the confirming call fails to reproduce was
decided using bars that had not happened yet, and `build` raises rather than
dropping it quietly - a strategy that needs the next ten bars to confirm a
break is not wrong, it is misdated, and its signal_ts belongs on the bar where
the decision could actually have been made.

The cost is one generator call per event. That is deliberate: the enumeration
pass is the only cheap one, and it is the one whose numbers are discarded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import polars as pl

from falsify import forward, schema

# What a generator must emit. `symbol` and `timeframe` are optional: a
# single-instrument generator can leave them to be filled in from the bars.
GENERATOR_OUTPUT = ("signal_ts", "direction", "ref_px", "atr_at_signal")

# UTC hour at which each session starts; each runs to the next one, and the
# last wraps at midnight. Sessions are a reference column, not a feature - they
# describe when an event happened, they are not a claim about it.
SESSIONS = (
    (0, "asia"),
    (8, "london"),
    (12, "overlap"),
    (16, "ny"),
    (21, "off"),
)

# Cost is measured, not assumed, so this is a placeholder rather than a
# default: bars.py will carry a realized spread and it will be read from there.
UNKNOWN_SPREAD = 0.0

# Unit separator. Nothing in a symbol, timeframe or ISO timestamp contains it,
# so the hashed payload cannot be made ambiguous by an unlucky field value.
_ID_SEPARATOR = "\x1f"

_MAX_REPORTED = 5

Generator = Callable[[pl.DataFrame], pl.DataFrame]


def build(
    bars: pl.DataFrame,
    generator: Generator,
    source: str,
    source_version: str,
) -> pl.DataFrame:
    """Bars plus a signal generator to a validated event table.

    `bars` needs [ts, open, high, low, close, symbol, timeframe]. `generator`
    returns [signal_ts, direction, ref_px, atr_at_signal] and any number of
    `f_`-prefixed features. `source` names the generator and `source_version`
    pins the version of it that produced these rows, because an event table
    outlives the code that made it.

    Raises ValueError if the generator is not causal, if it emits columns that
    are neither base nor `f_`-prefixed, or if the assembled table fails
    `schema.validate`.
    """
    _require(bars, forward.BAR_INPUTS, "bars")
    bars = bars.sort("ts")

    candidates = generator(bars)
    _require(candidates, GENERATOR_OUTPUT, "generator output")

    signals = _confirm(bars, generator, candidates)
    signals = _label_instrument(bars, signals)

    events = forward.measure_forward(bars, signals).with_columns(
        pl.Series("event_id", _event_ids(signals, source), dtype=pl.Utf8),
        pl.lit(source, dtype=pl.Utf8).alias("source"),
        pl.lit(source_version, dtype=pl.Utf8).alias("source_version"),
        _session_expr().alias("session"),
        pl.lit(UNKNOWN_SPREAD, dtype=pl.Float64).alias("spread_at_signal"),
    )
    events = _order_columns(events)

    result = schema.validate(events)
    if not result.ok:
        raise ValueError(
            "event table failed schema validation:\n  - "
            + "\n  - ".join(result.problems)
        )
    return events


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------
def _confirm(
    bars: pl.DataFrame, generator: Generator, candidates: pl.DataFrame
) -> pl.DataFrame:
    """Re-derive every candidate from bars that stop at its own signal bar.

    The enumeration pass supplies the timestamps to look at and nothing else;
    the rows returned here are the ones the confirming calls produced.
    """
    stamps = candidates["signal_ts"].unique().sort().to_list()
    if not stamps:
        return candidates.head(0)

    confirmed: list[pl.DataFrame] = []
    unconfirmed: list = []
    for signal_ts in stamps:
        visible = bars.filter(pl.col("ts") <= signal_ts)
        rows = generator(visible)
        _require(rows, GENERATOR_OUTPUT, "generator output")
        rows = rows.filter(pl.col("signal_ts") == signal_ts)
        if rows.height:
            confirmed.append(rows)
        else:
            unconfirmed.append(signal_ts)

    if unconfirmed:
        shown = ", ".join(str(ts) for ts in unconfirmed[:_MAX_REPORTED])
        hidden = len(unconfirmed) - _MAX_REPORTED
        more = f" (+{hidden} more)" if hidden > 0 else ""
        raise ValueError(
            f"{len(unconfirmed)} of {len(stamps)} signals disappeared when the "
            f"generator was re-run on bars up to their own signal_ts: {shown}"
            f"{more}. They were decided using bars that had not happened yet. "
            "Move signal_ts to the bar the decision could have been made on."
        )

    return pl.concat(confirmed, how="vertical")


def _label_instrument(bars: pl.DataFrame, signals: pl.DataFrame) -> pl.DataFrame:
    """Fill in symbol and timeframe when the generator did not emit them."""
    for col in ("symbol", "timeframe"):
        if col in signals.columns:
            continue
        values = bars[col].unique().to_list()
        if len(values) != 1:
            raise ValueError(
                f"generator did not emit {col!r} and bars carry {len(values)} "
                f"distinct values {values[:_MAX_REPORTED]}; a generator running "
                f"across more than one {col} has to label its own rows"
            )
        signals = signals.with_columns(pl.lit(values[0], dtype=pl.Utf8).alias(col))
    return signals


# ---------------------------------------------------------------------------
# Identity and reference columns
# ---------------------------------------------------------------------------
def _event_ids(signals: pl.DataFrame, source: str) -> list[str]:
    """A content hash, so the same event built twice gets the same id.

    Deliberately not a row number: prereg hashes the sorted event_ids to pin a
    registration to a dataset, and a row number would make an id depend on
    which other events happened to be in the frame.
    """
    return [
        hashlib.sha256(
            _ID_SEPARATOR.join(
                (source, symbol, timeframe, signal_ts.isoformat(), str(direction))
            ).encode("utf-8")
        ).hexdigest()
        for symbol, timeframe, signal_ts, direction in zip(
            signals["symbol"].to_list(),
            signals["timeframe"].to_list(),
            signals["signal_ts"].to_list(),
            signals["direction"].to_list(),
        )
    ]


def _session_expr() -> pl.Expr:
    """Session label from the UTC hour of signal_ts.

    The branches are tested in order against each session's END hour, so the
    first one that matches is the interval the timestamp falls in.
    """
    hour = pl.col("signal_ts").dt.hour()
    ends = [start for start, _ in SESSIONS[1:]] + [24]

    expr = None
    for (_, label), end in zip(SESSIONS, ends):
        branch = pl.lit(label, dtype=pl.Utf8)
        expr = (
            pl.when(hour < end).then(branch)
            if expr is None
            else expr.when(hour < end).then(branch)
        )
    # Unreachable - an hour is always < 24 - but a null here would sail past
    # the dtype check in schema.validate.
    return expr.otherwise(pl.lit(SESSIONS[-1][1], dtype=pl.Utf8))


# ---------------------------------------------------------------------------
# Frame plumbing
# ---------------------------------------------------------------------------
def _require(df: pl.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _order_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Base columns in schema order, then features, then anything else.

    Anything else is a validation failure, so it is kept rather than dropped:
    the error message should name the column the generator actually emitted.
    """
    base = [c for c in schema.BASE_COLUMNS if c in df.columns]
    features = sorted(schema.feature_columns(df))
    rest = [c for c in df.columns if c not in base and c not in features]
    return df.select(base + features + rest)
