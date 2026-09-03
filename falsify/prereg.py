"""Pre-registration and the trial log.

This module exists to make one thing impossible: looking at a result before
committing to what would count as a positive one.

The mechanism is deliberately blunt. `register()` writes a prediction to an
append-only log and returns a token. `run()` will not execute without a token,
and refuses to reuse one. There is no flag to disable it. If you want to peek
at the data, use the exploration API in `explore.py`, which tags everything it
touches as exploratory so it can never be quoted as a test.

The trial counter matters as much as the predictions. Deflated Sharpe needs an
honest n_trials, and an honest n_trials is one nobody had the chance to forget.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import polars as pl

LOG_PATH = Path(os.environ.get("FALSIFY_TRIAL_LOG", "data/trial_log.jsonl"))


@dataclass(frozen=True)
class Registration:
    token: str
    registered_at: str
    hypothesis: str          # plain English, what mechanism is claimed
    population: str          # the filter defining which events are included
    split_on: str            # the feature or rule being split
    metric: str              # e.g. "fwd_10"
    prediction: str          # what you expect, in words, including sign
    decision_rule: str       # what result would falsify it. Required.
    dataset_hash: str        # hash of the event table this applies to
    exploratory: bool = False
    notes: str = ""
    tags: list[str] = field(default_factory=list)


def _hash_dataset(df: pl.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(sorted(df.columns)).encode())
    h.update(str(df.height).encode())
    if "event_id" in df.columns:
        ids = df["event_id"].sort().to_list()
        h.update("".join(ids).encode())
    return h.hexdigest()[:16]


def _append(record: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def register(
    df: pl.DataFrame,
    *,
    hypothesis: str,
    population: str,
    split_on: str,
    metric: str,
    prediction: str,
    decision_rule: str,
    notes: str = "",
    tags: list[str] | None = None,
) -> Registration:
    """Commit a prediction. Must be called before the result is computed."""
    for name, value in [
        ("hypothesis", hypothesis),
        ("prediction", prediction),
        ("decision_rule", decision_rule),
    ]:
        if not value or len(value.strip()) < 15:
            raise ValueError(
                f"{name!r} is too short to be a real commitment. "
                "Write the sentence you would be embarrassed to have wrong."
            )

    reg = Registration(
        token=uuid.uuid4().hex,
        registered_at=datetime.now(timezone.utc).isoformat(),
        hypothesis=hypothesis,
        population=population,
        split_on=split_on,
        metric=metric,
        prediction=prediction,
        decision_rule=decision_rule,
        dataset_hash=_hash_dataset(df),
        notes=notes,
        tags=tags or [],
    )
    _append({"kind": "registration", **asdict(reg)})
    return reg


def _token_used(token: str) -> bool:
    if not LOG_PATH.exists():
        return False
    with LOG_PATH.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("kind") == "result" and rec.get("token") == token:
                return True
    return False


def run(
    reg: Registration,
    df: pl.DataFrame,
    fn: Callable[[pl.DataFrame], dict[str, Any]],
) -> dict[str, Any]:
    """Execute a registered test exactly once and log the result."""
    if _token_used(reg.token):
        raise RuntimeError(
            f"token {reg.token} already has a result. A registration is good for "
            "one look. Register a new one and let the trial count reflect it."
        )
    if _hash_dataset(df) != reg.dataset_hash:
        raise RuntimeError(
            "dataset does not match the one this test was registered against. "
            "Re-register, so the trial log records that the data changed."
        )

    result = fn(df)
    _append({
        "kind": "result",
        "token": reg.token,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    })
    return result


def trial_count(include_exploratory: bool = False) -> int:
    """The honest n_trials for deflated Sharpe."""
    if not LOG_PATH.exists():
        return 0
    n = 0
    with LOG_PATH.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("kind") != "registration":
                continue
            if rec.get("exploratory") and not include_exploratory:
                continue
            n += 1
    return n


def log_as_frame() -> pl.DataFrame:
    """The trial log as a table, for the viewer."""
    if not LOG_PATH.exists():
        return pl.DataFrame()
    rows = [json.loads(line) for line in LOG_PATH.open()]
    return pl.DataFrame(rows, infer_schema_length=None)
