"""Statistics.

Deliberately thin. Where a maintained library exists, wrap it rather than
reimplement: `sharpebench` for deflated Sharpe, `pypbo` for probability of
backtest overfitting. What this module adds is the part specific to the
event-table shape - differences between arms, cluster-robust errors across
correlated symbols, and permutation tests against a shuffled-label null.

One convention throughout: report the confidence interval on the DIFFERENCE
between arms, never two separate intervals. Two overlapping intervals do not
mean no difference, and two non-overlapping ones overstate it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ArmComparison:
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_permutation: float
    detectable_effect: float  # smallest |diff| this sample could resolve at t=2


def compare_arms(
    df: pl.DataFrame,
    *,
    arm_col: str,
    metric: str,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    seed: int = 0,
) -> ArmComparison:
    """Two-arm comparison with a bootstrap CI on the difference and a
    permutation test against the shuffled-label null."""
    rng = np.random.default_rng(seed)
    arms = df[arm_col].unique().sort().to_list()
    if len(arms) != 2:
        raise ValueError(f"{arm_col} must have exactly 2 levels, got {arms}")

    a = df.filter(pl.col(arm_col) == arms[0])[metric].drop_nulls().to_numpy()
    b = df.filter(pl.col(arm_col) == arms[1])[metric].drop_nulls().to_numpy()
    observed = b.mean() - a.mean()

    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = (
            rng.choice(b, b.size, replace=True).mean()
            - rng.choice(a, a.size, replace=True).mean()
        )
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    pooled = np.concatenate([a, b])
    perm = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        perm[i] = pooled[a.size:].mean() - pooled[: a.size].mean()
    p = float((np.abs(perm) >= abs(observed)).mean())

    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return ArmComparison(
        n_a=a.size, n_b=b.size,
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        diff=float(observed),
        ci_low=float(ci_low), ci_high=float(ci_high),
        p_permutation=p,
        detectable_effect=float(2 * se),
    )


def clustered_se(
    df: pl.DataFrame, *, metric: str, cluster_col: str = "symbol"
) -> tuple[float, float, float]:
    """Cluster-robust mean, SE and t. Twelve correlated majors are not twelve
    independent samples; the naive SE will be optimistic."""
    g = (
        df.group_by(cluster_col)
        .agg(pl.col(metric).mean().alias("m"), pl.len().alias("n"))
        .drop_nulls()
    )
    m = g["m"].to_numpy()
    w = g["n"].to_numpy().astype(float)
    w = w / w.sum()
    mean = float((m * w).sum())
    k = m.size
    se = float(np.sqrt(((m - mean) ** 2).sum() / (k * (k - 1))))
    return mean, se, mean / se if se > 0 else np.nan


def block_bootstrap_null(
    returns: np.ndarray, *, block: int = 20, n: int = 5_000, seed: int = 0
) -> np.ndarray:
    """Stationary block bootstrap. Preserves short-range autocorrelation and
    volatility clustering, which an iid shuffle destroys - and which is exactly
    what an intraday mean-reverting series has."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, returns.size))
    n_blocks = int(np.ceil(returns.size / block))
    for i in range(n):
        starts = rng.integers(0, returns.size - block, n_blocks)
        out[i] = np.concatenate([returns[s: s + block] for s in starts])[: returns.size]
    return out


def deflated_sharpe(per_trade_returns: np.ndarray, n_trials: int) -> dict:
    """Wrapper around sharpebench. Kept as a function so n_trials is always an
    explicit argument and never silently defaults to 1."""
    try:
        from sharpebench import bootstrap_dsr_ci, is_my_sharpe_real
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pip install sharpebench - do not hand-roll the deflation"
        ) from exc
    return {
        "verdict": is_my_sharpe_real(per_trade_returns, n_trials=n_trials),
        "ci": bootstrap_dsr_ci(per_trade_returns, n_trials=n_trials),
    }
