# falsify

An event-study harness for strategy research. **Not a backtest engine.**

Most backtesting tools make it cheap to try many configurations. That is the
problem, not the feature. Greedy selection over enough variants produces
all-green equity curves on pure noise, reliably, and the resulting Sharpe looks
fine until it meets out-of-sample data.

`falsify` inverts the incentive. Running a strategy is easy. Running one
*without recording the attempt* is impossible.

---

## The idea

One artifact sits at the centre: the **event table**. One row per signal event,
with every conditioning feature frozen at the signal bar and every outcome
measured forward from it — unstopped, untargeted, in ATR units.

Once it exists, most research questions become a groupby rather than a
backtest:

| question | operation |
|---|---|
| Does a filter predict anything? | split on `f_*`, compare `fwd_*` |
| Is a pullback stop tighter than 2 ATR? | histogram `swing_stop_dist` |
| Does being in profit predict continuation? | condition on `fwd_5`, look at `fwd_20` |
| Do breaks that retrace and re-break behave differently? | filter `rebreak_bar >= 0` |

Two properties make this more than a convenience.

**Measurement is separated from trade construction.** Outcomes contain no stop
and no target, so exit geometry cannot confound a question about drift. Stops
and targets are applied later, on top of the table, as a separate and clearly
labelled step. A negative result then means "no drift" rather than the
ambiguous "no drift, or the wrong stop".

**Forward returns are computed once.** Every hypothesis reads the same
precomputed columns, so the number of hypotheses tested is literally countable.
That number is what deflated Sharpe needs, and it is the number people
otherwise forget.

## Pre-registration is enforced, not suggested

```python
reg = prereg.register(
    events,
    hypothesis   = "Absorbed probe count predicts forward drift at 10 bars",
    population   = "all confirmed breaks, 2022-03 onward",
    split_on     = "f_rejects >= 2",
    metric       = "fwd_10",
    prediction   = "High arm exceeds low by more than 0.15 ATR",
    decision_rule= "Falsified if the bootstrap CI on the difference contains zero",
)
result = prereg.run(reg, events, my_test)   # will not run without a token
```

`run()` refuses a reused token, refuses a dataset that does not match the one
registered against, and `register()` refuses predictions too short to be real
commitments. There is no override flag. Exploration is not forbidden — it lives
in `explore.py` and is tagged `exploratory=True`, so it can be done freely and
can never later be quoted as a test.

## Statistics

Confidence intervals are reported on the **difference between arms**, never as
two separate intervals. Symbol-clustered standard errors are available because
twelve correlated majors are not twelve independent samples. Nulls include a
driftless random walk, AR(1), a stationary block bootstrap that preserves
volatility clustering, and label shuffling.

Deflated Sharpe and probability of backtest overfitting wrap `sharpebench` and
`pypbo` rather than being reimplemented. `n_trials` is always an explicit
argument, fed from the trial log.

## Data

**No market data is committed to this repository, ever.** Dukascopy's feed is
free to download but governed by their data usage agreement. What is committed
is `data/MANIFEST.md`: instrument, date range, row count and sha256 per file,
so anyone can reproduce the exact dataset without redistribution.

The fetcher is resumable by necessity. The feed tolerates roughly 5–10
requests/second before rate limiting, at one request per instrument-hour, so a
twelve-pair multi-year pull is on the order of a million requests and days of
wall-clock. Every completed hour is checkpointed.

## Layout

```
falsify/
├── falsify/
│   ├── schema.py     # event table contract + validation
│   ├── fetch.py      # resumable Dukascopy pull, manifest writer
│   ├── bars.py       # tick→bar, measured spread, session tagging
│   ├── events.py     # signal generator → event table
│   ├── forward.py    # the ONLY module allowed to see post-signal bars
│   ├── nulls.py      # random walk, AR(1), block bootstrap, label shuffle
│   ├── stats.py      # arm comparison, clustered SE, DSR wrappers
│   ├── prereg.py     # registration + append-only trial log
│   └── explore.py    # tagged exploration, never quotable as a result
├── strategies/       # signal generators
├── scripts/          # 01_fetch → 02_build_events → 03_register → 04_run_test
├── viewer/           # read-only results UI over the event table
└── data/             # gitignored except the manifest
```

## Status

Skeleton. `schema`, `prereg` and `stats` are implemented and tested. The
fetcher, bar builder, event builder, forward measurement and nulls are stubs
with their design constraints written into the docstrings.

## Licence

Apache-2.0 for the framework. Note that publishing a *method* is harmless;
publishing a specific edge is not. Crabel's opening-range breakout degraded
after publication and he stopped trading it as written.
