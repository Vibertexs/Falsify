# Dataset manifest

Data files are not committed. This manifest pins the exact bytes so a dataset
is reproducible without redistribution.

Regenerate with `python scripts/01_fetch.py --manifest`.

| symbol | timeframe | start | end | rows | sha256 | fetched |
|---|---|---|---|---|---|---|
| _(empty)_ | | | | | | |

## Source terms

Dukascopy historical data is free to download. Use is governed by Dukascopy's
data usage agreement. This repository ships code that downloads it, never the
data itself.
