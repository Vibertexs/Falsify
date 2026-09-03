"""Null generators.

Every statistic gets measured against these before it is trusted. A statistic
that fires at the same rate on noise as on signal has no information content.

  * random_walk: driftless GBM sampled as OHLC
  * ar1: a trending alternative, for power rather than size
  * block_bootstrap: resamples the real series in blocks, preserving
    autocorrelation and volatility clustering that a driftless walk lacks
  * label_shuffle: keeps the price path, permutes which bars are events
"""
