"""Tick to bar aggregation and cost measurement.

Bars carry a measured `spread` column rather than an assumed constant. The
21x round-trip cost gap between EURUSD and XAUUSD changes viability
thresholds, so cost is data, not a parameter.

TODO: resampling, session tagging, and hour-of-week deseasonalisation - the
last of which is a prerequisite, not a signal, and may change every
measurement already taken.
"""
