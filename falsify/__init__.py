"""falsify - an event-study harness for strategy research.

Not a backtest engine. A measurement harness that makes the falsification
funnel mechanical: null calibration, pre-registration, one look per
registration, and an honest trial count feeding the multiple-testing correction.
"""

__version__ = "0.1.0"

from falsify import schema, prereg, stats  # noqa: F401
