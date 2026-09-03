"""Forward outcome measurement.

Unstopped, untargeted, signed by direction, normalised by atr_at_signal.
Computes fwd/mfe/mae at each horizon plus the path-structure columns used to
define populations (retrace depth, re-break bar, swing stop distance).

This module is the only place allowed to look at bars after signal_ts.
"""
