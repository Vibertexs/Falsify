"""Event table construction.

A signal generator is any callable that takes a bar frame and returns rows of
(signal_ts, direction, ref_px, plus f_-prefixed features). This module attaches
the outcomes and validates the result against schema.py.

The generator NEVER sees future bars. The forward module never sees features.
"""
