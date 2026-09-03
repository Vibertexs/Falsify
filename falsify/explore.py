"""Exploration API.

Everything here writes to the trial log tagged exploratory=True. Use it freely
to look at the data - and know that anything found this way is a hypothesis,
not a result, and must be re-registered and confirmed on held-out data before
it counts.

The separation is the point: exploration is not forbidden, it is labelled.
"""
