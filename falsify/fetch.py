"""Dukascopy fetcher.

Constraints that shape this module:

  * The feed is free but governed by Dukascopy's data usage agreement. Data is
    NEVER committed to this repo. `data/` is gitignored. What IS committed is
    the manifest: instrument, date range, and a sha256 per file, so anyone can
    reproduce the exact dataset without redistributing it.
  * Roughly 5-10 requests/second before 429/503, one request per
    instrument-hour. Twelve pairs x fifteen years is ~1.5M requests, so this
    must be resumable. Every completed hour is checkpointed; a killed process
    resumes without refetching.
  * Files are .bi5, LZMA-compressed, 20-byte rows.

TODO: port the existing dukascopy_fetch.py here, keeping its realized-spread
measurement, and add the manifest writer.
"""
