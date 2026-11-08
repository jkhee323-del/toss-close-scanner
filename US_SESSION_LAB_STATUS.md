# Overseas market refresh safety lab — 2026-11-08

This is an **isolated GitHub experimental branch**, not a release. Do not
merge into `main` or overwrite a newer local V3.1 project without a
version-by-version review.

## Changes in this branch

- Local-exchange clock guard (US/Japan/Germany/France/Netherlands/Switzerland)
  and DST-aware conversions for timezone-aware timestamps.
- Validated OHLCV before writes (dates, non-null finite numbers,
  positive prices, high/low consistency, nonnegative volume).
- Merge one-year provider refresh with all existing history rather than
  discarding older training rows.
- Reject overlapping close-price disagreements larger than 25%;
  investigate splits, vendor corrections and bad cache instead of overwriting.
- Atomic replacement of price-history and candidate ranking CSVs.
- Keep the previous ranking table when too many ticker downloads fail.
- Added unit/integration regression tests for exchange session boundaries,
  bad OHLCV, partial network failures, and failed CSV replacements.

## Verified

GitHub Actions **US Session Lab Regression**:
https://github.com/jkhee323-del/toss-close-scanner/actions/runs/42486988345

- Python source compilation succeeded.
- 37 pytest regression cases passed on Python 3.13.

These are tests for **this repository and branch**, not the separate Windows
V3.1 ZIP. The local V3.1 previously had 39 tests in a different project state.

## Known limitations — not production-ready

- A 17:00 NY release buffer (and other exchange local buffers) reduces
  intraday-bar risk but cannot prove a vendor daily bar is final or detect
  partial bars that were cached in the past.
- Date filtering uses weekdays but does not consult full exchange holiday,
  unscheduled closure, or corporate-action calendars.
- A 25% overlap threshold is a safety stop, not a split-adjustment algorithm.
  Existing corrupted histories and duplicate-ticker series need independent
  review, re-download, and provenance checks.
- Network refresh was not exercised against live Yahoo data in this run.
- Not yet integrated with the user's latest offline Windows V3.1 project,
  medium/long-horizon experimental AI, foreign/institutional flows or CVD.
- Do not interpret any experimental model score as a calibrated probability
  of profit. No strategy performance or live trading accuracy is certified.

## Integration before release

1. Compare this branch with the actual local V3.1 code; port compatible
   changes one module at a time, preserving all working features.
2. Revalidate each downloaded US bar after the final session-close buffer.
3. Run the local V3.1 complete test suite and real-provider smoke checks.
4. Run purged walk-forward AI evaluation and out-of-sample trade P&L audits.
5. Build a separate development ZIP only after the merged copy passes tests.
