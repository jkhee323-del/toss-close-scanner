# Held-out-stock validation and scanner snapshot wiring — 2026-11-25

The fixed peer-rank candidate was evaluated on three disjoint symbol folds
from all 499 stored Korean history files at `eddd75a`. Each fold excludes
its evaluation stocks from training labels and from training peer ranks.
The quarterly training cutoff requires exits strictly before the start of
the evaluation quarter. The round-trip cost remains 0.2%.

Reproduce with `python -m src.long_term_walkforward --relative --symbol-folds`.
The CI version limits the input to a deterministic 80-file sample.

| Sessions | Candidate Brier | Constant Brier | AUC | Accuracy | Better folds | Top-ten hit rate | Mean net return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.251280 | 0.251444 | 0.5146 | 51.11% | 3/3 | 49.86% | +0.80% |
| 60 | 0.258649 | 0.259225 | 0.5348 | 45.17% | 3/3 | 55.61% | +3.85% |
| 120 | 0.268176 | 0.268464 | 0.5307 | 37.52% | 3/3 | 56.97% | +8.94% |
| 240 | 0.268234 | 0.268446 | 0.5291 | 35.89% | 2/3 | 63.54% | +20.94% |

Probability metrics weight each evaluated quarter/fold by observed outcomes.
Top-ten metrics weight each fold and quarter by selected trades with known
outcomes. Positions can overlap. One 240-session fold underperformed the
constant by 0.001098 Brier. The date range has been used for previous
research and the universe is drawn from current repository histories. This
measures transfer to held-out symbols within the same period. Confirmation
on untouched future dates remains outstanding.

## Snapshot configuration

The medium/long-term lab page can read the collector JSON from the
server-side `KRX_FLOW_SNAPSHOT` path. It displays a row only for the exact
eligible published market session and ticker with a full 20-session window.
When configured, a live KRX lookup can fill unavailable snapshot rows.

The short-term lab page can read trusted collector output from the
server-side `KIS_CVD_CAPTURE_DIR` directory, using `cvd-<ticker>.csv` for
the selected Korean ticker. It recomputes delta from signed executions and
displays the saved capture interval. Each saved interval is its own series.
Manual CSV upload remains available.

Actual provider authentication, unattended publishing and independent
scanner lift validation remain open. Local regression suite: 129 passed.
