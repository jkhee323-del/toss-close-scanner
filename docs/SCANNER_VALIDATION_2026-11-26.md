# Model and feed work — 2026-11-26

Base commit `5ce35b0c2eb898e06116ff571af830823d7236da` contains 499 Korean
history files, with Samsung Electronics ending on 2026-11-25.

## Model comparison

The existing feature set primarily covers 20 trading days. The additional
candidate includes 60/120/240-session returns, 120/240-session moving-average
distances, 120-session volatility and distance from the 240-session high.
Both feature sets are evaluated on the same observations after the longer
warm-up. Three disjoint stock groups provide held-out-symbol evaluation.

Two new experiments were run on this previously inspected retrospective.
Fixed 0.5 blending improved aggregate Brier at 20/60/120 sessions, but the
240-session candidate had Brier 0.250745 against constant 0.249981.
A second candidate estimates the blending weight using only a purged inner
validation window within each outer training period. The inner window uses
the latest 126 observed signal dates, and its fitting labels must mature
strictly before that window. Missing training coverage falls back to the
training positive rate. The coefficient is bounded between zero and one.

Run the second comparison with
`python -m src.long_term_walkforward --relative --symbol-folds --long-features --validate-blend`.
Add `--max-symbols 80` for the deterministic sample used in CI.

| Sessions | Full-universe candidate Brier | Constant Brier | Better stock groups | 80-symbol candidate Brier | Sample constant Brier | Sample better groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.251382 | 0.251504 | 3/3 | 0.251723 | 0.251722 | 1/3 |
| 60 | 0.255363 | 0.256246 | 3/3 | 0.256700 | 0.256408 | 1/3 |
| 120 | 0.261525 | 0.261951 | 2/3 | 0.270736 | 0.268656 | 2/3 |
| 240 | 0.249504 | 0.249981 | 3/3 | 0.266535 | 0.267305 | 2/3 |

The 12-month aggregate improved in both populations. One small-sample stock
group still failed. The shorter horizons lost aggregate superiority in the
80-symbol test. This candidate needs prospective evidence before deployment.
Model choice across these research runs has been informed by observed test
results, so the table remains retrospective. Overlapping holding periods,
current-universe membership, incomplete outcomes and modest AUCs limit the
inference. All returns deduct the existing 0.2% round-trip cost.
When the validated blend is zero, the model abstains from stock ranking.
Its probabilities still enter Brier/AUC/accuracy metrics, while the trade
selection count is zero and the return metric is unavailable.

## KIS authorization and capture integrity

The CVD command now accepts `KIS_APP_KEY` and `KIS_APP_SECRET` and requests a
WebSocket approval key through the broker's documented
[approval endpoint](https://github.com/koreainvestment/open-trading-api/blob/main/examples_user/auth/auth_functions.py).
An explicit `KIS_WS_APPROVAL_KEY` still works. Requests use a timeout,
reject redirects and provide sanitized failure messages.

Captured files retain exchange cumulative volume. The scanner checks this
column when present, detecting missing, repeated or reversed records after
capture. Legacy signed CSVs without this column continue to calculate CVD
with their original validation. This column is evidence for sequence
continuity within a capture window. Provider coverage still requires live
verification.

The lab-only `KIS read-only feed authorization` workflow tests access using
GitHub secrets without printing the issued key. It verifies authorization
only. Real executions must be collected during an eligible market session.

## Data-access status

The KRX collector was rerun on November 26. GitHub again failed at
`KRX_ID` and `KRX_PW` missing. The local runtime has none of the KRX or
KIS credential variables configured. Real source reconciliation and
scanner-effect attribution therefore remain pending.

Local regression tests passed 142 cases. Synthetic test messages exercise
parsing and state handling. They provide no live-feed performance evidence.

The KIS authorization workflow also failed at missing app credentials on
November 26. Both live sources require account setup before a real-data
success can be recorded.
