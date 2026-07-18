# Polymarket large-move Study

This follow-along example trains a small classic-ML model to estimate whether
a Polymarket outcome midpoint will move by at least 500 basis points during the
next hour. It uses public order-book snapshots, a temporally purged benchmark,
the bundled numeric logistic-regression runner, validation-only iteration, and
a one-shot held-out test.

It is a reproducibility example, not a trading recommendation.

## 1. Collect and label public data

Clone the
[Hedge Layer data collector](https://github.com/hedge-layer/data-collector),
install it with `uv sync`, and collect at least 25 hours of snapshots so a
one-hour future label is available across multiple time splits:

```bash
uv run hl-data-collector collect \
  --limit 100 \
  --workers 12 \
  --require-two-sided \
  --min-mid 0.02 \
  --max-mid 0.98 \
  --no-price-history \
  --no-last-trade \
  --quality-every 1440 \
  --interval-sec 60 \
  --db data/snapshots.sqlite
```

Stop the collector after enough history exists, then create the future-move
label and export CSV:

```bash
uv run hl-data-collector label \
  --db data/snapshots.sqlite \
  --horizon-min 60 \
  --threshold-bps 500

uv run hl-data-collector export \
  --db data/snapshots.sqlite \
  --out data/labeled.csv
```

`threshold-bps 500` means a 5% absolute midpoint move. The feature projection
in this example uses only values observable at prediction time.

## 2. Prepare purged time splits

Copy or link `labeled.csv` into this example's `data/` directory. The dates
below reproduce the July 3–4, 2026 Spark run; choose boundaries inside your own
collection window. Both gaps exceed the one-hour label horizon plus the
five-minute embargo:

```bash
python3 prepare_splits.py data/labeled.csv data \
  --training-end 2026-07-03T23:00:00Z \
  --validation-start 2026-07-04T00:06:00Z \
  --validation-end 2026-07-04T08:00:00Z \
  --test-start 2026-07-04T09:06:00Z
```

The script filters to quoteable, two-sided, non-extreme books; selects at most
5,000 deterministic rows per split; sorts them chronologically; and prints
row counts, class balance, and time ranges. It never invents labels.

## 3. Lock and validate the benchmark

```bash
tt-local studies lock polymarket.study.json
tt-local studies validate polymarket.study.json
```

The lock records exact hashes, columns, row counts, disjoint IDs, both target
classes, and the declared temporal boundaries. Change the data or StudySpec
and validation will fail instead of silently changing the benchmark.

## 4. Run validation trials

```bash
tt-local studies run \
  polymarket.study.json \
  logreg-balanced-c1.trial.json
```

The bundled runner fits median imputation, missing-value indicators, feature
scaling, and balanced logistic regression. TT Local exposes labels only for
training; it gives the runner label-free validation features and computes
trusted metrics after joining predictions by opaque row ID.

Every algorithm or parameter attempt needs a new immutable trial `id`. Compare
trials on validation only. Do not look at the test result while choosing the
model.

## 5. Promote and test once

When the candidate is final:

```bash
tt-local studies promote \
  polymarket.study.json \
  logreg-balanced-c1.trial.json

tt-local studies test polymarket.study.json
```

Promotion freezes the fitted model and predictor and verifies that replayed
validation probabilities exactly reproduce the selected trial. `studies test`
then consumes the locked held-out identity once, sends only IDs and allowlisted
features to the predictor, computes trusted metrics, and writes a durable
receipt. A second test attempt is rejected, including after a failed or crashed
post-claim attempt.

The one-shot ledger is a workflow guard, not an adversarial sandbox. A user
with filesystem access can delete local state or read the source test file.

## Reference Spark result

The example was run end to end on an NVIDIA DGX Spark (GB10, ARM64) from a
packed install of TT Local `0.2.9` at commit `de90b04`. The source export held
82,837 labeled public snapshots; each prepared split contained 5,000 rows.

| Split | Positive rows | Positive rate | Average precision | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Training | 78 | 1.56% | — | — |
| Validation | 147 | 2.94% | 0.2806 | 0.8733 |
| Test (one shot) | 67 | 1.34% | 0.1680 | 0.8136 |

The fitted trial took 1.5 seconds and held-out prediction took 0.54 seconds.
Candidate replay covered all 5,000 validation rows with maximum absolute
probability difference `0`.
