# TT Local

[![CI](https://github.com/tunedtensor/tuned-tensor-local/actions/workflows/ci.yml/badge.svg)](https://github.com/tunedtensor/tuned-tensor-local/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/@tuned-tensor/local)](https://www.npmjs.com/package/@tuned-tensor/local)

TT Local is a local-first experimentation system for fine-tuning and classic
machine learning. It can turn a behavior spec into a fine-tuned small
open-weight model with a paired evaluation report, or run reproducible
algorithm trials against a locked tabular benchmark. Specs, datasets, model
artifacts, events, reports, and dashboard state stay on local disk.

Usage docs:

https://tunedtensor.com/docs/local-training

## Install

```bash
npm install -g @tuned-tensor/local
tt-local info
```

The bundled SFT, DPO, Transformers evaluator, and classic-ML paths also need
`uv`:

```bash
uv --version
```

Custom training or evaluation workflows can use command entrypoints instead of
`uv`. Inference protocol v2 omits expected outputs from command payloads; TT
Local joins returned ID-keyed predictions to its references for scoring. See
`docs/architecture.md`.

The default uv project is included in the npm package at
`training/local-runner`; using bundled training does not require cloning this
repository.

## ML Study Benchmarks

TT Local has a separate, model-independent StudySpec foundation for classic ML
work. It locks predefined binary-classification CSV splits, then runs
independent bundled or command-backed algorithm trials against the same
training and validation benchmark.

Define the task and explicitly allowlist model inputs:

```json
{
  "schema_version": 1,
  "name": "Portfolio anomaly benchmark",
  "task": {
    "type": "binary_classification",
    "id_column": "id",
    "input_columns": ["spread_bps", "liquidity", "imbalance_3c"],
    "target_column": "big_move",
    "labels": { "negative": "0", "positive": "1" },
    "primary_metric": "average_precision"
  },
  "dataset": {
    "format": "csv",
    "splits": {
      "training": "data/training.csv",
      "validation": "data/validation.csv",
      "test": "data/test.csv"
    },
    "temporal": {
      "policy": "ordered_purged",
      "event_time_column": "observed_at",
      "label_end_time_column": "future_observed_at",
      "label_horizon_seconds": 3600,
      "embargo_seconds": 300
    }
  }
}
```

Then create and verify the adjacent integrity lock:

```bash
tt-local studies lock portfolio.study.json
tt-local studies validate portfolio.study.json
```

`studies lock` refuses to replace an existing lock unless `--force` is
explicit. `studies validate` is read-only. It checks exact file hashes,
ordered columns, row counts, the presence of both declared labels, duplicate
IDs, and ID overlap between splits. Portable relative CSV paths resolve from
the StudySpec. `studies run` performs this validation itself, so a separate
validate command is not required before every trial.

The optional `dataset.temporal` contract certifies time-indexed splits without
changing non-temporal tasks. Both time columns must contain strict RFC 3339 UTC
timestamps ending in `Z`, with up to nanosecond precision. For every row, the
observed label endpoint must be after its event time and no later than the
declared label horizon. For both training-to-validation and
validation-to-test, the next split's earliest event must be strictly later
than the previous split's latest event plus the full declared horizon and
embargo:

```text
next.min(event_time) >
previous.max(event_time) + label_horizon_seconds + embargo_seconds
```

TT Local scans extrema, so this check does not depend on physical CSV row
order and does not rewrite the files. The lock records the declared policy
plus exact event-time and observed label-end ranges for all three splits.
Successful iterative trial reports repeat only the training and validation
ranges while binding the full lock by hash. Purging uses the full declared
horizon rather than the latest available `label_end_time_column` value, which
may represent only partial future coverage.

The explicit `input_columns` allowlist is important: label-derived or
future-derived export columns must never become model inputs accidentally. The
declared label-end column is always forbidden as an input and is not projected
to a trial child. The event-time column is projected only when it is explicitly
allowlisted.

This is a declared horizon-purge certificate, not proof that a benchmark is
leakage-free. It does not prove complete sampling through the label horizon,
label correctness, feature availability or causality, chronological row order
inside a file, entity-group isolation, near-duplicate separation, cadence, or
class balance beyond requiring both labels. Test rows and labels remain
readable: the lock detects changes but does not seal the test set or stop trial
code from accessing it. Establish those additional properties upstream and
keep test data out of the trial loop until isolated evaluation is available.

Define one immutable algorithm attempt separately from the benchmark:

```json
{
  "schema_version": 1,
  "id": "logreg-c1-v1",
  "name": "Balanced numeric logistic regression",
  "runner": {
    "builtin": "numeric_logistic_regression",
    "timeout_ms": 300000
  },
  "parameters": {
    "c": 1,
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_seed": 42
  }
}
```

Then run it:

```bash
tt-local studies run portfolio.study.json logreg-c1-v1.trial.json
```

The bundled runner treats every allowlisted input as numeric, median-imputes
blank cells, adds missingness indicators, standardizes the features, and fits
scikit-learn logistic regression. Its four parameters are strict:
`c` must be between `1e-6` and `1e6`, `class_weight` is `none` or `balanced`,
`max_iter` is between `1` and `10000`, and `random_seed` is an unsigned
32-bit integer. It rejects nonnumeric or non-finite cells and features that
are entirely missing in training.

The first run uses `uv` to fetch a small isolated environment from the bundled
script lock; it does not install or load TT Local's Transformer/PyTorch
runtime. The fitted pipeline is saved as `model/model.joblib`, alongside a
manifest with normalized parameters, data counts, runtime versions, and the
model hash. `class_weight: "balanced"` is often useful for rare-event ranking,
but its probabilities should not be interpreted as the event's natural-rate
calibration without a separate calibration step.

TT Local creates
`.tt-local/study-trials/logreg-c1-v1/`, appends
`--input <trial-input.json> --output <predictions.json> --artifact-dir <model>`
to a command runner, and keeps the projected data, command log,
predictions, model files, implementation snapshot, deterministic provenance
manifests, and `trial-report.json` together. Those three flags are reserved for
command runners. A custom runner's `cwd` resolves from the trial-spec
directory; when omitted, it starts in its new trial directory. A trial ID is
write-once within the selected output root, including after a failed attempt,
so use a new ID for each algorithm or parameter change.

The runner input points to two CSVs. Training contains only the ID,
allowlisted features, and a target normalized to `0` or `1`. Validation
contains only the ID and allowlisted features.

For an algorithm that is not bundled, use a direct command runner instead:

```json
{
  "command": ["python3", "scripts/custom_trial.py"],
  "cwd": ".",
  "timeout_ms": 300000,
  "provenance": {
    "source_files": ["scripts/custom_trial.py"],
    "dependency_lock_files": ["uv.lock"]
  }
}
```

Every command runner must declare at least one source file and may declare an
empty dependency-lock list. These are exact portable paths relative to the
trial-spec directory, independent of `cwd`; directories, globs, symbolic
links, and paths outside that directory are rejected. TT Local snapshots the
declared bytes before launch and refuses to publish a report if an original or
snapshot changes during the trial. The bundled runner declares its packaged
script and adjacent lock automatically.

Parameters are caller-provided JSON: TT Local records and passes them, but a
custom runner must validate and apply them. For example:

```python
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--artifact-dir", required=True)
args = parser.parse_args()

request = json.loads(Path(args.input).read_text())
task = request["task"]
params = request["trial"]["parameters"]
id_dtype = {task["id_column"]: "string"}
training = pd.read_csv(request["datasets"]["training_csv"], dtype=id_dtype)
validation = pd.read_csv(request["datasets"]["validation_csv"], dtype=id_dtype)

model = LogisticRegression(
    C=float(params["c"]),
    random_state=int(params["random_seed"]),
    max_iter=1000,
)
model.fit(training[task["input_columns"]], training[task["target_column"]])
positive_index = list(model.classes_).index(1)
probabilities = model.predict_proba(
    validation[task["input_columns"]]
)[:, positive_index]
joblib.dump(model, Path(args.artifact_dir) / "model.joblib")

result = {
    "protocol_version": 1,
    "predictions": [
        {"id": str(row_id), "probability": float(probability)}
        for row_id, probability in zip(
            validation[task["id_column"]], probabilities, strict=True
        )
    ],
}
Path(args.output).write_text(json.dumps(result))
```

The command must write exactly one positive-class probability per validation
ID:

```json
{
  "protocol_version": 1,
  "predictions": [
    { "id": "validation-2", "probability": 0.82 },
    { "id": "validation-1", "probability": 0.06 }
  ]
}
```

Prediction order is irrelevant. TT Local rejects missing, duplicate, unknown,
non-finite, or out-of-range results and computes average precision, ROC AUC,
and F1 at the fixed `0.5` threshold itself. Candidate-supplied labels or
metrics are not accepted.

The report records the primary score, all three trusted metrics, trial
parameters, and hashes for the StudySpec, benchmark lock, projected data, and
predictions:

```json
{
  "trial": {
    "id": "logreg-c1-v1",
    "spec_sha256": "...",
    "parameters": {
      "c": 1,
      "class_weight": "balanced",
      "max_iter": 1000,
      "random_seed": 42
    }
  },
  "evaluation": {
    "primary_score": 0.84,
    "metrics": {
      "average_precision": 0.84,
      "roc_auc": 0.88,
      "f1_at_0_5": 0.73
    },
    "decision_threshold": 0.5,
    "predictions_sha256": "..."
  },
  "provenance": {
    "implementation": {
      "manifest": "implementation/manifest.json",
      "sha256": "...",
      "file_count": 2,
      "evidence": "bundled_locked"
    },
    "model": {
      "manifest": "model-manifest.json",
      "sha256": "...",
      "file_count": 2,
      "size_bytes": 18432
    }
  }
}
```

The v1 contract requires positive-class probabilities rather than arbitrary
anomaly scores; calibration quality remains the model author's
responsibility. A raw Isolation Forest or One-Class SVM score must be
converted to a defensible probability before it is returned. A general
ranking-score task is a future contract.

This is a target-free validation projection, not a sandbox or confidentiality
boundary. TT Local does not pass validation targets or test material through
the command protocol, but IDs may themselves carry information and same-user
code can inspect other host files, credentials under the home directory, or
the network. Test data is never scored by `studies run`, but remains readable
at its original path. Repeated validation scores are also an adaptive oracle.
Use this loop for algorithm development, not as a sealed test result.

The report binds the exact benchmark snapshot checked before launch. Its
`provenance` references `implementation/manifest.json`, which records and
hashes the snapshotted source and dependency locks, and
`model-manifest.json`, which independently inventories every regular file
under `model/`. Model directories may be empty for generic command runners;
symbolic links, special files, replacement of the model root, or files that
change while being read prevent report publication.

This evidence is deliberately narrower than a hermetic build attestation. TT
Local does not capture the interpreter, operating system, environment, native
libraries, or network inputs, and it cannot prove that a custom command
actually used its declared dependency lock. Same-user code may also mutate and
restore host files between checks. Use a locked command and immutable runtime
when those factors matter. Do not put secrets in command arguments, names, or
parameters: those values are persisted in inputs, reports, logs, and CLI
output. Declared source and lock files are also copied into the trial
directory, so they must not contain credentials or other secrets.

Once you have deliberately selected one bundled logistic-regression trial,
freeze it as the Study candidate:

```bash
tt-local studies promote portfolio.study.json logreg-c1-v1.trial.json
```

If the trial used `--output-root`, point promotion at its artifacts with
`--trial-directory`. Promotion revalidates the benchmark, trial report,
projected data, implementation snapshot, fitted-model manifests, and saved
runner metadata before it creates anything. It then copies the model, original
implementation manifest, predictor source, and dependency lock into
`.tt-local/study-candidates/<study-file>/`. Replay uses the installed bundled
predictor only after proving its source and lock are byte-identical to the
frozen copies and consistent with the manifest, then loads the copied model
against the label-free validation projection. The final directory and
`candidate.lock.json` appear only when
the probabilities reproduce the selected trial within `1e-12`. Failed
attempts clean up their private staging directory. The source and promoted
model files are separate files, not hard links.

Candidate selection is explicit and write-once per Study file; there is no
automatic “pick the best” step or `--force` option. This first promotion
contract supports only
`builtin:numeric_logistic_regression`. Command-backed trials need a separate
versioned saved-model prediction contract before they can be promoted safely.
Promotion still uses validation data and does not score or expose the held-out
test split. “Write-once” is a local workflow protocol, not host-level sealing:
the same operating-system user can still alter or delete the candidate
directory.

## First Local Run

On an NVIDIA Spark or another CUDA host, initialize both the behavior spec and
a durable project-local runner config:

```bash
tt-local --version
tt-local init --name "Support Bot" --model Qwen/Qwen3.5-2B --profile spark
```

Edit `tunedtensor.json` to replace the generated system prompt and example.
TT Local refuses to validate or train the unchanged placeholder so an
accidental run cannot spend GPU time on it.

Then preflight the exact Python environments and paths that the run will use:

```bash
tt-local doctor tunedtensor.json --config local-runner.json
tt-local validate tunedtensor.json --config local-runner.json
```

`doctor` checks the bundled uv environment, Torch/Transformers/PEFT imports,
the requested CUDA/MPS device, judge credentials when configured, writable
artifact/cache/store roots, and available disk space. GPU checks are skipped
for dry runs and are conditional for CPU/MPS configurations.

Download the base model explicitly, then start the run:

```bash
tt-local models prefetch tunedtensor.json --config local-runner.json
tt-local models verify-base tunedtensor.json --config local-runner.json
tt-local run tunedtensor.json --config local-runner.json
```

Prefetch progress is visible by default. Pass `--quiet` only when another
program is consuming the final JSON. `verify-base` is local-only: it checks
the cached snapshot structure, indexed weight shards, tokenizer files, and
non-empty weights without downloading a replacement. Cached Hugging Face blobs
whose ETags encode Git or LFS checksums are re-hashed as part of verification.

For a long run, start it in the background and watch it from the current or a
second terminal:

```bash
tt-local run tunedtensor.json --config local-runner.json --detach
tt-local runs watch <run-id> --config local-runner.json
```

The detached command returns the persisted run ID, process ID, log path, and
copy-paste watch/cancel commands before model work begins.

Inspect the evidence after the run:

```bash
tt-local runs list --config local-runner.json
tt-local runs report <run-id> --config local-runner.json
tt-local models verify local-<run-id> --config local-runner.json
```

Treat the report as evidence on this run's evaluation cases, not as a
guarantee of live application improvement.

When a prebuilt dataset provides both validation and test files, normal runs
evaluate the validation split. The test split remains a separate holdout.
A dataset with only a test split continues to evaluate that test split.

The model is registered immediately after its artifact passes verification,
so it remains discoverable if candidate evaluation or report generation later
fails.

## Serve a Trained Adapter

`tt-local serve` is the run dashboard. To serve a trained TT Local adapter as
an OpenAI-compatible model endpoint, use the nested model command:

```bash
tt-local models serve local-<run-id> \
  --config local-runner.json \
  --spec tunedtensor.json \
  --host 127.0.0.1 \
  --port 8000
```

The server verifies the stored artifact, loads the recorded base model and
PEFT adapter through the bundled uv environment, and binds to localhost by
default. Test it from another terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

Use `--device cuda|mps|cpu|auto` to override the configured inference device.
Non-loopback binds require both `--allow-remote` and
`--api-key-env <environment-variable>`; requests must then send that value as
a Bearer token. The server accepts multimodal images only as bounded `data:`
URIs and does not support streaming yet.

## Cache and Evaluation Semantics

`paths.modelCache` means Hugging Face `HF_HOME` everywhere. Hub snapshots live
under `<modelCache>/hub`; prefetch, training, evaluation, doctor, and serving
all use that same layout. Set `HF_TOKEN` in the environment or project `.env`
before prefetching a gated model. Do not commit `.env`.

For reproducible remote loading, set
`hyperparameters.base_model_revision` to an immutable Hugging Face commit.
Alternatively, `paths.baseModel` may point to a complete local Hugging Face
snapshot directory. It cannot point to a model archive or standalone weights
file; TT Local requires non-empty model weights, `config.json`, tokenizer
metadata and vocabulary, and every indexed shard. Local snapshot contents are
fingerprinted so a byte change invalidates dependent stages.

Shared baseline-cache entries are used only when the model and every local
input have a stable content identity. An unpinned remote model or remotely
hosted image bypasses the cache rather than risking stale evaluation results.

The safe default scoring mode is deterministic `exact_match`. To use an
OpenRouter judge, configure `evaluation.scoring.mode: "llm_judge"`, an `llm`
block, and its API-key environment variable. Judge mode fails preflight when
those are missing. Exact-match fallback occurs only when explicitly requested
with `evaluation.scoring.fallback: "exact_match"`; fallback-scored results are
marked and never written to the judge baseline cache.

Before the first real run, you can download the configured Hugging Face base
model into `paths.modelCache` explicitly:

```bash
tt-local models prefetch tunedtensor.json --config local-runner.json
tt-local models verify-base tunedtensor.json --config local-runner.json
```

Without this step, the first non-dry `tt-local run` downloads the base model
when baseline evaluation or training first loads it.

Individual `run --stage ...` commands finish in the terminal
`stage_completed` state, so `runs watch` exits instead of waiting for stages
that were intentionally not requested.

Relative paths in a runner config resolve from the config file's directory.
Relative dataset and image paths resolve from the behavior spec or request
file's directory. The published package includes `uv.lock` so bundled Python
dependencies are reproducible.

## DPO

TT Local supports first-class offline DPO for text causal-LM models. Set
`training_method` to `dpo` and provide a prebuilt preference JSONL training
dataset:

```json
{
  "training_method": "dpo",
  "dataset_prebuilt": {
    "training": "file://examples/dpo-preferences.jsonl",
    "format": "preference_jsonl"
  }
}
```

Each preference JSONL row must use explicit `prompt`, `chosen`, and `rejected`
string fields:

```json
{"prompt":"Summarize status: build passed.","chosen":"Build passed.","rejected":"The build failed."}
```

DPO v1 is text-only for the bundled `uv` trainer. Validation and reporting still
use the existing baseline-vs-candidate evaluation loop, so provide
`dataset_prebuilt.validation`, `dataset_prebuilt.test`, or normal
`spec_snapshot.examples` with reference outputs for evaluation.

Command-backed workflows may use external model ids by setting
`spec_snapshot.base_model` to an `external:<id>` or `command:<id>` value, for
example `external:karpathy/nanochat`. The bundled `uv` trainer still requires a
supported Hugging Face base model, but command trainers receive adapter-focused
hyperparameters without injected LoRA/model-loader defaults. Custom
hyperparameter keys are passed through to `TT_HYPERPARAMETERS_PATH`.
Set `hyperparameters.base_model_revision` to an immutable Hugging Face commit
when a run must load and record an exact base-model revision.

## Continuing From a Fine-Tuned Model

Start another loop from an existing TT Local model by passing the stored local
model id:

```bash
tt-local run tunedtensor.json --parent-model local-<previous-run-id>
```

You can also set the parent adapter artifact explicitly:

```bash
tt-local run tunedtensor.json --parent-model-artifact file:///path/to/model.tar.gz
```

For config-only workflows, put the same value in
`hyperparameters.parent_model_artifact`. The parent adapter becomes the
baseline for the new run, and the bundled SFT/DPO trainers load it before
continuing LoRA training.

Stored `--parent-model` references are verified before launch. TT Local pins
the child's base revision from the parent (or requires the same configured
local snapshot content), rejects conflicting base identity, and fingerprints
the parent artifact so changed bytes invalidate reuse. The explicit
`--parent-model-artifact` form is for caller-managed local artifacts and should
be used only when that provenance is understood.

Non-Hugging Face artifacts can describe their layout in `training.artifact`:

```json
{
  "training": {
    "backend": "command",
    "command": ["python", "train_adapter.py"],
    "artifact": {
      "framework": "nanochat",
      "format": "custom-directory",
      "entrypoint": "batch_command",
      "servable": false
    }
  }
}
```

## Local Development

```bash
npm install
npm run typecheck
npm test
npm run build
```

Release notes are in [CHANGELOG.md](CHANGELOG.md).

## License

Apache-2.0
