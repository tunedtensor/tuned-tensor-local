# Architecture

TT Local is organized around a small set of replaceable local
adapters. The first implementation targets a single Linux GPU host and is
packaged as a standalone CLI.

## Subsystems

- Local orchestrator: runs the workflow states in-process and records progress
  to local files.
- Local project spec: `tt-local init` creates a local `tunedtensor.json`, and
  `tt-local run` converts that spec into the same run request contract used by
  compatible hosted exports.
- Study benchmark validator: keeps classic ML task/data contracts separate
  from fine-tuning requests and writes deterministic integrity locks for
  predefined training, validation, and test CSVs.
- Study trial runner: projects one locked benchmark into a command-facing
  training CSV and label-blind validation CSV, validates ID-keyed
  probabilities, and writes trusted validation metrics and trial artifacts.
- Artifact store: writes datasets, logs, model outputs, and reports beneath a
  configured local artifact root.
- Local state store: keeps a SQLite metadata index for specs, runs, progress
  events, and model records while preserving JSON/JSONL files under a
  configured `storeRoot` for portable artifacts, recovery, and inspection.
- Training adapter: launches either a uv-managed Python process or an arbitrary
  command with local input, model cache, and output directories passed through
  environment variables. Bundled uv training supports SFT and text-only offline
  DPO; command training can implement custom methods.
- Evaluation adapter: runs local Hugging Face/PEFT inference, an arbitrary
  batch command, or per-example commands for baseline and candidate evaluation.
  Evaluation can also use an OpenRouter LLM judge to score locally generated
  outputs.
- Report writer: emits structured JSON reports that are portable across local
  and hosted environments.
- Dashboard/API server: serves local run/model/spec metadata from the local
  state store and accepts local run submissions.
- Artifact verifier: validates the trained adapter/full-model contract, writes
  an atomic SHA-256 manifest, and registers the model before downstream
  evaluation so successful training survives later-stage failures.
- Model inference server: loads the recorded base model plus the verified local
  adapter and exposes localhost `/health`, `/v1/models`, and OpenAI-compatible
  `/v1/chat/completions` endpoints.

## Repository Boundary

This repository should remain a clean open-source implementation. Private
deployment code, proprietary provider adapters, account projection code, and
billing integrations stay out of this project.

The hosted Tuned Tensor CLI can keep cloud-specific workflows such as auth,
accounts, billing, and managed runs. `tt-local` must be useful on its own: local
spec creation, validation, execution, run tracking, model inspection, and the
dashboard should not depend on the hosted CLI.

## First Milestone

The first runnable milestone should support:

- one local run at a time;
- one GPU device;
- local filesystem artifacts;
- SQLite-backed run/model/spec tracking with recoverable JSON files;
- standalone `tt-local init`, `validate`, `run`, `serve`, and inspection
  commands;
- uv-based or command-based training;
- native Transformers/PEFT, batch-command, or per-example command evaluation
  with optional OpenRouter judge scoring;
- verified model manifests and local adapter serving;
- local dashboard and CLI inspection commands;
- a tiny smoke-run fixture that finishes quickly.

## Configuration Shape

The public runner should accept a plain JSON run request and a local runner
configuration. The run request describes the training method, behavior spec,
base model, examples, datasets, and hyperparameters. The local runner
configuration describes artifact paths, store paths, uv or command entrypoint
settings, model cache paths, OpenRouter judge settings, and timeout limits.

## Study Benchmark Boundary

`StudySpec` is a parallel, model-independent versioned contract rather than
another variant of the fine-tuning run request. Its initial task is binary
classification with explicit ID, input, target, label, and primary-metric
fields. Algorithms, hyperparameters, trials, and search policy belong to later
trial records, not the benchmark definition.

`tt-local studies lock` validates three explicit CSV splits and writes an
adjacent deterministic lock. The lock contains the canonical StudySpec hash
and each split's exact byte hash, byte size, row count, and common ordered
columns. A StudySpec may also declare a fixed label horizon and embargo with
event-time and observed label-end columns. In that case the validator requires
strict UTC timestamps, checks every observed label endpoint is inside its
declared window, and requires each later split to begin strictly after the
previous split's maximum event time plus the full horizon and embargo. Its lock
then records the temporal policy and exact per-split event and label-end
ranges. It deliberately contains no absolute paths, environment state, UUIDs,
or class prevalence. `tt-local studies validate` only recomputes and compares
this evidence; it never rewrites the lock.

The validator rejects malformed CSV, ambiguous or incompatible headers,
missing role columns, undeclared labels, duplicate IDs, and IDs reused across
splits. Every split must contain both declared labels, but no balance policy is
imposed. Input columns are always allowlisted explicitly. This is an integrity
boundary, not proof that a benchmark is leakage-free. Temporal certification
audits only the caller-declared timestamps, fixed horizon, and split-boundary
inequality; it uses the full declared horizon even when observed future
evidence ends early. It does not prove complete label coverage, correct
labels, feature causality or availability, row ordering inside a file,
correlated entity-group isolation, near-duplicate separation, cadence, class
balance, or command isolation from test files. Test data remains readable.
Those need upstream data audits and isolated evaluation before an autonomous
research loop can claim a sealed test result.

`tt-local studies run` executes one versioned trial spec without opening the
fine-tuning run store. Each trial has a filesystem-safe immutable ID, finite
timeout, free-form parameter record, and either a bundled runner identifier or
a direct argv command with an optional relative working directory. Direct
commands also declare exact source files and zero or more dependency-lock
files, all relative to the trial-spec directory. TT Local claims a new
directory for that ID and records its projected inputs, command log, raw
predictions, optional model files, implementation snapshot, deterministic
manifests, and atomic report. The ID is write-once within that output root; a
failed ID is retained for diagnosis and never reused. By default a custom child
starts in this directory. Explicit working directories resolve from the
trial-spec directory.

The first bundled algorithm is `numeric_logistic_regression`: a deliberately
narrow supervised baseline for numeric binary-classification features. It
uses median imputation with missingness indicators, standard scaling, and
scikit-learn logistic regression. The runner accepts only regularization,
class weighting, iteration-limit, and seed parameters. Its PEP 723 script and
adjacent `uv` lock create a small isolated classic-ML environment, separate
from the Transformer training project. It writes an atomic fitted-pipeline
artifact and a manifest containing the normalized configuration, data counts,
runtime versions, and model hash before publishing predictions.

TT Local passes the child only:

- training CSV: ID, explicitly allowlisted inputs, and a target encoded as
  `0` or `1`;
- validation CSV: ID and explicitly allowlisted inputs, with no target;
- a model-artifact directory and caller-provided trial/task semantics.

The child does not receive the StudySpec, benchmark lock, original split
paths, validation labels, or any test path, hash, rows, or labels. It returns a
strict protocol-v1 JSON object containing exactly one positive-class
probability for every declared validation ID. Results are joined by ID, never
position. IDs are not audited for embedded label or future information. TT
Local rejects candidate labels and metrics, computes
tie-independent average precision and ROC AUC plus F1 at the fixed `0.5`
threshold, and publishes only aggregate metrics in the report.

The runner uses direct process arguments, process-group timeout teardown, a
reduced machine-learning environment, bounded prediction JSON, post-command
input hashes, and pre/post checks of its declared implementation files. TT
Local snapshots those implementation bytes before launch. The bundled runner
automatically binds its packaged PEP 723 script and adjacent lock; a custom
runner must declare at least one source file and may declare an empty lock
list. Exact files are used instead of directories or inferred import graphs.

After successful execution, TT Local independently inventories the model tree.
Both the implementation and model manifests are deterministic, contain
per-file sizes and SHA-256 hashes, and are referenced by digest from the
report. Model symlinks, special files, root replacement, drift while hashing,
and implementation drift prevent report publication. An empty model tree
remains valid for generic command runners.

These are protocol and provenance safeguards, not an execution sandbox,
credential/network isolation, or hermetic runtime attestation: same-user code
can inspect the host filesystem, escape the process group, mutate and restore
files between checks, or repeatedly query aggregate validation scores.
Declared lock evidence does not prove a custom command consumed that lock, and
the interpreter, operating system, native libraries, environment, and network
inputs are not captured. Names, parameters, and command arguments are
persisted and must not contain secrets.

For a promoted bundled candidate, `studies test` adds a local one-shot
held-out workflow. It completes candidate and validation replay preflight
before atomically claiming the path-independent test content and target
semantics in a global `TT_LOCAL_HOME` ledger. After the claim, only a
label-free projection reaches the frozen predictor; TT Local computes metrics
and publishes either a success receipt or metric-free failure evidence. A
crashed or incomplete claim remains consumed.

The ledger is a reproducibility guard, not sealed evaluation: the same user
can change the home path, delete the ledger, or read source data directly.
Strong process isolation, externally controlled test custody, runtime
attestation, validation query budgets, and entity-aware split certification
remain separate future boundaries.

## Evaluation Loop

The default Transformers loop intentionally compares like-for-like model artifacts:

- baseline loads the original Hugging Face base model from the behavior spec;
- training writes a Hugging Face/PEFT artifact under the run directory;
- candidate evaluation loads the same base model plus that fine-tuned artifact;
- the report compares the two eval reports and stores generated outputs.

The first native backend is `transformers`. `llama.cpp`/GGUF conversion is not
part of the default path because conversion would make the baseline/candidate
comparison less direct.

Custom model families can use `training.backend: "command"` and
`evaluation.inference.provider: "batch_command"`. Batch evaluators receive the
same versioned `--input`/`--output` JSON protocol as the bundled Transformers
evaluator, so a custom workflow can load any artifact format. Protocol v2 input
examples contain an opaque `id`, `input`, and optional input assets or modality,
but never the expected output. Evaluators return exactly one prediction for
each input using the same `id`, a string `actual`, and a non-negative integer
`latency_ms`. TT Local rejects missing, duplicate, unknown, or malformed
predictions and joins them to trusted prompts and expected outputs before
scoring.

This protocol boundary prevents accidental label access through evaluator input;
it is not a process sandbox. Command-backed evaluators still require a trusted
host until isolated execution is added.

`paths.modelCache` is a single cache contract: it is passed as Hugging Face
`HF_HOME`, and Hub snapshots live beneath `<modelCache>/hub`. The environment
is finalized before importing Hugging Face or Transformers in every bundled
Python process. Prefetch records the resolved snapshot revision, file count,
and byte count; `models verify-base` performs a local-only structural check of
the cached config, tokenizer, weights, and indexed shards.

An explicitly configured `paths.baseModel` is a complete Hugging Face snapshot
directory, never a generic weights file or archive. Its file contents are
fingerprinted for stage identity. Remote runs should set an immutable
`base_model_revision`; baseline caching is bypassed when the model or a remote
input lacks stable content identity.

## Local State Layout

The local store is intentionally transparent and easy to back up:

- `metadata.sqlite`
- `specs/<spec-id>/spec.json`
- `runs/<run-id>/request.json`
- `runs/<run-id>/state.json`
- `runs/<run-id>/progress.jsonl`
- `runs/<run-id>/run-report.json`
- `models/<model-id>/model.json`

Each successful real training stage also writes `artifact-manifest.json` with
the model contract and SHA-256/size entries for immutable files. Dry runs do
not create model records. `tt-local models verify` accepts a model ID, exact
artifact path, or its manifest path and re-hashes the model before it is served
or handed off.

The default Transformers/PEFT contract requires exact adapter weights plus a
non-empty `adapter_config.json`; a full-model contract requires recognized
Transformers model weights. Command backends may publish another layout only
with an explicit framework/format contract, and such artifacts are not
implicitly servable.

SQLite is the primary metadata index for CLI and dashboard listings. The
per-object JSON files remain the recoverable source for artifacts and
human-readable inspection. `tt-local store rebuild-index` can reconstruct the
SQLite index from the canonical per-object files.

## Dashboard API

The local server is a lightweight Node HTTP server backed by the local state
store. Current endpoints include:

- `GET /api/health`
- `GET /api/runs`, `GET /api/runs/:id`, `GET /api/runs/:id/events`
- `GET /api/runs/:id/events/stream`
- `GET /api/runs/:id/report`
- `POST /api/runs`
- `POST /api/runs/:id/cancel`
- `GET /api/models`, `GET /api/models/:id`
- `GET /api/specs`, `GET /api/specs/:id`

## Failure Handling

The orchestrator should fail fast when prerequisites are missing. Common checks
include uv availability, Python dependency resolution, GPU visibility, model
cache permissions, artifact directory writability, and dataset size limits for
the selected model.
