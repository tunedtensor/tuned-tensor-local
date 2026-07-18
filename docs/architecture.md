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
