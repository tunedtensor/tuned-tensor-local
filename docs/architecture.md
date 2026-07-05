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
same `--input`/`--output` JSON files as the bundled Transformers evaluator, so a
custom workflow can load any artifact format as long as it writes compatible
evaluation results.

## Local State Layout

The local store is intentionally transparent and easy to back up:

- `metadata.sqlite`
- `specs/<spec-id>/spec.json`
- `runs/<run-id>/request.json`
- `runs/<run-id>/state.json`
- `runs/<run-id>/progress.jsonl`
- `runs/<run-id>/run-report.json`
- `models/<model-id>/model.json`

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
