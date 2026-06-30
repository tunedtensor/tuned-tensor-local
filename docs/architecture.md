# Architecture

Tuned Tensor Local is organized around a small set of replaceable local
adapters. The first implementation targets a single Linux GPU host.

## Subsystems

- Local orchestrator: runs the workflow states in-process and records progress
  to local files.
- Artifact store: writes datasets, logs, model outputs, and reports beneath a
  configured local artifact root.
- Local state store: persists specs, runs, progress events, reports, and model
  records as JSON/JSONL under a configured `storeRoot` for the dashboard and CLI.
- Training adapter: launches a uv-managed Python process with local input,
  model cache, and output directories passed through environment variables.
- Serving and evaluation adapter: starts local model servers or direct inference
  processes for baseline and candidate evaluation. Evaluation can also use an
  OpenRouter LLM judge when configured.
- Report writer: emits structured JSON reports that are portable across local
  and hosted environments.
- Dashboard/API server: serves local run/model/spec metadata from the local
  state store and accepts local run submissions.

## Repository Boundary

This repository should remain a clean open-source implementation. Private
deployment code, proprietary provider adapters, account projection code, and
billing integrations stay out of this project.

## First Milestone

The first runnable milestone should support:

- one local run at a time;
- one GPU device;
- local filesystem artifacts;
- local file-backed run/model/spec tracking;
- uv-based training;
- heuristic, command, and optional OpenRouter judge evaluation;
- local dashboard and CLI inspection commands;
- a tiny smoke-run fixture that finishes quickly.

## Configuration Shape

The public runner should accept a plain JSON run request and a local runner
configuration. The run request describes the behavior spec, base model,
examples, and hyperparameters. The local runner configuration describes artifact
paths, store paths, uv entrypoint settings, model cache paths, OpenRouter judge
settings, and timeout limits.

## Local State Layout

The local store is intentionally transparent and easy to back up:

- `specs/<spec-id>/spec.json`
- `runs/<run-id>/request.json`
- `runs/<run-id>/state.json`
- `runs/<run-id>/progress.jsonl`
- `runs/<run-id>/run-report.json`
- `models/<model-id>/model.json`
- `catalog/*.jsonl`

Catalog files are append-only indexes for fast listing. `tt-local store
rebuild-index` can reconstruct them from the canonical per-object files.

## Dashboard API

The local server is a lightweight Node HTTP server, not a separate database.
Current endpoints include:

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
