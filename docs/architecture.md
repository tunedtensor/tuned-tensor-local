# Architecture

Tuned Tensor Local is organized around a small set of replaceable local
adapters. The first implementation targets a single Linux GPU host.

## Subsystems

- Local orchestrator: runs the workflow states in-process and records progress
  to local files.
- Artifact store: writes datasets, logs, model outputs, and reports beneath a
  configured local artifact root.
- Training adapter: launches a uv-managed Python process with local input,
  model cache, and output directories passed through environment variables.
- Serving and evaluation adapter: starts local model servers or direct inference
  processes for baseline and candidate evaluation.
- Report writer: emits structured JSON reports that are portable across local
  and hosted environments.

## Repository Boundary

This repository should remain a clean open-source implementation. Private
deployment code, proprietary provider adapters, account projection code, and
billing integrations stay out of this project.

## First Milestone

The first runnable milestone should support:

- one local run at a time;
- one GPU device;
- local filesystem artifacts;
- uv-based training;
- evaluation without external judge services;
- a tiny smoke-run fixture that finishes quickly.

## Configuration Shape

The public runner should accept a plain JSON run request and a local runner
configuration. The run request describes the behavior spec, base model,
examples, and hyperparameters. The local runner configuration describes artifact
paths, uv entrypoint settings, model cache paths, and timeout limits.

## Failure Handling

The orchestrator should fail fast when prerequisites are missing. Common checks
include uv availability, Python dependency resolution, GPU visibility, model
cache permissions, artifact directory writability, and dataset size limits for
the selected model.
