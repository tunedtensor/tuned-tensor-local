# Tuned Tensor Local

Tuned Tensor Local is an open-source local runner for fine-tuning small
open-weight models from behavior specs on a machine you control.

The project is starting as a clean standalone implementation. It is designed to
run without a required cloud provider, with local disk artifacts, local process
orchestration, and uv-managed Python GPU execution.

## Goals

- Run a complete fine-tuning workflow on a single GPU host.
- Keep training data, model artifacts, logs, and reports on local disk.
- Use uv for lightweight Python dependency and environment management.
- Produce portable run reports that can be inspected or integrated elsewhere.
- Keep the public implementation independent from Tuned Tensor's hosted runner.

## Non-Goals For The Initial Version

- Distributed scheduling.
- Hosted product billing, account, or deployment integrations.
- Managed cloud training or serving providers.
- A compatibility promise for private internal service code.

## Workflow

1. Read a behavior spec and run configuration.
2. Compile examples into training and evaluation datasets.
3. Run a baseline evaluation.
4. Launch local GPU fine-tuning through uv.
5. Run candidate evaluation against the tuned artifact.
6. Write a structured report under the local artifact root.

## Current Status

This repository now contains a preview local runner. It can validate run
requests, compile datasets, write local artifacts, run dry workflows, launch a
configured uv training process, and emit a structured `run-report.json`.

The training process is intentionally a separate contract: configure a uv
script or module that reads the local training environment variables and writes
model outputs under the run artifact directory.

This repo includes an initial compatible SFT script at
`training/sft-local/src/train.py`. It implements text SFT with Transformers +
PEFT LoRA and is launched by uv.

## Development

```bash
npm install
npm run typecheck
npm run build
npm test
```

## Local Smoke Run

```bash
npm run build
node dist/index.js validate examples/smoke-run-request.json --config examples/local-runner.json
node dist/index.js run examples/smoke-run-request.json --config examples/local-runner.json
```

The example config uses `dryRun: true`, so it verifies the orchestration and
artifact flow without starting GPU training.

## Commands

```bash
node dist/index.js info
node dist/index.js doctor --config examples/local-runner.json
node dist/index.js validate <request.json> --config <local-runner.json>
node dist/index.js run <request.json> --config <local-runner.json>
```

For Spark-specific notes, see [docs/spark.md](docs/spark.md).

## Real Training Config

Set `dryRun` to `false` and point `training.script` at the SFT script:

```json
{
  "artifactRoot": "/home/eve/tt-local-artifacts",
  "dryRun": false,
  "training": {
    "backend": "uv",
    "project": "training/sft-local",
    "script": "training/sft-local/src/train.py"
  },
  "paths": {
    "modelCache": "/home/eve/.cache/huggingface"
  }
}
```

## License

Apache-2.0
