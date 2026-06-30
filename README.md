# Tuned Tensor Local

Tuned Tensor Local is an open-source local runner for fine-tuning small
open-weight models from behavior specs on a machine you control.

The project is a standalone command line tool. It is designed to run without a
required cloud provider or the hosted Tuned Tensor CLI, with local disk
artifacts, local process orchestration, file-backed tracking, and uv-managed
Python GPU execution.

## Goals

- Run a complete fine-tuning workflow on a single GPU host.
- Keep training data, model artifacts, logs, and reports on local disk.
- Use uv for lightweight Python dependency and environment management.
- Produce portable run reports that can be inspected or integrated elsewhere.
- Keep the public implementation independent from Tuned Tensor's hosted runner.
- Let users create, validate, run, inspect, and serve local jobs entirely through
  `tt-local`.

## Non-Goals For The Initial Version

- Distributed scheduling.
- Hosted product billing, account, or deployment integrations.
- Managed cloud training or serving providers.
- A compatibility promise for private internal service code.

## Workflow

1. Read a behavior spec and run configuration.
2. Compile examples into training and evaluation datasets.
3. Run a baseline evaluation against the original Hugging Face model.
4. Launch local GPU fine-tuning through uv.
5. Run candidate evaluation against the fine-tuned Hugging Face/PEFT artifact.
6. Write a structured report under the local artifact root.

## Current Status

This repository now contains a preview local runner. It can validate run
local specs or compatible run requests, compile datasets, write local artifacts,
run dry workflows, launch a configured uv training process, persist run state to
a local file-backed store, serve a small dashboard/API, and emit a structured
`run-report.json`.

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
tt-local init --name "Local Assistant" --output /tmp/tunedtensor.json --force
tt-local validate /tmp/tunedtensor.json --config examples/local-runner.json
tt-local run /tmp/tunedtensor.json --config examples/local-runner.json
tt-local runs list --config examples/local-runner.json
tt-local serve --config examples/local-runner.json
```

The example config uses `dryRun: true`, so it verifies the orchestration and
artifact flow without starting GPU training. It writes artifacts to
`.tt-local/artifacts` and run/model/spec metadata to `.tt-local/store`.

## Commands

```bash
tt-local info
tt-local init --name "Support Bot" --model Qwen/Qwen3.5-2B
tt-local doctor --config examples/local-runner.json
tt-local validate <tunedtensor.json|request.json> --config <local-runner.json>
tt-local run <tunedtensor.json|request.json> --config <local-runner.json>
tt-local serve --config <local-runner.json>
tt-local runs list --config <local-runner.json>
tt-local runs get <run-id> --config <local-runner.json>
tt-local runs events <run-id> --config <local-runner.json>
tt-local runs watch <run-id> --config <local-runner.json>
tt-local runs report <run-id> --config <local-runner.json>
tt-local models list --config <local-runner.json>
tt-local specs list --config <local-runner.json>
tt-local store rebuild-index --config <local-runner.json>
```

For Spark-specific notes, see [docs/spark.md](docs/spark.md).

## Standalone CLI Workflow

Install or link the package, then use `tt-local` directly:

```bash
npm install -g tuned-tensor-local
tt-local init --name "Support Bot" --model Qwen/Qwen3.5-2B
tt-local validate --config examples/local-runner.json
tt-local run --config examples/local-runner.json
tt-local serve --config examples/local-runner.json
```

By default, `init`, `validate`, and `run` use `tunedtensor.json` in the current
directory. The local spec file is intentionally compatible with the hosted
behavior spec shape, but this tool does not require hosted auth or the
centralized `tt` CLI.

## Local Store

Set `storeRoot` in the runner config to keep dashboard state, runs, specs, model
records, progress events, and copied reports in a portable local directory:

```json
{
  "artifactRoot": ".tt-local/artifacts",
  "storeRoot": ".tt-local/store"
}
```

If `storeRoot` is omitted, the runner uses `TT_LOCAL_HOME` or
`~/.tuned-tensor-local`.

## Native Evaluation

For real non-dry runs, `tt-local` can evaluate with the same model family and
artifact format used for training. The baseline loads the original Hugging Face
model. The candidate loads that same base model plus the fine-tuned PEFT/HF
artifact produced by training. This avoids GGUF conversion and keeps the
comparison fair.

```json
{
  "evaluation": {
    "inference": {
      "provider": "transformers",
      "project": "training/sft-local",
      "script": "training/sft-local/src/evaluate.py",
      "maxNewTokens": 256,
      "temperature": 0,
      "topP": 1
    },
    "scoring": {
      "mode": "llm_judge",
      "fallback": "exact_match"
    },
    "maxExamples": 10
  }
}
```

`dryRun: true` skips model loading and keeps evaluation cheap.

## OpenRouter Judge

The local runner can use OpenRouter for LLM-as-judge scoring while keeping
inference, training, and artifacts local:

```json
{
  "evaluation": {
    "mode": "llm_judge",
    "maxExamples": 10
  },
  "llm": {
    "provider": "openrouter",
    "model": "openai/gpt-5.5",
    "apiKeyEnv": "OPENROUTER_API_KEY",
    "appName": "Tuned Tensor Local"
  }
}
```

OpenRouter scores generated local outputs. It does not generate the baseline or
candidate responses.

For command-backed evaluation, set `evaluation.mode` to `command` and provide
`baselineCommand` and/or `candidateCommand`. The command receives JSON on stdin
and should print either plain text or JSON with `content`, `output`, or `actual`.

## Real Training Config

Set `dryRun` to `false` and point `training.script` at the SFT script:

```json
{
  "artifactRoot": "/home/eve/tt-local-artifacts",
  "storeRoot": "/home/eve/tt-local-store",
  "dryRun": false,
  "training": {
    "backend": "uv",
    "project": "training/sft-local",
    "script": "training/sft-local/src/train.py"
  },
  "evaluation": {
    "inference": {
      "provider": "transformers",
      "project": "training/sft-local",
      "script": "training/sft-local/src/evaluate.py"
    },
    "scoring": {
      "mode": "llm_judge",
      "fallback": "exact_match"
    }
  },
  "paths": {
    "modelCache": "/home/eve/.cache/huggingface"
  }
}
```

## License

Apache-2.0
