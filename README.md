# TT Local

Tuning tensors locally.

[![CI](https://github.com/tunedtensor/tuned-tensor-local/actions/workflows/ci.yml/badge.svg)](https://github.com/tunedtensor/tuned-tensor-local/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/@tuned-tensor/local)](https://www.npmjs.com/package/@tuned-tensor/local)

TT Local is a standalone CLI for running Tuned Tensor-style
fine-tuning jobs on a machine you control. It stores specs, datasets, model
artifacts, progress events, reports, and dashboard state on local disk.

It does not require the hosted Tuned Tensor CLI, cloud orchestration, Docker,
or a managed database.

## What Happens In A Run

`tt-local` runs this loop:

1. Read a behavior spec from `tunedtensor.json`.
2. Compile examples into a local training dataset.
3. Evaluate the original Hugging Face base model.
4. Fine-tune locally with uv, Transformers, and PEFT.
5. Evaluate the fine-tuned Hugging Face/PEFT artifact.
6. Compare baseline vs tuned outputs and write a report.

OpenRouter can optionally score generated outputs as an LLM judge. It does not
generate the baseline or tuned responses.

## Install

Install the published CLI:

```bash
npm install -g @tuned-tensor/local
tt-local info
```

For local development or Spark:

```bash
git clone https://github.com/tunedtensor/tuned-tensor-local.git
cd tuned-tensor-local
npm install
npm run build
npm link
tt-local info
```

Real training also needs `uv`:

```bash
uv --version
```

If uv is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## First Run: Safe Dry Run

The included example config is a dry run. It validates the end-to-end local
workflow and writes artifacts without downloading models or starting GPU
training.

Create a local behavior spec:

```bash
tt-local init --name "Support Bot" --model Qwen/Qwen3.5-2B
```

Edit `tunedtensor.json` and replace the placeholder example with your task:

```json
{
  "input": "Classify sentiment: I love this product.",
  "output": "positive"
}
```

Validate the spec and config:

```bash
tt-local validate --config examples/local-runner.json
```

Run the dry workflow:

```bash
tt-local run --config examples/local-runner.json
```

Inspect the result:

```bash
tt-local runs list --config examples/local-runner.json
tt-local runs report <run-id> --config examples/local-runner.json
```

Start the local dashboard:

```bash
tt-local serve --config examples/local-runner.json
```

Open the printed local URL in your browser.

## Run Real Training

Create a config such as `local-runner.json` or `spark-runner.json`:

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
  },
  "paths": {
    "modelCache": "/home/eve/.cache/huggingface"
  }
}
```

If you use OpenRouter judging, set an API key:

```bash
export OPENROUTER_API_KEY="your_openrouter_key"
```

`tt-local` also loads a `.env` file from the working directory. Existing
environment variables win.

Run the job:

```bash
tt-local doctor --config spark-runner.json
tt-local validate --config spark-runner.json
tt-local run --config spark-runner.json
```

`tt-local run` prints live stage updates to stderr and writes the final JSON
summary to stdout. Use `--verbose` to stream Python subprocess output, or
`--quiet` for JSON-only automation.

Watch progress:

```bash
tt-local runs watch <run-id> --config spark-runner.json
```

View the final report:

```bash
tt-local runs report <run-id> --config spark-runner.json
```

Compare two completed runs:

```bash
tt-local runs compare <run-id-a> <run-id-b> --config spark-runner.json
```

The comparison aligns shared eval prompts, separates newly added eval examples,
and measures judge score noise for identical baseline outputs.

## Configuration Basics

The main config fields are:

- `artifactRoot`: datasets, logs, model outputs, eval JSON, and run reports.
- `storeRoot`: run/spec/model metadata used by CLI listing and the dashboard.
- `dryRun`: when `true`, skips model loading and training.
- `training`: uv/Python training entrypoint and environment.
- `evaluation.inference`: how baseline and candidate outputs are generated.
- `evaluation.scoring`: how generated outputs are scored.
- `paths.modelCache`: Hugging Face cache location.

Supported inference providers:

- `"transformers"`: run the included local evaluator with Hugging Face/PEFT.
- `"command"`: call `baselineCommand` or `candidateCommand` for custom inference.
- `"none"`: skip inference and record empty local responses.

Supported scoring modes:

- `"exact_match"`: normalized exact match.
- `"llm_judge"`: OpenRouter judge, with exact-match fallback by default.
- `"json_fields"`: field-level comparison for structured JSON outputs.

When `evaluation.scoring.mode` is `"llm_judge"`, `fallback` controls judge
failures. `"exact_match"` keeps the run going and records fallback reasoning.
`"fail"` fails the run.

## Evaluation Behavior

`evaluation.maxExamples` caps how many examples are scored. If it is unset,
the request hyperparameter `max_eval_examples` is used. An explicit
`maxExamples` in the config wins. When capped, the runner uses a deterministic
seeded sample rather than a prefix, so sorted or grouped eval files do not
bias results. The seed defaults to a hash of the run id and can be pinned with
`evaluation.sampleSeed`.

For spec-based runs without `dataset_prebuilt`, the runner holds out about 20%
of the spec examples for evaluation, with at least one training example and one
holdout when possible. With only one spec example, evaluation runs on that
example and reports `eval_split: "spec_examples"`.

For prebuilt datasets, `dataset_prebuilt.training` is always copied into the
run artifact as the training set. Evaluation uses `dataset_prebuilt.test` when
present, then `dataset_prebuilt.validation`. If neither is provided, a real
run fails because evaluation would otherwise measure the training split. Set
`evaluation.allowPrebuiltTrainingEval: true` to allow that explicitly.

Baseline evaluations are cached under `<storeRoot>/cache/baseline-evals/`.
Re-running an unchanged spec with the same model, eval examples, generation
settings, and scoring config reuses the previous baseline report. Set
`evaluation.baselineCache: false` to disable.

Training progress from Python metric lines and tqdm progress bars is recorded
as run events, so `tt-local runs watch` and per-run `progress.jsonl` show
epoch, loss, step, percentage, and ETA when the trainer emits them.

## Structured JSON Scoring

For structured tasks, use field-level scoring instead of whole-output exact
matching:

```json
{
  "evaluation": {
    "scoring": {
      "mode": "json_fields",
      "fields": ["triage", "priority", "should_process"]
    }
  }
}
```

If `fields` is omitted, the evaluator scores every key in the expected JSON
object. A configured field missing from the expected JSON is always scored as
incorrect, so a misconfigured field list cannot inflate scores.

Reports still include `exact_match_rate`, but `avg_score`, `pass_rate`, and
`json_field_metrics` are based on the selected JSON fields.

## Thinking-Mode Models

Some models, such as `Qwen/Qwen3.5-4B`, may open a hidden thinking block by
default and consume the token budget before producing a visible answer. Forward
template kwargs to both evaluation and training:

```json
{
  "evaluation": {
    "inference": {
      "chatTemplateKwargs": { "enable_thinking": false }
    }
  }
}
```

In `tunedtensor.json`, also set:

```json
{
  "hyperparameters": {
    "chat_template_kwargs": { "enable_thinking": false }
  }
}
```

## Multimodal Examples

Multimodal runs use chat JSONL with structured user content. For
`image_text_to_text` models such as `Qwen/Qwen3-VL-2B-Instruct`, image parts
are loaded by the local trainer and evaluator:

```json
{
  "messages": [
    { "role": "system", "content": "Answer chart questions concisely." },
    {
      "role": "user",
      "content": [
        { "type": "image", "image": "charts/example.png" },
        { "type": "text", "text": "What is the blue value?" }
      ]
    },
    { "role": "assistant", "content": "42" }
  ]
}
```

Image values in prebuilt JSONL may be absolute paths, paths relative to the
JSONL file, `file://` URIs, HTTP(S) URLs, or `data:` URIs. Behavior spec
examples can also use `input_assets` with `image`, `path`, `uri`, or
`data_uri`.

## Where Files Go

If `storeRoot` is omitted, `tt-local` uses `TT_LOCAL_HOME` or
`~/.tuned-tensor-local`.

Per-run artifacts include:

- `progress.jsonl`: stage changes and parsed training progress.
- `baseline-eval.json` and `candidate-eval.json`: generated outputs and scores.
- `baseline-eval.json.inference.log` and `candidate-eval.json.inference.log`:
  local inference subprocess output.
- `training/training.log`: uv fine-tuning output.
- `run-report.json`: baseline, candidate, comparison, and artifact references.

## Useful Commands

```bash
tt-local init --name "Support Bot" --model Qwen/Qwen3.5-2B
tt-local doctor --config spark-runner.json
tt-local validate --config spark-runner.json
tt-local run --config spark-runner.json
tt-local run --config spark-runner.json --verbose
tt-local runs list --config spark-runner.json
tt-local runs get <run-id> --config spark-runner.json
tt-local runs events <run-id> --config spark-runner.json
tt-local runs watch <run-id> --config spark-runner.json
tt-local runs report <run-id> --config spark-runner.json
tt-local runs compare <run-id-a> <run-id-b> --config spark-runner.json
tt-local models list --config spark-runner.json
tt-local serve --config spark-runner.json
```

## DGX Spark Notes

On the Spark host, confirm the GPU and uv are available:

```bash
nvidia-smi
uv --version
uv run python --version
```

Then clone, install, build, and link the CLI on the Spark:

```bash
git clone https://github.com/tunedtensor/tuned-tensor-local.git
cd tuned-tensor-local
npm install
npm run build
npm link
tt-local doctor --config examples/local-runner.json
```

Spark-specific details live in [docs/spark.md](docs/spark.md).

## Development

```bash
npm install
npm run typecheck
npm test
npm run build
```

CI runs those checks on pull requests and pushes to `main`. Publishing to npm
runs when a GitHub Release is published and requires an `NPM_TOKEN` repository
secret.

## Notes

- Real training uses `training/sft-local/src/train.py`.
- Real evaluation uses `training/sft-local/src/evaluate.py`.
- The default evaluation path uses Hugging Face/PEFT artifacts directly; it
  does not convert models to GGUF.
- `validate` and `run` warn about unknown hyperparameter keys and ignore them.
- Release notes are in [CHANGELOG.md](CHANGELOG.md).

## License

Apache-2.0
