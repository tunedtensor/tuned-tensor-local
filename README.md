# Tuned Tensor Local

[![CI](https://github.com/tunedtensor/tuned-tensor-local/actions/workflows/ci.yml/badge.svg)](https://github.com/tunedtensor/tuned-tensor-local/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/@tuned-tensor/local)](https://www.npmjs.com/package/@tuned-tensor/local)

Tuned Tensor Local is a standalone CLI for running Tuned Tensor-style fine-tuning
jobs on a machine you control. It stores specs, datasets, model artifacts,
progress, reports, and dashboard state on local disk.

It does not require the hosted Tuned Tensor CLI, cloud orchestration, Docker, or
a managed database.

## What It Does

`tt-local` runs this workflow locally:

1. Read a behavior spec from `tunedtensor.json`.
2. Compile examples into a training dataset.
3. Evaluate the original Hugging Face base model.
4. Fine-tune locally with uv, Transformers, and PEFT.
5. Evaluate the fine-tuned Hugging Face/PEFT artifact.
6. Compare baseline vs tuned outputs and write a local report.

OpenRouter can be used as an LLM judge for scoring generated outputs. It does
not generate the baseline or tuned responses.

## Install Locally

Recommended for development and Spark:

```bash
git clone https://github.com/tunedtensor/tuned-tensor-local.git
cd tuned-tensor-local
npm install
npm run build
npm link
```

You can install the published [@tuned-tensor/local npm package](https://www.npmjs.com/package/@tuned-tensor/local)
globally:

```bash
npm install -g @tuned-tensor/local
```

Check the CLI:

```bash
tt-local info
```

## Set Up DGX Spark

On the Spark host, confirm the GPU and uv are available:

```bash
nvidia-smi
uv --version
uv run python --version
```

If uv is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new shell, then rerun:

```bash
uv --version
```

Clone this repo on the Spark, install the Node dependencies, and build:

```bash
git clone https://github.com/tunedtensor/tuned-tensor-local.git
cd tuned-tensor-local
npm install
npm run build
npm link
```

Run the doctor:

```bash
tt-local doctor --config examples/local-runner.json
```

## Run Your First Job

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

Run the example dry workflow:

```bash
tt-local run --config examples/local-runner.json
```

The example config has `dryRun: true`, so it checks orchestration and artifact
writing without loading models or starting GPU training.

Inspect the run:

```bash
tt-local runs list --config examples/local-runner.json
tt-local runs report <run-id> --config examples/local-runner.json
```

Start the local dashboard:

```bash
tt-local serve --config examples/local-runner.json
```

Then open the printed local URL in your browser.

## Run Real Training On Spark

Create a Spark config, for example `spark-runner.json`:

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

For structured JSON tasks, use field-level scoring instead of whole-output
exact matching:

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

The evaluation report still includes `exact_match_rate`, but `avg_score`,
`pass_rate`, and `json_field_metrics` are based on the selected JSON fields.
If `fields` is omitted, the evaluator scores every key in the expected JSON
object for each example. A configured field that is missing from the expected
JSON is always scored as incorrect (with a reasoning note), so misconfigured
field lists cannot inflate scores.

The judge receives the spec's compiled system message (system prompt,
guidelines, and constraints) as task instructions and scores how well each
output fulfills the task; the expected output is treated as a reference
answer, not a fact checklist. Every eval report also includes `avg_token_f1`
(and the comparison includes `token_f1_delta`), a deterministic token-overlap
similarity against the reference outputs that is useful for free-text tasks
such as summarization, where exact match is always 0 and judge scores can be
noisy.

With `"mode": "llm_judge"`, `scoring.fallback` controls what happens when the
judge is unavailable, a judge request fails, or the judge returns malformed
JSON. With `"fallback": "exact_match"` (the default), the affected example is
scored by normalized exact match and its reasoning records the judge failure;
the run does not fail. With `"fallback": "fail"`, judge errors fail the run.

`evaluation.maxExamples` caps how many evaluation examples are scored. If it
is unset, the request hyperparameter `max_eval_examples` is used instead;
an explicit `maxExamples` in the config always wins. When the cap truncates
the eval set, the runner draws a deterministic seeded random sample (not a
prefix), so sorted or grouped eval files do not bias the subset. The seed
defaults to a hash of the run id, can be pinned with `evaluation.sampleSeed`,
and is recorded in each eval report as `eval_sample_seed`. Baseline and
candidate evaluations always score the same sampled examples in the same
order.

If you want OpenRouter judging, set your key:

```bash
export OPENROUTER_API_KEY="your_openrouter_key"
```

`tt-local` also loads a `.env` file from the working directory (existing
environment variables win), so the key can live in the project's `.env`
instead.

`validate` and `run` warn about hyperparameter keys the schema does not
recognize (for example `per_device_train_batch_size` instead of
`batch_size`); unknown keys are ignored rather than passed to training.

Run the job:

```bash
tt-local run --config spark-runner.json
```

`tt-local run` prints live stage updates to stderr and writes the final JSON
summary to stdout. Use `--verbose` to stream Python subprocess output, or
`--quiet` for script-only JSON output.

Training progress from Python metric lines and tqdm progress bars is also
recorded as run events, so `tt-local runs watch` and per-run `progress.jsonl`
show epoch, loss, step, percentage, and ETA updates when the trainer emits
them.

For spec-based runs (no `dataset_prebuilt`), the runner automatically holds
out about 20% of the spec examples (at least 1 holdout and at least 1 training
example) for evaluation, so baseline and candidate scores are not measured on
the training data. The split is deterministic: it uses the per-run sample seed
(a hash of the run id, or `evaluation.sampleSeed` when set), so re-running the
same run id reproduces the same split and baseline/candidate always score the
identical holdout. The training JSONL artifact contains only the training
split, `run_metadata.training_example_count` records the training split size,
and `eval_examples_total` records the holdout size. These runs report
`eval_split: "spec_holdout"`. With only 1 spec example there is nothing to
hold out, so evaluation runs on the training set and is reported as
`eval_split: "spec_examples"` (training-set evaluation due to insufficient
examples).

For prebuilt datasets, `dataset_prebuilt.training` is always copied into the
run artifact as the training set. Evaluation uses `dataset_prebuilt.test` when
present, then `dataset_prebuilt.validation`. If neither is provided, a real
(non-dry) run fails with an error, because evaluating on the training split
overstates improvement; set `evaluation.allowPrebuiltTrainingEval: true` to
evaluate on the training file anyway. Dry runs are allowed to fall back to the
training file without the override. Each eval report and the run's
`run_metadata` record the split that was evaluated in `eval_split`
(`spec_holdout`, `spec_examples`, `prebuilt_test`, `prebuilt_validation`, or
`prebuilt_training`), so training-set evaluation is always visible.

Multimodal runs use the same chat JSONL shape with structured user content.
For `image_text_to_text` models such as `Qwen/Qwen3-VL-2B-Instruct`, image
parts are loaded by the local SFT trainer and evaluator:

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
`data_uri`; use absolute paths, `file://` URIs, URLs, or `data:` URIs there so
the copied run artifact can still resolve the image.

Watch progress:

```bash
tt-local runs watch <run-id> --config spark-runner.json
```

View the final report:

```bash
tt-local runs report <run-id> --config spark-runner.json
```

## Where Files Go

The runner writes two local roots:

- `artifactRoot`: datasets, logs, model outputs, eval JSON, and `run-report.json`.
- `storeRoot`: run/spec/model metadata used by CLI listing and the dashboard.

If `storeRoot` is omitted, `tt-local` uses `TT_LOCAL_HOME` or
`~/.tuned-tensor-local`.

Per-run artifacts include:

- `progress.jsonl`: stage changes and parsed training progress for the run.
- `baseline-eval.json.inference.log` and `candidate-eval.json.inference.log`:
  local inference subprocess output.
- `training/training.log`: uv fine-tuning output.
- `run-report.json`: baseline, candidate, comparison, and artifact references.

## Useful Commands

```bash
tt-local init --name "Support Bot" --model Qwen/Qwen3.5-2B
tt-local validate --config spark-runner.json
tt-local run --config spark-runner.json
tt-local run --config spark-runner.json --verbose
tt-local runs list --config spark-runner.json
tt-local runs get <run-id> --config spark-runner.json
tt-local runs events <run-id> --config spark-runner.json
tt-local runs report <run-id> --config spark-runner.json
tt-local models list --config spark-runner.json
tt-local serve --config spark-runner.json
```

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

- Real training uses the included SFT script in `training/sft-local/src/train.py`.
- Real evaluation uses `training/sft-local/src/evaluate.py`.
- The default evaluation path uses Hugging Face/PEFT artifacts directly; it does
  not convert models to GGUF.
- Release notes are in [CHANGELOG.md](CHANGELOG.md).
- Spark-specific details live in [docs/spark.md](docs/spark.md).

## License

Apache-2.0
