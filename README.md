# Tuned Tensor Local

[![CI](https://github.com/tunedtensor/tuned-tensor-local/actions/workflows/ci.yml/badge.svg)](https://github.com/tunedtensor/tuned-tensor-local/actions/workflows/ci.yml)

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

After the package is published, you can install the CLI globally:

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

If you want OpenRouter judging, set your key:

```bash
export OPENROUTER_API_KEY="your_openrouter_key"
```

Run the job:

```bash
tt-local run --config spark-runner.json
```

`tt-local run` prints live stage updates to stderr and writes the final JSON
summary to stdout. Use `--verbose` to stream Python subprocess output, or
`--quiet` for script-only JSON output.

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

- `progress.jsonl`: stage changes for the run.
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
