# TT Local

[![CI](https://github.com/tunedtensor/tuned-tensor-local/actions/workflows/ci.yml/badge.svg)](https://github.com/tunedtensor/tuned-tensor-local/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/@tuned-tensor/local)](https://www.npmjs.com/package/@tuned-tensor/local)

> [!IMPORTANT]
> The standalone `tt-local` CLI is deprecated. Install
> [`@tuned-tensor/cli`](https://www.npmjs.com/package/@tuned-tensor/cli) and use
> `tt local ...` instead. The unified `tt` CLI exposes this complete workflow
> alongside the hosted workflow and is where new user-experience work will
> happen. This repository remains available in maintenance mode as the local
> runtime used by `tt`. See [DEPRECATION.md](DEPRECATION.md) for scope and
> migration details.

TT Local fine-tunes an open-weight language model on your own NVIDIA GPU. It
turns a small behavior spec or chat JSONL dataset into a verified LoRA adapter,
compares the base and tuned model on held-out examples, and can serve the
adapter through an OpenAI-compatible endpoint.

The current release deliberately has one supported path:

- text supervised fine-tuning (SFT);
- `Qwen/Qwen3.5-2B`;
- LoRA/PEFT on CUDA;
- deterministic exact-match or JSON-field evaluation;
- local files for datasets, state, reports, and model artifacts.

New model families and training methods should be added only after they have a
repeatable end-to-end GPU acceptance test.

## Requirements

- Node.js 22 or later
- [`uv`](https://docs.astral.sh/uv/)
- an NVIDIA CUDA host with enough free disk space for the model cache and run
  artifacts

The npm package includes the locked Python training environment.

```bash
npm install -g @tuned-tensor/cli
tt local info
```

## Fine-tune a model

Create a project on the GPU host:

```bash
mkdir support-adapter && cd support-adapter
tt local init --name "Support Adapter" --model Qwen/Qwen3.5-2B --profile spark
```

This creates `tunedtensor.json` and `local-runner.json`. Edit the generated
system prompt and replace both placeholder examples with representative,
different examples. Two examples are the minimum for a real run because
training and evaluation must not use the same row.

Preflight the machine, input, and cached base model:

```bash
tt local doctor tunedtensor.json
tt local validate tunedtensor.json
tt local models prefetch tunedtensor.json
tt local models verify-base tunedtensor.json
```

Commands discover `local-runner.json` beside the spec. Use
`--config /path/to/local-runner.json` to select another config explicitly.

Run the complete base-evaluate, train, tuned-evaluate, and report workflow:

```bash
tt local run tunedtensor.json
```

Inspect the evidence and verify the adapter:

```bash
tt local runs report <run-id>
tt local models verify local-<run-id>
```

To protect broad capability, configure a separate chat JSONL suite in
`local-runner.json`:

```json
{
  "evaluation": {
    "generalRegression": {
      "dataset": "evals/general.jsonl",
      "maxScoreDrop": 0.03,
      "maxPassRateDrop": 0.05
    }
  }
}
```

TT Local evaluates the protected base and candidate adapter on this suite,
records both reports, and marks whether the configured score and pass-rate
budgets were respected. The base result uses the existing immutable baseline
cache.

After a passing run, activation is a lightweight pointer to the existing model
record:

```bash
tt local models activate local-<run-id>
tt local models active
tt local serve active --config local-runner.json
tt local models rollback
tt local serve base --config local-runner.json
```

Activation re-verifies the model artifact and requires a completed run with a
passing general-regression result. Rollback restores the previous adapter or
the protected base; it does not copy or delete model files.

The report is evidence on the selected evaluation rows, not a guarantee of
general model improvement.

## Use a chat JSONL dataset

For more than a few examples, set `dataset_prebuilt` in `tunedtensor.json`:

```json
{
  "dataset_prebuilt": {
    "training": "data/train.jsonl",
    "validation": "data/validation.jsonl",
    "format": "chat_jsonl"
  }
}
```

Each line is a JSON object with a `messages` array. The final message must be
the assistant answer; preceding messages form the prompt:

```json
{"messages":[{"role":"system","content":"Return one sentiment label."},{"role":"user","content":"I loved it."},{"role":"assistant","content":"positive"}]}
```

TT Local masks every prompt token from the loss and preserves the complete
assistant answer when truncating a long example. Use a distinct validation or
test file for trustworthy evaluation.

## Serve the adapter

```bash
tt local serve local-<run-id> \
  --spec tunedtensor.json \
  --host 127.0.0.1 \
  --port 8000
```

`tt local models serve` is an equivalent nested command. Before launch, TT
Local re-hashes the stored artifact manifest and then loads the recorded base
model plus PEFT adapter with the bundled Python environment.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Classify: I loved it."}]}'
```

The default bind is localhost. A non-loopback bind requires `--allow-remote`
and `--api-key-env <environment-variable>`.

## Reproducibility and local data

Set `paths.modelCache` in `local-runner.json` to a durable Hugging Face cache.
Prefetch records the resolved model commit; set
`hyperparameters.base_model_revision` to a 40-character Hugging Face commit
SHA when an immutable revision is required before download.

Run state is intentionally plain and recoverable:

```text
.tt-local/
  artifacts/users/<user-id>/specs/<spec-id>/runs/<run-id>/...
  store/specs/<spec-id>/spec.json
  store/runs/<run-id>/{request.json,state.json,progress.jsonl,run-report.json}
  store/models/<model-id>/model.json
```

Real training writes an `artifact-manifest.json` containing file sizes and
SHA-256 hashes. Dry runs exercise orchestration but do not register a model.

See [docs/spark.md](docs/spark.md) for the DGX Spark workflow and
[docs/architecture.md](docs/architecture.md) for the design boundary.
