# TT Local

[![CI](https://github.com/tunedtensor/tuned-tensor-local/actions/workflows/ci.yml/badge.svg)](https://github.com/tunedtensor/tuned-tensor-local/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/@tuned-tensor/local)](https://www.npmjs.com/package/@tuned-tensor/local)

TT Local is local-first fine-tuning with built-in evaluation for small
open-weight models. On a compatible NVIDIA GPU on Linux, it turns a behavior
spec into a fine-tuned model and a paired, inspectable report comparing the run
baseline with the tuned candidate on representative evaluation cases. Specs,
datasets, model artifacts, events, reports, and dashboard state stay on local
disk.

Usage docs:

https://tunedtensor.com/docs/local-training

## Install

```bash
npm install -g @tuned-tensor/local
tt-local info
```

The bundled SFT, DPO, and Transformers evaluator path also needs `uv`:

```bash
uv --version
```

Custom training or evaluation workflows can use command entrypoints instead of
`uv`.

The default uv project is included in the npm package at
`training/local-runner`; using bundled training does not require cloning this
repository.

## First Local Run

On an NVIDIA Spark or another CUDA host, initialize both the behavior spec and
a durable project-local runner config:

```bash
tt-local --version
tt-local init --name "Support Bot" --model Qwen/Qwen3.5-2B --profile spark
```

Edit `tunedtensor.json` to replace the generated system prompt and example.
TT Local refuses to validate or train the unchanged placeholder so an
accidental run cannot spend GPU time on it.

Then preflight the exact Python environments and paths that the run will use:

```bash
tt-local doctor tunedtensor.json --config local-runner.json
tt-local validate tunedtensor.json --config local-runner.json
```

`doctor` checks the bundled uv environment, Torch/Transformers/PEFT imports,
the requested CUDA/MPS device, judge credentials when configured, writable
artifact/cache/store roots, and available disk space. GPU checks are skipped
for dry runs and are conditional for CPU/MPS configurations.

Download the base model explicitly, then start the run:

```bash
tt-local models prefetch tunedtensor.json --config local-runner.json
tt-local models verify-base tunedtensor.json --config local-runner.json
tt-local run tunedtensor.json --config local-runner.json
```

Prefetch progress is visible by default. Pass `--quiet` only when another
program is consuming the final JSON. `verify-base` is local-only: it checks
the cached snapshot structure, indexed weight shards, tokenizer files, and
non-empty weights without downloading a replacement. Cached Hugging Face blobs
whose ETags encode Git or LFS checksums are re-hashed as part of verification.

For a long run, start it in the background and watch it from the current or a
second terminal:

```bash
tt-local run tunedtensor.json --config local-runner.json --detach
tt-local runs watch <run-id> --config local-runner.json
```

The detached command returns the persisted run ID, process ID, log path, and
copy-paste watch/cancel commands before model work begins.

Inspect the evidence after the run:

```bash
tt-local runs list --config local-runner.json
tt-local runs report <run-id> --config local-runner.json
tt-local models verify local-<run-id> --config local-runner.json
```

Treat the report as evidence on this run's evaluation cases, not as a
guarantee of live application improvement.

The model is registered immediately after its artifact passes verification,
so it remains discoverable if candidate evaluation or report generation later
fails.

## Serve a Trained Adapter

`tt-local serve` is the run dashboard. To serve a trained TT Local adapter as
an OpenAI-compatible model endpoint, use the nested model command:

```bash
tt-local models serve local-<run-id> \
  --config local-runner.json \
  --spec tunedtensor.json \
  --host 127.0.0.1 \
  --port 8000
```

The server verifies the stored artifact, loads the recorded base model and
PEFT adapter through the bundled uv environment, and binds to localhost by
default. Test it from another terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

Use `--device cuda|mps|cpu|auto` to override the configured inference device.
Non-loopback binds require both `--allow-remote` and
`--api-key-env <environment-variable>`; requests must then send that value as
a Bearer token. The server accepts multimodal images only as bounded `data:`
URIs and does not support streaming yet.

## Cache and Evaluation Semantics

`paths.modelCache` means Hugging Face `HF_HOME` everywhere. Hub snapshots live
under `<modelCache>/hub`; prefetch, training, evaluation, doctor, and serving
all use that same layout. Set `HF_TOKEN` in the environment or project `.env`
before prefetching a gated model. Do not commit `.env`.

For reproducible remote loading, set
`hyperparameters.base_model_revision` to an immutable Hugging Face commit.
Alternatively, `paths.baseModel` may point to a complete local Hugging Face
snapshot directory. It cannot point to a model archive or standalone weights
file; TT Local requires non-empty model weights, `config.json`, tokenizer
metadata and vocabulary, and every indexed shard. Local snapshot contents are
fingerprinted so a byte change invalidates dependent stages.

Shared baseline-cache entries are used only when the model and every local
input have a stable content identity. An unpinned remote model or remotely
hosted image bypasses the cache rather than risking stale evaluation results.

The safe default scoring mode is deterministic `exact_match`. To use an
OpenRouter judge, configure `evaluation.scoring.mode: "llm_judge"`, an `llm`
block, and its API-key environment variable. Judge mode fails preflight when
those are missing. Exact-match fallback occurs only when explicitly requested
with `evaluation.scoring.fallback: "exact_match"`; fallback-scored results are
marked and never written to the judge baseline cache.

Before the first real run, you can download the configured Hugging Face base
model into `paths.modelCache` explicitly:

```bash
tt-local models prefetch tunedtensor.json --config local-runner.json
tt-local models verify-base tunedtensor.json --config local-runner.json
```

Without this step, the first non-dry `tt-local run` downloads the base model
when baseline evaluation or training first loads it.

Individual `run --stage ...` commands finish in the terminal
`stage_completed` state, so `runs watch` exits instead of waiting for stages
that were intentionally not requested.

Relative paths in a runner config resolve from the config file's directory.
Relative dataset and image paths resolve from the behavior spec or request
file's directory. The published package includes `uv.lock` so bundled Python
dependencies are reproducible.

## DPO

TT Local supports first-class offline DPO for text causal-LM models. Set
`training_method` to `dpo` and provide a prebuilt preference JSONL training
dataset:

```json
{
  "training_method": "dpo",
  "dataset_prebuilt": {
    "training": "file://examples/dpo-preferences.jsonl",
    "format": "preference_jsonl"
  }
}
```

Each preference JSONL row must use explicit `prompt`, `chosen`, and `rejected`
string fields:

```json
{"prompt":"Summarize status: build passed.","chosen":"Build passed.","rejected":"The build failed."}
```

DPO v1 is text-only for the bundled `uv` trainer. Validation and reporting still
use the existing baseline-vs-candidate evaluation loop, so provide
`dataset_prebuilt.test`, `dataset_prebuilt.validation`, or normal
`spec_snapshot.examples` with reference outputs for evaluation.

Command-backed workflows may use external model ids by setting
`spec_snapshot.base_model` to an `external:<id>` or `command:<id>` value, for
example `external:karpathy/nanochat`. The bundled `uv` trainer still requires a
supported Hugging Face base model, but command trainers receive adapter-focused
hyperparameters without injected LoRA/model-loader defaults. Custom
hyperparameter keys are passed through to `TT_HYPERPARAMETERS_PATH`.
Set `hyperparameters.base_model_revision` to an immutable Hugging Face commit
when a run must load and record an exact base-model revision.

## Continuing From a Fine-Tuned Model

Start another loop from an existing TT Local model by passing the stored local
model id:

```bash
tt-local run tunedtensor.json --parent-model local-<previous-run-id>
```

You can also set the parent adapter artifact explicitly:

```bash
tt-local run tunedtensor.json --parent-model-artifact file:///path/to/model.tar.gz
```

For config-only workflows, put the same value in
`hyperparameters.parent_model_artifact`. The parent adapter becomes the
baseline for the new run, and the bundled SFT/DPO trainers load it before
continuing LoRA training.

Stored `--parent-model` references are verified before launch. TT Local pins
the child's base revision from the parent (or requires the same configured
local snapshot content), rejects conflicting base identity, and fingerprints
the parent artifact so changed bytes invalidate reuse. The explicit
`--parent-model-artifact` form is for caller-managed local artifacts and should
be used only when that provenance is understood.

Non-Hugging Face artifacts can describe their layout in `training.artifact`:

```json
{
  "training": {
    "backend": "command",
    "command": ["python", "train_adapter.py"],
    "artifact": {
      "framework": "nanochat",
      "format": "custom-directory",
      "entrypoint": "batch_command",
      "servable": false
    }
  }
}
```

## Local Development

```bash
npm install
npm run typecheck
npm test
npm run build
```

Release notes are in [CHANGELOG.md](CHANGELOG.md).

## License

Apache-2.0
