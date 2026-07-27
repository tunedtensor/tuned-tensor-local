# DGX Spark

> [!IMPORTANT]
> The standalone `tt-local` command is deprecated. Install
> `@tuned-tensor/cli` and use the `tt local` commands shown below. See
> [the deprecation note](../DEPRECATION.md) for details.

DGX Spark is the reference host for the local workflow's first supported path:
`Qwen/Qwen3.5-2B` text SFT with a LoRA adapter.

## Check the host

Run on the Spark:

```bash
nvidia-smi
node --version
uv --version
```

TT Local requires Node 22+, `uv`, working CUDA PyTorch, and enough free space
for the Hugging Face cache plus run artifacts.

## Create a project

```bash
mkdir -p ~/tuned-tensor-runs/support-adapter
cd ~/tuned-tensor-runs/support-adapter
tt local init --name "Support Adapter" --model Qwen/Qwen3.5-2B --profile spark
```

Edit both generated examples in `tunedtensor.json`. For a meaningful run,
replace them with a larger, representative dataset and a separate validation
split.

The generated `local-runner.json` uses CUDA and project-local artifacts. A
durable Spark configuration can set:

```json
{
  "artifactRoot": "/home/eve/tuned-tensor-runs/artifacts",
  "storeRoot": "/home/eve/tuned-tensor-runs/store",
  "paths": {
    "modelCache": "/home/eve/.cache/huggingface"
  },
  "evaluation": {
    "inference": {
      "device": "cuda"
    },
    "scoring": {
      "mode": "exact_match"
    },
    "timeoutMs": 1800000
  }
}
```

Every Python stage uses the locked runtime included in the npm package; a
source checkout and a custom runner path are neither required nor supported.

## Preflight and run

```bash
tt local doctor tunedtensor.json
tt local validate tunedtensor.json
tt local models prefetch tunedtensor.json
tt local models verify-base tunedtensor.json
tt local run tunedtensor.json
```

`doctor` resolves the same bundled project and paths the run will use, imports
Torch/Transformers/PEFT, requires visible CUDA, checks writable storage, and
rejects unchanged placeholders. `validate` reads and normalizes the actual
dataset before any run state or artifact directory is claimed.

The runner provides these paths to Python:

- `SM_CHANNEL_TRAINING`: prepared chat JSONL directory;
- `TT_HYPERPARAMETERS_PATH`: generated SFT/LoRA parameters;
- `SM_OUTPUT_DIR`: logs and metrics;
- `SM_MODEL_DIR`: adapter output;
- `SM_CHANNEL_BASE_MODEL`: optional verified local model snapshot;
- `HF_HOME`: configured persistent model cache.

## Verify and serve

```bash
tt local runs report <run-id>
tt local models verify local-<run-id>
tt local serve local-<run-id> --spec tunedtensor.json --port 8000
```

In another shell:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Classify: I loved it."}]}'
```

If a run fails, start with `tt local runs events <run-id>` and
`tt local runs get <run-id>`. The run record reports its `artifact_dir`; the
main subprocess logs there are `training/training.log`,
`baseline-eval.json.inference.log`, and `candidate-eval.json.inference.log`.
The adapter is registered as soon as its manifest verifies, even if candidate
evaluation fails afterward.
