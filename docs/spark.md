# Running On DGX Spark

TT Local expects the Spark to behave like a single Linux GPU host:
uv must be installed, Python dependencies must resolve, and CUDA/PyTorch must be
able to see the GPU.

## Host Checks

Run these on the Spark host:

```bash
nvidia-smi
uv --version
uv run python --version
```

If uv is missing, install it with the official standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new shell or source the installer-updated profile before rerunning
`uv --version`.

Then run the project doctor command:

```bash
npm run build
tt-local doctor --config examples/local-runner.json
```

## Dry Run

The example config uses `dryRun: true`, so it validates orchestration, dataset
compilation, artifact writing, and report generation without launching training:

```bash
tt-local run examples/smoke-run-request.json --config examples/local-runner.json
```

## Real Training

For real training, set `dryRun: false` and configure the uv training entrypoint:

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

The runner sets:

- `SM_CHANNEL_TRAINING` to the local chat JSONL training directory;
- `TT_HYPERPARAMETERS_PATH` to the generated hyperparameter JSON file;
- `SM_OUTPUT_DIR` to the local training output directory;
- `SM_MODEL_DIR` to the local model output directory;
- `SM_CHANNEL_BASE_MODEL` when a local base-model artifact path is configured;
- `HF_HOME` when `paths.modelCache` is configured.

Real runs also use the Transformers/PEFT evaluator by default. Baseline
evaluation loads the original Hugging Face model, and candidate evaluation loads
that same base model plus the fine-tuned artifact from the run directory. Set
`paths.modelCache` to a persistent Spark-local cache so training and evaluation
reuse downloads.

The included first-pass SFT script can be run by the local runner through uv:

```bash
tt-local run examples/smoke-run-request.json --config examples/local-runner.json
```

The SFT uv project currently points Linux installs at the PyTorch CUDA 13.0
wheel index. If PyTorch does not publish a compatible wheel for the Spark's
architecture, use `training.env`, `training.project`, or the SFT
`pyproject.toml` to point uv at the NVIDIA/PyTorch package source that matches
the host.
