# TT Local

[![CI](https://github.com/tunedtensor/tuned-tensor-local/actions/workflows/ci.yml/badge.svg)](https://github.com/tunedtensor/tuned-tensor-local/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/@tuned-tensor/local)](https://www.npmjs.com/package/@tuned-tensor/local)

TT Local runs Tuned Tensor-style fine-tuning jobs on your own machine. It keeps
specs, datasets, model artifacts, events, reports, and dashboard state on local
disk.

Usage docs:

https://tunedtensor.com/docs/local-training

## Install

```bash
npm install -g @tuned-tensor/local
tt-local info
```

The bundled SFT and Transformers evaluator path also needs `uv`:

```bash
uv --version
```

Custom training or evaluation workflows can use command entrypoints instead of
`uv`.

Command-backed workflows may use external model ids by setting
`spec_snapshot.base_model` to an `external:<id>` or `command:<id>` value, for
example `external:karpathy/nanochat`. The bundled `uv` trainer still requires a
supported Hugging Face base model, but command trainers receive adapter-focused
hyperparameters without injected LoRA/model-loader defaults. Custom
hyperparameter keys are passed through to `TT_HYPERPARAMETERS_PATH`.

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
