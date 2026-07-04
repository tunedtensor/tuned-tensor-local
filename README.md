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
