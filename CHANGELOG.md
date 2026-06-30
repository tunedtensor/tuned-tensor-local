# Changelog

All notable changes to Tuned Tensor Local will be documented in this file.

## 0.1.1 - 2026-06-30

### Fixed

- Fixed the published/global CLI binary entrypoint when invoked through npm symlinks.
- Added live `tt-local run` stage logging, with `--verbose` subprocess streaming and `--quiet` JSON-only mode.
- Added baseline/candidate inference log files and surfaced key report, model, training log, and comparison paths in run output.
- Updated GitHub Actions pins and disabled uv action caching to avoid Node runtime and cache warnings.

## 0.1.0 - 2026-06-30

### Added

- Initial npm package under `@tuned-tensor/local`.
- Native Transformers/PEFT evaluation for local runs.
- Baseline vs fine-tuned comparison using the original Hugging Face base model and the trained HF/PEFT artifact.
- OpenRouter LLM-judge scoring for locally generated outputs.
- Step-by-step README guide for install, DGX Spark setup, first dry run, real training, and run inspection.

### Changed

- Default OpenRouter judge model is now `openai/gpt-5.5`.
- npm package contents now include `docs`, `examples`, and `training`.

### Notes

- The default evaluation path does not convert models to GGUF.
- Dry runs still skip model loading and GPU training.
