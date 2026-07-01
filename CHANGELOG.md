# Changelog

All notable changes to Tuned Tensor Local will be documented in this file.

## 0.1.4 - 2026-07-01

### Added

- Added first-class `image_text_to_text` training and evaluation support for multimodal local runs, including Qwen3-VL style image+text SFT with TRL, PEFT LoRA, and processor-based generation.
- Added multimodal chat JSONL asset handling for structured image parts, top-level `images`, relative prebuilt dataset paths, `file://` URIs, HTTP(S) URLs, and `data:` URIs.

### Fixed

- Corrected Qwen3.5 text model registry entries to use the CausalLM loader now that loader dispatch is active.

## 0.1.3 - 2026-07-01

### Added

- Added `evaluation.scoring.mode: "json_fields"` for structured JSON output scoring, including per-field accuracy, schema match rate, valid JSON rate, and all-fields match rate.
- Added parsed training progress events from trainer metric lines and tqdm progress bars so `tt-local runs watch` and run `progress.jsonl` show loss, epoch, step, percentage, and ETA updates during long jobs.

## 0.1.2 - 2026-07-01

### Fixed

- Evaluates prebuilt local runs against `dataset_prebuilt.test` when present, then `validation`, before falling back to `training`.
- Keeps the copied run training artifact tied to `dataset_prebuilt.training` so held-out eval files are not mixed into training artifacts.

## 0.1.1 - 2026-06-30

### Added

- Added live `tt-local run` stage logging to stderr while preserving final JSON output on stdout.
- Added `--verbose` subprocess streaming and `--quiet` JSON-only mode for local runs.
- Added baseline and candidate inference log files for Transformers evaluation subprocesses.
- Surfaced report, model, training log, evaluation artifact, and comparison paths in `tt-local run` output.

### Fixed

- Fixed the published/global CLI binary entrypoint when invoked through npm symlinks.
- Sanitized obvious token and API key patterns from streamed subprocess logs.
- Ignored generated `.tt-local/` state in git.
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
