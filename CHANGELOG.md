# Changelog

All notable changes to Tuned Tensor Local will be documented in this file.

## 0.1.5 - 2026-07-02

### Added

- Added `avg_token_f1` to eval reports and `token_f1_delta` to the comparison report: a cheap deterministic token-overlap reference-similarity metric for free-text tasks where exact match is always 0 and an LLM judge can be noisy.
- The OpenRouter judge now receives the spec's compiled system message (system prompt, guidelines, constraints) as `task_instructions` and scores conformance to the task instead of treating the expected output as a fact checklist. Previously a model fine-tuned toward a concise style was penalized for omitting reference details even when the spec asked for brevity.
- `tt-local` now loads `.env` from the working directory (without overriding already-set variables), so `OPENROUTER_API_KEY` works without manual sourcing.
- `validate` and `run` now warn about unknown hyperparameter keys (for example `per_device_train_batch_size` instead of `batch_size`), which were previously stripped silently and trained with defaults.

### Fixed

- Judge scoring with `scoring.fallback: "exact_match"` no longer fails the whole run when a judge request fails transiently; the affected example falls back to normalized exact match and records the judge failure in its reasoning.
- Fixed a crash-level race in atomic store writes: concurrent run-state updates in the same millisecond (typical when tqdm flushes several progress lines at training completion) collided on the same temp filename and killed the run with `ENOENT: rename` after training had succeeded. Temp filenames now include a UUID.
- The local evaluator no longer passes `temperature`/`top_p` to `model.generate` when decoding greedily, silencing the spurious transformers "generation flags are not valid" warning at `temperature: 0`.
- `--verbose` no longer prints consecutive duplicate subprocess log lines (tqdm redraws the same progress line several times per step).

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
