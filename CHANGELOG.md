# Changelog

All notable changes to TT Local will be documented in this file.

## 0.2.9 - 2026-07-14

### Changed

- Reframed TT Local as local-first fine-tuning with paired baseline-vs-tuned
  evaluation for small open-weight models, and aligned the README, package
  metadata, and CLI help with that positioning.

## 0.2.8 - 2026-07-14

### Added

- Added strict side-effect-free CLI help, `--version`, exact-runtime `doctor`
  checks, and a Spark initialization profile.
- Added atomic artifact manifests, `tt-local models verify`, and local
  OpenAI-compatible serving of verified PEFT adapters.
- Added local-only `tt-local models verify-base` snapshot validation and made
  deliberately staged runs terminate as `stage_completed`.
- Added process-group cancellation and immediate post-training model
  registration so a valid model survives later evaluation/report failures.
- Included the bundled Python `uv.lock` in the npm package.
- Added durable run/artifact ownership claims, workflow locks, detached runs,
  terminal stage/cancellation states, and `runs watch` lifecycle handling.

### Changed

- Standardized `paths.modelCache` as Hugging Face `HF_HOME` across prefetch,
  training, evaluation, doctor, and serving; prefetch now reports the resolved
  snapshot revision, file count, and byte size.
- Made `exact_match` the runnable scoring default. Explicit LLM-judge mode now
  requires valid configuration/credentials unless exact-match fallback was
  deliberately selected, and fallback results are never cached as judge data.
- Resume invalidation now fingerprints the complete request and effective
  runner configuration, local data/assets/base snapshot, immutable model
  revision, parent artifact, and bundled runtime rather than reusing outputs
  trained with stale inputs.
- Relative config paths resolve from the config file and relative spec data
  paths resolve from the spec/request file.
- Stored parent models now verify their artifact and immutable base identity;
  continuation runs inherit the recorded base revision and reject conflicts.
- Baseline caching now requires stable content identity and bypasses mutable
  remote models/assets. Optional progress reporters can no longer crash or
  orphan child processes.

### Fixed

- Prevented help, unknown flags, missing values, and option typos from starting
  work accidentally.
- Rejected empty or corrupt model outputs before registration and cleared stale
  terminal errors after successful retries.
- Rejected incomplete PEFT/full-model contracts, optimizer-only payloads,
  artifact-prefix collisions, and symlink escapes before publication.
- Disabled optional PyTorch native JIT kernels by default for bundled ML
  processes so packaged CUDA inference does not require system Python headers;
  explicit runner environments can opt back in.
- Made interactive model-server shutdown close cleanly without a Python
  traceback.
- Stopped reporting model checkpoint shard loading as optimizer progress.

## 0.2.7 - 2026-07-05

### Added

- Added `tt-local models prefetch [tunedtensor.json|request.json]` to download
  the configured Hugging Face base model into `paths.modelCache` before the
  first real run, making first-time setup explicit instead of hiding the
  download inside baseline evaluation or training.

## 0.2.6 - 2026-07-05

### Fixed

- Bundled uv training and Transformers evaluation now resolve
  `training/local-runner` relative to the installed npm package, so users can
  run real SFT/DPO workflows from `@tuned-tensor/local` without cloning the
  source repository.
- Tightened the published package contents to include the bundled runner source
  and `pyproject.toml` without Python bytecode cache files.

## 0.2.5 - 2026-07-05

### Added

- Added continued local fine-tuning from a previous TT Local PEFT adapter via
  `hyperparameters.parent_model_artifact`, `tt-local run --parent-model`, and
  `tt-local run --parent-model-artifact`.
- Added parent-model lineage to training reports, run metadata, and `tt-local
  run` JSON output.

### Changed

- Continued runs now evaluate the parent adapter as the baseline and then train
  the child adapter from that parent, so each follow-up loop compares against
  the model it actually continues from.
- `tt-local run --parent-model` now rejects stored parent models whose recorded
  base model differs from the new run's base model before launching training.

## 0.2.4 - 2026-07-05

### Added

- Added first-class offline DPO support for text causal-LM local runs via
  `training_method: "dpo"` and prebuilt `preference_jsonl` datasets with
  explicit `prompt`, `chosen`, and `rejected` fields.
- Added a bundled TRL `DPOTrainer` path alongside the existing SFT trainer,
  including DPO hyperparameters such as `dpo_beta`, `dpo_loss_type`,
  `dpo_label_smoothing`, `dpo_reference_free`, `max_prompt_length`, and
  `max_completion_length`.
- Added DPO examples and documentation for method-selected bundled training
  scripts.

### Changed

- Renamed the bundled uv training project from `training/sft-local` to
  `training/local-runner` now that it hosts both SFT and DPO workflows.
- Prebuilt local run preparation now fingerprints dataset file contents so
  stale staged artifacts are refreshed when prebuilt training, validation, or
  test files change under the same run id.

## 0.2.3 - 2026-07-05

### Added

- Added resumable `tt-local run --stage prepare|baseline|train|candidate|score|report|all`
  execution so users can rerun a single part of the local workflow without
  repeating training or inference. Stages reuse existing artifacts by default
  and accept `--force` to recompute.
- Added persisted stage artifacts, including `stage-metadata.json` and
  `training-report.json`, plus `--run-id` resume support and
  `--model-artifact` for candidate evaluation against an external adapter.
- Added scoring-only evaluation reuse for `--stage score`, allowing existing
  generated baseline/candidate outputs to be rescored with the current scorer,
  including OpenRouter LLM judge, without regenerating outputs.

### Fixed

- Prepared run artifacts now carry a source fingerprint and invalidate
  downstream baseline, training, candidate, and report artifacts when the input
  spec, dataset, model, eval seed, or eval limit changes under the same run id.

## 0.2.2 - 2026-07-04

### Added

- Added external command workflow support with `external:<id>` and
  `command:<id>` base model identifiers for custom trainers such as nanoGPT
  and nanochat adapters. The bundled uv trainer still requires a supported
  Hugging Face base model.
- Added command-workflow artifact metadata via `training.artifact`, surfaced in
  run reports as `training.artifact_metadata`, so non-Hugging Face layouts can
  describe their framework, format, entrypoint, and servability.

### Changed

- Command-backed training now receives adapter-focused hyperparameters without
  injecting bundled LoRA/model-loader defaults, while preserving custom
  hyperparameter keys in `TT_HYPERPARAMETERS_PATH`.
- Unknown hyperparameter warnings now clarify that keys are passed through but
  may be ignored by the default trainer.

## 0.2.1 - 2026-07-04

### Added

- Added command-based local training with `training.backend: "command"` and
  `training.command`, while preserving the existing uv script/module path.
- Added `evaluation.inference.provider: "batch_command"` with
  `evaluation.inference.command` for custom batch evaluators that consume the
  same `--input`/`--output` JSON files as the bundled Transformers evaluator.

## 0.2.0 - 2026-07-04

### Added

- Added a hybrid local state store: `metadata.sqlite` now indexes specs, runs,
  run events, and models for CLI/dashboard queries while preserving the
  transparent JSON/JSONL artifact layout. `tt-local store rebuild-index`
  rebuilds SQLite metadata from canonical per-object files. The SQLite index
  uses `better-sqlite3` instead of Node's experimental built-in SQLite API,
  avoiding runtime experimental warnings on Node 22.

## 0.1.9 - 2026-07-03

### Added

- Added a `tt-local label` command, porting the tuned-tensor-runs teacher
  labeling job to the local runner. It reads unlabeled JSONL
  (`{"input": "..."}` rows, optionally pre-labeled with `"output"`) or CSV
  (`--input-column`) sources, labels pending rows with an OpenRouter teacher
  model under the spec's system message, and writes a labeled
  `{"input", "output"}` JSONL, a per-row review file, and a job report under
  `<artifactRoot>/labeling/<job-id>/`. Supports `--dry-run`, `--output`,
  `--model`, `--spec`, and `--system-prompt`.
- Rows are sanitized before leaving the machine: secret-like content (API
  keys, private keys, connection strings, passwords) blocks the row; PII
  (emails, phones, SSNs, card numbers) is redacted.
- Added a `labeling` runner-config block (`model`, `maxTokens`, `temperature`,
  `concurrency`, `minIntervalMs`, `maxAttempts`, `maxRows`, `timeoutMs`); the
  teacher model falls back to `llm.model`.

### Changed

- `openRouterChat` now supports free-text responses (`responseFormat:
  "text"`), `temperature`/`maxTokens` overrides, and returns prompt/completion
  token usage; it throws a typed `OpenRouterHttpError` so callers can retry
  retryable statuses. Judge behavior (JSON object output, temperature 0) is
  unchanged by default.

## 0.1.8 - 2026-07-02

### Changed

- Renamed the user-facing project language to TT Local with the tagline
  "Tuning tensors locally", while keeping the `tt-local` CLI, npm package, repo,
  and filesystem identifiers unchanged.
- Reworked the README around a first-time user flow: install, safe dry run,
  real training, config basics, evaluation behavior, artifacts, Spark notes,
  and common commands.
- Simplified evaluation configuration by removing `evaluation.mode`. Inference
  is now selected with `evaluation.inference.provider`, and scoring is selected
  with `evaluation.scoring.mode`.
- Shared the uv/subprocess execution helper between training and evaluation so
  logging, timeout, and stderr handling are consistent.

### Added

- Added direct support for `evaluation.inference.provider: "command"` with
  `baselineCommand` and `candidateCommand` for custom inference adapters.

## 0.1.7 - 2026-07-02

### Added

- Added chat-template kwargs plumbing for thinking-mode models: `evaluation.inference.chatTemplateKwargs` (runner config) and the `chat_template_kwargs` hyperparameter (spec) are forwarded to `apply_chat_template` in the local evaluator and SFT trainer. Setting `{"enable_thinking": false}` stops models like `Qwen/Qwen3.5-4B` from opening a `<think>` block that consumes the whole `maxNewTokens` budget and truncates every output.

### Fixed

- Fixed a run-fatal report validation error when a comparison had regressions in only some taxonomy categories: zod v4 enum-keyed records are exhaustive, so the partial `regression_taxonomy` failed `runReportSchema` at the end of an otherwise-successful run. The taxonomy now always contains all categories with zero defaults.

## 0.1.6 - 2026-07-02

### Added

- Added `tt-local runs compare <run-id-a> <run-id-b>`: aligns two runs on their shared eval prompts and reports per-subset baseline/candidate averages, shared-subset deltas, new-example effects, and judge score noise measured on identical baseline outputs. Headline `avg_score_delta` values are not comparable across different eval subsets; the shared subset is.
- Added a baseline evaluation cache (default on, `evaluation.baselineCache: false` to disable): re-running a spec with an unchanged base model, eval examples, generation settings, and scoring config reuses the previous baseline report from `<storeRoot>/cache/baseline-evals/` instead of re-running inference and judge calls. Reports served from cache carry `cached: true` and a `cache_key`. Reports containing fallback-scored examples are never cached.
- Eval reports now record how each example was scored (`results[].scored_by`) and aggregate it as `judge_scored_count`, `fallback_scored_count`, and `judge_only_avg_score`, so transient judge failures that fall back to exact match can no longer silently skew `avg_score` without a trace. The comparison report includes `judge_only_avg_score_delta`.
- Comparison reports now classify each regressed example's judge reasoning into a coarse category (`factual`, `omission`, `style`, `fallback`, `other`) and aggregate counts as `regression_taxonomy`, answering "what kind of worse?" at a glance.

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
