# TT Local end-to-end workflow and UX review

Date: 2026-07-13
Reviewed release: `@tuned-tensor/local@0.2.7`
Repository commit: `3b18464` (`main`)
Scope: install, init, preflight, validation, base-model download, real fine-tuning,
evaluation, reports, persisted state, recovery, dashboard, and the handoff to model
inference.

## Executive summary

The core training and persistence path works on the DGX Spark once two configuration
workarounds are applied. A real SFT run completed, its PEFT adapter was loaded for
candidate evaluation on CUDA, the model archive was valid, the local store was
internally consistent, and the index could be rebuilt.

The current first-run UX is not ready to be presented as a smooth default workflow.
The most important blockers are:

1. Explicit prefetch and runtime model loading use different Hugging Face cache
   layouts, causing a duplicate multi-gigabyte download.
2. The public no-config quickstart and default evaluator do not select the bundled
   uv project, so `doctor` can pass immediately before `run` fails on `import torch`.
3. Resuming a run can silently reuse a model trained with old hyperparameters or
   old runner/evaluation configuration.
4. An empty model directory can be accepted, reported, and indexed as a completed
   model; there is no artifact manifest or verification command.
5. Subcommand help and unknown-option handling are unsafe: `tt-local run --help`
   starts a run.
6. There is no tested, documented inference-serving handoff for a TT Local adapter.

These should be addressed before optimizing lower-level polish.

## What was tested

### Automated and packaged checks

All existing checks passed:

- `npm run typecheck`
- `npm test`: 79 passed, 0 failed
- Node test coverage: 91.71% lines, 81.19% branches, 89.39% functions
- `npm pack --dry-run`: 75 files, 115.7 kB package
- fresh-prefix installation of the packed `0.2.7` package
- Python syntax compilation for all bundled runner scripts
- relevant `tuned-tensor-cli` model-serving tests: 29 passed, 0 failed

The high aggregate coverage does not cover the risky integration boundaries well:

- `src/index.ts`: 34.97% lines
- `src/doctor.ts`: 30.00% lines
- `src/prefetch.ts`: 75.32% lines, 35.71% branches
- no real Hugging Face download test
- no subprocess-level CLI contract tests
- no Python unit/integration tests
- CI covers Ubuntu x64, not Linux ARM64/DGX Spark or CUDA

### Mac dry and command-backed checks

A new project was initialized, validated, dry-run, listed, reported, served through
the dashboard, and rebuilt from its canonical files. JSON, JSONL, SQLite, spec, run,
event, report, and model records were inspected.

Additional reproductions:

- `doctor` returned nonzero on a dry-run/no-inference config only because the Mac
  lacks `nvidia-smi`.
- `validate` printed a generated run ID, but the subsequent spec-based `run`
  generated a different ID.
- a staged `prepare` invocation printed `"status": "completed"` while persisted
  state remained `preparing`.
- changing `n_epochs` from 3 to 9 under the same run ID reused every stage and left
  the persisted training hyperparameters at 3.
- a non-dry custom trainer that exited zero and wrote no files completed the run and
  registered an empty model directory.
- `tt-local run --help` queued and began a real run path.

### DGX Spark real workflow

Environment and result:

- audit workspace: `/home/eve/tt-local-audit-20260713-XnkDGt`
- run ID: `f9938d55-1c7d-40f1-9055-de8de5c57ddb`
- model ID: `local-f9938d55-1c7d-40f1-9055-de8de5c57ddb`
- resumed full workflow: 97.42 seconds
- baseline: about 30.9 seconds
- training stage: about 36.1 seconds; actual three-step trainer runtime 1.55 seconds
- candidate evaluation: about 30.1 seconds
- observed GPU memory: 3,760 MiB
- peak RSS: about 4.79 GB

Persistence checks:

- model archive: 18,821,402 bytes
- SHA-256: `ad5b329e89d35c271933c484924a056e4d76ad154823ff78df4e2597ab6404b4`
- `gzip -t` passed
- tar inspection passed
- adapter safetensors/config, tokenizer, chat template, and README were present
- the training-output tar and convenience copy had the same hash
- artifact/store request and report copies matched
- SQLite `integrity_check` returned `ok`
- store contained 1 spec, 1 run, 38 events, and 1 model
- `store rebuild-index`, `runs list/get/report/watch`, and `models list/get` passed
- the dashboard health, run, report, model, and spec APIs returned HTTP 200
- no audit process or server remains active

The successful run required both of these workarounds:

1. point `paths.baseModel` directly at the existing cached snapshot;
2. set `evaluation.inference.project` to `training/local-runner`.

## Workflow review

| Step | Result | UX assessment |
|---|---|---|
| Install package | Pass | Package contains the bundled scripts, but `tt-local --version` is unsupported and the package omits `uv.lock`. |
| `init` | Pass | Creates a runnable three-epoch placeholder and no runner config; accidental placeholder training is possible. |
| `doctor` | Misleading | Checks generic uv/Python and `nvidia-smi`, not the exact training/evaluator environments, Torch/CUDA, disk, cache, credentials, or writability. |
| `validate` | Partial | Schema validation passes, but does not validate placeholder content or runtime prerequisites and prints an ephemeral run ID. |
| `models prefetch` | Fail | Uses a cache layout different from training/evaluation and hides multi-GB progress unless `--verbose`. |
| Baseline | Workaround required | Default evaluator misses the bundled uv project; configured cache is also applied too late in Python. |
| Training | Pass with workaround | Real LoRA training and archive generation worked on Spark. |
| Candidate evaluation | Pass with workaround | Proved that the saved adapter tar can be extracted and loaded with its base model on CUDA. |
| Report and state | Pass with caveats | Reports and canonical copies persist, but stale errors and staged-state inconsistencies remain. |
| Recovery | Pass with caveats | SQLite rebuild works when canonical records exist; a trained artifact is not indexed until the final report completes. |
| Dashboard | Pass | `tt-local serve` is a state/report dashboard, not model inference. |
| Model inference server | Missing | `tt-local models serve` does not exist, and the separate hosted CLI handoff is not tested for TT Local adapters. |

## Release-blocking findings

### TT-L-001: prefetch and runtime caches are inconsistent

Severity: critical

`prefetch.py` calls `snapshot_download(cache_dir=paths.modelCache)`, which stores a
repository directly beneath that directory. Training treats the same value as
`HF_HOME`, whose Hub cache is beneath `<modelCache>/hub`. Evaluation passes the
cache in a JSON payload and changes `HF_HOME` only after importing Transformers and
Hugging Face libraries, so their cached path constants do not change.

Spark evidence:

- existing complete cache:
  `/home/eve/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B`
- existing size: 4,571,197,824 bytes
- prefetch began a second copy at:
  `/home/eve/.cache/huggingface/models--Qwen--Qwen3.5-2B`
- stopped after 101.92 seconds and 291,373,218 bytes of duplicate growth

Relevant code:

- `training/local-runner/src/prefetch.py:54-67`
- `src/process-training.ts:224-239`
- `training/local-runner/src/evaluate.py:15-18,303-312`
- `src/evaluation.ts:332-374`

Required adjustment:

- define one cache contract (`HF_HOME` or `HF_HUB_CACHE`) and use it identically in
  prefetch, training, baseline, candidate, and serving;
- set environment variables before Python imports;
- persist model revision/commit, snapshot path, file count, and byte count;
- add fresh, cached, interrupted, corrupt, gated, unwritable, and low-disk tests.

### TT-L-002: the documented/default packaged run does not select its Python project

Severity: critical

The homepage advertises `tt-local doctor` followed by `tt-local run` without a
config. Default config does not set the bundled uv project. The first Spark run and
a clean packaged run launched the evaluator as `uv run python .../evaluate.py` and
failed with `ModuleNotFoundError: No module named 'torch'`. `doctor` had passed.

Relevant code and docs:

- `src/contracts.ts:294-383`
- `src/process-runner.ts:30-49`
- `src/doctor.ts:43-109`
- sibling web repo `tuned-tensor/src/app/page.tsx:24-30,174-187`

Required adjustment:

- make both bundled training and bundled evaluation default to the installed
  `training/local-runner` project;
- make `doctor` resolve and execute the exact commands that `run` will use;
- run imports for Torch, Transformers, PEFT, and Hugging Face Hub and assert CUDA
  availability/device details when CUDA was requested.

### TT-L-003: resume invalidation omits hyperparameters and effective config

Severity: critical correctness

The stage source fingerprint includes the spec snapshot and selected dataset/eval
metadata, but not `request.hyperparameters` or most resolved training/evaluation
configuration. Existing baseline, training, and candidate files are then reused by
existence.

Reproduction: change `n_epochs` from 3 to 9 and rerun the same run ID. The CLI
reported reuse for prepare, baseline, training, and candidate; generated
`hyperparameters.json` still contained 3.

Relevant code:

- `src/orchestrator.ts:157-174,397-433`
- `src/orchestrator.ts:454-523,567-624`

Required adjustment:

- fingerprint the normalized full request, effective training config, effective
  evaluation/generation/scoring config, package/runtime version, model revision,
  dataset hashes, and parent artifact manifest;
- record per-stage input fingerprints and explain reuse/invalidation in CLI output;
- add a matrix of resume tests changing one input at a time.

### TT-L-004: model completion has no artifact validity contract

Severity: critical correctness

Training directories are created before the command runs. Any command that exits
zero can therefore fall back to an empty `training/model` directory. Candidate
evaluation with provider `none` or a permissive custom command does not prove the
artifact is loadable, yet the run and model are marked completed.

A non-dry `/usr/bin/true` trainer reproduced this: no model files were written, but
the run completed and `models list` registered the empty directory. Dry runs also
register empty simulated models.

Relevant code:

- `src/artifacts.ts:63-70`
- `src/process-training.ts:265-305`
- `src/orchestrator.ts:666-740`
- `src/store.ts:605-642`

Required adjustment:

- validate expected PEFT/full-model files after training;
- write an atomic `artifact-manifest.json` containing framework, format, base model
  and revision, files, sizes, and SHA-256 values;
- add `tt-local models verify <id-or-path>`;
- never register a dry-run model as real/servable;
- reject empty/custom artifacts unless their declared artifact contract passes.

### TT-L-005: CLI option parsing can start unintended work

Severity: critical UX/safety

Only top-level help is recognized. Unknown options are silently discarded.
Observed behavior:

```text
tt-local --version                  -> Unknown command
tt-local models prefetch --help     -> attempts the command
tt-local runs report --help         -> looks up run "--help"
tt-local run --help                 -> queues and starts a run
```

A typo such as `--dryrun` can therefore start a real local job.

Relevant code:

- `src/index.ts:100-188,305-320,373-439`

Required adjustment:

- use a strict command parser with per-command help, `--version`, required-value
  validation, mutually exclusive options, and rejection of unknown flags;
- add subprocess-level tests for every command before long-running work begins.

### TT-L-006: LLM-judge behavior can silently change metric semantics

Severity: critical evaluation correctness

The public real-run example sets `scoring.mode=llm_judge` and exports
`OPENROUTER_API_KEY`, but omits the required `llm` block. The implementation then
silently uses exact match. If `llm` exists but the key is initially absent, the
exact-match baseline is not marked as fallback and may be cached; adding the key
later can compare a cached exact-match baseline with a judge-scored candidate.

Relevant code and docs:

- `src/contracts.ts:350-390`
- `src/evaluation.ts:552-582,592-725,775-873`
- sibling web repo `tuned-tensor/src/app/docs/local-training/page.tsx:95-132`

Required adjustment:

- fail preflight when judge mode is requested but its model/config/key is absent,
  unless the user explicitly opts into fallback;
- record all fallback scoring as fallback and never cache it as a judge baseline;
- include judge availability in the cache key or invalidate on availability change;
- make the documented example assert `judge_scored_count > 0`.

## High-priority gaps

### Artifact discovery is coupled to successful evaluation/reporting

`training-report.json` is written after a successful trainer, but the model record is
created only in `completeRun`, after candidate evaluation and report generation. If
evaluation times out or the judge fails, the trained artifact remains on disk but is
absent from `models list`; index rebuild cannot recover a model record that was never
written.

Register a verified artifact immediately after training. Track training success,
evaluation success, and report success as separate states.

### `doctor` and `validate` do not preflight the real job

Missing checks include:

- exact installed training and evaluator commands;
- Python imports and CUDA-enabled Torch;
- requested device, GPU memory, and model fit;
- cache/artifact/store writability and free disk headroom;
- model reachability, revision, gated-model token, and partial cache state;
- configured judge/model/key;
- placeholder spec content and dataset readability/shape.

GPU checks should be conditional: a dry run, CPU command backend, or MPS workflow
should not fail solely because `nvidia-smi` is absent.

### No supported TT Local inference-serving endpoint

`tt-local serve` is the dashboard. `tt-local models serve` is unknown. The separate
`tt models serve` command was not installed on the Spark and has no real integration
test against the exact TT Local adapter tar. Code review also shows a likely mismatch:
TT Local nests archive files beneath `model/`, while the other CLI passes the archive
extraction root directly and its reference runtime is not designed around an explicit
base-model-plus-PEFT-adapter contract.

Add `tt-local models serve <local-model-id>` or a fully tested documented bridge.
Acceptance requires `/health`, `/v1/models`, and a real `/v1/chat/completions` request
using the saved artifact and its recorded base revision.

### Cancellation does not cancel subprocess work

`runs cancel` writes `cancel.requested` and changes metadata, but the orchestrator and
process runner never read the marker or terminate the child. Work can continue and a
later event can overwrite cancelled state. Cancellation must signal the process group,
wait for exit, and preserve a terminal cancelled state.

### Staged commands leave nonterminal persisted state

An individual stage prints `status: completed`, while the store remains in a running
state such as `preparing` or `training`. `runs watch` only exits for completed, failed,
or cancelled and can hang after a deliberately completed stage. Represent stage
completion explicitly and make watch aware of the requested terminal stage.

### Successful resume preserves a stale failure

The successful Spark resume is `completed`, but `runs get` and the dashboard still
expose the earlier `ModuleNotFoundError` in the `error` field, including after index
rebuild. Clear superseded errors when a run is retried and succeeds, while preserving
them in attempt history/events.

### Force/retry can retain stale model-directory files

Source refresh deletes report JSON files but not training model/output directories.
Stale files can be included in a later archive. Use a new attempt directory or clear
only the stage-owned directory before recomputation, then publish atomically.

## Medium-priority UX and documentation gaps

- `info` does not show the installed version; `--version` is unsupported.
- `init` creates placeholder training data that passes validation and defaults to
  three epochs, but creates no runner config.
- `validate` prints an unpersisted run ID different from the eventual spec-based run.
- prefetch progress is hidden by default and its help does not advertise `--verbose`.
- prefetch attempts are not indexed or discoverable through a cache-status command.
- `runs watch` is documented after a blocking foreground run, when it is no longer
  useful; document a second terminal, explicit run ID, or detached execution.
- `paths.modelCache`, artifact paths, and store paths need one clear path-resolution
  rule; current relative paths resolve from process CWD rather than config location.
- `HF_TOKEN` setup for gated models is not documented.
- “Files and Store” omits the model directory, copied tar, store report copy, and
  `models/<id>/model.json`.
- model tar is stored twice; deduplicate or document the durability reason.
- weight-loading progress is mislabeled as training progress; the Spark run reported
  0/320 to 320/320 shard loading before its real 3/3 training steps.
- the default 120-second timeout covers uv startup, model loading/download, and the
  whole evaluation batch. A one-example warm-cache Spark evaluation took about 30s.
- the published npm package omits `training/local-runner/uv.lock`, weakening runtime
  reproducibility and allowing dependency drift.
- a store-recovered report can retain stale absolute artifact URIs after artifacts
  are moved or removed.
- a successful resume updates artifact-side request data but may leave canonical
  store request/spec data stale.

## Recommended target workflow

The ideal first-run path should be short, explicit, and self-verifying:

```bash
tt-local --version
tt-local init --name "Support Bot" --model Qwen/Qwen3.5-2B --profile spark

# Edit tunedtensor.json, then:
tt-local doctor tunedtensor.json --config local-runner.json
tt-local models prefetch tunedtensor.json --config local-runner.json
tt-local models verify-base tunedtensor.json --config local-runner.json

tt-local run tunedtensor.json --config local-runner.json --detach
tt-local runs watch <run-id> --config local-runner.json
tt-local runs report <run-id> --config local-runner.json
tt-local models verify local-<run-id> --config local-runner.json

tt-local models serve local-<run-id> --spec tunedtensor.json
```

Suggested behavior:

- `init --profile spark` writes both a spec and a safe durable config.
- `doctor` prints an effective-plan summary: exact commands, paths, device, model
  revision, cached/download bytes, disk headroom, eval count, scoring mode, and
  expected artifact contract.
- prefetch shows byte progress by default and finishes with a verifiable snapshot
  manifest.
- `run` emits/persists the run ID before any model work and supports detach/watch.
- completed training makes a verified model discoverable even if evaluation fails.
- final output includes copy-paste report, verify, and serve commands.

## Acceptance test plan

1. **Clean packaged quickstart:** install the tarball without global ML packages;
   init, doctor, prefetch, run, report, verify, and serve with no source checkout.
2. **Fresh and warm Spark paths:** Linux ARM64/CUDA with empty cache, then cached
   retry; assert no duplicate bytes and identical model revision.
3. **Download resilience:** interrupt/resume, corrupt snapshot, gated model without
   token, unwritable cache, insufficient disk, and visible progress.
4. **Resume correctness:** change each hyperparameter/config/input under one run ID;
   assert exactly the dependent stages are invalidated.
5. **Artifact contract:** empty output, partial adapter, corrupt tar, wrong base model,
   and checksum mismatch must fail before model registration.
6. **Failure after training:** force candidate/judge/report failure and assert the
   verified trained model remains listed and servable.
7. **Dry-run contract:** no real model record, no cache download, terminal dry-run
   status, and valid orchestration artifacts.
8. **Judge correctness:** documented config performs judge calls, records the judge
   model, never caches silent fallback, and reports fallback prominently.
9. **Serving handoff:** exact emitted tar plus recorded base revision passes health,
   model listing, and a real chat request.
10. **CLI contract:** help/version, unknown flags, missing values, JSON mode, exit
    codes, staged watch, cancellation, and detached lifecycle.
11. **Persistence/recovery:** process restart, SQLite rebuild, moved roots, missing or
    corrupt canonical files, and stale-error clearing on successful retry.
12. **Platform matrix:** Ubuntu x64, Linux ARM64, Mac dry run/MPS as supported, plus a
    periodic real DGX Spark CUDA job.

## Residual audit data

The Spark audit intentionally did not delete generated data:

- audit workspace: about 74.7 MB
- interrupted duplicate prefetch data: about 291.2 MB
- total audit-related disk growth: about 366.4 MB

No existing project, run, or model was overwritten or deleted. Cleanup should be a
separate deliberate action after the cache-location fix is understood.
