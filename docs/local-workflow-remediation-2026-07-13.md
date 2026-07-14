# TT Local workflow remediation

Date: 2026-07-13
Source review: `docs/local-workflow-ux-review-2026-07-13.md`
Scope: the current, unreleased working-tree patch

This document records what the patch changes, how the intended first-run flow
should work, and what remains. “Implemented” below means code and focused tests
exist in the working tree. The validation record below distinguishes automated
local evidence from the packaged DGX Spark acceptance run.

## Critical findings

| Finding | Patch status | Remediation in this patch | Remaining qualification |
|---|---|---|---|
| TT-L-001: inconsistent Hugging Face caches | Implemented and warm-cache validated on DGX Spark | `paths.modelCache` now has one meaning: `HF_HOME`. Prefetch, training, evaluation, doctor, and serving resolve the Hub cache beneath `<HF_HOME>/hub`; Python sets cache variables before importing Hugging Face libraries. Prefetch reports the resolved snapshot revision, file count, and bytes. `models verify-base` performs a local-only snapshot check. The packaged Spark run reused the existing Hub snapshot and did not grow the legacy duplicate layout from the prior audit. | The fresh/interrupted/corrupt/gated/unwritable/low-disk matrix is not complete, and prefetch attempts are not indexed for later inspection. |
| TT-L-002: packaged defaults select the wrong Python environment | Implemented and packaged-host validated | Bundled training and Transformers evaluation default to the packaged `training/local-runner` uv project. `doctor` checks the effective commands, required imports, requested CUDA/MPS availability, storage writability/free space, placeholder input, judge configuration, and gated-model token requirements. The package now includes `uv.lock`. A fresh-prefix tarball ran both evaluator and trainer on Linux ARM64/CUDA. | There is no automated Linux ARM64/CUDA CI job, and doctor does not yet estimate whether a selected model and training configuration fit available VRAM. |
| TT-L-003: unsafe resume reuse | Implemented for identified inputs | The run request (including hyperparameters), effective runner/evaluation configuration, dataset and local multimodal-asset contents, TT Local/Node/platform identity, bundled `pyproject.toml`/`uv.lock`, local base snapshot, immutable remote revision, and parent-artifact contents participate in source or per-stage fingerprints. Reuse also verifies manifested outputs; invalidation removes stage-owned stale directories and dependent outputs. A stored `--parent-model` pins or validates the base identity before launch. | A remote base-model branch should still be explicitly pinned by commit for reproducible execution. The broader one-input-at-a-time and interruption matrix remains outstanding. |
| TT-L-004: no model artifact validity contract | Implemented for bundled formats | Real training must produce a non-empty model payload. An atomic `artifact-manifest.json` records the model contract plus file sizes and SHA-256 hashes; gzip/tar integrity is checked for archives. PEFT artifacts require exact adapter weights plus `adapter_config.json`; full-model contracts require recognized Transformers weights. Optimizer-only and incomplete adapters are rejected in fixtures. Dry runs do not create model records. A verified real artifact is registered immediately after training, and `models verify` rechecks the manifest before serving. | A live load/serve smoke test remains the final loadability proof for a particular framework/version combination. Explicit custom command-backend contracts remain caller-defined by design. |
| TT-L-005: unsafe CLI parsing | Implemented | Parsing is strict: top-level and nested help are side-effect free, `--version` is supported, unknown/duplicate/extra flags and missing values fail before work begins, and `--option=value` is accepted. Subprocess-level CLI tests cover the safety contract. | Continue adding each new command and option to the parser contract as the CLI grows. |
| TT-L-006: silent judge fallback | Implemented | `exact_match` is the safe default. Explicit `llm_judge` mode preflights its model/config/key and fails unless `fallback: "exact_match"` was deliberately configured. Fallback results are marked, excluded from judge baseline caching, and judge configuration/key availability participates in cache identity. | A real external-judge acceptance run should still assert `judge_scored_count > 0`; unit tests cannot prove provider availability. |

## High-priority findings

| Gap from the review | Patch response | Status |
|---|---|---|
| A trained artifact is undiscoverable when later evaluation/reporting fails | Register the manifested model immediately after successful training, independently of final report completion. | Implemented and covered by an integration test that forces candidate evaluation failure. |
| `doctor`/`validate` do not preflight the real job | Reject unchanged generated placeholders and invalid judge setups; run exact bundled-environment imports and device checks; check paths, disk, cache, and gated-model credentials. | Implemented; model-fit VRAM estimation remains open. |
| No supported TT Local inference handoff | Add `tt-local models serve`, which verifies the stored adapter and exposes `/health`, `/v1/models`, and `/v1/chat/completions` on localhost by default. Non-loopback binding requires explicit opt-in and bearer-token configuration. | Implemented; the newly trained Spark adapter passed all three live endpoints on CUDA. Streaming remains absent. |
| Cancellation only changes metadata | Poll the cancellation marker at stage boundaries and during long-running evaluation/training; terminate and await the child process group, then publish a terminal `cancelled` state. Late progress/failure/completion writes preserve the request until orchestration finalizes it. | Implemented with process-group, JSON-command, timeout, reporter-failure, and store-race tests; cross-platform process-tree coverage remains to expand. |
| Staged commands leave a nonterminal run | Persist `stage_completed` after an intentionally requested single stage so `runs watch` exits. | Implemented. |
| Successful retry retains stale failure state | Clear superseded run errors when active work restarts or a stage/run succeeds, while retaining event history. | Implemented. |
| Retry/force retains stale model files | Remove only the selected stage's owned directory and dependent outputs before recomputation, then regenerate manifests atomically. | Implemented; interruption-at-each-publication-point tests remain outstanding. |

## Improved target workflow

The following is the intended CUDA/Spark first-run path from an installed
package. Replace the generated prompt/example before validation; TT Local
rejects the unchanged placeholder.

```bash
npm install -g @tuned-tensor/local
tt-local --version
uv --version

tt-local init \
  --name "Support Bot" \
  --model Qwen/Qwen3.5-2B \
  --profile spark

# Edit tunedtensor.json, then run the exact preflight for this project.
tt-local doctor tunedtensor.json --config local-runner.json
tt-local validate tunedtensor.json --config local-runner.json

# Make the multi-gigabyte download explicit and verify it without network repair.
tt-local models prefetch tunedtensor.json --config local-runner.json
tt-local models verify-base tunedtensor.json --config local-runner.json

# Persist the run ID before model work, run in the background, and watch it.
tt-local run tunedtensor.json --config local-runner.json --detach
RUN_ID="<run-id returned by --detach>"
tt-local runs watch "$RUN_ID" --config local-runner.json

# Inspect results and independently verify the saved model contract.
tt-local runs report "$RUN_ID" --config local-runner.json
MODEL_ID="local-${RUN_ID}"
tt-local models verify "$MODEL_ID" --config local-runner.json

# Start the model endpoint. `tt-local serve` is the separate run dashboard.
tt-local models serve "$MODEL_ID" \
  --config local-runner.json \
  --spec tunedtensor.json \
  --host 127.0.0.1 \
  --port 8000 \
  --device cuda
```

From a second terminal, check the serving handoff:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

For a gated base model, export `HF_TOKEN` in the shell or an uncommitted project
environment before `doctor`/prefetch. Do not commit credentials, caches, model
artifacts, extracted runtimes, or serving logs.

## Validation record

This record is updated from commands run against this patch. Rows not exercised
are called out explicitly rather than inferred from unit coverage.

| Validation | Result |
|---|---|
| TypeScript typecheck | Pass: `npm run typecheck`. |
| Node test suite | Pass: `npm test` — 139 passed, 0 failed. |
| Build and package contents | Pass: `npm run build`; `npm pack --dry-run --json` reports 85 entries and includes `training/local-runner/src/serve.py` plus `training/local-runner/uv.lock`. |
| Python syntax and lockfile | Pass: `python3 -m py_compile training/local-runner/src/*.py`; `uv lock --check --project training/local-runner` resolved 85 packages without changing the lock. |
| Fresh-prefix packaged CLI | Pass in `/tmp/tt-local-package-validate.on7l0s`: fresh npm-prefix install, version/help/init/doctor/validate, and full dry run `36cf513f-8846-468f-bca4-4e095f11492e`. |
| DGX Spark cache flow | Pass in `/home/eve/tt-local-patch-test-20260713-BwhiDc`: `HF_HOME=/home/eve/.cache/huggingface`, Hub revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, 10 files, 4,571,198,095 verified bytes. The prior-audit legacy duplicate remained exactly 291,226,208 bytes; the patch created no second layout. |
| DGX Spark real run | Pass on NVIDIA GB10/CUDA: run `0f677a15-5574-4ff1-9aa2-b27780c1c9bd`, model `local-0f677a15-5574-4ff1-9aa2-b27780c1c9bd`, terminal `completed`, 157.992 seconds. `models verify` checked three manifested artifacts and a 35,103,917-byte PEFT tar with base revision pinned. An initial attempt exposed PyTorch 2.13's optional native-JIT dependency on system Python headers; the bundled environment now chooses the eager fallback by default, and the repeat passed. |
| Detach/watch/cancel lifecycle | Pass: successful detached PID 35830 exited after watch completed. Cancellation run `22ee1986-288c-438b-a26e-5f2bc6ebcdcd` moved through `cancel_requested` to terminal `cancelled`; PID 37720 exited and no TT Local/evaluator/trainer process remained. |
| Saved-model serving | Pass on `127.0.0.1:18080`: `/health` reported CUDA, `/v1/models` listed the new model, and chat returned `positive`; Ctrl-C stopped the endpoint and no server process remained. |
| Final diff review | `git diff --check` passes. A second-agent review returned GO with no P0/P1 findings; the modified/untracked secret-pattern scan was clean. |

## Honest residual gaps

The patch does not close the entire acceptance matrix from the review:

- Prefetch/download attempts are not indexed, so interrupted attempts and cache
  history are not discoverable through a dedicated status command.
- CI does not run Linux ARM64 or CUDA; the successful DGX Spark check remains a
  manual acceptance environment.
- There is no model-fit VRAM estimate before training.
- Store reports and manifests still need a complete relocation/moved-root
  strategy when artifact or store trees are copied elsewhere.
- The OpenAI-compatible serving endpoint does not implement streaming.
- The broader resilience matrix is outstanding: fresh/interrupted/corrupt
  downloads, every resume-input mutation, publication-point interruption,
  cross-platform process cancellation, moved roots, and missing/corrupt
  canonical files.
- Mutable remote model branches cannot be made reproducible by cache inspection
  alone; production specs should set an immutable `base_model_revision`.
- Bundled ML subprocesses intentionally receive a minimal environment. Proxy-only
  or private-package-index deployments need an explicit, documented credential
  forwarding mechanism for prefetch rather than inheriting the whole shell.
