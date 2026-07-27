# Architecture

This repository is the maintenance-mode local runtime behind the unified
`tt local` workflow. The standalone `tt-local` product surface is deprecated;
see [the deprecation note](../DEPRECATION.md). Its supported runtime boundary
is text SFT of `Qwen/Qwen3.5-2B` into a LoRA adapter on CUDA.

## Workflow

1. Parse a strict behavior spec or request.
2. Compile or validate chat JSONL and select distinct training/evaluation rows.
3. Evaluate the base model with the bundled Transformers runner.
4. Train a PEFT adapter with the bundled, locked `uv` project.
5. Verify and register the adapter before downstream evaluation.
6. Evaluate the adapter on the same held-out rows.
7. Write a base-versus-tuned report.

The orchestrator retains explicit stages and fingerprints internally so an
interrupted run can safely resume. These mechanics are reliability
implementation details, not a public plugin framework.

## Boundaries

The first version intentionally excludes:

- arbitrary training or evaluation commands;
- DPO, continued pretraining, and multimodal training;
- cloud labeling or LLM-as-judge calls;
- classic tabular model studies;
- a local web dashboard;
- unverified model-family claims.

The next model or method should arrive with its own locked dependencies,
resource defaults, data contract, and real CUDA acceptance test.

## Python boundary

Node owns input validation, state, process lifecycle, manifests, and reports.
The bundled Python project owns four small operations:

- `prefetch.py`: download and verify the pinned Hugging Face snapshot;
- `evaluate.py`: generate predictions without receiving reference answers;
- `train.py`: perform CUDA-only Qwen LoRA SFT;
- `serve.py`: load the base model and verified adapter for inference.

Training uses assistant-only loss. Prompt tokens have label `-100`, and
truncation removes older prompt tokens before it removes any answer token.
Tokenizer chat-template failures are fatal because a fallback would silently
change the trained or served prompt format.

## Evaluation

Baseline and candidate evaluation use the same input IDs, generation settings,
and deterministic scoring:

- baseline: recorded base-model snapshot;
- candidate: that base snapshot plus the run's verified PEFT adapter;
- scoring: normalized exact match or selected JSON fields.

The Python evaluator receives only opaque IDs and prompts. Node joins its
predictions to trusted references before scoring. This avoids accidental label
leakage across the process boundary.

## Storage and integrity

Canonical JSON and JSONL files are the state store; there is no database or
mirrored index. Writes are atomic and listings scan the small local metadata
tree deterministically.

Each real adapter has an atomic `artifact-manifest.json` with the expected PEFT
files, byte sizes, and SHA-256 hashes. A model is registered immediately after
training and manifest verification, so a later evaluation failure does not
hide a valid artifact. Serving verifies the manifest again.

## Configuration

The runner config controls only:

- state, artifact, and Hugging Face cache paths;
- CUDA or CPU inference for evaluation and serving;
- deterministic generation and scoring limits.

The training project, Python entrypoints, working directory, and child
environment are internal and fixed. Every stage uses the locked `uv` project
included in the npm package. The mutable virtual environment lives in the
user cache, keyed by the project and lockfile content, so a read-only global
npm install remains runnable.

Schemas are strict. Misspelled or obsolete fields fail validation instead of
silently reverting to defaults. Training itself is CUDA-only and fails fast
when PyTorch cannot see the GPU.
