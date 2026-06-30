# Tuned Tensor Local

Tuned Tensor Local is an open-source local runner for fine-tuning small
open-weight models from behavior specs on a machine you control.

The project is starting as a clean standalone implementation. It is designed to
run without a required cloud provider, with local disk artifacts, local process
orchestration, and Docker-based GPU execution.

## Goals

- Run a complete fine-tuning workflow on a single GPU host.
- Keep training data, model artifacts, logs, and reports on local disk.
- Use Docker for reproducible training and evaluation environments.
- Produce portable run reports that can be inspected or integrated elsewhere.
- Keep the public implementation independent from Tuned Tensor's hosted runner.

## Non-Goals For The Initial Version

- Distributed scheduling.
- Hosted product billing, account, or deployment integrations.
- Managed cloud training or serving providers.
- A compatibility promise for private internal service code.

## Planned Workflow

1. Read a behavior spec and run configuration.
2. Compile examples into training and evaluation datasets.
3. Run a baseline evaluation.
4. Launch local GPU fine-tuning in Docker.
5. Run candidate evaluation against the tuned artifact.
6. Write a structured report under the local artifact root.

## Current Status

This repository is an initial scaffold. The next milestone is a minimal local
orchestrator that can execute a tiny smoke run on one GPU machine.

## Development

```bash
npm install
npm run typecheck
npm run build
npm test
```

## License

Apache-2.0
