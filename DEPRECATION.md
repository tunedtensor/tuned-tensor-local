# Standalone CLI deprecation

The standalone `tt-local` command is deprecated in favor of the unified
Tuned Tensor CLI:

```bash
npm install -g @tuned-tensor/cli
tt local <command>
```

The local fine-tuning experiment proved the workflow: local CUDA training,
held-out evaluation, artifact verification, activation, rollback, and serving
now work through `tt local`. Keeping cloud and local workflows in one `tt`
surface lets the project focus product and user-experience improvements in one
place.

## What this means

- Do not start new scripts or documentation with the standalone `tt-local`
  command.
- Existing standalone commands continue to work for compatibility.
- The `@tuned-tensor/local` package remains in maintenance mode as the runtime
  used internally by `@tuned-tensor/cli`.
- Reliability and security fixes may still be made here, but new user-facing
  workflows and UX should be designed in
  [tuned-tensor-cli](https://github.com/tunedtensor/tuned-tensor-cli).
- There is no planned data migration. `tt local` uses the same
  `tunedtensor.json`, `local-runner.json`, `.tt-local` artifacts, and local
  store.

## Command migration

The command grammar is unchanged; add `local` after `tt`:

| Deprecated | Supported |
| --- | --- |
| `tt-local init ...` | `tt local init ...` |
| `tt-local doctor ...` | `tt local doctor ...` |
| `tt-local validate ...` | `tt local validate ...` |
| `tt-local run ...` | `tt local run ...` |
| `tt-local runs report ...` | `tt local runs report ...` |
| `tt-local models verify ...` | `tt local models verify ...` |
| `tt-local serve ...` | `tt local serve ...` |

After installing the unified CLI, existing projects can be used in place:

```bash
cd /path/to/existing/project
tt status
tt local doctor tunedtensor.json
tt local run tunedtensor.json
```
