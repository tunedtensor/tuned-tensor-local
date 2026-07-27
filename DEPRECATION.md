# `tt-local` is deprecated

Use the main Tuned Tensor CLI for local and cloud work:

```bash
npm install -g @tuned-tensor/cli
tt local <command>
```

Replace `tt-local ...` with `tt local ...`. The rest of the command stays the
same.

Existing projects need no migration. `tt local` uses the same
`tunedtensor.json`, `local-runner.json`, `.tt-local` artifacts, and local store.

The standalone command will continue to work for compatibility. This package
now serves as the local runtime behind `tt`; new user-facing work belongs in
the [main CLI](https://github.com/tunedtensor/tuned-tensor-cli).

## Examples

| Before | Now |
| --- | --- |
| `tt-local init ...` | `tt local init ...` |
| `tt-local doctor ...` | `tt local doctor ...` |
| `tt-local run ...` | `tt local run ...` |
| `tt-local runs report ...` | `tt local runs report ...` |
| `tt-local serve ...` | `tt local serve ...` |
