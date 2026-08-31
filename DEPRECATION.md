# `tt-local` is deprecated

Use the main Tuned Tensor CLI for local and cloud work:

```bash
npm install -g @tuned-tensor/cli
tt local <command>
```

For an adapter project created with this package, replace `tt-local ...` with
`tt local ...`. The rest of the command stays the same.

Those existing projects need no migration. `tt local` uses the same
`tunedtensor.json`, `local-runner.json`, `.tt-local` artifacts, and local store.

The standalone package is a frozen legacy adapter-only CLI. It continues to
support its Qwen 3.5 2B adapter workflow, but it is not the implementation of
current `tt` features. The Foundation Pipeline and newer certified models are
available only in the [main CLI](https://github.com/tunedtensor/tuned-tensor-cli).

## Examples

| Before | Now |
| --- | --- |
| `tt-local init ...` | `tt local init ...` |
| `tt-local doctor ...` | `tt local doctor ...` |
| `tt-local run ...` | `tt local run ...` |
| `tt-local runs report ...` | `tt local runs report ...` |
| `tt-local serve ...` | `tt local serve ...` |
