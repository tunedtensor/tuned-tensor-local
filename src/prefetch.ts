import { lstat, mkdir, readFile, readdir, stat } from "node:fs/promises";
import { join, resolve } from "node:path";
import type { FineTuneRunRequest, LocalRunnerConfig } from "./contracts.js";
import { fileUri, writeJson } from "./artifacts.js";
import { resolveTrainingModel } from "./model-registry.js";
import { buildEntrypointCommand, runLoggedProcess } from "./process-runner.js";
import type { LocalRunReporter } from "./run-reporter.js";
import { minimalMachineLearningEnvironment, withHuggingFaceCacheEnvironment } from "./huggingface-cache.js";

const PREFETCH_SCRIPT = "training/local-runner/src/prefetch.py";

export interface ModelPrefetchPayload {
  base_model: string;
  revision?: string;
  loader: string;
  trust_remote_code: boolean;
  requires_hf_token: boolean;
  model_cache?: string;
  local_files_only?: boolean;
}

export interface ModelPrefetchReport {
  ok: boolean;
  status: "completed" | "skipped";
  base_model: string;
  loader?: string;
  model_cache?: string;
  hf_home?: string;
  hub_cache?: string;
  snapshot_path?: string;
  snapshot_revision?: string;
  file_count?: number;
  size_bytes?: number;
  verified_blob_count?: number;
  local_base_model_path?: string;
  artifact_dir: string;
  input_path?: string;
  output_path?: string;
  log_uri?: string;
  command?: string[];
  reason?: string;
}

export function buildModelPrefetchPayload(
  request: FineTuneRunRequest,
  config: LocalRunnerConfig,
): ModelPrefetchPayload {
  const model = resolveTrainingModel(request.spec_snapshot.base_model);
  return {
    base_model: model.id,
    ...(request.hyperparameters.base_model_revision
      ? { revision: request.hyperparameters.base_model_revision }
      : {}),
    loader: model.loader,
    trust_remote_code: model.trustRemoteCode,
    requires_hf_token: model.requiresHfToken,
    ...(config.paths.modelCache ? { model_cache: resolve(config.paths.modelCache) } : {}),
  };
}

function safeName(value: string): string {
  return value.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "model";
}

function transformersWeightName(name: string): boolean {
  const lower = name.split(/[\\/]/).at(-1)?.toLowerCase() ?? "";
  return (lower.endsWith(".safetensors") && !lower.startsWith("adapter_"))
    || /^pytorch_model.*\.bin$/.test(lower);
}

export async function verifyLocalBaseModel(path: string): Promise<{ fileCount: number; sizeBytes: number }> {
  const root = await lstat(path).catch(() => null);
  if (!root) throw new Error(`paths.baseModel is set to ${path}, but that path does not exist.`);
  if (root.isSymbolicLink()) {
    throw new Error(`paths.baseModel must not itself be a symbolic link: ${path}`);
  }
  if (!root.isDirectory()) {
    throw new Error(
      `paths.baseModel must be a Hugging Face snapshot directory, not a standalone file or archive: ${path}`,
    );
  }
  const files: Array<{ path: string; size: number }> = [];
  const visit = async (directory: string): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const child = join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(child);
        continue;
      }
      const metadata = await stat(child);
      if (entry.isSymbolicLink() && !metadata.isFile()) {
        throw new Error(`Local base model contains a non-file symbolic link: ${child}`);
      }
      if (metadata.isFile()) files.push({ path: child, size: metadata.size });
    }
  };
  await visit(path);
  const nonEmpty = files.filter((file) => file.size > 0);
  const weights = nonEmpty.filter((file) => transformersWeightName(file.path));
  if (weights.length === 0) {
    throw new Error(`Local base-model directory contains no non-empty Transformers model weights: ${path}`);
  }
  const required = async (name: string) => {
    const metadata = await stat(join(path, name)).catch(() => null);
    return Boolean(metadata?.isFile() && metadata.size > 0);
  };
  if (!await required("config.json")) throw new Error(`Local base-model directory is missing config.json: ${path}`);
  try {
    JSON.parse(await readFile(join(path, "config.json"), "utf8"));
  } catch (error) {
    throw new Error(`Local base-model directory has an invalid config.json: ${path}`, { cause: error });
  }
  const vocabNames = [
    "tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model", "spiece.model", "vocab.json", "tokenizer.tiktoken",
  ];
  if (!await required("tokenizer_config.json") || !(await Promise.all(vocabNames.map(required))).some(Boolean)) {
    throw new Error(`Local base-model directory is missing tokenizer metadata or vocabulary: ${path}`);
  }
  for (const file of files.filter((candidate) => candidate.path.endsWith(".index.json"))) {
    const parsed = JSON.parse(await readFile(file.path, "utf8")) as { weight_map?: unknown };
    if (!parsed.weight_map || typeof parsed.weight_map !== "object" || Array.isArray(parsed.weight_map)) {
      throw new Error(`Local base-model weight index is invalid: ${file.path}`);
    }
    for (const shard of new Set(Object.values(parsed.weight_map as Record<string, unknown>))) {
      if (typeof shard !== "string" || !await required(shard)) {
        throw new Error(`Local base-model directory is missing indexed weight shard: ${String(shard)}`);
      }
    }
  }
  return {
    fileCount: files.length,
    sizeBytes: files.reduce((total, file) => total + file.size, 0),
  };
}

export async function prefetchBaseModel(args: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  reporter?: LocalRunReporter;
  localOnly?: boolean;
}): Promise<ModelPrefetchReport> {
  const artifactDir = resolve(
    args.config.artifactRoot,
    args.localOnly ? "verify-base" : "prefetch",
    `${safeName(args.request.spec_snapshot.base_model)}-${new Date().toISOString().replace(/[:.]/g, "-")}`,
  );
  await mkdir(artifactDir, { recursive: true });

  if (args.config.paths.baseModel) {
    const localPath = resolve(args.config.paths.baseModel);
    const verified = await verifyLocalBaseModel(localPath);
    return {
      ok: true,
      status: args.localOnly ? "completed" : "skipped",
      base_model: args.request.spec_snapshot.base_model,
      local_base_model_path: localPath,
      file_count: verified.fileCount,
      size_bytes: verified.sizeBytes,
      artifact_dir: artifactDir,
      reason: args.localOnly
        ? "Verified the configured local base-model artifact; no network access was used."
        : "paths.baseModel points at a verified local base-model artifact; no Hugging Face download is needed.",
    };
  }

  const payload: ModelPrefetchPayload = {
    ...buildModelPrefetchPayload(args.request, args.config),
    ...(args.localOnly ? { local_files_only: true } : {}),
  };
  const inputPath = join(artifactDir, "prefetch-input.json");
  const outputPath = join(artifactDir, "prefetch-output.json");
  const logPath = join(artifactDir, "prefetch.log");
  await writeJson(inputPath, payload);

  const entrypoint = buildEntrypointCommand({
    backend: "uv",
    project: "training/local-runner",
    script: PREFETCH_SCRIPT,
  }, {
    extraArgs: ["--input", inputPath, "--output", outputPath],
  });

  await args.reporter?.onEvent?.({
    stage: "model_prefetch",
    status: "running",
    message: args.localOnly
      ? "Verifying the Hugging Face base model is available in the local cache."
      : "Prefetching Hugging Face base model.",
    details: {
      base_model: payload.base_model,
      model_cache: payload.model_cache ?? null,
      command: entrypoint.displayCommand,
      log_path: logPath,
    },
  });

  const env = withHuggingFaceCacheEnvironment(
    minimalMachineLearningEnvironment(process.env, { includeHfToken: payload.requires_hf_token }),
    payload.model_cache,
  );

  const { exitCode } = await runLoggedProcess({
    command: entrypoint.command,
    commandArgs: entrypoint.commandArgs,
    env,
    logPath,
    // A base model can be several gigabytes. Surface the downloader's progress
    // by default; callers can still opt out by omitting the reporter (`--quiet`).
    reporter: args.reporter ? { ...args.reporter, verbose: true } : undefined,
    stage: "model_prefetch",
  });

  if (exitCode !== 0) {
    throw new Error(`Model prefetch exited with code ${exitCode}. See ${logPath}.`);
  }

  const output = JSON.parse(await readFile(outputPath, "utf8")) as {
    ok?: boolean;
    base_model?: string;
    loader?: string;
    model_cache?: string;
    hf_home?: string;
    hub_cache?: string;
    snapshot_path?: string;
    snapshot_revision?: string;
    file_count?: number;
    size_bytes?: number;
    verified_blob_count?: number;
  };

  await args.reporter?.onEvent?.({
    stage: "model_prefetch",
    status: "completed",
    message: "Base model is available in the local Hugging Face cache.",
    details: {
      base_model: output.base_model ?? payload.base_model,
      hf_home: output.hf_home ?? output.model_cache ?? payload.model_cache ?? null,
      hub_cache: output.hub_cache ?? null,
      snapshot_path: output.snapshot_path ?? null,
      snapshot_revision: output.snapshot_revision ?? null,
      file_count: output.file_count ?? null,
      size_bytes: output.size_bytes ?? null,
      verified_blob_count: output.verified_blob_count ?? null,
    },
  });

  return {
    ok: true,
    status: "completed",
    base_model: output.base_model ?? payload.base_model,
    loader: output.loader ?? payload.loader,
    model_cache: output.model_cache ?? payload.model_cache,
    hf_home: output.hf_home ?? output.model_cache ?? payload.model_cache,
    hub_cache: output.hub_cache,
    snapshot_path: output.snapshot_path,
    snapshot_revision: output.snapshot_revision,
    file_count: output.file_count,
    size_bytes: output.size_bytes,
    verified_blob_count: output.verified_blob_count,
    artifact_dir: artifactDir,
    input_path: inputPath,
    output_path: outputPath,
    log_uri: fileUri(logPath),
    command: entrypoint.displayCommand,
  };
}
