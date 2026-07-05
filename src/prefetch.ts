import { mkdir, readFile, stat } from "node:fs/promises";
import { join, resolve } from "node:path";
import type { FineTuneRunRequest, LocalRunnerConfig } from "./contracts.js";
import { fileUri, writeJson } from "./artifacts.js";
import { resolveTrainingModel } from "./model-registry.js";
import { buildEntrypointCommand, runLoggedProcess } from "./process-runner.js";
import type { LocalRunReporter } from "./run-reporter.js";

const PREFETCH_SCRIPT = "training/local-runner/src/prefetch.py";

export interface ModelPrefetchPayload {
  base_model: string;
  loader: string;
  trust_remote_code: boolean;
  requires_hf_token: boolean;
  model_cache?: string;
}

export interface ModelPrefetchReport {
  ok: boolean;
  status: "completed" | "skipped";
  base_model: string;
  loader?: string;
  model_cache?: string;
  snapshot_path?: string;
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
    loader: model.loader,
    trust_remote_code: model.trustRemoteCode,
    requires_hf_token: model.requiresHfToken,
    ...(config.paths.modelCache ? { model_cache: resolve(config.paths.modelCache) } : {}),
  };
}

function safeName(value: string): string {
  return value.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "model";
}

export async function prefetchBaseModel(args: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  reporter?: LocalRunReporter;
}): Promise<ModelPrefetchReport> {
  const artifactDir = resolve(
    args.config.artifactRoot,
    "prefetch",
    `${safeName(args.request.spec_snapshot.base_model)}-${new Date().toISOString().replace(/[:.]/g, "-")}`,
  );
  await mkdir(artifactDir, { recursive: true });

  if (args.config.paths.baseModel) {
    const localPath = resolve(args.config.paths.baseModel);
    try {
      await stat(localPath);
    } catch {
      throw new Error(`paths.baseModel is set to ${localPath}, but that path does not exist.`);
    }
    return {
      ok: true,
      status: "skipped",
      base_model: args.request.spec_snapshot.base_model,
      local_base_model_path: localPath,
      artifact_dir: artifactDir,
      reason: "paths.baseModel points at a local base-model artifact; no Hugging Face download is needed.",
    };
  }

  const payload = buildModelPrefetchPayload(args.request, args.config);
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
    message: "Prefetching Hugging Face base model.",
    details: {
      base_model: payload.base_model,
      model_cache: payload.model_cache ?? null,
      command: entrypoint.displayCommand,
      log_path: logPath,
    },
  });

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    HF_HOME: payload.model_cache ?? process.env.HF_HOME,
  };
  if (env.HF_HOME === undefined) delete env.HF_HOME;

  const { exitCode } = await runLoggedProcess({
    command: entrypoint.command,
    commandArgs: entrypoint.commandArgs,
    env,
    logPath,
    reporter: args.reporter,
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
    snapshot_path?: string;
  };

  await args.reporter?.onEvent?.({
    stage: "model_prefetch",
    status: "completed",
    message: "Base model is available in the local Hugging Face cache.",
    details: {
      base_model: output.base_model ?? payload.base_model,
      snapshot_path: output.snapshot_path ?? null,
    },
  });

  return {
    ok: true,
    status: "completed",
    base_model: output.base_model ?? payload.base_model,
    loader: output.loader ?? payload.loader,
    model_cache: output.model_cache ?? payload.model_cache,
    snapshot_path: output.snapshot_path,
    artifact_dir: artifactDir,
    input_path: inputPath,
    output_path: outputPath,
    log_uri: fileUri(logPath),
    command: entrypoint.displayCommand,
  };
}
