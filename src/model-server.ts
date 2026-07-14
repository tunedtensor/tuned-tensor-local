import { spawn } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { LocalRunnerConfig } from "./contracts.js";
import { minimalMachineLearningEnvironment, withHuggingFaceCacheEnvironment } from "./huggingface-cache.js";
import { resolveTrainingModel } from "./model-registry.js";
import { buildEntrypointCommand } from "./process-runner.js";
import type { LocalModelRecord } from "./store.js";

export interface LocalModelServeOptions {
  host?: string;
  port?: number;
  device?: "auto" | "cpu" | "cuda" | "mps";
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  systemPrompt?: string;
  allowRemote?: boolean;
  apiKeyEnv?: string;
  maxConcurrentRequests?: number;
  baseModelRevision?: string;
  baseModelArtifactUri?: string;
}

export interface LocalModelServerLaunch {
  command: string;
  commandArgs: string[];
  displayCommand: string[];
  cwd?: string;
  env: NodeJS.ProcessEnv;
  url: string;
  modelName: string;
  artifactPath: string;
}

function localArtifactPath(uri: string): string {
  if (uri.startsWith("file://")) return fileURLToPath(uri);
  if (/^[a-z][a-z0-9+.-]*:/i.test(uri)) {
    throw new Error(`TT Local serving requires a local file artifact, got: ${uri}`);
  }
  return resolve(uri);
}

function boundedNumber(name: string, value: number, minimum: number, maximum: number, integer = false): number {
  if (!Number.isFinite(value) || value < minimum || value > maximum || (integer && !Number.isInteger(value))) {
    throw new Error(`${name} must be between ${minimum} and ${maximum}`);
  }
  return value;
}

function isLoopbackHost(host: string): boolean {
  return host === "127.0.0.1" || host === "localhost" || host === "::1" || host === "[::1]";
}

function httpHost(host: string): string {
  return host.includes(":") && !host.startsWith("[") ? `[${host}]` : host;
}

export function buildLocalModelServerLaunch(args: {
  model: LocalModelRecord;
  config: LocalRunnerConfig;
  options?: LocalModelServeOptions;
}): LocalModelServerLaunch {
  const options = args.options ?? {};
  const host = options.host ?? "127.0.0.1";
  const remote = !isLoopbackHost(host);
  if (remote && !options.allowRemote) {
    throw new Error("Refusing a non-loopback model server bind without --allow-remote.");
  }
  const apiKey = options.apiKeyEnv ? process.env[options.apiKeyEnv]?.trim() : undefined;
  if (options.apiKeyEnv && !apiKey) throw new Error(`${options.apiKeyEnv} is not set.`);
  if (remote && !apiKey) {
    throw new Error("Non-loopback serving requires --api-key-env with a populated environment variable.");
  }
  const port = boundedNumber("port", options.port ?? 8000, 1, 65_535, true);
  const maxTokens = boundedNumber("maxTokens", options.maxTokens ?? 512, 1, 8192, true);
  const temperature = boundedNumber("temperature", options.temperature ?? 0, 0, 5);
  const topP = boundedNumber("topP", options.topP ?? 1, 0, 1);
  const maxConcurrentRequests = boundedNumber(
    "maxConcurrentRequests",
    options.maxConcurrentRequests ?? 1,
    1,
    8,
    true,
  );
  const artifactPath = localArtifactPath(args.model.artifact_uri);
  const recordedBaseModelPath = options.baseModelArtifactUri
    ? localArtifactPath(options.baseModelArtifactUri)
    : undefined;
  const configuredBaseModelPath = args.config.paths.baseModel
    ? resolve(args.config.paths.baseModel)
    : undefined;
  if (
    recordedBaseModelPath
    && configuredBaseModelPath
    && resolve(recordedBaseModelPath) !== configuredBaseModelPath
  ) {
    throw new Error(
      `Configured paths.baseModel ${configuredBaseModelPath} does not match the base model recorded for this artifact: `
      + recordedBaseModelPath,
    );
  }
  const localBaseModelPath = recordedBaseModelPath ?? configuredBaseModelPath;
  const resolvedModel = resolveTrainingModel(args.model.base_model);
  const entrypoint = buildEntrypointCommand({
    backend: "uv",
    project: args.config.evaluation.inference.project ?? "training/local-runner",
    cwd: args.config.evaluation.inference.cwd,
    script: "training/local-runner/src/serve.py",
  });
  const modelName = args.model.id;
  const env = withHuggingFaceCacheEnvironment({
    ...minimalMachineLearningEnvironment(process.env, { includeHfToken: resolvedModel.requiresHfToken }),
    ...args.config.evaluation.inference.env,
    TT_MODEL_ARTIFACT: artifactPath,
    TT_BASE_MODEL: localBaseModelPath ?? args.model.base_model,
    ...(options.baseModelRevision && !localBaseModelPath
      ? { TT_BASE_MODEL_REVISION: options.baseModelRevision }
      : {}),
    TT_MODEL_NAME: modelName,
    TT_MODEL_LOADER: resolvedModel.loader,
    TT_HOST: host,
    TT_PORT: String(port),
    TT_DEVICE: options.device ?? args.config.evaluation.inference.device,
    TT_MAX_TOKENS: String(maxTokens),
    TT_TEMPERATURE: String(temperature),
    TT_TOP_P: String(topP),
    TT_TRUST_REMOTE_CODE: String(resolvedModel.trustRemoteCode && args.config.evaluation.inference.trustRemoteCode),
    TT_MAX_CONCURRENT_REQUESTS: String(maxConcurrentRequests),
    TT_CHAT_TEMPLATE_KWARGS: JSON.stringify(args.config.evaluation.inference.chatTemplateKwargs ?? {}),
    ...(options.systemPrompt ? { TT_SYSTEM_PROMPT: options.systemPrompt } : {}),
    ...(apiKey ? { TT_API_KEY: apiKey } : {}),
  }, args.config.paths.modelCache);
  return {
    command: entrypoint.command,
    commandArgs: entrypoint.commandArgs,
    displayCommand: entrypoint.displayCommand,
    cwd: entrypoint.kind === "uv" ? args.config.evaluation.inference.cwd : undefined,
    env,
    url: `http://${httpHost(host)}:${port}`,
    modelName,
    artifactPath,
  };
}

export async function serveLocalModel(launch: LocalModelServerLaunch): Promise<void> {
  await new Promise<void>((resolveServer, reject) => {
    const child = spawn(launch.command, launch.commandArgs, {
      cwd: launch.cwd,
      env: launch.env,
      stdio: "inherit",
      detached: process.platform !== "win32",
    });
    let stopping = false;
    let forceKillTimer: NodeJS.Timeout | undefined;
    const killProcessGroup = (signal: NodeJS.Signals) => {
      if (child.pid && process.platform !== "win32") {
        try {
          process.kill(-child.pid, signal);
          return;
        } catch {
          // Fall through when the process group has already exited.
        }
      }
      child.kill(signal);
    };
    const stop = (signal: NodeJS.Signals) => {
      if (stopping) return;
      stopping = true;
      killProcessGroup(signal);
      forceKillTimer = setTimeout(() => killProcessGroup("SIGKILL"), 5_000);
      forceKillTimer.unref();
    };
    const onSigint = () => stop("SIGINT");
    const onSigterm = () => stop("SIGTERM");
    const cleanup = () => {
      process.off("SIGINT", onSigint);
      process.off("SIGTERM", onSigterm);
      if (forceKillTimer) clearTimeout(forceKillTimer);
    };
    process.once("SIGINT", onSigint);
    process.once("SIGTERM", onSigterm);
    child.once("error", (error) => {
      cleanup();
      reject(error);
    });
    child.once("close", (code, signal) => {
      cleanup();
      if (stopping || signal === "SIGINT" || signal === "SIGTERM") {
        resolveServer();
      } else if (code === 0) {
        resolveServer();
      } else {
        reject(new Error(`Local model server exited with code ${code ?? "unknown"}.`));
      }
    });
  });
}
