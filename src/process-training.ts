import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import type { FineTuneRunRequest, LocalRunnerConfig, TrainingReport } from "./contracts.js";
import type { RunArtifacts } from "./artifacts.js";
import { copyDatasetToTrainingChannel, fileUri, writeJson } from "./artifacts.js";
import { resolveTrainingModel } from "./model-registry.js";
import {
  buildBundledPythonCommand,
  runLoggedProcess,
  withBundledPythonEnvironment,
} from "./process-runner.js";
import { reportInBackground, type LocalRunReporter } from "./run-reporter.js";
import {
  minimalMachineLearningEnvironment,
  withOfflineHuggingFaceCacheEnvironment,
} from "./huggingface-cache.js";

export function buildTrainingHyperparameters(
  request: FineTuneRunRequest,
  options: { baseModelRevision?: string } = {},
): Record<string, string> {
  const model = resolveTrainingModel(request.spec_snapshot.base_model);
  const hyper = request.hyperparameters;
  return {
    base_model: model.id,
    ...(options.baseModelRevision ? { base_model_revision: options.baseModelRevision } : {}),
    n_epochs: String(hyper.n_epochs),
    learning_rate: String(hyper.learning_rate ?? model.defaultLearningRate),
    per_device_train_batch_size: String(hyper.batch_size ?? model.defaultPerDeviceBatchSize),
    gradient_accumulation_steps: String(
      hyper.gradient_accumulation_steps ?? model.defaultGradientAccumulationSteps,
    ),
    lora_rank: String(hyper.lora_rank ?? model.defaultLoraRank),
    lora_alpha: String(hyper.lora_alpha ?? model.defaultLoraAlpha),
    lora_dropout: String(hyper.lora_dropout ?? model.defaultLoraDropout),
    max_seq_length: String(hyper.max_seq_length ?? model.defaultMaxSeqLength),
  };
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

export interface TrainingProgressSnapshot {
  percent?: number;
  step?: number;
  total_steps?: number;
  elapsed?: string;
  eta?: string;
  rate?: string;
  loss?: number;
  grad_norm?: number;
  learning_rate?: number;
  epoch?: number;
}

function numberFromMetric(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function parseTrainingProgressLine(line: string): TrainingProgressSnapshot | null {
  // Transformers uses tqdm while loading checkpoint shards. Those counters
  // describe model startup, not optimizer steps, and previously appeared as
  // misleading "Training progress 320/320" events.
  if (/loading (?:checkpoint )?shards|loading weights/i.test(line)) return null;
  const snapshot: TrainingProgressSnapshot = {};
  const metricMatches = [...line.matchAll(/'([^']+)':\s*'([^']*)'/g)];
  for (const match of metricMatches) {
    const key = match[1];
    const value = numberFromMetric(match[2]);
    if (value === undefined) continue;
    if (key === "loss") snapshot.loss = value;
    if (key === "grad_norm") snapshot.grad_norm = value;
    if (key === "learning_rate") snapshot.learning_rate = value;
    if (key === "epoch") {
      snapshot.epoch = value;
      snapshot.percent = Math.max(0, Math.min(100, Math.round(value * 100)));
    }
  }

  const progress = line.match(/(\d+)%\|[^|]*\|\s*(\d+)\/(\d+)\s*\[([^<,\]]+)(?:<([^,\]]+))?,\s*([^\]]+)\]/);
  if (progress) {
    snapshot.percent = Number(progress[1]);
    snapshot.step = Number(progress[2]);
    snapshot.total_steps = Number(progress[3]);
    snapshot.elapsed = progress[4]?.trim();
    snapshot.eta = progress[5]?.trim();
    snapshot.rate = progress[6]?.trim();
  }

  return Object.keys(snapshot).length > 0 ? snapshot : null;
}

export function createTrainingProgressForwarder(reporter?: LocalRunReporter): (line: string) => void {
  let lastPercent = -1;
  let lastStep = -1;
  let lastEmittedAt = 0;
  let latestMetrics: TrainingProgressSnapshot = {};

  return (line: string) => {
    const parsed = parseTrainingProgressLine(line);
    if (!parsed) return;
    latestMetrics = { ...latestMetrics, ...parsed };
    const percent = latestMetrics.percent;
    const step = latestMetrics.step;
    const totalSteps = latestMetrics.total_steps;
    const now = Date.now();
    const shouldEmit = Boolean(reporter?.onEvent)
      && (
        (percent !== undefined && (percent >= lastPercent + 5 || percent === 100))
        || (step !== undefined && totalSteps !== undefined && step === totalSteps && step !== lastStep)
        || now - lastEmittedAt > 30_000
      );
    if (!shouldEmit) return;

    lastEmittedAt = now;
    if (percent !== undefined) lastPercent = percent;
    if (step !== undefined) lastStep = step;
    reportInBackground(() => reporter?.onEvent?.({
      stage: "training",
      status: "running",
      message: step !== undefined && totalSteps !== undefined
        ? `Training progress ${step}/${totalSteps}${percent !== undefined ? ` (${percent}%)` : ""}.`
        : `Training progress${percent !== undefined ? ` ${percent}%` : ""}.`,
      details: latestMetrics as Record<string, unknown>,
    }));
  };
}

export async function launchProcessTraining(args: {
  request: FineTuneRunRequest;
  artifacts: RunArtifacts;
  config: LocalRunnerConfig;
  baseModelRevision?: string;
  reporter?: LocalRunReporter;
  shouldCancel?: () => boolean | Promise<boolean>;
}): Promise<TrainingReport> {
  const { request, artifacts, config } = args;
  const jobName = `tt-local-${request.run_id}`;
  await mkdir(artifacts.trainingDir, { recursive: true });
  await copyDatasetToTrainingChannel(artifacts);
  const hyperparameters = buildTrainingHyperparameters(request, {
    baseModelRevision: args.baseModelRevision,
  });
  await writeJson(join(artifacts.trainingConfigDir, "hyperparameters.json"), hyperparameters);

  const entrypoint = buildBundledPythonCommand("train.py");

  if (config.dryRun) {
    await writeFile(
      artifacts.trainingLog,
      "Dry run enabled. Training process was not launched.\n",
      "utf8",
    );
    return {
      provider: "local-uv",
      training_job_name: jobName,
      model_artifact_uri: fileUri(artifacts.trainingModelDir),
      base_model_artifact_uri: config.paths.baseModel ? fileUri(config.paths.baseModel) : undefined,
      metrics: { dry_run: true },
      exit_code: 0,
      log_uri: fileUri(artifacts.trainingLog),
      command: entrypoint.displayCommand,
    };
  }

  const command = entrypoint.displayCommand;
  const inheritedEnv = minimalMachineLearningEnvironment(process.env);
  const childEnv: NodeJS.ProcessEnv = withBundledPythonEnvironment(
    withOfflineHuggingFaceCacheEnvironment({
      ...inheritedEnv,
      BACKEND: "local",
      SM_CHANNEL_TRAINING: resolve(artifacts.trainingInputDir),
      SM_CHANNEL_BASE_MODEL: config.paths.baseModel ? resolve(config.paths.baseModel) : undefined,
      SM_MODEL_DIR: resolve(artifacts.trainingModelDir),
      SM_OUTPUT_DIR: resolve(artifacts.trainingOutputDir),
      TT_HYPERPARAMETERS_PATH: resolve(artifacts.trainingConfigDir, "hyperparameters.json"),
    }, config.paths.modelCache),
  );
  for (const [key, value] of Object.entries(childEnv)) {
    if (value === undefined) delete childEnv[key];
  }

  if (config.paths.modelCache) await mkdir(resolve(config.paths.modelCache), { recursive: true });

  await args.reporter?.onEvent?.({
    stage: "training",
    status: "running",
    message: "Starting local training process.",
    details: {
      command,
      log_path: artifacts.trainingLog,
      model_dir: artifacts.trainingModelDir,
      output_dir: artifacts.trainingOutputDir,
    },
  });

  const forwardTrainingProgress = createTrainingProgressForwarder(args.reporter);
  const { exitCode } = await runLoggedProcess({
    command: entrypoint.command,
    commandArgs: entrypoint.commandArgs,
    env: childEnv,
    logPath: artifacts.trainingLog,
    reporter: args.reporter,
    stage: "training",
    onLine: (line) => forwardTrainingProgress(line),
    shouldCancel: args.shouldCancel,
  });

  const metricsPath = join(artifacts.trainingModelDir, "training-metrics.json");
  const metrics = await pathExists(metricsPath)
    ? JSON.parse(await readFile(metricsPath, "utf8")) as Record<string, unknown>
    : null;

  if (exitCode !== 0) {
    throw new Error(`Training process exited with code ${exitCode}. See ${artifacts.trainingLog}.`);
  }

  const modelTarPath = join(artifacts.trainingOutputDir, "model.tar.gz");
  const modelUri = await pathExists(modelTarPath)
    ? fileUri(modelTarPath)
    : fileUri(artifacts.trainingModelDir);
  await args.reporter?.onEvent?.({
    stage: "training",
    status: "running",
    message: "Training process completed.",
    details: {
      exit_code: exitCode,
      model_artifact_uri: modelUri,
      log_path: artifacts.trainingLog,
      metrics,
    },
  });

  return {
    provider: "local-uv",
    training_job_name: jobName,
    model_artifact_uri: modelUri,
    base_model_artifact_uri: config.paths.baseModel ? fileUri(config.paths.baseModel) : undefined,
    metrics,
    exit_code: exitCode,
    log_uri: fileUri(artifacts.trainingLog),
    command,
  };
}
