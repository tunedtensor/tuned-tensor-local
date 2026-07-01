import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { access, copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import type { FineTuneRunRequest, LocalRunnerConfig, TrainingReport } from "./contracts.js";
import type { RunArtifacts } from "./artifacts.js";
import { copyDatasetToTrainingChannel, fileUri, writeJson } from "./artifacts.js";
import { resolveTrainingModel } from "./model-registry.js";
import { forwardStreamLines, type LocalRunReporter } from "./run-reporter.js";

export function buildTrainingHyperparameters(request: FineTuneRunRequest): Record<string, string> {
  const model = resolveTrainingModel(request.spec_snapshot.base_model);
  const hyper = request.hyperparameters;
  return {
    run_id: request.run_id,
    base_model: model.id,
    model_family: model.family,
    model_loader: model.loader,
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
    save_adapter_only: String(hyper.save_adapter_only),
    requires_hf_token: String(model.requiresHfToken),
    trust_remote_code: String(model.trustRemoteCode),
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

function buildUvArgs(training: LocalRunnerConfig["training"]): string[] {
  const args: string[] = ["run"];
  if (training.project) args.push("--project", training.project);
  if (training.with?.length) {
    for (const dependency of training.with) args.push("--with", dependency);
  }
  args.push("python");
  if (training.module) {
    args.push("-m", training.module);
  } else if (training.script) {
    args.push(training.script);
  } else {
    args.push("training/sft-local/src/train.py");
  }
  args.push(...training.args);
  return args;
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

function createTrainingProgressForwarder(reporter?: LocalRunReporter): (line: string) => void {
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
    void reporter?.onEvent?.({
      stage: "training",
      status: "running",
      message: step !== undefined && totalSteps !== undefined
        ? `Training progress ${step}/${totalSteps}${percent !== undefined ? ` (${percent}%)` : ""}.`
        : `Training progress${percent !== undefined ? ` ${percent}%` : ""}.`,
      details: latestMetrics as Record<string, unknown>,
    });
  };
}

export async function launchProcessTraining(args: {
  request: FineTuneRunRequest;
  artifacts: RunArtifacts;
  config: LocalRunnerConfig;
  reporter?: LocalRunReporter;
}): Promise<TrainingReport> {
  const { request, artifacts, config } = args;
  const jobName = `tt-local-${request.run_id}`;
  await mkdir(artifacts.trainingDir, { recursive: true });
  await copyDatasetToTrainingChannel(artifacts);
  const hyperparameters = buildTrainingHyperparameters(request);
  await writeJson(join(artifacts.trainingConfigDir, "hyperparameters.json"), hyperparameters);

  if (config.dryRun) {
    await writeFile(
      artifacts.trainingLog,
      "Dry run enabled. uv training process was not launched.\n",
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
      command: ["uv", ...buildUvArgs(config.training)],
    };
  }

  const uvArgs = buildUvArgs(config.training);
  const command = ["uv", ...uvArgs];
  const logStream = createWriteStream(artifacts.trainingLog, { flags: "w" });
  const childEnv: NodeJS.ProcessEnv = {
    ...process.env,
    ...config.training.env,
    BACKEND: "local",
    SM_CHANNEL_TRAINING: resolve(artifacts.trainingInputDir),
    SM_CHANNEL_BASE_MODEL: config.paths.baseModel ? resolve(config.paths.baseModel) : undefined,
    SM_MODEL_DIR: resolve(artifacts.trainingModelDir),
    SM_OUTPUT_DIR: resolve(artifacts.trainingOutputDir),
    TT_HYPERPARAMETERS_PATH: resolve(artifacts.trainingConfigDir, "hyperparameters.json"),
    HF_HOME: config.paths.modelCache ? resolve(config.paths.modelCache) : process.env.HF_HOME,
  };
  for (const [key, value] of Object.entries(childEnv)) {
    if (value === undefined) delete childEnv[key];
  }

  if (config.paths.modelCache) await mkdir(config.paths.modelCache, { recursive: true });

  await args.reporter?.onEvent?.({
    stage: "training",
    status: "running",
    message: "Starting uv training process.",
    details: {
      command,
      log_path: artifacts.trainingLog,
      model_dir: artifacts.trainingModelDir,
      output_dir: artifacts.trainingOutputDir,
    },
  });

  let exitCode = 1;
  try {
    exitCode = await new Promise<number>((resolvePromise, reject) => {
      const child = spawn("uv", uvArgs, {
        cwd: config.training.cwd ? resolve(config.training.cwd) : process.cwd(),
        stdio: ["ignore", "pipe", "pipe"],
        env: childEnv,
      });
      const forwardTrainingProgress = createTrainingProgressForwarder(args.reporter);
      child.stdout.pipe(logStream, { end: false });
      child.stderr.pipe(logStream, { end: false });
      forwardStreamLines(child.stdout, (line) => {
        forwardTrainingProgress(line);
        if (args.reporter?.verbose) void args.reporter.onLog?.({ stage: "training", stream: "stdout", message: line });
      });
      forwardStreamLines(child.stderr, (line) => {
        forwardTrainingProgress(line);
        if (args.reporter?.verbose) void args.reporter.onLog?.({ stage: "training", stream: "stderr", message: line });
      });
      child.on("error", reject);
      child.on("close", (code) => resolvePromise(code ?? 1));
    });
  } finally {
    await new Promise<void>((resolveEnd) => {
      logStream.end(resolveEnd);
    });
  }

  const metricsPath = join(artifacts.trainingModelDir, "training-metrics.json");
  const metrics = await pathExists(metricsPath)
    ? JSON.parse(await readFile(metricsPath, "utf8")) as Record<string, unknown>
    : null;

  if (exitCode !== 0) {
    throw new Error(`uv training exited with code ${exitCode}. See ${artifacts.trainingLog}.`);
  }

  const modelTarPath = join(artifacts.trainingOutputDir, "model.tar.gz");
  const modelUri = await pathExists(modelTarPath)
    ? fileUri(modelTarPath)
    : fileUri(artifacts.trainingModelDir);
  if (await pathExists(modelTarPath)) {
    await copyFile(modelTarPath, join(artifacts.runDir, "model.tar.gz")).catch(() => undefined);
  }

  await args.reporter?.onEvent?.({
    stage: "training",
    status: "running",
    message: "uv training process completed.",
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
