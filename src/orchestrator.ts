import { copyFile, readFile, rm, stat, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";
import { dirname, resolve } from "node:path";
import {
  defaultArtifactPrefix,
  fileUri,
  prepareRunDirectories,
  readJson,
  resolveRunArtifacts,
  writeJson,
  type RunArtifacts,
} from "./artifacts.js";
import {
  evalReportSchema,
  fineTuneRunRequestSchema,
  localRunnerConfigSchema,
  runReportSchema,
  trainingReportSchema,
  type BehaviorSpecExample,
  type EvalReport,
  type EvalSplit,
  type FineTuneRunRequest,
  type LocalRunnerConfig,
  type RunReport,
  type TrainingReport,
} from "./contracts.js";
import { buildSystemMessage, compileSpecToJsonl, examplesFromChatJsonl, examplesFromSpec } from "./dataset.js";
import { compareEvalReports, deriveSampleSeed, evaluateExamples, rescoreEvalReport, splitSpecExamples } from "./evaluation.js";
import { launchProcessTraining } from "./process-training.js";
import type { LocalRunReporter } from "./run-reporter.js";
import { createLocalStore, type LocalRunStatus, type LocalStore } from "./store.js";

export type LocalRunStage = "prepare" | "baseline" | "train" | "candidate" | "score" | "report" | "all";

export interface LocalRunResult {
  request: FineTuneRunRequest;
  report: RunReport;
  reportPath: string;
  artifactDir: string;
}

export interface LocalStageRunResult {
  request: FineTuneRunRequest;
  stage: LocalRunStage;
  report?: RunReport;
  reportPath?: string;
  artifactDir: string;
  artifacts: {
    training_jsonl: string;
    stage_metadata: string;
    training_report: string;
    baseline_eval: string;
    candidate_eval: string;
    report: string;
  };
}

interface StageMetadata {
  run_id: string;
  behavior_spec_id: string;
  user_id: string;
  source_fingerprint: string;
  eval_split: EvalSplit;
  eval_sample_seed: number;
  eval_examples_total: number;
  eval_examples_used: number;
  max_eval_examples: number | null;
  training_example_count: number | null;
  dataset_prebuilt: boolean;
  dataset_uri: string;
  base_model_for_evaluation: string;
  system_prompt_sha256: string;
  prepared_at: string;
}

interface PreparedRun {
  request: FineTuneRunRequest;
  artifacts: RunArtifacts;
  metadata: StageMetadata;
  examples: BehaviorSpecExample[];
  system: string;
  baseModelForEvaluation: string;
  maxEvalExamples?: number;
}

export async function loadJsonFile<T>(path: string): Promise<T> {
  return JSON.parse(await readFile(path, "utf8")) as T;
}

export async function loadRunRequest(path: string): Promise<FineTuneRunRequest> {
  return fineTuneRunRequestSchema.parse(await loadJsonFile<unknown>(path));
}

export async function loadLocalRunnerConfig(path?: string): Promise<LocalRunnerConfig> {
  if (!path) return localRunnerConfigSchema.parse({});
  return localRunnerConfigSchema.parse(await loadJsonFile<unknown>(path));
}

function elapsed(started: number): { ms: number; seconds: number } {
  const ms = Math.max(0, Math.round(performance.now() - started));
  return { ms, seconds: Math.round((ms / 1000) * 1000) / 1000 };
}

function stripFileUri(path: string): string {
  return path.replace(/^file:\/\//, "");
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

function selectPrebuiltEvaluation(dataset: FineTuneRunRequest["dataset_prebuilt"]): {
  path: string;
  split: EvalSplit;
} {
  if (!dataset) throw new Error("dataset_prebuilt is required");
  if (dataset.test) return { path: stripFileUri(dataset.test), split: "prebuilt_test" };
  if (dataset.validation) return { path: stripFileUri(dataset.validation), split: "prebuilt_validation" };
  return { path: stripFileUri(dataset.training), split: "prebuilt_training" };
}

function artifactPrefix(request: FineTuneRunRequest): string {
  return request.artifacts?.prefix ?? defaultArtifactPrefix({
    userId: request.user_id,
    behaviorSpecId: request.behavior_spec_id,
    runId: request.run_id,
  });
}

function hashJson(value: unknown): string {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

function preparedSourceFingerprint(args: {
  request: FineTuneRunRequest;
  baseModelForEvaluation: string;
  evalSampleSeed: number;
  maxEvalExamples?: number;
}): string {
  return hashJson({
    spec_snapshot: args.request.spec_snapshot,
    dataset_prebuilt: args.request.dataset_prebuilt ?? null,
    base_model_for_evaluation: args.baseModelForEvaluation,
    eval_sample_seed: args.evalSampleSeed,
    max_eval_examples: args.maxEvalExamples ?? null,
  });
}

function statusForProgressStage(stage: string): LocalRunStatus {
  if (stage === "evaluating_baseline") return "evaluating_baseline";
  if (stage === "evaluating_candidate") return "evaluating_candidate";
  if (stage === "training") return "training";
  if (stage === "preparing") return "preparing";
  if (stage === "scoring") return "scoring";
  if (stage === "reporting") return "reporting";
  return "training";
}

async function readStageMetadata(path: string): Promise<StageMetadata | null> {
  try {
    const metadata = await readJson<Partial<StageMetadata>>(path);
    return typeof metadata.source_fingerprint === "string"
      ? metadata as StageMetadata
      : null;
  } catch {
    return null;
  }
}

async function clearDependentStageArtifacts(artifacts: RunArtifacts): Promise<void> {
  await Promise.all([
    rm(artifacts.baselineEvalJson, { force: true }),
    rm(artifacts.candidateEvalJson, { force: true }),
    rm(artifacts.trainingReportJson, { force: true }),
    rm(artifacts.runReportJson, { force: true }),
  ]);
}

function createStoreReporter(input: {
  request: FineTuneRunRequest;
  store: LocalStore;
  reporter?: LocalRunReporter;
}): LocalRunReporter {
  return {
    verbose: input.reporter?.verbose,
    async onEvent(event) {
      await input.store.updateRun({
        runId: input.request.run_id,
        status: statusForProgressStage(event.stage),
        stage: event.stage,
        message: event.message,
        details: event.details,
      });
      await input.reporter?.onEvent?.(event);
    },
    async onLog(log) {
      await input.reporter?.onLog?.(log);
    },
  };
}

async function updateRun(input: {
  store: LocalStore;
  reporter?: LocalRunReporter;
  request: FineTuneRunRequest;
  status: LocalRunStatus;
  stage: string;
  message: string;
  details?: Record<string, unknown>;
}) {
  const state = await input.store.updateRun({
    runId: input.request.run_id,
    status: input.status,
    stage: input.stage,
    message: input.message,
    details: input.details,
  });
  await input.reporter?.onEvent?.({
    stage: input.stage,
    status: input.status,
    message: input.message,
    details: input.details,
  });
  return state;
}

async function ensureRunRecord(args: {
  request: FineTuneRunRequest;
  artifacts: RunArtifacts;
  store: LocalStore;
  reporter?: LocalRunReporter;
}): Promise<void> {
  await prepareRunDirectories(args.artifacts);
  await writeJson(resolve(args.artifacts.runDir, "request.json"), args.request);
  try {
    await args.store.getRun(args.request.run_id);
  } catch {
    await args.store.startRun({ request: args.request, artifactDir: args.artifacts.runDir });
    await args.reporter?.onEvent?.({
      stage: "queued",
      status: "queued",
      message: "Run queued.",
      details: { run_id: args.request.run_id, artifact_dir: args.artifacts.runDir },
    });
  }
}

async function computePreparedRun(args: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  artifacts: RunArtifacts;
  writeArtifacts: boolean;
}): Promise<PreparedRun> {
  const { request, config, artifacts } = args;
  const evalSampleSeed = config.evaluation.sampleSeed ?? deriveSampleSeed(request.run_id);
  let examples = examplesFromSpec(request.spec_snapshot);
  let evalSplit: EvalSplit = "spec_examples";
  let trainingExampleCount: number | null = examples.length;

  if (request.dataset_prebuilt) {
    const trainingPath = stripFileUri(request.dataset_prebuilt.training);
    const evaluation = selectPrebuiltEvaluation(request.dataset_prebuilt);
    evalSplit = evaluation.split;
    if (evalSplit === "prebuilt_training" && !config.dryRun && !config.evaluation.allowPrebuiltTrainingEval) {
      throw new Error(
        "dataset_prebuilt has no test or validation split, so evaluation would run on the training data and "
        + "overstate improvement. Provide dataset_prebuilt.test or dataset_prebuilt.validation, or set "
        + "evaluation.allowPrebuiltTrainingEval=true to evaluate on the training split anyway.",
      );
    }
    if (args.writeArtifacts) await copyFile(trainingPath, artifacts.trainingJsonl);
    examples = await examplesFromChatJsonl(evaluation.path);
    trainingExampleCount = null;
  } else {
    const split = splitSpecExamples(request.spec_snapshot.examples, evalSampleSeed);
    let trainingExamples = request.spec_snapshot.examples;
    if (split.holdout.length > 0) {
      trainingExamples = split.train;
      examples = split.holdout;
      evalSplit = "spec_holdout";
    }
    trainingExampleCount = trainingExamples.length;
    if (args.writeArtifacts) {
      const jsonl = compileSpecToJsonl({ ...request.spec_snapshot, examples: trainingExamples });
      await writeFile(artifacts.trainingJsonl, `${jsonl}\n`, "utf8");
    }
  }

  const system = buildSystemMessage(request.spec_snapshot);
  const baseModelForEvaluation = config.paths.baseModel ?? request.spec_snapshot.base_model;
  const maxEvalExamples = config.evaluation.maxExamples ?? request.hyperparameters.max_eval_examples;
  const evalExamplesUsed = Math.min(maxEvalExamples ?? examples.length, examples.length);
  const metadata: StageMetadata = {
    run_id: request.run_id,
    behavior_spec_id: request.behavior_spec_id,
    user_id: request.user_id,
    source_fingerprint: preparedSourceFingerprint({
      request,
      baseModelForEvaluation,
      evalSampleSeed,
      maxEvalExamples,
    }),
    eval_split: evalSplit,
    eval_sample_seed: evalSampleSeed,
    eval_examples_total: examples.length,
    eval_examples_used: evalExamplesUsed,
    max_eval_examples: maxEvalExamples ?? null,
    training_example_count: trainingExampleCount,
    dataset_prebuilt: Boolean(request.dataset_prebuilt),
    dataset_uri: fileUri(artifacts.trainingJsonl),
    base_model_for_evaluation: baseModelForEvaluation,
    system_prompt_sha256: createHash("sha256").update(system).digest("hex"),
    prepared_at: new Date().toISOString(),
  };
  if (args.writeArtifacts) await writeJson(artifacts.stageMetadataJson, metadata);
  return {
    request,
    artifacts,
    metadata,
    examples,
    system,
    baseModelForEvaluation,
    maxEvalExamples,
  };
}

async function prepareStage(args: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  artifacts: RunArtifacts;
  store: LocalStore;
  reporter?: LocalRunReporter;
  force?: boolean;
}): Promise<PreparedRun> {
  const preparedExists = await pathExists(args.artifacts.stageMetadataJson)
    && await pathExists(args.artifacts.trainingJsonl);
  const prepared = await computePreparedRun({ ...args, writeArtifacts: false });
  const existingMetadata = preparedExists
    ? await readStageMetadata(args.artifacts.stageMetadataJson)
    : null;
  const canReuse = preparedExists
    && !args.force
    && existingMetadata?.source_fingerprint === prepared.metadata.source_fingerprint;
  if (canReuse) {
    await updateRun({
      ...args,
      status: "preparing",
      stage: "preparing",
      message: "Reusing prepared local run artifacts.",
      details: { artifact_dir: args.artifacts.runDir },
    });
    return prepared;
  }

  await updateRun({
    ...args,
    status: "preparing",
    stage: "preparing",
    message: "Preparing local run artifacts.",
    details: { artifact_dir: args.artifacts.runDir },
  });
  await clearDependentStageArtifacts(args.artifacts);
  return computePreparedRun({ ...args, writeArtifacts: true });
}

async function ensurePrepared(args: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  artifacts: RunArtifacts;
  store: LocalStore;
  reporter?: LocalRunReporter;
  forcePrepare?: boolean;
}): Promise<PreparedRun> {
  return prepareStage({
    request: args.request,
    config: args.config,
    artifacts: args.artifacts,
    store: args.store,
    reporter: args.reporter,
    force: args.forcePrepare,
  });
}

async function runBaselineStage(args: {
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  store: LocalStore;
  reporter?: LocalRunReporter;
  runReporter: LocalRunReporter;
  force?: boolean;
}): Promise<EvalReport> {
  if (!args.force && await pathExists(args.prepared.artifacts.baselineEvalJson)) {
    await updateRun({
      store: args.store,
      reporter: args.reporter,
      request: args.prepared.request,
      status: "evaluating_baseline",
      stage: "evaluating_baseline",
      message: "Reusing existing baseline evaluation.",
      details: { path: args.prepared.artifacts.baselineEvalJson },
    });
    return evalReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.baselineEvalJson));
  }
  await updateRun({
    store: args.store,
    reporter: args.reporter,
    request: args.prepared.request,
    status: "evaluating_baseline",
    stage: "evaluating_baseline",
    message: "Running baseline evaluation.",
    details: {
      examples: args.prepared.examples.length,
      eval_examples_used: args.prepared.metadata.eval_examples_used,
      eval_split: args.prepared.metadata.eval_split,
      model_id: args.prepared.baseModelForEvaluation,
    },
  });
  return evaluateExamples({
    kind: "baseline",
    modelId: args.prepared.baseModelForEvaluation,
    baseModelId: args.prepared.baseModelForEvaluation,
    examples: args.prepared.examples,
    system: args.prepared.system,
    config: args.config,
    outputPath: args.prepared.artifacts.baselineEvalJson,
    reporter: args.runReporter,
    maxExamples: args.prepared.maxEvalExamples,
    evalSplit: args.prepared.metadata.eval_split,
    sampleSeed: args.prepared.metadata.eval_sample_seed,
  });
}

async function runTrainStage(args: {
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  store: LocalStore;
  reporter?: LocalRunReporter;
  runReporter: LocalRunReporter;
  force?: boolean;
}): Promise<TrainingReport> {
  if (!args.force && await pathExists(args.prepared.artifacts.trainingReportJson)) {
    await updateRun({
      store: args.store,
      reporter: args.reporter,
      request: args.prepared.request,
      status: "training",
      stage: "training",
      message: "Reusing existing training result.",
      details: { path: args.prepared.artifacts.trainingReportJson },
    });
    return trainingReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.trainingReportJson));
  }
  await updateRun({
    store: args.store,
    reporter: args.reporter,
    request: args.prepared.request,
    status: "training",
    stage: "training",
    message: args.config.dryRun ? "Recording dry-run training result." : "Launching local training process.",
    details: { training_backend: args.config.training.backend, dry_run: args.config.dryRun },
  });
  const training = await launchProcessTraining({
    request: args.prepared.request,
    artifacts: args.prepared.artifacts,
    config: args.config,
    reporter: args.runReporter,
  });
  await writeJson(args.prepared.artifacts.trainingReportJson, training);
  return training;
}

async function writeExternalTrainingReport(args: {
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  modelArtifact: string;
}): Promise<TrainingReport> {
  const training = trainingReportSchema.parse({
    provider: args.config.training.backend === "command" ? "local-command" : "local-uv",
    training_job_name: `external-${args.prepared.request.run_id}`,
    model_artifact_uri: args.modelArtifact,
    base_model_artifact_uri: args.config.paths.baseModel ? fileUri(args.config.paths.baseModel) : undefined,
    artifact_metadata: {
      ...(args.config.training.artifact ?? {}),
      notes: args.config.training.artifact?.notes ?? "External model artifact supplied with --model-artifact.",
    },
    metrics: { external_model_artifact: true },
    exit_code: null,
    log_uri: fileUri(args.prepared.artifacts.trainingReportJson),
  });
  await writeJson(args.prepared.artifacts.trainingReportJson, training);
  return training;
}

async function runCandidateStage(args: {
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  store: LocalStore;
  reporter?: LocalRunReporter;
  runReporter: LocalRunReporter;
  force?: boolean;
  modelArtifact?: string;
}): Promise<EvalReport> {
  if (!args.force && !args.modelArtifact && await pathExists(args.prepared.artifacts.candidateEvalJson)) {
    await updateRun({
      store: args.store,
      reporter: args.reporter,
      request: args.prepared.request,
      status: "evaluating_candidate",
      stage: "evaluating_candidate",
      message: "Reusing existing candidate evaluation.",
      details: { path: args.prepared.artifacts.candidateEvalJson },
    });
    return evalReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.candidateEvalJson));
  }
  let training: TrainingReport;
  if (args.modelArtifact) {
    training = await writeExternalTrainingReport({
      prepared: args.prepared,
      config: args.config,
      modelArtifact: args.modelArtifact,
    });
  } else if (await pathExists(args.prepared.artifacts.trainingReportJson)) {
    training = trainingReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.trainingReportJson));
  } else {
    throw new Error("candidate stage requires training output or --model-artifact.");
  }
  const modelArtifact = training.model_artifact_uri;
  if (!modelArtifact) throw new Error("candidate stage requires a model_artifact_uri in training-report.json or --model-artifact.");
  await updateRun({
    store: args.store,
    reporter: args.reporter,
    request: args.prepared.request,
    status: "evaluating_candidate",
    stage: "evaluating_candidate",
    message: "Running candidate evaluation.",
    details: { model_artifact_uri: modelArtifact },
  });
  return evaluateExamples({
    kind: "candidate",
    modelId: modelArtifact,
    baseModelId: args.prepared.baseModelForEvaluation,
    adapterPath: modelArtifact,
    examples: args.prepared.examples,
    system: args.prepared.system,
    config: args.config,
    outputPath: args.prepared.artifacts.candidateEvalJson,
    reporter: args.runReporter,
    maxExamples: args.prepared.maxEvalExamples,
    evalSplit: args.prepared.metadata.eval_split,
    sampleSeed: args.prepared.metadata.eval_sample_seed,
  });
}

async function runScoreStage(args: {
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  store: LocalStore;
  reporter?: LocalRunReporter;
  runReporter: LocalRunReporter;
}): Promise<{ baseline: EvalReport; candidate: EvalReport }> {
  if (!await pathExists(args.prepared.artifacts.baselineEvalJson)) {
    throw new Error("score stage requires baseline-eval.json. Run --stage baseline first.");
  }
  if (!await pathExists(args.prepared.artifacts.candidateEvalJson)) {
    throw new Error("score stage requires candidate-eval.json. Run --stage candidate first.");
  }
  await updateRun({
    store: args.store,
    reporter: args.reporter,
    request: args.prepared.request,
    status: "scoring",
    stage: "scoring",
    message: "Rescoring existing baseline and candidate outputs.",
    details: { scoring_mode: args.config.evaluation.scoring.mode },
  });
  const baseline = await rescoreEvalReport({
    report: evalReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.baselineEvalJson)),
    config: args.config,
    outputPath: args.prepared.artifacts.baselineEvalJson,
    system: args.prepared.system,
    reporter: args.runReporter,
  });
  const candidate = await rescoreEvalReport({
    report: evalReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.candidateEvalJson)),
    config: args.config,
    outputPath: args.prepared.artifacts.candidateEvalJson,
    system: args.prepared.system,
    reporter: args.runReporter,
  });
  return { baseline, candidate };
}

async function runReportStage(args: {
  prepared: PreparedRun;
  store: LocalStore;
  reporter?: LocalRunReporter;
  startedAt: string;
  startedPerf: number;
  emitReportingEvent?: boolean;
}): Promise<RunReport> {
  if (!await pathExists(args.prepared.artifacts.baselineEvalJson)) {
    throw new Error("report stage requires baseline-eval.json. Run --stage baseline first.");
  }
  if (!await pathExists(args.prepared.artifacts.candidateEvalJson)) {
    throw new Error("report stage requires candidate-eval.json. Run --stage candidate first.");
  }
  if (!await pathExists(args.prepared.artifacts.trainingReportJson)) {
    throw new Error("report stage requires training-report.json. Run --stage train first, or --stage candidate --model-artifact <path>.");
  }
  if (args.emitReportingEvent) {
    await updateRun({
      store: args.store,
      reporter: args.reporter,
      request: args.prepared.request,
      status: "reporting",
      stage: "reporting",
      message: "Writing run report.",
      details: { report_path: args.prepared.artifacts.runReportJson },
    });
  }
  const baseline = evalReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.baselineEvalJson));
  const candidate = evalReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.candidateEvalJson));
  const training = trainingReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.trainingReportJson));
  const comparison = compareEvalReports(baseline, candidate);
  const completedAt = new Date().toISOString();
  const duration = elapsed(args.startedPerf);
  const report = runReportSchema.parse({
    run_id: args.prepared.request.run_id,
    behavior_spec_id: args.prepared.request.behavior_spec_id,
    user_id: args.prepared.request.user_id,
    run_number: args.prepared.request.run_number,
    base_model: args.prepared.request.spec_snapshot.base_model,
    fine_tuned_model_id: training.model_artifact_uri ?? training.training_job_name,
    status: "completed",
    baseline,
    candidate,
    comparison,
    training,
    artifact_uris: {
      dataset: fileUri(args.prepared.artifacts.trainingJsonl),
      baseline_eval: fileUri(args.prepared.artifacts.baselineEvalJson),
      candidate_eval: fileUri(args.prepared.artifacts.candidateEvalJson),
      report: fileUri(args.prepared.artifacts.runReportJson),
    },
    run_metadata: {
      base_model: args.prepared.request.spec_snapshot.base_model,
      fine_tuned_model_id: training.model_artifact_uri ?? training.training_job_name,
      dataset_prebuilt: args.prepared.metadata.dataset_prebuilt,
      dataset_uri: fileUri(args.prepared.artifacts.trainingJsonl),
      spec_example_count: args.prepared.request.spec_snapshot.examples.length,
      training_example_count: args.prepared.metadata.training_example_count,
      eval_examples_total: baseline.eval_examples_total,
      eval_examples_used: baseline.eval_examples_used,
      eval_split: baseline.eval_split,
      eval_sample_seed: baseline.eval_sample_seed ?? null,
      started_at: args.startedAt,
      completed_at: completedAt,
      elapsed_ms: duration.ms,
      elapsed_seconds: duration.seconds,
    },
    created_at: completedAt,
  });
  await writeJson(args.prepared.artifacts.runReportJson, report);
  await args.store.completeRun(report, args.prepared.artifacts.runDir, args.prepared.artifacts.runReportJson);
  await args.reporter?.onEvent?.({
    stage: "completed",
    status: "completed",
    message: "Run completed successfully.",
    details: {
      report_path: args.prepared.artifacts.runReportJson,
      model_id: `local-${args.prepared.request.run_id}`,
      avg_score_delta: comparison.avg_score_delta,
      elapsed_seconds: duration.seconds,
    },
  });
  return report;
}

export async function runLocalFineTuneStage(input: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  reporter?: LocalRunReporter;
  stage?: LocalRunStage;
  force?: boolean;
  modelArtifact?: string;
}): Promise<LocalStageRunResult> {
  const startedPerf = performance.now();
  const startedAt = new Date().toISOString();
  const stage = input.stage ?? "all";
  const prefix = artifactPrefix(input.request);
  const artifacts = resolveRunArtifacts({ artifactRoot: input.config.artifactRoot, prefix });
  const store = createLocalStore(input.config.storeRoot);
  await ensureRunRecord({ request: input.request, artifacts, store, reporter: input.reporter });
  const runReporter = createStoreReporter({ request: input.request, store, reporter: input.reporter });

  try {
    const prepared = await ensurePrepared({
      request: input.request,
      config: input.config,
      artifacts,
      store,
      reporter: input.reporter,
      forcePrepare: Boolean(input.force && (stage === "prepare" || stage === "all")),
    });
    let report: RunReport | undefined;

    if (stage === "prepare") {
      return stageResult({ request: input.request, stage, artifacts });
    }

    if (stage === "baseline" || stage === "all") {
      await runBaselineStage({
        prepared,
        config: input.config,
        store,
        reporter: input.reporter,
        runReporter,
        force: Boolean(input.force),
      });
      if (stage === "baseline") return stageResult({ request: input.request, stage, artifacts });
    }

    if (stage === "train" || stage === "all") {
      await runTrainStage({
        prepared,
        config: input.config,
        store,
        reporter: input.reporter,
        runReporter,
        force: Boolean(input.force),
      });
      if (stage === "train") return stageResult({ request: input.request, stage, artifacts });
    }

    if (stage === "candidate" || stage === "all") {
      await runCandidateStage({
        prepared,
        config: input.config,
        store,
        reporter: input.reporter,
        runReporter,
        force: Boolean(input.force),
        modelArtifact: input.modelArtifact,
      });
      if (stage === "candidate") return stageResult({ request: input.request, stage, artifacts });
    }

    if (stage === "score") {
      await runScoreStage({
        prepared,
        config: input.config,
        store,
        reporter: input.reporter,
        runReporter,
      });
      return stageResult({ request: input.request, stage, artifacts });
    }

    if (stage === "report" || stage === "all") {
      report = await runReportStage({
        prepared,
        store,
        reporter: input.reporter,
        startedAt,
        startedPerf,
        emitReportingEvent: stage === "report",
      });
      return stageResult({ request: input.request, stage, artifacts, report });
    }

    throw new Error(`Unknown run stage: ${stage}`);
  } catch (error) {
    await store.failRun(input.request.run_id, error instanceof Error ? error.message : String(error)).catch(() => undefined);
    await input.reporter?.onEvent?.({
      stage: "failed",
      status: "failed",
      message: error instanceof Error ? error.message : String(error),
    });
    throw error;
  }
}

function stageResult(args: {
  request: FineTuneRunRequest;
  stage: LocalRunStage;
  artifacts: RunArtifacts;
  report?: RunReport;
}): LocalStageRunResult {
  return {
    request: args.request,
    stage: args.stage,
    report: args.report,
    reportPath: args.report ? args.artifacts.runReportJson : undefined,
    artifactDir: dirname(args.artifacts.runReportJson),
    artifacts: {
      training_jsonl: args.artifacts.trainingJsonl,
      stage_metadata: args.artifacts.stageMetadataJson,
      training_report: args.artifacts.trainingReportJson,
      baseline_eval: args.artifacts.baselineEvalJson,
      candidate_eval: args.artifacts.candidateEvalJson,
      report: args.artifacts.runReportJson,
    },
  };
}

export async function runLocalFineTune(input: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  reporter?: LocalRunReporter;
}): Promise<LocalRunResult> {
  const result = await runLocalFineTuneStage({
    request: input.request,
    config: input.config,
    reporter: input.reporter,
    stage: "all",
  });
  if (!result.report || !result.reportPath) {
    throw new Error("Full local fine-tune run did not produce a report.");
  }
  return {
    request: result.request,
    report: result.report,
    reportPath: result.reportPath,
    artifactDir: result.artifactDir,
  };
}
