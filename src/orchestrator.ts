import { copyFile, readFile, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { dirname, resolve } from "node:path";
import {
  defaultArtifactPrefix,
  fileUri,
  prepareRunDirectories,
  resolveRunArtifacts,
  writeJson,
} from "./artifacts.js";
import {
  fineTuneRunRequestSchema,
  localRunnerConfigSchema,
  runReportSchema,
  type FineTuneRunRequest,
  type LocalRunnerConfig,
  type RunReport,
} from "./contracts.js";
import { buildSystemMessage, compileSpecToJsonl, examplesFromChatJsonl, examplesFromSpec } from "./dataset.js";
import { compareEvalReports, evaluateExamples } from "./evaluation.js";
import { launchProcessTraining } from "./process-training.js";
import type { LocalRunReporter } from "./run-reporter.js";
import { createLocalStore, type LocalRunStatus } from "./store.js";

export interface LocalRunResult {
  request: FineTuneRunRequest;
  report: RunReport;
  reportPath: string;
  artifactDir: string;
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

function selectPrebuiltEvaluationPath(dataset: FineTuneRunRequest["dataset_prebuilt"]): string {
  if (!dataset) throw new Error("dataset_prebuilt is required");
  return stripFileUri(dataset.test ?? dataset.validation ?? dataset.training);
}

export async function runLocalFineTune(input: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  reporter?: LocalRunReporter;
}): Promise<LocalRunResult> {
  const startedPerf = performance.now();
  const startedAt = new Date().toISOString();
  const { request, config, reporter } = input;
  const prefix = request.artifacts?.prefix ?? defaultArtifactPrefix({
    userId: request.user_id,
    behaviorSpecId: request.behavior_spec_id,
    runId: request.run_id,
  });
  const artifacts = resolveRunArtifacts({ artifactRoot: config.artifactRoot, prefix });
  const store = createLocalStore(config.storeRoot);
  await prepareRunDirectories(artifacts);
  await writeJson(resolve(artifacts.runDir, "request.json"), request);
  await store.startRun({ request, artifactDir: artifacts.runDir });

  try {
    async function updateRun(args: {
      status: LocalRunStatus;
      stage: string;
      message: string;
      details?: Record<string, unknown>;
    }) {
      const state = await store.updateRun({ runId: request.run_id, ...args });
      await reporter?.onEvent?.({
        stage: args.stage,
        status: args.status,
        message: args.message,
        details: args.details,
      });
      return state;
    }

    function statusForProgressStage(stage: string): LocalRunStatus {
      if (stage === "evaluating_baseline") return "evaluating_baseline";
      if (stage === "evaluating_candidate") return "evaluating_candidate";
      if (stage === "training") return "training";
      if (stage === "preparing") return "preparing";
      return "training";
    }

    const runReporter: LocalRunReporter = {
      verbose: reporter?.verbose,
      async onEvent(event) {
        await store.updateRun({
          runId: request.run_id,
          status: statusForProgressStage(event.stage),
          stage: event.stage,
          message: event.message,
          details: event.details,
        });
        await reporter?.onEvent?.(event);
      },
      async onLog(log) {
        await reporter?.onLog?.(log);
      },
    };

    await reporter?.onEvent?.({
      stage: "queued",
      status: "queued",
      message: "Run queued.",
      details: { run_id: request.run_id, artifact_dir: artifacts.runDir },
    });

    await updateRun({
      status: "preparing",
      stage: "preparing",
      message: "Preparing local run artifacts.",
      details: { artifact_dir: artifacts.runDir },
    });

    let examples = examplesFromSpec(request.spec_snapshot);
    if (request.dataset_prebuilt) {
      const trainingPath = stripFileUri(request.dataset_prebuilt.training);
      const evaluationPath = selectPrebuiltEvaluationPath(request.dataset_prebuilt);
      await copyFile(trainingPath, artifacts.trainingJsonl);
      examples = await examplesFromChatJsonl(evaluationPath);
    } else {
      const jsonl = compileSpecToJsonl(request.spec_snapshot);
      await writeFile(artifacts.trainingJsonl, `${jsonl}\n`, "utf8");
    }

    const system = buildSystemMessage(request.spec_snapshot);
    const baseModelForEvaluation = config.paths.baseModel ?? request.spec_snapshot.base_model;
    await updateRun({
      status: "evaluating_baseline",
      stage: "evaluating_baseline",
      message: "Running baseline evaluation.",
      details: {
        examples: examples.length,
        eval_examples_used: config.evaluation.maxExamples ?? examples.length,
        model_id: baseModelForEvaluation,
      },
    });
    const baseline = await evaluateExamples({
      kind: "baseline",
      modelId: baseModelForEvaluation,
      baseModelId: baseModelForEvaluation,
      examples,
      system,
      config,
      outputPath: artifacts.baselineEvalJson,
      reporter: runReporter,
    });

    await updateRun({
      status: "training",
      stage: "training",
      message: config.dryRun ? "Recording dry-run training result." : "Launching local uv training process.",
      details: { training_backend: config.training.backend, dry_run: config.dryRun },
    });
    const training = await launchProcessTraining({ request, artifacts, config, reporter: runReporter });

    await updateRun({
      status: "evaluating_candidate",
      stage: "evaluating_candidate",
      message: "Running candidate evaluation.",
      details: { model_artifact_uri: training.model_artifact_uri },
    });
    const candidate = await evaluateExamples({
      kind: "candidate",
      modelId: training.model_artifact_uri ?? training.training_job_name,
      baseModelId: baseModelForEvaluation,
      adapterPath: training.model_artifact_uri,
      examples,
      system,
      config,
      outputPath: artifacts.candidateEvalJson,
      reporter: runReporter,
    });
    const comparison = compareEvalReports(baseline, candidate);
    const completedAt = new Date().toISOString();
    const duration = elapsed(startedPerf);
    const report = runReportSchema.parse({
      run_id: request.run_id,
      behavior_spec_id: request.behavior_spec_id,
      user_id: request.user_id,
      run_number: request.run_number,
      base_model: request.spec_snapshot.base_model,
      fine_tuned_model_id: training.model_artifact_uri ?? training.training_job_name,
      status: "completed",
      baseline,
      candidate,
      comparison,
      training,
      artifact_uris: {
        dataset: fileUri(artifacts.trainingJsonl),
        baseline_eval: fileUri(artifacts.baselineEvalJson),
        candidate_eval: fileUri(artifacts.candidateEvalJson),
        report: fileUri(artifacts.runReportJson),
      },
      run_metadata: {
        base_model: request.spec_snapshot.base_model,
        fine_tuned_model_id: training.model_artifact_uri ?? training.training_job_name,
        dataset_prebuilt: Boolean(request.dataset_prebuilt),
        dataset_uri: fileUri(artifacts.trainingJsonl),
        spec_example_count: request.spec_snapshot.examples.length,
        training_example_count: request.dataset_prebuilt ? null : request.spec_snapshot.examples.length,
        eval_examples_total: baseline.eval_examples_total,
        eval_examples_used: baseline.eval_examples_used,
        started_at: startedAt,
        completed_at: completedAt,
        elapsed_ms: duration.ms,
        elapsed_seconds: duration.seconds,
      },
      created_at: completedAt,
    });
    await writeJson(artifacts.runReportJson, report);
    await store.completeRun(report, artifacts.runDir, artifacts.runReportJson);
    await reporter?.onEvent?.({
      stage: "completed",
      status: "completed",
      message: "Run completed successfully.",
      details: {
        report_path: artifacts.runReportJson,
        model_id: `local-${request.run_id}`,
        avg_score_delta: comparison.avg_score_delta,
        elapsed_seconds: duration.seconds,
      },
    });
    return {
      request,
      report,
      reportPath: artifacts.runReportJson,
      artifactDir: dirname(artifacts.runReportJson),
    };
  } catch (error) {
    await store.failRun(request.run_id, error instanceof Error ? error.message : String(error)).catch(() => undefined);
    await reporter?.onEvent?.({
      stage: "failed",
      status: "failed",
      message: error instanceof Error ? error.message : String(error),
    });
    throw error;
  }
}
