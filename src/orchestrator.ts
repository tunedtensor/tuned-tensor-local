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
import { createLocalStore } from "./store.js";

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

export async function runLocalFineTune(input: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
}): Promise<LocalRunResult> {
  const startedPerf = performance.now();
  const startedAt = new Date().toISOString();
  const { request, config } = input;
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
    await store.updateRun({
      runId: request.run_id,
      status: "preparing",
      stage: "preparing",
      message: "Preparing local run artifacts.",
      details: { artifact_dir: artifacts.runDir },
    });

    let examples = examplesFromSpec(request.spec_snapshot);
    if (request.dataset_prebuilt) {
      const trainingPath = request.dataset_prebuilt.training.replace(/^file:\/\//, "");
      await copyFile(trainingPath, artifacts.trainingJsonl);
      examples = await examplesFromChatJsonl(trainingPath);
    } else {
      const jsonl = compileSpecToJsonl(request.spec_snapshot);
      await writeFile(artifacts.trainingJsonl, `${jsonl}\n`, "utf8");
    }

    const system = buildSystemMessage(request.spec_snapshot);
    await store.updateRun({
      runId: request.run_id,
      status: "evaluating_baseline",
      stage: "evaluating_baseline",
      message: "Running baseline evaluation.",
      details: { examples: examples.length },
    });
    const baseline = await evaluateExamples({
      kind: "baseline",
      modelId: request.spec_snapshot.base_model,
      examples,
      system,
      config,
      outputPath: artifacts.baselineEvalJson,
    });

    await store.updateRun({
      runId: request.run_id,
      status: "training",
      stage: "training",
      message: config.dryRun ? "Recording dry-run training result." : "Launching local uv training process.",
      details: { training_backend: config.training.backend, dry_run: config.dryRun },
    });
    const training = await launchProcessTraining({ request, artifacts, config });

    await store.updateRun({
      runId: request.run_id,
      status: "evaluating_candidate",
      stage: "evaluating_candidate",
      message: "Running candidate evaluation.",
      details: { model_artifact_uri: training.model_artifact_uri },
    });
    const candidate = await evaluateExamples({
      kind: "candidate",
      modelId: training.model_artifact_uri ?? training.training_job_name,
      examples,
      system,
      config,
      outputPath: artifacts.candidateEvalJson,
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
    return {
      request,
      report,
      reportPath: artifacts.runReportJson,
      artifactDir: dirname(artifacts.runReportJson),
    };
  } catch (error) {
    await store.failRun(request.run_id, error instanceof Error ? error.message : String(error)).catch(() => undefined);
    throw error;
  }
}
