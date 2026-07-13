import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fineTuneRunRequestSchema, runReportSchema } from "../src/contracts.js";
import { createLocalStore, isTerminalRunState, type LocalStore } from "../src/store.js";

const runId = "33333333-3333-4333-8333-333333333333";
const specId = "44444444-4444-4444-8444-444444444444";

function requestFixture() {
  return fineTuneRunRequestSchema.parse({
    run_id: runId,
    user_id: "local-user",
    behavior_spec_id: specId,
    run_number: 1,
    spec_snapshot: {
      name: "Local Store Spec",
      description: "",
      system_prompt: "Return labels.",
      guidelines: [],
      constraints: [],
      base_model: "Qwen/Qwen3.5-2B",
      examples: [{ input: "Classify: good", output: "positive" }],
    },
    hyperparameters: {
      n_epochs: 1,
      augment: false,
      use_llm_judge: false,
      save_adapter_only: true,
    },
  });
}

function reportFixture(reportPath: string) {
  const evalReport = {
    kind: "baseline",
    model_id: "Qwen/Qwen3.5-2B",
    total: 1,
    eval_examples_total: 1,
    eval_examples_used: 1,
    eval_truncated: false,
    avg_score: 0,
    pass_rate: 0,
    exact_match_rate: 0,
    avg_latency_ms: 0,
    results: [{
      prompt: "Classify: good",
      expected: "positive",
      actual: "",
      passed: false,
      score: 0,
      reasoning: "test",
      latency_ms: 0,
    }],
    artifact_uri: `file://${reportPath}`,
    scoring_method: "heuristic",
  };
  return runReportSchema.parse({
    run_id: runId,
    behavior_spec_id: specId,
    user_id: "local-user",
    run_number: 1,
    base_model: "Qwen/Qwen3.5-2B",
    fine_tuned_model_id: `file://${reportPath}`,
    status: "completed",
    baseline: evalReport,
    candidate: { ...evalReport, kind: "candidate", model_id: `file://${reportPath}` },
    comparison: {
      avg_score_delta: 0,
      pass_rate_delta: 0,
      exact_match_rate_delta: 0,
      regressions: 0,
      improvements: 0,
      regressed_examples: [],
    },
    training: {
      provider: "local-uv",
      training_job_name: "test-job",
      model_artifact_uri: `file://${reportPath}`,
      metrics: { loss: 0.1 },
      exit_code: 0,
      log_uri: `file://${reportPath}`,
    },
    artifact_uris: {
      dataset: `file://${reportPath}`,
      baseline_eval: `file://${reportPath}`,
      candidate_eval: `file://${reportPath}`,
      report: `file://${reportPath}`,
    },
    run_metadata: {
      base_model: "Qwen/Qwen3.5-2B",
      fine_tuned_model_id: `file://${reportPath}`,
      dataset_prebuilt: false,
      dataset_uri: `file://${reportPath}`,
      spec_example_count: 1,
      training_example_count: 1,
      eval_examples_total: 1,
      eval_examples_used: 1,
      started_at: "2026-01-01T00:00:00.000Z",
      completed_at: "2026-01-01T00:00:01.000Z",
      elapsed_ms: 1000,
      elapsed_seconds: 1,
    },
    created_at: "2026-01-01T00:00:01.000Z",
  });
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function removeMetadataDb(store: LocalStore): Promise<void> {
  await Promise.all([
    rm(store.paths.metadataDb, { force: true }),
    rm(`${store.paths.metadataDb}-shm`, { force: true }),
    rm(`${store.paths.metadataDb}-wal`, { force: true }),
  ]);
}

test("local store persists runs, events, reports, specs, and model records", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-test-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const artifactDir = join(root, "artifacts", runId);
    const request = requestFixture();
    const reportPath = join(artifactDir, "run-report.json");
    await store.startRun({ request, artifactDir });
    await store.updateRun({
      runId,
      status: "training",
      stage: "training",
      message: "Training.",
      details: { dry_run: true },
    });

    const report = reportFixture(reportPath);
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    await store.completeRun(report, artifactDir, reportPath);

    const runs = await store.listRuns();
    assert.equal(runs[0]?.status, "completed");
    assert.equal((await store.getRun(runId.slice(0, 8))).id, runId);
    assert.equal((await store.getRunEvents(runId)).length, 4);
    assert.equal((await store.getRunReport(runId)).run_id, runId);
    assert.equal((await store.listModels())[0]?.run_id, runId);
    assert.equal((await store.getSpec(specId.slice(0, 8))).spec.name, "Local Store Spec");
    assert.equal(await exists(join(store.root, "catalog")), false);

    await store.rebuildIndexes();
    assert.equal((await store.listRuns())[0]?.id, runId);
    assert.match(await readFile(join(artifactDir, "progress.jsonl"), "utf8"), /Training/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("local store rebuilds SQLite metadata from canonical files", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-rebuild-test-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const artifactDir = join(root, "artifacts", runId);
    const request = requestFixture();
    const reportPath = join(artifactDir, "run-report.json");
    await store.startRun({ request, artifactDir });
    await store.updateRun({
      runId,
      status: "training",
      stage: "training",
      message: "Training.",
      details: { dry_run: true },
    });

    const report = reportFixture(reportPath);
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    await store.completeRun(report, artifactDir, reportPath);

    await removeMetadataDb(store);
    assert.equal((await store.listRuns()).length, 0);
    assert.equal((await store.getRunEvents(runId)).length, 4);

    await store.rebuildIndexes();
    assert.equal((await store.listRuns())[0]?.id, runId);
    assert.equal((await store.getRunEvents(runId)).length, 4);
    assert.equal((await store.listSpecs())[0]?.id, specId);
    assert.equal((await store.listModels())[0]?.id, `local-${runId}`);
    assert.equal(await exists(join(store.root, "catalog")), false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("dry-run completion does not create a model record", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-dry-model-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const request = requestFixture();
    const artifactDir = join(root, "artifacts", runId);
    const reportPath = join(artifactDir, "run-report.json");
    await store.startRun({ request, artifactDir });
    const real = reportFixture(reportPath);
    const report = runReportSchema.parse({
      ...real,
      training: { ...real.training, metrics: { dry_run: true } },
    });
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    const state = await store.completeRun(report, artifactDir, reportPath);
    assert.equal(state.model_id, undefined);
    assert.equal((await store.listModels()).length, 0);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("resuming work clears stale failure and completion fields", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-resume-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const request = requestFixture();
    const artifactDir = join(root, "artifacts", runId);
    await store.startRun({ request, artifactDir });
    const failed = await store.failRun(runId, "old failure");
    assert.equal(failed.error, "old failure");
    assert.ok(failed.completed_at);

    const resumed = await store.updateRun({
      runId,
      status: "preparing",
      stage: "preparing",
      message: "Resuming.",
    });
    assert.equal(resumed.error, undefined);
    assert.equal(resumed.completed_at, undefined);
    const persisted = await store.getRun(runId);
    assert.equal(persisted.error, undefined);
    assert.equal(persisted.completed_at, undefined);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("a cancellation marker wins over late progress, failure, and completion writes", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-store-cancel-race-"));
  try {
    const store = createLocalStore(join(root, "store"));
    const request = requestFixture();
    const artifactDir = join(root, "artifacts", runId);
    const reportPath = join(artifactDir, "run-report.json");
    await store.startRun({ request, artifactDir });
    await store.cancelRun(runId);

    const progressed = await store.updateRun({
      runId,
      status: "training",
      stage: "training",
      message: "Late training update.",
    });
    assert.equal(progressed.status, "cancelled");
    assert.equal(progressed.current_stage, "cancel_requested");
    assert.equal(isTerminalRunState(progressed), false);

    const failed = await store.failRun(runId, "late failure");
    assert.equal(failed.status, "cancelled");
    assert.equal(failed.error, undefined);

    const report = reportFixture(reportPath);
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    const completed = await store.completeRun(report, artifactDir, reportPath);
    assert.equal(completed.status, "cancelled");
    assert.equal(completed.current_stage, "cancel_requested");
    assert.equal((await store.listModels()).length, 0);

    const finalized = await store.finalizeCancellation(runId);
    assert.equal(finalized.current_stage, "cancelled");
    assert.equal(isTerminalRunState(finalized), true);
    await store.cancelRun(runId);
    const unchanged = await store.getRun(runId);
    assert.equal(unchanged.status, finalized.status);
    assert.equal(unchanged.current_stage, finalized.current_stage);
    assert.equal(unchanged.updated_at, finalized.updated_at);
    assert.equal(unchanged.completed_at, finalized.completed_at);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
