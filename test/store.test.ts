import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fineTuneRunRequestSchema, runReportSchema } from "../src/contracts.js";
import { createLocalStore } from "../src/store.js";

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
      metrics: { dry_run: true },
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
    assert.equal((await store.getRunEvents(runId)).length, 3);
    assert.equal((await store.getRunReport(runId)).run_id, runId);
    assert.equal((await store.listModels())[0]?.run_id, runId);
    assert.equal((await store.getSpec(specId.slice(0, 8))).spec.name, "Local Store Spec");

    await store.rebuildIndexes();
    assert.equal((await store.listRuns())[0]?.id, runId);
    assert.match(await readFile(join(artifactDir, "progress.jsonl"), "utf8"), /Training/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
