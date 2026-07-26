import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { activateModel, getActiveModel, rollbackActiveModel } from "../src/active-model.js";
import { fineTuneRunRequestSchema, runReportSchema } from "../src/contracts.js";
import { createLocalStore } from "../src/store.js";

function fixture(runId: string, specId: string, passed: boolean) {
  const request = fineTuneRunRequestSchema.parse({
    run_id: runId,
    user_id: "local-user",
    behavior_spec_id: specId,
    run_number: 1,
    spec_snapshot: {
      name: "Activation fixture",
      base_model: "Qwen/Qwen3.5-2B",
      examples: [
        { input: "one", output: "1" },
        { input: "two", output: "2" },
      ],
    },
    hyperparameters: { n_epochs: 1 },
  });
  const evalReport = {
    kind: "baseline" as const,
    model_id: request.spec_snapshot.base_model,
    total: 1,
    eval_examples_total: 1,
    eval_examples_used: 1,
    eval_truncated: false,
    avg_score: 1,
    pass_rate: 1,
    exact_match_rate: 1,
    avg_latency_ms: 1,
    results: [{
      prompt: "one",
      expected: "1",
      actual: "1",
      passed: true,
      score: 1,
      reasoning: "test",
      latency_ms: 1,
    }],
    artifact_uri: "file:///tmp/eval.json",
    scoring_method: "exact_match" as const,
  };
  const comparison = {
    avg_score_delta: passed ? 0 : -1,
    pass_rate_delta: passed ? 0 : -1,
    exact_match_rate_delta: passed ? 0 : -1,
    regressions: passed ? 0 : 1,
    improvements: 0,
    regressed_examples: [],
  };
  const report = runReportSchema.parse({
    run_id: runId,
    behavior_spec_id: specId,
    user_id: "local-user",
    run_number: 1,
    base_model: request.spec_snapshot.base_model,
    fine_tuned_model_id: `file:///tmp/${runId}`,
    status: "completed",
    baseline: evalReport,
    candidate: { ...evalReport, kind: "candidate", model_id: `file:///tmp/${runId}` },
    comparison,
    general_regression: {
      dataset_uri: "file:///tmp/general.jsonl",
      dataset_sha256: "a".repeat(64),
      baseline: { ...evalReport, eval_split: "general_regression" },
      candidate: {
        ...evalReport,
        kind: "candidate",
        model_id: `file:///tmp/${runId}`,
        eval_split: "general_regression",
      },
      comparison,
      policy: { max_score_drop: 0.03, max_pass_rate_drop: 0.05 },
      passed,
      failures: passed ? [] : ["regressed"],
    },
    training: {
      provider: "local-uv",
      training_job_name: runId,
      model_artifact_uri: `file:///tmp/${runId}`,
      metrics: { loss: 0.1 },
      exit_code: 0,
      log_uri: "file:///tmp/training.log",
    },
    artifact_uris: {
      dataset: "file:///tmp/train.jsonl",
      baseline_eval: "file:///tmp/baseline.json",
      candidate_eval: "file:///tmp/candidate.json",
      general_baseline_eval: "file:///tmp/general-baseline.json",
      general_candidate_eval: "file:///tmp/general-candidate.json",
      report: "file:///tmp/report.json",
    },
    run_metadata: {
      base_model: request.spec_snapshot.base_model,
      fine_tuned_model_id: `file:///tmp/${runId}`,
      dataset_prebuilt: false,
      dataset_uri: "file:///tmp/train.jsonl",
      spec_example_count: 2,
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
  return { request, report };
}

async function persistFixture(
  root: string,
  runId: string,
  specId: string,
  passed: boolean,
) {
  const store = createLocalStore(join(root, "store"));
  const artifactDir = join(root, "artifacts", runId);
  const { request, report } = fixture(runId, specId, passed);
  const reportPath = join(artifactDir, "run-report.json");
  await store.startRun({ request, artifactDir });
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await store.completeRun(report, artifactDir, reportPath);
  return { store, modelId: `local-${runId}` };
}

test("activation requires a passing general regression gate and rollback restores base", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-active-model-"));
  try {
    const passing = await persistFixture(
      root,
      "11111111-1111-4111-8111-111111111111",
      "22222222-2222-4222-8222-222222222222",
      true,
    );
    const failing = await persistFixture(
      root,
      "33333333-3333-4333-8333-333333333333",
      "44444444-4444-4444-8444-444444444444",
      false,
    );

    assert.equal((await getActiveModel(passing.store)).model, null);
    const activated = await activateModel(passing.store, passing.modelId);
    assert.equal(activated.model_id, passing.modelId);
    assert.equal((await getActiveModel(passing.store)).model?.id, passing.modelId);
    const reactivated = await activateModel(passing.store, passing.modelId);
    assert.equal(reactivated.previous_model_id, null);

    await assert.rejects(
      activateModel(passing.store, failing.modelId),
      /failed general regression/,
    );
    const rolledBack = await rollbackActiveModel(passing.store);
    assert.equal(rolledBack.model_id, null);
    assert.equal((await getActiveModel(passing.store)).model, null);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
