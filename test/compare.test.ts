import assert from "node:assert/strict";
import { test } from "node:test";
import { compareRuns } from "../src/compare.js";
import type { EvalExampleResult, EvalReport, RunReport } from "../src/contracts.js";

function exampleResult(overrides: Partial<EvalExampleResult> & { prompt: string }): EvalExampleResult {
  return {
    expected: "expected text",
    actual: "actual text",
    passed: false,
    score: 0.5,
    reasoning: "reasoning",
    latency_ms: 10,
    scored_by: "llm_judge",
    ...overrides,
  };
}

function evalReport(kind: "baseline" | "candidate", results: EvalExampleResult[]): EvalReport {
  const avg = results.reduce((sum, result) => sum + result.score, 0) / results.length;
  return {
    kind,
    model_id: "model",
    total: results.length,
    eval_examples_total: results.length,
    eval_examples_used: results.length,
    eval_truncated: false,
    avg_score: avg,
    pass_rate: 0,
    exact_match_rate: 0,
    avg_token_f1: 0.4,
    avg_latency_ms: 10,
    judge_scored_count: results.filter((result) => result.scored_by === "llm_judge").length,
    fallback_scored_count: results.filter((result) => result.scored_by === "exact_match_fallback").length,
    judge_only_avg_score: avg,
    results,
    artifact_uri: "file:///tmp/eval.json",
    scoring_method: "llm_judge",
  };
}

function runReport(id: string, baseline: EvalReport, candidate: EvalReport): RunReport {
  return {
    run_id: id,
    behavior_spec_id: "22222222-2222-4222-8222-222222222222",
    user_id: "local-user",
    run_number: 1,
    base_model: "Qwen/Qwen3.5-2B",
    fine_tuned_model_id: "file:///tmp/model.tar.gz",
    status: "completed",
    baseline,
    candidate,
    comparison: {
      avg_score_delta: candidate.avg_score - baseline.avg_score,
      pass_rate_delta: 0,
      exact_match_rate_delta: 0,
      regressions: 0,
      improvements: 0,
      regressed_examples: [],
    },
    training: {
      provider: "local-uv",
      training_job_name: `job-${id}`,
      metrics: null,
      exit_code: 0,
      log_uri: "file:///tmp/training.log",
    },
    artifact_uris: {
      dataset: "file:///tmp/train.jsonl",
      baseline_eval: "file:///tmp/baseline.json",
      candidate_eval: "file:///tmp/candidate.json",
      report: "file:///tmp/report.json",
    },
    run_metadata: {
      base_model: "Qwen/Qwen3.5-2B",
      fine_tuned_model_id: "file:///tmp/model.tar.gz",
      dataset_prebuilt: true,
      dataset_uri: "file:///tmp/train.jsonl",
      spec_example_count: 0,
      training_example_count: 100,
      eval_examples_total: baseline.eval_examples_total,
      eval_examples_used: baseline.eval_examples_used,
      started_at: "2026-07-02T10:00:00.000Z",
      completed_at: "2026-07-02T10:30:00.000Z",
      elapsed_ms: 1000,
      elapsed_seconds: 1,
    },
    created_at: "2026-07-02T10:30:00.000Z",
  };
}

test("compareRuns aligns shared prompts and isolates new-example effects", () => {
  // Run A judged prompts p1/p2; run B judged p1/p2 plus a new p3 that the
  // baseline aces, which inflates B's headline delta unfairly.
  const runA = runReport(
    "11111111-1111-4111-8111-111111111111",
    evalReport("baseline", [
      exampleResult({ prompt: "p1", actual: "base out 1", score: 0.7 }),
      exampleResult({ prompt: "p2", actual: "base out 2", score: 0.7 }),
    ]),
    evalReport("candidate", [
      exampleResult({ prompt: "p1", score: 0.6 }),
      exampleResult({ prompt: "p2", score: 0.6 }),
    ]),
  );
  const runB = runReport(
    "33333333-3333-4333-8333-333333333333",
    evalReport("baseline", [
      exampleResult({ prompt: "p1", actual: "base out 1", score: 0.8 }),
      exampleResult({ prompt: "p2", actual: "base out 2", score: 0.7 }),
      exampleResult({ prompt: "p3", actual: "base out 3", score: 0.9 }),
    ]),
    evalReport("candidate", [
      exampleResult({ prompt: "p1", score: 0.65 }),
      exampleResult({ prompt: "p2", score: 0.6 }),
      exampleResult({ prompt: "p3", score: 0.5 }),
    ]),
  );

  const comparison = compareRuns(runA, runB);
  assert.equal(comparison.shared.examples, 2);
  assert.equal(comparison.b_only.examples, 1);
  assert.equal(comparison.shared.run_a.candidate_avg_score, 0.6);
  assert.equal(comparison.shared.run_b.candidate_avg_score, 0.625);
  assert.ok(Math.abs((comparison.shared.candidate_avg_score_delta ?? 0) - 0.025) < 1e-9);
  assert.equal(comparison.b_only.candidate_avg_score, 0.5);
  // p1 and p2 baseline outputs are identical strings across runs, so their
  // judge score spread measures judge noise: |0.8-0.7| and |0.7-0.7|.
  assert.equal(comparison.judge_noise.identical_baseline_outputs, 2);
  assert.ok(Math.abs((comparison.judge_noise.mean_score_spread ?? 0) - 0.05) < 1e-9);
  assert.ok(Math.abs((comparison.judge_noise.max_score_spread ?? 0) - 0.1) < 1e-9);
  assert.ok(comparison.notes.some((note) => note.includes("not directly comparable")));
});

test("compareRuns surfaces fallback-scored examples in notes", () => {
  const results = [exampleResult({ prompt: "p1", score: 0, scored_by: "exact_match_fallback" })];
  const run = runReport(
    "11111111-1111-4111-8111-111111111111",
    evalReport("baseline", [exampleResult({ prompt: "p1", actual: "same", score: 0.7 })]),
    evalReport("candidate", results),
  );
  const other = runReport(
    "33333333-3333-4333-8333-333333333333",
    evalReport("baseline", [exampleResult({ prompt: "p1", actual: "same", score: 0.7 })]),
    evalReport("candidate", [exampleResult({ prompt: "p1", score: 0.6 })]),
  );
  const comparison = compareRuns(run, other);
  assert.equal(comparison.run_a.fallback_scored_count, 1);
  assert.ok(comparison.notes.some((note) => note.includes("fallback-scored")));
});
