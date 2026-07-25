import assert from "node:assert/strict";
import { test } from "node:test";
import { compareRuns } from "../src/compare.js";
import type { EvalExampleResult, EvalReport, RunReport } from "../src/contracts.js";

function result(
  prompt: string,
  score: number,
  expected = "expected text",
  actual = "actual text",
): EvalExampleResult {
  return {
    prompt,
    expected,
    actual,
    passed: score === 1,
    score,
    reasoning: "Scored by normalized exact match.",
    latency_ms: 10,
    scored_by: "exact_match",
  };
}

function evalReport(kind: "baseline" | "candidate", results: EvalExampleResult[]): EvalReport {
  const avg = results.length > 0
    ? results.reduce((sum, item) => sum + item.score, 0) / results.length
    : 0;
  return {
    kind,
    model_id: kind === "baseline" ? "Qwen/Qwen3.5-2B" : "local-adapter",
    total: results.length,
    eval_examples_total: results.length,
    eval_examples_used: results.length,
    eval_truncated: false,
    avg_score: avg,
    pass_rate: results.filter((item) => item.passed).length / Math.max(1, results.length),
    exact_match_rate: results.filter((item) => item.expected === item.actual).length
      / Math.max(1, results.length),
    avg_token_f1: 0.5,
    avg_latency_ms: 10,
    results,
    artifact_uri: "file:///tmp/eval.json",
    scoring_method: "exact_match",
    inference_provider: "transformers",
    scoring_mode: "exact_match",
  };
}

function runReport(
  id: string,
  baseline: EvalReport,
  candidate: EvalReport,
): RunReport {
  return {
    run_id: id,
    behavior_spec_id: "22222222-2222-4222-8222-222222222222",
    user_id: "local-user",
    run_number: 1,
    base_model: "Qwen/Qwen3.5-2B",
    fine_tuned_model_id: "local-adapter",
    status: "completed",
    baseline,
    candidate,
    comparison: {
      avg_score_delta: candidate.avg_score - baseline.avg_score,
      pass_rate_delta: candidate.pass_rate - baseline.pass_rate,
      exact_match_rate_delta: candidate.exact_match_rate - baseline.exact_match_rate,
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
      fine_tuned_model_id: "local-adapter",
      dataset_prebuilt: false,
      dataset_format: null,
      dataset_uri: "file:///tmp/train.jsonl",
      spec_example_count: candidate.results.length,
      training_example_count: candidate.results.length,
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

test("compareRuns aligns deterministic scores on shared prompts", () => {
  const runA = runReport(
    "11111111-1111-4111-8111-111111111111",
    evalReport("baseline", [result("p1", 0), result("p2", 1)]),
    evalReport("candidate", [
      result("p1", 0.5, "alpha beta", "gamma"),
      result("p2", 1, "yes", "yes"),
    ]),
  );
  const runB = runReport(
    "33333333-3333-4333-8333-333333333333",
    evalReport("baseline", [result("p1", 0), result("p2", 1), result("p3", 0)]),
    evalReport("candidate", [
      result("p1", 1, "alpha beta", "alpha beta"),
      result("p2", 0.5, "yes", "yes please"),
      result("p3", 1, "new", "new"),
    ]),
  );

  const comparison = compareRuns(runA, runB);
  assert.equal(comparison.shared.examples, 2);
  assert.equal(comparison.shared.run_a.candidate_avg_score, 0.75);
  assert.equal(comparison.shared.run_b.candidate_avg_score, 0.75);
  assert.equal(comparison.shared.candidate_avg_score_delta, 0);
  assert.ok((comparison.shared.candidate_avg_token_f1_delta ?? 0) > 0);
  assert.equal(comparison.b_only.examples, 1);
  assert.equal(comparison.b_only.candidate_avg_score, 1);
  assert.match(comparison.notes[0] ?? "", /not directly comparable/);
});

test("compareRuns returns null shared metrics when evaluation sets do not overlap", () => {
  const first = runReport(
    "11111111-1111-4111-8111-111111111111",
    evalReport("baseline", [result("a", 0)]),
    evalReport("candidate", [result("a", 1, "a", "a")]),
  );
  const second = runReport(
    "33333333-3333-4333-8333-333333333333",
    evalReport("baseline", [result("b", 0)]),
    evalReport("candidate", [result("b", 1, "b", "b")]),
  );

  const comparison = compareRuns(first, second);
  assert.equal(comparison.shared.examples, 0);
  assert.equal(comparison.shared.candidate_avg_score_delta, null);
  assert.equal(comparison.shared.candidate_avg_token_f1_delta, null);
  assert.equal(comparison.b_only.examples, 1);
});
