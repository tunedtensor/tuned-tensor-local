import assert from "node:assert/strict";
import { test } from "node:test";
import { evaluateGeneralRegressionGate } from "../src/general-regression.js";

const comparison = {
  avg_score_delta: -0.02,
  pass_rate_delta: -0.04,
  exact_match_rate_delta: -0.02,
  regressions: 1,
  improvements: 0,
  regressed_examples: [],
};

test("general regression gate accepts changes within both budgets", () => {
  assert.deepEqual(
    evaluateGeneralRegressionGate(comparison, {
      maxScoreDrop: 0.03,
      maxPassRateDrop: 0.05,
    }),
    { passed: true, failures: [] },
  );
});

test("general regression gate tolerates floating-point noise at the budget", () => {
  assert.equal(
    evaluateGeneralRegressionGate(
      {
        ...comparison,
        avg_score_delta: -0.1 - 0.2,
        pass_rate_delta: -0.2 - 0.3,
      },
      {
        maxScoreDrop: 0.3,
        maxPassRateDrop: 0.5,
      },
    ).passed,
    true,
  );
});

test("general regression gate reports each exceeded budget", () => {
  const gate = evaluateGeneralRegressionGate(comparison, {
    maxScoreDrop: 0.01,
    maxPassRateDrop: 0.03,
  });
  assert.equal(gate.passed, false);
  assert.equal(gate.failures.length, 2);
  assert.match(gate.failures[0]!, /Average score dropped 0\.0200/);
  assert.match(gate.failures[1]!, /Pass rate dropped 0\.0400/);
});
