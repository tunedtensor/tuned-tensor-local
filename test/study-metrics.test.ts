import assert from "node:assert/strict";
import test from "node:test";
import {
  computeBinaryClassificationMetrics,
  type BinaryScoredRow,
} from "../src/study-metrics.js";

function rows(
  positives: readonly boolean[],
  probabilities: readonly number[],
): BinaryScoredRow[] {
  return positives.map((positive, index) => ({
    id: `row-${index + 1}`,
    positive,
    probability: probabilities[index]!,
  }));
}

function assertMetrics(
  actual: ReturnType<typeof computeBinaryClassificationMetrics>,
  expected: ReturnType<typeof computeBinaryClassificationMetrics>,
): void {
  assert.ok(Math.abs(actual.average_precision - expected.average_precision) < 1e-15);
  assert.ok(Math.abs(actual.roc_auc - expected.roc_auc) < 1e-15);
  assert.ok(Math.abs(actual.f1_at_0_5 - expected.f1_at_0_5) < 1e-15);
}

test("binary metrics score perfect and reversed rankings", () => {
  const labels = [true, false, true, false];
  assert.deepEqual(
    computeBinaryClassificationMetrics(rows(labels, [0.9, 0.1, 0.8, 0.2])),
    { average_precision: 1, roc_auc: 1, f1_at_0_5: 1 },
  );
  assertMetrics(
    computeBinaryClassificationMetrics(rows(labels, [0.1, 0.9, 0.2, 0.8])),
    { average_precision: 5 / 12, roc_auc: 0, f1_at_0_5: 0 },
  );
});

test("binary metrics process equal probabilities as order-independent groups", () => {
  const allTied = computeBinaryClassificationMetrics(
    rows([true, false, true, false], [0.5, 0.5, 0.5, 0.5]),
  );
  assert.deepEqual(allTied, {
    average_precision: 0.5,
    roc_auc: 0.5,
    f1_at_0_5: 2 / 3,
  });

  const first = rows([true, true, false, false], [0.9, 0.5, 0.5, 0.1]);
  const reordered = [first[0]!, first[2]!, first[1]!, first[3]!];
  assertMetrics(computeBinaryClassificationMetrics(first), {
    average_precision: 5 / 6,
    roc_auc: 7 / 8,
    f1_at_0_5: 4 / 5,
  });
  assert.deepEqual(
    computeBinaryClassificationMetrics(reordered),
    computeBinaryClassificationMetrics(first),
  );
});

test("binary metrics keep ranking and the fixed decision threshold separate", () => {
  assert.deepEqual(
    computeBinaryClassificationMetrics(rows([true, false], [0.49, 0.1])),
    { average_precision: 1, roc_auc: 1, f1_at_0_5: 0 },
  );
});

test("binary metrics reject invalid probabilities and one-class inputs", () => {
  assert.throws(
    () => computeBinaryClassificationMetrics([]),
    /empty binary-classification split/,
  );
  assert.throws(
    () => computeBinaryClassificationMetrics(rows([true, true], [0.8, 0.2])),
    /requires both positive and negative/,
  );
  for (const probability of [-0.01, 1.01, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.throws(
      () => computeBinaryClassificationMetrics(rows([true, false], [probability, 0.2])),
      /finite probability between 0 and 1/,
    );
  }
});
