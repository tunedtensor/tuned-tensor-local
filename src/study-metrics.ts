export const BINARY_DECISION_THRESHOLD = 0.5;

export interface BinaryScoredRow {
  id: string;
  positive: boolean;
  probability: number;
}

export interface BinaryClassificationMetrics {
  average_precision: number;
  roc_auc: number;
  f1_at_0_5: number;
}

function clampUnit(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/**
 * Computes binary ranking metrics without choosing a threshold from validation
 * labels. Equal probabilities are processed together so row order cannot
 * change average precision or ROC AUC.
 */
export function computeBinaryClassificationMetrics(
  rows: readonly BinaryScoredRow[],
): BinaryClassificationMetrics {
  if (rows.length === 0) {
    throw new Error("Cannot score an empty binary-classification split");
  }

  let positiveCount = 0;
  for (const [index, row] of rows.entries()) {
    if (
      typeof row.probability !== "number"
      || !Number.isFinite(row.probability)
      || row.probability < 0
      || row.probability > 1
    ) {
      throw new Error(
        `Prediction for ID "${row.id}" at index ${index} must be a finite probability between 0 and 1`,
      );
    }
    if (row.positive) positiveCount += 1;
  }

  const negativeCount = rows.length - positiveCount;
  if (positiveCount === 0 || negativeCount === 0) {
    throw new Error(
      "Binary-classification scoring requires both positive and negative validation examples",
    );
  }

  const ranked = [...rows].sort((left, right) => right.probability - left.probability);
  let cursor = 0;
  let seen = 0;
  let cumulativePositive = 0;
  let lowerProbabilityNegatives = negativeCount;
  let averagePrecision = 0;
  let favorableAucPairs = 0;

  while (cursor < ranked.length) {
    const probability = ranked[cursor]!.probability;
    let end = cursor;
    let groupPositive = 0;
    let groupNegative = 0;

    while (end < ranked.length && ranked[end]!.probability === probability) {
      if (ranked[end]!.positive) groupPositive += 1;
      else groupNegative += 1;
      end += 1;
    }

    lowerProbabilityNegatives -= groupNegative;
    favorableAucPairs += (
      groupPositive * lowerProbabilityNegatives
      + 0.5 * groupPositive * groupNegative
    );

    seen += groupPositive + groupNegative;
    cumulativePositive += groupPositive;
    if (groupPositive > 0) {
      averagePrecision += (
        (groupPositive / positiveCount)
        * (cumulativePositive / seen)
      );
    }

    cursor = end;
  }

  let truePositive = 0;
  let falsePositive = 0;
  let falseNegative = 0;
  for (const row of rows) {
    const predictedPositive = row.probability >= BINARY_DECISION_THRESHOLD;
    if (predictedPositive && row.positive) truePositive += 1;
    else if (predictedPositive) falsePositive += 1;
    else if (row.positive) falseNegative += 1;
  }
  const f1Denominator = 2 * truePositive + falsePositive + falseNegative;

  return {
    average_precision: clampUnit(averagePrecision),
    roc_auc: clampUnit(favorableAucPairs / (positiveCount * negativeCount)),
    f1_at_0_5: clampUnit(
      f1Denominator === 0 ? 0 : (2 * truePositive) / f1Denominator,
    ),
  };
}
