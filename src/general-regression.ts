import type { ComparisonReport } from "./contracts.js";

export interface GeneralRegressionPolicy {
  maxScoreDrop: number;
  maxPassRateDrop: number;
}

export interface GeneralRegressionGate {
  passed: boolean;
  failures: string[];
}

export function evaluateGeneralRegressionGate(
  comparison: ComparisonReport,
  policy: GeneralRegressionPolicy,
): GeneralRegressionGate {
  const failures: string[] = [];
  const scoreDrop = Math.max(0, -comparison.avg_score_delta);
  const passRateDrop = Math.max(0, -comparison.pass_rate_delta);
  const tolerance = 1e-12;
  if (scoreDrop > policy.maxScoreDrop + tolerance) {
    failures.push(
      `Average score dropped ${scoreDrop.toFixed(4)}, `
      + `exceeding the ${policy.maxScoreDrop.toFixed(4)} budget.`,
    );
  }
  if (passRateDrop > policy.maxPassRateDrop + tolerance) {
    failures.push(
      `Pass rate dropped ${passRateDrop.toFixed(4)}, `
      + `exceeding the ${policy.maxPassRateDrop.toFixed(4)} budget.`,
    );
  }
  return { passed: failures.length === 0, failures };
}
