import type { EvalExampleResult, EvalReport, RunReport } from "./contracts.js";
import { tokenF1 } from "./evaluation.js";

interface SubsetSide {
  baseline_avg_score: number | null;
  candidate_avg_score: number | null;
  candidate_avg_token_f1: number | null;
}

interface RunSummary {
  run_id: string;
  started_at: string;
  base_model: string;
  training_example_count: number | null;
  eval_examples_used: number;
  baseline_avg_score: number;
  candidate_avg_score: number;
  candidate_avg_token_f1: number | null;
  avg_score_delta: number;
  fallback_scored_count: number;
}

export interface RunComparison {
  run_a: RunSummary;
  run_b: RunSummary;
  shared: {
    examples: number;
    run_a: SubsetSide;
    run_b: SubsetSide;
    candidate_avg_score_delta: number | null;
    candidate_avg_token_f1_delta: number | null;
  };
  b_only: SubsetSide & { examples: number };
  judge_noise: {
    identical_baseline_outputs: number;
    mean_score_spread: number | null;
    max_score_spread: number | null;
  };
  notes: string[];
}

function byPrompt(report: EvalReport): Map<string, EvalExampleResult> {
  return new Map(report.results.map((result) => [result.prompt, result]));
}

function avg(values: number[]): number | null {
  return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function summarizeRun(report: RunReport): RunSummary {
  return {
    run_id: report.run_id,
    started_at: report.run_metadata.started_at,
    base_model: report.base_model,
    training_example_count: report.run_metadata.training_example_count,
    eval_examples_used: report.run_metadata.eval_examples_used,
    baseline_avg_score: report.baseline.avg_score,
    candidate_avg_score: report.candidate.avg_score,
    candidate_avg_token_f1: report.candidate.avg_token_f1 ?? null,
    avg_score_delta: report.comparison.avg_score_delta,
    fallback_scored_count: (report.baseline.fallback_scored_count ?? 0)
      + (report.candidate.fallback_scored_count ?? 0),
  };
}

function subsetSide(
  prompts: string[],
  baseline: Map<string, EvalExampleResult>,
  candidate: Map<string, EvalExampleResult>,
): SubsetSide {
  const candidateResults = prompts
    .map((prompt) => candidate.get(prompt))
    .filter((result): result is EvalExampleResult => Boolean(result));
  return {
    baseline_avg_score: avg(prompts
      .map((prompt) => baseline.get(prompt)?.score)
      .filter((score): score is number => typeof score === "number")),
    candidate_avg_score: avg(candidateResults.map((result) => result.score)),
    candidate_avg_token_f1: avg(candidateResults.map((result) => tokenF1(result.expected, result.actual))),
  };
}

/**
 * Aligns two run reports on their shared eval prompts so recipe changes can
 * be compared apples-to-apples even when the eval set grew or shifted between
 * runs. Headline avg_score_delta values are not comparable across different
 * eval subsets; the shared subset is. Identical baseline outputs that appear
 * in both runs also give a free measurement of judge score noise.
 */
export function compareRuns(a: RunReport, b: RunReport): RunComparison {
  const aCandidate = byPrompt(a.candidate);
  const bCandidate = byPrompt(b.candidate);
  const aBaseline = byPrompt(a.baseline);
  const bBaseline = byPrompt(b.baseline);
  const sharedPrompts = [...bCandidate.keys()].filter((prompt) => aCandidate.has(prompt));
  const bOnlyPrompts = [...bCandidate.keys()].filter((prompt) => !aCandidate.has(prompt));

  const spreads: number[] = [];
  let identical = 0;
  for (const prompt of sharedPrompts) {
    const first = aBaseline.get(prompt);
    const second = bBaseline.get(prompt);
    if (!first || !second || first.actual !== second.actual) continue;
    identical += 1;
    spreads.push(Math.abs(first.score - second.score));
  }

  const sharedA = subsetSide(sharedPrompts, aBaseline, aCandidate);
  const sharedB = subsetSide(sharedPrompts, bBaseline, bCandidate);
  const runA = summarizeRun(a);
  const runB = summarizeRun(b);

  const notes: string[] = [];
  if (bOnlyPrompts.length > 0) {
    notes.push(
      `run_b evaluated ${bOnlyPrompts.length} example(s) run_a never saw; headline avg_score_delta values `
      + "are not directly comparable across the two runs. Use the shared subset for recipe comparisons.",
    );
  }
  if (runA.fallback_scored_count + runB.fallback_scored_count > 0) {
    notes.push(
      `${runA.fallback_scored_count + runB.fallback_scored_count} example(s) across both runs were `
      + "fallback-scored after judge failures; prefer judge_only_avg_score when comparing judge quality.",
    );
  }

  return {
    run_a: runA,
    run_b: runB,
    shared: {
      examples: sharedPrompts.length,
      run_a: sharedA,
      run_b: sharedB,
      candidate_avg_score_delta:
        sharedA.candidate_avg_score !== null && sharedB.candidate_avg_score !== null
          ? sharedB.candidate_avg_score - sharedA.candidate_avg_score
          : null,
      candidate_avg_token_f1_delta:
        sharedA.candidate_avg_token_f1 !== null && sharedB.candidate_avg_token_f1 !== null
          ? sharedB.candidate_avg_token_f1 - sharedA.candidate_avg_token_f1
          : null,
    },
    b_only: { examples: bOnlyPrompts.length, ...subsetSide(bOnlyPrompts, bBaseline, bCandidate) },
    judge_noise: {
      identical_baseline_outputs: identical,
      mean_score_spread: avg(spreads),
      max_score_spread: spreads.length > 0 ? Math.max(...spreads) : null,
    },
    notes,
  };
}
