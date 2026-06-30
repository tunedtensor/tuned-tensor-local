import { spawn } from "node:child_process";
import { performance } from "node:perf_hooks";
import type {
  BehaviorSpecExample,
  EvalExampleResult,
  EvalReport,
  LocalRunnerConfig,
} from "./contracts.js";
import { writeJson, fileUri } from "./artifacts.js";

function normalize(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

function scoreActual(expected: string, actual: string): number {
  return normalize(expected) === normalize(actual) ? 1 : 0;
}

async function runInferenceCommand(args: {
  command: string[];
  prompt: string;
  system: string;
  expected: string;
  timeoutMs: number;
}): Promise<string> {
  const [cmd, ...cmdArgs] = args.command;
  const payload = JSON.stringify({
    system: args.system,
    prompt: args.prompt,
    expected: args.expected,
  });

  return await new Promise((resolve, reject) => {
    const child = spawn(cmd, cmdArgs, {
      stdio: ["pipe", "pipe", "pipe"],
      env: process.env,
    });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`Inference command timed out after ${args.timeoutMs}ms`));
    }, args.timeoutMs);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(`Inference command exited ${code}: ${stderr.slice(0, 1000)}`));
        return;
      }
      const trimmed = stdout.trim();
      try {
        const parsed = JSON.parse(trimmed) as { content?: unknown; output?: unknown; actual?: unknown };
        const content = parsed.content ?? parsed.output ?? parsed.actual;
        resolve(typeof content === "string" ? content : trimmed);
      } catch {
        resolve(trimmed);
      }
    });
    child.stdin.end(payload);
  });
}

export async function evaluateExamples(args: {
  kind: "baseline" | "candidate";
  modelId: string;
  examples: BehaviorSpecExample[];
  system: string;
  config: LocalRunnerConfig["evaluation"];
  outputPath: string;
}): Promise<EvalReport> {
  const maxExamples = args.config.maxExamples ?? args.examples.length;
  const selected = args.examples.slice(0, maxExamples);
  const command = args.kind === "baseline"
    ? args.config.baselineCommand
    : args.config.candidateCommand;
  const mode: "command" | "heuristic" = args.config.mode === "command" && command ? "command" : "heuristic";
  const results: EvalExampleResult[] = [];

  for (const example of selected) {
    const started = performance.now();
    const actual = mode === "command" && command
      ? await runInferenceCommand({
          command,
          prompt: example.input,
          system: args.system,
          expected: example.output,
          timeoutMs: args.config.timeoutMs,
        })
      : "";
    const latencyMs = Math.max(0, Math.round(performance.now() - started));
    const score = scoreActual(example.output, actual);
    results.push({
      prompt: example.input,
      expected: example.output,
      actual,
      passed: score === 1,
      score,
      reasoning: mode === "heuristic"
        ? "No inference command configured; recorded an empty local response."
        : "Scored by normalized exact match against command output.",
      latency_ms: latencyMs,
    });
  }

  const total = results.length;
  const avgScore = total > 0
    ? results.reduce((sum, result) => sum + result.score, 0) / total
    : 0;
  const passRate = total > 0
    ? results.filter((result) => result.passed).length / total
    : 0;
  const avgLatency = total > 0
    ? Math.round(results.reduce((sum, result) => sum + result.latency_ms, 0) / total)
    : 0;
  const report: EvalReport = {
    kind: args.kind,
    model_id: args.modelId,
    total,
    eval_examples_total: args.examples.length,
    eval_examples_used: total,
    eval_truncated: args.examples.length > total,
    avg_score: avgScore,
    pass_rate: passRate,
    exact_match_rate: passRate,
    avg_latency_ms: avgLatency,
    results,
    artifact_uri: fileUri(args.outputPath),
    scoring_method: mode,
    judge_model_id: null,
  };
  await writeJson(args.outputPath, report);
  return report;
}

export function compareEvalReports(baseline: EvalReport, candidate: EvalReport) {
  let regressions = 0;
  let improvements = 0;
  const regressedExamples = [];
  const count = Math.min(baseline.results.length, candidate.results.length);
  for (let index = 0; index < count; index += 1) {
    const oldScore = baseline.results[index]?.score ?? 0;
    const newScore = candidate.results[index]?.score ?? 0;
    if (newScore < oldScore) {
      regressions += 1;
      regressedExamples.push({
        prompt: baseline.results[index]?.prompt ?? "",
        old_score: oldScore,
        new_score: newScore,
      });
    } else if (newScore > oldScore) {
      improvements += 1;
    }
  }
  return {
    avg_score_delta: candidate.avg_score - baseline.avg_score,
    pass_rate_delta: candidate.pass_rate - baseline.pass_rate,
    exact_match_rate_delta: candidate.exact_match_rate - baseline.exact_match_rate,
    regressions,
    improvements,
    regressed_examples: regressedExamples,
  };
}
