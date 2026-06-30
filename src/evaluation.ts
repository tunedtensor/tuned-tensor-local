import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import type {
  BehaviorSpecExample,
  EvalExampleResult,
  EvalReport,
  LocalRunnerConfig,
} from "./contracts.js";
import { writeJson, fileUri } from "./artifacts.js";
import { openRouterChat } from "./openrouter.js";
import { forwardStreamLines, type LocalRunReporter } from "./run-reporter.js";

function normalize(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

function scoreActual(expected: string, actual: string): number {
  return normalize(expected) === normalize(actual) ? 1 : 0;
}

function fileUriToPath(value?: string): string | undefined {
  if (!value) return undefined;
  return value.startsWith("file://") ? value.slice("file://".length) : value;
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

interface TransformersInferenceResult {
  provider: "transformers";
  model_id?: string;
  base_model?: string;
  adapter_path?: string;
  generation_config?: Record<string, unknown>;
  results: Array<{
    prompt: string;
    expected: string;
    actual: string;
    latency_ms: number;
  }>;
}

function buildUvInferenceArgs(config: LocalRunnerConfig, inputPath: string, outputPath: string): string[] {
  const inference = config.evaluation.inference;
  const args: string[] = ["run"];
  if (inference.project) args.push("--project", inference.project);
  for (const dependency of inference.with ?? []) args.push("--with", dependency);
  args.push("python");
  if (inference.module) {
    args.push("-m", inference.module);
  } else {
    args.push(inference.script);
  }
  args.push("--input", inputPath, "--output", outputPath);
  return args;
}

async function runTransformersInference(args: {
  kind: "baseline" | "candidate";
  modelId: string;
  baseModelId: string;
  adapterPath?: string;
  examples: BehaviorSpecExample[];
  system: string;
  config: LocalRunnerConfig;
  outputPath: string;
  reporter?: LocalRunReporter;
}): Promise<TransformersInferenceResult> {
  const inputPath = `${args.outputPath}.inference-input.json`;
  const outputPath = `${args.outputPath}.inference-output.json`;
  const logPath = `${args.outputPath}.inference.log`;
  await mkdir(dirname(inputPath), { recursive: true });
  await writeFile(inputPath, `${JSON.stringify({
    kind: args.kind,
    model_id: args.modelId,
    base_model: args.baseModelId,
    adapter_path: fileUriToPath(args.adapterPath),
    system: args.system,
    examples: args.examples,
    model_cache: args.config.paths.modelCache ? resolve(args.config.paths.modelCache) : undefined,
    trust_remote_code: args.config.evaluation.inference.trustRemoteCode,
    device: args.config.evaluation.inference.device,
    generation: {
      max_new_tokens: args.config.evaluation.inference.maxNewTokens,
      temperature: args.config.evaluation.inference.temperature,
      top_p: args.config.evaluation.inference.topP,
    },
  }, null, 2)}\n`, "utf8");

  const uvArgs = buildUvInferenceArgs(args.config, inputPath, outputPath);
  const command = ["uv", ...uvArgs];
  await args.reporter?.onEvent?.({
    stage: `evaluating_${args.kind}`,
    status: "running",
    message: `Starting ${args.kind} Transformers inference.`,
    details: {
      model_id: args.modelId,
      examples: args.examples.length,
      command,
      log_path: logPath,
    },
  });
  const logStream = createWriteStream(logPath, { flags: "w" });
  try {
    await new Promise<void>((resolvePromise, reject) => {
      const child = spawn("uv", uvArgs, {
        cwd: args.config.evaluation.inference.cwd ? resolve(args.config.evaluation.inference.cwd) : process.cwd(),
        stdio: ["ignore", "pipe", "pipe"],
        env: {
          ...process.env,
          ...args.config.evaluation.inference.env,
        },
      });
      let stderr = "";
      const timer = setTimeout(() => {
        child.kill("SIGTERM");
        reject(new Error(`Transformers inference timed out after ${args.config.evaluation.timeoutMs}ms`));
      }, args.config.evaluation.timeoutMs);
      child.stdout.pipe(logStream, { end: false });
      child.stderr.pipe(logStream, { end: false });
      forwardStreamLines(child.stdout, (line) => {
        if (args.reporter?.verbose) void args.reporter.onLog?.({ stage: `evaluating_${args.kind}`, stream: "stdout", message: line });
      });
      forwardStreamLines(child.stderr, (line) => {
        stderr += `${line}\n`;
        if (args.reporter?.verbose) void args.reporter.onLog?.({ stage: `evaluating_${args.kind}`, stream: "stderr", message: line });
      });
      child.on("error", (error) => {
        clearTimeout(timer);
        reject(error);
      });
      child.on("close", (code) => {
        clearTimeout(timer);
        if (code !== 0) {
          reject(new Error(`Transformers inference exited ${code}: ${stderr.slice(0, 1000)}`));
          return;
        }
        resolvePromise();
      });
    });
  } finally {
    await new Promise<void>((resolveEnd) => {
      logStream.end(resolveEnd);
    });
  }
  await args.reporter?.onEvent?.({
    stage: `evaluating_${args.kind}`,
    status: "running",
    message: `Finished ${args.kind} Transformers inference.`,
    details: { output_path: outputPath, log_path: logPath },
  });
  return JSON.parse(await readFile(outputPath, "utf8")) as TransformersInferenceResult;
}

async function judgeWithOpenRouter(args: {
  prompt: string;
  expected: string;
  actual: string;
  config: LocalRunnerConfig;
}): Promise<{ score: number; passed: boolean; reasoning: string; model: string | null }> {
  if (!args.config.llm) {
    throw new Error("evaluation.mode=llm_judge requires llm OpenRouter config");
  }
  const result = await openRouterChat([
    {
      role: "system",
      content: "You are a strict evaluator. Return JSON only with keys score (0 to 1), passed (boolean), and reasoning (string).",
    },
    {
      role: "user",
      content: JSON.stringify({
        prompt: args.prompt,
        expected: args.expected,
        actual: args.actual,
      }),
    },
  ], {
    model: args.config.llm.model,
    apiKeyEnv: args.config.llm.apiKeyEnv,
    appName: args.config.llm.appName,
    siteUrl: args.config.llm.siteUrl,
    timeoutMs: args.config.evaluation.timeoutMs,
  });
  const parsed = JSON.parse(result.content) as { score?: unknown; passed?: unknown; reasoning?: unknown };
  const score = typeof parsed.score === "number"
    ? Math.max(0, Math.min(1, parsed.score))
    : 0;
  return {
    score,
    passed: typeof parsed.passed === "boolean" ? parsed.passed : score === 1,
    reasoning: typeof parsed.reasoning === "string" ? parsed.reasoning : "OpenRouter judge returned no reasoning.",
    model: result.model ?? args.config.llm.model,
  };
}

function canUseOpenRouterJudge(config: LocalRunnerConfig): boolean {
  return Boolean(config.llm && process.env[config.llm.apiKeyEnv]);
}

export async function evaluateExamples(args: {
  kind: "baseline" | "candidate";
  modelId: string;
  baseModelId?: string;
  adapterPath?: string;
  examples: BehaviorSpecExample[];
  system: string;
  config: LocalRunnerConfig;
  outputPath: string;
  reporter?: LocalRunReporter;
}): Promise<EvalReport> {
  const maxExamples = args.config.evaluation.maxExamples ?? args.examples.length;
  const selected = args.examples.slice(0, maxExamples);
  const command = args.kind === "baseline"
    ? args.config.evaluation.baselineCommand
    : args.config.evaluation.candidateCommand;
  const inferenceProvider: "none" | "command" | "transformers" =
    args.config.dryRun
      ? "none"
      : args.config.evaluation.mode === "command" && command
        ? "command"
        : args.config.evaluation.inference.provider;
  const scoringMode: "exact_match" | "llm_judge" =
    args.config.dryRun || inferenceProvider === "none" || inferenceProvider === "command"
      ? "exact_match"
      : args.config.evaluation.mode === "llm_judge"
        ? "llm_judge"
        : args.config.evaluation.scoring.mode;
  const results: EvalExampleResult[] = [];
  let judgeModelId: string | null = null;
  let generationConfig: Record<string, unknown> | undefined;

  const inferred = inferenceProvider === "transformers"
    ? await runTransformersInference({
        kind: args.kind,
        modelId: args.modelId,
        baseModelId: args.baseModelId ?? args.modelId,
        adapterPath: args.adapterPath,
        examples: selected,
        system: args.system,
        config: args.config,
        outputPath: args.outputPath,
        reporter: args.reporter,
      })
    : null;
  generationConfig = inferred?.generation_config;

  for (const [index, example] of selected.entries()) {
    const started = performance.now();
    const commandActual = inferenceProvider === "command" && command
      ? await runInferenceCommand({
          command,
          prompt: example.input,
          system: args.system,
          expected: example.output,
          timeoutMs: args.config.evaluation.timeoutMs,
        })
      : undefined;
    const inferredResult = inferred?.results[index];
    const actual = commandActual ?? inferredResult?.actual ?? "";
    const exactScore = scoreActual(example.output, actual);
    const shouldJudge = scoringMode === "llm_judge" && canUseOpenRouterJudge(args.config);
    if (scoringMode === "llm_judge" && !shouldJudge && args.config.evaluation.scoring.fallback === "fail") {
      const keyName = args.config.llm?.apiKeyEnv ?? "OPENROUTER_API_KEY";
      throw new Error(`evaluation.scoring.mode=llm_judge requires ${keyName} or scoring.fallback=exact_match`);
    }
    if (shouldJudge && index === 0) {
      await args.reporter?.onEvent?.({
        stage: `evaluating_${args.kind}`,
        status: "running",
        message: `Scoring ${args.kind} outputs with OpenRouter judge.`,
        details: { model: args.config.llm?.model, examples: selected.length },
      });
    }
    const judged = shouldJudge
      ? await judgeWithOpenRouter({
          prompt: example.input,
          expected: example.output,
          actual,
          config: args.config,
        })
      : null;
    if (judged?.model) judgeModelId = judged.model;
    const latencyMs = inferredResult?.latency_ms ?? Math.max(0, Math.round(performance.now() - started));
    const score = judged?.score ?? exactScore;
    results.push({
      prompt: example.input,
      expected: example.output,
      actual,
      passed: judged?.passed ?? score === 1,
      score,
      reasoning: judged?.reasoning ?? (inferenceProvider === "none"
        ? "No inference command configured; recorded an empty local response."
        : scoringMode === "llm_judge"
          ? "OpenRouter judge unavailable; fell back to normalized exact match."
          : "Scored by normalized exact match."),
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
  const exactMatchRate = total > 0
    ? results.filter((result) => scoreActual(result.expected, result.actual) === 1).length / total
    : 0;
  const avgLatency = total > 0
    ? Math.round(results.reduce((sum, result) => sum + result.latency_ms, 0) / total)
    : 0;
  const scoringMethod = judgeModelId
    ? "llm_judge"
    : inferenceProvider === "command"
      ? "command"
      : inferenceProvider === "none"
        ? "heuristic"
        : "exact_match";
  const report: EvalReport = {
    kind: args.kind,
    model_id: args.modelId,
    total,
    eval_examples_total: args.examples.length,
    eval_examples_used: total,
    eval_truncated: args.examples.length > total,
    avg_score: avgScore,
    pass_rate: passRate,
    exact_match_rate: exactMatchRate,
    avg_latency_ms: avgLatency,
    results,
    artifact_uri: fileUri(args.outputPath),
    scoring_method: scoringMethod,
    judge_model_id: judgeModelId,
    inference_provider: inferenceProvider,
    scoring_mode: scoringMode,
    generation_config: generationConfig,
    log_uri: inferenceProvider === "transformers" ? fileUri(`${args.outputPath}.inference.log`) : undefined,
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
