import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import {
  evalReportSchema,
  type BehaviorSpecExample,
  type EvalExampleResult,
  type EvalReport,
  type EvalSplit,
  type LocalRunnerConfig,
} from "./contracts.js";
import { writeJson, fileUri } from "./artifacts.js";
import { resolveTrainingModel } from "./model-registry.js";
import { openRouterChat } from "./openrouter.js";
import { buildEntrypointCommand, runJsonStdInCommand, runLoggedProcess } from "./process-runner.js";
import type { LocalRunReporter } from "./run-reporter.js";
import { defaultLocalHome } from "./store.js";

function normalize(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

function scoreActual(expected: string, actual: string): number {
  return normalize(expected) === normalize(actual) ? 1 : 0;
}

/**
 * Token-overlap F1 between expected and actual output (bag-of-words, case and
 * whitespace insensitive). This is a cheap deterministic reference-similarity
 * signal for free-text tasks where exact match is always 0 and an LLM judge
 * can be noisy; both scoring paths keep working unchanged alongside it.
 */
export function tokenF1(expected: string, actual: string): number {
  const tokenize = (value: string): string[] => normalize(value).match(/[\p{L}\p{N}]+/gu) ?? [];
  const expectedTokens = tokenize(expected);
  const actualTokens = tokenize(actual);
  if (expectedTokens.length === 0 || actualTokens.length === 0) {
    return expectedTokens.length === actualTokens.length ? 1 : 0;
  }
  const counts = new Map<string, number>();
  for (const token of expectedTokens) counts.set(token, (counts.get(token) ?? 0) + 1);
  let overlap = 0;
  for (const token of actualTokens) {
    const remaining = counts.get(token) ?? 0;
    if (remaining > 0) {
      overlap += 1;
      counts.set(token, remaining - 1);
    }
  }
  if (overlap === 0) return 0;
  const precision = overlap / actualTokens.length;
  const recall = overlap / expectedTokens.length;
  return (2 * precision * recall) / (precision + recall);
}

/** Derives a deterministic 32-bit seed from an arbitrary string (FNV-1a). */
export function deriveSampleSeed(value: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Selects `count` examples with a seeded shuffle so that truncated evaluation
 * is a deterministic random sample instead of a dataset-order prefix. The
 * selected examples keep their original relative order, so baseline and
 * candidate evaluations that use the same seed score identical examples.
 */
export function sampleExamples<T>(examples: T[], count: number, seed: number): T[] {
  if (count >= examples.length) return examples;
  const indices = examples.map((_, index) => index);
  const random = mulberry32(seed);
  for (let index = indices.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    [indices[index], indices[swap]] = [indices[swap], indices[index]];
  }
  return indices.slice(0, count).sort((left, right) => left - right).map((index) => examples[index]);
}

/**
 * Deterministically splits spec examples into a training split and an eval
 * holdout so that default spec runs do not evaluate on their own training
 * data. The holdout is a seeded random sample of about `holdoutRatio` of the
 * examples, with at least 1 holdout and at least 1 training example. Both
 * splits preserve the original example order. With fewer than 2 examples the
 * holdout is empty and callers should fall back to training-set evaluation.
 */
export function splitSpecExamples<T>(
  examples: T[],
  seed: number,
  holdoutRatio = 0.2,
): { train: T[]; holdout: T[] } {
  if (examples.length < 2) return { train: examples, holdout: [] };
  const holdoutCount = Math.min(
    examples.length - 1,
    Math.max(1, Math.round(examples.length * holdoutRatio)),
  );
  const indices = examples.map((_, index) => index);
  const random = mulberry32(seed);
  for (let index = indices.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    [indices[index], indices[swap]] = [indices[swap], indices[index]];
  }
  const holdoutIndices = new Set(indices.slice(0, holdoutCount));
  return {
    train: examples.filter((_, index) => !holdoutIndices.has(index)),
    holdout: examples.filter((_, index) => holdoutIndices.has(index)),
  };
}

function extractJsonObject(value: string): Record<string, unknown> | null {
  const trimmed = value.trim();
  const candidates = [trimmed];
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fenced?.[1]) candidates.push(fenced[1].trim());
  const objectLike = trimmed.match(/\{[\s\S]*\}/);
  if (objectLike?.[0]) candidates.push(objectLike[0]);

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // Try the next candidate.
    }
  }
  return null;
}

function normalizeJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map((item) => normalizeJsonValue(item));
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, normalizeJsonValue(item)]),
    );
  }
  return value;
}

function jsonValuesEqual(left: unknown, right: unknown): boolean {
  return JSON.stringify(normalizeJsonValue(left)) === JSON.stringify(normalizeJsonValue(right));
}

interface JsonFieldScore {
  score: number;
  passed: boolean;
  reasoning: string;
  actualJsonValid: boolean;
  schemaMatch: boolean;
  fields: string[];
  fieldResults: Record<string, boolean>;
}

function scoreJsonFields(expected: string, actual: string, configuredFields?: string[]): JsonFieldScore {
  const expectedJson = extractJsonObject(expected);
  const actualJson = extractJsonObject(actual);
  if (!expectedJson) {
    const exactScore = scoreActual(expected, actual);
    return {
      score: exactScore,
      passed: exactScore === 1,
      reasoning: "Expected output is not a JSON object; fell back to normalized exact match.",
      actualJsonValid: Boolean(actualJson),
      schemaMatch: false,
      fields: [],
      fieldResults: {},
    };
  }

  const fields = configuredFields?.length
    ? configuredFields
    : Object.keys(expectedJson).sort();
  const expectedKeys = Object.keys(expectedJson).sort();
  const actualKeys = actualJson ? Object.keys(actualJson).sort() : [];
  const schemaMatch = Boolean(actualJson)
    && expectedKeys.length === actualKeys.length
    && expectedKeys.every((key, index) => key === actualKeys[index]);
  const fieldResults: Record<string, boolean> = {};
  const missingExpectedFields: string[] = [];

  for (const field of fields) {
    // A configured field that the expected output does not define cannot be
    // verified, so it must never count as correct.
    if (!Object.prototype.hasOwnProperty.call(expectedJson, field)) {
      missingExpectedFields.push(field);
      fieldResults[field] = false;
      continue;
    }
    fieldResults[field] = actualJson ? jsonValuesEqual(expectedJson[field], actualJson[field]) : false;
  }

  const correct = Object.values(fieldResults).filter(Boolean).length;
  const score = fields.length > 0 ? correct / fields.length : 0;
  const passed = fields.length > 0 && correct === fields.length;
  const missingNote = missingExpectedFields.length > 0
    ? ` Configured fields missing from expected output scored as incorrect: ${missingExpectedFields.join(", ")}.`
    : "";
  return {
    score,
    passed,
    reasoning: (actualJson
      ? `JSON field score: ${correct}/${fields.length} configured fields matched.`
      : "Actual output is not a JSON object.") + missingNote,
    actualJsonValid: Boolean(actualJson),
    schemaMatch,
    fields,
    fieldResults,
  };
}

function aggregateJsonFieldMetrics(scores: JsonFieldScore[], total: number) {
  if (total === 0 || scores.length === 0) return undefined;
  const fields = [...new Set(scores.flatMap((score) => score.fields))].sort();
  const field_accuracy: Record<string, { correct: number; total: number; accuracy: number }> = {};
  for (const field of fields) {
    const scored = scores.filter((score) => score.fields.includes(field));
    const correct = scored.filter((score) => score.fieldResults[field]).length;
    field_accuracy[field] = {
      correct,
      total: scored.length,
      accuracy: scored.length > 0 ? correct / scored.length : 0,
    };
  }
  const validJsonCount = scores.filter((score) => score.actualJsonValid).length;
  const schemaMatchCount = scores.filter((score) => score.schemaMatch).length;
  const allFieldsMatchCount = scores.filter((score) => score.passed).length;
  return {
    fields,
    valid_json_count: validJsonCount,
    valid_json_rate: validJsonCount / total,
    schema_match_count: schemaMatchCount,
    schema_match_rate: schemaMatchCount / total,
    all_fields_match_count: allFieldsMatchCount,
    all_fields_match_rate: allFieldsMatchCount / total,
    field_accuracy,
  };
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
  return runJsonStdInCommand({
    command: args.command,
    payload: {
      system: args.system,
      prompt: args.prompt,
      expected: args.expected,
    },
    timeoutMs: args.timeoutMs,
    timeoutMessage: `Inference command timed out after ${args.timeoutMs}ms`,
    errorPrefix: "Inference command",
  });
}

interface BatchInferenceResult {
  provider: "transformers" | "batch_command";
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

function buildBatchInferenceCommand(config: LocalRunnerConfig, inputPath: string, outputPath: string) {
  if (config.evaluation.inference.provider === "batch_command") {
    return buildEntrypointCommand(
      { ...config.evaluation.inference, backend: "command" },
      { extraArgs: ["--input", inputPath, "--output", outputPath] },
    );
  }
  return buildEntrypointCommand(config.evaluation.inference, {
    defaultScript: "training/sft-local/src/evaluate.py",
    extraArgs: ["--input", inputPath, "--output", outputPath],
  });
}

async function runBatchInference(args: {
  kind: "baseline" | "candidate";
  modelId: string;
  baseModelId: string;
  adapterPath?: string;
  examples: BehaviorSpecExample[];
  system: string;
  config: LocalRunnerConfig;
  outputPath: string;
  reporter?: LocalRunReporter;
}): Promise<BatchInferenceResult> {
  const provider = args.config.evaluation.inference.provider === "batch_command" ? "batch_command" : "transformers";
  const label = provider === "batch_command" ? "batch command" : "Transformers";
  const inputPath = `${args.outputPath}.inference-input.json`;
  const outputPath = `${args.outputPath}.inference-output.json`;
  const logPath = `${args.outputPath}.inference.log`;
  await mkdir(dirname(inputPath), { recursive: true });
  let modelLoader: string | undefined;
  try {
    modelLoader = resolveTrainingModel(args.baseModelId).loader;
  } catch {
    modelLoader = undefined;
  }
  await writeFile(inputPath, `${JSON.stringify({
    kind: args.kind,
    model_id: args.modelId,
    base_model: args.baseModelId,
    model_loader: modelLoader,
    adapter_path: fileUriToPath(args.adapterPath),
    system: args.system,
    examples: args.examples,
    model_cache: args.config.paths.modelCache ? resolve(args.config.paths.modelCache) : undefined,
    trust_remote_code: args.config.evaluation.inference.trustRemoteCode,
    device: args.config.evaluation.inference.device,
    chat_template_kwargs: args.config.evaluation.inference.chatTemplateKwargs,
    generation: {
      max_new_tokens: args.config.evaluation.inference.maxNewTokens,
      temperature: args.config.evaluation.inference.temperature,
      top_p: args.config.evaluation.inference.topP,
    },
  }, null, 2)}\n`, "utf8");

  const entrypoint = buildBatchInferenceCommand(args.config, inputPath, outputPath);
  const command = entrypoint.displayCommand;
  await args.reporter?.onEvent?.({
    stage: `evaluating_${args.kind}`,
    status: "running",
    message: `Starting ${args.kind} ${label} inference.`,
    details: {
      model_id: args.modelId,
      examples: args.examples.length,
      command,
      log_path: logPath,
    },
  });
  const result = await runLoggedProcess({
    command: entrypoint.command,
    commandArgs: entrypoint.commandArgs,
    cwd: args.config.evaluation.inference.cwd,
    env: {
      ...process.env,
      ...args.config.evaluation.inference.env,
    },
    logPath,
    timeoutMs: args.config.evaluation.timeoutMs,
    timeoutMessage: `${label} inference timed out after ${args.config.evaluation.timeoutMs}ms`,
    reporter: args.reporter,
    stage: `evaluating_${args.kind}`,
  });
  if (result.exitCode !== 0) {
    throw new Error(`${label} inference exited ${result.exitCode}: ${result.stderr.slice(0, 1000)}`);
  }
  await args.reporter?.onEvent?.({
    stage: `evaluating_${args.kind}`,
    status: "running",
    message: `Finished ${args.kind} ${label} inference.`,
    details: { output_path: outputPath, log_path: logPath },
  });
  return {
    ...JSON.parse(await readFile(outputPath, "utf8")) as Omit<BatchInferenceResult, "provider">,
    provider,
  };
}

/**
 * Builds the judge messages. The spec's compiled system message (system
 * prompt, guidelines, and constraints) is forwarded as task_instructions so
 * the judge scores conformance to the task, not just similarity to the
 * reference. Without it, a judge treats `expected` as a fact checklist and
 * systematically penalizes outputs trained toward a different style (for
 * example concise summaries) even when they follow the spec.
 */
export function buildJudgeMessages(args: {
  prompt: string;
  expected: string;
  actual: string;
  taskInstructions?: string;
}): Array<{ role: "system" | "user"; content: string }> {
  return [
    {
      role: "system",
      content: "You are a strict evaluator. Score how well the actual output fulfills the task instructions "
        + "for the given prompt. The expected output is a reference answer showing the desired style, length, "
        + "and content; treat it as one correct answer, not an exhaustive checklist. Penalize factual errors, "
        + "contradictions of the prompt, and violations of the task instructions. Do not penalize an output "
        + "merely for omitting secondary reference details when the task instructions call for that brevity. "
        + "Return JSON only with keys score (0 to 1), passed (boolean), and reasoning (string).",
    },
    {
      role: "user",
      content: JSON.stringify({
        ...(args.taskInstructions ? { task_instructions: args.taskInstructions } : {}),
        prompt: args.prompt,
        expected: args.expected,
        actual: args.actual,
      }),
    },
  ];
}

async function judgeWithOpenRouter(args: {
  prompt: string;
  expected: string;
  actual: string;
  taskInstructions?: string;
  config: LocalRunnerConfig;
}): Promise<{ score: number; passed: boolean; reasoning: string; model: string | null }> {
  if (!args.config.llm) {
    throw new Error("evaluation.scoring.mode=llm_judge requires llm OpenRouter config");
  }
  const result = await openRouterChat(buildJudgeMessages(args), {
    model: args.config.llm.model,
    apiKeyEnv: args.config.llm.apiKeyEnv,
    appName: args.config.llm.appName,
    siteUrl: args.config.llm.siteUrl,
    timeoutMs: args.config.evaluation.timeoutMs,
  });
  const parsed = extractJsonObject(result.content) as { score?: unknown; passed?: unknown; reasoning?: unknown } | null;
  if (!parsed) {
    throw new Error(`OpenRouter judge returned malformed JSON: ${result.content.slice(0, 200)}`);
  }
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

export type RegressionCategory = "factual" | "omission" | "style" | "fallback" | "other";

const FACTUAL_REASONING = /incorrect|error|misstate|mis-state|contradict|wrong|inaccurate|not supported|misattribut|invent|fabricat|halluc/i;
const OMISSION_REASONING = /omit|missing|leaves out|leave out|lacks|does not (?:mention|include|cover)|fails to (?:mention|include|cover)/i;
const STYLE_REASONING = /verbose|too long|too short|format|style|tone|first person|markdown|preamble/i;

/**
 * Coarse category for a judge reasoning string so comparison reports can
 * answer "what kind of worse?" without re-reading every example. Factual
 * problems dominate omissions, which dominate style notes; fallback marks
 * examples that were never judge-scored (their score is not comparable to
 * judged ones).
 */
export function classifyJudgeReasoning(reasoning: string | null, scoredBy?: string): RegressionCategory {
  if (scoredBy === "exact_match_fallback") return "fallback";
  if (!reasoning) return "other";
  if (reasoning.startsWith("LLM judge failed")) return "fallback";
  if (FACTUAL_REASONING.test(reasoning)) return "factual";
  if (OMISSION_REASONING.test(reasoning)) return "omission";
  if (STYLE_REASONING.test(reasoning)) return "style";
  return "other";
}

/**
 * Cache key for a baseline evaluation. Baseline outputs are deterministic for
 * a given model, example set, and generation settings, and judge scores only
 * depend on those plus the scoring configuration, so re-running a spec with
 * an unchanged baseline can reuse the previous report instead of paying for
 * inference and judge calls again. The package version participates so rubric
 * or evaluator changes invalidate old entries.
 */
export function baselineCacheKey(args: {
  modelId: string;
  system: string;
  examples: BehaviorSpecExample[];
  config: LocalRunnerConfig;
  packageVersion: string;
}): string {
  const payload = {
    v: 1,
    package_version: args.packageVersion,
    model_id: args.modelId,
    system: args.system,
    examples: args.examples.map((example) => ({
      input: example.input,
      output: example.output,
      assets: example.input_assets ?? null,
    })),
    // The full inference config participates: a different evaluator script,
    // project, or generation setting must produce a different cache entry.
    inference: args.config.evaluation.inference,
    scoring: {
      mode: args.config.evaluation.scoring.mode,
      fields: args.config.evaluation.scoring.fields ?? null,
      judgeModel: args.config.llm?.model ?? null,
    },
  };
  return createHash("sha256").update(JSON.stringify(payload)).digest("hex");
}

function baselineCachePath(config: LocalRunnerConfig, key: string): string {
  const root = config.storeRoot ? resolve(config.storeRoot) : defaultLocalHome();
  return join(root, "cache", "baseline-evals", `${key}.json`);
}

async function readBaselineCache(path: string): Promise<EvalReport | null> {
  try {
    return evalReportSchema.parse(JSON.parse(await readFile(path, "utf8")));
  } catch {
    return null;
  }
}

let cachedPackageVersion: string | null = null;

async function packageVersion(): Promise<string> {
  if (cachedPackageVersion === null) {
    try {
      const raw = await readFile(new URL("../package.json", import.meta.url), "utf8");
      cachedPackageVersion = String((JSON.parse(raw) as { version?: unknown }).version ?? "unknown");
    } catch {
      cachedPackageVersion = "unknown";
    }
  }
  return cachedPackageVersion;
}

function resolveEvaluationRuntime(args: {
  kind: "baseline" | "candidate";
  config: LocalRunnerConfig;
}): {
  command?: string[];
  inferenceProvider: "none" | "command" | "batch_command" | "transformers";
  scoringMode: "exact_match" | "llm_judge" | "json_fields";
} {
  const command = args.kind === "baseline"
    ? args.config.evaluation.baselineCommand
    : args.config.evaluation.candidateCommand;
  const provider = args.config.evaluation.inference.provider;
  const inferenceProvider: "none" | "command" | "batch_command" | "transformers" =
    args.config.dryRun
      ? "none"
      : provider === "command"
        ? "command"
        : provider;
  if (inferenceProvider === "command" && !command) {
    throw new Error(`evaluation.inference.provider=command requires evaluation.${args.kind}Command`);
  }
  if (inferenceProvider === "batch_command" && !args.config.evaluation.inference.command) {
    throw new Error("evaluation.inference.provider=batch_command requires evaluation.inference.command");
  }
  const configuredScoringMode: "exact_match" | "llm_judge" | "json_fields" =
    args.config.evaluation.scoring.mode;
  const scoringMode: "exact_match" | "llm_judge" | "json_fields" =
    (args.config.dryRun || inferenceProvider === "none") && configuredScoringMode === "llm_judge"
      ? "exact_match"
      : configuredScoringMode;
  return { command, inferenceProvider, scoringMode };
}

interface GeneratedEvalResult {
  prompt: string;
  expected: string;
  actual: string;
  latency_ms: number;
}

async function scoreGeneratedEvalResults(args: {
  kind: "baseline" | "candidate";
  modelId: string;
  generated: GeneratedEvalResult[];
  evalExamplesTotal: number;
  config: LocalRunnerConfig;
  outputPath: string;
  scoringMode: "exact_match" | "llm_judge" | "json_fields";
  inferenceProvider: "none" | "command" | "batch_command" | "transformers";
  system?: string;
  evalSplit?: EvalSplit;
  evalSampleSeed?: number | null;
  generationConfig?: Record<string, unknown>;
  logUri?: string;
  reporter?: LocalRunReporter;
}): Promise<EvalReport> {
  const results: EvalExampleResult[] = [];
  const jsonFieldScores: JsonFieldScore[] = [];
  let judgeModelId: string | null = null;
  const shouldJudge = args.scoringMode === "llm_judge" && canUseOpenRouterJudge(args.config);
  if (args.scoringMode === "llm_judge" && !shouldJudge && args.config.evaluation.scoring.fallback === "fail") {
    const keyName = args.config.llm?.apiKeyEnv ?? "OPENROUTER_API_KEY";
    throw new Error(`evaluation.scoring.mode=llm_judge requires ${keyName} or scoring.fallback=exact_match`);
  }

  for (const [index, generated] of args.generated.entries()) {
    const exactScore = scoreActual(generated.expected, generated.actual);
    if (shouldJudge && index === 0) {
      await args.reporter?.onEvent?.({
        stage: `evaluating_${args.kind}`,
        status: "running",
        message: `Scoring ${args.kind} outputs with OpenRouter judge.`,
        details: { model: args.config.llm?.model, examples: args.generated.length },
      });
    }
    let judged: { score: number; passed: boolean; reasoning: string; model: string | null } | null = null;
    let judgeFellBack = false;
    if (shouldJudge) {
      try {
        judged = await judgeWithOpenRouter({
          prompt: generated.prompt,
          expected: generated.expected,
          actual: generated.actual,
          taskInstructions: args.system?.trim() || undefined,
          config: args.config,
        });
      } catch (error) {
        if (args.config.evaluation.scoring.fallback === "fail") throw error;
        const message = error instanceof Error ? error.message : String(error);
        judgeFellBack = true;
        judged = {
          score: exactScore,
          passed: exactScore === 1,
          reasoning: `LLM judge failed (${message.slice(0, 300)}); scored by normalized exact match.`,
          model: null,
        };
      }
    }
    if (judged?.model) judgeModelId = judged.model;
    const jsonFieldScore = args.scoringMode === "json_fields"
      ? scoreJsonFields(generated.expected, generated.actual, args.config.evaluation.scoring.fields)
      : null;
    if (jsonFieldScore) jsonFieldScores.push(jsonFieldScore);
    const score = judged?.score ?? jsonFieldScore?.score ?? exactScore;
    const scoredBy: EvalExampleResult["scored_by"] = judged
      ? (judgeFellBack ? "exact_match_fallback" : "llm_judge")
      : jsonFieldScore
        ? "json_fields"
        : args.inferenceProvider === "none"
          ? "heuristic"
          : "exact_match";
    results.push({
      prompt: generated.prompt,
      expected: generated.expected,
      actual: generated.actual,
      passed: judged?.passed ?? jsonFieldScore?.passed ?? score === 1,
      score,
      reasoning: judged?.reasoning ?? jsonFieldScore?.reasoning ?? (args.inferenceProvider === "none"
        ? "No inference command configured; recorded an empty local response."
        : args.scoringMode === "llm_judge"
          ? "OpenRouter judge unavailable; fell back to normalized exact match."
          : "Scored by normalized exact match."),
      latency_ms: generated.latency_ms,
      scored_by: scoredBy,
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
  const avgTokenF1 = total > 0
    ? results.reduce((sum, result) => sum + tokenF1(result.expected, result.actual), 0) / total
    : 0;
  const judgeScored = results.filter((result) => result.scored_by === "llm_judge");
  const fallbackScoredCount = results.filter((result) => result.scored_by === "exact_match_fallback").length;
  const judgeOnlyAvgScore = judgeScored.length > 0
    ? judgeScored.reduce((sum, result) => sum + result.score, 0) / judgeScored.length
    : null;
  const avgLatency = total > 0
    ? Math.round(results.reduce((sum, result) => sum + result.latency_ms, 0) / total)
    : 0;
  const scoringMethod = judgeModelId
    ? "llm_judge"
    : args.scoringMode === "json_fields"
      ? "json_fields"
      : args.inferenceProvider === "command"
        ? "command"
        : args.inferenceProvider === "none"
          ? "heuristic"
          : "exact_match";
  const report: EvalReport = {
    kind: args.kind,
    model_id: args.modelId,
    total,
    eval_examples_total: args.evalExamplesTotal,
    eval_examples_used: total,
    eval_truncated: args.evalExamplesTotal > total,
    eval_split: args.evalSplit,
    eval_sample_seed: args.evalSampleSeed ?? null,
    avg_score: avgScore,
    pass_rate: passRate,
    exact_match_rate: exactMatchRate,
    avg_token_f1: avgTokenF1,
    avg_latency_ms: avgLatency,
    judge_scored_count: judgeScored.length,
    fallback_scored_count: fallbackScoredCount,
    judge_only_avg_score: judgeOnlyAvgScore,
    results,
    artifact_uri: fileUri(args.outputPath),
    scoring_method: scoringMethod,
    judge_model_id: judgeModelId,
    inference_provider: args.inferenceProvider,
    scoring_mode: args.scoringMode,
    json_field_metrics: args.scoringMode === "json_fields"
      ? aggregateJsonFieldMetrics(jsonFieldScores, total)
      : undefined,
    generation_config: args.generationConfig,
    log_uri: args.logUri,
  };
  await writeJson(args.outputPath, report);
  return report;
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
  maxExamples?: number;
  evalSplit?: EvalSplit;
  sampleSeed?: number;
}): Promise<EvalReport> {
  // Explicit config takes precedence over the per-run request hyperparameter
  // (passed by the orchestrator via args.maxExamples).
  const maxExamples = args.config.evaluation.maxExamples
    ?? args.maxExamples
    ?? args.examples.length;
  const truncated = args.examples.length > maxExamples;
  // When truncating, take a deterministic seeded sample (not a prefix) so a
  // sorted or grouped eval file does not bias the evaluated subset. Baseline
  // and candidate runs receive the same seed and therefore identical examples.
  const sampleSeed = args.config.evaluation.sampleSeed ?? args.sampleSeed ?? 0;
  const selected = truncated
    ? sampleExamples(args.examples, maxExamples, sampleSeed)
    : args.examples;
  const { command, inferenceProvider, scoringMode } = resolveEvaluationRuntime({
    kind: args.kind,
    config: args.config,
  });
  let generationConfig: Record<string, unknown> | undefined;

  // Baseline outputs and scores are fully determined by the cache key inputs,
  // so an unchanged baseline can reuse the previous report instead of paying
  // for inference and judge calls on every run of the same spec.
  const cacheEligible = args.kind === "baseline"
    && args.config.evaluation.baselineCache
    && !args.config.dryRun
    && inferenceProvider === "transformers";
  const cacheKey = cacheEligible
    ? baselineCacheKey({
        modelId: args.modelId,
        system: args.system,
        examples: selected,
        config: args.config,
        packageVersion: await packageVersion(),
      })
    : null;
  if (cacheKey) {
    const cached = await readBaselineCache(baselineCachePath(args.config, cacheKey));
    if (cached) {
      const report: EvalReport = {
        ...cached,
        cached: true,
        cache_key: cacheKey,
        artifact_uri: fileUri(args.outputPath),
        eval_split: args.evalSplit ?? cached.eval_split,
      };
      await writeJson(args.outputPath, report);
      await args.reporter?.onEvent?.({
        stage: `evaluating_${args.kind}`,
        status: "running",
        message: "Reusing cached baseline evaluation (identical model, examples, and scoring).",
        details: { cache_key: cacheKey, examples: report.total },
      });
      return report;
    }
  }

  const inferred = inferenceProvider === "transformers" || inferenceProvider === "batch_command"
    ? await runBatchInference({
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

  const generated: GeneratedEvalResult[] = [];
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
    const latencyMs = inferredResult?.latency_ms ?? Math.max(0, Math.round(performance.now() - started));
    generated.push({
      prompt: example.input,
      expected: example.output,
      actual,
      latency_ms: latencyMs,
    });
  }

  const report = await scoreGeneratedEvalResults({
    kind: args.kind,
    modelId: args.modelId,
    generated,
    evalExamplesTotal: args.examples.length,
    config: args.config,
    outputPath: args.outputPath,
    scoringMode,
    inferenceProvider,
    system: args.system,
    evalSplit: args.evalSplit,
    evalSampleSeed: truncated ? sampleSeed : null,
    generationConfig,
    logUri: inferenceProvider === "transformers" || inferenceProvider === "batch_command"
      ? fileUri(`${args.outputPath}.inference.log`)
      : undefined,
    reporter: args.reporter,
  });
  if (cacheKey) {
    // Do not cache reports with fallback-scored examples: a transient judge
    // failure would otherwise be replayed into every future run.
    if ((report.fallback_scored_count ?? 0) === 0) {
      await writeJson(baselineCachePath(args.config, cacheKey), { ...report, cache_key: cacheKey });
    }
  }
  return report;
}

export async function rescoreEvalReport(args: {
  report: EvalReport;
  config: LocalRunnerConfig;
  outputPath: string;
  system?: string;
  reporter?: LocalRunReporter;
}): Promise<EvalReport> {
  const configuredScoringMode = args.config.evaluation.scoring.mode;
  const scoringMode: "exact_match" | "llm_judge" | "json_fields" =
    args.config.dryRun && configuredScoringMode === "llm_judge"
      ? "exact_match"
      : configuredScoringMode;
  const inferenceProvider = args.report.inference_provider ?? "none";
  return scoreGeneratedEvalResults({
    kind: args.report.kind,
    modelId: args.report.model_id,
    generated: args.report.results.map((result) => ({
      prompt: result.prompt,
      expected: result.expected,
      actual: result.actual,
      latency_ms: result.latency_ms,
    })),
    evalExamplesTotal: args.report.eval_examples_total,
    config: args.config,
    outputPath: args.outputPath,
    scoringMode,
    inferenceProvider,
    system: args.system,
    evalSplit: args.report.eval_split,
    evalSampleSeed: args.report.eval_sample_seed ?? null,
    generationConfig: args.report.generation_config,
    logUri: args.report.log_uri,
    reporter: args.reporter,
  });
}

export function compareEvalReports(baseline: EvalReport, candidate: EvalReport) {
  let regressions = 0;
  let improvements = 0;
  const regressedExamples = [];
  // All categories are always present: zod v4 enum-keyed records are
  // exhaustive, so a partial taxonomy fails runReportSchema validation.
  const taxonomy: Record<RegressionCategory, number> = {
    factual: 0,
    omission: 0,
    style: 0,
    fallback: 0,
    other: 0,
  };
  const count = Math.min(baseline.results.length, candidate.results.length);
  for (let index = 0; index < count; index += 1) {
    const oldScore = baseline.results[index]?.score ?? 0;
    const newResult = candidate.results[index];
    const newScore = newResult?.score ?? 0;
    if (newScore < oldScore) {
      regressions += 1;
      const category = classifyJudgeReasoning(newResult?.reasoning ?? null, newResult?.scored_by);
      taxonomy[category] = (taxonomy[category] ?? 0) + 1;
      regressedExamples.push({
        prompt: baseline.results[index]?.prompt ?? "",
        old_score: oldScore,
        new_score: newScore,
        category,
      });
    } else if (newScore > oldScore) {
      improvements += 1;
    }
  }
  const judgeOnlyDelta = typeof baseline.judge_only_avg_score === "number"
    && typeof candidate.judge_only_avg_score === "number"
    ? candidate.judge_only_avg_score - baseline.judge_only_avg_score
    : null;
  return {
    avg_score_delta: candidate.avg_score - baseline.avg_score,
    pass_rate_delta: candidate.pass_rate - baseline.pass_rate,
    exact_match_rate_delta: candidate.exact_match_rate - baseline.exact_match_rate,
    token_f1_delta: (candidate.avg_token_f1 ?? 0) - (baseline.avg_token_f1 ?? 0),
    judge_only_avg_score_delta: judgeOnlyDelta,
    regressions,
    improvements,
    regression_taxonomy: regressions > 0 ? taxonomy : undefined,
    regressed_examples: regressedExamples,
  };
}
