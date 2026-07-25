import { createHash } from "node:crypto";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import {
  evalReportSchema,
  type BehaviorSpecExample,
  type EvalExampleResult,
  type EvalReport,
  type EvalSplit,
  type LocalRunnerConfig,
} from "./contracts.js";
import { fileUri, writeJson } from "./artifacts.js";
import {
  buildBundledPythonCommand,
  runLoggedProcess,
  withBundledPythonEnvironment,
} from "./process-runner.js";
import type { LocalRunReporter } from "./run-reporter.js";
import { defaultLocalHome } from "./store.js";
import {
  minimalMachineLearningEnvironment,
  withOfflineHuggingFaceCacheEnvironment,
} from "./huggingface-cache.js";

function normalize(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

function exactScore(expected: string, actual: string): number {
  return normalize(expected) === normalize(actual) ? 1 : 0;
}

/** Cheap deterministic free-text similarity reported beside exact match. */
export function tokenF1(expected: string, actual: string): number {
  const tokenize = (value: string): string[] =>
    normalize(value).match(/[\p{L}\p{N}]+/gu) ?? [];
  const expectedTokens = tokenize(expected);
  const actualTokens = tokenize(actual);
  if (expectedTokens.length === 0 || actualTokens.length === 0) {
    return expectedTokens.length === actualTokens.length ? 1 : 0;
  }
  const counts = new Map<string, number>();
  for (const token of expectedTokens) {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }
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
  return 2 * precision * recall / (precision + recall);
}

/** Deterministic 32-bit FNV-1a seed. */
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
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function sampleExamples<T>(examples: T[], count: number, seed: number): T[] {
  if (count >= examples.length) return examples;
  const indices = examples.map((_, index) => index);
  const random = mulberry32(seed);
  for (let index = indices.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    [indices[index], indices[swap]] = [indices[swap], indices[index]];
  }
  return indices
    .slice(0, count)
    .sort((left, right) => left - right)
    .map((index) => examples[index]);
}

/**
 * Inline specs get a deterministic holdout. A real run with fewer than two
 * examples is rejected by the shared validator before this function is used.
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
  const heldOut = new Set(indices.slice(0, holdoutCount));
  return {
    train: examples.filter((_, index) => !heldOut.has(index)),
    holdout: examples.filter((_, index) => heldOut.has(index)),
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
      // Try the next representation.
    }
  }
  return null;
}

function canonicalJson(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalJson);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, canonicalJson(item)]),
    );
  }
  return value;
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

function scoreJsonFields(
  expected: string,
  actual: string,
  configuredFields?: string[],
): JsonFieldScore {
  const expectedJson = extractJsonObject(expected);
  const actualJson = extractJsonObject(actual);
  if (!expectedJson) {
    const score = exactScore(expected, actual);
    return {
      score,
      passed: score === 1,
      reasoning: "Expected output is not JSON; used normalized exact match.",
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
  for (const field of fields) {
    fieldResults[field] = Object.prototype.hasOwnProperty.call(expectedJson, field)
      && Boolean(actualJson)
      && JSON.stringify(canonicalJson(expectedJson[field]))
        === JSON.stringify(canonicalJson(actualJson?.[field]));
  }
  const correct = Object.values(fieldResults).filter(Boolean).length;
  const score = fields.length > 0 ? correct / fields.length : 0;
  return {
    score,
    passed: fields.length > 0 && correct === fields.length,
    reasoning: actualJson
      ? `JSON field score: ${correct}/${fields.length} fields matched.`
      : "Actual output is not a JSON object.",
    actualJsonValid: Boolean(actualJson),
    schemaMatch,
    fields,
    fieldResults,
  };
}

function aggregateJsonFieldMetrics(scores: JsonFieldScore[], total: number) {
  if (total === 0 || scores.length === 0) return undefined;
  const fields = [...new Set(scores.flatMap((score) => score.fields))].sort();
  const field_accuracy: Record<string, {
    correct: number;
    total: number;
    accuracy: number;
  }> = {};
  for (const field of fields) {
    const applicable = scores.filter((score) => score.fields.includes(field));
    const correct = applicable.filter((score) => score.fieldResults[field]).length;
    field_accuracy[field] = {
      correct,
      total: applicable.length,
      accuracy: applicable.length > 0 ? correct / applicable.length : 0,
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

export const INFERENCE_PROTOCOL_VERSION = 2;

interface BatchInferenceResult {
  model_id?: string;
  base_model?: string;
  adapter_path?: string;
  generation_config?: Record<string, unknown>;
  results: Array<{ id: string; actual: string; latency_ms: number }>;
}

function parseBatchInferenceResult(
  value: unknown,
  expectedIds: string[],
): BatchInferenceResult {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Transformers inference output must be a JSON object");
  }
  const output = value as Record<string, unknown>;
  if (!Array.isArray(output.results) || output.results.length !== expectedIds.length) {
    throw new Error(
      `Transformers inference returned ${Array.isArray(output.results) ? output.results.length : 0} `
      + `predictions; expected ${expectedIds.length}`,
    );
  }
  const expected = new Set(expectedIds);
  const byId = new Map<string, BatchInferenceResult["results"][number]>();
  for (const [index, entry] of output.results.entries()) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      throw new Error(`Transformers prediction ${index} must be an object`);
    }
    const prediction = entry as Record<string, unknown>;
    if (typeof prediction.id !== "string" || !expected.has(prediction.id)) {
      throw new Error(`Transformers prediction ${index} has an unknown or missing id`);
    }
    if (byId.has(prediction.id)) {
      throw new Error(`Transformers inference returned duplicate id ${prediction.id}`);
    }
    if (typeof prediction.actual !== "string") {
      throw new Error(`Transformers prediction ${index} must include string actual`);
    }
    if (
      typeof prediction.latency_ms !== "number"
      || !Number.isInteger(prediction.latency_ms)
      || prediction.latency_ms < 0
    ) {
      throw new Error(
        `Transformers prediction ${index} must include non-negative integer latency_ms`,
      );
    }
    byId.set(prediction.id, {
      id: prediction.id,
      actual: prediction.actual,
      latency_ms: prediction.latency_ms,
    });
  }
  const optionalString = (key: string) =>
    typeof output[key] === "string" ? output[key] as string : undefined;
  return {
    ...(optionalString("model_id") ? { model_id: optionalString("model_id") } : {}),
    ...(optionalString("base_model") ? { base_model: optionalString("base_model") } : {}),
    ...(optionalString("adapter_path") ? { adapter_path: optionalString("adapter_path") } : {}),
    ...(output.generation_config
      && typeof output.generation_config === "object"
      && !Array.isArray(output.generation_config)
      ? { generation_config: output.generation_config as Record<string, unknown> }
      : {}),
    results: expectedIds.map((id) => byId.get(id)!),
  };
}

async function runTransformersInference(args: {
  kind: "baseline" | "candidate";
  modelId: string;
  baseModelId: string;
  baseModelRevision?: string;
  adapterPath?: string;
  examples: BehaviorSpecExample[];
  system: string;
  config: LocalRunnerConfig;
  outputPath: string;
  reporter?: LocalRunReporter;
  shouldCancel?: () => boolean | Promise<boolean>;
}): Promise<BatchInferenceResult> {
  const inputPath = `${args.outputPath}.inference-input.json`;
  const outputPath = `${args.outputPath}.inference-output.json`;
  const logPath = `${args.outputPath}.inference.log`;
  await mkdir(dirname(inputPath), { recursive: true });
  await writeFile(inputPath, `${JSON.stringify({
    protocol_version: INFERENCE_PROTOCOL_VERSION,
    kind: args.kind,
    model_id: args.modelId,
    base_model: args.baseModelId,
    base_model_revision: args.baseModelRevision,
    model_loader: "causal_lm",
    adapter_path: fileUriToPath(args.adapterPath),
    system: args.system,
    examples: args.examples.map((example, index) => ({
      id: String(index),
      input: example.input,
    })),
    model_cache: args.config.paths.modelCache
      ? resolve(args.config.paths.modelCache)
      : undefined,
    trust_remote_code: false,
    device: args.config.evaluation.inference.device,
    generation: {
      max_new_tokens: args.config.evaluation.inference.maxNewTokens,
      temperature: args.config.evaluation.inference.temperature,
      top_p: args.config.evaluation.inference.topP,
    },
  }, null, 2)}\n`, "utf8");
  await rm(outputPath, { force: true });
  const entrypoint = buildBundledPythonCommand(
    "evaluate.py",
    ["--input", inputPath, "--output", outputPath],
  );
  await args.reporter?.onEvent?.({
    stage: `evaluating_${args.kind}`,
    status: "running",
    message: `Starting ${args.kind} Transformers inference.`,
    details: {
      model_id: args.modelId,
      examples: args.examples.length,
      command: entrypoint.displayCommand,
      log_path: logPath,
    },
  });
  const result = await runLoggedProcess({
    command: entrypoint.command,
    commandArgs: entrypoint.commandArgs,
    env: withBundledPythonEnvironment(
      withOfflineHuggingFaceCacheEnvironment(
        minimalMachineLearningEnvironment(process.env),
        args.config.paths.modelCache,
      ),
    ),
    logPath,
    timeoutMs: args.config.evaluation.timeoutMs,
    timeoutMessage:
      `Transformers inference timed out after ${args.config.evaluation.timeoutMs}ms`,
    reporter: args.reporter,
    stage: `evaluating_${args.kind}`,
    shouldCancel: args.shouldCancel,
  });
  if (result.exitCode !== 0) {
    throw new Error(
      `Transformers inference exited ${result.exitCode}: ${result.stderr.slice(0, 1000)}`,
    );
  }
  let output: unknown;
  try {
    output = JSON.parse(await readFile(outputPath, "utf8"));
  } catch (error) {
    throw new Error(
      `Transformers inference did not write valid JSON: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
  return parseBatchInferenceResult(
    output,
    args.examples.map((_, index) => String(index)),
  );
}

/**
 * Deterministic local scoring needs no network credentials. This function is
 * kept as the shared preflight hook used by validate, doctor, and run.
 */
export function assertEvaluationScoringReady(config: LocalRunnerConfig): void {
  if (
    config.evaluation.scoring.mode === "json_fields"
    && config.evaluation.scoring.fields?.some((field) => !field.trim())
  ) {
    throw new Error("evaluation.scoring.fields must contain non-empty field names");
  }
}

export function baselineCacheKey(args: {
  modelId: string;
  baseModelRevision?: string;
  sourceFingerprint?: string;
  system: string;
  examples: BehaviorSpecExample[];
  evalExamplesTotal: number;
  evalSplit?: EvalSplit;
  evalSampleSeed?: number | null;
  config: LocalRunnerConfig;
  packageVersion: string;
}): string {
  return createHash("sha256").update(JSON.stringify({
    v: 6,
    inference_protocol_version: INFERENCE_PROTOCOL_VERSION,
    package_version: args.packageVersion,
    model_id: args.modelId,
    base_model_revision: args.baseModelRevision ?? null,
    source_fingerprint: args.sourceFingerprint ?? null,
    system: args.system,
    examples: args.examples,
    eval_examples_total: args.evalExamplesTotal,
    eval_split: args.evalSplit ?? null,
    eval_sample_seed: args.evalSampleSeed ?? null,
    inference: args.config.evaluation.inference,
    scoring: args.config.evaluation.scoring,
  })).digest("hex");
}

function baselineInputsAreStable(args: {
  baseModelRevision?: string;
  sourceFingerprint?: string;
  config: LocalRunnerConfig;
}): boolean {
  return args.config.paths.baseModel
    ? Boolean(args.sourceFingerprint)
    : Boolean(args.baseModelRevision);
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
  if (cachedPackageVersion !== null) return cachedPackageVersion;
  try {
    const raw = await readFile(new URL("../package.json", import.meta.url), "utf8");
    cachedPackageVersion = String(
      (JSON.parse(raw) as { version?: unknown }).version ?? "unknown",
    );
  } catch {
    cachedPackageVersion = "unknown";
  }
  return cachedPackageVersion;
}

interface GeneratedEvalResult {
  prompt: string;
  expected: string;
  actual: string;
  latency_ms: number;
}

async function scoreGenerated(args: {
  kind: "baseline" | "candidate";
  modelId: string;
  generated: GeneratedEvalResult[];
  evalExamplesTotal: number;
  config: LocalRunnerConfig;
  outputPath: string;
  inferenceProvider: "none" | "transformers";
  evalSplit?: EvalSplit;
  evalSampleSeed?: number | null;
  generationConfig?: Record<string, unknown>;
  logUri?: string;
}): Promise<EvalReport> {
  const jsonScores: JsonFieldScore[] = [];
  const results: EvalExampleResult[] = args.generated.map((generated) => {
    const json = args.config.evaluation.scoring.mode === "json_fields"
      ? scoreJsonFields(
          generated.expected,
          generated.actual,
          args.config.evaluation.scoring.fields,
        )
      : null;
    if (json) jsonScores.push(json);
    const score = json?.score ?? exactScore(generated.expected, generated.actual);
    return {
      prompt: generated.prompt,
      expected: generated.expected,
      actual: generated.actual,
      passed: json?.passed ?? score === 1,
      score,
      reasoning: json?.reasoning ?? (
        args.inferenceProvider === "none"
          ? "Dry run: model inference was not executed."
          : "Scored by normalized exact match."
      ),
      latency_ms: generated.latency_ms,
      scored_by: json
        ? "json_fields"
        : args.inferenceProvider === "none"
          ? "heuristic"
          : "exact_match",
    };
  });
  const total = results.length;
  const mean = (values: number[]) =>
    values.length > 0
      ? values.reduce((sum, value) => sum + value, 0) / values.length
      : 0;
  const report = evalReportSchema.parse({
    kind: args.kind,
    model_id: args.modelId,
    total,
    eval_examples_total: args.evalExamplesTotal,
    eval_examples_used: total,
    eval_truncated: args.evalExamplesTotal > total,
    eval_split: args.evalSplit,
    eval_sample_seed: args.evalSampleSeed ?? null,
    avg_score: mean(results.map((result) => result.score)),
    pass_rate: mean(results.map((result) => result.passed ? 1 : 0)),
    exact_match_rate: mean(
      results.map((result) => exactScore(result.expected, result.actual)),
    ),
    avg_token_f1: mean(
      results.map((result) => tokenF1(result.expected, result.actual)),
    ),
    avg_latency_ms: Math.round(mean(results.map((result) => result.latency_ms))),
    results,
    artifact_uri: fileUri(args.outputPath),
    scoring_method: args.config.evaluation.scoring.mode === "json_fields"
      ? "json_fields"
      : args.inferenceProvider === "none"
        ? "heuristic"
        : "exact_match",
    inference_provider: args.inferenceProvider,
    scoring_mode: args.config.evaluation.scoring.mode,
    json_field_metrics: args.config.evaluation.scoring.mode === "json_fields"
      ? aggregateJsonFieldMetrics(jsonScores, total)
      : undefined,
    generation_config: args.generationConfig,
    log_uri: args.logUri,
  });
  await writeJson(args.outputPath, report);
  return report;
}

export async function evaluateExamples(args: {
  kind: "baseline" | "candidate";
  modelId: string;
  baseModelId?: string;
  baseModelRevision?: string;
  sourceFingerprint?: string;
  adapterPath?: string;
  examples: BehaviorSpecExample[];
  system: string;
  config: LocalRunnerConfig;
  outputPath: string;
  reporter?: LocalRunReporter;
  maxExamples?: number;
  evalSplit?: EvalSplit;
  sampleSeed?: number;
  shouldCancel?: () => boolean | Promise<boolean>;
}): Promise<EvalReport> {
  assertEvaluationScoringReady(args.config);
  const maxExamples = args.config.evaluation.maxExamples
    ?? args.maxExamples
    ?? args.examples.length;
  const truncated = args.examples.length > maxExamples;
  const seed = args.config.evaluation.sampleSeed ?? args.sampleSeed ?? 0;
  const examples = truncated
    ? sampleExamples(args.examples, maxExamples, seed)
    : args.examples;
  const evalSampleSeed = truncated || args.evalSplit === "spec_holdout" ? seed : null;
  const cacheEligible = args.kind === "baseline"
    && args.config.evaluation.baselineCache
    && !args.config.dryRun
    && baselineInputsAreStable({
      baseModelRevision: args.baseModelRevision,
      sourceFingerprint: args.sourceFingerprint,
      config: args.config,
    });
  const cacheKey = cacheEligible
    ? baselineCacheKey({
        modelId: args.modelId,
        baseModelRevision: args.baseModelRevision,
        sourceFingerprint: args.sourceFingerprint,
        system: args.system,
        examples,
        evalExamplesTotal: args.examples.length,
        evalSplit: args.evalSplit,
        evalSampleSeed,
        config: args.config,
        packageVersion: await packageVersion(),
      })
    : null;
  if (cacheKey) {
    const cached = await readBaselineCache(baselineCachePath(args.config, cacheKey));
    if (cached) {
      const report = evalReportSchema.parse({
        ...cached,
        cached: true,
        cache_key: cacheKey,
        artifact_uri: fileUri(args.outputPath),
        eval_split: args.evalSplit ?? cached.eval_split,
        eval_examples_total: args.examples.length,
        eval_examples_used: examples.length,
        eval_truncated: truncated,
        eval_sample_seed: evalSampleSeed,
      });
      await writeJson(args.outputPath, report);
      return report;
    }
  }

  const inference = args.config.dryRun
    ? null
    : await runTransformersInference({
        kind: args.kind,
        modelId: args.modelId,
        baseModelId: args.baseModelId ?? args.modelId,
        baseModelRevision: args.baseModelRevision,
        adapterPath: args.adapterPath,
        examples,
        system: args.system,
        config: args.config,
        outputPath: args.outputPath,
        reporter: args.reporter,
        shouldCancel: args.shouldCancel,
      });
  const generated = examples.map((example, index) => ({
    prompt: example.input,
    expected: example.output,
    actual: inference?.results[index]?.actual ?? "",
    latency_ms: inference?.results[index]?.latency_ms ?? 0,
  }));
  const report = await scoreGenerated({
    kind: args.kind,
    modelId: args.modelId,
    generated,
    evalExamplesTotal: args.examples.length,
    config: args.config,
    outputPath: args.outputPath,
    inferenceProvider: inference ? "transformers" : "none",
    evalSplit: args.evalSplit,
    evalSampleSeed,
    generationConfig: inference?.generation_config,
    logUri: inference ? fileUri(`${args.outputPath}.inference.log`) : undefined,
  });
  if (cacheKey) {
    await writeJson(
      baselineCachePath(args.config, cacheKey),
      { ...report, cache_key: cacheKey },
    );
  }
  return report;
}

export function compareEvalReports(baseline: EvalReport, candidate: EvalReport) {
  let regressions = 0;
  let improvements = 0;
  const regressedExamples: Array<{
    prompt: string;
    old_score: number;
    new_score: number;
  }> = [];
  const count = Math.min(baseline.results.length, candidate.results.length);
  for (let index = 0; index < count; index += 1) {
    const oldScore = baseline.results[index]?.score ?? 0;
    const result = candidate.results[index];
    const newScore = result?.score ?? 0;
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
    exact_match_rate_delta:
      candidate.exact_match_rate - baseline.exact_match_rate,
    token_f1_delta:
      (candidate.avg_token_f1 ?? 0) - (baseline.avg_token_f1 ?? 0),
    regressions,
    improvements,
    regressed_examples: regressedExamples,
  };
}
