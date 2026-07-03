/**
 * Local teacher-labeling job, ported from the tuned-tensor-runs labeling
 * workflow. Reads an unlabeled JSONL or CSV source, sanitizes each row, sends
 * pending rows to an OpenRouter teacher model under the spec's system message
 * (the same message the fine-tuned model will see at inference), and writes a
 * labeled {"input","output"} JSONL plus a job report under the artifact root.
 *
 * Unlike the eval judge there is no JSON output schema (the teacher's output
 * IS the training label, free text) and no heuristic fallback — a failed call
 * must surface as a failed row, never a fabricated label.
 */

import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import type { LocalRunnerConfig } from "./contracts.js";
import {
  isRetryableOpenRouterError,
  openRouterChat,
  type OpenRouterOptions,
} from "./openrouter.js";
import {
  sanitizeLabelingRow,
  type SanitizationFinding,
  type SanitizationStatus,
} from "./labeling-sanitize.js";
import type { LocalRunReporter } from "./run-reporter.js";

/** Per-row input cap; larger documents should be chunked upstream. */
export const MAX_INPUT_CHARS = 32_000;
export const MAX_PARSE_ERRORS = 50;

export interface UnlabeledRow {
  rowIndex: number;
  input: string;
  /** Present when the source row was already labeled (JSONL only). */
  output?: string;
}

export interface FailedUnlabeledRow {
  rowIndex: number;
  input: string;
  error: string;
}

export interface ParseResult {
  rows: UnlabeledRow[];
  failedRows: FailedUnlabeledRow[];
  errors: string[];
}

function pushError(errors: string[], message: string): boolean {
  if (errors.length >= MAX_PARSE_ERRORS) {
    if (errors[errors.length - 1]?.startsWith("...") === false) {
      errors.push(`... additional errors truncated (showing first ${MAX_PARSE_ERRORS})`);
    }
    return false;
  }
  errors.push(message);
  return true;
}

function overLimitError(input: string): string | null {
  return input.length > MAX_INPUT_CHARS
    ? `input exceeds ${MAX_INPUT_CHARS.toLocaleString()} characters`
    : null;
}

/**
 * Parse a JSONL source of {"input": string, "output"?: string} rows. Rows
 * that already carry an "output" skip the teacher and land directly as
 * labeled. Collects errors instead of throwing; callers reject the job when
 * `errors` is non-empty.
 */
export function parseUnlabeledJsonl(text: string, maxRows: number): ParseResult {
  const lines = text.split("\n").filter((line) => line.trim().length > 0);

  const rows: UnlabeledRow[] = [];
  const failedRows: FailedUnlabeledRow[] = [];
  const errors: string[] = [];

  if (lines.length === 0) {
    errors.push("File is empty");
    return { rows, failedRows, errors };
  }
  if (lines.length > maxRows) {
    errors.push(
      `File has ${lines.length.toLocaleString()} rows; the limit is ${maxRows.toLocaleString()} per labeling job`,
    );
    return { rows, failedRows, errors };
  }

  for (let i = 0; i < lines.length; i++) {
    const lineLabel = `Line ${i + 1}`;
    let parsed: unknown;
    try {
      parsed = JSON.parse(lines[i]!);
    } catch {
      if (!pushError(errors, `${lineLabel}: invalid JSON`)) break;
      continue;
    }

    if (
      typeof parsed !== "object" ||
      parsed === null ||
      Array.isArray(parsed) ||
      typeof (parsed as Record<string, unknown>).input !== "string"
    ) {
      if (!pushError(errors, `${lineLabel}: row must be {"input": string}`)) break;
      continue;
    }

    const obj = parsed as Record<string, unknown>;
    const input = obj.input as string;
    if (input.trim().length === 0) {
      pushError(errors, `${lineLabel}: input is empty`);
      continue;
    }

    const limitError = overLimitError(input);
    if (limitError) {
      failedRows.push({ rowIndex: i, input, error: `${lineLabel}: ${limitError}` });
      continue;
    }

    if (obj.output !== undefined && typeof obj.output !== "string") {
      if (!pushError(errors, `${lineLabel}: "output" must be a string when present`)) break;
      continue;
    }

    const output = typeof obj.output === "string" ? obj.output : undefined;
    if (output !== undefined && output.trim().length === 0) {
      if (!pushError(errors, `${lineLabel}: "output" is empty — omit it to have the teacher label this row`)) break;
      continue;
    }

    rows.push(output !== undefined ? { rowIndex: i, input, output } : { rowIndex: i, input });
  }

  return { rows, failedRows, errors };
}

/**
 * Minimal RFC 4180 CSV reader (quoted fields, escaped quotes, CR/LF inside
 * quotes). tt-local deliberately avoids a CSV dependency; this covers the
 * upload formats the hosted labeling dialog accepts.
 */
export function parseCsvRecords(text: string): { fields: string[]; records: string[][]; errors: string[] } {
  const records: string[][] = [];
  const errors: string[] = [];
  let record: string[] = [];
  let field = "";
  let inQuotes = false;
  let sawAny = false;

  const endField = () => {
    record.push(field);
    field = "";
  };
  const endRecord = () => {
    endField();
    if (record.length > 1 || record[0]!.trim().length > 0) {
      records.push(record);
      sawAny = true;
    }
    record = [];
  };

  for (let i = 0; i < text.length; i++) {
    const char = text[i]!;
    if (inQuotes) {
      if (char === "\"") {
        if (text[i + 1] === "\"") {
          field += "\"";
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += char;
      }
      continue;
    }
    if (char === "\"" && field.length === 0) {
      inQuotes = true;
      continue;
    }
    if (char === ",") {
      endField();
      continue;
    }
    if (char === "\n" || char === "\r") {
      if (char === "\r" && text[i + 1] === "\n") i += 1;
      endRecord();
      continue;
    }
    field += char;
  }
  if (inQuotes) {
    errors.push("Unterminated quoted field at end of file");
  }
  if (field.length > 0 || record.length > 0) endRecord();

  if (!sawAny) {
    return { fields: [], records: [], errors };
  }
  const fields = records.shift() ?? [];

  for (let i = 0; i < records.length; i++) {
    if (records[i]!.length !== fields.length) {
      if (!pushError(errors, `Row ${i + 1}: expected ${fields.length} fields, found ${records[i]!.length}`)) break;
    }
  }
  return { fields, records, errors };
}

/**
 * Parse a CSV source, taking `inputColumn` as the input text of each row.
 * The header row is required.
 */
export function parseUnlabeledCsv(text: string, inputColumn: string, maxRows: number): ParseResult {
  const rows: UnlabeledRow[] = [];
  const failedRows: FailedUnlabeledRow[] = [];

  const { fields, records, errors } = parseCsvRecords(text);
  if (fields.length === 0 || records.length === 0) {
    return { rows, failedRows, errors: ["File is empty or has no header row"] };
  }
  const columnIndex = fields.indexOf(inputColumn);
  if (columnIndex === -1) {
    return {
      rows,
      failedRows,
      errors: [`Column "${inputColumn}" not found. Available columns: ${fields.join(", ")}`],
    };
  }
  if (records.length > maxRows) {
    return {
      rows,
      failedRows,
      errors: [
        `File has ${records.length.toLocaleString()} rows; the limit is ${maxRows.toLocaleString()} per labeling job`,
      ],
    };
  }

  for (let i = 0; i < records.length; i++) {
    const lineLabel = `Row ${i + 1}`;
    const value = records[i]![columnIndex];
    if (typeof value !== "string") {
      if (!pushError(errors, `${lineLabel}: missing value in column "${inputColumn}"`)) break;
      continue;
    }
    if (value.trim().length === 0) {
      pushError(errors, `${lineLabel}: input is empty`);
      continue;
    }
    const limitError = overLimitError(value);
    if (limitError) {
      failedRows.push({ rowIndex: i, input: value, error: `${lineLabel}: ${limitError}` });
      continue;
    }
    rows.push({ rowIndex: i, input: value });
  }

  return { rows, failedRows, errors };
}

/**
 * Defensive removal of `<think>` reasoning blocks so the training label is
 * only the teacher's final answer. Truncated output can leave a dangling
 * close tag; keep only what follows the last close tag in that case.
 */
export function stripModelThinking(raw: string): string {
  if (!raw) return raw;
  let text = raw.replace(/<think(?:ing)?\b[^>]*>[\s\S]*?<\/think(?:ing)?>/gi, "");
  if (/<\/think(?:ing)?>/i.test(text)) {
    const lastClose = text.toLowerCase().lastIndexOf("</think");
    text = text.slice(text.indexOf(">", lastClose) + 1);
  }
  return text.trim();
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

function retryDelayMs(attempt: number): number {
  if (process.env.NODE_ENV === "test") return 0;
  const base = 500;
  const exponential = base * (2 ** Math.max(0, attempt - 1));
  const jitter = Math.floor(Math.random() * base);
  return Math.min(15_000, exponential + jitter);
}

/**
 * Map over items with bounded concurrency and optional minimum spacing
 * between call starts (global across workers).
 */
export async function mapWithPacing<T, R>(
  items: readonly T[],
  pacing: { concurrency: number; minIntervalMs: number },
  mapper: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  if (items.length === 0) return [];
  const limit = Math.max(1, Math.min(pacing.concurrency, items.length));
  const results: R[] = new Array(items.length);
  const nextStartAt = { value: 0 };
  let nextIndex = 0;

  async function worker(): Promise<void> {
    while (true) {
      const index = nextIndex++;
      if (index >= items.length) return;
      if (pacing.minIntervalMs > 0) {
        const now = Date.now();
        const startAt = Math.max(now, nextStartAt.value);
        nextStartAt.value = startAt + pacing.minIntervalMs;
        if (startAt > now) await sleep(startAt - now);
      }
      results[index] = await mapper(items[index]!, index);
    }
  }

  await Promise.all(Array.from({ length: limit }, () => worker()));
  return results;
}

export interface TeacherOptions {
  model: string;
  apiKeyEnv: string;
  appName?: string;
  siteUrl?: string;
  maxTokens: number;
  temperature: number;
  maxAttempts: number;
  timeoutMs: number;
}

export interface TeacherLabelResult {
  output: string;
  promptTokens: number;
  completionTokens: number;
  latencyMs: number;
}

export function resolveTeacherOptions(
  config: LocalRunnerConfig,
  overrides: { model?: string } = {},
): TeacherOptions {
  // Same fallback default as the llm config block's judge model.
  const model = overrides.model ?? config.labeling.model ?? config.llm?.model ?? "openai/gpt-5.5";
  return {
    model,
    apiKeyEnv: config.llm?.apiKeyEnv ?? "OPENROUTER_API_KEY",
    appName: config.llm?.appName,
    siteUrl: config.llm?.siteUrl,
    maxTokens: config.labeling.maxTokens,
    temperature: config.labeling.temperature,
    maxAttempts: config.labeling.maxAttempts,
    timeoutMs: config.labeling.timeoutMs,
  };
}

export async function labelRow(args: {
  systemMessage: string;
  input: string;
  teacher: TeacherOptions;
}): Promise<TeacherLabelResult> {
  const started = Date.now();
  const request: OpenRouterOptions = {
    model: args.teacher.model,
    apiKeyEnv: args.teacher.apiKeyEnv,
    appName: args.teacher.appName,
    siteUrl: args.teacher.siteUrl,
    timeoutMs: args.teacher.timeoutMs,
    temperature: args.teacher.temperature,
    maxTokens: args.teacher.maxTokens,
    responseFormat: "text",
  };
  const messages = [
    { role: "system" as const, content: args.systemMessage },
    { role: "user" as const, content: args.input },
  ];

  let lastError: unknown;
  for (let attempt = 1; attempt <= args.teacher.maxAttempts; attempt++) {
    try {
      const response = await openRouterChat(messages, request);
      const output = stripModelThinking(response.content);
      if (!output) {
        throw new Error("Teacher returned an empty output");
      }
      return {
        output,
        promptTokens: response.promptTokens,
        completionTokens: response.completionTokens,
        latencyMs: Date.now() - started,
      };
    } catch (error) {
      lastError = error;
      if (attempt >= args.teacher.maxAttempts || !isRetryableOpenRouterError(error)) {
        throw error;
      }
      await sleep(retryDelayMs(attempt));
    }
  }
  throw lastError ?? new Error("Teacher retry loop exhausted");
}

export type LabelingRowStatus = "labeled" | "failed" | "pending";

export interface LabelingRowResult {
  row_index: number;
  input: string;
  output?: string;
  status: LabelingRowStatus;
  label_source?: "teacher" | "upload";
  error?: string;
  sanitization_status: SanitizationStatus;
  sanitization_findings: SanitizationFinding[];
  prompt_tokens: number;
  completion_tokens: number;
}

export interface LocalLabelingReport {
  job_id: string;
  status: "completed" | "failed" | "dry_run";
  source_path: string;
  source_format: "jsonl" | "csv";
  input_column?: string;
  teacher_model_id: string;
  row_count: number;
  labeled_count: number;
  prelabeled_count: number;
  failed_count: number;
  redacted_count: number;
  blocked_count: number;
  pending_count: number;
  prompt_tokens_total: number;
  completion_tokens_total: number;
  error?: string;
  artifact_uris: {
    labeled: string;
    rows: string;
    report: string;
  };
  created_at: string;
  completed_at: string;
}

export interface LocalLabelingResult {
  report: LocalLabelingReport;
  reportPath: string;
  artifactDir: string;
  labeledPath: string;
  rowsPath: string;
}

export interface RunLocalLabelingJobArgs {
  sourcePath: string;
  /** Inferred from the file extension when omitted. */
  format?: "jsonl" | "csv";
  /** Required for CSV sources. */
  inputColumn?: string;
  systemMessage: string;
  config: LocalRunnerConfig;
  /** Optional extra copy of the labeled JSONL (artifact copy is always written). */
  outputPath?: string;
  /** Teacher model override; falls back to labeling.model then llm.model. */
  model?: string;
  /** Parse and sanitize only; no teacher calls, no labeled.jsonl. */
  dryRun?: boolean;
  reporter?: LocalRunReporter;
}

function inferFormat(sourcePath: string, format?: "jsonl" | "csv"): "jsonl" | "csv" {
  if (format) return format;
  return sourcePath.toLowerCase().endsWith(".csv") ? "csv" : "jsonl";
}

/**
 * Sanitizes parsed rows and sorts them into their initial status: parse
 * failures and blocked rows are terminally failed, uploaded outputs are
 * already labeled, and the rest are pending teacher labeling.
 */
function prepareRowResults(parsed: ParseResult): {
  results: LabelingRowResult[];
  pendingRows: LabelingRowResult[];
} {
  const results: LabelingRowResult[] = [];
  const pendingRows: LabelingRowResult[] = [];

  for (const row of parsed.failedRows) {
    const sanitized = sanitizeLabelingRow({ input: row.input });
    results.push({
      row_index: row.rowIndex,
      input: sanitized.input,
      status: "failed",
      error: row.error,
      sanitization_status: sanitized.sanitizationStatus,
      sanitization_findings: sanitized.sanitizationFindings,
      prompt_tokens: 0,
      completion_tokens: 0,
    });
  }

  for (const row of parsed.rows) {
    const sanitized = sanitizeLabelingRow({ input: row.input, output: row.output });
    const result: LabelingRowResult = {
      row_index: row.rowIndex,
      input: sanitized.input,
      status: "pending",
      sanitization_status: sanitized.sanitizationStatus,
      sanitization_findings: sanitized.sanitizationFindings,
      prompt_tokens: 0,
      completion_tokens: 0,
    };
    if (sanitized.sanitizationStatus === "blocked") {
      result.status = "failed";
      result.error = sanitized.sanitizationError ?? "Sensitive content detected; row was not sent to the teacher";
    } else if (sanitized.output !== undefined) {
      result.status = "labeled";
      result.output = sanitized.output;
      result.label_source = "upload";
    } else {
      pendingRows.push(result);
    }
    results.push(result);
  }

  return { results, pendingRows };
}

export async function runLocalLabelingJob(args: RunLocalLabelingJobArgs): Promise<LocalLabelingResult> {
  const emit = (status: string, message: string, details?: Record<string, unknown>) => {
    void args.reporter?.onEvent?.({ stage: "labeling", status, message, details });
  };

  const systemMessage = args.systemMessage.trim();
  if (!systemMessage) {
    throw new Error("Labeling requires a non-empty system message: the teacher labels under the same instructions the tuned model will see");
  }
  const format = inferFormat(args.sourcePath, args.format);
  if (format === "csv" && !args.inputColumn) {
    throw new Error("CSV sources require --input-column to pick the input text column");
  }

  const dryRun = args.dryRun ?? args.config.dryRun;
  const teacher = resolveTeacherOptions(args.config, { model: args.model });
  if (!dryRun && !process.env[teacher.apiKeyEnv]) {
    throw new Error(`${teacher.apiKeyEnv} is not set`);
  }

  const sourceText = await readFile(resolve(args.sourcePath), "utf8");
  const parsed = format === "csv"
    ? parseUnlabeledCsv(sourceText, args.inputColumn!, args.config.labeling.maxRows)
    : parseUnlabeledJsonl(sourceText, args.config.labeling.maxRows);
  const rowCount = parsed.rows.length + parsed.failedRows.length;
  if (parsed.errors.length > 0 || rowCount === 0) {
    const detail = parsed.errors.length > 0 ? parsed.errors.join("; ") : "File contains no rows";
    throw new Error(`Labeling source failed validation: ${detail}`);
  }

  const { results, pendingRows } = prepareRowResults(parsed);

  const prelabeledCount = results.filter((row) => row.label_source === "upload").length;
  emit("running", `Parsed ${rowCount} rows: ${pendingRows.length} to label, ${prelabeledCount} pre-labeled, ${results.filter((r) => r.status === "failed").length} failed`, {
    source: args.sourcePath,
    format,
    teacher_model: teacher.model,
    ...(dryRun ? { dry_run: true } : {}),
  });

  if (!dryRun && pendingRows.length > 0) {
    let done = 0;
    await mapWithPacing(
      pendingRows,
      { concurrency: args.config.labeling.concurrency, minIntervalMs: args.config.labeling.minIntervalMs },
      async (result) => {
        try {
          const label = await labelRow({ systemMessage, input: result.input, teacher });
          result.status = "labeled";
          result.output = label.output;
          result.label_source = "teacher";
          result.prompt_tokens = label.promptTokens;
          result.completion_tokens = label.completionTokens;
        } catch (error) {
          result.status = "failed";
          result.error = (error instanceof Error ? error.message : String(error)).slice(0, 2000);
        }
        done += 1;
        if (done === pendingRows.length || done % 10 === 0) {
          emit("running", `Labeled ${done}/${pendingRows.length} pending rows`);
        }
      },
    );
  }

  results.sort((a, b) => a.row_index - b.row_index);
  const labeled = results.filter((row) => row.status === "labeled");
  const failedCount = results.filter((row) => row.status === "failed").length;
  const allFailed = !dryRun && rowCount > 0 && failedCount === rowCount;

  const jobId = randomUUID();
  const artifactDir = resolve(args.config.artifactRoot, "labeling", jobId);
  await mkdir(artifactDir, { recursive: true });
  const labeledPath = join(artifactDir, "labeled.jsonl");
  const rowsPath = join(artifactDir, "rows.jsonl");
  const reportPath = join(artifactDir, "labeling-report.json");

  if (!dryRun) {
    const labeledContent = labeled
      .map((row) => `${JSON.stringify({ input: row.input, output: row.output })}\n`)
      .join("");
    await writeFile(labeledPath, labeledContent, "utf8");
    if (args.outputPath) {
      const outputPath = resolve(args.outputPath);
      await mkdir(dirname(outputPath), { recursive: true });
      await writeFile(outputPath, labeledContent, "utf8");
    }
  }
  await writeFile(
    rowsPath,
    results.map((row) => `${JSON.stringify(row)}\n`).join(""),
    "utf8",
  );

  const now = new Date().toISOString();
  const report: LocalLabelingReport = {
    job_id: jobId,
    status: dryRun ? "dry_run" : allFailed ? "failed" : "completed",
    source_path: resolve(args.sourcePath),
    source_format: format,
    ...(args.inputColumn ? { input_column: args.inputColumn } : {}),
    teacher_model_id: teacher.model,
    row_count: rowCount,
    labeled_count: labeled.length,
    prelabeled_count: prelabeledCount,
    failed_count: failedCount,
    redacted_count: results.filter((row) => row.sanitization_status === "redacted").length,
    blocked_count: results.filter((row) => row.sanitization_status === "blocked").length,
    pending_count: results.filter((row) => row.status === "pending").length,
    prompt_tokens_total: results.reduce((sum, row) => sum + row.prompt_tokens, 0),
    completion_tokens_total: results.reduce((sum, row) => sum + row.completion_tokens, 0),
    ...(allFailed ? { error: "All rows failed teacher labeling" } : {}),
    artifact_uris: {
      labeled: labeledPath,
      rows: rowsPath,
      report: reportPath,
    },
    created_at: now,
    completed_at: now,
  };
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");

  emit(report.status, dryRun
    ? `Dry run: ${report.pending_count} rows would be sent to ${teacher.model}`
    : `Labeled ${report.labeled_count}/${report.row_count} rows (${report.failed_count} failed)`, {
    report_path: reportPath,
  });

  return { report, reportPath, artifactDir, labeledPath, rowsPath };
}
