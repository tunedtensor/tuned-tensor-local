import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  lstat,
  mkdir,
  open,
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { writeJsonAtomic } from "./artifacts.js";
import { minimalMachineLearningEnvironment } from "./huggingface-cache.js";
import { parseCsvRecords } from "./labeling.js";
import { runLoggedProcess } from "./process-runner.js";
import {
  computeBinaryClassificationMetrics,
  type BinaryClassificationMetrics,
  type BinaryScoredRow,
} from "./study-metrics.js";
import {
  loadStudySpec,
  validateStudyBenchmark,
  type BinaryClassificationStudyTask,
  type StudyBenchmarkLock,
  type StudySpec,
} from "./studies.js";

export const STUDY_TRIAL_PROTOCOL_VERSION = 1;
const MAX_PREDICTION_BYTES = 16 * 1024 * 1024;
const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const NUMERIC_LOGISTIC_REGRESSION_SCRIPT = join(
  packageRoot,
  "training/study-runner/numeric_logistic_regression.py",
);

export type StudyTrialJsonValue =
  | string
  | number
  | boolean
  | null
  | StudyTrialJsonValue[]
  | { [key: string]: StudyTrialJsonValue };

const studyTrialJsonValueSchema: z.ZodType<StudyTrialJsonValue> = z.lazy(() => z.union([
  z.string(),
  z.number().finite(),
  z.boolean(),
  z.null(),
  z.array(studyTrialJsonValueSchema),
  z.record(z.string(), studyTrialJsonValueSchema),
]));
const exactStringSchema = z.string().min(1).refine(
  (value) => value.trim() === value && !value.includes("\0"),
  { message: "must not have outer whitespace or contain a null character" },
);
const datasetIdSchema = z.string().min(1).refine(
  (value) => value.trim() === value,
  { message: "must not have outer whitespace" },
);
const relativeWorkingDirectorySchema = exactStringSchema.refine(
  (value) => (
    !isAbsolute(value)
    && !value.includes("\\")
    && !/^[a-zA-Z]:[\\/]/.test(value)
    && !/^[a-z][a-z0-9+.-]*:/i.test(value)
  ),
  { message: "must be a portable relative local path" },
);
const studyTrialTimeoutSchema = z.number().int().min(1_000).max(86_400_000);

const studyTrialCommandRunnerSchema = z.object({
  command: z.array(exactStringSchema).min(1).refine(
    (command) => !command.some((value) => (
      value === "--input"
      || value.startsWith("--input=")
      || value === "--output"
      || value.startsWith("--output=")
      || value === "--artifact-dir"
      || value.startsWith("--artifact-dir=")
    )),
    { message: "--input, --output, and --artifact-dir are reserved by the trial protocol" },
  ),
  cwd: relativeWorkingDirectorySchema.optional(),
  timeout_ms: studyTrialTimeoutSchema,
}).strict();

const studyTrialBuiltinRunnerSchema = z.object({
  builtin: z.literal("numeric_logistic_regression"),
  timeout_ms: studyTrialTimeoutSchema,
}).strict();

const numericLogisticRegressionParametersSchema = z.object({
  c: z.number().finite().min(1e-6).max(1e6),
  class_weight: z.enum(["none", "balanced"]),
  max_iter: z.number().int().min(1).max(10_000),
  random_seed: z.number().int().min(0).max(2 ** 32 - 1),
}).strict();

export const studyTrialSpecSchema = z.object({
  schema_version: z.literal(1),
  id: z.string().regex(
    /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/,
    "must start with an alphanumeric character and contain only alphanumerics, '.', '_', or '-'",
  ),
  name: exactStringSchema,
  runner: z.union([
    studyTrialCommandRunnerSchema,
    studyTrialBuiltinRunnerSchema,
  ]),
  parameters: z.record(z.string(), studyTrialJsonValueSchema).default({}),
}).strict().superRefine((trial, context) => {
  if (!("builtin" in trial.runner)) return;
  const parsed = numericLogisticRegressionParametersSchema.safeParse(trial.parameters);
  if (parsed.success) return;
  for (const issue of parsed.error.issues) {
    context.addIssue({
      code: "custom",
      path: ["parameters", ...issue.path],
      message: issue.message,
    });
  }
});

export const studyTrialPredictionSchema = z.object({
  id: datasetIdSchema,
  probability: z.number().finite().min(0).max(1),
}).strict();

export const studyTrialOutputSchema = z.object({
  protocol_version: z.literal(STUDY_TRIAL_PROTOCOL_VERSION),
  predictions: z.array(studyTrialPredictionSchema),
}).strict();

export type StudyTrialSpec = z.infer<typeof studyTrialSpecSchema>;
export type StudyTrialOutput = z.infer<typeof studyTrialOutputSchema>;

export interface StudyTrialReport {
  schema_version: 1;
  protocol_version: 1;
  trial: {
    id: string;
    name: string;
    spec_sha256: string;
    parameters: Record<string, StudyTrialJsonValue>;
  };
  study: {
    name: string;
    study_spec_sha256: string;
    benchmark_lock_sha256: string;
    task_type: "binary_classification";
    primary_metric: "average_precision" | "roc_auc" | "f1";
  };
  data: {
    training: {
      source_sha256: string;
      projected_sha256: string;
      row_count: number;
    };
    validation: {
      source_sha256: string;
      projected_sha256: string;
      row_count: number;
    };
  };
  evaluation: {
    score_semantics: "positive_class_probability";
    primary_score: number;
    metrics: BinaryClassificationMetrics;
    decision_threshold: 0.5;
    prediction_count: number;
    predictions_sha256: string;
  };
  execution: {
    command: string[];
    cwd: string;
    timeout_ms: number;
    duration_ms: number;
    log: "trial.log";
    artifact_directory: "model";
  };
}

interface LoadedTrialSpec {
  path: string;
  spec: StudyTrialSpec;
}

export interface StudyTrialRunnerCommand {
  command: string;
  baseArgs: string[];
  reportCommand: string[];
  requiredFiles: string[];
}

interface ParsedStudySplit {
  rows: string[][];
  columns: string[];
  sha256: string;
}

interface PreparedTrialData {
  trainingCsv: string;
  validationCsv: string;
  validationLabels: Array<{ id: string; positive: boolean }>;
  source: {
    training: { sha256: string; rowCount: number };
    validation: { sha256: string; rowCount: number };
  };
}

function schemaError(error: z.ZodError): string {
  return error.issues
    .map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`)
    .join("; ");
}

function sha256(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function canonicalJson(value: unknown): string {
  const normalize = (current: unknown): unknown => {
    if (Array.isArray(current)) return current.map(normalize);
    if (current !== null && typeof current === "object") {
      return Object.fromEntries(
        Object.entries(current as Record<string, unknown>)
          .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
          .map(([key, child]) => [key, normalize(child)]),
      );
    }
    return current;
  };
  return JSON.stringify(normalize(value));
}

async function assertRegularFile(path: string, description: string): Promise<void> {
  let metadata: Awaited<ReturnType<typeof stat>>;
  try {
    metadata = await stat(path);
  } catch (error) {
    throw new Error(`Cannot inspect ${description} at ${path}`, { cause: error });
  }
  if (!metadata.isFile()) {
    throw new Error(`${description} must be a regular file: ${path}`);
  }
}

async function assertDirectory(path: string, description: string): Promise<void> {
  let metadata: Awaited<ReturnType<typeof stat>>;
  try {
    metadata = await stat(path);
  } catch (error) {
    throw new Error(`Cannot inspect ${description} at ${path}`, { cause: error });
  }
  if (!metadata.isDirectory()) {
    throw new Error(`${description} must be a directory: ${path}`);
  }
}

async function readStableRegularFile(args: {
  path: string;
  description: string;
  maxBytes?: number;
  missingMessage?: string;
}): Promise<Uint8Array> {
  let handle: Awaited<ReturnType<typeof open>>;
  try {
    handle = await open(args.path, constants.O_RDONLY | constants.O_NOFOLLOW);
  } catch (error) {
    if (
      (error as NodeJS.ErrnoException).code === "ELOOP"
      || (error as NodeJS.ErrnoException).code === "EMLINK"
    ) {
      throw new Error(
        `${args.description} must be a regular, non-symbolic file: ${args.path}`,
        { cause: error },
      );
    }
    if (
      args.missingMessage
      && (error as NodeJS.ErrnoException).code === "ENOENT"
    ) {
      throw new Error(args.missingMessage, { cause: error });
    }
    throw new Error(`Cannot open ${args.description} at ${args.path}`, { cause: error });
  }
  try {
    const before = await handle.stat();
    if (!before.isFile()) {
      throw new Error(`${args.description} must be a regular, non-symbolic file: ${args.path}`);
    }
    if (args.maxBytes !== undefined && before.size > args.maxBytes) {
      throw new Error(
        `${args.description} exceeds the ${args.maxBytes}-byte limit: ${args.path}`,
      );
    }
    const bytes = await handle.readFile();
    const after = await handle.stat();
    if (
      before.dev !== after.dev
      || before.ino !== after.ino
      || before.size !== after.size
      || bytes.byteLength !== after.size
    ) {
      throw new Error(`${args.description} changed while it was being read: ${args.path}`);
    }
    if (args.maxBytes !== undefined && bytes.byteLength > args.maxBytes) {
      throw new Error(
        `${args.description} exceeds the ${args.maxBytes}-byte limit: ${args.path}`,
      );
    }
    const current = await lstat(args.path);
    if (
      !current.isFile()
      || current.isSymbolicLink()
      || current.dev !== after.dev
      || current.ino !== after.ino
      || current.size !== after.size
    ) {
      throw new Error(`${args.description} path changed while it was being read: ${args.path}`);
    }
    return bytes;
  } finally {
    await handle.close();
  }
}

export async function loadStudyTrialSpec(trialPath: string): Promise<LoadedTrialSpec> {
  const path = resolve(trialPath);
  await assertRegularFile(path, "study trial spec");
  let value: unknown;
  try {
    value = JSON.parse(await readFile(path, "utf8")) as unknown;
  } catch (error) {
    throw new Error(`Cannot read valid study trial spec JSON at ${path}`, { cause: error });
  }
  const parsed = studyTrialSpecSchema.safeParse(value);
  if (!parsed.success) {
    throw new Error(`Invalid study trial spec at ${path}: ${schemaError(parsed.error)}`);
  }
  return { path, spec: parsed.data };
}

function decodeUtf8(bytes: Uint8Array, split: "training" | "validation"): string {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new Error(`${split} dataset is not valid UTF-8`, { cause: error });
  }
}

async function readLockedSplit(args: {
  split: "training" | "validation";
  studyDirectory: string;
  source: string;
  expected: StudyBenchmarkLock["dataset"]["splits"]["training" | "validation"];
}): Promise<ParsedStudySplit> {
  const path = resolve(args.studyDirectory, args.source);
  const bytes = await readStableRegularFile({
    path,
    description: `${args.split} dataset`,
    maxBytes: args.expected.size_bytes,
  });
  const foundSha256 = sha256(bytes);
  if (
    foundSha256 !== args.expected.sha256
    || bytes.byteLength !== args.expected.size_bytes
  ) {
    throw new Error(
      `${args.split} dataset changed after benchmark validation; rerun the trial`,
    );
  }
  const parsed = parseCsvRecords(decodeUtf8(bytes, args.split), { strict: true });
  if (parsed.errors.length > 0) {
    throw new Error(`Invalid ${args.split} CSV: ${parsed.errors.join("; ")}`);
  }
  const columns = parsed.fields.map((field, index) => (
    index === 0 ? field.replace(/^\uFEFF/, "") : field
  ));
  if (parsed.records.length !== args.expected.row_count) {
    throw new Error(
      `${args.split} dataset row count changed after benchmark validation; rerun the trial`,
    );
  }
  return { rows: parsed.records, columns, sha256: foundSha256 };
}

function csvField(value: string): string {
  return /[",\r\n]/.test(value) ? `"${value.replaceAll("\"", "\"\"")}"` : value;
}

function serializeCsv(columns: readonly string[], rows: readonly (readonly string[])[]): string {
  return [
    columns.map(csvField).join(","),
    ...rows.map((row) => row.map(csvField).join(",")),
  ].join("\n") + "\n";
}

async function prepareTrialData(args: {
  study: StudySpec;
  studyPath: string;
  lock: StudyBenchmarkLock;
}): Promise<PreparedTrialData> {
  const studyDirectory = dirname(args.studyPath);
  const [training, validation] = await Promise.all([
    readLockedSplit({
      split: "training",
      studyDirectory,
      source: args.study.dataset.splits.training,
      expected: args.lock.dataset.splits.training,
    }),
    readLockedSplit({
      split: "validation",
      studyDirectory,
      source: args.study.dataset.splits.validation,
      expected: args.lock.dataset.splits.validation,
    }),
  ]);
  if (canonicalJson(training.columns) !== canonicalJson(args.lock.dataset.columns)) {
    throw new Error("Training dataset columns changed after benchmark validation; rerun the trial");
  }
  if (canonicalJson(validation.columns) !== canonicalJson(args.lock.dataset.columns)) {
    throw new Error("Validation dataset columns changed after benchmark validation; rerun the trial");
  }

  const task = args.study.task;
  const projectedColumns = [task.id_column, ...task.input_columns];
  const trainingIndexes = projectedColumns.map((column) => training.columns.indexOf(column));
  const validationIndexes = projectedColumns.map((column) => validation.columns.indexOf(column));
  const trainingTargetIndex = training.columns.indexOf(task.target_column);
  const validationTargetIndex = validation.columns.indexOf(task.target_column);
  const trainingRows = training.rows.map((row) => [
    ...trainingIndexes.map((index) => row[index]!),
    row[trainingTargetIndex] === task.labels.positive ? "1" : "0",
  ]);
  const validationRows = validation.rows.map((row) => (
    validationIndexes.map((index) => row[index]!)
  ));
  const validationLabels = validation.rows.map((row) => ({
    id: row[validationIndexes[0]!]!,
    positive: row[validationTargetIndex] === task.labels.positive,
  }));

  return {
    trainingCsv: serializeCsv(
      [...projectedColumns, task.target_column],
      trainingRows,
    ),
    validationCsv: serializeCsv(projectedColumns, validationRows),
    validationLabels,
    source: {
      training: { sha256: training.sha256, rowCount: training.rows.length },
      validation: { sha256: validation.sha256, rowCount: validation.rows.length },
    },
  };
}

export function defaultStudyTrialOutputRoot(studyPath: string): string {
  return join(dirname(resolve(studyPath)), ".tt-local", "study-trials");
}

export function buildStudyTrialRunnerCommand(
  runner: StudyTrialSpec["runner"],
): StudyTrialRunnerCommand {
  if ("builtin" in runner) {
    return {
      command: "uv",
      baseArgs: ["run", "--locked", NUMERIC_LOGISTIC_REGRESSION_SCRIPT],
      reportCommand: [`builtin:${runner.builtin}`],
      requiredFiles: [
        NUMERIC_LOGISTIC_REGRESSION_SCRIPT,
        `${NUMERIC_LOGISTIC_REGRESSION_SCRIPT}.lock`,
      ],
    };
  }
  const [command, ...baseArgs] = runner.command;
  return {
    command: command!,
    baseArgs,
    reportCommand: runner.command,
    requiredFiles: [],
  };
}

function trialProtocolInput(args: {
  trial: StudyTrialSpec;
  task: BinaryClassificationStudyTask;
  trainingPath: string;
  validationPath: string;
}): unknown {
  return {
    protocol_version: STUDY_TRIAL_PROTOCOL_VERSION,
    trial: {
      id: args.trial.id,
      name: args.trial.name,
      parameters: args.trial.parameters,
    },
    task: {
      type: args.task.type,
      id_column: args.task.id_column,
      input_columns: args.task.input_columns,
      target_column: args.task.target_column,
      target_values: { negative: 0, positive: 1 },
      prediction: {
        field: "probability",
        meaning: "probability_of_positive_target",
      },
    },
    datasets: {
      training_csv: args.trainingPath,
      validation_csv: args.validationPath,
    },
  };
}

async function parsePredictions(
  predictionPath: string,
  validationLabels: readonly { id: string; positive: boolean }[],
): Promise<{ bytes: Uint8Array; rows: BinaryScoredRow[] }> {
  const bytes = await readStableRegularFile({
    path: predictionPath,
    description: "trial predictions",
    maxBytes: MAX_PREDICTION_BYTES,
    missingMessage: `Trial command did not write predictions at ${predictionPath}`,
  });
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)) as unknown;
  } catch (error) {
    throw new Error(`Trial command wrote invalid prediction JSON at ${predictionPath}`, { cause: error });
  }
  const parsed = studyTrialOutputSchema.safeParse(value);
  if (!parsed.success) {
    throw new Error(`Invalid trial predictions: ${schemaError(parsed.error)}`);
  }
  if (parsed.data.predictions.length !== validationLabels.length) {
    throw new Error(
      `Trial returned ${parsed.data.predictions.length} predictions for `
      + `${validationLabels.length} validation examples`,
    );
  }

  const expectedIds = new Set(validationLabels.map((row) => row.id));
  const probabilities = new Map<string, number>();
  for (const prediction of parsed.data.predictions) {
    if (!expectedIds.has(prediction.id)) {
      throw new Error(`Trial returned unknown validation ID "${prediction.id}"`);
    }
    if (probabilities.has(prediction.id)) {
      throw new Error(`Trial returned duplicate validation ID "${prediction.id}"`);
    }
    probabilities.set(prediction.id, prediction.probability);
  }

  const missing = validationLabels.find((row) => !probabilities.has(row.id));
  if (missing) {
    throw new Error(`Trial did not return validation ID "${missing.id}"`);
  }
  return {
    bytes,
    rows: validationLabels.map((row) => ({
      ...row,
      probability: probabilities.get(row.id)!,
    })),
  };
}

function primaryScore(
  metric: BinaryClassificationStudyTask["primary_metric"],
  metrics: BinaryClassificationMetrics,
): number {
  if (metric === "f1") return metrics.f1_at_0_5;
  return metrics[metric];
}

async function claimTrialDirectory(outputRoot: string, trialId: string): Promise<string> {
  await mkdir(outputRoot, { recursive: true, mode: 0o700 });
  const rootMetadata = await lstat(outputRoot);
  if (!rootMetadata.isDirectory() || rootMetadata.isSymbolicLink()) {
    throw new Error(`Study trial output root must be a non-symbolic directory: ${outputRoot}`);
  }
  const trialDirectory = join(outputRoot, trialId);
  try {
    await mkdir(trialDirectory, { mode: 0o700 });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EEXIST") {
      throw new Error(
        `Study trial "${trialId}" already exists at ${trialDirectory}; use a new trial ID`,
      );
    }
    throw error;
  }
  return trialDirectory;
}

export async function runStudyTrial(args: {
  studyPath: string;
  trialPath: string;
  lockPath?: string;
  outputRoot?: string;
}): Promise<{
  trialDirectory: string;
  reportPath: string;
  report: StudyTrialReport;
}> {
  const loadedTrial = await loadStudyTrialSpec(args.trialPath);
  const validated = await validateStudyBenchmark({
    studyPath: args.studyPath,
    lockPath: args.lockPath,
  });
  const loadedStudy = await loadStudySpec(args.studyPath);
  if (sha256(JSON.stringify(loadedStudy.spec)) !== validated.lock.study_spec_sha256) {
    throw new Error("StudySpec changed after benchmark validation; rerun the trial");
  }
  const prepared = await prepareTrialData({
    study: loadedStudy.spec,
    studyPath: loadedStudy.path,
    lock: validated.lock,
  });
  const runnerCommand = buildStudyTrialRunnerCommand(loadedTrial.spec.runner);
  await Promise.all(runnerCommand.requiredFiles.map((path) => (
    assertRegularFile(path, "bundled study trial runtime")
  )));
  const configuredCwd = "cwd" in loadedTrial.spec.runner
    ? loadedTrial.spec.runner.cwd
    : undefined;
  const configuredRunnerCwd = configuredCwd
    ? resolve(dirname(loadedTrial.path), configuredCwd)
    : undefined;
  if (configuredRunnerCwd) {
    await assertDirectory(configuredRunnerCwd, "study trial working directory");
  }

  const outputRoot = resolve(
    args.outputRoot ?? defaultStudyTrialOutputRoot(loadedStudy.path),
  );
  const trialDirectory = await claimTrialDirectory(outputRoot, loadedTrial.spec.id);
  const trainingPath = join(trialDirectory, "training.csv");
  const validationPath = join(trialDirectory, "validation.csv");
  const inputPath = join(trialDirectory, "trial-input.json");
  const predictionPath = join(trialDirectory, "predictions.json");
  const artifactDirectory = join(trialDirectory, "model");
  const logPath = join(trialDirectory, "trial.log");
  const reportPath = join(trialDirectory, "trial-report.json");
  await mkdir(artifactDirectory, { mode: 0o700 });
  await Promise.all([
    writeFile(trainingPath, prepared.trainingCsv, { encoding: "utf8", mode: 0o400, flag: "wx" }),
    writeFile(validationPath, prepared.validationCsv, { encoding: "utf8", mode: 0o400, flag: "wx" }),
  ]);
  const projectedHashes = {
    training: sha256(prepared.trainingCsv),
    validation: sha256(prepared.validationCsv),
  };
  await writeFile(
    inputPath,
    `${JSON.stringify(trialProtocolInput({
      trial: loadedTrial.spec,
      task: loadedStudy.spec.task,
      trainingPath,
      validationPath,
    }), null, 2)}\n`,
    { encoding: "utf8", mode: 0o400, flag: "wx" },
  );

  const runnerCwd = configuredRunnerCwd ?? trialDirectory;
  const commandArgs = [
    ...runnerCommand.baseArgs,
    "--input",
    inputPath,
    "--output",
    predictionPath,
    "--artifact-dir",
    artifactDirectory,
  ];
  await rm(predictionPath, { force: true });
  const started = performance.now();
  const result = await runLoggedProcess({
    command: runnerCommand.command,
    commandArgs,
    cwd: runnerCwd,
    env: minimalMachineLearningEnvironment(process.env),
    logPath,
    timeoutMs: loadedTrial.spec.runner.timeout_ms,
    timeoutMessage: (
      `Study trial "${loadedTrial.spec.id}" timed out after `
      + `${loadedTrial.spec.runner.timeout_ms}ms`
    ),
    terminateProcessGroupOnExit: true,
    stage: "study-trial",
  });
  const durationMs = Math.round(performance.now() - started);
  if (result.exitCode !== 0) {
    throw new Error(
      `Study trial "${loadedTrial.spec.id}" exited with code ${result.exitCode}; see ${logPath}`,
    );
  }

  const [trainingAfter, validationAfter] = await Promise.all([
    readStableRegularFile({
      path: trainingPath,
      description: "projected training dataset",
      maxBytes: Buffer.byteLength(prepared.trainingCsv),
    }),
    readStableRegularFile({
      path: validationPath,
      description: "projected validation dataset",
      maxBytes: Buffer.byteLength(prepared.validationCsv),
    }),
  ]);
  if (
    sha256(trainingAfter) !== projectedHashes.training
    || sha256(validationAfter) !== projectedHashes.validation
  ) {
    throw new Error(
      `Study trial "${loadedTrial.spec.id}" modified its projected dataset inputs`,
    );
  }
  const parsedPredictions = await parsePredictions(
    predictionPath,
    prepared.validationLabels,
  );
  const metrics = computeBinaryClassificationMetrics(parsedPredictions.rows);
  const report: StudyTrialReport = {
    schema_version: 1,
    protocol_version: STUDY_TRIAL_PROTOCOL_VERSION,
    trial: {
      id: loadedTrial.spec.id,
      name: loadedTrial.spec.name,
      spec_sha256: sha256(canonicalJson(loadedTrial.spec)),
      parameters: loadedTrial.spec.parameters,
    },
    study: {
      name: loadedStudy.spec.name,
      study_spec_sha256: validated.lock.study_spec_sha256,
      benchmark_lock_sha256: sha256(canonicalJson(validated.lock)),
      task_type: loadedStudy.spec.task.type,
      primary_metric: loadedStudy.spec.task.primary_metric,
    },
    data: {
      training: {
        source_sha256: prepared.source.training.sha256,
        projected_sha256: projectedHashes.training,
        row_count: prepared.source.training.rowCount,
      },
      validation: {
        source_sha256: prepared.source.validation.sha256,
        projected_sha256: projectedHashes.validation,
        row_count: prepared.source.validation.rowCount,
      },
    },
    evaluation: {
      score_semantics: "positive_class_probability",
      primary_score: primaryScore(loadedStudy.spec.task.primary_metric, metrics),
      metrics,
      decision_threshold: 0.5,
      prediction_count: parsedPredictions.rows.length,
      predictions_sha256: sha256(parsedPredictions.bytes),
    },
    execution: {
      command: runnerCommand.reportCommand,
      cwd: configuredCwd ?? "<trial-directory>",
      timeout_ms: loadedTrial.spec.runner.timeout_ms,
      duration_ms: durationMs,
      log: "trial.log",
      artifact_directory: "model",
    },
  };
  await writeJsonAtomic(reportPath, report);
  return { trialDirectory, reportPath, report };
}
