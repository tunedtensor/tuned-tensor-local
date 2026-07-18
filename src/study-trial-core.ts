import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  lstat,
  mkdir,
  open,
  stat,
} from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { parseCsvRecords } from "./labeling.js";
import {
  type BinaryClassificationMetrics,
  type BinaryScoredRow,
} from "./study-metrics.js";
import {
  type StudyImplementationInputFile,
  type StudyImplementationReference,
  type StudyModelReference,
} from "./study-provenance.js";
import {
  type BinaryClassificationStudyTask,
  type StudyBenchmarkLock,
  type StudySpec,
} from "./studies.js";

export const STUDY_TRIAL_PROTOCOL_VERSION = 1;
const MAX_PREDICTION_BYTES = 16 * 1024 * 1024;
export const MAX_TRIAL_REPORT_BYTES = 4 * 1024 * 1024;
export const MAX_PROVENANCE_MANIFEST_BYTES = 64 * 1024 * 1024;
const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));
export const NUMERIC_LOGISTIC_REGRESSION_SCRIPT = join(
  packageRoot,
  "training/study-runner/numeric_logistic_regression.py",
);
export const NUMERIC_LOGISTIC_REGRESSION_LOCK = `${NUMERIC_LOGISTIC_REGRESSION_SCRIPT}.lock`;
export const NUMERIC_LOGISTIC_REGRESSION_LOGICAL_PATH =
  "training/study-runner/numeric_logistic_regression.py";
export const NUMERIC_LOGISTIC_REGRESSION_RUNNER_VERSION = 2;

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
export const exactStringSchema = z.string().min(1).refine(
  (value) => value.trim() === value && !value.includes("\0"),
  { message: "must not have outer whitespace or contain a null character" },
);
export const datasetIdSchema = z.string().min(1).refine(
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
const MAX_PROVENANCE_PATH_SEGMENTS = 64;
export const provenanceFilePathSchema = exactStringSchema.refine(
  (value) => {
    if (
      isAbsolute(value)
      || value.includes("\\")
      || /^[a-zA-Z]:[\\/]/.test(value)
      || /^[a-z][a-z0-9+.-]*:/i.test(value)
    ) {
      return false;
    }
    const segments = value.split("/");
    return segments.length <= MAX_PROVENANCE_PATH_SEGMENTS
      && segments.every((segment) => (
      segment.length > 0 && segment !== "." && segment !== ".."
      ));
  },
  { message: "must be a portable descendant file path" },
);
const studyTrialCommandProvenanceSchema = z.object({
  source_files: z.array(provenanceFilePathSchema).min(1).max(256),
  dependency_lock_files: z.array(provenanceFilePathSchema).max(32),
}).strict().superRefine((provenance, context) => {
  const seen = new Set<string>();
  for (const [field, paths] of [
    ["source_files", provenance.source_files],
    ["dependency_lock_files", provenance.dependency_lock_files],
  ] as const) {
    for (const [index, path] of paths.entries()) {
      if (seen.has(path)) {
        context.addIssue({
          code: "custom",
          path: [field, index],
          message: "must not duplicate or overlap another provenance path",
        });
      }
      seen.add(path);
    }
  }
});

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
  provenance: studyTrialCommandProvenanceSchema,
}).strict();

const studyTrialBuiltinRunnerSchema = z.object({
  builtin: z.literal("numeric_logistic_regression"),
  timeout_ms: studyTrialTimeoutSchema,
}).strict();

export const numericLogisticRegressionParametersSchema = z.object({
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

export const bundledRuntimeVersionsSchema = z.object({
  python: exactStringSchema,
  platform: exactStringSchema,
  numpy: exactStringSchema,
  scikit_learn: exactStringSchema,
  joblib: exactStringSchema,
}).strict();

export const bundledPredictionRuntimeEvidenceSchema = z.object({
  schema_version: z.literal(1),
  protocol_version: z.literal(STUDY_TRIAL_PROTOCOL_VERSION),
  runner: z.object({
    name: z.literal("numeric_logistic_regression"),
    version: z.literal(NUMERIC_LOGISTIC_REGRESSION_RUNNER_VERSION),
  }).strict(),
  runtime: bundledRuntimeVersionsSchema,
}).strict();

export type StudyTrialSpec = z.infer<typeof studyTrialSpecSchema>;
export type StudyTrialOutput = z.infer<typeof studyTrialOutputSchema>;
export type BundledPredictionRuntimeEvidence = z.infer<
  typeof bundledPredictionRuntimeEvidenceSchema
>;
type StudyTemporalCertification = NonNullable<StudyBenchmarkLock["dataset"]["temporal"]>;
export type StudyTrialTemporalEvidence = Omit<StudyTemporalCertification, "splits"> & {
  splits: Pick<StudyTemporalCertification["splits"], "training" | "validation">;
};

export const sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);
export const probabilityMetricSchema = z.number().finite().min(0).max(1);
export const nonnegativeIntegerSchema = z.number().int().nonnegative();
export const positiveIntegerSchema = z.number().int().positive();
export const bundledRunnerManifestSchema = z.object({
  schema_version: z.literal(1),
  runner: z.object({
    name: z.literal("numeric_logistic_regression"),
    version: z.literal(NUMERIC_LOGISTIC_REGRESSION_RUNNER_VERSION),
  }).strict(),
  parameters: numericLogisticRegressionParametersSchema,
  pipeline: z.object({
    imputer: z.object({
      strategy: z.literal("median"),
      add_indicator: z.literal(true),
    }).strict(),
    scaler: z.literal("standard"),
    classifier: z.object({
      name: z.literal("logistic_regression"),
      solver: z.literal("liblinear"),
      penalty: z.literal("l2"),
      tolerance: z.literal(1e-4),
    }).strict(),
  }).strict(),
  input_columns: z.array(exactStringSchema).min(1),
  training: z.object({
    row_count: positiveIntegerSchema,
    class_counts: z.object({
      negative: positiveIntegerSchema,
      positive: positiveIntegerSchema,
    }).strict(),
    missing_counts: z.record(z.string(), nonnegativeIntegerSchema),
    transformed_feature_count: positiveIntegerSchema,
  }).strict(),
  runtime: bundledRuntimeVersionsSchema,
  model: z.object({
    path: z.literal("model.joblib"),
    sha256: sha256Schema,
    size_bytes: positiveIntegerSchema,
  }).strict(),
}).strict();
const temporalRangeSchema = z.object({
  min: exactStringSchema,
  max: exactStringSchema,
}).strict();
const trialTemporalEvidenceSchema: z.ZodType<StudyTrialTemporalEvidence> = z.object({
  policy: z.literal("ordered_purged"),
  event_time_column: exactStringSchema,
  label_end_time_column: exactStringSchema,
  label_horizon_seconds: positiveIntegerSchema,
  embargo_seconds: nonnegativeIntegerSchema,
  splits: z.object({
    training: z.object({
      event_time: temporalRangeSchema,
      label_end_time: temporalRangeSchema,
    }).strict(),
    validation: z.object({
      event_time: temporalRangeSchema,
      label_end_time: temporalRangeSchema,
    }).strict(),
  }).strict(),
}).strict();
const studyImplementationReferenceSchema: z.ZodType<StudyImplementationReference> = z.object({
  manifest: z.literal("implementation/manifest.json"),
  sha256: sha256Schema,
  file_count: nonnegativeIntegerSchema,
  evidence: z.enum(["bundled_locked", "declared_files"]),
}).strict();
export const studyModelReferenceSchema: z.ZodType<StudyModelReference> = z.object({
  manifest: z.literal("model-manifest.json"),
  sha256: sha256Schema,
  file_count: nonnegativeIntegerSchema,
  size_bytes: nonnegativeIntegerSchema,
}).strict();

export const studyTrialReportSchema = z.object({
  schema_version: z.literal(1),
  protocol_version: z.literal(STUDY_TRIAL_PROTOCOL_VERSION),
  trial: z.object({
    id: studyTrialSpecSchema.shape.id,
    name: exactStringSchema,
    spec_sha256: sha256Schema,
    parameters: z.record(z.string(), studyTrialJsonValueSchema),
  }).strict(),
  study: z.object({
    name: exactStringSchema,
    study_spec_sha256: sha256Schema,
    benchmark_lock_sha256: sha256Schema,
    task_type: z.literal("binary_classification"),
    primary_metric: z.enum(["average_precision", "roc_auc", "f1"]),
  }).strict(),
  data: z.object({
    training: z.object({
      source_sha256: sha256Schema,
      projected_sha256: sha256Schema,
      row_count: positiveIntegerSchema,
    }).strict(),
    validation: z.object({
      source_sha256: sha256Schema,
      projected_sha256: sha256Schema,
      row_count: positiveIntegerSchema,
    }).strict(),
    temporal: trialTemporalEvidenceSchema.optional(),
  }).strict(),
  evaluation: z.object({
    score_semantics: z.literal("positive_class_probability"),
    primary_score: probabilityMetricSchema,
    metrics: z.object({
      average_precision: probabilityMetricSchema,
      roc_auc: probabilityMetricSchema,
      f1_at_0_5: probabilityMetricSchema,
    }).strict(),
    decision_threshold: z.literal(0.5),
    prediction_count: positiveIntegerSchema,
    predictions_sha256: sha256Schema,
  }).strict(),
  provenance: z.object({
    implementation: studyImplementationReferenceSchema,
    model: studyModelReferenceSchema,
  }).strict(),
  execution: z.object({
    command: z.array(exactStringSchema).min(1),
    cwd: exactStringSchema,
    timeout_ms: studyTrialTimeoutSchema,
    duration_ms: nonnegativeIntegerSchema,
    log: z.literal("trial.log"),
    artifact_directory: z.literal("model"),
  }).strict(),
}).strict();

export type StudyTrialReport = z.infer<typeof studyTrialReportSchema>;


export interface LoadedTrialSpec {
  path: string;
  spec: StudyTrialSpec;
}

export interface StudyTrialRunnerCommand {
  command: string;
  baseArgs: string[];
  reportCommand: string[];
  requiredFiles: string[];
}

export function studyImplementationInput(
  trial: LoadedTrialSpec,
): {
  evidence: "bundled_locked" | "declared_files";
  files: StudyImplementationInputFile[];
} {
  if ("builtin" in trial.spec.runner) {
    return {
      evidence: "bundled_locked",
      files: [
        {
          role: "source",
          path: NUMERIC_LOGISTIC_REGRESSION_LOGICAL_PATH,
          absolutePath: NUMERIC_LOGISTIC_REGRESSION_SCRIPT,
        },
        {
          role: "dependency_lock",
          path: `${NUMERIC_LOGISTIC_REGRESSION_LOGICAL_PATH}.lock`,
          absolutePath: NUMERIC_LOGISTIC_REGRESSION_LOCK,
        },
      ],
    };
  }
  const trialDirectory = dirname(trial.path);
  return {
    evidence: "declared_files",
    files: [
      ...trial.spec.runner.provenance.source_files.map((path) => ({
        role: "source" as const,
        path,
        absolutePath: resolve(trialDirectory, path),
        rootPath: trialDirectory,
      })),
      ...trial.spec.runner.provenance.dependency_lock_files.map((path) => ({
        role: "dependency_lock" as const,
        path,
        absolutePath: resolve(trialDirectory, path),
        rootPath: trialDirectory,
      })),
    ],
  };
}

interface ParsedStudySplit {
  rows: string[][];
  columns: string[];
  sha256: string;
}

export interface PreparedTrialData {
  trainingCsv: string;
  validationCsv: string;
  validationLabels: Array<{ id: string; positive: boolean }>;
  source: {
    training: { sha256: string; rowCount: number };
    validation: { sha256: string; rowCount: number };
  };
}

export function schemaError(error: z.ZodError): string {
  return error.issues
    .map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`)
    .join("; ");
}

export function sha256(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

export function canonicalJson(value: unknown): string {
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

export async function assertRegularFile(path: string, description: string): Promise<void> {
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

export async function assertDirectory(path: string, description: string): Promise<void> {
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

export async function readStableRegularFile(args: {
  path: string;
  description: string;
  maxBytes?: number;
  missingMessage?: string;
  requireSingleLink?: boolean;
}): Promise<Uint8Array> {
  let handle: Awaited<ReturnType<typeof open>>;
  try {
    handle = await open(
      args.path,
      constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK,
    );
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
    if (args.requireSingleLink && before.nlink !== 1) {
      throw new Error(`${args.description} must not have hard-link aliases: ${args.path}`);
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
      || before.mtimeMs !== after.mtimeMs
      || before.ctimeMs !== after.ctimeMs
      || before.nlink !== after.nlink
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
      || current.mtimeMs !== after.mtimeMs
      || current.ctimeMs !== after.ctimeMs
      || current.nlink !== after.nlink
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
  const bytes = await readStableRegularFile({
    path,
    description: "study trial spec",
    maxBytes: MAX_TRIAL_REPORT_BYTES,
  });
  let value: unknown;
  try {
    value = JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(bytes),
    ) as unknown;
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

export async function prepareTrialData(args: {
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
        NUMERIC_LOGISTIC_REGRESSION_LOCK,
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

export function trialProtocolInput(args: {
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

export async function readStudyTrialOutput(
  predictionPath: string,
  options: { requireSingleLink?: boolean } = {},
): Promise<{ bytes: Uint8Array; output: StudyTrialOutput }> {
  const bytes = await readStableRegularFile({
    path: predictionPath,
    description: "trial predictions",
    maxBytes: MAX_PREDICTION_BYTES,
    missingMessage: `Trial command did not write predictions at ${predictionPath}`,
    requireSingleLink: options.requireSingleLink,
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
  return { bytes, output: parsed.data };
}

export async function parsePredictions(
  predictionPath: string,
  expectedLabels: readonly { id: string; positive: boolean }[],
  options: {
    requireSingleLink?: boolean;
    splitDescription?: string;
  } = {},
): Promise<{ bytes: Uint8Array; rows: BinaryScoredRow[] }> {
  const { bytes, output } = await readStudyTrialOutput(predictionPath, options);
  const splitDescription = options.splitDescription ?? "validation";
  if (output.predictions.length !== expectedLabels.length) {
    throw new Error(
      `Trial returned ${output.predictions.length} predictions for `
      + `${expectedLabels.length} ${splitDescription} examples`,
    );
  }

  const expectedIds = new Set(expectedLabels.map((row) => row.id));
  const probabilities = new Map<string, number>();
  for (const prediction of output.predictions) {
    if (!expectedIds.has(prediction.id)) {
      throw new Error(
        `Trial returned unknown ${splitDescription} ID "${prediction.id}"`,
      );
    }
    if (probabilities.has(prediction.id)) {
      throw new Error(
        `Trial returned duplicate ${splitDescription} ID "${prediction.id}"`,
      );
    }
    probabilities.set(prediction.id, prediction.probability);
  }

  const missing = expectedLabels.find((row) => !probabilities.has(row.id));
  if (missing) {
    throw new Error(
      `Trial did not return ${splitDescription} ID "${missing.id}"`,
    );
  }
  return {
    bytes,
    rows: expectedLabels.map((row) => ({
      ...row,
      probability: probabilities.get(row.id)!,
    })),
  };
}

export function primaryScore(
  metric: BinaryClassificationStudyTask["primary_metric"],
  metrics: BinaryClassificationMetrics,
): number {
  if (metric === "f1") return metrics.f1_at_0_5;
  return metrics[metric];
}

export async function claimTrialDirectory(outputRoot: string, trialId: string): Promise<string> {
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
