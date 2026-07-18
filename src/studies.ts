import { createHash, randomUUID } from "node:crypto";
import { link, mkdir, readFile, realpath, rm, stat, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";
import { z } from "zod";
import { writeJsonAtomic } from "./artifacts.js";
import { parseCsvRecords } from "./labeling.js";

const exactStringSchema = z.string().min(1).refine(
  (value) => value.trim() === value,
  { message: "must not have outer whitespace" },
);
const columnNameSchema = exactStringSchema;
const relativeDatasetPathSchema = exactStringSchema.refine(
  (value) => (
    !isAbsolute(value)
    && !value.includes("\\")
    && !/^[a-zA-Z]:[\\/]/.test(value)
    && !/^[a-z][a-z0-9+.-]*:/i.test(value)
  ),
  { message: "must be a portable relative local path" },
);
const sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);
const UTC_TIMESTAMP_PATTERN = (
  /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?Z$/
);

function utcTimestampNanoseconds(value: string): bigint | undefined {
  const match = UTC_TIMESTAMP_PATTERN.exec(value);
  if (!match) return undefined;
  const milliseconds = Date.parse(`${match[1]}Z`);
  const normalized = Number.isFinite(milliseconds)
    ? new Date(milliseconds).toISOString()
    : "";
  if (normalized !== `${match[1]}.000Z`) return undefined;
  const fraction = (match[2] ?? "").padEnd(9, "0");
  return BigInt(milliseconds) * 1_000_000n + BigInt(fraction || "0");
}

const utcTimestampSchema = exactStringSchema.regex(
  UTC_TIMESTAMP_PATTERN,
  "must be an RFC 3339 UTC timestamp ending in Z",
).refine(
  (value) => utcTimestampNanoseconds(value) !== undefined,
  "must be a valid UTC calendar timestamp",
);
const studyTemporalPolicySchema = z.object({
  policy: z.literal("ordered_purged"),
  event_time_column: columnNameSchema,
  label_end_time_column: columnNameSchema,
  label_horizon_seconds: z.number().int().min(1).max(2 ** 32 - 1),
  embargo_seconds: z.number().int().min(0).max(2 ** 32 - 1),
}).strict();

export const binaryClassificationStudyTaskSchema = z.object({
  type: z.literal("binary_classification"),
  id_column: columnNameSchema,
  input_columns: z.array(columnNameSchema).min(1),
  target_column: columnNameSchema,
  labels: z.object({
    negative: exactStringSchema,
    positive: exactStringSchema,
  }).strict(),
  primary_metric: z.enum(["average_precision", "roc_auc", "f1"]),
}).strict().superRefine((task, ctx) => {
  if (task.labels.negative === task.labels.positive) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["labels"],
      message: "negative and positive labels must differ",
    });
  }
  const seenInputs = new Set<string>();
  for (const [index, column] of task.input_columns.entries()) {
    if (seenInputs.has(column)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["input_columns", index],
        message: `duplicate input column "${column}"`,
      });
    }
    seenInputs.add(column);
  }
  if (task.id_column === task.target_column) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["target_column"],
      message: "target column must differ from the ID column",
    });
  }
  for (const [index, column] of task.input_columns.entries()) {
    if (column === task.id_column || column === task.target_column) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["input_columns", index],
        message: `input column "${column}" must differ from ID and target columns`,
      });
    }
  }
});

export const studySpecSchema = z.object({
  schema_version: z.literal(1),
  name: exactStringSchema,
  task: binaryClassificationStudyTaskSchema,
  dataset: z.object({
    format: z.literal("csv"),
    splits: z.object({
      training: relativeDatasetPathSchema,
      validation: relativeDatasetPathSchema,
      test: relativeDatasetPathSchema,
    }).strict(),
    temporal: studyTemporalPolicySchema.optional(),
  }).strict(),
}).strict().superRefine((study, context) => {
  const temporal = study.dataset.temporal;
  if (!temporal) return;
  if (
    temporal.event_time_column === study.task.id_column
    || temporal.event_time_column === study.task.target_column
  ) {
    context.addIssue({
      code: "custom",
      path: ["dataset", "temporal", "event_time_column"],
      message: "event time column must differ from ID and target columns",
    });
  }
  if (temporal.event_time_column === temporal.label_end_time_column) {
    context.addIssue({
      code: "custom",
      path: ["dataset", "temporal", "label_end_time_column"],
      message: "label end time column must differ from event time column",
    });
  }
  if (
    temporal.label_end_time_column === study.task.id_column
    || temporal.label_end_time_column === study.task.target_column
  ) {
    context.addIssue({
      code: "custom",
      path: ["dataset", "temporal", "label_end_time_column"],
      message: "label end time column must differ from ID and target columns",
    });
  }
  if (study.task.input_columns.includes(temporal.label_end_time_column)) {
    context.addIssue({
      code: "custom",
      path: ["dataset", "temporal", "label_end_time_column"],
      message: "future label end time must not be a model input",
    });
  }
});

const studySplitSummarySchema = z.object({
  path: exactStringSchema,
  sha256: sha256Schema,
  size_bytes: z.number().int().nonnegative(),
  row_count: z.number().int().positive(),
}).strict();

const temporalRangeSchema = z.object({
  min: utcTimestampSchema,
  max: utcTimestampSchema,
}).strict().superRefine((range, context) => {
  const min = utcTimestampNanoseconds(range.min);
  const max = utcTimestampNanoseconds(range.max);
  if (min !== undefined && max !== undefined && max < min) {
    context.addIssue({
      code: "custom",
      path: ["max"],
      message: "must be at or after min",
    });
  }
});

const temporalSplitSummarySchema = z.object({
  event_time: temporalRangeSchema,
  label_end_time: temporalRangeSchema,
}).strict();

const studyTemporalCertificationSchema = studyTemporalPolicySchema.extend({
  splits: z.object({
    training: temporalSplitSummarySchema,
    validation: temporalSplitSummarySchema,
    test: temporalSplitSummarySchema,
  }).strict(),
}).strict();

export const studyBenchmarkLockSchema = z.object({
  schema_version: z.literal(1),
  study_spec_sha256: sha256Schema,
  study: z.object({
    name: exactStringSchema,
    task: binaryClassificationStudyTaskSchema,
  }).strict(),
  dataset: z.object({
    format: z.literal("csv"),
    columns: z.array(columnNameSchema).min(1),
    splits: z.object({
      training: studySplitSummarySchema,
      validation: studySplitSummarySchema,
      test: studySplitSummarySchema,
    }).strict(),
    temporal: studyTemporalCertificationSchema.optional(),
    total_row_count: z.number().int().positive(),
  }).strict(),
}).strict();

export type BinaryClassificationStudyTask = z.infer<typeof binaryClassificationStudyTaskSchema>;
export type StudySpec = z.infer<typeof studySpecSchema>;
export type StudyBenchmarkLock = z.infer<typeof studyBenchmarkLockSchema>;
export type StudySplitName = keyof StudySpec["dataset"]["splits"];

const STUDY_SPLITS: readonly StudySplitName[] = ["training", "validation", "test"];

interface InspectedSplit {
  columns: string[];
  ids: Set<string>;
  summary: StudyBenchmarkLock["dataset"]["splits"][StudySplitName];
  temporal?: InspectedTemporalSplit;
}

interface ParsedUtcTimestamp {
  nanoseconds: bigint;
  value: string;
}

interface InspectedTemporalSplit {
  summary: {
    event_time: { min: string; max: string };
    label_end_time: { min: string; max: string };
  };
  eventTime: { min: ParsedUtcTimestamp; max: ParsedUtcTimestamp };
  labelEndTime: { min: ParsedUtcTimestamp; max: ParsedUtcTimestamp };
}

function schemaError(error: z.ZodError): string {
  return error.issues
    .map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`)
    .join("; ");
}

function sha256(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function parseUtcTimestamp(
  value: string,
  description: string,
): ParsedUtcTimestamp {
  const match = UTC_TIMESTAMP_PATTERN.exec(value);
  if (!match) {
    throw new Error(`${description} must be an RFC 3339 UTC timestamp ending in Z`);
  }
  const nanoseconds = utcTimestampNanoseconds(value);
  if (nanoseconds === undefined) {
    throw new Error(`${description} is not a valid UTC calendar timestamp`);
  }
  return { value, nanoseconds };
}

function decodeUtf8(bytes: Uint8Array, split: StudySplitName): string {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new Error(`${split} dataset is not valid UTF-8`, { cause: error });
  }
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

function normalizedHeaders(fields: string[], split: StudySplitName): string[] {
  const columns = fields.map((field, index) => (
    index === 0 ? field.replace(/^\uFEFF/, "") : field
  ));
  const seen = new Set<string>();
  for (const [index, column] of columns.entries()) {
    if (!column) {
      throw new Error(`${split} dataset has a blank header at column ${index + 1}`);
    }
    if (column.trim() !== column) {
      throw new Error(`${split} dataset header "${column}" has outer whitespace`);
    }
    if (seen.has(column)) {
      throw new Error(`${split} dataset has duplicate header "${column}"`);
    }
    seen.add(column);
  }
  return columns;
}

function assertRequiredColumns(
  columns: string[],
  task: BinaryClassificationStudyTask,
  split: StudySplitName,
  temporal?: z.infer<typeof studyTemporalPolicySchema>,
): void {
  const available = new Set(columns);
  const required = [task.id_column, ...task.input_columns, task.target_column];
  if (temporal) {
    required.push(temporal.event_time_column, temporal.label_end_time_column);
  }
  for (const column of required) {
    if (!available.has(column)) {
      throw new Error(`${split} dataset is missing required column "${column}"`);
    }
  }
}

async function inspectSplit(args: {
  split: StudySplitName;
  source: string;
  studyDirectory: string;
  task: BinaryClassificationStudyTask;
  temporal?: z.infer<typeof studyTemporalPolicySchema>;
}): Promise<InspectedSplit> {
  const resolvedPath = resolve(args.studyDirectory, args.source);
  await assertRegularFile(resolvedPath, `${args.split} dataset`);
  let bytes: Uint8Array;
  try {
    bytes = await readFile(resolvedPath);
  } catch (error) {
    throw new Error(`Cannot read ${args.split} dataset at ${resolvedPath}`, { cause: error });
  }
  const parsed = parseCsvRecords(decodeUtf8(bytes, args.split), { strict: true });
  if (parsed.errors.length > 0) {
    throw new Error(`Invalid ${args.split} CSV: ${parsed.errors.join("; ")}`);
  }
  if (parsed.fields.length === 0) {
    throw new Error(`${args.split} dataset must contain a header row`);
  }
  if (parsed.records.length === 0) {
    throw new Error(`${args.split} dataset must contain at least one data row`);
  }
  const columns = normalizedHeaders(parsed.fields, args.split);
  assertRequiredColumns(columns, args.task, args.split, args.temporal);
  const idIndex = columns.indexOf(args.task.id_column);
  const targetIndex = columns.indexOf(args.task.target_column);
  const eventTimeIndex = args.temporal
    ? columns.indexOf(args.temporal.event_time_column)
    : -1;
  const labelEndTimeIndex = args.temporal
    ? columns.indexOf(args.temporal.label_end_time_column)
    : -1;
  const ids = new Set<string>();
  let sawNegativeLabel = false;
  let sawPositiveLabel = false;
  let eventTimeMin: ParsedUtcTimestamp | undefined;
  let eventTimeMax: ParsedUtcTimestamp | undefined;
  let labelEndTimeMin: ParsedUtcTimestamp | undefined;
  let labelEndTimeMax: ParsedUtcTimestamp | undefined;

  for (const [index, record] of parsed.records.entries()) {
    const recordNumber = index + 1;
    const id = record[idIndex]!;
    if (!id) {
      throw new Error(`${args.split} dataset record ${recordNumber} has an empty ID`);
    }
    if (id.trim() !== id) {
      throw new Error(`${args.split} dataset record ${recordNumber} ID has outer whitespace`);
    }
    if (ids.has(id)) {
      throw new Error(`${args.split} dataset has duplicate ID "${id}" at record ${recordNumber}`);
    }
    ids.add(id);

    const label = record[targetIndex]!;
    if (!label) {
      throw new Error(`${args.split} dataset record ${recordNumber} has an empty target label`);
    }
    if (label !== args.task.labels.negative && label !== args.task.labels.positive) {
      throw new Error(
        `${args.split} dataset record ${recordNumber} has undeclared label "${label}"`,
      );
    }
    sawNegativeLabel ||= label === args.task.labels.negative;
    sawPositiveLabel ||= label === args.task.labels.positive;

    if (args.temporal) {
      const eventTime = parseUtcTimestamp(
        record[eventTimeIndex]!,
        `${args.split} dataset record ${recordNumber} `
        + `"${args.temporal.event_time_column}"`,
      );
      const labelEndTime = parseUtcTimestamp(
        record[labelEndTimeIndex]!,
        `${args.split} dataset record ${recordNumber} `
        + `"${args.temporal.label_end_time_column}"`,
      );
      if (labelEndTime.nanoseconds <= eventTime.nanoseconds) {
        throw new Error(
          `${args.split} dataset record ${recordNumber} label end time `
          + `${labelEndTime.value} must be after event time ${eventTime.value}`,
        );
      }
      const declaredHorizonEnd = (
        eventTime.nanoseconds
        + BigInt(args.temporal.label_horizon_seconds) * 1_000_000_000n
      );
      if (labelEndTime.nanoseconds > declaredHorizonEnd) {
        throw new Error(
          `${args.split} dataset record ${recordNumber} label end time `
          + `${labelEndTime.value} exceeds the declared `
          + `${args.temporal.label_horizon_seconds}-second label horizon`,
        );
      }
      if (!eventTimeMin || eventTime.nanoseconds < eventTimeMin.nanoseconds) {
        eventTimeMin = eventTime;
      }
      if (!eventTimeMax || eventTime.nanoseconds > eventTimeMax.nanoseconds) {
        eventTimeMax = eventTime;
      }
      if (!labelEndTimeMin || labelEndTime.nanoseconds < labelEndTimeMin.nanoseconds) {
        labelEndTimeMin = labelEndTime;
      }
      if (!labelEndTimeMax || labelEndTime.nanoseconds > labelEndTimeMax.nanoseconds) {
        labelEndTimeMax = labelEndTime;
      }
    }
  }
  if (!sawNegativeLabel || !sawPositiveLabel) {
    throw new Error(
      `${args.split} dataset must contain both declared labels `
      + `"${args.task.labels.negative}" and "${args.task.labels.positive}"`,
    );
  }

  return {
    columns,
    ids,
    summary: {
      path: args.source,
      sha256: sha256(bytes),
      size_bytes: bytes.byteLength,
      row_count: parsed.records.length,
    },
    ...(args.temporal ? {
      temporal: {
        summary: {
          event_time: {
            min: eventTimeMin!.value,
            max: eventTimeMax!.value,
          },
          label_end_time: {
            min: labelEndTimeMin!.value,
            max: labelEndTimeMax!.value,
          },
        },
        eventTime: { min: eventTimeMin!, max: eventTimeMax! },
        labelEndTime: { min: labelEndTimeMin!, max: labelEndTimeMax! },
      },
    } : {}),
  };
}

export async function loadStudySpec(studyPath: string): Promise<{
  path: string;
  spec: StudySpec;
}> {
  const path = resolve(studyPath);
  await assertRegularFile(path, "StudySpec");
  let value: unknown;
  try {
    value = JSON.parse(await readFile(path, "utf8")) as unknown;
  } catch (error) {
    throw new Error(`Cannot read valid StudySpec JSON at ${path}`, { cause: error });
  }
  const parsed = studySpecSchema.safeParse(value);
  if (!parsed.success) {
    throw new Error(`Invalid StudySpec at ${path}: ${schemaError(parsed.error)}`);
  }
  return { path, spec: parsed.data };
}

async function buildStudyBenchmarkLockFromLoaded(
  loaded: Awaited<ReturnType<typeof loadStudySpec>>,
): Promise<StudyBenchmarkLock> {
  const studyDirectory = dirname(loaded.path);
  const inspected = {} as Record<StudySplitName, InspectedSplit>;
  const idOwners = new Map<string, StudySplitName>();
  let expectedColumns: string[] | undefined;

  for (const split of STUDY_SPLITS) {
    const current = await inspectSplit({
      split,
      source: loaded.spec.dataset.splits[split],
      studyDirectory,
      task: loaded.spec.task,
      temporal: loaded.spec.dataset.temporal,
    });
    if (expectedColumns && !isDeepStrictEqual(current.columns, expectedColumns)) {
      throw new Error(
        `${split} dataset columns do not match training dataset columns `
        + `(expected ${JSON.stringify(expectedColumns)}, found ${JSON.stringify(current.columns)})`,
      );
    }
    expectedColumns ??= current.columns;
    for (const id of current.ids) {
      const owner = idOwners.get(id);
      if (owner) {
        throw new Error(`ID "${id}" appears in both ${owner} and ${split} datasets`);
      }
      idOwners.set(id, split);
    }
    inspected[split] = current;
  }

  const temporal = loaded.spec.dataset.temporal;
  if (temporal) {
    const horizon = BigInt(temporal.label_horizon_seconds) * 1_000_000_000n;
    const embargo = BigInt(temporal.embargo_seconds) * 1_000_000_000n;
    for (const [previousName, nextName] of [
      ["training", "validation"],
      ["validation", "test"],
    ] as const) {
      const previous = inspected[previousName].temporal!;
      const next = inspected[nextName].temporal!;
      // Purge against the full declared horizon: observed label evidence may
      // end early when a source dataset has only partial future coverage.
      const declaredHorizonEnd = previous.eventTime.max.nanoseconds + horizon;
      const foundEmbargo = (
        next.eventTime.min.nanoseconds
        - declaredHorizonEnd
      );
      if (foundEmbargo <= embargo) {
        throw new Error(
          `${previousName}-to-${nextName} temporal boundary is not ordered and purged: `
          + `${previousName} has an event at ${previous.eventTime.max.value} with a `
          + `${temporal.label_horizon_seconds}-second label horizon, while `
          + `${nextName} starts at ${next.eventTime.min.value}; the next split must `
          + `start more than ${temporal.embargo_seconds} seconds after the declared `
          + "label horizon ends",
        );
      }
    }
  }

  return studyBenchmarkLockSchema.parse({
    schema_version: 1,
    study_spec_sha256: sha256(JSON.stringify(loaded.spec)),
    study: {
      name: loaded.spec.name,
      task: loaded.spec.task,
    },
    dataset: {
      format: loaded.spec.dataset.format,
      columns: expectedColumns!,
      splits: {
        training: inspected.training.summary,
        validation: inspected.validation.summary,
        test: inspected.test.summary,
      },
      ...(temporal ? {
        temporal: {
          ...temporal,
          splits: {
            training: inspected.training.temporal!.summary,
            validation: inspected.validation.temporal!.summary,
            test: inspected.test.temporal!.summary,
          },
        },
      } : {}),
      total_row_count: STUDY_SPLITS.reduce(
        (total, split) => total + inspected[split].summary.row_count,
        0,
      ),
    },
  });
}

export async function buildStudyBenchmarkLock(studyPath: string): Promise<StudyBenchmarkLock> {
  return buildStudyBenchmarkLockFromLoaded(await loadStudySpec(studyPath));
}

export function defaultStudyLockPath(studyPath: string): string {
  const path = resolve(studyPath);
  return path.toLowerCase().endsWith(".json")
    ? `${path.slice(0, -".json".length)}.lock.json`
    : `${path}.lock.json`;
}

async function canonicalPathIfPresent(path: string): Promise<string> {
  try {
    return await realpath(path);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return resolve(path);
    throw error;
  }
}

async function writeNewStudyLock(path: string, lock: StudyBenchmarkLock): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporary, `${JSON.stringify(lock, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await link(temporary, path);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EEXIST") {
      throw new Error(`Study benchmark lock already exists at ${path}; pass --force to replace it`);
    }
    throw error;
  } finally {
    await rm(temporary, { force: true });
  }
}

async function assertReplaceableStudyLock(path: string): Promise<void> {
  let metadata: Awaited<ReturnType<typeof stat>>;
  try {
    metadata = await stat(path);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    throw new Error(`Refusing to replace non-lock file at ${path}`, { cause: error });
  }
  if (!metadata.isFile()) {
    throw new Error(`Refusing to replace non-lock file at ${path}`);
  }
  let value: unknown;
  try {
    value = JSON.parse(await readFile(path, "utf8")) as unknown;
  } catch (error) {
    throw new Error(`Refusing to replace non-lock file at ${path}`, { cause: error });
  }
  if (!studyBenchmarkLockSchema.safeParse(value).success) {
    throw new Error(`Refusing to replace non-lock file at ${path}`);
  }
}

export async function writeStudyBenchmarkLock(args: {
  studyPath: string;
  outputPath?: string;
  force?: boolean;
}): Promise<{ lockPath: string; lock: StudyBenchmarkLock }> {
  const studyPath = resolve(args.studyPath);
  const lockPath = resolve(args.outputPath ?? defaultStudyLockPath(studyPath));
  const loaded = await loadStudySpec(studyPath);
  const protectedPaths = [
    loaded.path,
    ...STUDY_SPLITS.map((split) => resolve(
      dirname(loaded.path),
      loaded.spec.dataset.splits[split],
    )),
  ];
  const canonicalLockPath = await canonicalPathIfPresent(lockPath);
  const canonicalProtectedPaths = await Promise.all(protectedPaths.map(canonicalPathIfPresent));
  if (
    protectedPaths.includes(lockPath)
    || canonicalProtectedPaths.includes(canonicalLockPath)
  ) {
    throw new Error("Study lock output must not overwrite the StudySpec or a dataset split");
  }
  const lock = await buildStudyBenchmarkLockFromLoaded(loaded);
  if (args.force) {
    await assertReplaceableStudyLock(lockPath);
    await writeJsonAtomic(lockPath, lock);
  } else {
    await writeNewStudyLock(lockPath, lock);
  }
  return { lockPath, lock };
}

function driftValue(value: unknown): string {
  const rendered = JSON.stringify(value);
  if (rendered === undefined) return "<missing>";
  return rendered.length > 160 ? `${rendered.slice(0, 157)}...` : rendered;
}

function collectDrift(
  expected: unknown,
  found: unknown,
  path: string,
  output: string[],
): void {
  if (isDeepStrictEqual(expected, found) || output.length >= 20) return;
  if (Array.isArray(expected) && Array.isArray(found)) {
    const length = Math.max(expected.length, found.length);
    for (let index = 0; index < length; index += 1) {
      collectDrift(expected[index], found[index], `${path}[${index}]`, output);
    }
    return;
  }
  if (
    expected !== null
    && found !== null
    && typeof expected === "object"
    && typeof found === "object"
    && !Array.isArray(expected)
    && !Array.isArray(found)
  ) {
    const keys = [...new Set([
      ...Object.keys(expected as Record<string, unknown>),
      ...Object.keys(found as Record<string, unknown>),
    ])].sort();
    for (const key of keys) {
      collectDrift(
        (expected as Record<string, unknown>)[key],
        (found as Record<string, unknown>)[key],
        path ? `${path}.${key}` : key,
        output,
      );
    }
    return;
  }
  output.push(`${path}: expected ${driftValue(expected)}, found ${driftValue(found)}`);
}

async function readStudyBenchmarkLock(lockPath: string): Promise<StudyBenchmarkLock> {
  const path = resolve(lockPath);
  await assertRegularFile(path, "Study benchmark lock");
  let value: unknown;
  try {
    value = JSON.parse(await readFile(path, "utf8")) as unknown;
  } catch (error) {
    throw new Error(`Cannot read valid study benchmark lock JSON at ${path}`, { cause: error });
  }
  const parsed = studyBenchmarkLockSchema.safeParse(value);
  if (!parsed.success) {
    throw new Error(`Invalid study benchmark lock at ${path}: ${schemaError(parsed.error)}`);
  }
  return parsed.data;
}

export async function validateStudyBenchmark(args: {
  studyPath: string;
  lockPath?: string;
}): Promise<{ lockPath: string; lock: StudyBenchmarkLock }> {
  const studyPath = resolve(args.studyPath);
  const lockPath = resolve(args.lockPath ?? defaultStudyLockPath(studyPath));
  const expected = await readStudyBenchmarkLock(lockPath);
  const found = await buildStudyBenchmarkLock(studyPath);
  const drift: string[] = [];
  collectDrift(expected, found, "", drift);
  if (drift.length > 0) {
    throw new Error(
      `Benchmark lock drift detected:\n- ${drift.join("\n- ")}\n`
      + "Refresh the lock with `tt-local studies lock <study.json> --force` only if this change is intentional.",
    );
  }
  return { lockPath, lock: found };
}
