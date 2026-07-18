import { randomUUID } from "node:crypto";
import {
  lstat,
  mkdir,
  readdir,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { z } from "zod";
import { writeJsonAtomic } from "./artifacts.js";
import { minimalMachineLearningEnvironment } from "./huggingface-cache.js";
import { parseCsvRecords } from "./labeling.js";
import { runLoggedProcess } from "./process-runner.js";
import { computeBinaryClassificationMetrics } from "./study-metrics.js";
import {
  MAX_PROVENANCE_MANIFEST_BYTES,
  MAX_TRIAL_REPORT_BYTES,
  NUMERIC_LOGISTIC_REGRESSION_LOCK,
  NUMERIC_LOGISTIC_REGRESSION_LOGICAL_PATH,
  NUMERIC_LOGISTIC_REGRESSION_SCRIPT,
  STUDY_TRIAL_PROTOCOL_VERSION,
  bundledRunnerManifestSchema,
  canonicalJson,
  datasetIdSchema,
  defaultStudyTrialOutputRoot,
  exactStringSchema,
  loadStudyTrialSpec,
  nonnegativeIntegerSchema,
  parsePredictions,
  positiveIntegerSchema,
  prepareTrialData,
  primaryScore,
  probabilityMetricSchema,
  provenanceFilePathSchema,
  readStableRegularFile,
  schemaError,
  sha256,
  sha256Schema,
  studyModelReferenceSchema,
  studyTrialOutputSchema,
  studyTrialReportSchema,
  studyTrialSpecSchema,
  type StudyTrialTemporalEvidence,
} from "./study-trial-core.js";
import {
  captureDirectoryIdentity,
  inspectStudyModel,
  verifyDirectoryIdentity,
  verifyStudyImplementationSnapshot,
  type StudyFileEvidence,
  type StudyImplementationManifest,
} from "./study-provenance.js";
import {
  loadStudySpec,
  validateStudyBenchmark,
  type StudyBenchmarkLock,
} from "./studies.js";

// Study candidate contracts, verification, and publication.
const MAX_CANDIDATE_LOG_BYTES = 16 * 1024 * 1024;
const MAX_CANDIDATE_ARTIFACT_BYTES = 256 * 1024 * 1024;
const REPLAY_PROBABILITY_TOLERANCE = 1e-12;
const CANDIDATE_PREDICTOR_FILES = [
  "implementation-manifest.json",
  "numeric_logistic_regression.py",
  "numeric_logistic_regression.py.lock",
] as const;

const studyImplementationFileSchema = z.object({
  role: z.enum(["source", "dependency_lock"]),
  path: provenanceFilePathSchema,
  snapshot_path: provenanceFilePathSchema,
  size_bytes: nonnegativeIntegerSchema,
  sha256: sha256Schema,
}).strict();
const studyImplementationManifestSchema: z.ZodType<StudyImplementationManifest> = z.object({
  schema_version: z.literal(1),
  evidence: z.enum(["bundled_locked", "declared_files"]),
  files: z.array(studyImplementationFileSchema).max(288),
}).strict();
const candidateReplayRequestSchema = z.object({
  protocol_version: z.literal(STUDY_TRIAL_PROTOCOL_VERSION),
  task: z.object({
    type: z.literal("binary_classification"),
    id_column: exactStringSchema,
    input_columns: z.array(exactStringSchema).min(1),
    prediction: z.object({
      field: z.literal("probability"),
      meaning: z.literal("probability_of_positive_target"),
    }).strict(),
  }).strict(),
  dataset: z.object({
    prediction_csv: z.literal("replay/validation.csv"),
  }).strict(),
}).strict();

const candidateArtifactReferenceSchema = z.object({
  path: provenanceFilePathSchema,
  sha256: sha256Schema,
  size_bytes: nonnegativeIntegerSchema.max(MAX_CANDIDATE_ARTIFACT_BYTES),
}).strict();
const candidateArtifactAt = <Path extends string>(path: Path) => (
  candidateArtifactReferenceSchema.extend({ path: z.literal(path) }).strict()
);

export const studyCandidateLockSchema = z.object({
  schema_version: z.literal(1),
  protocol_version: z.literal(STUDY_TRIAL_PROTOCOL_VERSION),
  kind: z.literal("study_candidate"),
  study: z.object({
    name: exactStringSchema,
    study_spec_sha256: sha256Schema,
    benchmark_lock_sha256: sha256Schema,
    task_type: z.literal("binary_classification"),
    primary_metric: z.enum(["average_precision", "roc_auc", "f1"]),
  }).strict(),
  trial: z.object({
    id: studyTrialSpecSchema.shape.id,
    name: exactStringSchema,
    spec_sha256: sha256Schema,
    spec_artifact: candidateArtifactAt("selection/trial-spec.json"),
    report_artifact: candidateArtifactAt("selection/trial-report.json"),
  }).strict(),
  selection: z.object({
    validation_primary_score: probabilityMetricSchema,
    validation_prediction_count: positiveIntegerSchema,
    validation_predictions: candidateArtifactAt(
      "selection/validation-predictions.json",
    ),
  }).strict(),
  artifacts: z.object({
    model: studyModelReferenceSchema,
    predictor: z.object({
      implementation_manifest: candidateArtifactAt(
        "predictor/implementation-manifest.json",
      ),
      script: candidateArtifactAt("predictor/numeric_logistic_regression.py"),
      dependency_lock: candidateArtifactAt(
        "predictor/numeric_logistic_regression.py.lock",
      ),
    }).strict(),
  }).strict(),
  training_runtime: bundledRunnerManifestSchema.shape.runtime,
  replay: z.object({
    validation_projection: candidateArtifactAt("replay/validation.csv"),
    request: candidateArtifactAt("replay/predictor-input.json"),
    predictions: candidateArtifactAt("replay/predictions.json"),
    log: candidateArtifactAt("replay/predictor.log"),
    prediction_count: positiveIntegerSchema,
    probability_tolerance: z.literal(REPLAY_PROBABILITY_TOLERANCE),
    max_absolute_difference: z.number().finite().min(0),
  }).strict(),
}).strict().superRefine((candidate, context) => {
  if (
    candidate.selection.validation_prediction_count
    !== candidate.replay.prediction_count
  ) {
    context.addIssue({
      code: "custom",
      path: ["replay", "prediction_count"],
      message: "must match selection.validation_prediction_count",
    });
  }
  if (
    candidate.replay.max_absolute_difference
    > candidate.replay.probability_tolerance
  ) {
    context.addIssue({
      code: "custom",
      path: ["replay", "max_absolute_difference"],
      message: "must not exceed probability_tolerance",
    });
  }
});

export type StudyCandidateLock = z.infer<typeof studyCandidateLockSchema>;

interface CandidateArtifactReference<Path extends string = string> {
  path: Path;
  sha256: string;
  size_bytes: number;
}

function jsonText(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function assertJsonEqual(
  actual: unknown,
  expected: unknown,
  description: string,
): void {
  if (canonicalJson(actual) !== canonicalJson(expected)) {
    throw new Error(`${description} does not match the selected Study trial`);
  }
}

function assertBytesMatch(
  bytes: Uint8Array,
  expected: Pick<StudyFileEvidence, "size_bytes" | "sha256">,
  description: string,
): void {
  if (
    bytes.byteLength !== expected.size_bytes
    || sha256(bytes) !== expected.sha256
  ) {
    throw new Error(`${description} does not match its recorded provenance`);
  }
}

async function readStrictJson<T>(args: {
  path: string;
  description: string;
  maxBytes: number;
  schema: z.ZodType<T>;
  requireSingleLink?: boolean;
}): Promise<{ bytes: Uint8Array; value: T }> {
  const bytes = await readStableRegularFile({
    path: args.path,
    description: args.description,
    maxBytes: args.maxBytes,
    requireSingleLink: args.requireSingleLink,
  });
  return {
    bytes,
    value: parseStrictJsonBytes(
      bytes,
      `${args.description} at ${args.path}`,
      args.schema,
    ),
  };
}

function parseStrictJsonBytes<T>(
  bytes: Uint8Array,
  description: string,
  schema: z.ZodType<T>,
): T {
  let value: unknown;
  try {
    value = JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(bytes),
    ) as unknown;
  } catch (error) {
    throw new Error(
      `Cannot read valid JSON for ${description}`,
      { cause: error },
    );
  }
  const parsed = schema.safeParse(value);
  if (!parsed.success) {
    throw new Error(
      `Invalid ${description}: ${schemaError(parsed.error)}`,
    );
  }
  return parsed.data;
}

function candidateArtifactReference<Path extends string>(
  path: Path,
  bytes: string | Uint8Array,
): CandidateArtifactReference<Path> {
  return {
    path,
    sha256: sha256(bytes),
    size_bytes: typeof bytes === "string"
      ? Buffer.byteLength(bytes)
      : bytes.byteLength,
  };
}

async function writeCandidateArtifact<Path extends string>(args: {
  candidateDirectory: string;
  relativePath: Path;
  bytes: string | Uint8Array;
}): Promise<CandidateArtifactReference<Path>> {
  const destination = join(
    args.candidateDirectory,
    ...args.relativePath.split("/"),
  );
  await mkdir(dirname(destination), { recursive: true, mode: 0o700 });
  await writeFile(destination, args.bytes, { flag: "wx", mode: 0o400 });
  return candidateArtifactReference(args.relativePath, args.bytes);
}

async function copyVerifiedCandidateArtifact<Path extends string>(args: {
  candidateDirectory: string;
  relativePath: Path;
  sourcePath: string;
  expected: Pick<StudyFileEvidence, "size_bytes" | "sha256">;
  description: string;
}): Promise<CandidateArtifactReference<Path>> {
  const bytes = await readStableRegularFile({
    path: args.sourcePath,
    description: args.description,
    maxBytes: args.expected.size_bytes,
  });
  assertBytesMatch(bytes, args.expected, args.description);
  return writeCandidateArtifact({
    candidateDirectory: args.candidateDirectory,
    relativePath: args.relativePath,
    bytes,
  });
}

async function ensureRealDirectory(path: string, description: string): Promise<void> {
  try {
    await mkdir(path, { mode: 0o700 });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
  }
  const metadata = await lstat(path);
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
    throw new Error(`${description} must be a real directory: ${path}`);
  }
}

export function defaultStudyCandidateDirectory(studyPath: string): string {
  const path = resolve(studyPath);
  return join(
    dirname(path),
    ".tt-local",
    "study-candidates",
    basename(path),
  );
}

interface ClaimedStudyCandidateDirectory {
  candidateDirectory: string;
  finalDirectory: string;
  localDirectory: string;
  localDirectoryIdentity: Awaited<ReturnType<typeof captureDirectoryIdentity>>;
  candidateRoot: string;
  candidateRootIdentity: Awaited<ReturnType<typeof captureDirectoryIdentity>>;
}

async function claimStudyCandidateDirectory(
  studyPath: string,
): Promise<ClaimedStudyCandidateDirectory> {
  const studyDirectory = dirname(resolve(studyPath));
  const studyDirectoryMetadata = await lstat(studyDirectory);
  if (
    !studyDirectoryMetadata.isDirectory()
    || studyDirectoryMetadata.isSymbolicLink()
  ) {
    throw new Error(`Study directory must be a real directory: ${studyDirectory}`);
  }
  const localDirectory = join(studyDirectory, ".tt-local");
  const candidateRoot = join(localDirectory, "study-candidates");
  await ensureRealDirectory(localDirectory, "TT Local state directory");
  await ensureRealDirectory(candidateRoot, "Study candidate root");

  const finalDirectory = defaultStudyCandidateDirectory(studyPath);
  try {
    await lstat(finalDirectory);
    throw new Error(
      `A Study candidate selection already exists at ${finalDirectory}; `
      + "candidate selection is write-once",
    );
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  const candidateDirectory = join(
    candidateRoot,
    `.${basename(resolve(studyPath))}.promoting-${process.pid}-${randomUUID()}`,
  );
  await mkdir(candidateDirectory, { mode: 0o700 });
  return {
    candidateDirectory,
    finalDirectory,
    localDirectory,
    localDirectoryIdentity: await captureDirectoryIdentity(localDirectory),
    candidateRoot,
    candidateRootIdentity: await captureDirectoryIdentity(candidateRoot),
  };
}

async function verifyStudyCandidateAncestors(
  claimed: ClaimedStudyCandidateDirectory,
): Promise<void> {
  await Promise.all([
    verifyDirectoryIdentity({
      path: claimed.localDirectory,
      expected: claimed.localDirectoryIdentity,
      description: "TT Local state directory",
    }),
    verifyDirectoryIdentity({
      path: claimed.candidateRoot,
      expected: claimed.candidateRootIdentity,
      description: "Study candidate root",
    }),
  ]);
}

async function removeOwnedCandidateStaging(
  claimed: ClaimedStudyCandidateDirectory,
  expected: Awaited<ReturnType<typeof captureDirectoryIdentity>>,
): Promise<void> {
  try {
    await verifyDirectoryIdentity({
      path: claimed.candidateDirectory,
      expected,
      description: "Study candidate staging directory",
    });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    throw error;
  }
  await rm(claimed.candidateDirectory, { recursive: true });
}

function expectedTrialTemporalEvidence(
  lock: StudyBenchmarkLock,
): StudyTrialTemporalEvidence | undefined {
  const temporal = lock.dataset.temporal;
  if (!temporal) return undefined;
  return {
    policy: temporal.policy,
    event_time_column: temporal.event_time_column,
    label_end_time_column: temporal.label_end_time_column,
    label_horizon_seconds: temporal.label_horizon_seconds,
    embargo_seconds: temporal.embargo_seconds,
    splits: {
      training: temporal.splits.training,
      validation: temporal.splits.validation,
    },
  };
}

function studyCandidateArtifactReferences(
  candidate: StudyCandidateLock,
): CandidateArtifactReference[] {
  return [
    candidate.trial.spec_artifact,
    candidate.trial.report_artifact,
    candidate.selection.validation_predictions,
    candidate.artifacts.predictor.implementation_manifest,
    candidate.artifacts.predictor.script,
    candidate.artifacts.predictor.dependency_lock,
    candidate.replay.validation_projection,
    candidate.replay.request,
    candidate.replay.predictions,
    candidate.replay.log,
  ];
}

function candidateArtifactBytes(
  artifacts: ReadonlyMap<string, Uint8Array>,
  reference: CandidateArtifactReference,
): Uint8Array {
  const bytes = artifacts.get(reference.path);
  if (!bytes) {
    throw new Error(`Missing verified Study candidate artifact ${reference.path}`);
  }
  return bytes;
}

function candidatePredictionProbabilities(
  bytes: Uint8Array,
  description: string,
): Map<string, number> {
  const output = parseStrictJsonBytes(bytes, description, studyTrialOutputSchema);
  const probabilities = new Map<string, number>();
  for (const prediction of output.predictions) {
    if (probabilities.has(prediction.id)) {
      throw new Error(`${description} contains duplicate ID "${prediction.id}"`);
    }
    probabilities.set(prediction.id, prediction.probability);
  }
  return probabilities;
}

async function inspectCandidatePredictorDirectory(
  candidateDirectory: string,
): Promise<Awaited<ReturnType<typeof captureDirectoryIdentity>>> {
  const predictorDirectory = join(candidateDirectory, "predictor");
  const identity = await captureDirectoryIdentity(predictorDirectory);
  const entries = await readdir(predictorDirectory, { withFileTypes: true });
  const actual = entries.map((entry) => ({
    name: entry.name,
    type: entry.isFile() ? "file" : "other",
  })).sort((left, right) => left.name.localeCompare(right.name));
  const expected = CANDIDATE_PREDICTOR_FILES.map((name) => ({
    name,
    type: "file",
  }));
  assertJsonEqual(
    actual,
    expected,
    "Study candidate predictor file set",
  );
  return identity;
}

async function verifyStudyCandidateSemanticBindings(args: {
  candidateDirectory: string;
  candidate: StudyCandidateLock;
  artifacts: ReadonlyMap<string, Uint8Array>;
  model: Awaited<ReturnType<typeof inspectStudyModel>>;
}): Promise<void> {
  const trialSpec = parseStrictJsonBytes(
    candidateArtifactBytes(args.artifacts, args.candidate.trial.spec_artifact),
    "Study candidate trial spec",
    studyTrialSpecSchema,
  );
  if (
    !("builtin" in trialSpec.runner)
    || trialSpec.runner.builtin !== "numeric_logistic_regression"
  ) {
    throw new Error(
      "Study candidate trial spec must use the bundled numeric logistic-regression runner",
    );
  }
  const trialSpecSha256 = sha256(canonicalJson(trialSpec));
  assertJsonEqual(args.candidate.trial, {
    id: trialSpec.id,
    name: trialSpec.name,
    spec_sha256: trialSpecSha256,
    spec_artifact: args.candidate.trial.spec_artifact,
    report_artifact: args.candidate.trial.report_artifact,
  }, "Study candidate trial identity");

  const report = parseStrictJsonBytes(
    candidateArtifactBytes(args.artifacts, args.candidate.trial.report_artifact),
    "Study candidate trial report",
    studyTrialReportSchema,
  );
  assertJsonEqual(report.study, args.candidate.study, "Study candidate benchmark identity");
  assertJsonEqual(report.trial, {
    id: trialSpec.id,
    name: trialSpec.name,
    spec_sha256: trialSpecSha256,
    parameters: trialSpec.parameters,
  }, "Study candidate report trial identity");
  if (
    report.evaluation.primary_score
    !== primaryScore(report.study.primary_metric, report.evaluation.metrics)
  ) {
    throw new Error(
      "Study candidate report primary score does not match its primary metric",
    );
  }
  assertJsonEqual({
    command: report.execution.command,
    cwd: report.execution.cwd,
    timeout_ms: report.execution.timeout_ms,
    log: report.execution.log,
    artifact_directory: report.execution.artifact_directory,
  }, {
    command: ["builtin:numeric_logistic_regression"],
    cwd: "<trial-directory>",
    timeout_ms: trialSpec.runner.timeout_ms,
    log: "trial.log",
    artifact_directory: "model",
  }, "Study candidate report execution contract");
  assertJsonEqual(report.provenance.model, args.candidate.artifacts.model, (
    "Study candidate report model provenance"
  ));
  const implementationManifest = parseStrictJsonBytes(
    candidateArtifactBytes(
      args.artifacts,
      args.candidate.artifacts.predictor.implementation_manifest,
    ),
    "Study candidate predictor implementation manifest",
    studyImplementationManifestSchema,
  );
  if (
    report.provenance.implementation.evidence
      !== implementationManifest.evidence
    || report.provenance.implementation.file_count
      !== implementationManifest.files.length
    || report.provenance.implementation.sha256
      !== args.candidate.artifacts.predictor.implementation_manifest.sha256
  ) {
    throw new Error(
      "Study candidate predictor provenance does not match the selected trial report",
    );
  }
  assertJsonEqual(implementationManifest, {
    schema_version: 1,
    evidence: "bundled_locked",
    files: [
      {
        role: "source",
        path: NUMERIC_LOGISTIC_REGRESSION_LOGICAL_PATH,
        snapshot_path: (
          `implementation/source/${NUMERIC_LOGISTIC_REGRESSION_LOGICAL_PATH}`
        ),
        size_bytes: args.candidate.artifacts.predictor.script.size_bytes,
        sha256: args.candidate.artifacts.predictor.script.sha256,
      },
      {
        role: "dependency_lock",
        path: `${NUMERIC_LOGISTIC_REGRESSION_LOGICAL_PATH}.lock`,
        snapshot_path: (
          `implementation/dependency-lock/`
          + `${NUMERIC_LOGISTIC_REGRESSION_LOGICAL_PATH}.lock`
        ),
        size_bytes: args.candidate.artifacts.predictor.dependency_lock.size_bytes,
        sha256: args.candidate.artifacts.predictor.dependency_lock.sha256,
      },
    ],
  }, "Study candidate predictor implementation provenance");
  if (
    report.evaluation.primary_score
      !== args.candidate.selection.validation_primary_score
    || report.evaluation.prediction_count
      !== args.candidate.selection.validation_prediction_count
    || report.evaluation.predictions_sha256
      !== args.candidate.selection.validation_predictions.sha256
  ) {
    throw new Error(
      "Study candidate validation selection does not match the selected trial report",
    );
  }
  if (
    report.data.validation.projected_sha256
      !== args.candidate.replay.validation_projection.sha256
  ) {
    throw new Error(
      "Study candidate replay projection does not match the selected validation data",
    );
  }

  assertJsonEqual(
    args.model.manifest.files.map((file) => file.path),
    ["model.joblib", "runner-manifest.json"],
    "Study candidate bundled model layout",
  );
  const runnerManifestArtifact = await readStrictJson({
    path: join(args.candidateDirectory, "model", "runner-manifest.json"),
    description: "Study candidate bundled runner manifest",
    maxBytes: MAX_TRIAL_REPORT_BYTES,
    schema: bundledRunnerManifestSchema,
    requireSingleLink: true,
  });
  const runnerManifest = runnerManifestArtifact.value;
  assertJsonEqual(
    runnerManifest.parameters,
    trialSpec.parameters,
    "Study candidate bundled runner parameters",
  );
  assertJsonEqual(
    runnerManifest.runtime,
    args.candidate.training_runtime,
    "Study candidate training runtime",
  );
  if (
    runnerManifest.training.row_count !== report.data.training.row_count
    || (
      runnerManifest.training.class_counts.negative
      + runnerManifest.training.class_counts.positive
    ) !== runnerManifest.training.row_count
  ) {
    throw new Error(
      "Study candidate bundled runner training counts do not match the trial report",
    );
  }
  const modelFile = args.model.manifest.files.find(
    (file) => file.path === "model.joblib",
  )!;
  const runnerManifestFile = args.model.manifest.files.find(
    (file) => file.path === "runner-manifest.json",
  )!;
  assertBytesMatch(
    runnerManifestArtifact.bytes,
    runnerManifestFile,
    "Study candidate bundled runner manifest",
  );
  assertJsonEqual(runnerManifest.model, {
    path: "model.joblib",
    sha256: modelFile.sha256,
    size_bytes: modelFile.size_bytes,
  }, "Study candidate bundled runner saved-model evidence");

  const replayRequest = parseStrictJsonBytes(
    candidateArtifactBytes(args.artifacts, args.candidate.replay.request),
    "Study candidate replay request",
    candidateReplayRequestSchema,
  );
  assertJsonEqual(
    replayRequest.task.input_columns,
    runnerManifest.input_columns,
    "Study candidate replay input columns",
  );
  if (replayRequest.task.type !== args.candidate.study.task_type) {
    throw new Error("Study candidate replay task type does not match the benchmark");
  }

  let replayCsv: string;
  try {
    replayCsv = new TextDecoder("utf-8", { fatal: true }).decode(
      candidateArtifactBytes(
        args.artifacts,
        args.candidate.replay.validation_projection,
      ),
    );
  } catch (error) {
    throw new Error("Study candidate replay projection is not valid UTF-8", {
      cause: error,
    });
  }
  const projection = parseCsvRecords(replayCsv, { strict: true });
  if (projection.errors.length > 0) {
    throw new Error(
      `Invalid Study candidate replay projection: ${projection.errors.join("; ")}`,
    );
  }
  assertJsonEqual(projection.fields, [
    replayRequest.task.id_column,
    ...replayRequest.task.input_columns,
  ], "Study candidate replay projection columns");
  if (
    projection.records.length !== report.data.validation.row_count
    || projection.records.length
      !== args.candidate.selection.validation_prediction_count
  ) {
    throw new Error(
      "Study candidate replay row count does not match the selected validation data",
    );
  }
  const projectionIds = new Set<string>();
  for (const [index, record] of projection.records.entries()) {
    const parsedId = datasetIdSchema.safeParse(record[0]);
    if (!parsedId.success) {
      throw new Error(
        `Invalid Study candidate replay ID on row ${index + 1}: `
        + schemaError(parsedId.error),
      );
    }
    if (projectionIds.has(parsedId.data)) {
      throw new Error(
        `Study candidate replay projection contains duplicate ID "${parsedId.data}"`,
      );
    }
    projectionIds.add(parsedId.data);
  }

  const selectedProbabilities = candidatePredictionProbabilities(
    candidateArtifactBytes(
      args.artifacts,
      args.candidate.selection.validation_predictions,
    ),
    "Study candidate selected validation predictions",
  );
  const replayedProbabilities = candidatePredictionProbabilities(
    candidateArtifactBytes(args.artifacts, args.candidate.replay.predictions),
    "Study candidate replay predictions",
  );
  if (
    selectedProbabilities.size
      !== args.candidate.selection.validation_prediction_count
    || replayedProbabilities.size !== args.candidate.replay.prediction_count
  ) {
    throw new Error(
      "Study candidate prediction counts do not match the candidate lock",
    );
  }
  let maxAbsoluteDifference = 0;
  for (const id of projectionIds) {
    const selected = selectedProbabilities.get(id);
    const replayed = replayedProbabilities.get(id);
    if (selected === undefined || replayed === undefined) {
      throw new Error(
        `Study candidate predictions are missing validation ID "${id}"`,
      );
    }
    maxAbsoluteDifference = Math.max(
      maxAbsoluteDifference,
      Math.abs(selected - replayed),
    );
  }
  if (
    selectedProbabilities.size !== projectionIds.size
    || replayedProbabilities.size !== projectionIds.size
  ) {
    throw new Error(
      "Study candidate predictions contain IDs outside the replay projection",
    );
  }
  if (
    maxAbsoluteDifference !== args.candidate.replay.max_absolute_difference
  ) {
    throw new Error(
      "Study candidate replay difference does not match the frozen predictions",
    );
  }
}

export async function verifyStudyCandidateArtifacts(args: {
  candidateDirectory: string;
  lock?: unknown;
  requirePublishedLock?: boolean;
}): Promise<StudyCandidateLock> {
  const candidateDirectory = resolve(args.candidateDirectory);
  const candidateDirectoryIdentity = await captureDirectoryIdentity(
    candidateDirectory,
  );
  const [
    selectionDirectoryIdentity,
    predictorDirectoryIdentity,
    replayDirectoryIdentity,
    modelDirectoryIdentity,
  ] = await Promise.all([
    captureDirectoryIdentity(join(candidateDirectory, "selection")),
    inspectCandidatePredictorDirectory(candidateDirectory),
    captureDirectoryIdentity(join(candidateDirectory, "replay")),
    captureDirectoryIdentity(join(candidateDirectory, "model")),
  ]);
  const requirePublishedLock = args.requirePublishedLock ?? true;
  const storedLock = requirePublishedLock
    ? await readStrictJson({
        path: join(candidateDirectory, "candidate.lock.json"),
        description: "Study candidate lock",
        maxBytes: MAX_TRIAL_REPORT_BYTES,
        schema: studyCandidateLockSchema,
        requireSingleLink: true,
      })
    : undefined;
  const candidate = studyCandidateLockSchema.parse(
    args.lock ?? storedLock?.value,
  );
  if (storedLock && args.lock !== undefined) {
    assertJsonEqual(
      storedLock.value,
      candidate,
      "Published Study candidate lock",
    );
  }

  const artifactBytes = new Map<string, Uint8Array>();
  for (const reference of studyCandidateArtifactReferences(candidate)) {
    const bytes = await readStableRegularFile({
      path: join(candidateDirectory, ...reference.path.split("/")),
      description: `Study candidate artifact ${reference.path}`,
      maxBytes: reference.size_bytes,
      requireSingleLink: true,
    });
    assertBytesMatch(
      bytes,
      reference,
      `Study candidate artifact ${reference.path}`,
    );
    artifactBytes.set(reference.path, bytes);
  }

  const model = await inspectStudyModel({
    modelDirectory: join(candidateDirectory, "model"),
    expectedRoot: modelDirectoryIdentity,
    requireNonempty: true,
    requireSingleLink: true,
  });
  assertJsonEqual(
    model.reference,
    candidate.artifacts.model,
    "Study candidate model",
  );
  const modelManifest = await readStableRegularFile({
    path: join(candidateDirectory, "model-manifest.json"),
    description: "Study candidate model manifest",
    maxBytes: MAX_PROVENANCE_MANIFEST_BYTES,
    requireSingleLink: true,
  });
  if (
    sha256(modelManifest) !== model.reference.sha256
    || new TextDecoder().decode(modelManifest) !== model.text
  ) {
    throw new Error(
      "Study candidate model manifest does not match its model tree",
    );
  }

  await verifyStudyCandidateSemanticBindings({
    candidateDirectory,
    candidate,
    artifacts: artifactBytes,
    model,
  });
  await Promise.all([
    verifyDirectoryIdentity({
      path: candidateDirectory,
      expected: candidateDirectoryIdentity,
      description: "Study candidate directory",
    }),
    verifyDirectoryIdentity({
      path: join(candidateDirectory, "selection"),
      expected: selectionDirectoryIdentity,
      description: "Study candidate selection directory",
    }),
    verifyDirectoryIdentity({
      path: join(candidateDirectory, "predictor"),
      expected: predictorDirectoryIdentity,
      description: "Study candidate predictor directory",
    }),
    verifyDirectoryIdentity({
      path: join(candidateDirectory, "replay"),
      expected: replayDirectoryIdentity,
      description: "Study candidate replay directory",
    }),
    verifyDirectoryIdentity({
      path: join(candidateDirectory, "model"),
      expected: modelDirectoryIdentity,
      description: "Study candidate model directory",
    }),
  ]);
  return candidate;
}

export async function promoteStudyTrialCandidate(args: {
  studyPath: string;
  trialPath: string;
  lockPath?: string;
  trialDirectory?: string;
}): Promise<{
  candidateDirectory: string;
  lockPath: string;
  lock: StudyCandidateLock;
}> {
  const loadedTrial = await loadStudyTrialSpec(args.trialPath);
  if (
    !("builtin" in loadedTrial.spec.runner)
    || loadedTrial.spec.runner.builtin !== "numeric_logistic_regression"
  ) {
    throw new Error(
      "Study candidate promotion currently supports only "
      + 'runner.builtin="numeric_logistic_regression"; command-backed trials '
      + "need a reusable predictor contract",
    );
  }

  const validated = await validateStudyBenchmark({
    studyPath: args.studyPath,
    lockPath: args.lockPath,
  });
  const loadedStudy = await loadStudySpec(args.studyPath);
  const studySpecSha256 = sha256(JSON.stringify(loadedStudy.spec));
  if (studySpecSha256 !== validated.lock.study_spec_sha256) {
    throw new Error("StudySpec changed after benchmark validation");
  }
  const benchmarkLockSha256 = sha256(canonicalJson(validated.lock));
  const trialSpecSha256 = sha256(canonicalJson(loadedTrial.spec));
  const prepared = await prepareTrialData({
    study: loadedStudy.spec,
    studyPath: loadedStudy.path,
    lock: validated.lock,
  });
  if (Buffer.byteLength(prepared.validationCsv) > MAX_CANDIDATE_ARTIFACT_BYTES) {
    throw new Error(
      `Projected Study validation data exceeds the `
      + `${MAX_CANDIDATE_ARTIFACT_BYTES}-byte candidate artifact limit`,
    );
  }

  const trialDirectory = resolve(
    args.trialDirectory
      ?? join(
        defaultStudyTrialOutputRoot(loadedStudy.path),
        loadedTrial.spec.id,
      ),
  );
  const trialDirectoryIdentity = await captureDirectoryIdentity(trialDirectory);
  const trialSpecArtifact = await readStrictJson({
    path: loadedTrial.path,
    description: "study trial spec",
    maxBytes: MAX_TRIAL_REPORT_BYTES,
    schema: studyTrialSpecSchema,
  });
  assertJsonEqual(
    trialSpecArtifact.value,
    loadedTrial.spec,
    "Current Study trial spec",
  );

  const reportPath = join(trialDirectory, "trial-report.json");
  const reportArtifact = await readStrictJson({
    path: reportPath,
    description: "Study trial report",
    maxBytes: MAX_TRIAL_REPORT_BYTES,
    schema: studyTrialReportSchema,
  });
  const report = reportArtifact.value;
  assertJsonEqual(report.trial, {
    id: loadedTrial.spec.id,
    name: loadedTrial.spec.name,
    spec_sha256: trialSpecSha256,
    parameters: loadedTrial.spec.parameters,
  }, "Study trial report trial identity");
  assertJsonEqual(report.study, {
    name: loadedStudy.spec.name,
    study_spec_sha256: validated.lock.study_spec_sha256,
    benchmark_lock_sha256: benchmarkLockSha256,
    task_type: loadedStudy.spec.task.type,
    primary_metric: loadedStudy.spec.task.primary_metric,
  }, "Study trial report benchmark identity");
  assertJsonEqual(report.data, {
    training: {
      source_sha256: prepared.source.training.sha256,
      projected_sha256: sha256(prepared.trainingCsv),
      row_count: prepared.source.training.rowCount,
    },
    validation: {
      source_sha256: prepared.source.validation.sha256,
      projected_sha256: sha256(prepared.validationCsv),
      row_count: prepared.source.validation.rowCount,
    },
    ...(expectedTrialTemporalEvidence(validated.lock)
      ? { temporal: expectedTrialTemporalEvidence(validated.lock) }
      : {}),
  }, "Study trial report dataset evidence");
  assertJsonEqual({
    command: report.execution.command,
    cwd: report.execution.cwd,
    timeout_ms: report.execution.timeout_ms,
    log: report.execution.log,
    artifact_directory: report.execution.artifact_directory,
  }, {
    command: ["builtin:numeric_logistic_regression"],
    cwd: "<trial-directory>",
    timeout_ms: loadedTrial.spec.runner.timeout_ms,
    log: "trial.log",
    artifact_directory: "model",
  }, "Study trial report execution contract");

  const [trainingBytes, validationBytes] = await Promise.all([
    readStableRegularFile({
      path: join(trialDirectory, "training.csv"),
      description: "projected Study training data",
      maxBytes: Buffer.byteLength(prepared.trainingCsv),
    }),
    readStableRegularFile({
      path: join(trialDirectory, "validation.csv"),
      description: "projected Study validation data",
      maxBytes: Buffer.byteLength(prepared.validationCsv),
    }),
  ]);
  if (
    sha256(trainingBytes) !== sha256(prepared.trainingCsv)
    || trainingBytes.byteLength !== Buffer.byteLength(prepared.trainingCsv)
    || sha256(validationBytes) !== sha256(prepared.validationCsv)
    || validationBytes.byteLength !== Buffer.byteLength(prepared.validationCsv)
  ) {
    throw new Error("Study trial projected data changed after validation");
  }

  const originalPredictions = await parsePredictions(
    join(trialDirectory, "predictions.json"),
    prepared.validationLabels,
  );
  const originalMetrics = computeBinaryClassificationMetrics(
    originalPredictions.rows,
  );
  assertJsonEqual(report.evaluation, {
    score_semantics: "positive_class_probability",
    primary_score: primaryScore(
      loadedStudy.spec.task.primary_metric,
      originalMetrics,
    ),
    metrics: originalMetrics,
    decision_threshold: 0.5,
    prediction_count: originalPredictions.rows.length,
    predictions_sha256: sha256(originalPredictions.bytes),
  }, "Study trial report validation evaluation");

  const implementationArtifact = await readStrictJson({
    path: join(trialDirectory, "implementation", "manifest.json"),
    description: "Study implementation manifest",
    maxBytes: MAX_PROVENANCE_MANIFEST_BYTES,
    schema: studyImplementationManifestSchema,
  });
  if (sha256(implementationArtifact.bytes) !== report.provenance.implementation.sha256) {
    throw new Error("Study implementation manifest does not match the trial report");
  }
  assertJsonEqual({
    evidence: implementationArtifact.value.evidence,
    files: implementationArtifact.value.files.map((file) => ({
      role: file.role,
      path: file.path,
      snapshot_path: file.snapshot_path,
    })),
  }, {
    evidence: "bundled_locked",
    files: [
      {
        role: "source",
        path: NUMERIC_LOGISTIC_REGRESSION_LOGICAL_PATH,
        snapshot_path: (
          `implementation/source/${NUMERIC_LOGISTIC_REGRESSION_LOGICAL_PATH}`
        ),
      },
      {
        role: "dependency_lock",
        path: `${NUMERIC_LOGISTIC_REGRESSION_LOGICAL_PATH}.lock`,
        snapshot_path: (
          `implementation/dependency-lock/`
          + `${NUMERIC_LOGISTIC_REGRESSION_LOGICAL_PATH}.lock`
        ),
      },
    ],
  }, "Bundled Study implementation manifest");
  await verifyStudyImplementationSnapshot({
    trialDirectory,
    manifest: implementationArtifact.value,
    reference: report.provenance.implementation,
  });
  const implementationSource = implementationArtifact.value.files[0]!;
  const implementationLock = implementationArtifact.value.files[1]!;
  const [bundledPredictorSource, bundledPredictorLock] = await Promise.all([
    readStableRegularFile({
      path: NUMERIC_LOGISTIC_REGRESSION_SCRIPT,
      description: "installed bundled Study predictor source",
      maxBytes: implementationSource.size_bytes,
    }),
    readStableRegularFile({
      path: NUMERIC_LOGISTIC_REGRESSION_LOCK,
      description: "installed bundled Study predictor dependency lock",
      maxBytes: implementationLock.size_bytes,
    }),
  ]);
  assertBytesMatch(
    bundledPredictorSource,
    implementationSource,
    "Installed bundled Study predictor source",
  );
  assertBytesMatch(
    bundledPredictorLock,
    implementationLock,
    "Installed bundled Study predictor dependency lock",
  );

  const sourceModelDirectory = join(trialDirectory, "model");
  const sourceModel = await inspectStudyModel({
    modelDirectory: sourceModelDirectory,
    requireNonempty: true,
  });
  assertJsonEqual(
    sourceModel.reference,
    report.provenance.model,
    "Study model provenance",
  );
  assertJsonEqual(
    sourceModel.manifest.files.map((file) => file.path),
    ["model.joblib", "runner-manifest.json"],
    "Bundled Study model artifact layout",
  );
  const storedModelManifest = await readStableRegularFile({
    path: join(trialDirectory, "model-manifest.json"),
    description: "Study model manifest",
    maxBytes: MAX_PROVENANCE_MANIFEST_BYTES,
  });
  if (
    sha256(storedModelManifest) !== sourceModel.reference.sha256
    || new TextDecoder().decode(storedModelManifest) !== sourceModel.text
  ) {
    throw new Error("Study model manifest does not match the fitted model tree");
  }

  const runnerManifestArtifact = await readStrictJson({
    path: join(sourceModelDirectory, "runner-manifest.json"),
    description: "bundled runner manifest",
    maxBytes: MAX_TRIAL_REPORT_BYTES,
    schema: bundledRunnerManifestSchema,
  });
  const runnerManifest = runnerManifestArtifact.value;
  assertJsonEqual(
    runnerManifest.parameters,
    loadedTrial.spec.parameters,
    "Bundled runner parameters",
  );
  assertJsonEqual(
    runnerManifest.input_columns,
    loadedStudy.spec.task.input_columns,
    "Bundled runner input columns",
  );
  if (
    runnerManifest.training.row_count !== prepared.source.training.rowCount
    || (
      runnerManifest.training.class_counts.negative
      + runnerManifest.training.class_counts.positive
    ) !== runnerManifest.training.row_count
  ) {
    throw new Error("Bundled runner training counts do not match the Study trial");
  }
  assertJsonEqual(
    Object.keys(runnerManifest.training.missing_counts).sort(),
    [...loadedStudy.spec.task.input_columns].sort(),
    "Bundled runner missing-count columns",
  );
  const sourceModelFile = sourceModel.manifest.files.find(
    (file) => file.path === "model.joblib",
  )!;
  const sourceRunnerManifestFile = sourceModel.manifest.files.find(
    (file) => file.path === "runner-manifest.json",
  )!;
  assertBytesMatch(
    runnerManifestArtifact.bytes,
    sourceRunnerManifestFile,
    "Bundled runner manifest",
  );
  assertJsonEqual(runnerManifest.model, {
    path: "model.joblib",
    sha256: sourceModelFile.sha256,
    size_bytes: sourceModelFile.size_bytes,
  }, "Bundled runner saved-model evidence");

  await verifyDirectoryIdentity({
    path: trialDirectory,
    expected: trialDirectoryIdentity,
    description: "Study trial directory",
  });

  const claimedCandidate = await claimStudyCandidateDirectory(loadedStudy.path);
  const candidateDirectory = claimedCandidate.candidateDirectory;
  const candidateDirectoryIdentity = await captureDirectoryIdentity(
    candidateDirectory,
  );
  const candidateLockPath = join(candidateDirectory, "candidate.lock.json");
  try {
    const selectionTrial = await writeCandidateArtifact({
      candidateDirectory,
      relativePath: "selection/trial-spec.json",
      bytes: trialSpecArtifact.bytes,
    });
    const selectionReport = await writeCandidateArtifact({
      candidateDirectory,
      relativePath: "selection/trial-report.json",
      bytes: reportArtifact.bytes,
    });
    const selectionPredictions = await writeCandidateArtifact({
      candidateDirectory,
      relativePath: "selection/validation-predictions.json",
      bytes: originalPredictions.bytes,
    });

    const candidateModelDirectory = join(candidateDirectory, "model");
    await mkdir(candidateModelDirectory, { mode: 0o700 });
    const candidateModelDirectoryIdentity = await captureDirectoryIdentity(
      candidateModelDirectory,
    );
    for (const file of sourceModel.manifest.files) {
      await copyVerifiedCandidateArtifact({
        candidateDirectory,
        relativePath: `model/${file.path}`,
        sourcePath: join(sourceModelDirectory, ...file.path.split("/")),
        expected: file,
        description: `Study model artifact ${file.path}`,
      });
    }
    const candidateModelManifest = await writeCandidateArtifact({
      candidateDirectory,
      relativePath: "model-manifest.json",
      bytes: sourceModel.text,
    });
    if (candidateModelManifest.sha256 !== sourceModel.reference.sha256) {
      throw new Error("Copied Study model manifest changed during promotion");
    }
    const promotedModel = await inspectStudyModel({
      modelDirectory: candidateModelDirectory,
      expectedRoot: candidateModelDirectoryIdentity,
      requireNonempty: true,
      requireSingleLink: true,
    });
    assertJsonEqual(
      promotedModel.reference,
      sourceModel.reference,
      "Copied Study model",
    );

    const predictorImplementationManifest = await writeCandidateArtifact({
      candidateDirectory,
      relativePath: "predictor/implementation-manifest.json",
      bytes: implementationArtifact.bytes,
    });
    const predictorScript = await copyVerifiedCandidateArtifact({
      candidateDirectory,
      relativePath: "predictor/numeric_logistic_regression.py",
      sourcePath: join(
        trialDirectory,
        ...implementationSource.snapshot_path.split("/"),
      ),
      expected: implementationSource,
      description: "bundled Study predictor source",
    });
    const predictorLock = await copyVerifiedCandidateArtifact({
      candidateDirectory,
      relativePath: "predictor/numeric_logistic_regression.py.lock",
      sourcePath: join(
        trialDirectory,
        ...implementationLock.snapshot_path.split("/"),
      ),
      expected: implementationLock,
      description: "bundled Study predictor dependency lock",
    });
    const replayValidation = await writeCandidateArtifact({
      candidateDirectory,
      relativePath: "replay/validation.csv",
      bytes: prepared.validationCsv,
    });
    const replayInputText = jsonText({
      protocol_version: STUDY_TRIAL_PROTOCOL_VERSION,
      task: {
        type: loadedStudy.spec.task.type,
        id_column: loadedStudy.spec.task.id_column,
        input_columns: loadedStudy.spec.task.input_columns,
        prediction: {
          field: "probability",
          meaning: "probability_of_positive_target",
        },
      },
      dataset: { prediction_csv: "replay/validation.csv" },
    });
    const replayInput = await writeCandidateArtifact({
      candidateDirectory,
      relativePath: "replay/predictor-input.json",
      bytes: replayInputText,
    });
    const replayPredictionPath = join(
      candidateDirectory,
      "replay",
      "predictions.json",
    );
    const replayLogPath = join(candidateDirectory, "replay", "predictor.log");
    await verifyStudyCandidateAncestors(claimedCandidate);
    const replayResult = await runLoggedProcess({
      command: "uv",
      commandArgs: [
        "run",
        "--locked",
        NUMERIC_LOGISTIC_REGRESSION_SCRIPT,
        "--input",
        "replay/predictor-input.json",
        "--output",
        "replay/predictions.json",
        "--model-dir",
        "model",
      ],
      cwd: candidateDirectory,
      env: minimalMachineLearningEnvironment(process.env),
      logPath: replayLogPath,
      timeoutMs: loadedTrial.spec.runner.timeout_ms,
      timeoutMessage: (
        `Study candidate replay timed out after `
        + `${loadedTrial.spec.runner.timeout_ms}ms`
      ),
      terminateProcessGroupOnExit: true,
      stage: "study-candidate-replay",
    });
    if (replayResult.exitCode !== 0) {
      throw new Error(
        `Study candidate replay exited with code ${replayResult.exitCode}; `
        + `see ${replayLogPath}`,
      );
    }
    const replayPredictions = await parsePredictions(
      replayPredictionPath,
      prepared.validationLabels,
      { requireSingleLink: true },
    );
    let maxAbsoluteDifference = 0;
    for (const [index, original] of originalPredictions.rows.entries()) {
      const replayed = replayPredictions.rows[index]!;
      if (original.id !== replayed.id) {
        throw new Error("Study candidate replay changed validation prediction IDs");
      }
      maxAbsoluteDifference = Math.max(
        maxAbsoluteDifference,
        Math.abs(original.probability - replayed.probability),
      );
    }
    if (maxAbsoluteDifference > REPLAY_PROBABILITY_TOLERANCE) {
      throw new Error(
        `Study candidate replay differs from the selected validation predictions `
        + `(maximum absolute difference ${maxAbsoluteDifference}, allowed `
        + `${REPLAY_PROBABILITY_TOLERANCE})`,
      );
    }
    const replayMetrics = computeBinaryClassificationMetrics(
      replayPredictions.rows,
    );
    assertJsonEqual(
      replayMetrics,
      originalMetrics,
      "Study candidate replay metrics",
    );
    const replayPredictionArtifact = candidateArtifactReference(
      "replay/predictions.json",
      replayPredictions.bytes,
    );
    const replayLogBytes = await readStableRegularFile({
      path: replayLogPath,
      description: "Study candidate replay log",
      maxBytes: MAX_CANDIDATE_LOG_BYTES,
      requireSingleLink: true,
    });
    const replayLogArtifact = candidateArtifactReference(
      "replay/predictor.log",
      replayLogBytes,
    );

    const candidateLock: StudyCandidateLock = {
      schema_version: 1,
      protocol_version: STUDY_TRIAL_PROTOCOL_VERSION,
      kind: "study_candidate",
      study: {
        name: loadedStudy.spec.name,
        study_spec_sha256: studySpecSha256,
        benchmark_lock_sha256: benchmarkLockSha256,
        task_type: loadedStudy.spec.task.type,
        primary_metric: loadedStudy.spec.task.primary_metric,
      },
      trial: {
        id: loadedTrial.spec.id,
        name: loadedTrial.spec.name,
        spec_sha256: trialSpecSha256,
        spec_artifact: selectionTrial,
        report_artifact: selectionReport,
      },
      selection: {
        validation_primary_score: report.evaluation.primary_score,
        validation_prediction_count: report.evaluation.prediction_count,
        validation_predictions: selectionPredictions,
      },
      artifacts: {
        model: promotedModel.reference,
        predictor: {
          implementation_manifest: predictorImplementationManifest,
          script: predictorScript,
          dependency_lock: predictorLock,
        },
      },
      training_runtime: runnerManifest.runtime,
      replay: {
        validation_projection: replayValidation,
        request: replayInput,
        predictions: replayPredictionArtifact,
        log: replayLogArtifact,
        prediction_count: replayPredictions.rows.length,
        probability_tolerance: REPLAY_PROBABILITY_TOLERANCE,
        max_absolute_difference: maxAbsoluteDifference,
      },
    };
    studyCandidateLockSchema.parse(candidateLock);
    await verifyStudyCandidateArtifacts({
      candidateDirectory,
      lock: candidateLock,
      requirePublishedLock: false,
    });

    const validatedAfter = await validateStudyBenchmark({
      studyPath: loadedStudy.path,
      lockPath: args.lockPath,
    });
    if (
      sha256(canonicalJson(validatedAfter.lock)) !== benchmarkLockSha256
      || validatedAfter.lock.study_spec_sha256 !== studySpecSha256
    ) {
      throw new Error("Study benchmark changed during candidate promotion");
    }
    const trialAfter = await loadStudyTrialSpec(loadedTrial.path);
    if (sha256(canonicalJson(trialAfter.spec)) !== trialSpecSha256) {
      throw new Error("Study trial spec changed during candidate promotion");
    }
    await Promise.all([
      verifyDirectoryIdentity({
        path: trialDirectory,
        expected: trialDirectoryIdentity,
        description: "Study trial directory",
      }),
      verifyDirectoryIdentity({
        path: candidateDirectory,
        expected: candidateDirectoryIdentity,
        description: "Study candidate directory",
      }),
    ]);

    await writeJsonAtomic(candidateLockPath, candidateLock);
    await verifyStudyCandidateArtifacts({
      candidateDirectory,
      lock: candidateLock,
    });
    await verifyStudyCandidateAncestors(claimedCandidate);
    try {
      await lstat(claimedCandidate.finalDirectory);
      throw new Error(
        `A Study candidate selection already exists at `
        + `${claimedCandidate.finalDirectory}; candidate selection is write-once`,
      );
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    try {
      await rename(candidateDirectory, claimedCandidate.finalDirectory);
    } catch (error) {
      if (
        (error as NodeJS.ErrnoException).code === "EEXIST"
        || (error as NodeJS.ErrnoException).code === "ENOTEMPTY"
      ) {
        throw new Error(
          `A Study candidate selection already exists at `
          + `${claimedCandidate.finalDirectory}; candidate selection is write-once`,
          { cause: error },
        );
      }
      throw error;
    }
    return {
      candidateDirectory: claimedCandidate.finalDirectory,
      lockPath: join(claimedCandidate.finalDirectory, "candidate.lock.json"),
      lock: candidateLock,
    };
  } catch (error) {
    try {
      await removeOwnedCandidateStaging(
        claimedCandidate,
        candidateDirectoryIdentity,
      );
    } catch (cleanupError) {
      throw new AggregateError(
        [error, cleanupError],
        "Study candidate promotion failed and its staging directory could not be removed",
      );
    }
    throw error;
  }
}
