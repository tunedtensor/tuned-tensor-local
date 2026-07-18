import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import {
  link,
  lstat,
  mkdir,
  mkdtemp,
  rm,
  writeFile,
} from "node:fs/promises";
import { arch, platform, tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { minimalMachineLearningEnvironment } from "./huggingface-cache.js";
import { runLoggedProcess } from "./process-runner.js";
import {
  BINARY_DECISION_THRESHOLD,
  computeBinaryClassificationMetrics,
} from "./study-metrics.js";
import {
  defaultStudyCandidateDirectory,
  verifyStudyCandidateArtifacts,
  type StudyCandidateLock,
} from "./study-candidates.js";
import {
  MAX_TRIAL_REPORT_BYTES,
  NUMERIC_LOGISTIC_REGRESSION_RUNNER_VERSION,
  STUDY_TRIAL_PROTOCOL_VERSION,
  bundledPredictionRuntimeEvidenceSchema,
  bundledRunnerManifestSchema,
  canonicalJson,
  exactStringSchema,
  nonnegativeIntegerSchema,
  parsePredictions,
  positiveIntegerSchema,
  primaryScore,
  probabilityMetricSchema,
  provenanceFilePathSchema,
  readStableRegularFile,
  readStudyTrialOutput,
  schemaError,
  sha256,
  sha256Schema,
  studyTrialOutputSchema,
  studyTrialSpecSchema,
  type StudyTrialOutput,
  type StudyTrialSpec,
} from "./study-trial-core.js";
import {
  captureDirectoryIdentity,
  verifyDirectoryIdentity,
  type DirectoryIdentity,
} from "./study-provenance.js";
import {
  defaultStudyLockPath,
  inspectStudySplitBytes,
  loadStudySpec,
  studyBenchmarkLockSchema,
  type StudyBenchmarkLock,
  type StudySpec,
} from "./studies.js";
import { defaultLocalHome } from "./store.js";

export const STUDY_TEST_PROTOCOL_VERSION = 1;
const MAX_RUNTIME_EVIDENCE_BYTES = 64 * 1024;
const MAX_PREDICTOR_LOG_BYTES = 16 * 1024 * 1024;
const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));

const lockedTestEvidenceSchema = z.object({
  format: z.literal("csv"),
  sha256: sha256Schema,
  size_bytes: nonnegativeIntegerSchema,
  row_count: positiveIntegerSchema,
}).strict();

const claimedTaskSchema = z.object({
  type: z.literal("binary_classification"),
  id_column: exactStringSchema,
  target_column: exactStringSchema,
  labels: z.object({
    negative: exactStringSchema,
    positive: exactStringSchema,
  }).strict(),
}).strict();

export const studyTestClaimIdentitySchema = z.object({
  schema_version: z.literal(1),
  protocol_version: z.literal(STUDY_TEST_PROTOCOL_VERSION),
  kind: z.literal("study_test_claim"),
  task: claimedTaskSchema,
  test: lockedTestEvidenceSchema,
}).strict();

export type StudyTestClaimIdentity = z.infer<
  typeof studyTestClaimIdentitySchema
>;

const artifactReferenceSchema = z.object({
  path: provenanceFilePathSchema,
  sha256: sha256Schema,
  size_bytes: nonnegativeIntegerSchema,
}).strict();

const artifactAt = <Path extends string>(path: Path) => (
  artifactReferenceSchema.extend({ path: z.literal(path) }).strict()
);

const receiptClaimSchema = z.object({
  id: sha256Schema,
  task: claimedTaskSchema,
  test: lockedTestEvidenceSchema,
}).strict();

const receiptStudySchema = z.object({
  name: exactStringSchema,
  study_spec_sha256: sha256Schema,
  benchmark_lock_sha256: sha256Schema,
  primary_metric: z.enum(["average_precision", "roc_auc", "f1"]),
}).strict();

const receiptCandidateSchema = z.object({
  lock_sha256: sha256Schema,
  trial: z.object({
    id: studyTrialSpecSchema.shape.id,
    spec_sha256: sha256Schema,
  }).strict(),
  predictor: z.object({
    implementation_manifest: artifactAt(
      "predictor/implementation-manifest.json",
    ),
    script: artifactAt("predictor/numeric_logistic_regression.py"),
    dependency_lock: artifactAt(
      "predictor/numeric_logistic_regression.py.lock",
    ),
  }).strict(),
  model: z.object({
    outer: z.object({
      manifest: z.literal("model-manifest.json"),
      sha256: sha256Schema,
      file_count: positiveIntegerSchema,
      size_bytes: positiveIntegerSchema,
    }).strict(),
    runner: z.object({
      name: z.literal("numeric_logistic_regression"),
      version: z.literal(NUMERIC_LOGISTIC_REGRESSION_RUNNER_VERSION),
    }).strict(),
    inner: z.object({
      path: z.literal("model.joblib"),
      sha256: sha256Schema,
      size_bytes: positiveIntegerSchema,
    }).strict(),
  }).strict(),
}).strict();

const receiptProducerSchema = z.object({
  tt_local_version: exactStringSchema,
  node_version: exactStringSchema,
  platform: exactStringSchema,
  arch: exactStringSchema,
}).strict();

const receiptBaseShape = {
  schema_version: z.literal(1),
  protocol_version: z.literal(STUDY_TEST_PROTOCOL_VERSION),
  kind: z.literal("study_test_receipt"),
  claim: receiptClaimSchema,
  study: receiptStudySchema,
  candidate: receiptCandidateSchema,
  producer: receiptProducerSchema,
};

export const studyTestSuccessReceiptSchema = z.object({
  ...receiptBaseShape,
  status: z.literal("succeeded"),
  artifacts: z.object({
    request: artifactAt("predictor-input.json"),
    projection: artifactAt("test.csv"),
    predictions: artifactAt("predictions.json"),
  }).strict(),
  execution: z.object({
    command: z.tuple([
      z.literal("builtin:numeric_logistic_regression:predict"),
    ]),
    cwd: z.literal("<claim-directory>"),
    timeout_ms: positiveIntegerSchema,
    duration_ms: nonnegativeIntegerSchema,
    runtime: z.object({
      artifact: artifactAt("prediction-runtime.json"),
      evidence: bundledPredictionRuntimeEvidenceSchema,
    }).strict(),
    log: artifactAt("predictor.log"),
  }).strict(),
  evaluation: z.object({
    score_semantics: z.literal("positive_class_probability"),
    primary_metric: z.enum(["average_precision", "roc_auc", "f1"]),
    primary_score: probabilityMetricSchema,
    metrics: z.object({
      average_precision: probabilityMetricSchema,
      roc_auc: probabilityMetricSchema,
      f1_at_0_5: probabilityMetricSchema,
    }).strict(),
    decision_threshold: z.literal(BINARY_DECISION_THRESHOLD),
    prediction_count: positiveIntegerSchema,
  }).strict(),
}).strict().superRefine((receipt, context) => {
  const identity: StudyTestClaimIdentity = {
    schema_version: 1,
    protocol_version: STUDY_TEST_PROTOCOL_VERSION,
    kind: "study_test_claim",
    task: receipt.claim.task,
    test: receipt.claim.test,
  };
  if (receipt.claim.id !== studyTestClaimId(identity)) {
    context.addIssue({
      code: "custom",
      path: ["claim", "id"],
      message: "must match the canonical held-out test identity",
    });
  }
  if (receipt.evaluation.primary_metric !== receipt.study.primary_metric) {
    context.addIssue({
      code: "custom",
      path: ["evaluation", "primary_metric"],
      message: "must match study.primary_metric",
    });
  }
  if (
    receipt.evaluation.primary_score
    !== primaryScore(
      receipt.evaluation.primary_metric,
      receipt.evaluation.metrics,
    )
  ) {
    context.addIssue({
      code: "custom",
      path: ["evaluation", "primary_score"],
      message: "must match the declared primary metric",
    });
  }
  if (receipt.evaluation.prediction_count !== receipt.claim.test.row_count) {
    context.addIssue({
      code: "custom",
      path: ["evaluation", "prediction_count"],
      message: "must match the claimed test row count",
    });
  }
});

const studyTestFailurePhaseSchema = z.enum([
  "claim",
  "test",
  "projection",
  "prediction",
  "verification",
  "scoring",
  "publication",
]);

export const studyTestFailureReceiptSchema = z.object({
  ...receiptBaseShape,
  status: z.literal("failed"),
  error: z.object({
    phase: studyTestFailurePhaseSchema,
    message: z.string().min(1).max(4_096),
  }).strict(),
}).strict().superRefine((receipt, context) => {
  const identity: StudyTestClaimIdentity = {
    schema_version: 1,
    protocol_version: STUDY_TEST_PROTOCOL_VERSION,
    kind: "study_test_claim",
    task: receipt.claim.task,
    test: receipt.claim.test,
  };
  if (receipt.claim.id !== studyTestClaimId(identity)) {
    context.addIssue({
      code: "custom",
      path: ["claim", "id"],
      message: "must match the canonical held-out test identity",
    });
  }
});

export const studyTestReceiptSchema = z.union([
  studyTestSuccessReceiptSchema,
  studyTestFailureReceiptSchema,
]);

export type StudyTestSuccessReceipt = z.infer<
  typeof studyTestSuccessReceiptSchema
>;
export type StudyTestFailureReceipt = z.infer<
  typeof studyTestFailureReceiptSchema
>;
export type StudyTestReceipt = z.infer<typeof studyTestReceiptSchema>;

interface ArtifactReference<Path extends string = string> {
  path: Path;
  sha256: string;
  size_bytes: number;
}

interface FileState {
  dev: number;
  ino: number;
  size: number;
  mtimeMs: number;
  ctimeMs: number;
  nlink: number;
}

interface StudyTestPreflight {
  studyPath: string;
  study: StudySpec;
  studySpecSha256: string;
  lockPath: string;
  lock: StudyBenchmarkLock;
  benchmarkLockSha256: string;
  candidateDirectory: string;
  candidate: StudyCandidateLock;
  candidateLockSha256: string;
  trialSpec: StudyTrialSpec;
  runnerManifest: z.infer<typeof bundledRunnerManifestSchema>;
  timeoutMs: number;
  predictorCommandPath: string;
  modelCommandPath: string;
  claimIdentity: StudyTestClaimIdentity;
  claimId: string;
  producer: StudyTestSuccessReceipt["producer"];
}

interface StudyTestLedger {
  home: string;
  homeIdentity: DirectoryIdentity;
  root: string;
  rootIdentity: DirectoryIdentity;
}

interface ClaimedStudyTest {
  directory: string;
  directoryIdentity: DirectoryIdentity;
  ledger: StudyTestLedger;
}

interface PreparedClaimedTestData {
  labels: Array<{ id: string; positive: boolean }>;
  projectionCsv: string;
  sourcePath: string;
  sourceState: FileState;
}

interface StudyTestInternalHooks {
  afterClaim?: (context: { claimDirectory: string }) => void | Promise<void>;
  afterPrediction?: (
    context: { claimDirectory: string },
  ) => void | Promise<void>;
  beforeCommit?: (
    context: { claimDirectory: string },
  ) => void | Promise<void>;
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
    throw new Error(`${description} does not match its frozen evidence`);
  }
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
    throw new Error(`Cannot read valid JSON for ${description}`, {
      cause: error,
    });
  }
  const parsed = schema.safeParse(value);
  if (!parsed.success) {
    throw new Error(
      `Invalid ${description}: ${schemaError(parsed.error)}`,
    );
  }
  return parsed.data;
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
    value: parseStrictJsonBytes(bytes, args.description, args.schema),
  };
}

function artifactReference<Path extends string>(
  path: Path,
  bytes: string | Uint8Array,
): ArtifactReference<Path> {
  return {
    path,
    sha256: sha256(bytes),
    size_bytes: typeof bytes === "string"
      ? Buffer.byteLength(bytes)
      : bytes.byteLength,
  };
}

function assertArtifactBytes(
  bytes: Uint8Array,
  expected: Pick<ArtifactReference, "sha256" | "size_bytes">,
  description: string,
): void {
  if (
    bytes.byteLength !== expected.size_bytes
    || sha256(bytes) !== expected.sha256
  ) {
    throw new Error(`${description} does not match its recorded evidence`);
  }
}

async function writeFileOnceAtomic(
  path: string,
  value: string | Uint8Array,
): Promise<Uint8Array> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  const bytes = typeof value === "string" ? Buffer.from(value) : value;
  const temporary = join(
    dirname(path),
    `.${basename(path)}.${process.pid}.${randomUUID()}.tmp`,
  );
  try {
    await writeFile(temporary, bytes, {
      flag: "wx",
      mode: 0o400,
    });
    const verified = await readStableRegularFile({
      path: temporary,
      description: "Study test publication temporary file",
      maxBytes: bytes.byteLength,
      requireSingleLink: true,
    });
    if (
      verified.byteLength !== bytes.byteLength
      || sha256(verified) !== sha256(bytes)
    ) {
      throw new Error("Study test publication temporary file changed");
    }
    await link(temporary, path);
    return bytes;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EEXIST") {
      throw new Error(`Study test artifact already exists at ${path}`, {
        cause: error,
      });
    }
    throw error;
  } finally {
    await rm(temporary, { force: true }).catch(() => undefined);
  }
}

async function writeClaimArtifact<Path extends string>(args: {
  claimDirectory: string;
  relativePath: Path;
  value: string | Uint8Array;
}): Promise<ArtifactReference<Path>> {
  const bytes = await writeFileOnceAtomic(
    join(args.claimDirectory, ...args.relativePath.split("/")),
    args.value,
  );
  return artifactReference(args.relativePath, bytes);
}

function packageVersion(): string {
  try {
    const packageJson = JSON.parse(
      readFileSync(join(packageRoot, "package.json"), "utf8"),
    ) as { version?: unknown };
    return typeof packageJson.version === "string"
      ? packageJson.version
      : "unknown";
  } catch {
    return "unknown";
  }
}

function producerEvidence(): StudyTestSuccessReceipt["producer"] {
  return {
    tt_local_version: packageVersion(),
    node_version: process.version,
    platform: platform(),
    arch: arch(),
  };
}

function csvField(value: string): string {
  return /[",\r\n]/.test(value)
    ? `"${value.replaceAll("\"", "\"\"")}"`
    : value;
}

function serializeCsv(
  columns: readonly string[],
  rows: readonly (readonly string[])[],
): string {
  return [
    columns.map(csvField).join(","),
    ...rows.map((row) => row.map(csvField).join(",")),
  ].join("\n") + "\n";
}

function sameFileState(left: FileState, right: FileState): boolean {
  return (
    left.dev === right.dev
    && left.ino === right.ino
    && left.size === right.size
    && left.mtimeMs === right.mtimeMs
    && left.ctimeMs === right.ctimeMs
    && left.nlink === right.nlink
  );
}

async function captureRegularFileState(
  path: string,
  description: string,
): Promise<FileState> {
  const metadata = await lstat(path);
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    throw new Error(`${description} must be a regular, non-symbolic file`);
  }
  return {
    dev: metadata.dev,
    ino: metadata.ino,
    size: metadata.size,
    mtimeMs: metadata.mtimeMs,
    ctimeMs: metadata.ctimeMs,
    nlink: metadata.nlink,
  };
}

async function verifyRegularFileState(args: {
  path: string;
  expected: FileState;
  description: string;
}): Promise<void> {
  const current = await captureRegularFileState(args.path, args.description);
  if (!sameFileState(current, args.expected)) {
    throw new Error(`${args.description} changed during held-out evaluation`);
  }
}

export function studyTestClaimIdentity(
  lock: StudyBenchmarkLock,
): StudyTestClaimIdentity {
  return studyTestClaimIdentitySchema.parse({
    schema_version: 1,
    protocol_version: STUDY_TEST_PROTOCOL_VERSION,
    kind: "study_test_claim",
    task: {
      type: lock.study.task.type,
      id_column: lock.study.task.id_column,
      target_column: lock.study.task.target_column,
      labels: lock.study.task.labels,
    },
    test: {
      format: lock.dataset.format,
      sha256: lock.dataset.splits.test.sha256,
      size_bytes: lock.dataset.splits.test.size_bytes,
      row_count: lock.dataset.splits.test.row_count,
    },
  });
}

export function studyTestClaimId(
  identity: StudyTestClaimIdentity,
): string {
  return sha256(canonicalJson(studyTestClaimIdentitySchema.parse(identity)));
}

export function defaultStudyTestClaimRoot(): string {
  return join(defaultLocalHome(), "study-test-claims", "v1");
}

export function defaultStudyTestClaimDirectory(
  identity: StudyTestClaimIdentity,
): string {
  return join(defaultStudyTestClaimRoot(), studyTestClaimId(identity));
}

function alreadyConsumedStudyTestError(
  claimId: string,
  directory: string,
  cause?: unknown,
): Error {
  return new Error(
    `Held-out test identity ${claimId} has already been consumed at `
    + `${directory}; successful, failed, incomplete, and crashed claims are `
    + "all one-shot",
    cause === undefined ? undefined : { cause },
  );
}

async function rejectExistingStudyTestClaim(
  identity: StudyTestClaimIdentity,
): Promise<void> {
  const claimId = studyTestClaimId(identity);
  const directory = defaultStudyTestClaimDirectory(identity);
  try {
    await lstat(directory);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    throw error;
  }
  throw alreadyConsumedStudyTestError(claimId, directory);
}

async function ensureRealDirectory(
  path: string,
  description: string,
): Promise<DirectoryIdentity> {
  try {
    await mkdir(path, { recursive: true, mode: 0o700 });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
  }
  try {
    return await captureDirectoryIdentity(path);
  } catch (error) {
    throw new Error(`${description} must be a real directory: ${path}`, {
      cause: error,
    });
  }
}

async function prepareStudyTestLedger(): Promise<StudyTestLedger> {
  const home = defaultLocalHome();
  const homeIdentity = await ensureRealDirectory(
    home,
    "TT Local home directory",
  );
  const claims = join(home, "study-test-claims");
  await ensureRealDirectory(claims, "Study test claim parent");
  const root = defaultStudyTestClaimRoot();
  const rootIdentity = await ensureRealDirectory(
    root,
    "Study test claim root",
  );
  return { home, homeIdentity, root, rootIdentity };
}

async function verifyStudyTestLedger(ledger: StudyTestLedger): Promise<void> {
  await Promise.all([
    verifyDirectoryIdentity({
      path: ledger.home,
      expected: ledger.homeIdentity,
      description: "TT Local home directory",
    }),
    verifyDirectoryIdentity({
      path: ledger.root,
      expected: ledger.rootIdentity,
      description: "Study test claim root",
    }),
  ]);
}

async function claimStudyTest(
  preflight: StudyTestPreflight,
  ledger: StudyTestLedger,
): Promise<ClaimedStudyTest> {
  await verifyStudyTestLedger(ledger);
  const directory = join(ledger.root, preflight.claimId);
  try {
    await mkdir(directory, { mode: 0o700 });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EEXIST") {
      throw alreadyConsumedStudyTestError(
        preflight.claimId,
        directory,
        error,
      );
    }
    throw error;
  }
  await verifyStudyTestLedger(ledger);
  return {
    directory,
    directoryIdentity: await captureDirectoryIdentity(directory),
    ledger,
  };
}

async function readCandidateArtifact(args: {
  candidateDirectory: string;
  reference: ArtifactReference;
  description: string;
}): Promise<Uint8Array> {
  const bytes = await readStableRegularFile({
    path: join(
      args.candidateDirectory,
      ...args.reference.path.split("/"),
    ),
    description: args.description,
    maxBytes: args.reference.size_bytes,
    requireSingleLink: true,
  });
  assertArtifactBytes(bytes, args.reference, args.description);
  return bytes;
}

function assertReplayParity(
  expected: StudyTrialOutput,
  actual: StudyTrialOutput,
  tolerance: number,
): void {
  const expectedProbabilities = new Map(
    expected.predictions.map((prediction) => [
      prediction.id,
      prediction.probability,
    ]),
  );
  if (
    expectedProbabilities.size !== expected.predictions.length
    || actual.predictions.length !== expected.predictions.length
  ) {
    throw new Error("Study candidate preflight replay prediction count changed");
  }
  const seen = new Set<string>();
  for (const prediction of actual.predictions) {
    const expectedProbability = expectedProbabilities.get(prediction.id);
    if (
      expectedProbability === undefined
      || seen.has(prediction.id)
      || Math.abs(expectedProbability - prediction.probability) > tolerance
    ) {
      throw new Error(
        "Study candidate preflight replay did not reproduce frozen predictions",
      );
    }
    seen.add(prediction.id);
  }
}

async function preflightCandidateReplay(args: {
  candidateDirectory: string;
  candidate: StudyCandidateLock;
  timeoutMs: number;
}): Promise<void> {
  const temporary = await mkdtemp(
    join(tmpdir(), "tt-local-study-test-preflight-"),
  );
  try {
    const expectedBytes = await readCandidateArtifact({
      candidateDirectory: args.candidateDirectory,
      reference: args.candidate.replay.predictions,
      description: "Study candidate frozen replay predictions",
    });
    const expected = parseStrictJsonBytes(
      expectedBytes,
      "Study candidate frozen replay predictions",
      studyTrialOutputSchema,
    );
    const outputPath = join(temporary, "predictions.json");
    const runtimePath = join(temporary, "runtime.json");
    const logPath = join(temporary, "predictor.log");
    const result = await runLoggedProcess({
      command: "uv",
      commandArgs: [
        "run",
        "--locked",
        args.candidate.artifacts.predictor.script.path,
        "--input",
        args.candidate.replay.request.path,
        "--output",
        outputPath,
        "--model-dir",
        "model",
        "--runtime-output",
        runtimePath,
      ],
      cwd: args.candidateDirectory,
      env: minimalMachineLearningEnvironment(process.env),
      logPath,
      timeoutMs: args.timeoutMs,
      timeoutMessage: (
        `Study candidate preflight replay timed out after ${args.timeoutMs}ms`
      ),
      terminateProcessGroupOnExit: true,
      exclusiveLog: true,
      stage: "study-test-preflight",
    });
    if (result.exitCode !== 0) {
      throw new Error(
        `Study candidate preflight replay exited with code `
        + `${result.exitCode}; held-out test remains unclaimed`,
      );
    }
    const actual = await readStudyTrialOutput(outputPath, {
      requireSingleLink: true,
    });
    assertReplayParity(
      expected,
      actual.output,
      args.candidate.replay.probability_tolerance,
    );
    await readStrictJson({
      path: runtimePath,
      description: "Study candidate preflight runtime evidence",
      maxBytes: MAX_RUNTIME_EVIDENCE_BYTES,
      schema: bundledPredictionRuntimeEvidenceSchema,
      requireSingleLink: true,
    });
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

async function loadStudyTestPreflight(args: {
  studyPath: string;
  lockPath?: string;
}): Promise<StudyTestPreflight> {
  const loadedStudy = await loadStudySpec(args.studyPath);
  const lockPath = resolve(
    args.lockPath ?? defaultStudyLockPath(loadedStudy.path),
  );
  const lockArtifact = await readStrictJson({
    path: lockPath,
    description: "Study benchmark lock",
    maxBytes: MAX_TRIAL_REPORT_BYTES,
    schema: studyBenchmarkLockSchema,
  });
  const lock = lockArtifact.value;
  const studySpecSha256 = sha256(JSON.stringify(loadedStudy.spec));
  if (studySpecSha256 !== lock.study_spec_sha256) {
    throw new Error(
      "StudySpec does not match the stored benchmark lock; "
      + "held-out test remains unclaimed",
    );
  }
  assertJsonEqual(lock.study, {
    name: loadedStudy.spec.name,
    task: loadedStudy.spec.task,
  }, "Study benchmark lock identity");

  const claimIdentity = studyTestClaimIdentity(lock);
  await rejectExistingStudyTestClaim(claimIdentity);
  const benchmarkLockSha256 = sha256(canonicalJson(lock));
  const candidateDirectory = defaultStudyCandidateDirectory(loadedStudy.path);
  const candidate = await verifyStudyCandidateArtifacts({
    candidateDirectory,
  });
  assertJsonEqual(candidate.study, {
    name: loadedStudy.spec.name,
    study_spec_sha256: studySpecSha256,
    benchmark_lock_sha256: benchmarkLockSha256,
    task_type: loadedStudy.spec.task.type,
    primary_metric: loadedStudy.spec.task.primary_metric,
  }, "Study candidate benchmark binding");

  const trialSpecBytes = await readCandidateArtifact({
    candidateDirectory,
    reference: candidate.trial.spec_artifact,
    description: "Study candidate trial spec",
  });
  const trialSpec = parseStrictJsonBytes(
    trialSpecBytes,
    "Study candidate trial spec",
    studyTrialSpecSchema,
  );
  if (!("builtin" in trialSpec.runner)) {
    throw new Error("Held-out testing requires the bundled saved-model runner");
  }
  const runnerManifest = (
    await readStrictJson({
      path: join(candidateDirectory, "model", "runner-manifest.json"),
      description: "Study candidate runner manifest",
      maxBytes: MAX_TRIAL_REPORT_BYTES,
      schema: bundledRunnerManifestSchema,
      requireSingleLink: true,
    })
  ).value;
  const candidateLockSha256 = sha256(canonicalJson(candidate));
  await preflightCandidateReplay({
    candidateDirectory,
    candidate,
    timeoutMs: trialSpec.runner.timeout_ms,
  });
  const predictorCommandPath = join(
    candidateDirectory,
    ...candidate.artifacts.predictor.script.path.split("/"),
  );
  const modelCommandPath = join(candidateDirectory, "model");
  return {
    studyPath: loadedStudy.path,
    study: loadedStudy.spec,
    studySpecSha256,
    lockPath,
    lock,
    benchmarkLockSha256,
    candidateDirectory,
    candidate,
    candidateLockSha256,
    trialSpec,
    runnerManifest,
    timeoutMs: trialSpec.runner.timeout_ms,
    predictorCommandPath,
    modelCommandPath,
    claimIdentity,
    claimId: studyTestClaimId(claimIdentity),
    producer: producerEvidence(),
  };
}

async function prepareClaimedTestData(
  preflight: StudyTestPreflight,
): Promise<PreparedClaimedTestData> {
  const sourcePath = resolve(
    dirname(preflight.studyPath),
    preflight.study.dataset.splits.test,
  );
  const expected = preflight.lock.dataset.splits.test;
  const sourceStateBefore = await captureRegularFileState(
    sourcePath,
    "Held-out test dataset",
  );
  const bytes = await readStableRegularFile({
    path: sourcePath,
    description: "held-out test dataset",
    maxBytes: expected.size_bytes,
  });
  const inspected = inspectStudySplitBytes({
    split: "test",
    source: preflight.study.dataset.splits.test,
    bytes,
    task: preflight.study.task,
    temporal: preflight.study.dataset.temporal,
  });
  assertJsonEqual(
    inspected.summary,
    expected,
    "Held-out test dataset",
  );
  assertJsonEqual(
    inspected.columns,
    preflight.lock.dataset.columns,
    "Held-out test columns",
  );
  assertJsonEqual(
    inspected.temporal?.summary,
    preflight.lock.dataset.temporal?.splits.test,
    "Held-out test temporal evidence",
  );
  const sourceState = await captureRegularFileState(
    sourcePath,
    "Held-out test dataset",
  );
  if (!sameFileState(sourceStateBefore, sourceState)) {
    throw new Error(
      "Held-out test dataset changed while it was being prepared",
    );
  }
  const projectedColumns = [
    preflight.study.task.id_column,
    ...preflight.study.task.input_columns,
  ];
  const projectedIndexes = projectedColumns.map((column) => (
    inspected.columns.indexOf(column)
  ));
  const targetIndex = inspected.columns.indexOf(
    preflight.study.task.target_column,
  );
  const projectedRows = inspected.rows.map((row) => (
    projectedIndexes.map((index) => row[index]!)
  ));
  return {
    labels: inspected.rows.map((row) => ({
      id: row[projectedIndexes[0]!]!,
      positive: (
        row[targetIndex] === preflight.study.task.labels.positive
      ),
    })),
    projectionCsv: serializeCsv(projectedColumns, projectedRows),
    sourcePath,
    sourceState,
  };
}

function predictorRequest(study: StudySpec): unknown {
  return {
    protocol_version: STUDY_TRIAL_PROTOCOL_VERSION,
    task: {
      type: study.task.type,
      id_column: study.task.id_column,
      input_columns: study.task.input_columns,
      prediction: {
        field: "probability",
        meaning: "probability_of_positive_target",
      },
    },
    dataset: { prediction_csv: "test.csv" },
  };
}

async function verifyClaimArtifact(args: {
  claimed: ClaimedStudyTest;
  reference: ArtifactReference;
  description: string;
}): Promise<Uint8Array> {
  const bytes = await readStableRegularFile({
    path: join(args.claimed.directory, ...args.reference.path.split("/")),
    description: args.description,
    maxBytes: args.reference.size_bytes,
    requireSingleLink: true,
  });
  assertArtifactBytes(bytes, args.reference, args.description);
  return bytes;
}

function receiptCommon(preflight: StudyTestPreflight) {
  return {
    schema_version: 1 as const,
    protocol_version: STUDY_TEST_PROTOCOL_VERSION,
    kind: "study_test_receipt" as const,
    claim: {
      id: preflight.claimId,
      task: preflight.claimIdentity.task,
      test: preflight.claimIdentity.test,
    },
    study: {
      name: preflight.study.name,
      study_spec_sha256: preflight.studySpecSha256,
      benchmark_lock_sha256: preflight.benchmarkLockSha256,
      primary_metric: preflight.study.task.primary_metric,
    },
    candidate: {
      lock_sha256: preflight.candidateLockSha256,
      trial: {
        id: preflight.candidate.trial.id,
        spec_sha256: preflight.candidate.trial.spec_sha256,
      },
      predictor: preflight.candidate.artifacts.predictor,
      model: {
        outer: preflight.candidate.artifacts.model,
        runner: preflight.runnerManifest.runner,
        inner: preflight.runnerManifest.model,
      },
    },
    producer: preflight.producer,
  };
}

async function verifyFrozenInputs(args: {
  preflight: StudyTestPreflight;
  claimed: ClaimedStudyTest;
  testData: PreparedClaimedTestData;
  claimArtifact: ArtifactReference<"claim.json">;
  artifacts: ArtifactReference[];
}): Promise<void> {
  await Promise.all([
    verifyStudyTestLedger(args.claimed.ledger),
    verifyDirectoryIdentity({
      path: args.claimed.directory,
      expected: args.claimed.directoryIdentity,
      description: "Study test claim directory",
    }),
    verifyRegularFileState({
      path: args.testData.sourcePath,
      expected: args.testData.sourceState,
      description: "Held-out test dataset",
    }),
  ]);
  const loadedStudy = await loadStudySpec(args.preflight.studyPath);
  if (
    sha256(JSON.stringify(loadedStudy.spec))
      !== args.preflight.studySpecSha256
    || canonicalJson(loadedStudy.spec) !== canonicalJson(args.preflight.study)
  ) {
    throw new Error("StudySpec changed during held-out evaluation");
  }
  const currentLock = await readStrictJson({
    path: args.preflight.lockPath,
    description: "Study benchmark lock",
    maxBytes: MAX_TRIAL_REPORT_BYTES,
    schema: studyBenchmarkLockSchema,
  });
  if (
    sha256(canonicalJson(currentLock.value))
      !== args.preflight.benchmarkLockSha256
  ) {
    throw new Error("Study benchmark lock changed during held-out evaluation");
  }
  const currentCandidate = await verifyStudyCandidateArtifacts({
    candidateDirectory: args.preflight.candidateDirectory,
  });
  if (
    sha256(canonicalJson(currentCandidate))
      !== args.preflight.candidateLockSha256
  ) {
    throw new Error("Study candidate changed during held-out evaluation");
  }
  const claimBytes = await verifyClaimArtifact({
    claimed: args.claimed,
    reference: args.claimArtifact,
    description: "Study test claim identity",
  });
  assertJsonEqual(
    parseStrictJsonBytes(
      claimBytes,
      "Study test claim identity",
      studyTestClaimIdentitySchema,
    ),
    args.preflight.claimIdentity,
    "Study test claim identity",
  );
  for (const reference of args.artifacts) {
    await verifyClaimArtifact({
      claimed: args.claimed,
      reference,
      description: `Study test artifact ${reference.path}`,
    });
  }
}

async function publishFailureReceipt(args: {
  preflight: StudyTestPreflight;
  claimed: ClaimedStudyTest;
  phase: z.infer<typeof studyTestFailurePhaseSchema>;
}): Promise<string> {
  await verifyStudyTestLedger(args.claimed.ledger);
  await verifyDirectoryIdentity({
    path: args.claimed.directory,
    expected: args.claimed.directoryIdentity,
    description: "Study test claim directory",
  });
  const receipt = studyTestFailureReceiptSchema.parse({
    ...receiptCommon(args.preflight),
    status: "failed",
    error: {
      phase: args.phase,
      message: `Held-out Study test failed during ${args.phase}`,
    },
  });
  const path = join(args.claimed.directory, "failure-receipt.json");
  await writeFileOnceAtomic(
    path,
    jsonText(receipt),
  );
  return path;
}

export async function runStudyTest(args: {
  studyPath: string;
  lockPath?: string;
}): Promise<{
  claimDirectory: string;
  receiptPath: string;
  receipt: StudyTestSuccessReceipt;
}> {
  return runStudyTestWithHooks(args, {});
}

export async function runStudyTestWithHooks(
  args: {
    studyPath: string;
    lockPath?: string;
  },
  hooks: StudyTestInternalHooks,
): Promise<{
  claimDirectory: string;
  receiptPath: string;
  receipt: StudyTestSuccessReceipt;
}> {
  const preflight = await loadStudyTestPreflight(args);
  const ledger = await prepareStudyTestLedger();
  let claimed: ClaimedStudyTest | undefined;
  let phase: z.infer<typeof studyTestFailurePhaseSchema> = "claim";
  try {
    claimed = await claimStudyTest(preflight, ledger);
    await hooks.afterClaim?.({ claimDirectory: claimed.directory });
    const claimBytes = await writeFileOnceAtomic(
      join(claimed.directory, "claim.json"),
      jsonText(preflight.claimIdentity),
    );
    const claimArtifact = artifactReference("claim.json", claimBytes);

    phase = "test";
    const testData = await prepareClaimedTestData(preflight);
    phase = "projection";
    const projectionArtifact = await writeClaimArtifact({
      claimDirectory: claimed.directory,
      relativePath: "test.csv",
      value: testData.projectionCsv,
    });
    const requestValue = predictorRequest(preflight.study);
    const requestText = jsonText(requestValue);
    const requestArtifact = await writeClaimArtifact({
      claimDirectory: claimed.directory,
      relativePath: "predictor-input.json",
      value: requestText,
    });

    phase = "prediction";
    const outputPath = join(claimed.directory, "predictions.json");
    const runtimePath = join(
      claimed.directory,
      "prediction-runtime.json",
    );
    const logPath = join(claimed.directory, "predictor.log");
    const startedAt = Date.now();
    const predictionResult = await runLoggedProcess({
      command: "uv",
      commandArgs: [
        "run",
        "--locked",
        preflight.predictorCommandPath,
        "--input",
        "predictor-input.json",
        "--output",
        "predictions.json",
        "--model-dir",
        preflight.modelCommandPath,
        "--runtime-output",
        "prediction-runtime.json",
      ],
      cwd: claimed.directory,
      env: minimalMachineLearningEnvironment(process.env),
      logPath,
      timeoutMs: preflight.timeoutMs,
      timeoutMessage: (
        `Held-out Study prediction timed out after ${preflight.timeoutMs}ms`
      ),
      terminateProcessGroupOnExit: true,
      exclusiveLog: true,
      stage: "study-test-prediction",
    });
    const durationMs = Date.now() - startedAt;
    if (predictionResult.exitCode !== 0) {
      throw new Error(
        `Held-out Study predictor exited with code `
        + `${predictionResult.exitCode}`,
      );
    }
    phase = "verification";
    const runtimeArtifact = await readStrictJson({
      path: runtimePath,
      description: "Held-out prediction runtime evidence",
      maxBytes: MAX_RUNTIME_EVIDENCE_BYTES,
      schema: bundledPredictionRuntimeEvidenceSchema,
      requireSingleLink: true,
    });
    const parsedPredictions = await parsePredictions(
      outputPath,
      testData.labels,
      {
        requireSingleLink: true,
        splitDescription: "held-out test",
      },
    );
    const logBytes = await readStableRegularFile({
      path: logPath,
      description: "Held-out predictor log",
      maxBytes: MAX_PREDICTOR_LOG_BYTES,
      requireSingleLink: true,
    });
    const predictionsArtifact = artifactReference(
      "predictions.json",
      parsedPredictions.bytes,
    );
    const runtimeReference = artifactReference(
      "prediction-runtime.json",
      runtimeArtifact.bytes,
    );
    const logReference = artifactReference("predictor.log", logBytes);
    await hooks.afterPrediction?.({ claimDirectory: claimed.directory });

    phase = "scoring";
    const metrics = computeBinaryClassificationMetrics(
      parsedPredictions.rows,
    );
    await hooks.beforeCommit?.({ claimDirectory: claimed.directory });

    phase = "verification";
    await verifyFrozenInputs({
      preflight,
      claimed,
      testData,
      claimArtifact,
      artifacts: [
        projectionArtifact,
        requestArtifact,
        predictionsArtifact,
        runtimeReference,
        logReference,
      ],
    });
    parseStrictJsonBytes(
      await verifyClaimArtifact({
        claimed,
        reference: requestArtifact,
        description: "Held-out predictor request",
      }),
      "Held-out predictor request",
      z.object({
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
          prediction_csv: z.literal("test.csv"),
        }).strict(),
      }).strict(),
    );

    const receipt = studyTestSuccessReceiptSchema.parse({
      ...receiptCommon(preflight),
      status: "succeeded",
      artifacts: {
        request: requestArtifact,
        projection: projectionArtifact,
        predictions: predictionsArtifact,
      },
      execution: {
        command: ["builtin:numeric_logistic_regression:predict"],
        cwd: "<claim-directory>",
        timeout_ms: preflight.timeoutMs,
        duration_ms: durationMs,
        runtime: {
          artifact: runtimeReference,
          evidence: runtimeArtifact.value,
        },
        log: logReference,
      },
      evaluation: {
        score_semantics: "positive_class_probability",
        primary_metric: preflight.study.task.primary_metric,
        primary_score: primaryScore(
          preflight.study.task.primary_metric,
          metrics,
        ),
        metrics,
        decision_threshold: BINARY_DECISION_THRESHOLD,
        prediction_count: parsedPredictions.rows.length,
      },
    });
    phase = "publication";
    const receiptPath = join(claimed.directory, "receipt.json");
    await writeFileOnceAtomic(receiptPath, jsonText(receipt));
    return {
      claimDirectory: claimed.directory,
      receiptPath,
      receipt,
    };
  } catch (error) {
    if (claimed) {
      const failureReceiptPath = await publishFailureReceipt({
        preflight,
        claimed,
        phase,
      }).catch(() => undefined);
      throw new Error(
        `Held-out Study test failed after global claim `
        + `${preflight.claimId}; this test identity remains consumed.`
        + (
          failureReceiptPath
            ? ` Failure receipt: ${failureReceiptPath}`
            : ""
        ),
        { cause: error },
      );
    }
    throw error;
  }
}
