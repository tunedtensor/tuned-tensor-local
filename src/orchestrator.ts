import { lstat, mkdir, open, readFile, readdir, rename, rm, stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { createHash, randomUUID } from "node:crypto";
import { performance } from "node:perf_hooks";
import { homedir } from "node:os";
import { basename, dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  defaultArtifactPrefix,
  fileUri,
  assertArtifactManifest,
  ARTIFACT_WORKFLOW_LOCK_FILE,
  claimRunArtifactDirectory,
  prepareRunDirectories,
  readJson,
  resolveRunArtifacts,
  writeArtifactManifest,
  writeFileAtomic,
  writeJsonAtomic,
  type ArtifactManifestModel,
  type RunArtifacts,
} from "./artifacts.js";
import {
  evalReportSchema,
  fineTuneRunRequestSchema,
  localRunnerConfigSchema,
  runReportSchema,
  trainingReportSchema,
  type BehaviorSpecExample,
  type EvalReport,
  type EvalSplit,
  type FineTuneRunRequest,
  type LocalRunnerConfig,
  type RunReport,
  type TrainingReport,
} from "./contracts.js";
import {
  buildSystemMessage,
  compileSpecToJsonl,
  evaluationSuiteFromChatJsonl,
  examplesFromChatJsonl,
  examplesFromSpec,
  normalizeChatJsonlForRelocation,
} from "./dataset.js";
import {
  INFERENCE_PROTOCOL_VERSION,
  compareEvalReports,
  deriveSampleSeed,
  evaluateExamples,
  splitSpecExamples,
} from "./evaluation.js";
import { launchProcessTraining } from "./process-training.js";
import { assertUsableModelArtifact, localModelArtifactPath } from "./model-registry.js";
import { ProcessCancelledError } from "./process-runner.js";
import type { LocalRunReporter } from "./run-reporter.js";
import { createLocalStore, type LocalRunStatus, type LocalStore } from "./store.js";
import { withHuggingFaceCacheEnvironment } from "./huggingface-cache.js";
import { verifyLocalBaseModel } from "./prefetch.js";
import { evaluateGeneralRegressionGate } from "./general-regression.js";

export interface LocalRunResult {
  request: FineTuneRunRequest;
  report: RunReport;
  reportPath: string;
  artifactDir: string;
}

interface StageMetadata {
  run_id: string;
  behavior_spec_id: string;
  user_id: string;
  request_fingerprint: string;
  runtime_fingerprint: string;
  source_fingerprint: string;
  eval_split: EvalSplit;
  eval_sample_seed: number;
  eval_examples_total: number;
  eval_examples_used: number;
  max_eval_examples: number | null;
  training_example_count: number | null;
  dataset_prebuilt: boolean;
  dataset_format: NonNullable<FineTuneRunRequest["dataset_prebuilt"]>["format"] | null;
  dataset_fingerprints: Record<string, string>;
  dataset_uri: string;
  base_model_for_evaluation: string;
  base_model_revision: string | null;
  base_model_fingerprint: string | null;
  system_prompt_sha256: string;
  prepared_at: string;
}

interface PreparedRun {
  request: FineTuneRunRequest;
  artifacts: RunArtifacts;
  metadata: StageMetadata;
  examples: BehaviorSpecExample[];
  system: string;
  baseModelForEvaluation: string;
  maxEvalExamples?: number;
  generalRegression?: {
    datasetPath: string;
    datasetSha256: string;
    examples: BehaviorSpecExample[];
    system: string;
  };
}

export async function loadJsonFile<T>(path: string): Promise<T> {
  return JSON.parse(await readFile(path, "utf8")) as T;
}

export async function loadRunRequest(path: string): Promise<FineTuneRunRequest> {
  return fineTuneRunRequestSchema.parse(await loadJsonFile<unknown>(path));
}

export async function loadLocalRunnerConfig(path?: string): Promise<LocalRunnerConfig> {
  if (!path) return localRunnerConfigSchema.parse({});
  const configPath = resolve(path);
  const base = dirname(configPath);
  const config = localRunnerConfigSchema.parse(await loadJsonFile<unknown>(configPath));
  const configPathValue = (value: string | undefined): string | undefined => {
    if (!value) return undefined;
    if (value === "~") return homedir();
    if (value.startsWith("~/")) return resolve(homedir(), value.slice(2));
    return resolve(base, value);
  };
  return {
    ...config,
    artifactRoot: configPathValue(config.artifactRoot)!,
    storeRoot: configPathValue(config.storeRoot),
    paths: {
      baseModel: configPathValue(config.paths.baseModel),
      modelCache: configPathValue(config.paths.modelCache),
    },
    evaluation: {
      ...config.evaluation,
      generalRegression: config.evaluation.generalRegression
        ? {
            ...config.evaluation.generalRegression,
            dataset: configPathValue(config.evaluation.generalRegression.dataset)!,
          }
        : undefined,
    },
  };
}

function elapsed(started: number): { ms: number; seconds: number } {
  const ms = Math.max(0, Math.round(performance.now() - started));
  return { ms, seconds: Math.round((ms / 1000) * 1000) / 1000 };
}

function stripFileUri(path: string): string {
  return path.replace(/^file:\/\//, "");
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

function selectPrebuiltEvaluationSplit(dataset: FineTuneRunRequest["dataset_prebuilt"]): EvalSplit {
  if (!dataset) throw new Error("dataset_prebuilt is required");
  if (dataset.validation) return "prebuilt_validation";
  if (dataset.test) return "prebuilt_test";
  throw new Error("dataset_prebuilt requires a distinct validation or test split");
}

interface ValidatedDataset {
  trainingJsonl: string;
  trainingExampleCount: number;
  evaluationExamples: BehaviorSpecExample[];
  evalSplit: EvalSplit;
}

function assertMatchingDatasetSystems(
  jsonl: string,
  expectedSystem: string,
  split: string,
): void {
  for (const [index, line] of jsonl.split(/\r?\n/).entries()) {
    if (!line.trim()) continue;
    const row = JSON.parse(line) as {
      messages: Array<{ role: "system" | "user" | "assistant"; content: string }>;
    };
    const actualSystem = row.messages.find((message) => message.role === "system")?.content ?? "";
    if (actualSystem !== expectedSystem) {
      throw new Error(
        `Prebuilt ${split} row ${index + 1} system message must match the behavior spec system message.`,
      );
    }
  }
}

function addDryRunPlaceholders(request: unknown, dryRun: boolean): unknown {
  if (!dryRun || !request || typeof request !== "object" || Array.isArray(request)) return request;
  const candidate = request as Record<string, unknown>;
  if (candidate.dataset_prebuilt) return request;
  const snapshot = candidate.spec_snapshot;
  if (!snapshot || typeof snapshot !== "object" || Array.isArray(snapshot)) return request;
  const examples = (snapshot as Record<string, unknown>).examples;
  if (!Array.isArray(examples) || examples.length > 0) return request;
  return {
    ...candidate,
    spec_snapshot: {
      ...(snapshot as Record<string, unknown>),
      examples: [
        { input: "Dry-run placeholder input A", output: "Dry-run placeholder output A" },
        { input: "Dry-run placeholder input B", output: "Dry-run placeholder output B" },
      ],
    },
  };
}

async function validateDatasetInputs(
  request: FineTuneRunRequest,
  config: LocalRunnerConfig,
): Promise<ValidatedDataset> {
  const inputIdentity = (value: string) =>
    value.normalize("NFKC").trim().replace(/\s+/g, " ").toLocaleLowerCase("en");
  const evalSampleSeed = config.evaluation.sampleSeed
    ?? deriveSampleSeed(request.behavior_spec_id);
  if (!request.dataset_prebuilt) {
    const inlineExamples = examplesFromSpec(request.spec_snapshot);
    const seenInputs = new Map<string, number>();
    for (const [index, example] of inlineExamples.entries()) {
      const identity = inputIdentity(example.input);
      const previous = seenInputs.get(identity);
      if (previous !== undefined) {
        throw new Error(
          `Inline examples ${previous + 1} and ${index + 1} have duplicate inputs; `
          + "training and held-out evaluation require distinct prompts.",
        );
      }
      seenInputs.set(identity, index);
    }
    if (!config.dryRun && inlineExamples.length < 2) {
      throw new Error(
        "A real local fine-tune requires at least 2 inline examples so training and evaluation use distinct data.",
      );
    }
    const split = splitSpecExamples(inlineExamples, evalSampleSeed);
    const hasHoldout = split.holdout.length > 0;
    const trainingExamples = hasHoldout ? split.train : inlineExamples;
    const evaluationExamples = hasHoldout ? split.holdout : inlineExamples;
    return {
      trainingJsonl: compileSpecToJsonl({ ...request.spec_snapshot, examples: trainingExamples }),
      trainingExampleCount: trainingExamples.length,
      evaluationExamples,
      evalSplit: hasHoldout ? "spec_holdout" : "spec_examples",
    };
  }

  const dataset = request.dataset_prebuilt;
  const splitEntries = [
    ["training", dataset.training],
    ["validation", dataset.validation],
    ["test", dataset.test],
  ] as const;
  const normalized = new Map<string, { jsonl: string; examples: BehaviorSpecExample[] }>();
  const expectedSystem = buildSystemMessage(request.spec_snapshot);
  for (const [name, value] of splitEntries) {
    if (!value) continue;
    const path = stripFileUri(value);
    const jsonl = await normalizeChatJsonlForRelocation(path);
    assertMatchingDatasetSystems(jsonl, expectedSystem, name);
    normalized.set(name, {
      jsonl,
      examples: await examplesFromChatJsonl(path),
    });
  }

  const evaluationSplit = selectPrebuiltEvaluationSplit(dataset);
  const evaluationKey = evaluationSplit === "prebuilt_validation"
    ? "validation"
    : "test";
  const training = normalized.get("training");
  const evaluationData = normalized.get(evaluationKey);
  if (!training || !evaluationData) {
    throw new Error("Unable to load the configured prebuilt training and evaluation datasets.");
  }
  const trainingInputs = new Set(
    training.examples.map((example) => inputIdentity(example.input)),
  );
  for (const splitName of ["validation", "test"] as const) {
    const split = normalized.get(splitName);
    if (!split) continue;
    const splitInputs = new Set<string>();
    for (const example of split.examples) {
      const identity = inputIdentity(example.input);
      if (splitInputs.has(identity)) {
        throw new Error(
          `Prebuilt ${splitName} data contains duplicate inputs; `
          + "evaluation prompts must be unique so metrics and comparisons remain well-defined.",
        );
      }
      splitInputs.add(identity);
    }
    const overlap = split.examples.filter((example) =>
      trainingInputs.has(inputIdentity(example.input))
    );
    if (overlap.length > 0) {
      throw new Error(
        `Prebuilt training and ${splitName} data overlap on ${overlap.length} input(s); `
        + "evaluation examples must be held out from training.",
      );
    }
  }
  return {
    trainingJsonl: training.jsonl,
    trainingExampleCount: training.examples.length,
    evaluationExamples: evaluationData.examples,
    evalSplit: evaluationSplit,
  };
}

/**
 * Parses the public contracts and fully validates every dataset before a run
 * lock, store record, or artifact-directory claim is created.
 */
export async function validateLocalFineTuneInput(input: {
  request: unknown;
  config: unknown;
}): Promise<{ request: FineTuneRunRequest; config: LocalRunnerConfig }> {
  const config = localRunnerConfigSchema.parse(input.config);
  const request = fineTuneRunRequestSchema.parse(addDryRunPlaceholders(input.request, config.dryRun));
  await validateDatasetInputs(request, config);
  if (config.evaluation.generalRegression) {
    await evaluationSuiteFromChatJsonl(
      config.evaluation.generalRegression.dataset,
      config.evaluation.generalRegression.systemPrompt,
    );
  }
  return { request, config };
}

function artifactPrefix(request: FineTuneRunRequest): string {
  return defaultArtifactPrefix({
    userId: request.user_id,
    behaviorSpecId: request.behavior_spec_id,
    runId: request.run_id,
  });
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map((entry) => canonicalJson(entry)).join(",")}]`;
  const object = value as Record<string, unknown>;
  return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(object[key])}`).join(",")}}`;
}

function hashJson(value: unknown): string {
  return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));

async function hashFileIfPresent(path: string): Promise<string | null> {
  try {
    return await hashFile(path);
  } catch {
    return null;
  }
}

async function packageVersion(): Promise<string> {
  try {
    const parsed = JSON.parse(await readFile(resolve(packageRoot, "package.json"), "utf8")) as { version?: unknown };
    return typeof parsed.version === "string" ? parsed.version : "unknown";
  } catch {
    return "unknown";
  }
}

async function runtimeFingerprint(): Promise<string> {
  const bundledProject = resolve(packageRoot, "training/local-runner");
  return hashJson({
    tt_local_version: await packageVersion(),
    node_version: process.version,
    platform: process.platform,
    architecture: process.arch,
    runner_pyproject: await hashFileIfPresent(resolve(bundledProject, "pyproject.toml")),
    runner_uv_lock: await hashFileIfPresent(resolve(bundledProject, "uv.lock")),
  });
}

function preparedSourceFingerprint(args: {
  requestFingerprint: string;
  runtimeFingerprint: string;
  preparationConfig: unknown;
  baseModelRevision?: string;
  baseModelFingerprint?: string;
  datasetFingerprints: Record<string, string>;
}): string {
  return hashJson({
    request_fingerprint: args.requestFingerprint,
    runtime_fingerprint: args.runtimeFingerprint,
    preparation_config: args.preparationConfig,
    base_model_revision: args.baseModelRevision ?? null,
    base_model_fingerprint: args.baseModelFingerprint ?? null,
    dataset_fingerprints: args.datasetFingerprints,
  });
}

async function hashFile(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

async function datasetFingerprints(request: FineTuneRunRequest): Promise<Record<string, string>> {
  const dataset = request.dataset_prebuilt;
  const entries: Array<[string, string | undefined]> = [
    ["training", dataset?.training],
    ["validation", dataset?.validation],
    ["test", dataset?.test],
  ];
  const fingerprints: Record<string, string> = {};
  for (const [key, value] of entries) {
    if (!value) continue;
    const datasetPath = stripFileUri(value);
    fingerprints[key] = await hashFile(datasetPath);
  }
  return fingerprints;
}

async function resolveBaseModelRevision(
  request: FineTuneRunRequest,
  config: LocalRunnerConfig,
): Promise<string | undefined> {
  const explicit = request.hyperparameters.base_model_revision;
  if (explicit) return explicit;
  if (config.paths.baseModel) {
    const match = resolve(config.paths.baseModel).match(/[\\/]snapshots[\\/]([^\\/]+)(?:[\\/]|$)/);
    if (match?.[1]) return match[1];
  }
  if (!request.spec_snapshot.base_model.includes("/")) return undefined;
  const repository = `models--${request.spec_snapshot.base_model.replaceAll("/", "--")}`;
  const cacheEnvironment = withHuggingFaceCacheEnvironment(process.env, config.paths.modelCache);
  const refPath = resolve(cacheEnvironment.HF_HUB_CACHE!, repository, "refs", "main");
  try {
    const revision = (await readFile(refPath, "utf8")).trim();
    return revision || undefined;
  } catch {
    return undefined;
  }
}

/** Hash a local base model while permitting Hugging Face snapshot file links. */
export async function fingerprintLocalBaseModel(uri: string): Promise<string> {
  const root = localModelArtifactPath(uri);
  const rootMetadata = await lstat(root);
  if (rootMetadata.isSymbolicLink()) {
    throw new Error(`Local base model path must not itself be a symbolic link: ${root}`);
  }
  if (!rootMetadata.isDirectory()) {
    throw new Error(`Local base model must be a Hugging Face snapshot directory: ${root}`);
  }
  await verifyLocalBaseModel(root);
  const files: Array<{ path: string; size_bytes: number; sha256: string }> = [];
  const visit = async (directory: string): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const child = resolve(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(child);
        continue;
      }
      const childMetadata = await stat(child);
      if (entry.isSymbolicLink() && !childMetadata.isFile()) {
        throw new Error(`Local base model contains a non-file symbolic link: ${child}`);
      }
      if (!childMetadata.isFile()) continue;
      files.push({
        path: relative(root, child).split("\\").join("/"),
        size_bytes: childMetadata.size,
        sha256: await hashFile(child),
      });
    }
  };
  await visit(root);
  files.sort((left, right) => left.path.localeCompare(right.path));
  if (files.length === 0) throw new Error(`Local base model directory is empty: ${root}`);
  return hashJson({ kind: "directory", files });
}

function statusForProgressStage(stage: string): LocalRunStatus {
  if (stage === "evaluating_baseline") return "evaluating_baseline";
  if (stage === "evaluating_candidate") return "evaluating_candidate";
  if (stage === "training") return "training";
  if (stage === "preparing") return "preparing";
  if (stage === "reporting") return "reporting";
  return "training";
}

async function readStageMetadata(path: string): Promise<StageMetadata | null> {
  try {
    const metadata = await readJson<Partial<StageMetadata>>(path);
    return typeof metadata.source_fingerprint === "string"
      ? metadata as StageMetadata
      : null;
  } catch {
    return null;
  }
}

async function removePrefixedArtifacts(path: string): Promise<void> {
  const directory = dirname(path);
  const prefix = `${basename(path)}.`;
  const names = await readdir(directory).catch(() => []);
  await Promise.all([
    rm(path, { force: true }),
    ...names.filter((name) => name.startsWith(prefix)).map((name) => rm(resolve(directory, name), { recursive: true, force: true })),
  ]);
}

async function cleanupStageArtifacts(
  artifacts: RunArtifacts,
  stage: "prepare" | "baseline" | "train" | "candidate" | "report",
): Promise<void> {
  const removeReport = async () => rm(artifacts.runReportJson, { force: true });
  if (stage === "prepare") {
    await Promise.all([
      removePrefixedArtifacts(artifacts.baselineEvalJson),
      removePrefixedArtifacts(artifacts.candidateEvalJson),
      removePrefixedArtifacts(artifacts.generalBaselineEvalJson),
      removePrefixedArtifacts(artifacts.generalCandidateEvalJson),
      rm(artifacts.trainingDir, { recursive: true, force: true }),
      removePrefixedArtifacts(artifacts.trainingReportJson),
      rm(resolve(artifacts.runDir, "model.tar.gz"), { force: true }),
      removeReport(),
      rm(artifacts.artifactManifestJson, { force: true }),
    ]);
    return;
  }
  if (stage === "baseline") {
    await Promise.all([
      removePrefixedArtifacts(artifacts.baselineEvalJson),
      removePrefixedArtifacts(artifacts.generalBaselineEvalJson),
      removeReport(),
    ]);
    return;
  }
  if (stage === "train") {
    await Promise.all([
      rm(artifacts.trainingDir, { recursive: true, force: true }),
      removePrefixedArtifacts(artifacts.trainingReportJson),
      rm(resolve(artifacts.runDir, "model.tar.gz"), { force: true }),
      removePrefixedArtifacts(artifacts.candidateEvalJson),
      removePrefixedArtifacts(artifacts.generalCandidateEvalJson),
      removeReport(),
    ]);
    await prepareRunDirectories(artifacts);
    return;
  }
  if (stage === "candidate") {
    await Promise.all([
      removePrefixedArtifacts(artifacts.candidateEvalJson),
      removePrefixedArtifacts(artifacts.generalCandidateEvalJson),
      removeReport(),
    ]);
    return;
  }
  await removeReport();
}

function manifestRelativePath(artifacts: RunArtifacts, path: string): string {
  return relative(artifacts.runDir, path).split("\\").join("/");
}

async function verifyReusableArtifacts(
  artifacts: RunArtifacts,
  paths: string[],
  options: { verifyModel?: boolean } = {},
): Promise<boolean> {
  if (!await pathExists(artifacts.artifactManifestJson)) return false;
  await assertArtifactManifest(artifacts.artifactManifestJson, {
    requiredPaths: paths.map((path) => manifestRelativePath(artifacts, path)),
    scopeToRequired: true,
    verifyModel: options.verifyModel,
  });
  return true;
}

async function throwIfCancelled(store: LocalStore, request: FineTuneRunRequest): Promise<void> {
  if (await store.isCancellationRequested(request.run_id)) {
    throw new ProcessCancelledError(`Run ${request.run_id} was cancelled.`);
  }
}

async function acquireWorkflowLock(lockPath: string, description: string): Promise<() => Promise<void>> {
  await mkdir(dirname(lockPath), { recursive: true });
  const token = randomUUID();
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const handle = await open(lockPath, "wx", 0o600);
      try {
        await handle.writeFile(`${JSON.stringify({ token, pid: process.pid, created_at: new Date().toISOString() })}\n`);
      } finally {
        await handle.close();
      }
      return async () => {
        try {
          const owner = JSON.parse(await readFile(lockPath, "utf8")) as { token?: unknown };
          if (owner.token === token) await rm(lockPath, { force: true });
        } catch {
          // The lock may already have been removed during process shutdown.
        }
      };
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      const metadata = await stat(lockPath).catch(() => null);
      let owner: { pid?: unknown; created_at?: unknown } | null = null;
      try {
        owner = JSON.parse(await readFile(lockPath, "utf8")) as { pid?: unknown; created_at?: unknown };
      } catch {
        owner = null;
      }
      let ownerAlive = false;
      if (typeof owner?.pid === "number" && Number.isSafeInteger(owner.pid) && owner.pid > 0) {
        try {
          process.kill(owner.pid, 0);
          ownerAlive = true;
        } catch (killError) {
          ownerAlive = (killError as NodeJS.ErrnoException).code !== "ESRCH";
        }
      }
      const recent = metadata ? Date.now() - metadata.mtimeMs < 10_000 : true;
      if (ownerAlive || (!owner && recent)) {
        throw new Error(
          `${description} already has an active local workflow (lock: ${lockPath}).`,
        );
      }
      const stalePath = `${lockPath}.stale.${randomUUID()}`;
      try {
        await rename(lockPath, stalePath);
        await rm(stalePath, { force: true });
      } catch (renameError) {
        if ((renameError as NodeJS.ErrnoException).code !== "ENOENT") throw renameError;
      }
    }
  }
  throw new Error(`Unable to acquire local workflow lock for ${description}.`);
}

async function acquireRunLock(store: LocalStore, runId: string): Promise<() => Promise<void>> {
  return acquireWorkflowLock(resolve(store.paths.runsDir, runId, "workflow.lock"), `Run ${runId}`);
}

async function acquireArtifactLock(artifacts: RunArtifacts): Promise<() => Promise<void>> {
  return acquireWorkflowLock(
    join(artifacts.runDir, ARTIFACT_WORKFLOW_LOCK_FILE),
    `Artifact directory ${artifacts.runDir}`,
  );
}

function isDryTraining(training: TrainingReport): boolean {
  return training.metrics?.dry_run === true;
}

async function modelManifestContract(
  prepared: PreparedRun,
  training: TrainingReport,
): Promise<Omit<ArtifactManifestModel, "files"> | undefined> {
  if (isDryTraining(training) || !training.model_artifact_uri) return undefined;
  const inspection = await assertUsableModelArtifact(training.model_artifact_uri);
  const adapterWeights = inspection.adapter_weight_file_count > 0
    && inspection.adapter_weight_bytes > 0;
  if (!adapterWeights || !inspection.has_adapter_config) {
    throw new Error(
      `Local training must produce a PEFT adapter with adapter_model.safetensors or adapter_model.bin and `
      + `a non-empty adapter_config.json: ${inspection.path}`,
    );
  }
  const format = inspection.kind === "file" ? "tar.gz" : "huggingface-directory";
  return {
    artifact_kind: inspection.kind,
    format,
    framework: "transformers-peft",
    base_model: prepared.request.spec_snapshot.base_model,
    base_model_revision: prepared.metadata.base_model_revision
      ?? prepared.request.hyperparameters.base_model_revision,
    base_model_artifact_uri: training.base_model_artifact_uri,
    base_model_fingerprint: prepared.metadata.base_model_fingerprint ?? undefined,
    artifact_uri: training.model_artifact_uri,
    artifact_root: inspection.path,
    servable: true,
  };
}

async function refreshArtifactManifest(
  prepared: PreparedRun,
): Promise<void> {
  let model: Omit<ArtifactManifestModel, "files"> | undefined;
  if (await pathExists(prepared.artifacts.trainingReportJson)) {
    const training = trainingReportSchema.parse(await readJson<unknown>(prepared.artifacts.trainingReportJson));
    model = await modelManifestContract(prepared, training);
  }
  await writeArtifactManifest(prepared.artifacts, { model });
}

type FingerprintedStage = "baseline" | "train" | "candidate";

function stageOutputPath(prepared: PreparedRun, stage: FingerprintedStage): string {
  if (stage === "baseline") return prepared.artifacts.baselineEvalJson;
  if (stage === "train") return prepared.artifacts.trainingReportJson;
  return prepared.artifacts.candidateEvalJson;
}

function stageFingerprintPath(prepared: PreparedRun, stage: FingerprintedStage): string {
  return `${stageOutputPath(prepared, stage)}.stage.json`;
}

async function runnerFileFingerprints(): Promise<Record<string, string | null>> {
  const project = resolve(packageRoot, "training/local-runner");
  const bundledSource = resolve(packageRoot, "training/local-runner/src");
  const pythonSources = (await readdir(bundledSource, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && entry.name.endsWith(".py"))
    .map((entry) => resolve(bundledSource, entry.name));
  const candidates = [...new Set([
    ...pythonSources,
    resolve(project, "pyproject.toml"),
    resolve(project, "uv.lock"),
  ])];
  const fingerprints: Record<string, string | null> = {};
  for (const path of candidates.sort()) fingerprints[path] = await hashFileIfPresent(path);
  return fingerprints;
}

async function stageFingerprint(args: {
  stage: FingerprintedStage;
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  training?: TrainingReport;
}): Promise<string> {
  const common = {
    source_fingerprint: args.prepared.metadata.source_fingerprint,
    runtime_fingerprint: args.prepared.metadata.runtime_fingerprint,
    dry_run: args.config.dryRun,
  };
  if (args.stage === "train") {
    const entrypointFiles = await runnerFileFingerprints();
    return hashJson({
      ...common,
      paths: args.config.paths,
      entrypoint_files: entrypointFiles,
    });
  }
  const entrypointFiles = await runnerFileFingerprints();
  const evaluation = {
    inference_protocol_version: INFERENCE_PROTOCOL_VERSION,
    inference: args.config.evaluation.inference,
    scoring: args.config.evaluation.scoring,
    timeout_ms: args.config.evaluation.timeoutMs,
    baseline_cache: args.config.evaluation.baselineCache,
    general_regression: args.prepared.generalRegression
      ? {
          dataset_sha256: args.prepared.generalRegression.datasetSha256,
          system_prompt: args.prepared.generalRegression.system,
        }
      : null,
    model_cache: args.config.paths.modelCache,
  };
  return hashJson({
    ...common,
    evaluation,
    entrypoint_files: entrypointFiles,
  });
}

async function writeStageFingerprint(args: {
  stage: FingerprintedStage;
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  training?: TrainingReport;
}): Promise<void> {
  await writeJsonAtomic(stageFingerprintPath(args.prepared, args.stage), {
    schema_version: 1,
    stage: args.stage,
    fingerprint: await stageFingerprint(args),
    written_at: new Date().toISOString(),
  });
}

async function hasCurrentStageFingerprint(args: {
  stage: FingerprintedStage;
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  training?: TrainingReport;
}): Promise<boolean> {
  try {
    const record = await readJson<{ fingerprint?: unknown }>(stageFingerprintPath(args.prepared, args.stage));
    return record.fingerprint === await stageFingerprint(args);
  } catch {
    return false;
  }
}

async function canReuseStageArtifact(args: {
  stage: FingerprintedStage;
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  training?: TrainingReport;
  verifyModel?: boolean;
  additionalPaths?: string[];
}): Promise<boolean> {
  const output = stageOutputPath(args.prepared, args.stage);
  if (!await pathExists(output) || !await hasCurrentStageFingerprint(args)) return false;
  return verifyReusableArtifacts(
    args.prepared.artifacts,
    [output, stageFingerprintPath(args.prepared, args.stage), ...(args.additionalPaths ?? [])],
    { verifyModel: args.verifyModel },
  );
}

function createStoreReporter(input: {
  request: FineTuneRunRequest;
  store: LocalStore;
  reporter?: LocalRunReporter;
}): LocalRunReporter {
  return {
    verbose: input.reporter?.verbose,
    async onEvent(event) {
      const state = await input.store.updateRun({
        runId: input.request.run_id,
        status: statusForProgressStage(event.stage),
        stage: event.stage,
        message: event.message,
        details: event.details,
      });
      if (state.status === "cancelled") throw new ProcessCancelledError(`Run ${input.request.run_id} was cancelled.`);
      await input.reporter?.onEvent?.(event);
    },
    async onLog(log) {
      await input.reporter?.onLog?.(log);
    },
  };
}

async function updateRun(input: {
  store: LocalStore;
  reporter?: LocalRunReporter;
  request: FineTuneRunRequest;
  status: LocalRunStatus;
  stage: string;
  message: string;
  details?: Record<string, unknown>;
}) {
  const state = await input.store.updateRun({
    runId: input.request.run_id,
    status: input.status,
    stage: input.stage,
    message: input.message,
    details: input.details,
  });
  if (state.status === "cancelled") throw new ProcessCancelledError(`Run ${input.request.run_id} was cancelled.`);
  await input.reporter?.onEvent?.({
    stage: input.stage,
    status: input.status,
    message: input.message,
    details: input.details,
  });
  return state;
}

async function ensureRunRecord(args: {
  request: FineTuneRunRequest;
  artifacts: RunArtifacts;
  store: LocalStore;
  reporter?: LocalRunReporter;
}): Promise<void> {
  await prepareRunDirectories(args.artifacts);
  let existing: Awaited<ReturnType<LocalStore["getRun"]>> | null = null;
  try {
    existing = await args.store.getRun(args.request.run_id);
  } catch (error) {
    if (!(error instanceof Error) || !error.message.startsWith("Run not found:")) throw error;
  }
  if (existing) {
    if (
      existing.behavior_spec_id !== args.request.behavior_spec_id
      || existing.user_id !== args.request.user_id
      || existing.run_number !== args.request.run_number
    ) {
      throw new Error(
        `Run ${args.request.run_id} cannot be reused with different user, behavior spec, or run number identity.`,
      );
    }
    if (resolve(existing.artifact_dir) !== resolve(args.artifacts.runDir)) {
      throw new Error(
        `Run ${args.request.run_id} is already bound to artifact directory ${existing.artifact_dir}; `
        + `it cannot be resumed with ${args.artifacts.runDir}.`,
      );
    }
  } else {
    await args.store.startRun({ request: args.request, artifactDir: args.artifacts.runDir });
    await args.reporter?.onEvent?.({
      stage: "queued",
      status: "queued",
      message: "Run queued.",
      details: { run_id: args.request.run_id, artifact_dir: args.artifacts.runDir },
    });
  }
}

async function computePreparedRun(args: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  artifacts: RunArtifacts;
  writeArtifacts: boolean;
}): Promise<PreparedRun> {
  const { request, config, artifacts } = args;
  const evalSampleSeed = config.evaluation.sampleSeed
    ?? deriveSampleSeed(request.behavior_spec_id);
  const dataset = await validateDatasetInputs(request, config);
  const examples = dataset.evaluationExamples;
  if (args.writeArtifacts) {
    await writeFileAtomic(artifacts.trainingJsonl, `${dataset.trainingJsonl}\n`);
  }

  const system = buildSystemMessage(request.spec_snapshot);
  const baseModelForEvaluation = config.paths.baseModel ?? request.spec_snapshot.base_model;
  const maxEvalExamples = config.evaluation.maxExamples ?? request.hyperparameters.max_eval_examples;
  const evalExamplesUsed = Math.min(maxEvalExamples ?? examples.length, examples.length);
  const fingerprints = await datasetFingerprints(request);
  const generalRegressionConfig = config.evaluation.generalRegression;
  const generalRegression = generalRegressionConfig
    ? {
        datasetPath: generalRegressionConfig.dataset,
        datasetSha256: await hashFile(generalRegressionConfig.dataset),
        ...await evaluationSuiteFromChatJsonl(
          generalRegressionConfig.dataset,
          generalRegressionConfig.systemPrompt,
        ),
      }
    : undefined;
  const requestFingerprint = hashJson(request);
  const runtimeFingerprintValue = await runtimeFingerprint();
  const baseModelRevision = await resolveBaseModelRevision(request, config);
  const baseModelFingerprint = config.paths.baseModel
    ? await fingerprintLocalBaseModel(config.paths.baseModel)
    : undefined;
  const metadata: StageMetadata = {
    run_id: request.run_id,
    behavior_spec_id: request.behavior_spec_id,
    user_id: request.user_id,
    request_fingerprint: requestFingerprint,
    runtime_fingerprint: runtimeFingerprintValue,
    source_fingerprint: preparedSourceFingerprint({
      requestFingerprint,
      runtimeFingerprint: runtimeFingerprintValue,
      preparationConfig: {
        dry_run: config.dryRun,
        base_model_path: config.paths.baseModel,
        max_eval_examples: config.evaluation.maxExamples,
        eval_sample_seed: config.evaluation.sampleSeed,
      },
      baseModelRevision,
      baseModelFingerprint,
      datasetFingerprints: fingerprints,
    }),
    eval_split: dataset.evalSplit,
    eval_sample_seed: evalSampleSeed,
    eval_examples_total: examples.length,
    eval_examples_used: evalExamplesUsed,
    max_eval_examples: maxEvalExamples ?? null,
    training_example_count: dataset.trainingExampleCount,
    dataset_prebuilt: Boolean(request.dataset_prebuilt),
    dataset_format: request.dataset_prebuilt?.format ?? null,
    dataset_fingerprints: fingerprints,
    dataset_uri: fileUri(artifacts.trainingJsonl),
    base_model_for_evaluation: baseModelForEvaluation,
    base_model_revision: baseModelRevision ?? null,
    base_model_fingerprint: baseModelFingerprint ?? null,
    system_prompt_sha256: createHash("sha256").update(system).digest("hex"),
    prepared_at: new Date().toISOString(),
  };
  if (args.writeArtifacts) await writeJsonAtomic(artifacts.stageMetadataJson, metadata);
  return {
    request,
    artifacts,
    metadata,
    examples,
    system,
    baseModelForEvaluation,
    maxEvalExamples,
    generalRegression,
  };
}

function generalRegressionEvaluationConfig(config: LocalRunnerConfig): LocalRunnerConfig {
  const { maxExamples: _maxExamples, sampleSeed: _sampleSeed, ...evaluation } = config.evaluation;
  return { ...config, evaluation };
}

function generalRegressionRequiredPaths(
  prepared: PreparedRun,
  kind: "baseline" | "candidate",
): string[] {
  if (!prepared.generalRegression) return [];
  return [
    kind === "baseline"
      ? prepared.artifacts.generalBaselineEvalJson
      : prepared.artifacts.generalCandidateEvalJson,
  ];
}

async function prepareStage(args: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  artifacts: RunArtifacts;
  store: LocalStore;
  reporter?: LocalRunReporter;
}): Promise<PreparedRun> {
  const preparedExists = await pathExists(args.artifacts.stageMetadataJson)
    && await pathExists(args.artifacts.trainingJsonl);
  const prepared = await computePreparedRun({ ...args, writeArtifacts: false });
  await args.store.syncRunRequest(args.request, args.artifacts.runDir);
  await writeJsonAtomic(resolve(args.artifacts.runDir, "request.json"), args.request);
  const existingMetadata = preparedExists
    ? await readStageMetadata(args.artifacts.stageMetadataJson)
    : null;
  let canReuse = preparedExists
    && existingMetadata?.source_fingerprint === prepared.metadata.source_fingerprint;
  if (canReuse) {
    canReuse = await verifyReusableArtifacts(args.artifacts, [
      args.artifacts.stageMetadataJson,
      args.artifacts.trainingJsonl,
    ]);
  }
  if (canReuse) {
    await throwIfCancelled(args.store, args.request);
    await updateRun({
      ...args,
      status: "preparing",
      stage: "preparing",
      message: "Reusing prepared local run artifacts.",
      details: { artifact_dir: args.artifacts.runDir },
    });
    return prepared;
  }

  await updateRun({
    ...args,
    status: "preparing",
    stage: "preparing",
    message: "Preparing local run artifacts.",
    details: { artifact_dir: args.artifacts.runDir },
  });
  await throwIfCancelled(args.store, args.request);
  await args.store.invalidateRunOutputs(args.request.run_id, { report: true, model: true });
  await cleanupStageArtifacts(args.artifacts, "prepare");
  const refreshed = await computePreparedRun({ ...args, writeArtifacts: true });
  await refreshArtifactManifest(refreshed);
  return refreshed;
}

async function runBaselineStage(args: {
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  store: LocalStore;
  reporter?: LocalRunReporter;
  runReporter: LocalRunReporter;
}): Promise<EvalReport> {
  if (
    await canReuseStageArtifact({
      stage: "baseline",
      prepared: args.prepared,
      config: args.config,
      additionalPaths: generalRegressionRequiredPaths(args.prepared, "baseline"),
    })
  ) {
    await throwIfCancelled(args.store, args.prepared.request);
    await updateRun({
      store: args.store,
      reporter: args.reporter,
      request: args.prepared.request,
      status: "evaluating_baseline",
      stage: "evaluating_baseline",
      message: "Reusing existing baseline evaluation.",
      details: { path: args.prepared.artifacts.baselineEvalJson },
    });
    const report = evalReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.baselineEvalJson));
    await throwIfCancelled(args.store, args.prepared.request);
    return report;
  }
  await throwIfCancelled(args.store, args.prepared.request);
  await args.store.invalidateRunOutputs(args.prepared.request.run_id, { report: true });
  await cleanupStageArtifacts(args.prepared.artifacts, "baseline");
  await updateRun({
    store: args.store,
    reporter: args.reporter,
    request: args.prepared.request,
    status: "evaluating_baseline",
    stage: "evaluating_baseline",
    message: "Running baseline evaluation.",
    details: {
      examples: args.prepared.examples.length,
      eval_examples_used: args.prepared.metadata.eval_examples_used,
      eval_split: args.prepared.metadata.eval_split,
      model_id: args.prepared.baseModelForEvaluation,
    },
  });
  const report = await evaluateExamples({
    kind: "baseline",
    modelId: args.prepared.baseModelForEvaluation,
    baseModelId: args.prepared.baseModelForEvaluation,
    baseModelRevision: args.config.paths.baseModel
      ? undefined
      : args.prepared.metadata.base_model_revision ?? undefined,
    sourceFingerprint: args.prepared.metadata.base_model_fingerprint ?? undefined,
    examples: args.prepared.examples,
    system: args.prepared.system,
    config: args.config,
    outputPath: args.prepared.artifacts.baselineEvalJson,
    reporter: args.runReporter,
    maxExamples: args.prepared.maxEvalExamples,
    evalSplit: args.prepared.metadata.eval_split,
    sampleSeed: args.prepared.metadata.eval_sample_seed,
    shouldCancel: () => args.store.isCancellationRequested(args.prepared.request.run_id),
  });
  if (args.prepared.generalRegression) {
    await updateRun({
      store: args.store,
      reporter: args.reporter,
      request: args.prepared.request,
      status: "evaluating_baseline",
      stage: "evaluating_baseline",
      message: "Running baseline general regression evaluation.",
      details: {
        examples: args.prepared.generalRegression.examples.length,
        dataset: args.prepared.generalRegression.datasetPath,
      },
    });
    await evaluateExamples({
      kind: "baseline",
      modelId: args.prepared.baseModelForEvaluation,
      baseModelId: args.prepared.baseModelForEvaluation,
      baseModelRevision: args.config.paths.baseModel
        ? undefined
        : args.prepared.metadata.base_model_revision ?? undefined,
      sourceFingerprint: args.prepared.metadata.base_model_fingerprint ?? undefined,
      examples: args.prepared.generalRegression.examples,
      system: args.prepared.generalRegression.system,
      config: generalRegressionEvaluationConfig(args.config),
      outputPath: args.prepared.artifacts.generalBaselineEvalJson,
      reporter: args.runReporter,
      evalSplit: "general_regression",
      shouldCancel: () => args.store.isCancellationRequested(args.prepared.request.run_id),
    });
  }
  await throwIfCancelled(args.store, args.prepared.request);
  await writeStageFingerprint({ stage: "baseline", prepared: args.prepared, config: args.config });
  return report;
}

async function runTrainStage(args: {
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  store: LocalStore;
  reporter?: LocalRunReporter;
  runReporter: LocalRunReporter;
}): Promise<TrainingReport> {
  if (
    await canReuseStageArtifact({
      stage: "train",
      prepared: args.prepared,
      config: args.config,
      verifyModel: true,
    })
  ) {
    await throwIfCancelled(args.store, args.prepared.request);
    await updateRun({
      store: args.store,
      reporter: args.reporter,
      request: args.prepared.request,
      status: "training",
      stage: "training",
      message: "Reusing existing training result.",
      details: { path: args.prepared.artifacts.trainingReportJson },
    });
    const training = trainingReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.trainingReportJson));
    if (!isDryTraining(training)) {
      await modelManifestContract(args.prepared, training);
      await throwIfCancelled(args.store, args.prepared.request);
      await args.store.registerModel({
        request: args.prepared.request,
        training,
        artifactDir: args.prepared.artifacts.runDir,
      });
    }
    await throwIfCancelled(args.store, args.prepared.request);
    return training;
  }
  await throwIfCancelled(args.store, args.prepared.request);
  await args.store.invalidateRunOutputs(args.prepared.request.run_id, { report: true, model: true });
  await cleanupStageArtifacts(args.prepared.artifacts, "train");
  await updateRun({
    store: args.store,
    reporter: args.reporter,
    request: args.prepared.request,
    status: "training",
    stage: "training",
    message: args.config.dryRun ? "Recording dry-run training result." : "Launching local training process.",
    details: { training_backend: "local-uv", dry_run: args.config.dryRun },
  });
  const training = await launchProcessTraining({
    request: args.prepared.request,
    artifacts: args.prepared.artifacts,
    config: args.config,
    baseModelRevision: args.config.paths.baseModel
      ? undefined
      : args.prepared.metadata.base_model_revision ?? undefined,
    reporter: args.runReporter,
    shouldCancel: () => args.store.isCancellationRequested(args.prepared.request.run_id),
  });
  await throwIfCancelled(args.store, args.prepared.request);
  if (!isDryTraining(training)) {
    if (!training.model_artifact_uri) {
      throw new Error("Training process completed without a model_artifact_uri.");
    }
    await modelManifestContract(args.prepared, training);
  }
  await writeJsonAtomic(args.prepared.artifacts.trainingReportJson, training);
  await writeStageFingerprint({ stage: "train", prepared: args.prepared, config: args.config });
  await refreshArtifactManifest(args.prepared);
  await throwIfCancelled(args.store, args.prepared.request);
  if (!isDryTraining(training)) {
    await args.store.registerModel({
      request: args.prepared.request,
      training,
      artifactDir: args.prepared.artifacts.runDir,
    });
  }
  return training;
}

async function runCandidateStage(args: {
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  store: LocalStore;
  reporter?: LocalRunReporter;
  runReporter: LocalRunReporter;
}): Promise<EvalReport> {
  let verifiedTraining: TrainingReport | undefined;
  if (await pathExists(args.prepared.artifacts.trainingReportJson)) {
    const trainingCurrent = await canReuseStageArtifact({
      stage: "train",
      prepared: args.prepared,
      config: args.config,
      verifyModel: true,
    });
    if (trainingCurrent) {
      verifiedTraining = trainingReportSchema.parse(
        await readJson<unknown>(args.prepared.artifacts.trainingReportJson),
      );
    }
  }
  if (
    verifiedTraining
    && await canReuseStageArtifact({
      stage: "candidate",
      prepared: args.prepared,
      config: args.config,
      verifyModel: true,
      additionalPaths: [
        args.prepared.artifacts.trainingReportJson,
        stageFingerprintPath(args.prepared, "train"),
        ...generalRegressionRequiredPaths(args.prepared, "candidate"),
      ],
    })
  ) {
    await throwIfCancelled(args.store, args.prepared.request);
    await updateRun({
      store: args.store,
      reporter: args.reporter,
      request: args.prepared.request,
      status: "evaluating_candidate",
      stage: "evaluating_candidate",
      message: "Reusing existing candidate evaluation.",
      details: { path: args.prepared.artifacts.candidateEvalJson },
    });
    const training = verifiedTraining;
    if (!isDryTraining(training)) {
      await throwIfCancelled(args.store, args.prepared.request);
      await args.store.registerModel({
        request: args.prepared.request,
        training,
        artifactDir: args.prepared.artifacts.runDir,
      });
    }
    const report = evalReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.candidateEvalJson));
    await throwIfCancelled(args.store, args.prepared.request);
    return report;
  }
  if (!verifiedTraining) {
    throw new Error("Candidate evaluation requires current verified training output.");
  }
  await throwIfCancelled(args.store, args.prepared.request);
  await args.store.invalidateRunOutputs(args.prepared.request.run_id, { report: true });
  await cleanupStageArtifacts(args.prepared.artifacts, "candidate");
  const training = verifiedTraining;
  const modelArtifact = training.model_artifact_uri;
  if (!modelArtifact) throw new Error("candidate stage requires a model_artifact_uri in training-report.json.");
  if (!isDryTraining(training)) {
    await modelManifestContract(args.prepared, training);
    await refreshArtifactManifest(args.prepared);
    await throwIfCancelled(args.store, args.prepared.request);
    await args.store.registerModel({
      request: args.prepared.request,
      training,
      artifactDir: args.prepared.artifacts.runDir,
    });
  }
  await updateRun({
    store: args.store,
    reporter: args.reporter,
    request: args.prepared.request,
    status: "evaluating_candidate",
    stage: "evaluating_candidate",
    message: "Running candidate evaluation.",
    details: { model_artifact_uri: modelArtifact },
  });
  const report = await evaluateExamples({
    kind: "candidate",
    modelId: modelArtifact,
    baseModelId: args.prepared.baseModelForEvaluation,
    baseModelRevision: args.config.paths.baseModel
      ? undefined
      : args.prepared.metadata.base_model_revision ?? undefined,
    adapterPath: modelArtifact,
    examples: args.prepared.examples,
    system: args.prepared.system,
    config: args.config,
    outputPath: args.prepared.artifacts.candidateEvalJson,
    reporter: args.runReporter,
    maxExamples: args.prepared.maxEvalExamples,
    evalSplit: args.prepared.metadata.eval_split,
    sampleSeed: args.prepared.metadata.eval_sample_seed,
    shouldCancel: () => args.store.isCancellationRequested(args.prepared.request.run_id),
  });
  if (args.prepared.generalRegression) {
    await updateRun({
      store: args.store,
      reporter: args.reporter,
      request: args.prepared.request,
      status: "evaluating_candidate",
      stage: "evaluating_candidate",
      message: "Running candidate general regression evaluation.",
      details: {
        examples: args.prepared.generalRegression.examples.length,
        dataset: args.prepared.generalRegression.datasetPath,
      },
    });
    await evaluateExamples({
      kind: "candidate",
      modelId: modelArtifact,
      baseModelId: args.prepared.baseModelForEvaluation,
      baseModelRevision: args.config.paths.baseModel
        ? undefined
        : args.prepared.metadata.base_model_revision ?? undefined,
      adapterPath: modelArtifact,
      examples: args.prepared.generalRegression.examples,
      system: args.prepared.generalRegression.system,
      config: generalRegressionEvaluationConfig(args.config),
      outputPath: args.prepared.artifacts.generalCandidateEvalJson,
      reporter: args.runReporter,
      evalSplit: "general_regression",
      shouldCancel: () => args.store.isCancellationRequested(args.prepared.request.run_id),
    });
  }
  await throwIfCancelled(args.store, args.prepared.request);
  await writeStageFingerprint({
    stage: "candidate",
    prepared: args.prepared,
    config: args.config,
    training,
  });
  return report;
}

async function runReportStage(args: {
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  store: LocalStore;
  reporter?: LocalRunReporter;
  startedAt: string;
  startedPerf: number;
}): Promise<RunReport> {
  if (!await pathExists(args.prepared.artifacts.baselineEvalJson)) {
    throw new Error("Run reporting requires baseline-eval.json.");
  }
  if (!await pathExists(args.prepared.artifacts.candidateEvalJson)) {
    throw new Error("Run reporting requires candidate-eval.json.");
  }
  if (!await pathExists(args.prepared.artifacts.trainingReportJson)) {
    throw new Error("Run reporting requires training-report.json.");
  }
  if (
    args.prepared.generalRegression
    && (
      !await pathExists(args.prepared.artifacts.generalBaselineEvalJson)
      || !await pathExists(args.prepared.artifacts.generalCandidateEvalJson)
    )
  ) {
    throw new Error("Run reporting requires both general regression evaluations.");
  }
  const currentTraining = await canReuseStageArtifact({
    stage: "train",
    prepared: args.prepared,
    config: args.config,
    verifyModel: true,
  });
  const currentBaseline = await canReuseStageArtifact({
    stage: "baseline",
    prepared: args.prepared,
    config: args.config,
    additionalPaths: generalRegressionRequiredPaths(args.prepared, "baseline"),
  });
  const currentCandidate = await canReuseStageArtifact({
    stage: "candidate",
    prepared: args.prepared,
    config: args.config,
    verifyModel: true,
    additionalPaths: [
      args.prepared.artifacts.trainingReportJson,
      stageFingerprintPath(args.prepared, "train"),
      ...generalRegressionRequiredPaths(args.prepared, "candidate"),
    ],
  });
  if (!currentTraining || !currentBaseline || !currentCandidate) {
    throw new Error("report stage inputs are stale for the current request/config. Re-run baseline, train, and candidate as needed.");
  }
  await verifyReusableArtifacts(args.prepared.artifacts, [
    args.prepared.artifacts.baselineEvalJson,
    stageFingerprintPath(args.prepared, "baseline"),
    args.prepared.artifacts.candidateEvalJson,
    stageFingerprintPath(args.prepared, "candidate"),
    args.prepared.artifacts.trainingReportJson,
    stageFingerprintPath(args.prepared, "train"),
    ...generalRegressionRequiredPaths(args.prepared, "baseline"),
    ...generalRegressionRequiredPaths(args.prepared, "candidate"),
  ], { verifyModel: true });
  await throwIfCancelled(args.store, args.prepared.request);
  await args.store.invalidateRunOutputs(args.prepared.request.run_id, { report: true });
  await cleanupStageArtifacts(args.prepared.artifacts, "report");
  await updateRun({
    store: args.store,
    reporter: args.reporter,
    request: args.prepared.request,
    status: "reporting",
    stage: "reporting",
    message: "Writing run report.",
    details: { report_path: args.prepared.artifacts.runReportJson },
  });
  const baseline = evalReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.baselineEvalJson));
  const candidate = evalReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.candidateEvalJson));
  const training = trainingReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.trainingReportJson));
  const comparison = compareEvalReports(baseline, candidate);
  let generalRegression: RunReport["general_regression"];
  if (args.prepared.generalRegression) {
    const generalBaseline = evalReportSchema.parse(
      await readJson<unknown>(args.prepared.artifacts.generalBaselineEvalJson),
    );
    const generalCandidate = evalReportSchema.parse(
      await readJson<unknown>(args.prepared.artifacts.generalCandidateEvalJson),
    );
    const generalComparison = compareEvalReports(generalBaseline, generalCandidate);
    const policy = args.config.evaluation.generalRegression!;
    const gate = evaluateGeneralRegressionGate(generalComparison, policy);
    generalRegression = {
      dataset_uri: fileUri(args.prepared.generalRegression.datasetPath),
      dataset_sha256: args.prepared.generalRegression.datasetSha256,
      baseline: generalBaseline,
      candidate: generalCandidate,
      comparison: generalComparison,
      policy: {
        max_score_drop: policy.maxScoreDrop,
        max_pass_rate_drop: policy.maxPassRateDrop,
      },
      passed: gate.passed,
      failures: gate.failures,
    };
  }
  const completedAt = new Date().toISOString();
  const duration = elapsed(args.startedPerf);
  const report = runReportSchema.parse({
    run_id: args.prepared.request.run_id,
    behavior_spec_id: args.prepared.request.behavior_spec_id,
    user_id: args.prepared.request.user_id,
    run_number: args.prepared.request.run_number,
    base_model: args.prepared.request.spec_snapshot.base_model,
    fine_tuned_model_id: training.model_artifact_uri ?? training.training_job_name,
    status: "completed",
    baseline,
    candidate,
    comparison,
    general_regression: generalRegression,
    training,
    artifact_uris: {
      dataset: fileUri(args.prepared.artifacts.trainingJsonl),
      baseline_eval: fileUri(args.prepared.artifacts.baselineEvalJson),
      candidate_eval: fileUri(args.prepared.artifacts.candidateEvalJson),
      general_baseline_eval: generalRegression
        ? fileUri(args.prepared.artifacts.generalBaselineEvalJson)
        : undefined,
      general_candidate_eval: generalRegression
        ? fileUri(args.prepared.artifacts.generalCandidateEvalJson)
        : undefined,
      report: fileUri(args.prepared.artifacts.runReportJson),
    },
    run_metadata: {
      base_model: args.prepared.request.spec_snapshot.base_model,
      fine_tuned_model_id: training.model_artifact_uri ?? training.training_job_name,
      dataset_prebuilt: args.prepared.metadata.dataset_prebuilt,
      dataset_format: args.prepared.metadata.dataset_format,
      dataset_uri: fileUri(args.prepared.artifacts.trainingJsonl),
      spec_example_count: args.prepared.request.spec_snapshot.examples.length,
      training_example_count: args.prepared.metadata.training_example_count,
      eval_examples_total: baseline.eval_examples_total,
      eval_examples_used: baseline.eval_examples_used,
      eval_split: baseline.eval_split,
      eval_sample_seed: baseline.eval_sample_seed ?? null,
      started_at: args.startedAt,
      completed_at: completedAt,
      elapsed_ms: duration.ms,
      elapsed_seconds: duration.seconds,
    },
    created_at: completedAt,
  });
  await writeJsonAtomic(args.prepared.artifacts.runReportJson, report);
  await refreshArtifactManifest(args.prepared);
  await throwIfCancelled(args.store, args.prepared.request);
  const completedState = await args.store.completeRun(
    report,
    args.prepared.artifacts.runDir,
    args.prepared.artifacts.runReportJson,
  );
  if (completedState.status === "cancelled") {
    throw new ProcessCancelledError(`Run ${args.prepared.request.run_id} was cancelled.`);
  }
  await args.reporter?.onEvent?.({
    stage: "completed",
    status: "completed",
    message: "Run completed successfully.",
    details: {
      report_path: args.prepared.artifacts.runReportJson,
      ...(!isDryTraining(training) ? { model_id: `local-${args.prepared.request.run_id}` } : {}),
      avg_score_delta: comparison.avg_score_delta,
      general_regression_passed: generalRegression?.passed,
      elapsed_seconds: duration.seconds,
    },
  });
  return report;
}

export async function runLocalFineTune(input: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  reporter?: LocalRunReporter;
}): Promise<LocalRunResult> {
  const validated = await validateLocalFineTuneInput({
    request: input.request,
    config: input.config,
  });
  input = { ...input, ...validated };
  const startedPerf = performance.now();
  const startedAt = new Date().toISOString();
  const prefix = artifactPrefix(input.request);
  const artifacts = resolveRunArtifacts({ artifactRoot: input.config.artifactRoot, prefix });
  const store = createLocalStore(input.config.storeRoot);
  const releaseRunLock = await acquireRunLock(store, input.request.run_id);
  let releaseArtifactLock: (() => Promise<void>) | undefined;
  try {
    await claimRunArtifactDirectory({
      artifacts,
      runId: input.request.run_id,
      userId: input.request.user_id,
      behaviorSpecId: input.request.behavior_spec_id,
    });
    releaseArtifactLock = await acquireArtifactLock(artifacts);
    await ensureRunRecord({ request: input.request, artifacts, store, reporter: input.reporter });
    const runReporter = createStoreReporter({ request: input.request, store, reporter: input.reporter });
    let prepared: PreparedRun | undefined;

    try {
      await throwIfCancelled(store, input.request);
      prepared = await prepareStage({
        request: input.request,
        config: input.config,
        artifacts,
        store,
        reporter: input.reporter,
      });
      await throwIfCancelled(store, input.request);
      await runBaselineStage({
        prepared,
        config: input.config,
        store,
        reporter: input.reporter,
        runReporter,
      });
      await refreshArtifactManifest(prepared);
      await runTrainStage({
        prepared,
        config: input.config,
        store,
        reporter: input.reporter,
        runReporter,
      });
      await runCandidateStage({
        prepared,
        config: input.config,
        store,
        reporter: input.reporter,
        runReporter,
      });
      await refreshArtifactManifest(prepared);
      const report = await runReportStage({
        prepared,
        config: input.config,
        store,
        reporter: input.reporter,
        startedAt,
        startedPerf,
      });
      return {
        request: input.request,
        report,
        reportPath: artifacts.runReportJson,
        artifactDir: artifacts.runDir,
      };
    } catch (error) {
      const cancelled = error instanceof ProcessCancelledError
        || await store.isCancellationRequested(input.request.run_id).catch(() => false);
      if (cancelled) {
        const state = await store.getRun(input.request.run_id).catch(() => null);
        if (state?.status !== "cancelled") await store.cancelRun(input.request.run_id).catch(() => undefined);
        await store.finalizeCancellation(input.request.run_id).catch(() => undefined);
        await input.reporter?.onEvent?.({
          stage: "cancelled",
          status: "cancelled",
          message: "Run cancelled.",
        });
      } else {
        await store.failRun(input.request.run_id, error instanceof Error ? error.message : String(error)).catch(() => undefined);
        await input.reporter?.onEvent?.({
          stage: "failed",
          status: "failed",
          message: error instanceof Error ? error.message : String(error),
        });
      }
      throw error;
    }
  } finally {
    await releaseArtifactLock?.();
    await releaseRunLock();
  }
}
