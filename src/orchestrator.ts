import { copyFile, lstat, mkdir, open, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { createHash, randomUUID } from "node:crypto";
import { performance } from "node:perf_hooks";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";
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
  examplesFromChatJsonl,
  examplesFromSpec,
  localAssetPathsFromChatJsonl,
  normalizeChatJsonlForRelocation,
} from "./dataset.js";
import { compareEvalReports, deriveSampleSeed, evaluateExamples, rescoreEvalReport, splitSpecExamples } from "./evaluation.js";
import { launchProcessTraining } from "./process-training.js";
import { assertUsableModelArtifact, localModelArtifactPath } from "./model-registry.js";
import { ProcessCancelledError } from "./process-runner.js";
import type { LocalRunReporter } from "./run-reporter.js";
import { createLocalStore, type LocalRunStatus, type LocalStore } from "./store.js";
import { withHuggingFaceCacheEnvironment } from "./huggingface-cache.js";
import { verifyLocalBaseModel } from "./prefetch.js";

export type LocalRunStage = "prepare" | "baseline" | "train" | "candidate" | "score" | "report" | "all";

export interface LocalRunResult {
  request: FineTuneRunRequest;
  report: RunReport;
  reportPath: string;
  artifactDir: string;
}

export interface LocalStageRunResult {
  request: FineTuneRunRequest;
  stage: LocalRunStage;
  report?: RunReport;
  reportPath?: string;
  artifactDir: string;
  artifacts: {
    training_jsonl: string;
    stage_metadata: string;
    training_report: string;
    baseline_eval: string;
    candidate_eval: string;
    report: string;
    artifact_manifest: string;
  };
}

interface StageMetadata {
  run_id: string;
  behavior_spec_id: string;
  user_id: string;
  training_method: FineTuneRunRequest["training_method"];
  request_fingerprint: string;
  effective_config_fingerprint: string;
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
  parent_model_artifact: string | null;
  parent_model_fingerprint: string | null;
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
  const configPathValue = (value: string | undefined, preserveBundled = false): string | undefined => {
    if (!value) return undefined;
    if (preserveBundled && (value === "training/local-runner" || value.startsWith("training/local-runner/"))) {
      return value;
    }
    if (value === "~") return homedir();
    if (value.startsWith("~/")) return resolve(homedir(), value.slice(2));
    return resolve(base, value);
  };
  return {
    ...config,
    artifactRoot: configPathValue(config.artifactRoot)!,
    storeRoot: configPathValue(config.storeRoot),
    training: {
      ...config.training,
      cwd: configPathValue(config.training.cwd),
      project: configPathValue(config.training.project, true)!,
      script: configPathValue(config.training.script, true),
    },
    paths: {
      baseModel: configPathValue(config.paths.baseModel),
      modelCache: configPathValue(config.paths.modelCache),
    },
    evaluation: {
      ...config.evaluation,
      inference: {
        ...config.evaluation.inference,
        cwd: configPathValue(config.evaluation.inference.cwd),
        project: configPathValue(config.evaluation.inference.project, true)!,
        script: configPathValue(config.evaluation.inference.script, true)!,
      },
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

function selectPrebuiltEvaluation(dataset: FineTuneRunRequest["dataset_prebuilt"]): {
  path: string;
  split: EvalSplit;
} {
  if (!dataset) throw new Error("dataset_prebuilt is required");
  if (dataset.test) return { path: stripFileUri(dataset.test), split: "prebuilt_test" };
  if (dataset.validation) return { path: stripFileUri(dataset.validation), split: "prebuilt_validation" };
  return { path: stripFileUri(dataset.training), split: "prebuilt_training" };
}

function selectDpoEvaluation(request: FineTuneRunRequest): {
  path?: string;
  split: EvalSplit;
} {
  if (request.dataset_prebuilt?.test) {
    return { path: stripFileUri(request.dataset_prebuilt.test), split: "prebuilt_test" };
  }
  if (request.dataset_prebuilt?.validation) {
    return { path: stripFileUri(request.dataset_prebuilt.validation), split: "prebuilt_validation" };
  }
  return { split: "spec_examples" };
}

function artifactPrefix(request: FineTuneRunRequest): string {
  return request.artifacts?.prefix ?? defaultArtifactPrefix({
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

async function effectiveConfigFingerprint(config: LocalRunnerConfig): Promise<string> {
  const bundledProject = resolve(packageRoot, "training/local-runner");
  const projectRoots = [...new Set([
    bundledProject,
    config.training.project === "training/local-runner" ? bundledProject : config.training.project,
    config.evaluation.inference.project === "training/local-runner"
      ? bundledProject
      : config.evaluation.inference.project,
  ].filter((value): value is string => Boolean(value)).map((value) => resolve(value)))];
  const projectFingerprints: Record<string, { pyproject: string | null; uv_lock: string | null }> = {};
  for (const project of projectRoots) {
    projectFingerprints[project] = {
      pyproject: await hashFileIfPresent(resolve(project, "pyproject.toml")),
      uv_lock: await hashFileIfPresent(resolve(project, "uv.lock")),
    };
  }
  return hashJson({
    config,
    runtime: {
      tt_local_version: await packageVersion(),
      node_version: process.version,
      platform: process.platform,
      architecture: process.arch,
      project_fingerprints: projectFingerprints,
    },
  });
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
  parentModelFingerprint?: string;
  datasetFingerprints: Record<string, string>;
}): string {
  return hashJson({
    request_fingerprint: args.requestFingerprint,
    runtime_fingerprint: args.runtimeFingerprint,
    preparation_config: args.preparationConfig,
    base_model_revision: args.baseModelRevision ?? null,
    base_model_fingerprint: args.baseModelFingerprint ?? null,
    parent_model_fingerprint: args.parentModelFingerprint ?? null,
    dataset_fingerprints: args.datasetFingerprints,
  });
}

async function countJsonlRows(path: string): Promise<number> {
  const text = await readFile(path, "utf8");
  return text.split(/\r?\n/).filter((line) => line.trim()).length;
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
    for (const assetPath of await localAssetPathsFromChatJsonl(datasetPath)) {
      fingerprints[`${key}_asset:${assetPath}`] = await hashFile(assetPath);
    }
  }
  for (const [exampleIndex, example] of request.spec_snapshot.examples.entries()) {
    for (const [assetIndex, asset] of (example.input_assets ?? []).entries()) {
      const value = asset.image ?? asset.data_uri ?? asset.uri ?? asset.path;
      if (
        !value
        || value.startsWith("data:")
        || (/^[a-z][a-z0-9+.-]*:/i.test(value) && !value.startsWith("file://"))
      ) {
        continue;
      }
      const assetPath = stripFileUri(value);
      fingerprints[`spec_asset:${exampleIndex}:${assetIndex}:${assetPath}`] = await hashFile(assetPath);
    }
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

async function fingerprintLocalArtifact(uri: string): Promise<string> {
  const root = localModelArtifactPath(uri);
  const metadata = await lstat(root);
  if (metadata.isSymbolicLink()) {
    throw new Error(`Parent model artifact must not be a symbolic link: ${root}`);
  }
  if (metadata.isFile()) {
    return hashJson({ kind: "file", size_bytes: metadata.size, sha256: await hashFile(root) });
  }
  if (!metadata.isDirectory()) throw new Error(`Parent model artifact is not a file or directory: ${root}`);
  const files: Array<{ path: string; size_bytes: number; sha256: string }> = [];
  const visit = async (directory: string): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const child = resolve(directory, entry.name);
      if (entry.isSymbolicLink()) {
        throw new Error(`Parent model artifact must not contain symbolic links: ${child}`);
      }
      if (entry.isDirectory()) {
        await visit(child);
        continue;
      }
      const childMetadata = await stat(child);
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
  if (files.length === 0) throw new Error(`Parent model artifact directory is empty: ${root}`);
  return hashJson({ kind: "directory", files });
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
  if (stage === "scoring") return "scoring";
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

async function clearDependentStageArtifacts(artifacts: RunArtifacts): Promise<void> {
  await cleanupStageArtifacts(artifacts, "prepare");
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

async function cleanupStageArtifacts(artifacts: RunArtifacts, stage: LocalRunStage): Promise<void> {
  const removeReport = async () => rm(artifacts.runReportJson, { force: true });
  if (stage === "prepare") {
    await Promise.all([
      removePrefixedArtifacts(artifacts.baselineEvalJson),
      removePrefixedArtifacts(artifacts.candidateEvalJson),
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
      removeReport(),
    ]);
    await prepareRunDirectories(artifacts);
    return;
  }
  if (stage === "candidate") {
    await Promise.all([
      removePrefixedArtifacts(artifacts.candidateEvalJson),
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

function stringMetadata(
  value: Record<string, unknown> | null | undefined,
  key: string,
): string | undefined {
  const entry = value?.[key];
  return typeof entry === "string" && entry.length > 0 ? entry : undefined;
}

async function modelManifestContract(
  prepared: PreparedRun,
  training: TrainingReport,
): Promise<Omit<ArtifactManifestModel, "files"> | undefined> {
  if (isDryTraining(training) || !training.model_artifact_uri) return undefined;
  const metadata = training.artifact_metadata;
  const explicitContract = Boolean(metadata?.framework && metadata?.format);
  const inspection = await assertUsableModelArtifact(training.model_artifact_uri, {
    allowUnrecognizedPayload: explicitContract,
  });
  const adapterWeights = inspection.adapter_weight_file_count > 0
    && inspection.adapter_weight_bytes > 0;
  const fullModelWeights = inspection.full_model_weight_file_count > 0
    && inspection.full_model_weight_bytes > 0;
  if (!adapterWeights && !fullModelWeights && !explicitContract) {
    throw new Error(
      `Model artifact ${inspection.path} has no loadable adapter or Transformers full-model weights. `
      + "Custom trainers must set training.artifact.framework and training.artifact.format.",
    );
  }
  if ((metadata?.framework === "transformers-peft" || metadata?.servable === true)
    && (!adapterWeights || !inspection.has_adapter_config)) {
    throw new Error(
      `Model artifact ${inspection.path} requires adapter_model.safetensors or adapter_model.bin and a non-empty `
      + "adapter_config.json for a PEFT/servable contract.",
    );
  }
  if (metadata?.framework === "transformers-full" && !fullModelWeights) {
    throw new Error(`Model artifact ${inspection.path} has no Transformers full-model weights.`);
  }
  const implicitPeftAdapter = adapterWeights && inspection.has_adapter_config;
  const format = metadata?.format
    ?? (inspection.kind === "file" && inspection.path.endsWith(".tar.gz")
      ? "tar.gz"
      : inspection.kind === "directory" ? "huggingface-directory" : "file");
  return {
    artifact_kind: inspection.kind,
    format,
    framework: metadata?.framework ?? (implicitPeftAdapter ? "transformers-peft" : fullModelWeights ? "transformers-full" : "custom"),
    base_model: prepared.request.spec_snapshot.base_model,
    base_model_revision: stringMetadata(metadata, "base_model_revision")
      ?? stringMetadata(training.metrics, "base_model_revision")
      ?? prepared.metadata.base_model_revision
      ?? prepared.request.hyperparameters.base_model_revision,
    base_model_artifact_uri: training.base_model_artifact_uri,
    base_model_fingerprint: prepared.metadata.base_model_fingerprint ?? undefined,
    parent_model_artifact: prepared.request.hyperparameters.parent_model_artifact,
    artifact_uri: training.model_artifact_uri,
    artifact_root: inspection.path,
    servable: metadata?.servable ?? implicitPeftAdapter,
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

function configuredPath(path: string, cwd?: string): string | null {
  if (path === "training/local-runner" || path.startsWith("training/local-runner/")) {
    return resolve(packageRoot, path);
  }
  if (path.startsWith("file://")) return stripFileUri(path);
  if (isAbsolute(path)) return path;
  const fileLike = path.startsWith("./")
    || path.startsWith("../")
    || path.includes("/")
    || path.includes("\\")
    || /\.(?:py|js|mjs|cjs|ts|tsx|sh|bash|zsh|pl|rb)$/i.test(path);
  return fileLike ? resolve(cwd ?? process.cwd(), path) : null;
}

async function entrypointFileFingerprints(args: {
  values: Array<string | undefined>;
  cwd?: string;
  project?: string;
}): Promise<Record<string, string | null>> {
  const candidates = new Set<string>();
  for (const value of args.values) {
    if (!value) continue;
    const path = configuredPath(value, args.cwd);
    if (path) candidates.add(path);
  }
  if (args.project) {
    const project = configuredPath(args.project, args.cwd) ?? resolve(args.cwd ?? process.cwd(), args.project);
    candidates.add(resolve(project, "pyproject.toml"));
    candidates.add(resolve(project, "uv.lock"));
  }
  const fingerprints: Record<string, string | null> = {};
  for (const path of [...candidates].sort()) fingerprints[path] = await hashFileIfPresent(path);
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
    const entrypointFiles = await entrypointFileFingerprints({
      values: [
        args.config.training.script
          ?? (args.prepared.request.training_method === "dpo"
            ? "training/local-runner/src/train_dpo.py"
            : "training/local-runner/src/train.py"),
        ...(args.config.training.command ?? []),
      ],
      cwd: args.config.training.cwd,
      project: args.config.training.project,
    });
    return hashJson({
      ...common,
      training: args.config.training,
      paths: args.config.paths,
      entrypoint_files: entrypointFiles,
    });
  }
  const selectedCommand = args.stage === "baseline"
    ? args.config.evaluation.baselineCommand
    : args.config.evaluation.candidateCommand;
  const entrypointFiles = await entrypointFileFingerprints({
    values: [
      args.config.evaluation.inference.script,
      ...(args.config.evaluation.inference.command ?? []),
      ...(selectedCommand ?? []),
    ],
    cwd: args.config.evaluation.inference.cwd,
    project: args.config.evaluation.inference.project,
  });
  const evaluation = {
    command: args.stage === "baseline"
      ? args.config.evaluation.baselineCommand
      : args.config.evaluation.candidateCommand,
    inference: args.config.evaluation.inference,
    scoring: args.config.evaluation.scoring,
    timeout_ms: args.config.evaluation.timeoutMs,
    baseline_cache: args.config.evaluation.baselineCache,
    llm: args.config.llm,
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
  const evalSampleSeed = config.evaluation.sampleSeed ?? deriveSampleSeed(request.run_id);
  let examples = examplesFromSpec(request.spec_snapshot);
  let evalSplit: EvalSplit = "spec_examples";
  let trainingExampleCount: number | null = examples.length;

  if (request.training_method === "dpo") {
    if (!request.dataset_prebuilt) throw new Error("DPO training requires dataset_prebuilt.");
    const trainingPath = stripFileUri(request.dataset_prebuilt.training);
    if (args.writeArtifacts) await copyFile(trainingPath, artifacts.trainingJsonl);
    trainingExampleCount = await countJsonlRows(trainingPath);
    const evaluation = selectDpoEvaluation(request);
    evalSplit = evaluation.split;
    examples = evaluation.path
      ? await examplesFromChatJsonl(evaluation.path)
      : examplesFromSpec(request.spec_snapshot);
  } else if (request.dataset_prebuilt) {
    const trainingPath = stripFileUri(request.dataset_prebuilt.training);
    const evaluation = selectPrebuiltEvaluation(request.dataset_prebuilt);
    evalSplit = evaluation.split;
    if (evalSplit === "prebuilt_training" && !config.dryRun && !config.evaluation.allowPrebuiltTrainingEval) {
      throw new Error(
        "dataset_prebuilt has no test or validation split, so evaluation would run on the training data and "
        + "overstate improvement. Provide dataset_prebuilt.test or dataset_prebuilt.validation, or set "
        + "evaluation.allowPrebuiltTrainingEval=true to evaluate on the training split anyway.",
      );
    }
    if (args.writeArtifacts) {
      const normalizedTraining = await normalizeChatJsonlForRelocation(trainingPath);
      await writeFile(artifacts.trainingJsonl, `${normalizedTraining}\n`, "utf8");
    }
    examples = await examplesFromChatJsonl(evaluation.path);
    trainingExampleCount = null;
  } else {
    const split = splitSpecExamples(request.spec_snapshot.examples, evalSampleSeed);
    let trainingExamples = request.spec_snapshot.examples;
    if (split.holdout.length > 0) {
      trainingExamples = split.train;
      examples = split.holdout;
      evalSplit = "spec_holdout";
    }
    trainingExampleCount = trainingExamples.length;
    if (args.writeArtifacts) {
      const jsonl = compileSpecToJsonl({ ...request.spec_snapshot, examples: trainingExamples });
      await writeFile(artifacts.trainingJsonl, `${jsonl}\n`, "utf8");
    }
  }

  const system = buildSystemMessage(request.spec_snapshot);
  const baseModelForEvaluation = config.paths.baseModel ?? request.spec_snapshot.base_model;
  const parentModelArtifact = request.hyperparameters.parent_model_artifact;
  const parentModelFingerprint = parentModelArtifact
    ? await fingerprintLocalArtifact(parentModelArtifact)
    : undefined;
  const maxEvalExamples = config.evaluation.maxExamples ?? request.hyperparameters.max_eval_examples;
  const evalExamplesUsed = Math.min(maxEvalExamples ?? examples.length, examples.length);
  const fingerprints = await datasetFingerprints(request);
  const requestFingerprint = hashJson(request);
  const effectiveConfigFingerprintValue = await effectiveConfigFingerprint(config);
  const runtimeFingerprintValue = await runtimeFingerprint();
  const baseModelRevision = await resolveBaseModelRevision(request, config);
  const baseModelFingerprint = config.paths.baseModel
    ? await fingerprintLocalBaseModel(config.paths.baseModel)
    : undefined;
  const metadata: StageMetadata = {
    run_id: request.run_id,
    behavior_spec_id: request.behavior_spec_id,
    user_id: request.user_id,
    training_method: request.training_method,
    request_fingerprint: requestFingerprint,
    effective_config_fingerprint: effectiveConfigFingerprintValue,
    runtime_fingerprint: runtimeFingerprintValue,
    source_fingerprint: preparedSourceFingerprint({
      requestFingerprint,
      runtimeFingerprint: runtimeFingerprintValue,
      preparationConfig: {
        dry_run: config.dryRun,
        base_model_path: config.paths.baseModel,
        max_eval_examples: config.evaluation.maxExamples,
        eval_sample_seed: config.evaluation.sampleSeed,
        allow_prebuilt_training_eval: config.evaluation.allowPrebuiltTrainingEval,
      },
      baseModelRevision,
      baseModelFingerprint,
      parentModelFingerprint,
      datasetFingerprints: fingerprints,
    }),
    eval_split: evalSplit,
    eval_sample_seed: evalSampleSeed,
    eval_examples_total: examples.length,
    eval_examples_used: evalExamplesUsed,
    max_eval_examples: maxEvalExamples ?? null,
    training_example_count: trainingExampleCount,
    dataset_prebuilt: Boolean(request.dataset_prebuilt),
    dataset_format: request.dataset_prebuilt?.format ?? null,
    dataset_fingerprints: fingerprints,
    dataset_uri: fileUri(artifacts.trainingJsonl),
    base_model_for_evaluation: baseModelForEvaluation,
    base_model_revision: baseModelRevision ?? null,
    base_model_fingerprint: baseModelFingerprint ?? null,
    parent_model_artifact: parentModelArtifact ?? null,
    parent_model_fingerprint: parentModelFingerprint ?? null,
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
  };
}

async function prepareStage(args: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  artifacts: RunArtifacts;
  store: LocalStore;
  reporter?: LocalRunReporter;
  force?: boolean;
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
    && !args.force
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
  await clearDependentStageArtifacts(args.artifacts);
  const refreshed = await computePreparedRun({ ...args, writeArtifacts: true });
  await refreshArtifactManifest(refreshed);
  return refreshed;
}

async function ensurePrepared(args: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  artifacts: RunArtifacts;
  store: LocalStore;
  reporter?: LocalRunReporter;
  forcePrepare?: boolean;
}): Promise<PreparedRun> {
  return prepareStage({
    request: args.request,
    config: args.config,
    artifacts: args.artifacts,
    store: args.store,
    reporter: args.reporter,
    force: args.forcePrepare,
  });
}

async function runBaselineStage(args: {
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  store: LocalStore;
  reporter?: LocalRunReporter;
  runReporter: LocalRunReporter;
  force?: boolean;
}): Promise<EvalReport> {
  if (
    !args.force
    && await canReuseStageArtifact({
      stage: "baseline",
      prepared: args.prepared,
      config: args.config,
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
      model_id: args.prepared.metadata.parent_model_artifact ?? args.prepared.baseModelForEvaluation,
      parent_model_artifact: args.prepared.metadata.parent_model_artifact,
    },
  });
  const report = await evaluateExamples({
    kind: "baseline",
    modelId: args.prepared.metadata.parent_model_artifact ?? args.prepared.baseModelForEvaluation,
    baseModelId: args.prepared.baseModelForEvaluation,
    baseModelRevision: args.config.paths.baseModel
      ? undefined
      : args.prepared.metadata.base_model_revision ?? undefined,
    sourceFingerprint: args.prepared.metadata.source_fingerprint,
    adapterPath: args.prepared.metadata.parent_model_artifact ?? undefined,
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
  force?: boolean;
}): Promise<TrainingReport> {
  if (
    !args.force
    && await canReuseStageArtifact({
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
    details: { training_backend: args.config.training.backend, dry_run: args.config.dryRun },
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

async function writeExternalTrainingReport(args: {
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  modelArtifact: string;
}): Promise<TrainingReport> {
  const training = trainingReportSchema.parse({
    provider: args.config.training.backend === "command" ? "local-command" : "local-uv",
    training_job_name: `external-${args.prepared.request.run_id}`,
    model_artifact_uri: fileUri(localModelArtifactPath(args.modelArtifact)),
    base_model_artifact_uri: args.config.paths.baseModel ? fileUri(args.config.paths.baseModel) : undefined,
    parent_model_artifact_uri: args.prepared.metadata.parent_model_artifact ?? undefined,
    artifact_metadata: {
      ...(args.config.training.artifact ?? {}),
      notes: args.config.training.artifact?.notes ?? "External model artifact supplied with --model-artifact.",
    },
    metrics: { external_model_artifact: true },
    exit_code: null,
    log_uri: fileUri(args.prepared.artifacts.trainingReportJson),
  });
  await writeJsonAtomic(args.prepared.artifacts.trainingReportJson, training);
  return training;
}

async function runCandidateStage(args: {
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  store: LocalStore;
  reporter?: LocalRunReporter;
  runReporter: LocalRunReporter;
  force?: boolean;
  modelArtifact?: string;
}): Promise<EvalReport> {
  let verifiedTraining: TrainingReport | undefined;
  if (!args.modelArtifact && await pathExists(args.prepared.artifacts.trainingReportJson)) {
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
    !args.force
    && !args.modelArtifact
    && verifiedTraining
    && await canReuseStageArtifact({
      stage: "candidate",
      prepared: args.prepared,
      config: args.config,
      verifyModel: true,
      additionalPaths: [
        args.prepared.artifacts.trainingReportJson,
        stageFingerprintPath(args.prepared, "train"),
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
  if (!args.modelArtifact && !verifiedTraining) {
    throw new Error("candidate stage requires current verified training output. Run --stage train first.");
  }
  await throwIfCancelled(args.store, args.prepared.request);
  await args.store.invalidateRunOutputs(args.prepared.request.run_id, {
    report: true,
    model: Boolean(args.modelArtifact),
  });
  await cleanupStageArtifacts(args.prepared.artifacts, "candidate");
  let training: TrainingReport;
  if (args.modelArtifact) {
    training = await writeExternalTrainingReport({
      prepared: args.prepared,
      config: args.config,
      modelArtifact: args.modelArtifact,
    });
    await writeStageFingerprint({ stage: "train", prepared: args.prepared, config: args.config });
  } else if (verifiedTraining) {
    training = verifiedTraining;
  } else {
    throw new Error("candidate stage requires training output or --model-artifact.");
  }
  const modelArtifact = training.model_artifact_uri;
  if (!modelArtifact) throw new Error("candidate stage requires a model_artifact_uri in training-report.json or --model-artifact.");
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
  await throwIfCancelled(args.store, args.prepared.request);
  await writeStageFingerprint({
    stage: "candidate",
    prepared: args.prepared,
    config: args.config,
    training,
  });
  return report;
}

async function runScoreStage(args: {
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  store: LocalStore;
  reporter?: LocalRunReporter;
  runReporter: LocalRunReporter;
}): Promise<{ baseline: EvalReport; candidate: EvalReport }> {
  if (!await pathExists(args.prepared.artifacts.baselineEvalJson)) {
    throw new Error("score stage requires baseline-eval.json. Run --stage baseline first.");
  }
  if (!await pathExists(args.prepared.artifacts.candidateEvalJson)) {
    throw new Error("score stage requires candidate-eval.json. Run --stage candidate first.");
  }
  await verifyReusableArtifacts(args.prepared.artifacts, [
    args.prepared.artifacts.baselineEvalJson,
    stageFingerprintPath(args.prepared, "baseline"),
    args.prepared.artifacts.candidateEvalJson,
    stageFingerprintPath(args.prepared, "candidate"),
  ]);
  await throwIfCancelled(args.store, args.prepared.request);
  await args.store.invalidateRunOutputs(args.prepared.request.run_id, { report: true });
  await cleanupStageArtifacts(args.prepared.artifacts, "score");
  await updateRun({
    store: args.store,
    reporter: args.reporter,
    request: args.prepared.request,
    status: "scoring",
    stage: "scoring",
    message: "Rescoring existing baseline and candidate outputs.",
    details: { scoring_mode: args.config.evaluation.scoring.mode },
  });
  const rollbackPaths = [
    args.prepared.artifacts.baselineEvalJson,
    stageFingerprintPath(args.prepared, "baseline"),
    args.prepared.artifacts.candidateEvalJson,
    stageFingerprintPath(args.prepared, "candidate"),
  ];
  const originals = new Map<string, Buffer>();
  for (const path of rollbackPaths) originals.set(path, await readFile(path));
  try {
    const baseline = await rescoreEvalReport({
      report: evalReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.baselineEvalJson)),
      config: args.config,
      outputPath: args.prepared.artifacts.baselineEvalJson,
      system: args.prepared.system,
      reporter: args.runReporter,
    });
    const candidate = await rescoreEvalReport({
      report: evalReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.candidateEvalJson)),
      config: args.config,
      outputPath: args.prepared.artifacts.candidateEvalJson,
      system: args.prepared.system,
      reporter: args.runReporter,
    });
    await throwIfCancelled(args.store, args.prepared.request);
    await writeStageFingerprint({ stage: "baseline", prepared: args.prepared, config: args.config });
    const training = await pathExists(args.prepared.artifacts.trainingReportJson)
      ? trainingReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.trainingReportJson))
      : undefined;
    await writeStageFingerprint({
      stage: "candidate",
      prepared: args.prepared,
      config: args.config,
      training,
    });
    return { baseline, candidate };
  } catch (error) {
    await Promise.all([...originals].map(([path, contents]) => writeFileAtomic(path, contents)));
    throw error;
  }
}

async function runReportStage(args: {
  prepared: PreparedRun;
  config: LocalRunnerConfig;
  store: LocalStore;
  reporter?: LocalRunReporter;
  startedAt: string;
  startedPerf: number;
  emitReportingEvent?: boolean;
}): Promise<RunReport> {
  if (!await pathExists(args.prepared.artifacts.baselineEvalJson)) {
    throw new Error("report stage requires baseline-eval.json. Run --stage baseline first.");
  }
  if (!await pathExists(args.prepared.artifacts.candidateEvalJson)) {
    throw new Error("report stage requires candidate-eval.json. Run --stage candidate first.");
  }
  if (!await pathExists(args.prepared.artifacts.trainingReportJson)) {
    throw new Error("report stage requires training-report.json. Run --stage train first, or --stage candidate --model-artifact <path>.");
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
  });
  const currentCandidate = await canReuseStageArtifact({
    stage: "candidate",
    prepared: args.prepared,
    config: args.config,
    verifyModel: true,
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
  ], { verifyModel: true });
  await throwIfCancelled(args.store, args.prepared.request);
  await args.store.invalidateRunOutputs(args.prepared.request.run_id, { report: true });
  await cleanupStageArtifacts(args.prepared.artifacts, "report");
  if (args.emitReportingEvent) {
    await updateRun({
      store: args.store,
      reporter: args.reporter,
      request: args.prepared.request,
      status: "reporting",
      stage: "reporting",
      message: "Writing run report.",
      details: { report_path: args.prepared.artifacts.runReportJson },
    });
  }
  const baseline = evalReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.baselineEvalJson));
  const candidate = evalReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.candidateEvalJson));
  const training = trainingReportSchema.parse(await readJson<unknown>(args.prepared.artifacts.trainingReportJson));
  const comparison = compareEvalReports(baseline, candidate);
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
    training,
    artifact_uris: {
      dataset: fileUri(args.prepared.artifacts.trainingJsonl),
      baseline_eval: fileUri(args.prepared.artifacts.baselineEvalJson),
      candidate_eval: fileUri(args.prepared.artifacts.candidateEvalJson),
      report: fileUri(args.prepared.artifacts.runReportJson),
    },
    run_metadata: {
      base_model: args.prepared.request.spec_snapshot.base_model,
      fine_tuned_model_id: training.model_artifact_uri ?? training.training_job_name,
      parent_model_artifact: args.prepared.metadata.parent_model_artifact,
      training_method: args.prepared.request.training_method,
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
      elapsed_seconds: duration.seconds,
    },
  });
  return report;
}

export async function runLocalFineTuneStage(input: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  reporter?: LocalRunReporter;
  stage?: LocalRunStage;
  force?: boolean;
  modelArtifact?: string;
}): Promise<LocalStageRunResult> {
  const startedPerf = performance.now();
  const startedAt = new Date().toISOString();
  const stage = input.stage ?? "all";
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
      prepared = await ensurePrepared({
      request: input.request,
      config: input.config,
      artifacts,
      store,
      reporter: input.reporter,
      forcePrepare: Boolean(input.force && (stage === "prepare" || stage === "all")),
    });
    await throwIfCancelled(store, input.request);
    let report: RunReport | undefined;

    const completeRequestedStage = async (completedStage: Exclude<LocalRunStage, "all" | "report">) => {
      await throwIfCancelled(store, input.request);
      const message = `${completedStage} stage completed.`;
      const completedState = await store.updateRun({
        runId: input.request.run_id,
        status: "stage_completed",
        stage: `${completedStage}_completed`,
        message,
        details: { requested_stage: completedStage },
      });
      if (completedState.status === "cancelled") {
        throw new ProcessCancelledError(`Run ${input.request.run_id} was cancelled.`);
      }
      await input.reporter?.onEvent?.({
        stage: `${completedStage}_completed`,
        status: "completed",
        message,
        details: { requested_stage: completedStage },
      });
    };

    if (stage === "prepare") {
      await completeRequestedStage(stage);
      return stageResult({ request: input.request, stage, artifacts });
    }

    if (stage === "baseline" || stage === "all") {
      await runBaselineStage({
        prepared,
        config: input.config,
        store,
        reporter: input.reporter,
        runReporter,
        force: Boolean(input.force),
      });
      await refreshArtifactManifest(prepared);
      if (stage === "baseline") {
        await completeRequestedStage(stage);
        return stageResult({ request: input.request, stage, artifacts });
      }
    }

    if (stage === "train" || stage === "all") {
      await runTrainStage({
        prepared,
        config: input.config,
        store,
        reporter: input.reporter,
        runReporter,
        force: Boolean(input.force),
      });
      if (stage === "train") {
        await completeRequestedStage(stage);
        return stageResult({ request: input.request, stage, artifacts });
      }
    }

    if (stage === "candidate" || stage === "all") {
      await runCandidateStage({
        prepared,
        config: input.config,
        store,
        reporter: input.reporter,
        runReporter,
        force: Boolean(input.force),
        modelArtifact: input.modelArtifact,
      });
      await refreshArtifactManifest(prepared);
      if (stage === "candidate") {
        await completeRequestedStage(stage);
        return stageResult({ request: input.request, stage, artifacts });
      }
    }

    if (stage === "score") {
      await runScoreStage({
        prepared,
        config: input.config,
        store,
        reporter: input.reporter,
        runReporter,
      });
      await refreshArtifactManifest(prepared);
      await completeRequestedStage(stage);
      return stageResult({ request: input.request, stage, artifacts });
    }

    if (stage === "report" || stage === "all") {
      report = await runReportStage({
        prepared,
        config: input.config,
        store,
        reporter: input.reporter,
        startedAt,
        startedPerf,
        emitReportingEvent: stage === "report",
      });
      return stageResult({ request: input.request, stage, artifacts, report });
    }

    throw new Error(`Unknown run stage: ${stage}`);
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

function stageResult(args: {
  request: FineTuneRunRequest;
  stage: LocalRunStage;
  artifacts: RunArtifacts;
  report?: RunReport;
}): LocalStageRunResult {
  return {
    request: args.request,
    stage: args.stage,
    report: args.report,
    reportPath: args.report ? args.artifacts.runReportJson : undefined,
    artifactDir: dirname(args.artifacts.runReportJson),
    artifacts: {
      training_jsonl: args.artifacts.trainingJsonl,
      stage_metadata: args.artifacts.stageMetadataJson,
      training_report: args.artifacts.trainingReportJson,
      baseline_eval: args.artifacts.baselineEvalJson,
      candidate_eval: args.artifacts.candidateEvalJson,
      report: args.artifacts.runReportJson,
      artifact_manifest: args.artifacts.artifactManifestJson,
    },
  };
}

export async function runLocalFineTune(input: {
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
  reporter?: LocalRunReporter;
}): Promise<LocalRunResult> {
  const result = await runLocalFineTuneStage({
    request: input.request,
    config: input.config,
    reporter: input.reporter,
    stage: "all",
  });
  if (!result.report || !result.reportPath) {
    throw new Error("Full local fine-tune run did not produce a report.");
  }
  return {
    request: result.request,
    report: result.report,
    reportPath: result.reportPath,
    artifactDir: result.artifactDir,
  };
}
