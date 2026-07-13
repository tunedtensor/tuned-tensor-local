import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { copyFile, lstat, mkdir, open, readFile, readdir, realpath, rename, stat, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { createGunzip } from "node:zlib";

export interface RunArtifacts {
  root: string;
  prefix: string;
  runDir: string;
  trainingJsonl: string;
  stageMetadataJson: string;
  trainingReportJson: string;
  baselineEvalJson: string;
  candidateEvalJson: string;
  runReportJson: string;
  artifactManifestJson: string;
  progressJsonl: string;
  trainingDir: string;
  trainingInputDir: string;
  trainingConfigDir: string;
  trainingOutputDir: string;
  trainingModelDir: string;
  trainingLog: string;
}

const RUN_OWNER_FILE = ".tt-local-run-owner.json";
export const ARTIFACT_WORKFLOW_LOCK_FILE = ".tt-local-workflow.lock";

export interface ArtifactRunOwner {
  schema_version: 1;
  run_id: string;
  user_id: string;
  behavior_spec_id: string;
  created_at: string;
}

export function fileUri(path: string): string {
  return `file://${resolve(path)}`;
}

export function defaultArtifactPrefix(input: {
  userId: string;
  behaviorSpecId: string;
  runId: string;
}): string {
  return join("users", input.userId, "specs", input.behaviorSpecId, "runs", input.runId);
}

export function resolveRunArtifacts(args: {
  artifactRoot: string;
  prefix: string;
}): RunArtifacts {
  const root = resolve(args.artifactRoot);
  const safePrefix = args.prefix.replace(/^[/\\]+/, "");
  const runDir = resolve(root, safePrefix);
  const containment = relative(root, runDir);
  if (containment === ".." || containment.startsWith(`..${sep}`) || isAbsolute(containment)) {
    throw new Error(`Artifact prefix escapes artifactRoot: ${args.prefix}`);
  }
  const trainingDir = join(runDir, "training");
  return {
    root,
    prefix: safePrefix,
    runDir,
    trainingJsonl: join(runDir, "training.jsonl"),
    stageMetadataJson: join(runDir, "stage-metadata.json"),
    trainingReportJson: join(runDir, "training-report.json"),
    baselineEvalJson: join(runDir, "baseline-eval.json"),
    candidateEvalJson: join(runDir, "candidate-eval.json"),
    runReportJson: join(runDir, "run-report.json"),
    artifactManifestJson: join(runDir, "artifact-manifest.json"),
    progressJsonl: join(runDir, "progress.jsonl"),
    trainingDir,
    trainingInputDir: join(trainingDir, "input", "data", "training"),
    trainingConfigDir: join(trainingDir, "input", "config"),
    trainingOutputDir: join(trainingDir, "output"),
    trainingModelDir: join(trainingDir, "model"),
    trainingLog: join(trainingDir, "training.log"),
  };
}

/**
 * Creates the run directory one component at a time and rejects symlinked
 * descendants. Lexical `resolve()` containment alone is insufficient because
 * an existing child symlink could redirect later writes or recursive cleanup
 * outside `artifactRoot`.
 */
export async function ensureSafeRunDirectory(artifacts: RunArtifacts): Promise<void> {
  await mkdir(artifacts.root, { recursive: true });
  const canonicalRoot = await realpath(artifacts.root);
  const pathFromRoot = relative(artifacts.root, artifacts.runDir);
  const segments = pathFromRoot ? pathFromRoot.split(sep).filter(Boolean) : [];
  let current = artifacts.root;
  for (const segment of segments) {
    current = join(current, segment);
    let metadata = await lstat(current).catch((error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") return null;
      throw error;
    });
    if (!metadata) {
      await mkdir(current).catch((error: NodeJS.ErrnoException) => {
        if (error.code !== "EEXIST") throw error;
      });
      metadata = await lstat(current);
    }
    if (metadata.isSymbolicLink()) {
      throw new Error(`Artifact path must not contain symbolic links: ${current}`);
    }
    if (!metadata.isDirectory()) {
      throw new Error(`Artifact path component is not a directory: ${current}`);
    }
  }
  const canonicalRunDir = await realpath(artifacts.runDir);
  const containment = relative(canonicalRoot, canonicalRunDir);
  if (containment === ".." || containment.startsWith(`..${sep}`) || isAbsolute(containment)) {
    throw new Error(`Artifact run directory escapes artifactRoot: ${artifacts.runDir}`);
  }
}

/**
 * Durably binds an artifact directory to one run identity. The owner survives
 * workflow-lock release, preventing a later run with a colliding custom
 * prefix from deleting or silently reusing another run's artifacts.
 */
export async function claimRunArtifactDirectory(args: {
  artifacts: RunArtifacts;
  runId: string;
  userId: string;
  behaviorSpecId: string;
}): Promise<ArtifactRunOwner> {
  await ensureSafeRunDirectory(args.artifacts);
  const ownerPath = join(args.artifacts.runDir, RUN_OWNER_FILE);
  const proposed: ArtifactRunOwner = {
    schema_version: 1,
    run_id: args.runId,
    user_id: args.userId,
    behavior_spec_id: args.behaviorSpecId,
    created_at: new Date().toISOString(),
  };
  try {
    const handle = await open(ownerPath, "wx", 0o600);
    try {
      await handle.writeFile(`${JSON.stringify(proposed, null, 2)}\n`);
      await handle.sync();
    } finally {
      await handle.close();
    }
    return proposed;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
  }
  let existing: ArtifactRunOwner;
  try {
    existing = JSON.parse(await readFile(ownerPath, "utf8")) as ArtifactRunOwner;
  } catch (error) {
    throw new Error(`Artifact run owner claim is unreadable: ${ownerPath}`, { cause: error });
  }
  if (
    existing.schema_version !== 1
    || existing.run_id !== args.runId
    || existing.user_id !== args.userId
    || existing.behavior_spec_id !== args.behaviorSpecId
  ) {
    throw new Error(
      `Artifact directory ${args.artifacts.runDir} is already owned by run ${existing.run_id ?? "unknown"}; `
      + `run ${args.runId} must use a different artifacts.prefix.`,
    );
  }
  return existing;
}

export async function prepareRunDirectories(artifacts: RunArtifacts): Promise<void> {
  await ensureSafeRunDirectory(artifacts);
  await Promise.all([
    mkdir(artifacts.runDir, { recursive: true }),
    mkdir(artifacts.trainingInputDir, { recursive: true }),
    mkdir(artifacts.trainingConfigDir, { recursive: true }),
    mkdir(artifacts.trainingOutputDir, { recursive: true }),
    mkdir(artifacts.trainingModelDir, { recursive: true }),
  ]);
}

export async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

export async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await writeFileAtomic(path, `${JSON.stringify(value, null, 2)}\n`);
}

export async function writeFileAtomic(path: string, value: string | Uint8Array): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.${Date.now()}.${randomUUID()}.tmp`;
  await writeFile(temporary, value);
  await rename(temporary, path);
}

export async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await readFile(path, "utf8")) as T;
}

export async function appendProgress(artifacts: RunArtifacts, event: {
  stage: string;
  message: string;
  details?: Record<string, unknown>;
}): Promise<void> {
  await mkdir(artifacts.runDir, { recursive: true });
  await writeFile(
    artifacts.progressJsonl,
    `${JSON.stringify({ at: new Date().toISOString(), ...event })}\n`,
    { encoding: "utf8", flag: "a" },
  );
}

export async function copyDatasetToTrainingChannel(artifacts: RunArtifacts): Promise<string> {
  const destination = join(artifacts.trainingInputDir, "training.jsonl");
  await copyFile(artifacts.trainingJsonl, destination);
  return destination;
}

export interface ArtifactManifestFile {
  path: string;
  size_bytes: number;
  sha256: string;
}

export interface ArtifactManifest {
  schema_version: 1;
  generated_at: string;
  files: ArtifactManifestFile[];
  model?: ArtifactManifestModel;
}

export interface ArtifactManifestModel {
  artifact_kind: "file" | "directory";
  format: string;
  framework: string;
  base_model: string;
  base_model_revision?: string;
  base_model_artifact_uri?: string;
  base_model_fingerprint?: string;
  parent_model_artifact?: string;
  artifact_uri: string;
  artifact_root: string;
  servable: boolean;
  files: ArtifactManifestFile[];
}

export interface ArtifactManifestVerification {
  valid: boolean;
  checked: number;
  missing: string[];
  changed: string[];
  unexpected: string[];
}

export interface TarGzipVerification {
  entries: number;
  payload_entries: number;
  recognized_payload_entries: number;
  recognized_payload_bytes: number;
  adapter_weight_entries: number;
  adapter_weight_bytes: number;
  full_model_weight_entries: number;
  full_model_weight_bytes: number;
  adapter_config_entries: number;
  expanded_bytes: number;
}

const MANIFEST_IGNORED_PATHS = new Set([
  "artifact-manifest.json",
  // These are append-only/control-plane files rather than immutable stage
  // outputs. Including them would invalidate a manifest whenever the store
  // records the next stage event or refreshes the canonical request copy.
  "progress.jsonl",
  "request.json",
  "detached.log",
  RUN_OWNER_FILE,
  ARTIFACT_WORKFLOW_LOCK_FILE,
]);

function portableRelative(root: string, path: string): string {
  return relative(root, path).split(sep).join("/");
}

function shouldManifest(path: string): boolean {
  const atomicTemporary = /\.\d+\.\d+\.[0-9a-f-]{36}\.tmp$/i.test(path);
  return !MANIFEST_IGNORED_PATHS.has(path) && !atomicTemporary;
}

async function listArtifactFiles(root: string, current = root): Promise<string[]> {
  const files: string[] = [];
  for (const entry of await readdir(current, { withFileTypes: true }).catch(() => [])) {
    const absolute = join(current, entry.name);
    if (entry.isSymbolicLink()) {
      throw new Error(`Artifact tree must not contain symbolic links: ${absolute}`);
    }
    if (entry.isDirectory()) {
      files.push(...await listArtifactFiles(root, absolute));
      continue;
    }
    if (!entry.isFile()) continue;
    const relativePath = portableRelative(root, absolute);
    if (!shouldManifest(relativePath)) continue;
    const metadata = await stat(absolute).catch(() => null);
    if (metadata?.isFile()) files.push(relativePath);
  }
  return files.sort();
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

async function describeArtifactFile(root: string, path: string): Promise<ArtifactManifestFile> {
  const absolute = join(root, path);
  const metadata = await stat(absolute);
  return {
    path,
    size_bytes: metadata.size,
    sha256: await sha256File(absolute),
  };
}

function assertManifestFiles(value: unknown, label: string): asserts value is ArtifactManifestFile[] {
  if (!Array.isArray(value)) throw new Error(`Artifact manifest ${label} must be an array.`);
  const paths = new Set<string>();
  for (const file of value) {
    if (
      !file
      || typeof file.path !== "string"
      || !Number.isSafeInteger(file.size_bytes)
      || file.size_bytes < 0
      || typeof file.sha256 !== "string"
      || !/^[a-f0-9]{64}$/.test(file.sha256)
    ) {
      throw new Error(`Artifact manifest contains an invalid ${label} entry.`);
    }
    const manifestPath = file.path as string;
    const segments: string[] = manifestPath.split("/");
    if (
      !manifestPath
      || manifestPath.includes("\\")
      || manifestPath.includes("\0")
      || manifestPath.startsWith("/")
      || /^[a-z]:/i.test(manifestPath)
      || segments.some((segment) => !segment || segment === "." || segment === "..")
    ) {
      throw new Error(`Artifact manifest contains an unsafe ${label} path: ${manifestPath}`);
    }
    if (paths.has(manifestPath)) throw new Error(`Artifact manifest contains a duplicate ${label} path: ${manifestPath}`);
    paths.add(manifestPath);
  }
}

function parseArtifactManifest(value: unknown): ArtifactManifest {
  if (!value || typeof value !== "object") throw new Error("Artifact manifest must be an object.");
  const candidate = value as Partial<ArtifactManifest>;
  if (candidate.schema_version !== 1 || !Array.isArray(candidate.files)) {
    throw new Error("Unsupported or invalid artifact manifest.");
  }
  assertManifestFiles(candidate.files, "file");
  if (candidate.model) {
    const model = candidate.model;
    const hasBaseArtifact = model.base_model_artifact_uri !== undefined;
    const hasBaseFingerprint = model.base_model_fingerprint !== undefined;
    if (
      (model.artifact_kind !== "file" && model.artifact_kind !== "directory")
      || typeof model.format !== "string"
      || typeof model.framework !== "string"
      || typeof model.base_model !== "string"
      || (model.base_model_revision !== undefined && typeof model.base_model_revision !== "string")
      || (model.base_model_artifact_uri !== undefined && typeof model.base_model_artifact_uri !== "string")
      || (model.base_model_fingerprint !== undefined && typeof model.base_model_fingerprint !== "string")
      || hasBaseArtifact !== hasBaseFingerprint
      || typeof model.artifact_uri !== "string"
      || typeof model.artifact_root !== "string"
      || typeof model.servable !== "boolean"
      || (model.parent_model_artifact !== undefined && typeof model.parent_model_artifact !== "string")
      || !Array.isArray(model.files)
      || model.files.length === 0
    ) {
      throw new Error("Artifact manifest contains invalid model contract metadata.");
    }
    assertManifestFiles(model.files, "model file");
  }
  return candidate as ArtifactManifest;
}

/** Atomically snapshots every immutable file currently present in a run directory. */
export async function writeArtifactManifest(
  artifacts: RunArtifacts,
  options: { model?: Omit<ArtifactManifestModel, "files"> } = {},
): Promise<ArtifactManifest> {
  const paths = await listArtifactFiles(artifacts.runDir);
  let model: ArtifactManifestModel | undefined;
  if (options.model) {
    if (
      (options.model.base_model_artifact_uri === undefined)
      !== (options.model.base_model_fingerprint === undefined)
    ) {
      throw new Error("Model contracts must record a local base-model URI and fingerprint together.");
    }
    const modelRoot = resolve(options.model.artifact_root);
    const modelPaths = options.model.artifact_kind === "file"
      ? [portableRelative(dirname(modelRoot), modelRoot)]
      : await listArtifactFiles(modelRoot);
    const checksumRoot = options.model.artifact_kind === "file" ? dirname(modelRoot) : modelRoot;
    if (
      options.model.artifact_kind === "file"
      && (options.model.format === "tar.gz" || modelRoot.endsWith(".tar.gz"))
    ) {
      const archive = await verifyTarGzipArchive(modelRoot);
      if (
        (options.model.framework === "transformers-peft" || options.model.servable)
        && archive.adapter_weight_entries === 0
      ) {
        throw new Error(`PEFT model archive contains no adapter_model.safetensors or adapter_model.bin: ${modelRoot}`);
      }
      if (
        (options.model.framework === "transformers-peft" || options.model.servable)
        && archive.adapter_config_entries === 0
      ) {
        throw new Error(`PEFT model archive contains no adapter_config.json: ${modelRoot}`);
      }
      if (options.model.framework === "transformers-full" && archive.full_model_weight_entries === 0) {
        throw new Error(`Full-model archive contains no Transformers model weights: ${modelRoot}`);
      }
    }
    model = {
      ...options.model,
      artifact_root: modelRoot,
      files: await Promise.all(modelPaths.map((path) => describeArtifactFile(checksumRoot, path))),
    };
  }
  const manifest: ArtifactManifest = {
    schema_version: 1,
    generated_at: new Date().toISOString(),
    files: await Promise.all(paths.map((path) => describeArtifactFile(artifacts.runDir, path))),
    ...(model ? { model } : {}),
  };
  await writeJsonAtomic(artifacts.artifactManifestJson, manifest);
  return manifest;
}

/** Re-hashes a persisted run and reports missing, changed, and untracked files. */
export async function verifyArtifactManifest(
  manifestPath: string,
  options: {
    requiredPaths?: string[];
    allowUnexpected?: boolean;
    scopeToRequired?: boolean;
    verifyModel?: boolean;
  } = {},
): Promise<ArtifactManifestVerification> {
  const root = dirname(resolve(manifestPath));
  const manifest = parseArtifactManifest(JSON.parse(await readFile(manifestPath, "utf8")) as unknown);
  const expected = new Map(manifest.files.map((file) => [file.path, file]));
  const actualPaths = await listArtifactFiles(root);
  const actual = new Set(actualPaths);
  const missing: string[] = [];
  const changed: string[] = [];

  const normalizedRequired = (options.requiredPaths ?? [])
    .map((path) => path.split(sep).join("/").replace(/^\.\//, ""));
  const checkedEntries = options.scopeToRequired
    ? normalizedRequired.flatMap((path) => {
        const entry = expected.get(path);
        return entry ? [[path, entry] as const] : [];
      })
    : [...expected.entries()];
  for (const [path, expectedFile] of checkedEntries) {
    if (!actual.has(path)) {
      missing.push(path);
      continue;
    }
    const actualFile = await describeArtifactFile(root, path);
    if (actualFile.size_bytes !== expectedFile.size_bytes || actualFile.sha256 !== expectedFile.sha256) {
      changed.push(path);
    }
  }
  for (const normalized of normalizedRequired) {
    if (!expected.has(normalized) && !missing.includes(normalized)) missing.push(normalized);
  }
  const unexpected = options.allowUnexpected || options.scopeToRequired
    ? []
    : actualPaths.filter((path) => !expected.has(path));
  let modelChecked = 0;
  if (manifest.model && (options.verifyModel ?? !options.scopeToRequired)) {
    const modelRoot = resolve(manifest.model.artifact_root);
    const checksumRoot = manifest.model.artifact_kind === "file" ? dirname(modelRoot) : modelRoot;
    const modelActualPaths = manifest.model.artifact_kind === "file"
      ? [portableRelative(checksumRoot, modelRoot)]
      : await listArtifactFiles(modelRoot);
    const modelActual = new Set(modelActualPaths);
    for (const expectedFile of manifest.model.files) {
      const label = `model:${expectedFile.path}`;
      if (!modelActual.has(expectedFile.path)) {
        missing.push(label);
        continue;
      }
      const actualFile = await describeArtifactFile(checksumRoot, expectedFile.path);
      modelChecked += 1;
      if (actualFile.size_bytes !== expectedFile.size_bytes || actualFile.sha256 !== expectedFile.sha256) {
        changed.push(label);
      }
    }
    if (!options.allowUnexpected) {
      const expectedModelPaths = new Set(manifest.model.files.map((file) => file.path));
      unexpected.push(...modelActualPaths.filter((path) => !expectedModelPaths.has(path)).map((path) => `model:${path}`));
    }
    if (
      manifest.model.artifact_kind === "file"
      && (manifest.model.format === "tar.gz" || modelRoot.endsWith(".tar.gz"))
    ) {
      try {
        const archive = await verifyTarGzipArchive(modelRoot);
        if (
          (manifest.model.framework === "transformers-peft" || manifest.model.servable)
          && archive.adapter_weight_entries === 0
        ) {
          changed.push(`model:${portableRelative(checksumRoot, modelRoot)}:missing_adapter_weights`);
        }
        if (
          (manifest.model.framework === "transformers-peft" || manifest.model.servable)
          && archive.adapter_config_entries === 0
        ) {
          changed.push(`model:${portableRelative(checksumRoot, modelRoot)}:missing_adapter_config`);
        }
        if (manifest.model.framework === "transformers-full" && archive.full_model_weight_entries === 0) {
          changed.push(`model:${portableRelative(checksumRoot, modelRoot)}:missing_full_model_weights`);
        }
      } catch {
        changed.push(`model:${portableRelative(checksumRoot, modelRoot)}:invalid_archive`);
      }
    }
  }
  return {
    valid: missing.length === 0 && changed.length === 0 && unexpected.length === 0,
    checked: checkedEntries.length + modelChecked,
    missing: [...new Set(missing)].sort(),
    changed: changed.sort(),
    unexpected: unexpected.sort(),
  };
}

/** Validates gzip CRC/truncation, tar headers, paths, and bounded PAX metadata. */
export async function verifyTarGzipArchive(path: string): Promise<TarGzipVerification> {
  const maxEntries = 100_000;
  const maxExpandedBytes = 20 * 1024 * 1024 * 1024;
  const maxPaxBytes = 1024 * 1024;
  const safeMemberPath = (value: string): string => {
    const normalized = value.replaceAll("\\", "/");
    const segments = normalized.split("/").filter(Boolean);
    if (
      !normalized
      || normalized.startsWith("/")
      || /^[a-z]:\//i.test(normalized)
      || segments.includes("..")
    ) {
      throw new Error(`Unsafe tar member path ${JSON.stringify(normalized)} in ${path}.`);
    }
    return normalized;
  };
  const parsePax = (payload: Buffer, global: boolean): string | undefined => {
    let offset = 0;
    let memberPath: string | undefined;
    const allowedMetadata = new Set(["mtime", "atime", "ctime", "uid", "gid", "uname", "gname"]);
    while (offset < payload.length) {
      const separator = payload.indexOf(0x20, offset);
      if (separator < 0) throw new Error(`Malformed PAX record in ${path}.`);
      const lengthText = payload.subarray(offset, separator).toString("ascii");
      if (!/^[1-9][0-9]*$/.test(lengthText)) throw new Error(`Malformed PAX record length in ${path}.`);
      const length = Number.parseInt(lengthText, 10);
      const end = offset + length;
      if (!Number.isSafeInteger(length) || end > payload.length || payload[end - 1] !== 0x0a) {
        throw new Error(`Truncated PAX record in ${path}.`);
      }
      const record = payload.subarray(separator + 1, end - 1).toString("utf8");
      const equals = record.indexOf("=");
      if (equals <= 0) throw new Error(`Malformed PAX key/value in ${path}.`);
      const key = record.slice(0, equals);
      const value = record.slice(equals + 1);
      if (key === "path") {
        if (global) throw new Error(`Global PAX path overrides are unsupported in ${path}.`);
        memberPath = safeMemberPath(value);
      } else if (!allowedMetadata.has(key)) {
        // In particular, reject linkpath and size overrides: the validator
        // must never interpret a member differently from a downstream reader.
        throw new Error(`Unsupported PAX key ${JSON.stringify(key)} in ${path}.`);
      }
      offset = end;
    }
    return memberPath;
  };
  const gunzip = createReadStream(path).pipe(createGunzip());
  let buffered = Buffer.alloc(0);
  let payloadBytes = 0;
  let paddingBytes = 0;
  let paxType: "x" | "g" | null = null;
  let paxChunks: Buffer[] = [];
  let pendingPaxPath: string | undefined;
  let entries = 0;
  let payloadEntries = 0;
  let recognizedPayloadEntries = 0;
  let recognizedPayloadBytes = 0;
  let adapterWeightEntries = 0;
  let adapterWeightBytes = 0;
  let fullModelWeightEntries = 0;
  let fullModelWeightBytes = 0;
  let adapterConfigEntries = 0;
  let expandedBytes = 0;
  let zeroBlocks = 0;
  let ended = false;
  for await (const chunk of gunzip) {
    buffered = Buffer.concat([buffered, Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)]);
    while (true) {
      if (payloadBytes > 0) {
        if (buffered.length === 0) break;
        const consumed = Math.min(payloadBytes, buffered.length);
        if (paxType) paxChunks.push(buffered.subarray(0, consumed));
        buffered = buffered.subarray(consumed);
        payloadBytes -= consumed;
        continue;
      }
      if (paddingBytes > 0) {
        if (buffered.length === 0) break;
        const consumed = Math.min(paddingBytes, buffered.length);
        buffered = buffered.subarray(consumed);
        paddingBytes -= consumed;
        continue;
      }
      if (paxType) {
        const paxPath = parsePax(Buffer.concat(paxChunks), paxType === "g");
        if (paxType === "x") pendingPaxPath = paxPath;
        paxType = null;
        paxChunks = [];
        continue;
      }
      if (buffered.length < 512) break;
      const header = buffered.subarray(0, 512);
      buffered = buffered.subarray(512);
      if (header.every((byte) => byte === 0)) {
        zeroBlocks += 1;
        if (zeroBlocks >= 2) ended = true;
        continue;
      }
      if (ended) throw new Error(`Tar archive contains data after its end marker: ${path}.`);
      zeroBlocks = 0;
      const checksumText = header.subarray(148, 156).toString("ascii").replace(/\0/g, "").trim();
      if (!/^[0-7]+$/.test(checksumText)) throw new Error(`Invalid tar checksum field in ${path}.`);
      const expectedChecksum = Number.parseInt(checksumText, 8);
      let actualChecksum = 0;
      for (let index = 0; index < header.length; index += 1) {
        actualChecksum += index >= 148 && index < 156 ? 32 : header[index]!;
      }
      if (actualChecksum !== expectedChecksum) throw new Error(`Tar member checksum mismatch in ${path}.`);
      const sizeText = header.subarray(124, 136).toString("ascii").replace(/\0/g, "").trim();
      if (sizeText && !/^[0-7]+$/.test(sizeText)) throw new Error(`Invalid tar member size in ${path}.`);
      const size = sizeText ? Number.parseInt(sizeText, 8) : 0;
      if (!Number.isSafeInteger(size) || size < 0) throw new Error(`Invalid tar member size in ${path}.`);
      const type = String.fromCharCode(header[156] ?? 0);
      if (type !== "\0" && type !== "0" && type !== "5" && type !== "x" && type !== "g") {
        throw new Error(`Unsafe or unsupported tar member type ${JSON.stringify(type)} in ${path}.`);
      }
      const shortName = header.subarray(0, 100).toString("utf8").replace(/\0.*$/, "");
      const prefix = header.subarray(345, 500).toString("utf8").replace(/\0.*$/, "");
      const headerPath = safeMemberPath(prefix ? `${prefix}/${shortName}` : shortName);
      const memberPath = type === "x" || type === "g"
        ? headerPath
        : safeMemberPath(pendingPaxPath ?? headerPath);
      if (type !== "x" && type !== "g") pendingPaxPath = undefined;
      const segments = memberPath.split("/").filter(Boolean);
      const name = segments.at(-1)?.toLowerCase();
      const metadataOnly = name === "training-metrics.json"
        || name === "trainer_state.json"
        || name === "training_args.bin"
        || name === "config.json"
        || name === "generation_config.json"
        || name === "adapter_config.json"
        || name === "readme.md"
        || name?.startsWith("tokenizer")
        || name?.startsWith("special_tokens")
        || name?.startsWith("added_tokens")
        || name === "vocab.json"
        || name === "merges.txt";
      const recognizedPayload = Boolean(name)
        && name !== "training_args.bin"
        && (
          name!.endsWith(".safetensors")
          || name!.endsWith(".bin")
          || name!.endsWith(".pt")
          || name!.endsWith(".pth")
          || name!.endsWith(".gguf")
          || name!.endsWith(".onnx")
          || name!.endsWith(".ckpt")
          || name!.endsWith(".npz")
        );
      const adapterWeight = name === "adapter_model.safetensors" || name === "adapter_model.bin";
      const fullModelWeight = Boolean(name)
        && ((name!.endsWith(".safetensors") && !name!.startsWith("adapter_"))
          || /^pytorch_model.*\.bin$/.test(name!));
      if (size > 0 && name === "adapter_config.json" && (type === "\0" || type === "0")) {
        adapterConfigEntries += 1;
      }
      if ((type === "x" || type === "g") && size > maxPaxBytes) {
        throw new Error(`PAX metadata exceeds 1 MiB in ${path}.`);
      }
      if (size > 0 && !metadataOnly && (type === "\0" || type === "0")) payloadEntries += 1;
      if (size > 0 && recognizedPayload && (type === "\0" || type === "0")) {
        recognizedPayloadEntries += 1;
        recognizedPayloadBytes += size;
      }
      if (size > 0 && adapterWeight && (type === "\0" || type === "0")) {
        adapterWeightEntries += 1;
        adapterWeightBytes += size;
      }
      if (size > 0 && fullModelWeight && (type === "\0" || type === "0")) {
        fullModelWeightEntries += 1;
        fullModelWeightBytes += size;
      }
      expandedBytes += size;
      if (expandedBytes > maxExpandedBytes) throw new Error(`Tar archive exceeds 20 GiB expanded size: ${path}.`);
      payloadBytes = size;
      paddingBytes = (512 - (size % 512)) % 512;
      paxType = type === "x" || type === "g" ? type : null;
      paxChunks = [];
      entries += 1;
      if (entries > maxEntries) throw new Error(`Tar archive exceeds ${maxEntries} members: ${path}.`);
    }
  }
  if (payloadBytes !== 0 || paddingBytes !== 0 || paxType || buffered.some((byte) => byte !== 0)) {
    throw new Error(`Truncated tar archive: ${path}.`);
  }
  if (pendingPaxPath) throw new Error(`PAX path override has no following member in ${path}.`);
  if (!ended) throw new Error(`Tar archive is missing its end marker: ${path}.`);
  if (entries === 0) throw new Error(`Empty tar archive: ${path}.`);
  if (payloadEntries === 0) throw new Error(`Tar archive contains no model payload files: ${path}.`);
  return {
    entries,
    payload_entries: payloadEntries,
    recognized_payload_entries: recognizedPayloadEntries,
    recognized_payload_bytes: recognizedPayloadBytes,
    adapter_weight_entries: adapterWeightEntries,
    adapter_weight_bytes: adapterWeightBytes,
    full_model_weight_entries: fullModelWeightEntries,
    full_model_weight_bytes: fullModelWeightBytes,
    adapter_config_entries: adapterConfigEntries,
    expanded_bytes: expandedBytes,
  };
}

export async function assertArtifactManifest(
  manifestPath: string,
  options: {
    requiredPaths?: string[];
    allowUnexpected?: boolean;
    scopeToRequired?: boolean;
    verifyModel?: boolean;
  } = {},
): Promise<ArtifactManifestVerification> {
  const result = await verifyArtifactManifest(manifestPath, options);
  if (!result.valid) {
    const details = [
      result.missing.length ? `missing=${result.missing.join(",")}` : "",
      result.changed.length ? `changed=${result.changed.join(",")}` : "",
      result.unexpected.length ? `unexpected=${result.unexpected.join(",")}` : "",
    ].filter(Boolean).join(" ");
    throw new Error(`Artifact integrity verification failed${details ? `: ${details}` : "."}`);
  }
  return result;
}
