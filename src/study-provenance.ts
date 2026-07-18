import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  lstat,
  mkdir,
  open,
  readdir,
  writeFile,
} from "node:fs/promises";
import { dirname, join, relative, sep } from "node:path";
import { writeFileAtomic } from "./artifacts.js";

const MAX_SOURCE_FILES = 256;
const MAX_SOURCE_FILE_BYTES = 256 * 1024 * 1024;
const MAX_SOURCE_TOTAL_BYTES = 256 * 1024 * 1024;
const MAX_DEPENDENCY_FILES = 32;
const MAX_DEPENDENCY_FILE_BYTES = 64 * 1024 * 1024;
const MAX_DEPENDENCY_TOTAL_BYTES = 64 * 1024 * 1024;
const MAX_MODEL_FILES = 100_000;
const MAX_MODEL_DIRECTORIES = 100_000;
const MAX_MODEL_TOTAL_BYTES = 20 * 1024 * 1024 * 1024;
const MAX_TREE_DEPTH = 64;
const MAX_IMPLEMENTATION_DIRECTORIES = (
  (MAX_SOURCE_FILES + MAX_DEPENDENCY_FILES) * MAX_TREE_DEPTH
) + 3;
const READ_BUFFER_BYTES = 1024 * 1024;

export interface StudyFileEvidence {
  path: string;
  size_bytes: number;
  sha256: string;
}

export interface StudyImplementationFileEvidence extends StudyFileEvidence {
  role: "source" | "dependency_lock";
  snapshot_path: string;
}

export interface StudyImplementationManifest {
  schema_version: 1;
  evidence: "bundled_locked" | "declared_files";
  files: StudyImplementationFileEvidence[];
}

export interface StudyModelManifest {
  schema_version: 1;
  directory: "model";
  file_count: number;
  size_bytes: number;
  files: StudyFileEvidence[];
}

export interface StudyImplementationReference {
  manifest: "implementation/manifest.json";
  sha256: string;
  file_count: number;
  evidence: StudyImplementationManifest["evidence"];
}

export interface StudyModelReference {
  manifest: "model-manifest.json";
  sha256: string;
  file_count: number;
  size_bytes: number;
}

export interface StudyImplementationInputFile {
  role: "source" | "dependency_lock";
  path: string;
  absolutePath: string;
  rootPath?: string;
}

interface PreparedImplementationFile extends StudyImplementationFileEvidence {
  absolutePath: string;
}

export interface PreparedStudyImplementation {
  evidence: StudyImplementationManifest["evidence"];
  files: PreparedImplementationFile[];
}

export interface CapturedStudyImplementation {
  prepared: PreparedStudyImplementation;
  manifest: StudyImplementationManifest;
  reference: StudyImplementationReference;
}

export interface DirectoryIdentity {
  dev: number;
  ino: number;
}

interface StableFileResult {
  evidence: Omit<StudyFileEvidence, "path">;
  state: StableFileState;
  bytes?: Uint8Array;
}

interface StableFileState {
  dev: number;
  ino: number;
  size: number;
  mtimeMs: number;
  ctimeMs: number;
  nlink: number;
}

function sha256(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function jsonText(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function portableRelative(root: string, path: string): string {
  return relative(root, path).split(sep).join("/");
}

function sameIdentity(
  left: { dev: number; ino: number },
  right: { dev: number; ino: number },
): boolean {
  return left.dev === right.dev && left.ino === right.ino;
}

function sameFileState(
  left: StableFileState,
  right: StableFileState,
): boolean {
  return (
    sameIdentity(left, right)
    && left.size === right.size
    && left.mtimeMs === right.mtimeMs
    && left.ctimeMs === right.ctimeMs
    && left.nlink === right.nlink
  );
}

async function readStableRegularFile(args: {
  path: string;
  description: string;
  maxBytes: number;
  captureBytes?: boolean;
}): Promise<StableFileResult> {
  let handle: Awaited<ReturnType<typeof open>>;
  try {
    handle = await open(
      args.path,
      constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK,
    );
  } catch (error) {
    throw new Error(
      `${args.description} must be a readable regular, non-symbolic file: ${args.path}`,
      { cause: error },
    );
  }
  try {
    const before = await handle.stat();
    if (!before.isFile()) {
      throw new Error(`${args.description} must be a regular file: ${args.path}`);
    }
    if (before.size > args.maxBytes) {
      throw new Error(
        `${args.description} exceeds the ${args.maxBytes}-byte limit: ${args.path}`,
      );
    }

    const digest = createHash("sha256");
    const chunks: Buffer[] = [];
    const buffer = Buffer.allocUnsafe(READ_BUFFER_BYTES);
    let size = 0;
    while (true) {
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, null);
      if (bytesRead === 0) break;
      size += bytesRead;
      if (size > args.maxBytes) {
        throw new Error(
          `${args.description} exceeds the ${args.maxBytes}-byte limit: ${args.path}`,
        );
      }
      const chunk = buffer.subarray(0, bytesRead);
      digest.update(chunk);
      if (args.captureBytes) chunks.push(Buffer.from(chunk));
    }

    const after = await handle.stat();
    if (size !== after.size || !sameFileState(before, after)) {
      throw new Error(`${args.description} changed while it was being read: ${args.path}`);
    }
    const current = await lstat(args.path);
    if (
      !current.isFile()
      || current.isSymbolicLink()
      || !sameFileState(after, current)
    ) {
      throw new Error(`${args.description} path changed while it was being read: ${args.path}`);
    }
    return {
      evidence: {
        size_bytes: size,
        sha256: digest.digest("hex"),
      },
      state: {
        dev: after.dev,
        ino: after.ino,
        size: after.size,
        mtimeMs: after.mtimeMs,
        ctimeMs: after.ctimeMs,
        nlink: after.nlink,
      },
      ...(args.captureBytes ? { bytes: Buffer.concat(chunks, size) } : {}),
    };
  } finally {
    await handle.close();
  }
}

function roleLimits(role: StudyImplementationInputFile["role"]): {
  maxFiles: number;
  maxFileBytes: number;
  maxTotalBytes: number;
} {
  return role === "source"
    ? {
        maxFiles: MAX_SOURCE_FILES,
        maxFileBytes: MAX_SOURCE_FILE_BYTES,
        maxTotalBytes: MAX_SOURCE_TOTAL_BYTES,
      }
    : {
        maxFiles: MAX_DEPENDENCY_FILES,
        maxFileBytes: MAX_DEPENDENCY_FILE_BYTES,
        maxTotalBytes: MAX_DEPENDENCY_TOTAL_BYTES,
      };
}

function snapshotPath(file: StudyImplementationInputFile): string {
  const directory = file.role === "source" ? "source" : "dependency-lock";
  return `implementation/${directory}/${file.path}`;
}

function compareEvidence(
  expected: Pick<StudyFileEvidence, "size_bytes" | "sha256">,
  actual: Pick<StudyFileEvidence, "size_bytes" | "sha256">,
  description: string,
): void {
  if (
    expected.size_bytes !== actual.size_bytes
    || expected.sha256 !== actual.sha256
  ) {
    throw new Error(`${description} changed after provenance capture`);
  }
}

async function assertNonSymbolicDescendant(
  file: StudyImplementationInputFile,
): Promise<void> {
  if (!file.rootPath) return;
  const root = await lstat(file.rootPath);
  if (!root.isDirectory() || root.isSymbolicLink()) {
    throw new Error(
      `Study implementation provenance root must be a real directory: ${file.rootPath}`,
    );
  }
  let current = file.rootPath;
  const segments = file.path.split("/");
  for (const [index, segment] of segments.entries()) {
    current = join(current, segment);
    let metadata: Awaited<ReturnType<typeof lstat>>;
    try {
      metadata = await lstat(current);
    } catch (error) {
      throw new Error(
        `Study implementation ${file.role} must be a readable regular, non-symbolic file: ${file.absolutePath}`,
        { cause: error },
      );
    }
    if (metadata.isSymbolicLink()) {
      throw new Error(
        `Study implementation provenance paths must not contain symbolic links: ${file.path}`,
      );
    }
    if (index < segments.length - 1 && !metadata.isDirectory()) {
      throw new Error(
        `Study implementation provenance path has a non-directory component: ${file.path}`,
      );
    }
  }
}

export async function prepareStudyImplementation(args: {
  evidence: StudyImplementationManifest["evidence"];
  files: StudyImplementationInputFile[];
}): Promise<PreparedStudyImplementation> {
  const sorted = [...args.files].sort((left, right) => {
    if (left.role !== right.role) return left.role === "source" ? -1 : 1;
    return left.path < right.path ? -1 : left.path > right.path ? 1 : 0;
  });
  const seen = new Set<string>();
  const roleCounts = { source: 0, dependency_lock: 0 };
  const roleBytes = { source: 0, dependency_lock: 0 };
  const prepared: PreparedImplementationFile[] = [];

  for (const file of sorted) {
    if (seen.has(file.path)) {
      throw new Error(`Study implementation provenance repeats path: ${file.path}`);
    }
    seen.add(file.path);
    const limits = roleLimits(file.role);
    roleCounts[file.role] += 1;
    if (roleCounts[file.role] > limits.maxFiles) {
      throw new Error(
        `Study implementation provenance has too many ${file.role} files`,
      );
    }
    await assertNonSymbolicDescendant(file);
    const inspected = await readStableRegularFile({
      path: file.absolutePath,
      description: `study implementation ${file.role}`,
      maxBytes: limits.maxFileBytes,
    });
    roleBytes[file.role] += inspected.evidence.size_bytes;
    if (roleBytes[file.role] > limits.maxTotalBytes) {
      throw new Error(
        `Study implementation ${file.role} files exceed the total byte limit`,
      );
    }
    prepared.push({
      role: file.role,
      path: file.path,
      snapshot_path: snapshotPath(file),
      absolutePath: file.absolutePath,
      ...inspected.evidence,
    });
  }
  return { evidence: args.evidence, files: prepared };
}

export async function captureStudyImplementation(args: {
  trialDirectory: string;
  prepared: PreparedStudyImplementation;
}): Promise<CapturedStudyImplementation> {
  for (const file of args.prepared.files) {
    const limits = roleLimits(file.role);
    const current = await readStableRegularFile({
      path: file.absolutePath,
      description: `study implementation ${file.role}`,
      maxBytes: limits.maxFileBytes,
      captureBytes: true,
    });
    compareEvidence(file, current.evidence, `Study implementation ${file.path}`);
    const destination = join(args.trialDirectory, ...file.snapshot_path.split("/"));
    await mkdir(dirname(destination), { recursive: true, mode: 0o700 });
    await writeFile(destination, current.bytes!, {
      flag: "wx",
      mode: 0o400,
    });
  }

  const manifest: StudyImplementationManifest = {
    schema_version: 1,
    evidence: args.prepared.evidence,
    files: args.prepared.files.map((file) => ({
      role: file.role,
      path: file.path,
      snapshot_path: file.snapshot_path,
      size_bytes: file.size_bytes,
      sha256: file.sha256,
    })),
  };
  const text = jsonText(manifest);
  const manifestPath = join(args.trialDirectory, "implementation", "manifest.json");
  await mkdir(dirname(manifestPath), { recursive: true, mode: 0o700 });
  await writeFile(manifestPath, text, { flag: "wx", mode: 0o400 });
  return {
    prepared: args.prepared,
    manifest,
    reference: {
      manifest: "implementation/manifest.json",
      sha256: sha256(text),
      file_count: manifest.files.length,
      evidence: manifest.evidence,
    },
  };
}

async function collectTreePaths(args: {
  root: string;
  description: string;
  maxFiles: number;
  maxDirectories: number;
}): Promise<string[]> {
  const paths: string[] = [];
  let directoryCount = 0;
  const visit = async (directory: string, depth: number): Promise<void> => {
    if (depth > MAX_TREE_DEPTH) {
      throw new Error(`${args.description} exceeds the ${MAX_TREE_DEPTH}-level depth limit`);
    }
    directoryCount += 1;
    if (directoryCount > args.maxDirectories) {
      throw new Error(
        `${args.description} exceeds the ${args.maxDirectories}-directory limit`,
      );
    }
    const before = await lstat(directory);
    if (!before.isDirectory() || before.isSymbolicLink()) {
      throw new Error(`${args.description} must contain only real directories`);
    }
    const names = (await readdir(directory)).sort((left, right) => (
      left < right ? -1 : left > right ? 1 : 0
    ));
    for (const name of names) {
      const absolute = join(directory, name);
      const metadata = await lstat(absolute);
      if (metadata.isSymbolicLink()) {
        throw new Error(`${args.description} must not contain symbolic links: ${absolute}`);
      }
      if (metadata.isDirectory()) {
        await visit(absolute, depth + 1);
        continue;
      }
      if (!metadata.isFile()) {
        throw new Error(`${args.description} must contain only regular files: ${absolute}`);
      }
      paths.push(portableRelative(args.root, absolute));
      if (paths.length > args.maxFiles) {
        throw new Error(`${args.description} exceeds the ${args.maxFiles}-file limit`);
      }
    }
    const after = await lstat(directory);
    if (!after.isDirectory() || after.isSymbolicLink() || !sameIdentity(before, after)) {
      throw new Error(`${args.description} directory changed while it was being read`);
    }
  };
  await visit(args.root, 0);
  return paths;
}

async function describeTree(args: {
  root: string;
  description: string;
  maxFiles: number;
  maxDirectories: number;
  maxTotalBytes: number;
  expectedRoot?: DirectoryIdentity;
}): Promise<StudyFileEvidence[]> {
  const rootBefore = await lstat(args.root);
  if (
    !rootBefore.isDirectory()
    || rootBefore.isSymbolicLink()
    || (args.expectedRoot && !sameIdentity(rootBefore, args.expectedRoot))
  ) {
    throw new Error(`${args.description} directory was replaced`);
  }
  const paths = await collectTreePaths(args);
  const files: StudyFileEvidence[] = [];
  const states = new Map<string, StableFileState>();
  let totalBytes = 0;
  for (const path of paths) {
    const inspected = await readStableRegularFile({
      path: join(args.root, ...path.split("/")),
      description: args.description,
      maxBytes: args.maxTotalBytes,
    });
    totalBytes += inspected.evidence.size_bytes;
    if (totalBytes > args.maxTotalBytes) {
      throw new Error(
        `${args.description} exceeds the ${args.maxTotalBytes}-byte total limit`,
      );
    }
    files.push({ path, ...inspected.evidence });
    states.set(path, inspected.state);
  }
  const pathsAfter = await collectTreePaths(args);
  if (JSON.stringify(pathsAfter) !== JSON.stringify(paths)) {
    throw new Error(`${args.description} tree changed while it was being read`);
  }
  const rootAfter = await lstat(args.root);
  if (!rootAfter.isDirectory() || !sameIdentity(rootBefore, rootAfter)) {
    throw new Error(`${args.description} directory changed while it was being read`);
  }
  for (const path of paths) {
    const current = await lstat(join(args.root, ...path.split("/")));
    if (
      !current.isFile()
      || current.isSymbolicLink()
      || !sameFileState(states.get(path)!, current)
    ) {
      throw new Error(`${args.description} changed while the tree was being read`);
    }
  }
  return files;
}

export async function verifyStudyImplementation(args: {
  trialDirectory: string;
  captured: CapturedStudyImplementation;
}): Promise<void> {
  for (const file of args.captured.prepared.files) {
    const limits = roleLimits(file.role);
    const current = await readStableRegularFile({
      path: file.absolutePath,
      description: `study implementation ${file.role}`,
      maxBytes: limits.maxFileBytes,
    });
    compareEvidence(file, current.evidence, `Study implementation ${file.path}`);
  }

  const implementationRoot = join(args.trialDirectory, "implementation");
  const actual = await describeTree({
    root: implementationRoot,
    description: "study implementation snapshot",
    maxFiles: args.captured.manifest.files.length + 1,
    maxDirectories: MAX_IMPLEMENTATION_DIRECTORIES,
    maxTotalBytes: (
      MAX_SOURCE_TOTAL_BYTES
      + MAX_DEPENDENCY_TOTAL_BYTES
      + 1024 * 1024
    ),
  });
  const manifestText = jsonText(args.captured.manifest);
  const expected = [
    {
      path: "manifest.json",
      size_bytes: Buffer.byteLength(manifestText),
      sha256: sha256(manifestText),
    },
    ...args.captured.manifest.files.map((file) => ({
      path: file.snapshot_path.replace(/^implementation\//, ""),
      size_bytes: file.size_bytes,
      sha256: file.sha256,
    })),
  ].sort((left, right) => (
    left.path < right.path ? -1 : left.path > right.path ? 1 : 0
  ));
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error("Study implementation snapshot changed during trial execution");
  }
}

export async function captureDirectoryIdentity(path: string): Promise<DirectoryIdentity> {
  const metadata = await lstat(path);
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
    throw new Error(`Study path must be a real directory: ${path}`);
  }
  return { dev: metadata.dev, ino: metadata.ino };
}

export async function verifyDirectoryIdentity(args: {
  path: string;
  expected: DirectoryIdentity;
  description: string;
}): Promise<void> {
  const metadata = await lstat(args.path);
  if (
    !metadata.isDirectory()
    || metadata.isSymbolicLink()
    || !sameIdentity(metadata, args.expected)
  ) {
    throw new Error(`${args.description} was replaced during trial execution`);
  }
}

export async function writeStudyModelManifest(args: {
  trialDirectory: string;
  modelDirectory: string;
  expectedRoot: DirectoryIdentity;
}): Promise<{
  manifest: StudyModelManifest;
  reference: StudyModelReference;
}> {
  const files = await describeTree({
    root: args.modelDirectory,
    description: "study model artifacts",
    maxFiles: MAX_MODEL_FILES,
    maxDirectories: MAX_MODEL_DIRECTORIES,
    maxTotalBytes: MAX_MODEL_TOTAL_BYTES,
    expectedRoot: args.expectedRoot,
  });
  const manifest: StudyModelManifest = {
    schema_version: 1,
    directory: "model",
    file_count: files.length,
    size_bytes: files.reduce((total, file) => total + file.size_bytes, 0),
    files,
  };
  const text = jsonText(manifest);
  await writeFileAtomic(join(args.trialDirectory, "model-manifest.json"), text);
  return {
    manifest,
    reference: {
      manifest: "model-manifest.json",
      sha256: sha256(text),
      file_count: manifest.file_count,
      size_bytes: manifest.size_bytes,
    },
  };
}
