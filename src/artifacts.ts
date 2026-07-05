import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";

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
  progressJsonl: string;
  trainingDir: string;
  trainingInputDir: string;
  trainingConfigDir: string;
  trainingOutputDir: string;
  trainingModelDir: string;
  trainingLog: string;
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
  const runDir = isAbsolute(safePrefix) ? safePrefix : join(root, safePrefix);
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
    progressJsonl: join(runDir, "progress.jsonl"),
    trainingDir,
    trainingInputDir: join(trainingDir, "input", "data", "training"),
    trainingConfigDir: join(trainingDir, "input", "config"),
    trainingOutputDir: join(trainingDir, "output"),
    trainingModelDir: join(trainingDir, "model"),
    trainingLog: join(trainingDir, "training.log"),
  };
}

export async function prepareRunDirectories(artifacts: RunArtifacts): Promise<void> {
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
