import { appendFile, copyFile, mkdir, readdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { randomUUID } from "node:crypto";
import type { FineTuneRunRequest, RunReport, SpecSnapshot, TrainingReport } from "./contracts.js";

export type LocalRunStatus =
  | "queued"
  | "preparing"
  | "evaluating_baseline"
  | "training"
  | "evaluating_candidate"
  | "scoring"
  | "reporting"
  | "stage_completed"
  | "completed"
  | "failed"
  | "cancelled";

export interface LocalRunState {
  id: string;
  behavior_spec_id: string;
  user_id: string;
  run_number: number;
  status: LocalRunStatus;
  current_stage: string;
  status_message: string;
  artifact_dir: string;
  report_path?: string;
  model_id?: string;
  error?: string;
  base_model: string;
  spec_name: string;
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
}

export function isTerminalRunState(
  state: Pick<LocalRunState, "status" | "current_stage" | "completed_at">,
): boolean {
  return state.status === "stage_completed"
    || state.status === "completed"
    || state.status === "failed"
    || (state.status === "cancelled" && state.current_stage === "cancelled" && Boolean(state.completed_at));
}

export interface LocalRunEvent {
  id: string;
  run_id: string;
  stage: string;
  status: LocalRunStatus | "running" | "completed" | "failed";
  message: string;
  details?: Record<string, unknown>;
  occurred_at: string;
}

export interface LocalModelRecord {
  id: string;
  run_id: string;
  behavior_spec_id: string;
  name: string;
  provider: "local-uv";
  base_model: string;
  artifact_uri: string;
  artifact_dir: string;
  metrics: Record<string, unknown> | null;
  created_at: string;
}

export interface LocalSpecRecord {
  id: string;
  name: string;
  base_model: string;
  path: string;
  created_at: string;
  updated_at: string;
}

export interface LocalStore {
  root: string;
  paths: ReturnType<typeof localStorePaths>;
  ensure(): Promise<void>;
  importSpec(specId: string, spec: SpecSnapshot): Promise<LocalSpecRecord>;
  listSpecs(): Promise<LocalSpecRecord[]>;
  getSpec(id: string): Promise<LocalSpecRecord & { spec: SpecSnapshot }>;
  startRun(args: { request: FineTuneRunRequest; artifactDir: string }): Promise<LocalRunState>;
  syncRunRequest(request: FineTuneRunRequest, artifactDir?: string): Promise<LocalRunState>;
  updateRun(args: {
    runId: string;
    status: LocalRunStatus;
    stage: string;
    message: string;
    details?: Record<string, unknown>;
  }): Promise<LocalRunState>;
  completeRun(report: RunReport, artifactDir: string, reportPath: string): Promise<LocalRunState>;
  registerModel(args: {
    request: FineTuneRunRequest;
    training: TrainingReport;
    artifactDir: string;
    createdAt?: string;
  }): Promise<LocalModelRecord | null>;
  invalidateRunOutputs(runId: string, options: { report?: boolean; model?: boolean }): Promise<LocalRunState>;
  failRun(runId: string, error: string): Promise<LocalRunState>;
  cancelRun(runId: string): Promise<void>;
  finalizeCancellation(runId: string): Promise<LocalRunState>;
  isCancellationRequested(runId: string): Promise<boolean>;
  listRuns(): Promise<LocalRunState[]>;
  getRun(id: string): Promise<LocalRunState>;
  getRunEvents(id: string): Promise<LocalRunEvent[]>;
  getRunReport(id: string): Promise<RunReport>;
  listModels(): Promise<LocalModelRecord[]>;
  getModel(id: string): Promise<LocalModelRecord>;
}

export function defaultLocalHome(): string {
  return resolve(process.env.TT_LOCAL_HOME ?? join(homedir(), ".tuned-tensor-local"));
}

export function localStorePaths(root: string) {
  return {
    root,
    specsDir: join(root, "specs"),
    runsDir: join(root, "runs"),
    modelsDir: join(root, "models"),
    datasetsDir: join(root, "datasets"),
  };
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const tmp = `${path}.${process.pid}.${Date.now()}.${randomUUID()}.tmp`;
  await writeFile(tmp, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await rename(tmp, path);
}

async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await readFile(path, "utf8")) as T;
}

async function appendJsonl(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await appendFile(path, `${JSON.stringify(value)}\n`, "utf8");
}

async function readJsonl<T>(path: string): Promise<T[]> {
  if (!(await exists(path))) return [];
  return (await readFile(path, "utf8"))
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

async function copyIfExists(from: string, to: string): Promise<void> {
  if (!(await exists(from))) return;
  await mkdir(dirname(to), { recursive: true });
  await copyFile(from, to);
}

async function childDirectoryNames(path: string): Promise<string[]> {
  try {
    const entries = await readdir(path, { withFileTypes: true });
    return entries
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort();
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

function compareIds(left: { id: string }, right: { id: string }): number {
  if (left.id === right.id) return 0;
  return left.id < right.id ? -1 : 1;
}

function compareUpdatedAt(
  left: Pick<LocalRunState, "id" | "updated_at">,
  right: Pick<LocalRunState, "id" | "updated_at">,
): number {
  if (left.updated_at !== right.updated_at) return left.updated_at > right.updated_at ? -1 : 1;
  return compareIds(left, right);
}

function compareCreatedAt(
  left: Pick<LocalModelRecord, "id" | "created_at">,
  right: Pick<LocalModelRecord, "id" | "created_at">,
): number {
  if (left.created_at !== right.created_at) return left.created_at > right.created_at ? -1 : 1;
  return compareIds(left, right);
}

function findByIdOrPrefix<T extends { id: string }>(records: T[], id: string): T | undefined {
  return records.find((record) => record.id === id) ?? records.find((record) => record.id.startsWith(id));
}

export function createLocalStore(root = defaultLocalHome()): LocalStore {
  const resolvedRoot = resolve(root);
  const paths = localStorePaths(resolvedRoot);

  const runDir = (id: string) => join(paths.runsDir, id);
  const runStatePath = (id: string) => join(runDir(id), "state.json");
  const runEventsPath = (id: string) => join(runDir(id), "progress.jsonl");
  const runRequestPath = (id: string) => join(runDir(id), "request.json");
  const runReportPath = (id: string) => join(runDir(id), "run-report.json");
  const cancellationPath = (id: string) => join(runDir(id), "cancel.requested");
  const specDir = (id: string) => join(paths.specsDir, id);
  const specPath = (id: string) => join(specDir(id), "spec.json");
  const modelPath = (id: string) => join(paths.modelsDir, id, "model.json");

  async function ensure() {
    await Promise.all([
      mkdir(paths.specsDir, { recursive: true }),
      mkdir(paths.runsDir, { recursive: true }),
      mkdir(paths.modelsDir, { recursive: true }),
      mkdir(paths.datasetsDir, { recursive: true }),
    ]);
  }

  async function writeRunState(state: LocalRunState): Promise<LocalRunState> {
    await writeJsonAtomic(runStatePath(state.id), state);
    return state;
  }

  async function readSpecRecord(id: string): Promise<LocalSpecRecord | null> {
    const path = specPath(id);
    if (!(await exists(path))) return null;
    const [spec, fileStats, directoryStats] = await Promise.all([
      readJson<SpecSnapshot>(path),
      stat(path),
      stat(specDir(id)),
    ]);
    const createdAt = directoryStats.birthtimeMs > 0 ? directoryStats.birthtime : directoryStats.ctime;
    return {
      id,
      name: spec.name,
      base_model: spec.base_model,
      path,
      created_at: createdAt.toISOString(),
      updated_at: fileStats.mtime.toISOString(),
    };
  }

  async function listSpecRecords(): Promise<LocalSpecRecord[]> {
    await ensure();
    const records: LocalSpecRecord[] = [];
    for (const id of await childDirectoryNames(paths.specsDir)) {
      const record = await readSpecRecord(id);
      if (record) records.push(record);
    }
    return records.sort(compareUpdatedAt);
  }

  async function listRunRecords(): Promise<LocalRunState[]> {
    await ensure();
    const records: LocalRunState[] = [];
    for (const id of await childDirectoryNames(paths.runsDir)) {
      const path = runStatePath(id);
      if (await exists(path)) records.push(await readJson<LocalRunState>(path));
    }
    return records.sort(compareUpdatedAt);
  }

  async function listModelRecords(): Promise<LocalModelRecord[]> {
    await ensure();
    const records: LocalModelRecord[] = [];
    for (const id of await childDirectoryNames(paths.modelsDir)) {
      const path = modelPath(id);
      if (await exists(path)) records.push(await readJson<LocalModelRecord>(path));
    }
    return records.sort(compareCreatedAt);
  }

  async function importSpecRecord(specId: string, spec: SpecSnapshot): Promise<LocalSpecRecord> {
    await ensure();
    await mkdir(specDir(specId), { recursive: true });
    await writeJsonAtomic(specPath(specId), spec);
    const record = await readSpecRecord(specId);
    if (!record) throw new Error(`Failed to persist spec: ${specId}`);
    return record;
  }

  async function appendRunEvent(state: LocalRunState, event: Omit<LocalRunEvent, "id" | "run_id" | "occurred_at">): Promise<void> {
    const row: LocalRunEvent = {
      id: randomUUID(),
      run_id: state.id,
      occurred_at: new Date().toISOString(),
      ...event,
    };
    await appendJsonl(runEventsPath(state.id), row);
    await appendJsonl(join(state.artifact_dir, "progress.jsonl"), {
      at: row.occurred_at,
      stage: row.stage,
      status: row.status,
      message: row.message,
      ...(row.details ? { details: row.details } : {}),
    });
  }

  async function finalizeCancellationState(runId: string): Promise<LocalRunState> {
    const previous = await getRun(runId);
    if (previous.status === "cancelled" && previous.current_stage === "cancelled" && previous.completed_at) {
      return previous;
    }
    const now = new Date().toISOString();
    const cancelled: LocalRunState = {
      ...previous,
      status: "cancelled",
      current_stage: "cancelled",
      status_message: "Run cancelled.",
      error: undefined,
      completed_at: now,
      updated_at: now,
    };
    await writeRunState(cancelled);
    await appendRunEvent(cancelled, { stage: "cancelled", status: "cancelled", message: "Run cancelled." });
    return cancelled;
  }

  async function preserveCancellationRequestState(runId: string): Promise<LocalRunState> {
    const previous = await getRun(runId);
    if (isTerminalRunState(previous)) return previous;
    if (previous.status === "cancelled" && previous.current_stage === "cancel_requested") return previous;
    const requested: LocalRunState = {
      ...previous,
      status: "cancelled",
      current_stage: "cancel_requested",
      status_message: "Cancellation requested; waiting for the active worker to stop.",
      error: undefined,
      completed_at: undefined,
      updated_at: new Date().toISOString(),
    };
    return writeRunState(requested);
  }

  async function resolveRunId(id: string): Promise<string> {
    await ensure();
    if (await exists(runStatePath(id))) return id;
    const record = findByIdOrPrefix(await listRunRecords(), id);
    if (!record) throw new Error(`Run not found: ${id}`);
    return record.id;
  }

  async function getRun(id: string): Promise<LocalRunState> {
    return readJson<LocalRunState>(runStatePath(await resolveRunId(id)));
  }

  async function registerModelRecord(args: {
    request: FineTuneRunRequest;
    training: TrainingReport;
    artifactDir: string;
    createdAt?: string;
  }): Promise<LocalModelRecord | null> {
    if (!args.training.model_artifact_uri || args.training.metrics?.dry_run === true) return null;
    const previous = await getRun(args.request.run_id);
    const modelId = `local-${args.request.run_id}`;
    const existingModel = await exists(modelPath(modelId))
      ? await readJson<LocalModelRecord>(modelPath(modelId))
      : undefined;
    const now = new Date().toISOString();
    const model: LocalModelRecord = {
      id: modelId,
      run_id: args.request.run_id,
      behavior_spec_id: args.request.behavior_spec_id,
      name: `${args.request.spec_snapshot.base_model} (${args.request.run_id.slice(0, 8)})`,
      provider: args.training.provider,
      base_model: args.request.spec_snapshot.base_model,
      artifact_uri: args.training.model_artifact_uri,
      artifact_dir: args.artifactDir,
      metrics: args.training.metrics,
      created_at: existingModel?.created_at ?? args.createdAt ?? now,
    };
    await writeJsonAtomic(modelPath(model.id), model);
    const alreadyRegistered = previous.model_id === modelId;
    const state: LocalRunState = {
      ...previous,
      model_id: modelId,
      updated_at: now,
    };
    await writeRunState(state);
    if (!alreadyRegistered) {
      await appendRunEvent(state, {
        stage: "model_registered",
        status: "running",
        message: "Local model artifact registered.",
        details: { model_id: modelId, artifact_uri: model.artifact_uri },
      });
    }
    return model;
  }

  return {
    root: resolvedRoot,
    paths,
    ensure,

    importSpec: importSpecRecord,

    async listSpecs() {
      return listSpecRecords();
    },

    async getSpec(id) {
      const record = findByIdOrPrefix(await listSpecRecords(), id);
      if (!record) throw new Error(`Spec not found: ${id}`);
      return { ...record, spec: await readJson<SpecSnapshot>(record.path) };
    },

    async startRun({ request, artifactDir }) {
      await ensure();
      await importSpecRecord(request.behavior_spec_id, request.spec_snapshot);
      const now = new Date().toISOString();
      const state: LocalRunState = {
        id: request.run_id,
        behavior_spec_id: request.behavior_spec_id,
        user_id: request.user_id,
        run_number: request.run_number,
        status: "queued",
        current_stage: "queued",
        status_message: "Run queued.",
        artifact_dir: artifactDir,
        base_model: request.spec_snapshot.base_model,
        spec_name: request.spec_snapshot.name,
        created_at: now,
        updated_at: now,
        started_at: now,
      };
      await writeJsonAtomic(runRequestPath(request.run_id), request);
      await writeRunState(state);
      await appendRunEvent(state, { stage: "queued", status: "queued", message: "Run queued." });
      return state;
    },

    async syncRunRequest(request, artifactDir) {
      const previous = await getRun(request.run_id);
      if (
        previous.behavior_spec_id !== request.behavior_spec_id
        || previous.user_id !== request.user_id
        || previous.run_number !== request.run_number
      ) {
        throw new Error(
          `Run ${request.run_id} cannot be reused with different user, behavior spec, or run number identity.`,
        );
      }
      await importSpecRecord(request.behavior_spec_id, request.spec_snapshot);
      await writeJsonAtomic(runRequestPath(request.run_id), request);
      const state: LocalRunState = {
        ...previous,
        artifact_dir: artifactDir ?? previous.artifact_dir,
        base_model: request.spec_snapshot.base_model,
        spec_name: request.spec_snapshot.name,
        updated_at: new Date().toISOString(),
      };
      await writeRunState(state);
      return state;
    },

    async updateRun({ runId, status, stage, message, details }) {
      const previous = await getRun(runId);
      if (status !== "cancelled" && await exists(cancellationPath(previous.id))) {
        return preserveCancellationRequestState(previous.id);
      }
      const active = status !== "stage_completed" && status !== "completed" && status !== "failed" && status !== "cancelled";
      const successful = status === "stage_completed" || status === "completed";
      const now = new Date().toISOString();
      const state: LocalRunState = {
        ...previous,
        status,
        current_stage: stage,
        status_message: message,
        ...(active
          ? { error: undefined, completed_at: undefined }
          : successful
            ? { error: undefined, completed_at: now }
            : {}),
        updated_at: now,
      };
      await writeRunState(state);
      if (status !== "cancelled" && await exists(cancellationPath(previous.id))) {
        return preserveCancellationRequestState(previous.id);
      }
      await appendRunEvent(state, {
        stage,
        status: successful ? "completed" : status === "failed" ? "failed" : "running",
        message,
        details,
      });
      return state;
    },

    async completeRun(report, artifactDir, reportPath) {
      if (await exists(cancellationPath(report.run_id))) {
        return preserveCancellationRequestState(report.run_id);
      }
      const request = await readJson<FineTuneRunRequest>(runRequestPath(report.run_id));
      const completedAt = report.created_at ?? new Date().toISOString();
      const model = await registerModelRecord({
        request,
        training: report.training,
        artifactDir,
        createdAt: completedAt,
      });
      if (!model) {
        await rm(join(paths.modelsDir, `local-${report.run_id}`), { recursive: true, force: true });
      }
      const previous = await getRun(report.run_id);
      const state: LocalRunState = {
        ...previous,
        status: "completed",
        current_stage: "completed",
        status_message: "Run completed successfully.",
        report_path: reportPath,
        model_id: model?.id,
        error: undefined,
        completed_at: completedAt,
        updated_at: completedAt,
      };
      await writeRunState(state);
      if (await exists(cancellationPath(report.run_id))) {
        return preserveCancellationRequestState(report.run_id);
      }
      await copyIfExists(reportPath, runReportPath(report.run_id));
      await appendRunEvent(state, {
        stage: "completed",
        status: "completed",
        message: "Run completed successfully.",
        details: { report_path: reportPath, ...(model ? { model_id: model.id } : {}) },
      });
      return state;
    },

    registerModel: registerModelRecord,

    async invalidateRunOutputs(runId, options) {
      const previous = await getRun(runId);
      if (options.report) await rm(runReportPath(previous.id), { force: true });
      if (options.model) {
        await rm(join(paths.modelsDir, `local-${previous.id}`), { recursive: true, force: true });
      }
      const state: LocalRunState = {
        ...previous,
        ...(options.report ? { report_path: undefined } : {}),
        ...(options.model ? { model_id: undefined } : {}),
        error: undefined,
        completed_at: undefined,
        updated_at: new Date().toISOString(),
      };
      return writeRunState(state);
    },

    async failRun(runId, error) {
      const previous = await getRun(runId);
      const now = new Date().toISOString();
      if (await exists(cancellationPath(previous.id))) {
        return preserveCancellationRequestState(previous.id);
      }
      const state: LocalRunState = {
        ...previous,
        status: "failed",
        current_stage: "failed",
        status_message: error,
        error,
        completed_at: now,
        updated_at: now,
      };
      await writeRunState(state);
      if (await exists(cancellationPath(previous.id))) {
        return preserveCancellationRequestState(previous.id);
      }
      await appendRunEvent(state, { stage: "failed", status: "failed", message: error });
      return state;
    },

    async cancelRun(runId) {
      const state = await getRun(runId);
      if (isTerminalRunState(state) || (state.status === "cancelled" && state.current_stage === "cancel_requested")) {
        return;
      }
      await writeFile(cancellationPath(state.id), `${new Date().toISOString()}\n`, "utf8");
      const latest = await getRun(state.id);
      if (isTerminalRunState(latest)) {
        if (latest.status !== "cancelled") await rm(cancellationPath(state.id), { force: true });
        return;
      }
      const updated: LocalRunState = {
        ...latest,
        status: "cancelled",
        current_stage: "cancel_requested",
        status_message: "Cancellation requested; waiting for the active worker to stop.",
        error: undefined,
        completed_at: undefined,
        updated_at: new Date().toISOString(),
      };
      await writeRunState(updated);
      await appendRunEvent(updated, { stage: "cancel_requested", status: "cancelled", message: "Cancellation requested." });
    },

    async finalizeCancellation(runId) {
      return finalizeCancellationState(runId);
    },

    async isCancellationRequested(runId) {
      const state = await getRun(runId);
      return state.status === "cancelled" || await exists(cancellationPath(state.id));
    },

    async listRuns() {
      return listRunRecords();
    },

    getRun,

    async getRunEvents(id) {
      const state = await getRun(id);
      return readJsonl<LocalRunEvent>(runEventsPath(state.id));
    },

    async getRunReport(id) {
      const state = await getRun(id);
      if (state.report_path && await exists(state.report_path)) {
        return readJson<RunReport>(state.report_path);
      }
      const copiedReportPath = runReportPath(state.id);
      if (await exists(copiedReportPath)) return readJson<RunReport>(copiedReportPath);
      throw new Error(`Run has no report yet: ${id}`);
    },

    async listModels() {
      return listModelRecords();
    },

    async getModel(id) {
      const record = findByIdOrPrefix(await listModelRecords(), id);
      if (!record) throw new Error(`Model not found: ${id}`);
      return record;
    },
  };
}
