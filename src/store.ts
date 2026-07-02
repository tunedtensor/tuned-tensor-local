import { appendFile, copyFile, mkdir, readdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { randomUUID } from "node:crypto";
import type { FineTuneRunRequest, RunReport, SpecSnapshot } from "./contracts.js";

export type LocalRunStatus =
  | "queued"
  | "preparing"
  | "evaluating_baseline"
  | "training"
  | "evaluating_candidate"
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

export interface LocalRunEvent {
  id: string;
  run_id: string;
  stage: string;
  status: LocalRunStatus | "running" | "completed" | "failed";
  message: string;
  details?: Record<string, unknown>;
  occurred_at: string;
}

export interface LocalRunIndexRecord extends LocalRunState {
  catalog_updated_at: string;
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
  updateRun(args: {
    runId: string;
    status: LocalRunStatus;
    stage: string;
    message: string;
    details?: Record<string, unknown>;
  }): Promise<LocalRunState>;
  completeRun(report: RunReport, artifactDir: string, reportPath: string): Promise<LocalRunState>;
  failRun(runId: string, error: string): Promise<LocalRunState>;
  cancelRun(runId: string): Promise<void>;
  listRuns(): Promise<LocalRunIndexRecord[]>;
  getRun(id: string): Promise<LocalRunState>;
  getRunEvents(id: string): Promise<LocalRunEvent[]>;
  getRunReport(id: string): Promise<RunReport>;
  listModels(): Promise<LocalModelRecord[]>;
  getModel(id: string): Promise<LocalModelRecord>;
  rebuildIndexes(): Promise<void>;
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
    catalogDir: join(root, "catalog"),
    runsCatalog: join(root, "catalog", "runs.jsonl"),
    specsCatalog: join(root, "catalog", "specs.jsonl"),
    modelsCatalog: join(root, "catalog", "models.jsonl"),
    datasetsCatalog: join(root, "catalog", "datasets.jsonl"),
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

async function readJsonlLatestById<T extends { id: string }>(path: string): Promise<T[]> {
  if (!(await exists(path))) return [];
  const rows = (await readFile(path, "utf8"))
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
  return [...new Map(rows.map((row) => [row.id, row])).values()];
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

export function createLocalStore(root = defaultLocalHome()): LocalStore {
  const resolvedRoot = resolve(root);
  const paths = localStorePaths(resolvedRoot);

  const runDir = (id: string) => join(paths.runsDir, id);
  const runStatePath = (id: string) => join(runDir(id), "state.json");
  const runEventsPath = (id: string) => join(runDir(id), "progress.jsonl");
  const runRequestPath = (id: string) => join(runDir(id), "request.json");
  const runReportPath = (id: string) => join(runDir(id), "run-report.json");
  const specDir = (id: string) => join(paths.specsDir, id);
  const specPath = (id: string) => join(specDir(id), "spec.json");
  const modelPath = (id: string) => join(paths.modelsDir, id, "model.json");

  async function ensure() {
    await Promise.all([
      mkdir(paths.specsDir, { recursive: true }),
      mkdir(paths.runsDir, { recursive: true }),
      mkdir(paths.modelsDir, { recursive: true }),
      mkdir(paths.datasetsDir, { recursive: true }),
      mkdir(paths.catalogDir, { recursive: true }),
    ]);
  }

  async function writeRunState(state: LocalRunState): Promise<LocalRunState> {
    await writeJsonAtomic(runStatePath(state.id), state);
    await appendJsonl(paths.runsCatalog, { ...state, catalog_updated_at: new Date().toISOString() });
    return state;
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

  async function resolveRunId(id: string): Promise<string> {
    if (await exists(runStatePath(id))) return id;
    const records = await readJsonlLatestById<LocalRunIndexRecord>(paths.runsCatalog);
    const record = records.find((row) => row.id === id || row.id.startsWith(id));
    if (!record) throw new Error(`Run not found: ${id}`);
    return record.id;
  }

  async function getRun(id: string): Promise<LocalRunState> {
    return readJson<LocalRunState>(runStatePath(await resolveRunId(id)));
  }

  return {
    root: resolvedRoot,
    paths,
    ensure,

    async importSpec(specId, spec) {
      await ensure();
      const now = new Date().toISOString();
      const existingRecord = (await readJsonlLatestById<LocalSpecRecord>(paths.specsCatalog))
        .find((row) => row.id === specId);
      const record: LocalSpecRecord = {
        id: specId,
        name: spec.name,
        base_model: spec.base_model,
        path: specPath(specId),
        created_at: existingRecord?.created_at ?? now,
        updated_at: now,
      };
      await writeJsonAtomic(specPath(specId), spec);
      await appendJsonl(paths.specsCatalog, record);
      return record;
    },

    async listSpecs() {
      await ensure();
      return (await readJsonlLatestById<LocalSpecRecord>(paths.specsCatalog))
        .sort((a, b) => b.updated_at.localeCompare(a.updated_at));
    },

    async getSpec(id) {
      const records = await this.listSpecs();
      const record = records.find((row) => row.id === id || row.id.startsWith(id));
      if (!record) throw new Error(`Spec not found: ${id}`);
      return { ...record, spec: await readJson<SpecSnapshot>(record.path) };
    },

    async startRun({ request, artifactDir }) {
      await ensure();
      await this.importSpec(request.behavior_spec_id, request.spec_snapshot);
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

    async updateRun({ runId, status, stage, message, details }) {
      const previous = await getRun(runId);
      const state: LocalRunState = {
        ...previous,
        status,
        current_stage: stage,
        status_message: message,
        updated_at: new Date().toISOString(),
      };
      await writeRunState(state);
      await appendRunEvent(state, { stage, status: status === "completed" ? "completed" : "running", message, details });
      return state;
    },

    async completeRun(report, artifactDir, reportPath) {
      const previous = await getRun(report.run_id);
      const modelId = `local-${report.run_id}`;
      const completedAt = report.created_at ?? new Date().toISOString();
      const state: LocalRunState = {
        ...previous,
        status: "completed",
        current_stage: "completed",
        status_message: "Run completed successfully.",
        report_path: reportPath,
        model_id: modelId,
        completed_at: completedAt,
        updated_at: completedAt,
      };
      await writeRunState(state);
      await copyIfExists(reportPath, runReportPath(report.run_id));
      await appendRunEvent(state, {
        stage: "completed",
        status: "completed",
        message: "Run completed successfully.",
        details: { report_path: reportPath, model_id: modelId },
      });

      const model: LocalModelRecord = {
        id: modelId,
        run_id: report.run_id,
        behavior_spec_id: report.behavior_spec_id,
        name: `${report.run_metadata?.base_model ?? report.base_model} (${report.run_id.slice(0, 8)})`,
        provider: "local-uv",
        base_model: report.base_model,
        artifact_uri: report.fine_tuned_model_id,
        artifact_dir: artifactDir,
        metrics: report.training.metrics,
        created_at: completedAt,
      };
      await writeJsonAtomic(modelPath(model.id), model);
      await appendJsonl(paths.modelsCatalog, model);
      return state;
    },

    async failRun(runId, error) {
      const previous = await getRun(runId);
      const now = new Date().toISOString();
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
      await appendRunEvent(state, { stage: "failed", status: "failed", message: error });
      return state;
    },

    async cancelRun(runId) {
      const state = await getRun(runId);
      await writeFile(join(runDir(state.id), "cancel.requested"), `${new Date().toISOString()}\n`, "utf8");
      const updated: LocalRunState = {
        ...state,
        status: "cancelled",
        current_stage: "cancel_requested",
        status_message: "Cancellation requested.",
        updated_at: new Date().toISOString(),
      };
      await writeRunState(updated);
      await appendRunEvent(updated, { stage: "cancel_requested", status: "cancelled", message: "Cancellation requested." });
    },

    async listRuns() {
      await ensure();
      return (await readJsonlLatestById<LocalRunIndexRecord>(paths.runsCatalog))
        .sort((a, b) => b.updated_at.localeCompare(a.updated_at));
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
      await ensure();
      return (await readJsonlLatestById<LocalModelRecord>(paths.modelsCatalog))
        .sort((a, b) => b.created_at.localeCompare(a.created_at));
    },

    async getModel(id) {
      const models = await this.listModels();
      const record = models.find((row) => row.id === id || row.id.startsWith(id));
      if (!record) throw new Error(`Model not found: ${id}`);
      return record;
    },

    async rebuildIndexes() {
      await ensure();
      const runRecords: LocalRunIndexRecord[] = [];
      for (const entry of await readdir(paths.runsDir, { withFileTypes: true }).catch(() => [])) {
        if (!entry.isDirectory()) continue;
        const statePath = runStatePath(entry.name);
        if (await exists(statePath)) {
          const state = await readJson<LocalRunState>(statePath);
          runRecords.push({ ...state, catalog_updated_at: new Date().toISOString() });
        }
      }
      const specRecords: LocalSpecRecord[] = [];
      for (const entry of await readdir(paths.specsDir, { withFileTypes: true }).catch(() => [])) {
        if (!entry.isDirectory()) continue;
        const path = specPath(entry.name);
        if (await exists(path)) {
          const spec = await readJson<SpecSnapshot>(path);
          const now = new Date().toISOString();
          specRecords.push({ id: entry.name, name: spec.name, base_model: spec.base_model, path, created_at: now, updated_at: now });
        }
      }
      const modelRecords: LocalModelRecord[] = [];
      for (const entry of await readdir(paths.modelsDir, { withFileTypes: true }).catch(() => [])) {
        if (!entry.isDirectory()) continue;
        const path = modelPath(entry.name);
        if (await exists(path)) modelRecords.push(await readJson<LocalModelRecord>(path));
      }
      await writeFile(paths.runsCatalog, runRecords.map((row) => JSON.stringify(row)).join("\n") + (runRecords.length ? "\n" : ""), "utf8");
      await writeFile(paths.specsCatalog, specRecords.map((row) => JSON.stringify(row)).join("\n") + (specRecords.length ? "\n" : ""), "utf8");
      await writeFile(paths.modelsCatalog, modelRecords.map((row) => JSON.stringify(row)).join("\n") + (modelRecords.length ? "\n" : ""), "utf8");
      await writeFile(paths.datasetsCatalog, "", "utf8");
    },
  };
}
