import { appendFile, copyFile, mkdir, readdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { randomUUID } from "node:crypto";
import { createRequire } from "node:module";
import type BetterSqlite3 from "better-sqlite3";
import type { FineTuneRunRequest, RunReport, SpecSnapshot } from "./contracts.js";

const require = createRequire(import.meta.url);
const BetterSqlite = require("better-sqlite3") as typeof BetterSqlite3;

type MetadataDb = BetterSqlite3.Database;
type MetadataStatement = BetterSqlite3.Statement;

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
    metadataDb: join(root, "metadata.sqlite"),
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

type RunRow = {
  id: string;
  behavior_spec_id: string;
  user_id: string;
  run_number: number;
  status: LocalRunStatus;
  current_stage: string;
  status_message: string;
  artifact_dir: string;
  report_path: string | null;
  model_id: string | null;
  error: string | null;
  base_model: string;
  spec_name: string;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  catalog_updated_at: string;
};

type SpecRow = {
  id: string;
  name: string;
  base_model: string;
  path: string;
  created_at: string;
  updated_at: string;
};

type ModelRow = {
  id: string;
  run_id: string;
  behavior_spec_id: string;
  name: string;
  provider: "local-uv";
  base_model: string;
  artifact_uri: string;
  artifact_dir: string;
  metrics_json: string | null;
  created_at: string;
};

type EventRow = {
  id: string;
  run_id: string;
  stage: string;
  status: LocalRunStatus | "running" | "completed" | "failed";
  message: string;
  details_json: string | null;
  occurred_at: string;
};

function initMetadataDb(db: MetadataDb): void {
  db.exec(`
    PRAGMA journal_mode = WAL;
    PRAGMA foreign_keys = ON;
    PRAGMA busy_timeout = 5000;

    CREATE TABLE IF NOT EXISTS store_meta (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS specs (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      base_model TEXT NOT NULL,
      path TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS runs (
      id TEXT PRIMARY KEY,
      behavior_spec_id TEXT NOT NULL,
      user_id TEXT NOT NULL,
      run_number INTEGER NOT NULL,
      status TEXT NOT NULL,
      current_stage TEXT NOT NULL,
      status_message TEXT NOT NULL,
      artifact_dir TEXT NOT NULL,
      report_path TEXT,
      model_id TEXT,
      error TEXT,
      base_model TEXT NOT NULL,
      spec_name TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      started_at TEXT,
      completed_at TEXT,
      catalog_updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS run_events (
      id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      stage TEXT NOT NULL,
      status TEXT NOT NULL,
      message TEXT NOT NULL,
      details_json TEXT,
      occurred_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS models (
      id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      behavior_spec_id TEXT NOT NULL,
      name TEXT NOT NULL,
      provider TEXT NOT NULL,
      base_model TEXT NOT NULL,
      artifact_uri TEXT NOT NULL,
      artifact_dir TEXT NOT NULL,
      metrics_json TEXT,
      created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_runs_updated_at ON runs(updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_specs_updated_at ON specs(updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_run_events_run_time ON run_events(run_id, occurred_at);
  `);
}

function rowToRunState(row: RunRow): LocalRunState {
  return {
    id: row.id,
    behavior_spec_id: row.behavior_spec_id,
    user_id: row.user_id,
    run_number: row.run_number,
    status: row.status,
    current_stage: row.current_stage,
    status_message: row.status_message,
    artifact_dir: row.artifact_dir,
    base_model: row.base_model,
    spec_name: row.spec_name,
    created_at: row.created_at,
    updated_at: row.updated_at,
    ...(row.report_path ? { report_path: row.report_path } : {}),
    ...(row.model_id ? { model_id: row.model_id } : {}),
    ...(row.error ? { error: row.error } : {}),
    ...(row.started_at ? { started_at: row.started_at } : {}),
    ...(row.completed_at ? { completed_at: row.completed_at } : {}),
  };
}

function rowToRunIndex(row: RunRow): LocalRunIndexRecord {
  return { ...rowToRunState(row), catalog_updated_at: row.catalog_updated_at };
}

function rowToSpec(row: SpecRow): LocalSpecRecord {
  return {
    id: row.id,
    name: row.name,
    base_model: row.base_model,
    path: row.path,
    created_at: row.created_at,
    updated_at: row.updated_at,
  };
}

function rowToModel(row: ModelRow): LocalModelRecord {
  return {
    id: row.id,
    run_id: row.run_id,
    behavior_spec_id: row.behavior_spec_id,
    name: row.name,
    provider: row.provider,
    base_model: row.base_model,
    artifact_uri: row.artifact_uri,
    artifact_dir: row.artifact_dir,
    metrics: row.metrics_json ? JSON.parse(row.metrics_json) as Record<string, unknown> : null,
    created_at: row.created_at,
  };
}

function rowToEvent(row: EventRow): LocalRunEvent {
  return {
    id: row.id,
    run_id: row.run_id,
    stage: row.stage,
    status: row.status,
    message: row.message,
    occurred_at: row.occurred_at,
    ...(row.details_json ? { details: JSON.parse(row.details_json) as Record<string, unknown> } : {}),
  };
}

function withMetadataDb<T>(paths: ReturnType<typeof localStorePaths>, fn: (db: MetadataDb) => T): T {
  const db = new BetterSqlite(paths.metadataDb);
  try {
    initMetadataDb(db);
    return fn(db);
  } finally {
    db.close();
  }
}

function runTransaction<T>(db: MetadataDb, fn: () => T): T {
  db.exec("BEGIN IMMEDIATE");
  try {
    const result = fn();
    db.exec("COMMIT");
    return result;
  } catch (error) {
    db.exec("ROLLBACK");
    throw error;
  }
}

function bindRunState(stmt: MetadataStatement, state: LocalRunState, catalogUpdatedAt: string): void {
  stmt.run(
    state.id,
    state.behavior_spec_id,
    state.user_id,
    state.run_number,
    state.status,
    state.current_stage,
    state.status_message,
    state.artifact_dir,
    state.report_path ?? null,
    state.model_id ?? null,
    state.error ?? null,
    state.base_model,
    state.spec_name,
    state.created_at,
    state.updated_at,
    state.started_at ?? null,
    state.completed_at ?? null,
    catalogUpdatedAt,
  );
}

function upsertRun(db: MetadataDb, state: LocalRunState, catalogUpdatedAt = new Date().toISOString()): void {
  bindRunState(db.prepare(`
    INSERT INTO runs (
      id, behavior_spec_id, user_id, run_number, status, current_stage,
      status_message, artifact_dir, report_path, model_id, error, base_model,
      spec_name, created_at, updated_at, started_at, completed_at, catalog_updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      behavior_spec_id = excluded.behavior_spec_id,
      user_id = excluded.user_id,
      run_number = excluded.run_number,
      status = excluded.status,
      current_stage = excluded.current_stage,
      status_message = excluded.status_message,
      artifact_dir = excluded.artifact_dir,
      report_path = excluded.report_path,
      model_id = excluded.model_id,
      error = excluded.error,
      base_model = excluded.base_model,
      spec_name = excluded.spec_name,
      created_at = excluded.created_at,
      updated_at = excluded.updated_at,
      started_at = excluded.started_at,
      completed_at = excluded.completed_at,
      catalog_updated_at = excluded.catalog_updated_at
  `), state, catalogUpdatedAt);
}

function upsertSpec(db: MetadataDb, record: LocalSpecRecord): void {
  db.prepare(`
    INSERT INTO specs (id, name, base_model, path, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      name = excluded.name,
      base_model = excluded.base_model,
      path = excluded.path,
      created_at = excluded.created_at,
      updated_at = excluded.updated_at
  `).run(record.id, record.name, record.base_model, record.path, record.created_at, record.updated_at);
}

function upsertModel(db: MetadataDb, model: LocalModelRecord): void {
  db.prepare(`
    INSERT INTO models (
      id, run_id, behavior_spec_id, name, provider, base_model,
      artifact_uri, artifact_dir, metrics_json, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      run_id = excluded.run_id,
      behavior_spec_id = excluded.behavior_spec_id,
      name = excluded.name,
      provider = excluded.provider,
      base_model = excluded.base_model,
      artifact_uri = excluded.artifact_uri,
      artifact_dir = excluded.artifact_dir,
      metrics_json = excluded.metrics_json,
      created_at = excluded.created_at
  `).run(
    model.id,
    model.run_id,
    model.behavior_spec_id,
    model.name,
    model.provider,
    model.base_model,
    model.artifact_uri,
    model.artifact_dir,
    model.metrics ? JSON.stringify(model.metrics) : null,
    model.created_at,
  );
}

function insertEvent(db: MetadataDb, event: LocalRunEvent): void {
  db.prepare(`
    INSERT OR IGNORE INTO run_events (id, run_id, stage, status, message, details_json, occurred_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `).run(
    event.id,
    event.run_id,
    event.stage,
    event.status,
    event.message,
    event.details ? JSON.stringify(event.details) : null,
    event.occurred_at,
  );
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

  async function importLegacyCatalogsIfNeeded() {
    const imported = withMetadataDb(paths, (db) =>
      (db.prepare("SELECT value FROM store_meta WHERE key = ?").get("legacy_catalogs_imported") as { value?: string } | undefined)?.value
    );
    if (imported === "true") return;

    const [runs, specs, models] = await Promise.all([
      readJsonlLatestById<LocalRunIndexRecord>(paths.runsCatalog),
      readJsonlLatestById<LocalSpecRecord>(paths.specsCatalog),
      readJsonlLatestById<LocalModelRecord>(paths.modelsCatalog),
    ]);

    withMetadataDb(paths, (db) => runTransaction(db, () => {
      for (const run of runs) upsertRun(db, run, run.catalog_updated_at);
      for (const spec of specs) upsertSpec(db, spec);
      for (const model of models) upsertModel(db, model);
      db.prepare("INSERT OR REPLACE INTO store_meta (key, value) VALUES (?, ?)").run("legacy_catalogs_imported", "true");
    }));
  }

  async function ensure() {
    await Promise.all([
      mkdir(paths.specsDir, { recursive: true }),
      mkdir(paths.runsDir, { recursive: true }),
      mkdir(paths.modelsDir, { recursive: true }),
      mkdir(paths.datasetsDir, { recursive: true }),
      mkdir(paths.catalogDir, { recursive: true }),
    ]);
    withMetadataDb(paths, () => undefined);
    await importLegacyCatalogsIfNeeded();
  }

  async function writeRunState(state: LocalRunState): Promise<LocalRunState> {
    const catalogUpdatedAt = new Date().toISOString();
    await writeJsonAtomic(runStatePath(state.id), state);
    await appendJsonl(paths.runsCatalog, { ...state, catalog_updated_at: catalogUpdatedAt });
    withMetadataDb(paths, (db) => upsertRun(db, state, catalogUpdatedAt));
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
    withMetadataDb(paths, (db) => insertEvent(db, row));
  }

  async function resolveRunId(id: string): Promise<string> {
    await ensure();
    if (await exists(runStatePath(id))) return id;
    const records = withMetadataDb(paths, (db) => db.prepare(`
      SELECT * FROM runs WHERE id = ? OR id LIKE ? ORDER BY updated_at DESC
    `).all(id, `${id}%`) as RunRow[]).map(rowToRunIndex);
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
      const existingRecord = withMetadataDb(paths, (db) =>
        db.prepare("SELECT * FROM specs WHERE id = ?").get(specId) as SpecRow | undefined
      );
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
      withMetadataDb(paths, (db) => upsertSpec(db, record));
      return record;
    },

    async listSpecs() {
      await ensure();
      return withMetadataDb(paths, (db) =>
        (db.prepare("SELECT * FROM specs ORDER BY updated_at DESC").all() as SpecRow[]).map(rowToSpec)
      );
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
      withMetadataDb(paths, (db) => upsertModel(db, model));
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
      return withMetadataDb(paths, (db) =>
        (db.prepare("SELECT * FROM runs ORDER BY updated_at DESC").all() as RunRow[]).map(rowToRunIndex)
      );
    },

    getRun,

    async getRunEvents(id) {
      const state = await getRun(id);
      const events = withMetadataDb(paths, (db) => db.prepare(`
        SELECT * FROM run_events WHERE run_id = ? ORDER BY occurred_at ASC
      `).all(state.id) as EventRow[]).map(rowToEvent);
      if (events.length > 0) return events;
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
      return withMetadataDb(paths, (db) =>
        (db.prepare("SELECT * FROM models ORDER BY created_at DESC").all() as ModelRow[]).map(rowToModel)
      );
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
      const eventRecords: LocalRunEvent[] = [];
      for (const entry of await readdir(paths.runsDir, { withFileTypes: true }).catch(() => [])) {
        if (!entry.isDirectory()) continue;
        const statePath = runStatePath(entry.name);
        if (await exists(statePath)) {
          const state = await readJson<LocalRunState>(statePath);
          runRecords.push({ ...state, catalog_updated_at: new Date().toISOString() });
        }
        eventRecords.push(...await readJsonl<LocalRunEvent>(runEventsPath(entry.name)));
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
      withMetadataDb(paths, (db) => runTransaction(db, () => {
        db.exec("DELETE FROM run_events; DELETE FROM runs; DELETE FROM specs; DELETE FROM models;");
        for (const run of runRecords) upsertRun(db, run, run.catalog_updated_at);
        for (const spec of specRecords) upsertSpec(db, spec);
        for (const model of modelRecords) upsertModel(db, model);
        for (const event of eventRecords) insertEvent(db, event);
        db.prepare("INSERT OR REPLACE INTO store_meta (key, value) VALUES (?, ?)").run("legacy_catalogs_imported", "true");
      }));
    },
  };
}
