import { appendFile, readFile } from "node:fs/promises";
import { join } from "node:path";
import { writeJsonAtomic } from "./artifacts.js";
import { runReportSchema } from "./contracts.js";
import type { LocalModelRecord, LocalStore } from "./store.js";

export interface ActiveModelPointer {
  schema_version: 1;
  model_id: string | null;
  run_id: string | null;
  previous_model_id: string | null;
  activated_at: string;
  action: "activate" | "rollback";
}

export interface ActiveModelState {
  pointer: ActiveModelPointer | null;
  model: LocalModelRecord | null;
}

function pointerPath(store: LocalStore): string {
  return join(store.root, "active-model.json");
}

function historyPath(store: LocalStore): string {
  return join(store.root, "activation-history.jsonl");
}

async function readPointer(store: LocalStore): Promise<ActiveModelPointer | null> {
  try {
    const value = JSON.parse(await readFile(pointerPath(store), "utf8")) as ActiveModelPointer;
    if (
      value.schema_version !== 1
      || !["activate", "rollback"].includes(value.action)
      || (value.model_id !== null && typeof value.model_id !== "string")
      || (value.run_id !== null && typeof value.run_id !== "string")
      || (value.previous_model_id !== null && typeof value.previous_model_id !== "string")
    ) {
      throw new Error("Active model pointer has an invalid shape.");
    }
    return value;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw error;
  }
}

async function assertModelCanActivate(
  store: LocalStore,
  modelId: string,
): Promise<LocalModelRecord> {
  const model = await store.getModel(modelId);
  const run = await store.getRun(model.run_id);
  if (run.status !== "completed") {
    throw new Error(`Model ${model.id} belongs to run ${run.id}, which is not completed.`);
  }
  const report = runReportSchema.parse(
    JSON.parse(await readFile(join(model.artifact_dir, "run-report.json"), "utf8")),
  );
  if (!report.general_regression) {
    throw new Error(`Model ${model.id} has no general regression evaluation.`);
  }
  if (!report.general_regression.passed) {
    throw new Error(
      `Model ${model.id} failed general regression: `
      + (report.general_regression.failures.join(" ") || "regression budget exceeded"),
    );
  }
  return model;
}

async function publishPointer(
  store: LocalStore,
  pointer: ActiveModelPointer,
): Promise<ActiveModelPointer> {
  await store.ensure();
  await writeJsonAtomic(pointerPath(store), pointer);
  await appendFile(historyPath(store), `${JSON.stringify(pointer)}\n`, "utf8");
  return pointer;
}

export async function getActiveModel(store: LocalStore): Promise<ActiveModelState> {
  const pointer = await readPointer(store);
  if (!pointer?.model_id) return { pointer, model: null };
  return { pointer, model: await store.getModel(pointer.model_id) };
}

export async function activateModel(
  store: LocalStore,
  modelId: string,
): Promise<ActiveModelPointer> {
  const [model, current] = await Promise.all([
    assertModelCanActivate(store, modelId),
    readPointer(store),
  ]);
  if (current?.model_id === model.id) return current;
  return publishPointer(store, {
    schema_version: 1,
    model_id: model.id,
    run_id: model.run_id,
    previous_model_id: current?.model_id ?? null,
    activated_at: new Date().toISOString(),
    action: "activate",
  });
}

export async function rollbackActiveModel(store: LocalStore): Promise<ActiveModelPointer> {
  const current = await readPointer(store);
  if (!current) throw new Error("No active-model history exists to roll back.");
  const targetId = current.previous_model_id;
  const target = targetId ? await assertModelCanActivate(store, targetId) : null;
  return publishPointer(store, {
    schema_version: 1,
    model_id: target?.id ?? null,
    run_id: target?.run_id ?? null,
    previous_model_id: current.model_id,
    activated_at: new Date().toISOString(),
    action: "rollback",
  });
}
