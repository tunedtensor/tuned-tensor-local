#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fineTuneRunRequestSchema, localRunnerConfigSchema, specSnapshotSchema, type LocalRunnerConfig } from "./contracts.js";
import {
  loadLocalRunnerConfig,
  loadRunRequest,
  runLocalFineTune,
} from "./orchestrator.js";
import { runDoctor } from "./doctor.js";
import { createLocalStore } from "./store.js";
import { serveLocalDashboard } from "./server.js";

export * from "./contracts.js";
export * from "./dataset.js";
export * from "./orchestrator.js";
export * from "./openrouter.js";
export * from "./server.js";
export * from "./store.js";

export interface LocalRunnerInfo {
  name: "tuned-tensor-local";
  status: "local-runner-preview";
  description: string;
}

export function getLocalRunnerInfo(): LocalRunnerInfo {
  return {
    name: "tuned-tensor-local",
    status: "local-runner-preview",
    description: "Local fine-tuning runner for single-GPU uv/Python hosts.",
  };
}

function readOption(argv: string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  if (index === -1) return undefined;
  return argv[index + 1];
}

function hasFlag(argv: string[], name: string): boolean {
  return argv.includes(name);
}

function printHelp(): void {
  console.log(`tt-local

Commands:
  info
  doctor [--config local-runner.json]
  validate <request.json> [--config local-runner.json]
  run <request.json> [--config local-runner.json] [--dry-run]
  serve [--config local-runner.json] [--host 127.0.0.1] [--port 8787]
  runs list|get|events|watch|report|cancel|reconcile [args] [--config local-runner.json]
  models list|get [args] [--config local-runner.json]
  specs list|get|import [args] [--config local-runner.json]
  store rebuild-index [--config local-runner.json]

The run command writes local artifacts under config.artifactRoot, defaulting to
.tt-local/artifacts. The file-backed local store defaults to
~/.tuned-tensor-local unless config.storeRoot or TT_LOCAL_HOME is set.`);
}

async function configFromArgv(argv: string[]): Promise<LocalRunnerConfig> {
  const configPath = readOption(argv, "--config");
  return loadLocalRunnerConfig(configPath ? resolve(configPath) : undefined);
}

function printJson(value: unknown): void {
  console.log(JSON.stringify(value, null, 2));
}

async function sleep(ms: number): Promise<void> {
  await new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

function isTerminalStatus(status: string): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

async function main(argv: string[]): Promise<void> {
  const command = argv[2] ?? "info";

  if (command === "info" || command === "--help" || command === "-h") {
    const info = getLocalRunnerInfo();
    console.log(`${info.name}: ${info.description}`);
    console.log(`Status: ${info.status}`);
    if (command !== "info") printHelp();
    return;
  }

  if (command === "doctor") {
    const config = await configFromArgv(argv);
    const checks = await runDoctor(config);
    const ok = checks.every((check) => check.ok);
    printJson({ ok, checks });
    if (!ok) process.exitCode = 1;
    return;
  }

  if (command === "validate") {
    const requestPath = argv[3];
    if (!requestPath) throw new Error("validate requires <request.json>");
    const request = await loadRunRequest(resolve(requestPath));
    const config = await configFromArgv(argv);
    printJson({
      ok: true,
      run_id: request.run_id,
      base_model: request.spec_snapshot.base_model,
      artifact_root: config.artifactRoot,
      store_root: config.storeRoot,
      dry_run: config.dryRun,
    });
    return;
  }

  if (command === "run") {
    const requestPath = argv[3];
    if (!requestPath) throw new Error("run requires <request.json>");
    const configInput = await configFromArgv(argv);
    const config = localRunnerConfigSchema.parse({
      ...configInput,
      dryRun: hasFlag(argv, "--dry-run") ? true : configInput.dryRun,
    });
    const request = await loadRunRequest(resolve(requestPath));
    const result = await runLocalFineTune({ request, config });
    printJson({
      status: result.report.status,
      run_id: result.report.run_id,
      report_path: result.reportPath,
      artifact_dir: result.artifactDir,
      avg_score_delta: result.report.comparison.avg_score_delta,
    });
    return;
  }

  if (command === "serve") {
    const config = await configFromArgv(argv);
    const host = readOption(argv, "--host") ?? "127.0.0.1";
    const port = Number(readOption(argv, "--port") ?? "8787");
    const dashboard = await serveLocalDashboard({ host, port, config });
    console.log(`Tuned Tensor Local dashboard: ${dashboard.url}`);
    await new Promise<void>((resolveStop) => {
      const stop = () => {
        dashboard.close().then(resolveStop, resolveStop);
      };
      process.once("SIGINT", stop);
      process.once("SIGTERM", stop);
    });
    return;
  }

  if (command === "runs") {
    const subcommand = argv[3] ?? "list";
    const config = await configFromArgv(argv);
    const store = createLocalStore(config.storeRoot);
    if (subcommand === "list") return printJson(await store.listRuns());
    if (subcommand === "get") {
      const id = argv[4];
      if (!id) throw new Error("runs get requires <run-id>");
      return printJson(await store.getRun(id));
    }
    if (subcommand === "events") {
      const id = argv[4];
      if (!id) throw new Error("runs events requires <run-id>");
      return printJson(await store.getRunEvents(id));
    }
    if (subcommand === "report") {
      const id = argv[4];
      if (!id) throw new Error("runs report requires <run-id>");
      return printJson(await store.getRunReport(id));
    }
    if (subcommand === "cancel") {
      const id = argv[4];
      if (!id) throw new Error("runs cancel requires <run-id>");
      await store.cancelRun(id);
      return printJson({ ok: true, run_id: id });
    }
    if (subcommand === "watch") {
      const id = argv[4];
      if (!id) throw new Error("runs watch requires <run-id>");
      const printed = new Set<string>();
      while (true) {
        const events = await store.getRunEvents(id);
        for (const event of events) {
          if (printed.has(event.id)) continue;
          printed.add(event.id);
          console.log(`${event.occurred_at} ${event.stage} ${event.status}: ${event.message}`);
        }
        const run = await store.getRun(id);
        if (isTerminalStatus(run.status)) return;
        await sleep(1000);
      }
    }
    if (subcommand === "reconcile") {
      await store.rebuildIndexes();
      return printJson({ ok: true });
    }
    throw new Error(`Unknown runs command: ${subcommand}`);
  }

  if (command === "models") {
    const subcommand = argv[3] ?? "list";
    const config = await configFromArgv(argv);
    const store = createLocalStore(config.storeRoot);
    if (subcommand === "list") return printJson(await store.listModels());
    if (subcommand === "get") {
      const id = argv[4];
      if (!id) throw new Error("models get requires <model-id>");
      return printJson(await store.getModel(id));
    }
    throw new Error(`Unknown models command: ${subcommand}`);
  }

  if (command === "specs") {
    const subcommand = argv[3] ?? "list";
    const config = await configFromArgv(argv);
    const store = createLocalStore(config.storeRoot);
    if (subcommand === "list") return printJson(await store.listSpecs());
    if (subcommand === "get") {
      const id = argv[4];
      if (!id) throw new Error("specs get requires <spec-id>");
      return printJson(await store.getSpec(id));
    }
    if (subcommand === "import") {
      const path = argv[4];
      if (!path) throw new Error("specs import requires <spec-or-request.json>");
      const input = JSON.parse(await readFile(resolve(path), "utf8")) as unknown;
      const request = fineTuneRunRequestSchema.safeParse(input);
      if (request.success) return printJson(await store.importSpec(request.data.behavior_spec_id, request.data.spec_snapshot));
      const id = readOption(argv, "--id");
      if (!id) throw new Error("specs import requires --id when importing a raw spec snapshot");
      return printJson(await store.importSpec(id, specSnapshotSchema.parse(input)));
    }
    throw new Error(`Unknown specs command: ${subcommand}`);
  }

  if (command === "store") {
    const subcommand = argv[3];
    const config = await configFromArgv(argv);
    const store = createLocalStore(config.storeRoot);
    if (subcommand === "rebuild-index") {
      await store.rebuildIndexes();
      return printJson({ ok: true });
    }
    throw new Error(`Unknown store command: ${subcommand ?? ""}`);
  }

  console.error(`Unknown command: ${command}`);
  process.exitCode = 1;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main(process.argv).catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
