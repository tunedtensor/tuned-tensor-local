#!/usr/bin/env node

import { realpathSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { cwd } from "node:process";
import { fileURLToPath } from "node:url";
import { compareRuns } from "./compare.js";
import { fineTuneRunRequestSchema, localBehaviorSpecFileSchema, localRunnerConfigSchema, specSnapshotSchema, type LocalRunnerConfig } from "./contracts.js";
import { buildSystemMessage } from "./dataset.js";
import { runLocalLabelingJob } from "./labeling.js";
import {
  loadLocalRunnerConfig,
  runLocalFineTune,
} from "./orchestrator.js";
import { runDoctor } from "./doctor.js";
import { createLocalStore } from "./store.js";
import { serveLocalDashboard } from "./server.js";
import {
  DEFAULT_LOCAL_SPEC_PATH,
  initLocalSpecFile,
  loadLocalRunInput,
  runRequestFromLocalSpec,
} from "./local-project.js";
import { sanitizeLogLine, type LocalRunProgressEvent, type LocalRunReporter } from "./run-reporter.js";

export * from "./compare.js";
export * from "./contracts.js";
export * from "./dataset.js";
export * from "./labeling.js";
export * from "./labeling-sanitize.js";
export * from "./orchestrator.js";
export * from "./local-project.js";
export * from "./openrouter.js";
export * from "./run-reporter.js";
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

/**
 * Parses simple KEY=VALUE lines from .env content. Supports comments, blank
 * lines, an optional `export ` prefix, and single/double quoted values. Does
 * not support multiline values or variable expansion.
 */
export function parseDotEnv(content: string): Record<string, string> {
  const values: Record<string, string> = {};
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const match = line.match(/^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
    if (!match) continue;
    let value = match[2].trim();
    if ((value.startsWith("\"") && value.endsWith("\"")) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    values[match[1]] = value;
  }
  return values;
}

/**
 * Loads .env from the working directory into process.env without overriding
 * variables that are already set, so `OPENROUTER_API_KEY` and friends work
 * out of the box in project directories. Returns the names that were loaded.
 */
async function loadDotEnvFromCwd(): Promise<string[]> {
  let content: string;
  try {
    content = await readFile(resolve(cwd(), ".env"), "utf8");
  } catch {
    return [];
  }
  const loaded: string[] = [];
  for (const [key, value] of Object.entries(parseDotEnv(content))) {
    if (process.env[key] === undefined) {
      process.env[key] = value;
      loaded.push(key);
    }
  }
  return loaded;
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
  init [--name "My Local Model"] [--model Qwen/Qwen3.5-2B] [--output tunedtensor.json] [--force]
  doctor [--config local-runner.json]
  validate [tunedtensor.json|request.json] [--config local-runner.json]
  run [tunedtensor.json|request.json] [--config local-runner.json] [--dry-run] [--verbose] [--quiet]
  label <input.jsonl|input.csv> [--input-column col] [--spec tunedtensor.json] [--system-prompt "..."]
        [--model teacher-model] [--output labeled.jsonl] [--config local-runner.json] [--dry-run]
  serve [--config local-runner.json] [--host 127.0.0.1] [--port 8787]
  runs list|get|events|watch|report|compare|cancel|reconcile [args] [--config local-runner.json]
  models list|get [args] [--config local-runner.json]
  specs list|get|import [args] [--config local-runner.json]
  store rebuild-index [--config local-runner.json]

The run command writes local artifacts under config.artifactRoot, defaulting to
.tt-local/artifacts. The file-backed local store defaults to
~/.tuned-tensor-local unless config.storeRoot or TT_LOCAL_HOME is set.`);
}

const optionNamesWithValues = new Set([
  "--config",
  "--host",
  "--port",
  "--name",
  "--model",
  "--output",
  "--id",
  "--user-id",
  "--run-number",
  "--input-column",
  "--system-prompt",
  "--spec",
  "--format",
]);

function readPositionals(argv: string[], startIndex = 3): string[] {
  const positionals: string[] = [];
  for (let index = startIndex; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg) continue;
    if (optionNamesWithValues.has(arg)) {
      index += 1;
      continue;
    }
    if (arg.startsWith("--")) continue;
    positionals.push(arg);
  }
  return positionals;
}

function readNumberOption(argv: string[], name: string): number | undefined {
  const value = readOption(argv, name);
  return value ? Number(value) : undefined;
}

async function configFromArgv(argv: string[]): Promise<LocalRunnerConfig> {
  const configPath = readOption(argv, "--config");
  return loadLocalRunnerConfig(configPath ? resolve(configPath) : undefined);
}

function printJson(value: unknown): void {
  console.log(JSON.stringify(value, null, 2));
}

function shortValue(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) return `[${value.length} items]`;
  return null;
}

function formatEvent(event: LocalRunProgressEvent): string {
  const detailText = Object.entries(event.details ?? {})
    .filter(([key]) => key !== "metrics")
    .map(([key, value]) => {
      const formatted = key === "command" && Array.isArray(value)
        ? value.join(" ")
        : shortValue(value);
      return formatted ? `${key}=${formatted}` : null;
    })
    .filter((value): value is string => Boolean(value))
    .slice(0, 5)
    .join(" ");
  return sanitizeLogLine(`[tt-local] ${event.stage}: ${event.message}${detailText ? ` (${detailText})` : ""}`);
}

function createConsoleReporter(options: { verbose: boolean; quiet: boolean }): LocalRunReporter | undefined {
  if (options.quiet) return undefined;
  let lastLogLine = "";
  return {
    verbose: options.verbose,
    onEvent(event) {
      process.stderr.write(`${formatEvent(event)}\n`);
    },
    onLog(log) {
      const line = sanitizeLogLine(`[tt-local] ${log.stage}${log.stream ? ` ${log.stream}` : ""}: ${log.message}`);
      // tqdm redraws the same progress line several times per step; collapse
      // consecutive duplicates so --verbose output stays readable.
      if (line === lastLogLine) return;
      lastLogLine = line;
      process.stderr.write(`${line}\n`);
    },
  };
}

async function sleep(ms: number): Promise<void> {
  await new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

function isTerminalStatus(status: string): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

async function main(argv: string[]): Promise<void> {
  const command = argv[2] ?? "info";
  const loadedEnv = await loadDotEnvFromCwd();
  if (loadedEnv.length > 0 && !hasFlag(argv, "--quiet")) {
    process.stderr.write(`[tt-local] loaded ${loadedEnv.join(", ")} from .env\n`);
  }

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

  if (command === "init") {
    const outputPath = resolve(readOption(argv, "--output") ?? DEFAULT_LOCAL_SPEC_PATH);
    const spec = await initLocalSpecFile({
      outputPath,
      name: readOption(argv, "--name") ?? "Local Tuned Tensor Spec",
      baseModel: readOption(argv, "--model") ?? "Qwen/Qwen3.5-2B",
      force: hasFlag(argv, "--force"),
    });
    printJson({
      ok: true,
      path: outputPath,
      id: spec.id,
      name: spec.name,
      base_model: spec.base_model,
    });
    return;
  }

  if (command === "validate") {
    const inputPath = resolve(readPositionals(argv)[0] ?? DEFAULT_LOCAL_SPEC_PATH);
    const input = await loadLocalRunInput(inputPath, {
      userId: readOption(argv, "--user-id"),
      runNumber: readNumberOption(argv, "--run-number"),
    });
    const request = input.request;
    const config = await configFromArgv(argv);
    printJson({
      ok: true,
      input_kind: input.kind,
      input_path: input.path,
      run_id: request.run_id,
      behavior_spec_id: request.behavior_spec_id,
      base_model: request.spec_snapshot.base_model,
      artifact_root: config.artifactRoot,
      store_root: config.storeRoot,
      dry_run: config.dryRun,
      ...(input.warnings.length > 0 ? { warnings: input.warnings } : {}),
    });
    return;
  }

  if (command === "run") {
    const inputPath = resolve(readPositionals(argv)[0] ?? DEFAULT_LOCAL_SPEC_PATH);
    const configInput = await configFromArgv(argv);
    const config = localRunnerConfigSchema.parse({
      ...configInput,
      dryRun: hasFlag(argv, "--dry-run") ? true : configInput.dryRun,
    });
    const input = await loadLocalRunInput(inputPath, {
      userId: readOption(argv, "--user-id"),
      runNumber: readNumberOption(argv, "--run-number"),
    });
    if (!hasFlag(argv, "--quiet")) {
      for (const warning of input.warnings) {
        process.stderr.write(`[tt-local] warning: ${warning}\n`);
      }
    }
    const request = input.request;
    const result = await runLocalFineTune({
      request,
      config,
      reporter: createConsoleReporter({
        verbose: hasFlag(argv, "--verbose"),
        quiet: hasFlag(argv, "--quiet"),
      }),
    });
    printJson({
      status: result.report.status,
      input_kind: input.kind,
      run_id: result.report.run_id,
      behavior_spec_id: result.report.behavior_spec_id,
      report_path: result.reportPath,
      artifact_dir: result.artifactDir,
      model_id: `local-${result.report.run_id}`,
      fine_tuned_model_id: result.report.fine_tuned_model_id,
      training_log: result.report.training.log_uri,
      baseline_eval: result.report.artifact_uris.baseline_eval,
      candidate_eval: result.report.artifact_uris.candidate_eval,
      comparison: result.report.comparison,
    });
    return;
  }

  if (command === "label") {
    const sourcePath = readPositionals(argv)[0];
    if (!sourcePath) throw new Error("label requires <input.jsonl|input.csv>");
    const configInput = await configFromArgv(argv);
    const config = localRunnerConfigSchema.parse({
      ...configInput,
      dryRun: hasFlag(argv, "--dry-run") ? true : configInput.dryRun,
    });
    const format = readOption(argv, "--format");
    if (format !== undefined && format !== "jsonl" && format !== "csv") {
      throw new Error(`--format must be jsonl or csv, got: ${format}`);
    }
    let systemMessage = readOption(argv, "--system-prompt");
    if (!systemMessage) {
      const specPath = resolve(readOption(argv, "--spec") ?? DEFAULT_LOCAL_SPEC_PATH);
      let raw: string;
      try {
        raw = await readFile(specPath, "utf8");
      } catch {
        throw new Error(`Cannot read spec at ${specPath}. Pass --spec <tunedtensor.json> or --system-prompt "..." so the teacher labels under the model's instructions.`);
      }
      systemMessage = buildSystemMessage(localBehaviorSpecFileSchema.parse(JSON.parse(raw)));
    }
    const result = await runLocalLabelingJob({
      sourcePath: resolve(sourcePath),
      format,
      inputColumn: readOption(argv, "--input-column"),
      systemMessage,
      config,
      outputPath: readOption(argv, "--output"),
      model: readOption(argv, "--model"),
      reporter: createConsoleReporter({
        verbose: hasFlag(argv, "--verbose"),
        quiet: hasFlag(argv, "--quiet"),
      }),
    });
    printJson(result.report);
    if (result.report.status === "failed") process.exitCode = 1;
    return;
  }

  if (command === "serve") {
    const config = await configFromArgv(argv);
    const host = readOption(argv, "--host") ?? "127.0.0.1";
    const port = Number(readOption(argv, "--port") ?? "8787");
    const dashboard = await serveLocalDashboard({ host, port, config });
    console.log(`TT Local dashboard: ${dashboard.url}`);
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
    if (subcommand === "compare") {
      const idA = argv[4];
      const idB = argv[5];
      if (!idA || !idB) throw new Error("runs compare requires <run-id-a> <run-id-b>");
      const [reportA, reportB] = await Promise.all([
        store.getRunReport(idA),
        store.getRunReport(idB),
      ]);
      return printJson(compareRuns(reportA, reportB));
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
      const localSpec = localBehaviorSpecFileSchema.safeParse(input);
      if (localSpec.success) {
        const localRequest = runRequestFromLocalSpec(localSpec.data, {
          userId: readOption(argv, "--user-id"),
          runNumber: readNumberOption(argv, "--run-number"),
        });
        return printJson(await store.importSpec(localRequest.behavior_spec_id, localRequest.spec_snapshot));
      }
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

function isCliEntrypoint(): boolean {
  if (!process.argv[1]) return false;
  return realpathSync(fileURLToPath(import.meta.url)) === realpathSync(process.argv[1]);
}

if (isCliEntrypoint()) {
  main(process.argv).catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
