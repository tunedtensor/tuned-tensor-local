#!/usr/bin/env node

import { readFileSync, realpathSync } from "node:fs";
import { createHash } from "node:crypto";
import { mkdir, open, readFile, stat } from "node:fs/promises";
import { spawn } from "node:child_process";
import { basename, dirname, join, resolve } from "node:path";
import { cwd } from "node:process";
import { fileURLToPath } from "node:url";
import { compareRuns } from "./compare.js";
import { assertArtifactManifest, claimRunArtifactDirectory, defaultArtifactPrefix, resolveRunArtifacts } from "./artifacts.js";
import { fineTuneRunRequestSchema, localBehaviorSpecFileSchema, localRunnerConfigSchema, specSnapshotSchema, type FineTuneRunRequest, type LocalRunnerConfig, type SpecSnapshot } from "./contracts.js";
import { buildSystemMessage } from "./dataset.js";
import { assertEvaluationScoringReady } from "./evaluation.js";
import { runLocalLabelingJob } from "./labeling.js";
import {
  loadLocalRunnerConfig,
  fingerprintLocalBaseModel,
  runLocalFineTuneStage,
  type LocalRunStage,
} from "./orchestrator.js";
import { runDoctor } from "./doctor.js";
import { assertUsableModelArtifact, resolveTrainingModel } from "./model-registry.js";
import { buildLocalModelServerLaunch, serveLocalModel } from "./model-server.js";
import { prefetchBaseModel } from "./prefetch.js";
import { createLocalStore, isTerminalRunState, type LocalModelRecord, type LocalStore } from "./store.js";
import { serveLocalDashboard } from "./server.js";
import {
  DEFAULT_LOCAL_SPEC_PATH,
  assertLocalRunInputReady,
  initLocalRunnerConfigFile,
  initLocalSpecFile,
  loadLocalRunInput,
  runRequestFromLocalSpec,
} from "./local-project.js";
import { sanitizeLogLine, type LocalRunProgressEvent, type LocalRunReporter } from "./run-reporter.js";
import {
  validateStudyBenchmark,
  writeStudyBenchmarkLock,
} from "./studies.js";
import { promoteStudyTrialCandidate } from "./study-candidates.js";
import { runStudyTrial } from "./study-trials.js";

export * from "./compare.js";
export * from "./contracts.js";
export * from "./dataset.js";
export * from "./labeling.js";
export * from "./labeling-sanitize.js";
export * from "./model-server.js";
export * from "./orchestrator.js";
export * from "./local-project.js";
export * from "./openrouter.js";
export * from "./prefetch.js";
export * from "./run-reporter.js";
export * from "./server.js";
export * from "./store.js";
export * from "./study-metrics.js";
export * from "./study-candidates.js";
export * from "./study-trials.js";
export * from "./studies.js";

export interface LocalRunnerInfo {
  name: "tuned-tensor-local";
  status: "local-runner-preview";
  description: string;
  version: string;
}

function packageVersion(): string {
  try {
    const packageJson = JSON.parse(
      readFileSync(fileURLToPath(new URL("../package.json", import.meta.url)), "utf8"),
    ) as { version?: unknown };
    return typeof packageJson.version === "string" ? packageJson.version : "unknown";
  } catch {
    return "unknown";
  }
}

export const TT_LOCAL_VERSION = packageVersion();

export function getLocalRunnerInfo(): LocalRunnerInfo {
  return {
    name: "tuned-tensor-local",
    status: "local-runner-preview",
    description: "Local-first fine-tuning with baseline-vs-tuned evaluation for small open-weight models.",
    version: TT_LOCAL_VERSION,
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
  if (index !== -1) return argv[index + 1];
  const inline = argv.find((value) => value.startsWith(`${name}=`));
  return inline?.slice(name.length + 1);
}

function hasFlag(argv: string[], name: string): boolean {
  return argv.includes(name);
}

function printHelp(): void {
  console.log(`Usage: tt-local <command> [options]

Commands:
  info                              Show package and runner information
  init [--name "My Local Model"] [--model Qwen/Qwen3.5-2B] [--output tunedtensor.json] [--profile spark] [--force]
  doctor [tunedtensor.json|request.json] [--config local-runner.json]
  validate [tunedtensor.json|request.json] [--config local-runner.json]
  run [tunedtensor.json|request.json] [--config local-runner.json] [--stage all|prepare|baseline|train|candidate|score|report]
      [--run-id uuid] [--parent-model local-model-id] [--parent-model-artifact path-or-file-uri]
      [--model-artifact path-or-file-uri] [--force] [--dry-run] [--detach] [--verbose] [--quiet]
  label <input.jsonl|input.csv> [--input-column col] [--spec tunedtensor.json] [--system-prompt "..."]
        [--model teacher-model] [--output labeled.jsonl] [--config local-runner.json] [--dry-run]
  serve [--config local-runner.json] [--host 127.0.0.1] [--port 8787]
  runs list|get|events|watch|report|compare|cancel|reconcile [args] [--config local-runner.json]
  models list|get|verify|prefetch|verify-base|serve [args] [--config local-runner.json]
  specs list|get|import [args] [--config local-runner.json]
  studies lock|validate|run|promote <study.json> [args]
  store rebuild-index [--config local-runner.json]

Global options:
  -h, --help                       Show help
  -V, --version                    Show the installed version

The run command writes local artifacts under config.artifactRoot, defaulting to
.tt-local/artifacts. The file-backed local store defaults to
~/.tuned-tensor-local unless config.storeRoot or TT_LOCAL_HOME is set.`);
}

interface CliOptionDefinition {
  name: string;
  value?: string;
  description: string;
}

interface CliCommandDefinition {
  usage: string;
  description: string;
  options: readonly CliOptionDefinition[];
  minPositionals?: number;
  maxPositionals?: number;
  missingPositionalsMessage?: string;
}

interface CliCommandGroup {
  description: string;
  defaultSubcommand?: string;
  subcommands: Record<string, CliCommandDefinition>;
}

interface ParsedCli {
  command: string;
  subcommand?: string;
  positionals: string[];
  help: "top" | "command" | "group";
  definition?: CliCommandDefinition;
}

const CONFIG_OPTION = { name: "--config", value: "path", description: "Local runner config JSON path" } as const;
const USER_ID_OPTION = { name: "--user-id", value: "id", description: "Override the local user ID" } as const;
const RUN_NUMBER_OPTION = { name: "--run-number", value: "number", description: "Override the local run number" } as const;
const VERBOSE_OPTION = { name: "--verbose", description: "Stream subprocess output" } as const;
const QUIET_OPTION = { name: "--quiet", description: "Suppress progress output on stderr" } as const;

const COMMAND_DEFINITIONS: Record<string, CliCommandDefinition> = {
  info: {
    usage: "tt-local info",
    description: "Show the installed TT Local version and runner status.",
    options: [],
    maxPositionals: 0,
  },
  init: {
    usage: "tt-local init [options]",
    description: "Create a local tunedtensor.json behavior spec.",
    options: [
      { name: "--name", value: "name", description: "Behavior spec name" },
      { name: "--model", value: "model", description: "Base model ID" },
      { name: "--output", value: "path", description: "Output spec path" },
      { name: "--profile", value: "profile", description: "Write a durable runner config (spark)" },
      { name: "--config-output", value: "path", description: "Profile config output path" },
      { name: "--force", description: "Overwrite an existing output file" },
    ],
    maxPositionals: 0,
  },
  doctor: {
    usage: "tt-local doctor [tunedtensor.json|request.json] [--config path]",
    description: "Check the host and optional run input before starting work.",
    options: [CONFIG_OPTION],
    maxPositionals: 1,
  },
  validate: {
    usage: "tt-local validate [tunedtensor.json|request.json] [options]",
    description: "Validate a behavior spec or run request without executing it.",
    options: [CONFIG_OPTION, USER_ID_OPTION, RUN_NUMBER_OPTION],
    maxPositionals: 1,
  },
  run: {
    usage: "tt-local run [tunedtensor.json|request.json] [options]",
    description: "Run the baseline, fine-tuning, tuned evaluation, and report workflow.",
    options: [
      CONFIG_OPTION,
      USER_ID_OPTION,
      RUN_NUMBER_OPTION,
      { name: "--run-id", value: "uuid", description: "Resume or override a run ID" },
      { name: "--stage", value: "stage", description: "Run through a specific workflow stage" },
      { name: "--model-artifact", value: "uri", description: "Use an existing model artifact" },
      { name: "--parent-model", value: "model-id", description: "Continue training a stored local model" },
      { name: "--parent-model-artifact", value: "uri", description: "Continue training a model artifact" },
      { name: "--force", description: "Recompute the selected stage" },
      { name: "--dry-run", description: "Write representative artifacts without GPU work" },
      { name: "--detach", description: "Run in the background and return the run ID immediately" },
      VERBOSE_OPTION,
      QUIET_OPTION,
    ],
    maxPositionals: 1,
  },
  label: {
    usage: "tt-local label <input.jsonl|input.csv> [options]",
    description: "Label local rows with a teacher model.",
    options: [
      CONFIG_OPTION,
      { name: "--input-column", value: "column", description: "CSV input column" },
      { name: "--spec", value: "path", description: "Behavior spec path" },
      { name: "--system-prompt", value: "text", description: "Explicit teacher system prompt" },
      { name: "--model", value: "model", description: "Teacher model ID" },
      { name: "--output", value: "path", description: "Output JSONL path" },
      { name: "--format", value: "jsonl|csv", description: "Force the source format" },
      { name: "--dry-run", description: "Parse and sanitize without teacher calls" },
      VERBOSE_OPTION,
      QUIET_OPTION,
    ],
    minPositionals: 1,
    maxPositionals: 1,
    missingPositionalsMessage: "label requires <input.jsonl|input.csv>",
  },
  serve: {
    usage: "tt-local serve [--config path] [--host host] [--port port]",
    description: "Serve the local runs dashboard.",
    options: [
      CONFIG_OPTION,
      { name: "--host", value: "host", description: "Dashboard bind host" },
      { name: "--port", value: "port", description: "Dashboard bind port" },
    ],
    maxPositionals: 0,
  },
};

const COMMAND_GROUPS: Record<string, CliCommandGroup> = {
  runs: {
    description: "Inspect and manage locally stored runs.",
    defaultSubcommand: "list",
    subcommands: {
      list: { usage: "tt-local runs list [--config path]", description: "List local runs.", options: [CONFIG_OPTION], maxPositionals: 0 },
      get: { usage: "tt-local runs get <run-id> [--config path]", description: "Get a local run.", options: [CONFIG_OPTION], minPositionals: 1, maxPositionals: 1, missingPositionalsMessage: "runs get requires <run-id>" },
      events: { usage: "tt-local runs events <run-id> [--config path]", description: "List run events.", options: [CONFIG_OPTION], minPositionals: 1, maxPositionals: 1, missingPositionalsMessage: "runs events requires <run-id>" },
      watch: { usage: "tt-local runs watch <run-id> [--config path]", description: "Watch a run until it finishes.", options: [CONFIG_OPTION], minPositionals: 1, maxPositionals: 1, missingPositionalsMessage: "runs watch requires <run-id>" },
      report: { usage: "tt-local runs report <run-id> [--config path]", description: "Show the baseline-vs-tuned report, including deltas and regressions.", options: [CONFIG_OPTION], minPositionals: 1, maxPositionals: 1, missingPositionalsMessage: "runs report requires <run-id>" },
      compare: { usage: "tt-local runs compare <run-id-a> <run-id-b> [--config path]", description: "Compare two run reports.", options: [CONFIG_OPTION], minPositionals: 2, maxPositionals: 2, missingPositionalsMessage: "runs compare requires <run-id-a> <run-id-b>" },
      cancel: { usage: "tt-local runs cancel <run-id> [--config path]", description: "Request cancellation of a run.", options: [CONFIG_OPTION], minPositionals: 1, maxPositionals: 1, missingPositionalsMessage: "runs cancel requires <run-id>" },
      reconcile: { usage: "tt-local runs reconcile [--config path]", description: "Rebuild local store indexes.", options: [CONFIG_OPTION], maxPositionals: 0 },
    },
  },
  models: {
    description: "Inspect, verify, prefetch, or serve local models.",
    defaultSubcommand: "list",
    subcommands: {
      list: { usage: "tt-local models list [--config path]", description: "List local models.", options: [CONFIG_OPTION], maxPositionals: 0 },
      get: { usage: "tt-local models get <model-id> [--config path]", description: "Get a local model.", options: [CONFIG_OPTION], minPositionals: 1, maxPositionals: 1, missingPositionalsMessage: "models get requires <model-id>" },
      verify: {
        usage: "tt-local models verify <model-id-or-artifact-path> [--config path]",
        description: "Verify a stored model or manifested artifact path.",
        options: [CONFIG_OPTION],
        minPositionals: 1,
        maxPositionals: 1,
        missingPositionalsMessage: "models verify requires <model-id-or-artifact-path>",
      },
      prefetch: {
        usage: "tt-local models prefetch [tunedtensor.json|request.json] [options]",
        description: "Download the configured base model before a run.",
        options: [CONFIG_OPTION, USER_ID_OPTION, RUN_NUMBER_OPTION, VERBOSE_OPTION, QUIET_OPTION],
        maxPositionals: 1,
      },
      "verify-base": {
        usage: "tt-local models verify-base [tunedtensor.json|request.json] [options]",
        description: "Verify that the configured base-model snapshot is complete and locally available.",
        options: [CONFIG_OPTION, USER_ID_OPTION, RUN_NUMBER_OPTION, VERBOSE_OPTION, QUIET_OPTION],
        maxPositionals: 1,
      },
      serve: {
        usage: "tt-local models serve <model-id> [options]",
        description: "Serve a verified adapter through an OpenAI-compatible local API.",
        options: [
          CONFIG_OPTION,
          { name: "--host", value: "host", description: "Bind host (localhost by default)" },
          { name: "--port", value: "port", description: "Bind port" },
          { name: "--device", value: "device", description: "auto, cpu, cuda, or mps" },
          { name: "--max-tokens", value: "count", description: "Default response token limit" },
          { name: "--temperature", value: "number", description: "Default sampling temperature" },
          { name: "--top-p", value: "number", description: "Default nucleus sampling threshold" },
          { name: "--max-concurrent-requests", value: "count", description: "Concurrent generation limit" },
          { name: "--spec", value: "path", description: "Behavior spec whose instructions are enforced" },
          { name: "--no-spec-prompt", description: "Do not enforce the stored behavior-spec prompt" },
          { name: "--allow-remote", description: "Allow a non-loopback bind" },
          { name: "--api-key-env", value: "name", description: "Environment variable containing a bearer token" },
          { name: "--print-command", description: "Validate and print the launch plan without starting" },
        ],
        minPositionals: 1,
        maxPositionals: 1,
        missingPositionalsMessage: "models serve requires <model-id>",
      },
    },
  },
  specs: {
    description: "Inspect and import local behavior specs.",
    defaultSubcommand: "list",
    subcommands: {
      list: { usage: "tt-local specs list [--config path]", description: "List local behavior specs.", options: [CONFIG_OPTION], maxPositionals: 0 },
      get: { usage: "tt-local specs get <spec-id> [--config path]", description: "Get a local behavior spec.", options: [CONFIG_OPTION], minPositionals: 1, maxPositionals: 1, missingPositionalsMessage: "specs get requires <spec-id>" },
      import: {
        usage: "tt-local specs import <spec-or-request.json> [options]",
        description: "Import a behavior spec into the local store.",
        options: [CONFIG_OPTION, USER_ID_OPTION, RUN_NUMBER_OPTION, { name: "--id", value: "id", description: "ID for a raw spec snapshot" }],
        minPositionals: 1,
        maxPositionals: 1,
        missingPositionalsMessage: "specs import requires <spec-or-request.json>",
      },
    },
  },
  studies: {
    description: "Lock benchmarks, run validation trials, and promote a fitted Study candidate.",
    subcommands: {
      lock: {
        usage: "tt-local studies lock <study.json> [options]",
        description: "Validate predefined dataset splits and write a deterministic benchmark lock.",
        options: [
          { name: "--output", value: "path", description: "Benchmark lock output path" },
          { name: "--force", description: "Replace an existing benchmark lock" },
        ],
        minPositionals: 1,
        maxPositionals: 1,
        missingPositionalsMessage: "studies lock requires <study.json>",
      },
      validate: {
        usage: "tt-local studies validate <study.json> [options]",
        description: "Read-only verification of current study inputs against an existing benchmark lock.",
        options: [
          { name: "--lock", value: "path", description: "Benchmark lock path" },
        ],
        minPositionals: 1,
        maxPositionals: 1,
        missingPositionalsMessage: "studies validate requires <study.json>",
      },
      run: {
        usage: "tt-local studies run <study.json> <trial.json> [options]",
        description: "Run one bundled or command-backed trial and score its validation predictions.",
        options: [
          { name: "--lock", value: "path", description: "Benchmark lock path" },
          { name: "--output-root", value: "path", description: "Root for immutable per-trial artifacts" },
        ],
        minPositionals: 2,
        maxPositionals: 2,
        missingPositionalsMessage: "studies run requires <study.json> <trial.json>",
      },
      promote: {
        usage: "tt-local studies promote <study.json> <trial.json> [options]",
        description: "Freeze one fitted bundled trial and replay-verify it as the Study candidate.",
        options: [
          { name: "--lock", value: "path", description: "Benchmark lock path" },
          { name: "--trial-directory", value: "path", description: "Existing immutable trial artifact directory" },
        ],
        minPositionals: 2,
        maxPositionals: 2,
        missingPositionalsMessage: "studies promote requires <study.json> <trial.json>",
      },
    },
  },
  store: {
    description: "Maintain the local file-backed store.",
    subcommands: {
      "rebuild-index": { usage: "tt-local store rebuild-index [--config path]", description: "Rebuild store indexes from durable records.", options: [CONFIG_OPTION], maxPositionals: 0 },
    },
  },
};

function hasHelpFlag(argv: string[]): boolean {
  return argv.includes("--help") || argv.includes("-h");
}

function printCommandHelp(definition: CliCommandDefinition): void {
  console.log(`Usage: ${definition.usage}\n\n${definition.description}`);
  if (definition.options.length > 0) {
    console.log("\nOptions:");
    for (const option of definition.options) {
      const label = option.value ? `${option.name} <${option.value}>` : option.name;
      console.log(`  ${label.padEnd(34)} ${option.description}`);
    }
  }
  console.log("  -h, --help                        Show help");
}

function printGroupHelp(command: string, group: CliCommandGroup): void {
  console.log(`Usage: tt-local ${command} <command> [options]\n\n${group.description}\n\nCommands:`);
  for (const [name, definition] of Object.entries(group.subcommands)) {
    console.log(`  ${name.padEnd(16)} ${definition.description}`);
  }
  console.log("\nRun `tt-local " + command + " <command> --help` for command-specific help.");
}

function parseCommandArguments(tokens: string[], definition: CliCommandDefinition): string[] {
  const options = new Map(definition.options.map((option) => [option.name, option]));
  const seen = new Set<string>();
  const positionals: string[] = [];
  let optionsEnded = false;

  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index]!;
    if (!optionsEnded && token === "--") {
      optionsEnded = true;
      continue;
    }
    if (!optionsEnded && token.startsWith("-")) {
      const equalsIndex = token.indexOf("=");
      const name = equalsIndex === -1 ? token : token.slice(0, equalsIndex);
      const inlineValue = equalsIndex === -1 ? undefined : token.slice(equalsIndex + 1);
      const option = options.get(name);
      if (!option) throw new Error(`Unknown option: ${name}`);
      if (seen.has(name)) throw new Error(`Option ${name} may only be specified once.`);
      seen.add(name);
      if (option.value) {
        if (inlineValue !== undefined) {
          if (!inlineValue) throw new Error(`Option ${name} requires a value.`);
          continue;
        }
        const value = tokens[index + 1];
        if (value === undefined || value.startsWith("-")) {
          throw new Error(`Option ${name} requires a value.`);
        }
        index += 1;
      } else if (inlineValue !== undefined) {
        throw new Error(`Option ${name} does not accept a value.`);
      }
      continue;
    }
    positionals.push(token);
  }

  if (positionals.length < (definition.minPositionals ?? 0)) {
    throw new Error(definition.missingPositionalsMessage ?? `Missing required argument. Usage: ${definition.usage}`);
  }
  if (definition.maxPositionals !== undefined && positionals.length > definition.maxPositionals) {
    throw new Error(`Too many arguments. Usage: ${definition.usage}`);
  }
  return positionals;
}

function parseCli(argv: string[]): ParsedCli {
  const command = argv[2] ?? "info";
  if (command === "--help" || command === "-h") {
    return { command: "info", positionals: [], help: "top" };
  }
  if (command === "--version" || command === "-V") {
    if (argv.length > 3) throw new Error(`${command} does not accept arguments.`);
    return { command, positionals: [], help: "command" };
  }
  if (command.startsWith("-")) throw new Error(`Unknown option: ${command}`);

  const definition = COMMAND_DEFINITIONS[command];
  if (definition) {
    if (hasHelpFlag(argv.slice(3))) {
      return { command, positionals: [], help: "command", definition };
    }
    return {
      command,
      positionals: parseCommandArguments(argv.slice(3), definition),
      help: "top",
      definition,
    };
  }

  const group = COMMAND_GROUPS[command];
  if (!group) throw new Error(`Unknown command: ${command}`);
  if (argv[3] === "--help" || argv[3] === "-h") {
    return { command, positionals: [], help: "group" };
  }

  const candidate = argv[3];
  let subcommand: string;
  let tokenStart: number;
  if (candidate && !candidate.startsWith("-")) {
    subcommand = candidate;
    tokenStart = 4;
  } else if (group.defaultSubcommand) {
    subcommand = group.defaultSubcommand;
    tokenStart = 3;
  } else {
    throw new Error(`${command} requires a subcommand. Run 'tt-local ${command} --help'.`);
  }
  const subcommandDefinition = group.subcommands[subcommand];
  if (!subcommandDefinition) throw new Error(`Unknown ${command} command: ${subcommand}`);
  if (hasHelpFlag(argv.slice(tokenStart))) {
    return { command, subcommand, positionals: [], help: "command", definition: subcommandDefinition };
  }
  return {
    command,
    subcommand,
    positionals: parseCommandArguments(argv.slice(tokenStart), subcommandDefinition),
    help: "top",
    definition: subcommandDefinition,
  };
}

function readNumberOption(argv: string[], name: string): number | undefined {
  const value = readOption(argv, name);
  return value ? Number(value) : undefined;
}

function readRunStage(argv: string[]): LocalRunStage {
  const value = readOption(argv, "--stage") ?? "all";
  if (
    value === "prepare"
    || value === "baseline"
    || value === "train"
    || value === "candidate"
    || value === "score"
    || value === "report"
    || value === "all"
  ) {
    return value;
  }
  throw new Error(`--stage must be one of prepare, baseline, train, candidate, score, report, all; got: ${value}`);
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

function assertSupportedValidateShape(request: FineTuneRunRequest, config: LocalRunnerConfig): void {
  if (request.training_method !== "dpo" || config.training.backend === "command") return;
  const model = resolveTrainingModel(request.spec_snapshot.base_model);
  if (model.loader === "image_text_to_text") {
    throw new Error("Bundled DPO training supports text causal-LM models only in v1. Use a causal_lm base model or training.backend=command for a custom DPO trainer.");
  }
}

interface ParentModelSelection {
  artifactUri: string;
  modelId?: string;
  baseModel?: string;
  baseModelRevision?: string;
  baseModelArtifactUri?: string;
  baseModelFingerprint?: string;
}

async function parentModelSelectionFromArgv(
  argv: string[],
  config: LocalRunnerConfig,
): Promise<ParentModelSelection | undefined> {
  const parentModel = readOption(argv, "--parent-model");
  const parentModelArtifact = readOption(argv, "--parent-model-artifact");
  if (parentModel && parentModelArtifact) {
    throw new Error("Use only one of --parent-model or --parent-model-artifact.");
  }
  if (parentModelArtifact) return { artifactUri: parentModelArtifact };
  if (!parentModel) return undefined;
  const store = createLocalStore(config.storeRoot);
  const record = await store.getModel(parentModel);
  const verified = await verifyStoredModel(record, { requireServable: true });
  const contract = verified.contract as {
    base_model_revision?: unknown;
    base_model_artifact_uri?: unknown;
    base_model_fingerprint?: unknown;
  };
  const baseModelRevision = typeof contract.base_model_revision === "string"
    ? contract.base_model_revision
    : undefined;
  const baseModelArtifactUri = typeof contract.base_model_artifact_uri === "string"
    ? contract.base_model_artifact_uri
    : undefined;
  const baseModelFingerprint = typeof contract.base_model_fingerprint === "string"
    ? contract.base_model_fingerprint
    : undefined;
  if (baseModelArtifactUri && baseModelFingerprint) {
    if (!config.paths.baseModel) {
      throw new Error(
        `Parent model ${record.id} was trained from a local base snapshot. Configure paths.baseModel to the same `
        + "snapshot before continuing it.",
      );
    }
    const configuredFingerprint = await fingerprintLocalBaseModel(config.paths.baseModel);
    if (configuredFingerprint !== baseModelFingerprint) {
      throw new Error(`Configured paths.baseModel content does not match parent model ${record.id}.`);
    }
  } else if (!baseModelRevision) {
    throw new Error(
      `Parent model ${record.id} does not record an immutable base-model revision or local snapshot fingerprint. `
      + "Use --parent-model-artifact only if you intentionally accept responsibility for that unpinned artifact.",
    );
  }
  if (baseModelRevision && config.paths.baseModel) {
    const snapshotRevision = resolve(config.paths.baseModel).match(/[\\/]snapshots[\\/]([^\\/]+)(?:[\\/]|$)/)?.[1];
    if (!baseModelFingerprint && snapshotRevision !== baseModelRevision) {
      throw new Error(
        `Parent model ${record.id} requires base revision ${baseModelRevision}; configured paths.baseModel does not `
        + "identify that snapshot revision.",
      );
    }
  }
  return {
    artifactUri: record.artifact_uri,
    modelId: record.id,
    baseModel: record.base_model,
    baseModelRevision,
    baseModelArtifactUri,
    baseModelFingerprint,
  };
}

function withParentModelArtifact(
  request: FineTuneRunRequest,
  parentModel?: ParentModelSelection,
): FineTuneRunRequest {
  if (!parentModel) return request;
  if (parentModel.baseModel && parentModel.baseModel !== request.spec_snapshot.base_model) {
    throw new Error(
      `Parent model ${parentModel.modelId ?? ""} uses base model ${parentModel.baseModel}, `
      + `but this run uses ${request.spec_snapshot.base_model}. Continue from a model with the same base model.`,
    );
  }
  const requestedRevision = request.hyperparameters.base_model_revision;
  if (parentModel.baseModelRevision && requestedRevision && parentModel.baseModelRevision !== requestedRevision) {
    throw new Error(
      `Parent model ${parentModel.modelId ?? ""} uses base revision ${parentModel.baseModelRevision}, `
      + `but this run requests ${requestedRevision}.`,
    );
  }
  return fineTuneRunRequestSchema.parse({
    ...request,
    hyperparameters: {
      ...request.hyperparameters,
      ...(parentModel.baseModelRevision ? { base_model_revision: parentModel.baseModelRevision } : {}),
      parent_model_artifact: parentModel.artifactUri,
    },
  });
}

async function sleep(ms: number): Promise<void> {
  await new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

function detachedRunArgv(argv: string[], runId: string): string[] {
  const tokens = argv.slice(3);
  const output: string[] = [];
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index]!;
    if (token === "--detach") continue;
    if (token === "--run-id") {
      index += 1;
      continue;
    }
    if (token.startsWith("--run-id=")) continue;
    output.push(token);
  }
  return ["run", ...output, "--run-id", runId];
}

async function launchDetachedRun(args: {
  argv: string[];
  request: FineTuneRunRequest;
  config: LocalRunnerConfig;
}): Promise<{ pid: number; logPath: string; artifactDir: string }> {
  if (!process.argv[1]) throw new Error("Cannot determine the tt-local CLI entrypoint for --detach.");
  const prefix = args.request.artifacts?.prefix ?? defaultArtifactPrefix({
    userId: args.request.user_id,
    behaviorSpecId: args.request.behavior_spec_id,
    runId: args.request.run_id,
  });
  const artifacts = resolveRunArtifacts({ artifactRoot: args.config.artifactRoot, prefix });
  await claimRunArtifactDirectory({
    artifacts,
    runId: args.request.run_id,
    userId: args.request.user_id,
    behaviorSpecId: args.request.behavior_spec_id,
  });
  const store = createLocalStore(args.config.storeRoot);
  const existing = await store.getRun(args.request.run_id).catch(() => null);
  if (existing) {
    await store.syncRunRequest(args.request);
  } else {
    await store.startRun({ request: args.request, artifactDir: artifacts.runDir });
  }
  const logPath = resolve(artifacts.runDir, "detached.log");
  await mkdir(dirname(logPath), { recursive: true });
  const log = await open(logPath, "a");
  try {
    const child = spawn(process.execPath, [
      ...process.execArgv,
      process.argv[1],
      ...detachedRunArgv(args.argv, args.request.run_id),
    ], {
      detached: true,
      stdio: ["ignore", log.fd, log.fd],
      env: { ...process.env, TT_LOCAL_DETACHED: "1" },
    });
    await new Promise<void>((resolveSpawn, reject) => {
      child.once("spawn", resolveSpawn);
      child.once("error", reject);
    });
    if (!child.pid) throw new Error("Detached tt-local process did not report a PID.");
    child.unref();
    return { pid: child.pid, logPath, artifactDir: artifacts.runDir };
  } catch (error) {
    await store.failRun(
      args.request.run_id,
      `Failed to launch detached workflow: ${error instanceof Error ? error.message : String(error)}`,
    ).catch(() => undefined);
    throw error;
  } finally {
    await log.close();
  }
}

async function verifyStoredModel(model: LocalModelRecord, options: { requireServable?: boolean } = {}): Promise<{
  manifest_path: string;
  integrity: Awaited<ReturnType<typeof assertArtifactManifest>>;
  artifact: Awaited<ReturnType<typeof assertUsableModelArtifact>>;
  contract: unknown;
}> {
  const manifestPath = join(model.artifact_dir, "artifact-manifest.json");
  const integrity = await assertArtifactManifest(manifestPath, {
    requiredPaths: ["stage-metadata.json", "training-report.json"],
    scopeToRequired: true,
    verifyModel: true,
  });
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as {
    model?: {
      artifact_root?: unknown;
      base_model?: unknown;
      format?: unknown;
      framework?: unknown;
      servable?: unknown;
      base_model_artifact_uri?: unknown;
      base_model_fingerprint?: unknown;
    };
  };
  if (!manifest.model) {
    throw new Error(`Artifact manifest does not contain a model contract: ${manifestPath}`);
  }
  const customNonServable = manifest.model.framework !== "transformers-peft"
    && manifest.model.servable === false;
  const artifact = await assertUsableModelArtifact(model.artifact_uri, {
    allowUnrecognizedPayload: customNonServable && !options.requireServable,
  });
  if (
    typeof manifest.model.base_model_artifact_uri === "string"
    && typeof manifest.model.base_model_fingerprint === "string"
  ) {
    const actualBaseFingerprint = await fingerprintLocalBaseModel(manifest.model.base_model_artifact_uri);
    if (actualBaseFingerprint !== manifest.model.base_model_fingerprint) {
      throw new Error("Recorded local base-model content no longer matches the model artifact contract.");
    }
  }
  if (
    typeof manifest.model.artifact_root !== "string"
    || resolve(manifest.model.artifact_root) !== resolve(artifact.path)
  ) {
    throw new Error("Stored model record does not match the artifact covered by its manifest.");
  }
  if (manifest.model.base_model !== model.base_model) {
    throw new Error("Stored model base model does not match its artifact manifest.");
  }
  if (options.requireServable && manifest.model.servable !== true) {
    throw new Error(`Model ${model.id} is valid but its artifact contract does not mark it as locally servable.`);
  }
  return {
    manifest_path: manifestPath,
    integrity,
    artifact,
    contract: manifest.model,
  };
}

async function verifyModelArtifactPath(input: string): Promise<{
  manifest_path: string;
  integrity: Awaited<ReturnType<typeof assertArtifactManifest>>;
  artifact: Awaited<ReturnType<typeof assertUsableModelArtifact>>;
  contract: unknown;
}> {
  const inputPath = resolve(input);
  let manifestPath: string | undefined;
  let artifactUri = inputPath;
  if (basename(inputPath) === "artifact-manifest.json") {
    manifestPath = inputPath;
    const raw = JSON.parse(await readFile(inputPath, "utf8")) as { model?: { artifact_root?: unknown } };
    if (typeof raw.model?.artifact_root !== "string") {
      throw new Error(`Artifact manifest does not contain a model contract: ${inputPath}`);
    }
    artifactUri = raw.model.artifact_root;
  } else {
    const metadata = await stat(inputPath).catch(() => null);
    if (!metadata) throw new Error(`Model not found and artifact path does not exist: ${input}`);
    let current = metadata.isDirectory() ? inputPath : dirname(inputPath);
    for (let depth = 0; depth < 8; depth += 1) {
      const candidate = join(current, "artifact-manifest.json");
      const raw = await readFile(candidate, "utf8").catch(() => null);
      if (raw) {
        const parsed = JSON.parse(raw) as { model?: { artifact_root?: unknown } };
        if (
          typeof parsed.model?.artifact_root === "string"
          && resolve(parsed.model.artifact_root) === resolve(inputPath)
        ) {
          manifestPath = candidate;
          break;
        }
      }
      const parent = dirname(current);
      if (parent === current) break;
      current = parent;
    }
  }
  if (!manifestPath) {
    throw new Error(`No artifact manifest covering model path ${inputPath} was found in its parent run directory.`);
  }
  const integrity = await assertArtifactManifest(manifestPath, {
    requiredPaths: ["stage-metadata.json", "training-report.json"],
    scopeToRequired: true,
    verifyModel: true,
  });
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as { model?: Record<string, unknown> };
  if (!manifest.model) throw new Error(`Artifact manifest does not contain a model contract: ${manifestPath}`);
  const customNonServable = manifest.model.framework !== "transformers-peft"
    && manifest.model.servable === false;
  const artifact = await assertUsableModelArtifact(artifactUri, {
    allowUnrecognizedPayload: customNonServable,
  });
  if (
    typeof manifest.model.base_model_artifact_uri === "string"
    && typeof manifest.model.base_model_fingerprint === "string"
  ) {
    const actualBaseFingerprint = await fingerprintLocalBaseModel(manifest.model.base_model_artifact_uri);
    if (actualBaseFingerprint !== manifest.model.base_model_fingerprint) {
      throw new Error("Recorded local base-model content no longer matches the model artifact contract.");
    }
  }
  if (
    typeof manifest.model.artifact_root !== "string"
    || resolve(manifest.model.artifact_root) !== resolve(artifact.path)
  ) {
    throw new Error("Model path does not match the artifact covered by its manifest.");
  }
  return { manifest_path: manifestPath, integrity, artifact, contract: manifest.model };
}

async function modelSystemPrompt(args: {
  argv: string[];
  model: LocalModelRecord;
  store: LocalStore;
}): Promise<string | undefined> {
  const specPath = readOption(args.argv, "--spec");
  if (specPath && hasFlag(args.argv, "--no-spec-prompt")) {
    throw new Error("Use only one of --spec or --no-spec-prompt.");
  }
  if (hasFlag(args.argv, "--no-spec-prompt")) return undefined;

  let spec: SpecSnapshot;
  if (specPath) {
    const input = JSON.parse(await readFile(resolve(specPath), "utf8")) as unknown;
    const local = localBehaviorSpecFileSchema.safeParse(input);
    const request = fineTuneRunRequestSchema.safeParse(input);
    const snapshot = specSnapshotSchema.safeParse(input);
    const parsedSpec = local.success ? local.data : request.success ? request.data.spec_snapshot : snapshot.success ? snapshot.data : null;
    if (!parsedSpec) throw new Error(`--spec must contain a TT Local behavior spec or run request: ${resolve(specPath)}`);
    spec = parsedSpec;
  } else {
    const runRequestPath = join(args.store.paths.runsDir, args.model.run_id, "request.json");
    let persistedRequest: FineTuneRunRequest | null = null;
    try {
      persistedRequest = fineTuneRunRequestSchema.parse(JSON.parse(await readFile(runRequestPath, "utf8")));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    spec = persistedRequest?.spec_snapshot
      ?? (await args.store.getSpec(args.model.behavior_spec_id)).spec;
  }
  if (spec.base_model !== args.model.base_model) {
    throw new Error(
      `Behavior spec base model ${spec.base_model} does not match stored model base ${args.model.base_model}.`,
    );
  }
  const prompt = buildSystemMessage(spec);
  const metadata = JSON.parse(
    await readFile(join(args.model.artifact_dir, "stage-metadata.json"), "utf8"),
  ) as { system_prompt_sha256?: unknown };
  const promptHash = createHash("sha256").update(prompt).digest("hex");
  if (metadata.system_prompt_sha256 !== promptHash) {
    throw new Error(
      "Behavior spec instructions do not match the prompt fingerprint used for this trained model. "
      + "Pass the original --spec, or use --no-spec-prompt only when this change is intentional.",
    );
  }
  return prompt;
}

function modelServeDevice(argv: string[]): "auto" | "cpu" | "cuda" | "mps" | undefined {
  const value = readOption(argv, "--device");
  if (value === undefined || value === "auto" || value === "cpu" || value === "cuda" || value === "mps") {
    return value;
  }
  throw new Error(`--device must be one of auto, cpu, cuda, mps; got: ${value}`);
}

async function main(argv: string[]): Promise<void> {
  const cli = parseCli(argv);
  const command = cli.command;

  if (command === "--version" || command === "-V") {
    console.log(TT_LOCAL_VERSION);
    return;
  }
  if (cli.help === "top" && (argv[2] === "--help" || argv[2] === "-h")) {
    printHelp();
    return;
  }
  if (cli.help === "group") {
    printGroupHelp(command, COMMAND_GROUPS[command]!);
    return;
  }
  if (cli.help === "command" && cli.definition) {
    printCommandHelp(cli.definition);
    return;
  }

  const loadedEnv = await loadDotEnvFromCwd();
  if (loadedEnv.length > 0 && !hasFlag(argv, "--quiet")) {
    process.stderr.write(`[tt-local] loaded ${loadedEnv.join(", ")} from .env\n`);
  }

  if (command === "info") {
    const info = getLocalRunnerInfo();
    console.log(`${info.name}: ${info.description}`);
    console.log(`Version: ${info.version}`);
    console.log(`Status: ${info.status}`);
    return;
  }

  if (command === "doctor") {
    const config = await configFromArgv(argv);
    const request = cli.positionals[0]
      ? (await loadLocalRunInput(resolve(cli.positionals[0]))).request
      : undefined;
    const checks = await runDoctor(config, request);
    const ok = checks.every((check) => check.ok);
    printJson({ ok, checks });
    if (!ok) process.exitCode = 1;
    return;
  }

  if (command === "init") {
    const outputPath = resolve(readOption(argv, "--output") ?? DEFAULT_LOCAL_SPEC_PATH);
    const profile = readOption(argv, "--profile");
    if (profile !== undefined && profile !== "spark") {
      throw new Error(`--profile must be spark, got: ${profile}`);
    }
    const spec = await initLocalSpecFile({
      outputPath,
      name: readOption(argv, "--name") ?? "Local Tuned Tensor Spec",
      baseModel: readOption(argv, "--model") ?? "Qwen/Qwen3.5-2B",
      force: hasFlag(argv, "--force"),
    });
    const configPath = profile
      ? resolve(readOption(argv, "--config-output") ?? resolve(dirname(outputPath), "local-runner.json"))
      : undefined;
    if (profile && configPath) {
      await initLocalRunnerConfigFile({
        outputPath: configPath,
        profile,
        force: hasFlag(argv, "--force"),
      });
    }
    printJson({
      ok: true,
      path: outputPath,
      id: spec.id,
      name: spec.name,
      base_model: spec.base_model,
      config_path: configPath ?? null,
    });
    return;
  }

  if (command === "validate") {
    const inputPath = resolve(cli.positionals[0] ?? DEFAULT_LOCAL_SPEC_PATH);
    const input = await loadLocalRunInput(inputPath, {
      userId: readOption(argv, "--user-id"),
      runNumber: readNumberOption(argv, "--run-number"),
    });
    const request = input.request;
    const config = await configFromArgv(argv);
    assertLocalRunInputReady(request);
    assertSupportedValidateShape(request, config);
    assertEvaluationScoringReady(config);
    printJson({
      ok: true,
      input_kind: input.kind,
      input_path: input.path,
      run_id: input.kind === "request" ? request.run_id : null,
      behavior_spec_id: request.behavior_spec_id,
      training_method: request.training_method,
      base_model: request.spec_snapshot.base_model,
      dataset_format: request.dataset_prebuilt?.format ?? null,
      artifact_root: config.artifactRoot,
      store_root: config.storeRoot,
      dry_run: config.dryRun,
      ...(input.warnings.length > 0 ? { warnings: input.warnings } : {}),
    });
    return;
  }

  if (command === "run") {
    const inputPath = resolve(cli.positionals[0] ?? DEFAULT_LOCAL_SPEC_PATH);
    const configInput = await configFromArgv(argv);
    const config = localRunnerConfigSchema.parse({
      ...configInput,
      dryRun: hasFlag(argv, "--dry-run") ? true : configInput.dryRun,
    });
    const input = await loadLocalRunInput(inputPath, {
      runId: readOption(argv, "--run-id"),
      userId: readOption(argv, "--user-id"),
      runNumber: readNumberOption(argv, "--run-number"),
    });
    if (!hasFlag(argv, "--quiet")) {
      for (const warning of input.warnings) {
        process.stderr.write(`[tt-local] warning: ${warning}\n`);
      }
    }
    const runId = readOption(argv, "--run-id");
    const baseRequest = runId && input.kind === "request"
      ? fineTuneRunRequestSchema.parse({ ...input.request, run_id: runId })
      : input.request;
    const request = withParentModelArtifact(
      baseRequest,
      await parentModelSelectionFromArgv(argv, config),
    );
    assertLocalRunInputReady(request);
    assertSupportedValidateShape(request, config);
    assertEvaluationScoringReady(config);
    if (hasFlag(argv, "--detach")) {
      const detached = await launchDetachedRun({ argv, request, config });
      const configPath = readOption(argv, "--config");
      const configSuffix = configPath ? ` --config ${JSON.stringify(resolve(configPath))}` : "";
      printJson({
        status: "queued",
        detached: true,
        run_id: request.run_id,
        behavior_spec_id: request.behavior_spec_id,
        pid: detached.pid,
        log_path: detached.logPath,
        artifact_dir: detached.artifactDir,
        next: {
          watch: `tt-local runs watch ${request.run_id}${configSuffix}`,
          cancel: `tt-local runs cancel ${request.run_id}${configSuffix}`,
        },
      });
      return;
    }
    const stage = readRunStage(argv);
    const result = await runLocalFineTuneStage({
      request,
      config,
      stage,
      force: hasFlag(argv, "--force"),
      modelArtifact: readOption(argv, "--model-artifact"),
      reporter: createConsoleReporter({
        verbose: hasFlag(argv, "--verbose"),
        quiet: hasFlag(argv, "--quiet"),
      }),
    });
    if (result.report) {
      printJson({
        status: result.report.status,
        stage: result.stage,
        input_kind: input.kind,
        run_id: result.report.run_id,
        behavior_spec_id: result.report.behavior_spec_id,
        report_path: result.reportPath,
        artifact_dir: result.artifactDir,
        model_id: `local-${result.report.run_id}`,
        fine_tuned_model_id: result.report.fine_tuned_model_id,
        parent_model_artifact: result.report.run_metadata.parent_model_artifact ?? null,
        training_log: result.report.training.log_uri,
        baseline_eval: result.report.artifact_uris.baseline_eval,
        candidate_eval: result.report.artifact_uris.candidate_eval,
        comparison: result.report.comparison,
      });
    } else {
      printJson({
        status: "stage_completed",
        stage: result.stage,
        input_kind: input.kind,
        run_id: result.request.run_id,
        behavior_spec_id: result.request.behavior_spec_id,
        artifact_dir: result.artifactDir,
        parent_model_artifact: result.request.hyperparameters.parent_model_artifact ?? null,
        artifacts: result.artifacts,
      });
    }
    return;
  }

  if (command === "label") {
    const sourcePath = cli.positionals[0];
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

  if (command === "studies") {
    const subcommand = cli.subcommand!;
    const studyPath = cli.positionals[0];
    if (!studyPath) throw new Error(`studies ${subcommand} requires <study.json>`);
    if (subcommand === "lock") {
      const result = await writeStudyBenchmarkLock({
        studyPath,
        outputPath: readOption(argv, "--output"),
        force: hasFlag(argv, "--force"),
      });
      return printJson({
        ok: true,
        status: "locked",
        study_path: resolve(studyPath),
        lock_path: result.lockPath,
        benchmark_lock: result.lock,
      });
    }
    if (subcommand === "validate") {
      const result = await validateStudyBenchmark({
        studyPath,
        lockPath: readOption(argv, "--lock"),
      });
      return printJson({
        ok: true,
        status: "valid",
        study_path: resolve(studyPath),
        lock_path: result.lockPath,
        benchmark_lock: result.lock,
      });
    }
    if (subcommand === "run") {
      const trialPath = cli.positionals[1];
      if (!trialPath) throw new Error("studies run requires <study.json> <trial.json>");
      const result = await runStudyTrial({
        studyPath,
        trialPath,
        lockPath: readOption(argv, "--lock"),
        outputRoot: readOption(argv, "--output-root"),
      });
      return printJson({
        ok: true,
        status: "completed",
        study_path: resolve(studyPath),
        trial_spec_path: resolve(trialPath),
        trial_directory: result.trialDirectory,
        report_path: result.reportPath,
        trial_report: result.report,
      });
    }
    if (subcommand === "promote") {
      const trialPath = cli.positionals[1];
      if (!trialPath) {
        throw new Error("studies promote requires <study.json> <trial.json>");
      }
      const result = await promoteStudyTrialCandidate({
        studyPath,
        trialPath,
        lockPath: readOption(argv, "--lock"),
        trialDirectory: readOption(argv, "--trial-directory"),
      });
      return printJson({
        ok: true,
        status: "promoted",
        study_path: resolve(studyPath),
        trial_spec_path: resolve(trialPath),
        candidate_directory: result.candidateDirectory,
        candidate_lock_path: result.lockPath,
        candidate_lock: result.lock,
      });
    }
    throw new Error(`Unknown studies command: ${subcommand}`);
  }

  if (command === "runs") {
    const subcommand = cli.subcommand!;
    const config = await configFromArgv(argv);
    const store = createLocalStore(config.storeRoot);
    if (subcommand === "list") return printJson(await store.listRuns());
    if (subcommand === "get") {
      const id = cli.positionals[0];
      if (!id) throw new Error("runs get requires <run-id>");
      return printJson(await store.getRun(id));
    }
    if (subcommand === "events") {
      const id = cli.positionals[0];
      if (!id) throw new Error("runs events requires <run-id>");
      return printJson(await store.getRunEvents(id));
    }
    if (subcommand === "report") {
      const id = cli.positionals[0];
      if (!id) throw new Error("runs report requires <run-id>");
      return printJson(await store.getRunReport(id));
    }
    if (subcommand === "compare") {
      const idA = cli.positionals[0];
      const idB = cli.positionals[1];
      if (!idA || !idB) throw new Error("runs compare requires <run-id-a> <run-id-b>");
      const [reportA, reportB] = await Promise.all([
        store.getRunReport(idA),
        store.getRunReport(idB),
      ]);
      return printJson(compareRuns(reportA, reportB));
    }
    if (subcommand === "cancel") {
      const id = cli.positionals[0];
      if (!id) throw new Error("runs cancel requires <run-id>");
      await store.cancelRun(id);
      const run = await store.getRun(id);
      return printJson({ ok: true, run_id: run.id, status: run.status, current_stage: run.current_stage });
    }
    if (subcommand === "watch") {
      const id = cli.positionals[0];
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
        if (isTerminalRunState(run)) return;
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
    const subcommand = cli.subcommand!;
    const config = await configFromArgv(argv);
    const store = createLocalStore(config.storeRoot);
    if (subcommand === "list") return printJson(await store.listModels());
    if (subcommand === "get") {
      const id = cli.positionals[0];
      if (!id) throw new Error("models get requires <model-id>");
      return printJson(await store.getModel(id));
    }
    if (subcommand === "verify") {
      const id = cli.positionals[0];
      if (!id) throw new Error("models verify requires <model-id-or-artifact-path>");
      if (await stat(resolve(id)).then(() => true, () => false)) {
        return printJson({ ok: true, model: null, ...await verifyModelArtifactPath(id) });
      }
      const model = await store.getModel(id);
      const verified = await verifyStoredModel(model);
      return printJson({ ok: true, model, ...verified });
    }
    if (subcommand === "prefetch" || subcommand === "verify-base") {
      const inputPath = resolve(cli.positionals[0] ?? DEFAULT_LOCAL_SPEC_PATH);
      const input = await loadLocalRunInput(inputPath, {
        userId: readOption(argv, "--user-id"),
        runNumber: readNumberOption(argv, "--run-number"),
      });
      if (!hasFlag(argv, "--quiet")) {
        for (const warning of input.warnings) {
          process.stderr.write(`[tt-local] warning: ${warning}\n`);
        }
      }
      const report = await prefetchBaseModel({
        request: input.request,
        config,
        localOnly: subcommand === "verify-base",
        reporter: createConsoleReporter({
          verbose: hasFlag(argv, "--verbose"),
          quiet: hasFlag(argv, "--quiet"),
        }),
      });
      return printJson({
        ...report,
        input_kind: input.kind,
        input_path: input.path,
      });
    }
    if (subcommand === "serve") {
      const id = cli.positionals[0];
      if (!id) throw new Error("models serve requires <model-id>");
      const model = await store.getModel(id);
      const verified = await verifyStoredModel(model, { requireServable: true });
      const launch = buildLocalModelServerLaunch({
        model,
        config,
        options: {
          host: readOption(argv, "--host"),
          port: readNumberOption(argv, "--port"),
          device: modelServeDevice(argv),
          maxTokens: readNumberOption(argv, "--max-tokens"),
          temperature: readNumberOption(argv, "--temperature"),
          topP: readNumberOption(argv, "--top-p"),
          maxConcurrentRequests: readNumberOption(argv, "--max-concurrent-requests"),
          systemPrompt: await modelSystemPrompt({ argv, model, store }),
          allowRemote: hasFlag(argv, "--allow-remote"),
          apiKeyEnv: readOption(argv, "--api-key-env"),
          baseModelRevision: (() => {
            const contract = verified.contract as { base_model_revision?: unknown };
            return typeof contract.base_model_revision === "string" ? contract.base_model_revision : undefined;
          })(),
          baseModelArtifactUri: (() => {
            const contract = verified.contract as { base_model_artifact_uri?: unknown };
            return typeof contract.base_model_artifact_uri === "string" ? contract.base_model_artifact_uri : undefined;
          })(),
        },
      });
      if (hasFlag(argv, "--print-command")) {
        return printJson({
          ok: true,
          model_id: model.id,
          url: launch.url,
          command: launch.displayCommand,
          artifact_path: launch.artifactPath,
          manifest_path: verified.manifest_path,
          integrity: verified.integrity,
        });
      }
      process.stderr.write(`[tt-local] verified ${verified.integrity.checked} artifact file(s)\n`);
      process.stderr.write(`[tt-local] model API: ${launch.url}\n`);
      await serveLocalModel(launch);
      return;
    }
    throw new Error(`Unknown models command: ${subcommand}`);
  }

  if (command === "specs") {
    const subcommand = cli.subcommand!;
    const config = await configFromArgv(argv);
    const store = createLocalStore(config.storeRoot);
    if (subcommand === "list") return printJson(await store.listSpecs());
    if (subcommand === "get") {
      const id = cli.positionals[0];
      if (!id) throw new Error("specs get requires <spec-id>");
      return printJson(await store.getSpec(id));
    }
    if (subcommand === "import") {
      const path = cli.positionals[0];
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
    const subcommand = cli.subcommand!;
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
