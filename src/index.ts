#!/usr/bin/env node

import { readFileSync, realpathSync } from "node:fs";
import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { cwd } from "node:process";
import { fileURLToPath } from "node:url";
import { compareRuns } from "./compare.js";
import { assertArtifactManifest } from "./artifacts.js";
import { fineTuneRunRequestSchema, localBehaviorSpecFileSchema, localRunnerConfigSchema, type FineTuneRunRequest, type LocalRunnerConfig, type SpecSnapshot } from "./contracts.js";
import { buildSystemMessage } from "./dataset.js";
import {
  loadLocalRunnerConfig,
  fingerprintLocalBaseModel,
  runLocalFineTune,
  validateLocalFineTuneInput,
} from "./orchestrator.js";
import { runDoctor } from "./doctor.js";
import { assertUsableModelArtifact } from "./model-registry.js";
import { buildLocalModelServerLaunch, serveLocalModel } from "./model-server.js";
import { prefetchBaseModel } from "./prefetch.js";
import { createLocalStore, type LocalModelRecord, type LocalStore } from "./store.js";
import {
  DEFAULT_LOCAL_SPEC_PATH,
  assertLocalRunInputReady,
  initLocalRunnerConfigFile,
  initLocalSpecFile,
  loadLocalRunInput,
} from "./local-project.js";
import { sanitizeLogLine, type LocalRunProgressEvent, type LocalRunReporter } from "./run-reporter.js";

export * from "./compare.js";
export * from "./contracts.js";
export * from "./dataset.js";
export * from "./model-server.js";
export * from "./orchestrator.js";
export * from "./local-project.js";
export * from "./prefetch.js";
export * from "./run-reporter.js";
export * from "./store.js";

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
    description: "Local CUDA LoRA fine-tuning with held-out base-versus-tuned evaluation.",
    version: TT_LOCAL_VERSION,
  };
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
  doctor [tunedtensor.json] [--config local-runner.json]
  validate [tunedtensor.json] [--config local-runner.json]
  run [tunedtensor.json] [--config local-runner.json] [--dry-run] [--verbose] [--quiet]
  serve <model-id> [--config local-runner.json] [--host 127.0.0.1] [--port 8000]
  runs list|get|events|report|compare [args] [--config local-runner.json]
  models list|get|verify|prefetch|verify-base|serve [args] [--config local-runner.json]

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
const VERBOSE_OPTION = { name: "--verbose", description: "Stream subprocess output" } as const;
const QUIET_OPTION = { name: "--quiet", description: "Suppress progress output on stderr" } as const;
const MODEL_SERVE_OPTIONS = [
  CONFIG_OPTION,
  { name: "--host", value: "host", description: "Bind host (localhost by default)" },
  { name: "--port", value: "port", description: "Bind port" },
  { name: "--device", value: "device", description: "cuda or cpu" },
  { name: "--max-tokens", value: "count", description: "Default response token limit" },
  { name: "--temperature", value: "number", description: "Default sampling temperature" },
  { name: "--top-p", value: "number", description: "Default nucleus sampling threshold" },
  { name: "--max-concurrent-requests", value: "count", description: "Concurrent generation limit" },
  { name: "--spec", value: "path", description: "Behavior spec whose instructions are enforced" },
  { name: "--no-spec-prompt", description: "Do not enforce the stored behavior-spec prompt" },
  { name: "--allow-remote", description: "Allow a non-loopback bind" },
  { name: "--api-key-env", value: "name", description: "Environment variable containing a bearer token" },
  { name: "--print-command", description: "Validate and print the launch plan without starting" },
] as const satisfies readonly CliOptionDefinition[];

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
      { name: "--config", value: "path", description: "Runner config path (written with --profile)" },
      { name: "--force", description: "Overwrite an existing output file" },
    ],
    maxPositionals: 0,
  },
  doctor: {
    usage: "tt-local doctor [tunedtensor.json] [--config path]",
    description: "Check the host and optional run input before starting work.",
    options: [CONFIG_OPTION],
    maxPositionals: 1,
  },
  validate: {
    usage: "tt-local validate [tunedtensor.json] [options]",
    description: "Validate a local behavior spec without executing it.",
    options: [CONFIG_OPTION],
    maxPositionals: 1,
  },
  run: {
    usage: "tt-local run [tunedtensor.json] [options]",
    description: "Run the baseline, fine-tuning, tuned evaluation, and report workflow.",
    options: [
      CONFIG_OPTION,
      { name: "--dry-run", description: "Write representative artifacts without GPU work" },
      VERBOSE_OPTION,
      QUIET_OPTION,
    ],
    maxPositionals: 1,
  },
  serve: {
    usage: "tt-local serve <model-id> [options]",
    description: "Serve a verified model through an OpenAI-compatible local API.",
    options: MODEL_SERVE_OPTIONS,
    minPositionals: 1,
    maxPositionals: 1,
    missingPositionalsMessage: "serve requires <model-id>",
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
      report: { usage: "tt-local runs report <run-id> [--config path]", description: "Show the baseline-vs-tuned report, including deltas and regressions.", options: [CONFIG_OPTION], minPositionals: 1, maxPositionals: 1, missingPositionalsMessage: "runs report requires <run-id>" },
      compare: { usage: "tt-local runs compare <run-id-a> <run-id-b> [--config path]", description: "Compare two run reports.", options: [CONFIG_OPTION], minPositionals: 2, maxPositionals: 2, missingPositionalsMessage: "runs compare requires <run-id-a> <run-id-b>" },
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
        usage: "tt-local models prefetch [tunedtensor.json] [options]",
        description: "Download the configured base model before a run.",
        options: [CONFIG_OPTION, VERBOSE_OPTION, QUIET_OPTION],
        maxPositionals: 1,
      },
      "verify-base": {
        usage: "tt-local models verify-base [tunedtensor.json] [options]",
        description: "Verify that the configured base-model snapshot is complete and locally available.",
        options: [CONFIG_OPTION, VERBOSE_OPTION, QUIET_OPTION],
        maxPositionals: 1,
      },
      serve: {
        usage: "tt-local models serve <model-id> [options]",
        description: "Alias for `tt-local serve`.",
        options: MODEL_SERVE_OPTIONS,
        minPositionals: 1,
        maxPositionals: 1,
        missingPositionalsMessage: "models serve requires <model-id>",
      },
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

interface LocalConfigSelection {
  config: LocalRunnerConfig;
  path?: string;
}

async function selectedConfigPath(argv: string[], adjacentTo?: string): Promise<string | undefined> {
  const explicitPath = readOption(argv, "--config");
  if (explicitPath) return resolve(explicitPath);
  const candidate = join(adjacentTo ? dirname(resolve(adjacentTo)) : cwd(), "local-runner.json");
  const metadata = await stat(candidate).catch(() => null);
  return metadata?.isFile() ? candidate : undefined;
}

async function configSelectionFromArgv(argv: string[], adjacentTo?: string): Promise<LocalConfigSelection> {
  const path = await selectedConfigPath(argv, adjacentTo);
  return {
    config: await loadLocalRunnerConfig(path),
    ...(path ? { path } : {}),
  };
}

async function configFromArgv(argv: string[], adjacentTo?: string): Promise<LocalRunnerConfig> {
  return (await configSelectionFromArgv(argv, adjacentTo)).config;
}

async function loadCliBehaviorSpec(inputPath: string, runId?: string) {
  const input = await loadLocalRunInput(inputPath, {
    ...(runId ? { runId } : {}),
  });
  if (input.kind !== "spec") {
    throw new Error(`TT Local CLI expects a tunedtensor.json behavior spec, not a full run request: ${input.path}`);
  }
  return input;
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

async function verifyStoredModel(model: LocalModelRecord): Promise<{
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
  const artifact = await assertUsableModelArtifact(model.artifact_uri);
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
  const artifact = await assertUsableModelArtifact(artifactUri);
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
    if (!local.success) {
      throw new Error(`--spec must contain a tunedtensor.json behavior spec: ${resolve(specPath)}`);
    }
    spec = local.data;
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

function modelServeDevice(argv: string[]): "cpu" | "cuda" | undefined {
  const value = readOption(argv, "--device");
  if (value === undefined || value === "cpu" || value === "cuda") {
    return value;
  }
  throw new Error(`--device must be cuda or cpu; got: ${value}`);
}

async function serveStoredModelFromCli(args: {
  argv: string[];
  modelId: string;
  config: LocalRunnerConfig;
}): Promise<void> {
  const store = createLocalStore(args.config.storeRoot);
  const model = await store.getModel(args.modelId);
  const verified = await verifyStoredModel(model);
  const launch = buildLocalModelServerLaunch({
    model,
    config: args.config,
    options: {
      host: readOption(args.argv, "--host"),
      port: readNumberOption(args.argv, "--port"),
      device: modelServeDevice(args.argv),
      maxTokens: readNumberOption(args.argv, "--max-tokens"),
      temperature: readNumberOption(args.argv, "--temperature"),
      topP: readNumberOption(args.argv, "--top-p"),
      maxConcurrentRequests: readNumberOption(args.argv, "--max-concurrent-requests"),
      systemPrompt: await modelSystemPrompt({ argv: args.argv, model, store }),
      allowRemote: hasFlag(args.argv, "--allow-remote"),
      apiKeyEnv: readOption(args.argv, "--api-key-env"),
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
  if (hasFlag(args.argv, "--print-command")) {
    printJson({
      ok: true,
      model_id: model.id,
      url: launch.url,
      command: launch.displayCommand,
      artifact_path: launch.artifactPath,
      manifest_path: verified.manifest_path,
      integrity: verified.integrity,
    });
    return;
  }
  process.stderr.write(`[tt-local] verified ${verified.integrity.checked} artifact file(s)\n`);
  process.stderr.write(`[tt-local] model API: ${launch.url}\n`);
  await serveLocalModel(launch);
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

  if (command === "info") {
    const info = getLocalRunnerInfo();
    console.log(`${info.name}: ${info.description}`);
    console.log(`Version: ${info.version}`);
    console.log(`Status: ${info.status}`);
    return;
  }

  if (command === "doctor") {
    const inputPath = resolve(cli.positionals[0] ?? DEFAULT_LOCAL_SPEC_PATH);
    const configSelection = await configSelectionFromArgv(argv, inputPath);
    const hasDefaultSpec = cli.positionals[0]
      ? true
      : Boolean((await stat(inputPath).catch(() => null))?.isFile());
    const request = hasDefaultSpec
      ? (await loadCliBehaviorSpec(inputPath)).request
      : undefined;
    const checks = await runDoctor(configSelection.config, request);
    const ok = checks.every((check) => check.ok);
    printJson({
      ok,
      config_path: configSelection.path ?? null,
      checks,
    });
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
      ? resolve(readOption(argv, "--config") ?? resolve(dirname(outputPath), "local-runner.json"))
      : await selectedConfigPath(argv, outputPath);
    if (profile) {
      await initLocalRunnerConfigFile({
        outputPath: configPath!,
        profile,
        force: hasFlag(argv, "--force"),
      });
    } else if (configPath) {
      await loadLocalRunnerConfig(configPath);
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
    const input = await loadCliBehaviorSpec(inputPath);
    assertLocalRunInputReady(input.request);
    const configSelection = await configSelectionFromArgv(argv, inputPath);
    const validated = await validateLocalFineTuneInput({
      request: input.request,
      config: configSelection.config,
    });
    const request = validated.request;
    const config = validated.config;
    printJson({
      ok: true,
      input_path: input.path,
      config_path: configSelection.path ?? null,
      behavior_spec_id: request.behavior_spec_id,
      base_model: request.spec_snapshot.base_model,
      dataset_format: request.dataset_prebuilt?.format ?? null,
      artifact_root: config.artifactRoot,
      store_root: config.storeRoot,
      dry_run: config.dryRun,
    });
    return;
  }

  if (command === "run") {
    const inputPath = resolve(cli.positionals[0] ?? DEFAULT_LOCAL_SPEC_PATH);
    const configSelection = await configSelectionFromArgv(argv, inputPath);
    const configInput = configSelection.config;
    const config = localRunnerConfigSchema.parse({
      ...configInput,
      dryRun: hasFlag(argv, "--dry-run") ? true : configInput.dryRun,
    });
    const input = await loadCliBehaviorSpec(inputPath);
    assertLocalRunInputReady(input.request);
    const validated = await validateLocalFineTuneInput({
      request: input.request,
      config,
    });
    let request = validated.request;
    const reporter = createConsoleReporter({
      verbose: hasFlag(argv, "--verbose"),
      quiet: hasFlag(argv, "--quiet"),
    });
    if (
      !config.dryRun
      && !config.paths.baseModel
      && !request.hyperparameters.base_model_revision
    ) {
      const prefetch = await prefetchBaseModel({
        request,
        config,
        reporter,
      });
      if (!prefetch.snapshot_revision) {
        throw new Error("Base-model prefetch did not return an immutable snapshot revision.");
      }
      request = fineTuneRunRequestSchema.parse({
        ...request,
        hyperparameters: {
          ...request.hyperparameters,
          base_model_revision: prefetch.snapshot_revision,
        },
      });
    }
    const result = await runLocalFineTune({
      request,
      config,
      reporter,
    });
    printJson({
      status: result.report.status,
      run_id: result.report.run_id,
      behavior_spec_id: result.report.behavior_spec_id,
      report_path: result.reportPath,
      artifact_dir: result.artifactDir,
      ...(!config.dryRun ? {
        model_id: `local-${result.report.run_id}`,
        fine_tuned_model_id: result.report.fine_tuned_model_id,
      } : {}),
      training_log: result.report.training.log_uri,
      baseline_eval: result.report.artifact_uris.baseline_eval,
      candidate_eval: result.report.artifact_uris.candidate_eval,
      comparison: result.report.comparison,
    });
    return;
  }

  if (command === "serve") {
    const config = await configFromArgv(argv);
    await serveStoredModelFromCli({
      argv,
      modelId: cli.positionals[0]!,
      config,
    });
    return;
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
    throw new Error(`Unknown runs command: ${subcommand}`);
  }

  if (command === "models") {
    const subcommand = cli.subcommand!;
    const modelInputPath = subcommand === "prefetch" || subcommand === "verify-base"
      ? resolve(cli.positionals[0] ?? DEFAULT_LOCAL_SPEC_PATH)
      : undefined;
    const config = await configFromArgv(argv, modelInputPath);
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
      const input = await loadCliBehaviorSpec(modelInputPath!);
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
        input_path: input.path,
      });
    }
    if (subcommand === "serve") {
      const id = cli.positionals[0];
      if (!id) throw new Error("models serve requires <model-id>");
      await serveStoredModelFromCli({ argv, modelId: id, config });
      return;
    }
    throw new Error(`Unknown models command: ${subcommand}`);
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
