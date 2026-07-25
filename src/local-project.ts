import { createHash, randomUUID } from "node:crypto";
import { readFile, stat, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";
import { homedir } from "node:os";
import { mkdir } from "node:fs/promises";
import {
  fineTuneRunRequestSchema,
  localBehaviorSpecFileSchema,
  type FineTuneRunRequest,
  type LocalBehaviorSpecFile,
} from "./contracts.js";

export const DEFAULT_LOCAL_SPEC_PATH = "tunedtensor.json";

export interface CreateLocalSpecArgs {
  name: string;
  baseModel: string;
  outputPath: string;
  force?: boolean;
}

export type LocalRunnerProfile = "spark";

export interface CreateLocalRunnerConfigArgs {
  outputPath: string;
  profile: LocalRunnerProfile;
  force?: boolean;
}

export interface RunRequestFromSpecOptions {
  runId?: string;
}

export interface LocalRunInput {
  kind: "request" | "spec";
  path: string;
  request: FineTuneRunRequest;
  spec?: LocalBehaviorSpecFile;
}

function resolveLocalReference(value: unknown, baseDirectory: string): unknown {
  if (typeof value !== "string" || !value || isAbsolute(value) || /^[a-z][a-z0-9+.-]*:/i.test(value)) {
    return value;
  }
  if (value === "~") return homedir();
  if (value.startsWith("~/")) return resolve(homedir(), value.slice(2));
  return resolve(baseDirectory, value);
}

/** Resolve dataset paths relative to the spec/request file itself. */
export function resolveLocalRunInputPaths(raw: unknown, inputPath: string): unknown {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return raw;
  const baseDirectory = dirname(resolve(inputPath));
  const value = structuredClone(raw) as Record<string, unknown>;
  const dataset = value.dataset_prebuilt;
  if (dataset && typeof dataset === "object" && !Array.isArray(dataset)) {
    const fields = dataset as Record<string, unknown>;
    for (const key of ["training", "validation", "test"] as const) {
      if (fields[key] !== undefined) fields[key] = resolveLocalReference(fields[key], baseDirectory);
    }
  }
  return value;
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

export async function initLocalSpecFile(args: CreateLocalSpecArgs): Promise<LocalBehaviorSpecFile> {
  if (!args.force && await exists(args.outputPath)) {
    throw new Error(`Refusing to overwrite existing file: ${args.outputPath}`);
  }
  const spec = localBehaviorSpecFileSchema.parse({
    id: randomUUID(),
    name: args.name,
    description: "",
    system_prompt: "Describe the behavior this local model should learn.",
    guidelines: [
      "Return concise, task-specific answers.",
    ],
    constraints: [],
    base_model: args.baseModel,
    examples: [
      {
        input: "Replace this with representative training input.",
        output: "Replace this with the expected training output.",
      },
      {
        input: "Replace this with a different validation input.",
        output: "Replace this with the expected validation output.",
      },
    ],
    hyperparameters: {
      n_epochs: 1,
    },
  });
  await mkdir(dirname(args.outputPath), { recursive: true });
  await writeFile(args.outputPath, `${JSON.stringify(spec, null, 2)}\n`, "utf8");
  return spec;
}

export async function initLocalRunnerConfigFile(args: CreateLocalRunnerConfigArgs): Promise<Record<string, unknown>> {
  if (!args.force && await exists(args.outputPath)) {
    throw new Error(`Refusing to overwrite existing file: ${args.outputPath}`);
  }
  const config = {
    artifactRoot: ".tt-local/artifacts",
    storeRoot: ".tt-local/store",
    evaluation: {
      inference: {
        device: "cuda",
      },
      scoring: {
        mode: "exact_match",
      },
      timeoutMs: 1_800_000,
    },
  } satisfies Record<string, unknown>;
  await mkdir(dirname(args.outputPath), { recursive: true });
  await writeFile(args.outputPath, `${JSON.stringify(config, null, 2)}\n`, "utf8");
  return config;
}

export function generatedPlaceholderIssues(request: FineTuneRunRequest): string[] {
  const issues: string[] = [];
  if (/describe the behavior this local model should learn/i.test(request.spec_snapshot.system_prompt)) {
    issues.push("system_prompt still contains the generated placeholder");
  }
  request.spec_snapshot.examples.forEach((example, index) => {
    if (/replace this with/i.test(example.input) || /replace this with/i.test(example.output)) {
      issues.push(`examples[${index}] still contains generated placeholder text`);
    }
  });
  return issues;
}

export function assertLocalRunInputReady(request: FineTuneRunRequest): void {
  const issues = generatedPlaceholderIssues(request);
  if (issues.length > 0) {
    throw new Error(`Edit the generated behavior spec before training: ${issues.join("; ")}.`);
  }
}

export function runRequestFromLocalSpec(
  spec: LocalBehaviorSpecFile,
  options: RunRequestFromSpecOptions = {},
): FineTuneRunRequest {
  const {
    id,
    hyperparameters,
    dataset_prebuilt: datasetPrebuilt,
    ...specSnapshot
  } = spec;
  const behaviorSpecId = id ?? deterministicBehaviorSpecId(specSnapshot);
  return fineTuneRunRequestSchema.parse({
    run_id: options.runId ?? randomUUID(),
    user_id: "local-user",
    behavior_spec_id: behaviorSpecId,
    run_number: 1,
    spec_snapshot: specSnapshot,
    hyperparameters,
    dataset_prebuilt: datasetPrebuilt,
  });
}

function deterministicBehaviorSpecId(spec: FineTuneRunRequest["spec_snapshot"]): string {
  const namespace = Buffer.from("9f783f01340457bea061942004259253", "hex");
  const digest = createHash("sha1")
    .update(namespace)
    .update(JSON.stringify(spec))
    .digest()
    .subarray(0, 16);
  digest[6] = (digest[6]! & 0x0f) | 0x50;
  digest[8] = (digest[8]! & 0x3f) | 0x80;
  const hex = digest.toString("hex");
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20),
  ].join("-");
}

export async function loadLocalRunInput(
  path: string,
  options: RunRequestFromSpecOptions = {},
): Promise<LocalRunInput> {
  const raw = resolveLocalRunInputPaths(JSON.parse(await readFile(path, "utf8")) as unknown, path);
  const request = fineTuneRunRequestSchema.safeParse(raw);
  if (request.success) {
    return { kind: "request", path, request: request.data };
  }
  const spec = localBehaviorSpecFileSchema.parse(raw);
  return {
    kind: "spec",
    path,
    spec,
    request: runRequestFromLocalSpec(spec, options),
  };
}
