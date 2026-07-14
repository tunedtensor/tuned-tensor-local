import { randomUUID } from "node:crypto";
import { readFile, stat, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";
import { homedir } from "node:os";
import { mkdir } from "node:fs/promises";
import {
  fineTuneHyperparametersSchema,
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
  userId?: string;
  runNumber?: number;
}

export interface LocalRunInput {
  kind: "request" | "spec";
  path: string;
  request: FineTuneRunRequest;
  spec?: LocalBehaviorSpecFile;
  warnings: string[];
}

function resolveLocalReference(value: unknown, baseDirectory: string): unknown {
  if (typeof value !== "string" || !value || isAbsolute(value) || /^[a-z][a-z0-9+.-]*:/i.test(value)) {
    return value;
  }
  if (value === "~") return homedir();
  if (value.startsWith("~/")) return resolve(homedir(), value.slice(2));
  return resolve(baseDirectory, value);
}

/** Resolve file-bearing spec fields relative to the spec/request file itself. */
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
  const spec = value.spec_snapshot && typeof value.spec_snapshot === "object" && !Array.isArray(value.spec_snapshot)
    ? value.spec_snapshot as Record<string, unknown>
    : value;
  if (Array.isArray(spec.examples)) {
    for (const example of spec.examples) {
      if (!example || typeof example !== "object" || Array.isArray(example)) continue;
      const assets = (example as Record<string, unknown>).input_assets;
      if (!Array.isArray(assets)) continue;
      for (const asset of assets) {
        if (!asset || typeof asset !== "object" || Array.isArray(asset)) continue;
        const fields = asset as Record<string, unknown>;
        for (const key of ["image", "path", "uri"] as const) {
          if (fields[key] !== undefined) fields[key] = resolveLocalReference(fields[key], baseDirectory);
        }
      }
    }
  }
  return value;
}

/**
 * Custom command workflows may pass arbitrary hyperparameter keys through to
 * their adapter. Still surface unfamiliar keys because the bundled trainer
 * ignores them, and a typo like `per_device_train_batch_size` can otherwise
 * look valid while the run trains with defaults.
 */
export function unknownHyperparameterWarnings(raw: unknown): string[] {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return [];
  const hyperparameters = (raw as Record<string, unknown>).hyperparameters;
  if (!hyperparameters || typeof hyperparameters !== "object" || Array.isArray(hyperparameters)) return [];
  const knownKeys = new Set(Object.keys(fineTuneHyperparametersSchema.shape));
  return Object.keys(hyperparameters as Record<string, unknown>)
    .filter((key) => !knownKeys.has(key))
    .map((key) => `Unknown hyperparameter "${key}" will be passed through but may be ignored by the default trainer. Known keys: ${[...knownKeys].join(", ")}.`);
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
        input: "Replace this with a representative input.",
        output: "Replace this with the expected output.",
      },
    ],
    hyperparameters: {
      n_epochs: 1,
      save_adapter_only: true,
      augment: false,
      use_llm_judge: false,
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
    training: {
      project: "training/local-runner",
    },
    evaluation: {
      inference: {
        provider: "transformers",
        project: "training/local-runner",
        device: "cuda",
      },
      scoring: {
        mode: "exact_match",
        fallback: "fail",
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
    user_id: userId,
    run_number: runNumber,
    training_method: trainingMethod,
    hyperparameters,
    artifacts,
    dataset_prebuilt: datasetPrebuilt,
    ...specSnapshot
  } = spec;
  return fineTuneRunRequestSchema.parse({
    run_id: options.runId ?? randomUUID(),
    user_id: options.userId ?? userId,
    behavior_spec_id: id ?? randomUUID(),
    run_number: options.runNumber ?? runNumber,
    training_method: trainingMethod,
    spec_snapshot: specSnapshot,
    hyperparameters,
    artifacts,
    dataset_prebuilt: datasetPrebuilt,
  });
}

export async function loadLocalRunInput(
  path: string,
  options: RunRequestFromSpecOptions = {},
): Promise<LocalRunInput> {
  const raw = resolveLocalRunInputPaths(JSON.parse(await readFile(path, "utf8")) as unknown, path);
  const warnings = unknownHyperparameterWarnings(raw);
  const request = fineTuneRunRequestSchema.safeParse(raw);
  if (request.success) {
    return { kind: "request", path, request: request.data, warnings };
  }
  const spec = localBehaviorSpecFileSchema.parse(raw);
  return {
    kind: "spec",
    path,
    spec,
    request: runRequestFromLocalSpec(spec, options),
    warnings,
  };
}
