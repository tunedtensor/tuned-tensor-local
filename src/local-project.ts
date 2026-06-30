import { randomUUID } from "node:crypto";
import { readFile, stat, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
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
      n_epochs: 3,
      save_adapter_only: true,
      augment: false,
      use_llm_judge: false,
    },
  });
  await mkdir(dirname(args.outputPath), { recursive: true });
  await writeFile(args.outputPath, `${JSON.stringify(spec, null, 2)}\n`, "utf8");
  return spec;
}

export function runRequestFromLocalSpec(
  spec: LocalBehaviorSpecFile,
  options: RunRequestFromSpecOptions = {},
): FineTuneRunRequest {
  const {
    id,
    user_id: userId,
    run_number: runNumber,
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
  const raw = JSON.parse(await readFile(path, "utf8")) as unknown;
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
