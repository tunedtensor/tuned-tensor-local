import { lstat, readdir } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { verifyTarGzipArchive } from "./artifacts.js";

export interface TrainingModel {
  id: string;
  family: string;
  defaultLearningRate: number;
  defaultPerDeviceBatchSize: number;
  defaultGradientAccumulationSteps: number;
  defaultLoraRank: number;
  defaultLoraAlpha: number;
  defaultLoraDropout: number;
  defaultMaxSeqLength: number;
}

export const TRAINING_MODELS: TrainingModel[] = [
  {
    id: "Qwen/Qwen3.5-2B",
    family: "qwen3_5",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 8,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 2048,
  },
];

export function resolveTrainingModel(modelId: string): TrainingModel {
  const normalized = modelId.trim().toLowerCase();
  const model = TRAINING_MODELS.find((candidate) =>
    candidate.id.toLowerCase() === normalized
  );
  if (!model) {
    throw new Error(
      `Unsupported base model "${modelId}". Supported model: ${TRAINING_MODELS[0]!.id}`,
    );
  }
  return model;
}

export function canonicalizeTrainingModel(modelId: string): string {
  return resolveTrainingModel(modelId).id;
}

const CERTIFIED_TEXT_CONFIG = {
  model_type: "qwen3_5_text",
  hidden_size: 2048,
  num_hidden_layers: 24,
  num_attention_heads: 8,
  num_key_value_heads: 2,
  intermediate_size: 6144,
  vocab_size: 248320,
} as const;

/** Reject a same-family snapshot that is not the certified 2B architecture. */
export function assertCertifiedBaseModelConfig(
  value: unknown,
  label = "base-model config.json",
): void {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must contain a JSON object.`);
  }
  const config = value as Record<string, unknown>;
  const architectures = config.architectures;
  const textConfig = config.text_config;
  if (
    config.model_type !== "qwen3_5"
    || !Array.isArray(architectures)
    || !architectures.includes("Qwen3_5ForConditionalGeneration")
    || !textConfig
    || typeof textConfig !== "object"
    || Array.isArray(textConfig)
  ) {
    throw new Error(
      `${label} is not the certified Qwen/Qwen3.5-2B architecture.`,
    );
  }
  const text = textConfig as Record<string, unknown>;
  for (const [key, expected] of Object.entries(CERTIFIED_TEXT_CONFIG)) {
    if (text[key] !== expected) {
      throw new Error(
        `${label} is not the certified Qwen/Qwen3.5-2B architecture: `
        + `text_config.${key} must be ${JSON.stringify(expected)}.`,
      );
    }
  }
}

export interface ModelArtifactInspection {
  uri: string;
  path: string;
  kind: "file" | "directory";
  adapter_weight_file_count: number;
  adapter_weight_bytes: number;
  has_adapter_config: boolean;
}

export function localModelArtifactPath(uri: string): string {
  if (uri.startsWith("file://")) return fileURLToPath(new URL(uri));
  if (/^[a-z][a-z0-9+.-]*:/i.test(uri)) {
    throw new Error(`Model artifact must be a local path or file URI: ${uri}`);
  }
  return resolve(uri);
}

function adapterWeightName(name: string): boolean {
  const lower = name.toLowerCase();
  return lower === "adapter_model.safetensors" || lower === "adapter_model.bin";
}

async function inspectAdapterDirectory(path: string): Promise<{
  count: number;
  bytes: number;
  hasConfig: boolean;
}> {
  let count = 0;
  let bytes = 0;
  let hasConfig = false;
  for (const entry of await readdir(path, { withFileTypes: true })) {
    const child = resolve(path, entry.name);
    if (entry.isSymbolicLink()) {
      throw new Error(`Model artifact must not contain symbolic links: ${child}`);
    }
    if (entry.isDirectory()) {
      const nested = await inspectAdapterDirectory(child);
      count += nested.count;
      bytes += nested.bytes;
      hasConfig ||= nested.hasConfig;
      continue;
    }
    if (!entry.isFile()) continue;
    const metadata = await lstat(child);
    if (adapterWeightName(entry.name) && metadata.size > 0) {
      count += 1;
      bytes += metadata.size;
    }
    if (entry.name.toLowerCase() === "adapter_config.json" && metadata.size > 0) {
      hasConfig = true;
    }
  }
  return { count, bytes, hasConfig };
}

export async function inspectModelArtifact(uri: string): Promise<ModelArtifactInspection> {
  const path = localModelArtifactPath(uri);
  const metadata = await lstat(path).catch(() => null);
  if (!metadata) throw new Error(`Model artifact does not exist: ${path}`);
  if (metadata.isSymbolicLink()) {
    throw new Error(`Model artifact must not be a symbolic link: ${path}`);
  }
  if (metadata.isDirectory()) {
    const adapter = await inspectAdapterDirectory(path);
    return {
      uri,
      path,
      kind: "directory",
      adapter_weight_file_count: adapter.count,
      adapter_weight_bytes: adapter.bytes,
      has_adapter_config: adapter.hasConfig,
    };
  }
  if (!metadata.isFile() || !path.toLowerCase().endsWith(".tar.gz")) {
    throw new Error(
      `Model artifact must be a PEFT adapter directory or .tar.gz archive: ${path}`,
    );
  }
  const archive = await verifyTarGzipArchive(path);
  return {
    uri,
    path,
    kind: "file",
    adapter_weight_file_count: archive.adapter_weight_entries,
    adapter_weight_bytes: archive.adapter_weight_bytes,
    has_adapter_config: archive.adapter_config_entries > 0,
  };
}

export async function assertUsableModelArtifact(
  uri: string,
): Promise<ModelArtifactInspection> {
  const inspection = await inspectModelArtifact(uri);
  if (
    inspection.adapter_weight_file_count === 0
    || inspection.adapter_weight_bytes === 0
  ) {
    throw new Error(
      `PEFT model artifact ${inspection.path} contains no non-empty adapter_model weights.`,
    );
  }
  if (!inspection.has_adapter_config) {
    throw new Error(
      `PEFT model artifact ${inspection.path} contains adapter weights but no non-empty adapter_config.json.`,
    );
  }
  return inspection;
}
