import { lstat, readdir } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { verifyTarGzipArchive } from "./artifacts.js";

export interface TrainingModel {
  id: string;
  aliases: string[];
  family: string;
  loader: "causal_lm" | "image_text_to_text";
  defaultLearningRate: number;
  defaultPerDeviceBatchSize: number;
  defaultGradientAccumulationSteps: number;
  defaultLoraRank: number;
  defaultLoraAlpha: number;
  defaultLoraDropout: number;
  defaultMaxSeqLength: number;
  trustRemoteCode: boolean;
  requiresHfToken: boolean;
}

export const TRAINING_MODELS: TrainingModel[] = [
  {
    id: "Qwen/Qwen3.5-2B",
    aliases: ["qwen/qwen3.5-2b", "Qwen/Qwen3.5-2B-Base", "qwen/qwen3.5-2b-base"],
    family: "qwen3_5",
    loader: "causal_lm",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 8,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 2048,
    trustRemoteCode: true,
    requiresHfToken: false,
  },
  {
    id: "Qwen/Qwen3.5-4B",
    aliases: ["qwen/qwen3.5-4b", "Qwen/Qwen3.5-4B-Base", "qwen/qwen3.5-4b-base"],
    family: "qwen3_5",
    loader: "causal_lm",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 16,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 2048,
    trustRemoteCode: true,
    requiresHfToken: false,
  },
  {
    id: "Qwen/Qwen3-VL-2B-Instruct",
    aliases: ["qwen/qwen3-vl-2b-instruct", "Qwen/Qwen3-VL-2B", "qwen/qwen3-vl-2b"],
    family: "qwen3_vl",
    loader: "image_text_to_text",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 8,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 2048,
    trustRemoteCode: true,
    requiresHfToken: false,
  },
  {
    id: "google/gemma-4-E2B-it",
    aliases: ["google/gemma-4-E2B", "google/gemma-4-e2b", "google/gemma-4-e2b-it"],
    family: "gemma4",
    loader: "image_text_to_text",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 8,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 2048,
    trustRemoteCode: true,
    requiresHfToken: false,
  },
  {
    id: "google/gemma-4-E4B-it",
    aliases: ["google/gemma-4-E4B", "google/gemma-4-e4b", "google/gemma-4-e4b-it"],
    family: "gemma4",
    loader: "image_text_to_text",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 16,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 2048,
    trustRemoteCode: true,
    requiresHfToken: false,
  },
  {
    id: "meta-llama/Llama-3.2-3B-Instruct",
    aliases: ["meta-llama/llama-3.2-3b-instruct", "meta-llama/Llama-3.2-3B"],
    family: "llama3_2",
    loader: "causal_lm",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 16,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 2048,
    trustRemoteCode: true,
    requiresHfToken: true,
  },
  {
    id: "microsoft/Phi-4-mini-instruct",
    aliases: ["microsoft/phi-4-mini-instruct", "phi-4-mini-instruct"],
    family: "phi4",
    loader: "causal_lm",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 16,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 2048,
    trustRemoteCode: false,
    requiresHfToken: false,
  },
  {
    id: "ibm-granite/granite-3.3-2b-instruct",
    aliases: ["granite-3.3-2b-instruct"],
    family: "granite3_3",
    loader: "causal_lm",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 8,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 2048,
    trustRemoteCode: true,
    requiresHfToken: false,
  },
  {
    id: "bigcode/starcoder2-3b",
    aliases: ["starcoder2-3b"],
    family: "starcoder2",
    loader: "causal_lm",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 8,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 2048,
    trustRemoteCode: true,
    requiresHfToken: false,
  },
];

const EXTERNAL_MODEL_PREFIXES = ["external:", "command:"];

export function isExternalTrainingModel(modelId: string): boolean {
  const normalized = modelId.trim().toLowerCase();
  return EXTERNAL_MODEL_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}

export function resolveTrainingModel(modelId: string): TrainingModel {
  const normalized = modelId.trim().toLowerCase();
  const model = TRAINING_MODELS.find((candidate) =>
    candidate.id.toLowerCase() === normalized
    || candidate.aliases.some((alias) => alias.toLowerCase() === normalized)
  );
  if (!model) {
    const supported = TRAINING_MODELS.map((candidate) => candidate.id).join(", ");
    throw new Error(`Unsupported base model "${modelId}". Supported models: ${supported}`);
  }
  return model;
}

export function canonicalizeTrainingModel(modelId: string): string {
  if (isExternalTrainingModel(modelId)) {
    const trimmed = modelId.trim();
    const [prefix, ...rest] = trimmed.split(":");
    const id = rest.join(":").trim();
    if (!id) throw new Error(`External model "${modelId}" must include an id after "${prefix}:".`);
    return `${prefix.toLowerCase()}:${id}`;
  }
  return resolveTrainingModel(modelId).id;
}

export interface ModelArtifactInspection {
  uri: string;
  path: string;
  kind: "file" | "directory";
  file_count: number;
  total_bytes: number;
  payload_file_count: number;
  payload_bytes: number;
  recognized_payload_file_count: number;
  recognized_payload_bytes: number;
  adapter_weight_file_count: number;
  adapter_weight_bytes: number;
  full_model_weight_file_count: number;
  full_model_weight_bytes: number;
  has_adapter_config: boolean;
}

export function localModelArtifactPath(uri: string): string {
  if (uri.startsWith("file://")) return fileURLToPath(new URL(uri));
  if (/^[a-z][a-z0-9+.-]*:/i.test(uri)) {
    throw new Error(`Model artifact must be a local path or file URI: ${uri}`);
  }
  return resolve(uri);
}

function isModelPayloadFile(path: string): boolean {
  const name = path.split(/[\\/]/).at(-1)?.toLowerCase() ?? "";
  if (
    name === "training-metrics.json"
    || name === "trainer_state.json"
    || name === "training_args.bin"
    || name === "config.json"
    || name === "generation_config.json"
    || name === "adapter_config.json"
    || name === "readme.md"
    || name.startsWith("tokenizer")
    || name.startsWith("special_tokens")
    || name.startsWith("added_tokens")
    || name === "vocab.json"
    || name === "merges.txt"
  ) return false;
  return true;
}

function isRecognizedModelPayloadFile(path: string): boolean {
  const name = path.split(/[\\/]/).at(-1)?.toLowerCase() ?? "";
  if (name === "training_args.bin") return false;
  return name.endsWith(".safetensors")
    || name.endsWith(".bin")
    || name.endsWith(".pt")
    || name.endsWith(".pth")
    || name.endsWith(".gguf")
    || name.endsWith(".onnx")
    || name.endsWith(".ckpt")
    || name.endsWith(".npz");
}

function isAdapterWeightFile(path: string): boolean {
  const name = path.split(/[\\/]/).at(-1)?.toLowerCase() ?? "";
  return name === "adapter_model.safetensors" || name === "adapter_model.bin";
}

function isFullModelWeightFile(path: string): boolean {
  const name = path.split(/[\\/]/).at(-1)?.toLowerCase() ?? "";
  return (name.endsWith(".safetensors") && !name.startsWith("adapter_"))
    || /^pytorch_model.*\.bin$/.test(name);
}

async function inspectDirectory(path: string): Promise<{
  fileCount: number;
  totalBytes: number;
  payloadFileCount: number;
  payloadBytes: number;
  recognizedPayloadFileCount: number;
  recognizedPayloadBytes: number;
  adapterWeightFileCount: number;
  adapterWeightBytes: number;
  fullModelWeightFileCount: number;
  fullModelWeightBytes: number;
  hasAdapterConfig: boolean;
}> {
  let fileCount = 0;
  let totalBytes = 0;
  let payloadFileCount = 0;
  let payloadBytes = 0;
  let recognizedPayloadFileCount = 0;
  let recognizedPayloadBytes = 0;
  let adapterWeightFileCount = 0;
  let adapterWeightBytes = 0;
  let fullModelWeightFileCount = 0;
  let fullModelWeightBytes = 0;
  let hasAdapterConfig = false;
  for (const entry of await readdir(path, { withFileTypes: true })) {
    const child = resolve(path, entry.name);
    if (entry.isSymbolicLink()) {
      throw new Error(`Model artifact must not contain symbolic links: ${child}`);
    }
    if (entry.isDirectory()) {
      const nested = await inspectDirectory(child);
      fileCount += nested.fileCount;
      totalBytes += nested.totalBytes;
      payloadFileCount += nested.payloadFileCount;
      payloadBytes += nested.payloadBytes;
      recognizedPayloadFileCount += nested.recognizedPayloadFileCount;
      recognizedPayloadBytes += nested.recognizedPayloadBytes;
      adapterWeightFileCount += nested.adapterWeightFileCount;
      adapterWeightBytes += nested.adapterWeightBytes;
      fullModelWeightFileCount += nested.fullModelWeightFileCount;
      fullModelWeightBytes += nested.fullModelWeightBytes;
      hasAdapterConfig ||= nested.hasAdapterConfig;
      continue;
    }
    if (!entry.isFile()) continue;
    const metadata = await lstat(child);
    if (entry.name.toLowerCase() === "adapter_config.json" && metadata.size > 0) hasAdapterConfig = true;
    fileCount += 1;
    totalBytes += metadata.size;
    if (isModelPayloadFile(child)) {
      payloadFileCount += 1;
      payloadBytes += metadata.size;
    }
    if (isRecognizedModelPayloadFile(child)) {
      recognizedPayloadFileCount += 1;
      recognizedPayloadBytes += metadata.size;
    }
    if (isAdapterWeightFile(child)) {
      adapterWeightFileCount += 1;
      adapterWeightBytes += metadata.size;
    }
    if (isFullModelWeightFile(child)) {
      fullModelWeightFileCount += 1;
      fullModelWeightBytes += metadata.size;
    }
  }
  return {
    fileCount,
    totalBytes,
    payloadFileCount,
    payloadBytes,
    recognizedPayloadFileCount,
    recognizedPayloadBytes,
    adapterWeightFileCount,
    adapterWeightBytes,
    fullModelWeightFileCount,
    fullModelWeightBytes,
    hasAdapterConfig,
  };
}

export async function inspectModelArtifact(uri: string): Promise<ModelArtifactInspection> {
  const path = localModelArtifactPath(uri);
  const metadata = await lstat(path).catch(() => null);
  if (!metadata) throw new Error(`Model artifact does not exist: ${path}`);
  if (metadata.isSymbolicLink()) throw new Error(`Model artifact must not be a symbolic link: ${path}`);
  if (metadata.isFile()) {
    return {
      uri,
      path,
      kind: "file",
      file_count: 1,
      total_bytes: metadata.size,
      payload_file_count: 1,
      payload_bytes: metadata.size,
      recognized_payload_file_count: isRecognizedModelPayloadFile(path) ? 1 : 0,
      recognized_payload_bytes: isRecognizedModelPayloadFile(path) ? metadata.size : 0,
      adapter_weight_file_count: isAdapterWeightFile(path) ? 1 : 0,
      adapter_weight_bytes: isAdapterWeightFile(path) ? metadata.size : 0,
      full_model_weight_file_count: isFullModelWeightFile(path) ? 1 : 0,
      full_model_weight_bytes: isFullModelWeightFile(path) ? metadata.size : 0,
      has_adapter_config: false,
    };
  }
  if (metadata.isDirectory()) {
    const contents = await inspectDirectory(path);
    return {
      uri,
      path,
      kind: "directory",
      file_count: contents.fileCount,
      total_bytes: contents.totalBytes,
      payload_file_count: contents.payloadFileCount,
      payload_bytes: contents.payloadBytes,
      recognized_payload_file_count: contents.recognizedPayloadFileCount,
      recognized_payload_bytes: contents.recognizedPayloadBytes,
      adapter_weight_file_count: contents.adapterWeightFileCount,
      adapter_weight_bytes: contents.adapterWeightBytes,
      full_model_weight_file_count: contents.fullModelWeightFileCount,
      full_model_weight_bytes: contents.fullModelWeightBytes,
      has_adapter_config: contents.hasAdapterConfig,
    };
  }
  throw new Error(`Model artifact is not a regular file or directory: ${path}`);
}

export async function assertUsableModelArtifact(
  uri: string,
  options: { allowUnrecognizedPayload?: boolean } = {},
): Promise<ModelArtifactInspection> {
  let inspection = await inspectModelArtifact(uri);
  if (inspection.kind === "file" && inspection.path.toLowerCase().endsWith(".tar.gz")) {
    const archive = await verifyTarGzipArchive(inspection.path);
    inspection = {
      ...inspection,
      recognized_payload_file_count: archive.recognized_payload_entries,
      recognized_payload_bytes: archive.recognized_payload_bytes,
      adapter_weight_file_count: archive.adapter_weight_entries,
      adapter_weight_bytes: archive.adapter_weight_bytes,
      full_model_weight_file_count: archive.full_model_weight_entries,
      full_model_weight_bytes: archive.full_model_weight_bytes,
      has_adapter_config: archive.adapter_config_entries > 0,
    };
  }
  if (inspection.payload_file_count === 0 || inspection.payload_bytes === 0) {
    throw new Error(`Training completed without a usable model artifact: ${inspection.path} has no model payload files.`);
  }
  if (
    !options.allowUnrecognizedPayload
    && inspection.adapter_weight_file_count === 0
    && inspection.full_model_weight_file_count === 0
  ) {
    throw new Error(`Model artifact ${inspection.path} has no loadable adapter or Transformers full-model weights.`);
  }
  if (!options.allowUnrecognizedPayload && inspection.adapter_weight_file_count > 0 && !inspection.has_adapter_config) {
    throw new Error(`Model artifact ${inspection.path} has adapter weights but no non-empty adapter_config.json.`);
  }
  if (!options.allowUnrecognizedPayload && inspection.has_adapter_config && inspection.adapter_weight_file_count === 0) {
    throw new Error(`Model artifact ${inspection.path} has adapter_config.json but no adapter_model weights.`);
  }
  return inspection;
}
