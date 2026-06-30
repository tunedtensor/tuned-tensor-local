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
    id: "Qwen/Qwen3.5-4B",
    aliases: ["qwen/qwen3.5-4b", "Qwen/Qwen3.5-4B-Base", "qwen/qwen3.5-4b-base"],
    family: "qwen3_5",
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
  return resolveTrainingModel(modelId).id;
}
