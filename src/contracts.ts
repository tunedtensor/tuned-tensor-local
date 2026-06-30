import { z } from "zod";
import { canonicalizeTrainingModel } from "./model-registry.js";

export const documentInputAssetSchema = z.object({
  type: z.literal("image").default("image"),
  mime_type: z.string().min(1).optional(),
  image: z.string().min(1).optional(),
  data_uri: z.string().min(1).optional(),
  uri: z.string().min(1).optional(),
  path: z.string().min(1).optional(),
  page: z.number().int().min(1).optional(),
}).refine(
  (asset) => Boolean(asset.image || asset.data_uri || asset.uri || asset.path),
  { message: "image asset must include image, data_uri, uri, or path" },
);

export const behaviorSpecExampleSchema = z.object({
  input: z.string().min(1),
  output: z.string().min(1),
  input_assets: z.array(documentInputAssetSchema).optional(),
  modality: z.literal("document_ocr").optional(),
});

export const specSnapshotSchema = z.object({
  name: z.string().min(1),
  description: z.string().default(""),
  system_prompt: z.string().default(""),
  guidelines: z.array(z.string()).default([]),
  examples: z.array(behaviorSpecExampleSchema).default([]),
  constraints: z.array(z.string()).default([]),
  base_model: z.string().transform((value) => canonicalizeTrainingModel(value)),
});

export const datasetPrebuiltSchema = z.object({
  training: z.string().min(1),
  validation: z.string().min(1).optional(),
  test: z.string().min(1).optional(),
  format: z.enum(["chat_jsonl", "document_ocr_chat_jsonl"]).optional(),
});

export const fineTuneHyperparametersSchema = z.object({
  n_epochs: z.number().int().min(1).max(20).default(3),
  learning_rate: z.number().positive().optional(),
  batch_size: z.number().int().min(1).optional(),
  lora_rank: z.number().int().min(1).optional(),
  lora_alpha: z.number().int().min(1).optional(),
  lora_dropout: z.number().min(0).max(1).optional(),
  max_seq_length: z.number().int().min(128).max(32768).optional(),
  gradient_accumulation_steps: z.number().int().min(1).optional(),
  save_adapter_only: z.boolean().default(false),
  augment: z.boolean().default(false),
  use_llm_judge: z.boolean().default(false),
  max_eval_examples: z.number().int().min(1).optional(),
});

export const localArtifactsSchema = z.object({
  prefix: z.string().min(1).optional(),
});

export const fineTuneRunRequestSchema = z.object({
  run_id: z.string().uuid(),
  user_id: z.string().min(1),
  behavior_spec_id: z.string().uuid(),
  run_number: z.number().int().min(1),
  spec_snapshot: specSnapshotSchema,
  hyperparameters: fineTuneHyperparametersSchema.default({
    n_epochs: 3,
    save_adapter_only: false,
    augment: false,
    use_llm_judge: false,
  }),
  artifacts: localArtifactsSchema.optional(),
  dataset_prebuilt: datasetPrebuiltSchema.optional(),
}).superRefine((request, ctx) => {
  if (request.spec_snapshot.examples.length === 0 && !request.dataset_prebuilt) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["spec_snapshot", "examples"],
      message: "spec_snapshot.examples must have at least 1 example unless dataset_prebuilt is provided",
    });
  }
});

export const evalExampleResultSchema = z.object({
  prompt: z.string(),
  expected: z.string(),
  actual: z.string(),
  passed: z.boolean(),
  score: z.number().min(0).max(1),
  reasoning: z.string().nullable(),
  latency_ms: z.number().int().nonnegative(),
});

export const evalReportSchema = z.object({
  kind: z.enum(["baseline", "candidate"]),
  model_id: z.string(),
  total: z.number().int().nonnegative(),
  eval_examples_total: z.number().int().nonnegative(),
  eval_examples_used: z.number().int().nonnegative(),
  eval_truncated: z.boolean(),
  avg_score: z.number().min(0).max(1),
  pass_rate: z.number().min(0).max(1),
  exact_match_rate: z.number().min(0).max(1),
  avg_latency_ms: z.number().int().nonnegative(),
  results: z.array(evalExampleResultSchema),
  artifact_uri: z.string(),
  scoring_method: z.enum(["heuristic", "command"]),
  judge_model_id: z.string().nullable().optional(),
});

export const comparisonReportSchema = z.object({
  avg_score_delta: z.number(),
  pass_rate_delta: z.number(),
  exact_match_rate_delta: z.number(),
  regressions: z.number().int().nonnegative(),
  improvements: z.number().int().nonnegative(),
  regressed_examples: z.array(z.object({
    prompt: z.string(),
    old_score: z.number(),
    new_score: z.number(),
  })),
});

export const trainingReportSchema = z.object({
  provider: z.literal("local-uv"),
  training_job_name: z.string(),
  model_artifact_uri: z.string().optional(),
  base_model_artifact_uri: z.string().optional(),
  metrics: z.record(z.string(), z.unknown()).nullable(),
  exit_code: z.number().int().nullable(),
  log_uri: z.string(),
  command: z.array(z.string()).optional(),
});

export const runReportSchema = z.object({
  run_id: z.string().uuid(),
  behavior_spec_id: z.string().uuid(),
  user_id: z.string(),
  run_number: z.number().int().min(1),
  base_model: z.string(),
  fine_tuned_model_id: z.string(),
  status: z.enum(["completed", "failed"]),
  baseline: evalReportSchema,
  candidate: evalReportSchema,
  comparison: comparisonReportSchema,
  training: trainingReportSchema,
  artifact_uris: z.object({
    dataset: z.string(),
    baseline_eval: z.string(),
    candidate_eval: z.string(),
    report: z.string(),
  }),
  run_metadata: z.object({
    base_model: z.string(),
    fine_tuned_model_id: z.string(),
    dataset_prebuilt: z.boolean(),
    dataset_uri: z.string(),
    spec_example_count: z.number().int().nonnegative(),
    training_example_count: z.number().int().nonnegative().nullable(),
    eval_examples_total: z.number().int().nonnegative(),
    eval_examples_used: z.number().int().nonnegative(),
    started_at: z.string(),
    completed_at: z.string(),
    elapsed_ms: z.number().int().nonnegative(),
    elapsed_seconds: z.number().nonnegative(),
  }),
  created_at: z.string(),
});

export const commandSchema = z.array(z.string().min(1)).min(1);

export const localRunnerConfigSchema = z.object({
  artifactRoot: z.string().default(".tt-local/artifacts"),
  dryRun: z.boolean().default(false),
  training: z.object({
    backend: z.literal("uv").default("uv"),
    project: z.string().optional(),
    cwd: z.string().optional(),
    module: z.string().optional(),
    script: z.string().optional(),
    args: z.array(z.string()).default([]),
    with: z.array(z.string()).default([]),
    env: z.record(z.string(), z.string()).default({}),
  }).default({
    backend: "uv",
    args: [],
    with: [],
    env: {},
  }),
  paths: z.object({
    baseModel: z.string().optional(),
    modelCache: z.string().optional(),
  }).default({}),
  evaluation: z.object({
    mode: z.enum(["heuristic", "command"]).default("heuristic"),
    baselineCommand: commandSchema.optional(),
    candidateCommand: commandSchema.optional(),
    timeoutMs: z.number().int().min(100).default(120_000),
    maxExamples: z.number().int().min(1).optional(),
  }).default({
    mode: "heuristic",
    timeoutMs: 120_000,
  }),
});

export type DocumentInputAsset = z.infer<typeof documentInputAssetSchema>;
export type BehaviorSpecExample = z.infer<typeof behaviorSpecExampleSchema>;
export type SpecSnapshot = z.infer<typeof specSnapshotSchema>;
export type FineTuneHyperparameters = z.infer<typeof fineTuneHyperparametersSchema>;
export type FineTuneRunRequest = z.infer<typeof fineTuneRunRequestSchema>;
export type EvalExampleResult = z.infer<typeof evalExampleResultSchema>;
export type EvalReport = z.infer<typeof evalReportSchema>;
export type ComparisonReport = z.infer<typeof comparisonReportSchema>;
export type TrainingReport = z.infer<typeof trainingReportSchema>;
export type RunReport = z.infer<typeof runReportSchema>;
export type LocalRunnerConfig = z.infer<typeof localRunnerConfigSchema>;
