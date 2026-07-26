import { z } from "zod";
import { canonicalizeTrainingModel } from "./model-registry.js";

/**
 * TT Local deliberately has one training contract today: text supervised
 * fine-tuning with a LoRA adapter. Additional methods should only be added
 * after they have their own end-to-end CUDA acceptance test.
 */
export const datasetFormatSchema = z.literal("chat_jsonl");
export const baseModelRevisionSchema = z.string()
  .regex(/^[0-9a-f]{40}$/i, "base_model_revision must be a 40-character Hugging Face commit SHA")
  .transform((value) => value.toLowerCase());

export const behaviorSpecExampleSchema = z.object({
  input: z.string().min(1),
  output: z.string().min(1),
}).strict();

export const specSnapshotSchema = z.object({
  name: z.string().min(1),
  description: z.string().default(""),
  system_prompt: z.string().default(""),
  guidelines: z.array(z.string()).default([]),
  examples: z.array(behaviorSpecExampleSchema).default([]),
  constraints: z.array(z.string()).default([]),
  base_model: z.string().transform((value) => canonicalizeTrainingModel(value)),
}).strict();

export const datasetPrebuiltSchema = z.object({
  training: z.string().min(1),
  validation: z.string().min(1).optional(),
  test: z.string().min(1).optional(),
  format: datasetFormatSchema.default("chat_jsonl"),
}).strict().superRefine((dataset, context) => {
  if (!dataset.validation && !dataset.test) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["validation"],
      message: "Provide a distinct validation or test split for held-out evaluation",
    });
  }
});

export const fineTuneHyperparametersSchema = z.object({
  n_epochs: z.number().int().min(1).max(20).default(1),
  learning_rate: z.number().positive().optional(),
  batch_size: z.number().int().min(1).optional(),
  lora_rank: z.number().int().min(1).optional(),
  lora_alpha: z.number().int().min(1).optional(),
  lora_dropout: z.number().min(0).max(1).optional(),
  max_seq_length: z.number().int().min(128).max(32768).optional(),
  gradient_accumulation_steps: z.number().int().min(1).optional(),
  max_eval_examples: z.number().int().min(1).optional(),
  /** Immutable Hugging Face commit used by every stage of the run. */
  base_model_revision: baseModelRevisionSchema.optional(),
}).strict();

export const fineTuneRunRequestSchema = z.object({
  run_id: z.string().uuid(),
  user_id: z.string().min(1),
  behavior_spec_id: z.string().uuid(),
  run_number: z.number().int().min(1),
  spec_snapshot: specSnapshotSchema,
  hyperparameters: fineTuneHyperparametersSchema.default({ n_epochs: 1 }),
  dataset_prebuilt: datasetPrebuiltSchema.optional(),
}).strict().superRefine((request, context) => {
  if (request.spec_snapshot.examples.length === 0 && !request.dataset_prebuilt) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["spec_snapshot", "examples"],
      message: "Add examples or provide dataset_prebuilt training and validation files",
    });
  }
});

export const localBehaviorSpecFileSchema = specSnapshotSchema.extend({
  id: z.string().uuid().optional(),
  hyperparameters: fineTuneHyperparametersSchema.optional(),
  dataset_prebuilt: datasetPrebuiltSchema.optional(),
}).strict();

export const evalExampleResultSchema = z.object({
  prompt: z.string(),
  expected: z.string(),
  actual: z.string(),
  passed: z.boolean(),
  score: z.number().min(0).max(1),
  reasoning: z.string().nullable(),
  latency_ms: z.number().int().nonnegative(),
  scored_by: z.enum(["exact_match", "json_fields", "heuristic"]).optional(),
}).strict();

export const jsonFieldMetricsSchema = z.object({
  fields: z.array(z.string()),
  valid_json_count: z.number().int().nonnegative(),
  valid_json_rate: z.number().min(0).max(1),
  schema_match_count: z.number().int().nonnegative(),
  schema_match_rate: z.number().min(0).max(1),
  all_fields_match_count: z.number().int().nonnegative(),
  all_fields_match_rate: z.number().min(0).max(1),
  field_accuracy: z.record(z.string(), z.object({
    correct: z.number().int().nonnegative(),
    total: z.number().int().nonnegative(),
    accuracy: z.number().min(0).max(1),
  }).strict()),
}).strict();

export const evalSplitSchema = z.enum([
  "spec_examples",
  "spec_holdout",
  "prebuilt_test",
  "prebuilt_validation",
  "general_regression",
]);

export const evalReportSchema = z.object({
  kind: z.enum(["baseline", "candidate"]),
  model_id: z.string(),
  total: z.number().int().nonnegative(),
  eval_examples_total: z.number().int().nonnegative(),
  eval_examples_used: z.number().int().nonnegative(),
  eval_truncated: z.boolean(),
  eval_split: evalSplitSchema.optional(),
  eval_sample_seed: z.number().int().nullable().optional(),
  avg_score: z.number().min(0).max(1),
  pass_rate: z.number().min(0).max(1),
  exact_match_rate: z.number().min(0).max(1),
  avg_token_f1: z.number().min(0).max(1).optional(),
  avg_latency_ms: z.number().int().nonnegative(),
  cached: z.boolean().optional(),
  cache_key: z.string().optional(),
  results: z.array(evalExampleResultSchema),
  artifact_uri: z.string(),
  scoring_method: z.enum(["heuristic", "exact_match", "json_fields"]),
  inference_provider: z.enum(["none", "transformers"]).optional(),
  scoring_mode: z.enum(["exact_match", "json_fields"]).optional(),
  json_field_metrics: jsonFieldMetricsSchema.optional(),
  generation_config: z.record(z.string(), z.unknown()).optional(),
  log_uri: z.string().optional(),
}).strict();

export const comparisonReportSchema = z.object({
  avg_score_delta: z.number(),
  pass_rate_delta: z.number(),
  exact_match_rate_delta: z.number(),
  token_f1_delta: z.number().optional(),
  regressions: z.number().int().nonnegative(),
  improvements: z.number().int().nonnegative(),
  regressed_examples: z.array(z.object({
    prompt: z.string(),
    old_score: z.number(),
    new_score: z.number(),
  }).strict()),
}).strict();

export const trainingReportSchema = z.object({
  provider: z.literal("local-uv"),
  training_job_name: z.string(),
  model_artifact_uri: z.string().optional(),
  base_model_artifact_uri: z.string().optional(),
  metrics: z.record(z.string(), z.unknown()).nullable(),
  exit_code: z.number().int().nullable(),
  log_uri: z.string(),
  command: z.array(z.string()).optional(),
}).strict();

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
  general_regression: z.object({
    dataset_uri: z.string(),
    dataset_sha256: z.string().regex(/^[a-f0-9]{64}$/),
    baseline: evalReportSchema,
    candidate: evalReportSchema,
    comparison: comparisonReportSchema,
    policy: z.object({
      max_score_drop: z.number().min(0).max(1),
      max_pass_rate_drop: z.number().min(0).max(1),
    }).strict(),
    passed: z.boolean(),
    failures: z.array(z.string()),
  }).strict().optional(),
  training: trainingReportSchema,
  artifact_uris: z.object({
    dataset: z.string(),
    baseline_eval: z.string(),
    candidate_eval: z.string(),
    general_baseline_eval: z.string().optional(),
    general_candidate_eval: z.string().optional(),
    report: z.string(),
  }).strict(),
  run_metadata: z.object({
    base_model: z.string(),
    fine_tuned_model_id: z.string(),
    dataset_prebuilt: z.boolean(),
    dataset_format: datasetFormatSchema.nullable().optional(),
    dataset_uri: z.string(),
    spec_example_count: z.number().int().nonnegative(),
    training_example_count: z.number().int().nonnegative().nullable(),
    eval_examples_total: z.number().int().nonnegative(),
    eval_examples_used: z.number().int().nonnegative(),
    eval_split: evalSplitSchema.optional(),
    eval_sample_seed: z.number().int().nullable().optional(),
    started_at: z.string(),
    completed_at: z.string(),
    elapsed_ms: z.number().int().nonnegative(),
    elapsed_seconds: z.number().nonnegative(),
  }).strict(),
  created_at: z.string(),
}).strict();

const inferenceConfigSchema = z.object({
  maxNewTokens: z.number().int().min(1).default(256),
  temperature: z.number().min(0).default(0),
  topP: z.number().min(0).max(1).default(1),
  device: z.enum(["cuda", "cpu"]).default("cuda"),
}).strict();

const scoringConfigSchema = z.object({
  mode: z.enum(["exact_match", "json_fields"]).default("exact_match"),
  fields: z.array(z.string().min(1)).optional(),
}).strict();

const evaluationConfigSchema = z.object({
  inference: inferenceConfigSchema.default({
    maxNewTokens: 256,
    temperature: 0,
    topP: 1,
    device: "cuda",
  }),
  scoring: scoringConfigSchema.default({ mode: "exact_match" }),
  timeoutMs: z.number().int().min(100).default(1_800_000),
  maxExamples: z.number().int().min(1).optional(),
  sampleSeed: z.number().int().optional(),
  baselineCache: z.boolean().default(true),
  generalRegression: z.object({
    dataset: z.string().min(1),
    systemPrompt: z.string().optional(),
    maxScoreDrop: z.number().min(0).max(1).default(0.03),
    maxPassRateDrop: z.number().min(0).max(1).default(0.05),
  }).strict().optional(),
}).strict();

export const localRunnerConfigSchema = z.object({
  storeRoot: z.string().optional(),
  artifactRoot: z.string().default(".tt-local/artifacts"),
  dryRun: z.boolean().default(false),
  paths: z.object({
    baseModel: z.string().optional(),
    modelCache: z.string().optional(),
  }).strict().default({}),
  evaluation: evaluationConfigSchema.default({
    inference: {
      maxNewTokens: 256,
      temperature: 0,
      topP: 1,
      device: "cuda",
    },
    scoring: { mode: "exact_match" },
    timeoutMs: 1_800_000,
    baselineCache: true,
  }),
}).strict();

export type BehaviorSpecExample = z.infer<typeof behaviorSpecExampleSchema>;
export type SpecSnapshot = z.infer<typeof specSnapshotSchema>;
export type FineTuneHyperparameters = z.infer<typeof fineTuneHyperparametersSchema>;
export type FineTuneRunRequest = z.infer<typeof fineTuneRunRequestSchema>;
export type LocalBehaviorSpecFile = z.infer<typeof localBehaviorSpecFileSchema>;
export type EvalExampleResult = z.infer<typeof evalExampleResultSchema>;
export type EvalSplit = z.infer<typeof evalSplitSchema>;
export type EvalReport = z.infer<typeof evalReportSchema>;
export type ComparisonReport = z.infer<typeof comparisonReportSchema>;
export type TrainingReport = z.infer<typeof trainingReportSchema>;
export type RunReport = z.infer<typeof runReportSchema>;
export type LocalRunnerConfig = z.infer<typeof localRunnerConfigSchema>;
