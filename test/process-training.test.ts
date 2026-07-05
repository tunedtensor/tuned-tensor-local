import assert from "node:assert/strict";
import { test } from "node:test";
import { buildTrainingHyperparameters, parseTrainingProgressLine } from "../src/process-training.js";
import { fineTuneRunRequestSchema } from "../src/contracts.js";

test("parses trainer metric dictionaries as progress snapshots", () => {
  const parsed = parseTrainingProgressLine("{'loss': '0.351', 'grad_norm': '0.6381', 'learning_rate': '4.098e-08', 'epoch': '1'}");

  assert.deepEqual(parsed, {
    loss: 0.351,
    grad_norm: 0.6381,
    learning_rate: 4.098e-8,
    epoch: 1,
    percent: 100,
  });
});

test("parses tqdm progress lines as progress snapshots", () => {
  const parsed = parseTrainingProgressLine(" 98%|█████████▊| 238/244 [29:25<00:48,  8.02s/it]");

  assert.deepEqual(parsed, {
    percent: 98,
    step: 238,
    total_steps: 244,
    elapsed: "29:25",
    eta: "00:48",
    rate: "8.02s/it",
  });
});

test("builds text and multimodal model loader hyperparameters", () => {
  const base = {
    run_id: "11111111-1111-4111-8111-111111111111",
    user_id: "user",
    behavior_spec_id: "22222222-2222-4222-8222-222222222222",
    run_number: 1,
    spec_snapshot: {
      name: "Example",
      description: "",
      system_prompt: "",
      base_model: "Qwen/Qwen3.5-2B",
      examples: [{ input: "hello", output: "greeting" }],
    },
  };

  const textRequest = fineTuneRunRequestSchema.parse(base);
  const textHyperparameters = buildTrainingHyperparameters(textRequest);
  assert.equal(textHyperparameters.model_loader, "causal_lm");
  assert.equal(textHyperparameters.dpo_beta, undefined);

  const multimodalRequest = fineTuneRunRequestSchema.parse({
    ...base,
    spec_snapshot: {
      ...base.spec_snapshot,
      base_model: "Qwen/Qwen3-VL-2B-Instruct",
    },
  });
  assert.equal(buildTrainingHyperparameters(multimodalRequest).model_loader, "image_text_to_text");
});

test("builds DPO uv hyperparameters with LoRA defaults", () => {
  const request = fineTuneRunRequestSchema.parse({
    run_id: "12121212-1212-4212-8212-121212121212",
    user_id: "user",
    behavior_spec_id: "34343434-3434-4434-8434-343434343434",
    run_number: 1,
    training_method: "dpo",
    spec_snapshot: {
      name: "DPO",
      description: "",
      system_prompt: "",
      base_model: "Qwen/Qwen3.5-2B",
      examples: [{ input: "hello", output: "greeting" }],
    },
    hyperparameters: {
      n_epochs: 1,
      dpo_beta: 0.2,
      dpo_loss_type: "ipo",
      dpo_label_smoothing: 0.1,
      dpo_reference_free: true,
      max_prompt_length: 512,
      max_completion_length: 256,
    },
    dataset_prebuilt: {
      training: "file:///tmp/preferences.jsonl",
      format: "preference_jsonl",
    },
  });

  const hyperparameters = buildTrainingHyperparameters(request);
  assert.equal(hyperparameters.training_method, "dpo");
  assert.equal(hyperparameters.model_loader, "causal_lm");
  assert.equal(hyperparameters.dpo_beta, "0.2");
  assert.equal(hyperparameters.dpo_loss_type, "ipo");
  assert.equal(hyperparameters.dpo_label_smoothing, "0.1");
  assert.equal(hyperparameters.dpo_reference_free, "true");
  assert.equal(hyperparameters.max_prompt_length, "512");
  assert.equal(hyperparameters.max_completion_length, "256");
  assert.equal(hyperparameters.lora_rank, "16");
});

test("command training hyperparameters allow external models and avoid bundled model defaults", () => {
  const request = fineTuneRunRequestSchema.parse({
    run_id: "33333333-3333-4333-8333-333333333333",
    user_id: "user",
    behavior_spec_id: "44444444-4444-4444-8444-444444444444",
    run_number: 1,
    spec_snapshot: {
      name: "External",
      description: "",
      system_prompt: "",
      base_model: "external:karpathy/nanochat",
      examples: [{ input: "hello", output: "greeting" }],
    },
    hyperparameters: {
      n_epochs: 1,
      nanochat_depth: 1,
      custom_options: { compile: false },
    },
  });

  const hyperparameters = buildTrainingHyperparameters(request, { backend: "command" });
  assert.equal(hyperparameters.base_model, "external:karpathy/nanochat");
  assert.equal(hyperparameters.training_method, "sft");
  assert.equal(hyperparameters.model_mode, "external");
  assert.equal(hyperparameters.nanochat_depth, "1");
  assert.equal(hyperparameters.custom_options, "{\"compile\":false}");
  assert.equal(hyperparameters.n_epochs, "1");
  assert.equal(hyperparameters.model_family, undefined);
  assert.equal(hyperparameters.model_loader, undefined);
  assert.equal(hyperparameters.lora_rank, undefined);
});

test("command DPO hyperparameters pass through for external trainers", () => {
  const request = fineTuneRunRequestSchema.parse({
    run_id: "77777777-7777-4777-8777-777777777777",
    user_id: "user",
    behavior_spec_id: "88888888-8888-4888-8888-888888888888",
    run_number: 1,
    training_method: "dpo",
    spec_snapshot: {
      name: "External DPO",
      description: "",
      system_prompt: "",
      base_model: "external:karpathy/nanochat",
      examples: [{ input: "hello", output: "greeting" }],
    },
    hyperparameters: {
      n_epochs: 1,
      dpo_beta: 0.3,
    },
    dataset_prebuilt: {
      training: "file:///tmp/preferences.jsonl",
      format: "preference_jsonl",
    },
  });

  const hyperparameters = buildTrainingHyperparameters(request, { backend: "command" });
  assert.equal(hyperparameters.base_model, "external:karpathy/nanochat");
  assert.equal(hyperparameters.training_method, "dpo");
  assert.equal(hyperparameters.dpo_beta, "0.3");
  assert.equal(hyperparameters.model_mode, "external");
  assert.equal(hyperparameters.model_loader, undefined);
});

test("uv DPO rejects multimodal models", () => {
  const request = fineTuneRunRequestSchema.parse({
    run_id: "99999999-9999-4999-8999-999999999999",
    user_id: "user",
    behavior_spec_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    run_number: 1,
    training_method: "dpo",
    spec_snapshot: {
      name: "Vision DPO",
      description: "",
      system_prompt: "",
      base_model: "Qwen/Qwen3-VL-2B-Instruct",
      examples: [{ input: "hello", output: "greeting" }],
    },
    dataset_prebuilt: {
      training: "file:///tmp/preferences.jsonl",
      format: "preference_jsonl",
    },
  });

  assert.throws(
    () => buildTrainingHyperparameters(request),
    /DPO training supports text causal-LM models only/,
  );
});

test("uv training still rejects external models", () => {
  const request = fineTuneRunRequestSchema.parse({
    run_id: "55555555-5555-4555-8555-555555555555",
    user_id: "user",
    behavior_spec_id: "66666666-6666-4666-8666-666666666666",
    run_number: 1,
    spec_snapshot: {
      name: "External",
      description: "",
      system_prompt: "",
      base_model: "external:karpathy/nanochat",
      examples: [{ input: "hello", output: "greeting" }],
    },
  });

  assert.throws(
    () => buildTrainingHyperparameters(request),
    /Unsupported base model/,
  );
});
