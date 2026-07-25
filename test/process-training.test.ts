import assert from "node:assert/strict";
import { test } from "node:test";
import { fineTuneRunRequestSchema } from "../src/contracts.js";
import {
  buildTrainingHyperparameters,
  createTrainingProgressForwarder,
  parseTrainingProgressLine,
} from "../src/process-training.js";

test("builds only the certified Qwen LoRA SFT hyperparameters", () => {
  const request = fineTuneRunRequestSchema.parse({
    run_id: "11111111-1111-4111-8111-111111111111",
    user_id: "local-user",
    behavior_spec_id: "22222222-2222-4222-8222-222222222222",
    run_number: 1,
    spec_snapshot: {
      name: "Classifier",
      base_model: "qwen/qwen3.5-2b",
      system_prompt: "Return one label.",
      examples: [{ input: "hello", output: "greeting" }],
    },
    hyperparameters: {
      n_epochs: 3,
      learning_rate: 0.00002,
      batch_size: 2,
      gradient_accumulation_steps: 4,
      lora_rank: 8,
      lora_alpha: 16,
      lora_dropout: 0.1,
      max_seq_length: 1024,
    },
  });

  assert.deepEqual(buildTrainingHyperparameters(request, {
    baseModelRevision: "0123456789abcdef",
  }), {
    base_model: "Qwen/Qwen3.5-2B",
    base_model_revision: "0123456789abcdef",
    n_epochs: "3",
    learning_rate: "0.00002",
    per_device_train_batch_size: "2",
    gradient_accumulation_steps: "4",
    lora_rank: "8",
    lora_alpha: "16",
    lora_dropout: "0.1",
    max_seq_length: "1024",
  });
});

test("default Qwen LoRA training parameters stay small", () => {
  const request = fineTuneRunRequestSchema.parse({
    run_id: "33333333-3333-4333-8333-333333333333",
    user_id: "local-user",
    behavior_spec_id: "44444444-4444-4444-8444-444444444444",
    run_number: 1,
    spec_snapshot: {
      name: "Assistant-safe SFT",
      base_model: "Qwen/Qwen3.5-2B",
      examples: [{ input: "hello", output: "hi" }],
    },
  });

  const hyperparameters = buildTrainingHyperparameters(request);
  assert.deepEqual(hyperparameters, {
    base_model: "Qwen/Qwen3.5-2B",
    n_epochs: "1",
    learning_rate: "0.00001",
    per_device_train_batch_size: "1",
    gradient_accumulation_steps: "8",
    lora_rank: "16",
    lora_alpha: "32",
    lora_dropout: "0.05",
    max_seq_length: "2048",
  });
});

test("parses trainer metric dictionaries and tqdm progress", () => {
  assert.deepEqual(
    parseTrainingProgressLine(
      "{'loss': '0.351', 'grad_norm': '0.6381', 'learning_rate': '4.098e-08', 'epoch': '1'}",
    ),
    {
      loss: 0.351,
      grad_norm: 0.6381,
      learning_rate: 4.098e-8,
      epoch: 1,
      percent: 100,
    },
  );
  assert.deepEqual(
    parseTrainingProgressLine(" 98%|█████████▊| 238/244 [29:25<00:48,  8.02s/it]"),
    {
      percent: 98,
      step: 238,
      total_steps: 244,
      elapsed: "29:25",
      eta: "00:48",
      rate: "8.02s/it",
    },
  );
});

test("does not mistake checkpoint loading for optimizer progress", () => {
  assert.equal(
    parseTrainingProgressLine(
      "Loading checkpoint shards: 100%|████| 320/320 [00:02<00:00, 150it/s]",
    ),
    null,
  );
});

test("progress reporter rejections remain detached from training teardown", async () => {
  const forward = createTrainingProgressForwarder({
    async onEvent() {
      throw new Error("reporter unavailable");
    },
  });
  assert.doesNotThrow(() =>
    forward("100%|██████████| 1/1 [00:01<00:00, 1.00s/it]")
  );
  await new Promise((resolvePromise) => setImmediate(resolvePromise));
});
