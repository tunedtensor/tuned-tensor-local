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
  assert.equal(buildTrainingHyperparameters(textRequest).model_loader, "causal_lm");

  const multimodalRequest = fineTuneRunRequestSchema.parse({
    ...base,
    spec_snapshot: {
      ...base.spec_snapshot,
      base_model: "Qwen/Qwen3-VL-2B-Instruct",
    },
  });
  assert.equal(buildTrainingHyperparameters(multimodalRequest).model_loader, "image_text_to_text");
});
