import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  assertCertifiedBaseModelConfig,
  assertUsableModelArtifact,
  canonicalizeTrainingModel,
  resolveTrainingModel,
  TRAINING_MODELS,
} from "../src/model-registry.js";

test("registry certifies exactly the native text-only Qwen training path", () => {
  assert.deepEqual(TRAINING_MODELS.map((model) => model.id), ["Qwen/Qwen3.5-2B"]);
  assert.equal(canonicalizeTrainingModel(" qwen/QWEN3.5-2b "), "Qwen/Qwen3.5-2B");
  assert.deepEqual(resolveTrainingModel("Qwen/Qwen3.5-2B"), {
    id: "Qwen/Qwen3.5-2B",
    family: "qwen3_5",
    defaultLearningRate: 0.00001,
    defaultPerDeviceBatchSize: 1,
    defaultGradientAccumulationSteps: 8,
    defaultLoraRank: 16,
    defaultLoraAlpha: 32,
    defaultLoraDropout: 0.05,
    defaultMaxSeqLength: 2048,
  });
});

test("certified config rejects a larger same-family Qwen snapshot", () => {
  const config = {
    architectures: ["Qwen3_5ForConditionalGeneration"],
    model_type: "qwen3_5",
    text_config: {
      model_type: "qwen3_5_text",
      hidden_size: 2048,
      num_hidden_layers: 24,
      num_attention_heads: 8,
      num_key_value_heads: 2,
      intermediate_size: 6144,
      vocab_size: 248320,
    },
  };
  assert.doesNotThrow(() => assertCertifiedBaseModelConfig(config));
  assert.throws(
    () => assertCertifiedBaseModelConfig({
      ...config,
      text_config: { ...config.text_config, hidden_size: 2560 },
    }),
    /certified Qwen\/Qwen3\.5-2B architecture/,
  );
});

test("optimizer state cannot masquerade as a LoRA adapter", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-model-artifact-"));
  try {
    const model = join(root, "model");
    await mkdir(model);
    await writeFile(join(model, "optimizer.pt"), "optimizer state");
    await writeFile(join(model, "adapter_config.json"), "{}");
    await assert.rejects(
      assertUsableModelArtifact(model),
      /contains no non-empty adapter_model weights/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("a LoRA artifact requires both adapter weights and configuration", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-model-artifact-"));
  try {
    const model = join(root, "model");
    await mkdir(model);
    await writeFile(join(model, "adapter_model.safetensors"), "weights");
    await assert.rejects(
      assertUsableModelArtifact(model),
      /adapter weights but no non-empty adapter_config\.json/,
    );

    await writeFile(join(model, "adapter_config.json"), "{}");
    const inspection = await assertUsableModelArtifact(model);
    assert.equal(inspection.adapter_weight_file_count, 1);
    assert.equal(inspection.adapter_weight_bytes, 7);
    assert.equal(inspection.has_adapter_config, true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
