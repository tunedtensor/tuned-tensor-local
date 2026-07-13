import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { assertUsableModelArtifact } from "../src/model-registry.js";

test("optimizer state cannot masquerade as a PEFT adapter directory", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-model-contract-test-"));
  try {
    await mkdir(join(root, "model"));
    await writeFile(join(root, "model", "optimizer.pt"), "optimizer state");
    await writeFile(join(root, "model", "adapter_config.json"), "{}");
    await assert.rejects(
      assertUsableModelArtifact(join(root, "model")),
      /no loadable adapter or Transformers full-model weights/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("implicit PEFT artifacts require both adapter weights and configuration", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-model-contract-test-"));
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
    assert.equal(inspection.has_adapter_config, true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
