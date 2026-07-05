import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { test } from "node:test";
import { fineTuneRunRequestSchema, localRunnerConfigSchema } from "../src/contracts.js";
import { buildModelPrefetchPayload, prefetchBaseModel } from "../src/prefetch.js";

const request = fineTuneRunRequestSchema.parse({
  run_id: "11111111-1111-4111-8111-111111111111",
  user_id: "local-user",
  behavior_spec_id: "22222222-2222-4222-8222-222222222222",
  run_number: 1,
  spec_snapshot: {
    name: "Prefetch",
    description: "",
    system_prompt: "Return labels.",
    base_model: "qwen/qwen3.5-2b",
    examples: [{ input: "hello", output: "greeting" }],
  },
});

test("builds a model prefetch payload from spec and model cache config", () => {
  const config = localRunnerConfigSchema.parse({
    paths: {
      modelCache: ".cache/huggingface",
    },
  });

  assert.deepEqual(buildModelPrefetchPayload(request, config), {
    base_model: "Qwen/Qwen3.5-2B",
    loader: "causal_lm",
    trust_remote_code: true,
    requires_hf_token: false,
    model_cache: resolve(".cache/huggingface"),
  });
});

test("prefetch skips Hugging Face download when a local base model path is configured", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prefetch-test-"));
  try {
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      paths: {
        baseModel: join(root, "base-model"),
        modelCache: join(root, "cache"),
      },
    });
    await mkdir(join(root, "base-model"));

    const report = await prefetchBaseModel({ request, config });
    assert.equal(report.ok, true);
    assert.equal(report.status, "skipped");
    assert.equal(report.base_model, "Qwen/Qwen3.5-2B");
    assert.equal(report.local_base_model_path, join(root, "base-model"));
    assert.match(report.reason ?? "", /local base-model artifact/);
    assert.ok(report.artifact_dir.startsWith(join(root, "artifacts", "prefetch")));
    assert.equal(report.command, undefined);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
