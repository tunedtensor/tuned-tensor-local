import assert from "node:assert/strict";
import { test } from "node:test";
import { localRunnerConfigSchema } from "../src/contracts.js";
import { buildLocalModelServerLaunch } from "../src/model-server.js";
import type { LocalModelRecord } from "../src/store.js";

const model: LocalModelRecord = {
  id: "local-11111111-1111-4111-8111-111111111111",
  run_id: "11111111-1111-4111-8111-111111111111",
  behavior_spec_id: "22222222-2222-4222-8222-222222222222",
  name: "Qwen adapter",
  provider: "local-uv",
  base_model: "Qwen/Qwen3.5-2B",
  artifact_uri: "file:///tmp/model.tar.gz",
  artifact_dir: "/tmp/run",
  metrics: null,
  created_at: "2026-07-13T00:00:00.000Z",
};

test("serving launches the bundled text-only Qwen adapter with safe model settings", () => {
  const config = localRunnerConfigSchema.parse({
    paths: {
      baseModel: "/tmp/qwen-snapshot",
      modelCache: "/tmp/huggingface",
    },
    evaluation: {
      inference: {
        device: "cuda",
      },
      scoring: { mode: "exact_match" },
    },
  });
  const launch = buildLocalModelServerLaunch({
    model,
    config,
    options: {
      port: 8123,
      systemPrompt: "Be concise.",
      maxTokens: 64,
      maxConcurrentRequests: 2,
      baseModelArtifactUri: "file:///tmp/qwen-snapshot",
      baseModelRevision: "ignored-for-local-snapshot",
    },
  });

  assert.equal(launch.command, "uv");
  assert.ok(launch.commandArgs.includes("--project"));
  assert.ok(launch.commandArgs.some((value) =>
    value.endsWith("training/local-runner/src/serve.py")
  ));
  assert.equal(launch.env.TT_MODEL_ARTIFACT, "/tmp/model.tar.gz");
  assert.equal(launch.env.TT_BASE_MODEL, "/tmp/qwen-snapshot");
  assert.equal(launch.env.TT_BASE_MODEL_REVISION, undefined);
  assert.equal(launch.env.TT_MODEL_LOADER, "causal_lm");
  assert.equal(launch.env.TT_TRUST_REMOTE_CODE, "false");
  assert.equal(launch.env.TT_CHAT_TEMPLATE_KWARGS, undefined);
  assert.equal(launch.env.HF_HOME, "/tmp/huggingface");
  assert.equal(launch.env.HF_HUB_CACHE, "/tmp/huggingface/hub");
  assert.equal(launch.env.HF_HUB_OFFLINE, "1");
  assert.equal(launch.env.TRANSFORMERS_OFFLINE, "1");
  assert.ok(launch.env.UV_PROJECT_ENVIRONMENT);
  assert.equal(launch.env.TT_SYSTEM_PROMPT, "Be concise.");
  assert.equal(launch.env.TT_MAX_CONCURRENT_REQUESTS, "2");
  assert.equal(launch.url, "http://127.0.0.1:8123");
});

test("serving rejects unsafe artifacts, base-model mismatches, and invalid network bounds", () => {
  const config = localRunnerConfigSchema.parse({
    paths: { baseModel: "/tmp/qwen-snapshot" },
  });
  assert.throws(
    () => buildLocalModelServerLaunch({
      model: { ...model, artifact_uri: "s3://bucket/model" },
      config,
    }),
    /local file artifact/,
  );
  assert.throws(
    () => buildLocalModelServerLaunch({
      model,
      config,
      options: { baseModelArtifactUri: "file:///tmp/different-snapshot" },
    }),
    /does not match the base model recorded/,
  );
  assert.throws(
    () => buildLocalModelServerLaunch({ model, config, options: { port: 70_000 } }),
    /port must be between/,
  );
  assert.throws(
    () => buildLocalModelServerLaunch({ model, config, options: { maxConcurrentRequests: 9 } }),
    /maxConcurrentRequests must be between/,
  );
  assert.throws(
    () => buildLocalModelServerLaunch({ model, config, options: { host: "0.0.0.0" } }),
    /--allow-remote/,
  );
});

test("an explicitly remote bind requires and forwards only the selected bearer token", () => {
  const previous = process.env.TT_TEST_SERVE_KEY;
  process.env.TT_TEST_SERVE_KEY = "local-test-token";
  try {
    const launch = buildLocalModelServerLaunch({
      model,
      config: localRunnerConfigSchema.parse({}),
      options: {
        host: "0.0.0.0",
        allowRemote: true,
        apiKeyEnv: "TT_TEST_SERVE_KEY",
      },
    });
    assert.equal(launch.env.TT_API_KEY, "local-test-token");
    assert.equal(launch.env.TT_TEST_SERVE_KEY, undefined);
    assert.equal(launch.url, "http://0.0.0.0:8000");
  } finally {
    if (previous === undefined) delete process.env.TT_TEST_SERVE_KEY;
    else process.env.TT_TEST_SERVE_KEY = previous;
  }
});
