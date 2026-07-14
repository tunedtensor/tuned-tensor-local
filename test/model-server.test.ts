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

test("local model server loads the adapter with its recorded base model and shared cache", () => {
  const previousOpenRouter = process.env.OPENROUTER_API_KEY;
  const previousHfToken = process.env.HF_TOKEN;
  process.env.OPENROUTER_API_KEY = "must-not-leak";
  process.env.HF_TOKEN = "public-model-must-not-receive-this";
  try {
    const config = localRunnerConfigSchema.parse({
      paths: { modelCache: "/tmp/huggingface" },
      evaluation: {
        inference: {
          provider: "transformers",
          project: "training/local-runner",
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
        baseModelRevision: "revision-123",
      },
    });

    assert.equal(launch.command, "uv");
    assert.ok(launch.commandArgs.includes("--project"));
    assert.ok(launch.commandArgs.some((value) => value.endsWith("training/local-runner/src/serve.py")));
    assert.equal(launch.env.TT_MODEL_ARTIFACT, "/tmp/model.tar.gz");
    assert.equal(launch.env.TT_BASE_MODEL, "Qwen/Qwen3.5-2B");
    assert.equal(launch.env.TT_BASE_MODEL_REVISION, "revision-123");
    assert.equal(launch.env.TT_MODEL_LOADER, "causal_lm");
    assert.equal(launch.env.HF_HOME, "/tmp/huggingface");
    assert.equal(launch.env.TT_SYSTEM_PROMPT, "Be concise.");
    assert.equal(launch.env.OPENROUTER_API_KEY, undefined);
    assert.equal(launch.env.HF_TOKEN, undefined);
    assert.equal(launch.env.TT_MAX_CONCURRENT_REQUESTS, "1");
    assert.equal(launch.url, "http://127.0.0.1:8123");
  } finally {
    if (previousOpenRouter === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = previousOpenRouter;
    if (previousHfToken === undefined) delete process.env.HF_TOKEN;
    else process.env.HF_TOKEN = previousHfToken;
  }
});

test("only gated models inherit the Hugging Face token", () => {
  const config = localRunnerConfigSchema.parse({ evaluation: { scoring: { mode: "exact_match" } } });
  const previous = process.env.HF_TOKEN;
  process.env.HF_TOKEN = "gated-model-token";
  try {
    const launch = buildLocalModelServerLaunch({
      model: { ...model, base_model: "meta-llama/Llama-3.2-3B-Instruct" },
      config,
    });
    assert.equal(launch.env.HF_TOKEN, "gated-model-token");
  } finally {
    if (previous === undefined) delete process.env.HF_TOKEN;
    else process.env.HF_TOKEN = previous;
  }
});

test("local model server rejects non-local artifacts and invalid ports", () => {
  const config = localRunnerConfigSchema.parse({
    evaluation: { scoring: { mode: "exact_match" } },
  });
  assert.throws(
    () => buildLocalModelServerLaunch({ model: { ...model, artifact_uri: "s3://bucket/model" }, config }),
    /local file artifact/,
  );
  assert.throws(
    () => buildLocalModelServerLaunch({ model, config, options: { port: 70_000 } }),
    /port must be between/,
  );
  assert.throws(
    () => buildLocalModelServerLaunch({ model, config, options: { port: 8000.5 } }),
    /port must be between/,
  );
  assert.throws(
    () => buildLocalModelServerLaunch({ model, config, options: { host: "0.0.0.0" } }),
    /--allow-remote/,
  );
});

test("remote model serving requires an explicit bind opt-in and bearer token", () => {
  const config = localRunnerConfigSchema.parse({ evaluation: { scoring: { mode: "exact_match" } } });
  const previous = process.env.TT_TEST_SERVE_KEY;
  process.env.TT_TEST_SERVE_KEY = "local-test-token";
  try {
    const launch = buildLocalModelServerLaunch({
      model,
      config,
      options: {
        host: "0.0.0.0",
        allowRemote: true,
        apiKeyEnv: "TT_TEST_SERVE_KEY",
      },
    });
    assert.equal(launch.env.TT_API_KEY, "local-test-token");
    assert.equal(launch.env.TT_TEST_SERVE_KEY, undefined);
  } finally {
    if (previous === undefined) delete process.env.TT_TEST_SERVE_KEY;
    else process.env.TT_TEST_SERVE_KEY = previous;
  }
});
