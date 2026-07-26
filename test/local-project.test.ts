import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  fineTuneRunRequestSchema,
  localBehaviorSpecFileSchema,
  localRunnerConfigSchema,
} from "../src/contracts.js";
import {
  assertLocalRunInputReady,
  initLocalRunnerConfigFile,
  initLocalSpecFile,
  loadLocalRunInput,
  resolveLocalRunInputPaths,
  runRequestFromLocalSpec,
} from "../src/local-project.js";

test("init creates the certified Spark project with two placeholders", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-project-"));
  try {
    const specPath = join(root, "tunedtensor.json");
    const configPath = join(root, "local-runner.json");
    const spec = await initLocalSpecFile({
      outputPath: specPath,
      name: "Support Adapter",
      baseModel: "qwen/qwen3.5-2b",
    });
    await initLocalRunnerConfigFile({
      outputPath: configPath,
      profile: "spark",
    });

    assert.equal(spec.base_model, "Qwen/Qwen3.5-2B");
    assert.equal(spec.examples.length, 2);
    assert.throws(
      () => assertLocalRunInputReady(runRequestFromLocalSpec(spec)),
      /Edit the generated behavior spec/,
    );

    const config = localRunnerConfigSchema.parse(
      JSON.parse(await readFile(configPath, "utf8")),
    );
    assert.equal("training" in config, false);
    assert.equal("provider" in config.evaluation.inference, false);
    assert.equal(config.evaluation.inference.device, "cuda");
    assert.equal(config.evaluation.scoring.mode, "exact_match");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("local specs convert to strict SFT requests while preserving identity", () => {
  const request = runRequestFromLocalSpec(localBehaviorSpecFileSchema.parse({
    id: "77777777-7777-4777-8777-777777777777",
    name: "Preserved",
    system_prompt: "Return one label.",
    base_model: "Qwen/Qwen3.5-2B",
    examples: [
      { input: "good", output: "positive" },
      { input: "bad", output: "negative" },
    ],
    hyperparameters: { n_epochs: 2, batch_size: 1 },
  }), {
    runId: "88888888-8888-4888-8888-888888888888",
  });

  assert.equal(request.run_id, "88888888-8888-4888-8888-888888888888");
  assert.equal(request.behavior_spec_id, "77777777-7777-4777-8777-777777777777");
  assert.equal(request.user_id, "local-user");
  assert.equal(request.run_number, 1);
  assert.equal(request.hyperparameters.n_epochs, 2);
});

test("id-less local specs derive a stable behavior identity", () => {
  const spec = localBehaviorSpecFileSchema.parse({
    name: "Stable",
    system_prompt: "Return one label.",
    base_model: "Qwen/Qwen3.5-2B",
    examples: [
      { input: "good", output: "positive" },
      { input: "bad", output: "negative" },
    ],
  });
  const first = runRequestFromLocalSpec(spec);
  const second = runRequestFromLocalSpec(spec);
  assert.equal(first.behavior_spec_id, second.behavior_spec_id);
  assert.notEqual(first.run_id, second.run_id);
});

test("obsolete and misspelled settings fail instead of reverting to defaults", () => {
  assert.throws(
    () => localRunnerConfigSchema.parse({
      dry_run: true,
      evaluation: { inference: { device: "cdua" } },
    }),
    /Unrecognized key|Invalid option/,
  );
  assert.throws(
    () => localBehaviorSpecFileSchema.parse({
      name: "Typo",
      base_model: "Qwen/Qwen3.5-2B",
      examples: [
        { input: "a", output: "b" },
        { input: "c", output: "d" },
      ],
      hyperparameters: { n_epochs: 1, save_adapter_only: true },
    }),
    /Unrecognized key/,
  );
  assert.throws(
    () => fineTuneRunRequestSchema.parse({
      run_id: "11111111-1111-4111-8111-111111111111",
      user_id: "local-user",
      behavior_spec_id: "22222222-2222-4222-8222-222222222222",
      run_number: 1,
      training_method: "dpo",
      spec_snapshot: {
        name: "Unsupported",
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "a", output: "b" }],
      },
    }),
    /Unrecognized key/,
  );
  assert.throws(
    () => localBehaviorSpecFileSchema.parse({
      name: "Mutable revision",
      base_model: "Qwen/Qwen3.5-2B",
      examples: [
        { input: "a", output: "b" },
        { input: "c", output: "d" },
      ],
      hyperparameters: { base_model_revision: "main" },
    }),
    /40-character Hugging Face commit SHA/,
  );
});

test("loadLocalRunInput reads a complete request or derives one from a spec", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-input-"));
  try {
    const path = join(root, "tunedtensor.json");
    await writeFile(path, JSON.stringify({
      name: "Loaded",
      base_model: "Qwen/Qwen3.5-2B",
      system_prompt: "Return one label.",
      examples: [
        { input: "up", output: "positive" },
        { input: "down", output: "negative" },
      ],
    }), "utf8");
    const input = await loadLocalRunInput(path);
    assert.equal(input.kind, "spec");
    assert.equal(input.request.user_id, "local-user");
    assert.equal(input.request.run_number, 1);
    assert.equal(input.request.spec_snapshot.name, "Loaded");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("dataset paths resolve relative to the spec and file URIs stay absolute", () => {
  const inputPath = join(tmpdir(), "project", "tunedtensor.json");
  const resolved = resolveLocalRunInputPaths({
    dataset_prebuilt: {
      training: "data/train.jsonl",
      validation: "file:///shared/validation.jsonl",
    },
  }, inputPath) as {
    dataset_prebuilt: { training: string; validation: string };
  };
  assert.equal(
    resolved.dataset_prebuilt.training,
    join(tmpdir(), "project", "data", "train.jsonl"),
  );
  assert.equal(
    resolved.dataset_prebuilt.validation,
    "file:///shared/validation.jsonl",
  );
});
