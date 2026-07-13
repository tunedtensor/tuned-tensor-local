import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  fineTuneRunRequestSchema,
  localBehaviorSpecFileSchema,
} from "../src/contracts.js";
import {
  initLocalSpecFile,
  initLocalRunnerConfigFile,
  assertLocalRunInputReady,
  loadLocalRunInput,
  runRequestFromLocalSpec,
  resolveLocalRunInputPaths,
  unknownHyperparameterWarnings,
} from "../src/local-project.js";
import { parseDotEnv } from "../src/index.js";

test("initializes a standalone local spec file and converts it to a run request", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-project-test-"));
  try {
    const path = join(root, "tunedtensor.json");
    const spec = await initLocalSpecFile({
      outputPath: path,
      name: "Standalone Local",
      baseModel: "qwen/qwen3.5-2b",
    });
    assert.equal(spec.name, "Standalone Local");
    assert.equal(spec.base_model, "Qwen/Qwen3.5-2B");
    assert.ok(spec.id);
    assert.equal(spec.hyperparameters?.n_epochs, 1);

    const input = await loadLocalRunInput(path, {
      userId: "local-test-user",
      runNumber: 7,
    });
    assert.equal(input.kind, "spec");
    assert.equal(input.request.user_id, "local-test-user");
    assert.equal(input.request.run_number, 7);
    assert.equal(input.request.behavior_spec_id, spec.id);
    assert.equal(input.request.spec_snapshot.name, "Standalone Local");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("spark profile writes a safe durable local runner config", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-profile-test-"));
  try {
    const path = join(root, "local-runner.json");
    await initLocalRunnerConfigFile({ outputPath: path, profile: "spark" });
    const config = JSON.parse(await readFile(path, "utf8")) as Record<string, any>;
    assert.equal(config.storeRoot, ".tt-local/store");
    assert.equal(config.training.project, "training/local-runner");
    assert.equal(config.evaluation.inference.project, "training/local-runner");
    assert.equal(config.evaluation.inference.device, "cuda");
    assert.equal(config.evaluation.scoring.mode, "exact_match");
    assert.equal(config.evaluation.timeoutMs, 1_800_000);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("unchanged init placeholders are rejected before training", () => {
  const request = runRequestFromLocalSpec(localBehaviorSpecFileSchema.parse({
    name: "Placeholder",
    base_model: "Qwen/Qwen3.5-2B",
    system_prompt: "Describe the behavior this local model should learn.",
    examples: [{
      input: "Replace this with a representative input.",
      output: "Replace this with the expected output.",
    }],
  }));
  assert.throws(() => assertLocalRunInputReady(request), /Edit the generated behavior spec/);
});

test("runRequestFromLocalSpec preserves local spec metadata", () => {
  const request = runRequestFromLocalSpec({
    id: "77777777-7777-4777-8777-777777777777",
    user_id: "local-user",
    run_number: 3,
    name: "Preserved",
    description: "",
    system_prompt: "Return labels.",
    guidelines: [],
    constraints: [],
    base_model: "Qwen/Qwen3.5-2B",
    examples: [{ input: "good", output: "positive" }],
    hyperparameters: {
      n_epochs: 2,
      save_adapter_only: true,
      augment: false,
      use_llm_judge: false,
    },
  }, {
    runId: "88888888-8888-4888-8888-888888888888",
  });

  assert.equal(request.run_id, "88888888-8888-4888-8888-888888888888");
  assert.equal(request.behavior_spec_id, "77777777-7777-4777-8777-777777777777");
  assert.equal(request.run_number, 3);
  assert.equal(request.hyperparameters.n_epochs, 2);
});

test("runRequestFromLocalSpec preserves DPO training method", () => {
  const spec = localBehaviorSpecFileSchema.parse({
    id: "11111111-1111-4111-8111-111111111111",
    name: "DPO Spec",
    base_model: "Qwen/Qwen3.5-2B",
    training_method: "dpo",
    examples: [{ input: "eval", output: "answer" }],
    dataset_prebuilt: {
      training: "file:///tmp/preferences.jsonl",
      format: "preference_jsonl",
    },
  });

  const request = runRequestFromLocalSpec(spec);
  assert.equal(request.training_method, "dpo");
  assert.equal(request.dataset_prebuilt?.format, "preference_jsonl");
});

test("DPO request schema enforces preference training data and evaluation references", () => {
  assert.doesNotThrow(() => fineTuneRunRequestSchema.parse({
    run_id: "22222222-2222-4222-8222-222222222222",
    user_id: "user",
    behavior_spec_id: "33333333-3333-4333-8333-333333333333",
    run_number: 1,
    training_method: "dpo",
    spec_snapshot: {
      name: "DPO",
      description: "",
      system_prompt: "",
      base_model: "Qwen/Qwen3.5-2B",
      examples: [{ input: "eval", output: "answer" }],
    },
    dataset_prebuilt: {
      training: "file:///tmp/preferences.jsonl",
      format: "preference_jsonl",
    },
  }));

  assert.throws(
    () => fineTuneRunRequestSchema.parse({
      run_id: "44444444-4444-4444-8444-444444444444",
      user_id: "user",
      behavior_spec_id: "55555555-5555-4555-8555-555555555555",
      run_number: 1,
      training_method: "dpo",
      spec_snapshot: {
        name: "DPO",
        description: "",
        system_prompt: "",
        base_model: "Qwen/Qwen3.5-2B",
        examples: [],
      },
      dataset_prebuilt: {
        training: "file:///tmp/preferences.jsonl",
        format: "preference_jsonl",
      },
    }),
    /DPO evaluation requires/,
  );

  assert.throws(
    () => fineTuneRunRequestSchema.parse({
      run_id: "66666666-6666-4666-8666-666666666666",
      user_id: "user",
      behavior_spec_id: "77777777-7777-4777-8777-777777777777",
      run_number: 1,
      spec_snapshot: {
        name: "SFT",
        description: "",
        system_prompt: "",
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "eval", output: "answer" }],
      },
      dataset_prebuilt: {
        training: "file:///tmp/preferences.jsonl",
        format: "preference_jsonl",
      },
    }),
    /preference_jsonl datasets require training_method dpo/,
  );
});

test("unknownHyperparameterWarnings flags pass-through keys that the default trainer may ignore", () => {
  const warnings = unknownHyperparameterWarnings({
    hyperparameters: {
      n_epochs: 1,
      per_device_train_batch_size: 2,
    },
  });
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /per_device_train_batch_size/);
  assert.match(warnings[0], /passed through/);
  assert.match(warnings[0], /batch_size/);

  assert.deepEqual(unknownHyperparameterWarnings({ hyperparameters: { n_epochs: 1 } }), []);
  assert.deepEqual(unknownHyperparameterWarnings({}), []);
  assert.deepEqual(unknownHyperparameterWarnings(null), []);
});

test("loadLocalRunInput surfaces unknown hyperparameter warnings", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-warnings-"));
  try {
    const path = join(root, "tunedtensor.json");
    await writeFile(path, JSON.stringify({
      name: "Warning Spec",
      base_model: "Qwen/Qwen3.5-2B",
      examples: [{ input: "in", output: "out" }],
      hyperparameters: {
        n_epochs: 1,
        per_device_train_batch_size: 2,
      },
    }), "utf8");
    const input = await loadLocalRunInput(path);
    assert.equal(input.kind, "spec");
    assert.equal(input.warnings.length, 1);
    assert.match(input.warnings[0], /per_device_train_batch_size/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("local dataset and image paths resolve from the input file directory", () => {
  const inputPath = join(tmpdir(), "project", "tunedtensor.json");
  const resolved = resolveLocalRunInputPaths({
    dataset_prebuilt: {
      training: "data/train.jsonl",
      validation: "file:///shared/validation.jsonl",
    },
    examples: [{
      input: "read",
      output: "ok",
      input_assets: [{ type: "image", path: "images/page.png" }],
    }],
  }, inputPath) as any;
  assert.equal(resolved.dataset_prebuilt.training, join(tmpdir(), "project", "data", "train.jsonl"));
  assert.equal(resolved.dataset_prebuilt.validation, "file:///shared/validation.jsonl");
  assert.equal(resolved.examples[0].input_assets[0].path, join(tmpdir(), "project", "images", "page.png"));
});

test("parseDotEnv parses simple KEY=VALUE lines", () => {
  const values = parseDotEnv([
    "# comment",
    "",
    "OPENROUTER_API_KEY=sk-test-123",
    "export QUOTED=\"hello world\"",
    "SINGLE='one'",
    "not a valid line",
    "TRAILING = spaced value ",
  ].join("\n"));
  assert.deepEqual(values, {
    OPENROUTER_API_KEY: "sk-test-123",
    QUOTED: "hello world",
    SINGLE: "one",
    TRAILING: "spaced value",
  });
});
