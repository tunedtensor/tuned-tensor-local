import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { homedir, tmpdir } from "node:os";
import { test } from "node:test";
import { fineTuneRunRequestSchema, localRunnerConfigSchema } from "../src/contracts.js";
import { loadLocalRunnerConfig, runLocalFineTune, runLocalFineTuneStage } from "../src/orchestrator.js";
import { createLocalStore } from "../src/store.js";

function chatRow(input: string, output: string): string {
  return JSON.stringify({
    messages: [
      { role: "system", content: "Return labels." },
      { role: "user", content: input },
      { role: "assistant", content: output },
    ],
  });
}

function preferenceRow(prompt: string, chosen: string, rejected: string): string {
  return JSON.stringify({ prompt, chosen, rejected });
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

test("runs a dry local workflow and writes compatible artifacts", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-test-"));
  try {
    const request = fineTuneRunRequestSchema.parse({
      run_id: "11111111-1111-4111-8111-111111111111",
      user_id: "local-user",
      behavior_spec_id: "22222222-2222-4222-8222-222222222222",
      run_number: 1,
      spec_snapshot: {
        name: "Smoke",
        description: "",
        system_prompt: "Return labels.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [
          { input: "Classify: good", output: "positive" },
          { input: "Classify: bad", output: "negative" },
        ],
      },
      hyperparameters: {
        n_epochs: 1,
        augment: false,
        use_llm_judge: false,
        save_adapter_only: true,
      },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
    });
    const events: Array<{ stage: string; message: string }> = [];

    const result = await runLocalFineTune({
      request,
      config,
      reporter: {
        onEvent(event) {
          events.push({ stage: event.stage, message: event.message });
        },
      },
    });
    assert.equal(result.report.status, "completed");
    assert.equal(result.report.training.provider, "local-uv");
    // With 2 spec examples the runner holds one out for evaluation.
    assert.equal(result.report.baseline.total, 1);
    assert.equal(result.report.candidate.total, 1);
    assert.equal(result.report.baseline.eval_split, "spec_holdout");
    assert.equal(result.report.run_metadata.eval_split, "spec_holdout");
    assert.equal(result.report.run_metadata.training_example_count, 1);
    assert.equal(result.report.run_metadata.eval_examples_total, 1);

    const reportText = await readFile(result.reportPath, "utf8");
    assert.match(reportText, /local-uv/);
    const datasetText = await readFile(result.report.artifact_uris.dataset.replace(/^file:\/\//, ""), "utf8");
    // Training JSONL holds exactly one example; the other is the eval holdout.
    assert.match(datasetText, /Classify: (good|bad)/);
    const evaluatedPrompt = result.report.baseline.results[0]?.prompt ?? "";
    assert.ok(evaluatedPrompt);
    assert.ok(!datasetText.includes(evaluatedPrompt), "holdout example must not be in training JSONL");
    assert.deepEqual(events.map((event) => event.stage), [
      "queued",
      "preparing",
      "evaluating_baseline",
      "training",
      "evaluating_candidate",
      "completed",
    ]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("spec run with a single example evaluates the training set as spec_examples", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-spec-single-"));
  try {
    const request = fineTuneRunRequestSchema.parse({
      run_id: "99999999-9999-4999-8999-999999999999",
      user_id: "local-user",
      behavior_spec_id: "22222222-2222-4222-8222-222222222222",
      run_number: 1,
      spec_snapshot: {
        name: "Single",
        description: "",
        system_prompt: "Return labels.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "Classify: good", output: "positive" }],
      },
      hyperparameters: { n_epochs: 1 },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
    });

    const result = await runLocalFineTune({ request, config });
    assert.equal(result.report.baseline.eval_split, "spec_examples");
    assert.equal(result.report.run_metadata.eval_split, "spec_examples");
    assert.equal(result.report.run_metadata.training_example_count, 1);
    assert.equal(result.report.run_metadata.eval_examples_total, 1);
    assert.equal(result.report.baseline.results[0]?.prompt, "Classify: good");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("spec runs hold out ~20% of examples for evaluation deterministically", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-spec-holdout-"));
  try {
    const makeRequest = () => fineTuneRunRequestSchema.parse({
      run_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
      user_id: "local-user",
      behavior_spec_id: "22222222-2222-4222-8222-222222222222",
      run_number: 1,
      spec_snapshot: {
        name: "Holdout",
        description: "",
        system_prompt: "Return labels.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: Array.from({ length: 10 }, (_, index) => ({
          input: `Classify: item ${index}`,
          output: `label ${index}`,
        })),
      },
      hyperparameters: { n_epochs: 1 },
    });
    const makeConfig = (suffix: string) => localRunnerConfigSchema.parse({
      artifactRoot: join(root, `artifacts-${suffix}`),
      storeRoot: join(root, `store-${suffix}`),
      dryRun: true,
    });

    const first = await runLocalFineTune({ request: makeRequest(), config: makeConfig("a") });
    assert.equal(first.report.baseline.eval_split, "spec_holdout");
    assert.equal(first.report.run_metadata.eval_split, "spec_holdout");
    assert.equal(first.report.run_metadata.training_example_count, 8);
    assert.equal(first.report.run_metadata.eval_examples_total, 2);
    assert.equal(first.report.baseline.eval_examples_total, 2);
    assert.equal(first.report.baseline.total, 2);

    const datasetText = await readFile(first.report.artifact_uris.dataset.replace(/^file:\/\//, ""), "utf8");
    assert.equal(datasetText.trim().split("\n").length, 8);
    for (const entry of first.report.baseline.results) {
      assert.ok(!datasetText.includes(entry.prompt), `holdout prompt leaked into training JSONL: ${entry.prompt}`);
    }
    // Baseline and candidate evaluate the identical holdout.
    assert.deepEqual(
      first.report.baseline.results.map((entry) => entry.prompt),
      first.report.candidate.results.map((entry) => entry.prompt),
    );

    // The same run id yields the same split on a re-run.
    const second = await runLocalFineTune({ request: makeRequest(), config: makeConfig("b") });
    assert.deepEqual(
      second.report.baseline.results.map((entry) => entry.prompt),
      first.report.baseline.results.map((entry) => entry.prompt),
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

function prebuiltRequest(options: {
  runId: string;
  trainingPath: string;
  validationPath?: string;
  testPath?: string;
  maxEvalExamples?: number;
}) {
  return fineTuneRunRequestSchema.parse({
    run_id: options.runId,
    user_id: "local-user",
    behavior_spec_id: "44444444-4444-4444-8444-444444444444",
    run_number: 1,
    spec_snapshot: {
      name: "Prebuilt",
      description: "",
      system_prompt: "Return labels.",
      guidelines: [],
      constraints: [],
      base_model: "Qwen/Qwen3.5-2B",
      examples: [{ input: "spec input", output: "spec output" }],
    },
    ...(options.maxEvalExamples
      ? { hyperparameters: { n_epochs: 1, max_eval_examples: options.maxEvalExamples } }
      : {}),
    dataset_prebuilt: {
      training: `file://${options.trainingPath}`,
      ...(options.validationPath ? { validation: `file://${options.validationPath}` } : {}),
      ...(options.testPath ? { test: `file://${options.testPath}` } : {}),
      format: "chat_jsonl",
    },
  });
}

test("fails a real run when prebuilt dataset has no test or validation split", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prebuilt-training-guard-"));
  try {
    const trainingPath = join(root, "train.chat.jsonl");
    await writeFile(trainingPath, `${chatRow("train input", "train output")}\n`, "utf8");
    const request = prebuiltRequest({
      runId: "55555555-5555-4555-8555-555555555555",
      trainingPath,
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: false,
    });

    await assert.rejects(
      runLocalFineTune({ request, config }),
      /no test or validation split/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("dry run evaluates prebuilt training split and records provenance", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prebuilt-training-dry-"));
  try {
    const trainingPath = join(root, "train.chat.jsonl");
    await writeFile(trainingPath, `${chatRow("train input", "train output")}\n`, "utf8");
    const request = prebuiltRequest({
      runId: "66666666-6666-4666-8666-666666666666",
      trainingPath,
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
    });

    const result = await runLocalFineTune({ request, config });
    assert.equal(result.report.baseline.eval_split, "prebuilt_training");
    assert.equal(result.report.run_metadata.eval_split, "prebuilt_training");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("uses request max_eval_examples when config maxExamples is unset", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-max-eval-hyper-"));
  try {
    const testPath = join(root, "test.chat.jsonl");
    const trainingPath = join(root, "train.chat.jsonl");
    await writeFile(trainingPath, `${chatRow("train input", "train output")}\n`, "utf8");
    await writeFile(testPath, Array.from({ length: 4 }, (_, index) => chatRow(`test input ${index}`, `test output ${index}`)).join("\n"), "utf8");
    const request = prebuiltRequest({
      runId: "77777777-7777-4777-8777-777777777777",
      trainingPath,
      testPath,
      maxEvalExamples: 2,
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
    });

    const result = await runLocalFineTune({ request, config });
    assert.equal(result.report.baseline.eval_examples_total, 4);
    assert.equal(result.report.baseline.eval_examples_used, 2);
    assert.equal(result.report.baseline.eval_truncated, true);
    assert.equal(typeof result.report.baseline.eval_sample_seed, "number");
    assert.equal(result.report.run_metadata.eval_sample_seed, result.report.baseline.eval_sample_seed);
    assert.deepEqual(
      result.report.baseline.results.map((entry) => entry.prompt),
      result.report.candidate.results.map((entry) => entry.prompt),
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("config maxExamples overrides request max_eval_examples", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-max-eval-config-"));
  try {
    const testPath = join(root, "test.chat.jsonl");
    const trainingPath = join(root, "train.chat.jsonl");
    await writeFile(trainingPath, `${chatRow("train input", "train output")}\n`, "utf8");
    await writeFile(testPath, Array.from({ length: 4 }, (_, index) => chatRow(`test input ${index}`, `test output ${index}`)).join("\n"), "utf8");
    const request = prebuiltRequest({
      runId: "88888888-8888-4888-8888-888888888888",
      trainingPath,
      testPath,
      maxEvalExamples: 1,
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
      evaluation: { maxExamples: 3 },
    });

    const result = await runLocalFineTune({ request, config });
    assert.equal(result.report.baseline.eval_examples_used, 3);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("uses prebuilt test split for evaluation while preserving training artifact", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prebuilt-eval-test-"));
  try {
    const trainingPath = join(root, "train.chat.jsonl");
    const testPath = join(root, "test.chat.jsonl");
    await writeFile(trainingPath, `${chatRow("train input", "train output")}\n`, "utf8");
    await writeFile(testPath, `${chatRow("test input", "test output")}\n`, "utf8");

    const request = fineTuneRunRequestSchema.parse({
      run_id: "33333333-3333-4333-8333-333333333333",
      user_id: "local-user",
      behavior_spec_id: "44444444-4444-4444-8444-444444444444",
      run_number: 1,
      spec_snapshot: {
        name: "Prebuilt",
        description: "",
        system_prompt: "Return labels.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "spec input", output: "spec output" }],
      },
      dataset_prebuilt: {
        training: `file://${trainingPath}`,
        test: `file://${testPath}`,
        format: "chat_jsonl",
      },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
    });

    const result = await runLocalFineTune({ request, config });

    assert.equal(result.report.baseline.total, 1);
    assert.equal(result.report.baseline.results[0]?.prompt, "test input");
    assert.equal(result.report.candidate.results[0]?.expected, "test output");
    assert.equal(result.report.baseline.eval_split, "prebuilt_test");
    assert.equal(result.report.run_metadata.eval_split, "prebuilt_test");

    const datasetText = await readFile(result.report.artifact_uris.dataset.replace(/^file:\/\//, ""), "utf8");
    assert.match(datasetText, /train input/);
    assert.doesNotMatch(datasetText, /test input/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("prefers prebuilt validation over test for iterative evaluation", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prebuilt-eval-validation-"));
  try {
    const trainingPath = join(root, "train.chat.jsonl");
    const validationPath = join(root, "validation.chat.jsonl");
    const testPath = join(root, "test.chat.jsonl");
    await writeFile(trainingPath, `${chatRow("train input", "train output")}\n`, "utf8");
    await writeFile(validationPath, `${chatRow("validation input", "validation output")}\n`, "utf8");
    await writeFile(testPath, `${chatRow("test input", "test output")}\n`, "utf8");
    const request = prebuiltRequest({
      runId: "39393939-3939-4939-8939-393939393939",
      trainingPath,
      validationPath,
      testPath,
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
    });

    const result = await runLocalFineTune({ request, config });

    assert.equal(result.report.baseline.eval_split, "prebuilt_validation");
    assert.equal(result.report.run_metadata.eval_split, "prebuilt_validation");
    assert.equal(result.report.baseline.results[0]?.prompt, "validation input");
    assert.equal(result.report.candidate.results[0]?.expected, "validation output");
    assert.equal(result.report.baseline.results.some((entry) => entry.prompt === "test input"), false);
    assert.equal(result.report.candidate.results.some((entry) => entry.prompt === "test input"), false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("DPO dry run trains on preference JSONL and evaluates spec examples", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-dpo-spec-eval-"));
  try {
    const trainingPath = join(root, "preferences.jsonl");
    const preferenceRows = [
      preferenceRow("prefer short", "yes", "yes, and also here is a long aside"),
      preferenceRow("prefer JSON", "{\"ok\":true}", "ok"),
    ].join("\n");
    await writeFile(trainingPath, `${preferenceRows}\n`, "utf8");

    const request = fineTuneRunRequestSchema.parse({
      run_id: "41414141-4141-4441-8441-414141414141",
      user_id: "local-user",
      behavior_spec_id: "42424242-4242-4442-8442-424242424242",
      run_number: 1,
      training_method: "dpo",
      spec_snapshot: {
        name: "DPO Spec Eval",
        description: "",
        system_prompt: "Return concise answers.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "spec prompt", output: "spec answer" }],
      },
      hyperparameters: { n_epochs: 1, dpo_beta: 0.2 },
      dataset_prebuilt: {
        training: `file://${trainingPath}`,
        format: "preference_jsonl",
      },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
      evaluation: { scoring: { mode: "exact_match" } },
    });

    const result = await runLocalFineTune({ request, config });

    assert.equal(result.report.run_metadata.training_method, "dpo");
    assert.equal(result.report.run_metadata.dataset_format, "preference_jsonl");
    assert.equal(result.report.run_metadata.training_example_count, 2);
    assert.equal(result.report.baseline.eval_split, "spec_examples");
    assert.equal(result.report.baseline.results[0]?.prompt, "spec prompt");
    assert.equal(result.report.candidate.results[0]?.expected, "spec answer");

    const datasetText = await readFile(result.report.artifact_uris.dataset.replace(/^file:\/\//, ""), "utf8");
    assert.equal(datasetText, `${preferenceRows}\n`);
    assert.doesNotMatch(datasetText, /spec prompt/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("DPO evaluation prefers prebuilt validation over test and never uses preference rows", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-dpo-test-eval-"));
  try {
    const trainingPath = join(root, "preferences.jsonl");
    const validationPath = join(root, "validation.chat.jsonl");
    const testPath = join(root, "test.chat.jsonl");
    await writeFile(trainingPath, `${preferenceRow("preference prompt", "chosen", "rejected")}\n`, "utf8");
    await writeFile(validationPath, `${chatRow("validation prompt", "validation answer")}\n`, "utf8");
    await writeFile(testPath, `${chatRow("test prompt", "test answer")}\n`, "utf8");

    const request = fineTuneRunRequestSchema.parse({
      run_id: "51515151-5151-4551-8551-515151515151",
      user_id: "local-user",
      behavior_spec_id: "52525252-5252-4552-8552-525252525252",
      run_number: 1,
      training_method: "dpo",
      spec_snapshot: {
        name: "DPO Test Eval",
        description: "",
        system_prompt: "Return concise answers.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "spec prompt", output: "spec answer" }],
      },
      dataset_prebuilt: {
        training: `file://${trainingPath}`,
        validation: `file://${validationPath}`,
        test: `file://${testPath}`,
        format: "preference_jsonl",
      },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
      evaluation: { scoring: { mode: "exact_match" } },
    });

    const result = await runLocalFineTune({ request, config });

    assert.equal(result.report.baseline.eval_split, "prebuilt_validation");
    assert.equal(result.report.baseline.results[0]?.prompt, "validation prompt");
    assert.equal(result.report.baseline.results[0]?.expected, "validation answer");
    assert.notEqual(result.report.baseline.results[0]?.prompt, "preference prompt");

    const testOnlyRequest = fineTuneRunRequestSchema.parse({
      ...request,
      run_id: "53535353-5353-4553-8553-535353535353",
      dataset_prebuilt: {
        training: `file://${trainingPath}`,
        test: `file://${testPath}`,
        format: "preference_jsonl",
      },
    });
    const testOnlyResult = await runLocalFineTune({ request: testOnlyRequest, config });
    assert.equal(testOnlyResult.report.baseline.eval_split, "prebuilt_test");
    assert.equal(testOnlyResult.report.baseline.results[0]?.prompt, "test prompt");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("DPO prepare refreshes when preference file contents change", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-dpo-refresh-"));
  try {
    const trainingPath = join(root, "preferences.jsonl");
    await writeFile(trainingPath, `${preferenceRow("first preference", "chosen", "rejected")}\n`, "utf8");

    const request = fineTuneRunRequestSchema.parse({
      run_id: "61616161-6161-4661-8661-616161616161",
      user_id: "local-user",
      behavior_spec_id: "62626262-6262-4662-8662-626262626262",
      run_number: 1,
      training_method: "dpo",
      spec_snapshot: {
        name: "DPO Refresh",
        description: "",
        system_prompt: "Return concise answers.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "spec prompt", output: "spec answer" }],
      },
      dataset_prebuilt: {
        training: `file://${trainingPath}`,
        format: "preference_jsonl",
      },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
    });

    const prepared = await runLocalFineTuneStage({ request, config, stage: "prepare" });
    assert.match(await readFile(prepared.artifacts.training_jsonl, "utf8"), /first preference/);

    await writeFile(trainingPath, `${preferenceRow("second preference", "chosen", "rejected")}\n`, "utf8");
    await runLocalFineTuneStage({ request, config, stage: "prepare" });
    const refreshed = await readFile(prepared.artifacts.training_jsonl, "utf8");
    assert.match(refreshed, /second preference/);
    assert.doesNotMatch(refreshed, /first preference/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("prepare refreshes when training method changes", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-method-refresh-"));
  try {
    const trainingPath = join(root, "preferences.jsonl");
    await writeFile(trainingPath, `${preferenceRow("preference prompt", "chosen", "rejected")}\n`, "utf8");

    const base = {
      run_id: "71717171-7171-4771-8771-717171717171",
      user_id: "local-user",
      behavior_spec_id: "72727272-7272-4772-8772-727272727272",
      run_number: 1,
      spec_snapshot: {
        name: "Method Refresh",
        description: "",
        system_prompt: "Return labels.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "spec prompt", output: "spec answer" }],
      },
    };
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
    });

    const sftRequest = fineTuneRunRequestSchema.parse(base);
    const prepared = await runLocalFineTuneStage({ request: sftRequest, config, stage: "prepare" });
    assert.match(await readFile(prepared.artifacts.training_jsonl, "utf8"), /messages/);

    const dpoRequest = fineTuneRunRequestSchema.parse({
      ...base,
      training_method: "dpo",
      dataset_prebuilt: {
        training: `file://${trainingPath}`,
        format: "preference_jsonl",
      },
    });
    await runLocalFineTuneStage({ request: dpoRequest, config, stage: "prepare" });
    const refreshed = await readFile(prepared.artifacts.training_jsonl, "utf8");
    assert.match(refreshed, /preference prompt/);
    assert.doesNotMatch(refreshed, /messages/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("runs custom training and evaluation scripts as a minimal nanoGPT-style workflow", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-custom-workflow-"));
  try {
    const trainer = join(root, "tiny-nanogpt-train.mjs");
    await writeFile(trainer, `
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const rows = readFileSync(join(process.env.SM_CHANNEL_TRAINING, "training.jsonl"), "utf8")
  .trim()
  .split("\\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line));
const hyperparameters = JSON.parse(readFileSync(process.env.TT_HYPERPARAMETERS_PATH, "utf8"));
const completions = {};
for (const row of rows) {
  const user = row.messages.find((message) => message.role === "user")?.content;
  const assistant = row.messages.find((message) => message.role === "assistant")?.content;
  completions[user] = assistant;
}
mkdirSync(process.env.SM_MODEL_DIR, { recursive: true });
writeFileSync(join(process.env.SM_MODEL_DIR, "model.json"), JSON.stringify({
  architecture: "tiny-nanogpt-memorizer",
  completions,
}, null, 2));
writeFileSync(join(process.env.SM_MODEL_DIR, "training-metrics.json"), JSON.stringify({
  examples: rows.length,
  n_epochs: Number(hyperparameters.n_epochs),
  model_mode: hyperparameters.model_mode,
  model_family: hyperparameters.model_family ?? null,
  custom_context: hyperparameters.custom_context,
}, null, 2));
console.error("{'loss': '0.01', 'epoch': '1.0'}");
`, "utf8");

    const evaluator = join(root, "tiny-nanogpt-evaluate.mjs");
    await writeFile(evaluator, `
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const inputPath = process.argv[process.argv.indexOf("--input") + 1];
const outputPath = process.argv[process.argv.indexOf("--output") + 1];
const payload = JSON.parse(readFileSync(inputPath, "utf8"));
const modelPath = payload.adapter_path ? join(payload.adapter_path, "model.json") : null;
const model = modelPath ? JSON.parse(readFileSync(modelPath, "utf8")) : { completions: {} };
writeFileSync(outputPath, JSON.stringify({
  provider: "transformers",
  model_id: payload.model_id,
  base_model: payload.base_model,
  adapter_path: payload.adapter_path,
  generation_config: payload.generation,
  results: payload.examples.map((example) => ({
    prompt: example.input,
    expected: example.output,
    actual: model.completions[example.input] ?? "",
    latency_ms: 1,
  })),
}, null, 2));
`, "utf8");

    const trainingPath = join(root, "train.chat.jsonl");
    const testPath = join(root, "test.chat.jsonl");
    const rows = [
      chatRow("next token: A B", "C"),
      chatRow("next token: one two", "three"),
    ].join("\n");
    await writeFile(trainingPath, `${rows}\n`, "utf8");
    await writeFile(testPath, `${rows}\n`, "utf8");

    const request = fineTuneRunRequestSchema.parse({
      run_id: "99999999-1111-4111-8111-111111111111",
      user_id: "local-user",
      behavior_spec_id: "44444444-4444-4444-8444-444444444444",
      run_number: 1,
      spec_snapshot: {
        name: "Tiny nanoGPT",
        description: "",
        system_prompt: "Predict the next token.",
        guidelines: [],
        constraints: [],
        base_model: "external:karpathy/nanogpt",
        examples: [{ input: "spec input", output: "spec output" }],
      },
      hyperparameters: { n_epochs: 1, custom_context: "tiny" },
      dataset_prebuilt: {
        training: `file://${trainingPath}`,
        test: `file://${testPath}`,
        format: "chat_jsonl",
      },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: false,
      training: {
        backend: "command",
        command: [process.execPath, trainer],
        artifact: {
          framework: "nanogpt",
          format: "custom-directory",
          entrypoint: "batch_command",
          servable: false,
        },
      },
      evaluation: {
        inference: {
          provider: "batch_command",
          command: [process.execPath, evaluator],
        },
        scoring: { mode: "exact_match" },
      },
    });

    const result = await runLocalFineTune({ request, config });

    assert.equal(result.report.status, "completed");
    assert.equal(result.report.training.provider, "local-command");
    assert.equal(result.report.training.exit_code, 0);
    assert.equal(result.report.training.metrics?.examples, 2);
    assert.equal(result.report.training.metrics?.model_mode, "external");
    assert.equal(result.report.training.metrics?.model_family, null);
    assert.equal(result.report.training.metrics?.custom_context, "tiny");
    assert.deepEqual(result.report.training.artifact_metadata, {
      framework: "nanogpt",
      format: "custom-directory",
      entrypoint: "batch_command",
      servable: false,
    });
    assert.equal(result.report.base_model, "external:karpathy/nanogpt");
    assert.equal(result.report.baseline.avg_score, 0);
    assert.equal(result.report.candidate.avg_score, 1);
    assert.equal(result.report.candidate.inference_provider, "batch_command");
    assert.equal(result.report.candidate.results[0]?.actual, result.report.candidate.results[0]?.expected);

    const modelText = await readFile(result.report.training.model_artifact_uri.replace(/^file:\/\//, "") + "/model.json", "utf8");
    assert.match(modelText, /tiny-nanogpt-memorizer/);
    const manifest = JSON.parse(await readFile(join(result.artifactDir, "artifact-manifest.json"), "utf8"));
    assert.equal(manifest.schema_version, 1);
    assert.equal(manifest.model.artifact_kind, "directory");
    assert.equal(manifest.model.framework, "nanogpt");
    assert.equal(manifest.model.format, "custom-directory");
    assert.equal(manifest.model.base_model, "external:karpathy/nanogpt");
    assert.equal(manifest.model.servable, false);
    assert.ok(manifest.model.files.some((entry: { path: string }) => entry.path === "model.json"));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("runs a dry workflow one stage at a time and reuses artifacts by default", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-stage-workflow-"));
  try {
    const request = fineTuneRunRequestSchema.parse({
      run_id: "12121212-1212-4212-8212-121212121212",
      user_id: "local-user",
      behavior_spec_id: "34343434-3434-4434-8434-343434343434",
      run_number: 1,
      spec_snapshot: {
        name: "Staged",
        description: "",
        system_prompt: "Return labels.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [
          { input: "Classify: good", output: "positive" },
          { input: "Classify: bad", output: "negative" },
        ],
      },
      hyperparameters: { n_epochs: 1 },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
      evaluation: { scoring: { mode: "exact_match" } },
    });

    const prepared = await runLocalFineTuneStage({ request, config, stage: "prepare" });
    assert.equal(prepared.stage, "prepare");
    assert.equal(await exists(prepared.artifacts.training_jsonl), true);
    assert.equal(await exists(prepared.artifacts.stage_metadata), true);
    assert.equal(await exists(prepared.artifacts.baseline_eval), false);
    assert.equal(await exists(prepared.artifacts.training_report), false);
    const store = createLocalStore(config.storeRoot);
    assert.equal((await store.getRun(request.run_id)).status, "stage_completed");
    assert.equal((await store.getRun(request.run_id)).current_stage, "prepare_completed");

    await runLocalFineTuneStage({ request, config, stage: "baseline" });
    assert.equal(await exists(prepared.artifacts.baseline_eval), true);
    assert.equal(await exists(prepared.artifacts.candidate_eval), false);
    const baseline = JSON.parse(await readFile(prepared.artifacts.baseline_eval, "utf8"));
    await writeFile(prepared.artifacts.baseline_eval, `${JSON.stringify({ ...baseline, avg_score: 0.5 }, null, 2)}\n`, "utf8");

    await assert.rejects(
      runLocalFineTuneStage({ request, config, stage: "baseline" }),
      /Artifact integrity verification failed.*baseline-eval\.json/,
    );

    await runLocalFineTuneStage({ request, config, stage: "baseline", force: true });
    const recomputed = JSON.parse(await readFile(prepared.artifacts.baseline_eval, "utf8"));
    assert.equal(recomputed.avg_score, 0);

    await runLocalFineTuneStage({ request, config, stage: "train" });
    assert.equal(await exists(prepared.artifacts.training_report), true);
    await runLocalFineTuneStage({ request, config, stage: "candidate" });
    assert.equal(await exists(prepared.artifacts.candidate_eval), true);
    await runLocalFineTuneStage({ request, config, stage: "score" });
    assert.equal((await store.getRun(request.run_id)).status, "stage_completed");
    assert.equal((await store.getRun(request.run_id)).current_stage, "score_completed");
    const final = await runLocalFineTuneStage({ request, config, stage: "report" });

    assert.equal(final.report?.status, "completed");
    assert.equal(await exists(prepared.artifacts.report), true);
    assert.equal(final.report?.baseline.avg_score, 0);
    assert.equal(final.report?.candidate.avg_score, 0);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("prepare refreshes stale artifacts when the run input changes", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-stage-prepare-refresh-"));
  try {
    const baseRequest = {
      run_id: "13131313-1313-4313-8313-131313131313",
      user_id: "local-user",
      behavior_spec_id: "24242424-2424-4424-8424-242424242424",
      run_number: 1,
      spec_snapshot: {
        name: "Refresh",
        description: "",
        system_prompt: "Return labels.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "Classify: first", output: "one" }],
      },
      hyperparameters: { n_epochs: 1 },
    };
    const firstRequest = fineTuneRunRequestSchema.parse(baseRequest);
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
      evaluation: { scoring: { mode: "exact_match" } },
    });

    const prepared = await runLocalFineTuneStage({ request: firstRequest, config, stage: "prepare" });
    assert.match(await readFile(prepared.artifacts.training_jsonl, "utf8"), /Classify: first/);
    await runLocalFineTuneStage({ request: firstRequest, config, stage: "baseline" });
    assert.equal(await exists(prepared.artifacts.baseline_eval), true);

    const secondRequest = fineTuneRunRequestSchema.parse({
      ...baseRequest,
      spec_snapshot: {
        ...baseRequest.spec_snapshot,
        examples: [{ input: "Classify: second", output: "two" }],
      },
    });
    await runLocalFineTuneStage({ request: secondRequest, config, stage: "prepare" });
    const refreshed = await readFile(prepared.artifacts.training_jsonl, "utf8");
    assert.match(refreshed, /Classify: second/);
    assert.doesNotMatch(refreshed, /Classify: first/);
    assert.equal(await exists(prepared.artifacts.baseline_eval), false);

    await writeFile(prepared.artifacts.training_jsonl, "stale\n", "utf8");
    await runLocalFineTuneStage({ request: secondRequest, config, stage: "all", force: true });
    assert.doesNotMatch(await readFile(prepared.artifacts.training_jsonl, "utf8"), /^stale/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("candidate stage fails clearly when no training artifact is available", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-stage-candidate-missing-"));
  try {
    const request = fineTuneRunRequestSchema.parse({
      run_id: "56565656-5656-4565-8565-565656565656",
      user_id: "local-user",
      behavior_spec_id: "78787878-7878-4787-8787-787878787878",
      run_number: 1,
      spec_snapshot: {
        name: "Missing training",
        description: "",
        system_prompt: "Return labels.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [
          { input: "Classify: good", output: "positive" },
          { input: "Classify: bad", output: "negative" },
        ],
      },
      hyperparameters: { n_epochs: 1 },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
    });

    await runLocalFineTuneStage({ request, config, stage: "prepare" });
    await assert.rejects(
      runLocalFineTuneStage({ request, config, stage: "candidate" }),
      /requires current verified training output/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("candidate stage accepts an external model artifact", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-stage-external-model-"));
  try {
    const request = fineTuneRunRequestSchema.parse({
      run_id: "90909090-9090-4909-8909-909090909090",
      user_id: "local-user",
      behavior_spec_id: "abababab-abab-4aba-8aba-abababababab",
      run_number: 1,
      spec_snapshot: {
        name: "External candidate",
        description: "",
        system_prompt: "Return labels.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [
          { input: "Classify: good", output: "positive" },
          { input: "Classify: bad", output: "negative" },
        ],
      },
      hyperparameters: { n_epochs: 1 },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
      evaluation: { scoring: { mode: "exact_match" } },
    });

    const prepared = await runLocalFineTuneStage({ request, config, stage: "prepare" });
    await mkdir(join(root, "external-adapter"), { recursive: true });
    await writeFile(join(root, "external-adapter", "adapter_model.safetensors"), "model", "utf8");
    await writeFile(join(root, "external-adapter", "adapter_config.json"), "{}", "utf8");
    await runLocalFineTuneStage({
      request,
      config,
      stage: "candidate",
      modelArtifact: `file://${join(root, "external-adapter")}`,
    });
    const training = JSON.parse(await readFile(prepared.artifacts.training_report, "utf8"));
    assert.equal(training.metrics.external_model_artifact, true);
    assert.equal(training.model_artifact_uri, `file://${join(root, "external-adapter")}`);
    assert.equal(await exists(prepared.artifacts.candidate_eval), true);

    await mkdir(join(root, "replacement-adapter"), { recursive: true });
    await writeFile(join(root, "replacement-adapter", "adapter_model.safetensors"), "replacement", "utf8");
    await writeFile(join(root, "replacement-adapter", "adapter_config.json"), "{}", "utf8");
    await runLocalFineTuneStage({
      request,
      config,
      stage: "candidate",
      modelArtifact: `file://${join(root, "replacement-adapter")}`,
    });
    const replacement = JSON.parse(await readFile(prepared.artifacts.training_report, "utf8"));
    assert.equal(replacement.model_artifact_uri, `file://${join(root, "replacement-adapter")}`);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("continued run evaluates parent model as baseline", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-stage-parent-model-"));
  try {
    const parentArtifact = `file://${join(root, "parent-adapter")}`;
    await mkdir(join(root, "parent-adapter"), { recursive: true });
    await writeFile(join(root, "parent-adapter", "adapter_model.safetensors"), "parent", "utf8");
    await writeFile(join(root, "parent-adapter", "adapter_config.json"), "{}", "utf8");
    const request = fineTuneRunRequestSchema.parse({
      run_id: "14141414-1414-4414-8414-141414141414",
      user_id: "local-user",
      behavior_spec_id: "25252525-2525-4525-8525-252525252525",
      run_number: 2,
      spec_snapshot: {
        name: "Continued candidate",
        description: "",
        system_prompt: "Return labels.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [
          { input: "Classify: good", output: "positive" },
          { input: "Classify: bad", output: "negative" },
        ],
      },
      hyperparameters: {
        n_epochs: 1,
        parent_model_artifact: parentArtifact,
      },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
      evaluation: { scoring: { mode: "exact_match" } },
    });

    const result = await runLocalFineTuneStage({ request, config, stage: "all" });
    const baseline = JSON.parse(await readFile(result.artifacts.baseline_eval, "utf8"));
    const training = JSON.parse(await readFile(result.artifacts.training_report, "utf8"));

    assert.equal(baseline.model_id, parentArtifact);
    assert.equal(training.parent_model_artifact_uri, parentArtifact);
    assert.equal(result.report?.run_metadata.parent_model_artifact, parentArtifact);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("config paths resolve from the config file while bundled runner paths and home paths stay portable", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-config-paths-"));
  try {
    const configDir = join(root, "nested");
    await mkdir(configDir, { recursive: true });
    const configPath = join(configDir, "local-runner.json");
    await writeFile(configPath, `${JSON.stringify({
      artifactRoot: "./artifacts",
      storeRoot: "~/tt-local-store-test",
      paths: { baseModel: "../models/base", modelCache: "./cache" },
      training: { project: "./trainer", script: "./trainer/run.py", cwd: "." },
      evaluation: {
        inference: {
          project: "training/local-runner",
          script: "training/local-runner/src/evaluate.py",
        },
      },
    }, null, 2)}\n`, "utf8");

    const config = await loadLocalRunnerConfig(configPath);
    assert.equal(config.artifactRoot, join(configDir, "artifacts"));
    assert.equal(config.storeRoot, join(homedir(), "tt-local-store-test"));
    assert.equal(config.paths.baseModel, join(root, "models", "base"));
    assert.equal(config.paths.modelCache, join(configDir, "cache"));
    assert.equal(config.training.project, join(configDir, "trainer"));
    assert.equal(config.training.script, join(configDir, "trainer", "run.py"));
    assert.equal(config.training.cwd, configDir);
    assert.equal(config.evaluation.inference.project, "training/local-runner");
    assert.equal(config.evaluation.inference.script, "training/local-runner/src/evaluate.py");

    const defaults = await loadLocalRunnerConfig();
    assert.equal(defaults.training.project, "training/local-runner");
    assert.equal(defaults.evaluation.inference.project, "training/local-runner");
    assert.equal(defaults.evaluation.timeoutMs, 1_800_000);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("changing only n_epochs invalidates prepared and downstream artifacts", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-request-fingerprint-"));
  try {
    const base = {
      run_id: "61616161-6161-4161-8161-616161616161",
      user_id: "local-user",
      behavior_spec_id: "62626262-6262-4262-8262-626262626262",
      run_number: 1,
      spec_snapshot: {
        name: "Fingerprint",
        description: "",
        system_prompt: "Return labels.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "good", output: "positive" }],
      },
    };
    const first = fineTuneRunRequestSchema.parse({ ...base, hyperparameters: { n_epochs: 1 } });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
    });
    const result = await runLocalFineTuneStage({ request: first, config, stage: "train" });
    const before = JSON.parse(await readFile(result.artifacts.stage_metadata, "utf8"));
    assert.equal(await exists(result.artifacts.training_report), true);

    const second = fineTuneRunRequestSchema.parse({ ...base, hyperparameters: { n_epochs: 2 } });
    await runLocalFineTuneStage({ request: second, config, stage: "prepare" });
    const after = JSON.parse(await readFile(result.artifacts.stage_metadata, "utf8"));
    assert.notEqual(after.request_fingerprint, before.request_fingerprint);
    assert.notEqual(after.source_fingerprint, before.source_fingerprint);
    assert.equal(await exists(result.artifacts.training_report), false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("changing local base-model bytes invalidates prepared and downstream artifacts", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-base-fingerprint-"));
  try {
    const baseModel = join(root, "base-model");
    await mkdir(baseModel);
    await Promise.all([
      writeFile(join(baseModel, "config.json"), "{}"),
      writeFile(join(baseModel, "tokenizer_config.json"), "{}"),
      writeFile(join(baseModel, "tokenizer.json"), "{}"),
      writeFile(join(baseModel, "model.safetensors"), "weights-v1"),
    ]);
    const request = fineTuneRunRequestSchema.parse({
      run_id: "63636363-6363-4363-8363-636363636363",
      user_id: "local-user",
      behavior_spec_id: "64646464-6464-4464-8464-646464646464",
      run_number: 1,
      spec_snapshot: {
        name: "Local base fingerprint",
        description: "",
        system_prompt: "Return labels.",
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "good", output: "positive" }],
      },
      hyperparameters: { n_epochs: 1 },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      paths: { baseModel },
      dryRun: true,
    });
    const trained = await runLocalFineTuneStage({ request, config, stage: "train" });
    const before = JSON.parse(await readFile(trained.artifacts.stage_metadata, "utf8"));
    assert.ok(before.base_model_fingerprint);
    assert.equal(await exists(trained.artifacts.training_report), true);

    await writeFile(join(baseModel, "model.safetensors"), "weights-v2");
    await runLocalFineTuneStage({ request, config, stage: "prepare" });
    const after = JSON.parse(await readFile(trained.artifacts.stage_metadata, "utf8"));
    assert.notEqual(after.base_model_fingerprint, before.base_model_fingerprint);
    assert.notEqual(after.source_fingerprint, before.source_fingerprint);
    assert.equal(await exists(trained.artifacts.training_report), false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("changing a local multimodal asset invalidates prepared artifacts", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-image-fingerprint-"));
  try {
    const imagePath = join(root, "chart.png");
    await writeFile(imagePath, "image-v1");
    const request = fineTuneRunRequestSchema.parse({
      run_id: "65656565-6565-4565-8565-656565656565",
      user_id: "local-user",
      behavior_spec_id: "66666666-6666-4666-8666-666666666666",
      run_number: 1,
      spec_snapshot: {
        name: "Image fingerprint",
        description: "",
        system_prompt: "Read the chart.",
        base_model: "Qwen/Qwen3-VL-2B-Instruct",
        examples: [{
          input: "What is shown?",
          output: "A chart",
          input_assets: [{ type: "image", image: imagePath }],
        }],
      },
      hyperparameters: { n_epochs: 1 },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
    });
    const prepared = await runLocalFineTuneStage({ request, config, stage: "prepare" });
    const before = JSON.parse(await readFile(prepared.artifacts.stage_metadata, "utf8"));
    assert.ok(Object.keys(before.dataset_fingerprints).some((key) => key.startsWith("spec_asset:")));

    await writeFile(imagePath, "image-v2");
    await runLocalFineTuneStage({ request, config, stage: "prepare" });
    const after = JSON.parse(await readFile(prepared.artifacts.stage_metadata, "utf8"));
    assert.notEqual(after.source_fingerprint, before.source_fingerprint);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("successful command exit without model payload fails and registers no model", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-empty-model-"));
  try {
    const trainer = join(root, "empty-trainer.mjs");
    await writeFile(trainer, "process.exit(0);\n", "utf8");
    const request = fineTuneRunRequestSchema.parse({
      run_id: "71717171-7171-4171-8171-717171717171",
      user_id: "local-user",
      behavior_spec_id: "72727272-7272-4272-8272-727272727272",
      run_number: 1,
      spec_snapshot: {
        name: "Empty output",
        description: "",
        system_prompt: "Return labels.",
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "good", output: "positive" }],
      },
      hyperparameters: { n_epochs: 1 },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      training: {
        backend: "command",
        command: [process.execPath, trainer],
        artifact: { framework: "custom", format: "custom-directory", servable: false },
      },
      evaluation: { inference: { provider: "none" }, scoring: { mode: "exact_match" } },
    });
    await assert.rejects(
      runLocalFineTuneStage({ request, config, stage: "train" }),
      /without a usable model artifact|no model payload files/,
    );
    const store = createLocalStore(config.storeRoot);
    assert.equal((await store.listModels()).length, 0);
    const state = await store.getRun(request.run_id);
    assert.equal(state.status, "failed");
    assert.match(state.error ?? "", /model payload/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("trained model remains discoverable when candidate evaluation later fails", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-model-before-eval-"));
  try {
    const trainer = join(root, "trainer.mjs");
    await writeFile(trainer, `
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
mkdirSync(process.env.SM_MODEL_DIR, { recursive: true });
writeFileSync(join(process.env.SM_MODEL_DIR, "model.json"), JSON.stringify({ weights: [1] }));
`, "utf8");
    const evaluator = join(root, "fail-eval.mjs");
    await writeFile(evaluator, "process.exit(17);\n", "utf8");
    const request = fineTuneRunRequestSchema.parse({
      run_id: "81818181-8181-4181-8181-818181818181",
      user_id: "local-user",
      behavior_spec_id: "82828282-8282-4282-8282-828282828282",
      run_number: 1,
      spec_snapshot: {
        name: "Evaluation failure",
        description: "",
        system_prompt: "Return labels.",
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "good", output: "positive" }],
      },
      hyperparameters: { n_epochs: 1 },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      training: {
        backend: "command",
        command: [process.execPath, trainer],
        artifact: { framework: "custom", format: "custom-directory", servable: false },
      },
      evaluation: {
        inference: { provider: "batch_command", command: [process.execPath, evaluator] },
        scoring: { mode: "exact_match" },
      },
    });
    const trained = await runLocalFineTuneStage({ request, config, stage: "train" });
    const store = createLocalStore(config.storeRoot);
    assert.equal((await store.listModels())[0]?.run_id, request.run_id);

    await assert.rejects(
      runLocalFineTuneStage({ request, config, stage: "candidate" }),
      /inference exited 17/i,
    );
    assert.equal((await store.listModels())[0]?.run_id, request.run_id);
    const state = await store.getRun(request.run_id);
    assert.equal(state.status, "failed");
    assert.equal(state.model_id, `local-${request.run_id}`);
    assert.equal(await exists(trained.artifacts.training_report), true);
    assert.equal(await exists(trained.artifacts.artifact_manifest), true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("a cancellation request remains cancelled instead of being overwritten as failed", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-cancelled-stage-"));
  try {
    const request = fineTuneRunRequestSchema.parse({
      run_id: "91919191-9191-4191-8191-919191919191",
      user_id: "local-user",
      behavior_spec_id: "92929292-9292-4292-8292-929292929292",
      run_number: 1,
      spec_snapshot: {
        name: "Cancelled",
        description: "",
        system_prompt: "Return labels.",
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "good", output: "positive" }],
      },
      hyperparameters: { n_epochs: 1 },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
    });
    await runLocalFineTuneStage({ request, config, stage: "prepare" });
    const store = createLocalStore(config.storeRoot);
    await store.updateRun({
      runId: request.run_id,
      status: "preparing",
      stage: "preparing",
      message: "Resuming the run before cancellation.",
    });
    await store.cancelRun(request.run_id);
    await assert.rejects(
      runLocalFineTuneStage({ request, config, stage: "baseline" }),
      /cancelled/,
    );
    const state = await store.getRun(request.run_id);
    assert.equal(state.status, "cancelled");
    assert.equal(state.current_stage, "cancelled");
    assert.ok(state.completed_at);
    assert.equal(state.error, undefined);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("changing only scoring reuses generated evaluations for the score stage", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-rescore-stage-"));
  try {
    const request = fineTuneRunRequestSchema.parse({
      run_id: "93939393-9393-4393-8393-939393939393",
      user_id: "local-user",
      behavior_spec_id: "94949494-9494-4494-8494-949494949494",
      run_number: 1,
      spec_snapshot: {
        name: "Rescore",
        description: "",
        system_prompt: "Return JSON.",
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "classify", output: "{\"label\":\"positive\"}" }],
      },
      hyperparameters: { n_epochs: 1 },
    });
    const common = {
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      dryRun: true,
    };
    const exactConfig = localRunnerConfigSchema.parse({
      ...common,
      evaluation: { scoring: { mode: "exact_match" } },
    });
    const initial = await runLocalFineTune({ request, config: exactConfig });
    const baselineBefore = await readFile(initial.report.artifact_uris.baseline_eval.replace(/^file:\/\//, ""), "utf8");

    const jsonConfig = localRunnerConfigSchema.parse({
      ...common,
      evaluation: { scoring: { mode: "json_fields", fields: ["label"] } },
    });
    const scored = await runLocalFineTuneStage({ request, config: jsonConfig, stage: "score" });
    const baselineAfter = JSON.parse(await readFile(scored.artifacts.baseline_eval, "utf8")) as {
      scoring_mode?: string;
      results: Array<{ actual: string }>;
    };
    assert.equal(baselineAfter.scoring_mode, "json_fields");
    assert.equal(baselineAfter.results[0]?.actual, JSON.parse(baselineBefore).results[0]?.actual);
    assert.equal(await exists(scored.artifacts.training_report), true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
