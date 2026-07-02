import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { test } from "node:test";
import { fineTuneRunRequestSchema, localRunnerConfigSchema } from "../src/contracts.js";
import { runLocalFineTune } from "../src/orchestrator.js";

function chatRow(input: string, output: string): string {
  return JSON.stringify({
    messages: [
      { role: "system", content: "Return labels." },
      { role: "user", content: input },
      { role: "assistant", content: output },
    ],
  });
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
      evaluation: { mode: "heuristic" },
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
      evaluation: { mode: "heuristic" },
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
      evaluation: { mode: "heuristic" },
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
      evaluation: { mode: "heuristic" },
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
      evaluation: { mode: "heuristic" },
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
      evaluation: { mode: "heuristic" },
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
      evaluation: { mode: "heuristic", maxExamples: 3 },
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
      evaluation: { mode: "heuristic" },
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
