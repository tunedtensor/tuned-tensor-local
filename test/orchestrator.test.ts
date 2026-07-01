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
    assert.equal(result.report.baseline.total, 2);
    assert.equal(result.report.candidate.total, 2);

    const reportText = await readFile(result.reportPath, "utf8");
    assert.match(reportText, /local-uv/);
    const datasetText = await readFile(result.report.artifact_uris.dataset.replace(/^file:\/\//, ""), "utf8");
    assert.match(datasetText, /Classify: good/);
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

    const datasetText = await readFile(result.report.artifact_uris.dataset.replace(/^file:\/\//, ""), "utf8");
    assert.match(datasetText, /train input/);
    assert.doesNotMatch(datasetText, /test input/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
