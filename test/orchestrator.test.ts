import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  fineTuneRunRequestSchema,
  localRunnerConfigSchema,
  type FineTuneRunRequest,
} from "../src/contracts.js";
import {
  loadLocalRunnerConfig,
  runLocalFineTune,
  validateLocalFineTuneInput,
} from "../src/orchestrator.js";
import { defaultArtifactPrefix, resolveRunArtifacts } from "../src/artifacts.js";
import { createLocalStore } from "../src/store.js";

const behaviorSpecId = "22222222-2222-4222-8222-222222222222";

function chatRow(input: string, output: string): string {
  return JSON.stringify({
    messages: [
      { role: "system", content: "Return labels." },
      { role: "user", content: input },
      { role: "assistant", content: output },
    ],
  });
}

function requestFixture(options: {
  runId: string;
  examples?: Array<{ input: string; output: string }>;
  datasetPrebuilt?: {
    training: string;
    validation?: string;
    test?: string;
    format: "chat_jsonl";
  };
  nEpochs?: number;
}): FineTuneRunRequest {
  return fineTuneRunRequestSchema.parse({
    run_id: options.runId,
    user_id: "local-user",
    behavior_spec_id: behaviorSpecId,
    run_number: 1,
    spec_snapshot: {
      name: "Local SFT",
      description: "",
      system_prompt: "Return labels.",
      guidelines: [],
      constraints: [],
      base_model: "Qwen/Qwen3.5-2B",
      examples: options.examples ?? [
        { input: "Classify: good", output: "positive" },
        { input: "Classify: bad", output: "negative" },
      ],
    },
    hyperparameters: { n_epochs: options.nEpochs ?? 1 },
    ...(options.datasetPrebuilt ? { dataset_prebuilt: options.datasetPrebuilt } : {}),
  });
}

function configFixture(root: string, overrides: Record<string, unknown> = {}) {
  return localRunnerConfigSchema.parse({
    artifactRoot: join(root, "artifacts"),
    storeRoot: join(root, "store"),
    dryRun: true,
    ...overrides,
  });
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

test("runs the complete dry SFT workflow with distinct train and holdout data", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-sft-"));
  try {
    const request = requestFixture({
      runId: "11111111-1111-4111-8111-111111111111",
    });
    const config = configFixture(root);
    const events: string[] = [];

    const result = await runLocalFineTune({
      request,
      config,
      reporter: {
        onEvent(event) {
          events.push(event.stage);
        },
      },
    });

    assert.equal(result.report.status, "completed");
    assert.equal(result.report.training.provider, "local-uv");
    assert.equal(result.report.baseline.eval_split, "spec_holdout");
    assert.equal(result.report.run_metadata.training_example_count, 1);
    assert.equal(result.report.run_metadata.eval_examples_total, 1);
    assert.deepEqual(
      result.report.baseline.results.map((entry) => entry.prompt),
      result.report.candidate.results.map((entry) => entry.prompt),
    );

    const dataset = await readFile(
      result.report.artifact_uris.dataset.replace(/^file:\/\//, ""),
      "utf8",
    );
    const evaluatedPrompt = result.report.baseline.results[0]?.prompt;
    assert.ok(evaluatedPrompt);
    assert.equal(dataset.trim().split("\n").length, 1);
    assert.equal(dataset.includes(evaluatedPrompt), false);
    assert.deepEqual(events, [
      "queued",
      "preparing",
      "evaluating_baseline",
      "training",
      "evaluating_candidate",
      "reporting",
      "completed",
    ]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("repeated runs of one spec use the same held-out prompts", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-stable-holdout-"));
  try {
    const config = configFixture(root);
    const first = await runLocalFineTune({
      request: requestFixture({
        runId: "12121212-1212-4212-8212-121212121212",
        examples: [
          { input: "one", output: "1" },
          { input: "two", output: "2" },
          { input: "three", output: "3" },
          { input: "four", output: "4" },
        ],
      }),
      config,
    });
    const second = await runLocalFineTune({
      request: requestFixture({
        runId: "13131313-1313-4313-8313-131313131313",
        examples: [
          { input: "one", output: "1" },
          { input: "two", output: "2" },
          { input: "three", output: "3" },
          { input: "four", output: "4" },
        ],
      }),
      config,
    });

    assert.deepEqual(
      second.report.baseline.results.map((result) => result.prompt),
      first.report.baseline.results.map((result) => result.prompt),
    );
    assert.equal(second.report.run_metadata.eval_sample_seed, first.report.run_metadata.eval_sample_seed);
    assert.equal(
      second.report.baseline.eval_sample_seed,
      second.report.run_metadata.eval_sample_seed,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects a one-example real run before creating store or artifact state", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-validation-order-"));
  try {
    const request = requestFixture({
      runId: "33333333-3333-4333-8333-333333333333",
      examples: [{ input: "only input", output: "only output" }],
    });
    const config = configFixture(root, { dryRun: false });

    await assert.rejects(
      runLocalFineTune({ request, config }),
      /at least 2 inline examples/,
    );
    assert.equal(await exists(config.artifactRoot), false);
    assert.equal(await exists(config.storeRoot!), false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects duplicate inline prompts before creating store or artifact state", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-inline-overlap-"));
  try {
    const request = requestFixture({
      runId: "34343434-3434-4343-8343-343434343434",
      examples: [
        { input: "Duplicate   prompt", output: "first answer" },
        { input: " duplicate prompt ", output: "second answer" },
      ],
    });
    const config = configFixture(root, { dryRun: false });

    await assert.rejects(
      runLocalFineTune({ request, config }),
      /duplicate inputs.*distinct prompts/,
    );
    assert.equal(await exists(config.artifactRoot), false);
    assert.equal(await exists(config.storeRoot!), false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("dry-run validation may replace an empty placeholder dataset", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-dry-placeholder-"));
  try {
    const rawRequest = {
      run_id: "44444444-4444-4444-8444-444444444444",
      user_id: "local-user",
      behavior_spec_id: behaviorSpecId,
      run_number: 1,
      spec_snapshot: {
        name: "Placeholder",
        description: "",
        system_prompt: "",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [],
      },
      hyperparameters: { n_epochs: 1 },
    };
    const config = configFixture(root);
    const validated = await validateLocalFineTuneInput({ request: rawRequest, config });

    assert.equal(validated.request.spec_snapshot.examples.length, 2);
    assert.match(validated.request.spec_snapshot.examples[0]!.input, /Dry-run placeholder/);
    assert.equal(await exists(config.artifactRoot), false);
    assert.equal(await exists(config.storeRoot!), false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("parses every prebuilt split before claiming a run", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prebuilt-invalid-"));
  try {
    const training = join(root, "training.jsonl");
    const validation = join(root, "validation.jsonl");
    const malformedTest = join(root, "test.jsonl");
    await writeFile(training, `${chatRow("train", "answer")}\n`, "utf8");
    await writeFile(validation, `${chatRow("validate", "answer")}\n`, "utf8");
    await writeFile(malformedTest, "{not-json}\n", "utf8");
    const request = requestFixture({
      runId: "55555555-5555-4555-8555-555555555555",
      datasetPrebuilt: {
        training: `file://${training}`,
        validation: `file://${validation}`,
        test: `file://${malformedTest}`,
        format: "chat_jsonl",
      },
    });
    const config = configFixture(root);

    await assert.rejects(
      runLocalFineTune({ request, config }),
      /Invalid chat JSONL row 1: malformed JSON/,
    );
    assert.equal(await exists(config.artifactRoot), false);
    assert.equal(await exists(config.storeRoot!), false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects prebuilt rows whose system message differs from the spec", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prebuilt-system-"));
  try {
    const training = join(root, "training.jsonl");
    const validation = join(root, "validation.jsonl");
    const mismatched = JSON.stringify({
      messages: [
        { role: "system", content: "Use a different policy." },
        { role: "user", content: "train" },
        { role: "assistant", content: "answer" },
      ],
    });
    await writeFile(training, `${mismatched}\n`, "utf8");
    await writeFile(validation, `${chatRow("validate", "answer")}\n`, "utf8");
    const request = requestFixture({
      runId: "56565656-5656-4565-8565-565656565656",
      datasetPrebuilt: {
        training: `file://${training}`,
        validation: `file://${validation}`,
        format: "chat_jsonl",
      },
    });
    const config = configFixture(root);

    await assert.rejects(
      runLocalFineTune({ request, config }),
      /system message must match the behavior spec/,
    );
    assert.equal(await exists(config.artifactRoot), false);
    assert.equal(await exists(config.storeRoot!), false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects overlapping prebuilt training and validation before claiming a run", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prebuilt-overlap-"));
  try {
    const training = join(root, "training.jsonl");
    const validation = join(root, "validation.jsonl");
    await writeFile(training, `${chatRow("duplicate input", "training answer")}\n`, "utf8");
    await writeFile(validation, `${chatRow("duplicate input", "validation answer")}\n`, "utf8");
    const request = requestFixture({
      runId: "57575757-5757-4575-8575-575757575757",
      datasetPrebuilt: {
        training: `file://${training}`,
        validation: `file://${validation}`,
        format: "chat_jsonl",
      },
    });
    const config = configFixture(root);

    await assert.rejects(
      runLocalFineTune({ request, config }),
      /training and validation data overlap on 1 input/,
    );
    assert.equal(await exists(config.artifactRoot), false);
    assert.equal(await exists(config.storeRoot!), false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects duplicate prompts within a prebuilt evaluation split", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prebuilt-eval-duplicate-"));
  try {
    const training = join(root, "training.jsonl");
    const validation = join(root, "validation.jsonl");
    await writeFile(training, `${chatRow("train", "answer")}\n`, "utf8");
    await writeFile(
      validation,
      `${chatRow("same prompt", "first")}\n${chatRow("  SAME   PROMPT  ", "second")}\n`,
      "utf8",
    );
    const request = requestFixture({
      runId: "58585858-5858-4585-8585-585858585858",
      datasetPrebuilt: {
        training: `file://${training}`,
        validation: `file://${validation}`,
        format: "chat_jsonl",
      },
    });
    const config = configFixture(root);

    await assert.rejects(
      runLocalFineTune({ request, config }),
      /validation data contains duplicate inputs/,
    );
    assert.equal(await exists(config.artifactRoot), false);
    assert.equal(await exists(config.storeRoot!), false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("normalizes prebuilt training JSONL and evaluates the validation split", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prebuilt-valid-"));
  try {
    const training = join(root, "training.jsonl");
    const validation = join(root, "validation.jsonl");
    const trainRows = [
      chatRow("train one", "one"),
      chatRow("train two", "two"),
    ];
    await writeFile(training, `  ${trainRows[0]}  \n\n${trainRows[1]}\n`, "utf8");
    await writeFile(validation, `${chatRow("validation input", "validation output")}\n`, "utf8");
    const request = requestFixture({
      runId: "66666666-6666-4666-8666-666666666666",
      datasetPrebuilt: {
        training: `file://${training}`,
        validation: `file://${validation}`,
        format: "chat_jsonl",
      },
    });
    const config = configFixture(root);

    const result = await runLocalFineTune({ request, config });
    assert.equal(result.report.baseline.eval_split, "prebuilt_validation");
    assert.equal(result.report.baseline.results[0]?.prompt, "validation input");
    assert.equal(result.report.run_metadata.training_example_count, 2);
    const persisted = await readFile(
      result.report.artifact_uris.dataset.replace(/^file:\/\//, ""),
      "utf8",
    );
    assert.equal(persisted, `${trainRows.join("\n")}\n`);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("requires a separate prebuilt evaluation split for real runs", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prebuilt-guard-"));
  try {
    const training = join(root, "training.jsonl");
    await writeFile(training, `${chatRow("train", "answer")}\n`, "utf8");
    const config = configFixture(root, { dryRun: false });

    assert.throws(
      () => requestFixture({
        runId: "77777777-7777-4777-8777-777777777777",
        datasetPrebuilt: {
          training: `file://${training}`,
          format: "chat_jsonl",
        },
      }),
      /distinct validation or test split/,
    );
    assert.equal(await exists(config.artifactRoot), false);
    assert.equal(await exists(config.storeRoot!), false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("automatically resumes verified stages and rejects tampered artifacts", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-stages-"));
  try {
    const request = requestFixture({
      runId: "88888888-8888-4888-8888-888888888888",
    });
    const config = configFixture(root);
    const first = await runLocalFineTune({ request, config });
    const paths = {
      trainingJsonl: join(first.artifactDir, "training.jsonl"),
      baselineEval: join(first.artifactDir, "baseline-eval.json"),
      trainingReport: join(first.artifactDir, "training-report.json"),
      candidateEval: join(first.artifactDir, "candidate-eval.json"),
      report: join(first.artifactDir, "run-report.json"),
    };
    const messages: string[] = [];
    const resumed = await runLocalFineTune({
      request,
      config,
      reporter: {
        onEvent(event) {
          messages.push(event.message);
        },
      },
    });

    assert.equal(resumed.report.status, "completed");
    assert.equal(await exists(paths.trainingJsonl), true);
    assert.equal(await exists(paths.trainingReport), true);
    assert.equal(await exists(paths.candidateEval), true);
    assert.equal(await exists(paths.report), true);
    assert.ok(messages.includes("Reusing prepared local run artifacts."));
    assert.ok(messages.includes("Reusing existing baseline evaluation."));
    assert.ok(messages.includes("Reusing existing training result."));
    assert.ok(messages.includes("Reusing existing candidate evaluation."));

    const baseline = JSON.parse(await readFile(paths.baselineEval, "utf8")) as {
      avg_score: number;
    };
    await writeFile(
      paths.baselineEval,
      `${JSON.stringify({ ...baseline, avg_score: 0.5 }, null, 2)}\n`,
      "utf8",
    );
    await assert.rejects(
      runLocalFineTune({ request, config }),
      /Artifact integrity verification failed.*baseline-eval\.json/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("request changes invalidate prepared and downstream stage artifacts", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-request-fingerprint-"));
  try {
    const runId = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
    const first = requestFixture({ runId, nEpochs: 1 });
    const config = configFixture(root);
    const firstRun = await runLocalFineTune({ request: first, config });
    const metadataPath = join(firstRun.artifactDir, "stage-metadata.json");
    const trainingReportPath = join(firstRun.artifactDir, "training-report.json");
    const before = JSON.parse(await readFile(metadataPath, "utf8")) as {
      request_fingerprint: string;
      source_fingerprint: string;
    };

    const second = requestFixture({ runId, nEpochs: 2 });
    const secondRun = await runLocalFineTune({ request: second, config });
    const after = JSON.parse(await readFile(metadataPath, "utf8")) as {
      request_fingerprint: string;
      source_fingerprint: string;
    };
    assert.notEqual(after.request_fingerprint, before.request_fingerprint);
    assert.notEqual(after.source_fingerprint, before.source_fingerprint);
    assert.equal(secondRun.report.status, "completed");
    assert.equal(await exists(trainingReportPath), true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("local base-model byte changes invalidate prepared outputs", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-base-fingerprint-"));
  try {
    const baseModel = join(root, "base-model");
    await mkdir(baseModel);
    await Promise.all([
      writeFile(join(baseModel, "config.json"), JSON.stringify({
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
      })),
      writeFile(join(baseModel, "tokenizer_config.json"), "{}"),
      writeFile(join(baseModel, "tokenizer.json"), "{}"),
      writeFile(join(baseModel, "model.safetensors"), "weights-v1"),
    ]);
    const request = requestFixture({
      runId: "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
    });
    const config = configFixture(root, { paths: { baseModel } });
    const firstRun = await runLocalFineTune({ request, config });
    const metadataPath = join(firstRun.artifactDir, "stage-metadata.json");
    const before = JSON.parse(await readFile(metadataPath, "utf8")) as {
      base_model_fingerprint: string;
    };

    await writeFile(join(baseModel, "model.safetensors"), "weights-v2");
    const secondRun = await runLocalFineTune({ request, config });
    const after = JSON.parse(await readFile(metadataPath, "utf8")) as {
      base_model_fingerprint: string;
    };
    assert.notEqual(after.base_model_fingerprint, before.base_model_fingerprint);
    assert.equal(secondRun.report.status, "completed");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("config resolves data paths and rejects runner overrides", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-config-paths-"));
  try {
    const configDir = join(root, "nested");
    await mkdir(configDir);
    const configPath = join(configDir, "local-runner.json");
    await writeFile(configPath, `${JSON.stringify({
      artifactRoot: "./artifacts",
      storeRoot: "~/tt-local-store-test",
      paths: { baseModel: "../models/base", modelCache: "./cache" },
      evaluation: {
        inference: { device: "cuda" },
      },
    }, null, 2)}\n`, "utf8");

    const config = await loadLocalRunnerConfig(configPath);
    assert.equal(config.artifactRoot, join(configDir, "artifacts"));
    assert.equal(config.storeRoot, join(homedir(), "tt-local-store-test"));
    assert.equal(config.paths.baseModel, join(root, "models", "base"));
    assert.equal(config.paths.modelCache, join(configDir, "cache"));
    assert.equal("training" in config, false);
    assert.equal("project" in config.evaluation.inference, false);

    await writeFile(configPath, `${JSON.stringify({
      training: { project: "training/local-runner" },
    })}\n`, "utf8");
    await assert.rejects(loadLocalRunnerConfig(configPath), /Unrecognized key.*training/);

    await writeFile(configPath, `${JSON.stringify({
      evaluation: { inference: { env: { CUSTOM: "1" } } },
    })}\n`, "utf8");
    await assert.rejects(loadLocalRunnerConfig(configPath), /Unrecognized key.*env/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("a cancellation remains terminal cancellation instead of becoming failure", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-cancellation-"));
  try {
    const request = requestFixture({
      runId: "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
    });
    const config = configFixture(root);
    const store = createLocalStore(config.storeRoot);
    const artifacts = resolveRunArtifacts({
      artifactRoot: config.artifactRoot,
      prefix: defaultArtifactPrefix({
        userId: request.user_id,
        behaviorSpecId: request.behavior_spec_id,
        runId: request.run_id,
      }),
    });
    await store.startRun({ request, artifactDir: artifacts.runDir });
    await store.cancelRun(request.run_id);

    await assert.rejects(
      runLocalFineTune({ request, config }),
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
