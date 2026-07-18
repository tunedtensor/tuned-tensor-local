import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { comparisonReportSchema, localRunnerConfigSchema } from "../src/contracts.js";
import { baselineCacheKey, buildJudgeMessages, classifyJudgeReasoning, compareEvalReports, deriveSampleSeed, evaluateExamples, rescoreEvalReport, sampleExamples, splitSpecExamples, tokenF1 } from "../src/evaluation.js";

async function writeFakeEvaluator(root: string, actual: string): Promise<string> {
  const path = join(root, "fake-evaluate.py");
await writeFile(path, `
import argparse, json
import sys
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
payload = json.load(open(args.input))
if payload.get("protocol_version") != 2:
    raise ValueError("expected inference protocol version 2")
if any("output" in example for example in payload["examples"]):
    raise ValueError("inference examples must not contain expected outputs")
print("fake evaluator started")
print("fake evaluator loading model", file=sys.stderr)
json.dump({
  "provider": "transformers",
  "model_id": payload["model_id"],
  "base_model": payload["base_model"],
  "adapter_path": payload.get("adapter_path"),
  "generation_config": payload["generation"],
  "results": [
    {
      "id": example["id"],
      "prompt": "forged prompt",
      "expected": "forged expected output",
      "actual": ${JSON.stringify(actual)},
      "latency_ms": 12
    }
    for example in payload["examples"]
  ]
}, open(args.output, "w"))
`, "utf8");
  return path;
}

test("transformers evaluation adapter records generated outputs and exact-match scores", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-test-"));
  try {
    const script = await writeFakeEvaluator(root, "positive");
    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: {
          provider: "transformers",
          script,
          maxNewTokens: 8,
          temperature: 0,
          topP: 1,
        },
        scoring: { mode: "exact_match" },
      },
    });
    const report = await evaluateExamples({
      kind: "candidate",
      modelId: `file://${join(root, "adapter")}`,
      baseModelId: "Qwen/Qwen3.5-2B",
      baseModelRevision: "0123456789abcdef",
      adapterPath: `file://${join(root, "adapter")}`,
      examples: [{ input: "Classify: good", output: "positive" }],
      system: "Return labels.",
      config,
      outputPath: join(root, "candidate-eval.json"),
    });

    assert.equal(report.inference_provider, "transformers");
    assert.equal(report.scoring_method, "exact_match");
    assert.equal(report.results[0]?.prompt, "Classify: good");
    assert.equal(report.results[0]?.expected, "positive");
    assert.equal(report.results[0]?.actual, "positive");
    assert.equal(report.avg_score, 1);
    assert.equal(report.exact_match_rate, 1);
    assert.ok(report.log_uri);
    const inferencePayload = JSON.parse(
      await readFile(`${join(root, "candidate-eval.json")}.inference-input.json`, "utf8"),
    ) as Record<string, unknown>;
    assert.equal(inferencePayload.protocol_version, 2);
    assert.equal(inferencePayload.base_model_revision, "0123456789abcdef");
    assert.equal("output" in (inferencePayload.examples as Array<Record<string, unknown>>)[0], false);
    const logText = await readFile(report.log_uri.replace(/^file:\/\//, ""), "utf8");
    assert.match(logText, /fake evaluator started/);
    assert.match(logText, /fake evaluator loading model/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("transformers evaluation inherits the configured HF_HOME hub layout", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-cache-env-test-"));
  try {
    const script = join(root, "cache-env-evaluate.py");
    await writeFile(script, `
import argparse, json, os
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
payload = json.load(open(args.input))
if any("output" in example for example in payload["examples"]):
    raise ValueError("inference examples must not contain expected outputs")
print(json.dumps({key: os.environ.get(key) for key in [
  "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"
]}))
json.dump({
  "provider": "transformers",
  "model_id": payload["model_id"],
  "results": [
    {"id": e["id"], "actual": "positive", "latency_ms": 1}
    for e in payload["examples"]
  ]
}, open(args.output, "w"))
`, "utf8");
    const modelCache = join(root, "huggingface");
    const outputPath = join(root, "baseline-eval.json");
    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      paths: { modelCache },
      evaluation: {
        inference: {
          provider: "transformers",
          script,
          env: {
            HF_HOME: "/wrong/home",
            HF_HUB_CACHE: "/wrong/hub",
            HUGGINGFACE_HUB_CACHE: "/wrong/legacy",
            TRANSFORMERS_CACHE: "/wrong/transformers",
            PYTORCH_TRANSFORMERS_CACHE: "/wrong/pytorch-transformers",
            PYTORCH_PRETRAINED_BERT_CACHE: "/wrong/pytorch-bert",
          },
        },
        scoring: { mode: "exact_match" },
      },
    });
    await evaluateExamples({
      kind: "baseline",
      modelId: "Qwen/Qwen3.5-2B",
      baseModelId: "Qwen/Qwen3.5-2B",
      examples: [{ input: "Classify: good", output: "positive" }],
      system: "Return labels.",
      config,
      outputPath,
    });

    const inferenceInput = JSON.parse(await readFile(`${outputPath}.inference-input.json`, "utf8"));
    assert.equal(inferenceInput.model_cache, modelCache);
    const log = await readFile(`${outputPath}.inference.log`, "utf8");
    const cacheEnv = JSON.parse(log.trim()) as Record<string, string>;
    assert.equal(cacheEnv.HF_HOME, modelCache);
    assert.equal(cacheEnv.HF_HUB_CACHE, join(modelCache, "hub"));
    assert.equal(cacheEnv.HUGGINGFACE_HUB_CACHE, join(modelCache, "hub"));
    assert.equal(cacheEnv.TRANSFORMERS_CACHE, null);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("command inference provider runs configured evaluator commands", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-command-eval-test-"));
  try {
    const command = [
      process.execPath,
      "-e",
      "let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>{const p=JSON.parse(d);"
        + "if(p.protocol_version!==2||Object.hasOwn(p,'expected'))process.exit(2);"
        + "process.stdout.write(JSON.stringify({actual:p.prompt.includes('good')?'positive':'negative'}))})",
    ];
    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: { provider: "command" },
        candidateCommand: command,
        scoring: { mode: "exact_match" },
      },
    });
    const report = await evaluateExamples({
      kind: "candidate",
      modelId: "local-command-model",
      examples: [{ input: "Classify: good", output: "positive" }],
      system: "Return labels.",
      config,
      outputPath: join(root, "candidate-eval.json"),
    });

    assert.equal(report.inference_provider, "command");
    assert.equal(report.scoring_method, "command");
    assert.equal(report.results[0]?.actual, "positive");
    assert.equal(report.results[0]?.scored_by, "exact_match");
    assert.equal(report.avg_score, 1);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("batch inference joins reordered predictions by opaque id", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-reordered-batch-test-"));
  try {
    const script = join(root, "reordered-batch.mjs");
    await writeFile(script, `
import { readFileSync, writeFileSync } from "node:fs";
const inputPath = process.argv[process.argv.indexOf("--input") + 1];
const outputPath = process.argv[process.argv.indexOf("--output") + 1];
const payload = JSON.parse(readFileSync(inputPath, "utf8"));
if (payload.examples.some((example) => Object.hasOwn(example, "output"))) process.exit(2);
writeFileSync(outputPath, JSON.stringify({
  results: [...payload.examples].reverse().map((example) => ({
    id: example.id,
    actual: example.input === "first" ? "one" : "two",
    latency_ms: 1
  }))
}));
`, "utf8");
    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      evaluation: {
        inference: {
          provider: "batch_command",
          command: [process.execPath, script],
        },
        scoring: { mode: "exact_match" },
      },
    });

    const report = await evaluateExamples({
      kind: "candidate",
      modelId: "external:test",
      baseModelId: "external:test",
      examples: [
        { input: "first", output: "one" },
        { input: "second", output: "two" },
      ],
      system: "Predict.",
      config,
      outputPath: join(root, "candidate-eval.json"),
    });

    assert.deepEqual(report.results.map((result) => result.actual), ["one", "two"]);
    assert.equal(report.avg_score, 1);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("batch inference rejects incomplete or malformed prediction output", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-short-batch-test-"));
  try {
    const script = join(root, "short-batch.mjs");
    await writeFile(script, `
import { readFileSync, writeFileSync } from "node:fs";
const inputPath = process.argv[process.argv.indexOf("--input") + 1];
const outputPath = process.argv[process.argv.indexOf("--output") + 1];
const payload = JSON.parse(readFileSync(inputPath, "utf8"));
if (payload.examples.some((example) => Object.hasOwn(example, "output"))) process.exit(2);
writeFileSync(outputPath, JSON.stringify({
  results: [{ id: payload.examples[0].id, actual: "first", latency_ms: 1 }]
}));
`, "utf8");
    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      evaluation: {
        inference: {
          provider: "batch_command",
          command: [process.execPath, script],
        },
        scoring: { mode: "exact_match" },
      },
    });

    await assert.rejects(
      evaluateExamples({
        kind: "candidate",
        modelId: "external:test",
        baseModelId: "external:test",
        examples: [
          { input: "first", output: "one" },
          { input: "second", output: "two" },
        ],
        system: "Predict.",
        config,
        outputPath: join(root, "candidate-eval.json"),
      }),
      /returned 1 prediction; expected 2/,
    );

    await writeFile(script, `
import { writeFileSync } from "node:fs";
const outputPath = process.argv[process.argv.indexOf("--output") + 1];
writeFileSync(outputPath, JSON.stringify({
  results: [{ id: "0", actual: 42, latency_ms: 1 }]
}));
`, "utf8");
    await assert.rejects(
      evaluateExamples({
        kind: "candidate",
        modelId: "external:test",
        baseModelId: "external:test",
        examples: [{ input: "first", output: "one" }],
        system: "Predict.",
        config,
        outputPath: join(root, "malformed-candidate-eval.json"),
      }),
      /prediction 0 must include string actual/,
    );

    const staleOutputPath = join(root, "stale-candidate-eval.json");
    await writeFile(`${staleOutputPath}.inference-output.json`, JSON.stringify({
      results: [{ id: "0", actual: "stale", latency_ms: 1 }],
    }), "utf8");
    await writeFile(script, "process.exit(0);\n", "utf8");
    await assert.rejects(
      evaluateExamples({
        kind: "candidate",
        modelId: "external:test",
        baseModelId: "external:test",
        examples: [{ input: "first", output: "one" }],
        system: "Predict.",
        config,
        outputPath: staleOutputPath,
      }),
      /did not write valid JSON output/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("transformers evaluation payload includes image-text loader for multimodal models", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-mm-test-"));
  try {
    const script = await writeFakeEvaluator(root, "42");
    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: {
          provider: "transformers",
          script,
          maxNewTokens: 8,
          temperature: 0,
          topP: 1,
        },
        scoring: { mode: "exact_match" },
      },
    });
    await evaluateExamples({
      kind: "baseline",
      modelId: "Qwen/Qwen3-VL-2B-Instruct",
      baseModelId: "Qwen/Qwen3-VL-2B-Instruct",
      examples: [{
        input: "What is the blue value?",
        output: "42",
        input_assets: [{ type: "image", image: "charts/example.png" }],
        modality: "document_ocr",
      }],
      system: "Read charts.",
      config,
      outputPath: join(root, "baseline-eval.json"),
    });

    const payload = JSON.parse(await readFile(join(root, "baseline-eval.json.inference-input.json"), "utf8"));
    assert.equal(payload.protocol_version, 2);
    assert.equal(payload.model_loader, "image_text_to_text");
    assert.deepEqual(payload.examples[0].input_assets, [{ type: "image", image: "charts/example.png" }]);
    assert.equal(payload.examples[0].modality, "document_ocr");
    assert.equal("output" in payload.examples[0], false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("transformers evaluation can score structured JSON fields", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-json-fields-test-"));
  try {
    const script = await writeFakeEvaluator(
      root,
      "{\"triage\":\"reply\",\"priority\":\"low\",\"should_process\":true,\"summary\":\"Different words\",\"reason\":\"Different reason\"}",
    );
    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: {
          provider: "transformers",
          script,
          maxNewTokens: 8,
          temperature: 0,
          topP: 1,
        },
        scoring: {
          mode: "json_fields",
          fields: ["triage", "priority", "should_process"],
        },
      },
    });
    const report = await evaluateExamples({
      kind: "candidate",
      modelId: `file://${join(root, "adapter")}`,
      baseModelId: "Qwen/Qwen3.5-2B",
      adapterPath: `file://${join(root, "adapter")}`,
      examples: [{
        input: "Classify: urgent reply",
        output: "{\"triage\":\"reply\",\"priority\":\"normal\",\"should_process\":true,\"summary\":\"Sender asks for a reply.\",\"reason\":\"Direct request.\"}",
      }],
      system: "Return labels.",
      config,
      outputPath: join(root, "candidate-eval.json"),
    });

    assert.equal(report.scoring_method, "json_fields");
    assert.equal(report.scoring_mode, "json_fields");
    assert.equal(report.results[0]?.score, 2 / 3);
    assert.equal(report.results[0]?.passed, false);
    assert.equal(report.exact_match_rate, 0);
    assert.equal(report.json_field_metrics?.valid_json_rate, 1);
    assert.equal(report.json_field_metrics?.schema_match_rate, 1);
    assert.deepEqual(report.json_field_metrics?.field_accuracy.triage, {
      correct: 1,
      total: 1,
      accuracy: 1,
    });
    assert.deepEqual(report.json_field_metrics?.field_accuracy.priority, {
      correct: 0,
      total: 1,
      accuracy: 0,
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("sampleExamples takes a deterministic seeded sample preserving original order", () => {
  const examples = ["a", "b", "c", "d", "e", "f", "g", "h"];
  const first = sampleExamples(examples, 3, 42);
  const second = sampleExamples(examples, 3, 42);
  assert.deepEqual(first, second);
  assert.equal(first.length, 3);
  const positions = first.map((item) => examples.indexOf(item));
  assert.deepEqual(positions, [...positions].sort((left, right) => left - right));
  assert.deepEqual(sampleExamples(examples, 8, 42), examples);
  assert.deepEqual(sampleExamples(examples, 20, 42), examples);
  assert.equal(deriveSampleSeed("run-a"), deriveSampleSeed("run-a"));
  assert.notEqual(deriveSampleSeed("run-a"), deriveSampleSeed("run-b"));
});

test("splitSpecExamples deterministically holds out ~20% with min 1 train and 1 holdout", () => {
  const examples = Array.from({ length: 10 }, (_, index) => `example ${index}`);
  const first = splitSpecExamples(examples, 42);
  const second = splitSpecExamples(examples, 42);
  assert.deepEqual(first, second);
  assert.equal(first.holdout.length, 2);
  assert.equal(first.train.length, 8);
  // Splits are disjoint and cover all examples, preserving original order.
  assert.deepEqual([...first.train, ...first.holdout].sort(), [...examples].sort());
  for (const item of first.holdout) assert.ok(!first.train.includes(item));
  const trainPositions = first.train.map((item) => examples.indexOf(item));
  assert.deepEqual(trainPositions, [...trainPositions].sort((left, right) => left - right));

  // Different seeds can produce different holdouts.
  assert.notDeepEqual(splitSpecExamples(examples, 1).holdout, splitSpecExamples(examples, 7).holdout);

  // Two examples: one train, one holdout.
  const pair = splitSpecExamples(["a", "b"], 5);
  assert.equal(pair.train.length, 1);
  assert.equal(pair.holdout.length, 1);

  // Fewer than 2 examples: no holdout.
  assert.deepEqual(splitSpecExamples(["only"], 5), { train: ["only"], holdout: [] });
  assert.deepEqual(splitSpecExamples([], 5), { train: [], holdout: [] });
});

test("evaluation records eval split and seeded sample when maxExamples truncates", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-sample-test-"));
  try {
    const config = localRunnerConfigSchema.parse({
      dryRun: true,
      evaluation: {
        scoring: { mode: "exact_match" },
        maxExamples: 2,
      },
    });
    const examples = Array.from({ length: 6 }, (_, index) => ({
      input: `input ${index}`,
      output: `output ${index}`,
    }));
    const shared = {
      examples,
      system: "Return labels.",
      config,
      evalSplit: "prebuilt_test" as const,
      sampleSeed: 1234,
    };
    const baseline = await evaluateExamples({
      ...shared,
      kind: "baseline",
      modelId: "Qwen/Qwen3.5-2B",
      outputPath: join(root, "baseline-eval.json"),
    });
    const candidate = await evaluateExamples({
      ...shared,
      kind: "candidate",
      modelId: "local-adapter",
      outputPath: join(root, "candidate-eval.json"),
    });

    assert.equal(baseline.eval_truncated, true);
    assert.equal(baseline.eval_examples_used, 2);
    assert.equal(baseline.eval_examples_total, 6);
    assert.equal(baseline.eval_split, "prebuilt_test");
    assert.equal(baseline.eval_sample_seed, 1234);
    assert.deepEqual(
      baseline.results.map((result) => result.prompt),
      candidate.results.map((result) => result.prompt),
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("evaluation reports no sample seed when all examples are used", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-noseed-test-"));
  try {
    const config = localRunnerConfigSchema.parse({
      dryRun: true,
      evaluation: { scoring: { mode: "exact_match" } },
    });
    const report = await evaluateExamples({
      kind: "baseline",
      modelId: "Qwen/Qwen3.5-2B",
      examples: [{ input: "Classify: good", output: "positive" }],
      system: "Return labels.",
      config,
      outputPath: join(root, "baseline-eval.json"),
      evalSplit: "spec_examples",
      sampleSeed: 99,
    });
    assert.equal(report.eval_truncated, false);
    assert.equal(report.eval_sample_seed, null);
    assert.equal(report.eval_split, "spec_examples");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("json_fields scoring does not credit configured fields missing from expected output", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-json-missing-test-"));
  try {
    const script = await writeFakeEvaluator(root, "{\"triage\":\"reply\"}");
    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: { provider: "transformers", script },
        scoring: {
          mode: "json_fields",
          fields: ["triage", "not_in_expected"],
        },
      },
    });
    const report = await evaluateExamples({
      kind: "candidate",
      modelId: "Qwen/Qwen3.5-2B",
      baseModelId: "Qwen/Qwen3.5-2B",
      examples: [{ input: "Classify: urgent reply", output: "{\"triage\":\"reply\"}" }],
      system: "Return labels.",
      config,
      outputPath: join(root, "candidate-eval.json"),
    });

    assert.equal(report.results[0]?.score, 0.5);
    assert.equal(report.results[0]?.passed, false);
    assert.match(report.results[0]?.reasoning ?? "", /missing from expected output/);
    assert.match(report.results[0]?.reasoning ?? "", /not_in_expected/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("malformed judge output falls back to exact match without failing the run", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-judge-malformed-test-"));
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.OPENROUTER_API_KEY;
  try {
    const script = await writeFakeEvaluator(root, "positive");
    process.env.OPENROUTER_API_KEY = "test-key";
    globalThis.fetch = (async () => new Response(JSON.stringify({
      model: "openai/gpt-5.5",
      choices: [{ message: { content: "sorry, I cannot produce JSON right now" } }],
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })) as typeof fetch;

    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: { provider: "transformers", script },
        scoring: { mode: "llm_judge", fallback: "exact_match" },
      },
      llm: { provider: "openrouter", model: "openai/gpt-5.5", apiKeyEnv: "OPENROUTER_API_KEY" },
    });
    const report = await evaluateExamples({
      kind: "baseline",
      modelId: "Qwen/Qwen3.5-2B",
      baseModelId: "Qwen/Qwen3.5-2B",
      examples: [{ input: "Classify: good", output: "positive" }],
      system: "Return labels.",
      config,
      outputPath: join(root, "baseline-eval.json"),
    });

    assert.equal(report.results[0]?.score, 1);
    assert.equal(report.results[0]?.passed, true);
    assert.match(report.results[0]?.reasoning ?? "", /LLM judge failed/);
    assert.equal(report.judge_model_id, null);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) {
      delete process.env.OPENROUTER_API_KEY;
    } else {
      process.env.OPENROUTER_API_KEY = originalKey;
    }
    await rm(root, { recursive: true, force: true });
  }
});

test("judge request failure falls back to exact match with fallback=exact_match", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-judge-error-test-"));
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.OPENROUTER_API_KEY;
  try {
    const script = await writeFakeEvaluator(root, "negative");
    process.env.OPENROUTER_API_KEY = "test-key";
    globalThis.fetch = (async () => {
      throw new Error("network unreachable");
    }) as typeof fetch;

    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: { provider: "transformers", script },
        scoring: { mode: "llm_judge", fallback: "exact_match" },
      },
      llm: { provider: "openrouter", model: "openai/gpt-5.5", apiKeyEnv: "OPENROUTER_API_KEY" },
    });
    const report = await evaluateExamples({
      kind: "baseline",
      modelId: "Qwen/Qwen3.5-2B",
      baseModelId: "Qwen/Qwen3.5-2B",
      examples: [{ input: "Classify: good", output: "positive" }],
      system: "Return labels.",
      config,
      outputPath: join(root, "baseline-eval.json"),
    });

    assert.equal(report.results[0]?.score, 0);
    assert.equal(report.results[0]?.passed, false);
    assert.match(report.results[0]?.reasoning ?? "", /LLM judge failed/);
    assert.equal(report.results[0]?.scored_by, "exact_match_fallback");
    assert.equal(report.fallback_scored_count, 1);
    assert.equal(report.judge_scored_count, 0);
    assert.equal(report.judge_only_avg_score, null);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) {
      delete process.env.OPENROUTER_API_KEY;
    } else {
      process.env.OPENROUTER_API_KEY = originalKey;
    }
    await rm(root, { recursive: true, force: true });
  }
});

test("malformed judge output fails the run when fallback=fail", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-judge-fail-test-"));
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.OPENROUTER_API_KEY;
  try {
    const script = await writeFakeEvaluator(root, "positive");
    process.env.OPENROUTER_API_KEY = "test-key";
    globalThis.fetch = (async () => new Response(JSON.stringify({
      model: "openai/gpt-5.5",
      choices: [{ message: { content: "not json" } }],
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })) as typeof fetch;

    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: { provider: "transformers", script },
        scoring: { mode: "llm_judge", fallback: "fail" },
      },
      llm: { provider: "openrouter", model: "openai/gpt-5.5", apiKeyEnv: "OPENROUTER_API_KEY" },
    });
    await assert.rejects(evaluateExamples({
      kind: "baseline",
      modelId: "Qwen/Qwen3.5-2B",
      baseModelId: "Qwen/Qwen3.5-2B",
      examples: [{ input: "Classify: good", output: "positive" }],
      system: "Return labels.",
      config,
      outputPath: join(root, "baseline-eval.json"),
    }), /malformed JSON/);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) {
      delete process.env.OPENROUTER_API_KEY;
    } else {
      process.env.OPENROUTER_API_KEY = originalKey;
    }
    await rm(root, { recursive: true, force: true });
  }
});

test("transformers evaluation can score generated outputs with OpenRouter judge", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-judge-test-"));
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.OPENROUTER_API_KEY;
  try {
    const script = await writeFakeEvaluator(root, "mostly positive");
    process.env.OPENROUTER_API_KEY = "test-key";
    globalThis.fetch = (async () => new Response(JSON.stringify({
      model: "openai/gpt-5.5",
      choices: [{ message: { content: "{\"score\":0.75,\"passed\":true,\"reasoning\":\"close enough\"}" } }],
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })) as typeof fetch;

    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: {
          provider: "transformers",
          script,
          maxNewTokens: 8,
          temperature: 0,
          topP: 1,
        },
        scoring: { mode: "llm_judge", fallback: "fail" },
      },
      llm: {
        provider: "openrouter",
        model: "openai/gpt-5.5",
        apiKeyEnv: "OPENROUTER_API_KEY",
      },
    });
    const report = await evaluateExamples({
      kind: "baseline",
      modelId: "Qwen/Qwen3.5-2B",
      baseModelId: "Qwen/Qwen3.5-2B",
      examples: [{ input: "Classify: good", output: "positive" }],
      system: "Return labels.",
      config,
      outputPath: join(root, "baseline-eval.json"),
    });

    assert.equal(report.inference_provider, "transformers");
    assert.equal(report.scoring_method, "llm_judge");
    assert.equal(report.judge_model_id, "openai/gpt-5.5");
    assert.equal(report.results[0]?.score, 0.75);
    assert.equal(report.results[0]?.passed, true);
    assert.equal(report.exact_match_rate, 0);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) {
      delete process.env.OPENROUTER_API_KEY;
    } else {
      process.env.OPENROUTER_API_KEY = originalKey;
    }
    await rm(root, { recursive: true, force: true });
  }
});

test("rescoreEvalReport reuses generated outputs and rewrites scores", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-rescore-test-"));
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.OPENROUTER_API_KEY;
  try {
    process.env.OPENROUTER_API_KEY = "test-key";
    globalThis.fetch = (async () => new Response(JSON.stringify({
      model: "openai/gpt-5.5",
      choices: [{ message: { content: "{\"score\":0.25,\"passed\":false,\"reasoning\":\"wrong label\"}" } }],
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })) as typeof fetch;

    const exactConfig = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: { provider: "none" },
        scoring: { mode: "exact_match" },
      },
    });
    const original = await evaluateExamples({
      kind: "candidate",
      modelId: "local-model",
      examples: [{ input: "Classify: good", output: "positive" }],
      system: "Return labels.",
      config: exactConfig,
      outputPath: join(root, "candidate-eval.json"),
    });
    const withActual = {
      ...original,
      results: [{ ...original.results[0]!, actual: "mostly positive", latency_ms: 7 }],
    };

    const judgeConfig = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: { provider: "none" },
        scoring: { mode: "llm_judge", fallback: "fail" },
      },
      llm: { provider: "openrouter", model: "openai/gpt-5.5", apiKeyEnv: "OPENROUTER_API_KEY" },
    });
    const rescored = await rescoreEvalReport({
      report: withActual,
      config: judgeConfig,
      outputPath: join(root, "candidate-eval.json"),
      system: "Return labels.",
    });

    assert.equal(rescored.results[0]?.actual, "mostly positive");
    assert.equal(rescored.results[0]?.score, 0.25);
    assert.equal(rescored.results[0]?.scored_by, "llm_judge");
    assert.equal(rescored.judge_model_id, "openai/gpt-5.5");
    const persisted = JSON.parse(await readFile(join(root, "candidate-eval.json"), "utf8"));
    assert.equal(persisted.results[0].actual, "mostly positive");
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) {
      delete process.env.OPENROUTER_API_KEY;
    } else {
      process.env.OPENROUTER_API_KEY = originalKey;
    }
    await rm(root, { recursive: true, force: true });
  }
});

test("tokenF1 scores bag-of-words overlap", () => {
  assert.equal(tokenF1("Kelly and Mary will wear red dresses.", "Kelly and Mary will wear red dresses."), 1);
  assert.equal(tokenF1("alpha beta", "gamma delta"), 0);
  assert.equal(tokenF1("", ""), 1);
  assert.equal(tokenF1("alpha", ""), 0);
  // 2-token overlap of 4 expected + 2 actual tokens: p=1, r=0.5, f1=2/3.
  const partial = tokenF1("alpha beta gamma delta", "Alpha beta");
  assert.ok(Math.abs(partial - 2 / 3) < 1e-9);
  // Repeated tokens only match as often as they appear in the expected text.
  assert.ok(tokenF1("alpha beta", "alpha alpha alpha") < 1);
});

test("buildJudgeMessages forwards task instructions to the judge", () => {
  const messages = buildJudgeMessages({
    prompt: "Summarize this conversation.",
    expected: "A short summary.",
    actual: "Another summary.",
    taskInstructions: "Write one concise sentence.",
  });
  assert.equal(messages.length, 2);
  const payload = JSON.parse(messages[1].content) as Record<string, unknown>;
  assert.equal(payload.task_instructions, "Write one concise sentence.");
  assert.equal(payload.prompt, "Summarize this conversation.");
  assert.equal(payload.expected, "A short summary.");
  assert.equal(payload.actual, "Another summary.");

  const withoutInstructions = buildJudgeMessages({
    prompt: "p",
    expected: "e",
    actual: "a",
  });
  const bare = JSON.parse(withoutInstructions[1].content) as Record<string, unknown>;
  assert.equal("task_instructions" in bare, false);
});

async function writeCountingEvaluator(root: string, actual: string): Promise<{ script: string; countPath: string }> {
  const script = join(root, "counting-evaluate.py");
  const countPath = join(root, "invocations.txt");
  await writeFile(script, `
import argparse, json
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
payload = json.load(open(args.input))
if any("output" in example for example in payload["examples"]):
    raise ValueError("inference examples must not contain expected outputs")
with open(${JSON.stringify("COUNT_PATH")}, "a") as fh:
    fh.write("x")
json.dump({
  "provider": "transformers",
  "model_id": payload["model_id"],
  "results": [
    {"id": e["id"], "actual": ${JSON.stringify("ACTUAL")}, "latency_ms": 5}
    for e in payload["examples"]
  ]
}, open(args.output, "w"))
`.replace('"COUNT_PATH"', JSON.stringify(countPath)).replace('"ACTUAL"', JSON.stringify(actual)), "utf8");
  return { script, countPath };
}

test("default scoring is runnable exact_match while explicit llm_judge defaults to fail", () => {
  const config = localRunnerConfigSchema.parse({});
  assert.equal(config.evaluation.scoring.mode, "exact_match");
  assert.equal(config.evaluation.scoring.fallback, "fail");

  const judgeConfig = localRunnerConfigSchema.parse({
    evaluation: { scoring: { mode: "llm_judge" } },
  });
  assert.equal(judgeConfig.evaluation.scoring.mode, "llm_judge");
  assert.equal(judgeConfig.evaluation.scoring.fallback, "fail");
});

test("llm_judge missing config or API key fails before inference", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-judge-preflight-test-"));
  const originalKey = process.env.TT_LOCAL_MISSING_JUDGE_KEY;
  try {
    delete process.env.TT_LOCAL_MISSING_JUDGE_KEY;
    const { script, countPath } = await writeCountingEvaluator(root, "positive");
    const baseArgs = {
      kind: "baseline" as const,
      modelId: "Qwen/Qwen3.5-2B",
      baseModelId: "Qwen/Qwen3.5-2B",
      examples: [{ input: "Classify: good", output: "positive" }],
      system: "Return labels.",
    };
    const missingConfig = localRunnerConfigSchema.parse({
      dryRun: false,
      evaluation: {
        inference: { provider: "transformers", script },
        scoring: { mode: "llm_judge" },
      },
    });
    await assert.rejects(evaluateExamples({
      ...baseArgs,
      config: missingConfig,
      outputPath: join(root, "missing-config.json"),
    }), /cannot start because the llm OpenRouter configuration is missing/);

    const missingKey = localRunnerConfigSchema.parse({
      dryRun: false,
      evaluation: {
        inference: { provider: "transformers", script },
        scoring: { mode: "llm_judge" },
      },
      llm: {
        provider: "openrouter",
        model: "openai/gpt-5.5",
        apiKeyEnv: "TT_LOCAL_MISSING_JUDGE_KEY",
      },
    });
    await assert.rejects(evaluateExamples({
      ...baseArgs,
      config: missingKey,
      outputPath: join(root, "missing-key.json"),
    }), /TT_LOCAL_MISSING_JUDGE_KEY is not set/);

    await assert.rejects(readFile(countPath, "utf8"), /ENOENT/, "preflight must run before model inference");
  } finally {
    if (originalKey === undefined) {
      delete process.env.TT_LOCAL_MISSING_JUDGE_KEY;
    } else {
      process.env.TT_LOCAL_MISSING_JUDGE_KEY = originalKey;
    }
    await rm(root, { recursive: true, force: true });
  }
});

test("explicit judge fallback is marked and never enters the baseline cache", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-judge-unavailable-cache-test-"));
  try {
    const { script, countPath } = await writeCountingEvaluator(root, "positive");
    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: { provider: "transformers", script },
        scoring: { mode: "llm_judge", fallback: "exact_match" },
      },
    });
    const args = {
      kind: "baseline" as const,
      modelId: "Qwen/Qwen3.5-2B",
      baseModelId: "Qwen/Qwen3.5-2B",
      baseModelRevision: "revision-test",
      examples: [{ input: "Classify: good", output: "positive" }],
      system: "Return labels.",
      config,
    };
    const first = await evaluateExamples({ ...args, outputPath: join(root, "baseline-1.json") });
    const second = await evaluateExamples({ ...args, outputPath: join(root, "baseline-2.json") });

    for (const report of [first, second]) {
      assert.notEqual(report.cached, true);
      assert.equal(report.results[0]?.scored_by, "exact_match_fallback");
      assert.match(report.results[0]?.reasoning ?? "", /LLM judge unavailable/);
      assert.equal(report.fallback_scored_count, 1);
      assert.equal(report.judge_scored_count, 0);
    }
    assert.equal((await readFile(countPath, "utf8")).length, 2, "fallback result must not suppress later inference");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("baseline cache key includes judge configuration and key availability", () => {
  const originalKey = process.env.TT_LOCAL_CACHE_JUDGE_KEY;
  try {
    delete process.env.TT_LOCAL_CACHE_JUDGE_KEY;
    const config = localRunnerConfigSchema.parse({
      evaluation: {
        scoring: { mode: "llm_judge", fallback: "exact_match" },
      },
      llm: {
        provider: "openrouter",
        model: "openai/gpt-5.5",
        apiKeyEnv: "TT_LOCAL_CACHE_JUDGE_KEY",
      },
    });
    const args = {
      modelId: "Qwen/Qwen3.5-2B",
      system: "Return labels.",
      examples: [{ input: "Classify: good", output: "positive" }],
      config,
      packageVersion: "test",
    };
    const unavailable = baselineCacheKey(args);
    process.env.TT_LOCAL_CACHE_JUDGE_KEY = "secret-not-hashed";
    const available = baselineCacheKey(args);
    assert.notEqual(available, unavailable);

    const otherModel = baselineCacheKey({
      ...args,
      config: localRunnerConfigSchema.parse({
        evaluation: {
          scoring: { mode: "llm_judge", fallback: "exact_match" },
        },
        llm: {
          provider: "openrouter",
          model: "anthropic/claude-sonnet-4",
          apiKeyEnv: "TT_LOCAL_CACHE_JUDGE_KEY",
        },
      }),
    });
    assert.notEqual(otherModel, available);
    assert.notEqual(
      baselineCacheKey({ ...args, baseModelRevision: "revision-a" }),
      baselineCacheKey({ ...args, baseModelRevision: "revision-b" }),
    );
    assert.notEqual(
      baselineCacheKey({ ...args, sourceFingerprint: "source-a" }),
      baselineCacheKey({ ...args, sourceFingerprint: "source-b" }),
    );
  } finally {
    if (originalKey === undefined) {
      delete process.env.TT_LOCAL_CACHE_JUDGE_KEY;
    } else {
      process.env.TT_LOCAL_CACHE_JUDGE_KEY = originalKey;
    }
  }
});

test("transient judge fallback is retried instead of cached as a judge result", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-judge-transient-cache-test-"));
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.TT_LOCAL_TRANSIENT_JUDGE_KEY;
  let fetchCalls = 0;
  try {
    process.env.TT_LOCAL_TRANSIENT_JUDGE_KEY = "test-key";
    globalThis.fetch = (async () => {
      fetchCalls += 1;
      if (fetchCalls === 1) throw new Error("temporary judge outage");
      return new Response(JSON.stringify({
        model: "openai/gpt-5.5",
        choices: [{ message: { content: "{\"score\":0.8,\"passed\":true,\"reasoning\":\"good\"}" } }],
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }) as typeof fetch;
    const { script, countPath } = await writeCountingEvaluator(root, "mostly positive");
    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: { provider: "transformers", script },
        scoring: { mode: "llm_judge", fallback: "exact_match" },
      },
      llm: {
        provider: "openrouter",
        model: "openai/gpt-5.5",
        apiKeyEnv: "TT_LOCAL_TRANSIENT_JUDGE_KEY",
      },
    });
    const args = {
      kind: "baseline" as const,
      modelId: "Qwen/Qwen3.5-2B",
      baseModelId: "Qwen/Qwen3.5-2B",
      baseModelRevision: "revision-test",
      examples: [{ input: "Classify: good", output: "positive" }],
      system: "Return labels.",
      config,
    };
    const first = await evaluateExamples({ ...args, outputPath: join(root, "baseline-1.json") });
    const second = await evaluateExamples({ ...args, outputPath: join(root, "baseline-2.json") });

    assert.equal(first.results[0]?.scored_by, "exact_match_fallback");
    assert.notEqual(second.cached, true);
    assert.equal(second.results[0]?.scored_by, "llm_judge");
    assert.equal(second.results[0]?.score, 0.8);
    assert.equal((await readFile(countPath, "utf8")).length, 2);
    assert.equal(fetchCalls, 2);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) {
      delete process.env.TT_LOCAL_TRANSIENT_JUDGE_KEY;
    } else {
      process.env.TT_LOCAL_TRANSIENT_JUDGE_KEY = originalKey;
    }
    await rm(root, { recursive: true, force: true });
  }
});

test("baseline evaluation cache reuses the prior report for identical inputs", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-eval-cache-test-"));
  try {
    const { script, countPath } = await writeCountingEvaluator(root, "positive");
    const config = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: { provider: "transformers", script },
        scoring: { mode: "exact_match" },
      },
    });
    const args = {
      kind: "baseline" as const,
      modelId: "Qwen/Qwen3.5-2B",
      baseModelId: "Qwen/Qwen3.5-2B",
      baseModelRevision: "revision-test",
      examples: [{ input: "Classify: good", output: "positive" }],
      system: "Return labels.",
      config,
      sourceFingerprint: "source-a",
    };
    const first = await evaluateExamples({ ...args, outputPath: join(root, "baseline-1.json") });
    assert.notEqual(first.cached, true);
    assert.equal(first.avg_score, 1);

    const second = await evaluateExamples({ ...args, outputPath: join(root, "baseline-2.json") });
    assert.equal(second.cached, true);
    assert.equal(second.avg_score, first.avg_score);
    assert.deepEqual(second.results, first.results);
    assert.ok(second.cache_key);
    assert.equal((await readFile(countPath, "utf8")).length, 1, "evaluator must run only once");

    const changedSource = await evaluateExamples({
      ...args,
      sourceFingerprint: "source-b",
      outputPath: join(root, "baseline-changed-source.json"),
    });
    assert.notEqual(changedSource.cached, true);
    assert.equal((await readFile(countPath, "utf8")).length, 2, "changed input bytes must bypass the old cache key");

    // The candidate kind and disabled cache both bypass the cache.
    const noCacheConfig = localRunnerConfigSchema.parse({
      dryRun: false,
      storeRoot: join(root, "store"),
      evaluation: {
        inference: { provider: "transformers", script },
        scoring: { mode: "exact_match" },
        baselineCache: false,
      },
    });
    const third = await evaluateExamples({ ...args, config: noCacheConfig, outputPath: join(root, "baseline-3.json") });
    assert.notEqual(third.cached, true);
    assert.equal((await readFile(countPath, "utf8")).length, 3);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("classifyJudgeReasoning categorizes regression reasonings", () => {
  assert.equal(classifyJudgeReasoning("It incorrectly states that Meghan will wear a dress."), "factual");
  assert.equal(classifyJudgeReasoning("The summary omits the PDF detail."), "omission");
  assert.equal(classifyJudgeReasoning("The answer is too verbose for the requested style."), "style");
  assert.equal(classifyJudgeReasoning("LLM judge failed (fetch failed); scored by normalized exact match."), "fallback");
  assert.equal(classifyJudgeReasoning("Close but imperfect.", "exact_match_fallback"), "fallback");
  assert.equal(classifyJudgeReasoning(null), "other");
  // Factual signals dominate omission signals when both appear.
  assert.equal(classifyJudgeReasoning("It omits a detail and misstates the outcome."), "factual");
});

test("comparison with partial-category regressions passes schema validation", () => {
  // Regression test: zod v4 enum-keyed records are exhaustive, so a taxonomy
  // containing only the categories that occurred failed runReportSchema
  // validation at the end of an otherwise-successful run.
  const baseline = {
    kind: "baseline" as const,
    model_id: "m",
    total: 1,
    eval_examples_total: 1,
    eval_examples_used: 1,
    eval_truncated: false,
    avg_score: 0.9,
    pass_rate: 1,
    exact_match_rate: 0,
    avg_latency_ms: 10,
    results: [{
      prompt: "p1",
      expected: "e",
      actual: "a",
      passed: true,
      score: 0.9,
      reasoning: "good",
      latency_ms: 10,
      scored_by: "llm_judge" as const,
    }],
    artifact_uri: "file:///tmp/b.json",
    scoring_method: "llm_judge" as const,
  };
  const candidate = {
    ...baseline,
    kind: "candidate" as const,
    avg_score: 0.4,
    results: [{
      ...baseline.results[0],
      score: 0.4,
      reasoning: "The summary omits the key detail.",
    }],
  };
  const comparison = compareEvalReports(baseline, candidate);
  assert.equal(comparison.regressions, 1);
  assert.equal(comparison.regression_taxonomy?.omission, 1);
  assert.equal(comparison.regression_taxonomy?.factual, 0);
  const parsed = comparisonReportSchema.parse(comparison);
  assert.equal(parsed.regression_taxonomy?.omission, 1);
});
