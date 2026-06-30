import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { localRunnerConfigSchema } from "../src/contracts.js";
import { evaluateExamples } from "../src/evaluation.js";

async function writeFakeEvaluator(root: string, actual: string): Promise<string> {
  const path = join(root, "fake-evaluate.py");
  await writeFile(path, `
import argparse, json
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
payload = json.load(open(args.input))
json.dump({
  "provider": "transformers",
  "model_id": payload["model_id"],
  "base_model": payload["base_model"],
  "adapter_path": payload.get("adapter_path"),
  "generation_config": payload["generation"],
  "results": [
    {
      "prompt": example["input"],
      "expected": example["output"],
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
      adapterPath: `file://${join(root, "adapter")}`,
      examples: [{ input: "Classify: good", output: "positive" }],
      system: "Return labels.",
      config,
      outputPath: join(root, "candidate-eval.json"),
    });

    assert.equal(report.inference_provider, "transformers");
    assert.equal(report.scoring_method, "exact_match");
    assert.equal(report.results[0]?.actual, "positive");
    assert.equal(report.avg_score, 1);
    assert.equal(report.exact_match_rate, 1);
  } finally {
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
