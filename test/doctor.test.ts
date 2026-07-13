import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fineTuneRunRequestSchema, localRunnerConfigSchema } from "../src/contracts.js";
import { buildDoctorPythonPlans, runDoctor } from "../src/doctor.js";

test("doctor uses the configured uv projects and one HF_HOME contract", () => {
  const config = localRunnerConfigSchema.parse({
    paths: { modelCache: "/tmp/tt-hf-home" },
    training: { project: "training/local-runner" },
    evaluation: {
      inference: {
        provider: "transformers",
        project: "training/local-runner",
        device: "cuda",
      },
      scoring: { mode: "exact_match" },
    },
  });

  const plans = buildDoctorPythonPlans(config);
  assert.equal(plans.length, 2);
  for (const plan of plans) {
    assert.equal(plan.command, "uv");
    assert.ok(plan.args.includes("--project"));
    assert.ok(plan.args.some((value) => value.endsWith("training/local-runner")));
    assert.equal(plan.env.HF_HOME, "/tmp/tt-hf-home");
    assert.match(plan.args.at(-1) ?? "", /torch\.cuda\.is_available/);
  }
});

test("dry-run doctor skips Python and NVIDIA requirements but checks storage", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-doctor-"));
  try {
    const config = localRunnerConfigSchema.parse({
      dryRun: true,
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      paths: { modelCache: join(root, "cache") },
      evaluation: {
        inference: { provider: "none", device: "cpu" },
        scoring: { mode: "exact_match" },
      },
    });
    const checks = await runDoctor(config);
    assert.equal(checks.find((check) => check.name === "python-runtime")?.ok, true);
    assert.match(checks.find((check) => check.name === "python-runtime")?.message ?? "", /dryRun/);
    assert.equal(checks.find((check) => check.name === "nvidia-smi")?.ok, true);
    assert.equal(checks.filter((check) => check.name.endsWith("root")).every((check) => check.ok), true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("doctor rejects an unchanged generated placeholder spec", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-doctor-placeholder-"));
  try {
    const config = localRunnerConfigSchema.parse({
      dryRun: true,
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      paths: { modelCache: join(root, "cache") },
      evaluation: {
        inference: { provider: "none", device: "cpu" },
        scoring: { mode: "exact_match" },
      },
    });
    const request = fineTuneRunRequestSchema.parse({
      run_id: "11111111-1111-4111-8111-111111111111",
      user_id: "local-user",
      behavior_spec_id: "22222222-2222-4222-8222-222222222222",
      run_number: 1,
      spec_snapshot: {
        name: "Placeholder",
        base_model: "Qwen/Qwen3.5-2B",
        system_prompt: "Describe the behavior this local model should learn.",
        examples: [{
          input: "Replace this with a representative input.",
          output: "Replace this with the expected output.",
        }],
      },
    });
    const checks = await runDoctor(config, request);
    assert.equal(checks.find((check) => check.name === "spec-content")?.ok, false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("doctor requires HF_TOKEN for gated base models", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-doctor-gated-"));
  const previous = process.env.HF_TOKEN;
  delete process.env.HF_TOKEN;
  try {
    const config = localRunnerConfigSchema.parse({
      dryRun: true,
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      paths: { modelCache: join(root, "cache") },
      evaluation: { inference: { provider: "none", device: "cpu" }, scoring: { mode: "exact_match" } },
    });
    const request = fineTuneRunRequestSchema.parse({
      run_id: "33333333-3333-4333-8333-333333333333",
      user_id: "local-user",
      behavior_spec_id: "44444444-4444-4444-8444-444444444444",
      run_number: 1,
      spec_snapshot: {
        name: "Gated",
        base_model: "meta-llama/Llama-3.2-3B-Instruct",
        system_prompt: "Answer clearly.",
        examples: [{ input: "Hello", output: "Hi" }],
      },
    });
    const checks = await runDoctor(config, request);
    assert.equal(checks.find((check) => check.name === "hugging-face-token")?.ok, false);
  } finally {
    if (previous === undefined) delete process.env.HF_TOKEN;
    else process.env.HF_TOKEN = previous;
    await rm(root, { recursive: true, force: true });
  }
});

test("doctor rejects a standalone paths.baseModel file", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-doctor-base-file-"));
  try {
    const baseModel = join(root, "model.safetensors");
    await writeFile(baseModel, "weights");
    const config = localRunnerConfigSchema.parse({
      dryRun: true,
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      paths: { baseModel, modelCache: join(root, "cache") },
      evaluation: { inference: { provider: "none", device: "cpu" }, scoring: { mode: "exact_match" } },
    });
    const checks = await runDoctor(config);
    const localBase = checks.find((check) => check.name === "local-base-model");
    assert.equal(localBase?.ok, false);
    assert.match(localBase?.message ?? "", /snapshot directory/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
