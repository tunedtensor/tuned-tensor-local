import assert from "node:assert/strict";
import { chmod, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { test } from "node:test";
import { fineTuneRunRequestSchema, localRunnerConfigSchema } from "../src/contracts.js";
import { buildDoctorPythonPlans, runDoctor } from "../src/doctor.js";

test("doctor probes the locked uv runtime with one cache contract and mandatory CUDA", () => {
  const config = localRunnerConfigSchema.parse({
    paths: { modelCache: "/tmp/tt-hf-home" },
    evaluation: {
      inference: {
        device: "cpu",
      },
      scoring: { mode: "exact_match" },
    },
  });

  const plans = buildDoctorPythonPlans(config);
  assert.deepEqual(plans.map((plan) => plan.name), ["python-runtime"]);
  for (const plan of plans) {
    assert.equal(plan.command, "uv");
    assert.ok(plan.args.includes("--project"));
    assert.ok(plan.args.some((value) => value.endsWith("training/local-runner")));
    assert.equal(plan.env.HF_HOME, "/tmp/tt-hf-home");
    assert.equal(plan.env.HF_HUB_CACHE, "/tmp/tt-hf-home/hub");
    assert.ok(plan.env.UV_PROJECT_ENVIRONMENT);
    const probe = plan.args.at(-1) ?? "";
    assert.match(probe, /assert torch\.cuda\.is_available\(\)/);
    assert.match(probe, /TT Local training requires CUDA/);
    assert.match(probe, /compute_capability/);
  }
});

test("doctor surfaces a failed CUDA runtime probe", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-doctor-cuda-"));
  const previousPath = process.env.PATH;
  try {
    const bin = join(root, "bin");
    await mkdir(bin);
    const uv = join(bin, "uv");
    const nvidiaSmi = join(bin, "nvidia-smi");
    await writeFile(uv, `#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "uv 0.test"
  exit 0
fi
echo "TT Local training requires CUDA but torch.cuda.is_available() is false" >&2
exit 1
`, "utf8");
    await writeFile(nvidiaSmi, "#!/bin/sh\necho 'NVIDIA DGX Spark'\n", "utf8");
    await chmod(uv, 0o755);
    await chmod(nvidiaSmi, 0o755);
    process.env.PATH = `${bin}${delimiter}${previousPath ?? ""}`;

    const checks = await runDoctor(localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      paths: { modelCache: join(root, "cache") },
    }));
    const runtime = checks.find((check) => check.name === "python-runtime");
    assert.equal(runtime?.ok, false);
    assert.match(runtime?.message ?? "", /requires CUDA/);
    assert.equal(checks.find((check) => check.name === "nvidia-smi")?.ok, true);
  } finally {
    if (previousPath === undefined) delete process.env.PATH;
    else process.env.PATH = previousPath;
    await rm(root, { recursive: true, force: true });
  }
});

test("dry-run doctor skips Python and NVIDIA while still checking writable storage", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-doctor-dry-run-"));
  try {
    const checks = await runDoctor(localRunnerConfigSchema.parse({
      dryRun: true,
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      paths: { modelCache: join(root, "cache") },
    }));
    assert.equal(checks.find((check) => check.name === "python-runtime")?.ok, true);
    assert.match(checks.find((check) => check.name === "python-runtime")?.message ?? "", /dryRun/);
    assert.equal(checks.find((check) => check.name === "nvidia-smi")?.ok, true);
    assert.equal(
      checks.filter((check) => check.name.endsWith("root") || check.name === "model-cache")
        .every((check) => check.ok),
      true,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("doctor rejects an unchanged generated placeholder spec", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-doctor-placeholder-"));
  try {
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
    const checks = await runDoctor(localRunnerConfigSchema.parse({
      dryRun: true,
      artifactRoot: join(root, "artifacts"),
      storeRoot: join(root, "store"),
      paths: { modelCache: join(root, "cache") },
    }), request);
    assert.equal(checks.find((check) => check.name === "spec-content")?.ok, false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
