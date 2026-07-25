import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { chmod, mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  fileUri,
  prepareRunDirectories,
  resolveRunArtifacts,
  writeArtifactManifest,
} from "../src/artifacts.js";
import { fineTuneRunRequestSchema, trainingReportSchema } from "../src/contracts.js";
import { createLocalStore } from "../src/store.js";

const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const cliPath = join(projectRoot, "src", "index.ts");
const tsxLoader = import.meta.resolve("tsx");

interface CliResult {
  status: number | null;
  stdout: string;
  stderr: string;
}

function runCli(args: string[], cwd: string, env: NodeJS.ProcessEnv = {}): CliResult {
  const storeRoot = join(cwd, "store");
  const result = spawnSync(
    process.execPath,
    ["--import", tsxLoader, cliPath, ...args],
    {
      cwd,
      encoding: "utf8",
      env: {
        ...process.env,
        TT_LOCAL_HOME: storeRoot,
        ...env,
      },
    },
  );
  assert.equal(result.signal, null, result.error?.message);
  return {
    status: result.status,
    stdout: result.stdout,
    stderr: result.stderr,
  };
}

async function withTemporaryProject(
  callback: (root: string) => void | Promise<void>,
): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), "tt-local-cli-test-"));
  try {
    await callback(root);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

function assertNoWorkCreated(root: string): void {
  assert.equal(existsSync(join(root, "store")), false, "CLI help/error created a local store");
  assert.equal(existsSync(join(root, ".tt-local")), false, "CLI help/error created artifacts");
}

test("top-level help and version are available without loading project state", async () => {
  await withTemporaryProject(async (root) => {
    const packageJson = JSON.parse(await readFile(join(projectRoot, "package.json"), "utf8")) as { version: string };

    const help = runCli(["--help"], root);
    assert.equal(help.status, 0);
    assert.match(help.stdout, /^Usage: tt-local <command> \[options\]/);
    assert.match(help.stdout, /-V, --version/);
    assert.match(help.stdout, /serve <model-id>/);
    assert.doesNotMatch(
      help.stdout,
      /\blabel\b|\bspecs\b|dashboard|rebuild-index|\breconcile\b|parent-model|model-artifact|--stage|--run-id|--detach|\bwatch\b|\bcancel\b/i,
    );
    assert.equal(help.stderr, "");

    const version = runCli(["--version"], root);
    assert.equal(version.status, 0);
    assert.equal(version.stdout.trim(), packageJson.version);
    assert.equal(version.stderr, "");

    const info = runCli(["info"], root);
    assert.equal(info.status, 0);
    assert.match(info.stdout, new RegExp(`Version: ${packageJson.version.replaceAll(".", "\\.")}`));
    assertNoWorkCreated(root);
  });
});

test("command and nested-command help never execute work", async () => {
  await withTemporaryProject(async (root) => {
    await writeFile(join(root, ".env"), "TT_LOCAL_HELP_MUST_NOT_LOAD=true\n", "utf8");
    const cases = [
      { args: ["run", "--help"], usage: "tt-local run" },
      { args: ["serve", "--help"], usage: "tt-local serve" },
      { args: ["models", "prefetch", "--help"], usage: "tt-local models prefetch" },
      { args: ["models", "verify-base", "--help"], usage: "tt-local models verify-base" },
      { args: ["models", "verify", "--help"], usage: "tt-local models verify" },
      { args: ["models", "serve", "--help"], usage: "tt-local models serve" },
      { args: ["runs", "report", "--help"], usage: "tt-local runs report" },
      { args: ["models", "--help"], usage: "tt-local models <command>" },
    ];

    for (const { args, usage } of cases) {
      const result = runCli(args, root);
      assert.equal(result.status, 0, `${args.join(" ")}: ${result.stderr}`);
      assert.match(result.stdout, new RegExp(`Usage: ${usage.replaceAll(" ", "\\s+")}`));
      assert.equal(result.stderr, "");
      assertNoWorkCreated(root);
    }
  });
});

test("unknown options are rejected before run, nested, or store work", async () => {
  await withTemporaryProject((root) => {
    for (const args of [
      ["--dryrun"],
      ["run", "--dryrun"],
      ["run", "--parent-model", "model-id"],
      ["run", "--model-artifact", "artifact"],
      ["run", "--stage", "train"],
      ["run", "--force"],
      ["run", "--run-id", "11111111-1111-4111-8111-111111111111"],
      ["run", "--user-id", "someone"],
      ["run", "--detach"],
      ["models", "prefetch", "--dryrun"],
      ["serve", "model-id", "--public"],
      ["runs", "list", "--wat"],
    ]) {
      const result = runCli(args, root);
      assert.equal(result.status, 1);
      assert.match(
        result.stderr,
        /^Unknown option: --(?:dryrun|parent-model|model-artifact|stage|force|run-id|user-id|detach|public|wat)/,
      );
      assert.equal(result.stdout, "");
      assertNoWorkCreated(root);
    }
  });
});

test("options that require values fail clearly and before filesystem access", async () => {
  await withTemporaryProject((root) => {
    for (const args of [
      ["run", "--config"],
      ["serve", "model-id", "--port="],
      ["models", "prefetch", "--config"],
      ["init", "--name"],
    ]) {
      const result = runCli(args, root);
      assert.equal(result.status, 1);
      assert.match(result.stderr, /^Option --(?:config|port|name) requires a value\./);
      assert.equal(result.stdout, "");
      assertNoWorkCreated(root);
    }
  });
});

test("extra positional arguments and duplicate options are rejected", async () => {
  await withTemporaryProject((root) => {
    const extra = runCli(["runs", "get", "run-a", "run-b"], root);
    assert.equal(extra.status, 1);
    assert.match(extra.stderr, /^Too many arguments\. Usage: tt-local runs get/);

    const duplicate = runCli(["validate", "--config", "one.json", "--config=two.json"], root);
    assert.equal(duplicate.status, 1);
    assert.match(duplicate.stderr, /^Option --config may only be specified once\./);
    assertNoWorkCreated(root);
  });
});

test("removed generalized commands are no longer part of the CLI", async () => {
  await withTemporaryProject((root) => {
    const cases = [
      { args: ["label", "rows.jsonl"], message: /^Unknown command: label/ },
      { args: ["specs", "list"], message: /^Unknown command: specs/ },
      { args: ["store", "rebuild-index"], message: /^Unknown command: store/ },
      { args: ["runs", "reconcile"], message: /^Unknown runs command: reconcile/ },
      { args: ["runs", "watch", "run-id"], message: /^Unknown runs command: watch/ },
      { args: ["runs", "cancel", "run-id"], message: /^Unknown runs command: cancel/ },
    ];
    for (const { args, message } of cases) {
      const result = runCli(args, root);
      assert.equal(result.status, 1);
      assert.match(result.stderr, message);
      assert.equal(result.stdout, "");
      assertNoWorkCreated(root);
    }
  });
});

test("public workflow commands reject full run-request JSON", async () => {
  await withTemporaryProject(async (root) => {
    const requestPath = join(root, "request.json");
    await writeFile(requestPath, `${JSON.stringify({
      run_id: "99999999-9999-4999-8999-999999999999",
      user_id: "hosted-user",
      behavior_spec_id: "88888888-8888-4888-8888-888888888888",
      run_number: 42,
      spec_snapshot: {
        name: "Hosted request",
        description: "",
        system_prompt: "Answer briefly.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [
          { input: "One", output: "1" },
          { input: "Two", output: "2" },
        ],
      },
      hyperparameters: { n_epochs: 1 },
    })}\n`, "utf8");

    for (const command of ["doctor", "validate", "run"] as const) {
      const result = runCli([command, requestPath], root);
      assert.equal(result.status, 1);
      assert.match(result.stderr, /expects a tunedtensor\.json behavior spec, not a full run request/);
      assertNoWorkCreated(root);
    }
  });
});

test("adjacent config is discovered for init, doctor, validate, and dry-run", async () => {
  await withTemporaryProject(async (root) => {
    const project = join(root, "project");
    const specPath = join(project, "tunedtensor.json");
    const configPath = join(project, "local-runner.json");
    await mkdir(project, { recursive: true });
    await writeFile(configPath, `${JSON.stringify({
      artifactRoot: "artifacts",
      storeRoot: "state",
      dryRun: true,
    })}\n`, "utf8");

    const initialized = runCli(["init", "--output", specPath], root);
    assert.equal(initialized.status, 0, initialized.stderr);
    assert.equal(JSON.parse(initialized.stdout).config_path, configPath);

    await writeFile(specPath, `${JSON.stringify({
      id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
      name: "Adjacent config",
      description: "",
      system_prompt: "Answer the user accurately and briefly.",
      guidelines: [],
      constraints: [],
      base_model: "Qwen/Qwen3.5-2B",
      examples: [
        { input: "Say yes.", output: "Yes." },
        { input: "Say no.", output: "No." },
      ],
      hyperparameters: { n_epochs: 1 },
    })}\n`, "utf8");

    const doctor = runCli(["doctor", specPath], root);
    assert.ok(doctor.status === 0 || doctor.status === 1, doctor.stderr);
    assert.equal(JSON.parse(doctor.stdout).config_path, configPath);

    const defaultDoctor = runCli(["doctor"], project);
    assert.ok(defaultDoctor.status === 0 || defaultDoctor.status === 1, defaultDoctor.stderr);
    const defaultDoctorOutput = JSON.parse(defaultDoctor.stdout) as {
      config_path: string;
      checks: Array<{ name: string }>;
    };
    assert.equal(
      await realpath(defaultDoctorOutput.config_path),
      await realpath(configPath),
    );
    assert.ok(defaultDoctorOutput.checks.some((check) => check.name === "spec-content"));

    const validated = runCli(["validate", specPath], root);
    assert.equal(validated.status, 0, validated.stderr);
    const validation = JSON.parse(validated.stdout) as Record<string, unknown>;
    assert.equal(validation.config_path, configPath);
    assert.equal(validation.artifact_root, join(project, "artifacts"));
    assert.equal(validation.store_root, join(project, "state"));
    assert.equal(validation.dry_run, true);
    assert.equal("training_method" in validation, false);

    const run = runCli(["run", specPath, "--quiet"], root);
    assert.equal(run.status, 0, run.stderr);
    const output = JSON.parse(run.stdout) as Record<string, unknown>;
    assert.equal(output.status, "completed");
    assert.equal("model_id" in output, false);
    assert.equal("fine_tuned_model_id" in output, false);
    assert.ok(String(output.artifact_dir).startsWith(join(project, "artifacts")));

    const runs = runCli(["runs", "list"], project);
    assert.equal(runs.status, 0, runs.stderr);
    assert.equal(JSON.parse(runs.stdout).length, 1);

    const models = runCli(["models", "list"], project);
    assert.equal(models.status, 0, models.stderr);
    assert.deepEqual(JSON.parse(models.stdout), []);
  });
});

test("one-command real runs prefetch and pin an immutable base-model revision", async () => {
  await withTemporaryProject(async (root) => {
    const project = join(root, "project");
    const fakeBin = join(root, "bin");
    const callsPath = join(root, "uv-calls.log");
    const specPath = join(project, "tunedtensor.json");
    await mkdir(project, { recursive: true });
    await mkdir(fakeBin, { recursive: true });
    await writeFile(join(project, "local-runner.json"), `${JSON.stringify({
      artifactRoot: "artifacts",
      storeRoot: "state",
      dryRun: false,
    })}\n`, "utf8");
    await writeFile(specPath, `${JSON.stringify({
      id: "77777777-7777-4777-8777-777777777777",
      name: "Pinned run",
      description: "",
      system_prompt: "Answer accurately.",
      guidelines: [],
      constraints: [],
      base_model: "Qwen/Qwen3.5-2B",
      examples: [
        { input: "One", output: "1" },
        { input: "Two", output: "2" },
      ],
      hyperparameters: { n_epochs: 1 },
    })}\n`, "utf8");
    const fakeUv = join(fakeBin, "uv");
    await writeFile(fakeUv, `#!/usr/bin/env node
const fs = require("node:fs");
const args = process.argv.slice(2);
const isPrefetch = args.some((value) => value.endsWith("prefetch.py"));
fs.appendFileSync(${JSON.stringify(callsPath)}, isPrefetch ? "prefetch\\n" : "other\\n");
if (!isPrefetch) process.exit(23);
const outputIndex = args.indexOf("--output");
if (outputIndex === -1 || !args[outputIndex + 1]) process.exit(24);
fs.writeFileSync(args[outputIndex + 1], JSON.stringify({
  ok: true,
  base_model: "Qwen/Qwen3.5-2B",
  snapshot_revision: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  snapshot_path: "/fake/snapshot",
  file_count: 4,
  size_bytes: 100,
}) + "\\n");
`, "utf8");
    await chmod(fakeUv, 0o755);

    const result = runCli(["run", specPath, "--quiet"], root, {
      PATH: `${fakeBin}:${process.env.PATH ?? ""}`,
    });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /exited (?:with code )?23/);

    const calls = (await readFile(callsPath, "utf8")).trim().split("\n");
    assert.deepEqual(calls.slice(0, 2), ["prefetch", "other"]);
    assert.equal(calls.filter((call) => call === "prefetch").length, 1);

    const store = createLocalStore(join(project, "state"));
    const runs = await store.listRuns();
    assert.equal(runs.length, 1);
    const persisted = fineTuneRunRequestSchema.parse(JSON.parse(
      await readFile(join(store.paths.runsDir, runs[0]!.id, "request.json"), "utf8"),
    ));
    assert.equal(
      persisted.hyperparameters.base_model_revision,
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    );
  });
});

test("stored models are verified before a serving launch plan is produced", async () => {
  await withTemporaryProject(async (root) => {
    const runId = "11111111-1111-4111-8111-111111111111";
    const specId = "22222222-2222-4222-8222-222222222222";
    const modelStoreRoot = join(root, "model-state");
    await writeFile(join(root, "local-runner.json"), `${JSON.stringify({
      storeRoot: "model-state",
    })}\n`, "utf8");
    const request = fineTuneRunRequestSchema.parse({
      run_id: runId,
      user_id: "local-user",
      behavior_spec_id: specId,
      run_number: 1,
      spec_snapshot: {
        name: "CLI model verification",
        description: "",
        system_prompt: "Answer briefly.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "Hello", output: "Hi" }],
      },
      hyperparameters: {
        n_epochs: 1,
      },
    });
    const artifacts = resolveRunArtifacts({ artifactRoot: join(root, "artifacts"), prefix: "run" });
    const store = createLocalStore(modelStoreRoot);
    await prepareRunDirectories(artifacts);
    await store.startRun({ request, artifactDir: artifacts.runDir });
    const adapterWeights = join(artifacts.trainingModelDir, "adapter_model.safetensors");
    await writeFile(adapterWeights, "model weights", "utf8");
    await writeFile(join(artifacts.trainingModelDir, "adapter_config.json"), "{}\n", "utf8");
    await writeFile(artifacts.stageMetadataJson, `${JSON.stringify({
      system_prompt_sha256: createHash("sha256").update("Answer briefly.").digest("hex"),
    })}\n`, "utf8");
    const training = trainingReportSchema.parse({
      provider: "local-uv",
      training_job_name: "cli-test",
      model_artifact_uri: fileUri(artifacts.trainingModelDir),
      metrics: { loss: 0.1 },
      exit_code: 0,
      log_uri: fileUri(artifacts.trainingLog),
    });
    await writeFile(artifacts.trainingReportJson, `${JSON.stringify(training)}\n`, "utf8");
    await writeArtifactManifest(artifacts, {
      model: {
        artifact_kind: "directory",
        format: "huggingface-directory",
        framework: "transformers-peft",
        base_model: request.spec_snapshot.base_model,
        base_model_revision: "revision-a",
        artifact_uri: training.model_artifact_uri!,
        artifact_root: artifacts.trainingModelDir,
        servable: true,
      },
    });
    await store.registerModel({ request, training, artifactDir: artifacts.runDir });

    const modelId = `local-${runId}`;
    await writeFile(`${artifacts.candidateEvalJson}.inference.log`, "partial candidate output\n", "utf8");
    const verified = runCli(["models", "verify", modelId], root);
    assert.equal(verified.status, 0, verified.stderr);
    assert.equal(JSON.parse(verified.stdout).ok, true);

    const verifiedPath = runCli(["models", "verify", artifacts.trainingModelDir], root);
    assert.equal(verifiedPath.status, 0, verifiedPath.stderr);
    assert.equal(JSON.parse(verifiedPath.stdout).model, null);

    const verifiedManifest = runCli(["models", "verify", artifacts.artifactManifestJson], root);
    assert.equal(verifiedManifest.status, 0, verifiedManifest.stderr);

    const launch = runCli(["serve", modelId, "--print-command"], root);
    assert.equal(launch.status, 0, launch.stderr);
    const launchPlan = JSON.parse(launch.stdout) as { ok: boolean; url: string; command: string[] };
    assert.equal(launchPlan.ok, true);
    assert.equal(launchPlan.url, "http://127.0.0.1:8000");
    assert.ok(launchPlan.command.some((part) => part.endsWith("training/local-runner/src/serve.py")));

    const aliasLaunch = runCli(["models", "serve", modelId, "--print-command"], root);
    assert.equal(aliasLaunch.status, 0, aliasLaunch.stderr);
    assert.equal(JSON.parse(aliasLaunch.stdout).url, launchPlan.url);

    const unsupportedDevice = runCli(["serve", modelId, "--device", "mps", "--print-command"], root);
    assert.equal(unsupportedDevice.status, 1);
    assert.match(unsupportedDevice.stderr, /--device must be cuda or cpu/);

    const runRequestPath = join(store.paths.runsDir, runId, "request.json");
    await writeFile(runRequestPath, `${JSON.stringify({
      ...request,
      spec_snapshot: { ...request.spec_snapshot, system_prompt: "Changed after training." },
    })}\n`, "utf8");
    const mismatchedPrompt = runCli(["serve", modelId, "--print-command"], root);
    assert.equal(mismatchedPrompt.status, 1);
    assert.match(mismatchedPrompt.stderr, /do not match the prompt fingerprint/);
    await writeFile(runRequestPath, `${JSON.stringify(request)}\n`, "utf8");

    await writeFile(adapterWeights, "tampered weights", "utf8");
    const changed = runCli(["models", "verify", modelId], root);
    assert.equal(changed.status, 1);
    assert.match(changed.stderr, /Artifact integrity verification failed/);
  });
});
