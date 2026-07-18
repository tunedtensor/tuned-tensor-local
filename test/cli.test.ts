import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
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

function runCli(args: string[], cwd: string): CliResult {
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
      { args: ["models", "prefetch", "--help"], usage: "tt-local models prefetch" },
      { args: ["models", "verify-base", "--help"], usage: "tt-local models verify-base" },
      { args: ["models", "verify", "--help"], usage: "tt-local models verify" },
      { args: ["models", "serve", "--help"], usage: "tt-local models serve" },
      { args: ["runs", "report", "--help"], usage: "tt-local runs report" },
      { args: ["models", "--help"], usage: "tt-local models <command>" },
      { args: ["studies", "run", "--help"], usage: "tt-local studies run" },
      { args: ["studies", "validate", "--help"], usage: "tt-local studies validate" },
      { args: ["studies", "--help"], usage: "tt-local studies <command>" },
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

test("study benchmark lock and validation run end to end without opening the local store", async () => {
  await withTemporaryProject(async (root) => {
    const dataRoot = join(root, "data");
    await mkdir(dataRoot, { recursive: true });
    const header = "id,spread_bps,big_move\n";
    await writeFile(join(dataRoot, "training.csv"), `${header}train-1,1.2,0\ntrain-2,8.4,1\n`, "utf8");
    await writeFile(join(dataRoot, "validation.csv"), `${header}validation-1,1.4,0\nvalidation-2,7.9,1\n`, "utf8");
    await writeFile(join(dataRoot, "test.csv"), `${header}test-1,1.1,0\ntest-2,8.1,1\n`, "utf8");
    const studyPath = join(root, "portfolio.study.json");
    await writeFile(studyPath, `${JSON.stringify({
      schema_version: 1,
      name: "Portfolio anomaly benchmark",
      task: {
        type: "binary_classification",
        id_column: "id",
        input_columns: ["spread_bps"],
        target_column: "big_move",
        labels: { negative: "0", positive: "1" },
        primary_metric: "average_precision",
      },
      dataset: {
        format: "csv",
        splits: {
          training: "data/training.csv",
          validation: "data/validation.csv",
          test: "data/test.csv",
        },
      },
    }, null, 2)}\n`, "utf8");

    const locked = runCli(["studies", "lock", studyPath], root);
    assert.equal(locked.status, 0, locked.stderr);
    const lockedOutput = JSON.parse(locked.stdout) as { ok: boolean; lock_path: string };
    assert.equal(lockedOutput.ok, true);
    const lockText = await readFile(lockedOutput.lock_path, "utf8");

    const validated = runCli(["studies", "validate", studyPath], root);
    assert.equal(validated.status, 0, validated.stderr);
    assert.equal(JSON.parse(validated.stdout).ok, true);

    const runnerPath = join(root, "trial-runner.mjs");
    await writeFile(runnerPath, `
      import { readFileSync, writeFileSync } from "node:fs";
      const arg = (name) => process.argv[process.argv.indexOf(name) + 1];
      const input = JSON.parse(readFileSync(arg("--input"), "utf8"));
      const validation = readFileSync(input.datasets.validation_csv, "utf8");
      if (validation !== "id,spread_bps\\nvalidation-1,1.4\\nvalidation-2,7.9\\n") {
        throw new Error("validation projection contains unexpected data");
      }
      writeFileSync(arg("--output"), JSON.stringify({
        protocol_version: 1,
        predictions: [
          { id: "validation-2", probability: 0.9 },
          { id: "validation-1", probability: 0.1 }
        ]
      }));
    `, "utf8");
    const trialPath = join(root, "logreg.trial.json");
    await writeFile(trialPath, `${JSON.stringify({
      schema_version: 1,
      id: "logreg-v1",
      name: "Logistic regression",
      runner: {
        command: [process.execPath, runnerPath],
        cwd: ".",
        timeout_ms: 10_000,
        provenance: {
          source_files: ["trial-runner.mjs"],
          dependency_lock_files: [],
        },
      },
      parameters: { c: 1 },
    }, null, 2)}\n`, "utf8");
    const trialOutputRoot = join(root, "trial-output");
    const trial = runCli([
      "studies",
      "run",
      studyPath,
      trialPath,
      "--output-root",
      trialOutputRoot,
    ], root);
    assert.equal(trial.status, 0, trial.stderr);
    const trialResult = JSON.parse(trial.stdout) as {
      ok: boolean;
      trial_report: { evaluation: { primary_score: number } };
      report_path: string;
    };
    assert.equal(trialResult.ok, true);
    assert.equal(trialResult.trial_report.evaluation.primary_score, 1);
    assert.equal(existsSync(trialResult.report_path), true);

    await writeFile(
      join(dataRoot, "validation.csv"),
      `${header}validation-1,1.4,0\nvalidation-2,7.8,1\n`,
      "utf8",
    );
    const drifted = runCli(["studies", "validate", studyPath], root);
    assert.equal(drifted.status, 1);
    assert.match(drifted.stderr, /dataset\.splits\.validation\.sha256.*expected.*found/is);
    assert.equal(await readFile(lockedOutput.lock_path, "utf8"), lockText);
    assertNoWorkCreated(root);
  });
});

test("unknown options are rejected before run, nested, or store work", async () => {
  await withTemporaryProject((root) => {
    for (const args of [
      ["--dryrun"],
      ["run", "--dryrun"],
      ["models", "prefetch", "--dryrun"],
      ["models", "serve", "model-id", "--public"],
      ["runs", "list", "--wat"],
    ]) {
      const result = runCli(args, root);
      assert.equal(result.status, 1);
      assert.match(result.stderr, /^Unknown option: --(?:dryrun|public|wat)/);
      assert.equal(result.stdout, "");
      assertNoWorkCreated(root);
    }
  });
});

test("options that require values fail clearly and before filesystem access", async () => {
  await withTemporaryProject((root) => {
    for (const args of [
      ["run", "--config"],
      ["run", "--stage", "--quiet"],
      ["serve", "--port="],
      ["models", "prefetch", "--user-id"],
    ]) {
      const result = runCli(args, root);
      assert.equal(result.status, 1);
      assert.match(result.stderr, /^Option --(?:config|stage|port|user-id) requires a value\./);
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

test("stored models are verified before a serving launch plan is produced", async () => {
  await withTemporaryProject(async (root) => {
    const runId = "11111111-1111-4111-8111-111111111111";
    const specId = "22222222-2222-4222-8222-222222222222";
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
        augment: false,
        use_llm_judge: false,
        save_adapter_only: true,
      },
    });
    const artifacts = resolveRunArtifacts({ artifactRoot: join(root, "artifacts"), prefix: "run" });
    const store = createLocalStore(join(root, "store"));
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
        format: "peft-directory",
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

    const launch = runCli(["models", "serve", modelId, "--print-command"], root);
    assert.equal(launch.status, 0, launch.stderr);
    const launchPlan = JSON.parse(launch.stdout) as { ok: boolean; url: string; command: string[] };
    assert.equal(launchPlan.ok, true);
    assert.equal(launchPlan.url, "http://127.0.0.1:8000");
    assert.ok(launchPlan.command.some((part) => part.endsWith("training/local-runner/src/serve.py")));

    const childRunId = "33333333-3333-4333-8333-333333333333";
    const childRequest = fineTuneRunRequestSchema.parse({
      ...request,
      run_id: childRunId,
      run_number: 2,
    });
    const childPath = join(root, "child-request.json");
    await writeFile(childPath, `${JSON.stringify(childRequest)}\n`, "utf8");
    const continued = runCli([
      "run", childPath, "--parent-model", modelId, "--dry-run", "--stage", "prepare", "--quiet",
    ], root);
    assert.equal(continued.status, 0, continued.stderr);
    const persistedChild = fineTuneRunRequestSchema.parse(JSON.parse(
      await readFile(join(store.paths.runsDir, childRunId, "request.json"), "utf8"),
    ));
    assert.equal(persistedChild.hyperparameters.base_model_revision, "revision-a");
    assert.equal(persistedChild.hyperparameters.parent_model_artifact, training.model_artifact_uri);

    const mismatchPath = join(root, "mismatched-child-request.json");
    await writeFile(mismatchPath, `${JSON.stringify({
      ...childRequest,
      run_id: "44444444-4444-4444-8444-444444444444",
      hyperparameters: { ...childRequest.hyperparameters, base_model_revision: "revision-b" },
    })}\n`, "utf8");
    const mismatchedRevision = runCli([
      "run", mismatchPath, "--parent-model", modelId, "--dry-run", "--stage", "prepare", "--quiet",
    ], root);
    assert.equal(mismatchedRevision.status, 1);
    assert.match(mismatchedRevision.stderr, /uses base revision revision-a, but this run requests revision-b/);

    const runRequestPath = join(store.paths.runsDir, runId, "request.json");
    await writeFile(runRequestPath, `${JSON.stringify({
      ...request,
      spec_snapshot: { ...request.spec_snapshot, system_prompt: "Changed after training." },
    })}\n`, "utf8");
    const mismatchedPrompt = runCli(["models", "serve", modelId, "--print-command"], root);
    assert.equal(mismatchedPrompt.status, 1);
    assert.match(mismatchedPrompt.stderr, /do not match the prompt fingerprint/);
    await writeFile(runRequestPath, `${JSON.stringify(request)}\n`, "utf8");

    await writeFile(adapterWeights, "tampered weights", "utf8");
    const changed = runCli(["models", "verify", modelId], root);
    assert.equal(changed.status, 1);
    assert.match(changed.stderr, /Artifact integrity verification failed/);
  });
});
