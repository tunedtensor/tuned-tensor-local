import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import {
  chmod,
  cp,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import test from "node:test";
import {
  buildStudyTrialRunnerCommand,
  runStudyTrial,
  studyTrialSpecSchema,
  type StudyTrialSpec,
} from "../src/study-trials.js";
import { writeStudyBenchmarkLock } from "../src/studies.js";

function sha256(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

interface TrialFixture {
  root: string;
  studyPath: string;
  outputRoot: string;
  validationPath: string;
  writeTrial: (
    id: string,
    script: string,
    overrides?: Partial<StudyTrialSpec["runner"]>,
  ) => Promise<string>;
}

async function withTrialFixture(
  callback: (fixture: TrialFixture) => void | Promise<void>,
): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), "tt-local-study-trial-test-"));
  try {
    const dataDirectory = join(root, "data");
    await mkdir(dataDirectory);
    const header = "id,spread_bps,note,big_move,future_mid\n";
    await writeFile(
      join(dataDirectory, "training.csv"),
      `${header}train-1,1.2,quiet,0,0.10\ntrain-2,8.4,move,1,0.90\n`,
      "utf8",
    );
    const validationPath = join(dataDirectory, "validation.csv");
    await writeFile(
      validationPath,
      `${header}validation-1,1.4,quiet,0,0.20\nvalidation-2,7.9,move,1,0.80\n`,
      "utf8",
    );
    await writeFile(
      join(dataDirectory, "test.csv"),
      `${header}test-1,1.1,quiet,0,0.30\ntest-2,8.1,move,1,0.70\n`,
      "utf8",
    );
    const studyPath = join(root, "portfolio.study.json");
    await writeFile(studyPath, `${JSON.stringify({
      schema_version: 1,
      name: "Portfolio anomaly benchmark",
      task: {
        type: "binary_classification",
        id_column: "id",
        input_columns: ["spread_bps", "note"],
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
    await writeStudyBenchmarkLock({ studyPath });

    let scriptNumber = 0;
    const writeTrial: TrialFixture["writeTrial"] = async (id, script, overrides = {}) => {
      scriptNumber += 1;
      const scriptPath = join(root, `trial-runner-${scriptNumber}.mjs`);
      await writeFile(scriptPath, script, "utf8");
      const trialPath = join(root, `${id}.trial.json`);
      await writeFile(trialPath, `${JSON.stringify({
        schema_version: 1,
        id,
        name: `Trial ${id}`,
        runner: {
          command: [process.execPath, scriptPath],
          cwd: ".",
          timeout_ms: 10_000,
          provenance: {
            source_files: [`trial-runner-${scriptNumber}.mjs`],
            dependency_lock_files: [],
          },
          ...overrides,
        },
        parameters: { regularization: 0.25 },
      }, null, 2)}\n`, "utf8");
      return trialPath;
    };

    await callback({
      root,
      studyPath,
      outputRoot: join(root, "trial-output"),
      validationPath,
      writeTrial,
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

const validRunner = `
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
const value = (name) => process.argv[process.argv.indexOf(name) + 1];
const input = JSON.parse(readFileSync(value("--input"), "utf8"));
const training = readFileSync(input.datasets.training_csv, "utf8");
const validation = readFileSync(input.datasets.validation_csv, "utf8");
if (training !== "id,spread_bps,note,big_move\\ntrain-1,1.2,quiet,0\\ntrain-2,8.4,move,1\\n") {
  throw new Error("unexpected projected training data: " + training);
}
if (validation !== "id,spread_bps,note\\nvalidation-1,1.4,quiet\\nvalidation-2,7.9,move\\n") {
  throw new Error("unexpected projected validation data: " + validation);
}
if ("test" in input.datasets || "study" in input || "lock" in input) {
  throw new Error("private benchmark material leaked into trial input");
}
mkdirSync(value("--artifact-dir"), { recursive: true });
writeFileSync(join(value("--artifact-dir"), "model.txt"), "trained");
writeFileSync(value("--output"), JSON.stringify({
  protocol_version: 1,
  predictions: [
    { id: "validation-2", probability: 0.9 },
    { id: "validation-1", probability: 0.1 }
  ]
}));
`;

test("runs a label-blind command trial and computes trusted validation metrics", async () => {
  await withTrialFixture(async ({ studyPath, outputRoot, writeTrial }) => {
    const trialPath = await writeTrial("logreg-v1", validRunner);
    const result = await runStudyTrial({ studyPath, trialPath, outputRoot });

    assert.equal(result.trialDirectory, join(outputRoot, "logreg-v1"));
    assert.equal(result.report.evaluation.primary_score, 1);
    assert.deepEqual(result.report.evaluation.metrics, {
      average_precision: 1,
      roc_auc: 1,
      f1_at_0_5: 1,
    });
    assert.equal(result.report.evaluation.prediction_count, 2);
    assert.deepEqual(result.report.trial.parameters, { regularization: 0.25 });
    assert.equal(
      await readFile(join(result.trialDirectory, "model", "model.txt"), "utf8"),
      "trained",
    );
    const implementationManifestText = await readFile(
      join(result.trialDirectory, "implementation", "manifest.json"),
      "utf8",
    );
    assert.equal(
      result.report.provenance.implementation.sha256,
      sha256(implementationManifestText),
    );
    assert.equal(
      result.report.provenance.implementation.evidence,
      "declared_files",
    );
    assert.equal(result.report.provenance.implementation.file_count, 1);
    const implementationManifest = JSON.parse(implementationManifestText) as {
      evidence: string;
      files: Array<{
        role: string;
        path: string;
        snapshot_path: string;
        sha256: string;
      }>;
    };
    assert.equal(implementationManifest.evidence, "declared_files");
    assert.deepEqual(
      implementationManifest.files.map((file) => ({
        role: file.role,
        path: file.path,
        snapshot_path: file.snapshot_path,
      })),
      [{
        role: "source",
        path: "trial-runner-1.mjs",
        snapshot_path: "implementation/source/trial-runner-1.mjs",
      }],
    );
    assert.equal(
      await readFile(
        join(result.trialDirectory, "implementation", "source", "trial-runner-1.mjs"),
        "utf8",
      ),
      validRunner,
    );
    assert.equal(implementationManifest.files[0]!.sha256, sha256(validRunner));

    const modelManifestText = await readFile(
      join(result.trialDirectory, "model-manifest.json"),
      "utf8",
    );
    assert.equal(result.report.provenance.model.sha256, sha256(modelManifestText));
    assert.equal(result.report.provenance.model.file_count, 1);
    assert.equal(result.report.provenance.model.size_bytes, 7);
    assert.deepEqual(
      (JSON.parse(modelManifestText) as { files: unknown[] }).files,
      [{ path: "model.txt", size_bytes: 7, sha256: sha256("trained") }],
    );

    const validation = await readFile(join(result.trialDirectory, "validation.csv"), "utf8");
    assert.equal(validation.includes("big_move"), false);
    assert.equal(validation.includes("future_mid"), false);
    const training = await readFile(join(result.trialDirectory, "training.csv"), "utf8");
    assert.equal(training.includes("future_mid"), false);

    const input = JSON.parse(
      await readFile(join(result.trialDirectory, "trial-input.json"), "utf8"),
    ) as Record<string, unknown>;
    assert.equal("test" in (input.datasets as Record<string, unknown>), false);
    assert.equal("study" in input, false);
    assert.equal("lock" in input, false);
    assert.equal("artifact_directory" in input, false);

    const reportText = await readFile(result.reportPath, "utf8");
    assert.equal(reportText.includes("validation-1"), false);
    assert.equal(reportText.includes("validation-2"), false);
    assert.equal(reportText.includes("future_mid"), false);
    assert.equal(reportText.includes("big_move"), false);
  });
});

test("surfaces temporal certification without exposing its metadata to the runner", async () => {
  await withTrialFixture(async ({ root, studyPath, outputRoot, writeTrial }) => {
    const header = (
      "id,spread_bps,note,big_move,future_mid,"
      + "observed_at,future_observed_at\n"
    );
    await writeFile(
      join(root, "data", "training.csv"),
      `${header}`
      + "train-1,1.2,quiet,0,0.10,2026-07-03T00:00:00Z,2026-07-03T00:30:00Z\n"
      + "train-2,8.4,move,1,0.90,2026-07-03T00:10:00Z,2026-07-03T00:40:00Z\n",
      "utf8",
    );
    await writeFile(
      join(root, "data", "validation.csv"),
      `${header}`
      + "validation-1,1.4,quiet,0,0.20,2026-07-03T02:00:00Z,2026-07-03T02:30:00Z\n"
      + "validation-2,7.9,move,1,0.80,2026-07-03T02:10:00Z,2026-07-03T02:40:00Z\n",
      "utf8",
    );
    await writeFile(
      join(root, "data", "test.csv"),
      `${header}`
      + "test-1,1.1,quiet,0,0.30,2026-07-03T04:00:00Z,2026-07-03T04:30:00Z\n"
      + "test-2,8.1,move,1,0.70,2026-07-03T04:10:00Z,2026-07-03T04:40:00Z\n",
      "utf8",
    );
    const study = JSON.parse(await readFile(studyPath, "utf8")) as {
      dataset: {
        temporal?: {
          policy: string;
          event_time_column: string;
          label_end_time_column: string;
          label_horizon_seconds: number;
          embargo_seconds: number;
        };
      };
    };
    study.dataset.temporal = {
      policy: "ordered_purged",
      event_time_column: "observed_at",
      label_end_time_column: "future_observed_at",
      label_horizon_seconds: 3_600,
      embargo_seconds: 300,
    };
    await writeFile(studyPath, `${JSON.stringify(study, null, 2)}\n`, "utf8");
    await writeStudyBenchmarkLock({ studyPath, force: true });

    const trialPath = await writeTrial("temporal-logreg", validRunner);
    const result = await runStudyTrial({ studyPath, trialPath, outputRoot });
    assert.equal(result.report.data.temporal?.policy, "ordered_purged");
    assert.equal(result.report.data.temporal?.label_horizon_seconds, 3_600);
    assert.equal(result.report.data.temporal?.embargo_seconds, 300);
    assert.deepEqual(
      result.report.data.temporal?.splits.validation.event_time,
      {
        min: "2026-07-03T02:00:00Z",
        max: "2026-07-03T02:10:00Z",
      },
    );
    assert.equal("test" in result.report.data.temporal!.splits, false);
    const inputText = await readFile(join(result.trialDirectory, "trial-input.json"), "utf8");
    const input = JSON.parse(inputText) as Record<string, unknown>;
    assert.equal("temporal" in input, false);
    assert.equal(inputText.includes("future_observed_at"), false);
    assert.equal(
      (await readFile(join(result.trialDirectory, "validation.csv"), "utf8"))
        .includes("observed_at"),
      false,
    );

    const lockPath = join(root, "portfolio.study.lock.json");
    const forged = JSON.parse(await readFile(lockPath, "utf8")) as {
      dataset: { temporal: { label_horizon_seconds: number } };
    };
    forged.dataset.temporal.label_horizon_seconds = 3_599;
    await writeFile(lockPath, `${JSON.stringify(forged, null, 2)}\n`, "utf8");
    const marker = join(root, "forged-lock-runner-launched");
    const forgedTrialPath = await writeTrial("forged-temporal-lock", `
      import { writeFileSync } from "node:fs";
      writeFileSync(${JSON.stringify(marker)}, "yes");
    `);
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: forgedTrialPath, outputRoot }),
      /dataset\.temporal\.label_horizon_seconds.*expected.*found/is,
    );
    assert.equal(existsSync(marker), false);
    assert.equal(existsSync(join(outputRoot, "forged-temporal-lock")), false);
  });
});

test("defaults to the trial directory and reserves protocol-owned command flags", async () => {
  await withTrialFixture(async ({ studyPath, outputRoot, writeTrial }) => {
    const trialPath = await writeTrial(
      "default-cwd",
      validRunner,
      { cwd: undefined },
    );
    const result = await runStudyTrial({ studyPath, trialPath, outputRoot });
    assert.equal(result.report.execution.cwd, "<trial-directory>");
  });

  assert.throws(() => studyTrialSpecSchema.parse({
    schema_version: 1,
    id: "reserved",
    name: "Reserved flag",
    runner: {
      command: ["python3", "runner.py", "--output"],
      timeout_ms: 10_000,
      provenance: {
        source_files: ["runner.py"],
        dependency_lock_files: [],
      },
    },
    parameters: {},
  }), /reserved by the trial protocol/);
  assert.throws(() => studyTrialSpecSchema.parse({
    schema_version: 1,
    id: "reserved-inline",
    name: "Reserved inline flag",
    runner: {
      command: ["python3", "runner.py", "--output=elsewhere.json"],
      timeout_ms: 10_000,
      provenance: {
        source_files: ["runner.py"],
        dependency_lock_files: [],
      },
    },
    parameters: {},
  }), /reserved by the trial protocol/);
  assert.equal(studyTrialSpecSchema.safeParse({
    schema_version: 1,
    id: "nonfinite",
    name: "Nonfinite parameter",
    runner: {
      command: ["python3", "runner.py"],
      timeout_ms: 10_000,
      provenance: {
        source_files: ["runner.py"],
        dependency_lock_files: [],
      },
    },
    parameters: { learning_rate: Number.POSITIVE_INFINITY },
  }).success, false);
  const commandTrial = {
    schema_version: 1,
    id: "provenance",
    name: "Command provenance",
    runner: {
      command: ["python3", "runner.py"],
      timeout_ms: 10_000,
      provenance: {
        source_files: ["runner.py"],
        dependency_lock_files: ["uv.lock"],
      },
    },
    parameters: {},
  };
  assert.equal(studyTrialSpecSchema.safeParse({
    ...commandTrial,
    runner: {
      command: ["python3", "runner.py"],
      timeout_ms: 10_000,
    },
  }).success, false);
  for (const sourcePath of [
    "../runner.py",
    "/tmp/runner.py",
    "file://runner.py",
    "scripts\\runner.py",
    "scripts//runner.py",
  ]) {
    assert.equal(studyTrialSpecSchema.safeParse({
      ...commandTrial,
      runner: {
        ...commandTrial.runner,
        provenance: {
          ...commandTrial.runner.provenance,
          source_files: [sourcePath],
        },
      },
    }).success, false);
  }
  assert.equal(studyTrialSpecSchema.safeParse({
    ...commandTrial,
    runner: {
      ...commandTrial.runner,
      provenance: {
        source_files: ["runner.py"],
        dependency_lock_files: ["runner.py"],
      },
    },
  }).success, false);
});

test("accepts only the declared bundled runner shape and resolves its locked script", () => {
  const runner = {
    builtin: "numeric_logistic_regression" as const,
    timeout_ms: 300_000,
  };
  assert.equal(studyTrialSpecSchema.safeParse({
    schema_version: 1,
    id: "numeric-logreg",
    name: "Bundled numeric logistic regression",
    runner,
    parameters: {
      c: 1,
      class_weight: "balanced",
      max_iter: 1_000,
      random_seed: 42,
    },
  }).success, true);
  assert.equal(studyTrialSpecSchema.safeParse({
    schema_version: 1,
    id: "numeric-logreg-with-cwd",
    name: "Invalid bundled runner",
    runner: { ...runner, cwd: "." },
    parameters: {},
  }).success, false);
  assert.equal(studyTrialSpecSchema.safeParse({
    schema_version: 1,
    id: "mixed-runner",
    name: "Invalid mixed runner",
    runner: { ...runner, command: ["python3", "runner.py"] },
    parameters: {},
  }).success, false);

  const command = buildStudyTrialRunnerCommand(runner);
  assert.equal(command.command, "uv");
  assert.deepEqual(command.reportCommand, ["builtin:numeric_logistic_regression"]);
  assert.equal(command.baseArgs[0], "run");
  assert.equal(command.baseArgs[1], "--locked");
  assert.match(
    command.baseArgs[2]!,
    /training[/\\]study-runner[/\\]numeric_logistic_regression\.py$/,
  );
  assert.deepEqual(command.requiredFiles, [
    command.baseArgs[2],
    `${command.baseArgs[2]}.lock`,
  ]);
});

test("runs the bundled numeric logistic-regression trial deterministically", async () => {
  await withTrialFixture(async ({ root, studyPath, outputRoot }) => {
    const header = "id,spread_bps,volume,big_move,future_mid\n";
    await writeFile(
      join(root, "data", "training.csv"),
      `${header}`
      + "train-1,1.0,100,0,0.10\n"
      + "train-2,1.3,,0,0.20\n"
      + "train-3,2.0,80,0,0.30\n"
      + "train-4,7.0,20,1,0.80\n"
      + "train-5,8.0,,1,0.90\n",
      "utf8",
    );
    await writeFile(
      join(root, "data", "validation.csv"),
      `${header}`
      + "validation-1,1.2,,0,0.15\n"
      + "validation-2,7.5,15,1,0.85\n",
      "utf8",
    );
    await writeFile(
      join(root, "data", "test.csv"),
      `${header}test-1,1.1,90,0,0.12\ntest-2,7.8,10,1,0.88\n`,
      "utf8",
    );
    const study = JSON.parse(await readFile(studyPath, "utf8")) as {
      task: { input_columns: string[] };
    };
    study.task.input_columns = ["spread_bps", "volume"];
    await writeFile(studyPath, `${JSON.stringify(study, null, 2)}\n`, "utf8");
    await writeStudyBenchmarkLock({ studyPath, force: true });

    const writeBuiltinTrial = async (id: string): Promise<string> => {
      const trialPath = join(root, `${id}.trial.json`);
      await writeFile(trialPath, `${JSON.stringify({
        schema_version: 1,
        id,
        name: "Balanced numeric logistic regression",
        runner: {
          builtin: "numeric_logistic_regression",
          timeout_ms: 120_000,
        },
        parameters: {
          c: 1,
          class_weight: "balanced",
          max_iter: 1_000,
          random_seed: 42,
        },
      }, null, 2)}\n`, "utf8");
      return trialPath;
    };

    const first = await runStudyTrial({
      studyPath,
      trialPath: await writeBuiltinTrial("numeric-logreg-v1"),
      outputRoot,
    });
    const second = await runStudyTrial({
      studyPath,
      trialPath: await writeBuiltinTrial("numeric-logreg-v2"),
      outputRoot,
    });

    assert.equal(first.report.evaluation.primary_score, 1);
    assert.deepEqual(first.report.evaluation.metrics, {
      average_precision: 1,
      roc_auc: 1,
      f1_at_0_5: 1,
    });
    assert.deepEqual(
      first.report.execution.command,
      ["builtin:numeric_logistic_regression"],
    );
    assert.equal(
      first.report.provenance.implementation.sha256,
      second.report.provenance.implementation.sha256,
    );
    assert.equal(
      first.report.provenance.implementation.evidence,
      "bundled_locked",
    );
    assert.equal(first.report.provenance.implementation.file_count, 2);
    const implementationManifest = JSON.parse(
      await readFile(
        join(first.trialDirectory, "implementation", "manifest.json"),
        "utf8",
      ),
    ) as {
      evidence: string;
      files: Array<{ role: string; path: string; snapshot_path: string }>;
    };
    assert.equal(implementationManifest.evidence, "bundled_locked");
    assert.deepEqual(
      implementationManifest.files.map((file) => ({
        role: file.role,
        path: file.path,
      })),
      [
        {
          role: "source",
          path: "training/study-runner/numeric_logistic_regression.py",
        },
        {
          role: "dependency_lock",
          path: "training/study-runner/numeric_logistic_regression.py.lock",
        },
      ],
    );
    for (const file of implementationManifest.files) {
      assert.equal(existsSync(join(first.trialDirectory, file.snapshot_path)), true);
    }
    assert.equal(
      first.report.evaluation.predictions_sha256,
      second.report.evaluation.predictions_sha256,
    );
    assert.equal(
      existsSync(join(first.trialDirectory, "model", "model.joblib")),
      true,
    );
    const manifest = JSON.parse(
      await readFile(
        join(first.trialDirectory, "model", "runner-manifest.json"),
        "utf8",
      ),
    ) as {
      runner: { name: string; version: number };
      training: {
        class_counts: { negative: number; positive: number };
        missing_counts: Record<string, number>;
      };
      model: { sha256: string; size_bytes: number };
    };
    assert.deepEqual(manifest.runner, {
      name: "numeric_logistic_regression",
      version: 2,
    });
    assert.deepEqual(manifest.training.class_counts, {
      negative: 3,
      positive: 2,
    });
    assert.deepEqual(manifest.training.missing_counts, {
      spread_bps: 0,
      volume: 2,
    });
    assert.equal(
      (await readFile(join(first.trialDirectory, "training.csv"), "utf8"))
        .includes("future_mid"),
      false,
    );
    assert.equal(
      (await readFile(join(first.trialDirectory, "validation.csv"), "utf8"))
        .includes("big_move"),
      false,
    );
    assert.match(manifest.model.sha256, /^[0-9a-f]{64}$/);
    assert.ok(manifest.model.size_bytes > 0);
    const modelManifest = JSON.parse(
      await readFile(join(first.trialDirectory, "model-manifest.json"), "utf8"),
    ) as { files: Array<{ path: string }> };
    assert.deepEqual(
      modelManifest.files.map((file) => file.path),
      ["model.joblib", "runner-manifest.json"],
    );
    assert.equal(first.report.provenance.model.file_count, 2);
    assert.ok(first.report.provenance.model.size_bytes > 0);

    const predictionInput = {
      protocol_version: 1,
      task: {
        type: "binary_classification",
        id_column: "id",
        input_columns: ["spread_bps", "volume"],
        prediction: {
          field: "probability",
          meaning: "probability_of_positive_target",
        },
      },
      dataset: {
        prediction_csv: join(first.trialDirectory, "validation.csv"),
      },
    };
    const predictionInputPath = join(root, "saved-model-prediction-input.json");
    await writeFile(
      predictionInputPath,
      `${JSON.stringify(predictionInput, null, 2)}\n`,
      "utf8",
    );
    const bundledCommand = buildStudyTrialRunnerCommand({
      builtin: "numeric_logistic_regression",
      timeout_ms: 120_000,
    });
    const runSavedModel = (
      inputPath: string,
      modelDirectory: string,
      outputName: string,
    ) => spawnSync(bundledCommand.command, [
      ...bundledCommand.baseArgs,
      "--input",
      inputPath,
      "--output",
      join(root, outputName),
      "--model-dir",
      modelDirectory,
    ], { cwd: root, encoding: "utf8" });
    const savedModel = runSavedModel(
      predictionInputPath,
      join(first.trialDirectory, "model"),
      "saved-model-predictions.json",
    );
    assert.equal(savedModel.status, 0, savedModel.stderr);
    assert.deepEqual(
      JSON.parse(await readFile(join(root, "saved-model-predictions.json"), "utf8")),
      JSON.parse(await readFile(join(first.trialDirectory, "predictions.json"), "utf8")),
    );

    const unknownRequestPath = join(root, "unknown-prediction-input.json");
    await writeFile(unknownRequestPath, `${JSON.stringify({
      ...predictionInput,
      unexpected: true,
    }, null, 2)}\n`, "utf8");
    const unknownRequest = runSavedModel(
      unknownRequestPath,
      join(first.trialDirectory, "model"),
      "unknown-request-predictions.json",
    );
    assert.equal(unknownRequest.status, 1);
    assert.match(
      unknownRequest.stderr,
      /prediction input has unknown fields.*unexpected/i,
    );

    const driftModelDirectory = join(root, "drift-model");
    await cp(
      join(first.trialDirectory, "model"),
      driftModelDirectory,
      { recursive: true },
    );
    const driftManifestPath = join(driftModelDirectory, "runner-manifest.json");
    const driftManifestText = await readFile(driftManifestPath, "utf8");
    const oldManifest = JSON.parse(driftManifestText);
    oldManifest.runner.version = 1;
    await chmod(driftManifestPath, 0o600);
    await writeFile(
      driftManifestPath,
      `${JSON.stringify(oldManifest, null, 2)}\n`,
      "utf8",
    );
    const oldRunner = runSavedModel(
      predictionInputPath,
      driftModelDirectory,
      "old-runner-predictions.json",
    );
    assert.equal(oldRunner.status, 1);
    assert.match(
      oldRunner.stderr,
      /must identify numeric_logistic_regression version 2/i,
    );

    await writeFile(driftManifestPath, driftManifestText, "utf8");
    const driftModelPath = join(driftModelDirectory, "model.joblib");
    const driftModel = await readFile(driftModelPath);
    driftModel[0] = driftModel[0]! ^ 1;
    await chmod(driftModelPath, 0o600);
    await writeFile(driftModelPath, driftModel);
    const changedModel = runSavedModel(
      predictionInputPath,
      driftModelDirectory,
      "changed-model-predictions.json",
    );
    assert.equal(changedModel.status, 1);
    assert.match(
      changedModel.stderr,
      /saved model SHA-256 does not match/i,
    );
  });
});

test("bundled numeric trials fail clearly on invalid parameters and features", async () => {
  await withTrialFixture(async ({ root, studyPath, outputRoot }) => {
    const writeBuiltinTrial = async (
      id: string,
      parameters: Record<string, unknown>,
    ): Promise<string> => {
      const trialPath = join(root, `${id}.trial.json`);
      await writeFile(trialPath, `${JSON.stringify({
        schema_version: 1,
        id,
        name: `Invalid fixture ${id}`,
        runner: {
          builtin: "numeric_logistic_regression",
          timeout_ms: 120_000,
        },
        parameters,
      }, null, 2)}\n`, "utf8");
      return trialPath;
    };
    const validParameters = {
      c: 1,
      class_weight: "balanced",
      max_iter: 1_000,
      random_seed: 42,
    };
    const expectRunnerFailure = async (
      id: string,
      parameters: Record<string, unknown>,
      expectedLog: RegExp,
    ): Promise<void> => {
      await assert.rejects(
        runStudyTrial({
          studyPath,
          trialPath: await writeBuiltinTrial(id, parameters),
          outputRoot,
        }),
        /exited with code 1/,
      );
      assert.match(
        await readFile(join(outputRoot, id, "trial.log"), "utf8"),
        expectedLog,
      );
      assert.equal(existsSync(join(outputRoot, id, "predictions.json")), false);
    };
    const expectSpecFailure = async (
      id: string,
      parameters: Record<string, unknown>,
      expectedError: RegExp,
    ): Promise<void> => {
      await assert.rejects(
        runStudyTrial({
          studyPath,
          trialPath: await writeBuiltinTrial(id, parameters),
          outputRoot,
        }),
        expectedError,
      );
      assert.equal(existsSync(join(outputRoot, id)), false);
    };

    await expectSpecFailure(
      "unknown-parameter",
      { ...validParameters, solver: "lbfgs" },
      /parameters: unrecognized key.*solver/i,
    );
    await expectSpecFailure(
      "invalid-c",
      { ...validParameters, c: 0 },
      /parameters\.c: too small/i,
    );
    await expectRunnerFailure(
      "nonnumeric-feature",
      validParameters,
      /column "note" has nonnumeric value "quiet"/,
    );

    const header = "id,spread_bps,note,big_move,future_mid\n";
    await writeFile(
      join(root, "data", "training.csv"),
      `${header}train-1,1.2,,0,0.10\ntrain-2,8.4,,1,0.90\n`,
      "utf8",
    );
    await writeFile(
      join(root, "data", "validation.csv"),
      `${header}validation-1,1.4,,0,0.20\nvalidation-2,7.9,,1,0.80\n`,
      "utf8",
    );
    await writeFile(
      join(root, "data", "test.csv"),
      `${header}test-1,1.1,,0,0.30\ntest-2,8.1,,1,0.70\n`,
      "utf8",
    );
    const study = JSON.parse(await readFile(studyPath, "utf8")) as {
      task: { input_columns: string[] };
    };
    study.task.input_columns = ["note"];
    await writeFile(studyPath, `${JSON.stringify(study, null, 2)}\n`, "utf8");
    await writeStudyBenchmarkLock({ studyPath, force: true });
    await expectRunnerFailure(
      "all-missing-feature",
      validParameters,
      /training numeric features are entirely missing: "note"/,
    );
  });
});

test("implementation provenance fails before launch or report publication on drift", async () => {
  await withTrialFixture(async ({ root, studyPath, outputRoot, writeTrial }) => {
    const missingSource = await writeTrial(
      "missing-source",
      validRunner,
      {
        provenance: {
          source_files: ["missing-runner.mjs"],
          dependency_lock_files: [],
        },
      },
    );
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: missingSource, outputRoot }),
      /implementation source must be a readable regular, non-symbolic file/i,
    );
    assert.equal(existsSync(join(outputRoot, "missing-source")), false);

    const outsideDirectory = join(root, "outside-source");
    await mkdir(outsideDirectory);
    await writeFile(join(outsideDirectory, "runner.mjs"), validRunner, "utf8");
    await symlink(outsideDirectory, join(root, "linked-source"), "dir");
    const linkedSource = await writeTrial(
      "linked-source",
      validRunner,
      {
        provenance: {
          source_files: ["linked-source/runner.mjs"],
          dependency_lock_files: [],
        },
      },
    );
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: linkedSource, outputRoot }),
      /provenance paths must not contain symbolic links/i,
    );
    assert.equal(existsSync(join(outputRoot, "linked-source")), false);

    if (process.platform !== "win32") {
      const fifoPath = join(root, "runner.fifo");
      execFileSync("mkfifo", [fifoPath]);
      const fifoSource = await writeTrial(
        "fifo-source",
        validRunner,
        {
          provenance: {
            source_files: ["runner.fifo"],
            dependency_lock_files: [],
          },
        },
      );
      await assert.rejects(
        runStudyTrial({ studyPath, trialPath: fifoSource, outputRoot }),
        /implementation source must be a regular file/i,
      );
      assert.equal(existsSync(join(outputRoot, "fifo-source")), false);
    }

    const sourceDrift = await writeTrial("source-drift", `
      import { writeFileSync } from "node:fs";
      const output = process.argv[process.argv.indexOf("--output") + 1];
      writeFileSync(output, JSON.stringify({
        protocol_version: 1,
        predictions: [
          { id: "validation-1", probability: 0.1 },
          { id: "validation-2", probability: 0.9 }
        ]
      }));
      writeFileSync(new URL(import.meta.url), "mutated");
    `);
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: sourceDrift, outputRoot }),
      /implementation .* changed after provenance capture/i,
    );
    assert.equal(
      existsSync(join(outputRoot, "source-drift", "trial-report.json")),
      false,
    );

    const dependencyPath = join(root, "dependencies.lock");
    await writeFile(dependencyPath, "version=1\n", "utf8");
    const dependencyDrift = await writeTrial("dependency-drift", `
      import { writeFileSync } from "node:fs";
      const output = process.argv[process.argv.indexOf("--output") + 1];
      writeFileSync(output, JSON.stringify({
        protocol_version: 1,
        predictions: [
          { id: "validation-1", probability: 0.1 },
          { id: "validation-2", probability: 0.9 }
        ]
      }));
      writeFileSync(${JSON.stringify(dependencyPath)}, "version=2\\n");
    `);
    const dependencyTrial = JSON.parse(
      await readFile(dependencyDrift, "utf8"),
    ) as {
      runner: {
        command: string[];
        provenance: {
          source_files: string[];
          dependency_lock_files: string[];
        };
      };
    };
    dependencyTrial.runner.provenance = {
      source_files: [basename(dependencyTrial.runner.command[1]!)],
      dependency_lock_files: ["dependencies.lock"],
    };
    await writeFile(
      dependencyDrift,
      `${JSON.stringify(dependencyTrial, null, 2)}\n`,
      "utf8",
    );
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: dependencyDrift, outputRoot }),
      /implementation dependencies\.lock changed after provenance capture/i,
    );
    assert.equal(
      existsSync(join(outputRoot, "dependency-drift", "trial-report.json")),
      false,
    );
    assert.equal(
      await readFile(
        join(
          outputRoot,
          "dependency-drift",
          "implementation",
          "dependency-lock",
          "dependencies.lock",
        ),
        "utf8",
      ),
      "version=1\n",
    );

    const snapshotDrift = await writeTrial("snapshot-drift", `
      import { chmodSync, writeFileSync } from "node:fs";
      import { basename, dirname, join } from "node:path";
      import { fileURLToPath } from "node:url";
      const value = (name) => process.argv[process.argv.indexOf(name) + 1];
      writeFileSync(value("--output"), JSON.stringify({
        protocol_version: 1,
        predictions: [
          { id: "validation-1", probability: 0.1 },
          { id: "validation-2", probability: 0.9 }
        ]
      }));
      const trialDirectory = dirname(value("--artifact-dir"));
      const sourceName = basename(fileURLToPath(import.meta.url));
      const snapshot = join(
        trialDirectory,
        "implementation",
        "source",
        sourceName
      );
      chmodSync(snapshot, 0o600);
      writeFileSync(snapshot, "tampered");
    `);
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: snapshotDrift, outputRoot }),
      /implementation snapshot changed during trial execution/i,
    );
    assert.equal(
      existsSync(join(outputRoot, "snapshot-drift", "trial-report.json")),
      false,
    );
  });
});

test("manifests nested and empty model trees and rejects symbolic links", async () => {
  await withTrialFixture(async ({ studyPath, outputRoot, writeTrial }) => {
    const nested = await writeTrial("nested-model", `
      import { linkSync, mkdirSync, writeFileSync } from "node:fs";
      import { join } from "node:path";
      const value = (name) => process.argv[process.argv.indexOf(name) + 1];
      const model = value("--artifact-dir");
      mkdirSync(join(model, "nested"));
      writeFileSync(join(model, "z.txt"), "last");
      writeFileSync(join(model, "nested", "a.txt"), "first");
      linkSync(join(model, "z.txt"), join(model, "nested", "z-link.txt"));
      writeFileSync(value("--output"), JSON.stringify({
        protocol_version: 1,
        predictions: [
          { id: "validation-1", probability: 0.1 },
          { id: "validation-2", probability: 0.9 }
        ]
      }));
    `);
    const nestedResult = await runStudyTrial({
      studyPath,
      trialPath: nested,
      outputRoot,
    });
    const nestedManifest = JSON.parse(
      await readFile(join(nestedResult.trialDirectory, "model-manifest.json"), "utf8"),
    ) as { files: Array<{ path: string }>; file_count: number; size_bytes: number };
    assert.deepEqual(
      nestedManifest.files.map((file) => file.path),
      ["nested/a.txt", "nested/z-link.txt", "z.txt"],
    );
    assert.equal(nestedManifest.file_count, 3);
    assert.equal(nestedManifest.size_bytes, 13);

    const empty = await writeTrial("empty-model", `
      import { writeFileSync } from "node:fs";
      const output = process.argv[process.argv.indexOf("--output") + 1];
      writeFileSync(output, JSON.stringify({
        protocol_version: 1,
        predictions: [
          { id: "validation-1", probability: 0.1 },
          { id: "validation-2", probability: 0.9 }
        ]
      }));
    `);
    const emptyResult = await runStudyTrial({
      studyPath,
      trialPath: empty,
      outputRoot,
    });
    assert.deepEqual(emptyResult.report.provenance.model, {
      manifest: "model-manifest.json",
      sha256: emptyResult.report.provenance.model.sha256,
      file_count: 0,
      size_bytes: 0,
    });

    const symbolic = await writeTrial("symbolic-model", `
      import { symlinkSync, writeFileSync } from "node:fs";
      import { join } from "node:path";
      const value = (name) => process.argv[process.argv.indexOf(name) + 1];
      symlinkSync(value("--input"), join(value("--artifact-dir"), "linked-model"));
      writeFileSync(value("--output"), JSON.stringify({
        protocol_version: 1,
        predictions: [
          { id: "validation-1", probability: 0.1 },
          { id: "validation-2", probability: 0.9 }
        ]
      }));
    `);
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: symbolic, outputRoot }),
      /model artifacts must not contain symbolic links/i,
    );
    assert.equal(
      existsSync(join(outputRoot, "symbolic-model", "trial-report.json")),
      false,
    );

    const deep = await writeTrial("deep-model", `
      import { mkdirSync, writeFileSync } from "node:fs";
      import { join } from "node:path";
      const value = (name) => process.argv[process.argv.indexOf(name) + 1];
      let directory = value("--artifact-dir");
      for (let depth = 0; depth < 65; depth += 1) {
        directory = join(directory, "d");
        mkdirSync(directory);
      }
      writeFileSync(value("--output"), JSON.stringify({
        protocol_version: 1,
        predictions: [
          { id: "validation-1", probability: 0.1 },
          { id: "validation-2", probability: 0.9 }
        ]
      }));
    `);
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: deep, outputRoot }),
      /exceeds the 64-level depth limit/i,
    );
    assert.equal(
      existsSync(join(outputRoot, "deep-model", "trial-report.json")),
      false,
    );
  });
});

test("directory replacement cannot redirect report publication", async () => {
  await withTrialFixture(async ({ studyPath, outputRoot, writeTrial }) => {
    const modelReplacement = await writeTrial("replace-model", `
      import { mkdirSync, renameSync, writeFileSync } from "node:fs";
      const value = (name) => process.argv[process.argv.indexOf(name) + 1];
      const model = value("--artifact-dir");
      renameSync(model, model + ".moved");
      mkdirSync(model);
      writeFileSync(value("--output"), JSON.stringify({
        protocol_version: 1,
        predictions: [
          { id: "validation-1", probability: 0.1 },
          { id: "validation-2", probability: 0.9 }
        ]
      }));
    `);
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: modelReplacement, outputRoot }),
      /model artifacts directory was replaced/i,
    );
    assert.equal(
      existsSync(join(outputRoot, "replace-model", "trial-report.json")),
      false,
    );

    const trialReplacement = await writeTrial("replace-trial", `
      import { renameSync, symlinkSync, writeFileSync } from "node:fs";
      import { dirname } from "node:path";
      const value = (name) => process.argv[process.argv.indexOf(name) + 1];
      const trial = dirname(value("--artifact-dir"));
      const moved = trial + ".moved";
      renameSync(trial, moved);
      symlinkSync(moved, trial, "dir");
      writeFileSync(value("--output"), JSON.stringify({
        protocol_version: 1,
        predictions: [
          { id: "validation-1", probability: 0.1 },
          { id: "validation-2", probability: 0.9 }
        ]
      }));
    `);
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: trialReplacement, outputRoot }),
      /trial directory was replaced during trial execution/i,
    );
    assert.equal(
      existsSync(join(outputRoot, "replace-trial.moved", "trial-report.json")),
      false,
    );
  });
});

test("rejects malformed, incomplete, forged, and non-regular prediction outputs", async () => {
  await withTrialFixture(async ({ studyPath, outputRoot, writeTrial }) => {
    const cases = [
      {
        id: "missing",
        output: {
          protocol_version: 1,
          predictions: [{ id: "validation-1", probability: 0.1 }],
        },
        error: /returned 1 predictions for 2 validation examples/,
      },
      {
        id: "duplicate",
        output: {
          protocol_version: 1,
          predictions: [
            { id: "validation-1", probability: 0.1 },
            { id: "validation-1", probability: 0.9 },
          ],
        },
        error: /duplicate validation ID "validation-1"/,
      },
      {
        id: "unknown",
        output: {
          protocol_version: 1,
          predictions: [
            { id: "validation-1", probability: 0.1 },
            { id: "not-validation", probability: 0.9 },
          ],
        },
        error: /unknown validation ID "not-validation"/,
      },
      {
        id: "out-of-range",
        output: {
          protocol_version: 1,
          predictions: [
            { id: "validation-1", probability: -0.1 },
            { id: "validation-2", probability: 0.9 },
          ],
        },
        error: /expected number to be >=0/i,
      },
      {
        id: "forged-metric",
        output: {
          protocol_version: 1,
          metrics: { average_precision: 1 },
          predictions: [
            { id: "validation-1", probability: 0.1 },
            { id: "validation-2", probability: 0.9 },
          ],
        },
        error: /unrecognized key.*metrics/i,
      },
    ];

    for (const fixture of cases) {
      const script = `
        import { writeFileSync } from "node:fs";
        const output = process.argv[process.argv.indexOf("--output") + 1];
        writeFileSync(output, ${JSON.stringify(JSON.stringify(fixture.output))});
      `;
      const trialPath = await writeTrial(fixture.id, script);
      await assert.rejects(
        runStudyTrial({ studyPath, trialPath, outputRoot }),
        fixture.error,
      );
      assert.equal(
        existsSync(join(outputRoot, fixture.id, "trial-report.json")),
        false,
      );
    }

    const symlinkTrial = await writeTrial("symlink", `
      import { symlinkSync } from "node:fs";
      const input = process.argv[process.argv.indexOf("--input") + 1];
      const output = process.argv[process.argv.indexOf("--output") + 1];
      symlinkSync(input, output);
    `);
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: symlinkTrial, outputRoot }),
      /regular, non-symbolic file/,
    );

    if (process.platform !== "win32") {
      const fifoTrial = await writeTrial("fifo", `
        import { execFileSync } from "node:child_process";
        const output = process.argv[process.argv.indexOf("--output") + 1];
        execFileSync("mkfifo", [output]);
      `);
      await assert.rejects(
        runStudyTrial({ studyPath, trialPath: fifoTrial, outputRoot }),
        /predictions must be a regular, non-symbolic file/i,
      );
    }
  });
});

test("nonzero, missing-output, and timed-out commands cannot publish reports", async () => {
  await withTrialFixture(async ({ studyPath, outputRoot, writeTrial }) => {
    const nonzero = await writeTrial("nonzero", `
      import { writeFileSync } from "node:fs";
      const output = process.argv[process.argv.indexOf("--output") + 1];
      writeFileSync(output, JSON.stringify({
        protocol_version: 1,
        predictions: [
          { id: "validation-1", probability: 0.1 },
          { id: "validation-2", probability: 0.9 }
        ]
      }));
      process.exit(7);
    `);
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: nonzero, outputRoot }),
      /exited with code 7/,
    );
    assert.equal(existsSync(join(outputRoot, "nonzero", "trial-report.json")), false);

    const missing = await writeTrial("no-output", "process.exit(0);");
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: missing, outputRoot }),
      /did not write predictions/,
    );

    const timedOut = await writeTrial(
      "timeout",
      "setInterval(() => {}, 10_000);",
      { timeout_ms: 1_000 },
    );
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: timedOut, outputRoot }),
      /timed out after 1000ms/,
    );
    assert.equal(existsSync(join(outputRoot, "timeout", "trial-report.json")), false);
  });
});

test("benchmark drift prevents launch and trial IDs are immutable", async () => {
  await withTrialFixture(async ({
    root,
    studyPath,
    outputRoot,
    validationPath,
    writeTrial,
  }) => {
    const valid = await writeTrial("immutable", validRunner);
    await runStudyTrial({ studyPath, trialPath: valid, outputRoot });
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: valid, outputRoot }),
      /already exists.*use a new trial ID/i,
    );

    const marker = join(root, "launched");
    const driftTrial = await writeTrial("drift", `
      import { writeFileSync } from "node:fs";
      writeFileSync(${JSON.stringify(marker)}, "yes");
    `);
    const current = await readFile(validationPath, "utf8");
    await writeFile(validationPath, current.replace("7.9", "7.8"), "utf8");
    await assert.rejects(
      runStudyTrial({ studyPath, trialPath: driftTrial, outputRoot }),
      /benchmark lock drift detected/i,
    );
    assert.equal(existsSync(marker), false);
    assert.equal(existsSync(join(outputRoot, "drift")), false);
  });
});
