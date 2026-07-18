import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import {
  chmod,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  defaultStudyCandidateDirectory,
  promoteStudyTrialCandidate,
  studyCandidateLockSchema,
  verifyStudyCandidateArtifacts,
} from "../src/study-candidates.js";
import { runStudyTrial } from "../src/study-trials.js";
import { writeStudyBenchmarkLock } from "../src/studies.js";

const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const cliPath = join(projectRoot, "src", "index.ts");
const tsxLoader = import.meta.resolve("tsx");

interface CandidateFixture {
  root: string;
  studyPath: string;
  trialPath: string;
  outputRoot: string;
}

async function withCandidateFixture(
  callback: (fixture: CandidateFixture) => void | Promise<void>,
): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), "tt-local-study-candidate-test-"));
  try {
    const dataDirectory = join(root, "data");
    await mkdir(dataDirectory);
    const header = "id,spread_bps,volume,big_move,future_mid\n";
    await writeFile(
      join(dataDirectory, "training.csv"),
      `${header}`
      + "train-1,1.0,100,0,0.10\n"
      + "train-2,1.3,,0,0.20\n"
      + "train-3,2.0,80,0,0.30\n"
      + "train-4,7.0,20,1,0.80\n"
      + "train-5,8.0,,1,0.90\n",
      "utf8",
    );
    await writeFile(
      join(dataDirectory, "validation.csv"),
      `${header}`
      + "validation-1,1.2,,0,0.15\n"
      + "validation-2,7.5,15,1,0.85\n",
      "utf8",
    );
    await writeFile(
      join(dataDirectory, "test.csv"),
      `${header}`
      + "test-1,1.1,90,0,0.12\n"
      + "test-2,7.8,10,1,0.88\n",
      "utf8",
    );
    const studyPath = join(root, "portfolio.study.json");
    await writeFile(studyPath, `${JSON.stringify({
      schema_version: 1,
      name: "Portfolio anomaly benchmark",
      task: {
        type: "binary_classification",
        id_column: "id",
        input_columns: ["spread_bps", "volume"],
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

    const trialPath = join(root, "numeric-logreg.trial.json");
    await writeFile(trialPath, `${JSON.stringify({
      schema_version: 1,
      id: "numeric-logreg",
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
    await callback({
      root,
      studyPath,
      trialPath,
      outputRoot: join(root, "trial-output"),
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

async function runCandidateFixture(fixture: CandidateFixture) {
  return runStudyTrial({
    studyPath: fixture.studyPath,
    trialPath: fixture.trialPath,
    outputRoot: fixture.outputRoot,
  });
}

function runCandidateCli(args: string[], cwd: string) {
  return spawnSync(
    process.execPath,
    ["--import", tsxLoader, cliPath, ...args],
    { cwd, encoding: "utf8" },
  );
}

test("promotes a copied fitted candidate only after saved-model replay parity", async () => {
  await withCandidateFixture(async (fixture) => {
    const trial = await runCandidateFixture(fixture);
    const promoted = await promoteStudyTrialCandidate({
      studyPath: fixture.studyPath,
      trialPath: fixture.trialPath,
      trialDirectory: trial.trialDirectory,
    });

    assert.equal(
      promoted.candidateDirectory,
      defaultStudyCandidateDirectory(fixture.studyPath),
    );
    assert.equal(studyCandidateLockSchema.safeParse(promoted.lock).success, true);
    assert.equal(studyCandidateLockSchema.safeParse({
      ...promoted.lock,
      replay: {
        ...promoted.lock.replay,
        prediction_count: promoted.lock.replay.prediction_count + 1,
        max_absolute_difference: (
          promoted.lock.replay.probability_tolerance * 2
        ),
      },
    }).success, false);
    assert.equal(promoted.lock.trial.id, "numeric-logreg");
    assert.equal(promoted.lock.selection.validation_primary_score, 1);
    assert.equal(promoted.lock.replay.prediction_count, 2);
    assert.ok(
      promoted.lock.replay.max_absolute_difference
      <= promoted.lock.replay.probability_tolerance,
    );
    assert.deepEqual(promoted.lock.training_runtime, JSON.parse(
      await readFile(
        join(promoted.candidateDirectory, "model", "runner-manifest.json"),
        "utf8",
      ),
    ).runtime);

    const sourceModel = await stat(
      join(trial.trialDirectory, "model", "model.joblib"),
    );
    const promotedModel = await stat(
      join(promoted.candidateDirectory, "model", "model.joblib"),
    );
    assert.equal(promotedModel.nlink, 1);
    assert.notDeepEqual(
      { dev: promotedModel.dev, ino: promotedModel.ino },
      { dev: sourceModel.dev, ino: sourceModel.ino },
    );
    const replayCsv = await readFile(
      join(promoted.candidateDirectory, "replay", "validation.csv"),
      "utf8",
    );
    assert.equal(replayCsv.includes("big_move"), false);
    assert.equal(replayCsv.includes("future_mid"), false);
    const replayInput = JSON.parse(await readFile(
      join(promoted.candidateDirectory, "replay", "predictor-input.json"),
      "utf8",
    ));
    assert.deepEqual(Object.keys(replayInput.dataset), ["prediction_csv"]);
    assert.equal("target_column" in replayInput.task, false);
    assert.equal("target_values" in replayInput.task, false);

    await assert.rejects(
      promoteStudyTrialCandidate({
        studyPath: fixture.studyPath,
        trialPath: fixture.trialPath,
        trialDirectory: trial.trialDirectory,
      }),
      /candidate selection already exists.*write-once/i,
    );

    const candidateLockPath = join(
      promoted.candidateDirectory,
      "candidate.lock.json",
    );
    await chmod(candidateLockPath, 0o600);
    const writeCandidateLock = async (lock: unknown) => {
      await writeFile(
        candidateLockPath,
        `${JSON.stringify(lock, null, 2)}\n`,
        "utf8",
      );
    };

    const scoreDrift = structuredClone(promoted.lock);
    scoreDrift.selection.validation_primary_score = 0.5;
    await writeCandidateLock(scoreDrift);
    await assert.rejects(
      verifyStudyCandidateArtifacts({
        candidateDirectory: promoted.candidateDirectory,
      }),
      /validation selection does not match.*trial report/i,
    );

    const runtimeDrift = structuredClone(promoted.lock);
    runtimeDrift.training_runtime.python += "-tampered";
    await writeCandidateLock(runtimeDrift);
    await assert.rejects(
      verifyStudyCandidateArtifacts({
        candidateDirectory: promoted.candidateDirectory,
      }),
      /training runtime does not match/i,
    );

    const replayDifferenceDrift = structuredClone(promoted.lock);
    replayDifferenceDrift.replay.max_absolute_difference = (
      replayDifferenceDrift.replay.probability_tolerance / 2
    );
    await writeCandidateLock(replayDifferenceDrift);
    await assert.rejects(
      verifyStudyCandidateArtifacts({
        candidateDirectory: promoted.candidateDirectory,
      }),
      /replay difference does not match/i,
    );
    await writeCandidateLock(promoted.lock);

    const reportArtifactPath = join(
      promoted.candidateDirectory,
      promoted.lock.trial.report_artifact.path,
    );
    const reportArtifact = await readFile(reportArtifactPath);
    const forgedReport = JSON.parse(reportArtifact.toString("utf8"));
    forgedReport.evaluation.metrics.average_precision = 0.5;
    const forgedReportArtifact = Buffer.from(
      `${JSON.stringify(forgedReport, null, 2)}\n`,
    );
    await chmod(reportArtifactPath, 0o600);
    await writeFile(reportArtifactPath, forgedReportArtifact);
    const reportMetricDrift = structuredClone(promoted.lock);
    reportMetricDrift.trial.report_artifact.sha256 = createHash("sha256")
      .update(forgedReportArtifact)
      .digest("hex");
    reportMetricDrift.trial.report_artifact.size_bytes = (
      forgedReportArtifact.byteLength
    );
    await writeCandidateLock(reportMetricDrift);
    await assert.rejects(
      verifyStudyCandidateArtifacts({
        candidateDirectory: promoted.candidateDirectory,
      }),
      /report primary score does not match.*primary metric/i,
    );
    await writeFile(reportArtifactPath, reportArtifact);
    await chmod(reportArtifactPath, 0o400);
    await writeCandidateLock(promoted.lock);

    const predictorScriptPath = join(
      promoted.candidateDirectory,
      promoted.lock.artifacts.predictor.script.path,
    );
    const predictorScript = await readFile(predictorScriptPath);
    const forgedPredictorScript = Buffer.concat([
      predictorScript,
      Buffer.from("\n# forged after promotion\n"),
    ]);
    await chmod(predictorScriptPath, 0o600);
    await writeFile(predictorScriptPath, forgedPredictorScript);
    const predictorDrift = structuredClone(promoted.lock);
    predictorDrift.artifacts.predictor.script.sha256 = createHash("sha256")
      .update(forgedPredictorScript)
      .digest("hex");
    predictorDrift.artifacts.predictor.script.size_bytes = (
      forgedPredictorScript.byteLength
    );
    await writeCandidateLock(predictorDrift);
    await assert.rejects(
      verifyStudyCandidateArtifacts({
        candidateDirectory: promoted.candidateDirectory,
      }),
      /predictor implementation provenance does not match/i,
    );
    await writeFile(predictorScriptPath, predictorScript);
    await chmod(predictorScriptPath, 0o400);
    await writeCandidateLock(promoted.lock);

    const shadowModulePath = join(
      promoted.candidateDirectory,
      "predictor",
      "joblib.py",
    );
    await writeFile(shadowModulePath, "raise RuntimeError('shadowed')\n", "utf8");
    await assert.rejects(
      verifyStudyCandidateArtifacts({
        candidateDirectory: promoted.candidateDirectory,
      }),
      /predictor file set does not match/i,
    );
    await rm(shadowModulePath);

    const replayLogPath = join(
      promoted.candidateDirectory,
      promoted.lock.replay.log.path,
    );
    const replayLog = await readFile(replayLogPath, "utf8");
    await chmod(replayLogPath, 0o600);
    await writeFile(
      replayLogPath,
      `${replayLog[0] === "X" ? "Y" : "X"}${replayLog.slice(1)}`,
      "utf8",
    );
    await assert.rejects(
      verifyStudyCandidateArtifacts({
        candidateDirectory: promoted.candidateDirectory,
      }),
      /replay\/predictor\.log.*recorded provenance/i,
    );
  });
});

test("rejects command-backed trials before creating a candidate selection", async () => {
  await withCandidateFixture(async (fixture) => {
    const runnerPath = join(fixture.root, "runner.mjs");
    await writeFile(runnerPath, "process.exit(0);\n", "utf8");
    const commandTrialPath = join(fixture.root, "command.trial.json");
    await writeFile(commandTrialPath, `${JSON.stringify({
      schema_version: 1,
      id: "command-trial",
      name: "Command trial",
      runner: {
        command: [process.execPath, runnerPath],
        cwd: ".",
        timeout_ms: 10_000,
        provenance: {
          source_files: ["runner.mjs"],
          dependency_lock_files: [],
        },
      },
      parameters: {},
    }, null, 2)}\n`, "utf8");

    await assert.rejects(
      promoteStudyTrialCandidate({
        studyPath: fixture.studyPath,
        trialPath: commandTrialPath,
      }),
      /command-backed trials need a reusable predictor contract/i,
    );
    assert.equal(
      existsSync(defaultStudyCandidateDirectory(fixture.studyPath)),
      false,
    );
  });
});

test("runs and promotes the bundled candidate through the public CLI", async () => {
  await withCandidateFixture(async (fixture) => {
    const trial = runCandidateCli([
      "studies",
      "run",
      fixture.studyPath,
      fixture.trialPath,
      "--output-root",
      fixture.outputRoot,
    ], fixture.root);
    assert.equal(trial.status, 0, trial.stderr);
    const trialOutput = JSON.parse(trial.stdout);

    const promoted = runCandidateCli([
      "studies",
      "promote",
      fixture.studyPath,
      fixture.trialPath,
      "--trial-directory",
      trialOutput.trial_directory,
    ], fixture.root);
    assert.equal(promoted.status, 0, promoted.stderr);
    const output = JSON.parse(promoted.stdout);
    assert.equal(output.ok, true);
    assert.equal(output.status, "promoted");
    assert.equal(
      output.candidate_directory,
      defaultStudyCandidateDirectory(fixture.studyPath),
    );
    assert.equal(existsSync(output.candidate_lock_path), true);
    await verifyStudyCandidateArtifacts({
      candidateDirectory: output.candidate_directory,
    });
  });
});

test("strictly rejects report drift before claiming the candidate directory", async () => {
  await withCandidateFixture(async (fixture) => {
    const trial = await runCandidateFixture(fixture);
    const reportPath = join(trial.trialDirectory, "trial-report.json");
    const report = JSON.parse(await readFile(reportPath, "utf8"));
    report.untrusted_note = "ignore me";
    await chmod(reportPath, 0o600);
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");

    await assert.rejects(
      promoteStudyTrialCandidate({
        studyPath: fixture.studyPath,
        trialPath: fixture.trialPath,
        trialDirectory: trial.trialDirectory,
      }),
      /invalid Study trial report.*unrecognized key/i,
    );
    assert.equal(
      existsSync(defaultStudyCandidateDirectory(fixture.studyPath)),
      false,
    );
  });
});

test("allows only one concurrent promotion and cleans the losing staging directory", async () => {
  await withCandidateFixture(async (fixture) => {
    const trial = await runCandidateFixture(fixture);
    const attempts = await Promise.allSettled([
      promoteStudyTrialCandidate({
        studyPath: fixture.studyPath,
        trialPath: fixture.trialPath,
        trialDirectory: trial.trialDirectory,
      }),
      promoteStudyTrialCandidate({
        studyPath: fixture.studyPath,
        trialPath: fixture.trialPath,
        trialDirectory: trial.trialDirectory,
      }),
    ]);
    assert.equal(
      attempts.filter((attempt) => attempt.status === "fulfilled").length,
      1,
    );
    assert.equal(
      attempts.filter((attempt) => attempt.status === "rejected").length,
      1,
    );
    assert.deepEqual(
      await readdir(join(fixture.root, ".tt-local", "study-candidates")),
      ["portfolio.study.json"],
    );
  });
});
