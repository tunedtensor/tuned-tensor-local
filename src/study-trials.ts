import { mkdir, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { writeJsonAtomic } from "./artifacts.js";
import { minimalMachineLearningEnvironment } from "./huggingface-cache.js";
import { runLoggedProcess } from "./process-runner.js";
import { computeBinaryClassificationMetrics } from "./study-metrics.js";
import {
  STUDY_TRIAL_PROTOCOL_VERSION,
  assertDirectory,
  assertRegularFile,
  buildStudyTrialRunnerCommand,
  canonicalJson,
  claimTrialDirectory,
  defaultStudyTrialOutputRoot,
  loadStudyTrialSpec,
  parsePredictions,
  prepareTrialData,
  primaryScore,
  readStableRegularFile,
  sha256,
  studyImplementationInput,
  trialProtocolInput,
  type StudyTrialReport,
} from "./study-trial-core.js";
import {
  captureDirectoryIdentity,
  captureStudyImplementation,
  prepareStudyImplementation,
  verifyDirectoryIdentity,
  verifyStudyImplementation,
  writeStudyModelManifest,
} from "./study-provenance.js";
import { loadStudySpec, validateStudyBenchmark } from "./studies.js";

export {
  STUDY_TRIAL_PROTOCOL_VERSION,
  buildStudyTrialRunnerCommand,
  defaultStudyTrialOutputRoot,
  loadStudyTrialSpec,
  studyTrialOutputSchema,
  studyTrialPredictionSchema,
  studyTrialReportSchema,
  studyTrialSpecSchema,
  type StudyTrialJsonValue,
  type StudyTrialOutput,
  type StudyTrialReport,
  type StudyTrialRunnerCommand,
  type StudyTrialSpec,
} from "./study-trial-core.js";

export async function runStudyTrial(args: {
  studyPath: string;
  trialPath: string;
  lockPath?: string;
  outputRoot?: string;
}): Promise<{
  trialDirectory: string;
  reportPath: string;
  report: StudyTrialReport;
}> {
  const loadedTrial = await loadStudyTrialSpec(args.trialPath);
  const validated = await validateStudyBenchmark({
    studyPath: args.studyPath,
    lockPath: args.lockPath,
  });
  const loadedStudy = await loadStudySpec(args.studyPath);
  if (sha256(JSON.stringify(loadedStudy.spec)) !== validated.lock.study_spec_sha256) {
    throw new Error("StudySpec changed after benchmark validation; rerun the trial");
  }
  const prepared = await prepareTrialData({
    study: loadedStudy.spec,
    studyPath: loadedStudy.path,
    lock: validated.lock,
  });
  const runnerCommand = buildStudyTrialRunnerCommand(loadedTrial.spec.runner);
  await Promise.all(runnerCommand.requiredFiles.map((path) => (
    assertRegularFile(path, "bundled study trial runtime")
  )));
  const preparedImplementation = await prepareStudyImplementation(
    studyImplementationInput(loadedTrial),
  );
  const configuredCwd = "cwd" in loadedTrial.spec.runner
    ? loadedTrial.spec.runner.cwd
    : undefined;
  const configuredRunnerCwd = configuredCwd
    ? resolve(dirname(loadedTrial.path), configuredCwd)
    : undefined;
  if (configuredRunnerCwd) {
    await assertDirectory(configuredRunnerCwd, "study trial working directory");
  }

  const outputRoot = resolve(
    args.outputRoot ?? defaultStudyTrialOutputRoot(loadedStudy.path),
  );
  const trialDirectory = await claimTrialDirectory(outputRoot, loadedTrial.spec.id);
  const outputRootIdentity = await captureDirectoryIdentity(outputRoot);
  const trialDirectoryIdentity = await captureDirectoryIdentity(trialDirectory);
  const trainingPath = join(trialDirectory, "training.csv");
  const validationPath = join(trialDirectory, "validation.csv");
  const inputPath = join(trialDirectory, "trial-input.json");
  const predictionPath = join(trialDirectory, "predictions.json");
  const artifactDirectory = join(trialDirectory, "model");
  const logPath = join(trialDirectory, "trial.log");
  const reportPath = join(trialDirectory, "trial-report.json");
  await mkdir(artifactDirectory, { mode: 0o700 });
  const modelDirectoryIdentity = await captureDirectoryIdentity(artifactDirectory);
  const capturedImplementation = await captureStudyImplementation({
    trialDirectory,
    prepared: preparedImplementation,
  });
  await Promise.all([
    writeFile(trainingPath, prepared.trainingCsv, { encoding: "utf8", mode: 0o400, flag: "wx" }),
    writeFile(validationPath, prepared.validationCsv, { encoding: "utf8", mode: 0o400, flag: "wx" }),
  ]);
  const projectedHashes = {
    training: sha256(prepared.trainingCsv),
    validation: sha256(prepared.validationCsv),
  };
  await writeFile(
    inputPath,
    `${JSON.stringify(trialProtocolInput({
      trial: loadedTrial.spec,
      task: loadedStudy.spec.task,
      trainingPath,
      validationPath,
    }), null, 2)}\n`,
    { encoding: "utf8", mode: 0o400, flag: "wx" },
  );

  const runnerCwd = configuredRunnerCwd ?? trialDirectory;
  const commandArgs = [
    ...runnerCommand.baseArgs,
    "--input",
    inputPath,
    "--output",
    predictionPath,
    "--artifact-dir",
    artifactDirectory,
  ];
  await rm(predictionPath, { force: true });
  const started = performance.now();
  const result = await runLoggedProcess({
    command: runnerCommand.command,
    commandArgs,
    cwd: runnerCwd,
    env: minimalMachineLearningEnvironment(process.env),
    logPath,
    timeoutMs: loadedTrial.spec.runner.timeout_ms,
    timeoutMessage: (
      `Study trial "${loadedTrial.spec.id}" timed out after `
      + `${loadedTrial.spec.runner.timeout_ms}ms`
    ),
    terminateProcessGroupOnExit: true,
    stage: "study-trial",
  });
  const durationMs = Math.round(performance.now() - started);
  if (result.exitCode !== 0) {
    throw new Error(
      `Study trial "${loadedTrial.spec.id}" exited with code ${result.exitCode}; see ${logPath}`,
    );
  }
  await Promise.all([
    verifyDirectoryIdentity({
      path: outputRoot,
      expected: outputRootIdentity,
      description: "Study trial output root",
    }),
    verifyDirectoryIdentity({
      path: trialDirectory,
      expected: trialDirectoryIdentity,
      description: "Study trial directory",
    }),
  ]);
  await verifyStudyImplementation({
    trialDirectory,
    captured: capturedImplementation,
  });

  const [trainingAfter, validationAfter] = await Promise.all([
    readStableRegularFile({
      path: trainingPath,
      description: "projected training dataset",
      maxBytes: Buffer.byteLength(prepared.trainingCsv),
    }),
    readStableRegularFile({
      path: validationPath,
      description: "projected validation dataset",
      maxBytes: Buffer.byteLength(prepared.validationCsv),
    }),
  ]);
  if (
    sha256(trainingAfter) !== projectedHashes.training
    || sha256(validationAfter) !== projectedHashes.validation
  ) {
    throw new Error(
      `Study trial "${loadedTrial.spec.id}" modified its projected dataset inputs`,
    );
  }
  const parsedPredictions = await parsePredictions(
    predictionPath,
    prepared.validationLabels,
  );
  const metrics = computeBinaryClassificationMetrics(parsedPredictions.rows);
  await verifyDirectoryIdentity({
    path: trialDirectory,
    expected: trialDirectoryIdentity,
    description: "Study trial directory",
  });
  const modelProvenance = await writeStudyModelManifest({
    trialDirectory,
    modelDirectory: artifactDirectory,
    expectedRoot: modelDirectoryIdentity,
  });
  const report: StudyTrialReport = {
    schema_version: 1,
    protocol_version: STUDY_TRIAL_PROTOCOL_VERSION,
    trial: {
      id: loadedTrial.spec.id,
      name: loadedTrial.spec.name,
      spec_sha256: sha256(canonicalJson(loadedTrial.spec)),
      parameters: loadedTrial.spec.parameters,
    },
    study: {
      name: loadedStudy.spec.name,
      study_spec_sha256: validated.lock.study_spec_sha256,
      benchmark_lock_sha256: sha256(canonicalJson(validated.lock)),
      task_type: loadedStudy.spec.task.type,
      primary_metric: loadedStudy.spec.task.primary_metric,
    },
    data: {
      training: {
        source_sha256: prepared.source.training.sha256,
        projected_sha256: projectedHashes.training,
        row_count: prepared.source.training.rowCount,
      },
      validation: {
        source_sha256: prepared.source.validation.sha256,
        projected_sha256: projectedHashes.validation,
        row_count: prepared.source.validation.rowCount,
      },
      ...(validated.lock.dataset.temporal ? {
        temporal: {
          policy: validated.lock.dataset.temporal.policy,
          event_time_column: validated.lock.dataset.temporal.event_time_column,
          label_end_time_column: validated.lock.dataset.temporal.label_end_time_column,
          label_horizon_seconds: validated.lock.dataset.temporal.label_horizon_seconds,
          embargo_seconds: validated.lock.dataset.temporal.embargo_seconds,
          splits: {
            training: validated.lock.dataset.temporal.splits.training,
            validation: validated.lock.dataset.temporal.splits.validation,
          },
        },
      } : {}),
    },
    evaluation: {
      score_semantics: "positive_class_probability",
      primary_score: primaryScore(loadedStudy.spec.task.primary_metric, metrics),
      metrics,
      decision_threshold: 0.5,
      prediction_count: parsedPredictions.rows.length,
      predictions_sha256: sha256(parsedPredictions.bytes),
    },
    provenance: {
      implementation: capturedImplementation.reference,
      model: modelProvenance.reference,
    },
    execution: {
      command: runnerCommand.reportCommand,
      cwd: configuredCwd ?? "<trial-directory>",
      timeout_ms: loadedTrial.spec.runner.timeout_ms,
      duration_ms: durationMs,
      log: "trial.log",
      artifact_directory: "model",
    },
  };
  await Promise.all([
    verifyDirectoryIdentity({
      path: outputRoot,
      expected: outputRootIdentity,
      description: "Study trial output root",
    }),
    verifyDirectoryIdentity({
      path: trialDirectory,
      expected: trialDirectoryIdentity,
      description: "Study trial directory",
    }),
  ]);
  await writeJsonAtomic(reportPath, report);
  return { trialDirectory, reportPath, report };
}
