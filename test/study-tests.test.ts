import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import {
  chmod,
  cp,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  bundledPredictionRuntimeEvidenceSchema,
} from "../src/study-trials.js";
import {
  defaultStudyCandidateDirectory,
  promoteStudyTrialCandidate,
} from "../src/study-candidates.js";
import {
  defaultStudyTestClaimDirectory,
  defaultStudyTestClaimRoot,
  runStudyTest,
  runStudyTestWithHooks,
  studyTestClaimId,
  studyTestClaimIdentity,
  studyTestClaimIdentitySchema,
  studyTestFailureReceiptSchema,
  studyTestSuccessReceiptSchema,
} from "../src/study-tests.js";
import { runStudyTrial } from "../src/study-trials.js";
import {
  studyBenchmarkLockSchema,
  writeStudyBenchmarkLock,
} from "../src/studies.js";

let templateRoot = "";
const repositoryRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const cliPath = join(repositoryRoot, "src", "index.ts");
const tsxLoader = import.meta.resolve("tsx");

function sha256(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

async function writeTemplateProject(root: string): Promise<void> {
  const dataDirectory = join(root, "data");
  await mkdir(dataDirectory);
  const header = (
    "id,spread_bps,volume,big_move,future_mid,"
    + "observed_at,future_observed_at\n"
  );
  await writeFile(
    join(dataDirectory, "training.csv"),
    `${header}`
    + "train-1,1.0,100,0,0.10,2026-01-01T00:00:00Z,2026-01-01T00:30:00Z\n"
    + "train-2,1.3,90,0,0.20,2026-01-01T01:00:00Z,2026-01-01T01:30:00Z\n"
    + "train-3,2.0,80,0,0.30,2026-01-01T02:00:00Z,2026-01-01T02:30:00Z\n"
    + "train-4,7.0,20,1,0.80,2026-01-01T03:00:00Z,2026-01-01T03:30:00Z\n"
    + "train-5,8.0,10,1,0.90,2026-01-01T04:00:00Z,2026-01-01T04:30:00Z\n",
    "utf8",
  );
  await writeFile(
    join(dataDirectory, "validation.csv"),
    `${header}`
    + "validation-1,1.2,95,0,0.15,2026-01-02T00:00:00Z,2026-01-02T00:30:00Z\n"
    + "validation-2,7.5,15,1,0.85,2026-01-02T01:00:00Z,2026-01-02T01:30:00Z\n",
    "utf8",
  );
  await writeFile(
    join(dataDirectory, "test.csv"),
    `${header}`
    + "test-1,1.1,92,1,0.12,2026-01-03T00:00:00Z,2026-01-03T00:30:00Z\n"
    + "test-2,7.8,12,0,0.88,2026-01-03T01:00:00Z,2026-01-03T01:30:00Z\n",
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
      temporal: {
        policy: "ordered_purged",
        event_time_column: "observed_at",
        label_end_time_column: "future_observed_at",
        label_horizon_seconds: 3600,
        embargo_seconds: 300,
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
  const trial = await runStudyTrial({
    studyPath,
    trialPath,
    outputRoot: join(root, "trial-output"),
  });
  await promoteStudyTrialCandidate({
    studyPath,
    trialPath,
    trialDirectory: trial.trialDirectory,
  });
}

test.before(async () => {
  templateRoot = await mkdtemp(join(tmpdir(), "tt-local-study-test-template-"));
  await writeTemplateProject(templateRoot);
});

test.after(async () => {
  if (templateRoot) {
    await rm(templateRoot, { recursive: true, force: true });
  }
});

async function withCopiedProject(
  callback: (args: {
    caseRoot: string;
    projectRoot: string;
    studyPath: string;
    lockPath: string;
    testPath: string;
    home: string;
  }) => void | Promise<void>,
): Promise<void> {
  const caseRoot = await mkdtemp(join(tmpdir(), "tt-local-study-test-case-"));
  const projectRoot = join(caseRoot, "project");
  await cp(templateRoot, projectRoot, { recursive: true });
  const previousHome = process.env.TT_LOCAL_HOME;
  const homeAlias = join(caseRoot, "home-alias");
  await symlink(caseRoot, homeAlias, "dir");
  const home = join(homeAlias, "global-home");
  process.env.TT_LOCAL_HOME = home;
  try {
    await callback({
      caseRoot,
      projectRoot,
      studyPath: join(projectRoot, "portfolio.study.json"),
      lockPath: join(projectRoot, "portfolio.study.lock.json"),
      testPath: join(projectRoot, "data", "test.csv"),
      home,
    });
  } finally {
    if (previousHome === undefined) delete process.env.TT_LOCAL_HOME;
    else process.env.TT_LOCAL_HOME = previousHome;
    await rm(caseRoot, { recursive: true, force: true });
  }
}

async function lockedIdentity(lockPath: string) {
  const lock = studyBenchmarkLockSchema.parse(
    JSON.parse(await readFile(lockPath, "utf8")),
  );
  return studyTestClaimIdentity(lock);
}

async function assertArtifact(
  claimDirectory: string,
  reference: { path: string; sha256: string; size_bytes: number },
): Promise<Uint8Array> {
  const bytes = await readFile(join(claimDirectory, reference.path));
  assert.equal(bytes.byteLength, reference.size_bytes);
  assert.equal(sha256(bytes), reference.sha256);
  return bytes;
}

function runStudyTestCli(
  args: string[],
  cwd: string,
  home: string,
) {
  return spawnSync(
    process.execPath,
    ["--import", tsxLoader, cliPath, ...args],
    {
      cwd,
      encoding: "utf8",
      env: { ...process.env, TT_LOCAL_HOME: home },
    },
  );
}

test("claim identity is global, path-independent, and content-sensitive", async () => {
  await withCopiedProject(async ({ lockPath }) => {
    const lock = studyBenchmarkLockSchema.parse(
      JSON.parse(await readFile(lockPath, "utf8")),
    );
    const identity = studyTestClaimIdentity(lock);
    assert.deepEqual(studyTestClaimIdentitySchema.parse(identity), identity);
    assert.match(studyTestClaimId(identity), /^[a-f0-9]{64}$/);

    const renamed = structuredClone(lock);
    renamed.study.name = "A copied Study";
    renamed.dataset.splits.test.path = "renamed/test.csv";
    assert.equal(
      studyTestClaimId(studyTestClaimIdentity(renamed)),
      studyTestClaimId(identity),
    );
    const changedInputs = structuredClone(lock);
    changedInputs.study.task.input_columns = ["volume"];
    changedInputs.study.task.primary_metric = "roc_auc";
    assert.equal(
      studyTestClaimId(studyTestClaimIdentity(changedInputs)),
      studyTestClaimId(identity),
    );
    const changedTarget = structuredClone(lock);
    changedTarget.study.task.target_column = "another_target";
    assert.notEqual(
      studyTestClaimId(studyTestClaimIdentity(changedTarget)),
      studyTestClaimId(identity),
    );
    for (const changed of [
      {
        ...identity,
        test: { ...identity.test, sha256: "0".repeat(64) },
      },
      {
        ...identity,
        test: {
          ...identity.test,
          size_bytes: identity.test.size_bytes + 1,
        },
      },
      {
        ...identity,
        test: {
          ...identity.test,
          row_count: identity.test.row_count + 1,
        },
      },
    ]) {
      assert.notEqual(studyTestClaimId(changed), studyTestClaimId(identity));
    }
  });
});

test("evaluates held-out data and publishes strict evidence before returning metrics", async () => {
  await withCopiedProject(async ({
    home,
    projectRoot,
    studyPath,
    testPath,
  }) => {
    const execution = runStudyTestCli(
      ["studies", "test", studyPath],
      projectRoot,
      home,
    );
    assert.equal(execution.status, 0, execution.stderr);
    const output = JSON.parse(execution.stdout);
    assert.equal(output.ok, true);
    assert.equal(output.status, "tested");
    const result = {
      claimDirectory: output.claim_directory as string,
      receiptPath: output.receipt_path as string,
      receipt: output.receipt,
    };
    const receipt = studyTestSuccessReceiptSchema.parse(result.receipt);
    assert.deepEqual(output.evaluation, receipt.evaluation);
    assert.equal(existsSync(result.receiptPath), true);
    assert.equal(receipt.evaluation.primary_score, 0.5);
    assert.deepEqual(receipt.evaluation.metrics, {
      average_precision: 0.5,
      roc_auc: 0,
      f1_at_0_5: 0,
    });
    const candidate = JSON.parse(await readFile(
      join(
        defaultStudyCandidateDirectory(studyPath),
        "candidate.lock.json",
      ),
      "utf8",
    ));
    assert.equal(candidate.selection.validation_primary_score, 1);

    const projection = await readFile(
      join(result.claimDirectory, receipt.artifacts.projection.path),
      "utf8",
    );
    assert.equal(projection.startsWith("id,spread_bps,volume\n"), true);
    assert.equal(projection.includes("big_move"), false);
    assert.equal(projection.includes("future_mid"), false);
    const requestText = await readFile(
      join(result.claimDirectory, receipt.artifacts.request.path),
      "utf8",
    );
    const request = JSON.parse(requestText);
    assert.deepEqual(request.dataset, { prediction_csv: "test.csv" });
    assert.equal("target_column" in request.task, false);
    assert.equal("labels" in request.task, false);
    assert.equal(requestText.includes(testPath), false);
    assert.equal(requestText.includes(studyPath), false);

    const runtimeBytes = await assertArtifact(
      result.claimDirectory,
      receipt.execution.runtime.artifact,
    );
    assert.deepEqual(
      bundledPredictionRuntimeEvidenceSchema.parse(
        JSON.parse(new TextDecoder().decode(runtimeBytes)),
      ),
      receipt.execution.runtime.evidence,
    );
    await Promise.all([
      assertArtifact(result.claimDirectory, receipt.artifacts.request),
      assertArtifact(result.claimDirectory, receipt.artifacts.projection),
      assertArtifact(result.claimDirectory, receipt.artifacts.predictions),
      assertArtifact(result.claimDirectory, receipt.execution.log),
    ]);
    assert.deepEqual(
      (await readdir(result.claimDirectory))
        .filter((name) => name.includes(".tmp")),
      [],
    );
    assert.equal(
      existsSync(join(result.claimDirectory, "failure-receipt.json")),
      false,
    );

    const receiptBytes = await readFile(result.receiptPath);
    await chmod(
      join(
        defaultStudyCandidateDirectory(studyPath),
        "candidate.lock.json",
      ),
      0o600,
    );
    await writeFile(
      join(
        defaultStudyCandidateDirectory(studyPath),
        "candidate.lock.json",
      ),
      "not valid JSON\n",
    );
    await rm(testPath);
    const repeated = runStudyTestCli(
      ["studies", "test", studyPath],
      projectRoot,
      home,
    );
    assert.equal(repeated.status, 1);
    assert.equal(repeated.stdout, "");
    assert.match(repeated.stderr, /already been consumed.*one-shot/i);
    assert.deepEqual(await readFile(result.receiptPath), receiptBytes);
  });
});

test("copied Studies and concurrent callers share one atomic global claim", async () => {
  const caseRoot = await mkdtemp(join(tmpdir(), "tt-local-study-test-race-"));
  const firstRoot = join(caseRoot, "first");
  const secondRoot = join(caseRoot, "second");
  await Promise.all([
    cp(templateRoot, firstRoot, { recursive: true }),
    cp(templateRoot, secondRoot, { recursive: true }),
  ]);
  const previousHome = process.env.TT_LOCAL_HOME;
  process.env.TT_LOCAL_HOME = join(caseRoot, "global-home");
  try {
    const attempts = await Promise.allSettled([
      runStudyTest({ studyPath: join(firstRoot, "portfolio.study.json") }),
      runStudyTest({ studyPath: join(secondRoot, "portfolio.study.json") }),
    ]);
    assert.equal(
      attempts.filter((attempt) => attempt.status === "fulfilled").length,
      1,
    );
    const rejected = attempts.find(
      (attempt): attempt is PromiseRejectedResult => (
        attempt.status === "rejected"
      ),
    );
    assert.match(String(rejected?.reason), /already been consumed.*one-shot/i);
    const claimEntries = await readdir(defaultStudyTestClaimRoot());
    assert.equal(claimEntries.length, 1);
    const winner = attempts.find(
      (attempt): attempt is PromiseFulfilledResult<
        Awaited<ReturnType<typeof runStudyTest>>
      > => attempt.status === "fulfilled",
    )!;
    studyTestSuccessReceiptSchema.parse(winner.value.receipt);

    await rm(join(secondRoot, "data", "test.csv"), { force: true });
    await assert.rejects(
      runStudyTest({ studyPath: join(secondRoot, "portfolio.study.json") }),
      /already been consumed.*one-shot/i,
    );
  } finally {
    if (previousHome === undefined) delete process.env.TT_LOCAL_HOME;
    else process.env.TT_LOCAL_HOME = previousHome;
    await rm(caseRoot, { recursive: true, force: true });
  }
});

test("candidate preflight failures are retryable and do not open test data", async () => {
  await withCopiedProject(async ({ studyPath, testPath, lockPath }) => {
    const candidateLockPath = join(
      defaultStudyCandidateDirectory(studyPath),
      "candidate.lock.json",
    );
    const candidateLockBytes = await readFile(candidateLockPath);
    const testBytes = await readFile(testPath);
    const candidateLock = JSON.parse(candidateLockBytes.toString("utf8"));
    candidateLock.training_runtime.python = "tampered";
    await chmod(candidateLockPath, 0o600);
    await writeFile(
      candidateLockPath,
      `${JSON.stringify(candidateLock, null, 2)}\n`,
      "utf8",
    );
    await rm(testPath);
    await assert.rejects(
      runStudyTest({ studyPath }),
      (error: Error) => (
        /training runtime does not match/i.test(error.message)
        && !/test dataset|ENOENT/i.test(error.message)
      ),
    );
    const identity = await lockedIdentity(lockPath);
    assert.equal(
      existsSync(defaultStudyTestClaimDirectory(identity)),
      false,
    );

    await writeFile(candidateLockPath, candidateLockBytes);
    await chmod(candidateLockPath, 0o400);
    await writeFile(testPath, testBytes);
    const retried = await runStudyTest({ studyPath });
    assert.equal(retried.receipt.status, "succeeded");
  });
});

test("post-claim test drift publishes a terminal metric-free failure receipt", async () => {
  await withCopiedProject(async ({ studyPath, testPath }) => {
    const originalTest = await readFile(testPath);
    let claimDirectory = "";
    await assert.rejects(
      runStudyTestWithHooks(
        { studyPath },
        {
          afterClaim: async (context) => {
            claimDirectory = context.claimDirectory;
            await writeFile(
              testPath,
              Buffer.concat([originalTest, Buffer.from("drift")]),
            );
          },
        },
      ),
      /failed after global claim.*remains consumed/i,
    );
    assert.equal(existsSync(join(claimDirectory, "receipt.json")), false);
    const failureText = await readFile(
      join(claimDirectory, "failure-receipt.json"),
      "utf8",
    );
    const failure = studyTestFailureReceiptSchema.parse(
      JSON.parse(failureText),
    );
    assert.equal(failure.error.phase, "test");
    assert.equal(
      failure.error.message,
      "Held-out Study test failed during test",
    );
    assert.equal(failureText.includes(testPath), false);
    assert.equal(failureText.includes("drift"), false);
    assert.equal("evaluation" in failure, false);

    await writeFile(testPath, originalTest);
    await assert.rejects(
      runStudyTest({ studyPath }),
      /already been consumed.*one-shot/i,
    );
  });
});

test("captured outputs and frozen inputs cannot drift before publication", async () => {
  await withCopiedProject(async ({ caseRoot, lockPath, studyPath }) => {
    for (const artifact of ["runtime", "predictions"] as const) {
      process.env.TT_LOCAL_HOME = join(caseRoot, `${artifact}-home`);
      let claimDirectory = "";
      await assert.rejects(
        runStudyTestWithHooks(
          { studyPath },
          artifact === "runtime"
            ? {
                afterPrediction: async (context) => {
                  claimDirectory = context.claimDirectory;
                  const path = join(
                    claimDirectory,
                    "prediction-runtime.json",
                  );
                  const runtime = JSON.parse(await readFile(path, "utf8"));
                  runtime.runtime.python = "tampered";
                  await writeFile(
                    path,
                    `${JSON.stringify(runtime, null, 2)}\n`,
                    "utf8",
                  );
                },
              }
            : {
                beforeCommit: async (context) => {
                  claimDirectory = context.claimDirectory;
                  const path = join(claimDirectory, "predictions.json");
                  const predictions = JSON.parse(
                    await readFile(path, "utf8"),
                  );
                  predictions.predictions[0].probability = 0.123;
                  await writeFile(
                    path,
                    `${JSON.stringify(predictions, null, 2)}\n`,
                    "utf8",
                  );
                },
              },
        ),
        /failed after global claim.*remains consumed/i,
      );
      assert.equal(existsSync(join(claimDirectory, "receipt.json")), false);
      const failure = studyTestFailureReceiptSchema.parse(JSON.parse(
        await readFile(
          join(claimDirectory, "failure-receipt.json"),
          "utf8",
        ),
      ));
      assert.equal(failure.error.phase, "verification");
      assert.equal(
        failure.error.message,
        "Held-out Study test failed during verification",
      );
      assert.equal("evaluation" in failure, false);
    }

    process.env.TT_LOCAL_HOME = join(caseRoot, "lock-home");
    const lockBytes = await readFile(lockPath);
    const changedLock = JSON.parse(lockBytes.toString("utf8"));
    changedLock.study.name = "Changed after prediction";
    let claimDirectory = "";
    await assert.rejects(
      runStudyTestWithHooks(
        { studyPath },
        {
          beforeCommit: async (context) => {
            claimDirectory = context.claimDirectory;
            await chmod(lockPath, 0o600);
            await writeFile(
              lockPath,
              `${JSON.stringify(changedLock, null, 2)}\n`,
              "utf8",
            );
          },
        },
      ),
      /failed after global claim.*remains consumed/i,
    );
    assert.equal(existsSync(join(claimDirectory, "receipt.json")), false);
    const failure = studyTestFailureReceiptSchema.parse(JSON.parse(
      await readFile(
        join(claimDirectory, "failure-receipt.json"),
        "utf8",
      ),
    ));
    assert.equal(failure.error.phase, "verification");
    assert.equal("evaluation" in failure, false);
    await writeFile(lockPath, lockBytes);
  });
});

test("crash and symlink tombstones are consumed without opening test", async () => {
  await withCopiedProject(async ({
    caseRoot,
    studyPath,
    lockPath,
    testPath,
  }) => {
    const identity = await lockedIdentity(lockPath);
    const claimDirectory = defaultStudyTestClaimDirectory(identity);
    await mkdir(defaultStudyTestClaimRoot(), { recursive: true });
    await mkdir(claimDirectory);
    await rm(testPath);
    await assert.rejects(
      runStudyTest({ studyPath }),
      /already been consumed.*one-shot/i,
    );

    const secondHome = join(caseRoot, "symlink-home");
    process.env.TT_LOCAL_HOME = secondHome;
    const symlinkClaim = defaultStudyTestClaimDirectory(identity);
    const target = join(caseRoot, "symlink-target");
    await mkdir(defaultStudyTestClaimRoot(), { recursive: true });
    await mkdir(target);
    await symlink(target, symlinkClaim);
    await assert.rejects(
      runStudyTest({ studyPath }),
      /already been consumed.*one-shot/i,
    );
    assert.deepEqual(await readdir(target), []);
  });
});
