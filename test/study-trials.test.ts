import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  runStudyTrial,
  studyTrialSpecSchema,
  type StudyTrialSpec,
} from "../src/study-trials.js";
import { writeStudyBenchmarkLock } from "../src/studies.js";

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
    },
    parameters: { learning_rate: Number.POSITIVE_INFINITY },
  }).success, false);
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
