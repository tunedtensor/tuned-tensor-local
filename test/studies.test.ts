import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  buildStudyBenchmarkLock,
  defaultStudyLockPath,
  studyBenchmarkLockSchema,
  validateStudyBenchmark,
  writeStudyBenchmarkLock,
} from "../src/studies.js";

async function withTemporaryStudy(
  callback: (fixture: {
    root: string;
    studyPath: string;
    lockPath: string;
    splitPath: (split: "training" | "validation" | "test") => string;
  }) => void | Promise<void>,
): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), "tt-local-study-test-"));
  try {
    const splitPath = (split: "training" | "validation" | "test") => join(root, "data", `${split}.csv`);
    await mkdir(join(root, "data"), { recursive: true });
    await writeFile(splitPath("training"), [
      "id,spread_bps,note,big_move",
      'train-1,1.2,"quiet, market",0',
      'train-2,8.4,"line one',
      'line two",1',
    ].join("\n") + "\n", "utf8");
    await writeFile(splitPath("validation"), [
      "id,spread_bps,note,big_move",
      "validation-1,1.4,quiet,0",
      "validation-2,7.9,move,1",
    ].join("\n") + "\n", "utf8");
    await writeFile(splitPath("test"), [
      "id,spread_bps,note,big_move",
      "test-1,1.1,quiet,0",
      "test-2,8.1,move,1",
    ].join("\n") + "\n", "utf8");
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
    await callback({
      root,
      studyPath,
      lockPath: join(root, "portfolio.study.lock.json"),
      splitPath,
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

test("builds a deterministic benchmark lock from relative CSV splits", async () => {
  await withTemporaryStudy(async ({ root, studyPath }) => {
    const first = await buildStudyBenchmarkLock(studyPath);
    const second = await buildStudyBenchmarkLock(studyPath);

    assert.deepEqual(first, second);
    assert.equal(`${JSON.stringify(first, null, 2)}\n`, `${JSON.stringify(second, null, 2)}\n`);
    assert.doesNotThrow(() => studyBenchmarkLockSchema.parse(first));
    assert.equal(first.dataset.splits.training.row_count, 2);
    assert.equal(first.dataset.total_row_count, 6);
    assert.deepEqual(first.dataset.columns, ["id", "spread_bps", "note", "big_move"]);
    assert.equal(first.dataset.splits.training.path, "data/training.csv");
    assert.equal(JSON.stringify(first).includes(root), false);
    assert.equal(JSON.stringify(first).includes("class_count"), false);
    assert.equal(defaultStudyLockPath(studyPath), join(root, "portfolio.study.lock.json"));
  });
});

test("rejects duplicate and cross-split IDs, undeclared labels, and incompatible schemas", async () => {
  await withTemporaryStudy(async ({ studyPath, splitPath }) => {
    const training = await readFile(splitPath("training"), "utf8");
    await writeFile(splitPath("training"), training.replace("train-2", "train-1"), "utf8");
    await assert.rejects(buildStudyBenchmarkLock(studyPath), /training.*duplicate ID "train-1"/i);

    await writeFile(splitPath("training"), training, "utf8");
    const validation = await readFile(splitPath("validation"), "utf8");
    await writeFile(splitPath("validation"), validation.replace("validation-1", "train-1"), "utf8");
    await assert.rejects(buildStudyBenchmarkLock(studyPath), /ID "train-1".*training.*validation/i);

    await writeFile(splitPath("validation"), validation.replace(",move,1", ",move,unknown"), "utf8");
    await assert.rejects(buildStudyBenchmarkLock(studyPath), /validation.*undeclared label "unknown"/i);

    await writeFile(splitPath("validation"), validation.replace("spread_bps,note", "note,spread_bps"), "utf8");
    await assert.rejects(buildStudyBenchmarkLock(studyPath), /validation.*columns.*training/i);
  });
});

test("rejects ambiguous headers and IDs with outer whitespace", async () => {
  await withTemporaryStudy(async ({ studyPath, splitPath }) => {
    const training = await readFile(splitPath("training"), "utf8");
    await writeFile(
      splitPath("training"),
      training.replace("id,spread_bps,note,big_move", "id,spread_bps,spread_bps,big_move"),
      "utf8",
    );
    await assert.rejects(buildStudyBenchmarkLock(studyPath), /duplicate header "spread_bps"/i);

    await writeFile(splitPath("training"), training.replace("train-1", " train-1"), "utf8");
    await assert.rejects(buildStudyBenchmarkLock(studyPath), /training.*ID.*outer whitespace/i);
  });
});

test("rejects malformed quoted fields and splits missing either declared class", async () => {
  await withTemporaryStudy(async ({ studyPath, splitPath }) => {
    const training = await readFile(splitPath("training"), "utf8");
    await writeFile(splitPath("training"), training.replace("train-1", '"train-1"junk'), "utf8");
    await assert.rejects(buildStudyBenchmarkLock(studyPath), /unexpected character after closing quote/i);

    await writeFile(
      splitPath("training"),
      training.replace(/,1$/gm, ",0"),
      "utf8",
    );
    await assert.rejects(buildStudyBenchmarkLock(studyPath), /training.*both declared labels/i);
  });
});

test("never writes a benchmark lock over the StudySpec or a dataset split", async () => {
  await withTemporaryStudy(async ({ root, studyPath, splitPath }) => {
    const studyBefore = await readFile(studyPath, "utf8");
    const trainingBefore = await readFile(splitPath("training"), "utf8");
    const unrelatedPath = join(root, "notes.txt");
    await writeFile(unrelatedPath, "keep me\n", "utf8");

    await assert.rejects(
      writeStudyBenchmarkLock({ studyPath, outputPath: studyPath, force: true }),
      /must not overwrite/i,
    );
    await assert.rejects(
      writeStudyBenchmarkLock({ studyPath, outputPath: splitPath("training"), force: true }),
      /must not overwrite/i,
    );
    await assert.rejects(
      writeStudyBenchmarkLock({ studyPath, outputPath: unrelatedPath, force: true }),
      /refusing to replace non-lock file/i,
    );
    assert.equal(await readFile(studyPath, "utf8"), studyBefore);
    assert.equal(await readFile(splitPath("training"), "utf8"), trainingBefore);
    assert.equal(await readFile(unrelatedPath, "utf8"), "keep me\n");
  });
});

test("requires portable relative dataset paths in a StudySpec", async () => {
  await withTemporaryStudy(async ({ studyPath, splitPath }) => {
    const raw = JSON.parse(await readFile(studyPath, "utf8")) as {
      dataset: { splits: { training: string } };
    };
    raw.dataset.splits.training = splitPath("training");
    await writeFile(studyPath, `${JSON.stringify(raw, null, 2)}\n`, "utf8");
    await assert.rejects(buildStudyBenchmarkLock(studyPath), /portable relative local path/i);

    raw.dataset.splits.training = "C:\\data\\training.csv";
    await writeFile(studyPath, `${JSON.stringify(raw, null, 2)}\n`, "utf8");
    await assert.rejects(buildStudyBenchmarkLock(studyPath), /portable relative local path/i);

    raw.dataset.splits.training = "data\\training.csv";
    await writeFile(studyPath, `${JSON.stringify(raw, null, 2)}\n`, "utf8");
    await assert.rejects(buildStudyBenchmarkLock(studyPath), /portable relative local path/i);
  });
});

test("rejects non-regular dataset sources", async () => {
  await withTemporaryStudy(async ({ studyPath, splitPath }) => {
    await rm(splitPath("training"));
    await mkdir(splitPath("training"));
    await assert.rejects(buildStudyBenchmarkLock(studyPath), /training dataset must be a regular file/i);
  });
});

test("rejects non-regular StudySpec and benchmark lock sources", async () => {
  await withTemporaryStudy(async ({ studyPath, lockPath }) => {
    await writeStudyBenchmarkLock({ studyPath });
    await rm(lockPath);
    await mkdir(lockPath);
    await assert.rejects(
      validateStudyBenchmark({ studyPath }),
      /study benchmark lock must be a regular file/i,
    );

    await rm(lockPath, { recursive: true });
    await rm(studyPath);
    await mkdir(studyPath);
    await assert.rejects(buildStudyBenchmarkLock(studyPath), /StudySpec must be a regular file/i);
  });
});

test("only one concurrent non-force lock creation can publish", async () => {
  await withTemporaryStudy(async ({ studyPath }) => {
    const results = await Promise.allSettled(
      Array.from({ length: 8 }, () => writeStudyBenchmarkLock({ studyPath })),
    );
    assert.equal(results.filter((result) => result.status === "fulfilled").length, 1);
    const rejections = results.filter(
      (result): result is PromiseRejectedResult => result.status === "rejected",
    );
    assert.equal(rejections.length, 7);
    for (const rejection of rejections) {
      assert.match(String(rejection.reason), /already exists.*--force/i);
    }
  });
});

test("detects benchmark drift without changing the recorded lock", async () => {
  await withTemporaryStudy(async ({ studyPath, lockPath, splitPath }) => {
    const written = await writeStudyBenchmarkLock({ studyPath });
    assert.equal(written.lockPath, lockPath);
    const recorded = await readFile(lockPath, "utf8");
    await validateStudyBenchmark({ studyPath });

    const validation = await readFile(splitPath("validation"), "utf8");
    await writeFile(splitPath("validation"), validation.replace("7.9", "7.8"), "utf8");
    await assert.rejects(
      validateStudyBenchmark({ studyPath }),
      /dataset\.splits\.validation\.sha256.*expected.*found/is,
    );
    assert.equal(await readFile(lockPath, "utf8"), recorded);

    await assert.rejects(
      writeStudyBenchmarkLock({ studyPath }),
      /already exists.*--force/i,
    );
    const refreshed = await writeStudyBenchmarkLock({ studyPath, force: true });
    assert.notEqual(refreshed.lock.dataset.splits.validation.sha256, written.lock.dataset.splits.validation.sha256);
  });
});

test("detects semantic StudySpec drift against an existing lock", async () => {
  await withTemporaryStudy(async ({ studyPath }) => {
    await writeStudyBenchmarkLock({ studyPath });
    const raw = JSON.parse(await readFile(studyPath, "utf8")) as {
      task: { primary_metric: string };
    };
    raw.task.primary_metric = "roc_auc";
    await writeFile(studyPath, `${JSON.stringify(raw, null, 2)}\n`, "utf8");

    await assert.rejects(
      validateStudyBenchmark({ studyPath }),
      /study_spec_sha256.*expected.*found/is,
    );
  });
});
