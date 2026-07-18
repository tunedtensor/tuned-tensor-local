import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import {
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { test } from "node:test";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  buildEntrypointCommand,
  buildUvPythonArgs,
  ProcessCancelledError,
  runJsonStdInCommand,
  runLoggedProcess,
} from "../src/process-runner.js";

const repoRoot = dirname(dirname(fileURLToPath(import.meta.url)));

test("uv bundled local runner paths resolve relative to the package", () => {
  const args = buildUvPythonArgs(
    { backend: "uv", project: "training/local-runner" },
    { defaultScript: "training/local-runner/src/train.py" },
  );

  assert.deepEqual(args, [
    "run",
    "--project",
    join(repoRoot, "training/local-runner"),
    "python",
    join(repoRoot, "training/local-runner/src/train.py"),
  ]);
});

test("uv custom project and script paths stay caller-relative", () => {
  const args = buildUvPythonArgs({
    backend: "uv",
    project: "custom-runner",
    script: "scripts/train.py",
  });

  assert.deepEqual(args, [
    "run",
    "--project",
    "custom-runner",
    "python",
    "scripts/train.py",
  ]);
});

test("command entrypoints do not rewrite bundled-looking arguments", () => {
  const command = buildEntrypointCommand({
    backend: "command",
    command: ["python", "training/local-runner/src/train.py"],
  });

  assert.deepEqual(command.commandArgs, ["training/local-runner/src/train.py"]);
});

test("logged processes terminate their process group when cancellation is requested", async () => {
  let requested = false;
  const timer = setTimeout(() => { requested = true; }, 50);
  try {
    await assert.rejects(
      runLoggedProcess({
        command: process.execPath,
        commandArgs: ["-e", "setInterval(() => {}, 1000)"],
        stage: "training",
        shouldCancel: () => requested,
        cancelPollMs: 10,
        timeoutMs: 2_000,
      }),
      ProcessCancelledError,
    );
  } finally {
    clearTimeout(timer);
  }
});

test("process timeouts wait for child shutdown before returning", async () => {
  const started = performance.now();
  await assert.rejects(
    runLoggedProcess({
      command: process.execPath,
      commandArgs: [
        "-e",
        "process.on('SIGTERM', () => setTimeout(() => process.exit(0), 100)); setInterval(() => {}, 1000)",
      ],
      stage: "evaluation",
      timeoutMs: 500,
    }),
    /timed out after 500ms/,
  );
  assert.ok(performance.now() - started >= 550, "timeout returned before the child exited");
});

test("process timeouts force-kill descendants that outlive the direct child", async () => {
  if (process.platform === "win32") return;
  const root = await mkdtemp(join(tmpdir(), "tt-local-process-group-test-"));
  const marker = join(root, "descendant-survived");
  const descendant = `
    const { writeFileSync } = require("node:fs");
    process.on("SIGTERM", () => {});
    setTimeout(() => {
      writeFileSync(${JSON.stringify(marker)}, "survived");
      process.exit(0);
    }, 1000);
  `;
  const parent = `
    require("node:child_process").spawn(
      process.execPath,
      ["-e", ${JSON.stringify(descendant)}],
      { stdio: "ignore" }
    );
    setInterval(() => {}, 1000);
  `;
  try {
    await assert.rejects(
      runLoggedProcess({
        command: process.execPath,
        commandArgs: ["-e", parent],
        stage: "study-trial",
        timeoutMs: 500,
      }),
      /timed out after 500ms/,
    );
    await new Promise((resolveWait) => setTimeout(resolveWait, 700));
    assert.equal(existsSync(marker), false, "timed-out descendant survived its process group");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("logged processes can clean up descendants after a successful direct child", async () => {
  if (process.platform === "win32") return;
  const root = await mkdtemp(join(tmpdir(), "tt-local-process-exit-test-"));
  const marker = join(root, "descendant-survived");
  const descendant = `
    const { writeFileSync } = require("node:fs");
    setTimeout(() => {
      writeFileSync(${JSON.stringify(marker)}, "survived");
      process.exit(0);
    }, 500);
  `;
  const parent = `
    const child = require("node:child_process").spawn(
      process.execPath,
      ["-e", ${JSON.stringify(descendant)}],
      { stdio: "ignore" }
    );
    child.unref();
  `;
  try {
    const result = await runLoggedProcess({
      command: process.execPath,
      commandArgs: ["-e", parent],
      stage: "study-trial",
      terminateProcessGroupOnExit: true,
    });
    assert.equal(result.exitCode, 0);
    await new Promise((resolveWait) => setTimeout(resolveWait, 600));
    assert.equal(existsSync(marker), false, "successful child left a descendant running");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("exclusive process logs reject aliases before launching the child", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-process-log-test-"));
  const target = join(root, "target.log");
  const logPath = join(root, "predictor.log");
  const marker = join(root, "child-ran");
  await writeFile(target, "preserve me\n");
  await symlink(target, logPath);
  try {
    await assert.rejects(
      runLoggedProcess({
        command: process.execPath,
        commandArgs: [
          "-e",
          `require("node:fs").writeFileSync(${JSON.stringify(marker)}, "ran")`,
        ],
        stage: "study-test",
        logPath,
        exclusiveLog: true,
      }),
      /EEXIST/,
    );
    assert.equal(existsSync(marker), false);
    assert.equal(await readFile(target, "utf8"), "preserve me\n");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("JSON command inference terminates its process group when cancelled", async () => {
  let requested = false;
  const timer = setTimeout(() => { requested = true; }, 50);
  try {
    await assert.rejects(
      runJsonStdInCommand({
        command: [process.execPath, "-e", "process.stdin.resume(); setInterval(() => {}, 1000)"],
        payload: { input: "hello" },
        timeoutMs: 2_000,
        timeoutMessage: "inference timeout",
        errorPrefix: "inference",
        shouldCancel: () => requested,
        cancelPollMs: 10,
      }),
      ProcessCancelledError,
    );
  } finally {
    clearTimeout(timer);
  }
});

test("optional log reporter failures do not crash the child process", async () => {
  const result = await runLoggedProcess({
    command: process.execPath,
    commandArgs: ["-e", "console.log('progress')"],
    stage: "training",
    reporter: {
      verbose: true,
      async onLog() {
        throw new Error("reporter unavailable");
      },
    },
  });
  assert.equal(result.exitCode, 0);
});
