import assert from "node:assert/strict";
import { test } from "node:test";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildEntrypointCommand, buildUvPythonArgs } from "../src/process-runner.js";

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
