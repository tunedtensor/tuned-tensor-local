import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("deprecation guidance distinguishes the legacy binary from current tt", async () => {
  const [deprecation, architecture] = await Promise.all([
    readFile("DEPRECATION.md", "utf8"),
    readFile("docs/architecture.md", "utf8"),
  ]);

  assert.match(deprecation, /frozen legacy adapter-only/i);
  assert.match(deprecation, /Foundation Pipeline/);
  assert.doesNotMatch(deprecation, /serves as the local runtime behind `tt`/);
  assert.match(architecture, /frozen legacy adapter-only/i);
});