import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { localRunnerConfigSchema } from "../src/contracts.js";
import {
  parseCsvRecords,
  parseUnlabeledCsv,
  parseUnlabeledJsonl,
  runLocalLabelingJob,
  stripModelThinking,
} from "../src/labeling.js";
import { sanitizeText } from "../src/labeling-sanitize.js";

process.env.NODE_ENV = "test";

const MAX_ROWS = 1000;

test("parseUnlabeledJsonl accepts pending and pre-labeled rows", () => {
  const text = [
    JSON.stringify({ input: "classify: good" }),
    JSON.stringify({ input: "classify: bad", output: "negative" }),
  ].join("\n");
  const result = parseUnlabeledJsonl(text, MAX_ROWS);
  assert.deepEqual(result.errors, []);
  assert.equal(result.rows.length, 2);
  assert.equal(result.rows[0]?.output, undefined);
  assert.equal(result.rows[1]?.output, "negative");
});

test("parseUnlabeledJsonl collects row errors instead of throwing", () => {
  const text = [
    "not json",
    JSON.stringify({ notInput: "x" }),
    JSON.stringify({ input: "" }),
    JSON.stringify({ input: "ok", output: 42 }),
  ].join("\n");
  const result = parseUnlabeledJsonl(text, MAX_ROWS);
  assert.equal(result.rows.length, 0);
  assert.equal(result.errors.length, 4);
  assert.match(result.errors[0]!, /invalid JSON/);
});

test("parseUnlabeledJsonl routes oversized inputs to failedRows", () => {
  const text = JSON.stringify({ input: "x".repeat(32_001) });
  const result = parseUnlabeledJsonl(text, MAX_ROWS);
  assert.deepEqual(result.errors, []);
  assert.equal(result.rows.length, 0);
  assert.equal(result.failedRows.length, 1);
  assert.match(result.failedRows[0]!.error, /exceeds/);
});

test("parseUnlabeledJsonl enforces the row limit", () => {
  const text = Array.from({ length: 3 }, (_, i) => JSON.stringify({ input: `row ${i}` })).join("\n");
  const result = parseUnlabeledJsonl(text, 2);
  assert.equal(result.errors.length, 1);
  assert.match(result.errors[0]!, /limit is 2/);
});

test("parseCsvRecords handles quoted fields with commas, quotes, and newlines", () => {
  const text = 'id,text\n1,"hello, ""world"""\n2,"line one\nline two"\n';
  const parsed = parseCsvRecords(text);
  assert.deepEqual(parsed.errors, []);
  assert.deepEqual(parsed.fields, ["id", "text"]);
  assert.deepEqual(parsed.records, [
    ["1", 'hello, "world"'],
    ["2", "line one\nline two"],
  ]);
});

test("parseUnlabeledCsv picks the input column and reports a missing one", () => {
  const text = "id,prompt\n1,summarize this\n2,translate that\n";
  const ok = parseUnlabeledCsv(text, "prompt", MAX_ROWS);
  assert.deepEqual(ok.errors, []);
  assert.deepEqual(ok.rows.map((row) => row.input), ["summarize this", "translate that"]);

  const missing = parseUnlabeledCsv(text, "text", MAX_ROWS);
  assert.equal(missing.rows.length, 0);
  assert.match(missing.errors[0]!, /Column "text" not found/);
});

test("sanitizeText blocks secrets and redacts PII", () => {
  const blocked = sanitizeText("my key is sk-abcdefghijklmnopqrstuvwx");
  assert.equal(blocked.status, "blocked");
  const redacted = sanitizeText("email me at someone@example.com");
  assert.equal(redacted.status, "redacted");
  assert.match(redacted.text, /\[REDACTED_EMAIL\]/);
});

test("stripModelThinking removes think blocks and dangling close tags", () => {
  assert.equal(stripModelThinking("<think>hmm</think>answer"), "answer");
  assert.equal(stripModelThinking("truncated reasoning</think>\nanswer"), "answer");
  assert.equal(stripModelThinking("plain answer"), "plain answer");
});

interface MockCall {
  body: Record<string, unknown>;
}

function withMockedOpenRouter(
  respond: (call: MockCall, index: number) => Response,
): { calls: MockCall[]; restore: () => void } {
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.OPENROUTER_API_KEY;
  process.env.OPENROUTER_API_KEY = "test-key";
  const calls: MockCall[] = [];
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    const call = { body: JSON.parse(String(init?.body)) as Record<string, unknown> };
    calls.push(call);
    return respond(call, calls.length - 1);
  }) as typeof fetch;
  return {
    calls,
    restore: () => {
      globalThis.fetch = originalFetch;
      if (originalKey === undefined) {
        delete process.env.OPENROUTER_API_KEY;
      } else {
        process.env.OPENROUTER_API_KEY = originalKey;
      }
    },
  };
}

function teacherResponse(content: string): Response {
  return new Response(JSON.stringify({
    model: "test/teacher",
    choices: [{ message: { content } }],
    usage: { prompt_tokens: 10, completion_tokens: 5 },
  }), { status: 200, headers: { "Content-Type": "application/json" } });
}

async function testConfig(root: string) {
  return localRunnerConfigSchema.parse({
    artifactRoot: join(root, "artifacts"),
    llm: { model: "test/teacher" },
    labeling: { concurrency: 2 },
  });
}

test("runLocalLabelingJob labels pending rows, keeps uploads, blocks secrets", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-label-"));
  const sourcePath = join(root, "rows.jsonl");
  await writeFile(sourcePath, [
    JSON.stringify({ input: "classify: this is great" }),
    JSON.stringify({ input: "classify: awful", output: "negative" }),
    JSON.stringify({ input: "use my key sk-abcdefghijklmnopqrstuvwx please" }),
  ].join("\n"), "utf8");

  const mock = withMockedOpenRouter((call) => {
    const messages = call.body.messages as Array<{ role: string; content: string }>;
    assert.equal(messages[0]?.role, "system");
    assert.equal(messages[0]?.content, "You are a sentiment classifier.");
    assert.equal(call.body.response_format, undefined);
    assert.equal(call.body.temperature, 0.2);
    return teacherResponse("<think>ok</think>positive");
  });
  try {
    const result = await runLocalLabelingJob({
      sourcePath,
      systemMessage: "You are a sentiment classifier.",
      config: await testConfig(root),
      outputPath: join(root, "out", "labeled.jsonl"),
    });

    assert.equal(result.report.status, "completed");
    assert.equal(result.report.row_count, 3);
    assert.equal(result.report.labeled_count, 2);
    assert.equal(result.report.prelabeled_count, 1);
    assert.equal(result.report.failed_count, 1);
    assert.equal(result.report.blocked_count, 1);
    assert.equal(result.report.prompt_tokens_total, 10);
    assert.equal(result.report.completion_tokens_total, 5);
    assert.equal(mock.calls.length, 1);

    const labeled = (await readFile(result.labeledPath, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
    assert.deepEqual(labeled, [
      { input: "classify: this is great", output: "positive" },
      { input: "classify: awful", output: "negative" },
    ]);
    const copy = await readFile(join(root, "out", "labeled.jsonl"), "utf8");
    assert.equal(copy.trim().split("\n").length, 2);
  } finally {
    mock.restore();
  }
});

test("runLocalLabelingJob retries retryable errors and fails rows on hard errors", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-label-"));
  const sourcePath = join(root, "rows.jsonl");
  await writeFile(sourcePath, [
    JSON.stringify({ input: "first row" }),
    JSON.stringify({ input: "second row" }),
  ].join("\n"), "utf8");

  let firstRowAttempts = 0;
  const mock = withMockedOpenRouter((call) => {
    const messages = call.body.messages as Array<{ content: string }>;
    const input = messages[1]!.content;
    if (input === "first row") {
      firstRowAttempts += 1;
      if (firstRowAttempts === 1) {
        return new Response("rate limited", { status: 429 });
      }
      return teacherResponse("label one");
    }
    return new Response("bad request", { status: 400 });
  });
  try {
    const result = await runLocalLabelingJob({
      sourcePath,
      systemMessage: "Label the row.",
      config: await testConfig(root),
    });

    assert.equal(result.report.status, "completed");
    assert.equal(result.report.labeled_count, 1);
    assert.equal(result.report.failed_count, 1);
    assert.equal(firstRowAttempts, 2);

    const rows = (await readFile(result.rowsPath, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
    assert.equal(rows[0].status, "labeled");
    assert.equal(rows[0].label_source, "teacher");
    assert.equal(rows[1].status, "failed");
    assert.match(rows[1].error, /400/);
  } finally {
    mock.restore();
  }
});

test("runLocalLabelingJob marks the job failed when every row fails", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-label-"));
  const sourcePath = join(root, "rows.jsonl");
  await writeFile(sourcePath, JSON.stringify({ input: "only row" }), "utf8");

  const mock = withMockedOpenRouter(() => new Response("bad request", { status: 400 }));
  try {
    const result = await runLocalLabelingJob({
      sourcePath,
      systemMessage: "Label the row.",
      config: await testConfig(root),
    });
    assert.equal(result.report.status, "failed");
    assert.equal(result.report.error, "All rows failed teacher labeling");
  } finally {
    mock.restore();
  }
});

test("runLocalLabelingJob dry run parses and sanitizes without teacher calls", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-label-"));
  const sourcePath = join(root, "rows.csv");
  await writeFile(sourcePath, "id,text\n1,call me at 415-555-0123\n2,plain row\n", "utf8");

  const mock = withMockedOpenRouter(() => {
    throw new Error("teacher must not be called in dry run");
  });
  try {
    const result = await runLocalLabelingJob({
      sourcePath,
      inputColumn: "text",
      systemMessage: "Label the row.",
      config: await testConfig(root),
      dryRun: true,
    });
    assert.equal(result.report.status, "dry_run");
    assert.equal(result.report.pending_count, 2);
    assert.equal(result.report.redacted_count, 1);
    assert.equal(mock.calls.length, 0);
  } finally {
    mock.restore();
  }
});

test("runLocalLabelingJob rejects invalid sources and missing csv column option", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-label-"));
  const config = await testConfig(root);
  const jsonlPath = join(root, "bad.jsonl");
  await writeFile(jsonlPath, "not json", "utf8");

  await assert.rejects(
    runLocalLabelingJob({ sourcePath: jsonlPath, systemMessage: "x", config, dryRun: true }),
    /Labeling source failed validation/,
  );
  await assert.rejects(
    runLocalLabelingJob({ sourcePath: join(root, "rows.csv"), systemMessage: "x", config, dryRun: true }),
    /--input-column/,
  );
  await assert.rejects(
    runLocalLabelingJob({ sourcePath: jsonlPath, systemMessage: "   ", config, dryRun: true }),
    /non-empty system message/,
  );
});
