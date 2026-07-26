import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fineTuneRunRequestSchema } from "../src/contracts.js";
import {
  buildSystemMessage,
  compileSpecToJsonl,
  examplesFromChatJsonl,
  normalizeChatJsonlForRelocation,
} from "../src/dataset.js";

test("compiles a canonical Qwen text-SFT row with the assistant last", () => {
  const request = fineTuneRunRequestSchema.parse({
    run_id: "11111111-1111-4111-8111-111111111111",
    user_id: "local-user",
    behavior_spec_id: "22222222-2222-4222-8222-222222222222",
    run_number: 1,
    spec_snapshot: {
      name: "Classifier",
      system_prompt: "Be precise.",
      guidelines: ["Return one label."],
      constraints: ["No prose."],
      base_model: "qwen/qwen3.5-2b",
      examples: [{ input: "hello", output: "greeting" }],
    },
  });

  assert.equal(request.spec_snapshot.base_model, "Qwen/Qwen3.5-2B");
  assert.equal(
    buildSystemMessage(request.spec_snapshot),
    "Be precise.\n\nGuidelines:\n- Return one label.\n\nConstraints:\n- No prose.",
  );
  assert.deepEqual(JSON.parse(compileSpecToJsonl(request.spec_snapshot)), {
    messages: [
      {
        role: "system",
        content: "Be precise.\n\nGuidelines:\n- Return one label.\n\nConstraints:\n- No prose.",
      },
      { role: "user", content: "hello" },
      { role: "assistant", content: "greeting" },
    ],
  });
});

test("uses one non-empty system prompt when the spec provides no prose", () => {
  const request = fineTuneRunRequestSchema.parse({
    run_id: "11111111-1111-4111-8111-111111111111",
    user_id: "local-user",
    behavior_spec_id: "22222222-2222-4222-8222-222222222222",
    run_number: 1,
    spec_snapshot: {
      name: "Demonstrated behavior",
      system_prompt: "",
      guidelines: [],
      constraints: [],
      base_model: "Qwen/Qwen3.5-2B",
      examples: [{ input: "hello", output: "hi" }],
    },
  });

  assert.equal(
    buildSystemMessage(request.spec_snapshot),
    "Follow the demonstrated behavior.",
  );
});

test("normalizes and reads text-only prebuilt chat JSONL", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-dataset-"));
  try {
    const path = join(root, "training.jsonl");
    await writeFile(path, [
      JSON.stringify({
        extra: "discarded",
        messages: [
          { role: "system", content: "Classify." },
          { role: "user", content: "first" },
          { role: "assistant", content: "final answer" },
        ],
      }),
      "",
    ].join("\n"), "utf8");

    const normalized = await normalizeChatJsonlForRelocation(path);
    assert.equal((JSON.parse(normalized) as Record<string, unknown>).extra, undefined);
    assert.deepEqual(await examplesFromChatJsonl(path), [{
      input: "first",
      output: "final answer",
    }]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects malformed, empty, and structured-content chat rows", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-dataset-invalid-"));
  try {
    const path = join(root, "training.jsonl");
    await writeFile(path, "{not json}\n", "utf8");
    await assert.rejects(normalizeChatJsonlForRelocation(path), /row 1: malformed JSON/);

    await writeFile(path, "\n", "utf8");
    await assert.rejects(normalizeChatJsonlForRelocation(path), /contains no examples/);

    await writeFile(path, `${JSON.stringify({
      messages: [
        { role: "user", content: [{ type: "text", text: "hello" }] },
        { role: "assistant", content: "hi" },
      ],
    })}\n`, "utf8");
    await assert.rejects(
      normalizeChatJsonlForRelocation(path),
      /optional system message, one user message, and one assistant answer/,
    );

    await writeFile(path, `${JSON.stringify({
      messages: [{ role: "user", content: "hello" }],
    })}\n`, "utf8");
    await assert.rejects(
      examplesFromChatJsonl(path),
      /optional system message, one user message, and one assistant answer/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
