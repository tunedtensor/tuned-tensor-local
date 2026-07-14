import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { compileSpecToJsonl, buildSystemMessage, examplesFromChatJsonl } from "../src/dataset.js";
import { fineTuneRunRequestSchema } from "../src/contracts.js";

test("compiles behavior spec examples to chat JSONL", () => {
  const request = fineTuneRunRequestSchema.parse({
    run_id: "11111111-1111-4111-8111-111111111111",
    user_id: "user",
    behavior_spec_id: "22222222-2222-4222-8222-222222222222",
    run_number: 1,
    spec_snapshot: {
      name: "Example",
      description: "",
      system_prompt: "Be precise.",
      guidelines: ["Return one label."],
      constraints: ["No prose."],
      base_model: "qwen/qwen3.5-2b",
      examples: [{ input: "hello", output: "greeting" }],
    },
  });

  assert.equal(request.spec_snapshot.base_model, "Qwen/Qwen3.5-2B");
  assert.match(buildSystemMessage(request.spec_snapshot), /Guidelines:/);

  const lines = compileSpecToJsonl(request.spec_snapshot).split("\n");
  assert.equal(lines.length, 1);
  const row = JSON.parse(lines[0] ?? "");
  assert.equal(row.messages[0].role, "system");
  assert.equal(row.messages[1].content, "hello");
  assert.equal(row.messages[2].content, "greeting");
});

test("compiles and reads multimodal chat JSONL image assets", async () => {
  const request = fineTuneRunRequestSchema.parse({
    run_id: "11111111-1111-4111-8111-111111111111",
    user_id: "user",
    behavior_spec_id: "22222222-2222-4222-8222-222222222222",
    run_number: 1,
    spec_snapshot: {
      name: "Chart QA",
      description: "",
      system_prompt: "Read charts.",
      base_model: "qwen/qwen3-vl-2b",
      examples: [{
        input: "What is the blue value?",
        output: "42",
        input_assets: [{ type: "image", image: "charts/example.png", mime_type: "image/png" }],
      }],
    },
  });

  const line = compileSpecToJsonl(request.spec_snapshot);
  const row = JSON.parse(line);
  assert.deepEqual(row.messages[1].content, [
    { type: "image", image: "charts/example.png", mime_type: "image/png" },
    { type: "text", text: "What is the blue value?" },
  ]);

  const root = await mkdtemp(join(tmpdir(), "tt-local-dataset-mm-test-"));
  try {
    const path = join(root, "data.jsonl");
    await writeFile(path, `${line}\n`, "utf8");
    const examples = await examplesFromChatJsonl(path);
    assert.equal(examples[0]?.input, "What is the blue value?");
    assert.equal(examples[0]?.output, "42");
    assert.deepEqual(examples[0]?.input_assets, [{
      type: "image",
      image: join(root, "charts", "example.png"),
      mime_type: "image/png",
    }]);

    await writeFile(path, `${JSON.stringify({
      images: ["charts/top-level.png"],
      messages: [
        { role: "user", content: "What is shown?" },
        { role: "assistant", content: "A chart" },
      ],
    })}\n`, "utf8");
    const topLevelExamples = await examplesFromChatJsonl(path);
    assert.deepEqual(topLevelExamples[0]?.input_assets, [{
      type: "image",
      image: join(root, "charts", "top-level.png"),
    }]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
