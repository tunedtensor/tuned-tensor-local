import assert from "node:assert/strict";
import { test } from "node:test";
import { compileSpecToJsonl, buildSystemMessage } from "../src/dataset.js";
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
