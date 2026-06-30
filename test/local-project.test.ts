import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  initLocalSpecFile,
  loadLocalRunInput,
  runRequestFromLocalSpec,
} from "../src/local-project.js";

test("initializes a standalone local spec file and converts it to a run request", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-project-test-"));
  try {
    const path = join(root, "tunedtensor.json");
    const spec = await initLocalSpecFile({
      outputPath: path,
      name: "Standalone Local",
      baseModel: "qwen/qwen3.5-2b",
    });
    assert.equal(spec.name, "Standalone Local");
    assert.equal(spec.base_model, "Qwen/Qwen3.5-2B");
    assert.ok(spec.id);

    const input = await loadLocalRunInput(path, {
      userId: "local-test-user",
      runNumber: 7,
    });
    assert.equal(input.kind, "spec");
    assert.equal(input.request.user_id, "local-test-user");
    assert.equal(input.request.run_number, 7);
    assert.equal(input.request.behavior_spec_id, spec.id);
    assert.equal(input.request.spec_snapshot.name, "Standalone Local");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("runRequestFromLocalSpec preserves local spec metadata", () => {
  const request = runRequestFromLocalSpec({
    id: "77777777-7777-4777-8777-777777777777",
    user_id: "local-user",
    run_number: 3,
    name: "Preserved",
    description: "",
    system_prompt: "Return labels.",
    guidelines: [],
    constraints: [],
    base_model: "Qwen/Qwen3.5-2B",
    examples: [{ input: "good", output: "positive" }],
    hyperparameters: {
      n_epochs: 2,
      save_adapter_only: true,
      augment: false,
      use_llm_judge: false,
    },
  }, {
    runId: "88888888-8888-4888-8888-888888888888",
  });

  assert.equal(request.run_id, "88888888-8888-4888-8888-888888888888");
  assert.equal(request.behavior_spec_id, "77777777-7777-4777-8777-777777777777");
  assert.equal(request.run_number, 3);
  assert.equal(request.hyperparameters.n_epochs, 2);
});
