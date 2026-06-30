import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { test } from "node:test";
import { fineTuneRunRequestSchema, localRunnerConfigSchema } from "../src/contracts.js";
import { runLocalFineTune } from "../src/orchestrator.js";

test("runs a dry local workflow and writes compatible artifacts", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-test-"));
  try {
    const request = fineTuneRunRequestSchema.parse({
      run_id: "11111111-1111-4111-8111-111111111111",
      user_id: "local-user",
      behavior_spec_id: "22222222-2222-4222-8222-222222222222",
      run_number: 1,
      spec_snapshot: {
        name: "Smoke",
        description: "",
        system_prompt: "Return labels.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [
          { input: "Classify: good", output: "positive" },
          { input: "Classify: bad", output: "negative" },
        ],
      },
      hyperparameters: {
        n_epochs: 1,
        augment: false,
        use_llm_judge: false,
        save_adapter_only: true,
      },
    });
    const config = localRunnerConfigSchema.parse({
      artifactRoot: root,
      dryRun: true,
      evaluation: { mode: "heuristic" },
    });

    const result = await runLocalFineTune({ request, config });
    assert.equal(result.report.status, "completed");
    assert.equal(result.report.training.provider, "local-uv");
    assert.equal(result.report.baseline.total, 2);
    assert.equal(result.report.candidate.total, 2);

    const reportText = await readFile(result.reportPath, "utf8");
    assert.match(reportText, /local-uv/);
    const datasetText = await readFile(result.report.artifact_uris.dataset.replace(/^file:\/\//, ""), "utf8");
    assert.match(datasetText, /Classify: good/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
