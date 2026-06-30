import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fineTuneRunRequestSchema, localRunnerConfigSchema } from "../src/contracts.js";
import { serveLocalDashboard } from "../src/server.js";
import { createLocalStore } from "../src/store.js";

test("local dashboard serves health and run metadata from the file store", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-server-test-"));
  let dashboard: Awaited<ReturnType<typeof serveLocalDashboard>> | null = null;
  try {
    const config = localRunnerConfigSchema.parse({
      storeRoot: join(root, "store"),
      artifactRoot: join(root, "artifacts"),
      dryRun: true,
    });
    const request = fineTuneRunRequestSchema.parse({
      run_id: "55555555-5555-4555-8555-555555555555",
      user_id: "local-user",
      behavior_spec_id: "66666666-6666-4666-8666-666666666666",
      run_number: 1,
      spec_snapshot: {
        name: "Dashboard Spec",
        description: "",
        system_prompt: "Return labels.",
        guidelines: [],
        constraints: [],
        base_model: "Qwen/Qwen3.5-2B",
        examples: [{ input: "Classify: good", output: "positive" }],
      },
      hyperparameters: {
        n_epochs: 1,
        augment: false,
        use_llm_judge: false,
        save_adapter_only: true,
      },
    });
    await createLocalStore(config.storeRoot).startRun({
      request,
      artifactDir: join(root, "artifacts", request.run_id),
    });

    dashboard = await serveLocalDashboard({ host: "127.0.0.1", port: 0, config });
    const health = await fetch(new URL("/api/health", dashboard.url)).then((res) => res.json()) as { ok: boolean };
    assert.equal(health.ok, true);

    const runs = await fetch(new URL("/api/runs", dashboard.url)).then((res) => res.json()) as Array<{ id: string; spec_name: string }>;
    assert.equal(runs.length, 1);
    assert.equal(runs[0]?.spec_name, "Dashboard Spec");

    const detail = await fetch(new URL(`/api/runs/${request.run_id.slice(0, 8)}`, dashboard.url)).then((res) => res.json()) as { id: string };
    assert.equal(detail.id, request.run_id);
  } finally {
    await dashboard?.close();
    await rm(root, { recursive: true, force: true });
  }
});
