#!/usr/bin/env node

import { resolve } from "node:path";
import { localRunnerConfigSchema } from "./contracts.js";
import {
  loadLocalRunnerConfig,
  loadRunRequest,
  runLocalFineTune,
} from "./orchestrator.js";
import { runDoctor } from "./doctor.js";

export * from "./contracts.js";
export * from "./dataset.js";
export * from "./orchestrator.js";

export interface LocalRunnerInfo {
  name: "tuned-tensor-local";
  status: "local-runner-preview";
  description: string;
}

export function getLocalRunnerInfo(): LocalRunnerInfo {
  return {
    name: "tuned-tensor-local",
    status: "local-runner-preview",
    description: "Local fine-tuning runner for single-GPU uv/Python hosts.",
  };
}

function readOption(argv: string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  if (index === -1) return undefined;
  return argv[index + 1];
}

function hasFlag(argv: string[], name: string): boolean {
  return argv.includes(name);
}

function printHelp(): void {
  console.log(`tt-local

Commands:
  info
  doctor [--config local-runner.json]
  validate <request.json> [--config local-runner.json]
  run <request.json> [--config local-runner.json] [--dry-run]

The run command writes local artifacts under config.artifactRoot, defaulting to
.tt-local/artifacts. Without --dry-run, uv and the Python training dependencies are required.`);
}

async function main(argv: string[]): Promise<void> {
  const command = argv[2] ?? "info";

  if (command === "info" || command === "--help" || command === "-h") {
    const info = getLocalRunnerInfo();
    console.log(`${info.name}: ${info.description}`);
    console.log(`Status: ${info.status}`);
    if (command !== "info") printHelp();
    return;
  }

  if (command === "doctor") {
    const configPath = readOption(argv, "--config");
    const config = await loadLocalRunnerConfig(configPath ? resolve(configPath) : undefined);
    const checks = await runDoctor(config);
    const ok = checks.every((check) => check.ok);
    console.log(JSON.stringify({ ok, checks }, null, 2));
    if (!ok) process.exitCode = 1;
    return;
  }

  if (command === "validate") {
    const requestPath = argv[3];
    if (!requestPath) throw new Error("validate requires <request.json>");
    const configPath = readOption(argv, "--config");
    const request = await loadRunRequest(resolve(requestPath));
    const config = await loadLocalRunnerConfig(configPath ? resolve(configPath) : undefined);
    console.log(JSON.stringify({
      ok: true,
      run_id: request.run_id,
      base_model: request.spec_snapshot.base_model,
      artifact_root: config.artifactRoot,
      dry_run: config.dryRun,
    }, null, 2));
    return;
  }

  if (command === "run") {
    const requestPath = argv[3];
    if (!requestPath) throw new Error("run requires <request.json>");
    const configPath = readOption(argv, "--config");
    const configInput = await loadLocalRunnerConfig(configPath ? resolve(configPath) : undefined);
    const config = localRunnerConfigSchema.parse({
      ...configInput,
      dryRun: hasFlag(argv, "--dry-run") ? true : configInput.dryRun,
    });
    const request = await loadRunRequest(resolve(requestPath));
    const result = await runLocalFineTune({ request, config });
    console.log(JSON.stringify({
      status: result.report.status,
      run_id: result.report.run_id,
      report_path: result.reportPath,
      artifact_dir: result.artifactDir,
      avg_score_delta: result.report.comparison.avg_score_delta,
    }, null, 2));
    return;
  }

  console.error(`Unknown command: ${command}`);
  process.exitCode = 1;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main(process.argv).catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
