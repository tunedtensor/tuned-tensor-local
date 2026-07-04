import { spawn } from "node:child_process";
import type { LocalRunnerConfig } from "./contracts.js";

export interface DoctorCheck {
  name: string;
  ok: boolean;
  message: string;
}

async function runCommand(command: string, args: string[], timeoutMs = 10_000): Promise<{
  code: number | null;
  stdout: string;
  stderr: string;
  error?: string;
}> {
  return await new Promise((resolve) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      resolve({ code: null, stdout, stderr, error: `timed out after ${timeoutMs}ms` });
    }, timeoutMs);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => {
      clearTimeout(timer);
      resolve({ code: null, stdout, stderr, error: error.message });
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ code, stdout, stderr });
    });
  });
}

function firstLine(value: string): string {
  return value.trim().split(/\r?\n/)[0] ?? "";
}

export async function runDoctor(config: LocalRunnerConfig): Promise<DoctorCheck[]> {
  const checks: DoctorCheck[] = [];
  const needsUv = config.training.backend !== "command"
    || config.evaluation.inference.provider === "transformers";

  const nodeVersion = process.versions.node;
  const nodeMajor = Number(nodeVersion.split(".")[0]);
  checks.push({
    name: "node",
    ok: nodeMajor >= 22,
    message: `Node ${nodeVersion}`,
  });

  if (needsUv) {
    const uvVersion = await runCommand("uv", ["--version"]);
    checks.push({
      name: "uv",
      ok: uvVersion.code === 0,
      message: uvVersion.code === 0
        ? firstLine(uvVersion.stdout)
        : uvVersion.error ?? (firstLine(uvVersion.stderr) || "uv is not available"),
    });

    const pythonVersion = await runCommand("uv", ["run", "python", "--version"]);
    checks.push({
      name: "uv-python",
      ok: pythonVersion.code === 0,
      message: pythonVersion.code === 0
        ? firstLine(pythonVersion.stdout || pythonVersion.stderr)
        : pythonVersion.error ?? (firstLine(pythonVersion.stderr) || "uv could not run Python"),
    });
  } else {
    checks.push({
      name: "uv",
      ok: true,
      message: "uv is not required by the configured command entrypoints",
    });
  }

  const nvidiaSmi = await runCommand("nvidia-smi", []);
  checks.push({
    name: "nvidia-smi",
    ok: nvidiaSmi.code === 0,
    message: nvidiaSmi.code === 0
      ? firstLine(nvidiaSmi.stdout)
      : nvidiaSmi.error ?? (firstLine(nvidiaSmi.stderr) || "nvidia-smi not available"),
  });

  checks.push({
    name: "training-entrypoint",
    ok: config.training.backend === "command"
      ? Boolean(config.training.command || config.dryRun)
      : true,
    message: config.training.backend === "command" && config.training.command
      ? `will run command ${config.training.command.join(" ")}`
      : config.training.backend === "command" && config.dryRun
        ? "No training command configured; dryRun is enabled"
        : config.training.backend === "command"
          ? "No training command configured"
          : config.training.module
            ? `uv will run module ${config.training.module}`
            : config.training.script
              ? `uv will run script ${config.training.script}`
              : "uv will run the bundled SFT script",
  });

  return checks;
}
