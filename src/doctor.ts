import { constants } from "node:fs";
import { access, mkdir, statfs, unlink, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { delimiter, isAbsolute, join, resolve } from "node:path";
import { spawn } from "node:child_process";
import type { FineTuneRunRequest, LocalRunnerConfig } from "./contracts.js";
import { withHuggingFaceCacheEnvironment } from "./huggingface-cache.js";
import { buildEntrypointCommand } from "./process-runner.js";
import { resolveTrainingModel } from "./model-registry.js";
import { defaultLocalHome } from "./store.js";
import { verifyLocalBaseModel } from "./prefetch.js";

export interface DoctorCheck {
  name: string;
  ok: boolean;
  message: string;
  details?: Record<string, unknown>;
}

interface CommandResult {
  code: number | null;
  stdout: string;
  stderr: string;
  error?: string;
}

interface CommandOptions {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  timeoutMs?: number;
}

async function runCommand(command: string, args: string[], options: CommandOptions = {}): Promise<CommandResult> {
  const timeoutMs = options.timeoutMs ?? 10_000;
  let interrupted: string | undefined;
  const result = await new Promise<CommandResult>((resolveResult) => {
    let settled = false;
    let stdout = "";
    let stderr = "";
    let stopError: string | undefined;
    let forceKillTimer: NodeJS.Timeout | null = null;
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env ?? process.env,
      stdio: ["ignore", "pipe", "pipe"],
      detached: process.platform !== "win32",
    });
    const killProcessGroup = (signal: NodeJS.Signals) => {
      if (child.pid && process.platform !== "win32") {
        try {
          process.kill(-child.pid, signal);
          return;
        } catch {
          // The child may have exited between the check and signal.
        }
      }
      child.kill(signal);
    };
    const requestStop = (signal: NodeJS.Signals = "SIGTERM") => {
      killProcessGroup(signal);
      if (!forceKillTimer) {
        forceKillTimer = setTimeout(() => killProcessGroup("SIGKILL"), 5_000);
        forceKillTimer.unref();
      }
    };
    const onSigint = () => {
      interrupted = "interrupted by SIGINT";
      stopError = interrupted;
      requestStop("SIGINT");
    };
    const onSigterm = () => {
      interrupted = "interrupted by SIGTERM";
      stopError = interrupted;
      requestStop("SIGTERM");
    };
    const finish = (result: CommandResult) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (forceKillTimer) clearTimeout(forceKillTimer);
      process.off("SIGINT", onSigint);
      process.off("SIGTERM", onSigterm);
      resolveResult(result);
    };
    const timer = setTimeout(() => {
      stopError = `timed out after ${timeoutMs}ms`;
      requestStop();
    }, timeoutMs);
    timer.unref();
    process.once("SIGINT", onSigint);
    process.once("SIGTERM", onSigterm);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => {
      finish({ code: null, stdout, stderr, error: error.message });
    });
    child.on("close", (code) => {
      finish({ code: stopError ? null : code, stdout, stderr, error: stopError });
    });
  });
  if (interrupted) throw new Error(interrupted);
  return result;
}

function firstLine(value: string): string {
  return value.trim().split(/\r?\n/)[0] ?? "";
}

function commandText(command: string, args: string[]): string {
  return [command, ...args].map((part) => /\s/.test(part) ? JSON.stringify(part) : part).join(" ");
}

async function executableCheck(
  command: string,
  options: { cwd?: string; env?: NodeJS.ProcessEnv } = {},
): Promise<DoctorCheck> {
  const env = { ...process.env, ...options.env };
  const cwd = options.cwd ? resolve(options.cwd) : process.cwd();
  const candidates = isAbsolute(command) || command.includes("/")
    ? [isAbsolute(command) ? command : resolve(cwd, command)]
    : (env.PATH ?? "").split(delimiter).filter(Boolean).map((directory) =>
        join(isAbsolute(directory) ? directory : resolve(cwd, directory), command)
      );
  for (const candidate of candidates) {
    try {
      await access(candidate, constants.X_OK);
      return { name: `executable:${command}`, ok: true, message: candidate };
    } catch {
      // Continue searching PATH.
    }
  }
  return {
    name: `executable:${command}`,
    ok: false,
    message: `${command} is not executable or was not found on PATH`,
  };
}

async function writableDirectoryCheck(name: string, path: string): Promise<DoctorCheck> {
  const resolvedPath = resolve(path);
  const probePath = join(resolvedPath, `.tt-local-write-probe-${process.pid}-${Date.now()}`);
  try {
    await mkdir(resolvedPath, { recursive: true });
    await access(resolvedPath, constants.R_OK | constants.W_OK);
    await writeFile(probePath, "ok\n", { flag: "wx" });
    await unlink(probePath);
    const fs = await statfs(resolvedPath);
    const freeBytes = Number(fs.bavail) * Number(fs.bsize);
    return {
      name,
      ok: true,
      message: `${resolvedPath} is writable (${Math.round(freeBytes / (1024 ** 3) * 10) / 10} GiB available)`,
      details: { path: resolvedPath, free_bytes: freeBytes },
    };
  } catch (error) {
    await unlink(probePath).catch(() => undefined);
    return {
      name,
      ok: false,
      message: `${resolvedPath}: ${error instanceof Error ? error.message : String(error)}`,
      details: { path: resolvedPath },
    };
  }
}

interface PythonProbePlan {
  name: "training-python" | "evaluation-python";
  command: string;
  args: string[];
  cwd?: string;
  env: NodeJS.ProcessEnv;
}

function pythonProbeSource(device: LocalRunnerConfig["evaluation"]["inference"]["device"]): string {
  return [
    "import json",
    "import torch, transformers, peft, huggingface_hub",
    `requested = ${JSON.stringify(device)}`,
    "assert requested != 'cuda' or torch.cuda.is_available(), 'CUDA was requested but torch.cuda.is_available() is false'",
    "assert requested != 'mps' or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()), 'MPS was requested but is unavailable'",
    "print(json.dumps({'python_ok': True, 'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))",
  ].join("; ");
}

function pythonProbePlan(args: {
  name: PythonProbePlan["name"];
  entrypoint: LocalRunnerConfig["training"] | LocalRunnerConfig["evaluation"]["inference"];
  cwd?: string;
  env: Record<string, string>;
  device: LocalRunnerConfig["evaluation"]["inference"]["device"];
  modelCache?: string;
}): PythonProbePlan {
  const entrypoint = buildEntrypointCommand({
    ...args.entrypoint,
    backend: "uv",
    module: undefined,
    script: "-c",
    args: [],
  }, {
    extraArgs: [pythonProbeSource(args.device)],
  });
  const env = withHuggingFaceCacheEnvironment({
    ...process.env,
    ...args.env,
  }, args.modelCache);
  return {
    name: args.name,
    command: entrypoint.command,
    args: entrypoint.commandArgs,
    cwd: args.cwd,
    env,
  };
}

/** Build the exact uv environments that a configured run will use. */
export function buildDoctorPythonPlans(config: LocalRunnerConfig): PythonProbePlan[] {
  if (config.dryRun) return [];
  const plans: PythonProbePlan[] = [];
  if (config.training.backend === "uv") {
    plans.push(pythonProbePlan({
      name: "training-python",
      entrypoint: config.training,
      cwd: config.training.cwd,
      env: config.training.env,
      device: config.evaluation.inference.device,
      modelCache: config.paths.modelCache,
    }));
  }
  if (config.evaluation.inference.provider === "transformers") {
    plans.push(pythonProbePlan({
      name: "evaluation-python",
      entrypoint: config.evaluation.inference,
      cwd: config.evaluation.inference.cwd,
      env: config.evaluation.inference.env,
      device: config.evaluation.inference.device,
      modelCache: config.paths.modelCache,
    }));
  }
  return plans;
}

function configuredCommandChecks(config: LocalRunnerConfig): Array<{
  name: string;
  command?: string;
  cwd?: string;
  env?: NodeJS.ProcessEnv;
}> {
  const checks: Array<{ name: string; command?: string; cwd?: string; env?: NodeJS.ProcessEnv }> = [];
  if (config.training.backend === "command" && !config.dryRun) {
    checks.push({
      name: "training-command",
      command: config.training.command?.[0],
      cwd: config.training.cwd,
      env: config.training.env,
    });
  }
  const inference = config.evaluation.inference;
  if ((inference.provider === "command" || inference.provider === "batch_command") && !config.dryRun) {
    checks.push({
      name: "evaluation-command",
      command: inference.command?.[0],
      cwd: inference.cwd,
      env: inference.env,
    });
  }
  return checks;
}

function placeholderSpecCheck(request: FineTuneRunRequest): DoctorCheck {
  const placeholder = request.spec_snapshot.examples.some((example) =>
    /replace this with/i.test(example.input) || /replace this with/i.test(example.output)
  ) || /describe the behavior this local model should learn/i.test(request.spec_snapshot.system_prompt);
  return {
    name: "spec-content",
    ok: !placeholder,
    message: placeholder
      ? "The spec still contains generated placeholder content; edit it before training."
      : `${request.spec_snapshot.examples.length} spec example(s); base model ${request.spec_snapshot.base_model}`,
  };
}

function gatedModelTokenCheck(request: FineTuneRunRequest): DoctorCheck | null {
  let model: ReturnType<typeof resolveTrainingModel>;
  try {
    model = resolveTrainingModel(request.spec_snapshot.base_model);
  } catch {
    return null;
  }
  if (!model.requiresHfToken) return null;
  const available = Boolean(process.env.HF_TOKEN?.trim());
  return {
    name: "hugging-face-token",
    ok: available,
    message: available
      ? `HF_TOKEN is configured for gated model ${model.id}.`
      : `${model.id} is gated and requires a non-empty HF_TOKEN.`,
  };
}

function judgeCheck(config: LocalRunnerConfig): DoctorCheck | null {
  if (config.evaluation.scoring.mode !== "llm_judge") return null;
  if (config.evaluation.scoring.fallback === "exact_match" && !config.llm) {
    return {
      name: "llm-judge",
      ok: true,
      message: "LLM judge is not configured; explicit exact_match fallback will be used.",
    };
  }
  if (!config.llm) {
    return {
      name: "llm-judge",
      ok: false,
      message: "evaluation.scoring.mode=llm_judge requires an llm configuration, or explicitly set fallback=exact_match.",
    };
  }
  const available = Boolean(process.env[config.llm.apiKeyEnv]);
  return {
    name: "llm-judge",
    ok: available || config.evaluation.scoring.fallback === "exact_match",
    message: available
      ? `${config.llm.model} credentials found in ${config.llm.apiKeyEnv}`
      : `${config.llm.apiKeyEnv} is not set${config.evaluation.scoring.fallback === "exact_match" ? "; explicit exact_match fallback will be used" : ""}`,
  };
}

async function localBaseModelCheck(path: string): Promise<DoctorCheck> {
  const resolvedPath = resolve(path);
  try {
    const verified = await verifyLocalBaseModel(resolvedPath);
    return {
      name: "local-base-model",
      ok: true,
      message: `${resolvedPath} is a valid local Hugging Face snapshot directory.`,
      details: { path: resolvedPath, file_count: verified.fileCount, size_bytes: verified.sizeBytes },
    };
  } catch (error) {
    return {
      name: "local-base-model",
      ok: false,
      message: error instanceof Error ? error.message : String(error),
      details: { path: resolvedPath },
    };
  }
}

export async function runDoctor(config: LocalRunnerConfig, request?: FineTuneRunRequest): Promise<DoctorCheck[]> {
  const checks: DoctorCheck[] = [];
  const nodeVersion = process.versions.node;
  const nodeMajor = Number(nodeVersion.split(".")[0]);
  checks.push({ name: "node", ok: nodeMajor >= 22, message: `Node ${nodeVersion}` });

  checks.push(await writableDirectoryCheck("artifact-root", config.artifactRoot));
  checks.push(await writableDirectoryCheck("store-root", config.storeRoot ?? defaultLocalHome()));
  checks.push(await writableDirectoryCheck(
    "model-cache",
    config.paths.modelCache ?? process.env.HF_HOME ?? join(homedir(), ".cache", "huggingface"),
  ));
  if (config.paths.baseModel) checks.push(await localBaseModelCheck(config.paths.baseModel));

  if (request) {
    checks.push(placeholderSpecCheck(request));
    const gatedToken = gatedModelTokenCheck(request);
    if (gatedToken) checks.push(gatedToken);
  }
  const judge = judgeCheck(config);
  if (judge) checks.push(judge);

  const pythonPlans = buildDoctorPythonPlans(config);
  if (pythonPlans.length > 0) {
    const uvVersion = await runCommand("uv", ["--version"]);
    checks.push({
      name: "uv",
      ok: uvVersion.code === 0,
      message: uvVersion.code === 0
        ? firstLine(uvVersion.stdout)
        : uvVersion.error ?? (firstLine(uvVersion.stderr) || "uv is not available"),
    });

    if (uvVersion.code === 0) {
      const uniquePlans = new Map<string, PythonProbePlan>();
      for (const plan of pythonPlans) {
        const key = JSON.stringify([plan.command, plan.args, plan.cwd, plan.env]);
        if (!uniquePlans.has(key)) uniquePlans.set(key, plan);
      }
      for (const plan of uniquePlans.values()) {
        const result = await runCommand(plan.command, plan.args, {
          cwd: plan.cwd,
          env: plan.env,
          timeoutMs: 1_800_000,
        });
        checks.push({
          name: plan.name,
          ok: result.code === 0,
          message: result.code === 0
            ? firstLine(result.stdout)
            : result.error ?? (firstLine(result.stderr) || `${plan.command} exited ${result.code}`),
          details: { command: commandText(plan.command, plan.args), cwd: plan.cwd ?? process.cwd() },
        });
      }
    }
  } else {
    checks.push({
      name: "python-runtime",
      ok: true,
      message: config.dryRun
        ? "Python dependency checks skipped because dryRun is enabled."
        : "The configured command/none backends do not require the bundled Python runtime.",
    });
  }

  for (const entry of configuredCommandChecks(config)) {
    if (!entry.command) {
      checks.push({ name: entry.name, ok: false, message: `${entry.name} has no command configured` });
    } else {
      const check = await executableCheck(entry.command, { cwd: entry.cwd, env: entry.env });
      checks.push({ ...check, name: entry.name });
    }
  }

  const device = config.evaluation.inference.device;
  if (!config.dryRun && (device === "cuda" || device === "auto")) {
    const nvidiaSmi = await runCommand("nvidia-smi", []);
    checks.push({
      name: "nvidia-smi",
      ok: device === "auto" || nvidiaSmi.code === 0,
      message: nvidiaSmi.code === 0
        ? firstLine(nvidiaSmi.stdout)
        : device === "auto"
          ? "nvidia-smi is unavailable; device=auto may use MPS or CPU."
          : nvidiaSmi.error ?? (firstLine(nvidiaSmi.stderr) || "nvidia-smi not available"),
    });
  } else {
    checks.push({
      name: "nvidia-smi",
      ok: true,
      message: config.dryRun
        ? "GPU checks skipped because dryRun is enabled."
        : `nvidia-smi is not required for device=${device}.`,
    });
  }

  const trainingCommand = config.training.backend === "command"
    ? config.training.command
    : buildEntrypointCommand(config.training, {
        defaultScript: request?.training_method === "dpo"
          ? "training/local-runner/src/train_dpo.py"
          : "training/local-runner/src/train.py",
      }).displayCommand;
  checks.push({
    name: "effective-plan",
    ok: true,
    message: "Resolved the configured training and evaluation plan.",
    details: {
      training_command: trainingCommand ?? null,
      evaluation_provider: config.evaluation.inference.provider,
      evaluation_device: device,
      artifact_root: resolve(config.artifactRoot),
      store_root: resolve(config.storeRoot ?? defaultLocalHome()),
      model_cache: resolve(config.paths.modelCache ?? process.env.HF_HOME ?? join(homedir(), ".cache", "huggingface")),
      base_model: request?.spec_snapshot.base_model ?? null,
      scoring_mode: config.evaluation.scoring.mode,
    },
  });

  return checks;
}
