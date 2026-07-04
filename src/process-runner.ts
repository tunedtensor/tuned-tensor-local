import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { forwardStreamLines, type LocalRunReporter } from "./run-reporter.js";

export interface UvPythonEntrypointConfig {
  backend?: "uv" | "command";
  command?: string[];
  project?: string;
  cwd?: string;
  module?: string;
  script?: string;
  args?: string[];
  with?: string[];
}

export function buildUvPythonArgs(
  entrypoint: UvPythonEntrypointConfig,
  options: { defaultScript?: string; extraArgs?: string[] } = {},
): string[] {
  const args: string[] = ["run"];
  if (entrypoint.project) args.push("--project", entrypoint.project);
  for (const dependency of entrypoint.with ?? []) args.push("--with", dependency);
  args.push("python");
  if (entrypoint.module) {
    args.push("-m", entrypoint.module);
  } else if (entrypoint.script) {
    args.push(entrypoint.script);
  } else if (options.defaultScript) {
    args.push(options.defaultScript);
  } else {
    throw new Error("uv python entrypoint requires module, script, or defaultScript");
  }
  args.push(...(entrypoint.args ?? []), ...(options.extraArgs ?? []));
  return args;
}

export function buildEntrypointCommand(
  entrypoint: UvPythonEntrypointConfig,
  options: { defaultScript?: string; extraArgs?: string[] } = {},
): { command: string; commandArgs: string[]; displayCommand: string[]; kind: "uv" | "command" } {
  if (entrypoint.backend === "command") {
    const [command, ...baseArgs] = entrypoint.command ?? [];
    if (!command) {
      throw new Error("command entrypoint requires command");
    }
    const commandArgs = [...baseArgs, ...(entrypoint.args ?? []), ...(options.extraArgs ?? [])];
    return {
      command,
      commandArgs,
      displayCommand: [command, ...commandArgs],
      kind: "command",
    };
  }

  const commandArgs = buildUvPythonArgs(entrypoint, options);
  return {
    command: "uv",
    commandArgs,
    displayCommand: ["uv", ...commandArgs],
    kind: "uv",
  };
}

export interface LoggedProcessResult {
  exitCode: number;
  stderr: string;
}

export async function runLoggedProcess(args: {
  command: string;
  commandArgs: string[];
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  logPath?: string;
  timeoutMs?: number;
  timeoutMessage?: string;
  reporter?: LocalRunReporter;
  stage: string;
  onLine?: (line: string, stream: "stdout" | "stderr") => void;
}): Promise<LoggedProcessResult> {
  if (args.logPath) await mkdir(dirname(args.logPath), { recursive: true });
  const logStream = args.logPath ? createWriteStream(args.logPath, { flags: "w" }) : null;
  let stderr = "";

  try {
    const exitCode = await new Promise<number>((resolvePromise, reject) => {
      const child = spawn(args.command, args.commandArgs, {
        cwd: args.cwd ? resolve(args.cwd) : process.cwd(),
        stdio: ["ignore", "pipe", "pipe"],
        env: args.env ?? process.env,
      });
      let timedOut = false;
      const timer = args.timeoutMs
        ? setTimeout(() => {
            timedOut = true;
            child.kill("SIGTERM");
            reject(new Error(args.timeoutMessage ?? `${args.command} timed out after ${args.timeoutMs}ms`));
          }, args.timeoutMs)
        : null;

      if (logStream) {
        child.stdout.pipe(logStream, { end: false });
        child.stderr.pipe(logStream, { end: false });
      }

      forwardStreamLines(child.stdout, (line) => {
        args.onLine?.(line, "stdout");
        if (args.reporter?.verbose) {
          void args.reporter.onLog?.({ stage: args.stage, stream: "stdout", message: line });
        }
      });
      forwardStreamLines(child.stderr, (line) => {
        stderr += `${line}\n`;
        args.onLine?.(line, "stderr");
        if (args.reporter?.verbose) {
          void args.reporter.onLog?.({ stage: args.stage, stream: "stderr", message: line });
        }
      });
      child.on("error", (error) => {
        if (timer) clearTimeout(timer);
        reject(error);
      });
      child.on("close", (code) => {
        if (timer) clearTimeout(timer);
        if (timedOut) return;
        resolvePromise(code ?? 1);
      });
    });
    return { exitCode, stderr };
  } finally {
    if (logStream) {
      await new Promise<void>((resolveEnd) => {
        logStream.end(resolveEnd);
      });
    }
  }
}

export async function runJsonStdInCommand(args: {
  command: string[];
  payload: unknown;
  timeoutMs: number;
  timeoutMessage: string;
  errorPrefix: string;
}): Promise<string> {
  const [cmd, ...cmdArgs] = args.command;
  return await new Promise((resolvePromise, reject) => {
    const child = spawn(cmd, cmdArgs, {
      stdio: ["pipe", "pipe", "pipe"],
      env: process.env,
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
      reject(new Error(args.timeoutMessage));
    }, args.timeoutMs);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (timedOut) return;
      if (code !== 0) {
        reject(new Error(`${args.errorPrefix} exited ${code}: ${stderr.slice(0, 1000)}`));
        return;
      }
      const trimmed = stdout.trim();
      try {
        const parsed = JSON.parse(trimmed) as { content?: unknown; output?: unknown; actual?: unknown };
        const content = parsed.content ?? parsed.output ?? parsed.actual;
        resolvePromise(typeof content === "string" ? content : trimmed);
      } catch {
        resolvePromise(trimmed);
      }
    });
    child.stdin.end(JSON.stringify(args.payload));
  });
}
