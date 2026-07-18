import { spawn } from "node:child_process";
import { createWriteStream, type WriteStream } from "node:fs";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { forwardStreamLines, reportInBackground, type LocalRunReporter } from "./run-reporter.js";

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

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const bundledLocalRunnerPrefix = "training/local-runner";

function resolveBundledLocalRunnerPath(path: string | undefined): string | undefined {
  if (!path) return undefined;
  if (path === bundledLocalRunnerPrefix || path.startsWith(`${bundledLocalRunnerPrefix}/`)) {
    return join(packageRoot, path);
  }
  return path;
}

export function buildUvPythonArgs(
  entrypoint: UvPythonEntrypointConfig,
  options: { defaultScript?: string; extraArgs?: string[] } = {},
): string[] {
  const args: string[] = ["run"];
  const project = resolveBundledLocalRunnerPath(entrypoint.project);
  if (project) args.push("--project", project);
  for (const dependency of entrypoint.with ?? []) args.push("--with", dependency);
  args.push("python");
  if (entrypoint.module) {
    args.push("-m", entrypoint.module);
  } else if (entrypoint.script) {
    args.push(resolveBundledLocalRunnerPath(entrypoint.script) ?? entrypoint.script);
  } else if (options.defaultScript) {
    args.push(resolveBundledLocalRunnerPath(options.defaultScript) ?? options.defaultScript);
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

export class ProcessCancelledError extends Error {
  constructor(message = "Process cancelled.") {
    super(message);
    this.name = "ProcessCancelledError";
  }
}

async function openProcessLog(
  path: string,
  exclusive: boolean,
): Promise<WriteStream> {
  const stream = createWriteStream(path, {
    flags: exclusive ? "wx" : "w",
    mode: 0o600,
  });
  await new Promise<void>((resolveOpen, reject) => {
    const onOpen = () => {
      stream.off("error", onError);
      resolveOpen();
    };
    const onError = (error: Error) => {
      stream.off("open", onOpen);
      reject(error);
    };
    stream.once("open", onOpen);
    stream.once("error", onError);
  });
  return stream;
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
  shouldCancel?: () => boolean | Promise<boolean>;
  cancelPollMs?: number;
  terminateProcessGroupOnExit?: boolean;
  exclusiveLog?: boolean;
}): Promise<LoggedProcessResult> {
  if (args.logPath) await mkdir(dirname(args.logPath), { recursive: true });
  const logStream = args.logPath
    ? await openProcessLog(args.logPath, args.exclusiveLog ?? false)
    : null;
  let stderr = "";

  try {
    const exitCode = await new Promise<number>((resolvePromise, reject) => {
      const child = spawn(args.command, args.commandArgs, {
        cwd: args.cwd ? resolve(args.cwd) : process.cwd(),
        stdio: ["ignore", "pipe", "pipe"],
        env: args.env ?? process.env,
        detached: process.platform !== "win32",
      });
      let timedOut = false;
      let cancelled = false;
      let cancellationError: unknown;
      let cancellationCheckRunning = false;
      let forceKillTimer: NodeJS.Timeout | null = null;
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
        cancelled = true;
        requestStop("SIGINT");
      };
      const onSigterm = () => {
        cancelled = true;
        requestStop("SIGTERM");
      };
      process.once("SIGINT", onSigint);
      process.once("SIGTERM", onSigterm);
      const timer = args.timeoutMs
        ? setTimeout(() => {
            timedOut = true;
            requestStop();
          }, args.timeoutMs)
        : null;
      const cancellationTimer = args.shouldCancel
        ? setInterval(() => {
            if (cancellationCheckRunning || cancelled) return;
            cancellationCheckRunning = true;
            Promise.resolve(args.shouldCancel?.())
              .then((requested) => {
                if (!requested || cancelled) return;
                cancelled = true;
                requestStop();
              })
              .catch((error) => {
                cancellationError = error;
                requestStop();
              })
              .finally(() => { cancellationCheckRunning = false; });
          }, args.cancelPollMs ?? 250)
        : null;
      cancellationTimer?.unref();

      const clearProcessTimers = () => {
        if (timer) clearTimeout(timer);
        if (cancellationTimer) clearInterval(cancellationTimer);
        if (forceKillTimer) clearTimeout(forceKillTimer);
        process.off("SIGINT", onSigint);
        process.off("SIGTERM", onSigterm);
      };

      if (logStream) {
        child.stdout.pipe(logStream, { end: false });
        child.stderr.pipe(logStream, { end: false });
      }

      forwardStreamLines(child.stdout, (line) => {
        args.onLine?.(line, "stdout");
        if (args.reporter?.verbose) {
          reportInBackground(() => args.reporter?.onLog?.({ stage: args.stage, stream: "stdout", message: line }));
        }
      });
      forwardStreamLines(child.stderr, (line) => {
        stderr += `${line}\n`;
        args.onLine?.(line, "stderr");
        if (args.reporter?.verbose) {
          reportInBackground(() => args.reporter?.onLog?.({ stage: args.stage, stream: "stderr", message: line }));
        }
      });
      child.on("error", (error) => {
        clearProcessTimers();
        reject(timedOut
          ? new Error(args.timeoutMessage ?? `${args.command} timed out after ${args.timeoutMs}ms`)
          : error);
      });
      child.on("close", (code) => {
        if (
          timedOut
          || cancelled
          || cancellationError
          || args.terminateProcessGroupOnExit
        ) {
          // The direct child can close its stdio while descendants remain in
          // the process group and ignore SIGTERM. Complete teardown before
          // cancelling the force-kill timer.
          killProcessGroup("SIGKILL");
        }
        clearProcessTimers();
        if (timedOut) {
          reject(new Error(args.timeoutMessage ?? `${args.command} timed out after ${args.timeoutMs}ms`));
          return;
        }
        if (cancellationError) {
          reject(cancellationError);
          return;
        }
        if (cancelled) {
          reject(new ProcessCancelledError(`${args.stage} was cancelled.`));
          return;
        }
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
  shouldCancel?: () => boolean | Promise<boolean>;
  cancelPollMs?: number;
}): Promise<string> {
  const [cmd, ...cmdArgs] = args.command;
  if (!cmd) throw new Error(`${args.errorPrefix} has no executable.`);
  return await new Promise((resolvePromise, reject) => {
    const child = spawn(cmd, cmdArgs, {
      stdio: ["pipe", "pipe", "pipe"],
      env: process.env,
      detached: process.platform !== "win32",
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let cancelled = false;
    let cancellationCheckRunning = false;
    let cancellationError: unknown;
    let forceKillTimer: NodeJS.Timeout | null = null;
    const killProcessGroup = (signal: NodeJS.Signals) => {
      if (child.pid && process.platform !== "win32") {
        try {
          process.kill(-child.pid, signal);
          return;
        } catch {
          // Fall through when the process group has already exited.
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
      cancelled = true;
      requestStop("SIGINT");
    };
    const onSigterm = () => {
      cancelled = true;
      requestStop("SIGTERM");
    };
    const timer = setTimeout(() => {
      timedOut = true;
      requestStop();
    }, args.timeoutMs);
    timer.unref();
    const cancellationTimer = args.shouldCancel
      ? setInterval(() => {
          if (cancellationCheckRunning || cancelled) return;
          cancellationCheckRunning = true;
          Promise.resolve(args.shouldCancel?.())
            .then((requested) => {
              if (!requested || cancelled) return;
              cancelled = true;
              requestStop();
            })
            .catch((error) => {
              cancellationError = error;
              requestStop();
            })
            .finally(() => { cancellationCheckRunning = false; });
        }, args.cancelPollMs ?? 250)
      : null;
    cancellationTimer?.unref();
    process.once("SIGINT", onSigint);
    process.once("SIGTERM", onSigterm);
    const cleanup = () => {
      clearTimeout(timer);
      if (cancellationTimer) clearInterval(cancellationTimer);
      if (forceKillTimer) clearTimeout(forceKillTimer);
      process.off("SIGINT", onSigint);
      process.off("SIGTERM", onSigterm);
    };

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => {
      cleanup();
      reject(error);
    });
    child.on("close", (code) => {
      cleanup();
      if (timedOut) {
        reject(new Error(args.timeoutMessage));
        return;
      }
      if (cancellationError) {
        reject(cancellationError);
        return;
      }
      if (cancelled) {
        reject(new ProcessCancelledError(`${args.errorPrefix} was cancelled.`));
        return;
      }
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
    child.stdin.on("error", () => undefined);
    child.stdin.end(JSON.stringify(args.payload));
  });
}
