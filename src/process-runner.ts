import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { createWriteStream, readFileSync, type WriteStream } from "node:fs";
import { mkdir } from "node:fs/promises";
import { homedir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { forwardStreamLines, reportInBackground, type LocalRunReporter } from "./run-reporter.js";

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const bundledProject = join(packageRoot, "training/local-runner");
const bundledRuntimeHash = (() => {
  const hash = createHash("sha256");
  for (const name of ["pyproject.toml", "uv.lock"]) {
    hash.update(readFileSync(join(bundledProject, name)));
  }
  return hash.digest("hex").slice(0, 20);
})();
export const BUNDLED_PYTHON_ENVIRONMENT = join(
  resolve(process.env.XDG_CACHE_HOME ?? join(homedir(), ".cache")),
  "tuned-tensor-local",
  "uv",
  bundledRuntimeHash,
);
type BundledPythonEntrypoint =
  | "train.py"
  | "evaluate.py"
  | "prefetch.py"
  | "serve.py"
  | "-c";

/** Build a command for the one locked Python runtime shipped with TT Local. */
export function buildBundledPythonCommand(
  entrypoint: BundledPythonEntrypoint,
  args: string[] = [],
): { command: "uv"; commandArgs: string[]; displayCommand: string[] } {
  const target = entrypoint === "-c"
    ? entrypoint
    : join(bundledProject, "src", entrypoint);
  const commandArgs = [
    "run",
    "--frozen",
    "--project",
    bundledProject,
    "python",
    target,
    ...args,
  ];
  return {
    command: "uv",
    commandArgs,
    displayCommand: ["uv", ...commandArgs],
  };
}

/** Keep uv's mutable virtualenv outside a possibly read-only npm install. */
export function withBundledPythonEnvironment(
  env: NodeJS.ProcessEnv,
): NodeJS.ProcessEnv {
  return {
    ...env,
    UV_PROJECT_ENVIRONMENT: BUNDLED_PYTHON_ENVIRONMENT,
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
