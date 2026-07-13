export interface LocalRunProgressLog {
  stage: string;
  stream?: "stdout" | "stderr" | "info";
  message: string;
}

export interface LocalRunProgressEvent {
  stage: string;
  status: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface LocalRunReporter {
  verbose?: boolean;
  onEvent?(event: LocalRunProgressEvent): void | Promise<void>;
  onLog?(log: LocalRunProgressLog): void | Promise<void>;
}

/** Optional high-volume reporting must never crash or orphan the workload. */
export function reportInBackground(action: () => void | Promise<void>): void {
  try {
    void Promise.resolve(action()).catch(() => undefined);
  } catch {
    // Progress/log reporting is best effort. Cancellation is enforced through
    // the process runner's explicit polling callback.
  }
}

export function sanitizeLogLine(line: string): string {
  return line
    .replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, "$1[redacted]")
    .replace(/((?:api[_-]?key|token|secret|password)=)[^\s]+/gi, "$1[redacted]");
}

export function forwardStreamLines(
  stream: NodeJS.ReadableStream,
  onLine: (line: string) => void,
): void {
  let buffered = "";
  stream.setEncoding("utf8");
  stream.on("data", (chunk: string) => {
    buffered += chunk;
    const lines = buffered.split(/\r\n|\n|\r/);
    buffered = lines.pop() ?? "";
    for (const line of lines) {
      if (line.trim()) onLine(sanitizeLogLine(line));
    }
  });
  stream.on("end", () => {
    if (buffered.trim()) onLine(sanitizeLogLine(buffered));
  });
}
