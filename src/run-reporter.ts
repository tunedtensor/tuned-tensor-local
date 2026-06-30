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
    const lines = buffered.split(/\r?\n/);
    buffered = lines.pop() ?? "";
    for (const line of lines) {
      if (line.trim()) onLine(sanitizeLogLine(line));
    }
  });
  stream.on("end", () => {
    if (buffered.trim()) onLine(sanitizeLogLine(buffered));
  });
}
