/**
 * Secret/PII sanitization for labeling rows, ported from tuned-tensor-runs.
 * Even though tt-local runs on hardware the user controls, labeling sends row
 * content to OpenRouter — so secret-like content blocks the row and PII is
 * redacted before anything leaves the machine.
 */

export type SanitizationStatus = "clean" | "redacted" | "blocked";

export type SanitizationKind =
  | "api_key"
  | "bearer_token"
  | "connection_string"
  | "credit_card"
  | "email"
  | "password"
  | "phone"
  | "private_key"
  | "ssn";

export interface SanitizationFinding {
  kind: SanitizationKind;
  action: "redact" | "block";
  count: number;
}

export interface SanitizedText {
  text: string;
  status: SanitizationStatus;
  findings: SanitizationFinding[];
}

export interface SanitizedLabelingRow {
  input: string;
  output?: string;
  sanitizationStatus: SanitizationStatus;
  sanitizationFindings: SanitizationFinding[];
  sanitizationError?: string;
}

const BLOCK_PATTERNS: Array<{
  kind: SanitizationKind;
  replacement: string;
  regex: RegExp;
}> = [
  {
    kind: "private_key",
    replacement: "[REDACTED_PRIVATE_KEY]",
    regex:
      /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g,
  },
  {
    kind: "connection_string",
    replacement: "[REDACTED_CONNECTION_STRING]",
    regex:
      /\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis):\/\/[^\s:/@]+:[^\s@]+@[^\s]+/gi,
  },
  {
    kind: "bearer_token",
    replacement: "Bearer [REDACTED_TOKEN]",
    regex: /\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b/g,
  },
  {
    kind: "api_key",
    replacement: "[REDACTED_API_KEY]",
    regex:
      /\b(?:sk-[A-Za-z0-9]{20,}|(?:tt|ghp|gho|github_pat|xox[baprs])_[A-Za-z0-9_=-]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16}|AIza[0-9A-Za-z_-]{20,})\b/g,
  },
  {
    kind: "password",
    replacement: "$1 [REDACTED_SECRET]",
    regex:
      /\b((?:password|passwd|pwd|api[_-]?key|secret|token|access[_-]?token|refresh[_-]?token|client[_-]?secret)\s*[:=])\s*["']?[^"'\s,;]{8,}/gi,
  },
];

const REDACT_PATTERNS: Array<{
  kind: SanitizationKind;
  replacement: string;
  regex: RegExp;
}> = [
  {
    kind: "email",
    replacement: "[REDACTED_EMAIL]",
    regex: /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi,
  },
  {
    kind: "ssn",
    replacement: "[REDACTED_SSN]",
    regex: /\b\d{3}-\d{2}-\d{4}\b/g,
  },
  {
    kind: "phone",
    replacement: "[REDACTED_PHONE]",
    regex: /\b(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b/g,
  },
];

function addFinding(
  findings: Map<string, SanitizationFinding>,
  kind: SanitizationKind,
  action: "redact" | "block",
  count: number,
) {
  if (count <= 0) return;
  const key = `${kind}:${action}`;
  const existing = findings.get(key);
  findings.set(key, {
    kind,
    action,
    count: (existing?.count ?? 0) + count,
  });
}

function replacePattern(
  text: string,
  pattern: { kind: SanitizationKind; replacement: string; regex: RegExp },
  action: "redact" | "block",
  findings: Map<string, SanitizationFinding>,
): string {
  let count = 0;
  const next = text.replace(pattern.regex, (...args: unknown[]) => {
    count += 1;
    if (pattern.replacement.includes("$1") && typeof args[1] === "string") {
      return pattern.replacement.replace("$1", args[1]);
    }
    return pattern.replacement;
  });
  addFinding(findings, pattern.kind, action, count);
  return next;
}

function digitsOnly(value: string): string {
  return value.replace(/\D/g, "");
}

function passesLuhn(value: string): boolean {
  let sum = 0;
  let doubleDigit = false;
  for (let i = value.length - 1; i >= 0; i -= 1) {
    let digit = Number(value[i]);
    if (Number.isNaN(digit)) return false;
    if (doubleDigit) {
      digit *= 2;
      if (digit > 9) digit -= 9;
    }
    sum += digit;
    doubleDigit = !doubleDigit;
  }
  return sum > 0 && sum % 10 === 0;
}

function redactCreditCards(
  text: string,
  findings: Map<string, SanitizationFinding>,
): string {
  let count = 0;
  const next = text.replace(/\b(?:\d[ -]*?){13,19}\b/g, (match) => {
    const digits = digitsOnly(match);
    if (digits.length < 13 || digits.length > 19 || !passesLuhn(digits)) {
      return match;
    }
    count += 1;
    return "[REDACTED_CARD]";
  });
  addFinding(findings, "credit_card", "redact", count);
  return next;
}

export function sanitizeText(text: string): SanitizedText {
  const findings = new Map<string, SanitizationFinding>();
  let sanitized = text;

  for (const pattern of BLOCK_PATTERNS) {
    sanitized = replacePattern(sanitized, pattern, "block", findings);
  }

  sanitized = redactCreditCards(sanitized, findings);

  for (const pattern of REDACT_PATTERNS) {
    sanitized = replacePattern(sanitized, pattern, "redact", findings);
  }

  const allFindings = Array.from(findings.values());
  const hasBlock = allFindings.some((finding) => finding.action === "block");
  return {
    text: sanitized,
    status: hasBlock
      ? "blocked"
      : allFindings.length > 0
        ? "redacted"
        : "clean",
    findings: allFindings,
  };
}

export function mergeSanitizationFindings(
  findings: readonly SanitizationFinding[],
): SanitizationFinding[] {
  const merged = new Map<string, SanitizationFinding>();
  for (const finding of findings) {
    addFinding(merged, finding.kind, finding.action, finding.count);
  }
  return Array.from(merged.values());
}

export function sanitizeLabelingRow(row: {
  input: string;
  output?: string;
}): SanitizedLabelingRow {
  const input = sanitizeText(row.input);
  const output = row.output === undefined ? undefined : sanitizeText(row.output);
  const findings = mergeSanitizationFindings([
    ...input.findings,
    ...(output?.findings ?? []),
  ]);
  const status =
    input.status === "blocked" || output?.status === "blocked"
      ? "blocked"
      : input.status === "redacted" || output?.status === "redacted"
        ? "redacted"
        : "clean";

  return {
    input: input.text,
    ...(output !== undefined ? { output: output.text } : {}),
    sanitizationStatus: status,
    sanitizationFindings: findings,
    ...(status === "blocked"
      ? { sanitizationError: "Sensitive secret-like content was blocked" }
      : {}),
  };
}
