import { readFile } from "node:fs/promises";
import type { BehaviorSpecExample, SpecSnapshot } from "./contracts.js";

export interface ChatJsonlRow {
  messages: Array<{
    role: "system" | "user" | "assistant";
    content: string;
  }>;
}

interface NormalizedChatJsonl {
  rows: ChatJsonlRow[];
  jsonl: string;
}

async function loadNormalizedChatJsonl(path: string): Promise<NormalizedChatJsonl> {
  const text = await readFile(path, "utf8");
  const rows: ChatJsonlRow[] = [];
  const jsonLines: string[] = [];
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (!line.trim()) continue;
    let value: unknown;
    try {
      value = JSON.parse(line) as unknown;
    } catch (error) {
      throw new Error(`Invalid chat JSONL row ${index + 1}: malformed JSON`, { cause: error });
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error(`Invalid chat JSONL row ${index + 1}: expected an object`);
    }
    const messages = (value as { messages?: unknown }).messages;
    if (
      !Array.isArray(messages)
      || ![2, 3].includes(messages.length)
      || messages.some((message) =>
        !message
        || typeof message !== "object"
        || Array.isArray(message)
        || !["system", "user", "assistant"].includes(String((message as { role?: unknown }).role))
        || typeof (message as { content?: unknown }).content !== "string"
      )
    ) {
      throw new Error(
        `Invalid chat JSONL row ${index + 1}: expected an optional system message, one user message, and one assistant answer`,
      );
    }
    const row = { messages } as ChatJsonlRow;
    const offset = row.messages.length === 3 ? 1 : 0;
    if (
      (offset === 1 && row.messages[0]?.role !== "system")
      || row.messages[offset]?.role !== "user"
      || !row.messages[offset]?.content.trim()
      || row.messages[offset + 1]?.role !== "assistant"
      || !row.messages[offset + 1]?.content.trim()
    ) {
      throw new Error(
        `Invalid chat JSONL row ${index + 1}: expected an optional system message, one non-empty user message, and one non-empty assistant answer`,
      );
    }
    rows.push(row);
    jsonLines.push(JSON.stringify(row));
  }
  if (rows.length === 0) throw new Error(`Chat JSONL contains no examples: ${path}`);
  return {
    rows,
    jsonl: jsonLines.join("\n"),
  };
}

/** Parses and canonically rewrites text chat JSONL before copying it. */
export async function normalizeChatJsonlForRelocation(path: string): Promise<string> {
  return (await loadNormalizedChatJsonl(path)).jsonl;
}

export function buildSystemMessage(spec: SpecSnapshot): string {
  const parts: string[] = [];
  if (spec.system_prompt.trim()) parts.push(spec.system_prompt.trim());
  if (spec.guidelines.length > 0) {
    parts.push(`Guidelines:\n${spec.guidelines.map((guideline) => `- ${guideline}`).join("\n")}`);
  }
  if (spec.constraints.length > 0) {
    parts.push(`Constraints:\n${spec.constraints.map((constraint) => `- ${constraint}`).join("\n")}`);
  }
  return parts.join("\n\n") || "Follow the demonstrated behavior.";
}

export function exampleToChatRow(spec: SpecSnapshot, example: BehaviorSpecExample): ChatJsonlRow {
  const system = buildSystemMessage(spec);
  return {
    messages: [
      { role: "system", content: system },
      { role: "user", content: example.input },
      { role: "assistant", content: example.output },
    ],
  };
}

export function compileSpecToJsonl(spec: SpecSnapshot): string {
  return spec.examples.map((example) => JSON.stringify(exampleToChatRow(spec, example))).join("\n");
}

export function examplesFromSpec(spec: SpecSnapshot): BehaviorSpecExample[] {
  return spec.examples;
}

export async function examplesFromChatJsonl(path: string): Promise<BehaviorSpecExample[]> {
  const normalized = await loadNormalizedChatJsonl(path);
  const examples: BehaviorSpecExample[] = [];
  for (const [index, row] of normalized.rows.entries()) {
    const user = row.messages.at(-2)!;
    const assistant = row.messages.at(-1)!;
    examples.push({
      input: user.content,
      output: assistant.content,
    });
  }
  return examples;
}

export interface EvaluationSuite {
  examples: BehaviorSpecExample[];
  system: string;
}

/**
 * Loads an evaluation-only chat JSONL suite. Rows may omit the system message,
 * but when they include one it must be identical across the suite.
 */
export async function evaluationSuiteFromChatJsonl(
  path: string,
  configuredSystem?: string,
): Promise<EvaluationSuite> {
  const normalized = await loadNormalizedChatJsonl(path);
  const inputIdentities = new Set<string>();
  for (const row of normalized.rows) {
    const input = row.messages.at(-2)!.content;
    const identity = input.trim().replace(/\s+/g, " ").toLowerCase();
    if (inputIdentities.has(identity)) {
      throw new Error(
        "General regression data contains duplicate inputs; "
        + "evaluation prompts must be unique.",
      );
    }
    inputIdentities.add(identity);
  }
  const systems = new Set(
    normalized.rows
      .map((row) => row.messages.find((message) => message.role === "system")?.content)
      .filter((value): value is string => value !== undefined),
  );
  if (systems.size > 1) {
    throw new Error("General regression rows must use one consistent system message.");
  }
  const embeddedSystem = [...systems][0];
  if (
    configuredSystem !== undefined
    && embeddedSystem !== undefined
    && configuredSystem !== embeddedSystem
  ) {
    throw new Error(
      "evaluation.generalRegression.systemPrompt must match the system message embedded in the dataset.",
    );
  }
  return {
    examples: normalized.rows.map((row) => ({
      input: row.messages.at(-2)!.content,
      output: row.messages.at(-1)!.content,
    })),
    system: configuredSystem ?? embeddedSystem ?? "You are a helpful assistant.",
  };
}
