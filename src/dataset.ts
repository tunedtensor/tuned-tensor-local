import { readFile } from "node:fs/promises";
import type { BehaviorSpecExample, SpecSnapshot } from "./contracts.js";

export interface ChatJsonlRow {
  messages: Array<{
    role: "system" | "user" | "assistant";
    content: string | Array<Record<string, unknown>>;
  }>;
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
  return parts.join("\n\n");
}

export function exampleToChatRow(spec: SpecSnapshot, example: BehaviorSpecExample): ChatJsonlRow {
  const system = buildSystemMessage(spec);
  const userContent = example.input_assets?.length
    ? [
        ...example.input_assets.map((asset) => ({
          type: "image",
          image: asset.image ?? asset.data_uri ?? asset.uri ?? asset.path,
          ...(asset.mime_type ? { mime_type: asset.mime_type } : {}),
          ...(asset.page ? { page: asset.page } : {}),
        })),
        { type: "text", text: example.input },
      ]
    : example.input;

  return {
    messages: [
      { role: "system", content: system },
      { role: "user", content: userContent },
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
  const text = await readFile(path, "utf8");
  const examples: BehaviorSpecExample[] = [];
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (!line.trim()) continue;
    const row = JSON.parse(line) as ChatJsonlRow;
    const user = row.messages.find((message) => message.role === "user");
    const assistant = [...row.messages].reverse().find((message) => message.role === "assistant");
    if (!user || !assistant || typeof assistant.content !== "string") {
      throw new Error(`Invalid chat JSONL row ${index + 1}: expected user and assistant messages`);
    }
    examples.push({
      input: typeof user.content === "string" ? user.content : JSON.stringify(user.content),
      output: assistant.content,
    });
  }
  return examples;
}
