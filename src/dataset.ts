import { readFile } from "node:fs/promises";
import type { BehaviorSpecExample, SpecSnapshot } from "./contracts.js";

export interface ChatJsonlRow {
  messages: Array<{
    role: "system" | "user" | "assistant";
    content: string | Array<Record<string, unknown>>;
  }>;
  images?: string[];
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
    const inputAssets: BehaviorSpecExample["input_assets"] = [];
    let textInput = "";
    if (typeof user.content === "string") {
      textInput = user.content;
      if (Array.isArray(row.images)) {
        for (const image of row.images) {
          if (typeof image === "string" && image) {
            inputAssets.push({ type: "image", image });
          }
        }
      }
    } else {
      const topLevelImages = Array.isArray(row.images) ? row.images : [];
      let imageIndex = 0;
      const textParts: string[] = [];
      for (const part of user.content) {
        if (part.type === "text" && typeof part.text === "string") {
          textParts.push(part.text);
          continue;
        }
        if (part.type !== "image") continue;
        const image = typeof part.image === "string"
          ? part.image
          : typeof part.path === "string"
            ? part.path
            : typeof part.uri === "string"
              ? part.uri
              : typeof part.data_uri === "string"
                ? part.data_uri
                : topLevelImages[imageIndex];
        imageIndex += 1;
        if (image) {
          inputAssets.push({
            type: "image",
            image,
            ...(typeof part.mime_type === "string" ? { mime_type: part.mime_type } : {}),
            ...(typeof part.page === "number" ? { page: part.page } : {}),
          });
        }
      }
      textInput = textParts.join("\n").trim() || JSON.stringify(user.content);
    }
    examples.push({
      input: textInput,
      output: assistant.content,
      ...(inputAssets.length ? { input_assets: inputAssets } : {}),
    });
  }
  return examples;
}
