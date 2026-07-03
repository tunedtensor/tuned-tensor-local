export interface OpenRouterMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface OpenRouterOptions {
  model: string;
  apiKeyEnv: string;
  appName?: string;
  siteUrl?: string;
  timeoutMs?: number;
  /** Defaults to 0 (judge behavior). */
  temperature?: number;
  /** Omitted from the request when unset. */
  maxTokens?: number;
  /** Defaults to "json_object" (judge behavior); "text" for free-text output. */
  responseFormat?: "json_object" | "text";
}

export interface OpenRouterChatResult {
  content: string;
  model?: string;
  promptTokens: number;
  completionTokens: number;
}

export class OpenRouterHttpError extends Error {
  constructor(
    message: string,
    public readonly statusCode: number,
  ) {
    super(message);
    this.name = "OpenRouterHttpError";
  }
}

export function isRetryableOpenRouterError(error: unknown): boolean {
  if (error instanceof OpenRouterHttpError) {
    return [408, 409, 425, 429, 500, 502, 503, 504].includes(error.statusCode);
  }
  if (!(error instanceof Error)) return false;
  return error.name === "AbortError" || error.name === "TimeoutError" || error.name === "TypeError";
}

export async function openRouterChat(messages: OpenRouterMessage[], options: OpenRouterOptions): Promise<OpenRouterChatResult> {
  const apiKey = process.env[options.apiKeyEnv];
  if (!apiKey) {
    throw new Error(`${options.apiKeyEnv} is not set`);
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? 120_000);
  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
        ...(options.siteUrl ? { "HTTP-Referer": options.siteUrl } : {}),
        ...(options.appName ? { "X-Title": options.appName } : {}),
      },
      body: JSON.stringify({
        model: options.model,
        messages,
        temperature: options.temperature ?? 0,
        ...(options.maxTokens !== undefined ? { max_tokens: options.maxTokens } : {}),
        ...((options.responseFormat ?? "json_object") === "json_object"
          ? { response_format: { type: "json_object" } }
          : {}),
      }),
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new OpenRouterHttpError(
        `OpenRouter request failed (${response.status}): ${text.slice(0, 500)}`,
        response.status,
      );
    }
    const data = await response.json() as {
      model?: string;
      choices?: Array<{ message?: { content?: string } }>;
      usage?: { prompt_tokens?: unknown; completion_tokens?: unknown };
    };
    const content = data.choices?.[0]?.message?.content;
    if (!content) throw new Error("OpenRouter response did not include message content");
    return {
      content,
      model: data.model,
      promptTokens: numericTokenCount(data.usage?.prompt_tokens),
      completionTokens: numericTokenCount(data.usage?.completion_tokens),
    };
  } finally {
    clearTimeout(timer);
  }
}

function numericTokenCount(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}
