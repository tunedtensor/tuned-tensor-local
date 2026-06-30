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
}

export interface OpenRouterChatResult {
  content: string;
  model?: string;
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
        temperature: 0,
        response_format: { type: "json_object" },
      }),
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(`OpenRouter request failed (${response.status}): ${text.slice(0, 500)}`);
    }
    const data = await response.json() as {
      model?: string;
      choices?: Array<{ message?: { content?: string } }>;
    };
    const content = data.choices?.[0]?.message?.content;
    if (!content) throw new Error("OpenRouter response did not include message content");
    return { content, model: data.model };
  } finally {
    clearTimeout(timer);
  }
}
