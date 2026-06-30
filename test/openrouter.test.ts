import assert from "node:assert/strict";
import { test } from "node:test";
import { openRouterChat } from "../src/openrouter.js";

test("openRouterChat sends an OpenAI-compatible JSON chat request", async () => {
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.OPENROUTER_API_KEY;
  const calls: Array<{ input: RequestInfo | URL; init?: RequestInit }> = [];
  process.env.OPENROUTER_API_KEY = "test-key";
  globalThis.fetch = (async (input, init) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({
      model: "openai/gpt-5.5",
      choices: [{ message: { content: "{\"score\":1,\"passed\":true,\"reasoning\":\"ok\"}" } }],
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const result = await openRouterChat([
      { role: "user", content: "judge this" },
    ], {
      model: "openai/gpt-5.5",
      apiKeyEnv: "OPENROUTER_API_KEY",
      appName: "tt-local-test",
      siteUrl: "https://example.com",
      timeoutMs: 1000,
    });

    assert.equal(result.model, "openai/gpt-5.5");
    assert.equal(calls.length, 1);
    assert.equal(String(calls[0]?.input), "https://openrouter.ai/api/v1/chat/completions");
    assert.equal(calls[0]?.init?.headers && (calls[0].init.headers as Record<string, string>).Authorization, "Bearer test-key");
    const body = JSON.parse(String(calls[0]?.init?.body)) as { model: string; response_format: { type: string } };
    assert.equal(body.model, "openai/gpt-5.5");
    assert.equal(body.response_format.type, "json_object");
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) {
      delete process.env.OPENROUTER_API_KEY;
    } else {
      process.env.OPENROUTER_API_KEY = originalKey;
    }
  }
});
