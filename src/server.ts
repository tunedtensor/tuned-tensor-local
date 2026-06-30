import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFile, stat } from "node:fs/promises";
import type { AddressInfo } from "node:net";
import { createLocalStore } from "./store.js";
import { fineTuneRunRequestSchema, localRunnerConfigSchema, type LocalRunnerConfig } from "./contracts.js";
import { runLocalFineTune } from "./orchestrator.js";

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Cache-Control": "no-store",
  });
  res.end(`${JSON.stringify(body, null, 2)}\n`);
}

function sendText(res: ServerResponse, status: number, body: string, contentType = "text/plain; charset=utf-8"): void {
  res.writeHead(status, {
    "Content-Type": contentType,
    "Cache-Control": "no-store",
  });
  res.end(body);
}

async function readBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  const text = Buffer.concat(chunks).toString("utf8").trim();
  return text ? JSON.parse(text) : {};
}

const DASHBOARD_HTML = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tuned Tensor Local</title>
  <style>
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f7f4; color: #1e2320; }
    header { display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; border-bottom: 1px solid #d9ddd6; background: #ffffff; position: sticky; top: 0; }
    main { display: grid; grid-template-columns: minmax(280px, 420px) 1fr; min-height: calc(100vh - 65px); }
    aside { border-right: 1px solid #d9ddd6; padding: 16px; overflow: auto; }
    section { padding: 20px; overflow: auto; }
    button { border: 1px solid #a8b0a8; background: #ffffff; padding: 7px 10px; border-radius: 6px; cursor: pointer; }
    .run { width: 100%; text-align: left; display: grid; gap: 4px; margin-bottom: 8px; }
    .muted { color: #667066; font-size: 12px; }
    .pill { display: inline-block; border: 1px solid #b8c0b6; border-radius: 999px; padding: 2px 8px; font-size: 12px; background: #fff; }
    pre { background: #1f2420; color: #eef2ed; padding: 12px; overflow: auto; border-radius: 6px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .metric { border: 1px solid #d9ddd6; background: #fff; border-radius: 6px; padding: 12px; }
    @media (max-width: 760px) { main { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid #d9ddd6; } }
  </style>
</head>
<body>
  <header>
    <strong>Tuned Tensor Local</strong>
    <button id="refresh">Refresh</button>
  </header>
  <main>
    <aside><div id="runs"></div></aside>
    <section><div id="detail" class="muted">Select a run.</div></section>
  </main>
  <script>
    let selected = null;
    let events = null;
    const fmt = (v) => v == null ? "—" : String(v);
    async function api(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }
    async function loadRuns() {
      const runs = await api("/api/runs");
      document.getElementById("runs").innerHTML = runs.map(run => \`
        <button class="run" data-run="\${run.id}">
          <span><strong>\${run.spec_name}</strong> <span class="pill">\${run.status}</span></span>
          <span class="muted">\${run.id.slice(0, 8)} · \${run.current_stage} · \${run.base_model}</span>
        </button>\`).join("") || '<p class="muted">No runs yet.</p>';
      for (const el of document.querySelectorAll("[data-run]")) {
        el.onclick = () => showRun(el.getAttribute("data-run"));
      }
      if (!selected && runs[0]) showRun(runs[0].id);
    }
    async function showRun(id) {
      selected = id;
      if (events) events.close();
      const run = await api("/api/runs/" + id);
      let report = null;
      try { report = await api("/api/runs/" + id + "/report"); } catch {}
      document.getElementById("detail").innerHTML = \`
        <h2>\${run.spec_name}</h2>
        <p><span class="pill">\${run.status}</span> <span class="muted">\${run.id}</span></p>
        <div class="grid">
          <div class="metric"><div class="muted">Stage</div><strong>\${run.current_stage}</strong></div>
          <div class="metric"><div class="muted">Base Model</div><strong>\${run.base_model}</strong></div>
          <div class="metric"><div class="muted">Model ID</div><strong>\${fmt(run.model_id)}</strong></div>
          <div class="metric"><div class="muted">Updated</div><strong>\${run.updated_at}</strong></div>
        </div>
        <h3>Report</h3>
        <pre>\${report ? JSON.stringify({
          avg_score_delta: report.comparison?.avg_score_delta,
          pass_rate_delta: report.comparison?.pass_rate_delta,
          artifact: report.fine_tuned_model_id,
          metrics: report.training?.metrics,
        }, null, 2) : "Report not available yet."}</pre>
        <h3>Events</h3>
        <pre id="events"></pre>\`;
      const eventBox = document.getElementById("events");
      const history = await api("/api/runs/" + id + "/events");
      eventBox.textContent = history.map(e => \`\${e.occurred_at} \${e.stage}: \${e.message}\`).join("\\n");
      events = new EventSource("/api/runs/" + id + "/events/stream");
      events.onmessage = (msg) => {
        const e = JSON.parse(msg.data);
        eventBox.textContent += "\\n" + \`\${e.occurred_at} \${e.stage}: \${e.message}\`;
      };
    }
    document.getElementById("refresh").onclick = loadRuns;
    loadRuns().catch(err => document.getElementById("detail").textContent = err.message);
  </script>
</body>
</html>`;

async function streamRunEvents(req: IncomingMessage, res: ServerResponse, storeRoot: string, runId: string): Promise<void> {
  const store = createLocalStore(storeRoot);
  const run = await store.getRun(runId);
  const eventPath = `${store.paths.runsDir}/${run.id}/progress.jsonl`;
  let offset = 0;
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });
  const sendNew = async () => {
    const size = await stat(eventPath).then((s) => s.size).catch(() => 0);
    if (size <= offset) return;
    const text = await readFile(eventPath, "utf8");
    const chunk = text.slice(offset);
    offset = text.length;
    for (const line of chunk.split(/\r?\n/).filter(Boolean)) {
      res.write(`data: ${line}\n\n`);
    }
  };
  await sendNew();
  const timer = setInterval(() => { sendNew().catch(() => undefined); }, 1000);
  req.on("close", () => clearInterval(timer));
}

export async function serveLocalDashboard(options: {
  host?: string;
  port?: number;
  config?: LocalRunnerConfig;
} = {}): Promise<{ url: string; close(): Promise<void> }> {
  const host = options.host ?? "127.0.0.1";
  const port = options.port ?? 8787;
  const config = options.config ?? localRunnerConfigSchema.parse({});
  const storeRoot = config.storeRoot;
  const store = createLocalStore(storeRoot);
  await store.ensure();

  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
      const path = url.pathname;
      if (req.method === "GET" && path === "/") return sendText(res, 200, DASHBOARD_HTML, "text/html; charset=utf-8");
      if (req.method === "GET" && path === "/api/health") return sendJson(res, 200, { ok: true, store_root: store.root });
      if (req.method === "GET" && path === "/api/runs") return sendJson(res, 200, await store.listRuns());
      if (req.method === "GET" && path === "/api/models") return sendJson(res, 200, await store.listModels());
      if (req.method === "GET" && path === "/api/specs") return sendJson(res, 200, await store.listSpecs());

      const runMatch = path.match(/^\/api\/runs\/([^/]+)(?:\/(events|events\/stream|report|cancel))?$/);
      if (runMatch) {
        const id = runMatch[1];
        const tail = runMatch[2];
        if (req.method === "GET" && !tail) return sendJson(res, 200, await store.getRun(id));
        if (req.method === "GET" && tail === "events") return sendJson(res, 200, await store.getRunEvents(id));
        if (req.method === "GET" && tail === "events/stream") return streamRunEvents(req, res, store.root, id);
        if (req.method === "GET" && tail === "report") return sendJson(res, 200, await store.getRunReport(id));
        if (req.method === "POST" && tail === "cancel") {
          await store.cancelRun(id);
          return sendJson(res, 200, { ok: true });
        }
      }

      const modelMatch = path.match(/^\/api\/models\/([^/]+)$/);
      if (req.method === "GET" && modelMatch) return sendJson(res, 200, await store.getModel(modelMatch[1]));
      const specMatch = path.match(/^\/api\/specs\/([^/]+)$/);
      if (req.method === "GET" && specMatch) return sendJson(res, 200, await store.getSpec(specMatch[1]));

      if (req.method === "POST" && path === "/api/runs") {
        const body = await readBody(req) as { request?: unknown; config?: unknown };
        const request = fineTuneRunRequestSchema.parse(body.request ?? body);
        const runConfig = localRunnerConfigSchema.parse({ ...config, ...(body.config ?? {}) });
        void runLocalFineTune({ request, config: runConfig }).catch((error) => {
          console.error("[tt-local.serve.run]", error);
        });
        return sendJson(res, 202, { ok: true, run_id: request.run_id });
      }

      sendJson(res, 404, { error: "not_found" });
    } catch (error) {
      sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) });
    }
  });

  await new Promise<void>((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(port, host, () => resolveListen());
  });
  const address = server.address() as AddressInfo;
  return {
    url: `http://${host}:${address.port}/`,
    close: async () => await new Promise<void>((resolveClose) => server.close(() => resolveClose())),
  };
}
