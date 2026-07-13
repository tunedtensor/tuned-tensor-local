import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { test } from "node:test";
import { promisify } from "node:util";
import { fineTuneRunRequestSchema, localRunnerConfigSchema } from "../src/contracts.js";
import { buildModelPrefetchPayload, prefetchBaseModel } from "../src/prefetch.js";
import { minimalMachineLearningEnvironment, resolveHuggingFaceCacheLayout, withHuggingFaceCacheEnvironment } from "../src/huggingface-cache.js";

const execFileAsync = promisify(execFile);

const request = fineTuneRunRequestSchema.parse({
  run_id: "11111111-1111-4111-8111-111111111111",
  user_id: "local-user",
  behavior_spec_id: "22222222-2222-4222-8222-222222222222",
  run_number: 1,
  spec_snapshot: {
    name: "Prefetch",
    description: "",
    system_prompt: "Return labels.",
    base_model: "qwen/qwen3.5-2b",
    examples: [{ input: "hello", output: "greeting" }],
  },
});

test("builds a model prefetch payload from spec and model cache config", () => {
  const config = localRunnerConfigSchema.parse({
    paths: {
      modelCache: ".cache/huggingface",
    },
  });

  assert.deepEqual(buildModelPrefetchPayload(request, config), {
    base_model: "Qwen/Qwen3.5-2B",
    loader: "causal_lm",
    trust_remote_code: true,
    requires_hf_token: false,
    model_cache: resolve(".cache/huggingface"),
  });
});

test("includes an explicitly pinned base-model revision in prefetch", () => {
  const revised = fineTuneRunRequestSchema.parse({
    ...request,
    hyperparameters: {
      ...request.hyperparameters,
      base_model_revision: "0123456789abcdef",
    },
  });
  assert.equal(
    buildModelPrefetchPayload(revised, localRunnerConfigSchema.parse({})).revision,
    "0123456789abcdef",
  );
});

test("modelCache is HF_HOME and all Hugging Face clients share its hub directory", () => {
  const configured = join(tmpdir(), "tt-local-huggingface-home");
  const layout = resolveHuggingFaceCacheLayout(configured);
  assert.deepEqual(layout, {
    hfHome: resolve(configured),
    hubCache: join(resolve(configured), "hub"),
  });

  const env = withHuggingFaceCacheEnvironment({
    HF_HOME: "/old/home",
    HF_HUB_CACHE: "/old/hub",
    HUGGINGFACE_HUB_CACHE: "/legacy/hub",
    TRANSFORMERS_CACHE: "/transformers/cache",
    PYTORCH_TRANSFORMERS_CACHE: "/pytorch/transformers/cache",
    PYTORCH_PRETRAINED_BERT_CACHE: "/pytorch/bert/cache",
    KEEP_ME: "yes",
  }, configured);
  assert.equal(env.HF_HOME, layout.hfHome);
  assert.equal(env.HF_HUB_CACHE, layout.hubCache);
  assert.equal(env.HUGGINGFACE_HUB_CACHE, layout.hubCache);
  assert.equal(env.TRANSFORMERS_CACHE, undefined);
  assert.equal(env.PYTORCH_TRANSFORMERS_CACHE, undefined);
  assert.equal(env.PYTORCH_PRETRAINED_BERT_CACHE, undefined);
  assert.equal(env.KEEP_ME, "yes");
});

test("inherited Hugging Face settings are normalized even without modelCache", () => {
  const fromHome = withHuggingFaceCacheEnvironment({
    HF_HOME: "/cache/home",
    TRANSFORMERS_CACHE: "/split/transformers",
  });
  assert.equal(fromHome.HF_HOME, "/cache/home");
  assert.equal(fromHome.HF_HUB_CACHE, join("/cache/home", "hub"));
  assert.equal(fromHome.TRANSFORMERS_CACHE, undefined);

  const fromHub = withHuggingFaceCacheEnvironment({ HF_HUB_CACHE: "/cache/custom-hub" });
  assert.equal(fromHub.HF_HOME, "/cache");
  assert.equal(fromHub.HF_HUB_CACHE, join("/cache", "hub"));
});

test("bundled ML children do not inherit unrelated shell credentials", () => {
  const source = {
    PATH: "/bin",
    CUDA_VISIBLE_DEVICES: "0",
    HF_HOME: "/cache/huggingface",
    OPENROUTER_API_KEY: "judge-secret",
    AWS_SECRET_ACCESS_KEY: "cloud-secret",
    HF_TOKEN: "gated-secret",
    HTTPS_PROXY: "https://proxy.example.test",
    UV_INDEX_URL: "https://user:password@example.test/simple",
  };
  const publicModel = minimalMachineLearningEnvironment(source);
  assert.equal(publicModel.PATH, "/bin");
  assert.equal(publicModel.CUDA_VISIBLE_DEVICES, "0");
  assert.equal(publicModel.HF_HOME, "/cache/huggingface");
  assert.equal(publicModel.OPENROUTER_API_KEY, undefined);
  assert.equal(publicModel.AWS_SECRET_ACCESS_KEY, undefined);
  assert.equal(publicModel.HF_TOKEN, undefined);
  assert.equal(publicModel.HTTPS_PROXY, undefined);
  assert.equal(publicModel.UV_INDEX_URL, undefined);
  assert.equal(publicModel.TORCH_DISABLE_NATIVE_JIT, "1");
  assert.equal(
    minimalMachineLearningEnvironment(source, { includeHfToken: true }).HF_TOKEN,
    "gated-secret",
  );
  assert.equal(
    minimalMachineLearningEnvironment({ ...source, TORCH_DISABLE_NATIVE_JIT: "0" }).TORCH_DISABLE_NATIVE_JIT,
    "0",
  );
});

test("bundled prefetch imports Hugging Face after setting HF_HOME and downloads under hub", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prefetch-cache-contract-test-"));
  try {
    const fakePackage = join(root, "fake-python", "huggingface_hub");
    await mkdir(fakePackage, { recursive: true });
    await writeFile(join(fakePackage, "__init__.py"), `
import os
import hashlib
from pathlib import Path

class _Constants:
    HF_HOME = os.environ["HF_HOME"]
    HF_HUB_CACHE = os.environ["HF_HUB_CACHE"]

constants = _Constants()

def snapshot_download(**kwargs):
    assert "cache_dir" not in kwargs, "prefetch must use the HF_HOME-derived hub cache"
    assert isinstance(kwargs.get("local_files_only"), bool)
    assert kwargs.get("revision") == "0123456789abcdef"
    repository = Path(constants.HF_HUB_CACHE) / "models--Qwen--Qwen3.5-2B"
    snapshot = repository / "snapshots" / "fake-revision"
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True, exist_ok=True)
    blobs.mkdir(parents=True, exist_ok=True)
    def add(name, contents, lfs=False):
        digest = hashlib.sha256(contents).hexdigest() if lfs else hashlib.sha1(
            f"blob {len(contents)}\\0".encode() + contents
        ).hexdigest()
        blob = blobs / digest
        if not blob.exists():
            blob.write_bytes(contents)
        link = snapshot / name
        if not link.exists():
            link.symlink_to(blob)
    add("config.json", b"{}")
    add("tokenizer_config.json", b"{}")
    add("tokenizer.json", b"{}")
    add("pytorch_model.bin", b"weights", lfs=True)
    return str(snapshot)
`, "utf8");
    const hfHome = join(root, "huggingface");
    const inputPath = join(root, "input.json");
    const outputPath = join(root, "output.json");
    await writeFile(inputPath, JSON.stringify({
      base_model: "Qwen/Qwen3.5-2B",
      revision: "0123456789abcdef",
      loader: "causal_lm",
      model_cache: hfHome,
    }), "utf8");

    const result = await execFileAsync("uv", [
      "run",
      "python",
      resolve("training/local-runner/src/prefetch.py"),
      "--input",
      inputPath,
      "--output",
      outputPath,
    ], {
      cwd: root,
      env: {
        ...process.env,
        PYTHONPATH: join(root, "fake-python"),
        HF_HOME: "/wrong/home",
        HF_HUB_CACHE: "/wrong/hub",
        TRANSFORMERS_CACHE: "/wrong/transformers",
      },
    });
    const output = JSON.parse(await readFile(outputPath, "utf8"));
    assert.equal(output.hf_home, hfHome);
    assert.equal(output.hub_cache, join(hfHome, "hub"));
    assert.equal(output.snapshot_path, join(hfHome, "hub", "models--Qwen--Qwen3.5-2B", "snapshots", "fake-revision"));
    assert.equal(output.snapshot_revision, "fake-revision");
    assert.equal(output.file_count, 4);
    assert.ok(output.size_bytes > 0);
    assert.equal(output.verified_blob_count, 4);
    assert.match(result.stdout, new RegExp(join(hfHome, "hub").replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));

    await writeFile(inputPath, JSON.stringify({
      base_model: "Qwen/Qwen3.5-2B",
      revision: "0123456789abcdef",
      loader: "causal_lm",
      model_cache: hfHome,
      local_files_only: true,
    }), "utf8");
    const verification = await execFileAsync("uv", [
      "run",
      "python",
      resolve("training/local-runner/src/prefetch.py"),
      "--input",
      inputPath,
      "--output",
      outputPath,
    ], {
      cwd: root,
      env: { ...process.env, PYTHONPATH: join(root, "fake-python") },
    });
    assert.match(verification.stdout, /Verifying cached Qwen\/Qwen3\.5-2B/);
    assert.equal(JSON.parse(await readFile(outputPath, "utf8")).verified_blob_count, 4);

    const cachedWeight = join(
      hfHome,
      "hub",
      "models--Qwen--Qwen3.5-2B",
      "snapshots",
      "fake-revision",
      "pytorch_model.bin",
    );
    await writeFile(await realpath(cachedWeight), "corrupt", "utf8");
    await assert.rejects(
      execFileAsync("uv", [
        "run",
        "python",
        resolve("training/local-runner/src/prefetch.py"),
        "--input",
        inputPath,
        "--output",
        outputPath,
      ], {
        cwd: root,
        env: { ...process.env, PYTHONPATH: join(root, "fake-python") },
      }),
      /checksum mismatch/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("bundled evaluator sets the shared cache before importing Transformers", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-evaluator-import-order-test-"));
  try {
    const fakePython = join(root, "fake-python");
    await mkdir(join(fakePython, "PIL"), { recursive: true });
    const cacheAssertion = `
import os
assert os.environ["HF_HOME"] == os.environ["TT_EXPECTED_HF_HOME"]
assert os.environ["HF_HUB_CACHE"] == os.path.join(os.environ["TT_EXPECTED_HF_HOME"], "hub")
assert "TRANSFORMERS_CACHE" not in os.environ
`;
    await writeFile(join(fakePython, "torch.py"), `${cacheAssertion}\n`, "utf8");
    await writeFile(join(fakePython, "peft.py"), `${cacheAssertion}\nclass PeftModel: pass\n`, "utf8");
    await writeFile(join(fakePython, "PIL", "__init__.py"), `${cacheAssertion}\nclass Image: pass\n`, "utf8");
    await writeFile(join(fakePython, "transformers.py"), `${cacheAssertion}
class AutoModelForCausalLM: pass
class AutoModelForImageTextToText: pass
class AutoProcessor: pass
class AutoTokenizer: pass
`, "utf8");
    const hfHome = join(root, "huggingface");
    const source = `
import importlib.util
spec = importlib.util.spec_from_file_location("tt_local_evaluate", ${JSON.stringify(resolve("training/local-runner/src/evaluate.py"))})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.configure_hugging_face_cache(${JSON.stringify(hfHome)})
module.import_runtime_dependencies()
`;
    await execFileAsync("uv", ["run", "python", "-c", source], {
      cwd: root,
      env: {
        ...process.env,
        PYTHONPATH: fakePython,
        TT_EXPECTED_HF_HOME: hfHome,
        HF_HOME: "/wrong/home",
        HF_HUB_CACHE: "/wrong/hub",
        TRANSFORMERS_CACHE: "/wrong/transformers",
      },
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("prefetch skips Hugging Face download when a local base model path is configured", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prefetch-test-"));
  try {
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      paths: {
        baseModel: join(root, "base-model"),
        modelCache: join(root, "cache"),
      },
    });
    const baseModel = join(root, "base-model");
    await mkdir(baseModel);
    await Promise.all([
      writeFile(join(baseModel, "config.json"), "{}"),
      writeFile(join(baseModel, "tokenizer_config.json"), "{}"),
      writeFile(join(baseModel, "tokenizer.json"), "{}"),
      writeFile(join(baseModel, "model.safetensors"), "weights"),
    ]);

    const report = await prefetchBaseModel({ request, config });
    assert.equal(report.ok, true);
    assert.equal(report.status, "skipped");
    assert.equal(report.base_model, "Qwen/Qwen3.5-2B");
    assert.equal(report.local_base_model_path, join(root, "base-model"));
    assert.match(report.reason ?? "", /local base-model artifact/);
    assert.ok(report.artifact_dir.startsWith(join(root, "artifacts", "prefetch")));
    assert.equal(report.command, undefined);

    const verification = await prefetchBaseModel({ request, config, localOnly: true });
    assert.equal(verification.status, "completed");
    assert.equal(verification.file_count, 4);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects an empty configured local base-model directory", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prefetch-invalid-base-test-"));
  try {
    const baseModel = join(root, "base-model");
    await mkdir(baseModel);
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      paths: { baseModel },
    });
    await assert.rejects(
      prefetchBaseModel({ request, config, localOnly: true }),
      /no non-empty Transformers model weights/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects a standalone file as paths.baseModel", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prefetch-file-base-test-"));
  try {
    const baseModel = join(root, "model.safetensors");
    await writeFile(baseModel, "weights");
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      paths: { baseModel },
    });
    await assert.rejects(
      prefetchBaseModel({ request, config, localOnly: true }),
      /must be a Hugging Face snapshot directory/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects a snapshot-shaped directory that contains only optimizer state", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-prefetch-optimizer-base-test-"));
  try {
    const baseModel = join(root, "base-model");
    await mkdir(baseModel);
    await Promise.all([
      writeFile(join(baseModel, "config.json"), "{}"),
      writeFile(join(baseModel, "tokenizer_config.json"), "{}"),
      writeFile(join(baseModel, "tokenizer.json"), "{}"),
      writeFile(join(baseModel, "optimizer.pt"), "not model weights"),
    ]);
    const config = localRunnerConfigSchema.parse({
      artifactRoot: join(root, "artifacts"),
      paths: { baseModel },
    });
    await assert.rejects(
      prefetchBaseModel({ request, config, localOnly: true }),
      /no non-empty Transformers model weights/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
