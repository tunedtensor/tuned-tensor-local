import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";

/**
 * TT Local treats `paths.modelCache` as Hugging Face's `HF_HOME`. Model
 * snapshots therefore live below `<modelCache>/hub`, matching the default
 * layout used by huggingface_hub and Transformers.
 */
export interface HuggingFaceCacheLayout {
  hfHome: string;
  hubCache: string;
}

/**
 * Builds a deliberately small environment for bundled ML processes. Model
 * repositories that require `trust_remote_code` execute Python supplied by a
 * third party, so unrelated shell credentials must not be inherited. Values
 * added explicitly through runner `env` configuration are applied afterward.
 */
export function minimalMachineLearningEnvironment(
  env: NodeJS.ProcessEnv,
  options: { includeHfToken?: boolean } = {},
): NodeJS.ProcessEnv {
  const exact = new Set([
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TEMP", "TMP",
    "LANG", "LANGUAGE", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "XDG_CACHE_HOME",
    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "NO_PROXY", "no_proxy",
    "UV_CACHE_DIR", "UV_PYTHON",
    "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_ENDPOINT",
    "HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TRANSFORMERS_OFFLINE",
    "HF_HUB_ENABLE_HF_TRANSFER", "PYTHONUTF8", "PYTHONIOENCODING",
  ]);
  const prefixes = [
    "LC_", "CUDA_", "NVIDIA_", "NCCL_", "OMP_", "MKL_", "TORCH_",
    "PYTORCH_", "TOKENIZERS_", "HF_XET_",
  ];
  const result = Object.fromEntries(Object.entries(env).filter(([key]) =>
    exact.has(key) || prefixes.some((prefix) => key.startsWith(prefix))
  ));
  // PyTorch 2.13 can route ordinary CUDA operations through optional Triton
  // JIT kernels. Triton's first-use helper compilation requires system Python
  // development headers, which are not part of a normal packaged TT Local
  // install. Prefer PyTorch's eager fallback unless an advanced runner config
  // explicitly opts back in with TORCH_DISABLE_NATIVE_JIT=0.
  result.TORCH_DISABLE_NATIVE_JIT ??= "1";
  if (options.includeHfToken && env.HF_TOKEN) result.HF_TOKEN = env.HF_TOKEN;
  return result;
}

export function resolveHuggingFaceCacheLayout(modelCache: string): HuggingFaceCacheLayout {
  const hfHome = resolve(modelCache);
  return {
    hfHome,
    hubCache: join(hfHome, "hub"),
  };
}

/**
 * Returns a child-process environment in which current Hugging Face cache
 * variables agree and deprecated Transformers overrides are removed. Explicit
 * TT Local configuration wins over inherited shell variables, avoiding a
 * second cache tree during prefetch, training, evaluation, or serving.
 */
export function withHuggingFaceCacheEnvironment(
  env: NodeJS.ProcessEnv,
  modelCache?: string,
): NodeJS.ProcessEnv {
  const inheritedHub = env.HF_HUB_CACHE ?? env.HUGGINGFACE_HUB_CACHE;
  const canonicalHome = modelCache
    ?? env.HF_HOME
    ?? (inheritedHub ? dirname(resolve(inheritedHub)) : join(homedir(), ".cache", "huggingface"));
  const layout = resolveHuggingFaceCacheLayout(canonicalHome);
  const result: NodeJS.ProcessEnv = {
    ...env,
    HF_HOME: layout.hfHome,
    HF_HUB_CACHE: layout.hubCache,
    HUGGINGFACE_HUB_CACHE: layout.hubCache,
  };
  // These deprecated Transformers variables override HF_HOME when inherited
  // and also emit warnings when present. Removing them lets Transformers use
  // the canonical HF_HUB_CACHE value above.
  delete result.TRANSFORMERS_CACHE;
  delete result.PYTORCH_TRANSFORMERS_CACHE;
  delete result.PYTORCH_PRETRAINED_BERT_CACHE;
  return result;
}
