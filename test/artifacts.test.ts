import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { test } from "node:test";
import { gzipSync } from "node:zlib";
import {
  assertArtifactManifest,
  claimRunArtifactDirectory,
  fileUri,
  prepareRunDirectories,
  resolveRunArtifacts,
  verifyArtifactManifest,
  writeArtifactManifest,
  type ArtifactManifestModel,
} from "../src/artifacts.js";

function writeTarOctal(header: Buffer, offset: number, length: number, value: number): void {
  header.write(`${value.toString(8).padStart(length - 1, "0")}\0`, offset, length, "ascii");
}

function tarEntry(name: string, contents: Buffer): Buffer {
  const header = Buffer.alloc(512);
  header.write(name, 0, 100, "utf8");
  writeTarOctal(header, 100, 8, 0o644);
  writeTarOctal(header, 108, 8, 0);
  writeTarOctal(header, 116, 8, 0);
  writeTarOctal(header, 124, 12, contents.length);
  writeTarOctal(header, 136, 12, 0);
  header.fill(0x20, 148, 156);
  header.write("0", 156, 1, "ascii");
  header.write("ustar\0", 257, 6, "ascii");
  header.write("00", 263, 2, "ascii");
  const checksum = header.reduce((sum, byte) => sum + byte, 0);
  header.write(`${checksum.toString(8).padStart(6, "0")}\0 `, 148, 8, "ascii");

  const padding = Buffer.alloc((512 - (contents.length % 512)) % 512);
  return Buffer.concat([header, contents, padding]);
}

function tarGzipEntries(entries: Array<{ name: string; contents: Buffer }>): Buffer {
  return gzipSync(Buffer.concat([
    ...entries.map((entry) => tarEntry(entry.name, entry.contents)),
    Buffer.alloc(1024),
  ]));
}

function tarGzip(name: string, contents: Buffer): Buffer {
  return tarGzipEntries([
    { name, contents },
    ...(name === "adapter_config.json"
      ? []
      : [{ name: "adapter_config.json", contents: Buffer.from("{}") }]),
  ]);
}

function modelContract(modelPath: string): Omit<ArtifactManifestModel, "files"> {
  return {
    artifact_kind: "file",
    format: "tar.gz",
    framework: "transformers-peft",
    base_model: "Qwen/Qwen3.5-2B",
    base_model_revision: "revision-123",
    artifact_uri: fileUri(modelPath),
    artifact_root: modelPath,
    servable: true,
  };
}

test("writes and verifies an atomic artifact manifest with model contract metadata", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-artifacts-test-"));
  try {
    const artifacts = resolveRunArtifacts({ artifactRoot: root, prefix: "runs/run-1" });
    const modelPath = join(root, "adapter.tar.gz");
    await prepareRunDirectories(artifacts);
    await writeFile(artifacts.trainingJsonl, '{"messages":[]}\n', "utf8");
    await writeFile(modelPath, tarGzip("adapter_model.safetensors", Buffer.from("model weights")));

    const manifest = await writeArtifactManifest(artifacts, { model: modelContract(modelPath) });
    const persisted = JSON.parse(await readFile(artifacts.artifactManifestJson, "utf8")) as typeof manifest;

    assert.equal(persisted.schema_version, 1);
    assert.ok(persisted.generated_at);
    assert.deepEqual(persisted.files.map((file) => file.path), ["training.jsonl"]);
    assert.match(persisted.files[0]!.sha256, /^[a-f0-9]{64}$/);
    assert.deepEqual(persisted.model, manifest.model);
    assert.equal(persisted.model?.artifact_kind, "file");
    assert.equal(persisted.model?.format, "tar.gz");
    assert.equal(persisted.model?.framework, "transformers-peft");
    assert.equal(persisted.model?.base_model, "Qwen/Qwen3.5-2B");
    assert.equal(persisted.model?.base_model_revision, "revision-123");
    assert.equal(persisted.model?.artifact_uri, fileUri(modelPath));
    assert.equal(persisted.model?.artifact_root, modelPath);
    assert.equal(persisted.model?.servable, true);
    assert.equal(persisted.model?.files[0]?.path, basename(modelPath));
    assert.equal((await readdir(artifacts.runDir)).some((name) => name.endsWith(".tmp")), false);

    const verification = await verifyArtifactManifest(artifacts.artifactManifestJson);
    assert.equal(verification.valid, true);
    assert.deepEqual(verification.missing, []);
    assert.deepEqual(verification.changed, []);
    assert.deepEqual(verification.unexpected, []);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects an artifact whose checksum changes after manifest creation", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-artifacts-test-"));
  try {
    const artifacts = resolveRunArtifacts({ artifactRoot: root, prefix: "runs/run-2" });
    await prepareRunDirectories(artifacts);
    await writeFile(artifacts.trainingJsonl, '{"messages":[]}\n', "utf8");
    await writeArtifactManifest(artifacts);

    await writeFile(artifacts.trainingJsonl, '{"messages":[{"role":"user"}]}\n', "utf8");

    const verification = await verifyArtifactManifest(artifacts.artifactManifestJson);
    assert.equal(verification.valid, false);
    assert.deepEqual(verification.changed, ["training.jsonl"]);
    await assert.rejects(
      assertArtifactManifest(artifacts.artifactManifestJson),
      /Artifact integrity verification failed: changed=training\.jsonl/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("refuses corrupt and empty tar.gz model artifacts before writing a manifest", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-artifacts-test-"));
  try {
    const artifacts = resolveRunArtifacts({ artifactRoot: root, prefix: "runs/run-3" });
    await prepareRunDirectories(artifacts);
    await writeFile(artifacts.trainingJsonl, '{"messages":[]}\n', "utf8");

    const emptyPath = join(root, "empty-model.tar.gz");
    await writeFile(emptyPath, gzipSync(Buffer.alloc(1024)));
    await assert.rejects(
      writeArtifactManifest(artifacts, { model: modelContract(emptyPath) }),
      /Empty tar archive/,
    );

    const corruptPath = join(root, "corrupt-model.tar.gz");
    const corruptTar = Buffer.alloc(512);
    corruptTar.write("adapter_model.safetensors", 0, "utf8");
    await writeFile(corruptPath, gzipSync(Buffer.concat([corruptTar, Buffer.alloc(1024)])));
    await assert.rejects(
      writeArtifactManifest(artifacts, { model: modelContract(corruptPath) }),
      /Invalid tar checksum field|Tar member checksum mismatch/,
    );

    await assert.rejects(readFile(artifacts.artifactManifestJson), { code: "ENOENT" });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects artifact prefixes that escape the configured root", () => {
  assert.throws(
    () => resolveRunArtifacts({ artifactRoot: "/tmp/tt-local-root", prefix: "../outside" }),
    /escapes artifactRoot/,
  );
});

test("refuses symbolic links in artifact trees", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-artifacts-symlink-test-"));
  try {
    const artifacts = resolveRunArtifacts({ artifactRoot: root, prefix: "runs/run-symlink" });
    await prepareRunDirectories(artifacts);
    const outside = join(root, "outside.safetensors");
    await writeFile(outside, "weights", "utf8");
    await symlink(outside, join(artifacts.runDir, "linked.safetensors"));
    await assert.rejects(writeArtifactManifest(artifacts), /must not contain symbolic links/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects a symlink in an intermediate artifact prefix", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-artifacts-prefix-symlink-test-"));
  try {
    const outside = join(root, "outside");
    await mkdir(outside);
    await symlink(outside, join(root, "runs"));
    const artifacts = resolveRunArtifacts({ artifactRoot: root, prefix: "runs/run-symlink" });
    await assert.rejects(prepareRunDirectories(artifacts), /must not contain symbolic links/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("durably rejects two run identities that use the same artifact prefix", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-artifact-owner-test-"));
  try {
    const artifacts = resolveRunArtifacts({ artifactRoot: root, prefix: "shared-prefix" });
    const first = {
      artifacts,
      runId: "11111111-1111-4111-8111-111111111111",
      userId: "local-user",
      behaviorSpecId: "22222222-2222-4222-8222-222222222222",
    };
    await claimRunArtifactDirectory(first);
    await claimRunArtifactDirectory(first);
    await assert.rejects(
      claimRunArtifactDirectory({
        ...first,
        runId: "33333333-3333-4333-8333-333333333333",
      }),
      /already owned by run 11111111-1111-4111-8111-111111111111/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("requires recognized weights and adapter metadata for PEFT archives", async () => {
  const root = await mkdtemp(join(tmpdir(), "tt-local-artifact-contract-test-"));
  try {
    const artifacts = resolveRunArtifacts({ artifactRoot: root, prefix: "contract" });
    await prepareRunDirectories(artifacts);

    const notesOnly = join(root, "notes.tar.gz");
    await writeFile(notesOnly, tarGzipEntries([{ name: "notes.txt", contents: Buffer.from("hello") }]));
    await assert.rejects(
      writeArtifactManifest(artifacts, { model: modelContract(notesOnly) }),
      /no adapter_model\.safetensors or adapter_model\.bin/,
    );

    const missingConfig = join(root, "weights-only.tar.gz");
    await writeFile(missingConfig, tarGzipEntries([
      { name: "adapter_model.safetensors", contents: Buffer.from("weights") },
    ]));
    await assert.rejects(
      writeArtifactManifest(artifacts, { model: modelContract(missingConfig) }),
      /no adapter_config\.json/,
    );

    const optimizerOnly = join(root, "optimizer-only.tar.gz");
    await writeFile(optimizerOnly, tarGzipEntries([
      { name: "optimizer.pt", contents: Buffer.from("optimizer state") },
      { name: "adapter_config.json", contents: Buffer.from("{}") },
    ]));
    await assert.rejects(
      writeArtifactManifest(artifacts, { model: modelContract(optimizerOnly) }),
      /no adapter_model\.safetensors or adapter_model\.bin/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
