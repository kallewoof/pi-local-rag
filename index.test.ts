import { describe, it, expect, vi, beforeAll, afterAll } from "vitest";
import { mkdirSync, writeFileSync, rmSync, readFileSync } from "node:fs";
import { join } from "node:path";

// Hoisted so vi.mock factories can close over it
const TEST_HOME = vi.hoisted(() => `/tmp/pi-rag-test-${process.pid}`);

vi.mock("node:os", () => ({ homedir: () => TEST_HOME }));

// Prevent real ONNX model downloads; return a fixed 384-dim vector
vi.mock("@xenova/transformers", () => ({
  pipeline: vi.fn().mockResolvedValue(
    vi.fn().mockResolvedValue({ data: new Float32Array(384).fill(0.1) }),
  ),
}));

import { isIndexStale } from "./index.js";
import defaultExport from "./index.js";

// ─── Helpers ─────────────────────────────────────────────────────────────────

const RAG_DIR = `${TEST_HOME}/.pi/rag`;
const INDEX_FILE = join(RAG_DIR, "index.json");
const CONFIG_FILE = join(RAG_DIR, "config.json");
const DAY_MS = 24 * 60 * 60 * 1000;

const DEFAULT_CONFIG = { ragEnabled: true, ragTopK: 5, ragScoreThreshold: 0.1, ragAlpha: 0.4 };

function staleTimestamp() { return new Date(Date.now() - DAY_MS - 1_000).toISOString(); }
function freshTimestamp() { return new Date(Date.now() - 60_000).toISOString(); }

function writeIndex(data: object) {
  mkdirSync(RAG_DIR, { recursive: true });
  writeFileSync(INDEX_FILE, JSON.stringify(data));
}

function readIndex(): Record<string, any> {
  return JSON.parse(readFileSync(INDEX_FILE, "utf-8"));
}

/** Minimal chunk with a pre-filled vector to pass the `!index.chunks.length` guard */
function fakeChunk(file: string) {
  return {
    id: "test", file, content: "const x = 1;", lineStart: 1, lineEnd: 1,
    hash: "abc", indexed: new Date().toISOString(), tokens: 5,
    vector: new Array(384).fill(0.1),
  };
}

function makePi() {
  let hookFn: ((event: any, ctx: any) => Promise<any>) | undefined;
  const pi = {
    on: vi.fn((event: string, fn: any) => { if (event === "before_agent_start") hookFn = fn; }),
    registerCommand: vi.fn(),
    registerTool: vi.fn(),
  };
  const fireHook = (event = { prompt: "hello world", systemPrompt: "" }) => hookFn!(event, {});
  return { pi, fireHook };
}

// ─── isIndexStale ─────────────────────────────────────────────────────────────

describe("isIndexStale", () => {
  it("returns false when lastBuild is empty", () => {
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: "" } as any)).toBe(false);
  });

  it("returns false when index was built recently", () => {
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: freshTimestamp() } as any)).toBe(false);
  });

  it("returns true when lastBuild is more than 24h ago", () => {
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: staleTimestamp() } as any)).toBe(true);
  });

  it("respects a custom maxAgeMs", () => {
    const tenMinAgo = new Date(Date.now() - 10 * 60 * 1_000).toISOString();
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: tenMinAgo } as any, 5 * 60 * 1_000)).toBe(true);
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: tenMinAgo } as any, 15 * 60 * 1_000)).toBe(false);
  });
});

// ─── before_agent_start auto-rebuild ─────────────────────────────────────────

describe("before_agent_start auto-rebuild", () => {
  beforeAll(() => {
    mkdirSync(RAG_DIR, { recursive: true });
    writeFileSync(CONFIG_FILE, JSON.stringify(DEFAULT_CONFIG));
  });

  afterAll(() => {
    rmSync(TEST_HOME, { recursive: true, force: true });
  });

  it("does not update lastBuild when index is fresh", async () => {
    const freshBuild = freshTimestamp();
    writeIndex({ chunks: [fakeChunk("/some/file.ts")], files: {}, lastBuild: freshBuild });

    const { pi, fireHook } = makePi();
    defaultExport(pi as any);
    await fireHook();

    expect(readIndex().lastBuild).toBe(freshBuild);
  });

  it("updates lastBuild when index is stale and files exist on disk", async () => {
    const testFile = join(TEST_HOME, "sample.ts");
    writeFileSync(testFile, "export const answer = 42;\n");

    const staleBuild = staleTimestamp();
    writeIndex({
      chunks: [fakeChunk(testFile)],
      files: { [testFile]: { hash: "old", chunks: 1, indexed: staleBuild, size: 26, embedded: true } },
      lastBuild: staleBuild,
    });

    const { pi, fireHook } = makePi();
    defaultExport(pi as any);
    await fireHook();

    const updated = readIndex();
    expect(new Date(updated.lastBuild).getTime()).toBeGreaterThan(new Date(staleBuild).getTime());
  });

  it("does not update lastBuild when stale but all referenced files are gone", async () => {
    const staleBuild = staleTimestamp();
    const missingFile = join(TEST_HOME, "deleted.ts");
    writeIndex({
      chunks: [fakeChunk(missingFile)],
      files: { [missingFile]: { hash: "old", chunks: 1, indexed: staleBuild, size: 10, embedded: true } },
      lastBuild: staleBuild,
    });

    const { pi, fireHook } = makePi();
    defaultExport(pi as any);
    await fireHook();

    expect(readIndex().lastBuild).toBe(staleBuild);
  });
});
