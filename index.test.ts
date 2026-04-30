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
  let ragHandler: ((args: string, ctx: any) => Promise<any>) | undefined;
  const messages: string[] = [];
  const pi = {
    on: vi.fn((event: string, fn: any) => { if (event === "before_agent_start") hookFn = fn; }),
    registerCommand: vi.fn((name: string, def: any) => { if (name === "rag") ragHandler = def.handler; }),
    registerTool: vi.fn(),
    sendMessage: vi.fn((m: any) => { messages.push(m.content); }),
  };
  const ctx = {
    ui: {
      notify: vi.fn(),
      setStatus: vi.fn(),
      setWidget: vi.fn(),
    },
  };
  const fireHook = (event = { prompt: "hello world", systemPrompt: "" }) => hookFn!(event, {});
  const run = (args: string) => ragHandler!(args, ctx);
  return { pi, fireHook, run, messages };
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

// ─── /rag exclude subcommand ─────────────────────────────────────────────────

describe("/rag exclude subcommand", () => {
  beforeAll(() => {
    mkdirSync(RAG_DIR, { recursive: true });
  });

  afterAll(() => {
    rmSync(TEST_HOME, { recursive: true, force: true });
  });

  function readConfig(): Record<string, any> {
    return JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
  }

  it("adds a pattern to excludePatterns", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify(DEFAULT_CONFIG));
    const { pi, run } = makePi();
    defaultExport(pi as any);
    await run("exclude node_modules");
    expect(readConfig().excludePatterns).toEqual(["node_modules"]);
  });

  it("removes a pattern with leading dash", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify({ ...DEFAULT_CONFIG, excludePatterns: ["foo", "bar"] }));
    const { pi, run } = makePi();
    defaultExport(pi as any);
    await run("exclude -foo");
    expect(readConfig().excludePatterns).toEqual(["bar"]);
  });

  it("does not duplicate an already-present pattern", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify({ ...DEFAULT_CONFIG, excludePatterns: ["foo"] }));
    const { pi, run, messages } = makePi();
    defaultExport(pi as any);
    await run("exclude foo");
    expect(readConfig().excludePatterns).toEqual(["foo"]);
    expect(messages.some(m => /already excluded/i.test(m))).toBe(true);
  });

  it("reports error when removing a non-existent pattern", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify({ ...DEFAULT_CONFIG, excludePatterns: [] }));
    const { pi, run, messages } = makePi();
    defaultExport(pi as any);
    await run("exclude -ghost");
    expect(messages.some(m => /not found/i.test(m))).toBe(true);
  });

  it("lists current patterns when called with no argument", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify({ ...DEFAULT_CONFIG, excludePatterns: ["a", "b"] }));
    const { pi, run, messages } = makePi();
    defaultExport(pi as any);
    await run("exclude");
    const last = messages[messages.length - 1];
    expect(last).toContain("a");
    expect(last).toContain("b");
  });
});

// ─── /rag index auto-tracks paths ────────────────────────────────────────────

describe("/rag index auto-tracking", () => {
  beforeAll(() => {
    mkdirSync(RAG_DIR, { recursive: true });
  });

  afterAll(() => {
    rmSync(TEST_HOME, { recursive: true, force: true });
  });

  it("adds the indexed path to trackedPaths in config", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify(DEFAULT_CONFIG));
    const projDir = join(TEST_HOME, "proj-track");
    mkdirSync(projDir, { recursive: true });
    writeFileSync(join(projDir, "main.ts"), "export const a = 1;\n");

    const { pi, run } = makePi();
    defaultExport(pi as any);
    await run(`index ${projDir}`);

    const cfg = JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
    expect(cfg.trackedPaths).toContain(projDir);
  });

  it("does not duplicate when indexing the same path twice", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify(DEFAULT_CONFIG));
    const projDir = join(TEST_HOME, "proj-dedup");
    mkdirSync(projDir, { recursive: true });
    writeFileSync(join(projDir, "a.ts"), "export const a = 1;\n");

    const { pi, run } = makePi();
    defaultExport(pi as any);
    await run(`index ${projDir}`);
    await run(`index ${projDir}`);

    const cfg = JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
    expect(cfg.trackedPaths.filter((p: string) => p === projDir).length).toBe(1);
  });
});

// ─── /rag rebuild discovers new files ────────────────────────────────────────

describe("/rag rebuild new-file discovery", () => {
  beforeAll(() => {
    mkdirSync(RAG_DIR, { recursive: true });
  });

  afterAll(() => {
    rmSync(TEST_HOME, { recursive: true, force: true });
  });

  it("picks up files added after the initial index", async () => {
    const projDir = join(TEST_HOME, "proj-rebuild");
    mkdirSync(projDir, { recursive: true });
    writeFileSync(join(projDir, "first.ts"), "export const a = 1;\n");

    writeFileSync(CONFIG_FILE, JSON.stringify({ ...DEFAULT_CONFIG, trackedPaths: [projDir], excludePatterns: [] }));

    const { pi, run } = makePi();
    defaultExport(pi as any);
    await run(`index ${projDir}`);

    // Add a new file after indexing, then rebuild.
    const newFile = join(projDir, "second.ts");
    writeFileSync(newFile, "export const b = 2;\n");

    await run("rebuild");

    const idx = readIndex();
    expect(Object.keys(idx.files)).toContain(newFile);
  });

  it("drops files that match a newly-added exclude pattern", async () => {
    const projDir = join(TEST_HOME, "proj-rebuild-excl");
    mkdirSync(projDir, { recursive: true });
    writeFileSync(join(projDir, "keep.ts"), "export const k = 1;\n");
    writeFileSync(join(projDir, "drop.ts"), "export const d = 2;\n");

    writeFileSync(CONFIG_FILE, JSON.stringify({ ...DEFAULT_CONFIG, trackedPaths: [projDir], excludePatterns: [] }));

    const { pi, run } = makePi();
    defaultExport(pi as any);
    await run(`index ${projDir}`);
    await run("exclude drop.ts");
    await run("rebuild");

    const idx = readIndex();
    expect(Object.keys(idx.files)).toContain(join(projDir, "keep.ts"));
    expect(Object.keys(idx.files)).not.toContain(join(projDir, "drop.ts"));
  });

  it("prunes files that were deleted from disk", async () => {
    const projDir = join(TEST_HOME, "proj-rebuild-delete");
    mkdirSync(projDir, { recursive: true });
    const keep = join(projDir, "keep.ts");
    const gone = join(projDir, "gone.ts");
    writeFileSync(keep, "export const k = 1;\n");
    writeFileSync(gone, "export const g = 2;\n");

    writeFileSync(CONFIG_FILE, JSON.stringify(DEFAULT_CONFIG));

    const { pi, run } = makePi();
    defaultExport(pi as any);
    await run(`index ${projDir}`);

    rmSync(gone);
    await run("rebuild");

    const idx = readIndex();
    expect(Object.keys(idx.files)).toContain(keep);
    expect(Object.keys(idx.files)).not.toContain(gone);
    expect(idx.chunks.some((c: any) => c.file === gone)).toBe(false);
  });
});

// ─── /rag status output ──────────────────────────────────────────────────────

describe("/rag status output", () => {
  beforeAll(() => {
    mkdirSync(RAG_DIR, { recursive: true });
  });

  afterAll(() => {
    rmSync(TEST_HOME, { recursive: true, force: true });
  });

  it("lists tracked paths and exclude patterns", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify({
      ...DEFAULT_CONFIG,
      trackedPaths: ["/tmp/aaa", "/tmp/bbb"],
      excludePatterns: ["**/fixtures/**", "scratch/"],
    }));
    writeIndex({ chunks: [], files: {}, lastBuild: "" });

    const { pi, run, messages } = makePi();
    defaultExport(pi as any);
    await run("");

    const out = messages[messages.length - 1];
    expect(out).toMatch(/Tracked paths/i);
    expect(out).toContain("/tmp/aaa");
    expect(out).toContain("/tmp/bbb");
    expect(out).toMatch(/Exclude patterns/i);
    expect(out).toContain("**/fixtures/**");
    expect(out).toContain("scratch/");
  });

  it("shows '(none)' placeholders when both lists are empty", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify(DEFAULT_CONFIG));
    writeIndex({ chunks: [], files: {}, lastBuild: "" });

    const { pi, run, messages } = makePi();
    defaultExport(pi as any);
    await run("");

    const out = messages[messages.length - 1];
    expect(out).toMatch(/Tracked paths[\s\S]*\(none/);
    expect(out).toMatch(/Exclude patterns[\s\S]*\(none/);
  });
});

// ─── rag_index tool auto-tracks ──────────────────────────────────────────────

describe("rag_index tool", () => {
  beforeAll(() => {
    mkdirSync(RAG_DIR, { recursive: true });
  });

  afterAll(() => {
    rmSync(TEST_HOME, { recursive: true, force: true });
  });

  function captureTools() {
    const tools: any[] = [];
    let hookFn: any;
    const pi = {
      on: vi.fn((event: string, fn: any) => { if (event === "before_agent_start") hookFn = fn; }),
      registerCommand: vi.fn(),
      registerTool: vi.fn((def: any) => { tools.push(def); }),
      sendMessage: vi.fn(),
    };
    return { pi, tools, fireHook: () => hookFn };
  }

  it("auto-adds the indexed path to trackedPaths", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify(DEFAULT_CONFIG));
    const projDir = join(TEST_HOME, "tool-track");
    mkdirSync(projDir, { recursive: true });
    writeFileSync(join(projDir, "x.ts"), "export const x = 1;\n");

    const { pi, tools } = captureTools();
    defaultExport(pi as any);
    const ragIndex = tools.find(t => t.name === "rag_index");
    expect(ragIndex).toBeDefined();
    await ragIndex.execute("call-1", { path: projDir });

    const cfg = JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
    expect(cfg.trackedPaths).toContain(projDir);
  });

  it("respects excludePatterns when walking", async () => {
    const projDir = join(TEST_HOME, "tool-exclude");
    mkdirSync(projDir, { recursive: true });
    writeFileSync(join(projDir, "keep.ts"), "export const k = 1;\n");
    writeFileSync(join(projDir, "drop.ts"), "export const d = 2;\n");

    writeFileSync(CONFIG_FILE, JSON.stringify({ ...DEFAULT_CONFIG, excludePatterns: ["drop.ts"] }));
    writeIndex({ chunks: [], files: {}, lastBuild: "" });

    const { pi, tools } = captureTools();
    defaultExport(pi as any);
    const ragIndex = tools.find(t => t.name === "rag_index");
    await ragIndex.execute("call-1", { path: projDir });

    const idx = readIndex();
    expect(Object.keys(idx.files)).toContain(join(projDir, "keep.ts"));
    expect(Object.keys(idx.files)).not.toContain(join(projDir, "drop.ts"));
  });
});
