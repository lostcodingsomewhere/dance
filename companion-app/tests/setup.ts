import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement WebSocket — stub it so useAbletonState doesn't crash.
class FakeWebSocket {
  public onmessage: ((ev: MessageEvent) => void) | null = null;
  public onclose: (() => void) | null = null;
  public onerror: (() => void) | null = null;
  constructor(public url: string) {}
  close() {
    /* noop */
  }
}

// @ts-expect-error - assigning stub onto globalThis
globalThis.WebSocket = FakeWebSocket;

// Node 22's experimental built-in localStorage (enabled when --localstorage-file
// is in NODE_OPTIONS, even without a valid path) can shadow jsdom's with a
// broken impl — `window.localStorage.getItem is not a function`. Force a
// known-good in-memory Storage so components that touch it (e.g.
// StackMigrationPrompt) render in tests regardless of the host Node config.
class FakeStorage {
  private store = new Map<string, string>();
  get length() {
    return this.store.size;
  }
  clear() {
    this.store.clear();
  }
  getItem(key: string): string | null {
    return this.store.has(key) ? (this.store.get(key) as string) : null;
  }
  setItem(key: string, value: string) {
    this.store.set(key, String(value));
  }
  removeItem(key: string) {
    this.store.delete(key);
  }
  key(index: number): string | null {
    return Array.from(this.store.keys())[index] ?? null;
  }
}
const fakeStorage = new FakeStorage();
for (const target of [globalThis, typeof window !== "undefined" ? window : undefined]) {
  if (!target) continue;
  try {
    Object.defineProperty(target, "localStorage", {
      value: fakeStorage,
      configurable: true,
      writable: true,
    });
  } catch {
    /* property non-configurable on this host — best effort */
  }
}
