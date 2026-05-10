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
