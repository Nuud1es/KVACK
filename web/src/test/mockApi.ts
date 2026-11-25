import { vi } from "vitest";

type Handler = (req: { url: URL; method: string; body: unknown; headers: Headers }) => Response | Promise<Response>;

/** Replace global fetch with a router keyed "METHOD /path". Unmatched calls fail the test. */
export function mockApi(routes: Record<string, Handler>) {
  const calls: { key: string; body: unknown; headers: Headers }[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), "http://localhost"), init);
    const url = new URL(request.url);
    const method = request.method.toUpperCase();
    const key = `${method} ${url.pathname}`;
    let body: unknown = null;
    if (method !== "GET") {
      const text = await request.clone().text();
      try {
        body = text ? JSON.parse(text) : null;
      } catch {
        body = text;
      }
    }
    calls.push({ key, body, headers: request.headers });
    const handler = routes[key];
    if (!handler) throw new Error(`Unexpected request: ${key}`);
    return handler({ url, method, body, headers: request.headers });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock, count: (key: string) => calls.filter((c) => c.key === key).length };
}

export const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });

/** A streaming SSE response, delivered in the given pieces (to exercise chunk boundaries). */
export function sse(pieces: string[], headers: Record<string, string> = {}) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      pieces.forEach((p) => controller.enqueue(encoder.encode(p)));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream", ...headers } });
}

export const event = (e: Record<string, unknown>) => `event: ${String(e.event)}\ndata: ${JSON.stringify(e)}\n\n`;
