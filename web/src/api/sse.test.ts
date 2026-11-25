import { describe, expect, it, vi } from "vitest";
import { ApiError, parseSse, streamAsk, type AskEvent } from "./sse";
import { event, json, mockApi, sse } from "../test/mockApi";

describe("parseSse", () => {
  it("takes the event name from the SSE line when the data has none (server errors)", () => {
    const { events } = parseSse('event: error\ndata: {"detail":"LLM down"}\n\n');
    expect(events).toEqual([{ event: "error", detail: "LLM down" }]);
  });

  it("returns complete events and keeps the unfinished tail", () => {
    const first = event({ event: "token", text: "Hel" });
    const { events, rest } = parseSse(first + 'event: token\ndata: {"event":"token","te');
    expect(events).toEqual([{ event: "token", text: "Hel" }]);
    expect(rest).toBe('event: token\ndata: {"event":"token","te');
  });

  it("ignores keep-alive comments and malformed events", () => {
    const { events } = parseSse(": ping\n\n" + "event: token\ndata: {not json}\n\n" + event({ event: "token", text: "ok" }));
    expect(events).toEqual([{ event: "token", text: "ok" }]);
  });
});

describe("streamAsk", () => {
  it("delivers events across arbitrary chunk boundaries and returns the request id", async () => {
    const whole = event({ event: "token", text: "75C" }) + event({ event: "done", answer: "75C", answered: true, cited: [], prompt_version: "v" });
    const cut = 23; // splits the first event in the middle
    mockApi({ "POST /ask": () => sse([whole.slice(0, cut), whole.slice(cut)], { "x-request-id": "abc123" }) });
    sessionStorage.setItem("qacopilot.apiKey", "k");
    const seen: AskEvent[] = [];
    const { requestId } = await streamAsk("q", (e) => seen.push(e));
    expect(requestId).toBe("abc123");
    expect(seen.map((e) => e.event)).toEqual(["token", "done"]);
  });

  it("sends the API key", async () => {
    const api = mockApi({ "POST /ask": () => sse([]) });
    sessionStorage.setItem("qacopilot.apiKey", "secret");
    await streamAsk("q", () => {});
    expect(api.calls[0]?.headers.get("x-api-key")).toBe("secret");
  });

  it("throws an ApiError with the server's message", async () => {
    mockApi({ "POST /ask": () => json({ detail: "Invalid or missing API key" }, 401) });
    await expect(streamAsk("q", () => {})).rejects.toMatchObject({ status: 401, message: "Invalid or missing API key" });
    await expect(streamAsk("q", () => {})).rejects.toBeInstanceOf(ApiError);
  });

  it("can be aborted", async () => {
    const controller = new AbortController();
    vi.stubGlobal("fetch", vi.fn((_: unknown, init?: RequestInit) => new Promise((_res, rej) => init?.signal?.addEventListener("abort", () => rej(new DOMException("aborted", "AbortError"))))));
    const pending = streamAsk("q", () => {}, controller.signal);
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  });
});
