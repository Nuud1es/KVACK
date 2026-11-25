import { afterEach, describe, expect, it, vi } from "vitest";
import { api, errorMessage, NETWORK_MESSAGE } from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("network failures", () => {
  it("become an ordinary 503 error instead of an unhandled rejection", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    const { data, error, response } = await api.GET("/agent/runs", { params: { query: {} } });
    expect(data).toBeUndefined();
    expect(response.status).toBe(503);
    expect(errorMessage(error)).toBe(NETWORK_MESSAGE);
  });

  it("do not hide other errors (an abort or a bug still throws)", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("boom"); }));
    await expect(api.GET("/agent/runs", { params: { query: {} } })).rejects.toThrow("boom");
  });

  it("get a readable message when thrown directly (the streaming Ask path)", () => {
    expect(errorMessage(new TypeError("Failed to fetch"))).toBe(NETWORK_MESSAGE);
    expect(errorMessage(new Error("plain"))).toBe("plain");
  });
});
