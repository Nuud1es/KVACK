import createClient, { type Middleware } from "openapi-fetch";
import type { components, paths } from "./schema";

export type Schemas = components["schemas"];

const KEY = "qacopilot.apiKey";

// The key lives in sessionStorage: it is gone when the tab closes and never sent anywhere but the
// API. Wrapped in try/catch because storage can be blocked (private windows, strict settings).
export function getKey(): string {
  try {
    return sessionStorage.getItem(KEY) ?? "";
  } catch {
    return "";
  }
}
export function setKey(key: string): void {
  try {
    sessionStorage.setItem(KEY, key);
  } catch {
    /* the page still works for this load */
  }
}
export function clearKey(): void {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    /* nothing to clear */
  }
}

let onUnauthorized: () => void = () => {};
/** Called whenever the API answers 401, so the app can return to the sign-in screen. */
export function setUnauthorizedHandler(handler: () => void): void {
  onUnauthorized = handler;
}

export function notifyUnauthorized(): void {
  onUnauthorized();
}

const auth: Middleware = {
  onRequest({ request }) {
    const key = getKey();
    if (key) request.headers.set("x-api-key", key);
    return request;
  },
  onResponse({ response }) {
    if (response.status === 401) notifyUnauthorized();
    return response;
  },
};

export const NETWORK_MESSAGE =
  "Could not reach the server. It may be restarting: wait a moment, then reload the page and try again.";

/** True for what fetch throws when the connection fails (a TypeError), not for an abort. */
export function isNetworkError(error: unknown): boolean {
  return error instanceof TypeError;
}

// A dropped connection (API restarted, network down) makes fetch reject. Without this every page
// would need its own try/catch, and a missing one leaves a button stuck on "Working…" forever.
// Turn it into an ordinary 503 so each caller's existing error handling shows a message.
const network: Middleware = {
  onError({ error }) {
    // aborts and programming errors still propagate
    if (!isNetworkError(error)) return error instanceof Error ? error : new Error(String(error));
    return new Response(JSON.stringify({ detail: NETWORK_MESSAGE }), {
      status: 503,
      headers: { "content-type": "application/json" },
    });
  },
};

export const api = createClient<paths>({
  baseUrl: globalThis.location?.origin ?? "",
  // Looked up per call (not captured once) so tests can replace fetch.
  fetch: (request) => globalThis.fetch(request),
});
api.use(auth);
api.use(network);

/** A readable message from a FastAPI error body (`{detail: string | [{msg}]}`) or anything else. */
export function errorMessage(error: unknown): string {
  if (isNetworkError(error)) return NETWORK_MESSAGE;
  if (error instanceof Error) return error.message;
  if (error && typeof error === "object" && "detail" in error) {
    const detail = (error as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : "")).join("; ");
    }
  }
  return "Something went wrong";
}

/** Download every row of a list or count answer as a CSV file (the answer lists only 25). */
export async function downloadSpecs(query: Record<string, unknown>): Promise<void> {
  let response: Response;
  try {
    response = await fetch("/export/specs", {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": getKey() },
      body: JSON.stringify({ query }),
    });
  } catch (error) {
    throw new Error(isNetworkError(error) ? NETWORK_MESSAGE : "The download failed.");
  }
  if (response.status === 401) {
    notifyUnauthorized();
    throw new Error("Please sign in again.");
  }
  if (!response.ok) {
    const detail = await response.json().then((d) => d?.detail, () => null);
    throw new Error(typeof detail === "string" ? detail : "The download failed.");
  }
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = /filename="([^"]+)"/.exec(response.headers.get("content-disposition") ?? "")?.[1] ?? "specs.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
