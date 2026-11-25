import { errorMessage, getKey, notifyUnauthorized } from "./client";

export type Citation = {
  n: number;
  doc_id: string;
  filename: string;
  page: number;
  section: string | null;
  clause: string | null;
  score: number;
  snippet: string;
};

/** What a list or count answer carries so every row can be downloaded (see POST /export/specs). */
export type ExportRef = { query: Record<string, unknown>; count: number };

/** The events POST /ask streams (mirrors qacopilot.rag.answer; /ask is not in the OpenAPI types). */
export type AskEvent =
  | { event: "sources"; citations: Citation[] }
  | { event: "token"; text: string }
  | { event: "no_answer"; reason: "no_relevant_documents" | "not_in_documents" | "uncited" }
  | { event: "done"; answer: string; answered: boolean; cited: number[]; prompt_version: string; via?: "documents" | "certificates" | "coa_check" | "packaging" | "fact_spec"; export?: ExportRef | null }
  | { event: "error"; detail: string };

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

/**
 * Split buffered text into complete SSE events plus the unfinished remainder.
 * Comment lines (": ping" keep-alives) are ignored.
 */
export function parseSse(buffer: string): { events: AskEvent[]; rest: string } {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: AskEvent[] = [];
  for (const block of blocks) {
    const lines = block.split("\n");
    const name = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
    const data = lines
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (!data) continue;
    try {
      const parsed = JSON.parse(data) as Record<string, unknown>;
      // Dispatch is on the "event" field; fall back to the SSE event name if the data lacks it.
      if (typeof parsed.event !== "string" && name) parsed.event = name;
      events.push(parsed as AskEvent);
    } catch {
      /* a malformed event is skipped rather than breaking the stream */
    }
  }
  return { events, rest };
}

export async function streamAsk(
  question: string,
  onEvent: (event: AskEvent) => void,
  signal?: AbortSignal,
  context?: Record<string, unknown>,
): Promise<{ requestId: string | null }> {
  const response = await fetch("/ask", {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": getKey() },
    body: JSON.stringify(context ? { question, context } : { question }),
    signal,
  });
  if (response.status === 401) notifyUnauthorized();
  if (!response.ok || !response.body) {
    let detail: unknown = null;
    try {
      detail = await response.json();
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, detail ? errorMessage(detail) : `HTTP ${response.status}`);
  }
  const requestId = response.headers.get("x-request-id");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSse(buffer);
    buffer = parsed.rest;
    parsed.events.forEach(onEvent);
  }
  return { requestId };
}
