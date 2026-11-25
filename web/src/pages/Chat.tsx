import { Fragment, useRef, useState, type FormEvent } from "react";
import { api, downloadSpecs, errorMessage } from "../api/client";
import { streamAsk, type AskEvent, type Citation, type ExportRef } from "../api/sse";
import { PageViewer } from "../components/PageViewer";

type Message = {
  id: number;
  question: string;
  answer: string;
  citations: Citation[];
  cited: number[];
  status: "streaming" | "done" | "no_answer" | "error";
  note?: string;
  requestId: string | null;
  rating?: "up" | "down";
  via?: "documents" | "certificates" | "coa_check" | "packaging" | "fact_spec";
  exportRef?: ExportRef;
  exporting?: boolean;
  exportError?: string;
};

type Open = { docId: string; page: number; snippet: string } | null;

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [open, setOpen] = useState<Open>(null);
  const abort = useRef<AbortController | null>(null);
  const nextId = useRef(1);
  const busy = messages.some((m) => m.status === "streaming");

  const update = (id: number, change: (m: Message) => Message) =>
    setMessages((all) => all.map((m) => (m.id === id ? change(m) : m)));

  async function download(id: number, ref: ExportRef) {
    update(id, (m) => ({ ...m, exporting: true, exportError: undefined }));
    try {
      await downloadSpecs(ref.query);
      update(id, (m) => ({ ...m, exporting: false }));
    } catch (error) {
      update(id, (m) => ({ ...m, exporting: false, exportError: error instanceof Error ? error.message : "The download failed." }));
    }
  }

  function apply(id: number, e: AskEvent) {
    switch (e.event) {
      case "sources":
        update(id, (m) => ({ ...m, citations: e.citations }));
        break;
      case "token":
        update(id, (m) => ({ ...m, answer: m.answer + e.text }));
        break;
      case "no_answer":
        update(id, (m) => ({
          ...m,
          status: "no_answer",
          citations: [],
          note:
            e.reason === "no_relevant_documents"
              ? "Nothing relevant was found in the indexed documents."
              : e.reason === "uncited"
                ? "The model replied without citing a source, so that reply is not shown. Try rephrasing."
                : "The documents found do not answer this.",
        }));
        break;
      case "done":
        update(id, (m) => (m.status === "no_answer" ? m : { ...m, status: "done", cited: e.cited, via: e.via, exportRef: e.export ?? undefined }));
        break;
      case "error":
        update(id, (m) => ({ ...m, status: "error", note: e.detail }));
        break;
    }
  }

  async function send(e: FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question || busy) return;
    // A follow-up ("give me all from Tesco") builds on the last answer if it was a list or count.
    const last = messages[messages.length - 1];
    const context = last?.status === "done" ? last.exportRef?.query : undefined;
    const id = nextId.current++;
    setInput("");
    setMessages((all) => [
      ...all,
      { id, question, answer: "", citations: [], cited: [], status: "streaming", requestId: null },
    ]);
    const controller = new AbortController();
    abort.current = controller;
    try {
      const { requestId } = await streamAsk(question, (ev) => apply(id, ev), controller.signal, context);
      update(id, (m) => ({ ...m, requestId, status: m.status === "streaming" ? "done" : m.status }));
    } catch (err) {
      const stopped = controller.signal.aborted;
      update(id, (m) => ({
        ...m,
        status: stopped && m.answer ? "done" : "error",
        note: stopped ? "Stopped." : errorMessage(err),
      }));
    }
  }

  async function rate(m: Message, rating: "up" | "down") {
    if (!m.requestId || m.rating) return;
    update(m.id, (x) => ({ ...x, rating }));
    const { error } = await api.POST("/feedback", {
      body: { request_id: m.requestId, kind: "ask", rating },
    });
    if (error) update(m.id, (x) => ({ ...x, rating: undefined, note: errorMessage(error) }));
  }

  return (
    <section>
      <h1>Ask the documents</h1>
      {messages.length === 0 ? (
        <p className="muted">
          Ask about specifications, SOPs, certificates or customer codes. Every answer cites its source page. After a
          list or count you can refine it: "list whole duck specs", then "give me all from Tesco".
        </p>
      ) : (
        <p>
          <button type="button" className="link" disabled={busy} onClick={() => setMessages([])}>
            New topic
          </button>
        </p>
      )}
      {messages.map((m) => (
        <article key={m.id} className="card message" aria-label={`Question: ${m.question}`}>
          <p className="question">{m.question}</p>
          {m.status === "streaming" && !m.answer && <p className="muted">Thinking…</p>}
          {m.answer && m.status !== "no_answer" && <p className="answer">{withCitations(m, setOpen)}</p>}
          {m.status === "done" && m.via === "certificates" && (
            <p className="muted">From the certificates table: the latest certificate for each instrument.</p>
          )}
          {m.status === "done" && m.via === "coa_check" && (
            <p className="muted">From the spec check: limits compared in code against the matching specification.</p>
          )}
          {m.status === "done" && m.via === "fact_spec" && (
            <p className="muted">From the Fact Spec table: read from the spec headers in code.</p>
          )}
          {m.status === "done" && m.via === "packaging" && (
            <p className="muted">From the packaging table: counted in code across the current Fact Specs.</p>
          )}
          {m.status === "done" && m.exportRef && (
            <p>
              <button type="button" disabled={m.exporting} onClick={() => void download(m.id, m.exportRef as ExportRef)}>
                {m.exporting ? "Preparing…" : `Download all ${m.exportRef.count} specs (CSV)`}
              </button>
              {m.exportError && <span role="alert" className="error"> {m.exportError}</span>}
            </p>
          )}
          {m.note && <p className={m.status === "error" ? "error" : "muted"} role={m.status === "error" ? "alert" : undefined}>{m.note}</p>}
          {m.citations.length > 0 && (
            <details open={m.status === "done"}>
              <summary>{m.citations.length} sources</summary>
              <ul className="sources">
                {m.citations.map((c) => (
                  <li key={c.n} className={m.cited.includes(c.n) ? "cited" : ""}>
                    <button className="link" onClick={() => setOpen({ docId: c.doc_id, page: c.page, snippet: c.snippet })}>
                      [{c.n}] {c.filename} · page {c.page}
                    </button>
                    {(c.clause || c.section) && <span className="muted"> · {c.clause ? `clause ${c.clause}` : c.section}</span>}
                    <p className="snippet">{c.snippet}</p>
                  </li>
                ))}
              </ul>
            </details>
          )}
          {m.status === "streaming" && (
            <button className="secondary" onClick={() => abort.current?.abort()}>Stop</button>
          )}
          {m.status === "done" && m.requestId && (
            <div className="row" aria-label="Rate this answer">
              <button className={m.rating === "up" ? "chosen" : "secondary"} disabled={Boolean(m.rating)} aria-label="Good answer" onClick={() => void rate(m, "up")}>👍</button>
              <button className={m.rating === "down" ? "chosen" : "secondary"} disabled={Boolean(m.rating)} aria-label="Bad answer" onClick={() => void rate(m, "down")}>👎</button>
              {m.rating && <span className="muted">Thanks, noted.</span>}
            </div>
          )}
        </article>
      ))}
      <form className="ask" onSubmit={send}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="e.g. What core temperature must cooked duck reach before release?" aria-label="Your question" disabled={busy} />
        <button type="submit" disabled={busy || !input.trim()}>Ask</button>
      </form>
      {open && <PageViewer {...open} onClose={() => setOpen(null)} />}
    </section>
  );
}

/** Turn [n] markers in the answer into links to the matching source. */
function withCitations(m: Message, openSource: (o: Open) => void) {
  return m.answer.split(/(\[\d+\])/g).map((part, i) => {
    const n = /^\[(\d+)\]$/.exec(part)?.[1];
    const citation = n ? m.citations.find((c) => c.n === Number(n)) : undefined;
    if (!citation) return <Fragment key={i}>{part}</Fragment>;
    return (
      <button key={i} className="cite" aria-label={`Source ${citation.n}: ${citation.filename} page ${citation.page}`}
        onClick={() => openSource({ docId: citation.doc_id, page: citation.page, snippet: citation.snippet })}>
        {part}
      </button>
    );
  });
}
