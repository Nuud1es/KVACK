import { useCallback, useEffect, useState, type FormEvent } from "react";
import { api, errorMessage, type Schemas } from "../api/client";

type Summary = Schemas["RunSummary"];
type Run = Schemas["RunView"];
type Audit = Schemas["AuditEvent"];
type Draft = {
  message: { to: string[]; cc: string[]; subject: string; body: string };
  certificate_count: number;
};

export function Approvals({ me }: { me: string }) {
  const [filter, setFilter] = useState<"awaiting_approval" | "all">("awaiting_approval");
  const [runs, setRuns] = useState<Summary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const { data, error: e } = await api.GET("/agent/runs", {
      params: { query: filter === "all" ? {} : { status: filter } },
    });
    if (data) setRuns(data);
    else setError(errorMessage(e));
  }, [filter]);

  useEffect(() => {
    void load();
  }, [load]);

  async function start(e: FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setStarting(true);
    setError(null);
    const { data, error: err } = await api.POST("/agent/runs", { body: { query: query.trim() } });
    setStarting(false);
    if (data) {
      setQuery("");
      setSelected(data.run_id);
      await load();
    } else {
      setError(errorMessage(err));
    }
  }

  return (
    <section>
      <h1>Approvals</h1>
      <form className="ask" onSubmit={(e) => void start(e)}>
        <input value={query} onChange={(e) => setQuery(e.target.value)} disabled={starting} aria-label="Request for the agent"
          placeholder="e.g. Find all certificates expiring in the next 30 days and draft reminders" />
        <button type="submit" disabled={starting || !query.trim()}>{starting ? "Working…" : "Start"}</button>
      </form>
      {starting && <p className="muted">The agent is working. This can take up to a minute.</p>}
      {error && <p role="alert" className="error">{error}</p>}

      <div className="split">
        <div>
          <div className="row between">
            <h2>Queue</h2>
            <span className="row">
              <select value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)} aria-label="Show">
                <option value="awaiting_approval">Waiting for approval</option>
                <option value="all">All runs</option>
              </select>
              <button className="secondary" onClick={() => void load()}>Refresh</button>
            </span>
          </div>
          {runs.length === 0 ? <p className="muted">Nothing here.</p> : (
            <ul className="plain runs">
              {runs.map((r) => (
                <li key={r.run_id}>
                  <button className={`run ${r.run_id === selected ? "active" : ""}`} onClick={() => setSelected(r.run_id)}>
                    <span className={`badge ${tone(r.status)}`}>{r.status.replace(/_/g, " ")}</span>
                    <span className="run-query">{r.query}</span>
                    <span className="muted">{r.created_by} · {new Date(r.created_at).toLocaleString()}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>{selected ? <RunPanel key={selected} runId={selected} me={me} onChanged={load} /> : <p className="muted">Select a run to review it.</p>}</div>
      </div>
    </section>
  );
}

function RunPanel({ runId, me, onChanged }: { runId: string; me: string; onChanged: () => Promise<void> }) {
  const [run, setRun] = useState<Run | null>(null);
  const [audit, setAudit] = useState<Audit[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    const [r, a] = await Promise.all([
      api.GET("/agent/runs/{run_id}", { params: { path: { run_id: runId } } }),
      api.GET("/agent/runs/{run_id}/audit", { params: { path: { run_id: runId } } }),
    ]);
    if (r.data) setRun(r.data);
    else setError(errorMessage(r.error));
    if (a.data) setAudit(a.data);
  }, [runId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function act(kind: "approve" | "reject" | "retry-delivery") {
    setBusy(true);
    setError(null);
    const path = { params: { path: { run_id: runId } } };
    const result =
      kind === "approve" ? await api.POST("/agent/runs/{run_id}/approve", path)
      : kind === "retry-delivery" ? await api.POST("/agent/runs/{run_id}/retry-delivery", path)
      : await api.POST("/agent/runs/{run_id}/reject", { ...path, body: { reason: reason.trim() || null } });
    setBusy(false);
    setConfirming(false);
    setRejecting(false);
    if (result.error) {
      setError(result.response?.status === 409 ? "Someone else has already decided this run." : errorMessage(result.error));
    }
    await refresh();
    await onChanged();
  }

  if (!run) return error ? <p role="alert" className="error">{error}</p> : <p className="muted">Loading…</p>;
  const drafts = run.drafts as unknown as Draft[];
  const waiting = run.status === "awaiting_approval";

  return (
    <div className="card">
      <div className="row between">
        <strong>{run.query}</strong>
        <span className={`badge ${tone(run.status)}`}>{run.status.replace(/_/g, " ")}</span>
      </div>
      <p className="muted">Started by {run.created_by}{run.decided_by ? ` · decided by ${run.decided_by}` : ""}</p>
      {run.message && <p>{run.message}</p>}
      {run.route === "question" && run.answer && <p className="answer">{run.answer}</p>}

      {drafts.map((d, i) => (
        <div key={i} className="draft">
          <p><strong>To:</strong> {d.message.to.join(", ")}</p>
          <p><strong>Subject:</strong> {d.message.subject}</p>
          <pre className="page-text">{d.message.body}</pre>
        </div>
      ))}

      {waiting && (
        <div className="actions">
          {!confirming && !rejecting && (
            <>
              <button onClick={() => setConfirming(true)}>Approve…</button>
              <button className="secondary" onClick={() => setRejecting(true)}>Reject…</button>
            </>
          )}
          {confirming && (
            <div role="group" aria-label="Confirm approval">
              <p>Send {drafts.length} email{drafts.length === 1 ? "" : "s"} to {drafts.flatMap((d) => d.message.to).join(", ")} as <strong>{me}</strong>?</p>
              <button disabled={busy} onClick={() => void act("approve")}>Confirm and send</button>
              <button className="secondary" disabled={busy} onClick={() => setConfirming(false)}>Cancel</button>
            </div>
          )}
          {rejecting && (
            <div role="group" aria-label="Confirm rejection">
              <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="Reason (optional)" aria-label="Reason" />
              <button disabled={busy} onClick={() => void act("reject")}>Reject, send nothing</button>
              <button className="secondary" disabled={busy} onClick={() => setRejecting(false)}>Cancel</button>
            </div>
          )}
        </div>
      )}
      {run.status === "delivery_failed" && (
        <div className="actions">
          <button disabled={busy} onClick={() => void act("retry-delivery")}>Retry failed emails</button>
        </div>
      )}
      {run.deliveries.length > 0 && (
        <ul className="plain">
          {run.deliveries.map((d, i) => {
            const x = d as { status?: string; outcome?: string };
            return <li key={i} className={x.status === "delivered" ? "muted" : "error"}>Email {i + 1}: {x.status}{x.outcome ? ` (${x.outcome})` : ""}</li>;
          })}
        </ul>
      )}
      {error && <p role="alert" className="error">{error}</p>}

      <h3>History</h3>
      <table>
        <tbody>
          {audit.map((a, i) => (
            <tr key={i}><td className="muted">{new Date(a.ts).toLocaleTimeString()}</td><td>{a.actor}</td><td>{a.event.replace(/_/g, " ")}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function tone(status: string): string {
  if (status === "completed") return "good";
  if (status === "awaiting_approval") return "warn";
  if (status === "rejected" || status === "no_action") return "";
  return "bad"; // failed, delivery_failed, blocked
}
