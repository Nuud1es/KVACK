import { useCallback, useEffect, useRef, useState, type ChangeEvent } from "react";
import { api, errorMessage, type Schemas } from "../api/client";
import { ReviewQueue } from "../components/ReviewQueue";

type Doc = Schemas["DocumentSummary"];
type Due = Schemas["CertificateStatus"];
type Extraction = Schemas["ExtractionOut"];
type Check = Schemas["CoACheck"];
const EXTRACTABLE = new Set(["calibration_cert", "coa", "supplier_spec"]);
const WINDOWS = [30, 60, 90, 180];
const TYPES: Schemas["DocType"][] = [
  "calibration_cert",
  "coa",
  "supplier_spec",
  "fact_spec",
  "audit_cert",
  "sop",
  "haccp",
  "customer_cop",
  "other",
];

type Upload = { id: number; name: string; state: "uploading" | "done" | "error"; message: string };

const ACCEPTED = /\.(pdf|jpe?g|png)$/i;
const FOLDER_ATTRS = { webkitdirectory: "", directory: "" } as Record<string, string>;
const MAX_WAIT_MS = 30 * 60 * 1000; // a big bundle can take many minutes to read
let nextUploadId = 1;

export function Documents({ pollMs = 3000 }: { pollMs?: number }) {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState<Set<string>>(new Set()); // POST in flight (single certificates)
  const [shown, setShown] = useState<Extraction | null>(null);
  const [check, setCheck] = useState<{ name: string; result: Check } | null>(null);
  const [days, setDays] = useState(30);
  const [due, setDue] = useState<Due[] | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const [readAfter, setReadAfter] = useState(true);
  const [batch, setBatch] = useState<string | null>(null); // what a folder run is doing now
  const stopRequested = useRef(false);

  const loadDocs = useCallback(async () => {
    const { data, error: e } = await api.GET("/documents");
    if (data) setDocs(data);
    else setError(errorMessage(e));
  }, []);

  useEffect(() => {
    void loadDocs();
  }, [loadDocs]);

  // While a bundle is being read in the background, keep the list (and any open results) fresh.
  const running = docs.some((d) => d.extraction_status === "running");
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => {
      void loadDocs();
      setShown((cur) => {
        if (cur?.status === "running") void refreshShown(cur.doc_id);
        return cur;
      });
    }, pollMs);
    return () => clearInterval(timer);
  }, [running, loadDocs, pollMs]);

  useEffect(() => {
    let live = true;
    void api.GET("/certificates", { params: { query: { expiring_within: `${days}d` } } }).then(({ data }) => {
      if (live) setDue(data ?? []);
    });
    return () => {
      live = false;
    };
  }, [days, docs]);

  function upload(e: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (input.current) input.current.value = "";
    void uploadFiles(files, false);
  }

  function pickFolder(e: ChangeEvent<HTMLInputElement>) {
    const all = Array.from(e.target.files ?? []);
    if (folderInput.current) folderInput.current.value = "";
    const files = all.filter((f) => ACCEPTED.test(f.name));
    const skipped = all.length - files.length;
    if (files.length === 0) {
      setError(all.length ? "That folder has no PDFs or photos in it." : "No folder was chosen.");
      return;
    }
    setError(null);
    void uploadFiles(files, readAfter, skipped);
  }

  async function uploadFiles(files: File[], readFields: boolean, skipped = 0) {
    stopRequested.current = false;
    const added: { docId: string; docType: string }[] = [];
    let failed = 0;
    for (const [i, file] of files.entries()) {
      if (stopRequested.current) break;
      const shownName = file.webkitRelativePath || file.name;
      const id = nextUploadId++;
      if (files.length > 1) setBatch(`Uploading ${i + 1} of ${files.length}…`);
      setUploads((u) => [...u, { id, name: shownName, state: "uploading", message: "Reading and indexing…" }]);
      const { data, error: err } = await api.POST("/ingest", {
        body: { file: "" },
        bodySerializer: () => {
          const form = new FormData();
          form.append("file", file);
          return form;
        },
      });
      if (data) added.push({ docId: data.doc_id, docType: data.doc_type });
      else failed += 1;
      const result: Pick<Upload, "state" | "message"> = data
        ? {
            state: "done",
            message: `${label(data.doc_type)}${data.classified_by === "llm" ? " (chosen by the AI model: check it)" : ""}${data.ocr_pages ? `, ${data.ocr_pages} page(s) read by OCR` : ""}, ${data.chunks} chunks`,
          }
        : { state: "error", message: errorMessage(err) };
      setUploads((u) => u.map((x) => (x.id === id ? { ...x, ...result } : x)));
    }
    await loadDocs();
    const reading = readFields ? added.filter((a) => EXTRACTABLE.has(a.docType)) : [];
    for (const [i, a] of reading.entries()) {
      if (stopRequested.current) break;
      setBatch(`Reading the fields of ${i + 1} of ${reading.length}…`);
      await extractAndWait(a.docId);
      await loadDocs();
    }
    if (files.length > 1 || skipped) {
      const parts = [`${added.length} uploaded`];
      if (failed) parts.push(`${failed} failed`);
      if (skipped) parts.push(`${skipped} skipped (not a PDF or photo)`);
      if (reading.length) parts.push(`fields read for ${reading.length}`);
      if (stopRequested.current) parts.push("stopped early");
      setBatch(`Done: ${parts.join(", ")}.`);
    } else setBatch(null);
  }

  /** Start reading a document's fields and wait for it: a bundle runs in the background. */
  async function extractAndWait(docId: string) {
    const { data, error: err } = await api.POST("/documents/{doc_id}/extract", { params: { path: { doc_id: docId } } });
    if (!data) {
      setError(`Could not read the fields: ${errorMessage(err)}`);
      return;
    }
    let status = data.status;
    const until = Date.now() + MAX_WAIT_MS;
    while (status === "running" && Date.now() < until && !stopRequested.current) {
      await new Promise((r) => setTimeout(r, pollMs));
      const next = await api.GET("/documents/{doc_id}/extraction", { params: { path: { doc_id: docId } } });
      if (next.data) status = next.data.status;
    }
  }

  async function retype(doc: Doc, docType: Schemas["DocType"]) {
    if (docType === doc.doc_type) return;
    setError(null);
    const { error: err } = await api.POST("/documents/{doc_id}/reclassify", {
      params: { path: { doc_id: doc.doc_id } },
      body: { doc_type: docType },
    });
    if (err) setError(`${doc.filename}: ${errorMessage(err)}`);
    await loadDocs();
  }

  async function refreshShown(docId: string) {
    const { data } = await api.GET("/documents/{doc_id}/extraction", { params: { path: { doc_id: docId } } });
    if (data) setShown(data);
  }

  async function extract(doc: Doc) {
    setStarting((s) => new Set(s).add(doc.doc_id));
    setError(null);
    // One certificate: answered straight away. A bundle: 202 now, then it runs in the background.
    const { data, error: err } = await api.POST("/documents/{doc_id}/extract", { params: { path: { doc_id: doc.doc_id } } });
    setStarting((s) => {
      const next = new Set(s);
      next.delete(doc.doc_id);
      return next;
    });
    if (data) setShown(data);
    else setError(`${doc.filename}: ${errorMessage(err)}`);
    await loadDocs();
  }

  async function view(doc: Doc) {
    const { data, error: err } = await api.GET("/documents/{doc_id}/extraction", { params: { path: { doc_id: doc.doc_id } } });
    if (data) setShown(data);
    else setError(errorMessage(err));
  }

  async function checkSpec(doc: Doc) {
    setError(null);
    const { data, error: err } = await api.GET("/compliance/coa-check", { params: { query: { coa: doc.doc_id } } });
    if (data) setCheck({ name: doc.filename, result: data });
    else setError(`${doc.filename}: ${errorMessage(err)}`);
  }

  return (
    <section>
      <h1>Documents</h1>
      <div className="card">
        <label>
          Add PDFs or photos (scans and photos are read with OCR)
          <input ref={input} type="file" accept="application/pdf,image/jpeg,image/png" multiple onChange={(e) => void upload(e)} />
        </label>
        <div className="row">
          <button type="button" onClick={() => folderInput.current?.click()} disabled={uploads.some((u) => u.state === "uploading") || (batch !== null && !batch.startsWith("Done"))}>
            Add a folder…
          </button>
          <input
            ref={folderInput}
            type="file"
            multiple
            hidden
            aria-label="Choose a folder"
            onChange={pickFolder}
            {...FOLDER_ATTRS}
          />
          <label>
            <input type="checkbox" checked={readAfter} onChange={(e) => setReadAfter(e.target.checked)} /> Read the fields after uploading
            (calibration certificates, reports and specs)
          </label>
        </div>
        {batch && (
          <p className="muted" role="status">
            {batch}{" "}
            {!batch.startsWith("Done") && (
              <button type="button" className="link" onClick={() => { stopRequested.current = true; }}>Stop</button>
            )}
          </p>
        )}
        <ul className="plain">
          {uploads.map((u) => (
            <li key={u.id} className={u.state === "error" ? "error" : "muted"}>
              {u.state === "done" ? "✓" : u.state === "error" ? "✗" : "…"} {u.name}: {u.message}
            </li>
          ))}
        </ul>
      </div>
      {error && <p role="alert" className="error">{error}</p>}

      <ReviewQueue onChanged={() => void loadDocs()} />

      <h2>Calibration certificates due</h2>
      <div className="card">
        <label className="row">
          Within
          <select value={days} onChange={(e) => setDays(Number(e.target.value))} aria-label="Window in days">
            {WINDOWS.map((d) => <option key={d} value={d}>{d} days</option>)}
          </select>
        </label>
        {due === null ? <p className="muted">Loading…</p> : due.length === 0 ? (
          <p className="muted">Nothing is overdue or due in the next {days} days.</p>
        ) : (
          <table>
            <thead><tr><th>Serial</th><th>Instrument</th><th>Due</th><th>Status</th></tr></thead>
            <tbody>
              {due.map((c) => (
                <tr key={`${c.doc_id}:${c.start_page}`}>
                  <td>{c.instrument_serial}</td>
                  <td>{c.instrument_description ?? "–"}</td>
                  <td>{c.next_due}</td>
                  <td><span className={`badge ${c.status === "overdue" ? "bad" : "warn"}`}>
                    {c.status === "overdue" ? `${-c.days_left} days overdue` : `in ${c.days_left} days`}
                  </span>
                  {c.unverified && <span className="badge bad" title={`${c.review_reason ?? "Not confirmed"}. Confirm it under Needs review.`}>unverified</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <h2>Indexed documents</h2>
      {docs.length === 0 ? <p className="muted">No documents yet.</p> : (
        <table>
          <thead><tr><th>File</th><th>Type</th><th>Pages</th><th>Added</th><th>Fields</th></tr></thead>
          <tbody>
            {docs.map((d) => (
              <tr key={d.doc_id}>
                <td>{d.filename}{d.ocr_pages > 0 && <span className="badge warn" title="Some pages were read by OCR">OCR</span>}
                  {d.poor_ocr_pages > 0 && (
                    <span className="badge bad" title={`${d.poor_ocr_pages} page(s) were scanned poorly (low OCR confidence). Figures and dates may be misread: check them against the original, or rescan.`}>
                      poor scan
                    </span>
                  )}</td>
                <td>
                  <select
                    value={d.doc_type}
                    aria-label={`Type of ${d.filename}`}
                    onChange={(e) => void retype(d, e.target.value as Schemas["DocType"])}
                  >
                    {TYPES.map((t) => <option key={t} value={t}>{label(t)}</option>)}
                  </select>
                  {d.classified_by === "llm" && <span className="badge warn" title="The keyword rules were unsure, so the AI model chose this type. Check it.">AI</span>}
                  {d.classified_by === "user" && <span className="badge" title="Set by a person">set</span>}
                  {d.alternative_type && (
                    <span className="badge warn" title={`The AI model thought this might be ${label(d.alternative_type)}. Check the type.`}>
                      check
                    </span>
                  )}
                </td>
                <td>{d.pages}</td>
                <td>{new Date(d.ingested_at).toLocaleString()}</td>
                <td>
                  {!EXTRACTABLE.has(d.doc_type) ? <span className="muted">n/a</span> : starting.has(d.doc_id) ? (
                    <span className="muted">Extracting… this can take a minute or two</span>
                  ) : d.extraction_status === "running" ? (
                    <span className="muted" role="status">
                      Reading certificates: {d.extraction_done} of {d.extraction_total}…
                    </span>
                  ) : d.extracted ? (
                    <><button className="link" onClick={() => void view(d)}>View</button>{" · "}<button className="link" onClick={() => void extract(d)}>Re-run</button>
                      {d.doc_type === "coa" && <>{" · "}<button className="link" onClick={() => void checkSpec(d)}>Check against spec</button></>}</>
                  ) : (
                    <button className="secondary" onClick={() => void extract(d)}>Extract</button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {check && <CheckCard name={check.name} check={check.result} onClose={() => setCheck(null)} />}
      {shown && <ExtractionCard extraction={shown} onClose={() => setShown(null)} />}
    </section>
  );
}

const VERDICT = { pass: "good", fail: "bad", review: "warn" } as const;

function CheckCard({ name, check: c, onClose }: { name: string; check: Check; onClose: () => void }) {
  return (
    <div className="card" aria-label="Spec check">
      <div className="row between">
        <strong>
          {name} against {c.spec_filename ?? "the chosen spec"}
          {c.spec_revision ? ` (revision ${c.spec_revision})` : ""}
          <span className={`badge ${VERDICT[c.overall]}`}>{c.overall.toUpperCase()}</span>
        </strong>
        <button className="link" onClick={onClose}>Close</button>
      </div>
      {c.pairing === "matched" && <p className="muted">The spec was matched from the report's product name.</p>}
      {c.warnings.map((w) => (
        <p key={w} role="alert" className="badge warn">{w}</p>
      ))}
      <table>
        <thead><tr><th>Test</th><th>Result</th><th>Limit</th><th>Verdict</th><th>Why</th></tr></thead>
        <tbody>
          {c.rows.map((r) => (
            <tr key={r.test}>
              <td>{r.test}</td>
              <td>{r.result || "not tested"}</td>
              <td>{r.limit || "–"}{r.limit_source === "coa" ? " (from the report)" : ""}</td>
              <td><span className={`badge ${VERDICT[r.verdict]}`}>{r.verdict.toUpperCase()}</span></td>
              <td className="muted">{r.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const label = (type: string) => type.replace(/_/g, " ");


function ExtractionCard({ extraction: x, onClose }: { extraction: Extraction; onClose: () => void }) {
  const isCert = x.doc_type === "calibration_cert";
  const field = (data: Record<string, unknown>, key: string) => (data[key] == null ? "–" : String(data[key]));
  return (
    <div className="card" aria-label="Extracted fields">
      <div className="row between">
        <strong>
          Extracted fields
          {x.total > 1 && ` · ${x.done} of ${x.total} certificates read`}
        </strong>
        <button className="link" onClick={onClose}>Close</button>
      </div>
      {x.status === "running" && <p role="status" className="muted">Still working: {x.done} of {x.total} done. This page updates by itself.</p>}
      {x.failed > 0 && (
        <p role="alert" className="error">
          {x.failed} certificate{x.failed === 1 ? "" : "s"} could not be read{x.error ? `: ${x.error}` : ""}. The rest were saved; try Re-run for the missing ones.
        </p>
      )}
      {x.status === "failed" && x.failed === 0 && x.error && <p role="alert" className="error">{x.error}</p>}
      {x.poor_ocr_pages > 0 && (
        <p role="alert" className="badge bad">
          {x.poor_ocr_pages} page(s) are a poor scan. Dates and numbers read from them may be wrong even
          where a value was found: check them against the original, or rescan.
        </p>
      )}
      {x.ocr_pages > 0 && (
        <p role="alert" className="badge warn">
          {x.ocr_pages} page(s) were read by OCR: check the figures against the original (OCR can misread digits).
        </p>
      )}
      {isCert && x.segments.length > 0 ? (
        <table>
          <thead><tr><th>Pages</th><th>Serial</th><th>Instrument</th><th>Calibrated</th><th>Next due</th><th>Check</th></tr></thead>
          <tbody>
            {x.segments.map((seg) => (
              <tr key={seg.start_page}>
                <td>{seg.end_page && seg.end_page !== seg.start_page ? `${seg.start_page}–${seg.end_page}` : seg.start_page}</td>
                <td>{field(seg.data, "instrument_serial")}</td>
                <td>{field(seg.data, "instrument_description")}</td>
                <td>{field(seg.data, "calibration_date")}</td>
                <td>{field(seg.data, "next_due")}</td>
                <td>
                  {seg.warnings.length > 0 && <span className="badge warn" title={seg.warnings.join("; ")}>{seg.warnings[0]}</span>}
                  {seg.needs_review && <span className="badge bad" title={seg.review_reason ?? ""}>needs review</span>}
                  {!seg.needs_review && seg.verified_by && <span className="badge good">confirmed by {seg.verified_by}</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        x.data && <pre className="page-text">{JSON.stringify(x.data, null, 2)}</pre>
      )}
    </div>
  );
}
