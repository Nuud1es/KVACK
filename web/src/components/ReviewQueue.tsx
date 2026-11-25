import { useCallback, useEffect, useState } from "react";
import { api, errorMessage, type Schemas } from "../api/client";
import { PageViewer } from "./PageViewer";

type Item = Schemas["ReviewItem"];
type DocItem = Schemas["DocReviewItem"];

const DOC_LABELS: Record<string, string> = {
  product: "Product",
  batch: "Batch",
  report_date: "Report date (YYYY-MM-DD)",
  lab: "Laboratory",
  supplier: "Supplier",
  product_code: "Product code",
  shelf_life: "Shelf life",
  storage: "Storage",
};
const DOC_KIND: Record<string, string> = { coa: "Report", supplier_spec: "Specification" };

const FIELDS = [
  ["instrument_serial", "Serial"],
  ["instrument_description", "Instrument"],
  ["certificate_number", "Certificate no."],
  ["calibration_date", "Calibrated (YYYY-MM-DD)"],
  ["next_due", "Next due (YYYY-MM-DD)"],
] as const;
type Field = (typeof FIELDS)[number][0];

/**
 * Readings from poor scans (or with odd dates) that nobody has confirmed. Until a person does,
 * answers and reminders mark them as unverified. The original PDF is not stored, so the page text
 * is shown for reference; compare it with the paper or file you hold.
 */
export function ReviewQueue({ onChanged }: { onChanged?: () => void }) {
  const [items, setItems] = useState<Item[] | null>(null);
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [certs, others] = await Promise.all([api.GET("/certificates/review"), api.GET("/review/documents")]);
    if (certs.data && others.data) {
      setItems(certs.data);
      setDocs(others.data);
      setError(null);
    } else setError(errorMessage(certs.error ?? others.error));
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const done = (item: Item) => {
    setItems((cur) => (cur ?? []).filter((i) => !(i.doc_id === item.doc_id && i.start_page === item.start_page)));
    onChanged?.();
  };

  const doneDoc = (item: DocItem) => {
    setDocs((cur) => cur.filter((i) => !(i.doc_id === item.doc_id && i.start_page === item.start_page)));
    onChanged?.();
  };

  if (error) return <p role="alert" className="error">{error}</p>;
  if (!items || (items.length === 0 && docs.length === 0)) return null;
  return (
    <section aria-label="Needs review">
      <h2>Needs review ({items.length + docs.length})</h2>
      <p className="muted">
        These readings come from a poor scan, have odd dates or look incomplete, so a value may be
        misread or missing. Until you confirm or correct them, answers, checks and reminders mark
        them as unverified.
      </p>
      {items.map((item) => (
        <ReviewRow key={`${item.doc_id}:${item.start_page}`} item={item} onDone={() => done(item)} />
      ))}
      {docs.map((item) => (
        <DocReviewRow key={`${item.doc_id}:${item.start_page}`} item={item} onDone={() => doneDoc(item)} />
      ))}
    </section>
  );
}

function DocReviewRow({ item, onDone }: { item: DocItem; onDone: () => void }) {
  const keys = Object.keys(item.fields);
  const original: Record<string, string> = Object.fromEntries(keys.map((k) => [k, item.fields[k] ?? ""]));
  const [values, setValues] = useState(original);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPage, setShowPage] = useState(false);

  const changed = keys.filter((k) => (values[k] ?? "") !== (original[k] ?? ""));

  async function submit() {
    setBusy(true);
    setError(null);
    const corrections: Record<string, string | null> = {};
    for (const k of changed) corrections[k] = (values[k] ?? "").trim() === "" ? null : (values[k] ?? "").trim();
    const { data, error: e } = await api.POST("/documents/{doc_id}/extractions/{start_page}/review", {
      params: { path: { doc_id: item.doc_id, start_page: item.start_page } },
      body: { corrections: changed.length ? corrections : null, note: note.trim() || null },
    });
    setBusy(false);
    if (data) onDone();
    else setError(errorMessage(e));
  }

  return (
    <div className="card" aria-label={`Review ${item.filename}`}>
      <div className="row between">
        <strong>{DOC_KIND[item.doc_type] ?? item.doc_type} · {item.filename}</strong>
        <button className="link" onClick={() => setShowPage(true)}>Show page text</button>
      </div>
      {item.reason && <p className="badge warn">{item.reason}</p>}
      {item.min_ocr_confidence != null && <p className="muted">Lowest OCR confidence on these pages: {item.min_ocr_confidence}</p>}
      <div className="review-fields">
        {keys.map((k) => (
          <label key={k}>
            {DOC_LABELS[k] ?? k}
            <input value={values[k] ?? ""} onChange={(e) => setValues((v) => ({ ...v, [k]: e.target.value }))} />
          </label>
        ))}
        <label>
          Note (optional)
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="e.g. checked against the original" />
        </label>
      </div>
      <p className="muted">
        The list of results or limits cannot be edited here. If a row is missing, check the original
        and run the extraction again.
      </p>
      {error && <p role="alert" className="error">{error}</p>}
      <button disabled={busy} onClick={() => void submit()}>
        {busy ? "Saving…" : changed.length ? "Save correction and confirm" : "Confirm as checked"}
      </button>
      {showPage && <PageViewer docId={item.doc_id} page={item.start_page} onClose={() => setShowPage(false)} />}
    </div>
  );
}

function ReviewRow({ item, onDone }: { item: Item; onDone: () => void }) {
  const original: Record<Field, string> = {
    instrument_serial: item.instrument_serial,
    instrument_description: item.instrument_description ?? "",
    certificate_number: item.certificate_number ?? "",
    calibration_date: item.calibration_date,
    next_due: item.next_due ?? "",
  };
  const [values, setValues] = useState(original);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPage, setShowPage] = useState(false);

  const changed = FIELDS.filter(([key]) => values[key] !== original[key]);

  async function submit() {
    setBusy(true);
    setError(null);
    const corrections: Record<string, string | null> = {};
    for (const [key] of changed) corrections[key] = values[key].trim() === "" ? null : values[key].trim();
    const { data, error: e } = await api.POST("/documents/{doc_id}/certificates/{start_page}/review", {
      params: { path: { doc_id: item.doc_id, start_page: item.start_page } },
      body: { corrections: changed.length ? corrections : null, note: note.trim() || null },
    });
    setBusy(false);
    if (data) onDone();
    else setError(errorMessage(e));
  }

  const pages = item.end_page && item.end_page !== item.start_page ? `${item.start_page}–${item.end_page}` : item.start_page;
  return (
    <div className="card" aria-label={`Review ${item.filename} page ${item.start_page}`}>
      <div className="row between">
        <strong>{item.filename} · page {pages}</strong>
        <button className="link" onClick={() => setShowPage(true)}>Show page text</button>
      </div>
      {item.reason && <p className="badge warn">{item.reason}</p>}
      {item.min_ocr_confidence != null && <p className="muted">Lowest OCR confidence on these pages: {item.min_ocr_confidence}</p>}
      <div className="review-fields">
        {FIELDS.map(([key, label]) => (
          <label key={key}>
            {label}
            <input value={values[key]} onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))} />
          </label>
        ))}
        <label>
          Note (optional)
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="e.g. checked against the paper certificate" />
        </label>
      </div>
      {error && <p role="alert" className="error">{error}</p>}
      <button disabled={busy} onClick={() => void submit()}>
        {busy ? "Saving…" : changed.length ? "Save correction and confirm" : "Confirm as correct"}
      </button>
      {showPage && <PageViewer docId={item.doc_id} page={item.start_page} onClose={() => setShowPage(false)} />}
    </div>
  );
}
