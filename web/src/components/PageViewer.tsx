import { useEffect, useState } from "react";
import { api, errorMessage, type Schemas } from "../api/client";

type Props = { docId: string; page: number; snippet?: string; onClose: () => void };

/** Shows the text of one page, the way the model saw it; the cited snippet is highlighted. */
export function PageViewer({ docId, page, snippet, onClose }: Props) {
  const [data, setData] = useState<Schemas["PageOut"] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    void api
      .GET("/documents/{doc_id}/pages/{page}", { params: { path: { doc_id: docId, page } } })
      .then(({ data: d, error: e }) => {
        if (!live) return;
        if (d) setData(d);
        else setError(errorMessage(e));
      });
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => {
      live = false;
      window.removeEventListener("keydown", onKey);
    };
  }, [docId, page, onClose]);

  return (
    <div className="overlay" onClick={onClose}>
      <div className="dialog card" role="dialog" aria-modal="true" aria-label="Source page" onClick={(e) => e.stopPropagation()}>
        <div className="row between">
          <strong>{data ? `${data.filename} · page ${data.page}` : "Source page"}</strong>
          <button className="link" onClick={onClose}>Close</button>
        </div>
        {error && <p role="alert" className="error">{error}</p>}
        {!data && !error && <p className="muted">Loading…</p>}
        {data && (
          <>
            {data.method === "ocr" && <p className="badge warn">Read by OCR: check against the original</p>}
            <pre className="page-text">{highlight(data.text, snippet)}</pre>
          </>
        )}
      </div>
    </div>
  );
}

function highlight(text: string, snippet?: string) {
  const needle = snippet?.split("\n").find((l) => l.trim().length > 12)?.trim();
  const at = needle ? text.indexOf(needle) : -1;
  if (!needle || at < 0) return text;
  return (
    <>
      {text.slice(0, at)}
      <mark>{needle}</mark>
      {text.slice(at + needle.length)}
    </>
  );
}
