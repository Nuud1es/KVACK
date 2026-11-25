import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Documents } from "./Documents";
import { json, mockApi } from "../test/mockApi";

// every Documents render also asks for the review queue; most tests have nothing to review
const mock = (routes: Parameters<typeof mockApi>[0]) =>
  mockApi({ "GET /certificates/review": () => json([]), "GET /review/documents": () => json([]), ...routes });

const extraction = (over: object = {}) => ({
  doc_id: "d1", doc_type: "calibration_cert", status: "done", total: 1, done: 1, failed: 0, error: null,
  prompt_version: "extract_v5", model: "m", ocr_pages: 0, poor_ocr_pages: 0, data: { instrument_serial: "TP-1" },
  segments: [{ start_page: 1, end_page: 1, data: { instrument_serial: "TP-1", calibration_date: "2026-01-01", next_due: "2027-01-01" }, warnings: [] }],
  ...over,
});
const doc = (extracted: boolean, over: object = {}) => ({
  doc_id: "d1", filename: "cert.pdf", doc_type: "calibration_cert", classified_by: "rules", pages: 1,
  ocr_pages: 1, poor_ocr_pages: 0, chunks: 1, ingested_at: "2026-11-18T10:00:00Z", extracted,
  alternative_type: null, extraction_status: null, extraction_done: 0, extraction_total: 0, ...over,
});
const due = [{ doc_id: "d1", instrument_serial: "TP-1", instrument_description: "Probe", certificate_number: null, calibration_date: "2025-12-01", next_due: "2026-12-01", days_left: 13, status: "due_soon" }];

describe("Documents", () => {
  it("lists documents and certificates due, and extracts fields on request", async () => {
    let extracted = false;
    const api = mock({
      "GET /documents": () => json([doc(extracted)]),
      "GET /certificates": () => json(due),
      "POST /documents/d1/extract": () => {
        extracted = true;
        return json(extraction());
      },
    });
    render(<Documents />);
    expect(await screen.findByText("cert.pdf")).toBeInTheDocument();
    expect(screen.getByText("OCR")).toBeInTheDocument();
    expect(await screen.findByText("in 13 days")).toBeInTheDocument();
    expect(api.calls.find((c) => c.key === "GET /certificates")).toBeDefined();

    await userEvent.click(screen.getByRole("button", { name: "Extract" }));
    const card = await screen.findByLabelText("Extracted fields");
    expect(within(card).getByRole("cell", { name: "TP-1" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Re-run" })).toBeInTheDocument());
  });

  it("uploads a PDF and reports what was found", async () => {
    const api = mock({
      "GET /documents": () => json([]),
      "GET /certificates": () => json([]),
      "POST /ingest": () => json({ doc_id: "d9", filename: "scan.pdf", doc_type: "sop", classified_by: "rules", confidence: 0.8, pages: 2, ocr_pages: 2, chunks: 4 }),
    });
    render(<Documents />);
    const file = new File(["%PDF-1.4"], "scan.pdf", { type: "application/pdf" });
    await userEvent.upload(screen.getByLabelText(/Add PDFs or photos/), file);
    expect(await screen.findByText(/scan.pdf: sop, 2 page\(s\) read by OCR, 4 chunks/)).toBeInTheDocument();
    expect(api.count("POST /ingest")).toBe(1);
  });

  it("uploads the PDFs in a folder, skips other files, then reads their fields", async () => {
    let n = 0;
    const api = mock({
      "GET /documents": () => json([]),
      "GET /certificates": () => json([]),
      "POST /ingest": () => {
        n += 1;
        const type = n === 1 ? "calibration_cert" : "sop";
        return json({ doc_id: `d${n}`, filename: `f${n}.pdf`, doc_type: type, classified_by: "rules", confidence: 0.9, pages: 1, ocr_pages: 0, chunks: 2 });
      },
      "POST /documents/d1/extract": () => json(extraction()),
    });
    render(<Documents pollMs={5} />);
    const file = (name: string, type: string) => {
      const f = new File(["x"], name, { type });
      Object.defineProperty(f, "webkitRelativePath", { value: `Probes/${name}` });
      return f;
    };
    await userEvent.upload(screen.getByLabelText("Choose a folder"), [
      file("cert.pdf", "application/pdf"),
      file("manual.pdf", "application/pdf"),
      file("notes.txt", "text/plain"),
    ]);
    expect(await screen.findByText(/Done: 2 uploaded, 1 skipped \(not a PDF or photo\), fields read for 1/)).toBeInTheDocument();
    expect(screen.getByText(/Probes\/cert.pdf: calibration cert,/)).toBeInTheDocument();
    expect(api.count("POST /ingest")).toBe(2);
    expect(api.count("POST /documents/d1/extract")).toBe(1); // only the certificate; the SOP has no fields
  });

  it("does not read the fields when the box is unticked", async () => {
    const api = mock({
      "GET /documents": () => json([]),
      "GET /certificates": () => json([]),
      "POST /ingest": () => json({ doc_id: "d1", filename: "a.pdf", doc_type: "calibration_cert", classified_by: "rules", confidence: 0.9, pages: 1, ocr_pages: 0, chunks: 2 }),
    });
    render(<Documents pollMs={5} />);
    await userEvent.click(screen.getByLabelText(/Read the fields after uploading/));
    const f = new File(["x"], "a.pdf", { type: "application/pdf" });
    await userEvent.upload(screen.getByLabelText("Choose a folder"), [f]);
    await waitFor(() => expect(api.count("POST /ingest")).toBe(1));
    expect(api.count("POST /documents/d1/extract")).toBe(0);
  });

  it("says so when a folder has nothing to read", async () => {
    mock({ "GET /documents": () => json([]), "GET /certificates": () => json([]) });
    render(<Documents />);
    await userEvent.upload(screen.getByLabelText("Choose a folder"), [new File(["x"], "notes.txt", { type: "text/plain" })]);
    expect(await screen.findByRole("alert")).toHaveTextContent("no PDFs or photos");
  });

  it("shows an upload error from the server", async () => {
    mock({
      "GET /documents": () => json([]),
      "GET /certificates": () => json([]),
      "POST /ingest": () => json({ detail: "Only PDF files are accepted" }, 422),
    });
    render(<Documents />);
    await userEvent.upload(screen.getByLabelText(/Add PDFs or photos/), new File(["x"], "notes.pdf", { type: "application/pdf" }));
    expect(await screen.findByText(/notes.pdf: Only PDF files are accepted/)).toBeInTheDocument();
  });

  it("warns when extracted values came from OCR'd pages", async () => {
    mock({
      "GET /documents": () => json([doc(true)]),
      "GET /certificates": () => json([]),
      "GET /documents/d1/extraction": () =>
        json(extraction({ ocr_pages: 1 })),
    });
    render(<Documents />);
    await userEvent.click(await screen.findByRole("button", { name: "View" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("read by OCR");
  });

  it("marks types chosen by the AI model and lets a person correct the type", async () => {
    let type = "other";
    const api = mock({
      "GET /documents": () => json([doc(false, { doc_type: type, classified_by: type === "other" ? "llm" : "user" })]),
      "GET /certificates": () => json([]),
      "POST /documents/d1/reclassify": ({ body }) => {
        type = (body as { doc_type: string }).doc_type;
        return json({ doc_id: "d1", doc_type: type, chunks: 2 });
      },
    });
    render(<Documents />);
    expect(await screen.findByTitle(/AI model chose this type/)).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Type of cert.pdf"), "sop");
    await waitFor(() => expect(api.count("POST /documents/d1/reclassify")).toBe(1));
    expect(api.calls.find((c) => c.key === "POST /documents/d1/reclassify")?.body).toEqual({ doc_type: "sop" });
    expect(await screen.findByTitle("Set by a person")).toBeInTheDocument();
  });

  it("says when the AI model chose the type of an upload", async () => {
    mock({
      "GET /documents": () => json([]),
      "GET /certificates": () => json([]),
      "POST /ingest": () => json({ doc_id: "d2", filename: "x.pdf", doc_type: "supplier_spec", classified_by: "llm", confidence: 1, pages: 1, ocr_pages: 0, chunks: 3 }),
    });
    render(<Documents />);
    await userEvent.upload(screen.getByLabelText(/Add PDFs or photos/), new File(["%PDF"], "x.pdf", { type: "application/pdf" }));
    expect(await screen.findByText(/chosen by the AI model: check it/)).toBeInTheDocument();
  });

  it("reads a bundle in the background: shows progress, then every certificate", async () => {
    let polls = 0;
    let started = false;
    const seg = (n: number) => ({ start_page: n, end_page: n, data: { instrument_serial: `TP-${n}`, calibration_date: "2026-01-01", next_due: "2027-01-01" }, warnings: [] });
    mock({
      "GET /documents": () => json([
        !started ? doc(false)
        : polls > 1 ? doc(true, { extraction_status: "done", extraction_done: 3, extraction_total: 3 })
        : doc(false, { extraction_status: "running", extraction_done: polls, extraction_total: 3 }),
      ]),
      "GET /certificates": () => json([]),
      "POST /documents/d1/extract": () => {
        started = true;
        return json(extraction({ status: "running", total: 3, done: 0, segments: [], data: null }), 202);
      },
      "GET /documents/d1/extraction": () => {
        polls += 1;
        return json(polls > 1
          ? extraction({ status: "done", total: 3, done: 3, segments: [seg(1), seg(2), seg(3)] })
          : extraction({ status: "running", total: 3, done: 1, segments: [seg(1)] }));
      },
    });
    render(<Documents pollMs={20} />);
    await userEvent.click(await screen.findByRole("button", { name: "Extract" }));
    expect((await screen.findAllByRole("status"))[0]).toHaveTextContent(/of 3/);
    expect(await screen.findByText(/3 of 3 certificates read/)).toBeInTheDocument();
    const card = screen.getByLabelText("Extracted fields");
    for (const serial of ["TP-1", "TP-2", "TP-3"]) expect(within(card).getByRole("cell", { name: serial })).toBeInTheDocument();
  });

  it("says how many certificates could not be read and flags odd dates", async () => {
    mock({
      "GET /documents": () => json([doc(true)]),
      "GET /certificates": () => json([]),
      "GET /documents/d1/extraction": () =>
        json(extraction({
          total: 2, done: 1, failed: 1, error: "pages 2-2: invalid",
          segments: [{ start_page: 1, end_page: 1, data: { instrument_serial: "TP-1", calibration_date: "2026-01-01", next_due: "2029-01-01" }, warnings: ["The calibration interval is 1096 days (expected about a year)"] }],
        })),
    });
    render(<Documents />);
    await userEvent.click(await screen.findByRole("button", { name: "View" }));
    expect(await screen.findByText(/1 certificate could not be read: pages 2-2: invalid/)).toBeInTheDocument();
    expect(screen.getByText(/calibration interval is 1096 days/)).toBeInTheDocument();
  });

  it("flags a type the AI model disagreed with, and clears the flag once a person sets it", async () => {
    let alt: string | null = "supplier_spec";
    mock({
      "GET /documents": () => json([doc(false, { doc_type: "coa", alternative_type: alt, classified_by: alt ? "rules" : "user" })]),
      "GET /certificates": () => json([]),
      "POST /documents/d1/reclassify": () => {
        alt = null;
        return json({ doc_id: "d1", doc_type: "coa", chunks: 1 });
      },
    });
    render(<Documents />);
    expect(await screen.findByTitle(/might be supplier spec/)).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Type of cert.pdf"), "other");
    await waitFor(() => expect(screen.queryByTitle(/might be supplier spec/)).not.toBeInTheDocument());
  });

  it("warns about a poor scan in the list and in the extracted fields", async () => {
    mock({
      "GET /documents": () => json([doc(true, { poor_ocr_pages: 3 })]),
      "GET /certificates": () => json([]),
      "GET /documents/d1/extraction": () => json(extraction({ ocr_pages: 3, poor_ocr_pages: 3 })),
    });
    render(<Documents />);
    expect(await screen.findByTitle(/3 page\(s\) were scanned poorly/)).toHaveTextContent("poor scan");
    await userEvent.click(screen.getByRole("button", { name: "View" }));
    expect(await screen.findByText(/3 page\(s\) are a poor scan/)).toBeInTheDocument();
  });

  it("checks a CoA against its matched spec and explains an ambiguous match", async () => {
    let status = 200;
    mock({
      "GET /documents": () => json([doc(true, { doc_type: "coa" })]),
      "GET /certificates": () => json([]),
      "GET /compliance/coa-check": () =>
        status === 200
          ? json({
              coa_doc_id: "d1", spec_doc_id: "s2", overall: "fail", pairing: "matched",
              spec_filename: "spec_v2.pdf", spec_revision: 2, spec_effective: "2026-10-01",
              warnings: ["The report is dated 2026-09-20, before revision 2 took effect."],
              rows: [{ test: "Total viable count", result: "2.4E+05", limit: "< 1.0E+05", limit_source: "spec", verdict: "fail", reason: "above the limit" }],
            })
          : json({ detail: "2 different specs match; choose one. Candidates: a.pdf, b.pdf." }, status),
    });
    render(<Documents />);
    await userEvent.click(await screen.findByRole("button", { name: "Check against spec" }));
    const card = await screen.findByLabelText("Spec check");
    expect(card).toHaveTextContent("spec_v2.pdf (revision 2)");
    expect(card).toHaveTextContent("FAIL");
    expect(card).toHaveTextContent("before revision 2 took effect");
    expect(card).toHaveTextContent("above the limit");

    status = 409;
    await userEvent.click(screen.getByRole("button", { name: "Check against spec" }));
    expect(await screen.findByText(/Candidates: a.pdf, b.pdf/)).toBeInTheDocument();
  });

  const reviewItem = {
    doc_id: "d1", filename: "scan.pdf", start_page: 1, end_page: null, instrument_serial: "TP-7",
    instrument_description: "Digital probe", certificate_number: null, calibration_date: "2026-03-14",
    next_due: "2029-03-14", needs_review: true, reason: "1 page(s) are a poor scan, so the values may be misread",
    min_ocr_confidence: 58, verified_by: null, verified_at: null,
  };

  it("lists readings that need review, and a correction is sent and clears the item", async () => {
    let queue = [reviewItem];
    const api = mock({
      "GET /documents": () => json([doc(true)]),
      "GET /certificates": () => json([]),
      "GET /certificates/review": () => json(queue),
      "POST /documents/d1/certificates/1/review": () => {
        queue = [];
        return json({ ...reviewItem, next_due: "2027-03-14", needs_review: false, verified_by: "dev" });
      },
    });
    render(<Documents />);
    const card = await screen.findByLabelText("Review scan.pdf page 1");
    expect(card).toHaveTextContent("poor scan");
    expect(within(card).getByRole("button", { name: "Confirm as correct" })).toBeInTheDocument();

    const nextDue = within(card).getByLabelText("Next due (YYYY-MM-DD)");
    await userEvent.clear(nextDue);
    await userEvent.type(nextDue, "2027-03-14");
    await userEvent.type(within(card).getByLabelText("Note (optional)"), "checked the paper copy");
    await userEvent.click(within(card).getByRole("button", { name: "Save correction and confirm" }));

    await waitFor(() => expect(screen.queryByLabelText("Review scan.pdf page 1")).not.toBeInTheDocument());
    const sent = api.calls.find((c) => c.key === "POST /documents/d1/certificates/1/review");
    expect(sent?.body).toEqual({ corrections: { next_due: "2027-03-14" }, note: "checked the paper copy" });
  });

  it("confirms a reading as it is without sending any corrections", async () => {
    const api = mock({
      "GET /documents": () => json([doc(true)]),
      "GET /certificates": () => json([]),
      "GET /certificates/review": () => json([reviewItem]),
      "POST /documents/d1/certificates/1/review": () => json({ ...reviewItem, needs_review: false }),
    });
    render(<Documents />);
    await userEvent.click(await screen.findByRole("button", { name: "Confirm as correct" }));
    await waitFor(() => expect(api.count("POST /documents/d1/certificates/1/review")).toBe(1));
    expect(api.calls.find((c) => c.key.startsWith("POST"))?.body).toEqual({ corrections: null, note: null });
  });

  it("lists an incomplete report in the same queue and sends a correction for it", async () => {
    const specItem = {
      doc_id: "s1",
      filename: "spec.pdf",
      doc_type: "supplier_spec",
      start_page: 1,
      end_page: 1,
      reason: "Salmonella is printed on the page but no limit was read for it",
      fields: { supplier: null, product: "Duck", product_code: null, shelf_life: "14 days", storage: null },
      min_ocr_confidence: null,
      verified_by: null,
    };
    const api = mock({
      "GET /documents": () => json([doc(true)]),
      "GET /certificates": () => json([]),
      "GET /review/documents": () => json([specItem]),
      "POST /documents/s1/extractions/1/review": () => json(specItem),
    });
    render(<Documents />);
    const card = await screen.findByLabelText("Review spec.pdf");
    expect(card).toHaveTextContent("Salmonella is printed on the page but no limit was read");
    expect(screen.getByRole("heading", { name: "Needs review (1)" })).toBeInTheDocument();
    await userEvent.type(within(card).getByLabelText("Supplier"), "Acme Foods");
    await userEvent.click(within(card).getByRole("button", { name: "Save correction and confirm" }));
    await waitFor(() => expect(screen.queryByLabelText("Review spec.pdf")).not.toBeInTheDocument());
    const sent = api.calls.find((c) => c.key === "POST /documents/s1/extractions/1/review");
    expect(sent?.body).toEqual({ corrections: { supplier: "Acme Foods" }, note: null });
  });

  it("shows why a correction was refused and keeps the item", async () => {
    mock({
      "GET /documents": () => json([doc(true)]),
      "GET /certificates": () => json([]),
      "GET /certificates/review": () => json([reviewItem]),
      "POST /documents/d1/certificates/1/review": () => json({ detail: "Not a valid certificate: next_due" }, 422),
    });
    render(<Documents />);
    const card = await screen.findByLabelText("Review scan.pdf page 1");
    await userEvent.type(within(card).getByLabelText("Note (optional)"), "x");
    const nextDue = within(card).getByLabelText("Next due (YYYY-MM-DD)");
    await userEvent.clear(nextDue);
    await userEvent.type(nextDue, "soon");
    await userEvent.click(within(card).getByRole("button", { name: "Save correction and confirm" }));
    expect(await within(card).findByRole("alert")).toHaveTextContent("Not a valid certificate");
    expect(screen.getByLabelText("Review scan.pdf page 1")).toBeInTheDocument();
  });

  it("marks unverified readings in the due list", async () => {
    mock({
      "GET /documents": () => json([]),
      "GET /certificates": () =>
        json([{ doc_id: "d1", start_page: 1, instrument_serial: "TP-7", instrument_description: null, certificate_number: null,
          calibration_date: "2026-03-14", next_due: "2026-12-01", days_left: 12, status: "due_soon", unverified: true, review_reason: "poor scan" }]),
    });
    render(<Documents />);
    expect(await screen.findByText("unverified")).toBeInTheDocument();
  });
});
