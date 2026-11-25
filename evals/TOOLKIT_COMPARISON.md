# LLM extraction vs the toolkit's regex parser (Phase 3 comparison)

Run 2026-11-18. The toolkit's `calibration/extract.py` (per-supplier regex parsers) is the
reference. Inputs were the 11 certificate texts in the toolkit's own tests, which are taken from real
certificates (text layers and OCR output). Nothing from them is stored in this repo; this file holds
aggregate results only. The scratch script ran outside the repo.

| Field | Agreement with the regex parser (11 cases) |
|---|---|
| Calibration date | 11 / 11 |
| Instrument serial | 10 / 11 |
| Expiry / next due | 10 / 11 |

Prompt/schema `extract_v3`, model: the 30B-A3B Nemotron behind the proxy. Latency 22-140 s per
certificate (the model reasons even with thinking off); the synthetic set takes 15-20 s each.

## What the comparison found

1. **Truncation, not accuracy, was the first failure.** On real-format text the model spent its
   whole 4096-token budget reasoning and returned empty content; all retries failed the same way.
   `extract()` now detects `finish_reason == "length"`, retries with a doubled budget (default start
   8192, cap 16384) and `llm_timeout_s` is 300. Both Bizerba certificates went from failing to
   matching.
2. **Field descriptions decide recall.** With `next_due` described as "the date the next calibration
   is due", four TTP certificates returned null although "Valid To" was printed. Listing the label
   variants ("Valid To", "Next test due", "Suggested date for recalibration", ...) fixed all four.
   Same for `calibration_date` ("Valid From", "Date of test").
3. **Empty required strings must not validate.** Twice the model answered a required serial with
   `""`. `min_length=1` turns that into a validation error and a retry with the error fed back.
4. **Remaining differences are house rules, which belong in code.**
   - *Marel expiry:* the certificate prints only "Nov-26". The toolkit turns that into the
     anniversary of the authorised date. The model correctly returns null. Keep that rule as code.
   - *Multi-serial TTP layouts* (e.g. a head serial that has a "G20" prefix next to a conveyor
     serial): the regex parser encodes which column is the head. The model picked the conveyor
     serial. A second `conveyor_serial` field, or supplier-specific post-processing, is needed
     before these are filed automatically.

## Recommendation

Use the LLM for suppliers without a dedicated parser and for scanned or unfamiliar layouts, and keep
the regex parsers as the primary path for the five known suppliers (they are exact and instant).
Where both run, flag any disagreement for a person: that is a cheap, strong check. The toolkit's
parsers also split multi-certificate bundles, which this project does not do yet.

## Not covered

Real scanned certificates (only OCR text from the toolkit's fixtures), handwritten ones, and
bundles. The numbers come from 11 texts, so treat them as indicative.
