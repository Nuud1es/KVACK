# KVACK (Knowledge, Verification & Automation Compliance Kit) — GenAI document assistant for Gressingham quality team

An internal tool for Gressingham Foods that brings document filing, field extraction, Q&A and reminders into one service with a human-approved agent at the centre.

---

## 1. The problem

The quality team spends hours on paperwork that's almost all PDFs:

- Supplier specifications, certificates of analysis and audit certificates (BRCGS, etc.)
- Calibration certificates for probes, scales and metal detectors
- Lab results, SOPs, HACCP documents, customer codes of practice

People need to **file them**, **pull key fields out of them** (serial numbers, expiry dates, results against limits) and **answer questions from them** ("What's the max micro limit for product X in the Tesco CoP?" or "Which probe certs expire next month?").

Parts of this are already automated by separate modules in the AUTOMATION toolkit (`calibration`, `spec_organizer`, `reminder`, the OCR pipeline, File Organiser classification). KVACK wraps those as its first ingest/extraction implementations and adds retrieval, Q&A and an approval-gated agent on top. The existing Flask webapp keeps running alongside it.

## 2. What it does

| Capability | Example |
|---|---|
| **Ingest & classify** | Drop in a PDF → OCR if scanned → classified (calibration cert / supplier spec / CoA / SOP / other) |
| **Structured extraction** | Calibration cert → `{instrument_serial, cert_date, next_due, result: pass/fail}` as validated Pydantic objects |
| **RAG Q&A with citations** | "What temperature must cooked duck reach for release?" → answer + the exact SOP page it came from |
| **Agentic actions** | "Find all certs expiring in the next 30 days and draft reminders" → agent queries the DB, drafts Outlook emails, **waits for human approval** before sending |
| **Compliance checks** | Compare a CoA's results against spec limits and flag out-of-spec values |
| **Monitoring** | Langfuse traces: latency, token counts, retrieval quality, user feedback (thumbs up/down), eval scores per release |

**Design rule:** the LLM reads and writes language; code makes decisions. Expiry queries are SQL. CoA-vs-spec checks reuse the existing `parse_numeric_value` / `evaluate_result` logic. The LLM only extracts fields, answers questions with citations, and drafts text.

## 3. Deployment constraints and decisions

- **Development happens on either machine** (Windows work PC or the Linux laptop). Anything Windows-only sits behind an interface with a dev stub: mail goes through a `Mailer` interface (`OutlookMailer` on Windows via COM, `FileMailer` that writes `.eml` files for Linux/dev), and X: drive reads go through a document-source interface. Everything else is developed and tested on Linux; the Outlook sender is verified on Windows.
- **Production runs on a Windows machine.** The API and worker run natively on Windows (not in Linux containers) so they can use Outlook COM (`pythoncom.CoInitialize()` + `win32com`, as in the webapp's `_make_outlook_draft`) and read the X: drive. Reuse `OutlookClient` rather than writing new COM code.
- **Supporting services** (Postgres, Qdrant) run in Docker on the Windows host or on the Linux machine, whichever IT prefers. Config via env.
- **LLM is offloaded to the Linux machine** through an **OpenAI-compatible endpoint** (llama.cpp server / Ollama / vLLM). One `OpenAICompatibleProvider` behind the provider ABC covers it; the base URL and model name are config. Nothing leaves the site.
- **Langfuse is self-hosted** (on the Linux machine, via its own docker compose: Postgres, ClickHouse, Redis, S3-compatible store). It is the only monitoring stack; no Prometheus/Grafana in the core.
- **Human in the loop.** The agent never sends an email or writes to FoodOps without a person approving it.
- **Access control and audit.** Supplier specs are commercially sensitive. API-key auth at minimum, per-user identity on approvals, and an audit log of who approved what (Entra ID login is a stretch goal).
- **Data:** develop against synthetic or anonymised documents in the repo; real documents stay on site and out of git.

## 4. Architecture

```
┌──────────────────────┐        ┌────────────────────────────────────────────┐
│  React + TypeScript  │  HTTP  │       FastAPI service (Windows host)       │
│  (Vite, chat UI,     │◄──────►│  /ingest /ask /extract /agent /feedback    │
│   doc upload, review │  SSE   │  /health   Auth: API key                   │
│   & approval queue)  │        └───────┬───────────────┬────────────────────┘
└──────────────────────┘                │               │
                                        ▼               ▼
                          ┌──────────────────┐  ┌──────────────────────────┐
                          │ Ingestion worker │  │   LangGraph agent        │
                          │ OCR → classify → │  │  router → retrieve →     │
                          │ chunk → embed    │  │  tools → validate →      │
                          └────────┬─────────┘  │  human approval → act    │
                                   │            └──────┬───────────────────┘
                                   ▼                   │
                 ┌──────────────┐  ┌────────────┐      │   ┌────────────────────────┐
                 │   Qdrant     │  │ PostgreSQL │◄─────┘   │ Linux machine          │
                 │ (vectors)    │  │ (metadata, │          │  OpenAI-compatible LLM │
                 └──────────────┘  │  certs,    │          │  Langfuse (self-hosted)│
                                   │  feedback, │          └────────────────────────┘
        Outlook COM (Windows) ◄─── │  audit,    │
        X: drive (read)            │  checkpts) │
                                   └────────────┘
```

### LangGraph agent graph

```
            ┌──────────┐
  query ──► │  router  │──► "question"  ──► retrieve ──► answer_with_citations ──► END
            └──────────┘──► "extract"   ──► extract_fields ──► validate (Pydantic) ──┐
                       └──► "action"    ──► plan ──► tool calls ──► HUMAN APPROVAL ──┤
                                                                     (interrupt)     ▼
                                                                           execute ──► END
```

- LangGraph **checkpointing** (Postgres) so the approval step can pause for hours and resume.
- Tools: `search_documents`, `query_certificates` (SQL), `check_against_spec` (deterministic), `draft_email`, `send_email` (approval-gated, Outlook COM, idempotent so a retried approval can't double-send).
- Retry on failed validation (structured-output loop).

## 5. Repo structure

```
qa-copilot/
├── packages/qacopilot/              # installable library
│   ├── pyproject.toml
│   └── src/qacopilot/
│       ├── llm/                     # BaseLLMProvider ABC + OpenAICompatible (+ Ollama)
│       ├── ingest/                  # OCR, classifier, chunkers per doc type
│       ├── retrieval/               # Qdrant store, hybrid search, reranker
│       ├── extraction/              # Pydantic schemas: CalibrationCert, CoA, SupplierSpec
│       ├── agent/                   # LangGraph state, nodes, tools, graph builder
│       ├── mail/                    # Mailer interface: OutlookMailer (Windows COM) + FileMailer (dev)
│       ├── repositories/            # Postgres access (SQLAlchemy 2.0 + Alembic)
│       └── config.py                # pydantic-settings
├── services/
│   ├── api/                         # FastAPI app (routers, deps, auth)
│   └── worker/                      # ingestion worker
├── web/                             # React + TypeScript (Vite)
├── prompts/                         # versioned prompt templates
├── evals/                           # golden Q&A set, extraction test docs, eval runner
├── data/samples/                    # SYNTHETIC documents only
├── infra/
│   └── docker-compose.yml           # postgres, qdrant (Langfuse has its own compose)
├── .github/workflows/
│   ├── ci.yml                       # lint, type-check, tests, deterministic evals
│   └── evals-llm.yml                # LLM-judged evals on the self-hosted runner
└── README.md
```

## 6. Chunking strategy (per document type)

| Type | Strategy |
|---|---|
| Calibration cert / CoA | Not chunked for Q&A; extracted to structured rows, full text kept as one record per page |
| Supplier spec | Section-aware (heading-based), tables kept intact, product/supplier metadata on every chunk |
| SOP / HACCP | Heading-aware chunks with parent section title prepended; page number in metadata for citations |
| Customer CoP | Clause-numbered chunks; clause ID in metadata |

## 7. Build plan (around 6 weekends)

### Phase 1 — Foundations + feasibility (weekend 1)
- [x] Monorepo, `pyproject.toml`, `uv` (plain venv + pip used so far), `ruff` + `mypy` + `pytest`, pre-commit
- [x] `BaseLLMProvider` ABC + `OpenAICompatibleProvider` pointing at the Linux machine
- [x] **Feasibility spike on the real target model:** JSON-schema extraction accuracy, tool-calling reliability and latency on 10 sample docs. Decide model size before building further
- [x] FastAPI skeleton: `/health`, settings, structured logging
- [x] Docker Compose (postgres, qdrant); Langfuse self-hosted on the Linux machine with the first traced call
- [x] GitHub Actions CI: lint, type-check, tests

### Phase 2 — Ingestion, RAG and the eval baseline (weekend 2)
- [x] PDF text extraction + OCR fallback (wrap the existing pipeline)
- [x] Document classifier (rules-based first, then LLM; compare accuracy): LLM fills gaps and flags disagreements, never overrules a keyword match; see evals/results/classification_*.json
- [x] Per-type chunking, embeddings, Qdrant indexing with metadata
- [x] `/ask`: hybrid retrieval + reranking + answer with citations, streamed via SSE
- [x] **Golden set v1 (~15 questions) and eval runner; record the baseline scores now**

### Phase 3 — Structured extraction (weekend 3)
- [x] Pydantic schemas for calibration certs, CoAs and supplier specs
- [x] Structured-output extraction with validation + retry (reuse `calibration/extract.py` as a reference implementation and comparison baseline)
- [x] Store results in Postgres; `/certificates?expiring_within=30d`
- [x] Spec-limit checker (deterministic, reusing `common/business_logic.py`)

### Phase 4 — Agent (weekend 4)
- [x] LangGraph state + router node (deterministic `RuleRouter`; the `extract` route is the `/documents/{id}/extract` endpoint instead)
- [x] Tools (typed functions, not LLM-chosen) wrapping retrieval, the certificate repository, the spec checker and Outlook drafting
- [x] Human-in-the-loop `interrupt` before any action, Postgres checkpointer
- [x] `/agent/runs`: start, get state, approve/reject; audit-log every approval
- [x] `/feedback` endpoint writing thumbs up/down to Postgres and Langfuse scores

### Phase 5 — Front end (weekend 5)
- [x] React + TS (Vite): chat with streaming + citation links, document upload, approval queue (minimal, functional)
- [x] Typed API client generated from the OpenAPI schema (`openapi-typescript`)

### Phase 6 — Evaluation, CI, polish (weekend 6)
- [x] Grow golden set to ~30 questions + ~20 synthetic docs with expected fields, including adversarial cases: scanned/rotated PDFs, near-duplicate certs, out-of-scope questions where the right answer is "not in the documents"
- [x] **CI (GitHub-hosted):** deterministic checks only: extraction field accuracy on recorded outputs, retrieval hit@k, citation presence, prompt regression with a mock provider
- [x] **LLM-judged evals (faithfulness, answer relevance):** run on a **self-hosted GitHub runner on the Linux machine**, against the local LLM; on push to main and manual dispatch only. Keep the repo private or restrict the workflow so fork PRs can't execute code on that runner
- [x] Eval scores logged per release to Langfuse; fail the build if scores drop
- [x] README with architecture diagram, run instructions (screenshots still to add: the UI has not been captured in a browser) (Windows service + compose)

### Stretch goals
- [ ] Entra ID (Azure AD) login
- [ ] Microsoft Graph in place of Outlook COM (would allow a Linux deployment)
- [ ] Guardrails: prompt-injection checks on uploaded documents
- [ ] Cloud provider (Azure OpenAI / Claude) behind the same ABC, if the business ever approves one

## 8. Skills to touch up while building

| Skill | Quick way in |
|---|---|
| **Async Python** | Make every endpoint and provider call `async`; `asyncio.gather` |
| **Pydantic v2 & typing** | Type the whole package; get `mypy --strict` passing |
| **Testing** | `pytest` + fixtures; mock LLM providers; test LangGraph nodes in isolation |
| **SQL & SQLAlchemy 2.0** | Repository layer + Alembic migrations |
| **RAG evaluation** | Golden set from week 2; Ragas or custom metrics |
| **Observability** | Langfuse traces and scores |
| **CI/CD** | GitHub Actions, self-hosted runner |
| **Security & responsible AI** | Access control, audit log, prompt injection, human oversight |
| **System design talk-through** | Be able to explain the architecture in 5 minutes, including trade-offs (offloaded local LLM, Qdrant vs pgvector, deterministic logic vs LLM, why human approval) |

## 9. Interview / write-up stories

1. Keeping data on site: a local LLM served over an OpenAI-compatible API, swappable via config
2. How retrieval quality was measured and improved (baseline vs final eval scores)
3. Why the agent pauses for human approval, and how checkpointing makes that work
4. Why decisions are deterministic code and the LLM only reads and drafts
5. The provider abstraction: changing the model without touching business logic
