"""Application settings, read from environment variables (prefix ``QACOPILOT_``) or a .env file."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QACOPILOT_", env_file=".env", extra="ignore")

    # OpenAI-compatible LLM server (llama-server on the Linux machine)
    llm_base_url: str = "http://localhost:8080/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "nemotron"
    llm_timeout_s: float = 300.0  # a reasoning model can take minutes on a long extraction
    llm_connect_timeout_s: float = 10.0  # an unreachable server should fail fast, not hang
    # /metrics: if set, scrapers must send "Authorization: Bearer <token>"; if empty, the same API
    # key as the other endpoints applies (and nothing is required in local development).
    metrics_token: str = ""

    # Ask the LLM for the document type, but only when the keyword rules are unsure.
    llm_classify: bool = True

    # Let the model reason before extracting fields. Slower (seconds to minutes per document) and
    # usually unnecessary; measure with evals/run_extraction.py before turning it on.
    extraction_thinking: bool = False

    # Certificates of a bundle read at once. Leave at 1 unless the LLM server can serve parallel
    # requests (llama.cpp: --parallel N); then up to N cuts a bundle's time roughly N-fold.
    extraction_concurrency: int = 1

    # Thinking is off by default; enable per request where it earns its cost.
    llm_thinking_default: bool = False

    # Services started by infra/setup-services.sh on the Linux laptop
    database_url: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "qa_chunks"

    # Local embeddings (fastembed, CPU). Changing the dense model needs a re-index.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Hits scoring below this after reranking are dropped, so "nothing found" can be answered.
    # None = keep everything. 0.0 sits in the gap seen in evals/results/rerank.json (out-of-scope
    # tops < -5, in-scope tops > 4); re-check on real documents.
    rerank_min_score: float | None = 0.0

    # Langfuse tracing (self-hosted on the laptop). Tracing is on when both keys are set.
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # API auth. Empty string disables the check (local development only).
    # api_key is one shared key (user "api"); api_keys gives each person their own, e.g.
    # "alice:key1,bob:key2", so approvals and the audit log name who acted.
    api_key: str = ""
    api_keys: str = ""
    # accounts: people register with a name and 4-digit PIN (needs the database). With it on,
    # requests need a sign-in even when no API key is configured.
    accounts: bool = False
    # Names (comma separated) of the people who may manage accounts. Their accounts are created
    # from the terminal (python -m qa_api.users create), never through open registration.
    admins: str = ""

    # Mail. "file" writes .eml files (development); "outlook" needs Windows + Outlook.
    # outlook_mode "draft" saves to Drafts for a person to send; "send" sends immediately.
    mail_backend: Literal["file", "outlook"] = "file"
    mail_outbox_dir: str = "outbox"
    outlook_mode: Literal["draft", "send"] = "draft"
    reminder_recipients: str = ""  # comma-separated; who calibration reminders go to


@lru_cache
def get_settings() -> Settings:
    return Settings()
