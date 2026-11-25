"""Mail interface. The agent depends on ``Mailer``; Outlook (Windows) and files (dev) do it."""

import re
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field, field_validator

_ADDRESS = re.compile(r"^[^@\s<>,;]+@[^@\s<>,;]+\.[^@\s<>,;]+$")
_KEY = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


class EmailMessage(BaseModel):
    to: list[str] = Field(min_length=1)
    cc: list[str] = Field(default_factory=list)
    subject: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1)

    @field_validator("to", "cc")
    @classmethod
    def _addresses(cls, values: list[str]) -> list[str]:
        for v in values:
            if not _ADDRESS.match(v):
                raise ValueError(f"not a valid email address: {v!r}")
        return values

    @field_validator("subject")
    @classmethod
    def _no_header_injection(cls, value: str) -> str:
        if "\r" in value or "\n" in value:
            raise ValueError("subject must be a single line")
        return value


class DeliveryReceipt(BaseModel):
    backend: str
    outcome: Literal["file", "draft", "sent"]
    reference: str  # file path, or the Outlook EntryID


class MailerUnavailable(RuntimeError):
    """The backend cannot run here (e.g. Outlook COM on Linux)."""


def check_key(key: str) -> str:
    """Idempotency keys become file names, so they must be plain tokens."""
    if not _KEY.match(key):
        raise ValueError("idempotency key must be 8-128 characters of A-Z a-z 0-9 _ -")
    return key


class Mailer(ABC):
    @abstractmethod
    async def deliver(self, message: EmailMessage, *, key: str) -> DeliveryReceipt:
        """Deliver once. Calling again with the same ``key`` must not deliver a second time."""
