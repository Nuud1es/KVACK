"""Development mailer: writes each message as an .eml file instead of sending it."""

import asyncio
import os
from email.message import EmailMessage as MimeMessage
from pathlib import Path

from qacopilot.mail.base import DeliveryReceipt, EmailMessage, Mailer, check_key


class FileMailer(Mailer):
    def __init__(self, outbox: str | Path) -> None:
        self._outbox = Path(outbox)

    def _write(self, message: EmailMessage, key: str) -> DeliveryReceipt:
        self._outbox.mkdir(parents=True, exist_ok=True)
        path = self._outbox / f"{key}.eml"
        if path.exists():  # same key: already delivered
            return DeliveryReceipt(backend="file", outcome="file", reference=str(path))
        mime = MimeMessage()
        mime["To"] = ", ".join(message.to)
        if message.cc:
            mime["Cc"] = ", ".join(message.cc)
        mime["Subject"] = message.subject
        mime["X-QACopilot-Key"] = key
        mime.set_content(message.body)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(bytes(mime))
        os.replace(tmp, path)  # atomic: a reader never sees a half-written file
        return DeliveryReceipt(backend="file", outcome="file", reference=str(path))

    async def deliver(self, message: EmailMessage, *, key: str) -> DeliveryReceipt:
        return await asyncio.to_thread(self._write, message, check_key(key))
