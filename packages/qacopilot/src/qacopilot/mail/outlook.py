"""Outlook (COM) mailer for the Windows production machine.

Follows the toolkit's pattern: ``pythoncom.CoInitialize()`` in the worker thread, then
``Outlook.Application``. Not runnable on Linux; the COM session is injectable so the logic is
tested there, and the real thing is verified on Windows.
"""

import asyncio
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Literal

from qacopilot.mail.base import DeliveryReceipt, EmailMessage, Mailer, MailerUnavailable, check_key

Session = Callable[[], AbstractContextManager[Any]]


@contextmanager
def _com_session() -> Iterator[Any]:
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise MailerUnavailable("Outlook needs Windows with pywin32 installed") from exc
    pythoncom.CoInitialize()
    try:
        yield win32com.client.Dispatch("Outlook.Application")
    finally:
        pythoncom.CoUninitialize()


class OutlookMailer(Mailer):
    """``mode="draft"`` saves to Drafts for a person to send; ``mode="send"`` sends at once."""

    def __init__(self, mode: Literal["draft", "send"] = "draft", session: Session = _com_session):
        self._mode = mode
        self._session = session

    def _deliver(self, message: EmailMessage, key: str) -> DeliveryReceipt:
        with self._session() as outlook:
            mail = outlook.CreateItem(0)  # 0 = olMailItem
            mail.To = "; ".join(message.to)
            if message.cc:
                mail.CC = "; ".join(message.cc)
            mail.Subject = message.subject
            mail.Body = message.body
            # Searchable marker so a retried delivery can be recognised in the mailbox
            mail.UserProperties.Add("QACopilotKey", 1).Value = key
            if self._mode == "send":
                mail.Send()
                return DeliveryReceipt(backend="outlook", outcome="sent", reference=key)
            mail.Save()
            return DeliveryReceipt(backend="outlook", outcome="draft", reference=str(mail.EntryID))

    async def deliver(self, message: EmailMessage, *, key: str) -> DeliveryReceipt:
        # COM objects must live and die on one thread, so each delivery gets its own.
        return await asyncio.to_thread(self._deliver, message, check_key(key))
