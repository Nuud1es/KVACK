from qacopilot.config import Settings
from qacopilot.mail.base import (
    DeliveryReceipt,
    EmailMessage,
    Mailer,
    MailerUnavailable,
)
from qacopilot.mail.file import FileMailer
from qacopilot.mail.outlook import OutlookMailer


def make_mailer(settings: Settings | None = None) -> Mailer:
    s = settings or Settings()
    if s.mail_backend == "outlook":
        return OutlookMailer(s.outlook_mode)
    return FileMailer(s.mail_outbox_dir)


__all__ = [
    "DeliveryReceipt",
    "EmailMessage",
    "FileMailer",
    "Mailer",
    "MailerUnavailable",
    "OutlookMailer",
    "make_mailer",
]
