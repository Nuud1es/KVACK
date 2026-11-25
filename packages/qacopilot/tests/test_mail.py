from collections.abc import Iterator
from contextlib import contextmanager
from email import message_from_bytes
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from qacopilot.config import Settings
from qacopilot.mail import (
    EmailMessage,
    FileMailer,
    MailerUnavailable,
    OutlookMailer,
    make_mailer,
)

MSG = EmailMessage(
    to=["qa@example.com"], cc=["boss@example.com"], subject="Reminder", body="Hello\n"
)
KEY = "run1-abc12345"


async def test_file_mailer_writes_a_parsable_eml(tmp_path: Path) -> None:
    receipt = await FileMailer(tmp_path).deliver(MSG, key=KEY)
    assert receipt.outcome == "file" and receipt.reference == str(tmp_path / f"{KEY}.eml")
    parsed = message_from_bytes(Path(receipt.reference).read_bytes())
    assert parsed["To"] == "qa@example.com" and parsed["Cc"] == "boss@example.com"
    assert parsed["Subject"] == "Reminder" and parsed["X-QACopilot-Key"] == KEY
    assert "Hello" in parsed.get_payload()


async def test_file_mailer_is_idempotent_per_key(tmp_path: Path) -> None:
    mailer = FileMailer(tmp_path)
    await mailer.deliver(MSG, key=KEY)
    changed = MSG.model_copy(update={"body": "different"})
    await mailer.deliver(changed, key=KEY)  # same key: nothing is overwritten or duplicated
    assert [p.name for p in tmp_path.iterdir()] == [f"{KEY}.eml"]
    assert "different" not in (tmp_path / f"{KEY}.eml").read_text()


@pytest.mark.parametrize("key", ["../evil", "a/b/c/d/e/f", "short", "x" * 200, ""])
async def test_unsafe_keys_are_rejected(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError, match="idempotency key"):
        await FileMailer(tmp_path).deliver(MSG, key=key)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "bad",
    [
        {"to": ["not-an-address"]},
        {"to": []},
        {"to": ["a@b.co"], "subject": "x\nBcc: attacker@evil.com"},
        {"to": ["a@b.co, c@d.co"]},
        {"to": ["a@b.co"], "body": ""},
    ],
)
def test_message_validation_blocks_bad_input(bad: dict[str, Any]) -> None:
    fields = {"to": ["a@b.co"], "subject": "s", "body": "b", **bad}
    with pytest.raises(ValidationError):
        EmailMessage(**fields)


class FakeMailItem:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.EntryID = "ENTRY-1"
        self.UserProperties = self

    def Add(self, name: str, kind: int) -> "FakeMailItem":  # noqa: N802
        self.calls.append(f"prop:{name}")
        return self

    Value: Any = None

    def Save(self) -> None:  # noqa: N802
        self.calls.append("save")

    def Send(self) -> None:  # noqa: N802
        self.calls.append("send")


class FakeOutlook:
    def __init__(self) -> None:
        self.item = FakeMailItem()

    def CreateItem(self, kind: int) -> FakeMailItem:  # noqa: N802
        assert kind == 0
        return self.item


def _session(app: FakeOutlook, log: list[str]) -> Any:
    @contextmanager
    def session() -> Iterator[FakeOutlook]:
        log.append("open")
        yield app
        log.append("close")

    return session


async def test_outlook_draft_mode_saves_and_never_sends() -> None:
    app, log = FakeOutlook(), []
    receipt = await OutlookMailer("draft", _session(app, log)).deliver(MSG, key=KEY)
    assert receipt.outcome == "draft" and receipt.reference == "ENTRY-1"
    assert "save" in app.item.calls and "send" not in app.item.calls
    assert log == ["open", "close"]
    assert (app.item.To, app.item.CC, app.item.Subject) == (  # type: ignore[attr-defined]
        "qa@example.com",
        "boss@example.com",
        "Reminder",
    )


async def test_outlook_send_mode_sends() -> None:
    app = FakeOutlook()
    receipt = await OutlookMailer("send", _session(app, [])).deliver(MSG, key=KEY)
    assert receipt.outcome == "sent" and "send" in app.item.calls


async def test_outlook_unavailable_off_windows_is_a_clear_error() -> None:
    with pytest.raises(MailerUnavailable, match="pywin32"):
        await OutlookMailer().deliver(MSG, key=KEY)


def test_factory_picks_the_backend(tmp_path: Path) -> None:
    assert isinstance(
        make_mailer(Settings(mail_backend="file", mail_outbox_dir=str(tmp_path))), FileMailer
    )
    assert isinstance(make_mailer(Settings(mail_backend="outlook")), OutlookMailer)
