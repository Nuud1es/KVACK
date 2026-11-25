"""Accounts with a name and a short PIN: the input rules, hashing and session tokens (no I/O).

A 4-digit PIN has only 10,000 values, so the hash alone protects nothing against someone who
gets the database. What protects it is the lock-out after wrong tries (see ``MAX_FAILED``) and
keeping the app off the open internet. Say so wherever this is exposed.
"""

import hashlib
import hmac
import re
import secrets

MAX_FAILED = 5  # wrong PINs in a row before the account is locked
LOCK_MINUTES = 5
SESSION_HOURS = 12
NAME_MIN, NAME_MAX = 2, 40
RESERVED_NAMES = frozenset({"api", "dev", "admin", "system"})

_PIN = re.compile(r"[0-9]{4}")
_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}
# Checked when the name is unknown, so a wrong name takes as long as a wrong PIN.
_DUMMY = "scrypt$" + "00" * 16 + "$" + "00" * 32


class InvalidInput(ValueError):
    """A name or PIN the rules do not allow; the message is safe to show the person."""


def clean_name(raw: str) -> str:
    """Tidy a display name: collapse spaces, then allow letters, spaces and . ' - only."""
    name = " ".join(raw.split())
    if not NAME_MIN <= len(name) <= NAME_MAX:
        raise InvalidInput(f"Use a name of {NAME_MIN} to {NAME_MAX} characters.")
    if not name[0].isalpha() or not all(c.isalpha() or c in " .'-" for c in name):
        raise InvalidInput("A name can use letters, spaces and . ' - only.")
    return name


def name_key(name: str) -> str:
    """What makes two names the same person: case does not count."""
    return " ".join(name.split()).casefold()


def check_pin(pin: str) -> str:
    if not _PIN.fullmatch(pin):
        raise InvalidInput("The PIN must be exactly 4 digits.")
    return pin


def hash_pin(pin: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(pin.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_pin(pin: str, stored: str | None) -> bool:
    """True if ``pin`` matches ``stored``; with ``None`` it spends the same time and is False."""
    _, salt_hex, digest_hex = (stored or _DUMMY).split("$")
    digest = hashlib.scrypt(
        pin.encode(),
        salt=bytes.fromhex(salt_hex),
        **_SCRYPT,
    )
    return hmac.compare_digest(digest, bytes.fromhex(digest_hex)) and stored is not None


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_digest(token: str) -> str:
    """Only this is stored, so a database leak does not hand out working sessions."""
    return hashlib.sha256(token.encode()).hexdigest()


def admin_keys(raw: str) -> frozenset[str]:
    """The configured admin names (``QACOPILOT_ADMINS="Joe Rudling, Sam Jones"``), as match keys."""
    return frozenset(name_key(n) for n in raw.split(",") if n.strip())
