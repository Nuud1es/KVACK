"""Manage accounts from the terminal: ``python -m qa_api.users create "Joe Rudling"``.

This is how an administrator's own account is made (the web form refuses the names listed in
``QACOPILOT_ADMINS``, so nobody else can claim them), and the way back in if the administrator
forgets their PIN. The PIN is typed at a hidden prompt (or passed in the QA_PIN environment
variable when there is no terminal), never as an argument, so it is not in the shell history.
"""

import argparse
import asyncio
import getpass
import os
import sys

from qacopilot import auth
from qacopilot.config import get_settings
from qacopilot.repositories import UserRepository, make_engine


def _ask_pin() -> str:
    # Without a terminal (a tool that runs commands with no keyboard) the PIN can come from the
    # environment instead: QA_PIN=1234 python -m qa_api.users create "Name". Prefer the prompt.
    if (given := os.environ.get("QA_PIN")) is not None:
        pin = given
    else:
        if not sys.stdin.isatty():
            raise SystemExit(
                "No terminal to ask for the PIN. Run this in a normal terminal, or give it as "
                'QA_PIN=1234 (read -s -p "PIN: " P; QA_PIN=$P python -m qa_api.users ...).'
            )
        pin = getpass.getpass("PIN (4 digits): ")
        if getpass.getpass("Again: ") != pin:
            raise SystemExit("The two PINs differ.")
    try:
        return auth.check_pin(pin)
    except auth.InvalidInput as exc:
        raise SystemExit(str(exc)) from exc


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.database_url:
        print("QACOPILOT_DATABASE_URL is not set (run from qa-copilot/ so .env is found).")
        return 1
    engine = make_engine(settings)
    repo = UserRepository(engine)
    try:
        if args.command == "list":
            for u in await repo.list_users():
                state = "LOCKED" if u.locked else ""
                print(f"{u.id:>4}  {u.name:<30} {u.created_at:%Y-%m-%d %H:%M}  {state}")
            return 0
        try:
            name = auth.clean_name(args.name)
        except auth.InvalidInput as exc:
            print(exc)
            return 1
        if args.command == "create":
            if await repo.register(name, _ask_pin()) is None:
                print(f"{name} already exists. Use reset-pin to change the PIN.")
                return 1
            print(f"Created {name}.")
        else:  # reset-pin
            user_id = await repo.find(name)
            if user_id is None or await repo.set_pin(user_id, _ask_pin()) is None:
                print(f"No account called {name}.")
                return 1
            print(f"New PIN set for {name}; they are signed out everywhere.")
        return 0
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m qa_api.users", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show every account")
    for command in ("create", "reset-pin"):
        sub.add_parser(command).add_argument("name")
    sys.exit(asyncio.run(_run(parser.parse_args())))


if __name__ == "__main__":
    main()
