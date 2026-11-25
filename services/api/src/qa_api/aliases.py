"""Review and apply name aliases from the terminal.

    python -m qa_api.aliases suggest [--kind customer|supplier|person|all] [--out FILE]
    python -m qa_api.aliases apply FILE          # only rows whose approve column is yes
    python -m qa_api.aliases list
    python -m qa_api.aliases add customer "Sainsbury" "Sainsbury's"
    python -m qa_api.aliases remove customer "Sainsbury"

``suggest`` writes a CSV and changes nothing. Open it, put ``yes`` in the approve column of the rows
that really are the same name (change ``canonical`` if you prefer another spelling), save it, then
run ``apply``. The file holds your customer names: keep it out of git (``logs/`` is ignored).
"""

import argparse
import asyncio
import csv
import sys
from pathlib import Path

from qacopilot.aliases import Proposal, suggest, suggest_people
from qacopilot.config import get_settings
from qacopilot.repositories import DocumentRepository, make_engine

COLUMNS = ["approve", "kind", "raw", "canonical", "specs", "confidence", "reason"]
APPROVED = {"y", "yes", "1", "true", "x"}


async def _proposals(repo: DocumentRepository, kind: str) -> list[Proposal]:
    kinds = ["customer", "supplier", "person"] if kind == "all" else [kind]
    out: list[Proposal] = []
    for k in kinds:
        counts = await repo.name_counts(k)
        out += suggest_people(counts) if k == "person" else suggest(counts, k)
    return out


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.database_url:
        print("QACOPILOT_DATABASE_URL is not set (run from qa-copilot/ so .env is found).")
        return 1
    engine = make_engine(settings)
    repo = DocumentRepository(engine)
    try:
        if args.command == "suggest":
            rows = await _proposals(repo, args.kind)
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.writer(fh)
                writer.writerow(COLUMNS)
                for p in rows:
                    writer.writerow(
                        ["", p.kind, p.raw, p.canonical, p.specs, p.confidence, p.reason]
                    )
            safe = sum(p.confidence == "safe" for p in rows)
            print(f"{len(rows)} suggestions ({safe} safe, {len(rows) - safe} to check) -> {path}")
            print("Put yes in the approve column of the rows you accept, then run apply.")
        elif args.command == "apply":
            applied = 0
            with Path(args.file).open(newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    if (row.get("approve") or "").strip().lower() in APPROVED:
                        await repo.set_alias(
                            row["kind"].strip(), row["raw"], row["canonical"], by="alias file"
                        )
                        applied += 1
            print(f"Applied {applied} aliases.")
        elif args.command == "list":
            for kind, raw, canonical in await repo.list_aliases():
                print(f"{kind:<9} {raw:<40} -> {canonical}")
        elif args.command == "add":
            await repo.set_alias(args.kind, args.raw, args.canonical, by="terminal")
            print(f"{args.kind}: {args.raw} -> {args.canonical}")
        elif args.command == "remove":
            print("Removed." if await repo.remove_alias(args.kind, args.raw) else "No such alias.")
        return 0
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m qa_api.aliases", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("suggest")
    s.add_argument("--kind", choices=["customer", "supplier", "person", "all"], default="customer")
    s.add_argument("--out", default="logs/aliases_proposed.csv")
    sub.add_parser("apply").add_argument("file")
    sub.add_parser("list")
    a = sub.add_parser("add")
    for name in ("kind", "raw", "canonical"):
        a.add_argument(name)
    r = sub.add_parser("remove")
    r.add_argument("kind")
    r.add_argument("raw")
    sys.exit(asyncio.run(_run(parser.parse_args())))


if __name__ == "__main__":
    main()
