"""Compare a certificate of analysis against a supplier spec. Pure code, no LLM."""

import re
from collections.abc import Iterable

from pydantic import BaseModel

from qacopilot.compliance.limits import Verdict, evaluate_result, parse_limit
from qacopilot.extraction.schemas import CoA, SupplierSpec

# Names that mean the same test on different documents (compared after normalising).
_ALIASES = {
    "tvc": "totalviablecount",
    "aerobicplatecount": "totalviablecount",
    "apc": "totalviablecount",
    "totalplatecount": "totalviablecount",
    "aerobiccolonycount": "totalviablecount",
    "entero": "enterobacteriaceae",
    "enterobacteria": "enterobacteriaceae",
    "listeria": "listeriamonocytogenes",
    "lmonocytogenes": "listeriamonocytogenes",
    "salmonellaspp": "salmonella",
    "ecoli": "escherichiacoli",
    "staphaureus": "staphylococcusaureus",
    "saureus": "staphylococcusaureus",
}


def normalize_test_name(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return _ALIASES.get(key, key)


class CheckRow(BaseModel):
    test: str
    result: str
    limit: str
    limit_source: str  # "spec", "coa" (limit printed on the report) or "none"
    verdict: Verdict
    reason: str


def overall(rows: Iterable[CheckRow]) -> Verdict:
    verdicts = {r.verdict for r in rows}
    if Verdict.FAIL in verdicts:
        return Verdict.FAIL
    if Verdict.REVIEW in verdicts:
        return Verdict.REVIEW
    return Verdict.PASS


def cap_for_unverified_source(verdict: Verdict) -> Verdict:
    """A result built on text read by OCR can fail, but it cannot be a clean pass.

    OCR can drop a decimal point or a digit (4.1E+03 read as 41E+03), and a wrong number would
    otherwise be judged with full confidence. A person confirms the figures against the original.
    """
    return Verdict.REVIEW if verdict is Verdict.PASS else verdict


def check_coa_against_spec(coa: CoA, spec: SupplierSpec) -> list[CheckRow]:
    """One row per CoA result, plus a REVIEW row for each spec limit the CoA does not cover.

    The spec's limit wins. If the spec has none for a test, the limit printed on the CoA is used
    and marked as such; with neither, the row is REVIEW.
    """
    spec_limits = {normalize_test_name(m.test): m.limit for m in spec.micro_limits}
    rows: list[CheckRow] = []
    seen: set[str] = set()
    for res in coa.results:
        key = normalize_test_name(res.test)
        seen.add(key)
        if key in spec_limits:
            limit_text, source = spec_limits[key], "spec"
        elif res.limit:
            limit_text, source = res.limit, "coa"
        else:
            rows.append(
                CheckRow(
                    test=res.test,
                    result=res.result,
                    limit="",
                    limit_source="none",
                    verdict=Verdict.REVIEW,
                    reason="no limit in the spec or on the report",
                )  # fmt: skip
            )
            continue
        limit = parse_limit(limit_text)
        if limit is None:
            verdict, reason = Verdict.REVIEW, f"cannot read limit {limit_text!r}"
        else:
            verdict, reason = evaluate_result(res.result, limit)
        rows.append(
            CheckRow(
                test=res.test,
                result=res.result,
                limit=limit_text,
                limit_source=source,
                verdict=verdict,
                reason=reason,
            )  # fmt: skip
        )
    for m in spec.micro_limits:
        if normalize_test_name(m.test) not in seen:
            rows.append(
                CheckRow(
                    test=m.test,
                    result="",
                    limit=m.limit,
                    limit_source="spec",
                    verdict=Verdict.REVIEW,
                    reason="in the spec but not tested on this report",
                )  # fmt: skip
            )
    return rows
