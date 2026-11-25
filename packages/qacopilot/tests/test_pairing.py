"""Choosing the supplier spec (and revision) that applies to a certificate of analysis."""

from datetime import UTC, date, datetime

import pytest

from qacopilot.compliance import pair_spec
from qacopilot.compliance.pairing import (
    SpecCandidate,
    parse_effective,
    parse_revision,
    similarity,
)
from qacopilot.extraction import CoA

NOW = datetime(2026, 11, 1, tzinfo=UTC)


def spec(
    filename: str,
    product: str = "Duck breast fillet skin on",
    *,
    supplier: str | None = "Example Poultry Ltd",
    code: str | None = "4051234",
    revision: int | None = None,
    effective: str | None = None,
    limits: int = 4,
    added: int = 0,
) -> SpecCandidate:
    return SpecCandidate(
        doc_id=filename,
        filename=filename,
        supplier=supplier,
        product=product,
        product_code=code,
        revision=revision,
        effective=date.fromisoformat(effective) if effective else None,
        micro_limits=limits,
        ingested_at=NOW.replace(day=1 + added),
    )


def coa(product: str | None = "Duck breast fillet", report: str | None = "2026-11-05") -> CoA:
    return CoA(
        product=product,
        batch="B2613-01",
        report_date=date.fromisoformat(report) if report else None,
        results=[],
    )


@pytest.mark.parametrize(
    "text,revision,effective",
    [
        (
            "PRODUCT SPECIFICATION (Revision 2)\nEffective: 2026-10-01. Revision 2 supersedes",
            2,
            date(2026, 10, 1),
        ),
        ("Spec  Rev. 3   Issue date: 4 March 2026", 3, date(2026, 3, 4)),
        ("Version: 12\nValid from 01/02/2026", 12, date(2026, 2, 1)),
        ("Nothing about revisions here", None, None),
    ],
)
def test_revision_and_effective_date_are_read_from_the_text(
    text: str, revision: int | None, effective: date | None
) -> None:
    assert parse_revision(text) == revision
    assert parse_effective(text) == effective


def test_a_report_product_with_an_ocr_run_on_label_still_matches() -> None:
    run_on = "Duck breast fillet Batch: B2701-02"
    assert similarity(run_on, "Duck breast fillet skin on") == pytest.approx(0.6)
    p = pair_spec(coa(run_on), [spec("duck.pdf")])
    assert p.status == "paired"


def test_similarity_needs_every_word_of_the_report_product() -> None:
    assert similarity("Duck breast fillet", "Duck breast fillet skin on") == pytest.approx(0.6)
    assert similarity("Duck breast fillet", "Duck breast fillets") == 1.0  # plural
    assert similarity("Duck breast fillet", "Duck leg") == 0.0
    assert similarity("Chicken thigh", "Chicken thigh fillet") > 0


def test_single_spec_is_paired() -> None:
    p = pair_spec(coa(), [spec("v1.pdf"), spec("gravy.pdf", "Beef gravy", code=None)])
    assert p.status == "paired" and p.spec is not None and p.spec.filename == "v1.pdf"


def test_the_revision_in_force_on_the_report_date_is_used() -> None:
    specs = [
        spec("v1.pdf", added=1),
        spec("v2.pdf", revision=2, effective="2026-10-01", limits=0, added=2),
    ]
    after = pair_spec(coa(report="2026-11-05"), specs)
    assert after.spec is not None and after.spec.filename == "v2.pdf" and after.notes == []

    before = pair_spec(coa(report="2026-09-15"), specs)
    assert before.spec is not None and before.spec.filename == "v1.pdf"
    assert any("before revision 2 took effect" in n for n in before.notes)


def test_a_report_with_no_date_uses_the_newest_and_says_so() -> None:
    specs = [spec("v1.pdf"), spec("v2.pdf", revision=2, effective="2026-10-01")]
    p = pair_spec(coa(report=None), specs)
    assert p.spec is not None and p.spec.filename == "v2.pdf"
    assert any("no date" in n for n in p.notes)


def test_a_report_older_than_every_spec_is_flagged() -> None:
    p = pair_spec(coa(report="2025-01-01"), [spec("v2.pdf", revision=2, effective="2026-10-01")])
    assert p.status == "paired" and any("before the earliest spec" in n for n in p.notes)


def test_no_matching_product_means_no_spec() -> None:
    p = pair_spec(coa("Chicken thigh fillet"), [spec("duck.pdf")])
    assert p.status == "no_spec" and "Chicken thigh fillet" in p.notes[0]
    assert pair_spec(coa(None), [spec("duck.pdf")]).status == "no_spec"


def test_equally_good_but_different_specs_are_ambiguous_not_guessed() -> None:
    sauce = spec("sauce.pdf", "Duck breast fillet in orange sauce", code="4059999")
    skin_on = spec("skin_on.pdf")
    p = pair_spec(coa(), [skin_on, sauce])
    assert p.status == "ambiguous" and p.spec is None
    assert {c.filename for c in p.candidates} == {"skin_on.pdf", "sauce.pdf"}

    other_supplier = spec("other.pdf", supplier="Another Poultry Ltd")
    assert pair_spec(coa(), [skin_on, other_supplier]).status == "ambiguous"


def test_a_closer_product_name_beats_a_looser_one() -> None:
    exact = spec("exact.pdf", "Duck breast fillet", code="4050000")
    p = pair_spec(coa(), [spec("skin_on.pdf"), exact])
    assert p.status == "paired" and p.spec is not None and p.spec.filename == "exact.pdf"


def test_same_revision_filed_twice_uses_the_newest_and_says_so() -> None:
    a = spec("a.pdf", revision=2, effective="2026-10-01", added=1)
    b = spec("b.pdf", revision=2, effective="2026-10-01", added=2)
    p = pair_spec(coa(), [a, b])
    assert p.spec is not None and p.spec.filename == "b.pdf"
    assert any("same spec revision" in n for n in p.notes)
