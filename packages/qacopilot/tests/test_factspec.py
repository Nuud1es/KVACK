"""Fact Spec packaging: read in code at upload, counted in SQL, answered without search."""

from collections.abc import AsyncIterator
from datetime import date

import pytest
from sqlalchemy import text

from qacopilot.ingest import DocType, PageText, ParsedDocument, classify_text
from qacopilot.ingest.factspec import is_fact_spec, parse_fact_spec
from qacopilot.rag.packaging import answer_from_packaging
from qacopilot.repositories import DocumentRepository
from qacopilot.repositories.testing import DatabaseUnavailable, temporary_repository


def spec_text(
    nav: str = "4056877", version: int = 5, rows: str = "", label: str = "NAV CODE"
) -> str:
    """One page of a Fact Spec as the PDF text comes out: a table cell per line."""
    return f"""GRESSINGHAM FOODS
{label}
{nav}
PLU
6877
VERSION
{version}
06/03/2025
BRAND
Gressingham
SITE
Redgrave
SEASON
PRO
PRODUCT
Fresh Gressingham Duck Breast 170g x6
TYPE
Fresh
CUSTOMER
Morrisons
ALLERGENS
None
STATUS
ACTIVE
Personal Protective Equipment (PPE)
Page 1 of 2
PRODUCT DESCRIPTION
RAW MATERIAL
1
Fresh Duck Breast
Gressingham
Code
QTY
PACKAGING
Supplier
Dimension
Weight
{rows}
PROCESS STEPS:
Ducks are killed.
"""


ROWS = """PRIMARY PACKAGING
7904224
1
Pad - SuperLite 125 x 75 Black
McAirlaids
125x75mm
1.2g
79041
68
1
Blue Gas Flush Bag
Krehalon UK
530x1000 70µ
71.9g
SECONDARY PACKAGING
-
1
Gressingham Orange Crate
Gressingham
600x400x190mm
1650g
7903176
1
Pack Tare
71.9g
OCM / Case Tare
1727.3g
"""


def test_header_and_packaging_rows_are_read() -> None:
    spec = parse_fact_spec([spec_text(rows=ROWS)])
    assert (spec.nav_code, spec.plu, spec.version, spec.version_date) == (
        "4056877",
        "6877",
        5,
        date(2025, 3, 6),
    )
    assert (spec.customer, spec.status, spec.site) == ("Morrisons", "ACTIVE", "Redgrave")
    assert [(r.section, r.code, r.qty) for r in spec.packaging] == [
        ("PRIMARY", "7904224", "1"),
        ("PRIMARY", "7904168", "1"),  # the code the PDF wrapped over two lines
        ("SECONDARY", "7903176", "1"),  # the crate without a code is not a coded packaging row
    ]
    pad, bag, tags = spec.packaging
    assert (pad.description, pad.supplier, pad.dimension, pad.weight) == (
        "Pad - SuperLite 125 x 75 Black",
        "McAirlaids",
        "125x75mm",
        "1.2g",
    )
    assert bag.supplier == "Krehalon UK" and bag.weight == "71.9g"
    # the totals under the table are not part of the row
    assert tags.description == "" and tags.cells == ["7903176", "1"]


def test_older_layouts_and_misspellings_still_read() -> None:
    old = spec_text(label="EXCHEQUER", rows=ROWS).replace("VERSION", "VER5SION")
    assert is_fact_spec(old)
    assert parse_fact_spec([old]).nav_code == "4056877"
    assert parse_fact_spec([old]).version == 5


def test_other_documents_are_not_fact_specs() -> None:
    assert not is_fact_spec("CERTIFICATE OF ANALYSIS\nProduct: Duck\nBatch: B1\n")
    assert parse_fact_spec([]).packaging == []


def test_a_fact_spec_is_classified_as_one_and_a_supplier_spec_is_not() -> None:
    fact = classify_text(spec_text(rows=ROWS), "Fact Spec - 4056877 - Duck.pdf")
    assert fact.doc_type is DocType.FACT_SPEC
    supplier = classify_text(
        "PRODUCT SPECIFICATION\nSupplier: Acme\nProduct: Gravy\nShelf life: 12 months\n",
        "gravy.pdf",
    )
    assert supplier.doc_type is DocType.SUPPLIER_SPEC


@pytest.fixture
async def repo() -> AsyncIterator[DocumentRepository]:
    try:
        async with temporary_repository() as r:
            yield r
    except DatabaseUnavailable as exc:
        pytest.skip(str(exc))


def _rows(*codes: str, supplier: str = "Acme Packaging") -> str:
    body = "".join(f"{c}\n1\nItem {c}\n{supplier}\n10x10mm\n1g\n" for c in codes)
    return "PRIMARY PACKAGING\n" + body


async def _add(repo: DocumentRepository, doc_id: str, nav: str, version: int, *codes: str) -> None:
    doc = ParsedDocument(
        source=f"{doc_id}.pdf",
        pages=[PageText(page=1, text=spec_text(nav, version, _rows(*codes)), method="text")],
    )
    await repo.save_document(doc_id, f"Fact Spec - {nav}.pdf", DocType.FACT_SPEC, doc, 1)


async def test_tables_are_filled_replaced_and_removed_with_the_type(
    repo: DocumentRepository,
) -> None:
    await _add(repo, "a", "4000001", 1, "7900001", "7900002")
    assert [r.code for r in await repo.packaging_of_spec("a")] == ["7900001", "7900002"]
    await _add(repo, "a", "4000001", 1, "7900009")  # re-upload of the same document
    assert [r.code for r in await repo.packaging_of_spec("a")] == ["7900009"]
    await repo.set_document_type("a", DocType.OTHER, 1)
    assert await repo.packaging_of_spec("a") == [] and await repo.fact_spec_count() == 0


async def test_only_the_latest_version_of_a_product_counts(repo: DocumentRepository) -> None:
    await _add(repo, "v1", "4000001", 1, "7900001", "7900002")
    await _add(repo, "v2", "4000001", 2, "7900002", "7900003")  # newer: 7900001 is gone
    await _add(repo, "other", "4000002", 1, "7900002")
    assert await repo.fact_spec_count() == 2
    usage = {u.code: u.specs for u in await repo.packaging_usage(limit=10)}
    assert usage == {"7900002": 2, "7900003": 1}
    assert [s.doc_id for s in await repo.specs_using_code("7900001")] == []


async def test_a_blank_description_borrows_the_usual_one(repo: DocumentRepository) -> None:
    await _add(repo, "a", "4000001", 1, "7900001")
    bare = ParsedDocument(
        source="b.pdf",
        pages=[
            PageText(
                page=1,
                text=spec_text("4000002", 1, "PRIMARY PACKAGING\n7900001\n1\n"),
                method="text",
            )
        ],
    )
    await repo.save_document("b", "Fact Spec - 4000002.pdf", DocType.FACT_SPEC, bare, 1)
    (row,) = await repo.packaging_of_spec("b")
    assert row.description == "Item 7900001" and row.supplier == "Acme Packaging"


async def test_packaging_questions_are_answered_from_the_table(repo: DocumentRepository) -> None:
    await _add(repo, "a", "4000001", 1, "7900001", "7900002")
    await _add(repo, "b", "4000002", 1, "7900002")
    await _add(repo, "c", "4000003", 1, "7900002", "7900003")

    ranking = await answer_from_packaging("What is the most used packaging code?", repo)
    assert ranking is not None
    done = ranking[-1]
    assert done.via == "packaging" and "7900002" in done.answer.splitlines()[1]  # type: ignore[union-attr]
    assert "used by 3 specs" in done.answer  # type: ignore[union-attr]

    one = await answer_from_packaging("which specs use 7900001?", repo)
    assert one is not None and "Used by 1 of 3 current specs" in one[-1].answer  # type: ignore[union-attr]

    nav = await answer_from_packaging("what packaging does 4000003 use", repo)
    assert nav is not None
    assert "7900002" in nav[-1].answer and "7900003" in nav[-1].answer  # type: ignore[union-attr]

    missing = await answer_from_packaging("tell me about packaging code 7909999", repo)
    assert missing is not None and "does not appear" in missing[-1].answer  # type: ignore[union-attr]

    assert await answer_from_packaging("what is our shelf life policy?", repo) is None
    assert await answer_from_packaging("who is the best packaging supplier", repo) is None


async def test_without_fact_specs_the_question_goes_to_search(repo: DocumentRepository) -> None:
    assert await answer_from_packaging("most used packaging code", repo) is None


# --- header, counts and matching a question to one spec ---------------------------------------

from qacopilot.rag.answer import DoneEvent, TokenEvent  # noqa: E402
from qacopilot.rag.factspec_lookup import (  # noqa: E402
    answer_from_factspec,
    match_specs,
    resolve_spec,
    scope_for_question,
    with_note,
)
from qacopilot.repositories import SpecIndexEntry  # noqa: E402


def _entry(doc: str, nav: str, product: str, customer: str = "Acme") -> SpecIndexEntry:
    return SpecIndexEntry(
        doc_id=doc, filename=f"{doc}.pdf", nav_codes=[nav], product=product, customer=customer
    )


INDEX = [
    _entry("a", "4000001", "Fresh Gressingham Duck Breast 170g x6", "Morrisons"),
    _entry("b", "4000002", "Fresh Gressingham Duck Breast 170g x6 Confit", "Tesco"),
    _entry("c", "4000003", "Roast Turkey Crown 1.2kg", "Tesco"),
    _entry("d", "4000004", "Roast Turkey Crown 1.2kg", "Aldi"),
]


def test_a_nav_code_or_a_product_name_picks_one_spec() -> None:
    assert resolve_spec("what is the customer for 4000003", INDEX).doc_id == "c"  # type: ignore[union-attr]
    # the longer name wins when the question contains all of it
    assert (
        resolve_spec("storage of Fresh Gressingham Duck Breast 170g x6 Confit", INDEX).doc_id == "b"
    )  # type: ignore[union-attr]
    assert resolve_spec("storage of Fresh Gressingham Duck Breast 170g x6", INDEX).doc_id == "a"  # type: ignore[union-attr]
    assert resolve_spec("what is the weather", INDEX) is None
    assert resolve_spec("duck breast", INDEX) is None  # too little of any name


def test_the_same_product_for_two_customers_is_ambiguous_not_guessed() -> None:
    found = match_specs("process steps for Roast Turkey Crown 1.2kg", INDEX)
    assert {s.doc_id for s in found} == {"c", "d"}
    assert resolve_spec("process steps for Roast Turkey Crown 1.2kg", INDEX) is None


async def test_scope_note_and_rewrite_for_search(repo: DocumentRepository) -> None:
    await _add(repo, "a", "4000001", 1, "7900001")
    await _add(repo, "b", "4000002", 1, "7900001")
    scope, note, rewritten = await scope_for_question("storage temperature for 4000001", repo)
    assert scope == "a" and note is None
    assert "4000001" not in rewritten and "Fresh Gressingham Duck Breast" in rewritten
    # two specs with the same product name: no scope, and a note that says so
    scope, note, _ = await scope_for_question(
        "process steps for Fresh Gressingham Duck Breast 170g x6", repo
    )
    assert scope is None and note is not None and "2 current specs have this product name" in note


async def test_a_note_is_added_to_an_answer_but_not_to_a_refusal() -> None:
    async def answered():  # type: ignore[no-untyped-def]
        yield TokenEvent(text="x")
        yield DoneEvent(answer="x", answered=True, cited=[1])

    async def refused():  # type: ignore[no-untyped-def]
        yield DoneEvent(answer="", answered=False, cited=[])

    out = [e async for e in with_note(answered(), "Note: several")]
    assert out[-1].answer.endswith("Note: several")  # type: ignore[union-attr]
    assert any(isinstance(e, TokenEvent) and "Note: several" in e.text for e in out)
    assert [e async for e in with_note(refused(), "Note")][-1].answer == ""  # type: ignore[union-attr]


async def test_header_fields_and_counts_come_from_the_table(repo: DocumentRepository) -> None:
    await _add(repo, "a", "4000001", 3, "7900001")
    await _add(repo, "b", "4000002", 1, "7900001")

    customer = await answer_from_factspec("who is the customer for 4000001", repo)
    assert customer is not None and "Morrisons" in customer[-1].answer  # type: ignore[union-attr]
    assert customer[-1].via == "fact_spec"  # type: ignore[union-attr]

    version = await answer_from_factspec("what version is spec 4000001", repo)
    assert version is not None and version[-1].answer.startswith("Version 3")  # type: ignore[union-attr]

    status = await answer_from_factspec("is 4000001 active", repo)
    assert status is not None and "ACTIVE" in status[-1].answer  # type: ignore[union-attr]

    count = await answer_from_factspec("how many specs are there", repo)
    assert count is not None and "There are 2 current Fact Specs" in count[-1].answer  # type: ignore[union-attr]
    morrisons = await answer_from_factspec("how many specs are for Morrisons", repo)
    assert morrisons is not None and morrisons[-1].answer.startswith("2 of the 2")  # type: ignore[union-attr]

    summary = await answer_from_factspec("tell me about spec 4000002", repo)
    assert summary is not None and "Customer: Morrisons" in summary[-1].answer  # type: ignore[union-attr]

    unknown = await answer_from_factspec("who is the customer for 4999999", repo)
    assert (
        unknown is not None
        and "No current Fact Spec has the NAV code 4999999" in unknown[-1].answer
    )  # type: ignore[union-attr]
    assert await answer_from_factspec("what is the weather", repo) is None


async def test_packaging_totals_and_supplier_counts(repo: DocumentRepository) -> None:
    await _add(repo, "a", "4000001", 1, "7900001", "7900002")
    await _add(repo, "b", "4000002", 1, "7900002")

    codes = await answer_from_packaging("how many different packaging codes are there", repo)
    assert codes is not None and codes[-1].answer.startswith("2 different packaging codes")  # type: ignore[union-attr]
    typo = await answer_from_packaging("how much packaging suppliers do we have", repo)
    assert typo is not None and "1 different packaging suppliers" in typo[-1].answer  # type: ignore[union-attr]
    share = await answer_from_packaging("how many specs use packaging from Acme Packaging", repo)
    assert share is not None and share[-1].answer.startswith("2 of 2 current specs")  # type: ignore[union-attr]
    top = await answer_from_packaging("which supplier supplies the most packaging", repo)
    assert top is not None and "1. Acme Packaging: 2 specs, 2 codes" in top[-1].answer  # type: ignore[union-attr]


def test_the_product_group_is_taken_from_how_the_question_is_put() -> None:
    from qacopilot.rag.packaging import product_words

    assert product_words("what are the packaging codes used for whole duck") == ["whole", "duck"]
    assert product_words("what packaging does whole duck use") == ["whole", "duck"]
    assert product_words("which packaging is used on duck breast products?") == ["duck", "breast"]
    assert product_words("packaging codes for the turkey specs") == ["turkey"]
    assert product_words("what is the most used packaging code") == []
    assert product_words("top 3 packaging codes") == []


async def test_a_product_group_gets_its_own_codes(repo: DocumentRepository) -> None:
    async def add(doc_id: str, nav: str, product: str, *codes: str) -> None:
        text = spec_text(nav, 1, _rows(*codes)).replace(
            "Fresh Gressingham Duck Breast 170g x6", product
        )
        doc = ParsedDocument(
            source=f"{doc_id}.pdf", pages=[PageText(page=1, text=text, method="text")]
        )
        await repo.save_document(doc_id, f"Fact Spec - {nav}.pdf", DocType.FACT_SPEC, doc, 1)

    await add("a", "4000001", "Fresh Whole Duck 2kg", "7900001", "7900002")
    await add("b", "4000002", "Frozen Whole Duck 1.8kg", "7900001", "7900003")
    await add("c", "4000003", "Roast Turkey Crown", "7900009")

    found = await answer_from_packaging("what are the packaging codes used for whole duck", repo)
    assert found is not None
    answer = found[-1].answer  # type: ignore[union-attr]
    assert 'the 2 current specs matching "whole duck"' in answer
    assert "1. 7900001:" in answer and "used by 2 of 2 specs" in answer
    assert "7900002" in answer and "7900003" in answer
    assert "7900009" not in answer  # the turkey spec is not a whole duck spec
    assert found[-1].via == "packaging"  # type: ignore[union-attr]
    unknown = await answer_from_packaging("packaging codes used for kryptonite", repo)
    assert unknown is not None  # said plainly, not guessed by search
    assert "No current Fact Spec has a product name or customer matching" in unknown[-1].answer  # type: ignore[union-attr]


def test_a_misspelt_packaging_is_still_a_packaging_question() -> None:
    from qacopilot.rag.packaging import fix_typos, is_packaging_question

    for typo in ("packging", "pakaging", "packagin", "packagging", "pacakging"):
        assert is_packaging_question(fix_typos(f"what {typo} codes are used for whole duck"))
    assert fix_typos("what is the cheese packaging") == "what is the cheese packaging"
    assert fix_typos("the package arrived") == "the package arrived"
    assert not is_packaging_question(fix_typos("what is the weather"))
    assert not is_packaging_question(fix_typos("who is the customer for 4056877"))


def test_a_packaging_row_is_read_by_what_its_cells_look_like() -> None:
    """Rows from real specs, where extras shift the columns (cells after the code)."""
    from qacopilot.ingest.factspec import _row

    def read(*cells: str) -> tuple[str, str, str, str]:
        r = _row("PRIMARY", "7900000", list(cells), 1)
        return r.description, r.supplier, r.dimension, r.weight

    # a film thickness and a machine tag between the cells
    assert read(
        "1", "Film BDF28F 650x1450 28Mu Whole", "ULMA LINE", "Sealed Air", "28µ", "3.4g"
    ) == (
        "Film BDF28F 650x1450 28Mu Whole ULMA LINE",
        "Sealed Air",
        "28µ",
        "3.4g",
    )
    # a weight with no unit is a weight, not the supplier
    assert read("1", "Pad Black 110 x 200mm", "Sirane", "110x200mm", "2.6") == (
        "Pad Black 110 x 200mm",
        "Sirane",
        "110x200mm",
        "2.6",
    )
    # tolerances and ranges, whole or split over two cells
    assert (
        read("1", "Film Base", "Multivac UK Ltd", "1/3 footprint, 20mm", "12.8g ±0.5g")[3]
        == "12.8g ±0.5g"
    )
    assert read("1", "Film Base", "Multivac UK Ltd", "1/3 footprint, 20mm", "12.8g", "±0.5g")[
        :2
    ] == (
        "Film Base",
        "Multivac UK Ltd",
    )
    assert read("1", "Bag Clear Shrink", "Bemis Swansea", "250X500", "20 - 25g") == (
        "Bag Clear Shrink",
        "Bemis Swansea",
        "250X500",
        "20 - 25g",
    )
    # a stray extra number after the weight does not replace it
    assert read("1", "Strapping", "Qualitape", "9mm", "1.4g", "7382")[3] == "1.4g"
    # only a quantity: nothing else to read
    assert read("1") == ("", "", "", "")


async def test_a_group_that_matches_nothing_says_so_and_a_customer_is_a_group(
    repo: DocumentRepository,
) -> None:
    await _add(repo, "a", "4000001", 1, "7900001", "7900002")  # customer Morrisons (spec_text)
    none = await answer_from_packaging("what packaging codes are use for unicorn", repo)
    assert none is not None and none[-1].via == "packaging"  # type: ignore[union-attr]
    assert (
        'No current Fact Spec has a product name or customer matching "unicorn"' in none[-1].answer
    )  # type: ignore[union-attr]
    # a customer name works like a product group
    customer = await answer_from_packaging("what packaging do we use for Morrisons", repo)
    assert customer is not None
    assert 'the 1 current specs matching "Morrisons"' in customer[-1].answer  # type: ignore[union-attr]
    assert "7900001" in customer[-1].answer and "7900002" in customer[-1].answer  # type: ignore[union-attr]


async def test_a_count_is_narrowed_to_the_product_group_it_names(repo: DocumentRepository) -> None:
    """ "how many whole duck specs" once answered with every spec in the table."""

    async def add(doc_id: str, nav: str, product: str, *codes: str) -> None:
        text = spec_text(nav, 1, _rows(*codes)).replace(
            "Fresh Gressingham Duck Breast 170g x6", product
        )
        doc = ParsedDocument(
            source=f"{doc_id}.pdf", pages=[PageText(page=1, text=text, method="text")]
        )
        await repo.save_document(doc_id, f"Fact Spec - {nav}.pdf", DocType.FACT_SPEC, doc, 1)

    await add("a", "4000001", "Fresh Whole Duck 2kg", "7900001", "7900002")
    await add("b", "4000002", "Frozen Whole Duck 1.8kg", "7900001", "7900003")
    await add("c", "4000003", "Roast Turkey Crown", "7900009")
    # the header's TYPE of spec b is Frozen (the fixture writes Fresh for all)
    async with repo._sessions.begin() as session:
        await session.execute(text("UPDATE fact_specs SET spec_type = 'Frozen' WHERE doc_id = 'b'"))

    for q in (
        "how many whole duck specs are there",
        "how many specs are for whole duck",
        "how many whole duck products do we have",
        "number of whole duck specs",
        "count the whole duck specs",
    ):
        found = await answer_from_factspec(q, repo)
        assert found is not None and found[-1].answer.startswith("2 of the 3"), q  # type: ignore[union-attr]
        assert 'match "whole duck"' in found[-1].answer  # type: ignore[union-attr]
    # the other conditions still combine with the group
    frozen = await answer_from_factspec("how many frozen whole duck specs", repo)
    assert frozen is not None and frozen[-1].answer.startswith("1 of the 3")  # type: ignore[union-attr]
    # a real group with no frozen spec is "0 of", not "no such product"
    none_frozen = await answer_from_factspec("how many frozen turkey specs", repo)
    assert none_frozen is not None and none_frozen[-1].answer.startswith("0 of the 3")  # type: ignore[union-attr]
    # no qualifier: the total; a qualifier nothing matches: says so, never the total
    total = await answer_from_factspec("how many specs are there", repo)
    assert total is not None and "There are 3 current Fact Specs" in total[-1].answer  # type: ignore[union-attr]
    none = await answer_from_factspec("how many unicorn specs are there", repo)
    assert (
        none is not None and "No current Fact Spec has a product name matching" in none[-1].answer
    )  # type: ignore[union-attr]

    codes = await answer_from_packaging("how many packaging codes for whole duck", repo)
    assert codes is not None
    assert (
        '3 different packaging codes are used by the 2 current specs matching "whole duck"'
        in codes[-1].answer
    )  # type: ignore[union-attr]
    # the packaging totals without a group are unchanged
    every = await answer_from_packaging("how many different packaging codes are there", repo)
    assert every is not None and every[-1].answer.startswith("4 different packaging codes")  # type: ignore[union-attr]


async def test_a_product_group_is_matched_as_a_phrase(repo: DocumentRepository) -> None:
    """ "whole duck" means those words together, not "whole" and "duck" somewhere in the name."""

    async def add(doc_id: str, nav: str, product: str, *codes: str) -> None:
        text = spec_text(nav, 1, _rows(*codes)).replace(
            "Fresh Gressingham Duck Breast 170g x6", product
        )
        doc = ParsedDocument(
            source=f"{doc_id}.pdf", pages=[PageText(page=1, text=text, method="text")]
        )
        await repo.save_document(doc_id, f"Fact Spec - {nav}.pdf", DocType.FACT_SPEC, doc, 1)

    await add("a", "4000001", "Fresh Whole Duck 2kg", "7900001")
    await add("b", "4000002", "Frozen Whole Ducks 1.8kg", "7900001", "7900002")  # plural counts
    await add("c", "4000003", "Whole Aromatic Duck 1.2kg", "7900003")  # not the phrase
    await add("d", "4000004", "Duck Legs Whole Muscle", "7900004")  # words apart, other order
    await add("e", "4000005", "Roast Turkey Crown", "7900009")

    count = await answer_from_factspec("how many whole duck specs are there", repo)
    assert count is not None and count[-1].answer.startswith("2 of the 5")  # type: ignore[union-attr]
    assert "as a phrase" not in count[-1].answer  # type: ignore[union-attr]
    codes = await answer_from_packaging("what are the packaging codes used for whole duck", repo)
    assert codes is not None
    answer = codes[-1].answer  # type: ignore[union-attr]
    assert 'the 2 current specs matching "whole duck"' in answer
    assert "7900001" in answer and "7900002" in answer
    assert "7900003" not in answer and "7900004" not in answer  # words apart: not whole duck
    # a plural in the question finds the singular and the other way round
    assert (await repo.group_mode(["duck", "legs"])) == "phrase"
    assert (await repo.group_mode(["whole", "ducks"])) == "phrase"
    # only when no name has the phrase are "all the words" used, and the answer says so
    reversed_ = await answer_from_factspec("how many duck whole specs are there", repo)
    assert reversed_ is not None and "as a phrase" in reversed_[-1].answer  # type: ignore[union-attr]
    assert await repo.group_mode(["duck", "whole"]) == "words"
    assert await repo.group_mode(["unicorn"]) == "none"


# --- what the spec says: storage, shelf life, pallet, metal detection, weight, history ----------

TEMPLATE = """WEIGHT:
{weight}
Fixed Weight
TEMP:
{temp}
Shelf-life Information:
Max. Life:  {max_life}
Min. life into depot: {min_life}
PLU
Customer Code
PALLET
CONFIGURATION
{labels}
{values}
Metal Stick Sizes
Frequency
Method
Ferrous
Non-Ferrous
Stainless Steel
As per code of practice.
In line Metal Detection
{inline}
X Ray
1.5 mm
2.0 mm
2.0 mm
Pack Tare
20.10g
OCM / Case Tare
1.5kg
Product weight
170g
Version
number
Person
responsible
Page
changed
Amends made
Date
{history}
GRESSINGHAM FOODS
NAV CODE
Something unrelated that follows the table and is not history
"""


def template(**over: str) -> str:
    base = {
        "weight": "170g",
        "temp": "Keep refrigerated 0°C to 3°C",
        "max_life": "Kill +15 days",
        "min_life": "9 days",
        "labels": (
            "Packs per box\nBoxes per layer\nLayers per pallet\nCases per pallet\nType of Pallet"
        ),
        "values": "6\n10\n10\n100\nPLASTIC DIGIPAL PALLET",
        "inline": "2 mm\n2.5 mm\n3.5 mm",
        "history": (
            "1\nJames Walton\nAll\nSpecification originated\n12/06/2024\n"
            "2\nDaniel Paiva\n1\nPad code amended\n06/03/2025"
        ),
    }
    rows_without_tare = ROWS.split("7903176")[0]  # the template below has its own tare lines
    return spec_text(rows=rows_without_tare) + template_block({**base, **over})


def template_block(values: dict[str, str]) -> str:
    return TEMPLATE.format(**values)


def test_the_template_fields_are_read() -> None:
    spec = parse_fact_spec([template()])
    assert (spec.weight, spec.weight_type) == ("170g", "Fixed Weight")
    assert (spec.storage, spec.temp_min_c, spec.temp_max_c) == (
        "Keep refrigerated 0°C to 3°C",
        0.0,
        3.0,
    )
    assert (spec.max_life_days, spec.life_basis, spec.min_life_days) == (15, "kill", 9)
    assert (spec.cases_per_pallet, spec.layers_per_pallet, spec.boxes_per_layer) == (100, 10, 10)
    assert spec.pallet_type == "PLASTIC DIGIPAL PALLET"
    assert (spec.ferrous_mm, spec.non_ferrous_mm, spec.stainless_mm) == (2.0, 2.5, 3.5)
    assert spec.metal_detection["xray"] == [1.5, 2.0, 2.0]
    assert (spec.pack_tare_g, spec.case_tare_g, spec.product_weight) == (20.1, 1500.0, "170g")


def test_temperatures_in_the_ways_they_are_written() -> None:
    from qacopilot.ingest.factspec import _temperatures

    assert _temperatures("Keep refrigerated 0°C to 3°C") == (0.0, 3.0)
    assert _temperatures("Keep refrigerated 0°-3°C") == (0.0, 3.0)  # a hyphen is a range
    assert _temperatures("Keep frozen -18°C") == (-18.0, -18.0)  # a hyphen is a minus
    assert _temperatures("Keep frozen -18oC") == (-18.0, -18.0)  # a letter o for the degree sign
    assert _temperatures("Keep tempered -2°C to 3°C") == (-2.0, 3.0)
    assert _temperatures("Keep refrigerated") == (None, None)
    # a hyphen that is a minus however it is spaced (these were read as +18 and so missed)
    assert _temperatures("Keep frozen - 18°C") == (-18.0, -18.0)
    assert _temperatures("Keep frozen-18°C") == (-18.0, -18.0)
    assert _temperatures("Keep frozen \u2013 18\u00b0C") == (-18.0, -18.0)  # an en dash
    assert _temperatures("Keep at 18°C ambient") == (18.0, 18.0)  # ambient: no sign to restore
    assert _temperatures("Keep frozen 18°C") == (-18.0, -18.0)  # frozen is never above zero


def test_months_are_thirty_days_and_a_missing_figure_is_not_invented() -> None:
    spec = parse_fact_spec([template(max_life="Kill +18 months", min_life="9 months")])
    assert (spec.max_life_days, spec.min_life_days) == (540, 270)
    unit_less = parse_fact_spec([template(max_life="*SEE SHELF-LIFE TABLE")])
    assert unit_less.max_life_days is None and unit_less.max_life_text == "*SEE SHELF-LIFE TABLE"


def test_the_pallet_labels_vary_but_the_values_follow_them() -> None:
    spec = parse_fact_spec(
        [
            template(
                labels=(
                    "Bags per box\nBoxes per layer\nLayers per pallet\n"
                    "Cases per pallet\nType of Pallet"
                )
            )
        ]
    )
    assert spec.cases_per_pallet == 100 and spec.pallet["Bags per box"] == "6"
    bulk = parse_fact_spec(
        [template(values="N/A\nN/A\nN/A\nN/A\nPurple DOLAV", inline="N/A mm\nN/A mm\nN/A mm")]
    )
    assert bulk.cases_per_pallet is None and bulk.pallet_type == "Purple DOLAV"  # N/A is not 0
    assert bulk.ferrous_mm is None and "inline" not in bulk.metal_detection
    assert bulk.metal_detection_text and "N/A" in bulk.metal_detection_text


def test_the_version_history_is_read_and_stops_at_the_end_of_the_table() -> None:
    spec = parse_fact_spec([template()])
    assert [(h.version, h.person, h.changed_on) for h in spec.history] == [
        (1, "James Walton", date(2024, 6, 12)),
        (2, "Daniel Paiva", date(2025, 3, 6)),
    ]
    old = parse_fact_spec(
        [
            template(
                history=(
                    "Creation\nRui Sousa\nCreation\nNew product\n21.12.15\n"
                    "1\nDaniel Paiva\nAll\nUpdated\n05/05/17"
                )
            )
        ]
    )
    assert [(h.version, h.changed_on) for h in old.history] == [
        (None, date(2015, 12, 21)),
        (1, date(2017, 5, 5)),
    ]
    # text that is not a history row (here a long sentence before a date) is never taken for one
    junk = parse_fact_spec(
        [
            template(
                history=(
                    "Apply in each pallet 2 sheets A4 size indicating the Kill Date\n"
                    "Apply one each narrow side\n03/09/2024"
                )
            )
        ]
    )
    assert junk.history == []


def test_the_pack_tare_label_wrapped_over_two_lines_ends_the_packaging_table() -> None:
    text = spec_text(rows=ROWS.replace("7903176\n1\n", "7903176\n1\nPack\nTare\n11.2g\n"))
    row = next(r for r in parse_fact_spec([text]).packaging if r.code == "7903176")
    assert row.supplier == "" and "Tare" not in row.cells


def test_a_version_table_that_runs_across_pages_and_a_name_split_over_two_lines() -> None:
    """The table stopped at the first page header, so later rows (and who wrote them) were lost,
    and "Peter" / "Bringlesnap" on two lines was not read as a person."""
    history = (
        "0\nPaula Sousa\nAll\nNew product.\n05/03/12\n"
        "4\nDaniel Paiva\nAll\nLabel images changed\n22/03/17\n"
        "PLASTIC DIGIPAL\nPALLET\nGRESSINGHAM FOODS\nNAV CODE\n4053897\nPLU\n3897\n"
        "VERSION\n34\n31/08/2026\nBRAND\nGressingham\nPage 3 of 3\n"
        "5\nPeter\nBringlesnap\n1\nX Ray information added\n08/07/2022\n"
        "6\nPeter Bringlesnap\nAll\nPacking changed\nfrom box to crate\n04/10/2022"
    )
    spec = parse_fact_spec([template(history=history)])
    assert [(h.version, h.person) for h in spec.history] == [
        (0, "Paula Sousa"),
        (4, "Daniel Paiva"),
        (5, "Peter Bringlesnap"),  # a name split over two lines
        (6, "Peter Bringlesnap"),  # the table continued after the page header
    ]
    assert (
        spec.history[2].pages == "1"
        and spec.history[3].amends == "Packing changed from box to crate"
    )
    # the page header's own date (the spec's version date) is not a history row
    assert all(h.person != "Gressingham" for h in spec.history)
