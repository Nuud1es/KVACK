"""The structured query layer: conditions combine, nothing is silently dropped, and the answer
says how the question was understood."""

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from test_factspec import spec_text, template_block

from qacopilot.ingest import DocType, PageText, ParsedDocument
from qacopilot.llm import BaseLLMProvider, LLMResponse
from qacopilot.rag import specquery as sq
from qacopilot.rag.specquery import PackCond, SpecFilter, SpecQuery
from qacopilot.repositories import DocumentRepository
from qacopilot.repositories.testing import DatabaseUnavailable, temporary_repository


@pytest.fixture
async def repo() -> AsyncIterator[DocumentRepository]:
    try:
        async with temporary_repository() as r:
            yield r
    except DatabaseUnavailable as exc:
        pytest.skip(str(exc))


def _rows(*items: tuple[str, str, str, str]) -> str:
    """(code, description, supplier, section) as packaging table cells."""
    out, section = "", ""
    for code, desc, supplier, sect in items:
        if sect != section:
            out += f"{sect.upper()} PACKAGING\n"
            section = sect
        out += f"{code}\n1\n{desc}\n{supplier}\n10x10mm\n1.2g\n"
    return out


BAG = ("7900001", "Blue Gas Flush Bag", "Krehalon UK", "primary")
FILM = ("7900002", "Film Top Clear", "Multivac UK Ltd", "primary")
LABEL = ("7900003", "End case label plain", "Reflex", "secondary")
STRAP = ("7900004", "Strapping", "Qualitape", "secondary")


async def add(
    repo: DocumentRepository,
    doc_id: str,
    nav: str,
    product: str,
    customer: str,
    kind: str,
    allergens: str,
    version: int,
    *rows: tuple[str, str, str, str],
    extra: str = "",
) -> None:
    text = (
        spec_text(nav, version, _rows(*rows))
        .replace("Fresh Gressingham Duck Breast 170g x6", product)
        .replace("Morrisons", customer)
        .replace("TYPE\nFresh", f"TYPE\n{kind}")
        .replace("ALLERGENS\nNone", f"ALLERGENS\n{allergens}")
    ) + extra
    doc = ParsedDocument(source=f"{doc_id}.pdf", pages=[PageText(page=1, text=text, method="text")])
    await repo.save_document(doc_id, f"Fact Spec - {nav}.pdf", DocType.FACT_SPEC, doc, 1)


@pytest.fixture
async def specs(repo: DocumentRepository) -> DocumentRepository:
    await add(repo, "a", "4000001", "Fresh Whole Duck 2kg", "Tesco", "Fresh", "None", 3, BAG, LABEL)
    await add(
        repo, "b", "4000002", "Frozen Whole Duck 1.8kg", "Tesco", "Frozen", "Soy", 1, BAG, STRAP
    )
    await add(repo, "c", "4000003", "Duck Breast 170g", "Tesco", "Fresh", "None", 2, FILM, LABEL)
    await add(
        repo, "d", "4000004", "Duck Breast 300g", "Aldi", "Fresh", "None", 1, FILM, LABEL, STRAP
    )
    await add(repo, "e", "4000005", "Roast Turkey Crown", "Aldi", "Frozen", "Milk", 5, BAG, LABEL)
    await add(repo, "f", "4000006", "Roast Turkey Breast", "Sainsbury's", "Fresh", "None", 1, FILM)
    return repo


async def ask(repo: DocumentRepository, question: str, query: SpecQuery) -> str:
    vocab = await repo.spec_vocab()
    events = await sq.execute(repo, sq.validate(query, vocab, question))
    return events[-1].answer  # type: ignore[union-attr]


async def test_every_condition_narrows_the_count(specs: DocumentRepository) -> None:
    q = SpecQuery(intent="count", filter=SpecFilter(customer="Tesco", product="duck breast"))
    answer = await ask(specs, "how many Tesco duck breast specs", q)
    assert "1 of the 6 current Fact Specs match" in answer  # not all 3 Tesco specs
    assert 'Understood as: count of specs: customer Tesco, product "duck breast"' in answer

    q = SpecQuery(intent="count", filter=SpecFilter(customer="Tesco", spec_type="frozen"))
    assert "1 of the 6" in await ask(specs, "how many Tesco specs are frozen", q)


async def test_no_allergens_and_negated_packaging(specs: DocumentRepository) -> None:
    none = await ask(
        specs,
        "how many specs have no allergens",
        SpecQuery(intent="count", filter=SpecFilter(allergens="no")),
    )
    assert "4 of the 6" in none
    yes = await ask(
        specs,
        "how many declare allergens",
        SpecQuery(intent="count", filter=SpecFilter(allergens="yes")),
    )
    assert "2 of the 6" in yes
    no_bag = SpecQuery(
        intent="count", filter=SpecFilter(packaging=[PackCond(kind="gas flush bag", negate=True)])
    )
    assert "3 of the 6" in await ask(specs, "how many specs do not use a gas flush bag", no_bag)


async def test_two_packaging_conditions_must_both_hold(specs: DocumentRepository) -> None:
    both = SpecQuery(
        intent="list",
        filter=SpecFilter(packaging=[PackCond(code="7900001"), PackCond(code="7900003")]),
    )
    answer = await ask(specs, "which specs use both 7900001 and 7900003", both)
    assert "2 of the 6" in answer and "4000001" in answer and "4000005" in answer
    assert "4000002" not in answer  # bag but strapping, no label
    two_suppliers = SpecQuery(
        intent="count",
        filter=SpecFilter(
            packaging=[PackCond(supplier="Krehalon UK"), PackCond(supplier="Qualitape")]
        ),
    )
    assert "1 of the 6" in await ask(specs, "krehalon uk and qualitape", two_suppliers)


async def test_compare_common_rank_and_distinct(specs: DocumentRepository) -> None:
    cmp = SpecQuery(intent="compare", filter=SpecFilter(product="turkey"), compare_with="duck")
    answer = await ask(specs, "which has more specs, turkey or duck", cmp)
    assert '"turkey": 2 specs; "duck": 4 specs' in answer and '"duck" has more' in answer

    common = SpecQuery(intent="common", filter=SpecFilter(product="turkey"), compare_with="duck")
    answer = await ask(specs, "what packaging is common to turkey and duck", common)
    assert "7900001" in answer and "7900003" in answer and "7900004" not in answer

    rank = SpecQuery(intent="rank", filter=SpecFilter(product="whole duck"), group_by="customer")
    answer = await ask(specs, "which customers buy whole duck", rank)
    assert "1. Tesco: 2 specs" in answer

    distinct = SpecQuery(intent="distinct", distinct_of="customer")
    assert "3 different customers among all current specs" in await ask(
        specs, "how many different customers do we have", distinct
    )


async def test_section_and_kind_narrow_the_packaging_rows(specs: DocumentRepository) -> None:
    primary = SpecQuery(intent="packaging", filter=SpecFilter(customer="Tesco"), section="primary")
    answer = await ask(specs, "what primary packaging does Tesco use", primary)
    assert "7900001" in answer and "7900002" in answer
    assert "7900003" not in answer and "7900004" not in answer  # secondary rows are left out
    film = SpecQuery(intent="packaging", filter=SpecFilter(product="duck breast"), kind="film")
    answer = await ask(specs, "what film do we use for duck breast", film)
    assert "7900002" in answer and "7900001" not in answer


async def test_versions_dates_and_order(specs: DocumentRepository) -> None:
    v1 = await ask(
        specs, "specs on version 1", SpecQuery(intent="count", filter=SpecFilter(version=1))
    )
    assert "3 of the 6" in v1
    revised = await ask(specs, "most revised", SpecQuery(intent="most_revised"))
    assert revised.index("4000005") < revised.index("4000001")  # version 5 before version 3
    year = await ask(
        specs, "updated in 2025", SpecQuery(intent="count", filter=SpecFilter(year=2025))
    )
    assert "6 of the 6" in year  # the fixture dates every spec 06/03/2025


async def test_a_name_that_is_not_in_the_question_is_dropped_and_said(
    specs: DocumentRepository,
) -> None:
    """The model once copied "whole duck" from an example into an unrelated question."""
    q = SpecQuery(
        intent="count",
        filter=SpecFilter(product="whole duck", packaging=[PackCond(code="7900001")]),
    )
    answer = await ask(specs, "how many specs use 7900001", q)
    assert '(ignored product "whole duck" (not in the question).)' in answer
    assert "3 of the 6" in answer  # counted by the code alone


async def test_a_typo_is_corrected_and_an_unknown_name_is_reported(
    specs: DocumentRepository,
) -> None:
    typo = SpecQuery(intent="count", filter=SpecFilter(product="turkey"))
    assert "2 of the 6" in await ask(specs, "how many turky specs", typo)  # not 6 of 6
    near = SpecQuery(intent="count", filter=SpecFilter(customer="Tescoo"))
    answer = await ask(specs, "how many Tescoo specs", near)
    assert "Tescoo read as Tesco" in answer and "3 of the 6" in answer  # corrected, and said
    unknown = SpecQuery(intent="count", filter=SpecFilter(customer="Zorblax"))
    answer = await ask(specs, "how many Zorblax specs", unknown)
    assert 'No customer called "Zorblax"' in answer
    sainsburys = SpecQuery(intent="count", filter=SpecFilter(customer="Sainsburys"))
    assert "1 of the 6" in await ask(specs, "how many Sainsburys specs", sainsburys)
    missing_code = SpecQuery(
        intent="count", filter=SpecFilter(packaging=[PackCond(code="7999999")])
    )
    assert "No current spec uses packaging code 7999999" in await ask(
        specs, "who uses 7999999", missing_code
    )


async def test_which_questions_need_the_query_layer(specs: DocumentRepository) -> None:
    vocab = await specs.spec_vocab()
    needs = [
        "how many Tesco duck breast specs are there",
        "how many specs have no allergens",
        "which specs use both 7900001 and 7900003",
        "which has more specs, turkey or duck",
        "what primary packaging does Tesco use",
        "what packaging codes are used for frozen whole duck",
        "how many different customers do we have",
        "what is the oldest spec",
        "how many turky specs",
        "list the products for Aldi",
    ]
    for question in needs:
        assert sq.needs_engine(question, vocab), question
    single_shape = [
        "what is the most used packaging code",
        "what packaging does 4000001 use",
        "how many whole duck specs are there",
        "what is the weather today",
        "which probes are due for calibration",
    ]
    for question in single_shape:
        assert not sq.needs_engine(question, vocab), question


class _Scripted(BaseLLMProvider):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def chat(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:  # type: ignore[override]
        return LLMResponse(content=self.reply, finish_reason="stop")

    async def stream(self, messages: Sequence[Any], **kwargs: Any) -> AsyncIterator[str]:  # type: ignore[override]
        yield self.reply

    async def health(self) -> bool:
        return True


async def test_the_model_reads_the_question_and_code_runs_it(specs: DocumentRepository) -> None:
    llm = _Scripted('{"intent":"count","filter":{"customer":"Tesco","product":"duck breast"}}')
    events = await sq.answer_from_specquery("how many Tesco duck breast specs", specs, llm)
    assert events is not None and "1 of the 6" in events[-1].answer  # type: ignore[union-attr]


async def test_content_questions_and_unreadable_replies_fall_through(
    specs: DocumentRepository,
) -> None:
    content = _Scripted('{"intent":"content"}')
    assert (
        await sq.answer_from_specquery("storage temperature for frozen whole duck", specs, content)
        is None
    )
    garbled = _Scripted("not json at all")
    assert (
        await sq.answer_from_specquery("how many Tesco duck breast specs", specs, garbled) is None
    )
    unsupported = _Scripted('{"intent":"unsupported","reason":"unused codes are not recorded"}')
    events = await sq.answer_from_specquery(
        "which packaging codes are not used by any spec", specs, unsupported
    )
    assert events is not None and "I can't answer that from the spec tables" in events[-1].answer  # type: ignore[union-attr]


async def test_short_questions_get_a_menu_or_a_profile(specs: DocumentRepository) -> None:
    menu = await sq.answer_vague("help", specs)
    assert menu is not None and "Counts:" in menu[-1].answer  # type: ignore[union-attr]
    profile = await sq.answer_vague("tesco", specs)
    assert profile is not None and profile[-1].answer.startswith("Tesco: 3 current specs")  # type: ignore[union-attr]
    group = await sq.answer_vague("turkey", specs)
    assert group is not None and '"turkey": 2 current specs' in group[-1].answer  # type: ignore[union-attr]
    assert await sq.answer_vague("who approved spec 4000001", specs) is None  # a spec: not vague
    assert await sq.answer_vague("kryptonite", specs) is None


async def test_a_count_or_list_that_cannot_be_read_is_not_left_to_search(
    specs: DocumentRepository,
) -> None:
    vocab = await specs.spec_vocab()
    assert sq.looks_like_aggregate("how many specs mention blue crates", vocab)
    assert sq.looks_like_aggregate("list all the specs with a fancy lid", vocab)
    assert not sq.looks_like_aggregate("how many days is the max life for 4000001", vocab)
    assert not sq.looks_like_aggregate("what is the storage temperature", vocab)


async def test_what_the_question_names_is_added_when_the_model_leaves_it_out(
    specs: DocumentRepository,
) -> None:
    """The model sometimes returns a query without the supplier or the packaging type that the
    question plainly names; code adds them, and says so."""
    vocab = await specs.spec_vocab()

    def run(question: str, query: SpecQuery) -> sq.Understood:
        return sq.validate(query, vocab, question)

    # a packaging type ("film"), a section and a frozen/fresh word
    u = run(
        "what film do we use for duck breast",
        SpecQuery(intent="packaging", filter=SpecFilter(product="duck breast")),
    )
    assert u.query.kind == "film"
    u = run(
        "what primary packaging does Tesco use", SpecQuery(intent="packaging", filter=SpecFilter())
    )
    assert u.query.section == "primary" and u.query.filter.customer == "Tesco"
    u = run(
        "how many frozen turkey specs",
        SpecQuery(intent="count", filter=SpecFilter(product="turkey")),
    )
    assert u.query.filter.spec_type == "frozen"
    # a supplier named in the question joins the packaging condition ("Krehalon UK bags")
    u = run(
        "which specs use Krehalon UK bags",
        SpecQuery(intent="list", filter=SpecFilter(packaging=[PackCond(kind="bag")])),
    )
    assert [(c.supplier, c.kind) for c in u.query.filter.packaging] == [("Krehalon UK", "bag")]
    # a second code the model dropped
    u = run(
        "which specs use both 7900001 and 7900003",
        SpecQuery(intent="list", filter=SpecFilter(packaging=[PackCond(code="7900001")])),
    )
    assert {c.code for c in u.query.filter.packaging} == {"7900001", "7900003"}
    # "not frozen" is a negation: frozen is not added as a positive condition
    u = run(
        "duck breast specs that are not frozen",
        SpecQuery(intent="list", filter=SpecFilter(product="duck breast")),
    )
    assert u.query.filter.spec_type is None


async def test_the_supplier_and_the_type_both_narrow_the_answer(specs: DocumentRepository) -> None:
    vocab = await specs.spec_vocab()
    query = SpecQuery(intent="list", filter=SpecFilter(packaging=[PackCond(kind="label")]))
    u = sq.validate(query, vocab, "which specs use Reflex labels")
    answer = (await sq.execute(specs, u))[-1].answer  # type: ignore[union-attr]
    assert "uses label" not in answer or "Reflex" in answer
    assert "4 of the 6" in answer  # label rows from Reflex: specs a, c, d, e


async def test_an_answer_carries_a_download_of_every_row(specs: DocumentRepository) -> None:
    """The list stops at 25; the download has all of them, from the same validated query."""
    query = SpecQuery(
        intent="list",
        filter=SpecFilter(packaging=[PackCond(code="7900001"), PackCond(code="7900003")]),
    )
    vocab = await specs.spec_vocab()
    events = await sq.execute(
        specs, sq.validate(query, vocab, "which specs use both 7900001 and 7900003")
    )
    done = events[-1]
    assert done.export is not None and done.export.count == 2  # type: ignore[union-attr]
    # the browser sends the query back; it is validated again, and gives the same rows
    rows = await sq.export_specs(specs, SpecQuery.model_validate(done.export.query))  # type: ignore[union-attr]
    assert [r[0] for r in rows] == ["4000001", "4000005"]
    text = sq.csv_text(rows)
    assert text.startswith("﻿NAV code,Product,Customer,Type,Status,Version,Version date")
    assert "4000001" in text and "4000005" in text and "4000002" not in text


async def test_every_answer_with_a_count_has_a_download_but_a_comparison_has_none(
    specs: DocumentRepository,
) -> None:
    vocab = await specs.spec_vocab()
    count = await sq.execute(
        specs, sq.validate(SpecQuery(intent="count", filter=SpecFilter(customer="Aldi")), vocab)
    )
    assert count[-1].export.count == 2  # type: ignore[union-attr]
    cmp = await sq.execute(
        specs,
        sq.validate(
            SpecQuery(intent="compare", filter=SpecFilter(product="turkey"), compare_with="duck"),
            vocab,
        ),
    )
    assert cmp[-1].export is None  # type: ignore[union-attr]
    with pytest.raises(sq.ExportError, match="comparison"):
        await sq.export_specs(
            specs,
            SpecQuery(intent="compare", filter=SpecFilter(product="turkey"), compare_with="x"),
        )
    with pytest.raises(sq.ExportError, match="No customer called"):
        await sq.export_specs(
            specs, SpecQuery(intent="count", filter=SpecFilter(customer="Zorblax"))
        )


def test_a_cell_that_would_run_as_a_formula_is_made_plain_text() -> None:
    text = sq.csv_text([("=HYPERLINK(1)", "+1", "-2", "@x", "normal", 3, None, None, "f.pdf")])
    row = text.splitlines()[1]
    assert row.startswith("'=HYPERLINK(1),'+1,'-2,'@x,normal,3,,,f.pdf")


def detail(**over: str) -> str:
    """The rest of a spec's template (storage, life, pallet, metal detection, history)."""
    base = {
        "weight": "170g",
        "temp": "Keep refrigerated 0°C to 3°C",
        "max_life": "Kill +15 days",
        "min_life": "9 days",
        "labels": (
            "Packs per box\nBoxes per layer\nLayers per pallet\nCases per pallet\nType of Pallet"
        ),
        "values": "6\n10\n10\n100\nPLASTIC PALLET",
        "inline": "2 mm\n2.5 mm\n3.5 mm",
        "history": "1\nJames Walton\nAll\nOriginated\n12/06/2024",
    }
    return template_block({**base, **over})


@pytest.fixture
async def detailed(repo: DocumentRepository) -> DocumentRepository:
    await add(
        repo,
        "a",
        "4000001",
        "Fresh Whole Duck 2kg",
        "Tesco",
        "Fresh",
        "None",
        3,
        BAG,
        extra=detail(),
    )
    await add(
        repo,
        "b",
        "4000002",
        "Frozen Whole Duck 1.8kg",
        "Tesco",
        "Frozen",
        "None",
        1,
        BAG,
        extra=detail(
            temp="Keep frozen -18°C",
            max_life="Kill +18 months",
            values="6\n10\n10\n60\nCHEP PALLET",
            inline="3 mm\n4 mm\n4 mm",
            history=(
                "1\nDaniel Paiva\nAll\nOriginated\n01/02/2024\n"
                "2\nJames Walton\n1\nPad amended\n10/09/2025"
            ),
        ),
    )
    await add(
        repo,
        "c",
        "4000003",
        "Roast Turkey Crown",
        "Aldi",
        "Fresh",
        "None",
        2,
        FILM,
        extra=detail(
            max_life="Kill +5 days",
            values="N/A\nN/A\nN/A\nN/A\nPurple DOLAV",
            history="1\nDaniel Paiva\nAll\nOriginated\n03/03/2023",
        ),
    )
    return repo


async def test_numbers_the_spec_states_are_filters_and_sort_keys(
    detailed: DocumentRepository,
) -> None:
    from qacopilot.rag.specquery import NumCond

    long_life = SpecQuery(
        intent="list",
        filter=SpecFilter(numeric=[NumCond(field="max_life_days", op="gt", value=14)]),
    )
    answer = await ask(detailed, "which specs have a max life over 14 days", long_life)
    assert "2 of the 3" in answer and "4000001" in answer and "4000002" in answer
    assert "4000003" not in answer  # 5 days
    assert "max life days more than 14" in answer  # said how it was understood

    pallet = SpecQuery(
        intent="count",
        filter=SpecFilter(numeric=[NumCond(field="cases_per_pallet", op="gte", value=100)]),
    )
    assert "1 of the 3" in await ask(detailed, "specs with 100 or more cases per pallet", pallet)
    longest = SpecQuery(intent="list", order_by="max_life_days", descending=True, limit=1)
    answer = await ask(detailed, "which spec has the longest max life", longest)
    assert answer.index("4000002") < answer.index("max life days: 540")  # 18 months = 540 days
    assert "4000001" not in answer
    shortest = SpecQuery(intent="list", order_by="max_life_days", descending=False, limit=1)
    assert "4000003" in await ask(detailed, "which spec has the shortest max life", shortest)


async def test_the_storage_temperature_range_must_include_the_value(
    detailed: DocumentRepository,
) -> None:
    frozen = SpecQuery(intent="list", filter=SpecFilter(storage_c=-18))
    answer = await ask(detailed, "which specs are kept at -18", frozen)
    assert "1 of the 3" in answer and "4000002" in answer
    chilled = SpecQuery(intent="count", filter=SpecFilter(storage_c=2))
    assert "2 of the 3" in await ask(detailed, "which specs are kept at 2", chilled)  # 0 to 3


async def test_who_changed_a_spec_any_last_or_created(detailed: DocumentRepository) -> None:
    def who(role: str) -> SpecQuery:
        return SpecQuery(
            intent="list",
            filter=SpecFilter(person="James Walton", person_role=role),  # type: ignore[arg-type]
        )

    any_change = await ask(detailed, "which specs did James Walton change", who("any"))
    assert "2 of the 3" in any_change  # a and b
    last = await ask(detailed, "which specs did James Walton last update", who("last"))
    assert "2 of the 3" in last  # a (only entry) and b (his is the latest)
    created = await ask(detailed, "which specs did James Walton create", who("created"))
    assert "1 of the 3" in created and "4000001" in created  # b was created by Daniel Paiva
    ranking = await ask(
        detailed, "who has updated the most specs", SpecQuery(intent="rank", group_by="person")
    )
    assert (
        "1. Daniel Paiva: changed 2 specs" in ranking
        or "1. James Walton: changed 2 specs" in ranking
    )


async def test_an_unknown_person_is_reported_with_suggestions(detailed: DocumentRepository) -> None:
    query = SpecQuery(intent="list", filter=SpecFilter(person="Jaems Walton"))
    answer = await ask(detailed, "which specs did Jaems Walton update", query)
    assert "James Walton" in answer  # read as the nearest name, and said
    unknown = SpecQuery(intent="list", filter=SpecFilter(person="Zorblax Quux"))
    assert 'No person called "Zorblax Quux"' in await ask(
        detailed, "which specs did Zorblax Quux", unknown
    )


async def test_questions_about_the_fields_reach_the_query_layer(
    detailed: DocumentRepository,
) -> None:
    vocab = await detailed.spec_vocab()
    for question in (
        "which specs have a max life over 14 days",
        "how many specs have more than 100 cases per pallet",
        "which specs are kept at -18",
        "which specs did James Walton last update",
        "who has updated the most specs",
        "which spec has the longest max life",
    ):
        assert sq.needs_engine(question, vocab), question


async def test_one_spec_questions_are_answered_from_its_fields(
    detailed: DocumentRepository,
) -> None:
    from qacopilot.rag.factspec_lookup import answer_from_factspec

    async def ans(question: str) -> str:
        events = await answer_from_factspec(question, detailed)
        assert events is not None, question
        return events[-1].answer  # type: ignore[union-attr]

    assert "Kill +15 days" in await ans("what is the max life for 4000001")
    assert "(540 days)" in await ans("what is the shelf life of 4000002")  # months shown as days
    assert "Keep frozen -18°C" in await ans("what temperature should 4000002 be kept at")
    pallet = await ans("what is the pallet configuration for 4000002")
    assert "Cases per pallet: 60" in pallet and "CHEP PALLET" in pallet
    metal = await ans("what is the metal detection setting for 4000002")
    assert "ferrous 3 mm" in metal and "X-ray" in metal
    assert "pack tare 20.1 g" in await ans("what is the weight of 4000001")
    who = await ans("who last updated 4000002")
    assert "last changed by James Walton on 10/09/2025" in who and "created by Daniel Paiva" in who
    approval = await ans("who approved spec 4000001")
    assert "does not record an approver" in approval and "James Walton" in approval
    assert "no" not in (await ans("what are the process steps for 4000001")).lower()[:4]


async def test_a_number_the_model_invented_is_dropped_and_a_superlative_means_ordering(
    detailed: DocumentRepository,
) -> None:
    from qacopilot.rag.specquery import NumCond

    vocab = await detailed.spec_vocab()
    # the model once read "shortest max life" as "max life at most 0 days"
    wrong = SpecQuery(
        intent="list",
        filter=SpecFilter(numeric=[NumCond(field="max_life_days", op="lte", value=0)]),
    )
    u = sq.validate(wrong, vocab, "which spec has the shortest max life")
    assert u.query.filter.numeric == []
    assert u.query.order_by == "max_life_days" and u.query.descending is False
    assert u.query.limit == 1
    answer = (await sq.execute(detailed, u))[-1].answer  # type: ignore[union-attr]
    assert "4000003" in answer and "max life days: 5" in answer  # 5 days is the shortest

    # a real comparison keeps its number and is not turned into ordering
    real = SpecQuery(
        intent="count",
        filter=SpecFilter(numeric=[NumCond(field="max_life_days", op="gt", value=14)]),
    )
    kept = sq.validate(real, vocab, "how many specs have a max life of more than 14 days")
    assert [c.value for c in kept.query.filter.numeric] == [14] and kept.query.order_by is None
    # months are converted to days, so 540 is "18 months" in the question
    months = SpecQuery(
        intent="count",
        filter=SpecFilter(numeric=[NumCond(field="max_life_days", op="gte", value=540)]),
    )
    assert sq.validate(months, vocab, "max life of 18 months or more").query.filter.numeric
    # "kept at 5 degrees" is not "kept at -18"
    other = SpecQuery(intent="list", filter=SpecFilter(storage_c=-18))
    assert sq.validate(other, vocab, "which specs are kept at 5").query.filter.storage_c is None


async def test_conditions_the_question_does_not_mention_are_dropped_and_a_lost_product_restored(
    detailed: DocumentRepository,
) -> None:
    """ "how many turkey specs are for Wholesale" came back as "customer Wholesale, frozen" every
    time: the model dropped the product and invented a condition."""
    vocab = await detailed.spec_vocab()
    wrong = SpecQuery(
        intent="count",
        filter=SpecFilter(customer="Aldi", spec_type="frozen", allergens="no", year=2025),
    )
    u = sq.validate(wrong, vocab, "how many turkey specs are for Aldi")
    f = u.query.filter
    assert (f.spec_type, f.allergens, f.year) == (None, None, None)
    assert f.product == "turkey" and f.customer == "Aldi"  # restored from the question
    answer = (await sq.execute(detailed, u))[-1].answer  # type: ignore[union-attr]
    assert "1 of the 3" in answer and 'product "turkey"' in answer
    # stated conditions stay
    kept = SpecQuery(intent="count", filter=SpecFilter(spec_type="frozen", year=2025))
    k = sq.validate(kept, vocab, "how many frozen specs dated 2025 are there").query.filter
    assert (k.spec_type, k.year) == ("frozen", 2025)
    # a customer's own words are not mistaken for a product
    plain = SpecQuery(intent="count", filter=SpecFilter(customer="Tesco"))
    assert sq.validate(plain, vocab, "how many Tesco specs").query.filter.product is None


async def test_a_number_or_a_field_phrase_is_never_restored_as_a_product(
    detailed: DocumentRepository,
) -> None:
    """A question about "100 cases per pallet" was read with a product called "100"."""
    from qacopilot.rag.specquery import NumCond

    vocab = await detailed.spec_vocab()
    query = SpecQuery(
        intent="count",
        filter=SpecFilter(numeric=[NumCond(field="cases_per_pallet", op="gt", value=50)]),
    )
    u = sq.validate(query, vocab, "how many specs have more than 50 cases per pallet")
    assert u.query.filter.product is None
    # 100 and 60 cases per pallet are both over 50; the third spec has none (N/A)
    assert "2 of the 3" in (await sq.execute(detailed, u))[-1].answer  # type: ignore[union-attr]


async def test_the_role_of_a_person_comes_from_the_question_not_from_the_model(
    detailed: DocumentRepository,
) -> None:
    """ "which specs did X change" once came back as "latest change by X": 37 specs, not 218."""
    vocab = await detailed.spec_vocab()

    def role(question: str, given: str) -> str:
        query = SpecQuery(
            intent="list",
            filter=SpecFilter(person="James Walton", person_role=given),  # type: ignore[arg-type]
        )
        return sq.validate(query, vocab, question).query.filter.person_role

    assert role("which specs did James Walton change", "last") == "any"
    assert role("which specs has James Walton updated", "created") == "any"
    assert role("which specs did James Walton last update", "any") == "last"
    assert role("which specs did James Walton create", "any") == "created"
    assert role("which specs were written by James Walton", "last") == "created"


async def test_a_supplier_or_customer_name_is_never_a_product(detailed: DocumentRepository) -> None:
    """ "which products use Smart Packaging boxes" came back with the product "Smart Packaging"."""
    vocab = await detailed.spec_vocab()
    both = SpecQuery(intent="count", filter=SpecFilter(product="Tesco Whole Duck"))
    f = sq.validate(both, vocab, "how many Tesco Whole Duck specs").query.filter
    assert (f.customer, f.product) == ("Tesco", "Whole Duck")  # the name moves out of the product
    # only a name: nothing is left of the product
    only = SpecQuery(intent="count", filter=SpecFilter(customer="Tesco", product="Tesco"))
    assert sq.validate(only, vocab, "how many Tesco specs").query.filter.product is None
    # spellings are compared forgivingly: "Tescos" is still the customer, not a product word
    plain = SpecQuery(intent="count", filter=SpecFilter(customer="Tesco"))
    assert sq.validate(plain, vocab, "how many Tescos specs").query.filter.product is None


async def test_a_word_of_a_supplier_is_not_an_unknown_customer(
    detailed: DocumentRepository,
) -> None:
    """ "which products use Smart Packaging boxes" failed with 'No customer called "Smart"'."""
    vocab = await detailed.spec_vocab()
    q = SpecQuery(intent="list", filter=SpecFilter(customer="Multivac"))
    u = sq.validate(q, vocab, "which products use Multivac UK Ltd film")
    assert u.query.filter.customer is None and not u.problems
    still = sq.validate(
        SpecQuery(intent="count", filter=SpecFilter(customer="Zorblax")), vocab, "how many Zorblax"
    )
    assert still.problems  # an unknown name that is not part of a supplier is still reported


async def test_a_follow_up_refines_the_previous_answer(specs: DocumentRepository) -> None:
    """ "list whole duck specs", then "give me all from Tesco": whole duck AND Tesco."""
    first_llm = _Scripted('{"intent":"list","filter":{"product":"whole duck"}}')
    first = await sq.answer_from_specquery("list whole duck specs", specs, first_llm)
    assert first is not None
    context = first[-1].export.query  # type: ignore[union-attr]

    llm = _Scripted('{"intent":"list","filter":{"customer":"Tesco"}}')
    events = await sq.answer_from_specquery(
        "give me all from tesco", specs, llm, sq.SpecQuery.model_validate(context)
    )
    assert events is not None
    answer = events[-1].answer  # type: ignore[union-attr]
    assert "continuing from your last question" in answer
    assert 'product "whole duck"' in answer and "customer Tesco" in answer
    assert "2 of the 6" in answer  # both Tesco whole-duck specs, not every Tesco spec (3)
    # the answer carries the combined query, so the next follow-up builds on it
    combined = events[-1].export.query  # type: ignore[union-attr]
    assert (
        combined["filter"]["product"] == "whole duck" and combined["filter"]["customer"] == "Tesco"
    )

    frozen = _Scripted('{"intent":"count","filter":{"spec_type":"frozen"}}')
    more = await sq.answer_from_specquery(
        "only frozen", specs, frozen, sq.SpecQuery.model_validate(combined)
    )
    assert more is not None and "1 of the 6" in more[-1].answer  # type: ignore[union-attr]


async def test_a_question_with_its_own_subject_is_not_merged(specs: DocumentRepository) -> None:
    context = sq.SpecQuery(intent="list", filter=sq.SpecFilter(product="whole duck"))
    llm = _Scripted('{"intent":"count","filter":{"customer":"Aldi","product":"turkey"}}')
    events = await sq.answer_from_specquery("how many turkey specs from Aldi", specs, llm, context)
    assert events is not None
    assert "continuing" not in events[-1].answer  # type: ignore[union-attr]
    assert "1 of the 6" in events[-1].answer  # type: ignore[union-attr]
    # no earlier answer: a bare follow-up is handled like any other question
    assert (
        await sq.answer_from_specquery("from Tesco", specs, _Scripted('{"intent":"list"}')) is None
    )


def test_follow_up_wording() -> None:
    for yes in (
        "give me all from tesco",
        "only frozen",
        "and for Lidl",
        "what about Aldi",
        "from Tesco",
    ):
        assert sq.is_follow_up(yes), yes
    for no in (
        "how many whole duck specs are there",
        "list the packaging codes",
        "which customers have the most specs",
    ):
        assert not sq.is_follow_up(no), no


async def test_a_refinement_the_model_reads_as_content_still_continues(
    specs: DocumentRepository,
) -> None:
    """ "only frozen" came back as a content question; it must still refine the last list."""
    context = sq.SpecQuery(intent="list", filter=sq.SpecFilter(product="whole duck"))
    llm = _Scripted('{"intent":"content","filter":{"spec_type":"frozen"}}')
    events = await sq.answer_from_specquery("only frozen", specs, llm, context)
    assert events is not None
    answer = events[-1].answer  # type: ignore[union-attr]
    assert "list of specs" in answer and "continuing from your last question" in answer
    assert "1 of the 6" in answer


def test_cue_words_are_stripped_and_pointing_back_counts_as_a_follow_up() -> None:
    assert sq.strip_follow_up_cue("only frozen") == "frozen"
    assert sq.strip_follow_up_cue("give me all from tesco") == "from tesco"
    assert sq.strip_follow_up_cue("how many of those use Reflex") == "how many use Reflex"
    assert sq.strip_follow_up_cue("what about Lidl") == "Lidl"
    assert sq.is_follow_up("how many of those use Reflex")
    assert not sq.is_follow_up("how many whole duck specs use Reflex")


async def test_show_me_all_of_them_lists_what_was_counted(specs: DocumentRepository) -> None:
    context = sq.SpecQuery(intent="count", filter=sq.SpecFilter(customer="Tesco"))
    llm = _Scripted('{"intent":"list"}')
    events = await sq.answer_from_specquery("show me all of them", specs, llm, context)
    assert events is not None
    answer = events[-1].answer  # type: ignore[union-attr]
    assert "list of specs: customer Tesco" in answer and "3 of the 6" in answer
    assert "1. 4000" in answer  # the specs are listed, not just counted
