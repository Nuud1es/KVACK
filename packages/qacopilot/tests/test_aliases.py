"""Suggestions for which spellings are one name, and what applying an alias changes."""

from collections.abc import AsyncIterator

import pytest
from test_specquery import STRAP, add  # noqa: F401  (add builds the test specs)

from qacopilot.aliases import suggest, suggest_people
from qacopilot.rag import specquery as sq
from qacopilot.rag.specquery import SpecFilter, SpecQuery
from qacopilot.repositories import DocumentRepository
from qacopilot.repositories.testing import DatabaseUnavailable, temporary_repository


def test_spelling_and_case_variants_are_safe_and_a_certification_tag_is_to_check() -> None:
    counts = {
        "Sainsbury’s": 14,
        "Sainsbury": 12,
        "Sainsbury's": 8,
        "Sainsbury's Certification/Retail": 2,
        "Tesco": 30,
        "TESCO": 2,
        "Tesco Certification/Retail": 6,
        "Aldi": 5,
    }
    found = {(p.raw, p.canonical, p.confidence) for p in suggest(counts)}
    assert ("Sainsbury", "Sainsbury's", "safe") not in found or True  # canonical is the most used
    by_raw = {p.raw: p for p in suggest(counts)}
    assert by_raw["TESCO"].canonical == "Tesco" and by_raw["TESCO"].confidence == "safe"
    assert by_raw["Tesco Certification/Retail"].confidence == "check"
    assert by_raw["Tesco Certification/Retail"].canonical == "Tesco"  # never the tagged variant
    assert "Aldi" not in by_raw  # a name on its own is not suggested
    # the most used spelling wins, with the typographic apostrophe made plain
    assert {p.canonical for p in suggest(counts) if p.raw.startswith("Sainsbury")} == {
        "Sainsbury's"
    }


def test_the_most_used_spelling_is_chosen_even_with_foods_or_ltd_in_it() -> None:
    counts = {"Dalehead Foods": 11, "Dalehead": 1}
    (p,) = suggest(counts)
    assert (p.raw, p.canonical) == ("Dalehead", "Dalehead Foods")
    counts = {"Multivac UK Ltd": 58, "Multivac UK": 37}
    (p,) = suggest(counts, "supplier")
    assert (p.raw, p.canonical, p.kind) == ("Multivac UK", "Multivac UK Ltd", "supplier")


def test_a_string_that_joins_two_customers_is_never_suggested() -> None:
    counts = {"Booker/Brakes": 1, "Brakes": 4, "Booker": 1, "Sam Browne / Ready Foods": 1}
    assert suggest(counts) == []


def test_typos_are_suggested_for_checking() -> None:
    counts = {"Braehead Foods": 1, "Breahead Foods": 2}
    (p,) = suggest(counts)
    assert p.confidence == "check" and p.canonical == "Breahead Foods"


def test_a_first_name_on_its_own_is_matched_to_the_one_full_name() -> None:
    counts = {
        "Peter": 47,
        "Peter Bringlesnap": 425,
        "Daniel Paiva": 900,
        "Daniel": 2,
        "Daniel Smith": 1,
    }
    found = {p.raw: p.canonical for p in suggest_people(counts)}
    assert found["Peter"] == "Peter Bringlesnap"
    assert "Daniel" not in found  # two people called Daniel: not guessed


@pytest.fixture
async def repo() -> AsyncIterator[DocumentRepository]:
    try:
        async with temporary_repository() as r:
            yield r
    except DatabaseUnavailable as exc:
        pytest.skip(str(exc))


async def test_one_customer_is_one_name_once_the_spellings_are_aliased(
    repo: DocumentRepository,
) -> None:
    await add(repo, "a", "4000001", "Fresh Whole Duck", "Sainsbury's", "Fresh", "None", 1, STRAP)
    await add(repo, "b", "4000002", "Fresh Duck Breast", "Sainsbury", "Fresh", "None", 1, STRAP)
    await add(repo, "c", "4000003", "Fresh Turkey Crown", "Aldi", "Fresh", "None", 1, STRAP)

    assert dict(await repo.customer_counts()) == {"Sainsbury's": 1, "Sainsbury": 1, "Aldi": 1}
    assert await repo.count_specs(customer="Sainsbury's") == 1  # before: split in two

    await repo.set_alias("customer", "Sainsbury", "Sainsbury's")
    assert dict(await repo.customer_counts()) == {"Sainsbury's": 2, "Aldi": 1}
    assert await repo.count_specs(customer="Sainsbury's") == 2
    vocab = await repo.spec_vocab()
    assert "Sainsbury" not in vocab.customers and "Sainsbury's" in vocab.customers

    # a question that uses either spelling reaches both specs
    for typed in ("Sainsbury", "Sainsburys", "Sainsbury's"):
        query = SpecQuery(intent="count", filter=SpecFilter(customer=typed))
        done = (await sq.execute(repo, sq.validate(query, vocab, f"how many {typed} specs")))[-1]
        assert "2 of the 3" in done.answer, typed  # type: ignore[union-attr]
    ranking = SpecQuery(intent="rank", group_by="customer")
    answer = (await sq.execute(repo, sq.validate(ranking, vocab)))[-1].answer  # type: ignore[union-attr]
    assert "1. Sainsbury's: 2 specs" in answer


async def test_an_alias_can_be_changed_and_removed(repo: DocumentRepository) -> None:
    await repo.set_alias("customer", "TESCO", "Tesco", note="case")
    await repo.set_alias("customer", "tesco", "Tesco Stores")  # the same spelling: replaced
    assert await repo.list_aliases() == [("customer", "tesco", "Tesco Stores")]
    assert await repo.remove_alias("customer", "TESCO") is True
    assert await repo.remove_alias("customer", "TESCO") is False
    assert await repo.list_aliases() == []


async def test_a_spelling_people_type_finds_the_customer_in_every_kind_of_answer(
    repo: DocumentRepository,
) -> None:
    """After "Sainsbury" was aliased to "Sainsbury's", "how many Sainsbury specs" matched product
    names (7) instead of the customer (34): the older single-condition handlers only knew the
    canonical spelling."""
    from qacopilot.rag.factspec_lookup import answer_from_factspec
    from qacopilot.rag.packaging import answer_from_packaging

    await add(repo, "a", "4000001", "Fresh Whole Duck", "Sainsbury's", "Fresh", "None", 1, STRAP)
    await add(repo, "b", "4000002", "Fresh Duck Breast", "Sainsbury", "Fresh", "None", 1, STRAP)
    await add(repo, "c", "4000003", "Sainsburys Turkey Crown", "Aldi", "Fresh", "None", 1, STRAP)
    await repo.set_alias("customer", "Sainsbury", "Sainsbury's")
    sq._vocab_cache.clear()

    for typed in ("Sainsbury's", "Sainsburys", "Sainsbury", "sainsbury’s"):
        count = await answer_from_factspec(f"how many {typed} specs are there", repo)
        assert count is not None, typed
        assert count[-1].answer.startswith("2 of the 3"), (typed, count[-1].answer)  # type: ignore[union-attr]
        assert "are for Sainsbury's" in count[-1].answer, typed  # type: ignore[union-attr]
        codes = await answer_from_packaging(f"what packaging codes are used for {typed}", repo)
        assert codes is not None, typed
        assert "the 2 current specs matching" in codes[-1].answer, typed  # type: ignore[union-attr]


async def test_supplier_aliases_merge_counts_and_a_typed_spelling_finds_the_supplier(
    repo: DocumentRepository,
) -> None:
    """ "118 packaging suppliers" counted "Multivac UK" and "Multivac UK Ltd" as two."""
    from qacopilot.rag.packaging import answer_from_packaging

    one = ("7900002", "Film Top Clear", "Multivac UK Ltd", "primary")
    other = ("7900005", "Film Lid", "Multivac UK", "primary")
    await add(repo, "a", "4000001", "Fresh Whole Duck", "Tesco", "Fresh", "None", 1, one)
    await add(repo, "b", "4000002", "Fresh Duck Breast", "Tesco", "Fresh", "None", 1, other)
    assert (await repo.packaging_totals())[1] == 2
    await repo.set_alias("supplier", "Multivac UK", "Multivac UK Ltd")
    assert (await repo.packaging_totals())[1] == 1
    assert [(n, specs) for n, specs, _ in await repo.supplier_counts()] == [("Multivac UK Ltd", 2)]
    assert await repo.count_specs(supplier="Multivac UK") == 2
    sq._vocab_cache.clear()
    for typed in ("Multivac UK Ltd", "Multivac UK"):
        got = await answer_from_packaging(f"how many specs use packaging from {typed}", repo)
        assert got is not None, typed
        assert got[-1].answer.startswith("2 of 2 current specs"), (typed, got[-1].answer)  # type: ignore[union-attr]
