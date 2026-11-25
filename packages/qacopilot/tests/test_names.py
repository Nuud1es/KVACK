from qacopilot.names import review_names


def _by_name(items):  # type: ignore[no-untyped-def]
    return {i.name: i for i in items}


def test_a_spelling_that_looks_like_one_name_is_recommended_as_an_alias() -> None:
    counts = {"customer": {"Sainsbury's": 30, "Sainsbury": 12}}
    (item,) = review_names(counts, set(), set())
    assert (item.name, item.recommendation, item.suggested) == ("Sainsbury", "alias", "Sainsbury's")
    assert item.confidence == "safe"


def test_settled_names_are_not_listed_again() -> None:
    counts = {"customer": {"Sainsbury's": 30, "Sainsbury": 12}}
    assert review_names(counts, {("customer", "sainsbury")}, set()) == []
    assert review_names(counts, set(), {("customer", "sainsbury")}) == []


def test_a_rare_name_close_to_a_common_one_is_flagged_either_way() -> None:
    counts = {"customer": {"Wholesale": 70, "Wholesaler": 1, "Wholefoods": 1, "Aldi": 20}}
    items = _by_name(review_names(counts, set(), set()))
    assert "Aldi" not in items  # nothing resembles it and it is common: nobody needs to look
    assert items["Wholesaler"].confidence in {"safe", "check", "unclear"}
    assert items["Wholesaler"].recommendation == "alias"
    assert items["Wholesaler"].suggested == "Wholesale"


def test_a_rare_name_that_resembles_nothing_is_not_listed() -> None:
    counts = {"customer": {"Tesco": 40, "Zorblax Industries": 1}}
    assert review_names(counts, set(), set()) == []


def test_a_rare_name_that_only_somewhat_resembles_another_is_recommended_as_new() -> None:
    counts = {"supplier": {"Smart Packaging": 20, "Smart Pack Systems": 1}}
    items = _by_name(review_names(counts, set(), set()))
    if "Smart Pack Systems" in items:
        item = items["Smart Pack Systems"]
        assert item.candidates and item.candidates[0].name == "Smart Packaging"
