import pytest

from qacopilot.compliance import Limit, Verdict, evaluate_result, parse_limit, parse_numeric_value


@pytest.mark.parametrize(
    ("text", "number", "qualifier"),
    [
        ("10", 10.0, None),
        ("<10", 10.0, "<"),
        ("< 10 cfu/g", 10.0, "<"),
        (">1000", 1000.0, ">"),
        ("2.20E+03", 2200.0, None),
        ("<1.0E+05 cfu/g", 100000.0, "<"),
        ("1,200", 1200.0, None),
        ("<1,000 cfu/g", 1000.0, "<"),
        ("1,234,567.5", 1234567.5, None),
        ("-3", -3.0, None),
    ],
)
def test_parse_numeric_value(text: str, number: float, qualifier: str | None) -> None:
    v = parse_numeric_value(text)
    assert v is not None and v.number == number and v.qualifier == qualifier


@pytest.mark.parametrize("text", ["", "n/a", "Not detected", "abc 10", "1,5", "12,5", "1,23"])
def test_parse_numeric_value_rejects_non_numbers(text: str) -> None:
    assert parse_numeric_value(text) is None


def test_parse_limit() -> None:
    assert parse_limit("< 1.0E+05 cfu/g") == Limit(kind="max", number=1e5, text="< 1.0E+05 cfu/g")
    assert parse_limit("1.0E+03 cfu/g") is not None  # bare number = maximum
    assert parse_limit("> 5")  == Limit(kind="min", number=5.0, text="> 5")  # fmt: skip
    absent = parse_limit("Not detected in 25g")
    assert absent is not None and absent.kind == "absent"
    assert parse_limit("see supplier") is None


MAX_1000 = Limit(kind="max", number=1000.0)
MIN_5 = Limit(kind="min", number=5.0)
ABSENT = Limit(kind="absent")


@pytest.mark.parametrize(
    ("result", "limit", "verdict"),
    [
        ("2.2E+03", MAX_1000, Verdict.FAIL),  # the CoA example: 2200 > 1000
        ("999", MAX_1000, Verdict.PASS),
        ("1000", MAX_1000, Verdict.PASS),  # at the limit passes
        ("<10", MAX_1000, Verdict.PASS),
        ("<10000", MAX_1000, Verdict.REVIEW),  # could be either
        (">100000", MAX_1000, Verdict.FAIL),
        (">500", MAX_1000, Verdict.REVIEW),
        ("Not detected", MAX_1000, Verdict.PASS),
        ("see report", MAX_1000, Verdict.REVIEW),
        ("5", MIN_5, Verdict.PASS),
        ("4.9", MIN_5, Verdict.FAIL),
        (">6", MIN_5, Verdict.PASS),
        (">2", MIN_5, Verdict.REVIEW),
        ("<5", MIN_5, Verdict.FAIL),
        ("<8", MIN_5, Verdict.REVIEW),
        ("Not detected in 25g", ABSENT, Verdict.PASS),
        ("ND", ABSENT, Verdict.PASS),
        ("Detected", ABSENT, Verdict.FAIL),
        ("Positive", ABSENT, Verdict.FAIL),
        ("0", ABSENT, Verdict.PASS),
        ("10", ABSENT, Verdict.FAIL),
        ("<10", ABSENT, Verdict.REVIEW),
        ("??", ABSENT, Verdict.REVIEW),
    ],
)
def test_evaluate_result(result: str, limit: Limit, verdict: Verdict) -> None:
    assert evaluate_result(result, limit)[0] is verdict


def test_reason_is_given() -> None:
    verdict, reason = evaluate_result("2.2E+03", MAX_1000)
    assert verdict is Verdict.FAIL and "2200" in reason and "1000" in reason


def test_thousands_separator_cannot_turn_a_fail_into_a_pass() -> None:
    assert evaluate_result("1,200", MAX_1000)[0] is Verdict.FAIL  # not 1.2
    assert evaluate_result("1,5", MAX_1000)[0] is Verdict.REVIEW  # ambiguous, so a person decides


# The toolkit's evaluate_result(value, pass_max, warning_max) is two-tier. With both thresholds
# set to the limit it agrees with this checker wherever the answer is certain...
@pytest.mark.parametrize(
    ("result", "toolkit"),
    [("999", "pass"), ("1000", "pass"), ("1001", "fail"), ("<10", "pass"), ("<1000", "pass"),
     (">5000", "fail"), ("2.2E+03", "fail"), ("2.2E+02", "pass")],
)  # fmt: skip
def test_agrees_with_the_toolkit_where_the_answer_is_certain(result: str, toolkit: str) -> None:
    assert evaluate_result(result, MAX_1000)[0].value == toolkit


# ...and is deliberately more cautious where a censored result cannot decide the question.
@pytest.mark.parametrize(
    ("result", "toolkit", "here"),
    [
        ("<10000", "fail", Verdict.REVIEW),  # true value may well be under the limit
        (">500", "pass", Verdict.REVIEW),  # true value could exceed it
        (">1000", "pass", Verdict.FAIL),  # above 1000 can never meet a limit of 1000
    ],
)
def test_documented_differences_from_the_toolkit(result: str, toolkit: str, here: Verdict) -> None:
    assert evaluate_result(result, MAX_1000)[0] is here
