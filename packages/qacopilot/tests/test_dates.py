from datetime import date

import pytest

from qacopilot.extraction.dates import NEXT_DUE, find_labelled_date


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Next calibration due: 2027-04-02", date(2027, 4, 2)),
        ("Next Calibration Due  02/04/2027", date(2027, 4, 2)),  # day first
        ("NEXT TEST DUE 2.4.27", date(2027, 4, 2)),
        ("Recalibration due: 2 April 2027", date(2027, 4, 2)),
        ("Suggested date for Recalibration: 12 August 2027", date(2027, 8, 12)),
        ("Valid To: 15.07.27", date(2027, 7, 15)),
        ("Valid until 15-Jul-2027", date(2027, 7, 15)),
        ("Expiry date:\n1st Sept 2027", date(2027, 9, 1)),
    ],
)
def test_labelled_dates_in_the_usual_uk_forms(text: str, expected: date) -> None:
    assert find_labelled_date(text, NEXT_DUE) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Date of calibration: 2026-04-02",  # not a next-due label
        "Next calibration due: not stated",
        "Next due: 31/02/2027",  # not a real date
        "Valid To: 15.13.27",  # month 13
        "Next calibration due",
        "",
    ],
)
def test_nothing_is_guessed(text: str) -> None:
    assert find_labelled_date(text, NEXT_DUE) is None


def test_a_far_away_date_after_the_label_is_not_taken() -> None:
    far = "Next calibration due: " + "x" * 80 + " 2027-04-02"
    assert find_labelled_date(far, NEXT_DUE) is None


def test_a_bad_date_does_not_hide_a_good_one_later() -> None:
    assert find_labelled_date("Next due 31/02/2027 or 2027-04-02", NEXT_DUE) == date(2027, 4, 2)
