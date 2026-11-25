"""What the LLM is asked to read out of a document.

Values are kept as printed (``"<10"``, ``"2.2E+03"``, ``"12 months"``); turning them into
decisions is the job of ``qacopilot.compliance``. Every date field says ISO ``YYYY-MM-DD`` in its
description because the model otherwise answers "14 March 2026".
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

ISO = "ISO 8601 format YYYY-MM-DD"


class CalibrationCert(BaseModel):
    instrument_serial: str = Field(
        min_length=1,
        description=(
            "Serial number of the instrument or machine itself, exactly as printed (look in "
            "equipment or machine details; not a conveyor, model or certificate number). "
            "Never empty."
        ),
    )
    instrument_description: str | None = Field(
        default=None, description="What the instrument is, e.g. 'Digital probe thermometer'"
    )
    certificate_number: str | None = Field(
        default=None, description="Certificate number, as printed"
    )
    calibrated_by: str | None = Field(
        default=None, description="Company that issued the certificate"
    )
    calibration_date: date = Field(
        description=(
            "Date the instrument was calibrated or tested. Labels vary: 'Date of calibration', "
            "'Date of test', 'Calibration check date', 'Valid From', or just 'Date'. If the "
            f"certificate prints only one date, this is it. {ISO}"
        )
    )
    next_due: date | None = Field(
        default=None,
        description=(
            "Date the next calibration is due, or the date this certificate expires. Labels vary: "
            "'Next due', 'Next test due', 'Suggested date for recalibration', 'Valid To', "
            f"'Expiry'. {ISO}. Null if no such date is printed, or if only a month is given."
        ),
    )
    result: Literal["pass", "fail", "unknown"] = Field(
        description="Overall result printed on the certificate; 'unknown' if it is not stated"
    )


class TestResult(BaseModel):
    __test__ = False  # not a pytest class

    test: str = Field(description="Name of the test, e.g. 'Total viable count'")
    result: str = Field(description="The result exactly as printed, e.g. '<10' or '2.2E+03'")
    unit: str | None = Field(default=None, description="Unit, e.g. 'cfu/g'")
    limit: str | None = Field(
        default=None, description="The limit printed beside the result, as printed; null if none"
    )


class CoA(BaseModel):
    product: str | None = Field(default=None, description="Product name")
    batch: str = Field(min_length=1, description="Batch or lot number, as printed")
    report_date: date | None = Field(
        default=None, description=f"Date of the report, {ISO}; null if not stated"
    )
    lab: str | None = Field(default=None, description="Laboratory that issued the report")
    results: list[TestResult] = Field(description="One entry per test row on the report")


class MicroLimit(BaseModel):
    test: str = Field(description="Name of the test, e.g. 'Enterobacteriaceae'")
    limit: str = Field(description="The limit exactly as printed, e.g. '1.0E+03 cfu/g'")


class SupplierSpec(BaseModel):
    supplier: str | None = Field(default=None, description="Supplier name")
    product: str = Field(min_length=1, description="Product name")
    product_code: str | None = Field(default=None, description="Product code, as printed")
    shelf_life: str | None = Field(
        default=None, description="Shelf life as printed, e.g. '14 days from pack date'"
    )
    storage: str | None = Field(default=None, description="Storage conditions, as printed")
    allergens: list[str] = Field(
        default_factory=list, description="Declared allergens; empty list if none are declared"
    )
    micro_limits: list[MicroLimit] = Field(
        default_factory=list, description="Microbiological limits table; empty if there is none"
    )
