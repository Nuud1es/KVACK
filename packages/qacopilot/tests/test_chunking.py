from qacopilot.ingest import DocType, PageText, ParsedDocument, chunk_document


def _doc(*pages: str) -> ParsedDocument:
    return ParsedDocument(
        source="t.pdf",
        pages=[PageText(page=i, text=t, method="text") for i, t in enumerate(pages, start=1)],
    )


SOP = _doc(
    "SOP-014 Cooking and release\n\n"
    "1. Purpose\nEnsure cooked duck reaches the release temperature.\n\n"
    "2. Procedure\n"
    "2.1 Probing\nProbe the thickest part of the product.\n\n"
    "2.2 Release\nProduct must reach 75C core before release.",
    "3. Records\nRecord the result on form GF-12.",
)


def test_sop_chunks_are_heading_aware_with_parent_path_and_page() -> None:
    chunks = chunk_document(SOP, DocType.SOP, "sop14")
    by_section = {c.section: c for c in chunks}
    release = by_section["2. Procedure > 2.2 Release"]
    assert release.text.startswith("SOP-014 Cooking and release > 2. Procedure > 2.2 Release")
    assert "75C core" in release.text
    assert release.page == 1
    assert by_section["3. Records"].page == 2
    assert "Purpose" not in release.text


def test_title_line_is_only_in_the_prefix() -> None:
    chunks = chunk_document(SOP, DocType.SOP, "sop14")
    assert all(c.text.count("SOP-014 Cooking and release") == 1 for c in chunks)


def test_chunk_ids_are_stable_and_unique() -> None:
    a = chunk_document(SOP, DocType.SOP, "sop14")
    b = chunk_document(SOP, DocType.SOP, "sop14")
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)
    assert {c.id for c in a}.isdisjoint(c.id for c in chunk_document(SOP, DocType.SOP, "other"))


def test_cert_types_are_one_record_per_page() -> None:
    doc = _doc("CERTIFICATE OF ANALYSIS\nProduct: Duck   Batch: 1", "Page two results", "")
    chunks = chunk_document(doc, DocType.COA, "coa1")
    assert [c.page for c in chunks] == [1, 2]  # blank page skipped
    assert chunks[0].text.startswith("CERTIFICATE OF ANALYSIS")
    assert chunks[0].metadata["product"] == "Duck"


def test_spec_tables_stay_intact_and_metadata_is_on_every_chunk() -> None:
    table = "Test      Result    Limit\nTVC       100       1000\nEntero    <10       100"
    doc = _doc(
        "Product specification\nSupplier: Example Sauces Ltd\nProduct: Orange sauce\n\n"
        "MICROBIOLOGY\n" + table + "\n\nSHELF LIFE\n12 months ambient."
    )
    chunks = chunk_document(doc, DocType.SUPPLIER_SPEC, "spec1", max_chars=60)
    micro = next(c for c in chunks if c.section == "MICROBIOLOGY")
    assert table in micro.text  # not split, even though it exceeds max_chars
    assert all(c.metadata["supplier"] == "Example Sauces Ltd" for c in chunks)
    assert all(c.metadata["product"] == "Orange sauce" for c in chunks)


def test_long_paragraph_is_split_under_max_chars() -> None:
    sentence = "The line supervisor must check the probe before each shift. "
    doc = _doc("Title\n\n1. Checks\n" + sentence * 30)
    chunks = chunk_document(doc, DocType.SOP, "long", max_chars=300)
    assert len(chunks) > 3
    assert all(len(c.text) < 300 + 100 for c in chunks)
    assert all(c.section == "1. Checks" for c in chunks)


def test_customer_cop_is_chunked_per_clause() -> None:
    doc = _doc(
        "Retailer Code of Practice\n"
        "1.1 Suppliers shall hold current BRCGS certification.\n"
        "1.2 Cooked product shall not exceed 1000 cfu/g TVC at release.\n"
        "continued on the next line.\n"
        "2.1 Metal detection shall be verified hourly."
    )
    chunks = chunk_document(doc, DocType.CUSTOMER_COP, "cop")
    assert [c.clause for c in chunks] == ["1.1", "1.2", "2.1"]
    assert "continued on the next line" in chunks[1].text
    assert chunks[1].text.startswith("Retailer Code of Practice > Clause 1.2")


def test_empty_document_gives_no_chunks() -> None:
    assert chunk_document(_doc(""), DocType.SOP, "e") == []


def test_body_line_starting_with_a_number_is_not_a_heading() -> None:
    doc = _doc("Title\n\n1. Storage\nKeep chilled.\n12 months from pack date.")
    chunks = chunk_document(doc, DocType.SOP, "n")
    assert len(chunks) == 1
    assert "12 months from pack date" in chunks[0].text
