import pytest

pytest.importorskip("rich")

from socr.audit.heuristics import HeuristicsChecker
from socr.core.result import PageOutput, PageStatus

_FLAT_TABLE_TEXT = "\n".join(
    [
        "Forecast",
        "2026",
        "2027",
        "2028",
        "CountryA",
        "1.2",
        "1.3",
        "1.4",
        "CountryB",
        "2.1",
        "2.2",
        "2.3",
        "CountryC",
        "3.1",
        "3.2",
        "3.3",
    ]
)


def test_heuristics_flags_low_word_count_and_garbage() -> None:
    checker = HeuristicsChecker(min_word_count=10, max_garbage_ratio=0.1)
    text = "###\n" * 5  # low words, lots of garbage chars
    result = checker.check(text)
    assert result.passed is False
    assert any("Word count" in m.name for m in result.metrics)
    assert any("Garbage ratio" in m.name for m in result.metrics)


def test_heuristics_passes_clean_text() -> None:
    checker = HeuristicsChecker(min_word_count=3)
    text = "This is a short clean paragraph."
    result = checker.check(text)
    assert result.passed is True
    assert all(m.passed or m.severity == "info" for m in result.metrics)


def test_page_reprocessing_logic() -> None:
    page = PageOutput(
        page_num=1, text="ok", status=PageStatus.SUCCESS, confidence=0.5, audit_passed=False
    )
    assert page.needs_reprocessing() is True
    page.audit_passed = True
    assert page.needs_reprocessing() is False


def test_native_table_structure_flags_flat_cell_stream() -> None:
    result = HeuristicsChecker().check_native_table_structure(_FLAT_TABLE_TEXT)

    assert result.passed is False
    assert result.metrics[0].name == HeuristicsChecker.NATIVE_TABLE_STRUCTURE_METRIC


def test_native_table_structure_accepts_markdown_table() -> None:
    text = (
        "| Forecast | 2026 | 2027 | 2028 |\n"
        "| --- | --- | --- | --- |\n"
        "| CountryA | 1.2 | 1.3 | 1.4 |\n"
        "| CountryB | 2.1 | 2.2 | 2.3 |"
    )

    assert HeuristicsChecker().check_native_table_structure(text).passed is True


def test_native_table_structure_flags_malformed_markdown_table() -> None:
    text = "\n".join(
        [
            "| Name | 2026 | 2027 | 2028 | 2029 | 2030 |",
            "| --- | --- | --- | --- | --- | --- |",
            "| collapsed labels | 1.0 2.0 3.0 4.0 5.0 |",
            "| more labels | 1.1 2.1 3.1 4.1 5.1 |",
            "| another row | 1.2 2.2 3.2 4.2 5.2 |",
            "| final row | 1.3 2.3 3.3 4.3 5.3 |",
        ]
    )

    result = HeuristicsChecker().check_native_table_structure(text)

    assert result.passed is False
    assert "malformed Markdown" in result.metrics[0].value


def test_native_table_structure_accepts_markdown_with_section_and_footnote_rows() -> None:
    text = "\n".join(
        [
            "| Name | 2026 | 2027 | 2028 | 2029 | 2030 |",
            "| --- | --- | --- | --- | --- | --- |",
            "| CountryA | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 |",
            "| CountryB | 1.1 | 2.1 | 3.1 | 4.1 | 5.1 |",
            "| CountryC | 1.2 | 2.2 | 3.2 | 4.2 | 5.2 |",
            "| CountryD | 1.3 | 2.3 | 3.3 | 4.3 | 5.3 |",
            "| Panel note |",
            "| Sources cover 2026 2027 vintages |",
        ]
    )

    assert HeuristicsChecker().check_native_table_structure(text).passed is True


def test_native_table_structure_accepts_aligned_plain_rows() -> None:
    text = "\n".join(
        [
            "Forecast 2026 2027 2028",
            "CountryA 1.2 1.3 1.4",
            "CountryB 2.1 2.2 2.3",
            "CountryC 3.1 3.2 3.3",
        ]
    )

    assert HeuristicsChecker().check_native_table_structure(text).passed is True
