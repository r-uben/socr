from pathlib import Path

import pytest
pytest.importorskip("rich")

from socr.audit.heuristics import HeuristicsChecker
from socr.core.result import PageResult, PageStatus


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
    page = PageResult(page_num=1, text="ok", status=PageStatus.SUCCESS, confidence=0.5, audit_passed=False)
    assert page.needs_reprocessing() is True
    page.audit_passed = True
    assert page.needs_reprocessing() is False
