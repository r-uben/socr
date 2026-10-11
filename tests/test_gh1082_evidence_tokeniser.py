"""#1082 step 1 (reduced): the evidence tokeniser must never gain support for a
number the page does not print.

An adversarial review reproduced four fail-open paths in the first draft (minus
fold, decimal comma, pair split, edge stripping). Those rules were withdrawn;
the counterexamples below are regression guards and must all end NOT passed.
Only the leading-dot fix remains (docs/log/2026-10-11_1082-evidence-tokeniser.md).
"""

from __future__ import annotations

from collections import Counter

import fitz
import pytest

from socr.tables.source_evidence import (
    _tokens_from_plain_text,
    build_scanned_evidence,
    collect_table_tokens,
    verify_scanned_table,
)


def _md(*cells: str) -> str:
    header = "| v | " + " | ".join(f"c{i}" for i in range(len(cells))) + " |"
    sep = "|" + "---|" * (len(cells) + 1)
    return "\n".join([header, sep, "| row | " + " | ".join(cells) + " |"])


def _verify(model_cells: tuple[str, ...], witness_text: str):
    page = fitz.open().new_page(width=500, height=700)
    return verify_scanned_table(page, _md(*model_cells), ocr_image_fn=lambda pix: witness_text)


# ---- the one rule kept ----------------------------------------------------------------


def test_leading_decimal_point_is_not_eaten() -> None:
    assert _tokens_from_plain_text(".45")[0] == Counter({"0.45": 1})
    assert _tokens_from_plain_text("(.45,")[0] == Counter({"0.45": 1})


def test_eaten_leading_dot_no_longer_supports_the_bare_integer() -> None:
    # Old strip: ".45" -> "45", which supported a model 45 the page never printed.
    assert not _verify(("45",), ".45").passed
    assert _verify(("0.45",), ".45").passed


def test_non_decimal_leading_dots_still_stripped() -> None:
    assert _tokens_from_plain_text("..5 .Net")[0] == Counter({"5": 1})


# ---- review counterexamples: each must NOT pass ---------------------------------------


def test_markdown_link_in_model_cell_is_still_checked() -> None:
    assert not _verify(("[999](https://x/note)", "1"), "1").passed
    tokens = collect_table_tokens(_md("[999](https://x/note)", "1"))
    assert tokens is not None and tokens.numeric["999"] == 1


@pytest.mark.parametrize("evidence", ["~5", "--5--", "———5"])
def test_decorated_positive_does_not_support_negative_model(evidence: str) -> None:
    assert not _verify(("-5",), evidence).passed


def test_decimal_comma_number_does_not_support_two_numbers() -> None:
    assert not _verify(("1.234", "56"), "1.234,56").passed


def test_comma_list_does_not_support_a_decimal() -> None:
    assert not _verify(("1.5",), "1,5").passed


# ---- guards ---------------------------------------------------------------------------


def test_invented_number_and_count_shortfall_fail() -> None:
    assert not _verify(("0.94", "7.77"), "0.94").passed
    assert not _verify(("0.94", "0.94"), "0.94").passed


# ---- text layer gets no OCR-only treatment --------------------------------------------


def test_text_layer_path_has_no_ocr_only_treatment() -> None:
    page = fitz.open().new_page(width=500, height=700)
    page.insert_text((72, 100), "~5 --7-- 1,5")
    bundle = build_scanned_evidence(page, ocr_image_fn=lambda pix: "")
    assert "-5" not in bundle.numeric
    assert "-7" not in bundle.numeric
    assert "1.5" not in bundle.numeric
