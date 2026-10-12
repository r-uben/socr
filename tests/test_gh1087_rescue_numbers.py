"""#1087: the distrusted-layer rescue (#658b) is removed.

Reviewers bypassed three successive number grammars for the rescue, so it was
deleted (owner, 2026-10-12; docs/log/2026-10-12_1087-rescue-numbers.md). A table
whose only support is a distrusted text layer fails closed: not verifiable.
The repros below were all PASSED by the rescue on origin/main or in review.
"""

from __future__ import annotations

import fitz

from socr.tables.row_corroboration import corroborate_rows
from socr.tables.source_evidence import verify_scanned_table


class _LayerPage:
    """A blank page whose text layer is *layer_text* (injected as words)."""

    def __init__(self, page, layer_text: str) -> None:
        self._page = page
        self._words = [
            (72.0 + 40 * j, 10.0 + 12 * i, 100.0 + 40 * j, 20.0 + 12 * i, w, 0, i, j)
            for i, line in enumerate(layer_text.splitlines())
            for j, w in enumerate(line.split())
        ]
        self._text = layer_text

    def get_text(self, mode: str = "text", *args, **kwargs):
        return self._words if mode == "words" else self._text

    def __getattr__(self, name):
        return getattr(self._page, name)


def _verify_untrusted(layer_text: str, md: str, witness_text: str):
    doc = fitz.open()
    try:
        page = _LayerPage(doc.new_page(width=500, height=700), layer_text)
        return verify_scanned_table(
            page, md, ocr_image_fn=lambda pix: witness_text, native_trusted=False
        )
    finally:
        doc.close()


def _table(*vals: str) -> str:
    return _md("x", *vals)


def _md(header: str, *vals: str) -> str:
    return f"| label | {header} |\n|---|---|\n" + "\n".join(
        f"| row{chr(97 + i % 26)} | {v} |" for i, v in enumerate(vals)
    )


def _fails_closed(layer: str, md: str) -> bool:
    res = _verify_untrusted(layer, md, "")
    return not res.passed and not res.verifiable and not res.content_unverified


def test_unprinted_header_number_is_not_shipped() -> None:
    assert _fails_closed("-5", _md("999", "-5"))


def test_unprinted_number_among_many_rows_is_not_shipped() -> None:
    layer = "\n".join(str(100 + i) for i in range(49)) + "\n7"
    assert _fails_closed(layer, _md("x", *[str(100 + i) for i in range(49)], "8"))


def test_mixed_cell_and_prose_numbers_are_not_shipped() -> None:
    assert _fails_closed("-5\n888kg", _md("Year 999", "-5", "888kg"))
    assert _fails_closed("-5\nTotal: 999.", _md("x", "-5") + "\n\nTotal: 999.")


def test_a_layer_reproducing_every_row_is_still_not_enough() -> None:
    assert _fails_closed("2021\n-5", _md("2021", "-5"))


def test_shared_corroborate_rows_unchanged() -> None:
    # ``corroborate_rows`` / ``clears`` stay for core/manifest.py: still tolerates
    # the header number, still clears a reproduced row.
    words = [(72.0, 10.0, 100.0, 20.0, "-5", 0, 0, 0)]
    assert corroborate_rows(words, _md("999", "-5"), None).clears is True
