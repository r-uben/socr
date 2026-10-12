"""#1085: leftovers of the #1084 evidence tokeniser.

1. Fail-open: an en/em dash before an evidence number must not support a model
   negative (``strip_presentation`` folds it to a minus; it is shared, so the
   scanned-evidence side drops such a token instead).
2. Fail-closed: punctuation behind a leading dot must not hide the number.
"""

from __future__ import annotations

from collections import Counter

import fitz
import pytest

from socr.tables.native_verifier import strip_presentation
from socr.tables.source_evidence import (
    _strip_word_edges,
    _tokens_from_plain_text,
    verify_scanned_table,
)


def _md(*cells: str) -> str:
    header = "| v | " + " | ".join(f"c{i}" for i in range(len(cells))) + " |"
    sep = "|" + "---|" * (len(cells) + 1)
    return "\n".join([header, sep, "| row | " + " | ".join(cells) + " |"])


def _verify(model_cells: tuple[str, ...], witness_text: str):
    doc = fitz.open()
    try:
        page = doc.new_page(width=500, height=700)
        return verify_scanned_table(page, _md(*model_cells), ocr_image_fn=lambda pix: witness_text)
    finally:
        doc.close()


@pytest.mark.parametrize("evidence", ["—5", "–5"])
@pytest.mark.parametrize("model", ["-5", "5"])
def test_dash_led_evidence_supports_neither_sign(evidence: str, model: str) -> None:
    assert not _verify((model,), evidence).passed


@pytest.mark.parametrize("evidence", ["-5", "−5"])
def test_genuine_minus_still_supports_negative(evidence: str) -> None:
    assert _verify(("-5",), evidence).passed
    assert not _verify(("5",), evidence).passed


def test_trailing_dash_decoration_still_supports_positive() -> None:
    assert _verify(("5",), "5—").passed


def test_strip_presentation_still_folds_dashes() -> None:
    # Shared predicate (11 call sites): the #1085 fix must live in the evidence
    # side only. If this fails someone changed the shared fold.
    assert strip_presentation("—5") == "-5"
    assert strip_presentation("–5") == "-5"
    assert strip_presentation("−5") == "-5"


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("...(0.025)", "0.025"),
        (".(45)", "45"),
        (".,45", "45"),
        ('."45"', "45"),
        (".[45]", "45"),
        (".45", ".45"),
        ("..5", "5"),
    ],
)
def test_strip_word_edges_behind_leading_dot(word: str, expected: str) -> None:
    assert _strip_word_edges(word) == expected


def test_leading_decimal_still_zero_prefixed_not_bare() -> None:
    assert _tokens_from_plain_text(".45")[0] == Counter({"0.45": 1})
    assert _tokens_from_plain_text(".(45)")[0] == Counter({"45": 1})
    assert _verify(("0.025",), "...(0.025)").passed


# ---- review P0s -------------------------------------------------------------------------


class _LayerPage:
    """A blank page whose text layer is *layer_text* (the base-14 fonts cannot
    write an em dash, so the layer is injected rather than drawn)."""

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


_HDR_MD = "| label | 999 |\n|---|---|\n| row | -5 |"


def test_dash_led_reading_does_not_route_into_distrusted_layer_rescue() -> None:
    # The witness READ "—5" and the layer prints "—5": emptying the bundle used to
    # activate the rescue, which folds the dash and ignores the header row (999).
    res = _verify_untrusted("—5", _HDR_MD, "—5")
    assert not res.passed


def test_rescue_ignores_dash_led_layer_words() -> None:
    # Witness silent (nothing read) but the layer has "—5": the rescue must not
    # bind the model's -5 to it.
    res = _verify_untrusted("—5", "| label | x |\n|---|---|\n| row | -5 |", "")
    assert not res.passed


def test_rescue_still_works_for_ordinary_layer() -> None:
    # Sanity: the rescue path itself is intact for a layer with a plain -5.
    res = _verify_untrusted("-5", "| label | x |\n|---|---|\n| row | -5 |", "")
    assert res.passed and "header_binding_unverified" in res.reason


@pytest.mark.parametrize(
    "evidence",
    ["&amp;amp;mdash;5", "&mdash;5", "&amp;mdash;5", "&#8212;5", "[—5](https://x/n)"],
)
def test_encoded_dash_led_evidence_supports_nothing(evidence: str) -> None:
    assert not _verify(("-5",), evidence).passed
    assert not _verify(("5",), evidence).passed


def test_ambiguous_reading_is_a_reading_not_an_absence() -> None:
    # The layer legitimately prints -5, so the rescue WOULD bind the row (and
    # ignore the invented header 999). It must not be reached: the witness read
    # "—5", which is a reading that fails to support, not silence.
    assert not _verify_untrusted("-5", _HDR_MD, "—5").passed
    # Control: a truly silent witness does reach the rescue on the same layer.
    assert _verify_untrusted("-5", _HDR_MD, "").passed


def _table(*vals: str) -> str:
    return "| label | x |\n|---|---|\n" + "\n".join(f"| r{i} | {v} |" for i, v in enumerate(vals))


def test_rescue_declines_when_layer_has_dash_led_row_among_many() -> None:
    # Layer: 49 rows then "—5"; the model prints the same plus "-5". The rescue
    # tolerates one unsupported number in fifty, so deleting the word would pass.
    layer = "\n".join(str(100 + i) for i in range(49)) + "\n—5"
    vals = [str(100 + i) for i in range(49)] + ["-5"]
    assert not _verify_untrusted(layer, _table(*vals), "").passed
    # Control: the same table against a plain-minus layer is rescued.
    assert _verify_untrusted(layer.replace("—", "-"), _table(*vals), "").passed


def test_rescue_declines_when_layer_dash_led_row_is_omitted_by_model() -> None:
    # Layer rows 100, —5, 200; the model drops the middle row. Filtering the
    # word made the omission invisible.
    layer = "100\n—5\n200"
    assert not _verify_untrusted(layer, _table("100", "200"), "").passed
