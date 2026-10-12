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


def test_dash_led_reading_with_matching_layer_fails_closed() -> None:
    # The witness READ "—5" and the layer prints "—5" (#1087: no rescue exists).
    res = _verify_untrusted("—5", _HDR_MD, "—5")
    assert not res.passed


def test_dash_led_layer_words_do_not_support_a_negative() -> None:
    # Witness silent but the layer has "—5": nothing binds the model's -5 to it.
    res = _verify_untrusted("—5", "| label | x |\n|---|---|\n| row | -5 |", "")
    assert not res.passed


def test_layer_alone_never_passes_a_table_when_silent() -> None:
    # No rescue exists (#1087): a layer printing -5 and a silent witness fail closed.
    res = _verify_untrusted("-5", "| label | x |\n|---|---|\n| row | -5 |", "")
    assert not res.passed and not res.verifiable


@pytest.mark.parametrize(
    "evidence",
    ["&amp;amp;mdash;5", "&mdash;5", "&amp;mdash;5", "&#8212;5", "[—5](https://x/n)"],
)
def test_encoded_dash_led_evidence_supports_nothing(evidence: str) -> None:
    assert not _verify(("-5",), evidence).passed
    assert not _verify(("5",), evidence).passed


def test_ambiguous_reading_is_a_reading_not_an_absence() -> None:
    # The witness read "—5": a reading that fails to support, not silence.
    ok_md = _HDR_MD.replace("999", "x")
    assert not _verify_untrusted("-5", ok_md, "—5").passed
    # The outcome cannot tell a reading from silence (both reject), so the bundle
    # is checked: a dash-led reading counts as content evidence, silence does not.
    from socr.tables.source_evidence import build_scanned_evidence

    doc = fitz.open()
    try:
        page = doc.new_page(width=500, height=700)
        read = build_scanned_evidence(page, ocr_image_fn=lambda pix: "—5", include_text_layer=False)
        silent = build_scanned_evidence(page, ocr_image_fn=lambda pix: "", include_text_layer=False)
    finally:
        doc.close()
    assert read.has_content_evidence and read.ambiguous_numeric == 1
    assert not silent.has_content_evidence


def _table(*vals: str) -> str:
    return "| label | x |\n|---|---|\n" + "\n".join(
        f"| row{chr(97 + i % 26)} | {v} |" for i, v in enumerate(vals)
    )


def test_layer_dash_led_row_among_many_does_not_pass() -> None:
    # Layer: 49 rows then "—5"; the model prints the same plus "-5".
    layer = "\n".join(str(100 + i) for i in range(49)) + "\n—5"
    vals = [str(100 + i) for i in range(49)] + ["-5"]
    assert not _verify_untrusted(layer, _table(*vals), "").passed


def test_layer_dash_led_row_omitted_by_model_does_not_pass() -> None:
    # Layer rows 100, —5, 200; the model drops the middle row.
    layer = "100\n—5\n200"
    assert not _verify_untrusted(layer, _table("100", "200"), "").passed
