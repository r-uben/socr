"""Math recovery: retained crops, syntax gating, and byte-preserving splice."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest

from socr.math import recover as recover_module
from socr.math.detect_equations import EquationDetectionResult, EquationRegion
from socr.math.recover import (
    CorruptMathRegion,
    clean_latex,
    corrupt_math_line_rects,
    recover_math_regions,
    splice_math,
)

# '=' -> '¼', '(' -> 'ð', ')' -> 'Þ', '+' -> 'þ'
_EQ = "PðA or BÞ ¼ PðAÞ þ PðBÞ"


def _page(lines: list[tuple[float, str]]):
    """Build a page; each (y, text) is one line at x=72."""
    doc = fitz.open()
    page = doc.new_page()
    for y, text in lines:
        page.insert_text((72, y), text, fontsize=10)
    return doc, page


def _region(
    source: str,
    latex: str,
    crop: str | None = "equations/crop.png",
    equation_label: str = "",
):
    return CorruptMathRegion(
        rect=fitz.Rect(1, 1, 2, 2),
        source_text=source,
        crop_path=crop,
        raw_latex=latex,
        validation_ok=bool(latex),
        validation_reason="ok" if latex else "engine returned empty output",
        model_id="fixture-model",
        equation_label=equation_label,
        attempts=1,
    )


def test_clean_latex_strips_fences_and_dollars():
    assert clean_latex("```latex\nP(A) = 1\n```") == "P(A) = 1"
    assert clean_latex("$$P(A) = 1$$") == "P(A) = 1"
    assert clean_latex("$x$\n$y$") == "x\ny"
    assert clean_latex("  P(A)=1  ") == "P(A)=1"


def test_corrupt_math_lines_detected_and_prose_skipped():
    _doc, page = _page(
        [
            (100, "This is clean prose with no math at all."),
            (140, _EQ),
            (180, "More clean prose here."),
        ]
    )
    assert len(corrupt_math_line_rects(page)) == 1


def test_adjacent_corrupt_lines_merge_into_one_region():
    _doc, page = _page([(100, _EQ), (112, _EQ), (124, _EQ)])
    assert len(corrupt_math_line_rects(page)) == 1


def test_adjacent_geometry_does_not_merge_noncontiguous_native_lines():
    """An intervening extraction-order line must survive fallback recovery."""
    first = "PðAÞ ¼ 1"
    second = "QðBÞ ¼ 2"
    page = MagicMock()

    def _get_text(mode: str):
        if mode == "text":
            return f"{first}\nintervening prose\n{second}\n"
        assert mode == "dict"
        return {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {"bbox": (100, 100, 300, 112), "spans": [{"text": first}]},
                        {
                            "bbox": (100, 111, 300, 123),
                            "spans": [{"text": "intervening prose"}],
                        },
                        {"bbox": (100, 116, 300, 128), "spans": [{"text": second}]},
                    ],
                }
            ]
        }

    page.get_text.side_effect = _get_text
    with patch(
        "socr.math.detect_equations.detect_display_equations",
        return_value=EquationDetectionResult(page_num=1),
    ):
        groups = recover_module._recovery_groups(page)

    assert [group.source_text for group in groups] == [first, second]


def test_distant_corrupt_lines_stay_separate():
    _doc, page = _page([(100, _EQ), (400, _EQ)])
    assert len(corrupt_math_line_rects(page)) == 2


def test_no_corrupt_math_returns_empty():
    _doc, page = _page([(100, "Ordinary prose."), (140, "f(x) = a + b clean.")])
    assert corrupt_math_line_rects(page) == []


def test_recovery_source_uses_the_same_native_text_cleaning_boundary():
    """Zero-width and exotic spaces cannot make a real corrupt line unalignable."""
    raw = "P\u00adðAÞ\u200b\u2009¼\u00a01   "
    cleaned = "PðAÞ ¼ 1"
    page = MagicMock()

    def _get_text(mode: str):
        if mode == "text":
            return f"{cleaned}\n"
        assert mode == "dict"
        return {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "bbox": (100, 100, 300, 112),
                            "spans": [{"text": raw}],
                        }
                    ],
                }
            ]
        }

    page.get_text.side_effect = _get_text
    with patch(
        "socr.math.detect_equations.detect_display_equations",
        return_value=EquationDetectionResult(page_num=1),
    ):
        groups = recover_module._recovery_groups(page)

    assert [group.source_text for group in groups] == [cleaned]


def test_numbered_and_unnumbered_corrupt_regions_coexist_on_one_page():
    """One numbered row must not suppress a separate unnumbered corrupt display."""
    numbered_text = "PðAÞ ¼ 1"
    unnumbered_text = "QðBÞ ¼ 2"
    page = MagicMock()

    def _get_text(mode: str):
        if mode == "text":
            return f"{numbered_text}\n(A8)\n{unnumbered_text}\n"
        assert mode == "dict"
        return {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "bbox": (100, 100, 300, 112),
                            "spans": [{"text": numbered_text}],
                        },
                        {
                            "bbox": (100, 300, 300, 312),
                            "spans": [{"text": unnumbered_text}],
                        },
                    ],
                }
            ]
        }

    page.get_text.side_effect = _get_text
    detected = EquationDetectionResult(
        page_num=1,
        regions=[
            EquationRegion(
                page_num=1,
                bbox=(96, 96, 552, 116),
                source_bbox=(100, 100, 548, 112),
                has_eq_number=True,
                equation_label="(A8)",
                source_text=f"{numbered_text}\n(A8)",
            )
        ],
    )

    with patch(
        "socr.math.detect_equations.detect_display_equations",
        return_value=detected,
    ):
        groups = recover_module._recovery_groups(page)

    assert [group.source_text for group in groups] == [
        f"{numbered_text}\n(A8)",
        unnumbered_text,
    ]


def test_unaligned_numbered_row_does_not_fall_back_to_partial_fragments():
    """A complete numbered crop fails closed instead of shipping fragment OCR."""
    corrupt_text = "PðAÞ ¼ 1"
    page = MagicMock()

    def _get_text(mode: str):
        if mode == "text":
            return f"{corrupt_text}\nintervening prose\n(A8)\n"
        assert mode == "dict"
        return {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "bbox": (100, 100, 300, 112),
                            "spans": [{"text": corrupt_text}],
                        }
                    ],
                }
            ]
        }

    page.get_text.side_effect = _get_text
    detected = EquationDetectionResult(
        page_num=1,
        regions=[
            EquationRegion(
                page_num=1,
                bbox=(96, 96, 552, 116),
                source_bbox=(100, 100, 548, 112),
                has_eq_number=True,
                equation_label="(A8)",
                source_text=f"{corrupt_text}\n(A8)",
            )
        ],
    )

    with patch(
        "socr.math.detect_equations.detect_display_equations",
        return_value=detected,
    ):
        groups = recover_module._recovery_groups(page)

    assert [group.source_text for group in groups] == [f"{corrupt_text}\n(A8)"]


def test_clean_numbered_row_is_not_sent_through_corrupt_math_recovery():
    """Page-level corruption does not license rereading every numbered equation."""
    corrupt_text = "PðAÞ ¼ 1"
    clean_text = "Q(B) = 2"
    page = MagicMock()
    page.get_text.return_value = f"{corrupt_text}\n(A8)\n{clean_text}\n(A9)\n"
    detected = EquationDetectionResult(
        page_num=1,
        regions=[
            EquationRegion(
                page_num=1,
                bbox=(96, 96, 552, 116),
                source_bbox=(100, 100, 548, 112),
                has_eq_number=True,
                equation_label="(A8)",
                source_text=f"{corrupt_text}\n(A8)",
            ),
            EquationRegion(
                page_num=1,
                bbox=(96, 196, 552, 216),
                source_bbox=(100, 200, 548, 212),
                has_eq_number=True,
                equation_label="(A9)",
                source_text=f"{clean_text}\n(A9)",
            ),
        ],
    )

    with patch(
        "socr.math.detect_equations.detect_display_equations",
        return_value=detected,
    ):
        groups = recover_module._numbered_math_groups(page)

    assert [group.source_text for group in groups] == [f"{corrupt_text}\n(A8)"]


def test_numbered_equation_uses_complete_exact_source_slice(tmp_path: Path):
    class _NumberedPage:
        rect = fitz.Rect(0, 0, 600, 800)
        _lines = [
            ("Prose equation (25) reference.", "Helvetica", (60, 40, 280, 52)),
            ("x", "CMMI10", (180, 100, 190, 112)),
            ("¼", "CMSY10", (230, 100, 240, 112)),
            ("y", "CMMI10", (280, 100, 290, 112)),
            ("(A8)", "Helvetica", (520, 100, 548, 112)),
        ]

        def get_fonts(self):
            return [
                (1, "cff", "Type1", "CMMI10", "CMMI10", ""),
                (2, "cff", "Type1", "CMSY10", "CMSY10", ""),
            ]

        def get_text(self, mode: str):
            if mode == "text":
                return "\n".join(text for text, _font, _bbox in self._lines) + "\n"
            assert mode == "dict"
            return {
                "blocks": [
                    {
                        "type": 0,
                        "bbox": bbox,
                        "lines": [
                            {
                                "bbox": bbox,
                                "spans": [{"text": text, "font": font, "bbox": bbox}],
                            }
                        ],
                    }
                    for text, font, bbox in self._lines
                ]
            }

        def get_pixmap(self, matrix=None, clip=None):
            class _Pixmap:
                @staticmethod
                def tobytes(kind: str):
                    assert kind == "png"
                    return b"png"

            return _Pixmap()

    page = _NumberedPage()
    [region] = recover_math_regions(
        page,
        ocr=lambda png, **kwargs: "x = y",
        crop_dir=tmp_path / "equations",
    )

    assert region.source_text == "x\n¼\ny\n(A8)"
    assert region.source_text in page.get_text("text")
    assert "equation (25)" not in region.source_text


def test_recover_retains_crop_and_validates_candidate(tmp_path: Path):
    _doc, page = _page([(100, "prose"), (140, _EQ)])
    regions = recover_math_regions(
        page,
        ocr=lambda png, **kw: "P(A) = 1",
        crop_dir=tmp_path / "equations",
        page_num=7,
    )

    assert len(regions) == 1
    result = regions[0]
    assert result.raw_latex == "P(A) = 1"
    assert result.validation_ok is True
    assert result.validation_reason == "ok"
    assert result.crop_path == "equations/corrupt_math_p00007_r001.png"
    assert (tmp_path / result.crop_path).is_file()


def test_policy_disabled_model_retains_crop_without_ocr(tmp_path: Path):
    _doc, page = _page([(100, _EQ)])
    ocr = MagicMock(return_value="must not run")

    [result] = recover_math_regions(
        page,
        ocr=ocr,
        crop_dir=tmp_path / "equations",
        model_disabled_reason="model call skipped by test policy",
    )

    ocr.assert_not_called()
    assert result.attempts == 0
    assert result.crop_path is not None
    assert (tmp_path / result.crop_path).is_file()
    assert result.validation_ok is False
    assert result.validation_reason == "model call skipped by test policy"


def test_policy_reason_survives_when_crop_cannot_be_retained():
    _doc, page = _page([(100, _EQ)])
    ocr = MagicMock(return_value="must not run")

    [result] = recover_math_regions(
        page,
        ocr=ocr,
        crop_dir=None,
        model_disabled_reason="model call skipped by test policy",
    )

    ocr.assert_not_called()
    assert result.crop_path is None
    assert result.validation_ok is False
    assert result.validation_reason == (
        "model call skipped by test policy; crop could not be retained as visual ground truth"
    )


def test_recover_retries_empty_and_fails_closed_with_crop(tmp_path: Path):
    _doc, page = _page([(100, _EQ)])
    calls = 0

    def empty_ocr(png, **kwargs):
        nonlocal calls
        calls += 1
        return ""

    regions = recover_math_regions(
        page,
        ocr=empty_ocr,
        crop_dir=tmp_path / "equations",
    )

    assert calls == 2
    assert len(regions) == 1
    result = regions[0]
    assert result.validation_ok is False
    assert "empty" in result.validation_reason
    assert result.crop_path is not None
    assert (tmp_path / result.crop_path).is_file()


def test_recover_rejects_structurally_invalid_latex(tmp_path: Path):
    _doc, page = _page([(100, _EQ)])
    [result] = recover_math_regions(
        page,
        ocr=lambda png, **kwargs: r"\frac{a}{b",
        crop_dir=tmp_path / "equations",
    )

    assert result.validation_ok is False
    assert result.validation_reason.startswith("parse error:")
    assert result.crop_path is not None


def test_recover_render_failure_remains_explicit():
    page = MagicMock()
    text_dict = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "bbox": (72, 90, 250, 105),
                        "spans": [{"text": _EQ}],
                    }
                ],
            }
        ]
    }
    page.get_text.side_effect = lambda mode: _EQ if mode == "text" else text_dict
    page.get_pixmap.side_effect = RuntimeError("render unavailable")
    ocr = MagicMock(return_value="must not run")

    [result] = recover_math_regions(page, ocr=ocr)

    ocr.assert_not_called()
    assert result.source_text == _EQ
    assert result.crop_path is None
    assert result.validation_ok is False
    assert result.validation_reason == "crop render failed: render unavailable"
    assert result.raw_latex == ""


def test_splice_replaces_only_exact_corrupt_substring():
    prefix = "  Clean prose line one.  "
    suffix = "Clean prose line two.\t"
    native = prefix + "\n" + _EQ + "\n" + suffix
    out = splice_math(None, native, [_region(_EQ, r"P(A \text{ or } B) = P(A) + P(B)")])

    assert out.startswith(prefix + "\n")
    assert out.endswith("\n" + suffix)
    assert r"P(A \text{ or } B) = P(A) + P(B)" in out
    assert "syntax only, non-authoritative" in out
    assert "![Corrupt equation crop](equations/crop.png)" in out
    assert "¼" not in out and "ð" not in out


@pytest.mark.parametrize(
    "candidate",
    [
        "x = y",
        "x = y \\quad (A8)",
        "x = y \\tag{A8}",
        "x = y \\tag{(A8)}",
    ],
)
def test_splice_uses_exact_source_equation_label_once(candidate: str):
    native = "Prose.\n" + _EQ
    region = _region(_EQ, candidate, equation_label="(A8)")

    out = splice_math(None, native, [region])

    assert out.count(r"\tag{A8}") == 1
    assert "(A8)" not in out.replace(r"\tag{A8}", "")


def test_splice_empty_latex_keeps_visible_failure_and_crop():
    native = "Prose.\n" + _EQ
    out = splice_math(None, native, [_region(_EQ, "")])

    assert "Prose." in out
    assert "![Corrupt equation crop](equations/crop.png)" in out
    assert "corrupt equation unresolved" in out
    assert "¼" not in out and "ð" not in out


def test_splice_no_crop_retains_native_source_bytes():
    native = "Prose bytes stay exact.\n" + _EQ
    region = _region(_EQ, r"x = y", crop=None)

    out = splice_math(None, native, [region])

    assert region.source_aligned is True
    assert region.resolved is False
    assert out.startswith(native)
    assert "crop could not be retained; native text retained" in out
    assert "no crop was retained" in out


def test_splice_alignment_failure_is_unresolved_even_with_valid_latex():
    native = "Prose bytes stay exact.\n" + _EQ
    region = _region("different extracted text", r"x = y")

    out = splice_math(None, native, [region])

    assert region.source_aligned is False
    assert region.resolved is False
    assert out.startswith(native)
    assert "source could not be aligned; native text retained" in out
    assert "corrupt equation unresolved" in out
    assert "![Corrupt equation crop](equations/crop.png)" in out


def test_splice_alignment_failure_preserves_trailing_native_bytes():
    native = "Prose bytes stay exact.\n" + _EQ + "\n \t\n"
    out = splice_math(None, native, [_region("different extracted text", r"x = y")])

    assert out[: len(native)] == native
    assert "source could not be aligned; native text retained" in out


def test_identical_source_occurrences_align_and_replace_independently():
    native = f"{_EQ}\nintervening prose\n{_EQ}"
    first = _region(_EQ, r"P(A) = 1", crop="equations/first.png")
    second = _region(_EQ, r"P(A) = 2", crop="equations/second.png")

    out = splice_math(None, native, [first, second])

    assert first.source_aligned is True
    assert second.source_aligned is True
    assert "![Corrupt equation crop](equations/first.png)" in out
    assert "![Corrupt equation crop](equations/second.png)" in out
    assert _EQ not in out
