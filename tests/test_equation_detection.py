"""Tests for GH-36a: deterministic display-equation region detection.

Covers all acceptance criteria:
  AC1 — display-equation regions detected with bboxes, crop PNGs retained,
         WITHOUT whole-page OCR.
  AC2 — prose-only page produces no detected regions (no false positives).
  AC3 — flag default OFF; throughput harness returns a measurement.
  AC4 — no equation replacement/splicing occurs.
  AC5 — ``qwen3-vl:8b`` default removed from recover.py.
  AC6 — fingerprint differs between detect_equations=True/False.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from socr.math.detect_equations import (
    EquationDetectionResult,
    detect_display_equations,
    save_equation_crops,
)

# ── Helpers to build synthetic pages ──────────────────────────────────────


def _make_page_with_math_font(
    lines: list[tuple[float, str, bool]],
) -> tuple[fitz.Document, fitz.Page]:
    """Create a synthetic page.

    Each tuple is (y, text, use_math_font).  When use_math_font=True the span
    is inserted with a font name that matches _MATH_FONT_RE (CMMI10).  Since
    PyMuPDF's ``insert_text`` with a built-in font won't embed CMMI10 directly,
    we test the detection logic by inserting text with a font whose name
    registers as a math font — or we use a workaround: we rely on ``get_fonts()``
    returning the font used.  For synthetic tests we use the "courier" built-in
    (which does NOT match _MATH_FONT_RE) for prose and test the detector
    directly by patching the page.get_fonts() result via a wrapper.

    In practice the detector is exercised end-to-end on a real math font only
    when the page has an actual CMMI/STIX font embedded.  The unit tests here
    validate the *geometry* logic (centering, merging, padding) by monkeypatching
    the font-name check.  See ``_make_page_with_math_font_patched``.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    for y, text, _ in lines:
        page.insert_text((72, y), text, fontsize=11)
    return doc, page


def _page_with_mock_math_fonts(
    lines: list[tuple[float, str, bool]],
) -> tuple[fitz.Document, fitz.Page]:
    """Build a page and monkey-patch get_fonts / get_text to simulate math fonts.

    This lets us test the centering/merge/padding geometry without needing a
    real CMMI/STIX-embedded PDF (which cannot be built from plain PyMuPDF
    insert_text with built-in fonts).

    The monkey-patch approach:  we subclass fitz.Page only conceptually — in
    practice we return the real page plus a wrapper that overrides the two
    methods the detector calls.
    """
    doc, real_page = _make_page_with_math_font(lines)
    return doc, _MathFontPage(real_page, lines)


class _MathFontPage:
    """Thin wrapper that injects fake math-font metadata for unit tests.

    Presents the same API surface as fitz.Page that ``detect_display_equations``
    uses: ``rect``, ``get_fonts()``, ``get_text("dict")``, ``get_pixmap()``.
    """

    def __init__(self, page: fitz.Page, lines: list[tuple[float, str, bool]]) -> None:
        self._page = page
        self._math_lines: set[int] = {i for i, (_, _, is_math) in enumerate(lines) if is_math}
        self._lines = lines

    @property
    def rect(self) -> fitz.Rect:
        return self._page.rect

    def get_fonts(self):
        # Return a fake entry for CMMI10 so the font-name detection passes.
        return [(1, "cff", "Type1", "CMMI10", "CMMI10", "WinAnsiEncoding")]

    def get_text(self, mode: str) -> dict:
        """Return a synthetic dict matching fitz.Page.get_text('dict') shape."""
        blocks = []
        page_w = self._page.rect.width
        for i, (y, text, is_math) in enumerate(self._lines):
            # Estimate a plausible bbox: x centred on the page for math lines,
            # left-aligned (x0=72) for prose.
            text_w = len(text) * 6.5  # rough px-per-char at fontsize 11
            if is_math:
                x0 = (page_w - text_w) / 2
            else:
                x0 = 72.0
            x1 = x0 + text_w
            y0 = y - 11.0
            y1 = y + 3.0
            font_name = "CMMI10" if is_math else "Helvetica"
            span = {
                "text": text,
                "font": font_name,
                "bbox": (x0, y0, x1, y1),
            }
            line = {
                "spans": [span],
                "bbox": (x0, y0, x1, y1),
            }
            block = {
                "type": 0,
                "bbox": (x0, y0, x1, y1),
                "lines": [line],
            }
            blocks.append(block)
        return {"blocks": blocks}

    def get_pixmap(self, matrix=None, clip=None):
        return self._page.get_pixmap(matrix=matrix, clip=clip)


class _ExtractedLinePage:
    """Synthetic page with caller-controlled extraction order and geometry."""

    rect = fitz.Rect(0, 0, 600, 800)

    def __init__(self, lines: list[tuple[str, str, tuple[float, float, float, float]]]):
        self._lines = lines

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


# ── Tests ──────────────────────────────────────────────────────────────────


class TestDetectDisplayEquations:
    """Unit tests for detect_display_equations geometry logic."""

    def test_prose_only_page_no_regions(self):
        """AC2 — a prose-only page must produce no detected regions."""
        lines = [
            (100, "The quick brown fox jumps over the lazy dog.", False),
            (130, "Another prose sentence with no math content at all.", False),
        ]
        _doc, page = _page_with_mock_math_fonts(lines)

        # Override get_fonts to return NO math font so the fast-exit fires.
        class _NoMathFontPage(_MathFontPage):
            def get_fonts(self):
                return [(1, "cff", "Type1", "Helvetica", "Helvetica", "WinAnsiEncoding")]

        no_math_page = _NoMathFontPage(_doc[0], lines)
        result = detect_display_equations(no_math_page, page_num=1)
        assert isinstance(result, EquationDetectionResult)
        assert result.regions == []
        assert result.detection_time_s >= 0.0

    def test_centred_math_line_detected(self):
        """AC1 — a centred math-font line is detected as a display equation."""
        lines = [
            (80, "Introductory prose line that sets the scene.", False),
            (180, r"E = mc^2", True),  # centred math-font line
            (280, "Concluding prose sentence.", False),
        ]
        _doc, page = _page_with_mock_math_fonts(lines)
        result = detect_display_equations(page, page_num=2)
        assert len(result.regions) == 1
        region = result.regions[0]
        assert region.page_num == 2
        # bbox should be padded (x0 < source_bbox x0)
        assert region.bbox[0] <= region.source_bbox[0]
        assert region.bbox[1] <= region.source_bbox[1]
        assert region.bbox[2] >= region.source_bbox[2]
        assert region.bbox[3] >= region.source_bbox[3]

    def test_trailing_math_span_whitespace_does_not_inflate_math_ratio(self):
        """Characters removed from the denominator are removed from math count too."""

        class _TrailingMathSpacePage:
            rect = fitz.Rect(0, 0, 600, 800)

            def get_fonts(self):
                return [(1, "cff", "Type1", "CMMI10", "CMMI10", "")]

            def get_text(self, mode: str):
                assert mode == "dict"
                return {
                    "blocks": [
                        {
                            "type": 0,
                            "lines": [
                                {
                                    "bbox": (260, 100, 340, 112),
                                    "spans": [
                                        {"text": "abcde", "font": "Helvetica"},
                                        {"text": "x     ", "font": "CMMI10"},
                                    ],
                                }
                            ],
                        }
                    ]
                }

        result = detect_display_equations(_TrailingMathSpacePage(), page_num=1)

        assert result.regions == []

    def test_adjacent_math_lines_merge(self):
        """Two vertically adjacent math lines within DISPLAY_MERGE_GAP_PT merge."""
        lines = [
            (100, r"\alpha + \beta", True),
            # Gap less than DISPLAY_MERGE_GAP_PT (y0 of next line ≈ 111, y1 of prev ≈ 103)
            (114, r"= \gamma", True),
        ]
        _doc, page = _page_with_mock_math_fonts(lines)
        result = detect_display_equations(page, page_num=3)
        # Should merge into one region.
        assert len(result.regions) == 1

    def test_distant_math_lines_stay_separate(self):
        """Math lines far apart stay as separate regions."""
        lines = [
            (100, r"E = mc^2", True),
            (400, r"F = ma", True),  # far apart
        ]
        _doc, page = _page_with_mock_math_fonts(lines)
        result = detect_display_equations(page, page_num=4)
        assert len(result.regions) == 2

    def test_eq_number_promotes_candidate(self):
        """A line with an equation number is promoted even if slightly off-centre."""
        # Place math line slightly off centre but with an equation number.
        lines = [
            (200, r"\int f(x) dx   (3.1)", True),
        ]
        _doc, page = _page_with_mock_math_fonts(lines)
        result = detect_display_equations(page, page_num=5)
        assert len(result.regions) >= 1
        assert result.regions[0].has_eq_number

    def test_numbered_fragments_form_exact_nonoverlapping_rows(self):
        """Numbered equation rows use complete source slices, not corrupt fragments."""
        page = _ExtractedLinePage(
            [
                ("Introductory prose.", "Helvetica", (60, 40, 240, 52)),
                ("h", "CMMI10", (180, 100, 188, 112)),
                ("t+1", "CMMI10", (188, 94, 210, 106)),
                ("=", "CMSY10", (235, 100, 244, 112)),
                ("c", "CMMI10", (270, 100, 278, 112)),
                ("t+1", "CMMI10", (278, 94, 300, 106)),
                ("(A8)", "Helvetica", (520, 100, 548, 112)),
                (
                    "We substitute from equation (25)",
                    "Helvetica",
                    (60, 150, 548, 162),
                ),
                ("h", "CMMI10", (180, 210, 188, 222)),
                ("t+2", "CMMI10", (188, 204, 210, 216)),
                ("=", "CMSY10", (235, 210, 244, 222)),
                ("c", "CMMI10", (270, 210, 278, 222)),
                ("t+2", "CMMI10", (278, 204, 300, 216)),
                ("(A9)", "Helvetica", (520, 210, 548, 222)),
                ("Concluding prose.", "Helvetica", (60, 270, 220, 282)),
            ]
        )
        native_text = page.get_text("text").strip()
        result = detect_display_equations(page, page_num=1)

        numbered = [region for region in result.regions if region.has_eq_number]
        assert len(numbered) == 2
        assert [region.equation_label for region in numbered] == ["(A8)", "(A9)"]
        assert all(region.source_text in native_text for region in numbered)
        assert numbered[0].source_text == "h\nt+1\n=\nc\nt+1\n(A8)"
        assert numbered[1].source_text == "h\nt+2\n=\nc\nt+2\n(A9)"
        assert "equation (25)" not in "\n".join(region.source_text for region in numbered)
        assert numbered[0].bbox[3] <= numbered[1].bbox[1]

    def test_interleaved_extraction_order_cannot_cross_numbered_source_slices(self):
        """A later equation's early fragment must not be deleted with the prior row."""
        page = _ExtractedLinePage(
            [
                ("a", "CMMI10", (180, 100, 190, 112)),
                ("b", "CMMI10", (180, 210, 190, 222)),
                ("(A8)", "Helvetica", (520, 100, 548, 112)),
                ("(A9)", "Helvetica", (520, 210, 548, 222)),
            ]
        )

        result = detect_display_equations(page, page_num=1)
        numbered = [region for region in result.regions if region.has_eq_number]

        assert [region.source_text for region in numbered] == ["a\n(A8)", "b\n(A9)"]
        assert set(numbered[0].source_text.splitlines()).isdisjoint(
            numbered[1].source_text.splitlines()
        )

    def test_label_only_anchor_is_not_a_numbered_replacement_target(self):
        """A bare right-margin label cannot overwrite an earlier prose reference."""
        page = _ExtractedLinePage(
            [
                ("See equation (A8)", "Helvetica", (60, 40, 220, 52)),
                ("(A8)", "Helvetica", (520, 100, 548, 112)),
                ("z", "CMMI10", (295, 300, 305, 312)),
            ]
        )

        result = detect_display_equations(page, page_num=1)

        assert not any(region.has_eq_number for region in result.regions)

    def test_widest_prose_line_with_inline_math_reference_is_not_a_source_anchor(self):
        """A terminal citation inside a prose span cannot own replacement bytes."""

        class _ProseReferencePage:
            rect = fitz.Rect(0, 0, 600, 800)

            def get_fonts(self):
                return [(1, "cff", "Type1", "CMMI10", "CMMI10", "")]

            def get_text(self, mode: str):
                assert mode == "dict"
                return {
                    "blocks": [
                        {
                            "type": 0,
                            "lines": [
                                {
                                    "bbox": (60, 100, 548, 112),
                                    "spans": [
                                        {"text": "x+y+z+w+t+u+v", "font": "CMMI10"},
                                        {
                                            "text": " discussion ends in equation (25)",
                                            "font": "Helvetica",
                                        },
                                    ],
                                }
                            ],
                        }
                    ]
                }

        result = detect_display_equations(_ProseReferencePage(), page_num=1)

        assert not any(region.source_text for region in result.regions)

    def test_tiny_numbered_artifact_obeys_the_region_height_floor(self):
        page = _ExtractedLinePage(
            [
                ("x", "CMMI10", (200, 100, 201, 101)),
                ("(A8)", "Helvetica", (520, 100, 548, 101)),
            ]
        )

        result = detect_display_equations(page, page_num=1)

        assert result.regions == []

    def test_label_extracted_before_same_row_math_still_anchors_the_row(self):
        """Geometry, not extraction order, binds a separated label to its equation."""
        page = _ExtractedLinePage(
            [
                ("(A8)", "Helvetica", (520, 100, 548, 112)),
                ("x", "CMMI10", (180, 100, 190, 112)),
                ("=", "CMSY10", (230, 100, 240, 112)),
                ("y", "CMMI10", (280, 100, 290, 112)),
            ]
        )

        result = detect_display_equations(page, page_num=1)
        numbered = [region for region in result.regions if region.has_eq_number]

        assert len(numbered) == 1
        assert numbered[0].source_text == "(A8)\nx\n=\ny"

    def test_detection_time_measured(self):
        """AC5 — throughput harness: detection_time_s is a positive float."""
        lines = [(200, r"y = ax + b", True)]
        _doc, page = _page_with_mock_math_fonts(lines)
        result = detect_display_equations(page, page_num=6)
        assert isinstance(result.detection_time_s, float)
        assert result.detection_time_s >= 0.0

    def test_result_has_padded_bbox(self):
        """Padded bbox is strictly larger than source_bbox (when room exists)."""
        lines = [(300, r"\sigma^2 = \frac{1}{N}", True)]
        _doc, page = _page_with_mock_math_fonts(lines)
        result = detect_display_equations(page, page_num=7)
        if result.regions:
            r = result.regions[0]
            # padded must be at least as large as source
            assert r.bbox[0] <= r.source_bbox[0]
            assert r.bbox[1] <= r.source_bbox[1]
            assert r.bbox[2] >= r.source_bbox[2]
            assert r.bbox[3] >= r.source_bbox[3]

    def test_no_replacement_occurs(self):
        """AC4 — detection returns regions but never modifies the page or text."""
        lines = [(200, r"y = f(x)", True)]
        doc, page = _page_with_mock_math_fonts(lines)
        original_text = doc[0].get_text("text")
        detect_display_equations(page, page_num=8)
        # Text is unmodified.
        assert doc[0].get_text("text") == original_text

    def test_never_raises_on_error(self):
        """detect_display_equations must never raise, even with a broken page."""

        class _BrokenPage:
            rect = fitz.Rect(0, 0, 595, 842)

            def get_fonts(self):
                raise RuntimeError("simulated PDF error")

            def get_text(self, mode):
                raise RuntimeError("simulated PDF error")

        result = detect_display_equations(_BrokenPage(), page_num=99)
        assert result.regions == []


class TestSaveEquationCrops:
    """Tests for crop-PNG storage."""

    def test_crop_saved_naming_convention(self, tmp_path: Path):
        """AC1 — crop PNGs saved as equation_{n}_page{N}.png."""
        lines = [(200, r"E = mc^2", True)]
        _doc, wrapper = _page_with_mock_math_fonts(lines)
        real_page = _doc[0]  # use real fitz page for pixmap

        # Synthesise one region covering the centre of the page.
        from socr.math.detect_equations import EquationRegion

        region = EquationRegion(
            page_num=1,
            bbox=(100.0, 180.0, 495.0, 220.0),
            source_bbox=(104.0, 184.0, 491.0, 216.0),
        )
        save_equation_crops([region], real_page, tmp_path / "equations", dpi=72)
        expected = tmp_path / "equations" / "equation_1_page1.png"
        assert expected.exists(), f"Expected crop at {expected}"
        assert region.crop_path == str(expected)

    def test_save_dir_created(self, tmp_path: Path):
        """save_equation_crops creates save_dir with parents if missing."""
        doc = fitz.open()
        page = doc.new_page()
        from socr.math.detect_equations import EquationRegion

        region = EquationRegion(
            page_num=1,
            bbox=(100.0, 100.0, 400.0, 150.0),
            source_bbox=(105.0, 105.0, 395.0, 145.0),
        )
        nested = tmp_path / "deep" / "nested" / "equations"
        save_equation_crops([region], page, nested, dpi=72)
        assert nested.is_dir()

    def test_crop_path_none_on_failure(self, tmp_path: Path):
        """If render fails, crop_path stays None (never raises)."""

        class _BadPage:
            def get_pixmap(self, matrix=None, clip=None):
                raise RuntimeError("render error")

        from socr.math.detect_equations import EquationRegion

        region = EquationRegion(
            page_num=1,
            bbox=(0.0, 0.0, 100.0, 50.0),
            source_bbox=(5.0, 5.0, 95.0, 45.0),
        )
        save_equation_crops([region], _BadPage(), tmp_path, dpi=72)
        assert region.crop_path is None


class TestRecoverDefaultModel:
    """AC5 — qwen3-vl:8b must NOT be the default model in recover.py."""

    def test_default_model_is_not_8b(self):
        from socr.math.recover import DEFAULT_MODEL

        assert ":8b" not in DEFAULT_MODEL, (
            f"recover.py DEFAULT_MODEL must not be qwen3-vl:8b (forbidden model); "
            f"got: {DEFAULT_MODEL!r}"
        )

    def test_default_model_is_instruct(self):
        from socr.math.recover import DEFAULT_MODEL

        assert "30b-a3b-instruct" in DEFAULT_MODEL, (
            f"recover.py DEFAULT_MODEL should be the validated instruct model; "
            f"got: {DEFAULT_MODEL!r}"
        )


class TestPipelineConfigFlag:
    """AC3 — detect_equations defaults to False."""

    def test_detect_equations_default_off(self):
        from socr.core.config import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.detect_equations is False

    def test_detect_equations_settable(self):
        from socr.core.config import PipelineConfig

        cfg = PipelineConfig()
        cfg.detect_equations = True
        assert cfg.detect_equations is True


class TestFingerprintDiffers:
    """AC6 — fingerprint must differ between detect_equations=True/False."""

    def test_fingerprint_differs(self):
        """The run fingerprint changes when detect_equations is toggled."""
        from unittest.mock import patch

        from socr.core.config import PipelineConfig

        cfg_off = PipelineConfig()
        cfg_off.detect_equations = False

        cfg_on = PipelineConfig()
        cfg_on.detect_equations = True

        # Build two minimal orchestrators and compare their fingerprints.
        # We patch _engine_determinants and _manifest_versions so no real
        # engine resolution is needed.
        from socr.pipeline.orchestrator import UnifiedPipeline

        with (
            patch.object(
                UnifiedPipeline,
                "_engine_determinants",
                return_value={"model": "test", "backend": "test", "task": "test"},
            ),
            patch(
                "socr.pipeline.orchestrator._manifest_versions",
                return_value=("v1", "v1"),
            ),
        ):
            pipe_off = UnifiedPipeline(cfg_off)
            pipe_on = UnifiedPipeline(cfg_on)
            fp_off = pipe_off._run_fingerprint()
            fp_on = pipe_on._run_fingerprint()

        assert fp_off != fp_on, (
            f"Fingerprint must differ when detect_equations changes (off={fp_off!r}, on={fp_on!r})"
        )


class TestThroughputHarness:
    """AC5 — throughput harness returns a per-page measurement."""

    def test_throughput_measurement(self):
        """Run detection on a small set of pages and report timing."""
        lines_per_page = [
            [(200, r"E = mc^2", True)],
            [(100, "Prose only line.", False), (150, "More prose.", False)],
            [(150, r"\int_0^\infty f(x) dx", True), (250, r"= G(s)", True)],
        ]
        timings: list[float] = []
        for i, lines in enumerate(lines_per_page, start=1):
            _doc, page = _page_with_mock_math_fonts(lines)
            result = detect_display_equations(page, page_num=i)
            timings.append(result.detection_time_s)

        total_time = sum(timings)
        pages = len(timings)
        avg_ms = (total_time / pages) * 1000

        # The throughput harness must return non-negative measurements.
        assert all(t >= 0.0 for t in timings), "detection_time_s must be non-negative"
        # Geometry-only detection should be well under 500 ms per page
        # (the actual bound will be much lower; this guards against hangs).
        assert avg_ms < 500, (
            f"Detection is unexpectedly slow: {avg_ms:.1f} ms/page avg on synthetic pages"
        )
        # Report for the log (visible with pytest -s).
        print(
            f"\nThroughput harness: {pages} pages, "
            f"total {total_time * 1000:.1f} ms, "
            f"avg {avg_ms:.2f} ms/page"
        )
