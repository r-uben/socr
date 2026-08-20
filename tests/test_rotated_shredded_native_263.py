"""#263: a rotated page with NO detected table must not ship shredded native text.

The GH-147 rotation refusal lives inside ``if has_tables:`` in
``born_digital.py``, so ``text_direction_is_rotated`` was never consulted on a
rotated page without a table. On a rotated *figure* page the native layer is
character-level confetti -- and it shipped under a clean ``SUCCESS``.

Reference defect (Kaminska-Mumtaz-Sustek p38, measured 2026-08-20): 177 chars
of native text across 47 lines, 32 of them two characters or fewer --
``MC / O / F / round / a / ields / y / n / i / anges / h / c / requency``. The
caption is unreadable and no figure is referenced.

The mechanism, read off that page's own ``get_text("dict")``: the caption is
typeset as one PyMuPDF *line* per glyph run, all in the same x-column, each run
placed BEHIND the previous one in the writing direction ``(0, -1)``. PyMuPDF
starts a new line whenever the next run is not ahead of the current one along
the text direction, so the caption is split at every run boundary and emitted
in y-order, i.e. reversed. The fixtures below reproduce exactly that geometry;
the control places the identical runs in reading order and extracts the caption
verbatim.

Four binding groups:

  1. Fixture integrity. Every glyph box lies inside the page rectangle (a
     previous attempt at this ticket ran its rotated text off the top of the
     page, so the PDF clipped it and a sound extraction *looked* shredded), and
     the unrotated control reads back byte-for-byte.
  2. Detector. The shredded rotated figure page sets
     ``needs_ocr_enhancement``; the rotated CLEAN caption page and the upright
     page do not.
  3. Shipped output, hermetically through ``process()`` with the ladder
     rejecting everything: the page does not report SUCCESS and the shipped
     text carries none of the one- and two-character confetti.
  4. Reverse guard through ``process()``: a rotated page whose native text is
     clean prose still ships native as SUCCESS, for free.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.born_digital import BornDigitalDetector
from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.orchestrator import UnifiedPipeline

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

_CAPTION = (
    "Figure 4: Contribution of the three policy instruments to the high-frequency "
    "changes in yields around FOMC announcements. The units are basis points."
)

#: Glyph-run lengths, cycled. Mixed one-character and multi-character runs, the
#: shape the reference page shows (``O``, ``F``, ``round``, ``requency``): a
#: caption split into runs of one letter and of several.
_RUN_LENGTHS = (5, 1, 6, 2, 8, 1, 5, 3, 7, 1)

_FONT = "helv"
_SIZE = 12.0

#: The reference page's own caption column: x = 491.8, glyph height 11.9,
#: second column 14.4pt to the right. Copied so the fixture's geometry is the
#: measured geometry rather than an invented one.
_COL_X = 492.0
_COL_STEP = 14.5
_Y_TOP = 95.0
_Y_BOTTOM = 675.0


def _runs(text: str) -> list[str]:
    out: list[str] = []
    i = k = 0
    while i < len(text):
        n = _RUN_LENGTHS[k % len(_RUN_LENGTHS)]
        out.append(text[i : i + n])
        i += n
        k += 1
    return out


def _rotated_caption_pdf(path: Path, *, shredded: bool) -> None:
    """A rotated figure page: one full-page raster plus a 90-degree caption.

    ``shredded=True`` places each glyph run *behind* the previous one along the
    writing direction, which is what the reference PDF's typesetter emits and
    what makes PyMuPDF break the caption into one line per run. ``False``
    places the identical runs in reading order -- same font, same size, same
    column, same page -- and is the negative control.
    """
    doc = fitz.open()
    page = doc.new_page()  # 612 x 792, the reference page's size
    x = _COL_X
    y = _Y_TOP if shredded else _Y_BOTTOM
    for run in _runs(_CAPTION):
        width = fitz.get_text_length(run, fontname=_FONT, fontsize=_SIZE)
        if shredded:
            page.insert_text((x, y + width), run, fontsize=_SIZE, fontname=_FONT, rotate=90)
            y += width
        else:
            page.insert_text((x, y), run, fontsize=_SIZE, fontname=_FONT, rotate=90)
            y -= width
        if not (60.0 < y < 700.0):
            y = _Y_TOP if shredded else _Y_BOTTOM
            x += _COL_STEP
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 32, 32))
    pix.set_rect(pix.irect, (180, 90, 90))
    page.insert_image(fitz.Rect(52.0, 30.0, 470.0, 760.0), pixmap=pix)
    doc.save(str(path))
    doc.close()


def _upright_caption_pdf(path: Path) -> None:
    """The same caption and raster, unrotated: the untouched-page guard."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(
        fitz.Rect(60.0, 700.0, 550.0, 780.0), _CAPTION, fontsize=_SIZE, fontname=_FONT
    )
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 32, 32))
    pix.set_rect(pix.irect, (180, 90, 90))
    page.insert_image(fitz.Rect(52.0, 30.0, 470.0, 690.0), pixmap=pix)
    doc.save(str(path))
    doc.close()


def _confetti_lines(text: str) -> list[str]:
    """The ``MC / O / F / round / a`` signature: lines of <= 2 characters."""
    return [ln for ln in text.splitlines() if ln.strip() and len(ln.strip()) <= 2]


# ---------------------------------------------------------------------------
# Group 1: fixture integrity -- watch the fixture before trusting the verdict
# ---------------------------------------------------------------------------


class TestFixtureIntegrity:
    @pytest.mark.parametrize("shredded", [True, False])
    def test_every_glyph_box_lies_inside_the_page(self, tmp_path: Path, shredded: bool) -> None:
        pdf_path = tmp_path / f"rot_{shredded}.pdf"
        _rotated_caption_pdf(pdf_path, shredded=shredded)

        doc = fitz.open(pdf_path)
        page = doc[0]
        rect = page.rect
        boxes = [w[:4] for w in page.get_text("words")]
        doc.close()

        assert boxes, "fixture produced no words at all"
        outside = [
            b
            for b in boxes
            if not (rect.x0 <= b[0] and rect.y0 <= b[1] and b[2] <= rect.x1 and b[3] <= rect.y1)
        ]
        assert outside == [], (
            f"fixture text is clipped by the page edge, so any 'shredding' it shows is an "
            f"artifact of the fixture and not of the extractor; offending boxes: {outside}"
        )

    def test_control_reads_the_caption_back_verbatim(self, tmp_path: Path) -> None:
        """The rotated CONTROL is undamaged -- rotation alone extracts fine."""
        pdf_path = tmp_path / "rot_clean.pdf"
        _rotated_caption_pdf(pdf_path, shredded=False)

        assessment = BornDigitalDetector().detect_page(pdf_path, 1)

        assert assessment.text_is_rotated is True
        # Whitespace-insensitive: the caption is long enough to wrap into a
        # second rotated column, and the wrap falls mid-word exactly as it does
        # on a real two-column rotated caption. The property under test is that
        # the character stream is intact and in order, which is precisely what
        # the shredded page destroys.
        assert "".join(assessment.native_text.split()) == "".join(_CAPTION.split()), (
            f"control caption did not round-trip: {assessment.native_text!r}"
        )

    def test_shredded_fixture_reproduces_the_reference_signature(self, tmp_path: Path) -> None:
        """Same caption, same column, runs placed against the writing direction."""
        pdf_path = tmp_path / "rot_shredded.pdf"
        _rotated_caption_pdf(pdf_path, shredded=True)

        assessment = BornDigitalDetector().detect_page(pdf_path, 1)

        assert assessment.is_born_digital is True
        assert assessment.has_tables is False, "the whole point is that no table is detected"
        assert assessment.text_is_rotated is True
        assert len(_confetti_lines(assessment.native_text)) >= 5, (
            f"fixture did not shred; native text was {assessment.native_text!r}"
        )


# ---------------------------------------------------------------------------
# Group 2: detector
# ---------------------------------------------------------------------------


class TestDetectorConsultsRotationWithoutTables:
    def test_shredded_rotated_figure_page_is_not_trusted_native(self, tmp_path: Path) -> None:
        """#263: rotation is consulted even though ``has_tables`` is False."""
        pdf_path = tmp_path / "rot_shredded.pdf"
        _rotated_caption_pdf(pdf_path, shredded=True)

        assessment = BornDigitalDetector().detect_page(pdf_path, 1)

        assert assessment.has_tables is False
        assert assessment.needs_ocr_enhancement is True, (
            "a rotated page whose native layer is shredded confetti must be routed to OCR; "
            f"notes were {assessment.notes}"
        )

    def test_rotated_clean_caption_page_still_ships_native_for_free(self, tmp_path: Path) -> None:
        """Reverse guard: rotation alone must NOT route a page to a paid VLM."""
        pdf_path = tmp_path / "rot_clean.pdf"
        _rotated_caption_pdf(pdf_path, shredded=False)

        assessment = BornDigitalDetector().detect_page(pdf_path, 1)

        assert assessment.text_is_rotated is True
        assert assessment.has_tables is False
        assert assessment.needs_ocr_enhancement is False, (
            "a rotated page whose native text is clean prose must keep shipping native; "
            f"notes were {assessment.notes}"
        )
        assert getattr(assessment, "native_rotated_text_shredded", False) is False

    def test_upright_page_is_untouched(self, tmp_path: Path) -> None:
        """Reverse guard: an unrotated page is not in scope at all."""
        pdf_path = tmp_path / "upright.pdf"
        _upright_caption_pdf(pdf_path)

        assessment = BornDigitalDetector().detect_page(pdf_path, 1)

        assert assessment.text_is_rotated is False
        assert assessment.needs_ocr_enhancement is False
        assert getattr(assessment, "native_rotated_text_shredded", False) is False
        assert "Figure 4" in assessment.native_text


# ---------------------------------------------------------------------------
# Group 3 / 4: hermetic process()
# ---------------------------------------------------------------------------


def _config() -> PipelineConfig:
    return PipelineConfig(
        primary_engine=EngineType.DEEPSEEK,
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        audit_enabled=False,
        save_figures=False,
        write_manifest=False,
    )


def _rejecting_router(page_num: int, ladder, run_provider, judge, **kwargs):
    """Every rung rejected: the ladder produces nothing acceptable."""
    from socr.pipeline.agentic import PageDecision, ProviderAttempt

    prof = ladder[0]
    rejected = PageOutput(
        page_num=page_num,
        text="",
        status=PageStatus.ERROR,
        engine="qwen",
        audit_passed=False,
    )
    att = ProviderAttempt(
        engine=prof.engine,
        output=rejected,
        cost_usd=0.0,
        accepted=False,
        reason="judge reject",
        provider_id=prof.id,
        model=prof.model,
        backend=prof.backend,
    )
    return PageDecision(page_num=page_num, final_output=rejected, attempts=[att], accepted=False)


def _run(pdf_path: Path, out_dir: Path, *, router=_rejecting_router):
    pipeline = UnifiedPipeline(_config())
    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch("socr.pipeline.orchestrator.route_page", side_effect=router),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
    ):
        result = pipeline.process(pdf_path, out_dir)
    sidecar = json.loads((out_dir / pdf_path.stem / "pages" / "00001.json").read_text())
    return result, sidecar


class TestShippedOutput:
    def test_shredded_page_does_not_ship_success_or_confetti(self, tmp_path: Path) -> None:
        """The issue's regression assertion, at the surface that actually ships."""
        pdf_path = tmp_path / "rot_shredded.pdf"
        _rotated_caption_pdf(pdf_path, shredded=True)

        result, sidecar = _run(pdf_path, tmp_path)
        shipped_text = sidecar["winning_output"].get("text", "")

        assert sidecar["status"] != PageStatus.SUCCESS.value, (
            f"a page whose shipped text is rotated confetti must not report success; "
            f"shipped {shipped_text!r}"
        )
        assert _confetti_lines(shipped_text) == [], (
            f"shipped text still carries the one/two-character confetti: {shipped_text!r}"
        )
        assert result.status.value != "success", (
            f"the document must surface the page failure too; got {result.status}"
        )

        # Surfaced at every level, not just page status (the cardinal rule):
        # a greppable failure marker in the body, the page image so a human can
        # still read the caption socr refused, a durable audit event, and the
        # decision flag in the sidecar.
        assert shipped_text.startswith("[page 1 failed:"), shipped_text
        assert "![Shredded rotated page 1](" in shipped_text, shipped_text
        assert "rotated_text_shredded" in [ev["kind"] for ev in sidecar["audit_events"]], sidecar[
            "audit_events"
        ]
        assert sidecar["native_rotated_text_shredded"] is True
        assert sidecar["winning_output"]["failure_mode"] == "native_text_shredded"

    def test_rotated_clean_page_still_ships_native_success(self, tmp_path: Path) -> None:
        """Reverse guard at the shipping surface: no paid re-route, no demotion."""
        pdf_path = tmp_path / "rot_clean.pdf"
        _rotated_caption_pdf(pdf_path, shredded=False)

        _result, sidecar = _run(pdf_path, tmp_path)
        shipped_text = sidecar["winning_output"].get("text", "")

        assert sidecar["status"] == PageStatus.SUCCESS.value, (
            f"clean rotated prose must keep shipping native as success; shipped {shipped_text!r}"
        )
        assert _confetti_lines(shipped_text) == []
        assert "".join(_CAPTION.split()) in "".join(shipped_text.split())
        assert "rotated_text_shredded" not in [ev["kind"] for ev in sidecar["audit_events"]]

    def test_upright_page_still_ships_native_success(self, tmp_path: Path) -> None:
        """Reverse guard: the unrotated page is byte-for-byte unaffected."""
        pdf_path = tmp_path / "upright.pdf"
        _upright_caption_pdf(pdf_path)

        _result, sidecar = _run(pdf_path, tmp_path)
        shipped_text = sidecar["winning_output"].get("text", "")

        assert sidecar["status"] == PageStatus.SUCCESS.value
        assert _confetti_lines(shipped_text) == []
        assert "Figure 4" in shipped_text
