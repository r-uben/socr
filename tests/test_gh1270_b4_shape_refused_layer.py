"""#1270 (B4, class 3): agreement with a layer refused for its SHAPE alone is a check.

Before: ``_layer_is_damaged`` treated ``not is_born_digital`` as damage for every refusal reason,
so 137 pages carried "the reading's numbers agree ... confirms nothing" (13 contents pages refused
for an average word length of 2.5, 4 for spacing, 3 for too few characters).

The detector's shape refusals return BEFORE #961's invisible-text scan, the digit and encoding
corruption counts, the minus-as-digit and control-byte scans, and the prior-OCR answer. Counting the
refusal as proof of a clean layer would re-open PR #1219 (an earlier OCR run's layer "confirming" a
model reading). So the detector now RUNS all of them on a shape-refused page and the verdict trusts
the layer only when every one passed. Each guard has a control arm here and a mutant.

Only AGREEMENT is claimed: a dispute with such a layer stays UNKNOWN.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz
import pytest

from socr.core import born_digital
from socr.core.born_digital import BornDigitalDetector
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.core.verdict import (
    DAMAGED_LAYER_AGREES_REASON,
    SHAPE_REFUSED_LAYER_PREFIX,
    Check,
    CheckResult,
    text_layer_numbers_clause,
)
from socr.pipeline.agentic import PageDecision
from socr.pipeline.orchestrator import UnifiedPipeline

#: A contents-page shape: tokens of one or two characters, so the average word length (1.3) is
#: under ``MIN_AVG_WORD_LENGTH`` and the page is refused as a scan on shape alone.
CONTENTS_LINES = [f"A {i} B {i + 10} C" for i in range(1, 30)]
CONTENTS_TEXT = "\n".join(CONTENTS_LINES)


def _pdf(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontname="helv", fontsize=10)
        y += 13
    doc.save(path)
    doc.close()
    return path


def _assess(pdf: Path):
    return BornDigitalDetector().detect(str(pdf)).pages[0]


# --------------------------------------------------------------- the detector


def test_a_contents_page_is_refused_for_shape_and_shown_clean(tmp_path: Path) -> None:
    pa = _assess(_pdf(tmp_path / "toc.pdf", CONTENTS_LINES))
    assert pa.is_born_digital is False
    assert any("avg word length too short" in n for n in pa.notes), "premise: a shape refusal"
    assert pa.layer_refused_for_shape_alone is True


def test_a_blank_page_under_the_character_floor_is_a_shape_refusal(tmp_path: Path) -> None:
    pa = _assess(_pdf(tmp_path / "b.pdf", ["(This page intentionally blank.)"]))
    assert any("insufficient text layer" in n for n in pa.notes)
    assert pa.layer_refused_for_shape_alone is True


def test_a_born_digital_page_is_not_a_shape_refusal(tmp_path: Path) -> None:
    prose = ["The estimated effect of the policy on output is reported below for each case."] * 8
    pa = _assess(_pdf(tmp_path / "p.pdf", prose))
    assert pa.is_born_digital is True and pa.layer_refused_for_shape_alone is False


def test_only_the_four_shape_returns_stamp_the_flag() -> None:
    """Garbage ratio, CID artifacts, encoding and digit corruption, and raster dominance are
    DAMAGE or image evidence, not shape: their returns do not stamp the flag. Pinned on the
    source because those branches need a page the detector cannot be fed cheaply."""
    src = Path(born_digital.__file__).read_text()
    assert src.count("layer_refused_for_shape_alone=self._shape_refusal_layer_is_clean(") == 4


def _scan_with_invisible_text(path: Path) -> Path:
    """A page-sized raster with a short invisible (render mode 3) OCR layer over it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200), False)
    pix.set_rect(pix.irect, (250, 250, 250))
    page.insert_image(page.rect, pixmap=pix)
    for i in range(6):
        page.insert_text(
            (72, 100 + 14 * i), f"A {i} B {i + 10} C", fontname="helv", fontsize=10, render_mode=3
        )
    doc.save(path)
    doc.close()
    return path


def test_a_real_invisible_ocr_layer_over_a_raster_is_not_shown_clean(tmp_path: Path) -> None:
    """#961 through the real detector, no patching: the shape refusal returns before the
    invisible-text scan, and the helper must run it."""
    pa = _assess(_scan_with_invisible_text(tmp_path / "scan.pdf"))
    assert pa.is_born_digital is False
    assert pa.layer_refused_for_shape_alone is False


def _shape_page(tmp_path: Path):
    doc = fitz.open(_pdf(tmp_path / "toc.pdf", CONTENTS_LINES))
    return doc, doc[0]


@pytest.mark.parametrize(
    ("guard", "patches"),
    [
        pytest.param("prior_ocr", {}, id="prior-ocr"),
        pytest.param(
            "invisible_text",
            {"_has_invisible_text_over_raster": lambda self, page: True},
            id="invisible-text-over-raster",
        ),
        pytest.param(
            "digit_corruption",
            {"_digit": lambda *_: 1},
            id="digit-corruption",
        ),
        pytest.param(
            "encoding",
            {"_encoding_corruption_ratio": lambda self, text: 0.5},
            id="encoding-corruption",
        ),
        pytest.param("minus", {"_minus": lambda *_: 1}, id="minus-as-digit"),
        pytest.param("control", {"_control": lambda *_: 1}, id="control-byte"),
        pytest.param("scan_raises", {"_has_invisible_text_over_raster": "raise"}, id="scan-raises"),
    ],
)
def test_each_skipped_check_has_its_own_veto(tmp_path: Path, guard: str, patches: dict) -> None:
    """The control arms: the SAME shape-refused page is clean without the patch (premise) and
    unclean with exactly one skipped check failing."""
    doc, page = _shape_page(tmp_path)
    detector = BornDigitalDetector()
    raw = page.get_text("text")
    assert detector._shape_refusal_layer_is_clean(page, raw, (False, False)) is True, "premise"
    with ExitStack() as stack:
        prior = (True, False) if guard == "prior_ocr" else (False, False)
        for name, repl in patches.items():
            if name == "_digit":
                stack.enter_context(patch.object(born_digital, "count_digit_corruption", repl))
            elif name == "_minus":
                stack.enter_context(patch.object(born_digital, "count_minus_as_digit_hits", repl))
            elif name == "_control":
                stack.enter_context(
                    patch.object(born_digital, "count_control_byte_before_digit_hits", repl)
                )
            elif repl == "raise":

                def boom(self, page):
                    raise RuntimeError("scan failed")

                stack.enter_context(patch.object(BornDigitalDetector, name, boom))
            else:
                stack.enter_context(patch.object(BornDigitalDetector, name, repl))
        assert detector._shape_refusal_layer_is_clean(page, raw, prior) is False
    doc.close()


def test_the_minus_scan_runs_only_when_a_two_is_present(tmp_path: Path) -> None:
    doc, page = _shape_page(tmp_path)
    detector = BornDigitalDetector()
    with patch.object(born_digital, "count_minus_as_digit_hits", lambda *_: 1):
        assert (
            detector._shape_refusal_layer_is_clean(page, "A 1 B 11 C " * 5, (False, False)) is True
        )
        assert (
            detector._shape_refusal_layer_is_clean(page, "A 2 B 12 C " * 5, (False, False)) is False
        )
    doc.close()


# --------------------------------------------------------------- the verdict


def _p(**over) -> SimpleNamespace:
    base = dict(
        own_layer_text="A 1 B 11 C 2 D 12",
        own_layer_word_reference="",
        own_layer_figure_words="",
        is_born_digital=False,
        layer_refused_for_shape_alone=True,
        prior_ocr_layer_suspect=False,
        has_unrecovered_symbol_glyphs=False,
        control_byte_digit_hits=0,
        control_byte_scan_failed=False,
        minus_as_digit_hits=0,
        minus_as_digit_scan_failed=False,
        has_encoding_hygiene_suspect=False,
        table_ladder_disposition=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _out(text: str) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen")


AGREES = "A 1 B 11 C 2 D 12"
DISPUTES = "A 1 B 11 C 2 D 99"


def test_agreement_with_a_shape_refused_clean_layer_is_a_check() -> None:
    assert text_layer_numbers_clause(_out(AGREES), _p()) is None


def test_the_same_page_without_the_shape_fact_keeps_the_old_unknown() -> None:
    """Difference arm: ONLY ``layer_refused_for_shape_alone`` changes."""
    clause = text_layer_numbers_clause(_out(AGREES), _p(layer_refused_for_shape_alone=False))
    assert clause is not None and clause.result is CheckResult.UNKNOWN
    assert clause.reason == DAMAGED_LAYER_AGREES_REASON


@pytest.mark.parametrize(
    "flag",
    [
        {"minus_as_digit_hits": 2},
        {"minus_as_digit_scan_failed": True},
        {"control_byte_digit_hits": 1},
        {"control_byte_scan_failed": True},
        {"has_unrecovered_symbol_glyphs": True},
    ],
)
def test_a_damaged_layer_flag_overrides_the_shape_fact(flag: dict) -> None:
    clause = text_layer_numbers_clause(_out(AGREES), _p(**flag))
    assert clause is not None and clause.result is CheckResult.UNKNOWN
    assert clause.reason == DAMAGED_LAYER_AGREES_REASON


def test_a_prior_ocr_layer_gets_no_clause_here_as_before() -> None:
    assert text_layer_numbers_clause(_out(AGREES), _p(prior_ocr_layer_suspect=True)) is None


def test_a_dispute_with_a_shape_refused_layer_stays_unknown_and_says_why() -> None:
    clause = text_layer_numbers_clause(_out(DISPUTES), _p())
    assert clause is not None and clause.result is CheckResult.UNKNOWN
    assert clause.reason.startswith(SHAPE_REFUSED_LAYER_PREFIX)
    assert clause.check is Check.TEXT_LAYER_NUMBERS


def test_a_born_digital_layer_is_unchanged() -> None:
    page = _p(is_born_digital=True, layer_refused_for_shape_alone=False)
    assert text_layer_numbers_clause(_out(AGREES), page) is None
    clause = text_layer_numbers_clause(_out(DISPUTES), page)
    assert clause is not None and clause.result is CheckResult.FAIL


def test_the_drops_and_words_checks_still_read_a_shape_refused_layer_as_damaged() -> None:
    """Scope: only the numbers clause is widened."""
    from socr.core.verdict import _layer_is_damaged, _word_layer_is_damaged

    assert _layer_is_damaged(_p()) is True
    assert _word_layer_is_damaged(_p()) is True


# --------------------------------------------------- through the pipeline, both provider states


def _run(tmp_path: Path, *, shape_alone: bool, providers: list, resumed: bool = False):
    from socr.core.manifest import finalized_page_record

    pdf = tmp_path / "doc.pdf"
    if not pdf.exists():
        _pdf(pdf, CONTENTS_LINES)
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            agentic=True,
            judge_backend="heuristic",
            quiet=True,
            save_figures=False,
            write_manifest=False,
            native_first=False,
        )
    )
    pipeline._scan_root = pdf.parent
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    if not resumed:
        # the REAL wiring: detector -> DocumentAssessment -> PageState.apply_born_digital
        assessment = BornDigitalDetector().detect(str(pdf))
        state.apply_born_digital(assessment)
        assert ps.layer_refused_for_shape_alone is True, "premise: the wiring carries the fact"
        ps.layer_refused_for_shape_alone = shape_alone  # the ONLY thing the arms differ in
    reading = CONTENTS_TEXT

    def _route(page_num, *a, **kw):
        assert not resumed, "a resumed terminal page must not be re-read"
        out = PageOutput(
            page_num=page_num,
            text=reading,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        return PageDecision(page_num=page_num, final_output=out, attempts=[], accepted=True)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(pipeline, "_available_engines_for_agentic", return_value=providers)
        )
        stack.enter_context(patch("socr.pipeline.orchestrator.route_page", side_effect=_route))
        stack.enter_context(
            patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True)
        )
        stack.enter_context(patch.object(pipeline, "_resolve_judge_model", return_value=""))
        pipeline._phase_agentic(state, tmp_path / "out")
        pipeline._phase_assemble(state, tmp_path / "out")  # writes the terminal sidecar
    rec = finalized_page_record(state, 1)
    return rec.output.status, [(c.check.value, c.result.value) for c in rec.verdict.clauses]


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "no-provider"])
def test_the_shipped_contents_page_differs_only_by_the_clause(tmp_path: Path, providers) -> None:
    with_fact = _run(tmp_path / "a", shape_alone=True, providers=providers)
    without = _run(tmp_path / "b", shape_alone=False, providers=providers)
    if providers:
        assert ("text_layer_numbers", "UNKNOWN") in without[1], "premise: the arm is marked"
        assert ("text_layer_numbers", "UNKNOWN") not in with_fact[1]
        assert with_fact[0] is PageStatus.SUCCESS and without[0] is PageStatus.WARNING
    else:
        assert with_fact == without


def test_the_cleared_contents_page_stays_cleared_on_resume(tmp_path: Path) -> None:
    """The page is SUCCESS, so run 2 skips it from its terminal sidecar; the restored page has
    no shape fact (it is runtime only) and must keep the verdict its first run published."""
    first = _run(tmp_path, shape_alone=True, providers=[PROFILE_QWEN_LOCAL])
    again = _run(tmp_path, shape_alone=True, providers=[PROFILE_QWEN_LOCAL], resumed=True)
    assert first[0] is PageStatus.SUCCESS and first[1] == []
    assert again == first


# ------------------------------------------- review of PR #1325: the vetoes the early return skipped


def _verdict_for(pa, layer: str, reading: str):
    """The detector's real flag fed into the real numbers clause: the end-to-end arm."""
    page = _p(
        own_layer_text=layer,
        layer_refused_for_shape_alone=pa.layer_refused_for_shape_alone,
        layer_glyphs_undamaged=pa.layer_glyphs_undamaged,
    )
    return text_layer_numbers_clause(_out(reading), page)


def test_a_full_page_raster_with_a_visible_baked_in_layer_is_not_shown_clean(tmp_path: Path):
    """Carbon: a raster that fills the page (GH-35-FU) with a SHORT visible OCR layer over it is
    refused for shape, and the layer is an earlier OCR's. The shape flag must stay False, so a
    reading that repeats the OCR's digits does not become a CHECK."""
    pdf = tmp_path / "scan.pdf"
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200), False)
    pix.set_rect(pix.irect, (250, 250, 250))
    page.insert_image(page.rect, pixmap=pix)
    for i in range(6):
        page.insert_text((72, 100 + 14 * i), f"A {i} B {i + 10} C", fontname="helv", fontsize=10)
    doc.save(pdf)
    doc.close()
    pa = _assess(pdf)
    assert pa.is_born_digital is False and any("avg word length" in n for n in pa.notes)
    assert pa.layer_refused_for_shape_alone is False
    clause = _verdict_for(pa, AGREES, AGREES)
    assert clause is not None and clause.reason == DAMAGED_LAYER_AGREES_REASON


def test_a_contents_page_with_mojibake_math_is_not_shown_clean(tmp_path: Path):
    """Kimi: ``PðAÞ ¼ 2`` on a short-token contents page. The control arm is the same page
    without the line (shown clean, premise)."""
    clean = _assess(_pdf(tmp_path / "c.pdf", CONTENTS_LINES))
    assert clean.layer_refused_for_shape_alone is True
    pa = _assess(_pdf(tmp_path / "k.pdf", ["PðAÞ ¼ 2", *CONTENTS_LINES[:11]]))
    assert pa.is_born_digital is False and any("avg word length" in n for n in pa.notes)
    assert pa.layer_refused_for_shape_alone is False
    assert pa.layer_glyphs_undamaged is False
    clause = _verdict_for(pa, "PðAÞ ¼ 2", "PðAÞ ¼ 2")
    assert clause is not None and clause.result is CheckResult.UNKNOWN


def test_a_symbolmt_private_use_minus_is_not_shown_clean(tmp_path: Path):
    """Carbon: ``\\uf02d3.5`` in SymbolMT (the minus is a private-use glyph), reading ``3.5``.
    The comparator splits the PUA char off and sees agreement, so the layer must not be trusted.
    Spans are patched in: a PDF written by PyMuPDF cannot carry a Word-style SymbolMT PUA."""
    doc, page = _shape_page(tmp_path)
    detector = BornDigitalDetector()
    raw = page.get_text("text")  # the layer text is clean; the SPAN says SymbolMT PUA
    span = {"font": "SymbolMT", "text": "\uf02d3.5"}
    assert detector._shape_refusal_layer_is_clean(page, raw, (False, False)) is True, "control"
    with patch.object(born_digital, "iter_page_spans", lambda p: [span]):
        assert detector._shape_refusal_layer_is_clean(page, raw, (False, False)) is False
        assert detector._layer_glyphs_are_undamaged(page, raw) is False
    doc.close()


@pytest.mark.parametrize(
    ("guard", "patches"),
    [
        pytest.param("raster", {"_raster_coverage": lambda self, page: 1.0}, id="raster-dominance"),
        pytest.param("no_font", {"_count_fonts": lambda self, page: 0}, id="no-font"),
        pytest.param(
            "garbled_math",
            {"_layer_glyphs_are_undamaged": lambda self, page, raw=None: False},
            id="glyph-scan",
        ),
    ],
)
def test_each_review_veto_stands_alone(tmp_path: Path, guard: str, patches: dict) -> None:
    doc, page = _shape_page(tmp_path)
    detector = BornDigitalDetector()
    raw = page.get_text("text")
    assert detector._shape_refusal_layer_is_clean(page, raw, (False, False)) is True, "premise"
    with ExitStack() as stack:
        for name, repl in patches.items():
            stack.enter_context(patch.object(BornDigitalDetector, name, repl))
        assert detector._shape_refusal_layer_is_clean(page, raw, (False, False)) is False
    doc.close()


@pytest.mark.parametrize(
    ("name", "patch_target", "repl"),
    [
        pytest.param("garbage", "_garbage_ratio", lambda self, t: 1.0, id="garbage-ratio"),
        pytest.param("mojibake", "corrupt_math_glyph_count", lambda t: 1, id="any-mojibake-glyph"),
        pytest.param(
            "garbled",
            "detect_garbled_math",
            lambda page, t: born_digital.GarbledMathSignals(private_use=1),
            id="garbled-math-signal",
        ),
    ],
)
def test_the_glyph_scan_has_a_veto_per_signal(tmp_path: Path, name, patch_target, repl) -> None:
    doc, page = _shape_page(tmp_path)
    detector = BornDigitalDetector()
    raw = page.get_text("text")
    assert detector._layer_glyphs_are_undamaged(page, raw) is True, "premise"
    target = BornDigitalDetector if patch_target == "_garbage_ratio" else born_digital
    with patch.object(target, patch_target, repl):
        assert detector._layer_glyphs_are_undamaged(page, raw) is False
    doc.close()
