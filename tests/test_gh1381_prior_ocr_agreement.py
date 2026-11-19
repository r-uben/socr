"""#1381: an exact, independent fresh reading settles a scan's ``prior_ocr_layer`` clause.

The owner's ruling (2026-11-18): on a scanned page that carries a text layer from an earlier
OCR pass, exact two-way agreement between socr's fresh reading and that layer confirms the page.
Everything that is not exact agreement keeps the page warned. Two kinds of test:

* clause level: one fixture, one change at a time, each adversarial case a DIFFERENCE from the
  agreeing control (never an absolute outcome);
* ``process()`` level, hermetic per ``CLAUDE.md`` (ladder, judge and engine pinned; the
  provider-dependent outcome is never pinned): the same document run twice, the reading being
  the only thing that differs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import socr
from socr.core.marks import CheckResult
from socr.core.result import PageOutput, PageStatus
from socr.core.state import PageState
from socr.core.verdict import (
    PRIOR_OCR_AGREEMENT_KIND,
    PRIOR_OCR_AGREEMENT_REASON,
    Check,
    TraceResult,
    _prior_ocr_layer_clause,
    _prior_ocr_layer_outcome,
    independent_page_evidence,
    prior_ocr_layer_agreement,
    stamp_text_digest,
)
from socr.core.pass_trace import TraceEntry

LAYER = "\n".join(
    [
        "The Committee met and reviewed conditions in the economy.",
        "Discount rates were raised from 3 to 3.5 per cent at the Banks.",
        "Part II covered inflation of -.7 percent and growth of 1.1 percent.",
        "Members discussed the outlook for prices and employment.",
    ]
)


def test_loaded_source_is_this_checkout() -> None:
    """The mutation canary, in the house shape."""
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _page(layer: str = LAYER, *, engine: str = "qwen", **flags) -> PageState:
    p = PageState(page_num=1)
    p.prior_ocr_layer_suspect = True
    p.prior_ocr_layer = True
    p.text_hidden_layer_ink_accounted = True
    p.native_text = layer
    p.best_output = PageOutput(page_num=1, text="x", status=PageStatus.SUCCESS, engine=engine)
    for name, value in flags.items():
        setattr(p, name, value)
    return p


def _read(text: str, engine: str = "qwen") -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine=engine)


def _agrees(text: str, p: PageState | None = None, engine: str = "qwen") -> bool:
    return prior_ocr_layer_agreement(_read(text, engine), p or _page(engine=engine))


def _clause_result(text: str, p: PageState | None = None, engine: str = "qwen"):
    clause = _prior_ocr_layer_clause(_read(text, engine), p or _page(engine=engine))
    return None if clause is None else clause.result


# --- the rule: control, then one difference at a time --------------------------------


def test_exact_agreement_settles_the_clause() -> None:
    assert _agrees(LAYER)
    outcome = _prior_ocr_layer_outcome(_read(LAYER), _page())
    assert outcome.clause is None
    assert outcome.entry.result is TraceResult.PASS
    assert outcome.entry.check == Check.PRIOR_OCR_LAYER.value
    assert outcome.entry.reason == PRIOR_OCR_AGREEMENT_REASON


def test_only_layout_whitespace_is_normalised() -> None:
    assert _agrees(LAYER.replace("\n", "\n\n").replace(" and ", "  and\u00a0"))
    assert _agrees("\t" + LAYER.replace("\n", " ") + "\u2003")


@pytest.mark.parametrize(
    ("layer_form", "reading_form"),
    [
        ("(7)", "7"),  # a negative lost
        ("(7)", "-7"),
        ("-7", "7"),
        ("- 7", "7"),
        ("-7", "\u20137"),  # en dash: a different dash is a disagreement
        ("-7", "\u20147"),  # em dash
        ("-7", "\u20107"),  # U+2010
        ("-7", "\u20117"),  # U+2011
        ("-7", "\u20127"),  # U+2012
        ("-7", "\u22127"),  # minus sign
        ("-7", "\uff0d7"),  # fullwidth minus
        ("-.7", "-7"),
        ("-.7", ".7"),
        ("1.1", "11"),
        ("1.1", "1 1"),
        ("10 2", "102"),
        ("7%", "7"),
        ("+7", "7"),
        ("\u00b17", "7"),
        ("< 2", "> 2"),
        ("\u2264 2", "\u2265 2"),
        ("a = b", "a \u2260 b"),
        ("$7", "\u00a37"),
        ("1/2", "1 2"),
        ("10^2", "10 2"),
        ("3.5", "3.6"),
        ("Part II", "Part I"),
        ("Part II", "Part III"),
        ("3.5", "35"),
        ("inflation", "deflation"),
        ("economy.", "economy"),
        ("economy.", "economy,"),
        ("economy.", "**economy**."),
    ],
)
def test_any_character_difference_keeps_the_page_warned(layer_form: str, reading_form: str) -> None:
    layer = f"The note holds {layer_form} in the table."
    reading = f"The note holds {reading_form} in the table."
    assert _agrees(layer, _page(layer)), "the control must agree, or the case below is vacuous"
    assert not _agrees(reading, _page(layer)), (layer_form, reading_form)
    assert _clause_result(reading, _page(layer)) is CheckResult.UNKNOWN


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t.replace("economy.", "economy today."),
        lambda t: t.replace("reviewed ", ""),
        lambda t: t + "\nThe meeting adjourned.",
        lambda t: t.replace("met and", "met and so"),
        lambda t: "\n".join(t.splitlines()[:2]),
        lambda t: t.replace("Committee met and reviewed", "reviewed met and Committee"),
        lambda t: t.replace("of -.7 percent", "-.7 of percent"),
        lambda t: "\n".join(reversed(t.splitlines())),
    ],
)
def test_added_dropped_or_reordered_words_keep_the_page_warned(change) -> None:
    assert _agrees(LAYER)
    assert not _agrees(change(LAYER))


def test_an_empty_layer_stays_unknown() -> None:
    """Empty text equals an empty layer token for token: only the guard refuses it."""
    for layer in ("", "   \n"):
        assert not _agrees("", _page(layer))
        assert not _agrees("  ", _page(layer))
        assert not _agrees(LAYER, _page(layer))


def test_a_failure_marker_does_not_agree_with_a_layer() -> None:
    assert not _agrees("[page 1 failed: unreadable]")


def test_hyphenated_line_breaks_stay_unknown() -> None:
    layer = LAYER.replace("conditions in", "condi-\ntions in")
    # both sides carry the identical break: the tokens are equal, only the guard refuses it
    assert not _agrees(layer, _page(layer))
    assert _agrees(
        layer.replace("condi-\ntions", "conditions"),
        _page(layer.replace("condi-\ntions", "conditions")),
    )


# --- nothing ships that was not compared ----------------------------------------------


@pytest.mark.parametrize(
    "appended",
    [
        "> [page 1 note] the rate was 9.31 percent.",
        "<!-- rate 9.9 -->",
        "![Figure 1](figures/figure_1_page1.png)",
        "**Figure 1** (page 1)",
        "<table><tr><td>1</td></tr></table>",
    ],
)
def test_content_outside_the_layer_never_ships_under_a_confirmation(appended: str) -> None:
    assert _agrees(LAYER)
    assert not _agrees(LAYER + "\n\n" + appended)
    assert not _agrees(appended + "\n\n" + LAYER)


# --- completeness ---------------------------------------------------------------------


def test_the_layer_must_account_for_the_ink() -> None:
    """A layer that is a fragment agrees with a reading that is the same fragment."""
    fragment = "Page 3"
    assert _agrees(fragment, _page(fragment))
    assert not _agrees(fragment, _page(fragment, text_hidden_layer_ink_accounted=False))
    assert _clause_result(fragment, _page(fragment, text_hidden_layer_ink_accounted=False)) is (
        CheckResult.UNKNOWN
    )


def _scan(path: Path, layer_lines: list[str], ink_lines: list[str]) -> Path:
    """A scan: the page's ink is a raster of *ink_lines*; the layer is invisible text over it."""
    import fitz

    doc = fitz.open()
    ink = fitz.open()
    src = ink.new_page(width=612, height=792)
    for row, line in enumerate(ink_lines):
        src.insert_text((72, 100 + row * 14), line, fontsize=10)
    png = src.get_pixmap(dpi=150).tobytes("png")
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=png)
    for row, line in enumerate(layer_lines):
        page.insert_text((72, 100 + row * 14), line, fontsize=10, render_mode=3)
    doc.save(path)
    return path


ROWS = [f"Row {i} value {i * 7}.5 and {i * 3}.25" for i in range(1, 46)]


def test_the_hidden_layer_ink_fact_measures_completeness_on_a_scan(tmp_path: Path) -> None:
    import fitz

    from socr.core.text_coverage import analyze_page

    def fact(layer_lines, ink_lines, name) -> tuple[bool, bool]:
        with fitz.open(_scan(tmp_path / name, layer_lines, ink_lines)) as doc:
            cov = analyze_page(doc[0])
        return cov.hidden_layer_ink_accounted, cov.ink_fully_accounted

    whole = fact(ROWS, ROWS, "whole.pdf")
    fragment = fact(["Page 3"], ROWS, "fragment.pdf")
    assert whole[0] is True and fragment[0] is False
    # the #1378 fact is never true on a scan (the raster): the reason this one exists
    assert whole[1] is False


# --- the gates: what may NOT settle the clause ---------------------------------------


def test_a_failed_layer_scan_stays_unknown() -> None:
    assert _agrees(LAYER)
    for flag in ("prior_ocr_layer_scan_failed", "invisible_text_scan_failed"):
        assert not _agrees(LAYER, _page(**{flag: True})), flag
    # a failed scan alone: not a positive finding, so nothing to confirm
    alone = _page(prior_ocr_layer=False, prior_ocr_layer_scan_failed=True)
    assert not _agrees(LAYER, alone)
    assert _clause_result(LAYER, alone) is CheckResult.UNKNOWN


def test_a_page_that_is_not_suspect_is_untouched() -> None:
    assert not _agrees(LAYER, _page(prior_ocr_layer_suspect=False, prior_ocr_layer=False))


@pytest.mark.parametrize(
    "engine",
    [
        "marker",
        "consensus(marker)",
        "native",
        "native+equations",
        "chart_asset",
        "reconciled(qwen,glm)",
        "",
    ],
)
def test_a_reading_that_may_derive_from_the_layer_never_counts(engine: str) -> None:
    assert _agrees(LAYER, engine="qwen")
    page = _page(engine=engine)
    assert not prior_ocr_layer_agreement(_read(LAYER, engine), page)
    assert _clause_result(LAYER, page, engine) is CheckResult.UNKNOWN


def test_consensus_of_an_image_reader_is_still_a_reading() -> None:
    assert _agrees(LAYER, engine="consensus(qwen)")


@pytest.mark.parametrize(
    "change",
    [
        {"table_floor_body": True},
        {"scanned_prose_recovered": True},
        {"status": PageStatus.ERROR},
    ],
)
def test_a_floor_body_under_a_model_label_never_counts(change: dict) -> None:
    control = _read(LAYER)
    assert prior_ocr_layer_agreement(control, _page())
    floor = PageOutput(**{**control.__dict__, **change})
    assert not prior_ocr_layer_agreement(floor, _page())


def test_a_layer_slice_spliced_into_the_text_never_counts() -> None:
    """The equation sidecar block appends a slice of the layer to the reading."""
    assert _agrees(LAYER)
    assert not _agrees(LAYER, _page(layer_text_spliced=True))
    # and structurally: reading plus its appended slice is not the layer
    assert not _agrees(LAYER + "\n\n" + LAYER.splitlines()[1])


def test_a_whole_document_slice_is_not_a_page_major_reading() -> None:
    page = _page()
    page.best_output = None
    assert not prior_ocr_layer_agreement(_read(LAYER), page)


def test_the_winner_must_be_the_pages_own_attempt() -> None:
    assert not prior_ocr_layer_agreement(_read(LAYER, "qwen"), _page(engine="glm"))


def test_agreement_is_not_evidence_for_any_other_clause() -> None:
    from socr.core.verdict import text_layer_numbers_outcome, text_layer_words_outcome

    page = _page()
    page.own_layer_text = LAYER
    page.own_layer_word_reference = LAYER
    assert _agrees(LAYER, page)
    assert text_layer_numbers_outcome(_read(LAYER), page).entry.result is TraceResult.NOT_RUN
    assert text_layer_words_outcome(_read(LAYER), page).entry.result is TraceResult.NOT_RUN


# --- the page_judge evidence: the one entry that counts, exactly ----------------------


def _agreement_entry(**changes) -> TraceEntry:
    entry = TraceEntry(
        Check.PRIOR_OCR_LAYER.value, TraceResult.PASS, "page", PRIOR_OCR_AGREEMENT_REASON
    )
    return TraceEntry(**{**entry.__dict__, **changes})


def test_independent_evidence_accepts_only_the_exact_agreement_entry() -> None:
    text = "the page"
    good = stamp_text_digest(_agreement_entry(), text)
    assert independent_page_evidence([good], text)
    # not computed on THIS text (digest): states nothing about the bytes that ship
    assert not independent_page_evidence([good], text + " changed")
    assert not independent_page_evidence([_agreement_entry()], text)  # unstamped
    # not a PASS
    for result in (TraceResult.NOT_RUN, TraceResult.NOT_APPLICABLE):
        assert not independent_page_evidence([_agreement_entry(result=result)], text), result
    # another reason, another scope, another check
    assert not independent_page_evidence(
        [stamp_text_digest(_agreement_entry(reason="no"), text)], text
    )
    assert not independent_page_evidence(
        [stamp_text_digest(_agreement_entry(scope="table:1"), text)], text
    )
    assert not independent_page_evidence(
        [stamp_text_digest(_agreement_entry(check=Check.TEXT_LAYER_WORDS.value), text)], text
    )
    # and an UNKNOWN anywhere else still wins
    other = TraceEntry(Check.TEXT_LAYER_NUMBERS.value, TraceResult.UNKNOWN, "page", "x")
    assert not independent_page_evidence([good, other], text)


def test_the_meter_does_not_count_a_confirmed_page_unchecked() -> None:
    from socr.benchmark.acceptance import PRIOR_OCR_UNCHECKED, Page, _prior_ocr_unchecked

    def violations(sidecar: dict) -> list[str]:
        return _prior_ocr_unchecked(Page("d", 1, sidecar, "", None))

    suspect = {"prior_ocr_layer_suspect": True}
    assert violations(suspect) == [PRIOR_OCR_UNCHECKED]
    assert violations({**suspect, PRIOR_OCR_AGREEMENT_KIND: True}) == []
    assert violations({**suspect, PRIOR_OCR_AGREEMENT_KIND: "yes"}) == [PRIOR_OCR_UNCHECKED]
    assert violations({}) == []


# --- process(): the reading is the only thing that differs ----------------------------


class _ReadingEngine:
    name = "qwen"

    def __init__(self, text: str) -> None:
        self._text = text

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        return [
            PageOutput(page_num=n, text=self._text, status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


def _run(out_dir: Path, pdf: Path, reading: str, monkeypatch: pytest.MonkeyPatch) -> dict:
    from test_gh1156_part_a_interim import _FixedJudge

    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch
    from socr.pipeline.orchestrator import UnifiedPipeline

    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _ReadingEngine(reading))
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=True,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=False,
                describe_figure_crops=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._build_page_judge = lambda state: _FixedJudge(True)
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        pipe.process(pdf, output_dir=out_dir)
    sidecar = json.loads(next(iter(out_dir.rglob("pages/00001.json"))).read_text())
    log = next(iter(out_dir.rglob("audit_log.json")), None)
    events = json.loads(log.read_text()) if log is not None else {}
    return {"sidecar": sidecar, "events": events}


def _doc_status(out_dir: Path) -> str:
    meta = json.loads(next(iter(out_dir.rglob("scan/metadata.json"))).read_text())
    return str(meta.get("status", "")).lower()


def _verdict_checks(sidecar: dict) -> set[str]:
    return {c["check"] for c in (sidecar.get("verdict") or {}).get("clauses", [])}


def _event_kinds(events: object) -> list[str]:
    rows = events.get("events", events) if isinstance(events, dict) else events
    return [e.get("kind") for e in rows if isinstance(e, dict)]


SCAN_LINES = [
    "The Committee met and reviewed conditions in the economy.",
    "Discount rates were raised from 3 to 3.5 per cent at the Banks.",
    "Members discussed the outlook for prices and employment.",
    "The Secretary read the minutes of the previous meeting.",
]


def _layer_of(pdf: Path) -> str:
    import fitz

    with fitz.open(pdf) as doc:
        return doc[0].get_text()


def test_process_the_reading_alone_decides_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = _scan(tmp_path / "scan.pdf", SCAN_LINES, SCAN_LINES)
    layer_text = _layer_of(pdf)
    agree = _run(tmp_path / "agree", pdf, layer_text, monkeypatch)
    differ = _run(tmp_path / "differ", pdf, layer_text.replace("3.5", "3.6"), monkeypatch)
    dash = _run(tmp_path / "dash", pdf, layer_text.replace("3.5", "\u22123.5"), monkeypatch)

    assert agree["sidecar"].get("prior_ocr_layer_suspect") is True, (
        "fixture is not a prior-OCR scan"
    )
    assert agree["sidecar"].get("text_hidden_layer_ink_accounted") is True
    assert agree["sidecar"].get(PRIOR_OCR_AGREEMENT_KIND) is True
    for arm in (differ, dash):
        assert arm["sidecar"].get(PRIOR_OCR_AGREEMENT_KIND) is not True

    # every level that records clauses: the verdict, the audit log, the manifest, the sidecar
    assert _verdict_checks(agree["sidecar"]) == set()
    assert agree["sidecar"]["status"] == "success"
    assert differ["sidecar"]["status"] == "warning"
    assert "prior_ocr_layer" in _verdict_checks(differ["sidecar"])
    assert PRIOR_OCR_AGREEMENT_KIND in _event_kinds(agree["events"])
    assert PRIOR_OCR_AGREEMENT_KIND not in _event_kinds(differ["events"])
    assert "prior_ocr_layer_retained" in _event_kinds(differ["events"])
    assert "prior_ocr_layer_retained" not in _event_kinds(agree["events"])
    # the document says what its pages say: clean for the confirmed page, warned for the other
    assert _doc_status(tmp_path / "agree") == "completed"
    assert _doc_status(tmp_path / "differ") != "completed"
    manifest = json.loads(next(iter((tmp_path / "agree").rglob("manifest.json"))).read_text())
    assert any(e.get("prior_ocr_layer_agreement") is True for e in manifest["entries"].values())
    # the layer stays retained and suspect: confirming the page does not rehabilitate the layer
    assert agree["sidecar"]["prior_ocr_layer_suspect"] is True
    assert agree["sidecar"]["prior_ocr_layer_text"]


def test_process_a_layer_that_is_a_fragment_of_the_page_is_not_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewer's case: layer ``Page 3`` over a raster of 45 rows, reading ``Page 3``."""
    pdf = _scan(tmp_path / "frag.pdf", ["Page 3"], ROWS)
    out = _run(tmp_path / "frag", pdf, "Page 3", monkeypatch)
    assert out["sidecar"].get(PRIOR_OCR_AGREEMENT_KIND) is not True
    assert out["sidecar"]["status"] == "warning"


def test_process_refinalize_keeps_a_confirmed_page_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``socr-refinalize-own-text`` reaches the verdict a fresh run reached."""
    from socr.core.born_digital import BornDigitalDetector
    from socr.pipeline.refinalize_own_text import refinalize_page

    pdf = _scan(tmp_path / "scan.pdf", SCAN_LINES, SCAN_LINES)
    agree = _run(tmp_path / "agree", pdf, _layer_of(pdf), monkeypatch)
    assert agree["sidecar"].get(PRIOR_OCR_AGREEMENT_KIND) is True
    pa = BornDigitalDetector().detect(pdf).pages[0]
    assert refinalize_page(agree["sidecar"], pa) is None
    # the difference: the same sidecar without the run's record is re-derived as unconfirmed
    stripped = {k: v for k, v in agree["sidecar"].items() if k != "prior_ocr_layer_text"}
    assert refinalize_page(stripped, pa) is not None


def test_the_equation_block_marks_the_page_as_spliced(tmp_path: Path) -> None:
    """The equation sidecar block appends the layer's own slice: the page is then flagged, and
    the flag alone (not only the text difference) keeps it out of the confirmation."""
    from unittest.mock import MagicMock, patch

    from socr.core.audit_log import AuditEvent
    from socr.core.config import PipelineConfig
    from socr.core.state import DocumentState
    from socr.math.equation_latex import EQUATION_SIDECAR_HEADER, EquationLatexResult
    from socr.pipeline.orchestrator import UnifiedPipeline

    crop = tmp_path / "equation_1_page1.png"
    crop.write_bytes(b"fakepng")
    cfg = PipelineConfig()
    cfg.recover_clean_equations = True
    cfg.detect_equations = True
    pipe = UnifiedPipeline(cfg)
    handle = MagicMock()
    handle.path = tmp_path / "doc.pdf"
    state = DocumentState(handle=handle)
    state.pages[1] = _page(LAYER)
    state.events.append(
        AuditEvent(
            page_num=1,
            kind="equation_region_detected",
            engine="detect_equations",
            detail="t",
            data={
                "source_bbox": [1.0, 2.0, 4.0, 5.0],
                "padded_bbox": [1.0, 2.0, 4.0, 5.0],
                "has_eq_number": False,
                "crop_path": str(crop),
                "detection_time_s": 0.0,
            },
        )
    )
    out = _read(LAYER)
    block = f"![eq]({crop})\n{EQUATION_SIDECAR_HEADER}\n```latex\nE = mc^2\n```"
    result = EquationLatexResult(
        region_index=0,
        page_num=1,
        crop_path=str(crop),
        raw_latex="E = mc^2",
        validation_ok=True,
        validation_reason="ok",
        latex_attached=True,
        model_id="m",
        sidecar_block=block,
    )
    assert state.pages[1].layer_text_spliced is False
    with patch("socr.math.equation_latex.process_equation_region", return_value=result):
        pipe._attach_equation_latex_sidecars(state, [out])
    assert out.text != LAYER, "the lane appended nothing, so this test shows nothing"
    assert state.pages[1].layer_text_spliced is True
