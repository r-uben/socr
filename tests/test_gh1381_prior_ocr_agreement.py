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
    prior_ocr_layer_agreement,
)

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


def test_markdown_decoration_alone_does_not_break_agreement() -> None:
    decorated = "# " + LAYER.replace("\n", "\n\n").replace("Banks.", "**Banks**.")
    assert _agrees(decorated)


@pytest.mark.parametrize(
    ("label", "changed"),
    [
        ("one digit differs", LAYER.replace("3.5", "3.6")),
        ("one short token differs (II -> I)", LAYER.replace("Part II", "Part I")),
        ("one short token differs (II -> III)", LAYER.replace("Part II", "Part III")),
        ("sign or point differs (-.7 -> -7)", LAYER.replace("-.7", "-7")),
        ("sign dropped (-.7 -> .7)", LAYER.replace("-.7", ".7")),
        ("decimal point lost (1.1 -> 11)", LAYER.replace("1.1", "11")),
        ("decimal point lost (3.5 -> 35)", LAYER.replace("3.5", "35")),
        ("extra word in the reading", LAYER.replace("economy.", "economy today.")),
        ("word missing from the reading", LAYER.replace("reviewed ", "")),
        ("extra sentence in the reading", LAYER + "\nThe meeting adjourned."),
        ("extra short token in the reading", LAYER.replace("met and", "met and so")),
        ("long word changed", LAYER.replace("inflation", "deflation")),
        ("reading is only a part of the layer", "\n".join(LAYER.splitlines()[:2])),
        (
            "long words reordered",
            LAYER.replace("Committee met and reviewed", "reviewed met and Committee"),
        ),
        (
            "short words reordered across a long word",
            LAYER.replace("of -.7 percent", "-.7 of percent"),
        ),
        ("lines reordered", "\n".join(reversed(LAYER.splitlines()))),
    ],
)
def test_any_difference_keeps_the_page_warned(label: str, changed: str) -> None:
    assert _agrees(LAYER), "the control must agree, or every case below is vacuous"
    assert not _agrees(changed), label
    assert _clause_result(changed) is CheckResult.UNKNOWN, label
    assert _clause_result(LAYER) is None


def test_a_layer_longer_than_the_reading_stays_unknown() -> None:
    assert not _agrees("The Committee met.", _page(LAYER))
    assert not _agrees(LAYER, _page("The Committee met."))


def test_an_empty_layer_stays_unknown() -> None:
    for layer in ("", "   \n"):
        assert not _agrees(LAYER, _page(layer))
        assert not _agrees("", _page(layer))


def test_the_reading_must_have_some_content() -> None:
    assert not _agrees("")
    assert not _agrees("[page 1 failed: unreadable]")


@pytest.mark.parametrize(
    "wrapped",
    [
        "<table><tr><td>" + " </td><td>".join(LAYER.splitlines()) + "</td></tr></table>",
        "<p>" + LAYER + "</p>",
    ],
)
def test_an_html_wrapped_reading_stays_unknown(wrapped: str) -> None:
    assert not _agrees(wrapped)


def test_hyphenated_line_breaks_stay_unknown() -> None:
    layer = LAYER.replace("conditions in", "condi-\ntions in")
    # both sides carry the break identically: still not confirmed, the join is a guess
    assert not _agrees(layer, _page(layer))
    # the reading joined it: the layer's two fragments are not the reading's word
    assert not _agrees(LAYER, _page(layer))


# --- the gates: what may NOT settle the clause ---------------------------------------


def test_a_failed_layer_scan_stays_unknown() -> None:
    assert _agrees(LAYER)
    for flag in ("prior_ocr_layer_scan_failed", "invisible_text_scan_failed"):
        assert not _agrees(LAYER, _page(**{flag: True})), flag
    # a failed scan alone: not a positive finding, so nothing to confirm
    assert not _agrees(LAYER, _page(prior_ocr_layer=False, prior_ocr_layer_scan_failed=True))
    assert _clause_result(
        LAYER, _page(prior_ocr_layer=False, prior_ocr_layer_scan_failed=True)
    ) is (CheckResult.UNKNOWN)


def test_a_page_that_is_not_suspect_is_untouched() -> None:
    p = _page(prior_ocr_layer_suspect=False, prior_ocr_layer=False)
    assert not _agrees(LAYER, p)


@pytest.mark.parametrize(
    "engine",
    [
        "marker",  # reads the embedded layer unless --force-ocr: independence not provable
        "consensus(marker)",
        "native",  # the layer itself
        "native+equations",
        "chart_asset",
        "reconciled(qwen,glm)",
        "",
    ],
)
def test_a_reading_that_may_derive_from_the_layer_never_counts(engine: str) -> None:
    """A native lane, a chart picture, a reconciliation and marker all agree with the layer by
    construction or cannot be shown independent of it."""
    assert _agrees(LAYER, engine="qwen")
    page = _page(engine=engine)
    assert not prior_ocr_layer_agreement(_read(LAYER, engine), page)
    assert _clause_result(LAYER, page, engine) is CheckResult.UNKNOWN


@pytest.mark.parametrize(
    "change",
    [
        {"table_floor_body": True},
        {"scanned_prose_recovered": True},
        {"status": PageStatus.ERROR},
    ],
)
def test_a_floor_body_under_a_model_label_never_counts(change: dict) -> None:
    """The floor replaces the model's text with the layer's prose and keeps the engine label."""
    control = _read(LAYER)
    assert prior_ocr_layer_agreement(control, _page())
    floor = PageOutput(**{**control.__dict__, **change})
    assert not prior_ocr_layer_agreement(floor, _page())


def test_consensus_of_an_image_reader_is_still_a_reading() -> None:
    assert _agrees(LAYER, engine="consensus(qwen)")


def test_a_whole_document_slice_is_not_a_page_major_reading() -> None:
    """A whole-document run hands the engine the PDF and its layer; its per-page slot is empty."""
    page = _page()
    page.best_output = None
    assert not prior_ocr_layer_agreement(_read(LAYER), page)


def test_the_winner_must_be_the_pages_own_attempt() -> None:
    page = _page(engine="glm")  # the page's attempt is glm; the shipped text claims qwen
    assert not prior_ocr_layer_agreement(_read(LAYER, "qwen"), page)


def test_agreement_is_not_evidence_for_any_other_clause() -> None:
    """The layer stays suspect: the numbers and words checks still refuse it."""
    from socr.core.verdict import text_layer_numbers_outcome, text_layer_words_outcome

    page = _page()
    page.own_layer_text = LAYER
    page.own_layer_word_reference = LAYER
    assert _agrees(LAYER, page)
    assert text_layer_numbers_outcome(_read(LAYER), page).entry.result is TraceResult.NOT_RUN
    assert text_layer_words_outcome(_read(LAYER), page).entry.result is TraceResult.NOT_RUN


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


def test_process_the_reading_alone_decides_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_gh1156_part_a_interim import _prior_ocr_pdf

    pdf = _prior_ocr_pdf(tmp_path / "scan.pdf", pages=1)
    # the layer of that fixture, read back from the same PDF the run reads
    import fitz

    with fitz.open(pdf) as doc:
        layer_text = doc[0].get_text()
    agree = _run(tmp_path / "agree", pdf, layer_text, monkeypatch)
    differ = _run(tmp_path / "differ", pdf, layer_text.replace("3 to 3", "3 to 5"), monkeypatch)

    agree_key = agree["sidecar"].get(PRIOR_OCR_AGREEMENT_KIND) is True
    differ_key = differ["sidecar"].get(PRIOR_OCR_AGREEMENT_KIND) is True
    assert agree_key and not differ_key, (agree["sidecar"].keys(), differ["sidecar"].keys())

    # every level that records clauses: the verdict, the audit log, the manifest, the sidecar
    assert _verdict_checks(agree["sidecar"]) == set()
    assert agree["sidecar"]["status"] == "success"
    assert differ["sidecar"]["status"] == "warning"
    assert "prior_ocr_layer" in _verdict_checks(differ["sidecar"])
    assert PRIOR_OCR_AGREEMENT_KIND in _event_kinds(agree["events"])
    assert PRIOR_OCR_AGREEMENT_KIND not in _event_kinds(differ["events"])
    assert "prior_ocr_layer_retained" in _event_kinds(differ["events"])
    assert "prior_ocr_layer_retained" not in _event_kinds(agree["events"])
    manifest = json.loads(next(iter((tmp_path / "agree").rglob("manifest.json"))).read_text())
    assert any(e.get("prior_ocr_layer_agreement") is True for e in manifest["entries"].values())
    # the document says what its pages say: clean for the confirmed page, warned for the other
    assert _doc_status(tmp_path / "agree") == "completed"
    assert _doc_status(tmp_path / "differ") != "completed"
    # the layer stays retained and suspect: confirming the page does not rehabilitate the layer
    assert agree["sidecar"]["prior_ocr_layer_suspect"] is True
    assert agree["sidecar"]["prior_ocr_layer_text"]
