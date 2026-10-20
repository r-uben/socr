"""#1156 Part B: an earlier OCR layer ships its prose, every number withheld.

The owner's ruling (2026-10-17, temporary until a second reader can confirm these pages):
a scan whose text layer is an EARLIER OCR run, and whose model reading was rejected or
never made, keeps that layer's prose under the canonical mark ("earlier OCR, not re-read"),
every number in it replaced by ``[number withheld, see image]`` unless an independent
reading confirms it, the original retained in the sidecar and never shipped; the page is
MARKED, never CHECKED; tables never ship as loose text. The orchestrator (2026-10-19): this
replaces #1027's delete as well -- what decides is whether the layer reads.

Every pin is a DIFFERENCE between two runs in one process (CLAUDE.md, #257): the flag on
and off, the readable line present and absent, the table event present and absent, the
masked ending enabled and disabled. Hermetic: no provider, no Ollama.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.audit_log import AuditEvent  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    PageEnding,
    PagePrimaryReason,
    SelectionProvenance,
    _select_and_finalize_page,
)
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState, PageState  # noqa: E402
from socr.tables.prior_ocr_mask import (  # noqa: E402
    INDEPENDENT_READER,
    NUMBER_WITHHELD,
    NUMBER_WITHHELD_NO_IMAGE,
    numeric_tokens_in_order,
)

LAYER = (
    "Raised discount rates from 3 to 3 Y, per cent and $400 million\n"
    "P\nr\ni\nc\ne\n"
    "The Committee met on July 12-13, 1983 and reviewed the outlook."
)
PNG = (
    "![Scanned page carrying an earlier, unchecked OCR layer 1]"
    "(figures/prior_ocr_layer_page_p1.png)"
)
MARK = "> [page 1 unchecked: prior_ocr_layer=UNKNOWN; scope=page; reason=earlier OCR, not re-read]"
_DIGIT = re.compile(r"[0-9²³¹¼-¾⁰-₟⅐-↉]")


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _layer_lines(text: str) -> str:
    """The shipped lines that came from the layer: not socr's mark, not its image ref."""
    return "\n".join(
        ln
        for ln in text.splitlines()
        if not ln.startswith("> [page")
        and not ln.startswith("![")
        and not re.fullmatch(r"\*\*Figure \d+\*\* \(page \d+\)", ln)
    )


def _state(
    *,
    detected: bool = True,
    invisible: bool = False,
    scan_failed: bool = False,
    attempts: bool = True,
    layer: str = LAYER,
    png: str = PNG,
) -> DocumentState:
    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = layer
    p.prior_ocr_layer = detected
    p.invisible_text_over_raster = invisible
    p.prior_ocr_layer_scan_failed = scan_failed
    p.prior_ocr_layer_suspect = detected or invisible or scan_failed
    p.needs_ocr_enhancement = True
    if invisible:
        p.invisible_scan_png_ref = png
    else:
        p.prior_ocr_page_png_ref = png
    if attempts:
        rejected = PageOutput(
            page_num=1,
            text="A MODEL READING THE JUDGE REJECTED",
            status=PageStatus.WARNING,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.AUDIT_FAILED,
        )
        p.attempts = [rejected]
        p.best_output = rejected
    state.pages[1] = p
    return state


# ---------------------------------------------------------------------------
# The ending
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invisible", [False, True], ids=["drawing-order", "render-mode-3"])
@pytest.mark.parametrize("attempts", [True, False], ids=["rejected", "never-ran"])
def test_a_readable_layer_ships_its_prose_with_every_number_withheld(
    attempts: bool, invisible: bool
) -> None:
    """The ruling, on both detector routes and both "no accepted reading" shapes."""
    control = _select_and_finalize_page(_state(detected=False, attempts=attempts), 1)
    record = _select_and_finalize_page(
        _state(detected=not invisible, invisible=invisible, attempts=attempts), 1
    )
    text = record.output.text

    assert "400" in control.output.text, "control: an unflagged page keeps its numbers"
    assert record.selection_provenance is SelectionProvenance.PRIOR_OCR_LAYER_MASKED
    assert record.disposition.ending is PageEnding.DEMOTED_NATIVE
    assert record.disposition.primary_reason is PagePrimaryReason.PRIOR_OCR_LAYER_MASKED
    # The prose is kept ...
    assert "Raised discount rates from" in text and "reviewed the outlook" in text
    # ... every number withheld, one token per number, the count recorded ...
    assert not _DIGIT.search(_layer_lines(text)), _layer_lines(text)
    assert record.output.numbers_withheld == len(numeric_tokens_in_order(LAYER)) == 6
    assert text.count(NUMBER_WITHHELD) == 6
    # ... the shreds dropped, the canonical mark first, the page image last.
    assert "\nP\n" not in f"\n{text}\n"
    assert text.splitlines()[0] == MARK
    assert text.rstrip().endswith(PNG.split("/")[-1].rstrip(")") + ")")
    # MARKED, never a clean pass, never the rejected reading, never "socr kept" it.
    assert record.output.status is PageStatus.WARNING and record.output.audit_passed is False
    assert record.output.failure_mode is FailureMode.PRIOR_OCR_LAYER_UNVERIFIED
    assert "A MODEL READING THE JUDGE REJECTED" not in text
    assert "socr kept this model reading" not in text


def test_finalize_refinalize_and_again_are_byte_identical() -> None:
    """Lesson 1: the transform runs once, on the retained original, before any socr line.

    ``_phase_assemble`` re-finalises every page with its saved body; a pass that masked
    THAT body would mask socr's own mark (``> [page [number withheld...``), double it, and
    cut the count to the one digit left in it.
    """
    state = _state()
    first = _select_and_finalize_page(state, 1)
    second = _select_and_finalize_page(state, 1, saved_text=first.output.text)
    third = _select_and_finalize_page(state, 1, saved_text=second.output.text)

    assert first.output.text == second.output.text == third.output.text
    assert first.output.numbers_withheld == second.output.numbers_withheld == 6
    assert third.output.numbers_withheld == 6
    assert second.output.text.count(MARK) == 1
    assert "[page [number withheld" not in second.output.text


def test_a_layer_where_no_word_survives_keeps_its_floor() -> None:
    """Argon (design review): a shredded layer plus a lone folio is not prose to keep."""
    shreds = "P\nr\ni\nc\ne\n19"
    for invisible, floor in (
        (True, SelectionProvenance.INVISIBLE_SCAN_UNREAD),
        (False, SelectionProvenance.PRIOR_OCR_LAYER_UNREAD),
    ):
        kept = _select_and_finalize_page(
            _state(detected=not invisible, invisible=invisible, layer=shreds), 1
        )
        readable = _select_and_finalize_page(
            _state(detected=not invisible, invisible=invisible, layer=shreds + "\nPrices rose."),
            1,
        )
        assert kept.selection_provenance is floor
        assert "19" not in _layer_lines(kept.output.text).replace("page 1", "")
        assert readable.selection_provenance is SelectionProvenance.PRIOR_OCR_LAYER_MASKED


def test_a_page_where_the_run_saw_a_table_keeps_its_floor() -> None:
    """Never loose table text: a table the run saw but could not place stays withheld."""
    plain = _state(invisible=True, detected=False)
    tabled = _state(invisible=True, detected=False)
    tabled.events.append(
        AuditEvent(page_num=1, kind="source_evidence_table_reject", engine="qwen", detail="x")
    )
    other_page = _state(invisible=True, detected=False)
    other_page.events.append(
        AuditEvent(page_num=2, kind="source_evidence_table_reject", engine="qwen", detail="x")
    )

    assert (
        _select_and_finalize_page(plain, 1).selection_provenance
        is SelectionProvenance.PRIOR_OCR_LAYER_MASKED
    )
    assert (
        _select_and_finalize_page(other_page, 1).selection_provenance
        is SelectionProvenance.PRIOR_OCR_LAYER_MASKED
    )
    assert (
        _select_and_finalize_page(tabled, 1).selection_provenance
        is SelectionProvenance.INVISIBLE_SCAN_UNREAD
    )


def test_a_failed_scan_alone_is_not_the_masked_ending() -> None:
    """Part A's rule stands: suspicion grants no subtraction; only a positive finding."""
    record = _select_and_finalize_page(_state(detected=False, scan_failed=True), 1)
    assert record.selection_provenance is not SelectionProvenance.PRIOR_OCR_LAYER_MASKED


def test_without_a_page_image_the_token_names_the_pdf_and_carries_no_digit() -> None:
    record = _select_and_finalize_page(_state(png=""), 1)
    text = record.output.text
    assert record.selection_provenance is SelectionProvenance.PRIOR_OCR_LAYER_MASKED
    assert NUMBER_WITHHELD not in text and text.count(NUMBER_WITHHELD_NO_IMAGE) == 6
    assert not _DIGIT.search(_layer_lines(text))
    assert "![" not in text


# ---------------------------------------------------------------------------
# The seam for a second reader (#1208)
# ---------------------------------------------------------------------------


def _with_record(record: dict) -> DocumentState:
    state = _state()
    state.pages[1].reading_agreement = record
    return state


_ALL = [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    ("record", "unmasked"),
    [
        ({"provenance": INDEPENDENT_READER, "agreed": True, "compared": 6, "confirmed": _ALL}, 6),
        ({"provenance": INDEPENDENT_READER, "agreed": True, "compared": 6, "confirmed": [2]}, 0),
        ({"provenance": INDEPENDENT_READER, "agreed": False, "compared": 6, "confirmed": _ALL}, 0),
        ({"provenance": "prior_ocr", "agreed": True, "compared": 6, "confirmed": _ALL}, 0),
        ({"provenance": "qwen_judge", "agreed": True, "compared": 6, "confirmed": _ALL}, 0),
        ({"provenance": INDEPENDENT_READER, "agreed": True, "compared": 5, "confirmed": _ALL}, 0),
        (
            {
                "provenance": INDEPENDENT_READER,
                "agreed": True,
                "compared": 6,
                # ``True == 1``: read as a position, it would complete the set.
                "confirmed": [True, 0, 2, 3, 4, 5],
            },
            0,
        ),
    ],
    ids=[
        "independent-whole-page",
        "subset",
        "disputed",
        "the-layer",
        "same-model-judge",
        "other-text",
        "bool",
    ],
)
def test_only_an_independent_whole_page_agreement_unmasks(record: dict, unmasked: int) -> None:
    """The seam for #1208. A subset is refused even when "agreed": a reader that confirmed
    the "3" of "3 1/2" but not its fraction would otherwise ship a wrong whole number."""
    out = _select_and_finalize_page(_with_record(record), 1).output
    assert out.numbers_withheld == 6 - unmasked
    if unmasked:
        assert "$400 million" in out.text and NUMBER_WITHHELD not in out.text
    else:
        assert "400" not in _layer_lines(out.text)
    # The page is MARKED either way: a second reader does not lift the mark in Part B.
    assert out.text.splitlines()[0] == MARK


# ---------------------------------------------------------------------------
# End to end, on the provider path the Fed run takes
# ---------------------------------------------------------------------------


def _png(w: int, h: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (w, h), 245).save(buf, format="PNG")
    return buf.getvalue()


#: A number-free paragraph wrapped over printed lines with a hyphen break, then a
#: separate paragraph carrying the numbers. #1074's reflow joins a paragraph only when
#: every printed line of it is in the body verbatim, so it can act on the first and never
#: on the masked second; if it ran on this ending, a fresh run would join the first and a
#: re-finalisation from the sidecar (which has no geometry) would not.
_SCAN_PARAGRAPHS = [
    [
        "The Committee reviewed the economic situa-",
        "tion and agreed that conditions warranted",
        "further restraint on bank credit growth.",
    ],
    [
        "It raised the discount rate from 3 to 3 Y, per cent.",
        "A total of $400 million was added to reserves.",
    ],
]


def _scan_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 120
    for paragraph in _SCAN_PARAGRAPHS:
        for line in paragraph:
            page.insert_text((72, y), line, fontsize=10)
            y += 13
        y += 26
    page.insert_image(page.rect, stream=_png(300, 400))
    # A born-digital second page, so the document has text even when page 1 is floored
    # and the run writes its ``.md`` in both arms.
    plain = doc.new_page(width=612, height=792)
    plain.insert_text((72, 120), "Appendix: the staff report follows in a separate volume.")
    doc.save(path)
    doc.close()
    return path


class _StubReadingEngine:
    name = "qwen"

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        return [
            PageOutput(page_num=n, text="A MODEL READING", status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


class _RejectingJudge:
    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=False, reason="fixed judge")


def _run(tmp_path: Path, monkeypatch, out_name: str, *, masked: bool = True) -> Path:
    from socr.core import manifest
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch
    from socr.pipeline.orchestrator import UnifiedPipeline

    # Built ONCE per test: a PDF saved twice differs in its timestamp, so its hash -- part
    # of every page fingerprint -- would differ between the two arms being compared.
    pdf = tmp_path / "scan.pdf"
    if not pdf.exists():
        _scan_pdf(pdf)
    out = tmp_path / out_name
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _StubReadingEngine())
        if not masked:  # the code as it was before Part B: the floors only
            m.setattr(manifest, "prior_ocr_masked_body", lambda p: ("", 0))
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
                save_figures=True,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._build_page_judge = lambda state: _RejectingJudge()
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        pipe.process(pdf, output_dir=out)
    return next(iter(out.rglob("pages"))).parent


def _read(doc_dir: Path) -> dict:
    side = json.loads((doc_dir / "pages" / "00001.json").read_text())
    return {
        "fragment": (doc_dir / "pages" / "00001.md").read_text(),
        "doc": next(iter(doc_dir.glob("*.md"))).read_text(),
        "side": side,
        "meta": json.loads((doc_dir / "metadata.json").read_text()),
        "manifest": json.loads((doc_dir / "manifest.json").read_text()),
        "kinds": sorted(e["kind"] for e in side.get("audit_events", [])),
        "audit_kinds": sorted(
            e["kind"]
            for e in json.loads((doc_dir / "audit_log.json").read_text())["events"]
            if e.get("page_num") == 1
        ),
        # Whole events, not just their kinds: an engine or detail written differently is a
        # difference (Argon, diff review: the tool named "native" where a run names "qwen").
        "events": sorted(json.dumps(e, sort_keys=True) for e in side.get("audit_events", [])),
        "audit_events": sorted(
            json.dumps(e, sort_keys=True)
            for e in json.loads((doc_dir / "audit_log.json").read_text())["events"]
            if e.get("page_num") == 1
        ),
        "root": json.loads((doc_dir.parent / "metadata.json").read_text()),
    }


def test_e2e_the_rejected_scan_ships_masked_prose_and_keeps_the_original(
    tmp_path: Path, monkeypatch
) -> None:
    from socr.core.result import DocumentStatus

    got = _read(_run(tmp_path, monkeypatch, "on"))
    floor = _read(_run(tmp_path, monkeypatch, "off", masked=False))

    assert "earlier OCR, not re-read — see image" in floor["fragment"], "control: the floor"
    text = got["fragment"]
    assert text.splitlines()[0].startswith("> [page 1 unchecked: prior_ocr_layer=UNKNOWN")
    assert "agreed that conditions warranted" in text
    assert not _DIGIT.search(_layer_lines(text)), _layer_lines(text)
    assert "A MODEL READING" not in text
    assert "figures/prior_ocr_layer_page_p1.png" in text
    side = got["side"]
    assert "3 Y," in side["prior_ocr_layer_text"] and "$400 million" in side["prior_ocr_layer_text"]
    assert side["winning_output"]["numbers_withheld"] == 3  # 3, 3, 400
    assert side["disposition"]["primary_reason"] == "prior_ocr_layer_masked"
    assert side["status"] == "warning" and side["audit_passed"] is False
    assert got["manifest"]["entries"]["1"]["disposition"] == side["disposition"]
    assert got["meta"]["status"] != DocumentStatus.SUCCESS.value
    assert "page_failed" not in got["audit_kinds"] and "page_failed" in floor["audit_kinds"]
    assert "prior_ocr_layer_retained" in got["audit_kinds"]


def test_e2e_a_resume_reproduces_the_same_bytes_and_count(tmp_path: Path, monkeypatch) -> None:
    """A masked page is WARNING, so a resume re-runs it from the PDF: same source, same
    transform, same bytes and the same counter -- never a re-mask of the saved body."""
    first = _read(_run(tmp_path, monkeypatch, "out"))
    again = _read(_run(tmp_path, monkeypatch, "out"))
    assert again["fragment"] == first["fragment"]
    assert again["doc"] == first["doc"]
    assert again["side"]["winning_output"] == first["side"]["winning_output"]


# ---------------------------------------------------------------------------
# Re-finalising a finished run from its sidecars, with no new OCR
# ---------------------------------------------------------------------------

#: Sidecar keys the re-finalisation must reproduce exactly as a fresh run writes them.
_RECORD_KEYS = (
    "disposition",
    "status",
    "failure_mode",
    "audit_passed",
    "engine",
    "provider",
    "verdict",
    "winning_output",
    "page_fingerprint",
    "prior_ocr_layer_text",
)


def test_refinalizing_a_floored_run_equals_a_fresh_run(tmp_path: Path, monkeypatch) -> None:
    """The finished Fed run was made before Part B. Re-finalising it must give exactly
    what running Part B fresh gives: page, document, sidecar, manifest, metadata, audit."""
    from socr.pipeline.refinalize_prior_ocr import refinalize_document

    fresh = _read(_run(tmp_path, monkeypatch, "fresh"))
    old_dir = _run(tmp_path, monkeypatch, "old", masked=False)
    before = _read(old_dir)
    changes = refinalize_document(old_dir)
    after = _read(old_dir)

    assert before["side"]["disposition"]["primary_reason"] == "prior_ocr_layer_unread"
    assert [(c.page_num, c.after[0]) for c in changes] == [(1, "prior_ocr_layer_masked")]
    assert after["fragment"] == fresh["fragment"]
    assert after["doc"] == fresh["doc"]
    for key in _RECORD_KEYS:
        assert after["side"].get(key) == fresh["side"].get(key), key
    assert after["events"] == fresh["events"]
    assert after["audit_events"] == fresh["audit_events"]
    assert {k: v for k, v in after["root"]["files"]["scan.pdf"].items() if k == "error"} == {
        k: v for k, v in fresh["root"]["files"]["scan.pdf"].items() if k == "error"
    }
    assert after["manifest"]["entries"]["1"] == fresh["manifest"]["entries"]["1"]
    assert after["meta"]["status"] == fresh["meta"]["status"]
    # Key PRESENCE too: a document with no error carries no ``error`` key, not a null.
    assert ("error" in after["meta"], after["meta"].get("error")) == (
        "error" in fresh["meta"],
        fresh["meta"].get("error"),
    )
    assert "page(s) 1 produced no usable output" in before["meta"].get("error", "")
    assert after["side"]["refinalized_from"]["disposition"] == before["side"]["disposition"]
    # A second pass finds nothing left to do.
    assert refinalize_document(old_dir) == []


# ---------------------------------------------------------------------------
# Argon, diff review
# ---------------------------------------------------------------------------


def test_a_table_event_added_after_the_first_finalisation_does_not_flip_the_ending() -> None:
    """``_phase_assemble`` adds events between the first finalisation and the last; the
    page's ending must not change under the saved body it already shipped."""
    state = _state()
    first = _select_and_finalize_page(state, 1)
    state.events.append(
        AuditEvent(page_num=1, kind="table_ladder_unverified", engine="", detail="late")
    )
    again = _select_and_finalize_page(state, 1, saved_text=first.output.text)
    assert first.selection_provenance is SelectionProvenance.PRIOR_OCR_LAYER_MASKED
    assert again.selection_provenance is SelectionProvenance.PRIOR_OCR_LAYER_MASKED
    assert again.output.text == first.output.text
    assert again.output.numbers_withheld == first.output.numbers_withheld


@pytest.mark.parametrize("attempts", [True, False], ids=["rejected", "never-ran"])
def test_a_structure_class_page_never_ships_its_layer_as_prose(attempts: bool) -> None:
    """The lookahead declines on some structure-class pages (no model ran, or a flagged
    model table kept); the layer's table rows must not ship as loose text there."""
    plain = _state(attempts=attempts)
    tabled = _state(attempts=attempts)
    tabled.pages[1].has_tables = True
    assert (
        _select_and_finalize_page(plain, 1).selection_provenance
        is SelectionProvenance.PRIOR_OCR_LAYER_MASKED
    )
    assert (
        _select_and_finalize_page(tabled, 1).selection_provenance
        is not SelectionProvenance.PRIOR_OCR_LAYER_MASKED
    )


def test_figure_extraction_skips_only_a_page_that_ships_its_own_image() -> None:
    from socr.core.manifest import page_ships_its_own_page_image

    p = PageState(page_num=1)
    p.prior_ocr_page_png_ref = PNG
    assert page_ships_its_own_page_image(p, f"prose\n\n{PNG}")
    assert not page_ships_its_own_page_image(p, "a model reading that ships no page image")
    p2 = PageState(page_num=1)
    p2.scanned_figure_png_ref = "![f](figures/scanned_figure_page_p1.png)"
    assert page_ships_its_own_page_image(p2, "")
    assert not page_ships_its_own_page_image(PageState(page_num=1), "anything")
