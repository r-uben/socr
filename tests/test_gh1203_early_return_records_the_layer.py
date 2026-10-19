"""#1203: a page rejected for ANOTHER reason still records what its text layer is.

``_assess_page_signals`` returns ten times before #1156's drawing-order check runs. Each
of those returns is a born-digital rejection, so the page is routed to OCR either way --
but it recorded no ``prior_ocr_layer``, and therefore got no page mark, no sidecar
retention, no document bucket and no acceptance-meter lane.
Measured: 307 pages of the 455-PDF Fed census's own 355 (~3,960 of the corpus at x12.9)
and 80 pages of the corpus probe set, zero in the other direction, zero routing changes.

Two things the FIRST round of this file got wrong, both corrected here (round 1 review):

* it said nothing changes what ships, because every native floor sits inside
  ``manifest.py``'s ``if p.is_born_digital and p.native_text:`` branch. Most do;
  ``UNVERIFIABLE_TABLE_SCANNED`` does not, and #649's prose floor shipped the earlier OCR
  run's paragraphs from there. Pinned by
  ``test_the_scanned_prose_floor_withholds_the_layers_own_prose``, which drives the REAL
  judge chain, because a fixed judge cannot produce the rejection that reaches it.
* an ACCEPTED reading on one of these pages ships ``warning`` /
  ``prior_ocr_layer_unverified`` -- not SUCCESS -- so the page status is part of what this
  change adds, and the per-page resume ledger never admits such a page (#1192).

Every pin here is a DIFFERENCE between two runs in ONE process that differ in exactly one
thing -- whether the detector's #1156 route recognises the layer -- never an absolute
outcome measured on this machine (CLAUDE.md's no-provider trap, #257). Hermetic: a stub
engine, a fixed judge, the provider ladder and the judge model pinned, no Ollama.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.born_digital import BornDigitalDetector  # noqa: E402
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402

#: Tokens of the fixture's own text layer. A number among them, because what the owner's
#: ruling protects is a number the earlier OCR run read wrong.
_LAYER_TOKENS = ("3 Y,", "$400 million")


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _png(w: int, h: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (w, h), 245).save(buf, format="PNG")
    return buf.getvalue()


#: The two damage classes are written into the text layer, not into the page geometry, so
#: the page is the SAME prior-OCR scan in both arms and only the early return differs.
#: ``cid`` is PyMuPDF's unmapped-glyph artefact; ``digit`` is #136's eaten leading digit
#: ("(/997)" for "(1997)"), which is what an old OCR run of a typewritten Fed page does.
_EARLY_RETURN_DAMAGE = {
    "cid": ("(cid:3)(cid:7)(cid:11) ", "CID font mapping artifacts"),
    "digit": ("Reported in (/997) and again in (/985). ", "digit corruption"),
}

#: The control: the same page with NO early-return damage, which is the page Part A
#: measured. Not in the parametrised set above -- it has no early return to still fire.
_UNDAMAGED = ("", "")


def _early_return_prior_ocr_pdf(path: Path, damage: str, *, pages: int = 2) -> Path:
    """A painted-under prior-OCR scan whose text layer ALSO trips an early return.

    The raster is inserted after the text and covers the page, which is the #1156 family
    (painted over, render mode 0). Each page's visible-word test still has to pass, so the
    prose carries a per-page nonce exactly as the Part A fixture does -- without it the
    repeated-slot scan reads the body itself as page furniture.
    """
    prefix, _reason = _EARLY_RETURN_DAMAGE.get(damage, _UNDAMAGED)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=612, height=792)
        nonce = "w" * (i + 1)
        sentences = [
            f"{nonce} {prefix}The Committee met and reviewed conditions in the economy.",
            f"{nonce} Discount rates were raised from 3 to 3 Y, per cent at the Banks.",
            f"{nonce} Open market operations totalled $400 million over the period.",
            f"{nonce} The Secretary read the minutes of the previous meeting.",
        ]
        for row, sentence in enumerate(sentences):
            page.insert_text((72, 120 + row * 24), sentence, fontsize=10)
        page.insert_image(page.rect, stream=_png(300, 400))
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# The detector: the flag no longer depends on which return fired
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("damage", sorted(_EARLY_RETURN_DAMAGE))
def test_an_early_return_page_still_records_its_layer(tmp_path: Path, damage: str) -> None:
    """The fix, and its two premises, in one place.

    The premises matter as much as the conclusion: without the first the page never took
    an early return and the test would pass on main, and without the second the page is
    not a prior-OCR layer at all.
    """
    pdf = _early_return_prior_ocr_pdf(tmp_path / f"{damage}.pdf", damage)
    page = BornDigitalDetector().detect(pdf).pages[0]

    # Premise 1: an early return fired, and it still fires FOR ITS OWN REASON.
    assert page.is_born_digital is False
    assert page.needs_ocr_enhancement is False, (
        "the page reached the end of the signals body, so no early return fired and this "
        "test cannot show anything about #1203"
    )
    _prefix, reason = _EARLY_RETURN_DAMAGE[damage]
    assert any(reason in note for note in page.notes), (
        f"the early return's own reason is gone from the notes: {page.notes}"
    )

    # The fix: the layer is recorded anyway.
    assert page.prior_ocr_layer is True, (
        "a page rejected for another reason recorded no prior-OCR layer, so it gets no "
        "mark, no retention and no document bucket (#1203)"
    )
    # ... and retained, because ``native_text`` is empty on a rejected page BY DESIGN, so
    # the sidecar's existing retention had nothing to write.
    assert page.native_text == "", "a rejected page must ship no native text"
    assert any(tok in page.prior_ocr_layer_text for tok in _LAYER_TOKENS), (
        f"the layer was not retained for Part B: {page.prior_ocr_layer_text[:120]!r}"
    )
    assert "Committee" in page.prior_ocr_layer_text
    # The audit log's own trace of why, in the SAME wording the born-digital path uses:
    # without it the only record of the family on these pages is a boolean.
    assert any("earlier OCR reading" in note for note in page.notes), (
        f"the #1156 reason is not in the notes: {page.notes}"
    )


def test_a_failed_scan_on_an_early_return_page_is_unknown_not_clean(
    tmp_path: Path, monkeypatch
) -> None:
    """The fail-closed arm, on the page class #1203 adds.

    Part A pins it on a born-digital page. Here the predicate raises on a page that an
    early return also rejects: the page must record "unknown" (which marks it and keeps
    the document out of SUCCESS) and must still retain its layer, because a failed scan
    is not a reason to throw the page's own text away.
    """

    def _raise(self, page, slots, folios, **kw):
        raise RuntimeError("#1203: a deliberately unreadable page")

    monkeypatch.setattr(BornDigitalDetector, "_has_prior_ocr_layer", _raise)
    page = (
        BornDigitalDetector()
        .detect(_early_return_prior_ocr_pdf(tmp_path / "f.pdf", "cid"))
        .pages[0]
    )

    assert page.is_born_digital is False, "fixture premise: an early return still fired"
    assert page.prior_ocr_layer is False
    assert page.prior_ocr_layer_scan_failed is True, (
        "an unreadable page read as clean, which is the fail-open direction"
    )
    assert any("scan failed: unknown" in note for note in page.notes), page.notes
    assert any(tok in page.prior_ocr_layer_text for tok in _LAYER_TOKENS), (
        "a failed scan dropped the page's own layer"
    )


def test_the_born_digital_path_is_unchanged_by_the_new_stamping_point(tmp_path: Path) -> None:
    """The control arm: an undamaged prior-OCR page keeps Part A's exact outcome.

    The flag is now computed before the signals body and stamped after it. If that
    stamping lost or re-derived the answer, THIS page -- the one Part A measured -- is
    where it would show.
    """
    pdf = _early_return_prior_ocr_pdf(tmp_path / "clean.pdf", "clean")
    page = BornDigitalDetector().detect(pdf).pages[0]

    assert page.is_born_digital is True
    assert page.prior_ocr_layer is True
    assert page.needs_ocr_enhancement is True, "Part A's routing of a flagged page is gone"
    # EXACTLY once (#1203 round 1, r-uben nit): the note is appended at two stamping
    # points, and dropping ``and not assessment.is_born_digital`` from the second gate
    # made this page carry it twice while every assertion still passed.
    assert sum("earlier OCR reading" in note for note in page.notes) == 1, page.notes
    assert any(tok in page.native_text for tok in _LAYER_TOKENS)
    # The retention field stays EMPTY here: ``native_text`` already carries the layer and
    # is what the sidecar writes, so writing it twice would change every flagged sidecar.
    assert page.prior_ocr_layer_text == ""


# ---------------------------------------------------------------------------
# process(): the difference the fix makes, end to end
# ---------------------------------------------------------------------------


def test_the_retention_survives_a_flush_and_a_restore_on_an_early_return_page(
    tmp_path: Path, monkeypatch
) -> None:
    """Part A's retention contract, for the page class that had no carrier for it.

    The page state is taken FROM THE DETECTOR, so the combination it pins -- rejected,
    no native text, a retained layer -- is production's, not the test's invention. Both
    halves matter: the sidecar must carry the layer although ``native_text`` is empty,
    and a restore must put it back where the re-flush reads it, or the next write drops
    the one thing Part B re-finalizes from.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.document import DocumentHandle
    from socr.core.result import FailureMode
    from socr.core.state import DocumentState, PageState
    from socr.pipeline.orchestrator import UnifiedPipeline

    pa = (
        BornDigitalDetector()
        .detect(_early_return_prior_ocr_pdf(tmp_path / "e.pdf", "cid"))
        .pages[0]
    )
    assert pa.is_born_digital is False and pa.native_text == "", "fixture premise"
    assert pa.prior_ocr_layer_text, "fixture premise: the detector retained the layer"

    # Hermetic: the run fingerprint resolves the engine and the judge, and both reach a
    # live Ollama host unless they are pinned (CLAUDE.md's no-provider trap).
    pipe = UnifiedPipeline(
        PipelineConfig(
            quiet=True,
            write_manifest=False,
            judge_backend="heuristic",
            save_figures=False,
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
        )
    )
    monkeypatch.setattr(type(pipe), "_resolve_judge_model", lambda *a, **k: "", raising=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    p = PageState(page_num=1)
    p.is_born_digital = pa.is_born_digital
    p.native_text = pa.native_text
    p.prior_ocr_layer = pa.prior_ocr_layer
    p.prior_ocr_layer_text = pa.prior_ocr_layer_text
    p.prior_ocr_layer_suspect = True
    p.best_output = PageOutput(
        page_num=1,
        text="A MODEL READING OF THE PAGE",
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )
    state.pages[1] = p
    sidecar = json.loads(
        Path(pipe._flush_page_sidecar(state, 1, out_dir, terminal=True)).read_text()
    )

    assert any(tok in sidecar.get("prior_ocr_layer_text", "") for tok in _LAYER_TOKENS), (
        "the sidecar dropped the layer of a page whose native_text is empty"
    )
    assert all(tok not in sidecar["winning_output"]["text"] for tok in _LAYER_TOKENS)

    resumed = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    resumed.pages[1] = PageState(page_num=1)
    pipe._restore_terminal_page_state(
        resumed, 1, PageOutput.from_dict(sidecar["winning_output"]), out_dir
    )
    assert any(tok in (resumed.pages[1].prior_ocr_layer_text or "") for tok in _LAYER_TOKENS), (
        "the restore lost the retention, so the next flush drops it"
    )
    # ... and NOT into ``native_text``. Without this the re-flush below passes through
    # ``native_text`` and never reads the carrier, which made the whole second half of
    # this test vacuous (Argon, diff review); and a rejected page carrying the layer in
    # ``native_text`` is what makes it native-eligible under ``--native-only``.
    assert not resumed.pages[1].native_text, (
        "the restore put the earlier OCR layer into the field that decides native eligibility"
    )
    again = json.loads(
        Path(pipe._flush_page_sidecar(resumed, 1, out_dir, terminal=True)).read_text()
    )
    assert any(tok in again.get("prior_ocr_layer_text", "") for tok in _LAYER_TOKENS), (
        "the re-flush wiped the retention"
    )


def test_no_page_image_is_rendered_where_nothing_can_reference_it(
    tmp_path: Path, monkeypatch
) -> None:
    """#1203 round 1 (Argon P2b): the render had no reader on this page class.

    Both readers of ``prior_ocr_page_png_ref`` sit inside ``manifest.py``'s
    ``if p.is_born_digital and p.native_text:`` branch, so on a page an early rejection
    produced it was a page rasterised and written per page for nothing. Pinned as a
    DIFFERENCE between the two page classes, on the rejected arm where the render used to
    fire: the born-digital prior-OCR page still gets its image, and the early-return page
    does not.
    """
    early = _run(tmp_path, monkeypatch, "png-early", accept=False, route_live=True)
    assert early["side"].get("prior_ocr_layer_suspect") is True, "fixture premise"
    assert not list((tmp_path / "out-png-early").rglob("*prior_ocr_layer_page*")), (
        "a page image was rendered for an ending that cannot reference it"
    )

    # The control: Part A's own page class still renders and still SHIPS it.
    from socr.core.manifest import is_page_failed_marker

    born = _run_born_digital_floor(tmp_path, monkeypatch, "png-born")
    assert list((tmp_path / "out-png-born").rglob("*prior_ocr_layer_page*")), (
        "the born-digital flagged page lost the image its floor marker points at"
    )
    body = [ln for ln in born["text"].splitlines() if ln.strip() and not ln.startswith(">")]
    assert is_page_failed_marker(body[0]) and "see image" in born["text"]


def _run_born_digital_floor(tmp_path: Path, monkeypatch, tag: str) -> dict:
    """Part A's own page: a painted-under scan with NO early-return damage, reading
    rejected, so it reaches the #1156 floor that references the page image."""
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _StubReadingEngine())
        pipe = orch.UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=True,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._build_page_judge = lambda state: _FixedJudge(False)
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        out = tmp_path / f"out-{tag}"
        pipe.process(_early_return_prior_ocr_pdf(tmp_path / f"{tag}.pdf", "clean"), output_dir=out)
    return {"text": next(iter(out.rglob("pages/00001.md"))).read_text()}


def test_the_retained_layer_is_cleaned_like_the_native_reading() -> None:
    """ "Cleaned exactly as ``_assess_page_signals`` cleans it" was unfalsifiable (r-uben 4).

    Dropping ``clean_native_text`` from ``_retained_layer_text`` left the whole suite
    green, because the publisher-injected characters that clean removes cannot be put into
    a synthetic fixture: ``insert_text`` with a base-14 font silently substitutes a middle
    dot for a zero-width space, a soft hyphen and a thin space alike (measured -- none of
    the five candidates survived extraction). So the contract is pinned on the HELPER, with
    a page whose text layer does carry them, which is strictly stronger than a character
    the PDF writer replaces before it is ever stored.
    """

    class _PageWithInjectedCharacters:
        def get_text(self, _kind):
            return "The Com\u200bmittee met\u00ad and\u2009reviewed conditions."

    retained = BornDigitalDetector()._retained_layer_text(_PageWithInjectedCharacters())

    assert "\u200b" not in retained and "\u00ad" not in retained, (
        "the retained layer keeps publisher-injected characters the native reading strips, "
        "so Part B would re-finalize from different bytes"
    )
    assert "Committee" in retained and "reviewed conditions" in retained, (
        "the clean removed more than the injected characters"
    )


def test_an_unreadable_layer_retains_nothing_rather_than_raising() -> None:
    """The guard on that helper: every ``get_text`` in the module is wrapped, and this one
    runs on pages the detector has already decided it cannot trust."""

    class _RaisingPage:
        def get_text(self, _kind):
            raise RuntimeError("#1203: an unreadable layer")

    assert BornDigitalDetector()._retained_layer_text(_RaisingPage()) == ""


def test_the_drawing_order_scan_runs_exactly_once_per_page(tmp_path: Path, monkeypatch) -> None:
    """ "Once per page" was a claim the suite could not fail (#1203 round 1, r-uben 4).

    The answer is computed in ``_assess_page`` and handed to the signals body on a side
    channel. Delete the hand-over and nothing breaks: the body simply computes it again,
    so every page of the corpus pays for two drawing-order scans and the docstring's
    "exactly once" becomes false silently. Counted here at ``detect()``, over a document
    with both a flagged and an unflagged page, so the count is per page and not per
    document.
    """
    calls: list[int] = []
    real = BornDigitalDetector._has_prior_ocr_layer

    def _counting(self, page, slots, folios, **kw):
        calls.append(page.number)
        return real(self, page, slots, folios, **kw)

    monkeypatch.setattr(BornDigitalDetector, "_has_prior_ocr_layer", _counting)
    pdf = _early_return_prior_ocr_pdf(tmp_path / "count.pdf", "cid", pages=3)
    assessment = BornDigitalDetector().detect(pdf)

    assert len(assessment.pages) == 3, "fixture premise"
    assert len(calls) == 3, (
        f"the drawing-order scan ran {len(calls)} times for 3 pages; the side channel is "
        "not being used, so every page is scanned twice"
    )
    assert sorted(calls) == [0, 1, 2], f"one page was scanned twice and another not at all: {calls}"


def test_a_direct_signals_call_is_never_answered_from_the_previous_page(tmp_path: Path) -> None:
    """The side channel is CONSUMED, so a stale value cannot answer the next page.

    ``_assess_page`` hands the #1156 answer to ``_assess_page_signals`` on
    ``self._prior_ocr_this_page``. A caller that reaches the signals body without going
    through ``_assess_page`` -- any test, and any future second entry point -- would
    otherwise be answered by whichever page was assessed last. Pinned on the pair that
    makes the failure visible: a flagged page, then a clean one, in that order.
    """
    import fitz as _fitz

    flagged = _early_return_prior_ocr_pdf(tmp_path / "flagged.pdf", "cid")
    clean = tmp_path / "clean-born.pdf"
    doc = _fitz.open()
    page = doc.new_page(width=612, height=792)
    for row in range(6):
        page.insert_text(
            (72, 120 + row * 24),
            f"Line {row} of an ordinary born-digital page with no raster over it at all.",
            fontsize=10,
        )
    doc.save(clean)
    doc.close()

    det = BornDigitalDetector()
    assert det.detect(flagged).pages[0].prior_ocr_layer is True, "fixture premise"

    with fitz.open(clean) as cdoc:
        det._read_document_furniture(cdoc)
        direct = det._assess_page_signals(cdoc[0], 1, (1.0, 0.0))
    assert direct.prior_ocr_layer is False, (
        "a clean page was answered from the previously assessed page's side channel"
    )


def _body_without_marks(text: str) -> str:
    """The shipped reading, with the verdict banner's quoted lines removed.

    The mark is the ONE thing the flag is meant to add, so comparing bodies means
    comparing everything else: blockquote lines are the banner, and every other
    non-empty line is content.
    """
    return "\n".join(
        ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith(">")
    )


class _StubReadingEngine:
    """One reading per page, so a fixed judge can accept or reject it."""

    name = "qwen"

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        return [
            PageOutput(
                page_num=n,
                text="A MODEL READING OF THE PAGE",
                status=PageStatus.SUCCESS,
                engine="qwen",
            )
            for n in page_nums
        ]


class _FixedJudge:
    def __init__(self, accept: bool) -> None:
        self._accept = accept

    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=self._accept, reason="fixed judge")


def _run(tmp_path: Path, monkeypatch, tag: str, *, accept: bool, route_live: bool) -> dict:
    """``process()`` on the early-return fixture, with the #1156 route live or neutralised.

    ``route_live=False`` is the ONLY difference between the arms: the detector's
    drawing-order predicate answers False, which is what main records on these pages.
    Everything a provider decides is pinned (ladder, engines, judge, judge model), so the
    two arms are comparable in CI, where there is no provider at all.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch
    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf = _early_return_prior_ocr_pdf(tmp_path / f"{tag}.pdf", "cid")
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _StubReadingEngine())
        if not route_live:
            m.setattr(
                BornDigitalDetector,
                "_has_prior_ocr_layer",
                lambda self, page, slots, folios, **kw: False,
            )
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=True,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._build_page_judge = lambda state: _FixedJudge(accept)
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        result = pipe.process(pdf, output_dir=tmp_path / f"out-{tag}")
    out = tmp_path / f"out-{tag}"
    side = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    text = next(iter(out.rglob("pages/00001.md"))).read_text()
    # The audit log is written only when the run produced events, so a genuinely clean
    # control arm has no file. Absent means "no event of any kind", which is what the
    # control side of the pin asserts; the live side asserts a PRESENT kind, and that
    # cannot pass without the file.
    logs = list(out.rglob("audit_log.json"))
    events = json.loads(logs[0].read_text())["events"] if logs else []
    return {
        "status": result.status,
        "side": side,
        "text": text,
        "kinds": sorted({e.get("kind") for e in events if e.get("page_num") == 1}),
        # The ROUTING answer, which must be IDENTICAL across the arms: #1203 is a
        # surfacing and retention change, and these pages were re-read either way.
        "engines": [a.get("engine") for a in side.get("attempts", [])],
        "needs_ocr": side.get("needs_ocr_enhancement"),
    }


@pytest.mark.parametrize("accept", [True, False])
def test_the_flag_the_retention_and_the_mark_appear_only_with_the_route(
    tmp_path: Path, monkeypatch, accept: bool
) -> None:
    """The difference pin at ``process()``, on both judge arms.

    Both judge arms: the reading ships, under the page's mark, and the page enters the
    document's prior-OCR bucket (``prior_ocr_layer_retained``), so the document can no
    longer report a clean SUCCESS over it. On main an early-return page reached none of
    that, because it carried no flag at all. What does NOT change is the shipped reading
    itself -- see the body comparison at the end.
    """
    live = _run(tmp_path, monkeypatch, f"live-{accept}", accept=accept, route_live=True)
    main = _run(tmp_path, monkeypatch, f"main-{accept}", accept=accept, route_live=False)

    # Routing is unchanged -- the claim the measurement had to confirm as exactly zero.
    assert live["engines"] == main["engines"], (
        f"the flag changed which engines ran: {live['engines']} vs {main['engines']}"
    )
    assert live["needs_ocr"] == main["needs_ocr"], "the flag changed the enhancement route"

    # The flag and the retention.
    assert live["side"].get("prior_ocr_layer_suspect") is True
    assert main["side"].get("prior_ocr_layer_suspect") is None, (
        "the control arm already carried the flag, so the pin shows nothing"
    )
    assert any(tok in live["side"].get("prior_ocr_layer_text", "") for tok in _LAYER_TOKENS)
    assert "prior_ocr_layer_text" not in main["side"]

    # Retained AND unshipped, which is the pair that matters: the same tokens are in the
    # sidecar and in neither arm's page. On its own "not in the text" would be vacuous
    # here (nothing in this fixture could put them there), so it is asserted beside the
    # sidecar's own copy of them (Argon, diff review).
    assert all(tok not in live["text"] for tok in _LAYER_TOKENS), "the earlier OCR layer shipped"

    # The document bucket, and the reason event behind the CLI line, in BOTH judge arms:
    # the page has a ``best_output`` and no failed marker, so it is a retained page
    # whichever way the judge went.
    assert "prior_ocr_layer_retained" in live["kinds"]
    assert "prior_ocr_layer_retained" not in main["kinds"]
    assert "A MODEL READING OF THE PAGE" in live["text"], "the model's reading did not ship"
    assert "prior_ocr_layer=UNKNOWN" in live["text"], "the reading of a scan shipped unmarked"
    assert "prior_ocr_layer=UNKNOWN" not in main["text"]

    # And the body of the reading is UNCHANGED. Part A's withholding floor, and every
    # other native floor, sits inside ``if p.is_born_digital and p.native_text:``
    # (manifest.py:4348), which an early-return page satisfies neither half of -- so
    # #1203 adds the mark, the retention and the bucket WITHOUT touching what ships.
    # Measured the same way by ``scratch/1203/ending_matrix.py`` over all four attempt
    # states; pinned here so a later change to that branch cannot start withholding a
    # reading on ~4,200 Fed pages unnoticed.
    assert _body_without_marks(live["text"]) == _body_without_marks(main["text"]), (
        "the flag changed the shipped reading, which #1203 is not allowed to do"
    )

    # The document status, as a DIFFERENCE. On the accepted arm main reports a clean
    # SUCCESS over this page and the flagged run cannot; on the rejected arm neither
    # does, so only the accepted arm discriminates and only it is asserted both ways.
    from socr.core.result import DocumentStatus

    assert live["status"] is not DocumentStatus.SUCCESS
    if accept:
        assert main["status"] is DocumentStatus.SUCCESS, (
            "main did not report SUCCESS over an unflagged accepted reading, so the "
            "document-status pin shows nothing"
        )


# ---------------------------------------------------------------------------
# Round 1: the #649 scanned-prose floor, which sits OUTSIDE the branch the first
# round reasoned about
# ---------------------------------------------------------------------------

_TABLE_LAYER_ROWS = [("Reserves", "512", "-380", "-18"), ("Borrowings", "204", "-77", "-9")]
#: Tokens of the LAYER's own reading: two withheld numeric cells, and a prose sentence
#: the #649 floor does ship on the control arm.
_TABLE_LAYER_TOKENS = ("The Committee reviewed conditions",)


def _scanned_table_pdf(path: Path) -> Path:
    """An early-return page with a RULED table, painted under a page raster.

    Ruled on purpose: ``find_tables()`` is what ``is_table_page`` asks in the judge chain,
    and an unruled column layout reported ``has_tables=False``, so the table chain was
    never entered at all (the first attempt at this reproduction).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=612, height=792)
        nonce = "w" * (i + 1)
        page.insert_text((72, 90), f"{nonce} Reported in (/997) and again in (/985).", fontsize=10)
        cols, rows_y = [72, 220, 330, 440, 540], [125, 145, 165, 185]
        for x in cols:
            page.draw_line(fitz.Point(x, rows_y[0]), fitz.Point(x, rows_y[-1]))
        for yy in rows_y:
            page.draw_line(fitz.Point(cols[0], yy), fitz.Point(cols[-1], yy))
        for label, x in (("Item", 76), ("Level", 224), ("Change", 334), ("Net", 444)):
            page.insert_text((x, 140), label, fontsize=9)
        y = 160
        for row in _TABLE_LAYER_ROWS:
            for value, x in zip(row, (76, 224, 334, 444)):
                page.insert_text((x, y), value, fontsize=9)
            y += 20
        # Digit-free prose: ``partition_prose_bands`` withholds any band carrying a
        # numeral, so a paragraph numbered "0" is withheld as a numeric row and the
        # control arm then has no prose for the fix to be seen withholding.
        for extra, line in enumerate(
            [
                "The Committee reviewed conditions in the economy and prices.",
                "Members discussed the outlook for employment and output.",
                "The Secretary read the minutes of the previous meeting.",
            ]
        ):
            page.insert_text((72, 230 + extra * 18), f"{nonce} {line}")
        page.insert_image(page.rect, stream=_png(300, 400))
    doc.save(path)
    doc.close()
    return path


class _TableEngine:
    name = "qwen"

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        table = (
            "| Item | Level | Change | Net |\n| --- | --- | --- | --- |\n"
            "| Reserves | 521 | 380 | 18 |\n| Borrowings | 204 | -77 | -9 |\n"
        )
        return [
            PageOutput(
                page_num=n,
                text=f"RESERVE AGGREGATES, MEMBER BANKS\n\n{table}",
                status=PageStatus.SUCCESS,
                engine="qwen",
            )
            for n in page_nums
        ]


def _run_real_judge(tmp_path: Path, monkeypatch, tag: str, *, route_live: bool) -> dict:
    """``process()`` with the REAL ``_build_page_judge`` chain and no witness backend.

    The judge chain is the point: the #649 floor is reached through
    ``SourceEvidenceTableJudge``'s rejection, which a fixed judge cannot produce. The
    classical-OCR witness is patched ABSENT in both arms -- it is absent on CI and on the
    cluster, and a box that has tesseract would take a different branch, so pinning it
    makes the two arms comparable everywhere.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch
    from socr.tables import source_evidence as se

    # ``monkeypatch.context()``, not the fixture itself: the two arms run in ONE test, so
    # a patch held by the fixture until teardown would still be in place for the second
    # arm. It was: the live arm ran with the detection route still neutralised and the
    # pin passed for the wrong reason.
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _TableEngine())
        m.setattr(
            se,
            "classical_ocr_with_state",
            lambda pix: ("", se.WITNESS_PACKAGE_MISSING, "test: no witness backend"),
        )
        m.setattr(
            se,
            "classical_ocr_backend_state",
            lambda: (se.WITNESS_PACKAGE_MISSING, "test: no witness backend"),
        )
        if not route_live:
            m.setattr(
                BornDigitalDetector,
                "_has_prior_ocr_layer",
                lambda self, page, slots, folios, **kw: False,
            )
        pipe = orch.UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=True,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        out = tmp_path / f"out-{tag}"
        pipe.process(_scanned_table_pdf(tmp_path / f"{tag}.pdf"), output_dir=out)
    side = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    return {"text": next(iter(out.rglob("pages/00001.md"))).read_text(), "side": side}


def test_the_scanned_prose_floor_withholds_the_layers_own_prose(tmp_path, monkeypatch) -> None:
    """#1203 round 1 (Argon P1, r-uben should-fix 2).

    ``native_prose_floor_text`` is a native floor OUTSIDE ``is_born_digital and
    native_text`` -- it is reached by ``not p.is_born_digital and
    scanned_table_evidence_failed``, which is an early-return Fed scan whose table could
    not be verified. It shipped the earlier OCR run's own paragraphs, which #1156's
    interim ruling forbids on a page with no accepted reading. The difference pin: with
    the route neutralised the page still ships that prose (main's behaviour), and with it
    live the page falls to the bare marker plus its image -- and the prose is still in the
    sidecar, where Part B reads it.
    """
    main = _run_real_judge(tmp_path, monkeypatch, "main", route_live=False)
    live = _run_real_judge(tmp_path, monkeypatch, "live", route_live=True)

    assert "the paragraphs below are this page's own text layer" in main["text"], (
        "the control arm did not reach the #649 prose floor, so this pin shows nothing"
    )
    assert any(tok in main["text"] for tok in _TABLE_LAYER_TOKENS), (
        "the control arm shipped no layer text, so there is nothing for the fix to withhold"
    )

    assert "the paragraphs below are this page's own text layer" not in live["text"]
    assert "unverifiable table — see image" in live["text"]
    assert all(tok not in live["text"] for tok in _TABLE_LAYER_TOKENS), (
        "the earlier OCR layer's prose still ships on a flagged page"
    )
    # Withheld, not destroyed: the layer is retained unshipped for Part B.
    assert "The Committee reviewed conditions" in live["side"].get("prior_ocr_layer_text", ""), (
        "the prose was withheld AND lost, which the ruling does not allow"
    )


# ---------------------------------------------------------------------------
# Round 1, the owner's ruling (2026-10-18) on r-uben's blocker 1: WITHHOLD
# ---------------------------------------------------------------------------

_COLUMN_ROWS = [
    ("Reserves", "512", "-380", "-18"),
    ("Borrowings", "204", "-77", "-9"),
    ("Excess", "308", "-303", "-9"),
    ("Required", "204", "-77", "-9"),
    ("Nonborrowed", "308", "-226", "-11"),
    ("Monetary base", "721", "-154", "-23"),
    ("Total", "1,029", "-457", "-34"),
]
_COLUMN_X = (76, 230, 340, 450)
#: The layer's three cells that the DISAGREEING model reading contradicts: two dropped
#: minus signs and one transposed pair, which only a reading of the pixels could settle.
_MODEL_ONLY_CELLS = ("521", "380", "18")
_LAYER_ONLY_CELLS = ("512", "-380", "-18")


def _column_table_pdf(path: Path, *, early_return: bool = True) -> Path:
    """An early-return page with a COLUMN-LAID-OUT table, painted under a raster.

    No ruling lines, on purpose, and this is the whole reproduction: a RULED table reports
    a detected table REGION and takes another path, where both arms withheld and nothing
    differed (my first attempt at this). The unruled one reports ``has_tables=True`` with
    ``detected_table_count=0`` -- ``is_table_page`` true, no table geometry -- which is the
    state in which ``NativeTableVerifierJudge`` runs and therefore the state whose skip the
    ruling is about.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=612, height=792)
        nonce = "w" * (i + 1)
        if early_return:
            page.insert_text(
                (72, 90), f"{nonce} Reported in (/997) and again in (/985).", fontsize=10
            )
        else:
            # No early-return damage: the page is BORN-DIGITAL and flagged, which is
            # #1202's own class and #1208's territory, not this rule's.
            page.insert_text((72, 90), f"{nonce} Minutes of the Federal Open Market.", fontsize=10)
        page.insert_text((72, 108), f"{nonce} RESERVE AGGREGATES, MEMBER BANKS", fontsize=10)
        y = 130
        for value, x in zip(("Item", "Level", "Change", "Net"), _COLUMN_X):
            page.insert_text((x, y), value, fontsize=9)
        y += 18
        for row in _COLUMN_ROWS:
            for value, x in zip(row, _COLUMN_X):
                page.insert_text((x, y), value, fontsize=9)
            y += 18
        for extra, line in enumerate(
            [
                "The Committee reviewed conditions in the economy and prices.",
                "Members discussed the outlook for employment and output.",
            ]
        ):
            page.insert_text((72, y + 16 + extra * 18), f"{nonce} {line}", fontsize=10)
        page.insert_image(page.rect, stream=_png(300, 400))
    doc.save(path)
    doc.close()
    return path


def _table_markdown(rows) -> str:
    return (
        "| Item | Level | Change | Net |\n| --- | --- | --- | --- |\n"
        + "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows)
        + "\n"
    )


class _ColumnTableEngine:
    """The stub reading, reporting the engine it stands in for.

    Argon P2 (round 4): this hardcoded ``engine="qwen"`` for every rung, so the escalation
    arm's "second model" was indistinguishable from the first in the output and the arm did
    not really exercise a second provider. The name now comes from the ``EngineType`` the
    factory was asked for, and the per-rung counter is shared across instances so the
    rung-dependent bodies still line up.
    """

    table = ""
    #: Per-rung bodies, when an arm needs the two rungs to differ (Argon 3: rung 1 splits
    #: the table in two blocks, rung 2 emits it unified). Consumed in order; the last
    #: entry repeats.
    per_rung: list[str] | None = None

    def __init__(self, name: str = "qwen", rung: list[int] | None = None) -> None:
        self.name = name
        self._rung_box = rung if rung is not None else [0]

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        body = self.table
        if self.per_rung:
            body = self.per_rung[min(self._rung_box[0], len(self.per_rung) - 1)]
        out = [
            PageOutput(
                page_num=n,
                text=f"RESERVE AGGREGATES, MEMBER BANKS\n\n{body}",
                status=PageStatus.SUCCESS,
                engine=self.name,
            )
            for n in page_nums
        ]
        self._rung_box[0] += 1
        return out


def _run_table_arm(
    tmp_path: Path,
    monkeypatch,
    tag: str,
    *,
    route_live: bool,
    agree: bool,
    witness: bool,
    pdf: Path | None = None,
    first_rung_witness_error: bool = False,
    witness_empty: bool = False,
    per_rung_tables: list[str] | None = None,
    swap_rows: bool = False,
    verifier_raises: bool = False,
    escalate: bool = False,
    html_table: str | None = None,
) -> dict:
    """``process()`` with the real judge chain, over the axes the rule needs.

    ``first_rung_witness_error`` gives the page a two-rung ladder and makes the witness
    fail on its FIRST call only: the transient shape of Argon's P1a, where rung 1 emits
    the page's no-witness event and rung 2's witness then reads the page.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL
    from socr.pipeline import agentic as agentic_mod
    from socr.pipeline import orchestrator as orch
    from socr.tables import source_evidence as se

    if swap_rows:
        # Grok P0: two rows EXCHANGED. The token multiset is identical to the correct
        # table, so a multiset verdict passes it; a row-level check cannot.
        first, second = _COLUMN_ROWS[0], _COLUMN_ROWS[1]
        model_rows = [(first[0], *second[1:]), (second[0], *first[1:]), *_COLUMN_ROWS[2:]]
    elif agree:
        model_rows = list(_COLUMN_ROWS)
    else:
        model_rows = [("Reserves", "521", "380", "18"), *_COLUMN_ROWS[1:]]
    with monkeypatch.context() as m:
        _rung_box = [0]
        _engines: dict[str, _ColumnTableEngine] = {}

        def _engine_for(engine_type):
            name = getattr(engine_type, "value", str(engine_type))
            if name not in _engines:
                eng = _ColumnTableEngine(name, _rung_box)
                eng.table = html_table if html_table is not None else _table_markdown(model_rows)
                eng.per_rung = list(per_rung_tables) if per_rung_tables else None
                _engines[name] = eng
            return _engines[name]

        m.setattr(orch, "get_engine", _engine_for)
        if witness:
            # A witness that READS the page's pixels. On this fixture the pixels carry the
            # LAYER's numbers, which is what a real tesseract on a real scan reads -- so
            # with ``agree=True`` the witness corroborates the model and with
            # ``agree=False`` it contradicts it.
            # ONE LINE PER ROW, which is what ``pytesseract.image_to_string`` produces
            # on a table. Round 4 removed the row-level confirmation this once fed; the
            # arm is kept because an AGREEING witness must still not release the table.
            # ``witness_empty``: the backend RAN and returned nothing. r-uben's third
            # arm in his should-fix 9 table, and distinct from both others -- the
            # package is installed (so not ``no witness backend``) and the call did not
            # fail (so not an exec error); it simply read no text off the raster.
            reading = "" if witness_empty else "\n".join(" ".join(r) for r in _COLUMN_ROWS)
            calls: list[int] = []

            def _witness(pix, _reading=reading, _calls=calls):
                _calls.append(1)
                if first_rung_witness_error and len(_calls) == 1:
                    return "", se.WITNESS_EXEC_ERROR, "test: transient witness failure"
                return _reading, se.WITNESS_READING, ""

            m.setattr(se, "classical_ocr_with_state", _witness)
            m.setattr(se, "classical_ocr_backend_state", lambda: (se.WITNESS_READING, ""))
        else:
            m.setattr(
                se,
                "classical_ocr_with_state",
                lambda pix: ("", se.WITNESS_PACKAGE_MISSING, "test: no witness backend"),
            )
            m.setattr(
                se,
                "classical_ocr_backend_state",
                lambda: (se.WITNESS_PACKAGE_MISSING, "test: no witness backend"),
            )
        if verifier_raises:
            # Grok P0 / Argon P1b: the gate RAISES, so its rejection reason is
            # ``table_verifier_error``, which ``_source_ev_rejected`` never matched -- the
            # path an unconfirmed table used to reach shipping around rounds 1 and 2.
            def _raise(*_a, **_k):
                raise RuntimeError("test: verify_scanned_table raised")

            m.setattr(agentic_mod, "verify_scanned_table", _raise, raising=False)
            m.setattr(se, "verify_scanned_table", _raise)
        if not route_live:
            m.setattr(
                BornDigitalDetector,
                "_has_prior_ocr_layer",
                lambda self, page, slots, folios, **kw: False,
            )
        pipe = orch.UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=True,
            )
        )
        pipe._available_engines_for_agentic = lambda: (
            [PROFILE_QWEN_LOCAL, PROFILE_GEMINI]
            if (first_rung_witness_error or escalate or per_rung_tables)
            else [PROFILE_QWEN_LOCAL]
        )
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        out = tmp_path / f"out-{tag}"
        pipe.process(pdf or _column_table_pdf(tmp_path / f"{tag}.pdf"), output_dir=out)
    side = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    text = next(iter(out.rglob("pages/00001.md"))).read_text()
    return {
        "text": text,
        "side": side,
        "ships_table": "| Reserves |" in text,
        "page_status": side["winning_output"]["status"],
        "failure_mode": side["winning_output"].get("failure_mode"),
    }


@pytest.mark.parametrize("agree", [False, True])
def test_an_unwitnessed_suspect_table_is_withheld(tmp_path: Path, monkeypatch, agree: bool) -> None:
    """Owner ruling 2026-10-18 on r-uben's blocker 1.

    Since #1202 a suspect page under an opaque raster skips row-level verification --
    correctly, the layer is not evidence -- which leaves the PIXELS as the only possible
    witness. Where nothing can read them the heuristic leaf accepted and #1048 shipped the
    model's table marked unverified. The ruling: WITHHOLD it. The page ships its picture
    plus a mark, and the reading stays in the sidecar for a later reader to release.

    Both parametrisations matter, and the second is the one that rules OUT "re-run the
    native verifier" as the fix:

    * ``agree=False`` -- the model contradicts the layer. A layer-based check would veto a
      reading that may well be right (#1202's own -350 case).
    * ``agree=True`` -- the model matches the layer. A layer-based check would PASS it, on
      the strength of a layer nobody trusts. Here the control arm does exactly that, and
      the ruling withholds it anyway.
    """
    main = _run_table_arm(
        tmp_path, monkeypatch, f"m-{agree}", route_live=False, agree=agree, witness=False
    )
    live = _run_table_arm(
        tmp_path, monkeypatch, f"l-{agree}", route_live=True, agree=agree, witness=False
    )

    if agree:
        # The premise of the Option-A-killing arm: without the flag the LAYER passed this
        # table and it shipped, carrying the layer's own cells.
        assert main["ships_table"] is True, "the control arm did not ship, so nothing is shown"
        assert any(c in main["text"] for c in _LAYER_ONLY_CELLS)

    assert live["ships_table"] is False, (
        "an unwitnessed table on a suspect page shipped; the ruling is to withhold it"
    )
    assert all(c not in live["text"] for c in _MODEL_ONLY_CELLS)
    assert "unverifiable table — see image" in live["text"]
    assert ".png" in live["text"], "the mark promises an image the page did not ship"
    assert live["page_status"] == "error"
    assert live["failure_mode"] == "no_witness_backend", (
        "the withheld page blames the model instead of recording that nothing read it"
    )
    # Withheld, not destroyed.
    retained = live["side"].get("withheld_table_text") or ""
    assert "| Reserves |" in retained, "the withheld reading is gone, so nothing can release it"
    assert "withheld_table_text" not in main["side"]


def test_a_witness_that_contradicts_the_model_withholds_the_table(
    tmp_path: Path, monkeypatch
) -> None:
    """Arm (iii). Round 1 pinned the OPPOSITE of this, and it was wrong (r-uben 1b).

    The witness runs and reads the page's pixels as 512/-380/-18; the model read
    521/380/18. "A witness ran" was satisfied, so round 1's rule let the table ship
    ``warning`` / ``table_unverified`` -- on a box with tesseract, main withholds that
    page. The rule is "an independent reader CONFIRMED the cells", and a reader that
    contradicted them confirmed nothing.
    """
    live = _run_table_arm(
        tmp_path, monkeypatch, "iii-live", route_live=True, agree=False, witness=True
    )
    assert live["ships_table"] is False, (
        "a table the witness CONTRADICTED shipped; confirmation is what the rule needs"
    )
    assert all(c not in live["text"] for c in _MODEL_ONLY_CELLS)
    assert "unverifiable table — see image" in live["text"]
    # The cause is the witness's own finding, not a missing install.
    assert live["failure_mode"] == "hallucination"
    assert "| Reserves |" in (live["side"].get("withheld_table_text") or "")


def test_even_an_agreeing_witness_does_not_release_the_table(tmp_path: Path, monkeypatch) -> None:
    """Round 4 (orchestrator decision): there is no release path in this PR.

    Rounds 2 and 3 shipped the table when the scanned-table gate's witness agreed with it.
    All three seats then broke the check that was supposed to make that agreement
    trustworthy -- the subsequence skipped words; dropped, shifted and merged cells passed;
    a dropped row passed; swapped header years passed -- and it could only fire at all on a
    box with Tesseract, which CI and Rorqual are not. So the agreement no longer releases
    anything: EVERY table on one of these pages is withheld and retained, which is the
    owner's ruling taken literally. #1208 builds the comparator that can release them.
    """
    live = _run_table_arm(
        tmp_path, monkeypatch, "agree-live", route_live=True, agree=True, witness=True
    )
    assert live["ships_table"] is False, (
        "an agreeing witness released the table; this PR has no release path"
    )
    assert "unverifiable table — see image" in live["text"]
    assert "| Reserves |" in (live["side"].get("withheld_table_text") or ""), (
        "withheld without retention, which is a deletion"
    )


@pytest.mark.parametrize("second_rung_agrees", [True, False])
def test_a_transient_witness_error_keeps_the_witnesss_own_cause(
    tmp_path: Path, monkeypatch, second_rung_agrees: bool
) -> None:
    """#658's cause through a two-rung ladder (Argon P1a/P1b, rounds 2-3).

    Rung 1's witness hits a transient execution error -- which emits the page's no-witness
    event -- and rung 2's witness then reads the page. Round 2 keyed the withhold on that
    event existing, so a table rung 2 had corroborated was destroyed; and the recorded
    cause was overwritten with ``no_witness_backend`` even when rung 2's witness had
    DISPUTED the table.

    Round 4 removes the release path, so BOTH arms are now withheld -- the question this
    test still answers is the cause, which must follow the witness's own finding and not
    the earlier missing install.
    """
    live = _run_table_arm(
        tmp_path,
        monkeypatch,
        f"v-{second_rung_agrees}",
        route_live=True,
        agree=second_rung_agrees,
        witness=True,
        first_rung_witness_error=True,
    )
    assert live["ships_table"] is False
    assert "| Reserves |" in (live["side"].get("withheld_table_text") or "")
    if second_rung_agrees:
        assert live["failure_mode"] == "no_witness_backend", (
            "nothing refuted this table, so the cause is the reader that never ran"
        )
    else:
        assert live["failure_mode"] == "hallucination", (
            "the cause was overwritten with a missing install, although a witness read "
            "the page and refused named cells"
        )


def test_a_layer_with_no_numeric_columns_still_withholds(tmp_path: Path, monkeypatch) -> None:
    """Arm (vi): the rule must not be answered from the untrusted layer (Argon P1b).

    Round 1 recorded the skipped row check only when ``_is_table_page`` was true, and that
    predicate asks ``has_numeric_columns`` of the LAYER. A layer with one value per row
    fails it, so the rule never fired and the unwitnessed table shipped. Nothing in the
    rule reads the layer now, so this page is withheld like any other.
    """
    pdf = tmp_path / "one-column-layer.pdf"
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=612, height=792)
        nonce = "w" * (i + 1)
        page.insert_text((72, 90), f"{nonce} Reported in (/997) and again in (/985).", fontsize=10)
        y = 130
        for row in _COLUMN_ROWS:
            # ONE numeric column in the layer, where the model emits four.
            page.insert_text((76, y), f"{row[0]} {row[1]}", fontsize=9)
            y += 18
        page.insert_text((72, y + 20), "The Committee reviewed conditions.", fontsize=10)
        page.insert_image(page.rect, stream=_png(300, 400))
    doc.save(pdf)
    doc.close()

    live = _run_table_arm(
        tmp_path, monkeypatch, "vi-live", route_live=True, agree=False, witness=False, pdf=pdf
    )
    assert live["side"].get("prior_ocr_layer_suspect") is True, "fixture premise: flagged"
    assert live["ships_table"] is False, (
        "an unwitnessed table shipped because the LAYER had no numeric columns"
    )
    assert "| Reserves |" in (live["side"].get("withheld_table_text") or "")


def test_a_suspect_page_with_no_table_is_not_touched(tmp_path: Path, monkeypatch) -> None:
    """r-uben's nit, as a behaviour: the rule applies to TABLES.

    Round 1's guard for this was ``_is_table_page``, which reads the layer; dropping it
    survived the suite. Round 2's docstring then said the rule "asks whether the WINNER
    carries a table block" -- it did not, that term had been removed as unkillable, and
    r-uben caught the stale sentence. Round 3's final gate DOES ask exactly that, of the
    shipped text (``find_table_blocks`` on the winner), so a flagged prose page is outside
    the rule by construction. Pinned, because "nothing happens here" is the claim a later
    change breaks silently.
    """
    live = _run(tmp_path, monkeypatch, "no-table", accept=True, route_live=True)
    assert live["side"].get("prior_ocr_layer_suspect") is True, "fixture premise: flagged"
    assert "withheld_table_text" not in live["side"]
    assert "unverifiable table — see image" not in live["text"]


@pytest.mark.parametrize(
    "arm,witness_kw",
    [
        # The three arms r-uben's should-fix 9 names. They differ in WHY no reader
        # vouched for the cells, and the Fed run meets all three: CI and Rorqual have no
        # Tesseract at all; a transient failure of an installed one is the exec error;
        # and a backend that ran and read nothing off the raster is the empty reading
        # (the shape `tesseract-is-the-weaker-reader-on-crops` measured on 27 of 91
        # figure crops).
        ("no-witness", {"witness": False}),
        ("exec-error", {"witness": True, "first_rung_witness_error": True}),
        ("empty-reading", {"witness": True, "witness_empty": True}),
    ],
)
def test_a_withheld_page_does_not_also_claim_its_table_shipped(
    tmp_path: Path, monkeypatch, arm: str, witness_kw: dict
) -> None:
    """Argon P2 / r-uben should-fix 9: a withheld page was carrying #1048's stamping.

    #1048's record means "the table shipped marked unverified". On a page whose table this
    run WITHHELD, nothing shipped -- and the sidecar still said
    ``unchecked_table_kind: unverified``, ``table_ladder_superseded`` and that sentence,
    beside a marker-only body. Two readings of the same page in one file.
    """
    # ``agree=True``: the arm where the SETTLING actually promotes an attempt and writes
    # its claim. With a disagreeing table nothing is promoted, so the strip is never
    # exercised and the assertions below pass for the wrong reason (measured).
    live = _run_table_arm(
        tmp_path, monkeypatch, f"p2-stamp-{arm}", route_live=True, agree=True, **witness_kw
    )
    assert live["ships_table"] is False, "fixture premise: the table is withheld"
    side = live["side"]
    # The page's own record must not say a table shipped. Measured: without the clears,
    # every withheld arm records these (``unverified`` / ``disputed`` and
    # ``table_unverified``) beside a marker-only body.
    assert side.get("unchecked_table_kind") in (None, ""), side.get("unchecked_table_kind")
    assert side.get("table_ladder_superseded") in (None, ""), side.get("table_ladder_superseded")
    assert side.get("unchecked_table_values") in (None, "", []), side.get("unchecked_table_values")
    assert "marked unverified" not in live["text"]
    # r-uben 9 / Argon P2, round 5. Round 4 removed the strip after measuring the claim
    # in the page BODY only and concluding no arm could produce it. It was in the
    # SIDECAR the whole time: on head 92b2c420 the no-witness arm wrote
    # ``winning_output.judge_reason = "ladder exhausted; table shipped marked unverified
    # (#1048); ..."``, the same sentence into ``attempts_summary[0].rejection_reason``,
    # and ``judge_accepted: true``; the exec-error and empty-reading arms wrote
    # ``judge_accepted: true`` beside a marker-only body too. So on every arm at least
    # one of these three assertions was failing before the fix, and the assertions are
    # on the SIDECAR -- not the body, which is what round 4 measured and why it
    # concluded wrongly.
    assert "table shipped marked" not in json.dumps(side), (
        "the sidecar still claims this page's table shipped"
    )
    assert side["winning_output"].get("judge_accepted") is not True, (
        f"[{arm}] the sidecar records an accepted judgement for a page with a marker body"
    )
    assert not any(
        "table shipped marked" in (att.get("rejection_reason") or "")
        for att in (side.get("attempts_summary") or [])
    ), f"[{arm}] attempts_summary still claims the table shipped"
    assert not side["winning_output"].get("judge_outcome"), side["winning_output"].get(
        "judge_outcome"
    )


def test_a_born_digital_flagged_page_does_not_grow_the_retention_key(
    tmp_path: Path, monkeypatch
) -> None:
    """Argon P2, second half: the key belongs to the pages this rule withholds.

    A BORN-DIGITAL flagged page with the same table is #1202's class and #1208's
    territory; the rule's scope is early-return pages. If the capture ran on it, a page
    whose table SHIPS would carry a "withheld" reading in its sidecar.
    """
    pdf = _column_table_pdf(tmp_path / "born-digital-flagged.pdf", early_return=False)
    live = _run_table_arm(
        tmp_path, monkeypatch, "p2-born", route_live=True, agree=False, witness=False, pdf=pdf
    )
    assert live["side"].get("prior_ocr_layer_suspect") is True, "fixture premise: flagged"
    # The sidecar does not record ``is_born_digital``, so the premise is asserted where it
    # is decided -- the detector's own assessment of the same file.
    assert BornDigitalDetector().detect(pdf).pages[0].is_born_digital is True, (
        "fixture premise: a born-digital page, not an early-return one"
    )
    assert "withheld_table_text" not in live["side"]


def test_a_clean_born_digital_table_page_is_untouched(tmp_path: Path, monkeypatch) -> None:
    """The control the ruling asks for: no raster, no prior-OCR layer, same table.

    If the rule leaked to ordinary born-digital table pages it would withhold tables across
    the corpus, so the page that carries none of the conditions is pinned explicitly.
    """
    pdf = tmp_path / "clean-table.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 90), "RESERVE AGGREGATES, MEMBER BANKS", fontsize=10)
    y = 130
    for value, x in zip(("Item", "Level", "Change", "Net"), _COLUMN_X):
        page.insert_text((x, y), value, fontsize=9)
    y += 18
    for row in _COLUMN_ROWS:
        for value, x in zip(row, _COLUMN_X):
            page.insert_text((x, y), value, fontsize=9)
        y += 18
    page.insert_text((72, y + 20), "The Committee reviewed conditions in the economy.", fontsize=10)
    doc.save(pdf)
    doc.close()

    live = _run_table_arm(
        tmp_path, monkeypatch, "clean", route_live=True, agree=False, witness=False, pdf=pdf
    )
    assert live["side"].get("prior_ocr_layer_suspect") is None, "fixture premise: not suspect"
    assert "withheld_table_text" not in live["side"], (
        "the no-witness withhold fired on a clean born-digital page"
    )
    assert "unverifiable table — see image" not in live["text"]


def test_the_withheld_table_survives_a_flush_a_restore_and_a_re_flush(
    tmp_path: Path, monkeypatch
) -> None:
    """The ruling's release path, end to end: retained, restored, and still there after.

    The sidecar is rebuilt from ``PageState`` on every flush, so a field that does not come
    back from a restore is a key that silently disappears -- which is exactly what happened
    to ``prior_ocr_layer_text`` in Part A's first round. Three halves, all required:

    1. the live run's own re-finalize keeps the key (the assertion below reads the sidecar
       AFTER ``process()`` has assembled, so this is the post-re-finalize state);
    2. ``_restore_terminal_page_state`` brings the reading back into its field;
    3. the next flush still writes it.

    Half 2 is defence, not a live path, and the last assertion is why: a withheld page is
    ERROR, so the per-page ledger never admits it and a normal resume does not reach that
    restore. It is pinned so a later change to the ledger or to the page's status cannot
    make it live unnoticed.
    """
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState, PageState

    live = _run_table_arm(
        tmp_path, monkeypatch, "survive", route_live=True, agree=False, witness=False
    )
    sidecar = live["side"]
    assert "| Reserves |" in (sidecar.get("withheld_table_text") or ""), (
        "the live run's re-finalize dropped the withheld reading"
    )
    assert sidecar["winning_output"]["status"] != "success", (
        "a withheld page became terminal SUCCESS, which puts it on the ledger's skip path"
    )

    # The restore reads the sidecar from ``out_dir``, so the chain has to start by writing
    # one there. The state it is written from carries the LIVE run's own values, not
    # invented ones, which is also the flush half of the pin.
    pipe = _hermetic_pipeline()
    out_dir = tmp_path / "restored"
    out_dir.mkdir()
    flushed_state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    source = PageState(page_num=1)
    source.is_born_digital = False
    source.native_text = ""
    source.prior_ocr_layer = True
    source.prior_ocr_layer_suspect = True
    source.withheld_table_text = sidecar["withheld_table_text"]
    source.best_output = PageOutput.from_dict(sidecar["winning_output"])
    flushed_state.pages[1] = source
    written = json.loads(
        Path(pipe._flush_page_sidecar(flushed_state, 1, out_dir, terminal=True)).read_text()
    )
    assert "| Reserves |" in (written.get("withheld_table_text") or ""), (
        "the flush dropped the withheld reading"
    )

    resumed = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    resumed.pages[1] = PageState(page_num=1)
    pipe._restore_terminal_page_state(
        resumed, 1, PageOutput.from_dict(sidecar["winning_output"]), out_dir
    )
    assert "| Reserves |" in (resumed.pages[1].withheld_table_text or ""), (
        "the restore lost the withheld reading, so the next flush drops it"
    )

    again = json.loads(
        Path(pipe._flush_page_sidecar(resumed, 1, out_dir, terminal=True)).read_text()
    )
    assert "| Reserves |" in (again.get("withheld_table_text") or ""), (
        "the re-flush wiped the withheld reading"
    )
    # And it is still only in the sidecar: nothing about a restore puts it on the page.
    assert "| Reserves |" not in (again["winning_output"]["text"] or "")


def _hermetic_pipeline():
    """A pipeline whose run fingerprint resolves no engine and no judge (CLAUDE.md)."""
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    pipe = UnifiedPipeline(
        PipelineConfig(
            quiet=True,
            write_manifest=False,
            judge_backend="heuristic",
            save_figures=False,
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
        )
    )
    pipe._resolve_judge_model = lambda *a, **k: ""
    return pipe


# ---------------------------------------------------------------------------
# Round 3: ONE final gate, and confirmation that is POSITIONAL
# ---------------------------------------------------------------------------


def test_a_row_swap_is_not_confirmed_by_a_multiset_agreeing_witness(
    tmp_path: Path, monkeypatch
) -> None:
    """Grok P0. Two rows exchanged; the witness reads the page correctly.

    Round 2 minted the confirmation from ``verify_table_tokens``, a token MULTISET, which
    a swap leaves identical — so the swapped table shipped as "confirmed", carrying
    Reserves' values on Borrowings' row. The row check that would have caught it is the
    native verifier's, and that one is skipped here because it reads the suspect layer.
    Confirmation is now a ROW-level check against the witness's own reading.
    """
    live = _run_table_arm(
        tmp_path, monkeypatch, "swap", route_live=True, agree=False, witness=True, swap_rows=True
    )
    assert live["ships_table"] is False, "a row swap shipped as confirmed"
    assert "| Reserves |" in (live["side"].get("withheld_table_text") or "")
    # The swapped cells are nowhere in the body.
    assert "204" not in live["text"] or "unverifiable table" in live["text"]


def test_a_raised_verifier_withholds_with_retention(tmp_path: Path, monkeypatch) -> None:
    """Grok P0 / Argon P1b: ``verify_scanned_table`` raises.

    Its rejection reason is then ``table_verifier_error``, which the upstream condition
    (`"source_evidence_table" in att.reason`) never matched, so `_source_ev_rejected`
    stayed False and the unconfirmed table shipped with no retention. The gate now runs on
    the table that is about to ship and does not care which upstream condition fired.
    """
    live = _run_table_arm(
        tmp_path,
        monkeypatch,
        "raise",
        route_live=True,
        agree=False,
        witness=False,
        verifier_raises=True,
    )
    assert live["ships_table"] is False, "a raised verifier shipped an unconfirmed table"
    assert all(c not in live["text"] for c in _MODEL_ONLY_CELLS)
    assert "| Reserves |" in (live["side"].get("withheld_table_text") or ""), (
        "withheld without retention, which is a deletion"
    )
    # r-uben 8: the CAUSE, asserted. A verifier that raised means nothing read the pixels,
    # so the page must not blame the model for inventing the table.
    assert live["failure_mode"] == "no_witness_backend", (
        "a crashed verifier was recorded as a hallucination"
    )


def test_an_escalated_second_model_table_is_not_shipped_unconfirmed(
    tmp_path: Path, monkeypatch
) -> None:
    """Argon P1a: escalation swaps in an unjudged second-model table.

    ``_escalate_table_page`` grades candidates against the page's own native rows — the
    SUSPECT LAYER here — and ships the winner without the scanned-table gate judging it,
    so a table that scores 100% against the earlier OCR run replaced a confirmed one. The
    final gate sees whatever ends up in the winner, so an unjudged table arrives
    unconfirmed and is withheld.
    """
    live = _run_table_arm(
        tmp_path,
        monkeypatch,
        "escalate",
        route_live=True,
        agree=False,
        witness=False,
        escalate=True,
    )
    assert live["ships_table"] is False
    assert "| Reserves |" in (live["side"].get("withheld_table_text") or "")


def test_the_scan_failed_page_with_no_raster_is_not_confirmed(tmp_path: Path, monkeypatch) -> None:
    """r-uben should-fix 6: the confirmation must never be minted from the layer.

    On a page with no page-covering raster the evidence bundle is
    ``build_born_digital_evidence``, whose evidence INCLUDES the PDF text layer — which on
    this page is the earlier OCR run. Round 2's flag was set from that bundle's verdict, so
    a model table agreeing with the layer was recorded as independently confirmed: the
    exact licence the ruling removed, written down as provenance. Round 4 removed the
    confirmation path altogether, so this page is withheld like every other -- but the
    fixture is kept because it is the one page class where a "pixel reading" is not
    independent of the layer at all (the pixels show the layer's own ink), which is the
    constraint #1208's comparator inherits.

    The page is reached through the scan-failed half of the union (the positive detectors
    need 0.90 raster cover), so the detector is made to raise.
    """
    pdf = tmp_path / "no-raster.pdf"
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=612, height=792)
        nonce = "w" * (i + 1)
        page.insert_text((72, 90), f"{nonce} Reported in (/997) and again in (/985).", fontsize=10)
        y = 130
        for value, x in zip(("Item", "Level", "Change", "Net"), _COLUMN_X):
            page.insert_text((x, y), value, fontsize=9)
        y += 18
        for row in _COLUMN_ROWS:
            for value, x in zip(row, _COLUMN_X):
                page.insert_text((x, y), value, fontsize=9)
            y += 18
        # NO page-covering raster.
    doc.save(pdf)
    doc.close()

    def _raise(self, page, slots, folios, **kw):
        raise RuntimeError("#1203: scan failed")

    monkeypatch.setattr(BornDigitalDetector, "_has_prior_ocr_layer", _raise)
    # WITH a witness, deliberately: this page has no covering raster, so a witness reading
    # of its pixels reads the LAYER's own ink and would agree with it. A no-witness arm
    # would pass for the wrong reason (nothing to confirm with).
    live = _run_table_arm(
        tmp_path, monkeypatch, "no-raster", route_live=True, agree=True, witness=True, pdf=pdf
    )
    from socr.core.pdf import open_pdf
    from socr.tables.source_evidence import page_has_page_covering_raster

    with open_pdf(pdf) as _doc:
        assert page_has_page_covering_raster(_doc[0]) is False, "fixture premise: no raster"
    assert live["side"].get("prior_ocr_layer_suspect") is True, "fixture premise: scan-failed"
    assert live["ships_table"] is False
    assert "| Reserves |" in (live["side"].get("withheld_table_text") or "")


def test_a_non_suspect_scan_is_unchanged_by_the_final_gate(tmp_path: Path, monkeypatch) -> None:
    """Control: the same page with the detection route neutralised keeps main's ending."""
    main = _run_table_arm(
        tmp_path, monkeypatch, "ctl-main", route_live=False, agree=True, witness=False
    )
    assert "withheld_table_text" not in main["side"]
    assert main["ships_table"] is True, (
        "an unflagged scan lost its table, so the gate is not scoped to suspect pages"
    )


def test_settling_can_never_ship_an_unconfirmed_rung_over_a_confirmed_one(
    tmp_path: Path, monkeypatch
) -> None:
    """Argon §3: ``settle_scanned_table_decision`` can un-accept a CONFIRMED rung.

    Its rule is about how many marked table blocks each candidate accounts for, not about
    who confirmed what: rung 1 splitting the table into two blocks and rung 2 emitting it
    unified makes it promote rung 1. Rounds 1 and 2 put the withhold upstream of the
    settling, so whichever way it went was final.

    The answer this PR takes is the second one the brief allows: the final gate runs AFTER
    the settling, so whatever it hands over is judged on its own confirmation. The two
    invariants are asserted directly — never ship unconfirmed, never withhold without
    retention — because which rung settling picks is its business, not this rule's.
    """
    split = (
        "| Item | Level |\n| --- | --- |\n| Reserves | 512 |\n\n"
        "| Item | Change | Net |\n| --- | --- | --- |\n| Reserves | -380 | -18 |\n"
    )
    unified = _table_markdown(list(_COLUMN_ROWS))
    live = _run_table_arm(
        tmp_path,
        monkeypatch,
        "settle",
        route_live=True,
        agree=True,
        witness=True,
        per_rung_tables=[split, unified],
    )
    assert live["ships_table"] is False, (
        "settling shipped a table; with no release path nothing on these pages ships one"
    )
    assert (live["side"].get("withheld_table_text") or "").strip(), (
        "withheld without retention, which is a deletion"
    )


def test_an_html_table_is_withheld_too(tmp_path: Path, monkeypatch) -> None:
    """Argon P0 (round 4): the gate read only markdown pipe tables.

    `find_table_blocks` does not see `<table>`, so a model that emitted its table as HTML
    walked through the final gate and shipped on a page where no reader had confirmed
    anything. It matters on Rorqual: the engines there do emit HTML tables.

    The gate now detects every table form the scanned-table gate itself recognises --
    markdown via `find_table_blocks`, HTML via `html_table_for_check` -- and the retention
    is written on this floor too.
    """
    html = (
        "<table><tr><th>Item</th><th>Level</th></tr>"
        "<tr><td>Reserves</td><td>521</td></tr>"
        "<tr><td>Borrowings</td><td>204</td></tr></table>"
    )
    live = _run_table_arm(
        tmp_path,
        monkeypatch,
        "html",
        route_live=True,
        agree=False,
        witness=False,
        html_table=html,
    )
    assert "<table" not in live["text"], "an HTML table shipped on an unconfirmed page"
    assert "521" not in live["text"]
    assert "unverifiable table — see image" in live["text"]
    assert "<table" in (live["side"].get("withheld_table_text") or ""), (
        "the HTML table was withheld without retention, which is a deletion"
    )


def test_an_upstream_floor_still_retains_the_rejected_rungs_table(
    tmp_path: Path, monkeypatch
) -> None:
    """Argon P0, round 5: a withhold that keeps nothing is a deletion.

    The ladder's own floor (``orchestrator.py``, at the routing site) runs BEFORE this
    PR's gate and replaces the body with the marker. The gate read only
    ``ps.best_output``, so when the winner it left behind carried no table the gate
    returned early -- and the table the ladder had refused was gone from the run.

    The repro is Argon's: rung 1 emits a table with a RAGGED row, which the
    source-evidence judge rejects and ``promote_unchecked_table_attempt`` cannot promote;
    rung 2 emits prose only. Measured on head 92b2c420: body is the marker,
    ``withheld_table_text`` is absent. The ruling forbids losing a model reading, so the
    retention now covers every attempt, not only the winner.
    """
    ragged = (
        "| Item | Level | Change | Net |\n| --- | --- | --- | --- |\n"
        "| Reserves | 521 | 380 |\n"
        + "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in _COLUMN_ROWS[1:])
        + "\n"
    )
    live = _run_table_arm(
        tmp_path,
        monkeypatch,
        "p0-upstream",
        route_live=True,
        agree=True,
        witness=False,
        per_rung_tables=[ragged, "The Committee reviewed conditions in the economy.\n"],
    )
    assert live["ships_table"] is False, "fixture premise: no table ships"
    assert "unverifiable table — see image" in live["text"], live["text"][:200]
    retained = live["side"].get("withheld_table_text") or ""
    assert "| Reserves | 521 | 380 |" in retained, (
        "the rejected rung's table was withheld without retention, which is a deletion"
    )
    # The retention's first segment is a COMPLETE candidate body, not a bare block: the
    # body here is the marker, so that rung's prose around the table ships nowhere else
    # and #1208 needs it to know what the table was a table OF.
    assert retained.startswith("RESERVE AGGREGATES, MEMBER BANKS"), repr(retained[:80])
    # This page IS withheld -- the body is the marker -- so it must also carry the
    # withhold's record, not just the retention. Without this the gate could satisfy the
    # retention through the retain-only path and leave #1048's claim standing.
    assert "table shipped marked" not in json.dumps(live["side"])
    assert live["side"]["winning_output"].get("judge_accepted") is not True
    assert live["side"].get("table_ladder_disposition") in (None, "")
    # The shipped bytes are the marker, so "the page judge completed" is not a statement
    # about them. This is also what separates a WITHHELD page from a retain-only one,
    # which keeps its judge record: a gate that reached the retention through the
    # retain-only path would leave this standing.
    assert not live["side"]["winning_output"].get("judge_outcome"), live["side"][
        "winning_output"
    ].get("judge_outcome")


def test_every_attempts_distinct_table_is_retained_not_only_the_winners(
    tmp_path: Path, monkeypatch
) -> None:
    """The second half of Argon P0: a rung with TWO tables, a winner with one.

    Retaining ``ps.best_output.text`` alone keeps whichever tables the winner happened to
    carry. Measured on head 92b2c420 with these bodies: the retention held rung 2's single
    table and rung 1's second table was gone.

    The dedup is pinned here too: rung 1's FIRST table is identical to rung 2's, and it
    must appear once, not twice. ``table_grid_identity`` is order-sensitive, so this is a
    claim about identical readings only -- two rungs that differ in one cell are two
    readings and both are kept.
    """
    second = _table_markdown([("Borrowings", "12", "3", "9"), ("Excess", "4", "1", "3")])
    live = _run_table_arm(
        tmp_path,
        monkeypatch,
        "p0-twotable",
        route_live=True,
        agree=True,
        witness=False,
        per_rung_tables=[
            _table_markdown(_COLUMN_ROWS) + "\n\nSome prose.\n\n" + second,
            _table_markdown(_COLUMN_ROWS),
        ],
    )
    assert live["ships_table"] is False, "fixture premise: no table ships"
    retained = live["side"].get("withheld_table_text") or ""
    assert "| Borrowings | 12 | 3 | 9 |" in retained, (
        "the rejected rung's SECOND table was lost; retention followed the winner only"
    )
    assert retained.count("| Monetary base | 721 | -154 | -23 |") == 1, (
        f"the shared table was retained twice: {retained}"
    )


def test_an_unparsable_html_table_is_retained_too(tmp_path: Path, monkeypatch) -> None:
    """r-uben 10: the unparsed-HTML half of the gate's detector is load-bearing.

    ``html_table_for_check`` returns ``(converted, an HTML table remains unparsed)``.
    ``convert_html_tables`` never deletes content, so a ``<table>`` it cannot read as rows
    survives the conversion and ``find_table_blocks`` cannot see it either. Dropping the
    ``_html_unparsed`` conjunct left all 237 tests of round 4 green while losing the
    reading at ``process()``: the scanned gate still floored the page, and nothing was
    retained.

    ``<table>\n</table>`` is the minimal such block -- no rows to parse.
    """
    live = _run_table_arm(
        tmp_path,
        monkeypatch,
        "html-unparsed",
        route_live=True,
        agree=False,
        witness=False,
        html_table="<table>\n</table>\n\nReserves were 521 on the week.\n",
    )
    assert "unverifiable table — see image" in live["text"], live["text"][:200]
    assert "<table" not in live["text"], "an unreadable HTML table shipped"
    assert "<table" in (live["side"].get("withheld_table_text") or ""), (
        "an HTML table the house parser could not read was floored with no retention"
    )


def test_a_corroborated_table_is_not_recorded_as_a_fabrication(tmp_path: Path, monkeypatch) -> None:
    """Argon P1, round 5: the floor had two causes where the page has three.

    ``HALLUCINATION`` means a witness READ the pixels and the model's numbers were not
    there. The floor picked it for everything that was not ``NO_WITNESS_BACKEND``, so a
    page where a witness read the pixels and AGREED -- withheld only because nothing
    compares its CELLS -- was recorded as a fabrication. Measured on head 92b2c420:
    ``failure_mode: hallucination`` on the ``witness=True, agree=True`` arm.

    A DIFFERENCE, not a pinned tuple: the corroborated arm and the contradicted arm run in
    the same process over the same fixture, and only the witness's verdict changes. Both
    withhold; the recorded causes must differ, and only the contradicted one may accuse
    the model.
    """
    corroborated = _run_table_arm(
        tmp_path, monkeypatch, "p1-agree", route_live=True, agree=True, witness=True
    )
    contradicted = _run_table_arm(
        tmp_path, monkeypatch, "p1-contra", route_live=True, agree=False, witness=True
    )
    assert corroborated["ships_table"] is False and contradicted["ships_table"] is False, (
        "fixture premise: both arms withhold"
    )
    assert contradicted["failure_mode"] == FailureMode.HALLUCINATION.value, contradicted[
        "failure_mode"
    ]
    assert corroborated["failure_mode"] != contradicted["failure_mode"], (
        "a witness-corroborated table is recorded with the cause that means a witness "
        "contradicted it"
    )
    assert corroborated["failure_mode"] == FailureMode.TABLE_WITHHELD_UNCONFIRMED.value, (
        corroborated["failure_mode"]
    )
    # The cause must survive into the operator-facing clause, not just the enum: a mode
    # with no ``_MODE_CLAUSES`` row is invisible in the page verdict.
    from socr.core.verdict import _MODE_CLAUSES

    assert FailureMode.TABLE_WITHHELD_UNCONFIRMED in _MODE_CLAUSES

    # The floor stamps the cause in TWO places and both are load-bearing: the sidecar is
    # rebuilt by the manifest from the page flag, and the WINNER OBJECT is written
    # through to the page blob cache, which a replay reads back. A mutant that reverts
    # only the in-memory branch leaves the sidecar correct and the cache saying
    # ``hallucination`` -- measured, and it is why this assertion is on the blobs.
    def _cache_modes(tag: str) -> set[str]:
        root = tmp_path / f"out-{tag}"
        return {
            json.loads(blob.read_text()).get("failure_mode")
            for blob in root.rglob("cache/*/*.json")
        }

    assert _cache_modes("p1-agree") == {FailureMode.TABLE_WITHHELD_UNCONFIRMED.value}, (
        f"the replay cache accuses the model: {_cache_modes('p1-agree')}"
    )
    assert _cache_modes("p1-contra") == {FailureMode.HALLUCINATION.value}, (
        f"the contradicted arm lost its honest cause: {_cache_modes('p1-contra')}"
    )


def test_the_withheld_unconfirmed_cause_survives_a_resume(tmp_path: Path, monkeypatch) -> None:
    """The new cause is DERIVED on resume from the persisted mode, like #658's.

    No new sidecar key -- the P6 key set stays frozen -- so the page flag and the recorded
    mode can never disagree. A page restored from a sidecar written before this ticket
    records ``hallucination`` and restores ``False``, which is what that record meant.
    """
    from socr.core.result import PageOutput, PageStatus
    from socr.core.state import PageState

    for mode, expected in (
        (FailureMode.TABLE_WITHHELD_UNCONFIRMED, True),
        (FailureMode.HALLUCINATION, False),
        (FailureMode.NO_WITNESS_BACKEND, False),
    ):
        ps = PageState(page_num=1)
        ps.scanned_table_evidence_failed = True
        out = PageOutput(page_num=1, text="x", status=PageStatus.ERROR, failure_mode=mode)
        ps.scanned_table_withheld_unconfirmed = (
            ps.scanned_table_evidence_failed
            and out.failure_mode is FailureMode.TABLE_WITHHELD_UNCONFIRMED
        )
        assert ps.scanned_table_withheld_unconfirmed is expected, mode
        assert PageOutput.from_dict(out.to_dict()).failure_mode is mode


def test_the_disputed_arm_is_what_makes_the_value_clears_non_vacuous(
    tmp_path: Path, monkeypatch
) -> None:
    """r-uben's nit on round 4: the ``unchecked_table_values`` assertions were vacuous.

    ``test_a_withheld_page_does_not_also_claim_its_table_shipped`` runs the no-witness
    arm, where #1048's settling fills ``unchecked_table_kind`` but leaves the VALUES
    empty -- so asserting they are empty after the gate proved nothing about the clears.

    Only a DISPUTED arm fills them: a witness that read the pixels and refused cells.
    Measured with the gate disabled, this fixture leaves ``unchecked_table_values:
    "521, 380, 18"`` and the matching list; the no-witness arm leaves ``""``. The two
    arms run here so the difference is in one place, and the gate must clear both.
    """
    from socr.pipeline import orchestrator as orch

    def _arm(tag, **kw):
        return _run_table_arm(tmp_path, monkeypatch, tag, route_live=True, **kw)["side"]

    with monkeypatch.context() as m:
        m.setattr(orch.UnifiedPipeline, "_withhold_unconfirmed_table", lambda *a, **k: False)
        ungated = _arm("values-off", agree=False, witness=True)
    assert ungated.get("unchecked_table_values"), (
        "fixture premise: the disputed arm must FILL the values, or the clear below is "
        f"vacuous -- got {ungated.get('unchecked_table_values')!r}"
    )
    assert ungated.get("unchecked_table_value_list"), ungated.get("unchecked_table_value_list")

    gated = _arm("values-on", agree=False, witness=True)
    assert gated.get("unchecked_table_values") in (None, ""), gated.get("unchecked_table_values")
    assert gated.get("unchecked_table_value_list") in (None, []), gated.get(
        "unchecked_table_value_list"
    )
    assert gated.get("unchecked_table_kind") in (None, ""), gated.get("unchecked_table_kind")


def test_a_losing_rungs_unparsable_html_table_is_retained(tmp_path: Path, monkeypatch) -> None:
    """The unparsed-HTML branch of ``_table_blocks_for_retention``, which only a
    NON-WINNER attempt can reach.

    ``test_an_unparsable_html_table_is_retained_too`` pins the GATE's detector, but there
    the winner carries the HTML and the retention is its whole text -- so the helper's
    own branch is never what saves it, and a mutant that disables that branch survives
    (measured). The branch matters when a LOSING rung emitted the unreadable ``<table>``
    and the winner's table is markdown: then the only way rung 1's reading survives is
    for the helper to isolate it as a block.
    """
    live = _run_table_arm(
        tmp_path,
        monkeypatch,
        "html-losing-rung",
        route_live=True,
        agree=True,
        witness=False,
        per_rung_tables=["<table>\n</table>\n", _table_markdown(_COLUMN_ROWS)],
    )
    assert live["ships_table"] is False, "fixture premise: no table ships"
    retained = live["side"].get("withheld_table_text") or ""
    assert "| Monetary base |" in retained, "the winner's own table was not retained"
    assert "<table" in retained, (
        "the losing rung's unreadable HTML table was dropped from the retention"
    )


def test_a_table_the_page_does_not_ship_is_retained_even_with_no_floor(
    tmp_path: Path, monkeypatch
) -> None:
    """Argon's round-5 "variant B", the one repro of its three that actually runs.

    Rung 1 emits a SHORT table whose numbers the witness corroborates, so the
    source-evidence gate passes it -- and the PAGE judge then refuses it for word count.
    Rung 2 emits enough prose to win. Nothing rejected a table, so no floor fires; the
    winner carries no table; and measured on this fixture the page shipped rung 2's prose
    with ``withheld_table_text`` unset. A table a model read on this page was gone from
    the run and #1208 could never release it.

    The page's ENDING is main's and must not move -- flooring a page whose accepted
    reading is prose would be new content loss. So this pins a difference in the
    retention ONLY: same body, same status, the reading now on disk.
    """
    short_table = "| Item | Level |\n| --- | --- |\n| Reserves | 512 |\n"
    prose = " ".join(f"word{i}" for i in range(35)) + "\n"
    live = _run_table_arm(
        tmp_path,
        monkeypatch,
        "variant-b",
        route_live=True,
        agree=True,
        witness=True,
        per_rung_tables=[short_table, prose],
    )
    assert "unverifiable table — see image" not in live["text"], (
        "the retain-only path floored a page whose accepted reading is prose"
    )
    assert "word34" in live["text"], "fixture premise: the prose rung's reading ships"
    assert "| Reserves | 512 |" not in live["text"], "fixture premise: no table ships"
    assert "| Reserves | 512 |" in (live["side"].get("withheld_table_text") or ""), (
        "a table a model read on this page is neither shipped nor retained"
    )
    # ...and the page keeps its OWN judge record. The clears below the withhold describe
    # a page whose body was replaced by a marker; this page's body is its model reading,
    # and erasing "the judge completed and refused it" would be a second false record in
    # the opposite direction.
    assert live["side"]["winning_output"].get("judge_outcome") == "page_judge_completed", live[
        "side"
    ]["winning_output"].get("judge_outcome")


def test_an_unclosed_html_table_does_not_duplicate_the_block_beside_it() -> None:
    """The no-closing-tag fallback must isolate the tail, not re-dump the whole text.

    Dumping the whole text appends any table already captured above a second time, so
    one reading lands twice in ``withheld_table_text`` -- once isolated and once inside
    the dump -- and #1208 reads a one-table page as a two-table page.
    """
    from socr.pipeline.orchestrator import _table_blocks_for_retention

    blocks = _table_blocks_for_retention("| A | B |\n| --- | --- |\n| 1 | 2 |\n\n<table")

    assert len(blocks) == 2, blocks
    assert blocks[0][1] == "| A | B |\n| --- | --- |\n| 1 | 2 |"
    assert blocks[1][1] == "<table", blocks[1][1]
    assert "| 1 | 2 |" not in blocks[1][1], "the parsed table was re-dumped inside the tail"


def test_two_readings_that_differ_only_by_an_empty_row_are_both_kept() -> None:
    """``table_grid_identity`` alone collapses readings the retention must keep.

    ``_parse_grid`` drops a row whose populated cells are all ``-`` and a row that is
    entirely blank. So a rung that read a missing-data row as ``| - | - |`` and a rung
    that omitted the row produce the SAME grid identity (measured), and a dedup keyed on
    it alone silently drops one of the two readings. The identity therefore carries the
    block's own bytes as well.
    """
    from socr.pipeline.orchestrator import _table_blocks_for_retention

    with_dashes = _table_blocks_for_retention("| A | B |\n| --- | --- |\n| 1 | 2 |\n| - | - |\n")
    without = _table_blocks_for_retention("| A | B |\n| --- | --- |\n| 1 | 2 |\n")
    blank = _table_blocks_for_retention("| A | B |\n| --- | --- |\n| 1 | 2 |\n|  |  |\n")

    assert with_dashes[0][0] != without[0][0], "the dash row collapsed into its absence"
    assert blank[0][0] != without[0][0], "the blank row collapsed into its absence"
    # A genuinely identical reading must still dedupe, or the retention stores every
    # rung of an agreeing ladder.
    assert (
        without[0][0] == _table_blocks_for_retention("| A | B |\n| --- | --- |\n| 1 | 2 |\n")[0][0]
    )
