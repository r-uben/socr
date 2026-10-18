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
from socr.core.result import PageOutput, PageStatus  # noqa: E402

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


def _column_table_pdf(path: Path) -> Path:
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
        page.insert_text((72, 90), f"{nonce} Reported in (/997) and again in (/985).", fontsize=10)
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
    name = "qwen"
    table = ""

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        return [
            PageOutput(
                page_num=n,
                text=f"RESERVE AGGREGATES, MEMBER BANKS\n\n{self.table}",
                status=PageStatus.SUCCESS,
                engine="qwen",
            )
            for n in page_nums
        ]


def _run_table_arm(
    tmp_path: Path,
    monkeypatch,
    tag: str,
    *,
    route_live: bool,
    agree: bool,
    witness: bool,
    pdf: Path | None = None,
) -> dict:
    """``process()`` with the real judge chain, over the four axes the ruling needs."""
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch
    from socr.tables import source_evidence as se

    model_rows = (
        list(_COLUMN_ROWS) if agree else [("Reserves", "521", "380", "18"), *_COLUMN_ROWS[1:]]
    )
    with monkeypatch.context() as m:
        engine = _ColumnTableEngine()
        engine.table = _table_markdown(model_rows)
        m.setattr(orch, "get_engine", lambda engine_type: engine)
        if witness:
            # A witness that READS something: its reading is of the page pixels, which on
            # this fixture carry the layer's own numbers -- what a real tesseract on a real
            # scan does. The arm only has to show the no-witness rule does not fire.
            m.setattr(
                se,
                "classical_ocr_with_state",
                lambda pix: (" ".join(c for r in _COLUMN_ROWS for c in r), se.WITNESS_READING, ""),
            )
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


def test_a_witness_that_ran_leaves_the_table_alone(tmp_path: Path, monkeypatch) -> None:
    """Arm (iii): with a witness backend present the rule must not fire.

    The ruling is about tables nothing could check. Where the classical-OCR witness runs,
    the page keeps today's ending -- otherwise this would withhold every suspect table on
    a box that CAN check them, which is a different and much larger decision.
    """
    live = _run_table_arm(
        tmp_path, monkeypatch, "w-live", route_live=True, agree=False, witness=True
    )
    assert live["ships_table"] is True, "a witnessed table was withheld by the no-witness rule"
    assert live["page_status"] == "warning"
    assert live["failure_mode"] == "table_unverified"
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
