"""#1203: a page rejected for ANOTHER reason still records what its text layer is.

``_assess_page_signals`` returns ten times before #1156's drawing-order check runs. Each
of those returns is a born-digital rejection, so the page is routed to OCR either way --
but it recorded no ``prior_ocr_layer``, and therefore got no page mark, no sidecar
retention, no document bucket and no acceptance-meter lane. NOT the withholding floor:
that, and every other native floor, sits inside ``manifest.py``'s ``if p.is_born_digital
and p.native_text:`` branch, which such a page satisfies neither half of -- so this fix
adds marks and retention and changes nothing about what ships.
Measured: 325 of 7,941 prefiltered pages on the 455-PDF Fed census (~4,200 of the corpus),
80 of 661 pages on the corpus probe set, zero in the other direction.

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
