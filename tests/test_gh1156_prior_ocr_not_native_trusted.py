"""#1156: a page whose text layer is a PRIOR OCR reading is not native-trusted.

A historical scan carrying a baked-in OCR layer stayed ``is_born_digital=True`` after
detection, so ``_build_page_judge``'s trust source reported it trusted, the source-evidence
verifier deferred (``verify_scanned_table``), and the model's table was graded against that
same prior layer -- the layer under suspicion was the authority.

``PageState.prior_ocr_layer_suspect`` now carries the suspicion, and the trust source
answers ``False`` for such a page however born-digital its structure looks.

Kept separate from ``invisible_text_over_raster`` on purpose: that flag alone grants #1027
the right to DELETE the page's native text. Suspicion must not grant deletion rights, or a
detector false positive destroys good born-digital text. The last two tests pin that
separation, so widening the detector (#1156 PR2) cannot widen the destructive floor.

Hermetic: no model, no provider, no Ollama. Every behavioural pin is a DIFFERENCE between
two runs in the same process that change exactly one thing (CLAUDE.md, #257).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    SelectionProvenance,
    _select_page_output_tagged,
)
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState, PageState  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402
from socr.tables.source_evidence import (  # noqa: E402
    page_has_native_words,
    verify_scanned_table,
)

_PROSE = "The estimated effect of the policy change on output is reported below. "
_OCR_MARK = "MODEL-READING-MARK"

# Numbers that appear nowhere on the fixture page: real evidence must reject this.
HALLUCINATED = "\n".join(
    [
        "| Variable | Coefficient | SE |",
        "| --- | --- | --- |",
        "| growth | 0.8172 | 0.0413 |",
        "| inflation | -0.2946 | 0.0517 |",
    ]
)


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# The trust source
# ---------------------------------------------------------------------------


def _pipeline_and_state(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "a text layer long enough to be a real one.")
    doc.save(pdf)
    doc.close()
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            judge_backend="heuristic",
        )
    )
    return pipeline, DocumentState(handle=DocumentHandle.from_path(pdf))


def test_prior_ocr_suspicion_flips_the_trust_source(tmp_path: Path) -> None:
    """One page, one judge, one flag changed: trusted -> untrusted."""
    pipeline, state = _pipeline_and_state(tmp_path)
    judge = pipeline._build_page_judge(state)
    supplier = getattr(judge, "_native_trusted", None)
    assert callable(supplier), "the judge was built with no trust source; the fix is inert"

    state.pages[1].is_born_digital = True
    state.pages[1].prior_ocr_layer_suspect = False
    before = supplier(1)
    state.pages[1].prior_ocr_layer_suspect = True
    after = supplier(1)

    assert before is True
    assert after is False, (
        "a page whose text layer is a prior OCR reading still reported trusted-native, so "
        "the source-evidence verifier defers to that layer and grades model tables on it"
    )


def test_an_unknown_page_still_reports_none(tmp_path: Path) -> None:
    """Control: #1156 must not turn 'cannot tell' into 'untrusted' (GH-163's rule)."""
    pipeline, state = _pipeline_and_state(tmp_path)
    supplier = pipeline._build_page_judge(state)._native_trusted
    assert supplier(99) is None


# ---------------------------------------------------------------------------
# The consequence: the raster witness stops deferring to the prior layer
# ---------------------------------------------------------------------------


def _scanned_page_with_prior_layer(tmp_path: Path):
    """A full-page raster with a words-bearing text layer over it: words are present,
    so the pre-#1156 path deferred on trust, not on their absence."""
    from PIL import Image

    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    buf = io.BytesIO()
    Image.new("L", (int(page.rect.width * 3), int(page.rect.height * 3)), color=245).save(
        buf, format="PNG"
    )
    page.insert_image(page.rect, stream=buf.getvalue())
    for i in range(12):
        page.insert_text((72, 100 + i * 16), f"rn1 garbled ocr line {i} lll 000", fontsize=10)
    pdf = tmp_path / "scan.pdf"
    doc.save(pdf)
    doc.close()
    return fitz.open(pdf)


def test_the_witness_runs_on_a_prior_ocr_page_and_defers_without_the_flag(
    tmp_path: Path,
) -> None:
    """End to end through the real trust source: deferral flips with the flag alone."""
    pipeline, state = _pipeline_and_state(tmp_path / "wire")
    supplier = pipeline._build_page_judge(state)._native_trusted
    state.pages[1].is_born_digital = True

    doc = _scanned_page_with_prior_layer(tmp_path / "scan")
    try:
        assert page_has_native_words(doc[0]), "fixture must carry words, or deferral is vacuous"

        state.pages[1].prior_ocr_layer_suspect = False
        deferred = verify_scanned_table(
            doc[0], HALLUCINATED, ocr_image_fn=lambda _img: "", native_trusted=supplier(1)
        )
        state.pages[1].prior_ocr_layer_suspect = True
        checked = verify_scanned_table(
            doc[0], HALLUCINATED, ocr_image_fn=lambda _img: "", native_trusted=supplier(1)
        )

        assert deferred.deferred is True
        assert checked.deferred is False, (
            "the prior OCR layer was still the authority: the verifier deferred to it"
        )
        assert checked.passed is False, (
            f"a table supported by nothing on the page was accepted: {checked.reason}"
        )
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Suspicion must not grant deletion rights (#1027 stays narrower)
# ---------------------------------------------------------------------------


def _page(*, over_raster: bool, suspect: bool, detected: bool = False) -> DocumentState:
    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = _PROSE * 4
    p.needs_ocr_enhancement = True
    p.invisible_text_over_raster = over_raster
    p.prior_ocr_layer_suspect = suspect
    # ``detected`` is the #1156 route's POSITIVE verdict, which is what the withholding
    # floor keys on. ``suspect`` is the union flag, which must never reach it alone.
    p.prior_ocr_layer = detected
    rejected = PageOutput(
        page_num=1,
        text=_OCR_MARK,
        status=PageStatus.WARNING,
        engine="gemini",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )
    p.attempts = [rejected]
    p.best_output = rejected
    state.pages[1] = p
    return state


def test_suspicion_alone_does_not_delete_the_native_text() -> None:
    """Every destructive floor keys on a POSITIVE detection, never on suspicion.

    Four arms of the same page, differing only in which key is set. Four separate review
    rounds found a flag meaning "trust this less" removing a check instead of adding one,
    so each key's deletion rights are pinned here against the others:

    * suspicion ALONE (the union flag, e.g. from another route's failed scan) keeps the
      page's text -- a detector false positive must not destroy clean born-digital prose;
    * #1027's key (``invisible_text_over_raster``) keeps its own floor, unchanged;
    * #1156's POSITIVE detection withholds the layer, which is the owner's interim ruling;
    * #1156's FAILED scan ("unknown") keeps the text, because unknown is not a finding.
    """
    suspect_only, prov_suspect = _select_page_output_tagged(
        _page(over_raster=False, suspect=True), 1
    )
    floor, prov_floor = _select_page_output_tagged(_page(over_raster=True, suspect=True), 1)
    detected, prov_detected = _select_page_output_tagged(
        _page(over_raster=False, suspect=True, detected=True), 1
    )

    assert prov_suspect is SelectionProvenance.NATIVE_FALLBACK, (
        "suspicion alone reached a destructive floor; a false positive would then "
        "delete clean born-digital text"
    )
    assert "estimated effect" in suspect_only.text

    assert prov_floor is SelectionProvenance.INVISIBLE_SCAN_UNREAD
    assert "estimated effect" not in floor.text

    assert prov_detected is SelectionProvenance.PRIOR_OCR_LAYER_UNREAD
    assert "estimated effect" not in detected.text, (
        "the earlier OCR layer shipped; the interim rule is marker plus page image only"
    )
    assert "earlier OCR, not re-read" in detected.text
    assert detected.failure_mode is FailureMode.PRIOR_OCR_LAYER_UNVERIFIED


def test_the_withheld_page_ships_no_number_of_the_old_layer() -> None:
    """The ruling is "no prior-layer text and NO numbers", so pin the numbers too.

    A prose-only fixture cannot show this: a floor that stripped letters and kept digits
    would pass the test above. The layer here is numeric, and not one of its figures may
    appear in anything the page ships.
    """
    state = _page(over_raster=False, suspect=True, detected=True)
    layer = "Series A 2.2 per cent; Series B 3.1 per cent; total 1957 units 48,392."
    state.pages[1].native_text = layer
    shipped, provenance = _select_page_output_tagged(state, 1)

    assert provenance is SelectionProvenance.PRIOR_OCR_LAYER_UNREAD
    digits = [tok for tok in ("2.2", "3.1", "48,392", "1957") if tok in shipped.text]
    assert digits == [], f"the withheld layer's numbers reached the page: {digits}"
    # The page number itself is the one numeral the marker may carry, and it comes from
    # ``page_num``, not from the layer.
    assert "earlier OCR, not re-read" in shipped.text


def test_a_failed_scan_is_suspect_but_not_deletable() -> None:
    """'Unknown' is not trust, and it is not 'known garbage' either (#961's own scoping).

    Pins the derivation in ``DocumentState``: a scan that FAILED makes the page suspect
    while leaving the #1027 key False.
    """
    from socr.core.born_digital import PageAssessment

    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    state.pages[1] = PageState(page_num=1)

    from socr.core.born_digital import DocumentAssessment

    assessment = DocumentAssessment(
        path=Path("x.pdf"),
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=True,
                native_text=_PROSE,
                confidence=1.0,
                invisible_text_over_raster=False,
                invisible_text_scan_failed=True,
            )
        ],
    )
    state.apply_born_digital(assessment)
    ps = state.pages[1]
    assert ps.prior_ocr_layer_suspect is True
    assert ps.invisible_text_over_raster is False


# ---------------------------------------------------------------------------
# The suspect layer must not VETO the reading that replaced it (Astra P1)
# ---------------------------------------------------------------------------


class _AcceptingInner:
    """Stands in for the model/heuristic leaf: accepts whatever it is given."""

    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=True, reason="inner accepted")


def _table_page_with_prior_layer(tmp_path: Path):
    """A table page whose text layer reads 2.2 where the page itself reads 2.9.

    The page carries a page-covering raster because a prior-OCR scan is one: that
    raster is what gives the outer judge an INDEPENDENT reading, and round 2 made the
    row-level bypass conditional on having one. A fixture without it is not this page.
    """
    from PIL import Image

    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # The text goes down FIRST and the scan is painted over it: that is what a prior-OCR
    # page is, and round 3 made the bypass require exactly that evidence. A fixture with
    # the raster underneath is a born-digital page on a background, where the pixels are
    # no reading of the table at all.
    page.insert_text((72, 100), "Series", fontsize=10)
    page.insert_text((300, 100), "Value", fontsize=10)
    page.insert_text((72, 130), "inflation", fontsize=10)
    page.insert_text((300, 130), "2.2", fontsize=10)
    page.draw_line((72, 110), (400, 110))
    buf = io.BytesIO()
    Image.new("L", (int(page.rect.width * 3), int(page.rect.height * 3)), color=245).save(
        buf, format="PNG"
    )
    page.insert_image(page.rect, stream=buf.getvalue())
    pdf = tmp_path / "table.pdf"
    doc.save(pdf)
    doc.close()
    return fitz.open(pdf)


MODEL_TABLE = "\n".join(
    [
        "| Series | Value |",
        "| --- | --- |",
        "| inflation | 2.9 |",
    ]
)


def test_the_suspect_layer_cannot_veto_the_reading_that_replaced_it(tmp_path: Path) -> None:
    """Difference pin: one page, one judge, only the suspicion flag changes.

    The witness and the model agree on 2.9; the prior layer says 2.2. Before this, the
    native verifier still graded the reading against that layer and rejected it
    (``value_guard_multiset_mismatch``) -- the layer under suspicion vetoing its
    replacement, and every such rejection feeds #1027's delete.
    """
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.agentic import NativeTableVerifierJudge

    doc = _table_page_with_prior_layer(tmp_path / "veto")
    try:
        out = PageOutput(
            page_num=1,
            text=MODEL_TABLE,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )

        def judge(suspect: bool):
            return NativeTableVerifierJudge(
                inner=_AcceptingInner(),
                get_fitz_page=lambda _n: doc[0],
                is_table_page=lambda _n: True,
                native_evidence_suspect=lambda _n: suspect,
            ).assess(out, PROFILE_QWEN_LOCAL)

        graded_on_the_layer = judge(False)
        layer_excluded = judge(True)

        assert graded_on_the_layer.accept is False, (
            "fixture is inert: the prior layer did not veto the reading even without the "
            "flag, so the flag cannot be shown to stop it"
        )
        assert layer_excluded.accept is True, (
            "the prior OCR layer still graded the reading that replaced it"
        )
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# A DETECTED hit reaches the terminal artifacts
# ---------------------------------------------------------------------------


def _state_from_detection(*, detected: bool, route: str = "prior_ocr_layer") -> DocumentState:
    """``route``: which detection key the assessment carries, #1156's or #961's.

    Both exist and each is a separate propagation path through ``apply_born_digital``;
    Astra's mutant removed one while the suite only ever exercised the other.
    """
    from socr.core.born_digital import DocumentAssessment, PageAssessment

    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    state.pages[1] = PageState(page_num=1)
    state.apply_born_digital(
        DocumentAssessment(
            path=Path("x.pdf"),
            pages=[
                PageAssessment(
                    page_num=1,
                    is_born_digital=True,
                    native_text=_PROSE * 4,
                    confidence=1.0,
                    invisible_text_over_raster=detected and route == "invisible_text_over_raster",
                    invisible_text_scan_failed=False,
                    prior_ocr_layer=detected and route == "prior_ocr_layer",
                )
            ],
        )
    )
    return state


@pytest.mark.parametrize("route", ["prior_ocr_layer", "invisible_text_over_raster"])
def test_a_detected_hit_reaches_the_sidecar_and_the_shipped_text(
    tmp_path: Path, route: str
) -> None:
    """Difference pin from DETECTION, not from a hand-set flag.

    Astra's mutant removed detected-hit propagation while keeping the scan-failure path,
    and every test still passed: the suite only ever set the flag by hand or through a
    failed scan. This drives a detected hit to both terminal artifacts.
    """
    import json

    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            judge_backend="heuristic",
            save_figures=False,
            write_manifest=False,
        )
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = {}
    for detected in (False, True):
        state = _state_from_detection(detected=detected, route=route)
        rejected = PageOutput(
            page_num=1,
            text=_OCR_MARK,
            status=PageStatus.WARNING,
            engine="gemini",
            audit_passed=False,
            failure_mode=FailureMode.AUDIT_FAILED,
        )
        state.pages[1].attempts = [rejected]
        state.pages[1].best_output = rejected
        state.pages[1].needs_ocr_enhancement = True

        supplier = pipeline._build_page_judge(state)._native_trusted
        suspect_to_judge = supplier(1)

        sidecar = pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
        payload = json.loads(Path(sidecar).read_text())
        shipped, _prov = _select_page_output_tagged(state, 1)
        payloads[detected] = (payload.get("prior_ocr_layer_suspect"), suspect_to_judge, shipped)

    quiet_flag, quiet_trust, quiet_text = payloads[False]
    hit_flag, hit_trust, hit_text = payloads[True]

    assert quiet_flag is None and hit_flag is True, (
        "a DETECTED prior-OCR hit did not reach the sidecar, so a resume restores a page "
        "the meter and the judge believe is trustworthy"
    )
    assert quiet_trust is True and hit_trust is False
    assert "estimated effect" in quiet_text.text
    # The hit reaches the shipped TEXT too: whichever route found it, the earlier OCR
    # layer does not ship. The ROUTE decides which floor owns the page -- #961/#1027 keep
    # their own marker on their own key -- so the mode is pinned per route rather than
    # collapsed to one value both would satisfy.
    assert "estimated effect" not in hit_text.text, (
        "the detected hit did not reach the shipped text either"
    )
    expected = (
        FailureMode.PRIOR_OCR_LAYER_UNVERIFIED
        if route == "prior_ocr_layer"
        else FailureMode.INVISIBLE_SCAN_UNREAD
    )
    assert hit_text.failure_mode is expected
    assert quiet_text.failure_mode is not expected


# ---------------------------------------------------------------------------
# Suspicion must never make the check weaker (Astra round 2)
# ---------------------------------------------------------------------------


def _born_digital_table_page(tmp_path: Path):
    """A genuine born-digital table page: no raster, so no independent witness exists."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for x, label in ((72, "Series"), (260, "Value"), (400, "SE")):
        page.insert_text((x, 100), label, fontsize=10)
    page.draw_line((60, 108), (470, 108))
    y = 130
    for row in _ROWS:
        for x, cell in zip((72, 260, 400), row):
            page.insert_text((x, y), cell, fontsize=10)
        y += 22
    page.draw_line((60, y), (470, y))
    pdf = tmp_path / "born.pdf"
    doc.save(pdf)
    doc.close()
    return fitz.open(pdf)


_ROWS = (("Alpha", "2.1", "0.4"), ("Beta", "3.1", "0.5"), ("Gamma", "4.2", "0.6"))


def _markdown(rows) -> str:
    head = "| Series | Value | SE |\n| --- | --- | --- |"
    return "\n".join([head] + [f"| {n} | {v} | {s} |" for n, v, s in rows])


#: Alpha's and Beta's values swapped: the page's numbers, in the wrong rows. The
#: aggregate multiset over the whole table is unchanged, which is why an aggregate
#: check accepts it and row-level verification does not.
SWAPPED = _markdown((("Alpha", "3.1", "0.4"), ("Beta", "2.1", "0.5"), ("Gamma", "4.2", "0.6")))


def test_a_detection_error_does_not_turn_a_swap_into_an_acceptance(tmp_path: Path) -> None:
    """A DETECTION ERROR makes a born-digital page suspect (fail towards suspicion).

    That must not weaken anything. The page's own values are Alpha=2.1, Beta=3.1 and the
    candidate swaps them: row-level verification rejects it, while the outer judge's
    aggregate multiset — rebuilt from that same native text, because a page with no
    raster has no independent witness — accepts it. Dropping row-level verification here
    would turn a rejection into an acceptance, which is what this pins.
    """
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.agentic import NativeTableVerifierJudge, SourceEvidenceTableJudge

    doc = _born_digital_table_page(tmp_path / "swap")
    try:
        out = PageOutput(
            page_num=1,
            text=SWAPPED,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )

        def chain(suspect: bool):
            native = NativeTableVerifierJudge(
                inner=_AcceptingInner(),
                get_fitz_page=lambda _n: doc[0],
                is_table_page=lambda _n: True,
                native_evidence_suspect=lambda _n: suspect,
            )
            judge = SourceEvidenceTableJudge(
                inner=native,
                get_fitz_page=lambda _n: doc[0],
                native_trusted=lambda _n: not suspect,
            )
            return judge.assess(out, PROFILE_QWEN_LOCAL)

        clean = chain(False)
        after_detection_error = chain(True)

        assert clean.accept is False, "fixture is inert: the swap was not caught even normally"
        assert after_detection_error.accept is False, (
            "a detection error on a born-digital page turned a rejected swap into an "
            "acceptance: suspicion made the check weaker"
        )
    finally:
        doc.close()


def test_a_transparent_cover_is_not_an_independent_witness(tmp_path: Path) -> None:
    """Round 2 (Astra P1), the same principle a second time.

    Dropping row verification is safe only when something independent read the page.
    An image drawn with alpha 0 satisfies "the page is covered by images" while leaving
    the pixels unchanged, so a suspect page could skip the row check with nothing having
    read it -- and the swapped values came back accepted.
    """
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.agentic import NativeTableVerifierJudge

    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for x, label in ((72, "Series"), (260, "Value"), (400, "SE")):
        page.insert_text((x, 100), label, fontsize=10)
    page.draw_line((60, 108), (470, 108))
    y = 130
    for row in _ROWS:
        for x, cell in zip((72, 260, 400), row):
            page.insert_text((x, y), cell, fontsize=10)
        y += 22
    page.draw_line((60, y), (470, y))
    # a full-page image that paints nothing
    page.draw_rect(page.rect, color=None, fill=(1, 1, 1), fill_opacity=0.0)
    buf = io.BytesIO()
    from PIL import Image

    Image.new("L", (300, 400), color=245).save(buf, format="PNG")
    page.insert_image(page.rect, stream=buf.getvalue())
    pdf = tmp_path / "transparent.pdf"
    doc.save(pdf)
    doc.close()
    doc = fitz.open(pdf)

    try:
        out = PageOutput(
            page_num=1,
            text=SWAPPED,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        decision = NativeTableVerifierJudge(
            inner=_AcceptingInner(),
            get_fitz_page=lambda _n: doc[0],
            is_table_page=lambda _n: True,
            native_evidence_suspect=lambda _n: True,
        ).assess(out, PROFILE_QWEN_LOCAL)
        assert decision.accept is False, (
            "a page covered only by a transparent image counted as independently read, "
            "so the row check was dropped and a swap was accepted"
        )
    finally:
        doc.close()
