"""#165 / #140: unresolved-math accounting follows the OUTCOME, not the flags.

A born-digital page can extract clean prose and unusable mathematics: the math
glyphs come back as private-use codepoints because the embedded font carries no
usable ToUnicode map. `has_unmapped_math_glyphs` detects that damage.

The accounting on top of it did not. `native_math_unrecovered` was emitted from
the native branch and SUPPRESSED whenever `--detect-equations` and
`--recover-clean-equations` were both set -- a statement about the run's
configuration, with no check that a region was ever detected, let alone
recovered. Two flags therefore silenced the only durable record of lost
mathematics without recovering a glyph. The same emitter also ran BEFORE the
region lanes, so a page later recovered in full still carried the warning.

The pair (a)/(b) below is the falsifier for the first half: identical damage and
identical (zero) recovery, run with the flags on and with them off, must produce
the identical outcome. On the old code they produce opposite ones.

What counts as recovery here is deliberately narrow, and section (c) pins the
narrowness rather than hiding it: a resolved region, an accepted reading, or
even an absence of PUA in the shipped text is not on its own a coverage proof --
omission removes PUA exactly as well as recovery does. Only the corrupt-region
lane's demonstrably complete, aligned, still-present replacements clear the
signal.

Hermetic: `_available_engines_for_agentic` is patched explicitly,
`_resolve_judge_model` returns "", `route_page` is asserted un-called, and the
only equation "model" is a deterministic double supplied at the recovery
boundary. Nothing here reaches a provider.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.born_digital import (  # noqa: E402
    BornDigitalDetector,
    count_pua_chars,
)
from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.result import (  # noqa: E402
    DocumentStatus,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState  # noqa: E402
from socr.math.accounting import (  # noqa: E402
    UNRESOLVED_MATH_KIND,
    corrupt_region_evidence,
    unresolved_math_detail,
)
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

PUA = chr(0xF766)  # UniMath script-T, the exact shape #92's detector fingerprints

#: The damaged native layer. The PUA run IS the mathematics: an extractor that
#: cannot map the font hands back private-use codepoints, and the prose around
#: them reads perfectly.
DAMAGED_EQUATION = f"{PUA}{PUA} = {PUA}o(T)"
PUA_NATIVE_TEXT = (
    "Optimists set a lift-off date and the heterogeneous agents consume more,\n"
    f"{DAMAGED_EQUATION}\n"
    "presenting a comprehensive analysis with several full sentences of prose."
)
CLEAN_NATIVE_TEXT = (
    "Optimists set a lift-off date and the heterogeneous agents consume more,\n"
    "T T = f o ( T ) written entirely in mapped characters,\n"
    "presenting a comprehensive analysis with several full sentences of prose."
)


# ---------------------------------------------------------------------------
# Grounding: the fixture text is damaged by the REAL detector's own measure.
# Without this the rest of the file could be asserting on a string nothing in
# production would ever flag.
# ---------------------------------------------------------------------------


def _prose_pdf(path: Path) -> Path:
    """A born-digital page. Its own text layer is clean ASCII on purpose.

    A standard PDF base font cannot encode a private-use codepoint at all --
    ``insert_text`` drops it -- which is why the damaged layer is supplied
    through the extractor seam below, exactly as #92's own detector test does.
    """
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in [
        "This is a born-digital academic paper with more than enough words to clear",
        "the born-digital floor and be classified as a clean native text layer here,",
        "presenting a comprehensive analysis with several full sentences of prose.",
    ]:
        page.insert_text((72, y), line, fontsize=11, fontname="helv")
        y += 16
    doc.save(str(path))
    doc.close()
    return path


def test_the_real_detector_calls_this_fixture_damaged(tmp_path: Path, monkeypatch) -> None:
    """The canary. `PUA_NATIVE_TEXT` must be damaged by production's own
    detector, and `CLEAN_NATIVE_TEXT` must not be -- otherwise every assertion
    below is about a string the pipeline would never flag."""
    pdf = _prose_pdf(tmp_path / "doc.pdf")
    detector = BornDigitalDetector()

    for text, expected in ((PUA_NATIVE_TEXT, True), (CLEAN_NATIVE_TEXT, False)):
        doc = fitz.open(str(pdf))
        page = doc[0]
        original = page.get_text

        def fake_get_text(*args, _text=text, **kwargs):
            mode = args[0] if args else kwargs.get("option", "text")
            return _text if mode == "text" else original(*args, **kwargs)

        monkeypatch.setattr(page, "get_text", fake_get_text)
        assessment = detector._assess_page(page, 1)
        assert assessment.is_born_digital
        assert assessment.has_unmapped_math_glyphs is expected, text
        doc.close()

    assert count_pua_chars(PUA_NATIVE_TEXT) == 3
    assert count_pua_chars(CLEAN_NATIVE_TEXT) == 0


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _write_crops(crop_dir: Path, regions: list) -> None:
    """Materialise each region's crop as a real 1x1 PNG on disk."""
    crop_dir.mkdir(parents=True, exist_ok=True)
    for region in regions:
        if not region.crop_path:
            continue
        target = crop_dir / Path(region.crop_path).name
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8))
        pix.set_rect(pix.irect, (255, 255, 255))
        pix.save(str(target))


def _pipeline(*, detect_equations: bool, recover_clean_equations: bool) -> UnifiedPipeline:
    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.DEEPSEEK,
            enabled_engines=list(EngineType),
            agentic=True,
            quiet=True,
            native_first=True,
            detect_equations=detect_equations,
            recover_clean_equations=recover_clean_equations,
        )
    )


def _state(pdf: Path, *, native_text: str, damaged: bool, corrupt: bool = False) -> DocumentState:
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    ps.is_born_digital = True
    ps.native_text = native_text
    ps.has_unmapped_math_glyphs = damaged
    ps.has_corrupt_math = corrupt
    return state


def _run(
    tmp_path: Path,
    *,
    native_text: str = PUA_NATIVE_TEXT,
    damaged: bool = True,
    corrupt: bool = False,
    detect_equations: bool = False,
    recover_clean_equations: bool = False,
    regions: list | None = None,
    providers: list | None = None,
    tag: str = "run",
):
    """Route one page through the real agentic loop, then the real assemble.

    Returns ``(pipeline, state, result, out_dir)``. No provider is reachable:
    the ladder is supplied explicitly, the judge model resolves to "", and
    ``route_page`` is asserted never to have been called.
    """
    from socr.core.providers import PROFILE_QWEN_LOCAL

    work = tmp_path / tag
    work.mkdir(parents=True, exist_ok=True)
    pdf = _prose_pdf(work / "doc.pdf")
    out_dir = work / "out"

    pipeline = _pipeline(
        detect_equations=detect_equations, recover_clean_equations=recover_clean_equations
    )
    pipeline._scan_root = pdf.parent
    state = _state(pdf, native_text=native_text, damaged=damaged, corrupt=corrupt)

    ladder = [PROFILE_QWEN_LOCAL] if providers is None else providers
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(pipeline, "_available_engines_for_agentic", return_value=ladder)
        )
        stack.enter_context(patch.object(pipeline, "_resolve_judge_model", return_value=""))
        route = stack.enter_context(patch("socr.pipeline.orchestrator.route_page"))
        if regions is not None:

            def _recover(*_args, crop_dir=None, **_kwargs):
                # The crops the real lane retains, actually written. Without
                # them the assembled body's image refs point at nothing and the
                # phantom-image sweep removes them -- which is a genuine loss of
                # the evidence pointer, so a fixture that skipped this would
                # measure the sweep rather than the accounting.
                if crop_dir is not None:
                    _write_crops(Path(crop_dir), regions)
                return regions

            stack.enter_context(
                patch("socr.math.recover.recover_math_regions", side_effect=_recover)
            )
        pipeline._phase_agentic(state, out_dir)
        result = pipeline._phase_assemble(state, out_dir)
        route.assert_not_called()

    return pipeline, state, result, out_dir


def _unresolved_events(state: DocumentState) -> list:
    return [e for e in state.events if getattr(e, "kind", "") == UNRESOLVED_MATH_KIND]


def _final_status(state: DocumentState, page_num: int = 1) -> PageStatus:
    from socr.core.manifest import finalized_page_record

    return finalized_page_record(state, page_num).output.status


def _region(
    source: str,
    *,
    crop: str | None = "equations/p1_r1.png",
    latex: str = r"\hat{T} = f_o(T)",
    valid: bool = True,
):
    """A real ``CorruptMathRegion``, not a mock.

    ``splice_math`` reads these fields and sets ``source_aligned`` itself, so
    the alignment in every test below is computed by production code against
    real text rather than asserted into existence.
    """
    from socr.math.recover import CorruptMathRegion

    return CorruptMathRegion(
        rect=None,
        source_text=source,
        crop_path=crop,
        raw_latex=latex if valid else "",
        validation_ok=valid,
        validation_reason="" if valid else "unbalanced delimiters",
        model_id="test-double",
        attempts=1,
    )


# ---------------------------------------------------------------------------
# (a) / (b): the flags do not decide. The outcome does.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flags_on", [True, False])
def test_zero_recovery_is_reported_whatever_the_flags_say(tmp_path: Path, flags_on: bool) -> None:
    """(a) and (b), each on its own terms.

    Damage detected, nothing recovered. The page warns, the document is
    AUDIT_FAILED with its output retained, the note names the page and the real
    reason, and none of that depends on two flags that recovered nothing.
    """
    _, state, result, _ = _run(
        tmp_path,
        detect_equations=flags_on,
        recover_clean_equations=flags_on,
        tag=f"flags_{flags_on}",
    )

    events = _unresolved_events(state)
    assert len(events) == 1, f"expected exactly one unresolved-math record, got {events}"
    assert events[0].page_num == 1
    assert events[0].data["regions_covered"] == 0
    assert events[0].data["residual_pua_chars"] == 3

    assert _final_status(state) is PageStatus.WARNING
    assert state.status is DocumentStatus.AUDIT_FAILED
    assert result.pages[0].text, "the document lost its prose; this is a warning, not a floor"
    assert DAMAGED_EQUATION in result.pages[0].text, (
        "the damaged source bytes were dropped rather than shipped and flagged"
    )
    assert "unrecovered math glyphs on page(s) 1" in (result.error or "")
    assert "enable --detect-equations" not in (result.error or ""), (
        "the note still advises enabling flags that may already have run"
    )


def test_the_two_flag_settings_produce_the_same_outcome(tmp_path: Path) -> None:
    """The #165 falsifier proper: a DIFFERENCE pinned, not a value.

    Same damage, same (zero) recovery, only the two recovery flags change. On
    the old accounting the flags-on run emitted nothing and the flags-off run
    emitted the event, so this comparison fails there. Config fingerprints and
    timings are free to differ; the reported outcome is not.
    """
    _, on_state, on_result, _ = _run(
        tmp_path, detect_equations=True, recover_clean_equations=True, tag="cmp_on"
    )
    _, off_state, off_result, _ = _run(
        tmp_path, detect_equations=False, recover_clean_equations=False, tag="cmp_off"
    )

    def summary(state, result):
        return (
            sorted(e.page_num for e in _unresolved_events(state)),
            [e.data["reason"] for e in _unresolved_events(state)],
            _final_status(state),
            state.status,
            result.error,
        )

    assert summary(on_state, on_result) == summary(off_state, off_result)


# ---------------------------------------------------------------------------
# (c): what clears the signal, and the negatives that must not
# ---------------------------------------------------------------------------


def test_complete_aligned_recovery_clears_this_warning_only(tmp_path: Path) -> None:
    """Every damaged span replaced by a retained, aligned, validated reading.

    The unresolved-math record goes. The corrupt-hybrid record does NOT: a LaTeX
    candidate passed a syntax gate, which is not a proof that it says what the
    crop says. Pinning `status == SUCCESS` here would delete that separate
    guarantee, so the assertion is the absence of THIS warning.
    """
    _, state, result, _ = _run(
        tmp_path, corrupt=True, regions=[_region(DAMAGED_EQUATION)], tag="covered"
    )

    assert _unresolved_events(state) == [], (
        "a fully covered page still reports unrecovered math glyphs"
    )
    assert "unrecovered math glyphs" not in (result.error or "")

    body = result.pages[0].text
    assert count_pua_chars(body) == 0
    assert r"\hat{T} = f_o(T)" in body

    # The existing, separate uncertainty survives untouched.
    assert state.status is DocumentStatus.AUDIT_FAILED
    assert "corrupt equation candidate unverified" in (result.error or "")


@pytest.mark.parametrize(
    ("label", "regions"),
    [
        ("rejected reading", [_region(DAMAGED_EQUATION, valid=False)]),
        ("no crop retained", [_region(DAMAGED_EQUATION, crop=None)]),
        ("source never aligned", [_region("a span that is not on this page")]),
        ("no region enumerated", []),
    ],
)
def test_incomplete_recovery_keeps_the_warning(tmp_path: Path, label: str, regions: list) -> None:
    """The negative pair for (c), one row per way recovery can fall short.

    Each of these produces a recovery attempt, an event, and in three of the
    four cases a nonzero region count -- none of which is coverage.
    """
    _, state, result, _ = _run(
        tmp_path, corrupt=True, regions=regions, tag=f"neg_{label.replace(' ', '_')}"
    )

    assert len(_unresolved_events(state)) == 1, (
        f"{label}: the damage was reported as resolved by a recovery that did not cover it"
    )
    assert state.status is DocumentStatus.AUDIT_FAILED
    assert "unrecovered math glyphs on page(s) 1" in (result.error or "")


def test_two_regions_one_recovered_still_warns(tmp_path: Path) -> None:
    """A recovered region elsewhere on the page does not vouch for a failed one.

    This is the shape a `recovered_regions > 0` test would pass and a coverage
    test must fail: one span is genuinely fixed, and the page is still damaged.
    """
    second = f"{PUA}z + {PUA}w"
    native = f"{PUA_NATIVE_TEXT}\nand also the second display {second} appears here."
    _, state, result, _ = _run(
        tmp_path,
        native_text=native,
        corrupt=True,
        regions=[
            _region(DAMAGED_EQUATION),
            _region(second, crop="equations/p1_r2.png", valid=False),
        ],
        tag="two_regions",
    )

    events = _unresolved_events(state)
    assert len(events) == 1
    assert events[0].data["regions_total"] == 2
    assert events[0].data["regions_covered"] == 1
    assert "1 of 2 damaged region(s)" in events[0].data["reason"]
    assert state.status is DocumentStatus.AUDIT_FAILED
    assert "unrecovered math glyphs on page(s) 1" in (result.error or "")


def test_omission_is_not_recovery(tmp_path: Path) -> None:
    """The trap the helper exists to avoid, stated directly at the seam.

    A body with no private-use codepoints left in it looks identical whether the
    mathematics was transcribed or simply deleted. Absence of PUA therefore
    clears nothing on its own.
    """
    assert (
        unresolved_math_detail(
            has_unmapped_math_glyphs=True,
            evidence=None,
            text="the equation was dropped entirely and this prose remains",
        )
        is not None
    )


# ---------------------------------------------------------------------------
# (d): a table page. Reporting must not touch selection.
# ---------------------------------------------------------------------------


def test_math_reporting_never_changes_which_table_ships(tmp_path: Path) -> None:
    """(d): the same accepted table wins, byte for byte, damaged or not.

    A reporting guard that reached into selection would replace a good model
    table with native text the moment math went unresolved. Pinned as a
    DIFFERENCE: two finalizations of the same page, changing only whether the
    math-damage signal is set.
    """
    from socr.core.manifest import finalized_page_record

    grid = "| Year | Value |\n|---|---|\n| 2018 | 1.0 |"
    outputs = {}
    for damaged in (True, False):
        work = tmp_path / f"tbl_{damaged}"
        work.mkdir(parents=True, exist_ok=True)
        pdf = _prose_pdf(work / "doc.pdf")
        state = _state(pdf, native_text=PUA_NATIVE_TEXT, damaged=damaged)
        state.pages[1].has_tables = True
        accepted = PageOutput(
            page_num=1,
            text=grid,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
            confidence=0.9,
        )
        state.pages[1].attempts.append(accepted)
        state.pages[1].best_output = accepted
        outputs[damaged] = finalized_page_record(state, 1).output

    damaged_out, clean_out = outputs[True], outputs[False]
    assert damaged_out.text == clean_out.text == grid, (
        "the accepted table was replaced because math was unresolved"
    )
    assert damaged_out.engine == clean_out.engine == "qwen"
    assert damaged_out.failure_mode is clean_out.failure_mode
    assert damaged_out.audit_passed is clean_out.audit_passed is True, (
        "audit_passed selects the winner; a reporting guard must not touch it"
    )
    # The only difference is the report itself.
    assert clean_out.status is PageStatus.SUCCESS
    assert damaged_out.status is PageStatus.WARNING
    assert any("unmapped math glyphs" in n for n in damaged_out.audit_notes)


# ---------------------------------------------------------------------------
# (e): no provider at all
# ---------------------------------------------------------------------------


def test_unresolved_math_is_reported_with_no_provider(tmp_path: Path) -> None:
    """(e): an empty ladder exits routing early. The damage is still reported.

    The accounting must not be a passenger on a code path that only runs when a
    model is reachable -- CI has no provider, and neither does an offline run.
    """
    _, state, result, _ = _run(
        tmp_path,
        detect_equations=True,
        recover_clean_equations=True,
        providers=[],
        tag="noprovider",
    )

    assert len(_unresolved_events(state)) == 1
    assert state.status is not DocumentStatus.SUCCESS
    assert "unrecovered math glyphs on page(s) 1" in (result.error or "")


# ---------------------------------------------------------------------------
# Idempotency, the math-free control, and the resume round trip
# ---------------------------------------------------------------------------


def test_a_math_free_page_is_untouched(tmp_path: Path) -> None:
    """The control. No damage signal, no note, no event, no sparse field, and a
    clean SUCCESS -- so the guard above cannot be a stamp on every page."""
    _, state, result, out_dir = _run(
        tmp_path, native_text=CLEAN_NATIVE_TEXT, damaged=False, tag="clean"
    )

    assert _unresolved_events(state) == []
    assert "unrecovered math glyphs" not in (result.error or "")
    assert _final_status(state) is PageStatus.SUCCESS
    assert state.status is DocumentStatus.SUCCESS

    sidecar = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
    assert "has_unmapped_math_glyphs" not in sidecar, (
        "the sparse field was written on an unaffected page, changing every "
        "sidecar in the corpus for no information"
    )
    assert "math_recovery_evidence" not in sidecar


def test_the_body_is_byte_identical_with_and_without_the_damage_signal(tmp_path: Path) -> None:
    """The guard reports; it never rewrites. Same input, same shipped markdown.

    Only the accounting differs, so the final `.md` of a damaged run must equal
    that of the identical undamaged run byte for byte.
    """
    _, _, damaged_result, damaged_dir = _run(tmp_path, damaged=True, tag="bytes_damaged")
    _, _, clean_result, clean_dir = _run(tmp_path, damaged=False, tag="bytes_clean")

    assert damaged_result.pages[0].text == clean_result.pages[0].text

    def final_md(d: Path) -> str:
        return next(p for p in d.rglob("*.md") if p.parent.name != "pages").read_text()

    assert final_md(damaged_dir) == final_md(clean_dir)


def test_finalizing_twice_adds_one_note_and_one_event(tmp_path: Path) -> None:
    """Every finalization seam re-runs the guard on its own output, and assemble
    can be re-entered. A guard that appended would grow a note per pass."""
    from socr.core.manifest import finalized_page_record

    pipeline, state, _, out_dir = _run(tmp_path, tag="idem")

    once = finalized_page_record(state, 1).output
    twice = finalized_page_record(state, 1).output
    assert once.audit_notes == twice.audit_notes
    assert len([n for n in twice.audit_notes if "unmapped math glyphs" in n]) == 1

    pipeline._phase_assemble(state, out_dir)
    assert len(_unresolved_events(state)) == 1, "assemble re-entry duplicated the record"


def test_the_report_survives_a_resume(tmp_path: Path) -> None:
    """A resumed page is not re-assessed and not re-recovered.

    Without persistence and replay the damage is reported on the run that found
    it and never again -- and the resumed run reports a clean SUCCESS on a
    document whose mathematics is known to be missing.
    """
    pipeline, state, _, out_dir = _run(tmp_path, tag="resume")
    assert _unresolved_events(state)

    sidecar = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
    assert sidecar["has_unmapped_math_glyphs"] is True
    assert UNRESOLVED_MATH_KIND in [e.get("kind") for e in sidecar["audit_events"]]

    resumed = DocumentState(handle=DocumentHandle.from_path(state.handle.path))
    assert not resumed.events
    page_out = PageOutput(
        page_num=1,
        text=PUA_NATIVE_TEXT,
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=True,
    )
    pipeline._restore_terminal_page_state(resumed, 1, page_out, out_dir)

    assert resumed.pages[1].has_unmapped_math_glyphs is True
    assert _unresolved_events(resumed), "the record vanished on resume"
    assert _final_status(resumed) is PageStatus.WARNING


def test_a_resume_does_not_invent_coverage(tmp_path: Path) -> None:
    """Missing evidence restores as UNKNOWN, which is unresolved.

    A sidecar written before this field existed has no `math_recovery_evidence`.
    Reading that absence as "recovered" would silently retire the debt for the
    entire existing corpus.
    """
    assert (
        unresolved_math_detail(has_unmapped_math_glyphs=True, evidence=None, text=CLEAN_NATIVE_TEXT)
        is not None
    )


def test_complete_evidence_survives_the_round_trip(tmp_path: Path) -> None:
    """The other direction: a covered page restored from its sidecar must not
    acquire a stale warning it did not have."""
    pipeline, state, _, out_dir = _run(
        tmp_path, corrupt=True, regions=[_region(DAMAGED_EQUATION)], tag="resume_covered"
    )
    assert _unresolved_events(state) == []

    sidecar = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
    assert sidecar["math_recovery_evidence"]["regions_total"] == 1

    resumed = DocumentState(handle=DocumentHandle.from_path(state.handle.path))
    page_out = PageOutput(
        page_num=1,
        text=state.pages[1].best_output.text,
        status=PageStatus.WARNING,
        engine="native+math",
        audit_passed=False,
    )
    pipeline._restore_terminal_page_state(resumed, 1, page_out, out_dir)

    assert resumed.pages[1].math_recovery_evidence is not None
    assert _unresolved_events(resumed) == []
    assert (
        unresolved_math_detail(
            has_unmapped_math_glyphs=resumed.pages[1].has_unmapped_math_glyphs,
            evidence=resumed.pages[1].math_recovery_evidence,
            text=page_out.text,
        )
        is None
    )


def test_a_fresh_recovery_outranks_a_restored_one(tmp_path: Path) -> None:
    """Evidence is taken from the sidecar only when this run recorded none.

    Reprocessing a page that fails this time must not be vouched for by the
    coverage the previous run proved.
    """
    regions = [_region(DAMAGED_EQUATION, valid=False)]
    fresh = corrupt_region_evidence(regions)
    assert fresh["covered_crops"] == []
    assert (
        unresolved_math_detail(has_unmapped_math_glyphs=True, evidence=fresh, text=PUA_NATIVE_TEXT)
        is not None
    )


def test_a_witness_removed_from_the_assembled_body_is_missing() -> None:
    """The document-level reconciliation's own question, at unit level.

    Assemble's per-page reduction reads each page's finalized output, but the
    image sweeps that run afterwards operate on the ASSEMBLED body. If one of
    them removes a recovered region's crop reference, the evidence pointer is
    gone from the document while the page's own copy still had it -- so the
    reconciliation re-asks, against `final_text`, whether the replacements are
    still there. It asks about witnesses only: run in full against the whole
    document, the residual-PUA term would read another page's damage as this
    page's and name the wrong page.
    """
    from socr.math.accounting import LANE_CORRUPT_REGION, missing_coverage_witnesses
    from socr.math.recover import _CORRUPT_CANDIDATE_HEADER

    crop = "equations/p1_r1.png"
    evidence = {"lane": LANE_CORRUPT_REGION, "regions_total": 1, "covered_crops": [crop]}
    intact = f"prose\n![Corrupt equation crop]({crop})\n{_CORRUPT_CANDIDATE_HEADER}\n$$\nx\n$$"

    assert missing_coverage_witnesses(evidence, intact) == []
    assert missing_coverage_witnesses(evidence, "prose with the crop reference swept away") == [
        crop
    ]
    # A refusal wears the same crop reference; only the candidate header
    # distinguishes it, so the recogniser must not accept the crop alone.
    refusal = f"prose\n![Corrupt equation crop]({crop})\n[corrupt equation unresolved: bad latex]"
    assert missing_coverage_witnesses(evidence, refusal) == [crop]
    # A lane that proves no span coverage has no witnesses to lose.
    assert missing_coverage_witnesses({"lane": "clean_region", "covered_crops": [crop]}, "") == []
