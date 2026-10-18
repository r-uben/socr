"""#1156 round 4: the evidence guards, pinned by BEHAVIOUR at the gate.

Round 3 added three guards that stop the table-judge chain reading a suspect page's own
text layer, and pinned them with a test that counted the predicate's name in the module
source and compared a literal to itself. That is grep, not behaviour -- and it is why a
guard that CRASHED the ladder went green: the binding guard returned a bare ``None`` where
both callers unpack a pair, so every judged table on a flagged page became
``TABLE_UNVERIFIED / witness_preparation_error`` with no rung run at all. Both P0
directions looked closed because the whole ladder was being skipped.

These drive ``_run_table_judge_gate`` itself, on the ``test_ladder_binding_evidence``
row-shift fixture, with a clean-page control in which binding DOES fire.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from test_ladder_binding_evidence import (  # noqa: E402
    _CORRECT_MD,
    _SHIFTED_MD,
    _accept_rung,
    _bo,
    _events_of_kind,
    _make_pipeline,
    _make_state,
    _reject_rung,
    _row_shift_pdf,
)

from socr.core.result import FailureMode


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _run(tmp_path: Path, *, suspect: bool, rung, markdown=None, judge_outcome: str = ""):
    """Drive the gate on one page.

    ``markdown`` selects the binding RELATION to the page's own words: ``_SHIFTED_MD``
    contradicts them (same values, swapped row labels) and ``_CORRECT_MD`` matches, which
    is the only one that can produce a binding PASS. ``judge_outcome`` is what makes a
    credential mintable at all -- a credential stands in for a page judge that TIMED OUT,
    so without the typed outcome the credential assertion cannot fail either way.
    """
    pipeline = _make_pipeline()
    tag = f"{'suspect' if suspect else 'clean'}-{judge_outcome or 'none'}"
    state = _make_state(_row_shift_pdf(tmp_path, f"{tag}.pdf"))
    ps = state.pages[1]
    ps.prior_ocr_layer_suspect = suspect
    ps.prior_ocr_layer = suspect
    bo = _bo(_SHIFTED_MD if markdown is None else markdown)
    if judge_outcome:
        bo = replace(bo, judge_outcome=judge_outcome)
    with patch.object(pipeline, "_transcribe_cell_token", return_value=None):
        pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])
    return state, ps, bo


def test_the_ladder_still_runs_on_a_flagged_page(tmp_path: Path) -> None:
    """The round-4 P0 itself: an abstention must not fail the ladder.

    ``witness_preparation_error`` is an INFRASTRUCTURE cause: it says socr could not
    prepare a witness, not that anything was wrong with the table. A guard that produces
    it has not refused evidence, it has refused to look -- and the page is then demoted
    under a fake cause while the real question goes unasked.
    """
    state, ps, _bo_out = _run(tmp_path, suspect=True, rung=_accept_rung("high"))

    unverified = _events_of_kind(state, "table_ladder_unverified")
    causes = [(e.data or {}).get("cause") for e in unverified]
    assert "witness_preparation_error" not in causes, (
        f"the ladder crashed on a flagged page instead of abstaining: {causes}"
    )
    assert ps.table_ladder_disposition is not FailureMode.TABLE_UNVERIFIED or not unverified


def test_a_binding_pass_does_not_overrule_a_visual_rejection_on_a_flagged_page(
    tmp_path: Path,
) -> None:
    """(a) The positive direction, with the arm that makes it a test.

    Round 5 (all three seats): the first version of this was vacuous three ways.
    ``_SHIFTED_MD`` against this PDF yields CONTRADICT, never PASS, so a guard that
    abstained only on CONTRADICT would have satisfied it; ``_bo()`` sets no judge outcome,
    so a credential could not be minted on either arm and that assertion could not fail;
    and there was no control showing binding DOES overrule when the page is clean.

    ``_CORRECT_MD`` matches the page's own words, which is the only relation that produces
    a binding PASS, and the timeout outcome is what makes a credential mintable. The clean
    arm then shows the overrule happening -- accepted, ``verified_by_geometry``, credential
    minted -- and the suspect arm shows it refused, because on that page the words are an
    earlier OCR run.
    """
    from socr.core.result import JUDGE_OUTCOME_TIMEOUT
    from socr.judge.table_verdict import REASON_VERIFIED_BY_GEOMETRY

    clean_state, clean_ps, clean_bo = _run(
        tmp_path,
        suspect=False,
        rung=_reject_rung(),
        markdown=_CORRECT_MD,
        judge_outcome=JUDGE_OUTCOME_TIMEOUT,
    )
    accepted = _events_of_kind(clean_state, "table_ladder_accepted")
    assert accepted, (
        "control: binding does not overrule the visual rejection on a CLEAN page, so the "
        "suspect arm below proves nothing"
    )
    assert any(
        REASON_VERIFIED_BY_GEOMETRY in str((e.data or {}).get("reason", ""))
        or REASON_VERIFIED_BY_GEOMETRY in (e.detail or "")
        for e in accepted
    ), f"control: the overrule did not come from geometry: {[e.detail for e in accepted]}"
    assert getattr(clean_bo, "table_acceptance_credential", None), (
        "control: no credential was minted, so the suspect arm's credential assertion cannot fail"
    )
    assert clean_ps is not None

    state, ps, bo = _run(
        tmp_path,
        suspect=True,
        rung=_reject_rung(),
        markdown=_CORRECT_MD,
        judge_outcome=JUDGE_OUTCOME_TIMEOUT,
    )

    assert ps.table_ladder_disposition is not FailureMode.NONE, (
        "a visually REJECTED table was let through on a flagged page"
    )
    assert not _events_of_kind(state, "table_ladder_accepted"), (
        "the layer's binding evidence overruled the visual rejection"
    )
    assert not getattr(bo, "table_acceptance_credential", None), (
        "a credential was minted over a visual rejection, on the layer's say-so"
    )


def test_the_layer_does_not_convict_a_table_the_visual_rung_passed(tmp_path: Path) -> None:
    """(b) The destructive direction, the ``-350`` shape.

    The row-shift fixture is a page whose text layer contradicts the emitted table -- the
    same relation as a layer that dropped a minus. On a CLEAN page that contradiction is
    real evidence and caps the table at UNVERIFIED (the control below). On a flagged page
    the contradiction is the earlier OCR run disagreeing with a reading that may well be
    right, so it must not clamp or withhold.
    """
    clean_state, clean_ps, _ = _run(tmp_path, suspect=False, rung=_accept_rung("high"))
    assert clean_ps.table_ladder_disposition is FailureMode.TABLE_UNVERIFIED, (
        "control: binding does not fire on the clean page, so the pin below shows nothing"
    )
    assert _events_of_kind(clean_state, "table_ladder_unverified"), "control: no contradiction"

    state, ps, _ = _run(tmp_path, suspect=True, rung=_accept_rung("high"))

    assert ps.table_ladder_disposition is not FailureMode.TABLE_UNVERIFIED, (
        "the suspect layer convicted a table the visual rung passed"
    )
    assert not _events_of_kind(state, "table_ladder_unverified"), (
        "the layer's contradiction clamped a visually accepted table"
    )
    assert _events_of_kind(state, "table_ladder_accepted"), (
        "the table was neither accepted nor convicted -- the ladder did not resolve it"
    )


@pytest.mark.parametrize(
    ("refs", "expect_see_image", "expect_ref"),
    [
        (
            {
                "prior_ocr_layer": True,
                "invisible_text_over_raster": True,
                "invisible_scan_png_ref": "![Scan page 1](figures/invisible_scan_page_p1.png)",
            },
            True,
            "invisible_scan_page_p1",
        ),
        # A D3-floor ref is NOT this floor's image: the body must not carry it and the
        # marker must not promise it (round 4, r-uben should-fix).
        (
            {
                "prior_ocr_layer": True,
                "d3_floor_png_ref": "![Failed table page 1](figures/failed_table_p1.png)",
            },
            False,
            "failed_table_p1",
        ),
    ],
)
def test_the_image_helper_reaches_both_callers(
    tmp_path: Path, refs: dict, expect_see_image: bool, expect_ref: str
) -> None:
    """#1156 round 4 (r-uben should-fix): pin the helper AT ITS CALLERS, not only as a unit.

    Both reverts survived the suite with only the unit test in place: the disputed grid's
    ref list dropping ``invisible_scan_png_ref`` -- which is the ref every page carrying
    BOTH flags actually gets -- and the floor's old ``ships_an_image`` predicate, which
    counted a ref the floor does not ship.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_gh1156_part_a_interim import _grid_layer_text, _prior_ocr_pdf

    from socr.core.document import DocumentHandle
    from socr.core.manifest import REJECTION_AMBIGUOUS_DEFERRED, _select_and_finalize_page
    from socr.core.result import PageOutput, PageStatus
    from socr.core.state import DocumentState, PageState

    state = DocumentState(
        handle=DocumentHandle(path=_prior_ocr_pdf(tmp_path / "refs.pdf"), page_count=2)
    )
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = _grid_layer_text()
    p.needs_ocr_enhancement = True
    p.has_tables = True
    for name, value in refs.items():
        setattr(p, name, value)
    p.prior_ocr_layer_suspect = True
    rejected = PageOutput(
        page_num=1,
        text="| Metric | 1956 |\n| --- | --- |\n| Rate | 99.9 |\n",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_outcome="page_judge_completed",
        judge_accepted=False,
        rejection_class=REJECTION_AMBIGUOUS_DEFERRED,
    )
    p.attempts = [rejected]
    p.best_output = rejected
    state.pages[1] = p

    body = _select_and_finalize_page(state, 1).output.text
    if expect_see_image:
        assert expect_ref in body, f"the disputed grid shipped without its page image: {body[:200]}"
    else:
        assert expect_ref not in body, (
            "the page shipped the D3 floor's image, which belongs to a different ending"
        )
        assert "see image" not in body, "the marker promised an image the body lacks"


def test_the_contradiction_withhold_abstains_on_a_flagged_page(tmp_path: Path) -> None:
    """#1156 round 4: the DESTRUCTIVE guard, driven at its own seam.

    ``_run_table_judge_gate`` above does not reach this one -- the withhold runs later, on
    tables the ladder left UNVERIFIED -- so the gate tests cannot pin it and its mutant
    survived them. The row-shift fixture is again the ``-350`` relation: the page's own
    text contradicts the emitted table. On a clean page that is real evidence and the
    table is withheld; on a flagged page it is the earlier OCR run disagreeing with a
    reading that may be right, so nothing may be destroyed on its say-so.
    """

    def withheld(*, suspect: bool) -> bool:
        pipeline = _make_pipeline()
        state = _make_state(_row_shift_pdf(tmp_path, f"wh-{suspect}.pdf"))
        ps = state.pages[1]
        ps.prior_ocr_layer_suspect = suspect
        ps.prior_ocr_layer = suspect
        bo = _bo(_SHIFTED_MD)
        ps.table_ladder_disposition = FailureMode.TABLE_UNVERIFIED
        pipeline._withhold_contradicted_unverified_tables(state, 1, ps, bo)
        return ps.table_ladder_disposition is FailureMode.TABLE_WITHHELD

    assert withheld(suspect=False) is True, (
        "control: the contradiction does not fire on the clean page, so the pin below shows nothing"
    )
    assert withheld(suspect=True) is False, (
        "the suspect layer convicted a table and withheld it, destroying a reading that "
        "may be correct -- the -350 shape"
    )


def test_the_floors_marker_only_promises_an_image_the_floor_ships(tmp_path: Path) -> None:
    """#1156 round 4 (r-uben should-fix, the floor half).

    The disputed-grid test above reaches ``STRUCTURE_CLASS_GRID_FLAGGED``; this one
    reaches the FLOOR, which is where the wording predicate lives. A page carrying only
    ``d3_floor_png_ref`` is the case that separates the two rules: the old predicate
    counted that ref, so the marker said "see image" while the body carried none, and the
    reader was pointed at a picture that is not there.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_gh1156_part_a_interim import _grid_layer_text, _prior_ocr_pdf

    from socr.core.document import DocumentHandle
    from socr.core.manifest import _select_and_finalize_page
    from socr.core.state import DocumentState, PageState

    def floored(**refs) -> str:
        state = DocumentState(
            handle=DocumentHandle(path=_prior_ocr_pdf(tmp_path / "floor.pdf"), page_count=2)
        )
        p = PageState(page_num=1)
        p.is_born_digital = True
        p.native_text = _grid_layer_text()
        p.needs_ocr_enhancement = True
        p.prior_ocr_layer = True
        p.prior_ocr_layer_suspect = True
        for name, value in refs.items():
            setattr(p, name, value)
        state.pages[1] = p
        return _select_and_finalize_page(state, 1).output.text

    d3_only = floored(d3_floor_png_ref="![Failed table page 1](figures/failed_table_p1.png)")
    assert "failed_table_p1" not in d3_only, "the floor shipped the D3 floor's own image"
    assert "see image" not in d3_only, (
        "the marker promised an image the floor does not ship -- the reader is pointed at "
        "a picture that is not there"
    )
    assert "not transcribed" in d3_only

    own = floored(prior_ocr_page_png_ref="![Scanned page 1](figures/prior_ocr_layer_page_p1.png)")
    assert "prior_ocr_layer_page_p1" in own, "control: the floor's OWN image must still ship"
    assert "see image" in own


# #1156 round 4: the BOUNDARY re-read guard (``_record_unresolved_binding_boundary``) is
# NOT pinned here, and that is recorded rather than papered over. Its event only fires on
# geometry where ``bind()`` leaves a numeric cell unresolved AND the page re-read cannot
# settle it; the fixture in ``test_ladder_binding_evidence`` that produces it does not
# reproduce outside its own class, and a test that fails on the live source would "kill"
# every mutant pointed at it for the wrong reason -- which is exactly what my first
# attempt did. The guard is one line, identical in shape to the two pinned above, and the
# follow-up issue carries it. See ``leads/1156/part-a-r4-report.md``.
