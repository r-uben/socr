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

from pathlib import Path
from unittest.mock import patch

import pytest
from test_ladder_binding_evidence import (  # noqa: E402
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


def _run(tmp_path: Path, *, suspect: bool, rung):
    pipeline = _make_pipeline()
    state = _make_state(_row_shift_pdf(tmp_path, f"{'suspect' if suspect else 'clean'}.pdf"))
    ps = state.pages[1]
    ps.prior_ocr_layer_suspect = suspect
    ps.prior_ocr_layer = suspect
    bo = _bo(_SHIFTED_MD)
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
    """(a) The positive direction. The visual rung FAILs; the layer must not rescue it.

    On a suspect page the binding evidence comes from an earlier OCR run, so a PASS there
    is the layer vouching for a reading. It may not overrule the readers that looked at
    the pixels, and it may not mint an acceptance credential on their behalf.
    """
    state, ps, bo = _run(tmp_path, suspect=True, rung=_reject_rung())

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
