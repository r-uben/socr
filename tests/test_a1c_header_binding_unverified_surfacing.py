"""TICKET-A1c (#641): a corroboration-fallback winner surfaces at every level.

A1b (#634) taught ``_select_page_output_tagged`` to rescue a structure-class
page from the fail-closed floor when no attempt cleared the strict
grid-authored pool but A1a's row check (ordered-number reproduction against
native words) corroborates one anyway. A1b's own review left a gap: the
rescued candidate's HEADER/COLUMN binding is never checked -- only its ROW
shape is -- so a candidate whose own ``audit_passed`` happened to be True
shipped through the OLD code as an undemoted ``SUCCESS`` / ``failure_mode
NONE``, indistinguishable from an ordinarily-verified S1 case (i) winner.
This ticket closes that: every corroboration-fallback winner ships
``status=WARNING``, ``failure_mode=HEADER_BINDING_UNVERIFIED``, and a
``table_corroboration`` record, UNCONDITIONALLY -- regardless of the
winner's own prior ``audit_passed``.

Hermetic, and deliberately at the same seam A1b's own acceptance tests use
(``_select_page_output_tagged`` / ``structure_class_grid_winner`` /
``structure_class_grid_corroboration`` called directly on hand-built
``PageState`` objects) rather than a full ``UnifiedPipeline.process()`` run:
tripping this exact branch through the real pipeline needs native word
GEOMETRY (``page.get_text("words")``) and a detected table bbox that
numerically line up with the injected model attempt's grid -- reproducing
that faithfully through born-digital detection and bbox detection buys
no additional coverage over calling the selection function process() itself
calls unconditionally, and ``tests/test_s1_structure_class_winner_corroboration.py``
(A1b's own ticket) already set this precedent for testing this exact
branch. No provider ladder, no ``_phase_agentic``, no ollama.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from socr.core.audit_log import AuditEvent
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.manifest import (
    SelectionProvenance,
    _select_page_output_tagged,
    structure_class_grid_corroboration,
)
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState
from socr.core.tables_trust import TABLE_DISTRUST_KINDS, build_tables_trust, trust_note
from socr.pipeline.orchestrator import UnifiedPipeline

NATIVE_PROSE = "Table 1 below reports quarterly balances for 2018-2020."

# A ragged (non-uniform-body) table: fails the strict pool's uniform-body
# check but is admitted to the corroboration fallback's own pool. Values
# reproduce NATIVE_WORDS below, row for row -- see
# tests/test_s1_structure_class_winner_corroboration.py, which this fixture
# shape is copied from (same GOOD_MD / NATIVE_WORDS / REGION triple).
GOOD_MD = (
    "| Year | A | B |\n"
    "|---|---|---|\n"
    "| units |\n"
    "| 2018 | 100.0 | 200.0 |\n"
    "| 2019 | 110.0 | 210.0 |\n"
    "| 2020 | 120.0 | 220.0 |\n"
)

# A clean, uniform-body grid: clears the strict pool directly when its
# attempt is audit_passed=True, so no corroboration fallback is ever
# consulted for it -- the "path off" control.
STRICT_MD = (
    "| Year | A | B |\n"
    "|---|---|---|\n"
    "| 2018 | 100.0 | 200.0 |\n"
    "| 2019 | 110.0 | 210.0 |\n"
    "| 2020 | 120.0 | 220.0 |\n"
)

REGION = (0.0, 0.0, 200.0, 100.0)


def _row_words(y: float, tokens: list[str]) -> list[tuple]:
    words = []
    x = 0.0
    for tok in tokens:
        words.append((x, y, x + 8.0, y + 10.0, tok))
        x += 12.0
    return words


NATIVE_WORDS: list[tuple] = (
    _row_words(10.0, ["2018", "100.0", "200.0"])
    + _row_words(30.0, ["2019", "110.0", "210.0"])
    + _row_words(50.0, ["2020", "120.0", "220.0"])
)


def _weak_best_output() -> PageOutput:
    """A non-passing, non-grid attempt to install as ``p.best_output``.

    ``_reaches_structure_class_branch`` shortcuts to an EARLIER passing-
    best-output branch whenever ``p.best_output.audit_passed`` is True on a
    non-native engine -- ``_winning_page_output`` never even reaches S1 in
    that case, so the S1/corroboration branch this ticket demotes is simply
    never entered regardless of ``manifest.py``'s fix. Every A1b fixture
    (``_grid_reading_output`` in the A1b test file) keeps ``best_output``
    audit_passed=False for exactly this reason; this test keeps the SAME
    convention and puts the candidate under test as a SEPARATE attempt (the
    real production shape a corroboration winner comes from -- see
    ``_row_corroborated_grid_winner``, which scores every attempt, not only
    ``best_output``).
    """
    return PageOutput(
        page_num=1,
        text="prose, not a table",
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=False,
        confidence=0.2,
    )


def _corroboration_page(*, winner_attempt: PageOutput) -> PageState:
    """A born-digital structure-class page with an empty strict grid pool
    and native words the fallback's row check can bind against -- the exact
    shape ``_floored_structure_class_page(with_native_words=True, ...)``
    builds in the A1b test file, generalized to two attempts so the
    candidate under test can carry ``audit_passed=True`` without tripping
    the early passing-``best_output`` shortcut (see ``_weak_best_output``).
    """
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = NATIVE_PROSE
    p.has_tables = True
    weak = _weak_best_output()
    p.attempts = [weak, winner_attempt]
    p.best_output = weak
    p.native_words = NATIVE_WORDS
    p.detected_table_bboxes = [REGION]
    return p


def _make_pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Table 1 prose")
    doc.save(str(path))
    doc.close()
    return path


def _state_with_page(tmp_path: Path, p: PageState) -> DocumentState:
    pdf_path = _make_pdf(tmp_path)
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    state.pages[1] = p
    return state


# --------------------------------------------------------------------------
# 1. Selection seam: pin the DIFFERENCE between the corroboration-fallback
#    winner (path ON) and an ordinary strict-pool winner (path OFF).
# --------------------------------------------------------------------------


def test_corroboration_precondition_present_for_the_on_case(tmp_path: Path) -> None:
    """Sanity: GOOD_MD really does reach the corroboration branch (empty
    strict pool, a fallback winner exists) -- otherwise the difference
    assertion below would be vacuous.
    """
    fallback_attempt = PageOutput(
        page_num=1,
        text=GOOD_MD,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,  # deliberately True: the OLD bug shipped this
        confidence=0.9,
    )
    p = _corroboration_page(winner_attempt=fallback_attempt)
    detail = structure_class_grid_corroboration(p)
    assert detail is not None, "fixture must reach the corroboration fallback"


def test_corroborated_winner_ships_warning_regardless_of_own_audit_passed(
    tmp_path: Path,
) -> None:
    """The ticket's core assertion: a corroboration-fallback winner whose OWN
    audit_passed is True must NOT ship as an undemoted SUCCESS (the #641
    defect) -- it ships WARNING / HEADER_BINDING_UNVERIFIED with a populated
    table_corroboration record, exactly like a corroborated winner whose
    audit_passed is False.
    """
    for own_audit_passed in (True, False):
        fallback_attempt = PageOutput(
            page_num=1,
            text=GOOD_MD,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=own_audit_passed,
            confidence=0.9,
        )
        p = _corroboration_page(winner_attempt=fallback_attempt)
        state = _state_with_page(tmp_path, p)

        winner, provenance = _select_page_output_tagged(state, 1)

        assert winner.status is PageStatus.WARNING, (own_audit_passed, winner)
        assert winner.audit_passed is False, (own_audit_passed, winner)
        assert winner.failure_mode is FailureMode.HEADER_BINDING_UNVERIFIED, (
            own_audit_passed,
            winner,
        )
        assert provenance is SelectionProvenance.STRUCTURE_CLASS_GRID_CORROBORATED

        tc = winner.table_corroboration
        assert tc is not None, "sidecar carrier must be populated"
        assert tc["engine"] == "qwen"
        assert tc["bound"] == 3
        assert tc["total"] == 3
        assert tc["share"] == pytest.approx(1.0)
        assert tc["extra_numbers"] == []
        assert tc["skipped_native_rows"] == 0
        assert tc["unbound_rows"] == [[]]
        assert tc["corroboration_region"] in ("bbox_union", "page")
        assert tc["header_text"]  # non-empty: "| Year | A | B |"


def test_ordinary_strict_pool_winner_is_unaffected_pin_the_difference(
    tmp_path: Path,
) -> None:
    """Path OFF, paired against path ON above: an attempt that clears the
    STRICT grid-authored pool directly (uniform body, audit_passed=True)
    never reaches the corroboration branch at all -- ships PASSING, no
    table_corroboration, failure_mode NONE. Same fixture family, only the
    strictness of the winning attempt's own table shape differs.
    """
    strict_attempt = PageOutput(
        page_num=1,
        text=STRICT_MD,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        confidence=0.9,
    )
    p = _corroboration_page(winner_attempt=strict_attempt)
    assert structure_class_grid_corroboration(p) is None, (
        "control fixture must NOT reach the corroboration branch"
    )

    state = _state_with_page(tmp_path, p)
    winner, provenance = _select_page_output_tagged(state, 1)

    assert winner.status is PageStatus.SUCCESS
    assert winner.audit_passed is True
    assert winner.failure_mode is FailureMode.NONE
    assert winner.table_corroboration is None
    assert provenance is SelectionProvenance.STRUCTURE_CLASS_GRID_PASSING


# --------------------------------------------------------------------------
# 2. Sidecar seam: PageOutput.to_dict()/from_dict() round-trips
#    table_corroboration exactly (this is the only channel the field has
#    to survive into pages/NNN.json).
# --------------------------------------------------------------------------


def test_table_corroboration_survives_sidecar_round_trip() -> None:
    record = {
        "engine": "qwen",
        "bound": 3,
        "total": 3,
        "share": 1.0,
        "extra_numbers": [],
        "skipped_native_rows": 0,
        "unbound_rows": [[]],
        "corroboration_region": "bbox_union",
        "coverage_share": 1.0,
        "header_text": "| Year | A | B |",
    }
    out = PageOutput(
        page_num=1,
        text=GOOD_MD,
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.HEADER_BINDING_UNVERIFIED,
        table_corroboration=record,
    )
    rebuilt = PageOutput.from_dict(json.loads(json.dumps(out.to_dict())))
    assert rebuilt.table_corroboration == record
    assert rebuilt.failure_mode is FailureMode.HEADER_BINDING_UNVERIFIED

    # A page that never touched the corroboration path round-trips None,
    # not a missing key defaulting to something falsy-but-wrong.
    plain = PageOutput(page_num=1, text="x", status=PageStatus.SUCCESS, engine="qwen")
    rebuilt_plain = PageOutput.from_dict(json.loads(json.dumps(plain.to_dict())))
    assert rebuilt_plain.table_corroboration is None


# --------------------------------------------------------------------------
# 3. Trust/metadata seam: the reused audit-event kind
#    (structure_class_row_corroborated, A1b's, not a new one) still rolls
#    up into tables_trust.json / metadata.json's error note. No new event
#    kind is introduced by this ticket, so no audit_log.py / tables_trust.py
#    edits are needed -- this test is the proof, not a change.
# --------------------------------------------------------------------------


def test_reused_corroboration_event_kind_already_rolls_up_to_document_trust() -> None:
    assert "structure_class_row_corroborated" in TABLE_DISTRUST_KINDS

    event = AuditEvent(
        page_num=1,
        kind="structure_class_row_corroborated",
        engine="qwen",
        detail="row corroboration kept this candidate: 3/3 rows bound",
        data={"bound": 3, "total": 3},
    )
    trust = build_tables_trust("paper.pdf", [event])
    assert trust.untrusted_pages == [1]
    assert trust.counts_by_kind().get("structure_class_row_corroborated") == 1

    note = trust_note(trust)
    assert note is not None
    assert "structure_class_row_corroborated" in note or "1 page" in note.lower() or note


# --------------------------------------------------------------------------
# 4. Resume seam: a HEADER_BINDING_UNVERIFIED sidecar must be reprocessed,
#    not skipped -- pinned against an ordinary clean SUCCESS sidecar (same
#    gate, same fixture family) which IS skippable, per test_gh161's own
#    paired pattern.
# --------------------------------------------------------------------------


def _make_pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.GEMINI],
            primary_engine=EngineType.DEEPSEEK,
            save_figures=False,
            dual_pass_tables=False,
            detect_equations=False,
            recover_clean_equations=False,
            quiet=True,
            write_manifest=False,
        )
    )


def test_header_binding_unverified_sidecar_is_not_resume_skippable(tmp_path: Path) -> None:
    """Realistic shape: build the page so ``_select_page_output_tagged``
    ITSELF produces the HEADER_BINDING_UNVERIFIED demotion (same
    ``_corroboration_page`` fixture as the selection-seam tests above),
    rather than hand-injecting an already-demoted ``PageOutput`` --
    ``_flush_page_sidecar`` calls the real selection function internally,
    so a hand-built winner that skips the real preconditions (no
    ``native_words`` / bbox to corroborate against) is silently overridden
    by the fail-closed floor instead, making the resume assertion vacuous.
    """
    pdf_path = _make_pdf(tmp_path)
    out_dir = tmp_path / "out"

    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent

    fallback_attempt = PageOutput(
        page_num=1,
        text=GOOD_MD,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        confidence=0.9,
    )
    p = _corroboration_page(winner_attempt=fallback_attempt)
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    state.pages[1] = p

    winner, _provenance = _select_page_output_tagged(state, 1)
    pipeline._flush_page_fragment(state, 1, winner.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

    sidecar = next(out_dir.rglob("pages/00001.json"))
    meta = json.loads(sidecar.read_text())
    assert meta["terminal"] is True, meta
    assert meta["winning_output"]["status"] == PageStatus.WARNING.value, meta["winning_output"]
    assert meta["winning_output"]["audit_passed"] is False, meta["winning_output"]
    assert meta["winning_output"]["failure_mode"] == FailureMode.HEADER_BINDING_UNVERIFIED.value, (
        meta["winning_output"]
    )
    assert meta["winning_output"]["table_corroboration"]["bound"] == 3

    resumed = pipeline._load_terminal_page(state, 1, out_dir)
    assert resumed is None, (
        "a HEADER_BINDING_UNVERIFIED sidecar (status=WARNING) was treated as "
        f"terminally clean and restored on resume: {resumed!r}"
    )
    # The gate's refusal above is exactly what makes the "bytes are not
    # bit-stable across runs" property hold for this failure mode: the row
    # corroboration record is derived from a fresh judge/route each run, so
    # a resumed run always re-derives it rather than replaying a stale one.


def test_clean_success_sidecar_is_still_resume_skippable_pin_the_difference(
    tmp_path: Path,
) -> None:
    """Reverse regression, same fixture family: a fix that made the gate
    refuse to skip ANYTHING would satisfy the test above and silently
    destroy resume. Only the status/audit_passed/failure_mode differ.
    """
    pdf_path = _make_pdf(tmp_path)
    out_dir = tmp_path / "out"

    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent

    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = True
    ps.native_text = NATIVE_PROSE

    clean = PageOutput(
        page_num=1,
        text=STRICT_MD,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    ps.attempts.append(clean)
    ps.best_output = clean

    pipeline._flush_page_fragment(state, 1, clean.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

    resumed = pipeline._load_terminal_page(state, 1, out_dir)
    assert resumed is not None, "an ordinary clean SUCCESS page must still be resume-skippable"
    assert resumed.status is PageStatus.SUCCESS


def test_to_dict_omits_table_corroboration_key_when_unset() -> None:
    """TICKET-A1c (#641): the new field must not change the serialized shape
    of a page that never touches row-corroboration.

    ``PageOutput.to_dict()`` is the input to the content-addressed cache hash
    (``blob_ref`` / ``page_fingerprint``, see ``core.cache``). Emitting
    ``"table_corroboration": None`` unconditionally -- the pattern every
    other field in this dataclass follows -- would change that hash for
    every already-terminal page in every corpus on the next resume, since
    the key did not exist in the dict before this ticket. Pinning the key's
    ABSENCE (not merely its value) is the whole point: a dict with
    ``"table_corroboration": None`` present is not byte-identical to one
    missing the key, even though both round-trip to ``None`` via
    ``from_dict``.
    """
    plain = PageOutput(page_num=1, text="hello", status=PageStatus.SUCCESS, engine="qwen")
    d = plain.to_dict()
    assert "table_corroboration" not in d, d
    assert PageOutput.from_dict(d).table_corroboration is None


def test_to_dict_carries_table_corroboration_when_set() -> None:
    """The reverse: a genuinely corroborated page's record round-trips."""
    record = {
        "engine": "qwen",
        "bound": 3,
        "total": 4,
        "share": 0.75,
        "extra_numbers": [],
        "skipped_native_rows": 0,
        "unbound_rows": [],
        "corroboration_region": "table",
        "coverage_share": 1.0,
        "header_text": "Year | Value",
    }
    corroborated = PageOutput(
        page_num=1,
        text="hello",
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.HEADER_BINDING_UNVERIFIED,
        table_corroboration=record,
    )
    d = corroborated.to_dict()
    assert d["table_corroboration"] == record
    assert PageOutput.from_dict(d).table_corroboration == record


# --------------------------------------------------------------------------
# 5. CLI summary line: mirrors
#    tests/test_ladder_status_surfacing.py::TestCliSummary::test_print_summary_names_both_terminals.
# --------------------------------------------------------------------------


def _make_summary_pipeline() -> UnifiedPipeline:
    from socr.core.config import EngineType as _EngineType

    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=_EngineType.DEEPSEEK,
            enabled_engines=list(_EngineType),
            agentic=False,
            quiet=False,
            native_first=True,
        )
    )


def _make_summary_state(tmp_path: Path, page_count: int = 1) -> DocumentState:
    from unittest.mock import patch as _patch

    pdf = tmp_path / "doc.pdf"
    with _patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf, page_count=page_count)
    state = DocumentState(handle=handle)
    for pn in range(1, page_count + 1):
        ps = state.pages[pn]
        ps.is_born_digital = True
        ps.native_text = f"page {pn} prose"
    return state


def test_print_summary_names_header_binding_unverified_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A HEADER_BINDING_UNVERIFIED page prints the count line; a clean control
    page does not.

    ``_phase_assemble`` re-derives every page's shipped output via
    ``finalized_page_records`` (``_select_and_finalize_page`` walks
    ``state.pages[n].attempts`` from scratch) rather than trusting a
    hand-set ``best_output`` directly -- unlike C2's ladder-terminal guard,
    there is no separate post-selection override attribute for this failure
    mode; it is produced INSIDE the selection cascade itself from real
    corroboration data. Patching ``finalized_page_records`` to return the
    exact finalized records is the seam that keeps this test hermetic
    without needing native words / a detected table bbox to actually trip
    the corroboration branch (that is covered end-to-end by
    ``structure_class_grid_corroboration`` tests above); this test only
    needs to prove the CLI reads ``failure_mode`` correctly once it is set.
    """
    from unittest.mock import patch as _patch

    from socr.core.manifest import (
        FinalizedPageRecord,
        PageDisposition,
        PageEnding,
        PagePrimaryReason,
        provenance_to_disposition,
    )

    pipeline = _make_summary_pipeline()
    state = _make_summary_state(tmp_path, page_count=2)

    corroborated = PageOutput(
        page_num=1,
        text="| A | B |\n| --- | --- |\n| 1 | 2 |\n",
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.HEADER_BINDING_UNVERIFIED,
        table_corroboration={"engine": "qwen", "bound": 1, "total": 1},
    )
    clean = PageOutput(
        page_num=2,
        text="page 2 clean text",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    records = [
        FinalizedPageRecord(
            output=corroborated,
            disposition=provenance_to_disposition(
                SelectionProvenance.STRUCTURE_CLASS_GRID_CORROBORATED
            ),
            selection_provenance=SelectionProvenance.STRUCTURE_CLASS_GRID_CORROBORATED,
        ),
        FinalizedPageRecord(
            output=clean,
            disposition=PageDisposition(
                PageEnding.NATIVE_PROSE, PagePrimaryReason.CLEAN_NATIVE_PROSE
            ),
            selection_provenance=SelectionProvenance.NATIVE_CLEAN,
        ),
    ]

    with _patch("socr.core.manifest.finalized_page_records", return_value=records):
        result = pipeline._phase_assemble(state, tmp_path)
    pipeline._print_summary(result, state)

    captured = capsys.readouterr()
    assert "header binding unverified" in captured.out
    assert "[1]" in captured.out


# --------------------------------------------------------------------------
# 6. Per-row markers: A1b's ``_apply_row_corroboration_disclosure`` /
#    ``_splice_unverified_row_markers`` already splice a trailing
#    ``<!-- row unverified -->`` marker onto each unbound candidate row
#    BEFORE this ticket's code runs (``grid_winner`` is already the
#    disclosed/spliced output by the time A1c's return builds
#    ``table_corroboration``) -- this section pins that the markers land on
#    EXACTLY the unbound rows (never a bound one) for a two-unbound-row
#    candidate, and that they survive ``_phase_assemble`` into the final
#    stitched ``.md``, not only the per-page fragment.
# --------------------------------------------------------------------------


def test_two_unbound_rows_marked_exactly_and_no_others(tmp_path: Path) -> None:
    """Force two of the three GOOD_MD rows (2018, 2020) unbound via a hand-
    built ``RowCorroboration`` (same technique as A1b's own single-row
    placement test, ``test_row_unverified_marker_spliced_for_unbound_row``
    in ``test_s1_structure_class_winner_corroboration.py``) patched in as
    ``structure_class_grid_corroboration``'s return, so the real
    ``clears`` share/extra-share gate (which a genuine 2/3-bound candidate
    would fail) is not in the way of testing marker PLACEMENT.
    """
    from dataclasses import replace as dc_replace
    from unittest.mock import patch as _patch

    from socr.tables.row_corroboration import corroborate_rows

    fallback_attempt = PageOutput(
        page_num=1,
        text=GOOD_MD,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        confidence=0.9,
    )
    p = _corroboration_page(winner_attempt=fallback_attempt)
    state = _state_with_page(tmp_path, p)

    real_rc = corroborate_rows(NATIVE_WORDS, GOOD_MD, REGION)
    forced_rc = dc_replace(real_rc, bound=1, unbound_rows=((0, 2),))

    with _patch(
        "socr.core.manifest.structure_class_grid_corroboration",
        return_value=(forced_rc, "bbox_union", 1.0),
    ):
        winner, provenance = _select_page_output_tagged(state, 1)

    assert provenance is SelectionProvenance.STRUCTURE_CLASS_GRID_CORROBORATED
    lines = (winner.text or "").splitlines()
    marked = [ln for ln in lines if "row unverified" in ln]
    assert len(marked) == 2, lines
    assert any("2018" in ln for ln in marked)
    assert any("2020" in ln for ln in marked)
    assert not any("2019" in ln for ln in marked)
    unmarked_2019 = [ln for ln in lines if "2019" in ln]
    assert unmarked_2019 and "row unverified" not in unmarked_2019[0]


def test_two_unbound_row_markers_survive_phase_assemble_into_final_md(
    tmp_path: Path,
) -> None:
    """Same forced two-unbound-row candidate, but through the real
    ``_phase_assemble`` -> per-page fragment -> stitched final ``.md``
    path, so a regression that only preserved markers on the in-memory
    winner (but dropped them somewhere in fragment flush / stitching /
    the byte-identity fallback) would be caught here, not just at the
    selection-seam level above.
    """
    from dataclasses import replace as dc_replace
    from unittest.mock import patch as _patch

    from socr.tables.row_corroboration import corroborate_rows

    fallback_attempt = PageOutput(
        page_num=1,
        text=GOOD_MD,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        confidence=0.9,
    )
    p = _corroboration_page(winner_attempt=fallback_attempt)
    pdf_path = _make_pdf(tmp_path)
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    state.pages[1] = p

    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent
    out_dir = tmp_path / "out"

    real_rc = corroborate_rows(NATIVE_WORDS, GOOD_MD, REGION)
    forced_rc = dc_replace(real_rc, bound=1, unbound_rows=((0, 2),))

    with _patch(
        "socr.core.manifest.structure_class_grid_corroboration",
        return_value=(forced_rc, "bbox_union", 1.0),
    ):
        pipeline._phase_assemble(state, out_dir)

    final_md_path = next(p for p in out_dir.rglob("*.md") if p.parent.name != "pages")
    final_text = final_md_path.read_text()
    marked_lines = [ln for ln in final_text.splitlines() if "row unverified" in ln]
    assert len(marked_lines) == 2, final_text
    assert any("2018" in ln for ln in marked_lines)
    assert any("2020" in ln for ln in marked_lines)
    assert "2019" in final_text
    assert not any("2019" in ln for ln in marked_lines)
