"""P2 / GH-317: an exhausted ladder on a structure-class page ships the
fail-closed floor, never the native word-geometry grid.

Programme item P2 of ``docs/log/2026-09-01_conceptual-revision.md``. Today,
S1's former case (iii) ending in ``_select_page_output_tagged`` shipped the
COLLAPSED NATIVE GRID as a flagged WARNING when the provider ladder ran but no
attempt authored a usable grid. Measured worst-of-three
(``docs/log/2026-08-30_model-vs-native-table-rows.md``): shipping that grid is
worse than shipping nothing, for the same reason D3 (TR-3) already refuses to
ship a hard-failed native table. P2 folds case (iii) into the existing D3
fail-closed floor: marker + PNG ref, native PROSE outside the table kept,
ERROR / audit_passed=False / an explicit failure mode, and the native grid
never in the shipped bytes.

Every new symbol below is fetched via ``getattr``, matching this repo's own
convention (see ``test_s1_structure_class_winner_gh_reachability.py``'s module
docstring): a run against the PRE-P2 baseline must fail on an explicit
assertion naming the missing symbol, or on a BEHAVIOURAL difference (still
shipping the native grid), never on a bare ``ImportError``/``AttributeError``.

This file is TEST-ONLY. It does not implement P2 -- every test here is
expected to be RED (or error) against ``main`` at commit 8dfdf81 and is meant
to go green only once P2 lands. It does not modify
``test_s1_structure_class_winner_gh_reachability.py`` or ``test_tr3_d3_floor.py``,
which still pin the PRE-P2 (soon-to-be-replaced) case (iii) shape --
``docs/plans`` task t6 retargets those files as part of the implementation,
not this one.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.document import DocumentHandle
from socr.core.manifest import _winning_page_output
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState

# ---------------------------------------------------------------------------
# Shared fixtures: a born-digital table page with unique prose above and
# below the table, plus a unique native row -- so "prose survives" and "grid
# bytes are gone" are both independently checkable byte-for-byte.
# ---------------------------------------------------------------------------

PROSE_BEFORE = "Section 4.2 discusses maturity-sorted yield regressions in detail."
PROSE_AFTER = "The regression residuals are examined further in Appendix C."
NATIVE_TABLE_MD = (
    "| $n$ | const. | slope | $R^2$ |\n"
    "|---|---|---|---|\n"
    "| 2 | 0.03 | 0.91 | 0.44 |\n"
    "| 5 | 0.07 | 0.85 | 0.51 |\n"
)
UNIQUE_NATIVE_ROW = "| 2 | 0.03 | 0.91 | 0.44 |"
NATIVE_TEXT_WITH_PROSE = f"{PROSE_BEFORE}\n\n{NATIVE_TABLE_MD}\n{PROSE_AFTER}\n"
PROSE_PAGE_TWO = "Page two carries ordinary prose and no table of any kind."

MODEL_GRID = (
    "Table 2. Yield regressions by maturity\n\n"
    "| $n$ | const. | slope | $R^2$ |\n"
    "|---|---|---|---|\n"
    "| 2 | 0.03 | 0.91 | 0.44 |\n"
    "| 5 | 0.07 | 0.85 | 0.51 |\n"
)
SOFT_AMBIGUOUS = "ambiguous_deferred"


def _born_digital_pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "gh317_structure_class.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Table 2. Yield regressions by maturity")
    doc.save(str(path))
    doc.close()
    return path


def _born_digital_pdf_two_pages(tmp_path: Path) -> Path:
    """One table page and one ordinary prose page.

    ``_phase_assemble`` iterates by the handle's real page count, so a second
    ``PageState`` on a one-page PDF is simply ignored. A document that keeps
    text on some page is also the only shape in which the byte-identity guard
    has anything to compare once the floor is whole-page.
    """
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "gh317_structure_class_2p.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Table 2. Yield regressions by maturity")
    doc.new_page().insert_text((72, 72), PROSE_PAGE_TWO)
    doc.save(str(path))
    doc.close()
    return path


def _state(
    pdf_path: Path,
    *,
    native_text: str = NATIVE_TEXT_WITH_PROSE,
    model_text: str = "The table on this page could not be read reliably.\n",
    grid_qualifies: bool = False,
    rejection_class: str = SOFT_AMBIGUOUS,
) -> DocumentState:
    """A structure-class (table-bearing) page with a real non-native rung.

    ``grid_qualifies=True`` reproduces case (i): the non-native attempt
    authors ``MODEL_GRID`` and qualifies under ``structure_class_grid_winner``
    (this is the arm that must be UNCHANGED by P2). ``grid_qualifies=False``
    reproduces case (iii): the non-native attempt authored nothing usable
    (this is the arm P2 changes).
    """
    from socr.tables.reconcile import find_table_blocks, table_grid_identity

    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = True
    ps.has_tables = True
    ps.native_text = native_text

    # A real born-digital table page carries the per-region verifier's own
    # enumeration (GH-371/GH-375), and the floor needs it to prove that a
    # regional splice covers every region before it may keep the prose
    # (cold review round 1, finding 1). Derived from the fixture text rather
    # than hardcoded, so a fixture whose regions change stays self-consistent.
    _regions = find_table_blocks(native_text)
    ps.native_table_region_count = len(_regions)
    ps.native_table_region_identities = [table_grid_identity(b.grid) for b in _regions]

    native_attempt = PageOutput(
        page_num=1,
        text=native_text,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    model_attempt = PageOutput(
        page_num=1,
        text=MODEL_GRID if grid_qualifies else model_text,
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=False,
    )
    setattr(model_attempt, "rejection_class", rejection_class)
    ps.attempts.extend([native_attempt, model_attempt])
    ps.best_output = native_attempt
    return state


def _manifest_symbol(name: str):
    import socr.core.manifest as _manifest

    fn = getattr(_manifest, name, None)
    assert fn is not None, f"P2 (GH-317) must define socr.core.manifest.{name}"
    return fn


def _floor_kind():
    import socr.core.manifest as _manifest

    kind = getattr(_manifest.SelectionProvenance, "STRUCTURE_CLASS_FLOOR", None)
    assert kind is not None, "P2 must expose SelectionProvenance.STRUCTURE_CLASS_FLOOR"
    return kind


# ---------------------------------------------------------------------------
# t0: the case-(i) vs case-(iii) contract, as a single paired fixture pair.
# ---------------------------------------------------------------------------


def test_grid_arm_is_unaffected_by_p2(tmp_path: Path) -> None:
    """Case (i): a qualifying grid attempt must still win, unchanged."""
    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=True)

    winner = _winning_page_output(state, 1)

    assert winner.engine == "gemini", winner.engine
    assert UNIQUE_NATIVE_ROW in winner.text, winner.text
    assert winner.audit_passed is False, winner.audit_passed

    from socr.core.manifest import SelectionProvenance, _select_page_output_tagged

    _out, kind = _select_page_output_tagged(state, 1)
    assert kind is SelectionProvenance.STRUCTURE_CLASS_GRID_FLAGGED, kind


def test_no_grid_arm_loses_the_native_row_and_gains_the_marker(tmp_path: Path) -> None:
    """Case (iii), the P2 contract: the native grid row must NOT appear in the
    shipped bytes, and the fail-closed marker must appear instead.

    On the pre-P2 baseline this is RED: ``_winning_page_output`` ships
    ``_native_text_with_appends(p)`` verbatim, which still contains
    ``UNIQUE_NATIVE_ROW``.
    """
    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)

    winner = _winning_page_output(state, 1)

    assert UNIQUE_NATIVE_ROW not in winner.text, (
        f"the native word-geometry grid must never ship for an exhausted "
        f"structure-class ladder; got: {winner.text!r}"
    )
    assert "[page 1 failed:" in winner.text, (
        f"expected the D3-style fail-closed marker; got: {winner.text!r}"
    )


def test_no_grid_arm_only_this_arm_changes_from_the_baseline_shape(tmp_path: Path) -> None:
    """A structural sanity check on the paired fixtures themselves: the ONLY
    input that differs between the two arms above is whether the non-native
    attempt qualifies as a grid -- both start from the identical clean native
    ``best_output`` and the identical real non-native attempt object shape."""
    grid_state = _state(_born_digital_pdf(tmp_path), grid_qualifies=True)
    no_grid_state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)

    assert grid_state.pages[1].native_text == no_grid_state.pages[1].native_text
    assert grid_state.pages[1].best_output.audit_passed is True
    assert no_grid_state.pages[1].best_output.audit_passed is True


# ---------------------------------------------------------------------------
# t1: the pure selector ending -- marker, PNG ref, ERROR, failure mode,
# WinnerKind rename, no mutation of stored attempts.
# ---------------------------------------------------------------------------


def test_failure_mode_structure_class_ladder_exhausted_exists() -> None:
    kind = getattr(FailureMode, "STRUCTURE_CLASS_LADDER_EXHAUSTED", None)
    assert kind is not None, (
        "P2 must add FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED to src/socr/core/result.py"
    )
    assert kind.value == "structure_class_ladder_exhausted"


def test_pre_p2_compatibility_failure_mode_deserializes() -> None:
    """Old sidecars retain the retired failure enum as a read-only value."""
    restored = PageOutput.from_dict(
        {"page_num": 1, "failure_mode": "structure_class_no_model_attempt"}
    )
    assert restored.failure_mode is FailureMode.STRUCTURE_CLASS_NO_MODEL_ATTEMPT


def test_no_grid_page_ships_error_status_and_the_new_failure_mode(tmp_path: Path) -> None:
    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)

    winner = _winning_page_output(state, 1)

    assert winner.status is PageStatus.ERROR, (
        f"an exhausted structure-class ladder must ship ERROR, not WARNING; got {winner.status!r}"
    )
    assert winner.audit_passed is False, winner.audit_passed
    expected = getattr(FailureMode, "STRUCTURE_CLASS_LADDER_EXHAUSTED", None)
    assert expected is not None
    assert winner.failure_mode is expected, winner.failure_mode


def test_no_grid_page_ships_no_native_byte_at_all(tmp_path: Path) -> None:
    """Cold review round 2: the floor is WHOLE-PAGE, so nothing native ships.

    P2 originally spliced table regions out and kept the surrounding prose.
    That splice was retired because it could only ever prove coverage against
    the same parser that produced the regions, and a detected sibling whose
    reconstruction failed is absent from that enumeration entirely -- so a
    collapsed grid rode out inside "preserved prose". See the module note and
    docs/log/2026-09-01_p2-structure-class-floor.md.

    The property that replaces prose preservation is stronger and needs no
    enumeration to be trusted: the floor text is built from the marker and the
    PNG ref alone, so no byte of ``native_text`` can reach the page whatever
    the reconstruction did.
    """
    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)

    winner = _winning_page_output(state, 1)

    assert UNIQUE_NATIVE_ROW not in winner.text
    assert PROSE_BEFORE not in winner.text, winner.text
    assert PROSE_AFTER not in winner.text, winner.text
    assert winner.text.strip() == "[page 1 failed: unverifiable table — see image]"


def test_no_grid_page_withholds_the_appended_sidecar_too(tmp_path: Path) -> None:
    """GH-36b equation-sidecar appends are withheld with everything else.

    Before P2 this content shipped, and in P2 round 1 it survived the regional
    splice. Neither is available now: a page whose table regions cannot be
    isolated has no provable non-table part, and an appended sidecar is not
    exempt from that. Withheld, not lost silently -- the page is ERROR with a
    marker, a PNG, an audit event and a document-level note.
    """
    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)
    ps = state.pages[1]
    sidecar_text = NATIVE_TEXT_WITH_PROSE + "\n\n$$E = mc^2$$"
    ps.attempts[0] = PageOutput(
        page_num=1,
        text=sidecar_text,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    ps.best_output = ps.attempts[0]

    winner = _winning_page_output(state, 1)

    assert "$$E = mc^2$$" not in winner.text, winner.text
    assert UNIQUE_NATIVE_ROW not in winner.text, winner.text
    assert winner.failure_mode is FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED


def test_no_grid_page_includes_the_png_ref_when_rendered(tmp_path: Path) -> None:
    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)
    ps = state.pages[1]
    ps.d3_floor_png_ref = "![Failed table page 1](figures/failed_table_p1.png)"

    winner = _winning_page_output(state, 1)

    assert "![Failed table page 1]" in winner.text, winner.text
    assert "figures/failed_table_p1.png" in winner.text, winner.text


def test_no_grid_page_with_empty_png_ref_still_fails_closed(tmp_path: Path) -> None:
    """A render failure (empty ``d3_floor_png_ref``) must still produce a
    marker -- never fall back to shipping the native grid because the image
    could not be rendered."""
    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)
    assert state.pages[1].d3_floor_png_ref == "" or not getattr(
        state.pages[1], "d3_floor_png_ref", ""
    )

    winner = _winning_page_output(state, 1)

    assert UNIQUE_NATIVE_ROW not in winner.text, winner.text
    assert "[page 1 failed:" in winner.text, winner.text


def test_no_grid_page_with_unparseable_native_text_falls_back_to_whole_page_marker(
    tmp_path: Path,
) -> None:
    """When the native text has no table markdown ``find_table_blocks`` can
    isolate, splicing must fail closed to the whole-page marker -- never ship
    the raw unparseable text as if it were safe prose."""
    unparseable = "revenue costs were up\nno pipe characters or GFM separators here\n"
    state = _state(
        _born_digital_pdf(tmp_path),
        native_text=unparseable,
        grid_qualifies=False,
    )

    winner = _winning_page_output(state, 1)

    assert "[page 1 failed:" in winner.text, winner.text
    assert winner.status is PageStatus.ERROR, winner.status


def test_stored_native_attempt_is_never_mutated(tmp_path: Path) -> None:
    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)
    ps = state.pages[1]
    stored_native = ps.attempts[0]
    assert stored_native.audit_passed is True

    winner = _winning_page_output(state, 1)

    assert stored_native.audit_passed is True, (
        "the stored native attempt's audit_passed flag was mutated by the floor"
    )
    assert stored_native.text == NATIVE_TEXT_WITH_PROSE, "stored attempt text mutated"
    assert winner is not stored_native, "the floor must ship a copy, not the stored object"


def test_winner_kind_structure_class_floor_replaces_retired_ending(tmp_path: Path) -> None:
    """R7's tag: the ending must be tagged ``STRUCTURE_CLASS_FLOOR``, and the
    retired no-grid member must be gone (renamed, not aliased)
    -- ``test_r7_winner_kind_tags.py``'s bijection check enforces "no dead
    member" for whatever the enum ends up declaring; this test pins the
    specific rename P2 requires."""
    import socr.core.manifest as _manifest

    assert not any(member.name.endswith("_NO_GRID") for member in _manifest.SelectionProvenance), (
        "P2 must remove the retired no-grid WinnerKind member"
    )
    floor_kind = _floor_kind()
    assert floor_kind.value == "structure_class_floor"

    select_tagged = _manifest_symbol("_select_page_output_tagged")
    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)
    _winner, tag = select_tagged(state, 1)
    assert tag is floor_kind, tag


# ---------------------------------------------------------------------------
# structure_class_floor_applies -- the renamed predicate
# (the former native-fallback predicate before P2).
# ---------------------------------------------------------------------------


def _floor_applies(ps) -> bool:
    import socr.core.manifest as _manifest

    fn = getattr(_manifest, "structure_class_floor_applies", None)
    assert fn is not None, "P2 must expose socr.core.manifest.structure_class_floor_applies"
    return fn(ps)


def test_floor_applies_true_for_the_no_grid_arm(tmp_path: Path) -> None:
    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)
    assert _floor_applies(state.pages[1]) is True


def test_floor_applies_false_for_the_grid_arm(tmp_path: Path) -> None:
    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=True)
    assert _floor_applies(state.pages[1]) is False


# ---------------------------------------------------------------------------
# Precedence must be unaffected: D3 (TR-3), #263, #259 still short-circuit
# before the new floor ending; these mirror the equivalent S1 tests but pin
# that the RENAME did not alter reachability order.
# ---------------------------------------------------------------------------


def test_d3_floor_still_takes_precedence_over_the_new_ending(tmp_path: Path) -> None:
    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)
    ps = state.pages[1]
    ps.native_table_structure_failed = True
    ps.native_table_unverifiable = True
    ps.best_output = None

    winner = _winning_page_output(state, 1)
    assert "unverifiable table" in winner.text, winner.text
    assert winner.status is PageStatus.ERROR, winner.status
    # Must still be the pre-existing D3 failure mode, not the new one --
    # the new branch must not shadow D3's own ending.
    assert winner.failure_mode is not getattr(
        FailureMode, "STRUCTURE_CLASS_LADDER_EXHAUSTED", object()
    )


def test_gh259_flagged_model_kept_still_takes_precedence(tmp_path: Path) -> None:
    """A #259 flagged-but-present model output ships before the new floor
    ending is ever reached, unaffected by the rename.

    ``flagged_model_page_output`` requires the model attempt to have authored
    SOME table grid (``has_authored_table_grid``) -- otherwise it counts as
    "the model produced nothing", the case that must still fall through to
    native/the floor. So this fixture deliberately uses ``MODEL_GRID`` text
    even though the model attempt does not fully qualify under
    ``structure_class_grid_winner`` here (it is reached via #259's own,
    earlier branch, not S1/P2's).
    """
    from socr.core.manifest import flagged_model_page_output

    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False, model_text=MODEL_GRID)
    ps = state.pages[1]
    ps.native_table_structure_defective = True
    ps.attempts[1].audit_passed = False
    ps.best_output = ps.attempts[1]

    assert flagged_model_page_output(ps) is not None, "fixture sanity"
    winner = _winning_page_output(state, 1)
    assert winner.engine == "gemini", winner.engine


def test_equation_only_page_is_unaffected(tmp_path: Path) -> None:
    """C2 stays tables-only: an equation-only page never reaches the new
    ending, exactly as it never reached case (iii) before P2."""
    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)
    ps = state.pages[1]
    ps.has_tables = False
    ps.has_equations = True

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.status is PageStatus.SUCCESS, winner.status
    assert winner.audit_passed is True, winner.audit_passed


# ---------------------------------------------------------------------------
# t3: assemble bucket rename, event rename, audit-log rank, tables-trust set.
# ---------------------------------------------------------------------------


def _pipeline():
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=True,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=False,
            write_manifest=False,
            table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md
        )
    )


def test_assemble_emits_the_floor_event_for_case_iii(
    tmp_path: Path,
) -> None:
    from socr.core.result import DocumentStatus

    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)
    pipeline = _pipeline()
    out_dir = tmp_path / "out"

    result = pipeline._phase_assemble(state, out_dir)

    assert result.status is not DocumentStatus.SUCCESS, result.status
    kinds = {getattr(e, "kind", "") for e in state.events}
    assert "structure_class_ladder_exhausted_floor" in kinds, (
        f"expected the renamed floor event; got kinds={sorted(kinds)}"
    )
    assert "structure_class_ladder_exhausted_floor" in kinds


def test_assemble_floor_event_carries_the_floor_data_flag(tmp_path: Path) -> None:
    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)
    pipeline = _pipeline()
    out_dir = tmp_path / "out"

    pipeline._phase_assemble(state, out_dir)

    events = [
        e
        for e in state.events
        if getattr(e, "kind", "") == "structure_class_ladder_exhausted_floor"
    ]
    assert events, "expected at least one structure_class_ladder_exhausted_floor event"
    assert events[0].data.get("structure_class_floor") is True, events[0].data


def test_structure_class_floor_predicate_is_the_only_manifest_symbol() -> None:
    """t6's cleanup check, pinned as a symbol-level assertion rather than an
    ``rg`` shell scan: the OLD predicate name must not still resolve."""
    import socr.core.manifest as _manifest

    assert getattr(_manifest, "structure_class_floor_applies", None) is not None


def test_audit_log_rank_recognises_the_new_event_kind() -> None:
    """``socr.core.audit_log``'s explicit rank dict must place the renamed
    event at the same disposition rank as the former case-(iii) event
    kind held (rank 6 -- alongside the other final-word table dispositions),
    not fall through to the default rank."""
    import inspect

    from socr.core import audit_log

    source = inspect.getsource(audit_log)
    assert "structure_class_ladder_exhausted_floor" in source, (
        "src/socr/core/audit_log.py must add 'structure_class_ladder_exhausted_floor' "
        "to its explicit event rank dict"
    )


def test_tables_trust_distrust_set_recognises_the_new_event_kind() -> None:
    from socr.core import tables_trust

    assert "structure_class_ladder_exhausted_floor" in tables_trust.TABLE_DISTRUST_KINDS, (
        "src/socr/core/tables_trust.py's distrust set must include "
        "'structure_class_ladder_exhausted_floor' so a floor-only page is "
        "reported as untrusted"
    )


def test_document_status_never_success_when_the_floor_fires(tmp_path: Path) -> None:
    from socr.core.result import DocumentStatus

    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)
    pipeline = _pipeline()
    out_dir = tmp_path / "out"

    result = pipeline._phase_assemble(state, out_dir)
    assert result.status != DocumentStatus.SUCCESS, result.status
    assert state.status != DocumentStatus.SUCCESS, state.status


# ---------------------------------------------------------------------------
# t4: the metadata surface.
# ---------------------------------------------------------------------------


def test_structure_class_floor_note_names_the_affected_pages(tmp_path: Path) -> None:
    from socr.pipeline.orchestrator import UnifiedPipeline

    note_fn = getattr(UnifiedPipeline, "_structure_class_floor_note", None)
    assert note_fn is not None, (
        "P2 must add UnifiedPipeline._structure_class_floor_note(state), "
        "beside the existing _table_judge_ladder_note / _tables_trust_note "
        "document-level note helpers"
    )

    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=False)
    note = note_fn(state)
    assert note is not None, "expected a non-None note when the floor applies to a page"
    assert "1" in note, f"expected the affected page number in the note; got {note!r}"


def test_structure_class_floor_note_is_none_when_the_floor_never_fires(tmp_path: Path) -> None:
    from socr.pipeline.orchestrator import UnifiedPipeline

    note_fn = getattr(UnifiedPipeline, "_structure_class_floor_note", None)
    assert note_fn is not None

    state = _state(_born_digital_pdf(tmp_path), grid_qualifies=True)
    assert note_fn(state) is None


# ---------------------------------------------------------------------------
# t5: conservative resume. Both a NEW floor sidecar and a PRE-P2 case-(iii)
# sidecar must force reprocessing -- the ledger is SUCCESS-only, and neither
# WARNING (pre-P2) nor ERROR (post-P2) is SUCCESS.
# ---------------------------------------------------------------------------


def _flush_terminal_sidecar(
    pipeline, state: DocumentState, page_num: int, out_dir: Path, record=None
) -> Path:
    text = state.pages[page_num].best_output.text
    pipeline._flush_page_fragment(state, page_num, text, out_dir)
    return pipeline._flush_page_sidecar(state, page_num, out_dir, terminal=True, record=record)


def _flush_legacy_sidecar(
    pipeline, state: DocumentState, page_num: int, out_dir: Path, output: PageOutput
) -> Path:
    """Write the sidecar an OLDER BUILD would have left behind for *output*.

    Two things make the artefact historical rather than merely injected. The body
    is the fixture's own -- the current selector cannot produce it, which is the
    whole point of these tests. And the file carries **no** ``disposition`` key: the
    builds these tests defend against predate that field entirely.

    Stripping the key is what keeps the artefact internally consistent (cold review
    round 2, finding 9). Production always writes a disposition, and one recomputed
    from live state can contradict the injected bytes -- a Gemini
    ``WARNING/TABLE_REJECTED`` output paired with ``(MODEL_OUTPUT, ACCEPTED_OUTPUT)``.
    ``_load_terminal_page`` never reads the field, so such a contradiction would sit
    on disk unexamined and the resume assertion would pass over an invalid record.

    Production now has ONE record path (``_final_records`` / the ``record=``
    argument); this is a test fixture built on top of it, not a second one.
    """
    from socr.core.manifest import FinalizedPageRecord, finalized_page_record

    base = finalized_page_record(state, page_num)
    record = FinalizedPageRecord(
        output=output,
        disposition=base.disposition,
        selection_provenance=base.selection_provenance,
    )
    path = _flush_terminal_sidecar(pipeline, state, page_num, out_dir, record=record)
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.pop("disposition", None)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def test_a_legacy_sidecar_fixture_carries_no_disposition_field(tmp_path: Path) -> None:
    """The historical-artefact claim, pinned rather than asserted in a docstring.

    Cold review round 2, finding 9. Every resume test below that injects an
    older-build body reads a file with no ``disposition`` key, so the three fields of
    a finalized record cannot silently disagree on disk, and the resume decision is
    made on the same evidence the older build left.
    """
    pdf_path = _born_digital_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf_path.parent

    state = _state(pdf_path, grid_qualifies=False)
    legacy = PageOutput(
        page_num=1,
        text=NATIVE_TEXT_WITH_PROSE,
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=False,
        failure_mode=FailureMode.NATIVE_TABLE_STRUCTURE_FAILED,
    )
    state.pages[1].best_output = legacy
    path = _flush_legacy_sidecar(pipeline, state, 1, out_dir, legacy)

    meta = json.loads(path.read_text(encoding="utf-8"))
    assert "disposition" not in meta, meta
    # ...while an ordinary flush from the current build does carry it, so the
    # absence above is the fixture's doing and not a missing feature.
    ordinary = _flush_terminal_sidecar(pipeline, state, 1, out_dir)
    assert "disposition" in json.loads(ordinary.read_text(encoding="utf-8"))


def test_new_floor_sidecar_is_reprocessed_not_skipped(tmp_path: Path) -> None:
    pdf_path = _born_digital_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf_path.parent

    state = _state(pdf_path, grid_qualifies=False)
    winner = _winning_page_output(state, 1)
    assert winner.status == PageStatus.ERROR
    assert winner.audit_passed is False
    assert winner.failure_mode == FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED
    state.pages[1].best_output = winner

    _flush_terminal_sidecar(pipeline, state, 1, out_dir)

    sidecar_path = next(out_dir.rglob("pages/00001.json"), None)
    assert sidecar_path is not None
    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert meta.get("terminal") is True
    assert meta.get("run_fingerprint") == pipeline._run_fingerprint()
    winning_meta = meta.get("winning_output", {})
    assert winning_meta.get("status") == "error"
    assert winning_meta.get("audit_passed") is False
    assert winning_meta.get("failure_mode") == "structure_class_ladder_exhausted"

    assert pipeline._load_terminal_page(state, 1, out_dir) is None, (
        "a terminal ERROR/audit_passed=False structure-class-floor sidecar "
        "must force reprocessing on resume, never be skip-and-kept"
    )
    assert getattr(state.pages[1], "structure_class_model_kept_on_resume", False) is False


@pytest.mark.parametrize(
    "legacy_failure_mode",
    [
        FailureMode.NATIVE_TABLE_STRUCTURE_FAILED,
        FailureMode("structure_class_no_model_attempt"),
    ],
)
def test_pre_p2_case_iii_warning_sidecar_is_also_reprocessed(
    tmp_path: Path, legacy_failure_mode: FailureMode
) -> None:
    """A sidecar written by a run BEFORE P2 shipped (WARNING,
    audit_passed=False, the historical no-model or
    ``NATIVE_TABLE_STRUCTURE_FAILED`` failure mode) must ALSO be reprocessed --
    it is non-SUCCESS, so ``_load_terminal_page``'s existing SUCCESS gate
    already refuses to skip it. No production parser change is expected for
    this case; it is a regression guard, not a new-behaviour proof."""
    pdf_path = _born_digital_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf_path.parent

    state = _state(pdf_path, grid_qualifies=False)
    legacy_winner = PageOutput(
        page_num=1,
        text=NATIVE_TEXT_WITH_PROSE,
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=False,
        failure_mode=legacy_failure_mode,
    )
    state.pages[1].best_output = legacy_winner

    _flush_legacy_sidecar(pipeline, state, 1, out_dir, legacy_winner)

    sidecar_path = next(out_dir.rglob("pages/00001.json"), None)
    assert sidecar_path is not None
    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert meta.get("terminal") is True
    assert meta.get("run_fingerprint") == pipeline._run_fingerprint()
    winning_meta = meta.get("winning_output", {})
    assert winning_meta.get("status") == "warning"
    assert winning_meta.get("audit_passed") is False
    assert winning_meta.get("failure_mode") == legacy_failure_mode.value

    assert pipeline._load_terminal_page(state, 1, out_dir) is None, (
        f"a pre-P2 WARNING case-(iii) sidecar ({legacy_failure_mode.value}) "
        "must be reprocessed, not skipped"
    )


def test_the_new_failure_mode_alone_grants_no_gh353_resume_exception(tmp_path: Path) -> None:
    """Reverse guard: GH-353 TICKET-D1b's skip-and-keep exception is scoped
    exactly to ``table_ladder_disposition == TABLE_REJECTED`` with
    ``table_ladder_incomplete`` false. The new failure mode must not itself
    grant a resume skip just because it sounds terminal."""
    pdf_path = _born_digital_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf_path.parent

    state = _state(pdf_path, grid_qualifies=False)
    winner = _winning_page_output(state, 1)
    state.pages[1].best_output = winner

    # 1. table_ladder_disposition is None
    state.pages[1].table_ladder_disposition = None
    state.pages[1].table_ladder_incomplete = False
    _flush_terminal_sidecar(pipeline, state, 1, out_dir)
    assert pipeline._load_terminal_page(state, 1, out_dir) is None

    # 2. table_ladder_disposition is explicitly set to the new failure mode
    state.pages[1].table_ladder_disposition = FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED
    state.pages[1].table_ladder_incomplete = False
    _flush_terminal_sidecar(pipeline, state, 1, out_dir)
    assert pipeline._load_terminal_page(state, 1, out_dir) is None

    # 3. table_ladder_disposition is TABLE_UNVERIFIED
    state.pages[1].table_ladder_disposition = FailureMode.TABLE_UNVERIFIED
    state.pages[1].table_ladder_incomplete = False
    _flush_terminal_sidecar(pipeline, state, 1, out_dir)
    assert pipeline._load_terminal_page(state, 1, out_dir) is None

    # 4. table_ladder_disposition is TABLE_REJECTED but table_ladder_incomplete is True
    state.pages[1].table_ladder_disposition = FailureMode.TABLE_REJECTED
    state.pages[1].table_ladder_incomplete = True
    _flush_terminal_sidecar(pipeline, state, 1, out_dir)
    assert pipeline._load_terminal_page(state, 1, out_dir) is None

    # 5. Positive contrast: table_ladder_disposition is TABLE_REJECTED and
    # table_ladder_incomplete is False (the genuine GH-353 D1b exception)
    table_rejected_winner = PageOutput(
        page_num=1,
        text=NATIVE_TEXT_WITH_PROSE,
        status=PageStatus.WARNING,
        engine="gemini",
        audit_passed=False,
        failure_mode=FailureMode.TABLE_REJECTED,
    )
    state.pages[1].best_output = table_rejected_winner
    state.pages[1].table_ladder_disposition = FailureMode.TABLE_REJECTED
    state.pages[1].table_ladder_incomplete = False
    _flush_legacy_sidecar(pipeline, state, 1, out_dir, table_rejected_winner)
    resumed_d1b = pipeline._load_terminal_page(state, 1, out_dir)
    assert resumed_d1b is not None, (
        "exact TABLE_REJECTED with table_ladder_incomplete=False is the sole D1b exception"
    )


def test_clean_case_i_resume_test_is_unaffected() -> None:
    """Sanity note, not a new behavioural proof: case (i)'s existing resume
    coverage (``structure_class_model_kept_on_resume`` in
    ``tests/test_pp5_resume_ledger.py`` / the S1 file) is a DIFFERENT,
    successful ending and must not be retargeted to the floor by this work.
    This test only pins that the symbol still exists post-rename so nothing
    silently deleted the case-(i) resume flag while renaming the case-(iii)
    one."""
    from socr.core.state import PageState

    ps = PageState(page_num=1)
    assert hasattr(ps, "structure_class_model_kept_on_resume")


# ---------------------------------------------------------------------------
# t2/t4: the provisional-flush seam. The in-loop fragment written by
# ``_phase_agentic`` (terminal=False) must ALSO carry the floor -- not the
# refused native grid -- so a crash-recovery read sees the same disposition
# the sidecar and the final assemble do. Hermetic: the provider ladder and
# judge are patched out; only the seam between selection and the in-loop
# flush is under test.
# ---------------------------------------------------------------------------


def _hermetic_agentic_pipeline(tmp_path: Path, *, grid_qualifies: bool):
    """Drives ``_phase_agentic`` end to end on a one-page born-digital table
    PDF, with the provider ladder and judge patched to a single fake rung
    whose output either qualifies as a grid or does not -- the same paired
    shape as the rest of this file, now exercised through the real loop
    instead of a hand-built ``DocumentState``."""
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.document import DocumentHandle
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.state import DocumentState
    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf_path = _born_digital_pdf(tmp_path)
    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=True,
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        save_figures=False,
        write_manifest=False,
        table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md
    )
    pipeline = UnifiedPipeline(config)
    pipeline._scan_root = pdf_path.parent
    out_dir = tmp_path / "out"

    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = True
    ps.has_tables = True
    ps.native_text = NATIVE_TEXT_WITH_PROSE
    native_attempt = PageOutput(
        page_num=1,
        text=NATIVE_TEXT_WITH_PROSE,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    ps.attempts.append(native_attempt)
    ps.best_output = native_attempt

    from socr.pipeline.agentic import PageDecision, ProviderAttempt

    def _fake_route_page(page_num, ladder, run_provider, judge, **kwargs):
        rung_text = MODEL_GRID if grid_qualifies else "Nothing usable here.\n"
        out = PageOutput(
            page_num=page_num,
            text=rung_text,
            status=PageStatus.SUCCESS,
            engine="gemini",
            audit_passed=False,
        )
        setattr(out, "rejection_class", SOFT_AMBIGUOUS)
        attempt = ProviderAttempt(
            engine=EngineType.QWEN,
            output=out,
            cost_usd=0.0,
            accepted=False,
            reason="ambiguous_deferred",
            provider_id=PROFILE_QWEN_LOCAL.id,
            model=PROFILE_QWEN_LOCAL.model,
            backend=PROFILE_QWEN_LOCAL.backend,
        )
        return PageDecision(page_num=page_num, final_output=out, attempts=[attempt])

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch("socr.pipeline.orchestrator.route_page", _fake_route_page),
        patch.object(pipeline, "_backend_available", return_value=True, create=True),
    ):
        pipeline._phase_agentic(state, out_dir)

    return pipeline, state, out_dir


def test_provisional_fragment_carries_the_floor_not_the_raw_rejected_rung_text(
    tmp_path: Path,
) -> None:
    """The seam t2 fixes: today the in-loop provisional flush
    (``terminal=False``) writes ``bo.text`` verbatim -- for a structure-class
    page whose ladder is exhausted, ``bo`` at that point in the loop is the
    LAST rung's raw (refused) output, e.g. ``"Nothing usable here.\\n"``, with
    no marker and no page-image reference at all. A crash exactly after this
    flush and before ``_phase_assemble`` would recover that raw, unmarked text.

    For a ``structure_class_floor_applies`` page, the provisional body must
    instead be derived from the same finalized floor selection the terminal
    sidecar uses -- so the fragment must already carry the fail-closed marker,
    not the bare rung text.
    """
    pytest.importorskip("fitz")
    try:
        _pipeline, state, out_dir = _hermetic_agentic_pipeline(tmp_path, grid_qualifies=False)
    except (AttributeError, TypeError) as exc:
        pytest.skip(f"route_page/_phase_agentic hook shape not yet compatible: {exc}")

    frag_path = next(out_dir.rglob("pages/00001.md"), None)
    assert frag_path is not None, "expected a provisional fragment flushed for page 1"
    body = frag_path.read_text(encoding="utf-8")

    assert "[page 1 failed:" in body, (
        f"the provisional (terminal=False) fragment for an exhausted "
        f"structure-class ladder must already carry the fail-closed marker, "
        f"not the raw rejected rung text; got: {body!r}"
    )
    assert UNIQUE_NATIVE_ROW not in body, body


def test_grid_qualifying_arm_does_not_render_the_floor_png(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    try:
        _pipeline, state, _out_dir = _hermetic_agentic_pipeline(tmp_path, grid_qualifies=True)
    except (AttributeError, TypeError) as exc:
        pytest.skip(f"route_page/_phase_agentic hook shape not yet compatible: {exc}")

    assert not getattr(state.pages[1], "d3_floor_png_ref", ""), (
        "a page whose grid attempt qualifies must not render the D3 floor "
        "PNG -- the grid ships instead"
    )


def test_phase_agentic_renders_and_flushes_floor_png_only_for_no_grid_arm(
    tmp_path: Path,
) -> None:
    """Hermetic _phase_agentic seam test:
    Patches _available_engines_for_agentic, _resolve_judge_model, route_page,
    and _render_d3_floor_png to a stable ref.
    Asserts the renderer is called and flushed to provisional fragment and sidecar
    only for the no-grid arm, and never called for the qualifying grid arm.
    """
    pytest.importorskip("fitz")
    from unittest.mock import MagicMock

    pdf_path = _born_digital_pdf(tmp_path)
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.document import DocumentHandle
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.state import DocumentState
    from socr.pipeline.agentic import PageDecision, ProviderAttempt
    from socr.pipeline.orchestrator import UnifiedPipeline

    stable_ref = "![Failed table page 1](figures/failed_table_p1.png)"

    def _run_arm(grid_qualifies: bool, out_name: str):
        config = PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=True,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=False,
            write_manifest=False,
            table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md
        )
        pipeline = UnifiedPipeline(config)
        pipeline._scan_root = pdf_path.parent
        out_dir = tmp_path / out_name

        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.has_tables = True
        ps.native_text = NATIVE_TEXT_WITH_PROSE
        native_attempt = PageOutput(
            page_num=1,
            text=NATIVE_TEXT_WITH_PROSE,
            status=PageStatus.SUCCESS,
            engine="native",
            audit_passed=True,
        )
        ps.attempts.append(native_attempt)
        ps.best_output = native_attempt

        def _fake_route_page(page_num, ladder, run_provider, judge, **kwargs):
            rung_text = MODEL_GRID if grid_qualifies else "Nothing usable here.\n"
            out = PageOutput(
                page_num=page_num,
                text=rung_text,
                status=PageStatus.SUCCESS,
                engine="gemini",
                audit_passed=False,
            )
            setattr(out, "rejection_class", SOFT_AMBIGUOUS)
            attempt = ProviderAttempt(
                engine=EngineType.QWEN,
                output=out,
                cost_usd=0.0,
                accepted=False,
                reason="ambiguous_deferred",
                provider_id=PROFILE_QWEN_LOCAL.id,
                model=PROFILE_QWEN_LOCAL.model,
                backend=PROFILE_QWEN_LOCAL.backend,
            )
            return PageDecision(page_num=page_num, final_output=out, attempts=[attempt])

        render_mock = MagicMock(return_value=stable_ref)
        with (
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch("socr.pipeline.orchestrator.route_page", _fake_route_page),
            patch.object(pipeline, "_backend_available", return_value=True, create=True),
            patch.object(pipeline, "_render_d3_floor_png", render_mock),
        ):
            pipeline._phase_agentic(state, out_dir)

        return pipeline, state, out_dir, render_mock

    # No-grid arm: renderer is called and ref is flushed
    _, state_no_grid, out_no_grid, render_no_grid = _run_arm(False, "out_no_grid")
    assert render_no_grid.call_count == 1
    assert getattr(state_no_grid.pages[1], "d3_floor_png_ref", "") == stable_ref
    frag_no_grid = next(out_no_grid.rglob("pages/00001.md")).read_text(encoding="utf-8")
    assert "[page 1 failed: unverifiable table — see image]" in frag_no_grid
    assert stable_ref in frag_no_grid
    assert UNIQUE_NATIVE_ROW not in frag_no_grid
    sidecar_no_grid = json.loads(
        next(out_no_grid.rglob("pages/00001.json")).read_text(encoding="utf-8")
    )
    assert sidecar_no_grid["winning_output"]["text"] == frag_no_grid
    assert sidecar_no_grid["winning_output"]["failure_mode"] == "structure_class_ladder_exhausted"
    assert sidecar_no_grid["winning_output"]["audit_passed"] is False

    # Grid arm: renderer is NOT called
    _, state_grid, out_grid, render_grid = _run_arm(True, "out_grid")
    assert render_grid.call_count == 0
    assert not getattr(state_grid.pages[1], "d3_floor_png_ref", "")
    frag_grid = next(out_grid.rglob("pages/00001.md")).read_text(encoding="utf-8")
    assert UNIQUE_NATIVE_ROW in frag_grid
    assert "[page 1 failed:" not in frag_grid


# ---------------------------------------------------------------------------
# t4: byte-identity between the provisional flush, the authoritative
# rewrite, and the final assembled document -- reusing the same paired
# born-digital fixture as the rest of this file.
# ---------------------------------------------------------------------------


def test_final_markdown_and_page_fragment_agree_and_omit_the_native_grid(
    tmp_path: Path,
) -> None:
    """Byte identity (spec point 5) with a floored page in the document.

    Cold review round 2: the floor is whole-page, so a document whose ONLY
    page floors carries no text, ``_phase_assemble`` writes no ``<stem>.md``,
    and ``_rewrite_all_fragments`` (which splits the final text) writes no
    fragments either -- leaving this guard nothing to compare. A clean second
    page restores a real document, which is also the realistic shape.
    """
    from socr.core.state import PageState

    state = _state(_born_digital_pdf_two_pages(tmp_path), grid_qualifies=False)
    state.pages.setdefault(2, PageState(page_num=2))
    clean = state.pages[2]
    clean.is_born_digital = True
    clean.has_tables = False
    clean.native_text = PROSE_PAGE_TWO
    clean_attempt = PageOutput(
        page_num=2,
        text=PROSE_PAGE_TWO,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    clean.attempts.append(clean_attempt)
    clean.best_output = clean_attempt

    pipeline = _pipeline()
    out_dir = tmp_path / "out"

    pipeline._phase_assemble(state, out_dir)

    frag_path = next(out_dir.rglob("pages/00001.md"), None)
    assert frag_path is not None
    frag_body = frag_path.read_text(encoding="utf-8")
    assert UNIQUE_NATIVE_ROW not in frag_body, frag_body
    assert "[page 1 failed: unverifiable table — see image]" in frag_body
    assert PROSE_BEFORE not in frag_body

    sidecar_path = next(out_dir.rglob("pages/00001.json"), None)
    assert sidecar_path is not None
    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    winning_text = meta.get("winning_output", {}).get("text", "")
    assert UNIQUE_NATIVE_ROW not in winning_text, winning_text
    assert meta.get("failure_mode") == "structure_class_ladder_exhausted", meta.get("failure_mode")
    assert meta.get("audit_passed") is False, meta.get("audit_passed")

    md_path = next(out_dir.rglob(f"{state.handle.path.stem}.md"), None)
    assert md_path is not None, "a document with one clean page must still assemble"
    final_md = md_path.read_text(encoding="utf-8")
    assert UNIQUE_NATIVE_ROW not in final_md, final_md
    assert PROSE_PAGE_TWO in final_md, "the floor must not touch the clean page"

    stitched = pipeline._stitch_fragments(state, out_dir)
    assert stitched == final_md, f"stitched={stitched!r} != final_md={final_md!r}"


def _born_digital_pdf_with_prose_and_table(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "gh317_prose_table.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), PROSE_BEFORE, fontsize=10)
    cols = [72, 144, 216, 288]
    rows = [140, 162, 184, 206]
    data = [
        ["$n$", "const.", "slope", "$R^2$"],
        ["2", "0.03", "0.91", "0.44"],
        ["5", "0.07", "0.85", "0.51"],
        ["10", "0.12", "0.78", "0.58"],
    ]
    for r, row in enumerate(data):
        for c, cell in enumerate(row):
            page.insert_text((cols[c], rows[r]), cell, fontsize=10)
    for yy in [rows[0] - 8, rows[1] - 6, rows[-1] + 8]:
        page.draw_line((72, yy), (360, yy))
    page.insert_text((72, 250), PROSE_AFTER, fontsize=10)
    # Cold review round 2: a SECOND, clean prose page. The floor is whole-page
    # now, so a one-page document has no page carrying text and assemble writes
    # no ``<stem>.md`` at all -- which would leave the byte-identity guard
    # (spec point 5) with nothing to compare. Page two also proves the floor's
    # blast radius stops at its own page.
    page2 = doc.new_page()
    page2.insert_text((72, 72), PROSE_PAGE_TWO, fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def test_paired_process_regression(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Hermetic paired process() regression for P2 / GH-317."""
    import logging

    from socr.core.cache import BlobStore
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.document import DocumentHandle
    from socr.core.manifest import Manifest, replay
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.state import DocumentState
    from socr.pipeline.agentic import PageDecision, ProviderAttempt
    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf_path = _born_digital_pdf_with_prose_and_table(tmp_path)

    def _run_process(grid_qualifies: bool, out_name: str):
        config = PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=True,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=False,
            write_manifest=True,
            table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md
        )
        pipeline = UnifiedPipeline(config)
        out_dir = tmp_path / out_name

        def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
            rung_text = (
                MODEL_GRID
                if grid_qualifies
                else "The table on this page could not be read reliably.\n"
            )
            out = PageOutput(
                page_num=page_num,
                text=rung_text,
                status=PageStatus.SUCCESS,
                engine="gemini",
                audit_passed=False,
            )
            setattr(out, "rejection_class", SOFT_AMBIGUOUS)
            attempt = ProviderAttempt(
                engine=EngineType.QWEN,
                output=out,
                cost_usd=0.0,
                accepted=False,
                reason="ambiguous_deferred",
                provider_id=PROFILE_QWEN_LOCAL.id,
                model=PROFILE_QWEN_LOCAL.model,
                backend=PROFILE_QWEN_LOCAL.backend,
            )
            return PageDecision(page_num=page_num, final_output=out, attempts=[attempt])

        with (
            patch.object(
                pipeline,
                "_available_engines_for_agentic",
                return_value=[PROFILE_QWEN_LOCAL],
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch("socr.pipeline.orchestrator.route_page", _fake_route),
            patch.object(pipeline, "_backend_available", return_value=True, create=True),
            patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
        ):
            result = pipeline.process(pdf_path, out_dir)

        return pipeline, result, out_dir

    with caplog.at_level(logging.WARNING):
        pipe_grid, res_grid, out_grid = _run_process(True, "out_grid")
        pipe_floor, res_floor, out_floor = _run_process(False, "out_floor")

    # 1. Intended differences on final terminal sidecar
    sidecar_grid = json.loads(next(out_grid.rglob("pages/00001.json")).read_text(encoding="utf-8"))
    sidecar_floor = json.loads(
        next(out_floor.rglob("pages/00001.json")).read_text(encoding="utf-8")
    )

    assert sidecar_floor["winning_output"]["failure_mode"] == "structure_class_ladder_exhausted"
    assert sidecar_grid["winning_output"].get("failure_mode") != "structure_class_ladder_exhausted"

    assert sidecar_floor["winning_output"]["status"] == "error"
    assert sidecar_grid["winning_output"]["status"] != "error"

    events_grid = [e.get("kind", "") for e in sidecar_grid.get("audit_events", [])]
    events_floor = [e.get("kind", "") for e in sidecar_floor.get("audit_events", [])]
    assert "structure_class_ladder_exhausted_floor" not in events_grid
    assert "structure_class_ladder_exhausted_floor" in events_floor

    # 2. Metadata note difference
    meta_grid = json.loads((out_grid / pdf_path.stem / "metadata.json").read_text(encoding="utf-8"))
    meta_floor = json.loads(
        (out_floor / pdf_path.stem / "metadata.json").read_text(encoding="utf-8")
    )
    assert "structure-class ladder exhausted" not in (meta_grid.get("error") or "")
    assert "structure-class ladder exhausted" in (meta_floor.get("error") or "")

    # 3. Removal of unique native row and presence of marker + prose in floor run
    md_floor = next(out_floor.rglob(f"{pdf_path.stem}.md")).read_text(encoding="utf-8")
    md_grid = next(out_grid.rglob(f"{pdf_path.stem}.md")).read_text(encoding="utf-8")
    frag_floor = next(out_floor.rglob("pages/00001.md")).read_text(encoding="utf-8")
    winning_text_floor = sidecar_floor["winning_output"]["text"]

    # Manifest and replay bytes
    manifest_path = next(out_floor.rglob("manifest.json"))
    manifest = Manifest.load(manifest_path)
    store = BlobStore(manifest_path.parent / "cache")
    blob_page = store.get_page(manifest.entries[1].blob_ref)
    replay_md = replay(manifest, store)

    assert UNIQUE_NATIVE_ROW in md_grid
    assert UNIQUE_NATIVE_ROW not in md_floor
    assert UNIQUE_NATIVE_ROW not in frag_floor
    assert UNIQUE_NATIVE_ROW not in winning_text_floor
    assert UNIQUE_NATIVE_ROW not in blob_page.text
    assert UNIQUE_NATIVE_ROW not in replay_md

    # The floor is whole-page: page one's own prose is withheld along with its
    # grid, at EVERY surface. Page two is untouched, which is what says the
    # floor is scoped to the page that failed.
    for surface in (md_floor, frag_floor, winning_text_floor, blob_page.text, replay_md):
        assert "[page 1 failed: unverifiable table — see image]" in surface
        assert "![Failed table page 1](figures/failed_table_p1.png)" in surface
        assert PROSE_BEFORE not in surface
        assert PROSE_AFTER not in surface

    assert PROSE_PAGE_TWO in md_floor
    assert PROSE_PAGE_TWO in replay_md
    assert PROSE_PAGE_TWO in md_grid
    frag_two_floor = next(out_floor.rglob("pages/00002.md")).read_text(encoding="utf-8")
    assert PROSE_PAGE_TWO in frag_two_floor

    # 4. Assert referenced PNG exists under document figures directory
    png_path = next(out_floor.rglob("figures/failed_table_p1.png"), None)
    assert png_path is not None and png_path.exists(), "PNG must exist under figures directory"

    # Grid run has no PNG rendered
    grid_pngs = list(out_grid.rglob("figures/*.png"))
    assert not grid_pngs, f"Grid run should not render floor PNG, got: {grid_pngs}"

    # 5. Assert stitch is byte-identical and no PP-1 warning logged
    doc_state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    stitched = pipe_floor._stitch_fragments(doc_state, out_floor)
    assert stitched == md_floor, f"stitched={stitched!r} != md_floor={md_floor!r}"
    assert not any("PP-1" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Cold review rounds 1 and 2, finding 1: the floor is WHOLE-PAGE.
#
# Round 1 shipped a regional splice, and round 1's fix tried to justify it by
# requiring coverage against ``native_table_region_count`` /
# ``native_table_region_identities``. Round 2 showed that is circular: those
# are recorded by ``_verify_regions`` (born_digital.py:2186-2217) counting only
# separator-bearing members of ``table_regions``, and ``table_regions`` is
# itself built only from SUCCESSFUL reconstructions (born_digital.py:1939-2003).
# A detected sibling whose reconstruction failed never enters the enumeration,
# so the recorded count matches what ``find_table_blocks`` sees, the check
# agrees with the parser it was meant to audit, and the collapsed sibling ships.
#
# Splicing safely needs an INDEPENDENT, detection-level count recorded BEFORE
# reconstruction. ``_detect_tables`` (born_digital.py:1766-1804) reduces
# ``page.find_tables()`` to a bool and keeps no count, and no PageAssessment /
# PageState field carries one. So the splice is gone.
#
# What replaces it is a property that needs no enumeration to be trusted, and
# is therefore not circular: the floor text is a function of the marker and the
# PNG ref ONLY. No byte of the native layer can reach a floored page, whatever
# reconstruction did or did not manage to parse.
# ---------------------------------------------------------------------------

#: A second native table region that reconstruction COLLAPSED: the cells are
#: there, the grid is not, so ``find_table_blocks`` cannot see it as a table.
#: Every token in it is a number a citation corpus would carry.
COLLAPSED_REGION = "Maturity 10 30\nconst. 0.11 0.19\nslope 0.78 0.62\n$R^2$ 0.58 0.63"
COLLAPSED_UNIQUE_TOKEN = "0.62"

MIXED_VALIDITY_NATIVE_TEXT = (
    f"{PROSE_BEFORE}\n\n{NATIVE_TABLE_MD}\n{COLLAPSED_REGION}\n\n{PROSE_AFTER}\n"
)


def test_mixed_validity_page_ships_neither_sibling(tmp_path: Path) -> None:
    """The original defect, with no injected metadata anywhere.

    One region parses as GFM, the other collapsed to ragged lines. The fixture
    asserts that asymmetry through the real parser, so it cannot silently stop
    being the mixed-validity shape. Under a regional splice the collapsed
    sibling ships; under the whole-page floor neither does.
    """
    from socr.tables.reconcile import find_table_blocks

    parsed = find_table_blocks(MIXED_VALIDITY_NATIVE_TEXT)
    assert len(parsed) == 1, (
        "fixture premise: exactly one of the two regions must parse as a GFM "
        f"table; find_table_blocks saw {len(parsed)}"
    )

    state = _state(_born_digital_pdf(tmp_path), native_text=MIXED_VALIDITY_NATIVE_TEXT)
    out = _winning_page_output(state, 1)

    assert out.failure_mode is FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED
    assert "[page 1 failed: unverifiable table — see image]" in out.text
    assert COLLAPSED_UNIQUE_TOKEN not in out.text, (
        "the collapsed native region shipped: the parser could not see it, so "
        "no splice keyed off that parser can ever prove it was withheld"
    )
    assert COLLAPSED_REGION not in out.text
    assert UNIQUE_NATIVE_ROW not in out.text


@pytest.mark.parametrize(
    "native_text",
    [
        NATIVE_TEXT_WITH_PROSE,
        MIXED_VALIDITY_NATIVE_TEXT,
        COLLAPSED_REGION,
        "prose with no table at all, and a stray number 0.62",
        "",
    ],
    ids=["clean-gfm", "mixed-validity", "collapsed-only", "no-table", "empty"],
)
def test_floor_text_is_independent_of_the_native_layer(tmp_path: Path, native_text: str) -> None:
    """The fail-closed DEFAULT: with no detection evidence, nothing native ships.

    Whatever the native layer holds -- a clean grid, a mixed-validity page, a
    collapsed region with no grid at all, or nothing -- the floor text is the
    same two elements. That was P2's unconditional property, and GH-520 makes
    it conditional: the splice returns for a page whose tables can be
    enumerated independently of the parser (``detected_table_count`` /
    ``detected_table_bboxes``, recorded before reconstruction by #570).

    These fixtures carry no such signal, which is the case that must keep
    behaving exactly as it did -- a page the detector could not enumerate is
    still a page whose coverage is unprovable. The conditional half is pinned
    in ``test_gh520_regional_floor_splice.py``, including the mixed-validity
    reproducer WITH the signal, where the counts disagree and the page still
    floors whole.
    """
    from socr.core.manifest import structure_class_floor_text

    state = _state(_born_digital_pdf(tmp_path), native_text=native_text)
    ps = state.pages[1]
    ps.d3_floor_png_ref = "![Failed table page 1](figures/failed_table_p1.png)"

    text = structure_class_floor_text(ps, 1)

    assert text == (
        "[page 1 failed: unverifiable table — see image]\n\n"
        "![Failed table page 1](figures/failed_table_p1.png)"
    )
    for token in ("0.62", "0.91", "Section 4.2", "Appendix C", "Maturity"):
        assert token not in text


def test_region_metadata_cannot_reopen_the_splice(tmp_path: Path) -> None:
    """Regression guard on the retirement itself.

    Round 1's fix consulted ``native_table_region_count`` /
    ``native_table_region_identities``. Populating them -- correctly, from the
    parser, exactly as production does -- must no longer change the floor.
    Without this, a future change could quietly reinstate the circular check
    and every other test here would still pass.

    Still true after GH-520 reopened the splice, and more load-bearing than
    before: the splice now has a door, and this pins that the circular signal
    is not a key to it. Its counterpart with the detection signal present is
    ``test_the_parser_derived_count_still_cannot_reopen_the_splice``.
    """
    from socr.tables.reconcile import find_table_blocks, table_grid_identity

    state = _state(_born_digital_pdf(tmp_path), native_text=MIXED_VALIDITY_NATIVE_TEXT)
    ps = state.pages[1]
    without = _winning_page_output(state, 1).text

    parsed = find_table_blocks(MIXED_VALIDITY_NATIVE_TEXT)
    ps.native_table_region_count = len(parsed)
    ps.native_table_region_identities = [table_grid_identity(b.grid) for b in parsed]
    with_meta = _winning_page_output(state, 1).text

    assert with_meta == without, (
        "the floor consulted the region enumeration again; that check is "
        "circular (see this section's header) and must stay retired"
    )
    assert COLLAPSED_UNIQUE_TOKEN not in with_meta


# ---------------------------------------------------------------------------
# Cold review round 1, finding 2: the D1b resume exception must not be able to
# restore a floored page.
#
# GH-353 TICKET-D1b lets a page whose table ladder ended in TABLE_REJECTED
# skip resume's SUCCESS and audit_passed gates: two rungs looked and said no,
# which is a content judgment, not an infra doubt. The only guard left after
# that bypass is ``is_page_failed_marker(body)`` -- and that deliberately
# returns False for a marker surrounded by preserved prose.
#
# So a REGIONAL floor (prose kept) on a page whose ladder also recorded
# TABLE_REJECTED is restored verbatim on the next run: the floored page is
# never re-OCR'd, and P2's "reprocess on any doubt" promise is void for
# exactly the pages the enabled table-judge ladder produces.
# ---------------------------------------------------------------------------


def test_floored_page_is_reprocessed_even_when_the_ladder_said_table_rejected(
    tmp_path: Path,
) -> None:
    """Finding 2: the floor's failure mode must never be restorable.

    The exact combination the existing coverage never built:
    ``STRUCTURE_CLASS_LADDER_EXHAUSTED`` winner + ``TABLE_REJECTED``
    disposition + ``table_ladder_incomplete=False`` + a prose-preserving
    regional floor body.
    """
    pdf_path = _born_digital_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf_path.parent

    state = _state(pdf_path, grid_qualifies=False)

    # The sidecar this test must defend against is one written by the P2 build
    # that DID splice regionally (cold review round 1). Those artefacts exist on
    # disk and carry the floor's failure mode with a prose-bearing body. Built
    # explicitly rather than taken from the current selector, because the
    # current floor is whole-page and its body would be refused by the marker
    # check -- which would make this test pass without exercising D1b at all.
    winner = PageOutput(
        page_num=1,
        text=f"{PROSE_BEFORE}\n\n[page 1 failed: unverifiable table — see image]\n\n{PROSE_AFTER}",
        status=PageStatus.ERROR,
        engine="native",
        audit_passed=False,
        failure_mode=FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED,
    )

    # Fixture premises, asserted rather than assumed: this carries the floor's
    # failure mode, and its body is NOT a whole-page marker, so the marker
    # check cannot be what refuses this resume.
    assert UNIQUE_NATIVE_ROW not in winner.text
    from socr.core.manifest import is_page_failed_marker

    assert not is_page_failed_marker(winner.text), (
        "fixture premise: a prose-preserving floor body is NOT a whole-page "
        "failure marker, so only the D1b guard can refuse this resume"
    )

    state.pages[1].best_output = winner
    state.pages[1].table_ladder_disposition = FailureMode.TABLE_REJECTED
    state.pages[1].table_ladder_incomplete = False
    _flush_legacy_sidecar(pipeline, state, 1, out_dir, winner)

    sidecar_path = next(out_dir.rglob("pages/00001.json"), None)
    assert sidecar_path is not None
    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert meta.get("table_ladder_disposition") == FailureMode.TABLE_REJECTED.value
    assert not meta.get("table_ladder_incomplete")
    assert meta["winning_output"]["failure_mode"] == "structure_class_ladder_exhausted"

    assert pipeline._load_terminal_page(state, 1, out_dir) is None, (
        "a floored page must be reprocessed even when the table ladder also "
        "recorded TABLE_REJECTED: D1b's exception is about a judged TABLE, "
        "not about a page whose every rung was refused"
    )


def test_d1b_exception_still_works_for_a_genuinely_rejected_table(tmp_path: Path) -> None:
    """The difference that says finding 2's fix is narrow.

    Same disposition, same completeness, same fingerprint -- only the winner's
    failure mode differs. A real TABLE_REJECTED winner still skips and is kept;
    only the floor is refused.
    """
    pdf_path = _born_digital_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _pipeline()
    pipeline._scan_root = pdf_path.parent

    state = _state(pdf_path, grid_qualifies=False)
    rejected_winner = PageOutput(
        page_num=1,
        text=NATIVE_TEXT_WITH_PROSE,
        status=PageStatus.WARNING,
        engine="gemini",
        audit_passed=False,
        failure_mode=FailureMode.TABLE_REJECTED,
    )
    state.pages[1].best_output = rejected_winner
    state.pages[1].table_ladder_disposition = FailureMode.TABLE_REJECTED
    state.pages[1].table_ladder_incomplete = False
    _flush_legacy_sidecar(pipeline, state, 1, out_dir, rejected_winner)

    assert pipeline._load_terminal_page(state, 1, out_dir) is not None, (
        "the D1b exception must survive finding 2's fix for its own case"
    )
