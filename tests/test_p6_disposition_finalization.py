"""P6 Stage A: the disposition is computed ONCE per page AFTER the shared
finalization guards, not from raw selection provenance.

Design: §8 Q3 ruling (Codex) -- the public disposition names what SHIPPED, not
what selection chose, without moving the emission/ladder guards themselves.
Plan task t4. Hermetic: constructs ``PageOutput``/``PageState`` fixtures
directly (the ``test_gh292_hybrid_bucket_matches_the_tag.py`` /
``test_gh226_table_emission_guard.py`` pattern), no provider, no pipeline run.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    PageDisposition,
    PageEnding,
    PagePrimaryReason,
    SelectionProvenance,
    finalized_page_records,
    page_disposition,
    provenance_to_disposition,
)
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402


def _pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text(
        (54, 72), "born-digital prose long enough to count as a real text layer here."
    )
    doc.save(path)
    doc.close()
    return path


def _state_with_marker_emission(tmp_path) -> DocumentState:
    """A page whose SELECTED output is an ordinary passing model reading, but
    whose text trips ``_apply_table_emission_guard`` (GH-226/GH-302) -- an
    invalid table emission. Selection provenance says "clean success"; what
    ships is a fail-closed marker.
    """
    state = DocumentState(handle=DocumentHandle.from_path(_pdf(tmp_path)))
    p = state.pages[1]
    p.is_born_digital = True
    p.native_text = "native prose"
    best = PageOutput(
        page_num=1,
        text="| a | b |\n| --- |\n| 1 | 2 |",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    p.attempts.append(best)
    p.best_output = best
    return state


def _state_with_ordinary_pass(tmp_path) -> DocumentState:
    state = DocumentState(handle=DocumentHandle.from_path(_pdf(tmp_path)))
    p = state.pages[1]
    p.is_born_digital = True
    p.native_text = "native prose"
    best = PageOutput(
        page_num=1,
        text="ordinary clean model prose, nothing table-shaped here at all.",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    p.attempts.append(best)
    p.best_output = best
    return state


def _state_with_ladder_rejection(tmp_path) -> DocumentState:
    """The ladder guard (GH-353 C3) demotes an otherwise-SUCCESS page's status
    without replacing its text -- a content-only demotion.
    """
    state = DocumentState(handle=DocumentHandle.from_path(_pdf(tmp_path)))
    p = state.pages[1]
    p.is_born_digital = True
    p.native_text = "native prose"
    best = PageOutput(
        page_num=1,
        text="ordinary clean model prose with a table the ladder later rejected.",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    p.attempts.append(best)
    p.best_output = best
    p.table_ladder_disposition = FailureMode.TABLE_REJECTED
    return state


def test_a_marker_replacement_forces_fail_closed_regardless_of_selection(tmp_path) -> None:
    """The load-bearing case: selection provenance ships PASSING_BEST_OUTPUT,
    but the final disposition must be (FAIL_CLOSED_MARKER, INVALID_TABLE_EMISSION)
    because that is what the reader actually receives.
    """
    state = _state_with_marker_emission(tmp_path)
    d = page_disposition(state, 1)
    assert d.ending is PageEnding.FAIL_CLOSED_MARKER
    assert d.primary_reason is PagePrimaryReason.INVALID_TABLE_EMISSION

    rec = finalized_page_records(state)[0]
    assert rec.disposition == d
    assert rec.selection_provenance == SelectionProvenance.PASSING_BEST_OUTPUT


def test_an_ordinary_pass_is_unaffected_by_the_guard(tmp_path) -> None:
    """Control: the guard must not fire on text that has no table defect, or
    the marker-replacement test above would be measuring nothing.
    """
    state = _state_with_ordinary_pass(tmp_path)
    d = page_disposition(state, 1)
    assert d.ending is PageEnding.MODEL_OUTPUT
    assert d.primary_reason is PagePrimaryReason.ACCEPTED_OUTPUT


def test_a_content_only_ladder_demotion_does_not_change_ending_or_reason(tmp_path) -> None:
    """§ constraints: "If the content-defect arm only demotes status without
    replacing bytes, or the ladder guard only demotes status, the selected
    ending/reason remains and those verdicts stay outside PageDisposition".
    """
    demoted = page_disposition(_state_with_ladder_rejection(tmp_path), 1)
    clean = page_disposition(_state_with_ordinary_pass(tmp_path), 1)
    assert demoted.ending == clean.ending
    assert demoted.primary_reason == clean.primary_reason


def test_page_disposition_is_the_only_public_surface_no_provenance_leaks() -> None:
    import inspect

    sig = inspect.signature(page_disposition)
    ret = sig.return_annotation
    assert ret in ("PageDisposition", PageDisposition), (
        f"page_disposition must return PageDisposition only, got annotation {ret!r}"
    )


def test_finalized_page_records_builds_one_record_per_page_in_one_pass(tmp_path) -> None:
    """t4: a record's output and disposition must be impossible to obtain
    from different selector/guard passes -- so asking for the disposition of
    the SAME page via two different entry points must always agree.
    """
    state = _state_with_ordinary_pass(tmp_path)
    records = finalized_page_records(state)
    assert len(records) == state.handle.page_count

    rec = next(r for r in records if r.output.page_num == 1 or True)
    standalone = page_disposition(state, 1)
    assert rec.disposition == standalone


def test_finalized_page_records_saved_body_snapshot_can_diverge_from_pre_transform(
    tmp_path,
) -> None:
    """A pre-transform snapshot and a post-transform (``saved_body``) snapshot
    may legitimately disagree when the saved body itself trips the emission
    guard -- t8's "final-body table validation replaces a post-transform page"
    regression case, pinned here at the record layer.
    """
    state = _state_with_ordinary_pass(tmp_path)
    pre = finalized_page_records(state)[0]
    assert pre.disposition.ending is PageEnding.MODEL_OUTPUT

    bad_body = "## Page 1\n\n| a | b |\n| --- |\n| 1 | 2 |\n"
    post = finalized_page_records(state, saved_body=bad_body)[0]
    assert post.disposition.ending is PageEnding.FAIL_CLOSED_MARKER
    assert post.disposition.primary_reason is PagePrimaryReason.INVALID_TABLE_EMISSION


# ---------------------------------------------------------------------------
# Focused acceptance criteria tests
# ---------------------------------------------------------------------------


def test_two_structure_grid_rows_normalize_to_one_primary_reason() -> None:
    """Both structure-grid rows normalize to STRUCTURE_CLASS primary reason."""
    from socr.core.manifest import SelectionProvenance, provenance_to_disposition

    d_passing = provenance_to_disposition(SelectionProvenance.STRUCTURE_CLASS_GRID_PASSING)
    d_flagged = provenance_to_disposition(SelectionProvenance.STRUCTURE_CLASS_GRID_FLAGGED)

    assert d_passing.ending is PageEnding.MODEL_OUTPUT
    assert d_flagged.ending is PageEnding.MODEL_OUTPUT
    assert d_passing.primary_reason is PagePrimaryReason.STRUCTURE_CLASS
    assert d_flagged.primary_reason is PagePrimaryReason.STRUCTURE_CLASS
    assert d_passing == d_flagged


def test_d3_model_versus_d3_floor_differ_by_ending() -> None:
    """D3 model kept vs D3 floor share NATIVE_TABLE_UNVERIFIABLE reason, differ by ending."""
    from socr.core.manifest import SelectionProvenance, provenance_to_disposition

    d_model = provenance_to_disposition(SelectionProvenance.UNVERIFIABLE_TABLE_MODEL_KEPT)
    d_floor = provenance_to_disposition(SelectionProvenance.UNVERIFIABLE_TABLE_NATIVE)

    assert d_model.primary_reason is PagePrimaryReason.NATIVE_TABLE_UNVERIFIABLE
    assert d_floor.primary_reason is PagePrimaryReason.NATIVE_TABLE_UNVERIFIABLE
    assert d_model.ending is PageEnding.MODEL_OUTPUT
    assert d_floor.ending is PageEnding.FAIL_CLOSED_MARKER
    assert d_model != d_floor


def test_ordinary_passing_model_disposition() -> None:
    from socr.core.manifest import SelectionProvenance, provenance_to_disposition

    d = provenance_to_disposition(SelectionProvenance.PASSING_BEST_OUTPUT)
    assert d.ending is PageEnding.MODEL_OUTPUT
    assert d.primary_reason is PagePrimaryReason.ACCEPTED_OUTPUT


def test_clean_native_disposition() -> None:
    from socr.core.manifest import SelectionProvenance, provenance_to_disposition

    d = provenance_to_disposition(SelectionProvenance.NATIVE_CLEAN)
    assert d.ending is PageEnding.NATIVE_PROSE
    assert d.primary_reason is PagePrimaryReason.CLEAN_NATIVE_PROSE


def test_demoted_native_disposition() -> None:
    from socr.core.manifest import SelectionProvenance, provenance_to_disposition

    d = provenance_to_disposition(SelectionProvenance.NATIVE_FALLBACK)
    assert d.ending is PageEnding.DEMOTED_NATIVE
    assert d.primary_reason is PagePrimaryReason.DEMOTED_NATIVE_RECOVERY_EXHAUSTION


def test_already_present_emission_marker(tmp_path) -> None:
    """An already-present failure marker is preserved with FAIL_CLOSED_MARKER / INVALID_TABLE_EMISSION."""
    state = DocumentState(handle=DocumentHandle.from_path(_pdf(tmp_path)))
    p = state.pages[1]
    p.is_born_digital = False
    marker_output = PageOutput(
        page_num=1,
        text="[page 1 failed: invalid table emission — width mismatch]",
        status=PageStatus.ERROR,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.TABLE_EMISSION_INVALID,
    )
    p.attempts.append(marker_output)
    p.best_output = marker_output

    rec = finalized_page_records(state)[0]
    assert rec.disposition.ending is PageEnding.FAIL_CLOSED_MARKER
    assert rec.disposition.primary_reason is PagePrimaryReason.INVALID_TABLE_EMISSION
    assert page_disposition(state, 1) == rec.disposition


def test_content_only_emission_demotion_preserves_selected_disposition(tmp_path) -> None:
    """A content defect demotes status to ERROR without replacing bytes; disposition is unchanged."""
    state = DocumentState(handle=DocumentHandle.from_path(_pdf(tmp_path)))
    p = state.pages[1]
    p.is_born_digital = True
    p.native_text = "native prose"
    # All blank dashes in table body -> table_content_defect fires but does not replace text
    empty_table_text = "| Header 1 | Header 2 |\n| --- | --- |\n| - | - |"
    best = PageOutput(
        page_num=1,
        text=empty_table_text,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    p.attempts.append(best)
    p.best_output = best

    rec = finalized_page_records(state)[0]
    assert rec.output.status is PageStatus.ERROR
    assert rec.output.failure_mode is FailureMode.TABLE_EMISSION_INVALID
    assert rec.output.text == empty_table_text
    # Disposition is NOT converted to INVALID_TABLE_EMISSION marker
    assert rec.disposition.ending is PageEnding.MODEL_OUTPUT
    assert rec.disposition.primary_reason is PagePrimaryReason.ACCEPTED_OUTPUT
    assert page_disposition(state, 1) == rec.disposition


@pytest.mark.parametrize(
    "terminal",
    [FailureMode.TABLE_REJECTED, FailureMode.TABLE_UNVERIFIED],
)
def test_each_ladder_terminal_preserves_selected_disposition(tmp_path, terminal) -> None:
    """Each ladder terminal guard demotes status without changing the public disposition."""
    state = DocumentState(handle=DocumentHandle.from_path(_pdf(tmp_path)))
    p = state.pages[1]
    p.is_born_digital = True
    p.native_text = "native prose"
    best = PageOutput(
        page_num=1,
        text="clean model prose with a table judged by ladder.",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    p.attempts.append(best)
    p.best_output = best
    p.table_ladder_disposition = terminal

    rec = finalized_page_records(state)[0]
    assert rec.output.status is PageStatus.WARNING
    assert rec.output.failure_mode is terminal
    assert rec.disposition.ending is PageEnding.MODEL_OUTPUT
    assert rec.disposition.primary_reason is PagePrimaryReason.ACCEPTED_OUTPUT
    assert page_disposition(state, 1) == rec.disposition


def test_assert_only_marker_replacement_changes_public_ending_and_reason(tmp_path) -> None:
    """Comprehensive assertion: among all guard interactions, ONLY a text-replacing
    emission marker overrides the public disposition.
    """
    from socr.core.manifest import provenance_to_disposition

    # 1. Clean pass -> unchanged
    st_clean = _state_with_ordinary_pass(tmp_path)
    rec_clean = finalized_page_records(st_clean)[0]
    assert rec_clean.disposition == provenance_to_disposition(rec_clean.selection_provenance)

    # 2. Content-only demotion -> unchanged
    st_content = DocumentState(handle=DocumentHandle.from_path(_pdf(tmp_path)))
    st_content.pages[1].is_born_digital = True
    st_content.pages[1].native_text = "prose"
    out_content = PageOutput(
        page_num=1,
        text="| H1 | H2 |\n| --- | --- |\n| - | - |",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    st_content.pages[1].attempts.append(out_content)
    st_content.pages[1].best_output = out_content
    rec_content = finalized_page_records(st_content)[0]
    assert rec_content.disposition == provenance_to_disposition(rec_content.selection_provenance)

    # 3. Ladder demotion -> unchanged
    st_ladder = _state_with_ladder_rejection(tmp_path)
    rec_ladder = finalized_page_records(st_ladder)[0]
    assert rec_ladder.disposition == provenance_to_disposition(rec_ladder.selection_provenance)

    # 4. Marker replacement -> OVERRIDDEN
    st_marker = _state_with_marker_emission(tmp_path)
    rec_marker = finalized_page_records(st_marker)[0]
    assert rec_marker.selection_provenance == SelectionProvenance.PASSING_BEST_OUTPUT
    assert rec_marker.disposition != provenance_to_disposition(rec_marker.selection_provenance)
    assert rec_marker.disposition.ending is PageEnding.FAIL_CLOSED_MARKER
    assert rec_marker.disposition.primary_reason is PagePrimaryReason.INVALID_TABLE_EMISSION


def test_final_body_table_validation_replaces_post_transform_page_all_surfaces_agree(
    tmp_path,
) -> None:
    """Regression test (t8): When final-body table validation replaces a post-transform
    page with an invalid table emission failure marker, its final disposition, sidecar,
    manifest entry/blob, fragment, and stitched .md all agree on the replaced failure
    marker and (FAIL_CLOSED_MARKER, INVALID_TABLE_EMISSION) disposition.
    """
    import json
    from unittest.mock import patch

    from ocr_output_contract import split_native_pages

    from socr.core.cache import BlobStore
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.manifest import Manifest
    from socr.core.result import DocumentStatus
    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf_path = _pdf(tmp_path)
    state = _state_with_ordinary_pass(tmp_path)
    out_dir = tmp_path / "out"

    pipeline = UnifiedPipeline(
        PipelineConfig(
            save_figures=True,
            write_manifest=True,
            quiet=True,
            enabled_engines=[EngineType.GEMINI],
            primary_engine=EngineType.GEMINI,
            # docs/log/2026-09-03_p1-prep-latch-and-audit.md (cold review round 1)
            table_judge_ladder=False,
        )
    )
    pipeline._scan_root = pdf_path.parent

    # Simulate post-transform (e.g. figure phase / captioning) introducing an invalid table emission
    invalid_table_body = "## Page 1\n\n| a | b |\n| --- |\n| 1 | 2 |\n"

    with patch.object(
        pipeline,
        "_describe_and_embed_figures",
        return_value=invalid_table_body,
    ):
        result = pipeline._phase_assemble(state, out_dir)

    # 1. Output Markdown file (.md)
    md_files = [p for p in out_dir.rglob("*.md") if "pages" not in p.parts]
    assert len(md_files) == 1
    md_content = md_files[0].read_text(encoding="utf-8")
    md_pages = split_native_pages(md_content)
    assert len(md_pages) == 1
    shipped_page_text = md_pages[0]
    assert shipped_page_text.startswith(
        "[page 1 failed: invalid table emission — table_width_mismatch]"
    )

    # 2. Fragment (pages/00001.md)
    frag_path = next(out_dir.rglob("pages/00001.md"))
    frag_text = frag_path.read_text(encoding="utf-8")
    assert frag_text == shipped_page_text

    # 3. Sidecar (pages/00001.json)
    sidecar_path = next(out_dir.rglob("pages/00001.json"))
    sidecar_meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar_meta["winning_output"]["text"] == shipped_page_text
    assert sidecar_meta["status"] == "error"
    assert sidecar_meta["failure_mode"] == "table_emission_invalid"
    assert sidecar_meta["disposition"] == {
        "ending": "fail_closed_marker",
        "primary_reason": "invalid_table_emission",
    }

    # 4. Manifest (manifest.json) + Blob Store
    manifest_path = next(out_dir.rglob("manifest.json"))
    manifest = Manifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    entry = manifest.entries[1]
    assert entry.disposition == PageDisposition(
        ending=PageEnding.FAIL_CLOSED_MARKER,
        primary_reason=PagePrimaryReason.INVALID_TABLE_EMISSION,
    )
    blob_store = BlobStore(manifest_path.parent / "cache")
    blob_output = blob_store.get_page(entry.blob_ref)
    assert blob_output is not None
    assert blob_output.text == shipped_page_text
    assert blob_output.status is PageStatus.ERROR
    assert blob_output.failure_mode is FailureMode.TABLE_EMISSION_INVALID

    # 5. Result object
    assert result.status is DocumentStatus.AUDIT_FAILED
