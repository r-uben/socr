"""P6 Stage A: persisting `PageDisposition` in the sidecar and manifest must
not change resume behaviour, and old data must still load.

Design: §3 of ``docs/log/2026-09-02_p6-selector-collapse-design.md`` ("Writing
the tag into the sidecar would be new provenance... it does not affect
``run_fingerprint``... so old sidecars stay valid") and the GH-525 precedent
named directly by the task spec ("a field that changes nothing must not
invalidate the run"). Plan tasks t6-t7.

Hermetic: drives ``_flush_page_sidecar`` / ``_load_terminal_page`` /
``build_manifest`` directly, exactly like
``test_gh161_resume_ledger_audit_gate.py`` -- no provider ladder, no
``_phase_agentic``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.cache import BlobStore  # noqa: E402
from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    ManifestEntry,
    PageDisposition,
    PageEnding,
    PagePrimaryReason,
    build_manifest,
    page_disposition,
)
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402


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


def _real_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "page 1 text " * 10)
    doc.save(str(path))
    doc.close()
    return path


def _clean_state(pdf_path: Path) -> DocumentState:
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.native_text = ""
    accepted = PageOutput(
        page_num=1,
        text="accepted body text long enough to look like real content here.",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    ps.attempts.append(accepted)
    ps.best_output = accepted
    return state


# ---------------------------------------------------------------------------
# ManifestEntry: additive field, old data still loads
# ---------------------------------------------------------------------------


def test_manifest_entry_round_trips_disposition() -> None:
    entry = ManifestEntry(
        page_num=1,
        blob_ref="deadbeef",
        fingerprint=__import__("socr.core.manifest", fromlist=["PageFingerprint"]).PageFingerprint(
            pdf_file_hash="h", page_num=1, render_dpi=200, engine="native"
        ),
        disposition=PageDisposition(
            ending=PageEnding.NATIVE_PROSE, primary_reason=PagePrimaryReason.CLEAN_NATIVE_PROSE
        ),
    )
    payload = entry.to_dict()
    assert payload["disposition"] == {
        "ending": "native_prose",
        "primary_reason": "clean_native_prose",
    }
    assert ManifestEntry.from_dict(payload) == entry


def test_an_old_manifest_entry_without_disposition_loads_as_none() -> None:
    from socr.core.manifest import PageFingerprint

    old_payload = {
        "page_num": 1,
        "blob_ref": "deadbeef",
        "fingerprint": dataclasses.asdict(
            PageFingerprint(pdf_file_hash="h", page_num=1, render_dpi=200, engine="native")
        ),
        "journal": [],
    }
    entry = ManifestEntry.from_dict(old_payload)
    assert entry.disposition is None


def test_the_only_new_key_between_old_and_new_entries_is_disposition() -> None:
    """The additive-key exception the decision log must state explicitly:
    parsed equality must hold everywhere ELSE once ``disposition`` is removed.
    """
    from socr.core.manifest import PageFingerprint

    fp = PageFingerprint(pdf_file_hash="h", page_num=1, render_dpi=200, engine="native")
    old = ManifestEntry(page_num=1, blob_ref="deadbeef", fingerprint=fp)
    new = ManifestEntry(
        page_num=1,
        blob_ref="deadbeef",
        fingerprint=fp,
        disposition=PageDisposition(
            ending=PageEnding.NATIVE_PROSE, primary_reason=PagePrimaryReason.CLEAN_NATIVE_PROSE
        ),
    )
    old_payload = old.to_dict()
    new_payload = new.to_dict()
    assert set(new_payload.keys()) - set(old_payload.keys()) == {"disposition"}
    new_payload_without_disposition = {k: v for k, v in new_payload.items() if k != "disposition"}
    assert new_payload_without_disposition == old_payload


def test_build_manifest_writes_the_disposition_the_page_actually_has(tmp_path: Path) -> None:
    pdf_path = _real_pdf(tmp_path)
    state = _clean_state(pdf_path)
    blobs = BlobStore(tmp_path / "blobs")

    manifest = build_manifest(state, blobs)
    entry = manifest.entries[1]
    assert entry.disposition == page_disposition(state, 1)


# ---------------------------------------------------------------------------
# Sidecar: additive field, old sidecars still load, resume unaffected
# ---------------------------------------------------------------------------


def test_sidecar_carries_top_level_disposition_matching_page_disposition(tmp_path: Path) -> None:
    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent

    state = _clean_state(pdf_path)
    pipeline._flush_page_fragment(state, 1, state.pages[1].best_output.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

    sidecar = next(out_dir.rglob("pages/00001.json"))
    meta = json.loads(sidecar.read_text())

    assert "disposition" in meta, "the sidecar must carry the page's disposition at top level"
    got = PageDisposition.from_dict(meta["disposition"])
    assert got == page_disposition(state, 1)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda meta: meta, id="unmodified"),
        pytest.param(lambda meta: {**meta, "disposition": None}, id="disposition_explicitly_null"),
        pytest.param(
            lambda meta: {k: v for k, v in meta.items() if k != "disposition"},
            id="disposition_key_absent_old_sidecar",
        ),
    ],
)
def test_resume_decision_is_identical_regardless_of_the_disposition_field(
    tmp_path: Path, mutate
) -> None:
    """GH-525's precedent, one field over: a field that carries no invalidation
    signal must not change ``_load_terminal_page``'s verdict. This is a
    DIFFERENCE test, not a pinned tuple (see the repo's no-provider trap note
    in CLAUDE.md) -- both the unmodified and the field-stripped sidecar must
    agree with each other, whatever that shared verdict is.
    """
    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent

    state = _clean_state(pdf_path)
    pipeline._flush_page_fragment(state, 1, state.pages[1].best_output.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

    sidecar_path = next(out_dir.rglob("pages/00001.json"))
    baseline_meta = json.loads(sidecar_path.read_text())
    mutated_meta = mutate(baseline_meta)
    sidecar_path.write_text(json.dumps(mutated_meta))

    fresh_state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    resumed = pipeline._load_terminal_page(fresh_state, 1, out_dir)

    reference_state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    sidecar_path.write_text(json.dumps(baseline_meta))
    reference = pipeline._load_terminal_page(reference_state, 1, out_dir)

    assert (resumed is None) == (reference is None), (
        f"the disposition field's presence/shape changed the resume verdict: "
        f"mutated={resumed!r} baseline={reference!r}"
    )
    if resumed is not None and reference is not None:
        assert resumed.text == reference.text
        assert resumed.status == reference.status
        assert resumed.audit_passed == reference.audit_passed


def test_a_repeated_unchanged_sidecar_write_is_byte_identical(tmp_path: Path) -> None:
    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent
    state = _clean_state(pdf_path)

    pipeline._flush_page_fragment(state, 1, state.pages[1].best_output.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
    first = next(out_dir.rglob("pages/00001.json")).read_bytes()

    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
    second = next(out_dir.rglob("pages/00001.json")).read_bytes()

    assert first == second


def test_sidecar_only_additive_key_is_disposition(tmp_path: Path) -> None:
    """Old vs new parsed sidecar comparison has exactly one additive key: disposition."""
    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent
    state = _clean_state(pdf_path)

    pipeline._flush_page_fragment(state, 1, state.pages[1].best_output.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
    sidecar_path = next(out_dir.rglob("pages/00001.json"))
    meta = json.loads(sidecar_path.read_text())

    assert "disposition" in meta
    without_disp = {k: v for k, v in meta.items() if k != "disposition"}

    # Canonical pre-disposition sidecar key set
    expected_pre_keys = {
        "page_num",
        "status",
        "failure_mode",
        "audit_passed",
        "terminal",
        "engine",
        "provider",
        "cost_usd",
        "page_cost_usd",
        "winning_output",
        "run_fingerprint",
        "socr_version",
        "socr_source_digest",
        "input_checksum",
        "page_fingerprint",
        "needs_ocr_enhancement",
        "native_table_structure_failed",
        "native_rotated_text_shredded",
        "rotated_shred_png_ref",
        "native_table_structure_defective",
        "native_table_emission_defect",
        "native_table_content_defect",
        "native_table_header_unattributed",
        "native_table_unverifiable",
        "native_table_unverifiable_ordinals",
        "native_table_region_count",
        "native_table_region_identities",
        "d3_floor_png_ref",
        "scanned_table_evidence_failed",
        "chart_asset_render_failed",
        "equation_lane_retry_pending",
        "chart_asset_detection_failed",
        "judge_rejected",
        "structure_class_model_kept",
        "table_ladder_disposition",
        "table_ladder_incomplete",
        "binding_adjudication",
        "audit_events",
        "figure_refs",
    }
    assert set(without_disp.keys()) == expected_pre_keys
    assert set(meta.keys()) - expected_pre_keys == {"disposition"}


def test_flush_page_sidecar_accepts_record_and_never_serializes_provenance(tmp_path: Path) -> None:
    """_flush_page_sidecar accepts a finalized-page record directly and writes its output
    and public disposition without serializing selection provenance."""
    from socr.core.manifest import FinalizedPageRecord, SelectionProvenance

    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent
    state = _clean_state(pdf_path)

    custom_output = PageOutput(
        page_num=1,
        text="custom record body",
        status=PageStatus.SUCCESS,
        engine="test-engine",
        audit_passed=True,
    )
    custom_disp = PageDisposition(
        ending=PageEnding.MODEL_OUTPUT,
        primary_reason=PagePrimaryReason.ACCEPTED_OUTPUT,
    )
    rec = FinalizedPageRecord(
        output=custom_output,
        disposition=custom_disp,
        selection_provenance=SelectionProvenance.PASSING_BEST_OUTPUT,
    )

    pipeline._flush_page_fragment(state, 1, custom_output.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True, record=rec)

    sidecar_path = next(out_dir.rglob("pages/00001.json"))
    raw_text = sidecar_path.read_text()
    meta = json.loads(raw_text)

    assert meta["winning_output"]["text"] == "custom record body"
    assert meta["disposition"] == {"ending": "model_output", "primary_reason": "accepted_output"}
    # SelectionProvenance must never be serialized
    assert "selection_provenance" not in meta
    assert "PASSING_BEST_OUTPUT" not in raw_text
    assert "selection_provenance" not in meta["winning_output"]
    assert "selection_provenance" not in meta["disposition"]


def test_page_blob_key_receives_only_winning_output(tmp_path: Path) -> None:
    """_page_blob_key receives only winning_output; adding disposition does not enter it."""
    from socr.pipeline.orchestrator import _page_blob_key

    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent
    state = _clean_state(pdf_path)

    pipeline._flush_page_fragment(state, 1, state.pages[1].best_output.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

    sidecar_path = next(out_dir.rglob("pages/00001.json"))
    meta = json.loads(sidecar_path.read_text())

    winning = meta["winning_output"]
    expected_fp = _page_blob_key(winning)
    assert meta["page_fingerprint"] == expected_fp

    # Adding or mutating top-level disposition cannot affect _page_blob_key
    meta["disposition"]["primary_reason"] = "something_else"
    assert _page_blob_key(winning) == expected_fp


# ---------------------------------------------------------------------------
# Comprehensive Resume Matrix: Difference Testing Over All Shapes & Providers
# ---------------------------------------------------------------------------


def _setup_sidecar_shape(
    tmp_path: Path,
    shape: str,
    suffix: str = "",
) -> tuple[UnifiedPipeline, DocumentState, Path, Path]:
    from socr.core.result import FailureMode

    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / f"out_{shape}_{suffix}"
    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent

    state = _clean_state(pdf_path)
    ps = state.pages[1]

    terminal = True
    body_text = "body text content for testing resume shapes."

    if shape == "clean_success":
        ps.best_output = PageOutput(
            page_num=1,
            text=body_text,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
    elif shape == "audit_rejected":
        ps.best_output = PageOutput(
            page_num=1,
            text=body_text,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=False,
        )
    elif shape == "error_status":
        ps.best_output = PageOutput(
            page_num=1,
            text=body_text,
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
        )
    elif shape == "warning_status":
        ps.best_output = PageOutput(
            page_num=1,
            text=body_text,
            status=PageStatus.WARNING,
            engine="native",
            audit_passed=False,
        )
    elif shape == "failure_marker":
        body_text = "[page 1 failed: timeout during extraction]"
        ps.best_output = PageOutput(
            page_num=1,
            text=body_text,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=False,
        )
    elif shape == "ladder_rejected_d1b":
        ps.table_ladder_disposition = FailureMode.TABLE_REJECTED
        ps.table_ladder_incomplete = False
        ps.best_output = PageOutput(
            page_num=1,
            text=body_text,
            status=PageStatus.WARNING,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.TABLE_REJECTED,
        )
    elif shape == "ladder_unverified":
        ps.table_ladder_disposition = FailureMode.TABLE_UNVERIFIED
        ps.table_ladder_incomplete = False
        ps.best_output = PageOutput(
            page_num=1,
            text=body_text,
            status=PageStatus.WARNING,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.TABLE_UNVERIFIED,
        )
    elif shape == "ladder_incomplete":
        ps.table_ladder_disposition = FailureMode.TABLE_REJECTED
        ps.table_ladder_incomplete = True
        ps.best_output = PageOutput(
            page_num=1,
            text=body_text,
            status=PageStatus.WARNING,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.TABLE_REJECTED,
        )
    elif shape == "structure_floor":
        ps.best_output = PageOutput(
            page_num=1,
            text=body_text,
            status=PageStatus.WARNING,
            engine="native",
            audit_passed=False,
            failure_mode=FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED,
        )
    elif shape == "equation_retry_pending":
        ps.equation_lane_retry_pending = True
        ps.best_output = PageOutput(
            page_num=1,
            text=body_text,
            status=PageStatus.SUCCESS,
            engine="native",
            audit_passed=True,
        )
    elif shape == "provisional":
        terminal = False
        ps.best_output = PageOutput(
            page_num=1,
            text=body_text,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
    else:
        raise ValueError(f"Unknown shape: {shape}")

    pipeline._flush_page_fragment(state, 1, body_text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=terminal)

    sidecar_path = next(out_dir.rglob("pages/00001.json"))
    return pipeline, state, pdf_path, sidecar_path


@pytest.mark.parametrize(
    "shape",
    [
        "clean_success",
        "audit_rejected",
        "error_status",
        "warning_status",
        "failure_marker",
        "ladder_rejected_d1b",
        "ladder_unverified",
        "ladder_incomplete",
        "structure_floor",
        "equation_retry_pending",
        "provisional",
    ],
)
@pytest.mark.parametrize("provider_state", ["providerless", "with_provider"])
def test_resume_difference_matrix_across_all_shapes_and_providers(
    tmp_path: Path,
    shape: str,
    provider_state: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive _load_terminal_page over every existing resume-sidecar shape plus
    the same sidecar with disposition removed or explicitly null.
    Assert identical return-versus-None decisions and identical reconstructed
    PageOutput values under both providerless and active provider states.
    """
    if provider_state == "providerless":
        monkeypatch.setattr(UnifiedPipeline, "_available_engines_for_agentic", lambda self: [])
        monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")
        monkeypatch.setattr(
            UnifiedPipeline,
            "_equation_lane_provider",
            lambda self: (None, "no provider"),
        )
    else:
        from socr.core.providers import PROFILE_GEMINI

        monkeypatch.setattr(
            UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_GEMINI]
        )
        monkeypatch.setattr(
            UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "mock-model"
        )
        monkeypatch.setattr(
            UnifiedPipeline,
            "_equation_lane_provider",
            lambda self: (PROFILE_GEMINI, "test"),
        )

    pipeline, state, pdf_path, sidecar_path = _setup_sidecar_shape(
        tmp_path, shape, suffix=provider_state
    )
    out_dir = sidecar_path.parent.parent

    baseline_meta = json.loads(sidecar_path.read_text())

    # 1. Baseline with disposition
    state_baseline = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    res_baseline = pipeline._load_terminal_page(state_baseline, 1, out_dir)

    # 2. Mutated: disposition explicitly null
    sidecar_path.write_text(json.dumps({**baseline_meta, "disposition": None}))
    state_null = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    res_null = pipeline._load_terminal_page(state_null, 1, out_dir)

    # 3. Mutated: disposition key absent (old sidecar shape)
    without_disp = {k: v for k, v in baseline_meta.items() if k != "disposition"}
    sidecar_path.write_text(json.dumps(without_disp))
    state_absent = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    res_absent = pipeline._load_terminal_page(state_absent, 1, out_dir)

    # Restore baseline
    sidecar_path.write_text(json.dumps(baseline_meta))

    # Assert invariant: return-versus-None decisions must be identical
    assert (res_null is None) == (res_baseline is None), (
        f"Shape {shape} in {provider_state}: null disposition changed resume decision "
        f"(null={res_null!r} vs baseline={res_baseline!r})"
    )
    assert (res_absent is None) == (res_baseline is None), (
        f"Shape {shape} in {provider_state}: absent disposition changed resume decision "
        f"(absent={res_absent!r} vs baseline={res_baseline!r})"
    )

    if res_baseline is not None:
        assert res_null is not None and res_absent is not None
        assert res_null.text == res_baseline.text == res_absent.text
        assert res_null.status == res_baseline.status == res_absent.status
        assert res_null.audit_passed == res_baseline.audit_passed == res_absent.audit_passed
        assert res_null.failure_mode == res_baseline.failure_mode == res_absent.failure_mode
        assert res_null.engine == res_baseline.engine == res_absent.engine
