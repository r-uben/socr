"""TICKET-B3: Agentic manifest enrichment tests.

Verifies that:
1. Journal entries include provider_id, model, backend, cost_usd, accepted,
   confidence, judge_model fields.
2. A ladder snapshot is recorded when agentic mode ran.
3. Replay still works (blob-based, 0 model calls) after the enrichment.
4. Budget-skipped rungs produce stub journal entries (skip_reason populated).
5. judge_model flows from state into the manifest and each journal entry.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")  # PyMuPDF; skip whole module if unavailable

from socr.core.cache import BlobStore  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import Manifest, build_manifest, replay  # noqa: E402
from socr.core.result import (  # noqa: E402
    DocumentStatus,
    EngineResult,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pdf(path, n_pages: int = 2):
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {i + 1} content")
    doc.save(str(path))
    doc.close()
    return path


def _agentic_state(
    pdf_path,
    n_pages: int = 2,
    judge_model: str = "qwen3.5:cloud",
) -> DocumentState:
    """A DocumentState that simulates an agentic-mode run.

    Each page has one accepted attempt carrying provider_id/model/backend/confidence
    (as set by _phase_agentic after B3). judge_model is recorded on state.
    """
    handle = DocumentHandle.from_path(pdf_path)
    state = DocumentState(handle=handle)

    for i in range(1, n_pages + 1):
        po = PageOutput(
            page_num=i,
            text=f"OCR text page {i}",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
            cost_usd=0.0,
            confidence=0.95,
            provider_id="qwen-local-instruct",
            provider_model="qwen3-vl:30b-a3b-instruct",
            provider_backend="ollama",
        )
        state.pages[i].attempts.append(po)
        state.pages[i].best_output = po

    # Simulate a ladder snapshot as _phase_agentic would set it.
    state.agentic_ladder = [
        {
            "provider_id": "qwen-local-instruct",
            "model": "qwen3-vl:30b-a3b-instruct",
            "backend": "ollama",
            "cost_per_page_usd": 0.0,
            "tier": "local",
        },
        {
            "provider_id": "gemini",
            "model": "gemini-3-flash-preview",
            "backend": "gemini-api",
            "cost_per_page_usd": 0.0002,
            "tier": "cloud",
        },
    ]
    state.agentic_judge_model = judge_model

    state.engine_runs.append(
        EngineResult(
            document_path=pdf_path,
            engine="qwen",
            status=DocumentStatus.SUCCESS,
            cost=0.0,
        )
    )
    return state


# ---------------------------------------------------------------------------
# Test 1: journal records provider_id, model, backend, confidence, judge_model
# ---------------------------------------------------------------------------


def test_journal_records_provider_id_model_backend(tmp_path):
    """Journal entries for agentic-mode pages must carry provider identity fields."""
    pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
    state = _agentic_state(pdf, n_pages=2, judge_model="qwen3.5:cloud")
    store = BlobStore(tmp_path / "cache")

    manifest = build_manifest(state, store, dpi=120)

    for page_num in (1, 2):
        journal = manifest.entries[page_num].journal
        assert len(journal) == 1, f"expected 1 journal entry for page {page_num}"
        entry = journal[0]
        assert entry["provider_id"] == "qwen-local-instruct"
        assert entry["model"] == "qwen3-vl:30b-a3b-instruct"
        assert entry["backend"] == "ollama"
        assert entry["cost_usd"] == 0.0
        assert entry["accepted"] is True
        assert "failure_mode" in entry
        assert "reason" in entry
        # Reviewer finding #2: confidence and judge_model must be in journal
        assert entry["confidence"] == pytest.approx(0.95)
        assert entry["judge_model"] == "qwen3.5:cloud"


def test_journal_confidence_zero_when_heuristic_judge(tmp_path):
    """When the heuristic judge is used (no VLM), confidence stays 0.0 and judge_model is ''."""
    pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
    state = _agentic_state(pdf, n_pages=1, judge_model="")
    # Overwrite confidence to 0 (heuristic judge doesn't set it)
    state.pages[1].attempts[0].confidence = 0.0

    store = BlobStore(tmp_path / "cache")
    manifest = build_manifest(state, store, dpi=120)

    entry = manifest.entries[1].journal[0]
    assert entry["confidence"] == 0.0
    assert entry["judge_model"] == ""


# ---------------------------------------------------------------------------
# Test 2: ladder snapshot appears in manifest.to_dict() when agentic mode ran
# ---------------------------------------------------------------------------


def test_ladder_snapshot_in_manifest(tmp_path):
    """When state.agentic_ladder is populated, manifest.to_dict() includes it."""
    pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
    state = _agentic_state(pdf, n_pages=1)
    store = BlobStore(tmp_path / "cache")

    manifest = build_manifest(state, store, dpi=120)

    d = manifest.to_dict()
    assert "agentic_ladder" in d, "ladder snapshot missing from manifest dict"
    ladder = d["agentic_ladder"]
    assert isinstance(ladder, list)
    assert len(ladder) == 2  # local + cloud rungs

    local_rung = ladder[0]
    assert local_rung["provider_id"] == "qwen-local-instruct"
    assert local_rung["model"] == "qwen3-vl:30b-a3b-instruct"
    assert local_rung["backend"] == "ollama"
    assert local_rung["cost_per_page_usd"] == 0.0
    assert local_rung["tier"] == "local"

    cloud_rung = ladder[1]
    assert cloud_rung["provider_id"] == "gemini"
    assert cloud_rung["tier"] == "cloud"


def test_judge_model_in_manifest_dict(tmp_path):
    """When a VLM judge is used, agentic_judge_model appears in manifest.to_dict()."""
    pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
    state = _agentic_state(pdf, n_pages=1, judge_model="qwen3.5:cloud")
    store = BlobStore(tmp_path / "cache")

    manifest = build_manifest(state, store, dpi=120)
    d = manifest.to_dict()

    assert d.get("agentic_judge_model") == "qwen3.5:cloud"


def test_no_ladder_snapshot_when_not_agentic(tmp_path):
    """When agentic mode was not used, agentic_ladder must be absent from to_dict()."""
    pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
    handle = DocumentHandle.from_path(pdf)
    state = DocumentState(handle=handle)

    po = PageOutput(
        page_num=1,
        text="native text",
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=True,
    )
    state.pages[1].attempts.append(po)
    state.pages[1].best_output = po
    state.engine_runs.append(
        EngineResult(
            document_path=pdf,
            engine="gemini",
            status=DocumentStatus.SUCCESS,
        )
    )
    # agentic_ladder is empty list — should not appear in to_dict()
    assert state.agentic_ladder == []

    store = BlobStore(tmp_path / "cache")
    manifest = build_manifest(state, store, dpi=120)

    d = manifest.to_dict()
    assert "agentic_ladder" not in d
    assert "agentic_judge_model" not in d


def test_ladder_snapshot_roundtrips_via_json(tmp_path):
    """Manifest with ladder snapshot and judge_model survives save/load round-trip."""
    pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
    state = _agentic_state(pdf, n_pages=1, judge_model="qwen3.5:cloud")
    store = BlobStore(tmp_path / "cache")

    manifest = build_manifest(state, store, dpi=120)
    manifest.save(tmp_path / "manifest.json")

    reloaded = Manifest.load(tmp_path / "manifest.json")
    assert reloaded.agentic_ladder is not None
    assert len(reloaded.agentic_ladder) == 2
    assert reloaded.agentic_ladder[0]["provider_id"] == "qwen-local-instruct"
    assert reloaded.agentic_judge_model == "qwen3.5:cloud"


# ---------------------------------------------------------------------------
# Test 3: replay still works (0 model calls) after enrichment
# ---------------------------------------------------------------------------


def test_replay_zero_model_calls(tmp_path):
    """Enriched manifest must not break blob-based replay.

    After B3 enrichment replay is still blob-only: no engine invoked, just
    fetching cached page blobs by their content hash.
    """
    from ocr_output_contract import assemble_pages

    pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=2)
    state = _agentic_state(pdf, n_pages=2)
    store = BlobStore(tmp_path / "cache")

    manifest = build_manifest(state, store, dpi=120)
    manifest.save(tmp_path / "manifest.json")

    # Cold replay from disk only — simulates `socr replay` on HPC
    reloaded = Manifest.load(tmp_path / "manifest.json")
    cold_store = BlobStore(tmp_path / "cache")
    out = replay(reloaded, cold_store)

    expected = assemble_pages(["OCR text page 1", "OCR text page 2"])
    assert out == expected


# ---------------------------------------------------------------------------
# Test 4: journal with no agentic fields (backward compat via getattr defaults)
# ---------------------------------------------------------------------------


def test_journal_backward_compat_no_provider_fields(tmp_path):
    """Pages whose attempts lack provider_id/model/backend get empty strings in journal.

    This covers non-agentic runs where the PageOutput was never enriched.
    """
    pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
    handle = DocumentHandle.from_path(pdf)
    state = DocumentState(handle=handle)

    # Plain PageOutput — no provider_id/model/backend set (non-agentic)
    po = PageOutput(
        page_num=1,
        text="plain OCR text",
        status=PageStatus.SUCCESS,
        engine="marker",
        audit_passed=True,
        failure_mode=FailureMode.NONE,
    )
    state.pages[1].attempts.append(po)
    state.pages[1].best_output = po
    state.engine_runs.append(
        EngineResult(
            document_path=pdf,
            engine="marker",
            status=DocumentStatus.SUCCESS,
        )
    )

    store = BlobStore(tmp_path / "cache")
    manifest = build_manifest(state, store, dpi=120)

    journal = manifest.entries[1].journal
    assert len(journal) == 1
    entry = journal[0]
    # Provider fields default to empty — no crash, no KeyError
    assert entry["provider_id"] == ""
    assert entry["model"] == ""
    assert entry["backend"] == ""
    assert entry["engine"] == "marker"
    assert entry["accepted"] is True
    assert entry["confidence"] == 0.0
    assert entry["judge_model"] == ""


# ---------------------------------------------------------------------------
# Test 5: budget-skipped rungs produce stub journal entries with skip_reason
# ---------------------------------------------------------------------------


def test_budget_skip_stub_in_journal(tmp_path):
    """A budget-exceeded rung must appear in the journal with skip_reason set.

    _phase_agentic in the orchestrator creates a stub ProviderAttempt when a
    rung is skipped due to budget exhaustion. The orchestrator then copies
    skip_reason to PageOutput.skip_reason, which flows into the journal.
    """
    pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
    handle = DocumentHandle.from_path(pdf)
    state = DocumentState(handle=handle)

    # Simulate what _phase_agentic produces: two attempts for page 1.
    # Attempt 1: local provider accepted (paid the local, free rung).
    accepted_po = PageOutput(
        page_num=1,
        text="OCR text from local",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        cost_usd=0.0,
        confidence=0.9,
        provider_id="qwen-local-instruct",
        provider_model="qwen3-vl:30b-a3b-instruct",
        provider_backend="ollama",
        skip_reason="",
    )
    # Attempt 2: cloud rung skipped due to budget — stub attempt.
    skipped_po = PageOutput(
        page_num=1,
        text="",
        status=PageStatus.ERROR,
        engine="gemini",
        audit_passed=False,
        cost_usd=0.0,
        confidence=0.0,
        provider_id="gemini",
        provider_model="gemini-3-flash-preview",
        provider_backend="gemini-api",
        skip_reason="budget exceeded",
    )
    state.pages[1].attempts.extend([accepted_po, skipped_po])
    state.pages[1].best_output = accepted_po
    state.agentic_ladder = [
        {
            "provider_id": "qwen-local-instruct",
            "model": "q",
            "backend": "ollama",
            "cost_per_page_usd": 0.0,
            "tier": "local",
        },
        {
            "provider_id": "gemini",
            "model": "g",
            "backend": "gemini-api",
            "cost_per_page_usd": 0.0002,
            "tier": "cloud",
        },
    ]
    state.agentic_judge_model = ""
    state.engine_runs.append(
        EngineResult(
            document_path=pdf,
            engine="qwen",
            status=DocumentStatus.SUCCESS,
            cost=0.0,
        )
    )

    store = BlobStore(tmp_path / "cache")
    manifest = build_manifest(state, store, dpi=120)

    journal = manifest.entries[1].journal
    assert len(journal) == 2, "both attempts (accepted + skipped) must appear in journal"

    accepted_entry = journal[0]
    assert accepted_entry["provider_id"] == "qwen-local-instruct"
    assert accepted_entry["accepted"] is True
    assert accepted_entry["reason"] in ("", "none")  # no skip reason for accepted

    skipped_entry = journal[1]
    assert skipped_entry["provider_id"] == "gemini"
    assert skipped_entry["accepted"] is False
    assert skipped_entry["reason"] == "budget exceeded"


# ---------------------------------------------------------------------------
# GH-271: the explicit region hybrid must win only over rejected whole-page text
# ---------------------------------------------------------------------------


def test_corrupt_math_hybrid_does_not_promote_rejected_whole_page_candidate(tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    page = state.pages[1]
    page.is_born_digital = True
    page.native_text = "native prose with corrupt equation"
    rejected = PageOutput(
        page_num=1,
        text="fluent but rejected whole-page candidate",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )
    hybrid = PageOutput(
        page_num=1,
        text="native prose plus crop-backed region candidate",
        status=PageStatus.WARNING,
        engine="native+math",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )
    page.attempts.extend([rejected, hybrid])
    page.best_output = rejected
    page.corrupt_math_hybrid = hybrid

    winner = _winning_page_output(state, 1)

    assert winner.engine == "native+math"
    assert winner.text == hybrid.text
    assert winner.status == PageStatus.WARNING
    assert winner.audit_passed is False
    assert rejected.text not in winner.text


def test_corrupt_math_hybrid_cannot_bypass_hard_table_floor(tmp_path):
    pdf = _make_pdf(tmp_path / "doc.pdf", n_pages=1)
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    page = state.pages[1]
    page.is_born_digital = True
    page.native_text = "collapsed native table"
    page.has_tables = True
    page.native_table_structure_failed = True
    page.native_table_unverifiable = True
    hybrid = PageOutput(
        page_num=1,
        text="native prose plus crop-backed equation",
        status=PageStatus.WARNING,
        engine="native+math",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )
    page.attempts.append(hybrid)
    page.best_output = hybrid
    page.corrupt_math_hybrid = hybrid

    winner = _winning_page_output(state, 1)

    assert winner.status == PageStatus.ERROR
    assert "failed: unverifiable table" in winner.text
    assert hybrid.text not in winner.text


# ---------------------------------------------------------------------------
# GH-200: D3 floor widening -- native_table_header_unattributed must reach
# the same fail-closed floor as native_table_unverifiable (TR-3), because
# TR-3 is blind to header loss by construction (see header_attribution.py).
# ---------------------------------------------------------------------------

from socr.core.manifest import _winning_page_output  # noqa: E402


def _build_state_header_defect(
    page_num: int = 1,
    native_text: str = "collapsed| table |",
    header_unattributed: bool = True,
    png_ref: str = "",
) -> DocumentState:
    """A page whose header-attribution check found a HARD verdict (not TR-3).

    native_table_unverifiable stays False throughout -- TR-3's numeric
    multiset check is blind to header loss by construction, so this is
    exactly the case the D3 floor widening exists for.
    """
    from socr.core.document import DocumentHandle

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DocumentHandle, "__post_init__", lambda self: None)
        handle = DocumentHandle(path=__import__("pathlib").Path("/tmp/fake.pdf"), page_count=1)
    state = DocumentState(handle=handle)
    ps = state.pages[page_num]
    ps.is_born_digital = True
    ps.native_text = native_text
    ps.has_tables = True
    ps.native_table_structure_failed = True
    ps.native_table_unverifiable = False
    ps.native_table_header_unattributed = header_unattributed
    ps.d3_floor_png_ref = png_ref
    ps.attempts.append(
        PageOutput(
            page_num=page_num,
            text="ragged row-major attempt",
            status=PageStatus.WARNING,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.NATIVE_TABLE_STRUCTURE_FAILED,
        )
    )
    return state


def test_d3_floor_fires_on_header_defect() -> None:
    """A header-only defect (TR-3 stays False) must still reach the D3 floor.

    Before the GH-200 widening this fell through to the native_is_fallback
    WARNING branch (manifest.py:340-350) and SHIPPED the header-destroyed
    native table text -- verified fact from the ratified spec.
    """
    state = _build_state_header_defect(header_unattributed=True)
    winner = _winning_page_output(state, 1, None)

    assert winner is not None
    assert "[page 1 failed:" in winner.text
    assert "collapsed| table |" not in winner.text
    assert winner.status == PageStatus.ERROR
    assert winner.audit_passed is False
    assert winner.failure_mode == FailureMode.NATIVE_TABLE_STRUCTURE_FAILED


def test_d3_floor_header_defect_false_keeps_native_fallback_warning() -> None:
    """Paired negative: with the header flag False, the existing
    native_is_fallback WARNING behaviour (manifest.py:340-350) is unchanged."""
    state = _build_state_header_defect(header_unattributed=False)
    winner = _winning_page_output(state, 1, None)

    assert winner is not None
    assert "[page 1 failed:" not in winner.text
    assert winner.status == PageStatus.WARNING
    assert winner.audit_passed is False


def test_d3_floor_without_png_ships_marker_alone() -> None:
    """PNG render failure degrades to the marker alone, never plausible table text."""
    state = _build_state_header_defect(header_unattributed=True, png_ref="")
    winner = _winning_page_output(state, 1, None)

    assert winner is not None
    assert winner.text.strip() == "[page 1 failed: unverifiable table — see image]"
    assert winner.status == PageStatus.ERROR
    assert winner.audit_passed is False
