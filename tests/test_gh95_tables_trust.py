"""GH-95: consumer-visible table distrust (tables_trust.json + doc-level note).

The complaint: dual-pass detects untrusted tables and records them in
``audit_log.json``, but the assembled markdown looks clean and ``metadata.json``
named exactly one page on a run where 19 carried table flags. These tests pin the
derivation, the top-level summary a consumer gates on, and the guarantee that a
clean run produces nothing.

Hermetic: everything here is a pure derivation over ``AuditEvent`` objects. No
engine, no provider ladder, no ollama - so no need to patch
``_available_engines_for_agentic``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from socr.core.audit_log import AuditEvent
from socr.core.tables_trust import (
    TABLE_DISTRUST_KINDS,
    TRUST_NOTE_PREFIX,
    build_tables_trust,
    trust_note,
)
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.reconcile import PATCH_ELIGIBLE_NOTE


def _flagged(page: int, note: str | None = None) -> AuditEvent:
    return AuditEvent(
        page_num=page,
        kind="dualpass_flagged",
        engine="qwen",
        detail=f"table 0: mismatch on p{page}",
        data={"note": note} if note else {},
    )


# ----------------------------------------------------------------------
# Derivation
# ----------------------------------------------------------------------


def test_clean_run_yields_an_empty_index_and_no_note():
    """Prose-only pages stay unmarked - the issue's second acceptance clause."""
    events = [
        AuditEvent(page_num=1, kind="native_fallback"),
        AuditEvent(page_num=2, kind="native_table_verifier_exact_pass"),
        AuditEvent(page_num=3, kind="figure_cap_reached"),
    ]

    trust = build_tables_trust("doc.pdf", events)

    assert trust.pages == {}
    assert trust.untrusted_pages == []
    assert trust.flag_count == 0
    assert trust.summary_line() == ""
    assert trust_note(trust) is None


def test_a_resolved_disagreement_is_not_distrust():
    """A patched table ships corrected, so it must not be flagged as untrusted."""
    events = [
        AuditEvent(page_num=7, kind="dualpass_patched"),
        AuditEvent(page_num=8, kind="table_header_repair"),
    ]

    trust = build_tables_trust("doc.pdf", events)

    assert trust.pages == {}


def test_page_failed_alone_does_not_enter_the_table_index():
    """page_failed is not table-specific and is already surfaced elsewhere."""
    trust = build_tables_trust("doc.pdf", [AuditEvent(page_num=46, kind="page_failed")])

    assert trust.pages == {}


def test_a_failed_closed_table_page_is_still_captured():
    """Page 46 of the reference run carried table_region_unverifiable too."""
    events = [
        AuditEvent(page_num=46, kind="page_failed"),
        AuditEvent(page_num=46, kind="table_region_unverifiable"),
    ]

    trust = build_tables_trust("doc.pdf", events)

    assert trust.untrusted_pages == [46]
    assert trust.pages[46].reasons == ["table_region_unverifiable"]


def test_multiple_reasons_on_one_page_collapse_to_one_entry():
    events = [
        _flagged(13),
        AuditEvent(page_num=13, kind="native_table_verifier_warn"),
        AuditEvent(page_num=13, kind="value_guard_row_count_warning"),
    ]

    trust = build_tables_trust("doc.pdf", events)

    assert trust.untrusted_pages == [13]
    assert trust.flag_count == 3  # flags count events, not pages
    assert trust.pages[13].to_dict()["reasons"] == [
        "dualpass_flagged",
        "native_table_verifier_warn",
        "value_guard_row_count_warning",
    ]


def test_patch_eligibility_is_read_from_the_named_constant():
    """Not parsed out of prose - the note is a shared constant."""
    trust = build_tables_trust("doc.pdf", [_flagged(13, PATCH_ELIGIBLE_NOTE)])

    assert trust.pages[13].patch_eligible is True
    assert trust.to_dict()["patch_eligible_pages"] == [13]


def test_a_non_patchable_flag_is_not_marked_eligible():
    trust = build_tables_trust(
        "doc.pdf",
        [_flagged(13, "crop reading malformed or column count differs — not patched")],
    )

    assert trust.pages[13].patch_eligible is False
    assert trust.to_dict()["patch_eligible_pages"] == []


def test_repetition_truncation_counts_as_distrust():
    """GH-97's event belongs in the trust index: the table was altered."""
    trust = build_tables_trust(
        "doc.pdf", [AuditEvent(page_num=62, kind="table_row_repetition_truncated")]
    )

    assert trust.untrusted_pages == [62]


# ----------------------------------------------------------------------
# The consumer contract: gate on top-level keys, not by walking pages
# ----------------------------------------------------------------------


def test_top_level_summary_is_enough_to_gate_on():
    events = [_flagged(13, PATCH_ELIGIBLE_NOTE), _flagged(51), _flagged(51)]

    payload = build_tables_trust("efo.pdf", events).to_dict()

    assert payload["pdf_filename"] == "efo.pdf"
    assert payload["untrusted_page_count"] == 2
    assert payload["untrusted_pages"] == [13, 51]
    assert payload["table_flags_n"] == 3
    assert payload["counts_by_kind"] == {"dualpass_flagged": 3}
    assert payload["patch_eligible_pages"] == [13]


def test_sidecar_is_json_serialisable_with_string_page_keys():
    payload = build_tables_trust("doc.pdf", [_flagged(13)]).to_dict()

    round_tripped = json.loads(json.dumps(payload))

    assert "13" in round_tripped["pages"]
    assert round_tripped["pages"]["13"]["flagged"] is True


def test_saved_sidecar_matches_the_built_payload(tmp_path):
    trust = build_tables_trust("doc.pdf", [_flagged(13), _flagged(46)])
    path = tmp_path / "nested" / "tables_trust.json"

    trust.save(path)

    assert json.loads(path.read_text(encoding="utf-8")) == trust.to_dict()


# ----------------------------------------------------------------------
# Document-level note
# ----------------------------------------------------------------------


def test_doc_level_note_reports_counts_and_points_at_the_sidecar():
    note = trust_note(build_tables_trust("doc.pdf", [_flagged(13), _flagged(46)]))

    assert note is not None
    assert note.startswith(TRUST_NOTE_PREFIX)
    assert "2 page(s)" in note
    assert "tables_trust.json" in note


def test_orchestrator_note_helper_derives_from_state_events():
    state = SimpleNamespace(handle=SimpleNamespace(filename="doc.pdf"), events=[_flagged(13)])

    note = UnifiedPipeline._tables_trust_note(state)

    assert note is not None
    assert "1 page(s)" in note


def test_orchestrator_note_helper_is_none_on_a_clean_state():
    state = SimpleNamespace(handle=SimpleNamespace(filename="doc.pdf"), events=[])

    assert UnifiedPipeline._tables_trust_note(state) is None


def test_orchestrator_note_helper_survives_a_malformed_state():
    """Non-fatal: a note that cannot be derived must not fail the run."""

    class _Broken:
        @property
        def events(self):
            raise RuntimeError("boom")

    assert UnifiedPipeline._tables_trust_note(_Broken()) is None


# ----------------------------------------------------------------------
# Taxonomy sanity: the kinds we claim to watch must be kinds the code emits
# ----------------------------------------------------------------------


def test_watched_kinds_are_real_emitted_kinds():
    """Guards against a typo silently disabling a distrust signal.

    Two documented exceptions:
      - ``dualpass_flagged`` is built as f"dualpass_{action}" with
        action="flagged", so it is not a literal anywhere in the source.
      - ``table_row_repetition_truncated`` is emitted by GH-97, which ships on its
        own branch. Watching it here is forward-compatible (an unemitted kind
        simply never matches) and avoids a follow-up edit once #97 merges.
    """
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "socr"
    literals = set()
    for path in src.rglob("*.py"):
        literals.update(re.findall(r'kind="([a-z_]+)"', path.read_text(encoding="utf-8")))

    dynamic = {"dualpass_flagged"}
    # GH-97 ships on its own branch. Listing it here is forward-compatible in both
    # directions: before that merge the kind is never emitted (and simply never
    # matches an event), after it the kind becomes a literal and drops out of
    # ``literals`` subtraction harmlessly. Deliberately NOT a tripwire that fails
    # once #97 lands - a stale comment is cheaper than a red build on main.
    pending = {"table_row_repetition_truncated"}
    unknown = TABLE_DISTRUST_KINDS - literals - dynamic - pending

    assert unknown == set(), f"watched kinds never emitted anywhere: {sorted(unknown)}"


def test_doc_level_note_is_bounded_regardless_of_document_size():
    """The reference run flagged 20 of 20 table pages; the field must not grow
    with the page count."""
    many = [_flagged(n) for n in range(1, 201)]

    note = trust_note(build_tables_trust("doc.pdf", many))

    assert note is not None
    assert len(note) < 100
    assert "200 page(s)" in note
