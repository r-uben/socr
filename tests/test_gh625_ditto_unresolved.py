"""#625: ditto marks are preserved verbatim, and their presence is surfaced.

Owner ruling (2026-09-08, option 3): a shipped table cell whose entire content
is a ditto mark (``"``, ``''``, ``”``, ``″``, ``〃``) is kept byte-for-byte --
NO fill-down, because a fill-down that guesses wrong manufactures a number,
the exact failure mode this corpus most wants to avoid. What must change is
visibility: the table-level flag with the column index and the count of
ditto cells, carried into the page sidecar, ``tables_trust.json``, the
document metadata note, the CLI report, and restored on resume.

Falsification: main's ``detect_ditto_columns`` does not exist and
``PageOutput`` has no ``table_ditto_columns`` field, so every assertion below
about the flag being set, surfaced, and round-tripped fails on main. A table
with a legitimate quoted-string cell (``"n/a"``) must NOT be flagged (it is
not a bare ditto mark), and a table with no ditto mark must ship byte-
identical output with no artifacts at all.

Hermetic: pure text/dataclass round-trips, no ollama, no provider ladder.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from ocr_output_contract import doc_dir_for, relative_key
from test_gh659_label_unverified_finalization import _pipeline, _state

from socr.core.audit_log import AuditEvent
from socr.core.manifest import (
    FinalizedPageRecord,
    PageDisposition,
    PageEnding,
    PagePrimaryReason,
    SelectionProvenance,
    _apply_ditto_guard,
)
from socr.core.result import PageOutput, PageStatus
from socr.core.tables_trust import TABLE_DISTRUST_KINDS, build_tables_trust
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.ditto import DITTO_UNRESOLVED_KIND, detect_ditto_columns

# The issue's own repro shape: the annual FOMC swap-arrangement renewal table.
# Header + 15 body rows; only the first body row carries a real Term value,
# the other 14 carry a ditto mark -- column index 2, count 14.
_SWAP_HEADER = "| Institution | Amount | Term | Date |\n| --- | --- | --- | --- |\n"
_SWAP_FIRST_ROW = "| Austrian National Bank | 250.0 | 12 mos. | 12/4/79 |\n"
_SWAP_DITTO_ROWS = "".join(f'| Bank {i} | {1000 + i}.0 | " | 12/4/79 |\n' for i in range(14))
SWAP_TABLE = _SWAP_HEADER + _SWAP_FIRST_ROW + _SWAP_DITTO_ROWS

QUOTED_STRING_TABLE = (
    '| Country | Status |\n| --- | --- |\n| Freedonia | "n/a" |\n| Sylvania | "n/a" |\n'
)

NO_DITTO_TABLE = "| Country | Amount |\n| --- | --- |\n| Germany | 10 |\n| France | 20 |\n"


# --------------------------------------------------------------------------
# 1. Detection: the pure text scan.
# --------------------------------------------------------------------------


def test_swap_arrangement_shape_flags_column_2_count_14() -> None:
    columns = detect_ditto_columns(SWAP_TABLE, page_num=7)
    assert len(columns) == 1
    col = columns[0]
    assert col.table_id == "p7-t0"
    assert col.column_index == 2
    assert col.ditto_cells == 14


def test_legitimate_quoted_string_cell_is_not_flagged() -> None:
    assert detect_ditto_columns(QUOTED_STRING_TABLE, page_num=1) == []


def test_table_with_no_ditto_mark_is_not_flagged() -> None:
    assert detect_ditto_columns(NO_DITTO_TABLE, page_num=1) == []


def test_detection_never_alters_the_source_text() -> None:
    before = SWAP_TABLE
    detect_ditto_columns(SWAP_TABLE, page_num=7)
    assert SWAP_TABLE == before  # pure function; the module docstring's own claim


# --------------------------------------------------------------------------
# 2. The manifest guard: SUCCESS -> WARNING, field set, text untouched; a
#    clean page is returned unchanged (byte-identical, no artifacts).
# --------------------------------------------------------------------------


def test_guard_demotes_success_to_warning_and_sets_the_field() -> None:
    out = PageOutput(page_num=7, text=SWAP_TABLE, status=PageStatus.SUCCESS, engine="qwen")
    guarded = _apply_ditto_guard(out, page_num=7)
    assert guarded.status is PageStatus.WARNING
    assert guarded.text == SWAP_TABLE  # no fill-down, ever
    assert guarded.table_ditto_columns == [
        {"table_id": "p7-t0", "column_index": 2, "ditto_cells": 14}
    ]


def test_guard_leaves_a_clean_page_byte_identical_with_no_artifacts() -> None:
    out = PageOutput(page_num=1, text=NO_DITTO_TABLE, status=PageStatus.SUCCESS, engine="qwen")
    guarded = _apply_ditto_guard(out, page_num=1)
    assert guarded is out  # identity: nothing rebuilt for a clean page
    assert guarded.table_ditto_columns == []


def test_guard_never_upgrades_a_more_severe_status() -> None:
    out = PageOutput(page_num=7, text=SWAP_TABLE, status=PageStatus.ERROR, engine="qwen")
    guarded = _apply_ditto_guard(out, page_num=7)
    assert guarded.status is PageStatus.ERROR
    assert guarded.table_ditto_columns  # still recorded, just not re-promoted


# --------------------------------------------------------------------------
# 3. ``PageOutput`` serialization: field round-trips; empty list omitted so a
#    clean page's content-addressed fingerprint is unchanged (no spurious
#    resume reprocess across every pre-#625 corpus).
# --------------------------------------------------------------------------


def test_field_round_trips_through_to_dict_from_dict() -> None:
    out = PageOutput(
        page_num=7,
        text=SWAP_TABLE,
        status=PageStatus.WARNING,
        engine="qwen",
        table_ditto_columns=[{"table_id": "p7-t0", "column_index": 2, "ditto_cells": 14}],
    )
    restored = PageOutput.from_dict(out.to_dict())
    assert restored.table_ditto_columns == out.table_ditto_columns


def test_empty_field_is_omitted_from_the_dict() -> None:
    out = PageOutput(page_num=1, text=NO_DITTO_TABLE, status=PageStatus.SUCCESS, engine="qwen")
    assert "table_ditto_columns" not in out.to_dict()


# --------------------------------------------------------------------------
# 4. ``tables_trust.json``: the kind is a distrust entry, resolved off the
#    FINAL winning candidate (same shape as LABEL_UNVERIFIED_KIND, #659).
# --------------------------------------------------------------------------


def test_ditto_kind_is_a_table_distrust_kind() -> None:
    assert DITTO_UNRESOLVED_KIND in TABLE_DISTRUST_KINDS


def test_flagged_page_shows_up_in_tables_trust_json() -> None:
    events = [
        AuditEvent(
            page_num=7,
            kind=DITTO_UNRESOLVED_KIND,
            engine="native",
            detail="table p7-t0 column 2: 14 ditto-mark cell(s)",
            data={"table_id": "p7-t0", "column_index": 2, "ditto_cells": 14},
        )
    ]
    trust = build_tables_trust("doc.pdf", events)
    assert 7 in trust.untrusted_pages


def test_a_later_clean_candidate_retires_the_flag_from_trust() -> None:
    """Same terminal-diagnosis contract as #659: the caller's CURRENT winner
    set decides resolution, never the raw event history alone.
    """
    events = [
        AuditEvent(
            page_num=7,
            kind=DITTO_UNRESOLVED_KIND,
            engine="native",
            detail="table p7-t0 column 2: 14 ditto-mark cell(s)",
            data={"table_id": "p7-t0", "column_index": 2, "ditto_cells": 14},
        )
    ]
    trust = build_tables_trust(
        "doc.pdf",
        events,
        ditto_unresolved_pages=frozenset(),  # current winner is clean
    )
    assert 7 not in trust.untrusted_pages


# --------------------------------------------------------------------------
# 5. Terminal-diagnosis helpers read the FINAL winning output.
# --------------------------------------------------------------------------


def _disposition() -> PageDisposition:
    return PageDisposition(PageEnding.NATIVE_PROSE, PagePrimaryReason.CLEAN_NATIVE_PROSE)


def _record(output: PageOutput) -> FinalizedPageRecord:
    return FinalizedPageRecord(
        output=output,
        disposition=_disposition(),
        selection_provenance=SelectionProvenance.NATIVE_CLEAN,
    )


def test_ditto_unresolved_pages_reads_the_final_output_field() -> None:
    flagged = PageOutput(
        page_num=7,
        text=SWAP_TABLE,
        status=PageStatus.WARNING,
        engine="qwen",
        table_ditto_columns=[{"table_id": "p7-t0", "column_index": 2, "ditto_cells": 14}],
    )
    clean = PageOutput(page_num=8, text="prose only", status=PageStatus.SUCCESS, engine="qwen")
    records = [_record(flagged), _record(clean)]
    assert UnifiedPipeline._ditto_unresolved_pages(records) == [7]


def test_ditto_unresolved_note_names_the_page_and_the_count() -> None:
    flagged = PageOutput(
        page_num=7,
        text=SWAP_TABLE,
        status=PageStatus.WARNING,
        engine="qwen",
        table_ditto_columns=[{"table_id": "p7-t0", "column_index": 2, "ditto_cells": 14}],
    )
    note = UnifiedPipeline._ditto_unresolved_note([_record(flagged)])
    assert note is not None
    assert "7" in note
    assert "14" in note


def test_ditto_unresolved_note_is_none_on_a_clean_run() -> None:
    clean = PageOutput(page_num=1, text="prose only", status=PageStatus.SUCCESS, engine="qwen")
    assert UnifiedPipeline._ditto_unresolved_note([_record(clean)]) is None


# --------------------------------------------------------------------------
# 6. Resume: the kind is replayed, not silently dropped.
# --------------------------------------------------------------------------


def test_ditto_kind_survives_resume_replay() -> None:
    assert DITTO_UNRESOLVED_KIND in UnifiedPipeline.resume_restore_kinds()


def _one_col_table(cell: str) -> str:
    return f"| Name | Value |\n| --- | --- |\n| A | 12 |\n| B | {cell} |\n"


def test_changed_count_after_real_restore(tmp_path: Path) -> None:
    """Astra P2 (round 1, e40495d): a REPLAYED event with a STALE count must
    not survive the retire/readd dedup just because its (page, table_id,
    column_index) identity still matches.

    Simulates a genuine resume: an earlier run's terminal sidecar carries a
    ``DITTO_UNRESOLVED_KIND`` event recording 14 ditto cells in this table's
    column; ``_restore_terminal_page_state`` (the real method, reading a real
    sidecar file) replays it into ``state.events``. The CURRENT winning
    candidate for the same page/table/column carries only ONE ditto cell.
    Every reader of the outcome -- the audit event, ``tables_trust.json``'s
    detail, the document metadata note, the CLI line -- must report 1, not
    the stale 14, because the field lives on the FINAL winning candidate
    (owner ruling: no fill-down, no history-only truth).
    """
    pipeline = _pipeline()
    state = _state(tmp_path, page_count=1)
    stale_data = {"table_id": "p1-t0", "column_index": 1, "ditto_cells": 14}
    stale_event = AuditEvent(
        page_num=1,
        kind=DITTO_UNRESOLVED_KIND,
        engine="native",
        detail="table p1-t0 column 1: 14 ditto-mark cell(s)",
        data=stale_data,
    )
    doc_dir = doc_dir_for(tmp_path, relative_key(state.handle.path, state.handle.path.parent))
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "00001.json").write_text(json.dumps({"audit_events": [stale_event.to_dict()]}))

    current_output = _apply_ditto_guard(
        PageOutput(
            page_num=1,
            text=_one_col_table('"'),
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        ),
        page_num=1,
    )
    assert current_output.table_ditto_columns == [
        {"table_id": "p1-t0", "column_index": 1, "ditto_cells": 1}
    ]
    # Round-trip through the sidecar shape, same as a real resume load.
    restored_output = PageOutput.from_dict(current_output.to_dict())
    pipeline._restore_terminal_page_state(state, 1, restored_output, tmp_path)
    assert any(e.kind == DITTO_UNRESOLVED_KIND for e in state.events)  # the stale replay landed

    record = FinalizedPageRecord(
        output=restored_output,
        disposition=_disposition(),
        selection_provenance=SelectionProvenance.NATIVE_CLEAN,
    )
    with patch("socr.core.manifest.finalized_page_records", return_value=[record]):
        result = pipeline._phase_assemble(state, tmp_path)

    error = result.error or ""
    assert "1 ditto-mark cell(s)" in error
    assert "14 ditto-mark cell(s)" not in error

    ditto_events = [e for e in state.events if e.kind == DITTO_UNRESOLVED_KIND]
    assert len(ditto_events) == 1
    assert ditto_events[0].data["ditto_cells"] == 1
