"""GH-97: degenerate table-row repetition guard.

Regression cases are taken from the OBR EFO November 2022 reference run, where a
15-character empty row repeated 865 times reached 867 of the 3177 lines of the
assembled document (27% of it) with no audit event and no doc-level signal. The
pre-existing ``OutputNormalizer._RE_LINE_REPEAT`` could not catch it: that rule
has a 20-character floor and runs at the engine boundary, before table assembly.
"""

from __future__ import annotations

from socr.core.audit_log import AuditEvent
from socr.core.normalizer import (
    MAX_CONSECUTIVE_IDENTICAL_TABLE_ROWS,
    OutputNormalizer,
    collapse_repeated_table_rows,
)
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.orchestrator import UnifiedPipeline

# The exact rows observed in the reference run.
_P62_ROW = "| | | | | | | |"  # 15 chars — below the generic rule's 20-char floor
_P48_ROW = "|  |  |  |  |  |  |  |  |  |  |  |  |"  # 37 chars


def test_p62_row_is_below_the_generic_rules_width_floor():
    """Documents *why* the generic normalizer rule cannot catch this case."""
    assert len(_P62_ROW) < 20
    text = "\n".join([_P62_ROW] * 20)
    # The generic rule leaves it completely untouched.
    assert OutputNormalizer()._RE_LINE_REPEAT.sub(r"\1\n", text) == text


def test_collapses_the_p62_runaway():
    body = ["| Real GDP | 1.7 | -0.9 |", "| RPI | 0.5 | 1.8 |"]
    text = "\n".join(body + [_P62_ROW] * 865)

    cleaned, removed = collapse_repeated_table_rows(text)

    assert removed == 865 - MAX_CONSECUTIVE_IDENTICAL_TABLE_ROWS
    assert cleaned.count(_P62_ROW) == MAX_CONSECUTIVE_IDENTICAL_TABLE_ROWS
    # Real content survives untouched.
    for line in body:
        assert line in cleaned


def test_collapses_the_p48_contentless_table():
    text = "\n".join([_P48_ROW] * 16)

    cleaned, removed = collapse_repeated_table_rows(text)

    assert removed == 16 - MAX_CONSECUTIVE_IDENTICAL_TABLE_ROWS
    assert cleaned.count(_P48_ROW) == MAX_CONSECUTIVE_IDENTICAL_TABLE_ROWS


def test_no_content_can_be_lost():
    """Every removed line is byte-identical to one that is kept.

    The citation-corpus invariant: a dropped number is worse than a missing one,
    so the guard must be incapable of losing a distinct cell value.
    """
    text = "\n".join(
        [
            "| a | 1.0 |",
            "| a | 1.0 |",
            "| a | 1.0 |",
            "| a | 1.0 |",
            "| b | 2.0 |",
        ]
    )
    cleaned, removed = collapse_repeated_table_rows(text)

    assert removed == 2
    assert set(cleaned.split("\n")) == set(text.split("\n"))


def test_leaves_a_legitimate_table_alone():
    text = "\n".join(
        [
            "| | 2022-23 | 2023-24 |",
            "| :--- | :--- | :--- |",
            "| Total effect | 42.8 | 30.5 |",
            "| Direct effect | 64.2 | 39.8 |",
            "| of which: | | |",
            "| Tax decisions | -0.2 | 5.5 |",
        ]
    )
    cleaned, removed = collapse_repeated_table_rows(text)

    assert removed == 0
    assert cleaned == text


def test_only_consecutive_duplicates_are_collapsed():
    """A row legitimately reused far apart in a table is not a runaway."""
    row = "| of which: | | |"
    text = "\n".join([row, "| A | 1.0 |", row, "| B | 2.0 |", row])

    cleaned, removed = collapse_repeated_table_rows(text)

    assert removed == 0
    assert cleaned.count(row) == 3


def test_non_table_lines_are_never_touched():
    text = "\n".join(["some prose"] * 40)

    cleaned, removed = collapse_repeated_table_rows(text)

    assert removed == 0
    assert cleaned == text


def test_prose_between_rows_resets_the_run():
    text = "\n".join([_P62_ROW, _P62_ROW, "a paragraph", _P62_ROW, _P62_ROW])

    cleaned, removed = collapse_repeated_table_rows(text)

    assert removed == 0
    assert cleaned == text


def test_empty_text_is_a_noop():
    assert collapse_repeated_table_rows("") == ("", 0)


# ----------------------------------------------------------------------
# Orchestrator seam. Called directly rather than through ``_phase_agentic``:
# driving the agentic loop in CI would need the provider ladder patched, and
# this hook has no engine dependency at all.
# ----------------------------------------------------------------------


class _FakeState:
    def __init__(self):
        self.events: list[AuditEvent] = []


def _page(text: str) -> PageOutput:
    return PageOutput(page_num=62, text=text, status=PageStatus.SUCCESS, engine="qwen")


def test_seam_truncates_and_records_an_audit_event():
    state = _FakeState()
    page = _page("\n".join(["| Real GDP | 1.7 |"] + [_P62_ROW] * 865))

    UnifiedPipeline._guard_agentic_page_table_repetition(state, 62, page)

    assert page.text.count(_P62_ROW) == MAX_CONSECUTIVE_IDENTICAL_TABLE_ROWS
    assert "| Real GDP | 1.7 |" in page.text

    assert len(state.events) == 1
    event = state.events[0]
    assert event.kind == "table_row_repetition_truncated"
    assert event.page_num == 62
    assert event.engine == "qwen"
    assert event.data["rows_removed"] == 863


def test_seam_is_silent_on_a_clean_page():
    state = _FakeState()
    original = "| A | 1.0 |\n| B | 2.0 |"
    page = _page(original)

    UnifiedPipeline._guard_agentic_page_table_repetition(state, 62, page)

    assert page.text == original
    assert state.events == []


def test_seam_handles_empty_page_text():
    state = _FakeState()
    page = _page("")

    UnifiedPipeline._guard_agentic_page_table_repetition(state, 62, page)

    assert state.events == []


# ----------------------------------------------------------------------
# Document-level surface: a consumer gating on metadata must see this without
# parsing audit_log.json (the GH-95 complaint).
# ----------------------------------------------------------------------


def test_doc_level_note_names_every_affected_page():
    events = [
        AuditEvent(page_num=62, kind="table_row_repetition_truncated"),
        AuditEvent(page_num=48, kind="table_row_repetition_truncated"),
        AuditEvent(page_num=48, kind="table_row_repetition_truncated"),
        AuditEvent(page_num=13, kind="dualpass_flagged"),
    ]

    note = UnifiedPipeline._repetition_truncated_note(events)

    assert note is not None
    assert "48, 62" in note
    assert "13" not in note


def test_doc_level_note_is_none_on_a_clean_run():
    events = [AuditEvent(page_num=13, kind="dualpass_flagged")]

    assert UnifiedPipeline._repetition_truncated_note(events) is None


# ----------------------------------------------------------------------
# GH-98: scope. The guard's premise is VLM degeneration.
# ----------------------------------------------------------------------


def test_a_native_page_is_never_truncated():
    """A character-exact native extraction cannot have degenerated.

    And there the collapse is unsafe: repeated rows may be real content. A
    wide-format appendix table with five consecutive all-zero rows and no per-row
    label is legitimate on a native page; truncating it to two drops three rows.
    """
    state = _FakeState()
    original = "\n".join(["| Region | 0.0 | 0.0 |"] * 5)
    page = _page(original)

    UnifiedPipeline._guard_agentic_page_table_repetition(state, 62, page, is_native=True)

    assert page.text == original, "native text must pass through untouched"
    assert state.events == []


def test_row_multiplicity_is_preserved_on_native_pages():
    """The gap #97's own content-safety test could not see.

    ``test_no_content_can_be_lost`` asserts set equality, so it is blind to
    multiplicity: five identical rows collapsing to two passes it. On a VLM page
    that is the intended behaviour; on a native page it is content loss.
    """
    state = _FakeState()
    row = "| Region | 0.0 | 0.0 |"
    page = _page("\n".join([row] * 5))

    UnifiedPipeline._guard_agentic_page_table_repetition(state, 62, page, is_native=True)

    assert page.text.count(row) == 5


def test_a_vlm_page_is_still_truncated():
    """The scoping must not disarm the guard on the pages it was built for."""
    state = _FakeState()
    page = _page("\n".join(["| Real GDP | 1.7 |"] + [_P62_ROW] * 865))

    UnifiedPipeline._guard_agentic_page_table_repetition(state, 62, page, is_native=False)

    assert page.text.count(_P62_ROW) == MAX_CONSECUTIVE_IDENTICAL_TABLE_ROWS
    assert "table_row_repetition_truncated" in [e.kind for e in state.events]


def test_the_default_still_guards():
    """Callers that do not pass is_native get the protective behaviour."""
    state = _FakeState()
    page = _page("\n".join([_P62_ROW] * 20))

    UnifiedPipeline._guard_agentic_page_table_repetition(state, 62, page)

    assert page.text.count(_P62_ROW) == MAX_CONSECUTIVE_IDENTICAL_TABLE_ROWS
