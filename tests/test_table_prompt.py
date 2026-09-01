"""TICKET-A0: table judge prompt policy file + loader.

The prompt is data (``prompts/table_judge.md``), not code, mirroring the page
judge's ``load_judge_prompt``. Pins: the six-code closed enum appears verbatim
in the template (A1's parser treats an unknown code as an S1 failure, so the
model must only ever see these six spellings); GH-359 ruling 4: a prior-
findings payload is never injected (judge input is crop + markdown only);
the emitted markdown is always embedded, never dropped.
"""

from __future__ import annotations

import pytest

from socr.judge.table_prompt import (
    build_table_judge_prompt,
    load_table_judge_prompt,
    load_table_judge_scope_note,
    table_judge_prompt_scope,
)

FINDING_CODES = [
    "MISSING_VALUE",
    "FABRICATED_VALUE",
    "WRONG_BINDING",
    "HEADER_MANGLED",
    "STRUCTURE_MERGED",
    "NOT_A_TABLE",
]


def test_prompt_file_contains_all_six_finding_codes():
    template = load_table_judge_prompt()
    for code in FINDING_CODES:
        assert code in template, f"missing finding code {code!r} in table_judge.md"


def test_prompt_declares_verdict_and_confidence_schema():
    template = load_table_judge_prompt()
    assert '"verdict"' in template
    assert '"confidence"' in template
    assert '"findings"' in template
    assert "PASS" in template and "FAIL" in template


def test_prompt_states_judge_only_the_table_region():
    template = load_table_judge_prompt()
    assert "table region" in template.lower()


def test_prompt_states_empty_cell_rule():
    template = load_table_judge_prompt()
    assert "empty-cell rule" in template.lower() or "genuinely blank" in template.lower()


def test_loader_returns_raw_template_unrendered():
    template = load_table_judge_prompt()
    assert "{{EMITTED_MARKDOWN}}" in template
    assert "{{SCOPE_NOTE}}" in template
    assert "independently" in template.lower()


def test_build_prompt_embeds_markdown_and_independent_look_note():
    markdown = "| a | b |\n|---|---|\n| 1 | 2 |"
    rendered = build_table_judge_prompt(markdown)
    assert markdown in rendered
    assert "{{EMITTED_MARKDOWN}}" not in rendered
    assert "{{SCOPE_NOTE}}" not in rendered
    assert "{{PRIOR_FINDINGS}}" not in rendered
    assert "independently" in rendered.lower()
    assert "not given" in rendered.lower()
    assert "multiple tables may be visible" not in rendered.lower()


def test_build_prompt_does_not_inject_findings():
    """GH-359 ruling 4: a complaint payload must not reach the judge."""
    markdown = "| a | b |\n|---|---|\n| 1 | 2 |"
    prior_findings = [
        {"code": "WRONG_BINDING", "where": "row 2, col Coef", "detail": "shifted one column left"},
    ]
    rendered = build_table_judge_prompt(markdown, prior_findings)
    assert markdown in rendered
    assert "row 2, col Coef" not in rendered
    assert "shifted one column left" not in rendered


def test_build_prompt_with_empty_findings_list_still_independent():
    rendered = build_table_judge_prompt("| a |\n|---|\n| 1 |", prior_findings=[])
    assert "independently" in rendered.lower()


def test_build_prompt_findings_argument_does_not_raise():
    rendered = build_table_judge_prompt(
        "| a |\n|---|\n| 1 |",
        prior_findings=[{"code": "NOT_A_TABLE", "detail": "unique-payload-xyz"}],
    )
    assert "unique-payload-xyz" not in rendered


_PAGE_SCOPE_PHRASE = "multiple tables may be visible"


def test_page_scope_fragment_is_policy_not_code():
    """GH-381: the scope instruction must not have a second copy in Python."""
    from pathlib import Path

    note = load_table_judge_scope_note("page")
    assert _PAGE_SCOPE_PHRASE in note.lower()
    prompt_py = Path(__file__).resolve().parents[1] / "src" / "socr" / "judge" / "table_prompt.py"
    assert _PAGE_SCOPE_PHRASE not in prompt_py.read_text(encoding="utf-8").lower()


def test_located_scope_note_is_empty():
    assert load_table_judge_scope_note("located") == ""


def test_unknown_scope_raises():
    with pytest.raises(ValueError, match="unknown table-judge prompt scope"):
        load_table_judge_scope_note("union")


def test_page_scope_splices_the_fragment_and_drops_the_placeholder():
    markdown = "| a | b |\n|---|---|\n| 1 | 2 |"
    rendered = build_table_judge_prompt(markdown, scope="page")
    assert markdown in rendered
    assert "{{SCOPE_NOTE}}" not in rendered
    assert "{{EMITTED_MARKDOWN}}" not in rendered
    flat = " ".join(rendered.split()).lower()
    assert _PAGE_SCOPE_PHRASE in flat
    assert "whose content matches the emitted markdown" in flat


def test_context_manager_selects_page_scope_for_rung_shaped_calls():
    """Rungs call build_table_judge_prompt(markdown, prior_findings) with no
    scope argument. The gate's context manager is how the fragment reaches
    them without a fourth RungCallable parameter (GH-359 ruling 4)."""
    markdown = "| a |\n|---|\n| 1 |"
    with table_judge_prompt_scope("page"):
        rendered = build_table_judge_prompt(markdown, None)
    assert _PAGE_SCOPE_PHRASE in rendered.lower()
    # Context resets: a later located call must not leak the page note.
    located = build_table_judge_prompt(markdown, None)
    assert _PAGE_SCOPE_PHRASE not in located.lower()


def test_prompt_digest_includes_the_page_scope_fragment(monkeypatch):
    """A wording-only edit to the fragment must invalidate resume."""
    from socr.pipeline.orchestrator import _table_judge_prompt_digest

    digest_a = _table_judge_prompt_digest()
    monkeypatch.setattr(
        "socr.judge.table_prompt.load_table_judge_scope_note",
        lambda scope: "CHANGED-FRAGMENT" if scope == "page" else "",
    )
    digest_b = _table_judge_prompt_digest()
    assert digest_a != digest_b
