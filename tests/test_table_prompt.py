"""TICKET-A0: table judge prompt policy file + loader.

The prompt is data (``prompts/table_judge.md``), not code, mirroring the page
judge's ``load_judge_prompt``. Pins: the six-code closed enum appears verbatim
in the template (A1's parser treats an unknown code as an S1 failure, so the
model must only ever see these six spellings); the loader renders with and
without a prior-findings payload (the tiebreak injection slot A3 uses); the
emitted markdown is always embedded, never dropped.
"""

from __future__ import annotations

from socr.judge.table_prompt import (
    build_table_judge_prompt,
    load_table_judge_prompt,
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
    assert "{{PRIOR_FINDINGS}}" in template


def test_build_prompt_without_findings_embeds_markdown_and_neutral_note():
    markdown = "| a | b |\n|---|---|\n| 1 | 2 |"
    rendered = build_table_judge_prompt(markdown)
    assert markdown in rendered
    assert "{{EMITTED_MARKDOWN}}" not in rendered
    assert "{{PRIOR_FINDINGS}}" not in rendered
    assert "no prior findings" in rendered.lower()


def test_build_prompt_with_findings_injects_tiebreak_payload():
    markdown = "| a | b |\n|---|---|\n| 1 | 2 |"
    prior_findings = [
        {"code": "WRONG_BINDING", "where": "row 2, col Coef", "detail": "shifted one column left"},
    ]
    rendered = build_table_judge_prompt(markdown, prior_findings)
    assert markdown in rendered
    assert "WRONG_BINDING" in rendered
    assert "row 2, col Coef" in rendered
    assert "shifted one column left" in rendered
    assert "no prior findings" not in rendered.lower()


def test_build_prompt_with_empty_findings_list_uses_neutral_note():
    rendered = build_table_judge_prompt("| a |\n|---|\n| 1 |", prior_findings=[])
    assert "no prior findings" in rendered.lower()


def test_build_prompt_findings_missing_optional_keys_do_not_raise():
    rendered = build_table_judge_prompt(
        "| a |\n|---|\n| 1 |", prior_findings=[{"code": "NOT_A_TABLE"}]
    )
    assert "NOT_A_TABLE" in rendered
