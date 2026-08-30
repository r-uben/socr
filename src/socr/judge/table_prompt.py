"""Table judge prompt: policy as data, loaded from ``prompts/table_judge.md``.

Mirrors ``socr.judge.judge.load_judge_prompt`` — the prompt template lives in the
policy file, not in code, so wording can be iterated without touching ladder
control flow. ``build_table_judge_prompt`` additionally fills two injection
slots: the emitted markdown for the table under judgment, and (in tiebreak mode,
when a later rung is confirming an earlier FAIL) the prior rung's findings.

``prior_findings`` is duck-typed as a sequence of mappings with ``code`` /
``where`` / ``detail`` keys rather than importing ``TableJudgeVerdict`` — this
module has no dependency on the verdict schema (``socr.judge.table_verdict``,
TICKET-A1) so either can land independently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "table_judge.md"

_MARKDOWN_PLACEHOLDER = "{{EMITTED_MARKDOWN}}"
_FINDINGS_PLACEHOLDER = "{{PRIOR_FINDINGS}}"

_NO_PRIOR_FINDINGS_NOTE = "This is the first rung to judge this table; there are no prior findings."


def load_table_judge_prompt() -> str:
    """Read the raw table-judge prompt template (policy lives in the .md)."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _render_prior_findings(prior_findings: Sequence[Mapping[str, str]] | None) -> str:
    if not prior_findings:
        return _NO_PRIOR_FINDINGS_NOTE
    lines = [
        "A prior rung FAILed this table with the findings below. Confirm, refute, "
        "or reclassify each one against what you actually see in the image — do "
        "not simply repeat them:"
    ]
    for finding in prior_findings:
        code = finding.get("code", "?")
        where = finding.get("where", "?")
        detail = finding.get("detail", "")
        lines.append(f"- {code} at {where}: {detail}")
    return "\n".join(lines)


def build_table_judge_prompt(
    markdown: str,
    prior_findings: Sequence[Mapping[str, str]] | None = None,
) -> str:
    """Render the full table-judge prompt for one rung call.

    ``markdown`` is the emitted table under judgment. ``prior_findings`` is the
    tiebreak-mode injection (rung 2 confirming a rung-1 FAIL); pass ``None`` or
    an empty sequence for a first-look call, which renders a neutral note
    instead of leaving the slot blank.
    """
    template = load_table_judge_prompt()
    rendered = template.replace(_FINDINGS_PLACEHOLDER, _render_prior_findings(prior_findings))
    rendered = rendered.replace(_MARKDOWN_PLACEHOLDER, markdown)
    return rendered
