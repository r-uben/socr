"""Table judge prompt: policy as data, loaded from ``prompts/table_judge.md``.

Mirrors ``socr.judge.judge.load_judge_prompt`` — the prompt template lives in the
policy file, not in code, so wording can be iterated without touching ladder
control flow. ``build_table_judge_prompt`` fills the emitted-markdown slot.

GH-359 ruling 4: judge input is crop + markdown, nothing else. The
``prior_findings`` argument is accepted for RungCallable compatibility and
is ignored — a B-escalation must not prime the next judge toward FAIL.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "table_judge.md"

_MARKDOWN_PLACEHOLDER = "{{EMITTED_MARKDOWN}}"


def load_table_judge_prompt() -> str:
    """Read the raw table-judge prompt template (policy lives in the .md)."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def build_table_judge_prompt(
    markdown: str,
    prior_findings: Sequence[Mapping[str, str]] | None = None,
) -> str:
    """Render the full table-judge prompt for one rung call.

    ``markdown`` is the emitted table under judgment. ``prior_findings`` is
    ignored (GH-359 ruling 4); the argument remains so rung call sites that
    still pass it do not have to change.
    """
    del prior_findings
    # GH-381: no findings slot. ``prompts/table_judge.md`` carries the
    # independent-look sentence as policy text, so the old replace was a no-op
    # against a placeholder that no longer exists -- and it kept a SECOND copy
    # of that sentence in code, where only the .md copy could ever take effect.
    # One authority for the wording; the ruling-4 guarantee lives in the caller
    # passing no findings at all, not in a dead string substitution.
    return load_table_judge_prompt().replace(_MARKDOWN_PLACEHOLDER, markdown)
