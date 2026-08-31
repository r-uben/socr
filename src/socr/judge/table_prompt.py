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

#: GH-359 ruling 4: leftover placeholder from the dropped findings-injection
#: slot. If a future edit reintroduces ``{{PRIOR_FINDINGS}}`` in the policy
#: file, it is filled with this independent-look sentence rather than a
#: complaint payload.
_FINDINGS_PLACEHOLDER = "{{PRIOR_FINDINGS}}"
_INDEPENDENT_LOOK_NOTE = (
    "Judge independently from the crop image and the emitted markdown. "
    "You are not given any prior verdict or findings."
)


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
    template = load_table_judge_prompt()
    rendered = template.replace(_FINDINGS_PLACEHOLDER, _INDEPENDENT_LOOK_NOTE)
    rendered = rendered.replace(_MARKDOWN_PLACEHOLDER, markdown)
    return rendered
