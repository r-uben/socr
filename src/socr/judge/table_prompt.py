"""Table judge prompt: policy as data, loaded from ``prompts/table_judge.md``.

Mirrors ``socr.judge.judge.load_judge_prompt`` — the prompt template lives in the
policy file, not in code, so wording can be iterated without touching ladder
control flow. ``build_table_judge_prompt`` fills the emitted-markdown slot and
the scope-note slot.

GH-359 ruling 4: judge input is crop + markdown, nothing else. The
``prior_findings`` argument is accepted for RungCallable compatibility and
is ignored — a B-escalation must not prime the next judge toward FAIL.

GH-373: ``{{SCOPE_NOTE}}`` is a live placeholder. The page-scope instruction
lives in ``prompts/table_judge_scope_page.md``, not in this module. Located
scope splices the empty string. The gate selects the fragment via
``table_judge_prompt_scope`` so rung callables keep receiving crop + markdown
only (ruling 4) while a wording-only edit to the fragment still takes effect.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PROMPT_PATH = _PROMPTS_DIR / "table_judge.md"
_SCOPE_NOTE_PATHS = {
    "located": None,  # empty splice — the located prompt is the template as-is
    "page": _PROMPTS_DIR / "table_judge_scope_page.md",
}

_MARKDOWN_PLACEHOLDER = "{{EMITTED_MARKDOWN}}"
_SCOPE_PLACEHOLDER = "{{SCOPE_NOTE}}"
#: Cold review round 2, N1: the coordinate contract is spliced, not
#: restated. The same fragment goes into the blind-transcription prompt.
_GRAMMAR_PLACEHOLDER = "{{CELL_REF_GRAMMAR}}"
#: Cold review round 4: the worked examples go to the READER prompt only.
#: They necessarily show cells with contents in them, and the blind
#: transcription prompt may carry no cell contents at all.
_EXAMPLES_PLACEHOLDER = "{{CELL_REF_EXAMPLES}}"

_SCOPE: ContextVar[str] = ContextVar("socr_table_judge_scope", default="located")


def load_table_judge_prompt() -> str:
    """Read the raw table-judge prompt template (policy lives in the .md)."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def load_table_judge_scope_note(scope: str) -> str:
    """Read the policy fragment for ``scope``. Empty string for ``located``.

    Raises ``ValueError`` on an unknown scope so a typo cannot silently
    splice nothing. ``FileNotFoundError`` (an ``OSError``) on a missing
    fragment file so the fingerprint digest can treat it as unreadable.
    """
    if scope not in _SCOPE_NOTE_PATHS:
        raise ValueError(f"unknown table-judge prompt scope: {scope!r}")
    path = _SCOPE_NOTE_PATHS[scope]
    if path is None:
        return ""
    return path.read_text(encoding="utf-8").strip()


@contextmanager
def table_judge_prompt_scope(scope: str) -> Iterator[None]:
    """Select which ``{{SCOPE_NOTE}}`` fragment ``build_table_judge_prompt`` splices.

    The gate sets this around ``run_table_ladder`` so the rung callables —
    which still receive only crop + markdown — render the fragment that
    matches the image they are shown.
    """
    if scope not in _SCOPE_NOTE_PATHS:
        raise ValueError(f"unknown table-judge prompt scope: {scope!r}")
    token = _SCOPE.set(scope)
    try:
        yield
    finally:
        _SCOPE.reset(token)


def build_table_judge_prompt(
    markdown: str,
    prior_findings: Sequence[Mapping[str, str]] | None = None,
    *,
    scope: str | None = None,
) -> str:
    """Render the full table-judge prompt for one rung call.

    ``markdown`` is the emitted table under judgment. ``prior_findings`` is
    ignored (GH-359 ruling 4); the argument remains so rung call sites that
    still pass it do not have to change. ``scope`` selects the
    ``{{SCOPE_NOTE}}`` fragment (``located`` or ``page``); ``None`` reads
    the ``table_judge_prompt_scope`` context, defaulting to ``located``.
    """
    del prior_findings
    # GH-381: no findings slot. ``prompts/table_judge.md`` carries the
    # independent-look sentence as policy text, so the old replace was a no-op
    # against a placeholder that no longer exists -- and it kept a SECOND copy
    # of that sentence in code, where only the .md copy could ever take effect.
    # One authority for the wording; the ruling-4 guarantee lives in the caller
    # passing no findings at all, not in a dead string substitution.
    #
    # GH-373: ``{{SCOPE_NOTE}}`` is live. The page-scope sentence lives in
    # ``prompts/table_judge_scope_page.md``. This replace is the only
    # substitution; there is no Python copy of that sentence.
    resolved = _SCOPE.get() if scope is None else scope
    note = load_table_judge_scope_note(resolved)
    from socr.judge.table_verdict import load_cell_ref_examples, load_cell_ref_grammar

    return (
        load_table_judge_prompt()
        .replace(_GRAMMAR_PLACEHOLDER, load_cell_ref_grammar())
        .replace(_EXAMPLES_PLACEHOLDER, load_cell_ref_examples())
        .replace(_SCOPE_PLACEHOLDER, note)
        .replace(_MARKDOWN_PLACEHOLDER, markdown)
    )
