"""GH-142: every `process` flag is classified, and CI keeps it that way.

Two flags were found promising behaviour the default agentic path does not
deliver -- `--max-cost-per-page` against $0.00-priced cloud rungs, and
`--no-audit` (#139) -- both found incidentally while looking for something
else. A flag that lies is worse than a missing flag: the user believes a
constraint is in force and scripts around it.

The sweep this ticket asked for found a THIRD, deliberately rather than by
accident: `--no-judge-hard-pages` set `config.judge_hard_pages`, which gates
nothing anywhere. Its only remaining reader is `_run_fingerprint`, so toggling
it changed the run identity without changing behaviour -- invalidating terminal
pages and forcing a reprocess that produces the same bytes. It now rejects, as
#139 did for `--no-audit`.

The lasting deliverable is not that sweep, though; it is this file. Every option
on `process` must appear in `CLASSIFIED` below with a status and a reason. A new
flag fails the test until someone classifies it, which converts "someone
remembered to check" into something CI enforces. The issue's own words: a third
instance found by accident would be a process failure, not bad luck.

What this file does NOT claim: that a flag marked AGENTIC demonstrably changes
agentic output. Proving that for ~38 flags means ~38 pipeline runs, and several
need a provider. The mechanical half that IS checked here is the one that caught
all three known instances -- a flag whose config field has no consumer capable of
gating anything. Per-flag behavioural pins stay the job of each flag's own test.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import click
import pytest

from socr.cli import cli
from socr.core.config import PipelineConfig

_SRC = Path(__file__).resolve().parents[1] / "src" / "socr"

# CLI option name -> PipelineConfig field, where the two differ (cubic P2 on
# #516). These four were SKIPPED by the first version for want of a matching
# dataclass field -- which quietly exempted `--primary`, `--dpi` and
# `--fallback` from the very check this file exists to apply.
# `fallback` maps to the BACKING field, not to the alias: `fallback_engine` is a
# property whose setter writes `fallback_chain`, so checking for reads of the
# property name would find none and wrongly convict a working flag.
_FIELD_ALIASES = {
    "primary": "primary_engine",
    "fallback": "fallback_chain",
    "dpi": "render_dpi",
    # `--hpc-sequential` writes config.hpc.*, a nested section rather than a
    # top-level field, so it is checked by its nested attribute instead.
    "hpc_sequential": "hpc",
}

# --- status vocabulary ------------------------------------------------------
AGENTIC = "agentic"  # read on, or by a helper of, the default agentic path
NON_AGENTIC = "non-agentic"  # by design; must say so in its own help text
PLUMBING = "plumbing"  # not a pipeline knob (paths, output, verbosity, mode)
REJECTED = "rejected"  # incoherent on this path; the CLI refuses it

# --- the classification -----------------------------------------------------
# Each entry: option name -> (status, reason). The reason is the point: it is
# what a future reader checks against the code, and what makes a wrong entry
# correctable rather than merely present.
CLASSIFIED: dict[str, tuple[str, str]] = {
    # -- mode / plumbing
    "agentic": (PLUMBING, "selects the path itself"),
    "config_path": (PLUMBING, "loads the config file"),
    "output_dir": (PLUMBING, "where output is written"),
    "dry_run": (PLUMBING, "reports what would run"),
    "quiet": (PLUMBING, "console verbosity"),
    "verbose": (PLUMBING, "console verbosity"),
    "reprocess": (PLUMBING, "bypasses the resume gate"),
    "profile": (PLUMBING, "named preset applied before anything else"),
    "unified": (PLUMBING, "legacy alias for the unified pipeline"),
    # -- read on the agentic path or by a helper it calls
    "cost_budget": (AGENTIC, "read in _phase_agentic"),
    "max_cost_per_page": (AGENTIC, "read in _phase_agentic (see #154, #160)"),
    "strict_local": (AGENTIC, "read in _phase_agentic"),
    "native_only": (AGENTIC, "read in _phase_agentic"),
    "no_native_first": (AGENTIC, "native_first, read in _phase_agentic"),
    "detect_equations": (AGENTIC, "read in _phase_agentic"),
    "recover_clean_equations": (AGENTIC, "read in _phase_agentic"),
    "no_dual_pass_tables": (AGENTIC, "dual_pass_tables, read in _phase_agentic"),
    "qwen_backend": (AGENTIC, "read in _phase_agentic"),
    "qwen_vllm_url": (AGENTIC, "read in _phase_agentic"),
    "judge_backend": (AGENTIC, "read by _build_page_judge"),
    "judge_model": (AGENTIC, "read by _build_page_judge"),
    "table_judge_ladder": (AGENTIC, "read by the table judge ladder"),
    "auto_patch_tables": (AGENTIC, "read by _reread_page_tables"),
    "timeout": (AGENTIC, "provider/judge/crop deadlines"),
    "dpi": (AGENTIC, "render_dpi; page rasterisation and the fingerprint"),
    "save_figures": (AGENTIC, "gates the figure phase in _phase_assemble"),
    "describe_figures": (AGENTIC, "gates the caption engine in the figure phase"),
    "write_manifest": (AGENTIC, "manifest is written after assemble"),
    "primary": (AGENTIC, "primary_engine; the ladder's starting rung"),
    "qwen_model": (AGENTIC, "resolved model for the qwen rung"),
    "qwen_vllm_model": (AGENTIC, "resolved model for the vLLM qwen rung"),
    "math_model": (AGENTIC, "equation phases"),
    "clean_equation_model": (AGENTIC, "equation phases"),
    "recover_corrupt_math": (AGENTIC, "corrupt-math recovery routing"),
    # -- non-agentic by design
    "fallback": (NON_AGENTIC, "fallback_engine; the agentic ladder supersedes it"),
    "hpc_sequential": (NON_AGENTIC, "HPC lane only"),
    # -- rejected outright
    "no_audit": (REJECTED, "GH-139: no consumer in any mode; raises UsageError"),
    "no_judge_hard_pages": (REJECTED, "GH-142: gates nothing; raises UsageError"),
}

_PROCESS_OPTIONS = sorted(
    p.name for p in cli.commands["process"].params if isinstance(p, click.Option)
)


def test_every_process_flag_is_classified() -> None:
    """The guard that stops the class from recurring.

    A flag added to `process` without an entry here fails, which is the point:
    classifying it is a deliberate act, not something to remember.
    """
    unclassified = [name for name in _PROCESS_OPTIONS if name not in CLASSIFIED]
    assert not unclassified, (
        f"new process flag(s) with no GH-142 classification: {unclassified}. "
        "Add each to CLASSIFIED with a status and the reason, having checked "
        "what it actually does on the DEFAULT agentic path -- not what its help "
        "text says."
    )


def test_the_classification_has_no_stale_entries() -> None:
    """The other direction: an entry for a flag that no longer exists is a
    classification of nothing, and would hide a real gap behind a full-looking
    table."""
    stale = [name for name in CLASSIFIED if name not in _PROCESS_OPTIONS]
    assert not stale, f"CLASSIFIED names flags that `process` no longer has: {stale}"


def _config_field_for(option_name: str) -> str | None:
    fields = {f.name for f in dataclasses.fields(PipelineConfig)}
    for candidate in (
        _FIELD_ALIASES.get(option_name, option_name),
        option_name.removeprefix("no_"),
    ):
        if candidate in fields:
            return candidate
    return None


class _ConfigReads(ast.NodeVisitor):
    """Attribute LOADS of ``config.<field>``, excluding the run fingerprint.

    cubic P2 on #516: the first version grepped source text, so a comment or a
    docstring mentioning ``config.foo`` certified a dead flag as live, and a
    read spelled across a line break was invisible. Text matching is the wrong
    instrument for "is this value used"; the parse tree is the right one.

    The fingerprint exclusion is structural rather than a string test: any read
    inside ``_run_fingerprint`` records the value into the run identity, which
    is precisely the not-gating-anything case this check is for.
    """

    def __init__(self, field: str) -> None:
        self.field = field
        self.reads: list[int] = []
        self._skip_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        skip = node.name == "_run_fingerprint"
        self._skip_depth += skip
        self.generic_visit(node)
        self._skip_depth -= skip

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if (
            self._skip_depth == 0
            and node.attr == self.field
            and isinstance(node.ctx, ast.Load)
            and _names_config(node.value)
        ):
            self.reads.append(node.lineno)
        self.generic_visit(node)


def _names_config(node: ast.AST) -> bool:
    """True for ``config`` / ``self.config`` / ``cfg`` / ``self.cfg``."""
    if isinstance(node, ast.Name):
        return node.id in {"config", "cfg"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"config", "cfg"}
    return False


def _config_read_sites(field: str) -> list[str]:
    sites: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a broken file is its own failure
            continue
        visitor = _ConfigReads(field)
        visitor.visit(tree)
        sites.extend(f"{path.relative_to(_SRC)}:{line}" for line in visitor.reads)
    return sites


@pytest.mark.parametrize(
    "option", [n for n, (status, _) in CLASSIFIED.items() if status in (AGENTIC, NON_AGENTIC)]
)
def test_a_live_flag_has_a_consumer_beyond_the_fingerprint(option: str) -> None:
    """The mechanical check that caught all three known instances.

    A flag whose config field is read ONLY by `_run_fingerprint` gates nothing.
    That is exactly what `--no-judge-hard-pages` was, and it is strictly worse
    than inert: it changes the run identity, so it invalidates terminal pages
    and forces a reprocess that produces the same output.

    Deliberately weak in one direction and honest about it -- a field can be
    read and still not gate anything useful. It cannot catch that; it catches
    the shape all three known liars had.
    """
    field = _config_field_for(option)
    if field is None:
        pytest.skip(f"{option} has no PipelineConfig field (handled at the CLI layer)")

    # Attribute LOADS only, and never from inside `_run_fingerprint`. Both
    # exclusions are structural: assignments are `ast.Store`, so they cannot be
    # counted, and the fingerprint is excluded by the function it lives in
    # rather than by matching its dict-key text.
    #
    # Both exclusions were found by probing this check against the bug it was
    # written for. With neither, restoring `--no-judge-hard-pages` still PASSED:
    # the CLI's own `config.judge_hard_pages = False` counted as a read site.
    # Writing a field is not reading it.
    gating = _config_read_sites(field)
    assert gating, (
        f"{option} -> config.{field} is only ASSIGNED or copied into the run "
        "fingerprint, never read to gate anything. That is the GH-142 failure "
        "shape: toggling it changes the run identity and forces a reprocess "
        "without changing behaviour. Either give it a consumer, or reject it at "
        "the CLI as #139 and #142 did."
    )


@pytest.mark.parametrize(
    "option", [n for n, (status, _) in CLASSIFIED.items() if status == REJECTED]
)
def test_a_rejected_flag_actually_refuses(option: str, tmp_path: Path) -> None:
    """A flag classified REJECTED must raise, not be quietly accepted.

    Without this, the table could say "rejected" while the CLI kept taking it --
    which is the same lie one level up.
    """
    from click.testing import CliRunner

    pdf = tmp_path / "d.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    flag = "--" + option.replace("_", "-")

    result = CliRunner().invoke(cli, ["process", str(pdf), flag])

    assert result.exit_code != 0, f"{flag} was accepted silently despite being classified rejected"
    assert "GH-" in (result.output or ""), (
        f"{flag} is refused but the message does not point at the ticket "
        f"explaining why: {result.output!r}"
    )


def test_every_classification_is_well_formed() -> None:
    """A mistyped status would bypass every status-specific check below it.

    cubic P2 on #516: `test_a_live_flag_has_a_consumer...` and
    `test_a_rejected_flag_actually_refuses` both select by status, so a typo
    like "agentc" silently exempts a flag from both while still satisfying
    `test_every_process_flag_is_classified`. An empty reason is the same
    failure in slower motion: it looks classified and tells a later reader
    nothing to check against.
    """
    allowed = {AGENTIC, NON_AGENTIC, PLUMBING, REJECTED}
    bad_status = {n: st for n, (st, _) in CLASSIFIED.items() if st not in allowed}
    assert not bad_status, (
        f"unknown status(es) -- these flags are silently exempt from every "
        f"status-specific check: {bad_status}"
    )

    no_reason = [n for n, (_, why) in CLASSIFIED.items() if not why.strip()]
    assert not no_reason, (
        f"classified with no reason: {no_reason}. The reason is what a future "
        "reader checks against the code; without it the entry asserts nothing."
    )
