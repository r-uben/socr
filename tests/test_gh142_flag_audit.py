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

import dataclasses
import re
import subprocess
from pathlib import Path

import click
import pytest

from socr.cli import cli
from socr.core.config import PipelineConfig

_SRC = Path(__file__).resolve().parents[1] / "src" / "socr"

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
    for candidate in (option_name, option_name.removeprefix("no_")):
        if candidate in fields:
            return candidate
    return None


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

    found = subprocess.run(
        ["grep", "-rn", rf"config\.{field}\b", str(_SRC)],
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert found, f"{option} -> config.{field} is never read anywhere; the flag is dead"

    # Two kinds of match are not consumption, and excluding them is what makes
    # this check bite. Verified by probe: with only the fingerprint line
    # excluded, restoring the `--no-judge-hard-pages` bug still PASSED, because
    # the CLI's own `config.judge_hard_pages = False` counted as a read site.
    # Writing a field is not reading it.
    gating = [
        line
        for line in found
        if f'"{field}":' not in line  # the run-fingerprint dict
        and not re.search(rf"config\.{field}\s*=[^=]", line)  # an assignment
    ]
    assert gating, (
        f"{option} -> config.{field} is only ASSIGNED or copied into the run "
        "fingerprint, never read to gate anything. That is the GH-142 failure "
        "shape: toggling it changes the run identity and forces a reprocess "
        "without changing behaviour. Either give it a consumer, or reject it at "
        f"the CLI as #139 and #142 did. Sites found:\n" + "\n".join(found)
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
