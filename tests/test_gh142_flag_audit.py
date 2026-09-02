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
    "dpi": "render_dpi",
}

# Flags that gate BY CONSTRUCTION rather than through a config read, with the
# construct named. An explicit exemption, never a silent skip: the first version
# of this file skipped four flags for want of a field mapping, which is exactly
# how a flag escapes the audit.
#
# `--hpc-sequential` is the one such case. Its `if hpc_sequential:` branch in
# `cli.py` builds HPCPipeline directly off the LOCAL variable, so the flag has a
# real effect while the two config fields it also writes -- `hpc.enabled` and
# `hpc.sequential` -- are read nowhere in the source tree. Those dead fields are
# a smaller finding of this same sweep, filed separately; they are not what
# makes the flag work, so convicting the flag on them would be wrong.
_GATES_BY_CONSTRUCTION = {
    "hpc_sequential": "cli.py builds HPCPipeline from the local variable, not from config",
}

# Reads that do NOT make a flag live for `process`. A value serialised into a
# benchmark report is not consumed by any run -- this is how `--fallback` passed
# the audit while no execution path read it (cubic P2 on #516).
#
# The directory exclusion alone did NOT deliver that (cubic P2, round two): the
# motivating site is `benchmark_calibrate` in `cli.py`, which no `benchmark/`
# prefix covers. The exclusion is by FUNCTION for that reason, the same way the
# fingerprint one is -- naming the construct rather than hoping a path prefix
# happens to contain it.
_NON_EXECUTION_DIRS = ("benchmark/",)
#: `_warn_inert_config` reads the inert fields to REPORT that they are ignored
#: (GH-525). Counting that as a consumer would certify the very fields whose
#: deadness this audit established -- the run's own "I am ignoring this" would
#: become the evidence that it is not.
_NON_EXECUTION_FUNCTIONS = frozenset({"benchmark_calibrate", "_warn_inert_config"})

# Functions that compute the run FINGERPRINT rather than gate behaviour. A read
# reached only through one of these records the value into the run identity,
# which is the not-gating-anything case this audit exists to catch. The two
# helpers are named because `_run_fingerprint` calls out to them, so excluding
# only its own subtree let a fingerprint-only field pass (cubic P2 on #516).
_FINGERPRINT_FUNCTIONS = frozenset(
    {"_run_fingerprint", "_engine_determinants", "_resolve_primary_engine"}
)

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
    # Landed on main after this branch was cut, and CI caught it here before it
    # reached the audit's blind spot -- the guard's first real firing, on the
    # exact case it was written for. Verified live rather than assumed: read at
    # orchestrator.py:462 and :1549, both gating on `and self.config.agentic`.
    "no_equation_region_lane": (AGENTIC, "equation_region_lane, gated in the agentic lane"),
    # -- non-agentic by design
    "fallback": (REJECTED, "GH-142: no execution reader; raises UsageError"),
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
    """The config path a flag writes: ``"quiet"``, or ``"hpc.sequential"``."""
    fields = {f.name for f in dataclasses.fields(PipelineConfig)}
    for candidate in (
        _FIELD_ALIASES.get(option_name, option_name),
        option_name.removeprefix("no_"),
    ):
        # A dotted alias names a nested section; only its ROOT has to be a
        # field of PipelineConfig.
        if candidate.split(".", 1)[0] in fields:
            return candidate
    return None


class _ConfigReads(ast.NodeVisitor):
    """Attribute LOADS of ``config.<field>``, excluding the run fingerprint.

    cubic P2 on #516: the first version grepped source text, so a comment or a
    docstring mentioning ``config.foo`` certified a dead flag as live, and a
    read spelled across a line break was invisible. Text matching is the wrong
    instrument for "is this value used"; the parse tree is the right one.

    The fingerprint exclusion is structural rather than a string test: a read
    inside ``_run_fingerprint`` -- or inside the helpers it calls, which is
    where several such reads actually live -- records the value into the run
    identity, which is precisely the not-gating-anything case this check is
    for. See ``_FINGERPRINT_FUNCTIONS``.
    """

    def __init__(self, path: str) -> None:
        # "quiet" or "hpc.sequential": the last element is the attribute read,
        # anything before it is the section chain below `config`.
        *self.section, self.field = path.split(".")
        self.reads: list[int] = []
        self._skip_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        skip = node.name in _FINGERPRINT_FUNCTIONS or node.name in _NON_EXECUTION_FUNCTIONS
        self._skip_depth += skip
        self.generic_visit(node)
        self._skip_depth -= skip

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if (
            self._skip_depth == 0
            and node.attr == self.field
            and isinstance(node.ctx, ast.Load)
            and _is_config_path(node.value, self.section)
        ):
            self.reads.append(node.lineno)
        self.generic_visit(node)


def _is_config_path(node: ast.AST, section: list[str]) -> bool:
    """True when *node* is ``config`` / ``cfg`` followed by *section*.

    With ``section == ["hpc"]`` this matches ``config.hpc`` and
    ``self.config.hpc`` but not ``config`` alone, so a nested field is only
    credited to reads of that nested field.
    """
    for name in reversed(section):
        if not isinstance(node, ast.Attribute) or node.attr != name:
            return False
        node = node.value
    if isinstance(node, ast.Name):
        return node.id in {"config", "cfg"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"config", "cfg"}
    return False


def _flag_constructs(option: str, class_name: str, *, command: str = "process") -> bool:
    """True when ``if <option>:`` inside *command* SELECTS *class_name*.

    Three narrowings, each closing a way the check could pass without proving
    anything (cubic P2 on #516, round three):

    - only within the `command` function, so a same-named branch in another
      subcommand cannot satisfy the exemption for `process`;
    - only the TAKEN body (`node.body`), never `node.orelse` -- walking the
      whole `If` counted the else branch, and here the else builds
      `UnifiedPipeline`, so the opposite branch's constructor would have
      satisfied it;
    - a call node, not a name, so a comment, an import or a type annotation
      mentioning the class proves nothing.
    """
    tree = ast.parse((_SRC / "cli.py").read_text())
    command_fn = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == command
        ),
        None,
    )
    if command_fn is None:
        return False

    for node in ast.walk(command_fn):
        if not isinstance(node, ast.If):
            continue
        if not (isinstance(node.test, ast.Name) and node.test.id == option):
            continue
        for statement in node.body:
            for inner in ast.walk(statement):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == class_name
                ):
                    return True
    return False


def _config_read_sites(field_path: str) -> list[str]:
    sites: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        rel = str(path.relative_to(_SRC))
        if rel.startswith(_NON_EXECUTION_DIRS):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a broken file is its own failure
            continue
        visitor = _ConfigReads(field_path)
        visitor.visit(tree)
        sites.extend(f"{rel}:{line}" for line in visitor.reads)
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
    if option in _GATES_BY_CONSTRUCTION:
        # Not a skip, and not a rubber stamp either (cubic P2 on #516): a
        # non-empty justification string proved nothing, so the audit would have
        # stayed green if the HPC branch stopped building HPCPipeline. The
        # construct the exemption NAMES is what gets asserted.
        assert _GATES_BY_CONSTRUCTION[option].strip()
        assert _flag_constructs(option, "HPCPipeline"), (
            f"{option} is exempted from the config-consumer check because it "
            "gates by construction -- but its branch no longer constructs "
            "HPCPipeline, so it now gates nothing at all. Re-audit it."
        )
        return

    field = _config_field_for(option)
    assert field is not None, (
        f"{option} maps to no PipelineConfig field. Either add it to "
        "_FIELD_ALIASES, or -- if it gates by construction rather than through "
        "config -- to _GATES_BY_CONSTRUCTION with the construct named. Leaving "
        "it unmapped would exempt it from this audit silently."
    )

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

    # A value-taking option needs one, or click refuses it for the wrong reason
    # and the test would pass on click's own error rather than ours.
    param = next(p for p in cli.commands["process"].params if p.name == option)
    args = ["process", str(pdf), flag]
    if not getattr(param, "is_flag", False):
        choices = getattr(param.type, "choices", None)
        args.append(choices[0] if choices else "qwen")

    result = CliRunner().invoke(cli, args)

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


def test_an_ignored_field_reporter_does_not_certify_a_flag() -> None:
    """The other exclusion, and the leg the benchmark test does not cover.

    cubic P3 on #529: `_warn_inert_config` reads BOTH inert fields to report
    that they are ignored. The benchmark control happens to cover
    `fallback_chain`; nothing covered `judge_hard_pages`, so renaming or
    dropping `_warn_inert_config` would silently let that reporter read certify
    a dead field as live -- the run's own "I am ignoring this" becoming the
    evidence that it does not.
    """
    import test_gh142_flag_audit as module

    assert module._config_read_sites("judge_hard_pages") == [], (
        "a read that exists to report the field is ignored is being counted as "
        "an execution consumer"
    )

    original = module._NON_EXECUTION_FUNCTIONS
    try:
        module._NON_EXECUTION_FUNCTIONS = frozenset()
        without = module._config_read_sites("judge_hard_pages")
    finally:
        module._NON_EXECUTION_FUNCTIONS = original

    assert without, (
        "the control failed: with the exclusion removed there is no reporter "
        "read left to exclude, so the assertion above proves nothing"
    )


def test_a_benchmark_report_read_does_not_certify_a_flag() -> None:
    """The exclusion the audit's guarantee rests on, verified directly.

    cubic P2 on #516 (round two): the first attempt excluded the `benchmark/`
    DIRECTORY, but the motivating site -- `benchmark_calibrate` serialising
    `config.fallback_chain` into a report -- lives in `cli.py`, which no path
    prefix covers. The claim was documented and unenforced.

    `fallback_chain` is the case: outside the fingerprint, its only read in the
    whole tree is that serialisation, of a FRESH PipelineConfig that never sees
    the user's value. Asserting through a flag would not isolate this (the flag
    is rejected now, so it has no reads at all either way), so this measures the
    function directly.
    """
    import test_gh142_flag_audit as module

    assert module._config_read_sites("fallback_chain") == [], (
        "a benchmark-report serialisation is being counted as an execution "
        "consumer; a flag read only there would pass the audit"
    )

    original = module._NON_EXECUTION_FUNCTIONS
    try:
        module._NON_EXECUTION_FUNCTIONS = frozenset()
        without = module._config_read_sites("fallback_chain")
    finally:
        module._NON_EXECUTION_FUNCTIONS = original

    assert without, (
        "the control failed: with the exclusion removed there is no benchmark "
        "read left to exclude, so the test above proves nothing"
    )


@pytest.mark.parametrize(
    "option", [n for n, (status, _) in CLASSIFIED.items() if status == REJECTED]
)
def test_a_rejected_flag_does_not_advertise_the_old_behaviour(option: str) -> None:
    """GH-524: `--help` is the surface most users read, and it still lied.

    All three rejected flags raise on invoke, but two of them described live
    behaviour in `common_options` -- "Fallback OCR engine", "Disable VLM judge
    on hard pages". A user who reads the help and never invokes gets exactly
    the failure #142 is about: believing a constraint exists.

    The wording is not pinned; the CLAIM is. A help string must say the flag is
    removed, which is what `--no-audit` already did and what the other two now
    copy.
    """
    param = next(p for p in cli.commands["process"].params if p.name == option)
    help_text = (param.help or "").upper()

    assert "REMOVED" in help_text or "REJECT" in help_text, (
        f"--{option.replace('_', '-')} raises on invoke but its help text still "
        f"advertises behaviour it does not have: {param.help!r}"
    )


@pytest.mark.parametrize(
    "option", [n for n, (status, _) in CLASSIFIED.items() if status == REJECTED]
)
def test_the_readme_does_not_advertise_a_rejected_flag(option: str) -> None:
    """GH-528: the third surface to tell the same lie.

    `--fallback` and `--no-judge-hard-pages` raise on invoke (GH-142), and their
    `--help` was corrected (GH-524) -- and the README's CLI reference still
    listed `--fallback` as "Fallback engine". Three surfaces, fixed one ticket
    at a time, each after someone noticed.

    So this checks the README rather than fixing it again: a rejected flag that
    appears there must be marked removed. Absent is fine too -- a reference that
    omits a dead flag is not lying about it.
    """
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    flag = "--" + option.replace("_", "-")

    # A reference entry is a line whose FIRST TOKEN is the flag, once the
    # markdown decoration is stripped. Not any prose that happens to name it --
    # a sentence explaining why it was removed is not an advertisement and
    # should not have to shout REMOVED to pass.
    #
    # cubic P2 on #531: matching the raw line's prefix let any other layout
    # escape -- `` `--fallback` `` in backticks, a `- --fallback` list item, a
    # table cell -- and the check would return green while the README told the
    # exact lie it exists to prevent. A format tweak must not be able to
    # silently disable it.
    def _entry_flag(line: str) -> str:
        stripped = line.strip().lstrip("-*|` \t")
        head = stripped.split(maxsplit=1)[0] if stripped.split() else ""
        return "--" + head.lstrip("-").strip("`,|")

    mentions = [line for line in readme.splitlines() if _entry_flag(line) == flag]
    if not mentions:
        return

    for line in mentions:
        assert "REMOVED" in line.upper() or "REJECT" in line.upper(), (
            f"README lists {flag} as if it worked: {line.strip()!r}. It raises "
            "on invoke; a reader of the docs alone would believe the constraint "
            "exists."
        )
