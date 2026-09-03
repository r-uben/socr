"""P1 prep item 2 (plan tasks t10-t12): the golden/replay/byte-identity flag
audit's regression guard.

The fingerprint binds ``table_judge_ladder`` (among other things), so
flipping the DEFAULT reprocesses every document -- expected, and not what
this guards against. The real risk (design record, "Flip mechanics"): any
golden/byte-identity/replay test that default-constructs ``PipelineConfig``
without pinning the flag would, after a future default flip, run the ladder
gate in CI with unreachable rungs -- turning every table page UNVERIFIED and
changing goldens in a way that additionally becomes MACHINE-DEPENDENT
(ollama present locally, absent in CI: the #253/#257 trap, at suite scale).

This file is the AST regression guard (t12), not the audit itself (t10-t11).
The audit -- walking every in-scope module, running each fixture under both
flag states with an empty rung list, classifying moved vs. unaffected, and
pinning every moved constructor with the one-line comment naming
``docs/log/2026-09-03_p1-prep-latch-and-audit.md`` -- is implementation
work belonging to t10/t11 and the decision log, not to a test file. The tuple
below is the resulting deliberate scope: a new golden/replay/byte-identity
module must be audited and added explicitly.

Design contract this guard enforces, independent of the exact module list:
  * every ``PipelineConfig(...)`` call in an enumerated module must forward
    an explicit ``table_judge_ladder=`` keyword (True or False; the point is
    that the flag is not left to its default so a future default flip
    cannot silently move the test).
  * shared ``_config``/``_make_config``-style helper constructors must
    accept and forward an explicit ``table_judge_ladder`` parameter, not
    hide it behind an unanalyzable ``**overrides``/``**defaults`` mapping.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from socr.core.config import PipelineConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The exact golden/replay/byte-identity module paths audited in t10/t11 per
# docs/log/2026-09-03_p1-prep-latch-and-audit.md. Keep this explicit: a glob
# would silently widen the future-default regression surface.
#
# Cold review round 1, finding 2: the list is paths, NOT ``test_*.py`` names.
# ``tests/p6_corpus_fixture.py`` is the P6 assemble corpus builder -- a shared
# helper that pytest never collects, so any ``test_*.py``-shaped scan would miss
# it while ``tests/test_p6_stage_ab_difference.py`` compares its five captured
# surfaces byte-for-byte against a stored pre-change baseline. It is the ONLY
# non-``test_*.py`` module under ``tests/`` that constructs ``PipelineConfig``
# (enumerated mechanically); if another appears, audit it and add it here.
_AUDITED_GOLDEN_MODULES: tuple[str, ...] = (
    "tests/p6_corpus_fixture.py",
    "tests/test_p6_cold_review_round2.py",
    "tests/test_p6_disposition_finalization.py",
    "tests/test_p3_judged_bytes_ship.py",
    "tests/test_p35_cold_review_round1.py",
    "tests/test_p35_cold_review_round2.py",
    "tests/test_p35_cold_review_round4.py",
    "tests/test_p35_cold_review_round5.py",
    "tests/test_ladder_e2e.py",
    "tests/test_table_repair_parity.py",
    "tests/test_gh317_structure_class_floor.py",
    "tests/test_gh190_empty_table_surfacing.py",
    "tests/test_gh259_flagged_model_table_wins.py",
)


def _iter_pipelineconfig_calls(tree: ast.AST, source_names: set[str]):
    """Yield every AST ``Call`` node that constructs ``PipelineConfig``.

    Matches a bare ``PipelineConfig(...)`` call bound to one of
    ``source_names`` (the names ``PipelineConfig`` was imported as in this
    module, including aliases), and ``<module>.PipelineConfig(...)``
    attribute-access forms regardless of the module alias.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in source_names:
            yield node
        elif isinstance(func, ast.Attribute) and func.attr == "PipelineConfig":
            yield node


def _imported_pipelineconfig_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "PipelineConfig":
                    names.add(alias.asname or alias.name)
    return names


def _has_pinned_kwarg(call: ast.Call) -> bool:
    return any(kw.arg == "table_judge_ladder" for kw in call.keywords)


def _has_star_kwargs_only(call: ast.Call) -> bool:
    """A call whose only keyword material is a ``**mapping`` -- the guard
    cannot see inside it, so per t11 this shape is itself a violation: the
    flag must be forwarded as a NAMED keyword, never hidden in an opaque
    mapping."""
    return bool(call.keywords) and all(kw.arg is None for kw in call.keywords)


def find_unpinned_constructors(source: str, filename: str = "<memory>") -> list[str]:
    """Return ``filename:lineno`` for every unpinned ``PipelineConfig(...)`` call.

    Public so both this module's own guard test and the negative-control
    tests below exercise the exact same finder -- there is no separate,
    unverified "prose-only" claim about what the guard would catch.
    """
    tree = ast.parse(source, filename=filename)
    names = _imported_pipelineconfig_names(tree) | {"PipelineConfig"}
    violations = []
    for call in _iter_pipelineconfig_calls(tree, names):
        if _has_star_kwargs_only(call) or not _has_pinned_kwarg(call):
            violations.append(f"{filename}:{call.lineno}")
    return violations


# ---------------------------------------------------------------------------
# The guard: every enumerated golden/replay/byte-identity module pins the flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_path", _AUDITED_GOLDEN_MODULES)
def test_enumerated_golden_module_pins_the_ladder_flag(module_path: str) -> None:
    path = _REPO_ROOT / module_path
    if not path.exists():
        pytest.fail(
            f"{module_path} is enumerated in _AUDITED_GOLDEN_MODULES but no longer exists -- "
            "update the audited list (and the decision log) rather than leaving a dead entry"
        )
    source = path.read_text(encoding="utf-8")
    violations = find_unpinned_constructors(source, filename=module_path)
    assert not violations, (
        f"{module_path} constructs PipelineConfig without an explicit table_judge_ladder= "
        f"pin at: {violations}. Per docs/log/2026-09-03_p1-prep-latch-and-audit.md, every "
        "golden/byte-identity/replay fixture must pin the flag explicitly to False (with a "
        "one-line comment naming that audit) so a future default flip cannot silently move "
        "this test or make it machine-dependent (the #253/#257 trap)."
    )


def test_audited_list_matches_the_decision_log() -> None:
    """The decision log (t15) must record the SAME module list this guard
    enforces -- otherwise the audit's paper trail and the guard's actual
    coverage can silently diverge."""
    log_path = _REPO_ROOT / "docs" / "log" / "2026-09-03_p1-prep-latch-and-audit.md"
    if not log_path.exists():
        pytest.skip(
            "decision log not yet written (t15) -- this check is only meaningful once it exists"
        )
    text = log_path.read_text(encoding="utf-8")
    missing = [m for m in _AUDITED_GOLDEN_MODULES if m not in text]
    assert not missing, (
        f"the decision log does not mention {missing}, which this guard enforces -- "
        "the audited list recorded in the log must match the guard's tuple verbatim"
    )


# ---------------------------------------------------------------------------
# Negative-control: the finder itself is exercised on synthetic sources, so a
# broken/no-op scanner would fail even if every real module happened to
# already be pinned (or not yet contain the risky shape).
# ---------------------------------------------------------------------------


def test_finder_flags_a_direct_unpinned_constructor() -> None:
    source = (
        "from socr.core.config import PipelineConfig\n\n"
        "def test_something():\n"
        "    cfg = PipelineConfig(agentic=True, quiet=True)\n"
    )
    violations = find_unpinned_constructors(source, filename="negative_control.py")
    assert violations == ["negative_control.py:4"]


def test_finder_flags_an_overrides_helper_hiding_the_flag_in_star_kwargs() -> None:
    source = (
        "from socr.core.config import PipelineConfig\n\n"
        "def _make_config(**overrides):\n"
        "    kwargs = dict(agentic=True, quiet=True)\n"
        "    kwargs.update(overrides)\n"
        "    return PipelineConfig(**kwargs)\n"
    )
    violations = find_unpinned_constructors(source, filename="negative_control.py")
    assert violations == ["negative_control.py:6"]


def test_finder_accepts_an_explicit_named_pin() -> None:
    source = (
        "from socr.core.config import PipelineConfig\n\n"
        "def test_something():\n"
        "    cfg = PipelineConfig(agentic=True, table_judge_ladder=False)\n"
    )
    assert find_unpinned_constructors(source, filename="ok.py") == []


def test_finder_accepts_an_aliased_import() -> None:
    source = (
        "from socr.core.config import PipelineConfig as Cfg\n\n"
        "def test_something():\n"
        "    c = Cfg(agentic=True, table_judge_ladder=True)\n"
    )
    assert find_unpinned_constructors(source, filename="ok.py") == []


def test_finder_flags_a_module_qualified_attribute_call() -> None:
    source = (
        "from socr.core import config\n\n"
        "def test_something():\n"
        "    c = config.PipelineConfig(agentic=True)\n"
    )
    violations = find_unpinned_constructors(source, filename="qualified.py")
    assert violations == ["qualified.py:4"]


# ---------------------------------------------------------------------------
# Default-value contract (unchanged by this task -- the flag stays off)
# ---------------------------------------------------------------------------


def test_default_pipeline_config_has_the_ladder_on() -> None:
    """P1 (owner ruling Q3, 2026-09-03): the default flipped to True.

    This module's REAL job -- the AST guard below, which forces every golden /
    byte-identity / replay module to pin the flag explicitly -- is unchanged
    and is what made the flip safe. This one assertion existed to record the
    pre-flip state; it now records the post-flip one.
    """
    assert PipelineConfig().table_judge_ladder is True
