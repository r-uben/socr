"""GH-168: an absent CLI option must not clobber a loaded config value.

`build_config` assigned seven options unconditionally, so a value loaded from
`--config` / `--profile` was overwritten by the CLI's own default even when the
user never mentioned the option. Measured before the fix, a config file setting
all seven lost every one:

    cost_budget 5.0 -> 0.0          max_cost_per_page 0.25 -> 0.0
    write_manifest True -> False    timeout 999 -> 1800
    judge_backend heuristic -> auto judge_model my-judge -> ""
    save_figures True -> False

A silently ignored setting is the "flag that lies" failure #142 names: the user
believes a budget is in force and scripts around it.

Driven through the real Click command, because the defect is precisely about
what Click passes when an option is absent -- calling `build_config` directly
cannot distinguish "not supplied" from "supplied with the default".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from socr.cli import cli

CONFIG = """
cost_budget: 5.0
max_cost_per_page: 0.25
write_manifest: true
timeout: 999
judge_backend: heuristic
judge_model: my-judge
save_figures: true
describe_figures: true
# GH-469: four `is_flag` options of the same class, missed by GH-168.
reprocess: true
quiet: true
verbose: true
"""

LOADED = {
    "cost_budget": 5.0,
    "max_cost_per_page": 0.25,
    "write_manifest": True,
    "timeout": 999,
    "judge_backend": "heuristic",
    "judge_model": "my-judge",
    "save_figures": True,
    # GH-168 review: the `--describe-figures` else-branch clears BOTH of these
    # when the flag is absent, so both need a preserve assertion.
    "describe_figures": True,
    "reprocess": True,
    "quiet": True,
    "verbose": True,
}


def _run_with(tmp_path: Path, config_text: str, extra: list[str], monkeypatch):
    fitz = pytest.importorskip("fitz")

    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(config_text)
    pdf = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "born digital text long enough to be a text layer.")
    doc.save(str(pdf))
    doc.close()

    seen: dict = {}

    class _Stub:
        def __init__(self, config):
            seen["config"] = config

        def process(self, *_a, **_k):
            raise SystemExit(0)

        def _resolve_output_root(self, *_a, **_k):
            return tmp_path / "out"

    monkeypatch.setattr("socr.pipeline.orchestrator.UnifiedPipeline", _Stub)
    CliRunner().invoke(cli, ["process", str(pdf), "--config", str(cfg), *extra])
    assert "config" in seen, "the pipeline was never constructed, so nothing was measured"
    return seen["config"]


def _run(tmp_path: Path, extra: list[str], monkeypatch: pytest.MonkeyPatch):
    """Invoke `socr process --config <CONFIG>` and capture the config it built.

    GH-479 review: a thin wrapper over `_run_with` rather than a near-identical
    copy, so the PDF fixture and the capturing stub live in ONE place and cannot
    drift apart.
    """
    return _run_with(tmp_path, CONFIG, extra, monkeypatch)


@pytest.mark.parametrize(("field", "expected"), sorted(LOADED.items()))
def test_an_absent_option_preserves_the_loaded_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, expected
) -> None:
    config = _run(tmp_path, [], monkeypatch)
    assert getattr(config, field) == expected, (
        f"{field}: the CLI default overwrote the value loaded from --config "
        f"({getattr(config, field)!r} instead of {expected!r})"
    )


@pytest.mark.parametrize(
    ("flag", "field", "value"),
    [
        (["--cost-budget", "1.5"], "cost_budget", 1.5),
        (["--max-cost-per-page", "0.05"], "max_cost_per_page", 0.05),
        (["--timeout", "60"], "timeout", 60),
        (["--judge-backend", "vlm"], "judge_backend", "vlm"),
        (["--judge-model", "other"], "judge_model", "other"),
        # The figures pair: `--describe-figures` implies save_figures. The
        # `--save-figures` row below deliberately expects describe_figures to
        # stay TRUE -- see its own comment. (GH-479: this comment used to say
        # the opposite, contradicting the row it introduces.)
        (["--describe-figures"], "describe_figures", True),
        (["--describe-figures"], "save_figures", True),
        # NOT `describe_figures False`: the user typed --save-figures, not
        # "do not describe". Clearing a `describe_figures: true` loaded from the
        # file because an unrelated flag was given is the very clobbering this
        # ticket is about, so the loaded value stands.
        (["--save-figures"], "describe_figures", True),
    ],
)
def test_an_explicit_option_still_overrides_the_loaded_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: list[str], field: str, value
) -> None:
    """The other half: preserving loaded values must not disable the CLI."""
    config = _run(tmp_path, flag, monkeypatch)
    assert getattr(config, field) == value, (
        f"{field}: an explicit {flag[0]} did not override the config file"
    )


def test_a_loaded_dry_run_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GH-469: `dry_run` needs its own test, and the reason is the evidence.

    It cannot ride in the shared fixture above: a preserved `dry_run: true`
    makes the CLI list files instead of constructing the pipeline, so every
    other assertion there loses the config it measures. That the pipeline is
    NEVER constructed is exactly the proof the loaded value survived -- under
    the old unconditional assignment it was cleared to False and the run
    proceeded.
    """
    fitz = pytest.importorskip("fitz")

    d = tmp_path / "dry"
    d.mkdir(parents=True, exist_ok=True)
    cfg = d / "c.yaml"
    cfg.write_text("dry_run: true\n")
    pdf = d / "d.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "born digital text long enough to be a text layer.")
    doc.save(str(pdf))
    doc.close()

    built: dict = {}

    class _Stub:
        def __init__(self, config):
            built["config"] = config

        def process(self, *_a, **_k):
            raise AssertionError("dry_run was cleared: the pipeline ran anyway")

        def _resolve_output_root(self, *_a, **_k):
            # The dry-run path calls this. Omitting it made the command crash
            # with AttributeError right after printing its first line -- and the
            # test still passed, because it only looked for "Would process".
            # #473 review: a test that observes a crash and calls it a pass is
            # the vacuity this whole backlog is about.
            return d / "out"

    monkeypatch.setattr("socr.pipeline.orchestrator.UnifiedPipeline", _Stub)
    result = CliRunner().invoke(cli, ["process", str(pdf), "--config", str(cfg)])

    assert result.exit_code == 0, (
        f"the dry-run path did not complete cleanly: {result.output!r}\n{result.exception!r}"
    )

    # The dry-run listing is the evidence: it only happens when the loaded
    # `dry_run: true` survived. Asserted on the OUTPUT rather than the exit
    # code, which the dry-run path does not define as part of this contract.
    assert "Would process" in result.output, (
        f"the CLI default cleared a `dry_run: true` loaded from --config; the "
        f"run proceeded instead of listing: {result.output!r}"
    )
    assert "config" not in built or built["config"].dry_run is True


# GH-479: preserve was pinned, override was not. A gate that simply ignored the
# CLI would satisfy every preserve test in this file, so the other direction --
# a YAML `false` that an explicit flag must flip ON -- is the half that proves
# `_explicitly_given` still lets the user win.
FALSE_CONFIG = """
reprocess: false
quiet: false
verbose: false
write_manifest: false
save_figures: false
"""


@pytest.mark.parametrize(
    ("flag", "field"),
    [
        (["--reprocess"], "reprocess"),
        (["--quiet"], "quiet"),
        (["--verbose"], "verbose"),
        (["--write-manifest"], "write_manifest"),
        (["--save-figures"], "save_figures"),
    ],
)
def test_an_explicit_flag_overrides_a_loaded_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: list[str], field: str
) -> None:
    """A YAML `false` must not survive the user explicitly asking for it."""
    config = _run_with(tmp_path / field, FALSE_CONFIG, flag, monkeypatch)
    assert getattr(config, field) is True, (
        f"{flag[0]} did not override `{field}: false` from --config"
    )


# GH-479 review: an anchor asserting `<field> is False` with no flag was WRONG
# and is removed rather than reworded. Every one of these fields defaults to
# False in `PipelineConfig`, so that assertion passed identically whether the
# YAML `false` loaded or the field simply kept its default -- an anti-vacuity
# anchor that was itself vacuous, and whose docstring claimed the opposite.
#
# Loading IS proven, by `test_an_absent_option_preserves_the_loaded_value`
# above: it asserts a loaded `true` survives for these same fields, which is
# only possible if the YAML reached the config. Duplicating that here with the
# wrong polarity added nothing.
