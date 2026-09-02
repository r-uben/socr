"""GH-517: `hpc.enabled` selects the HPC lane, or it should not exist.

Found by the GH-142 flag sweep. `--hpc-sequential` wrote `config.hpc.enabled`
and `config.hpc.sequential` and then built `HPCPipeline` from the LOCAL
variable, so the flag worked while both fields were read nowhere in the source
tree. `hpc.enabled: true` in a config file did nothing, and the lane was
reachable only through the flag.

A config key that looks like a switch and is not one misleads exactly the way a
flag does -- which is GH-142's whole thesis, one layer down. Its sibling
`hpc.audit_enabled` IS read, which made the dead ones more convincing, not less.

Both halves are handled, differently, because they are not the same case:

- `hpc.enabled` becomes REAL: it selects the pipeline, from YAML as well as from
  the flag.
- `hpc.sequential` is REFUSED when false. There is one HPC mode -- sequential
  model loading for a single GPU is what HPCPipeline implements -- so `false`
  asks for something that does not exist. Running the sequential pipeline anyway
  would be the flag-that-lies failure at the YAML layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from socr.cli import cli


def _pdf(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "d.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    return pdf


def _config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "socr.yaml"
    path.write_text(body)
    return path


def _run(tmp_path: Path, body: str):
    """Invoke `process` with a config file, capturing which pipeline was built."""
    from unittest.mock import patch

    built: list[str] = []

    class _Recorder:
        def __init__(self, config):
            built.append(type(self).__name__)
            self.config = config

        def process(self, *_a, **_k):
            raise RuntimeError("stop here: the pipeline choice is what is measured")

    hpc = type("HPCPipeline", (_Recorder,), {})
    unified = type("UnifiedPipeline", (_Recorder,), {})

    with (
        patch("socr.pipeline.hpc_pipeline.HPCPipeline", hpc),
        patch("socr.pipeline.orchestrator.UnifiedPipeline", unified),
    ):
        result = CliRunner().invoke(
            cli, ["process", str(_pdf(tmp_path)), "--config", str(_config(tmp_path, body))]
        )
    return built, result


def test_a_config_file_can_reach_the_hpc_lane(tmp_path: Path) -> None:
    """The fix: `hpc.enabled: true` selects HPCPipeline."""
    built, _result = _run(tmp_path / "enabled", "hpc:\n  enabled: true\n  sequential: true\n")

    assert built == ["HPCPipeline"], (
        f"hpc.enabled: true did not select the HPC lane (built {built}); the "
        "key looks like a switch and controls nothing"
    )


def test_the_default_still_takes_the_agentic_pipeline(tmp_path: Path) -> None:
    """Control. Without it, a change that always chose HPCPipeline would satisfy
    the test above while breaking every ordinary run."""
    built, _result = _run(tmp_path / "default", "quiet: true\n")

    assert built == ["UnifiedPipeline"], (
        f"a default config no longer takes the agentic path: {built}"
    )


def test_hpc_enabled_false_is_not_the_hpc_lane(tmp_path: Path) -> None:
    """The other direction: the key must be read, not merely present."""
    built, _result = _run(tmp_path / "off", "hpc:\n  enabled: false\n  sequential: true\n")

    assert built == ["UnifiedPipeline"], f"hpc.enabled: false selected the HPC lane anyway: {built}"


def test_a_non_sequential_hpc_config_is_refused(tmp_path: Path) -> None:
    """`sequential: false` asks for a mode that does not exist.

    Silently running the sequential pipeline would be the failure this ticket
    family is about: a setting that reads as a choice and is not one. The error
    must say which key and what to do, not just fail.
    """
    built, result = _run(tmp_path / "nonseq", "hpc:\n  enabled: true\n  sequential: false\n")

    assert result.exit_code != 0, (
        f"hpc.sequential: false was accepted (built {built}); the run would have "
        "used sequential loading while the config said otherwise"
    )
    assert "hpc.sequential" in result.output, (
        f"the refusal does not name the key that caused it: {result.output!r}"
    )
    assert built == [], f"a pipeline was built before the config was rejected: {built}"


def test_batch_refuses_an_hpc_config_rather_than_ignoring_it(tmp_path: Path) -> None:
    """cubic P2 on #535: making `hpc.enabled` authoritative for ONE command.

    `process` honours it now. `batch` does not implement the HPC lane at all --
    `HPCPipeline` has no `process_batch` -- so accepting the config there would
    run the agentic pipeline while the config asked for HPC. That is the same
    silent-ignore this ticket exists to remove, newly created by fixing it one
    command over.

    Refused rather than routed: routing would mean inventing a batch loop for a
    pipeline that has none, which is a feature rather than a bug fix.
    """
    tmp = tmp_path / "batch_hpc"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "d.pdf").write_bytes(b"%PDF-1.4\n")

    result = CliRunner().invoke(
        cli,
        [
            "batch",
            str(tmp),
            "--config",
            str(_config(tmp, "hpc:\n  enabled: true\n  sequential: true\n")),
        ],
    )

    assert result.exit_code != 0, (
        "batch accepted an HPC config and would have run the agentic pipeline "
        "while the config asked for HPC"
    )
    assert "hpc.enabled" in result.output, (
        f"the refusal does not name the setting that caused it: {result.output!r}"
    )


def test_batch_still_works_without_an_hpc_config(tmp_path: Path) -> None:
    """Control: the refusal must be scoped to HPC configs, not to batch."""
    from unittest.mock import patch

    tmp = tmp_path / "batch_plain"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "d.pdf").write_bytes(b"%PDF-1.4\n")

    built: list[str] = []

    class _Unified:
        def __init__(self, config):
            built.append("UnifiedPipeline")
            self.config = config
            self.last_outcome = None

        def process_batch(self, *_a, **_k):
            return []

    with patch("socr.pipeline.orchestrator.UnifiedPipeline", _Unified):
        result = CliRunner().invoke(
            cli, ["batch", str(tmp), "--config", str(_config(tmp, "quiet: true\n"))]
        )

    assert built == ["UnifiedPipeline"], (
        f"an ordinary batch no longer builds the agentic pipeline: {built}, "
        f"output={result.output!r}"
    )
