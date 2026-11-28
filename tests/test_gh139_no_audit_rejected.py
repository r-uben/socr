"""GH-139: ``--no-audit`` must fail loudly, and the dead field behind it is gone.

The flag advertised "Skip quality audit stage" and set ``audit_enabled=False``. The
issue framed it as inert *on the agentic path*, with the real gates living in the
multi-engine and single-engine branches. That framing is out of date: #298 deleted
those branches, so every consumer disappeared and the flag was inert in **every**
mode -- including HPC, since the CLI never wired it to the separate, still-live
``HPCConfig.audit_enabled``.

Resolution 1 of the issue's own preference order: reject it. A flag that lies is
worse than a missing flag, because the user believes a constraint is in force and
scripts around it (#142). The option is retained purely to produce an explanatory
error rather than click's bare "no such option".

``PipelineConfig.audit_enabled`` is deleted outright rather than kept as a
vestigial constant. Keeping it was justified on the grounds that dropping its run
fingerprint key would force a corpus-wide reprocess -- but ``_run_fingerprint``
already hashes ``socr_source_digest``, which covers every shipped ``.py``
*including comments*, so any source change invalidates every fingerprint anyway.
There is no cache-preserving option to protect, and retention would have kept a
public lie (the README advertised the field as configurable).

Testing note (CLAUDE.md): nothing here drives a provider. Each test pins a
DIFFERENCE -- before this fix, passing the flag and omitting it were identical.
"""

from __future__ import annotations

import dataclasses
import pathlib

import click
import pytest
from click.testing import CliRunner

from socr.cli import build_config
from socr.core.config import PipelineConfig


def test_passing_the_flag_now_differs_from_omitting_it():
    """The regression: both paths used to be indistinguishable."""
    assert isinstance(build_config(), PipelineConfig)

    with pytest.raises(click.UsageError):
        build_config(no_audit=True)


def test_the_error_explains_why_rather_than_just_refusing():
    """A user scripting the flag must learn what to do instead."""
    with pytest.raises(click.UsageError) as excinfo:
        build_config(no_audit=True)

    message = str(excinfo.value)
    assert "GH-139" in message
    # Names the reason the flag cannot be honoured on the default path...
    assert "judge" in message.lower()
    # ...and points at the knobs that DO reduce spend.
    assert "--strict-local" in message
    assert "--cost-budget" in message
    # ...without overclaiming: the HPC lane DOES have a separable audit stage,
    # gated by a different setting this flag never wrote to. Saying "no audit is
    # removable anywhere" would be a fresh lie in place of the old one.
    assert "hpc.audit_enabled" in message


@pytest.mark.parametrize("command", ["process", "batch"])
def test_the_real_cli_rejects_the_flag(tmp_path, command):
    """Exercised through click, not just the config builder it happens to call.

    ``build_config`` is an implementation detail; what a user's script invokes is
    the command. A rejection that only fired in the helper would leave the lie
    intact at the surface that matters.
    """
    from socr.cli import cli

    target = tmp_path / ("doc.pdf" if command == "process" else "corpus")
    if command == "process":
        target.write_bytes(b"%PDF-1.4\n")
    else:
        target.mkdir()

    result = CliRunner().invoke(cli, [command, str(target), "--no-audit"])

    assert result.exit_code != 0
    assert "GH-139" in result.output


def test_the_field_is_gone_from_the_config():
    """The vestigial field is deleted, not merely defended.

    Replaces an earlier source-scanning guard: asserting the field's absence is a
    real contract, whereas scanning source text for reads was brittle -- it could
    not see ``from_file``'s generic ``setattr``, and false-failed on the removal
    code itself.
    """
    names = {f.name for f in dataclasses.fields(PipelineConfig)}
    assert "audit_enabled" not in names

    with pytest.raises(TypeError):
        PipelineConfig(audit_enabled=False)


def test_the_hpc_field_of_the_same_name_survives():
    """Guards the one thing the removal must not touch.

    ``HPCConfig.audit_enabled`` is a different field on a different path, genuinely
    read in ``hpc_pipeline.py``. Deleting it would silently disable a real gate.
    """
    from socr.core.config import HPCConfig

    assert HPCConfig().audit_enabled is True


def test_a_config_file_naming_the_removed_field_says_so(tmp_path):
    """A stale config must be told the setting is gone, not that it made a typo.

    ``from_file`` restores fields generically, so before the removal
    ``audit_enabled: false`` in YAML landed on the config, disabled nothing, and
    silently changed the run fingerprint. Every spelling is refused, not just
    ``false``: YAML ``0`` loads as int and ``"false"`` as str.
    """
    for spelling in ("false", "true", "no", "0", '"false"'):
        stale = tmp_path / "stale.yaml"
        stale.write_text(f"audit_enabled: {spelling}\n")
        with pytest.raises(ValueError, match="GH-139") as excinfo:
            PipelineConfig.from_file(stale)
        # Distinguishable from the generic unknown-key error, which would tell the
        # user they mistyped something rather than that the setting was removed.
        assert "Removed setting" in str(excinfo.value)

    # The DIFFERENCE: a file that does not mention the key loads fine.
    fine = tmp_path / "fine.yaml"
    fine.write_text("render_dpi: 200\n")
    assert PipelineConfig.from_file(fine).render_dpi == 200


def test_the_source_digest_is_what_makes_dropping_the_key_free():
    """Dropping the fingerprint key is safe only because the digest already moved.

    Pinned so a future reader does not "restore" the key for cache-stability
    reasons that do not hold. ``_socr_source_digest`` hashes every shipped ``.py``
    including comments, so ANY source edit already invalidates every stored
    fingerprint -- there was never a cache-preserving option to protect.

    Verified by recomputing the digest over the package's current bytes rather than
    by inspecting the fingerprint string (an opaque hash, where a substring check
    would pass vacuously) and without mutating any source file (a killed test run
    must not leave a stray byte in the installed package).
    """
    import hashlib

    import socr
    from socr.pipeline import orchestrator

    root = pathlib.Path(socr.__file__).resolve().parent
    expected = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        expected.update(str(path.relative_to(root)).encode("utf-8"))
        expected.update(b"\x00")
        expected.update(path.read_bytes())
        expected.update(b"\x00")

    assert orchestrator._socr_source_digest() == expected.hexdigest(), (
        "the run fingerprint must depend on socr's own source bytes; if it does "
        "not, the argument for deleting the audit key needs re-examining"
    )
