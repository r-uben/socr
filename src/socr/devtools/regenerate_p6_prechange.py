"""Regenerate the P6 pre-change capture from a source checkout.

This is intentionally a developer tool, rather than part of socr's runtime
pipeline.  It must be run from (or below) a checkout containing
``tests/p6_corpus_fixture.py``.  The requested revision contributes only its
``src/`` tree; the corpus fixture always comes from the checkout that launches
the command.

Typical use::

    PYTHONPATH=/Users/rubenffuertes/repos/tools/socr-p6c/src uv run socr-regenerate-p6-prechange

``--print-only`` reports the normalized capture's SHA-256 and does not write
the checked-in fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# Keep in step with ``tests/p6_stage_c_oracle.py``: only fields that vary by
# environmental source build or PDF generation timestamp are excised.
# ``disposition`` is captured, not stripped -- it is the public field the P6
# stage-C comparison exists to check.
VOLATILE_KEYS = frozenset(
    {
        "socr_source_digest",
        "run_fingerprint",
        "input_checksum",
        "pdf_file_hash",
    }
)

# Runs ``capture()`` inside the exported tree.  The isolated interpreter makes
# the source selection explicit: the exported package is placed ahead of any
# installed package, while the fixture is supplied separately from this
# checkout.  The assertion on ``socr.__file__`` prevents an editable or
# otherwise installed socr package from silently satisfying the import.
_RUNNER = r"""
import json, sys, tempfile
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
fixture_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(fixture_root))
sys.path.insert(0, str(source_root))

import socr

package_root = source_root / "socr"
try:
    Path(socr.__file__).resolve().relative_to(package_root)
except ValueError as exc:
    raise RuntimeError(
        f"archived capture imported socr from {socr.__file__!s}, "
        f"not from {package_root!s}"
    ) from exc

import p6_corpus_fixture as fixture

expected_fixture = fixture_root / "p6_corpus_fixture.py"
if Path(fixture.__file__).resolve() != expected_fixture:
    raise RuntimeError(
        f"archived capture imported the corpus fixture from {fixture.__file__!s}, "
        f"not from {expected_fixture!s}"
    )

with tempfile.TemporaryDirectory() as td:
    print(json.dumps(fixture.capture(Path(td)), default=str))
"""


def _clean_environment() -> dict[str, str]:
    """Return the small environment needed by git and the hermetic runner."""
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }


def _source_checkout() -> tuple[Path, Path]:
    """Find the launching checkout and its corpus fixture.

    The module is packaged, so its own ``__file__`` cannot identify the source
    checkout after installation.  Requiring the fixture on the cwd's ancestor
    path also gives a direct, actionable error when the command is run from a
    wheel, another repository, or an arbitrary directory.
    """
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        fixture = candidate / "tests" / "p6_corpus_fixture.py"
        if fixture.is_file():
            return candidate, fixture
    raise RuntimeError(
        "socr-regenerate-p6-prechange is source-checkout-only: "
        "tests/p6_corpus_fixture.py was not found from the current directory"
    )


def normalize(obj):
    if isinstance(obj, dict):
        return {key: normalize(value) for key, value in obj.items() if key not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [normalize(value) for value in obj]
    return obj


def capture_at(rev: str) -> dict:
    repo, fixture = _source_checkout()
    with tempfile.TemporaryDirectory() as td:
        export = Path(td)
        archive = export / "src.tar"
        try:
            with archive.open("wb") as fh:
                subprocess.run(
                    ["git", "archive", rev, "src"],
                    cwd=repo,
                    stdout=fh,
                    check=True,
                    env=_clean_environment(),
                )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"git archive failed for revision {rev!r}") from exc

        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if not members or any(
                member.name != "src" and not member.name.startswith("src/") for member in members
            ):
                raise RuntimeError("git archive did not contain only the requested src/ tree")
            tar.extractall(export)  # noqa: S202 -- archive was produced by our git checkout

        proc = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER, str(export / "src"), str(fixture.parent)],
            capture_output=True,
            text=True,
            check=False,
            env=_clean_environment(),
        )
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or "no diagnostic output"
            raise RuntimeError(f"capture at {rev} failed with exit {proc.returncode}: {detail}")
        output = proc.stdout.strip()
        if not output:
            raise RuntimeError(f"capture at {rev} produced no JSON output")
        try:
            return json.loads(output.splitlines()[-1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"capture at {rev} produced invalid JSON: {output!r}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", default="HEAD", help="pre-change revision (default HEAD)")
    parser.add_argument("--print-only", action="store_true", help="report the hash, write nothing")
    args = parser.parse_args(argv)

    try:
        repo, _ = _source_checkout()
        normalized = normalize(capture_at(args.rev))
    except (OSError, RuntimeError) as exc:
        print(f"socr-regenerate-p6-prechange: error: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(normalized, indent=2, sort_keys=True)
    digest = hashlib.sha256(rendered.encode()).hexdigest()
    pin = repo / "tests" / "fixtures" / "p6" / "prechange_assemble.json"

    if args.print_only:
        print(f"{args.rev}: sha256={digest}")
        if pin.exists():
            current = hashlib.sha256(pin.read_text().encode()).hexdigest()
            print(f"checked in: sha256={current}")
            print("MATCH" if current == digest else "DIFFERS")
        return 0

    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(rendered)
    print(f"wrote {pin} ({args.rev}, sha256={digest})")
    return 0
