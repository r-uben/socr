"""Regenerate `tests/fixtures/p6/prechange_assemble.json` from the pre-change sources.

The pin that `tests/test_p6_stage_ab_difference.py` compares against is a capture of
`tests/p6_corpus_fixture.py` run on a source tree that does NOT contain the P6 stage
A/B change. Checked in as a script (cold review round 2, finding 7) so the pin is
reproducible rather than asserted: anyone can rebuild it and diff.

    PYTHONPATH=<worktree>/src ~/venvs/socr/bin/python tests/regenerate_p6_prechange.py

`--rev` picks the pre-change revision (default `HEAD`); `--print-only` writes nothing
and reports the SHA-256 of the normalized capture, which is what a reviewer needs to
confirm the checked-in file's provenance. Not a test module: pytest collects only
`test_*.py`, so this is never run by the suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PIN = HERE / "fixtures" / "p6" / "prechange_assemble.json"

#: Must match `VOLATILE_KEYS` in tests/test_p6_stage_ab_difference.py. Duplicated
#: rather than imported because this script must not import the current tree's test
#: module to normalize a capture taken from a different tree.
VOLATILE_KEYS = frozenset(
    {
        "disposition",
        "socr_source_digest",
        "run_fingerprint",
        "input_checksum",
        "pdf_file_hash",
    }
)

#: Runs `capture()` inside the exported tree. Kept as a string because it executes
#: under a DIFFERENT `sys.path`: the pre-change `src/`, plus this `tests/` directory
#: for the corpus fixture itself (which is written to import from both trees).
_RUNNER = """
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import p6_corpus_fixture as fixture
with tempfile.TemporaryDirectory() as td:
    print(json.dumps(fixture.capture(Path(td)), default=str))
"""


def normalize(obj):
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    return obj


def capture_at(rev: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        export = Path(td)
        archive = export / "src.tar"
        with archive.open("wb") as fh:
            subprocess.run(["git", "archive", rev, "src"], cwd=REPO, stdout=fh, check=True)
        with tarfile.open(archive) as tar:
            tar.extractall(export)  # noqa: S202 -- our own git archive
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(export / "src"), str(HERE)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit(f"capture at {rev} failed with exit {proc.returncode}")
        return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", default="HEAD", help="pre-change revision (default HEAD)")
    parser.add_argument("--print-only", action="store_true", help="report the hash, write nothing")
    args = parser.parse_args()

    normalized = normalize(capture_at(args.rev))
    rendered = json.dumps(normalized, indent=2, sort_keys=True)
    digest = hashlib.sha256(rendered.encode()).hexdigest()

    if args.print_only:
        print(f"{args.rev}: sha256={digest}")
        if PIN.exists():
            current = hashlib.sha256(PIN.read_text().encode()).hexdigest()
            print(f"checked in: sha256={current}")
            print("MATCH" if current == digest else "DIFFERS")
        return 0

    PIN.parent.mkdir(parents=True, exist_ok=True)
    PIN.write_text(rendered)
    print(f"wrote {PIN} ({args.rev}, sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
