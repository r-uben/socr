"""The committed lane-comparison records must never carry corpus content.

The measurements are committed as derived verdicts only: the corpus is copyrighted
and this repo is public. An earlier revision of the 2026-08-20 commit claimed a guard
like this existed when it did not, and shipped nine absolute paths exposing the
corpus location. This is the guard, so the claim is true and stays true.

Scope, stated precisely because an earlier version of THIS file overstated it. The
key allowlist is applied to every ``*verdict*.json`` in ``docs/log`` -- matching on
the word rather than on one exact suffix, so a file named ``-verdict.json`` or
``-verdicts-v2.json`` is covered too. The absolute-path check is applied to every
``*lane-comparison*`` file regardless of extension, so the prose record and the
runner are covered as well as the JSON.

What it still cannot do is stop a leak in a file named nothing like either pattern,
or corpus prose sitting inside an allowed field. It is a tripwire on the shapes this
record actually takes, not a proof of confidentiality.
"""

from __future__ import annotations

import json
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "docs" / "log"
VERDICT_FILES = sorted(LOG.glob("*verdict*.json"))
RECORD_FILES = sorted(LOG.glob("*lane-comparison*"))
MANIFEST = LOG / "2026-08-20_lane-comparison-manifest.json"

ALLOWED_VERDICT_KEYS = {
    "doc",
    "page",
    "kind",
    "shipped_engine",
    "page_status",
    "flags",
    "candidates",
    "decimals",
}

# Any of these beginning a path component means a filesystem location escaped into
# the record. ``/Users/`` and ``/private/tmp`` alone were the 2026-08-20 checks and
# would pass a Linux home directory straight through.
ABSOLUTE_PATH_MARKERS = ("/Users/", "/home/", "/private/", "/var/folders/", "/tmp/")


def test_at_least_one_record_and_one_verdict_file_are_committed():
    """Otherwise every loop below would pass vacuously over an empty set."""
    assert VERDICT_FILES, "no *verdict*.json found in docs/log"
    assert RECORD_FILES, "no *lane-comparison* files found in docs/log"


def test_verdicts_carry_no_extracted_text_or_images():
    for path in VERDICT_FILES:
        rows = json.loads(path.read_text())
        assert rows, f"{path.name} is empty"
        for row in rows:
            extra = set(row) - ALLOWED_VERDICT_KEYS
            assert not extra, f"unexpected key(s) in {path.name}: {sorted(extra)}"


def test_no_absolute_paths_anywhere_in_the_record():
    """Absolute paths leak the corpus location and the local username."""
    for path in {*RECORD_FILES, *VERDICT_FILES, MANIFEST}:
        blob = path.read_text()
        for marker in ABSOLUTE_PATH_MARKERS:
            assert marker not in blob, f"{path.name} exposes a path under {marker}"


def test_manifest_references_documents_by_basename_only():
    for entry in json.loads(MANIFEST.read_text()):
        pdf = str(entry.get("pdf", ""))
        assert pdf and not pdf.startswith("/"), f"absolute path committed: {pdf}"
        assert "/" not in pdf, f"path component committed: {pdf}"
