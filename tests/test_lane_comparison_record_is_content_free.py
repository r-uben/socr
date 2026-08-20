"""The committed lane-comparison record must never carry corpus content.

The 2026-08-20 measurement is committed as derived verdicts only: the corpus is
copyrighted and this repo is public. An earlier revision of that commit claimed a
guard like this existed when it did not, and shipped nine absolute paths exposing
the corpus location. This is the guard, so the claim is true and stays true.
"""

from __future__ import annotations

import json
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "docs" / "log"
VERDICTS = LOG / "2026-08-20_lane-comparison-verdicts.json"
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


def test_verdicts_carry_no_extracted_text_or_images():
    rows = json.loads(VERDICTS.read_text())
    assert rows, "verdict file is empty"
    for row in rows:
        extra = set(row) - ALLOWED_VERDICT_KEYS
        assert not extra, f"unexpected key(s) in verdict record: {sorted(extra)}"


def test_no_absolute_paths_anywhere_in_the_record():
    """Absolute paths leak the corpus location and the local username."""
    for path in (VERDICTS, MANIFEST):
        blob = path.read_text()
        assert "/Users/" not in blob, f"{path.name} exposes a home directory"
        assert "/private/tmp" not in blob, f"{path.name} exposes a session scratchpad"


def test_manifest_references_documents_by_basename_only():
    for entry in json.loads(MANIFEST.read_text()):
        pdf = str(entry.get("pdf", ""))
        assert pdf and not pdf.startswith("/"), f"absolute path committed: {pdf}"
        assert "/" not in pdf, f"path component committed: {pdf}"
