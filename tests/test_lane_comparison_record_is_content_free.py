"""The committed lane-comparison records must never carry corpus content.

The measurements are committed as derived verdicts only: the corpus is copyrighted
and this repo is public. An earlier revision of the 2026-08-20 commit claimed a guard
like this existed when it did not, and shipped nine absolute paths exposing the
corpus location. This is the guard, so the claim is true and stays true.

WHAT IT CHECKS, and equally WHAT IT DOES NOT -- stated exhaustively because two
earlier revisions of this docstring overstated it, and an overstated guard is worse
than an honest narrow one:

Covered.
  * Every ``*verdict*.json`` in ``docs/log`` -- matched on the word, so a file named
    ``-verdict.json`` or ``-verdicts-v2.json`` is caught, not just one exact suffix.
    Each row must carry ONLY allowlisted keys, and each value must match the SHAPE
    that key is supposed to hold. The shape check is what stops corpus prose or a
    base64 image being parked inside an allowed field, which a key-name check alone
    permits.
  * Every ``*lane-comparison*`` file whatever its extension, scanned for filesystem
    paths.

NOT covered, and a reader should assume these are open.
  * Path detection is a NAMED-PREFIX scan, not an absolute-path parser. It knows the
    prefixes below. A location under a prefix nobody listed passes.
  * Corpus prose written into a matched non-JSON record -- this very file's ``.md``
    companions are hand-written prose, and no test can tell an author's sentence from
    a page's sentence. The prose records are protected by review, not by this.
  * Anything in a file matching neither pattern.

It is a tripwire on the shapes these records actually take. It is not a proof of
confidentiality and must not be cited as one.
"""

from __future__ import annotations

import json
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "docs" / "log"
VERDICT_FILES = sorted(LOG.glob("*verdict*.json"))
RECORD_FILES = sorted(LOG.glob("*lane-comparison*"))
MANIFEST = LOG / "2026-08-20_lane-comparison-manifest.json"

# key -> (allowed python types, max length for any string it contains).
# The length bound is the part that matters: a page of prose and a base64 image are
# both just "a long string" sitting in a field whose name is on the allowlist.
VERDICT_SCHEMA = {
    "doc": ((str,), 120),
    "page": ((int,), 0),
    "kind": ((str,), 40),
    "shipped_engine": ((str, type(None)), 40),
    "page_status": ((str, type(None)), 40),
    "flags": ((list,), 80),
    "candidates": ((list,), 40),
    "decimals": ((dict,), 40),
}

# A named-prefix scan, deliberately not a general path parser: a regex loose enough
# to catch every absolute path also matches ordinary prose like "/api/tags".
# ``/Volumes/`` is here because a reviewer walked an external-drive corpus path past
# the first version of this list.
ABSOLUTE_PATH_MARKERS = (
    "/Users/",
    "/home/",
    "/private/",
    "/var/folders/",
    "/tmp/",
    "/Volumes/",
    "/mnt/",
    "/media/",
)


def _strings_in(value):
    """Every string reachable inside a verdict value, keys of dicts included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings_in(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings_in(key)
            yield from _strings_in(item)


def test_at_least_one_record_and_one_verdict_file_are_committed():
    """Otherwise every loop below would pass vacuously over an empty set."""
    assert VERDICT_FILES, "no *verdict*.json found in docs/log"
    assert RECORD_FILES, "no *lane-comparison* files found in docs/log"


def test_verdict_rows_carry_only_allowlisted_keys():
    for path in VERDICT_FILES:
        rows = json.loads(path.read_text())
        assert rows, f"{path.name} is empty"
        for row in rows:
            extra = set(row) - set(VERDICT_SCHEMA)
            assert not extra, f"unexpected key(s) in {path.name}: {sorted(extra)}"


def test_verdict_values_cannot_smuggle_prose_or_images():
    """A key-name allowlist alone lets a page of text ride inside ``flags``."""
    for path in VERDICT_FILES:
        for row in json.loads(path.read_text()):
            for key, value in row.items():
                types, limit = VERDICT_SCHEMA[key]
                assert isinstance(value, types), (
                    f"{path.name}: {key} is {type(value).__name__}, expected "
                    f"{'/'.join(x.__name__ for x in types)}"
                )
                for text in _strings_in(value):
                    assert len(text) <= limit, (
                        f"{path.name}: {key} holds a {len(text)}-character string; "
                        f"the cap is {limit}, and anything longer is corpus content "
                        "or a blob rather than a label"
                    )


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
