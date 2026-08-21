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
    Each row must carry ONLY allowlisted keys, and each value is bounded five ways:
    its own type, the type of every element inside it, the length of any one string,
    the number of elements, and the serialised size of the whole value. A key-name
    check alone permits a payload inside an allowed field; a per-string cap alone is
    defeated by chopping it into legal pieces; any string cap is blind to a numeric
    byte array. These bounds close all three PER VALUE.
  * The whole file, not only its values: a row count and a byte size. Per-value bounds
    are per value, so extra rows each carrying a legal payload assemble a page out of
    individually-innocent parts. An earlier revision without these was walked that way.
  * CLOSED VOCABULARIES, which are what actually reduce the channel to nothing. Bounds
    alone were not enough: with only the 21 real rows, two legal strings per ``flags``
    entry carry 1,932 characters -- more than a whole shipped page. So every string in
    a verdict row must now come from a fixed set (flags, engines, kinds, statuses) or
    be a prefix of a document name already committed in the manifest. There is no
    free-text field left for a payload to occupy.
  * Every ``*lane-comparison*`` file whatever its extension, scanned for filesystem
    paths -- BOTH as raw text and as decoded JSON strings. Raw-text scanning alone is
    defeated by JSON's own escaping: a backslash-escaped solidus is valid JSON,
    decodes to an absolute path, and contains no bare ``/Users/`` substring for a
    text scan to find.

NOT covered, and a reader should assume these are open.
  * Path detection is a NAMED-PREFIX scan, not an absolute-path parser. It knows the
    prefixes below. A location under a prefix nobody listed passes.
  * Corpus prose written into a matched non-JSON record -- this very file's ``.md``
    companions are hand-written prose, and no test can tell an author's sentence from
    a page's sentence. The prose records are protected by review, not by this.
  * Anything in a file matching neither pattern.
  * Meaning. Every string must come from a committed vocabulary, but WHICH permitted
    value sits in which row is unconstrained, so a low-rate covert channel exists in
    the choice itself. Closing that is not worth the brittleness.
  * The prose records, as above -- the vocabularies apply to JSON only.
  * Path spellings nobody listed, in any encoding the decoder does not produce.

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

# key -> (allowed types for the value, allowed types for elements inside it,
#         max length of any single string, max number of elements, max serialised
#         size of the whole value).
#
# The per-string cap alone is not enough and an earlier revision of this file wrongly
# claimed it was: a reviewer defeated it by splitting a base64 image across two
# under-cap ``flags`` entries, and again by splitting prose across three. An unbounded
# list of short chunks carries as much as one long string. The element-type and
# serialised-size bounds are what actually close that, and the element-type bound is
# what stops a payload arriving as a numeric array with no strings in it at all.
#
# The bounds are DERIVED, not invented -- a first pass guessed round numbers generous
# enough that the segmented payloads still walked through. Widest value actually
# present across the committed verdict files: doc 60 chars, flags 32 chars / 2 items /
# 69 serialised, candidates 6 / 2 / 20, decimals 6 / 2 / 30. Each cap below is roughly
# double its observed maximum -- headroom for a longer flag name or a fifth engine,
# and nowhere near enough for a page of prose. Re-derive them if the record's shape
# changes rather than raising them to make a new file pass.
#
# The bounds alone are NOT sufficient and an earlier revision claimed they were: a
# reviewer filled the 21 real rows with legal-length strings and carried 1,932
# characters -- larger than an actual shipped page in this corpus. Bounds cap the size
# of a payload; only the closed vocabularies below stop there being a place to put one.
# File-level bounds, also derived: both committed verdict files hold 21 rows and just
# under 5.9 KB. Doubling leaves room for a larger campaign. These bound the FILE; they
# do not by themselves bound what it can carry -- see the vocabularies below.
MAX_VERDICT_ROWS = 48
MAX_VERDICT_BYTES = 12_000

# Closed vocabularies. Every one is small and finite in the pipeline itself, so an
# unrecognised value means either corpus content or a genuine pipeline change -- and a
# genuine change should be added here deliberately, in the same commit that introduces
# it, rather than the guard being loosened to admit it.
KNOWN_FLAGS = frozenset(
    {
        "native_table_structure_defective",
        "native_table_structure_failed",
        "native_table_unverifiable",
        "native_table_header_unattributed",
        "native_rotated_text_shredded",
    }
)
KNOWN_ENGINES = frozenset({"gemini", "native", "nougat", "qwen"})
KNOWN_KINDS = frozenset({"equation", "figure", "table"})
KNOWN_STATUSES = frozenset({"error", "success", "warning", "partial"})

VERDICT_SCHEMA = {
    #                value types          element types  str  items  bytes
    "doc": ((str,), (), 80, 0, 96),
    "page": ((int,), (), 0, 0, 6),
    "kind": ((str,), (), 16, 0, 20),
    "shipped_engine": ((str, type(None)), (), 16, 0, 20),
    "page_status": ((str, type(None)), (), 16, 0, 20),
    "flags": ((list,), (str,), 48, 4, 100),
    "candidates": ((list,), (str,), 16, 6, 80),
    "decimals": ((dict,), (int,), 16, 6, 80),
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


def test_verdict_files_are_bounded_as_files_not_only_per_value():
    """Per-value bounds are per value; a payload can be spread across extra rows."""
    for path in VERDICT_FILES:
        rows = json.loads(path.read_text())
        assert len(rows) <= MAX_VERDICT_ROWS, (
            f"{path.name} holds {len(rows)} rows against a cap of {MAX_VERDICT_ROWS}; "
            "a campaign this large needs the cap re-derived, deliberately"
        )
        size = len(path.read_bytes())
        assert size <= MAX_VERDICT_BYTES, (
            f"{path.name} is {size} bytes against a cap of {MAX_VERDICT_BYTES}"
        )


def test_verdict_values_cannot_smuggle_prose_or_images():
    """A key-name allowlist alone lets a page of text ride inside ``flags``.

    Four bounds, because a reviewer walked something past each of the first three:
    the value's own type, the type of every element in it (a numeric byte array holds
    no strings to inspect), the length of any one string, the number of elements, and
    the serialised size of the whole value (which is what defeats a payload chopped
    into individually-legal pieces).
    """
    for path in VERDICT_FILES:
        for row in json.loads(path.read_text()):
            for key, value in row.items():
                types, elem_types, max_str, max_items, max_bytes = VERDICT_SCHEMA[key]
                assert isinstance(value, types) and not isinstance(value, bool), (
                    f"{path.name}: {key} is {type(value).__name__}, expected "
                    f"{'/'.join(x.__name__ for x in types)}"
                )
                blob = json.dumps(value, ensure_ascii=False)
                assert len(blob) <= max_bytes, (
                    f"{path.name}: {key} serialises to {len(blob)} characters against a "
                    f"cap of {max_bytes}; a value this large is content, not a label"
                )
                if isinstance(value, (list, dict)):
                    assert len(value) <= max_items, (
                        f"{path.name}: {key} holds {len(value)} entries, cap {max_items}"
                    )
                    elements = value.values() if isinstance(value, dict) else value
                    for element in elements:
                        assert isinstance(element, elem_types), (
                            f"{path.name}: {key} holds a {type(element).__name__}, "
                            f"expected {'/'.join(x.__name__ for x in elem_types)}"
                        )
                for text in _strings_in(value):
                    assert len(text) <= max_str, (
                        f"{path.name}: {key} holds a {len(text)}-character string; "
                        f"the cap is {max_str}, and anything longer is corpus content "
                        "or a blob rather than a label"
                    )


def _decoded_strings(path):
    """Every string a JSON consumer would actually see, escapes resolved.

    Scanning raw file text is not enough: JSON may escape the solidus, so a value
    that decodes to an absolute path can carry no bare ``/Users/`` substring for a
    text scan to find. A reviewer walked exactly that past the previous version.
    """
    if path.suffix != ".json":
        return
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())


def _manifest_document_names():
    names = set()
    for entry in json.loads(MANIFEST.read_text()):
        for key in ("pdf", "name"):
            value = str(entry.get(key, ""))
            if value:
                names.add(value[:-4] if value.endswith(".pdf") else value)
    return names


def test_every_verdict_string_comes_from_a_closed_vocabulary():
    """Bounds cap a payload's size; this removes anywhere to put one.

    Filling only the real rows with legal-length strings carried 1,932 characters past
    the size bounds -- more than an actual page of this corpus. Every string in a
    verdict row must therefore come from a fixed set, or be a prefix of a document
    name already committed in the manifest (the runner truncates long names).
    """
    document_names = _manifest_document_names()
    assert document_names, "manifest carries no document names to anchor against"
    for path in VERDICT_FILES:
        for row in json.loads(path.read_text()):
            doc = row.get("doc", "")
            assert any(name.startswith(doc) for name in document_names), (
                f"{path.name}: doc {doc!r} is not a prefix of any manifest document; "
                "a verdict may only name documents the manifest already names"
            )
            assert row.get("kind") in KNOWN_KINDS, f"{path.name}: kind {row.get('kind')!r}"
            for key, vocabulary in (
                ("page_status", KNOWN_STATUSES),
                ("shipped_engine", KNOWN_ENGINES),
            ):
                value = row.get(key)
                assert value is None or value in vocabulary, (
                    f"{path.name}: {key} {value!r} is not a known value"
                )
            for flag in row.get("flags") or []:
                assert flag in KNOWN_FLAGS, f"{path.name}: unknown flag {flag!r}"
            for engine in list(row.get("candidates") or []) + list(row.get("decimals") or {}):
                assert engine in KNOWN_ENGINES, f"{path.name}: unknown engine {engine!r}"


def test_no_absolute_paths_anywhere_in_the_record():
    """Absolute paths leak the corpus location and the local username."""
    for path in {*RECORD_FILES, *VERDICT_FILES, MANIFEST}:
        blob = path.read_text()
        for marker in ABSOLUTE_PATH_MARKERS:
            assert marker not in blob, f"{path.name} exposes a path under {marker}"
            for text in _decoded_strings(path):
                assert marker not in text, (
                    f"{path.name} hides a path under {marker} behind JSON escaping"
                )


def test_manifest_references_documents_by_basename_only():
    for entry in json.loads(MANIFEST.read_text()):
        pdf = str(entry.get("pdf", ""))
        assert pdf and not pdf.startswith("/"), f"absolute path committed: {pdf}"
        assert "/" not in pdf, f"path component committed: {pdf}"
