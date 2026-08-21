"""The committed lane-comparison records must never carry corpus content.

The measurements are committed as derived verdicts only: the corpus is copyrighted
and this repo is public. An earlier revision of the 2026-08-20 commit claimed a guard
like this existed when it did not, and shipped nine absolute paths exposing the
corpus location. This is the guard, so the claim is true and stays true.

WHAT IT CHECKS, and equally WHAT IT DOES NOT. Stated exhaustively because four
successive revisions of this docstring overstated it, each time in a way a reviewer
then walked a payload through -- a per-string cap defeated by chopping the payload up,
a size cap defeated by using the rows already present, a "no free-text field left"
claim defeated by encoding a whole page into integers. An overstated guard is worse
than an honest narrow one, because it gets quoted as if it were a proof.

Covered.
  * Every ``*verdict*.json`` in ``docs/log`` -- matched on the word, so ``-verdict``
    and ``-verdicts-v2`` are caught, not one exact suffix.
  * Row IDENTITY, which is the load-bearing check. Each ``doc`` is resolved through a
    FINITE table of permitted spellings -- the manifest name, with and without the
    ``.pdf`` suffix, and the two widths the runner has truncated to -- and the SET of
    resolved ``(document, page, kind)`` triples must equal the manifest's selected
    pages exactly. Not a prefix match: an earlier revision compared with
    ``name.startswith(doc)``, so 21 rows carrying 21 distinct prefixes of one document
    all matched the same page, dropped 20 real pages, and passed a check whose
    docstring claimed exact identity.
  * Row SHAPE: exactly the schema's keys, no more and no fewer.
  * Every remaining string against a CAMPAIGN-APPROVED vocabulary -- flags, engines,
    kinds, statuses. These are the values this measurement may record, NOT the
    pipeline's vocabularies, which are larger. An unrecognised value means either
    corpus content or a pipeline change that belongs in this file, added deliberately.
  * Integer RANGE. Unbounded integers are a payload channel: a reviewer compressed an
    entire 1,675-byte page into 37 ``decimals`` values and passed every other check.
  * Value bounds -- type, element type, string length, element count, serialised size.
  * File bounds -- row count and byte size.
  * The MANIFEST's own page records: exact key set, integer types, derived ranges on
    ``se``/``math``/``imgs``. Those three counts were unbounded, and a reviewer
    compressed a real 1,675-byte page into the 63 of them that already exist.
  * Filesystem paths in every ``*lane-comparison*`` file, scanned BOTH as raw text and
    as decoded JSON strings. Raw-text scanning alone is defeated by JSON escaping: a
    backslash-escaped solidus decodes to an absolute path while carrying no bare
    marker substring.

NOT covered. Assume these are open.
  * A NUMERIC CHANNEL in ``decimals``, reduced but not closed. Each value is a count
    in [0, 512], about nine bits, over at most four engines on 21 rows -- on the order
    of 90 bytes if every count were chosen adversarially. A paragraph at most, not a
    page, and bounding it further would mean asserting counts this measurement has not
    made.
  * WHICH permitted value sits in which row. Choosing among approved terms is itself a
    low-rate channel. Closing it is not worth the brittleness.
  * SELF-CONSISTENCY IS NOT PROVENANCE. The manifest anchors these checks and is
    mutable in the very commit that changes them -- appending a document there and
    referencing it here passes. What this proves is that the verdicts describe the
    study the manifest describes. That the manifest describes the real study is
    checked by review, not by code.
  * Corpus prose in a hand-written ``.md`` record. No test distinguishes an author's
    sentence from a page's. Those files are protected by review.
  * Path spellings nobody listed, in any encoding the decoder does not produce.
  * Files matching neither pattern.

It is a tripwire on the shapes these records take. It is not a proof of
confidentiality and must not be cited as one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "docs" / "log"
VERDICT_FILES = sorted(LOG.glob("*verdict*.json"))
RECORD_FILES = sorted(LOG.glob("*lane-comparison*"))
MANIFEST = LOG / "2026-08-20_lane-comparison-manifest.json"

# Bounds are DERIVED from the widest value actually committed, at roughly double, not
# guessed: doc 60 chars, manifest names 128, flags 32/2/69, candidates 6/2/20,
# decimals 6/2/30, page 44, largest decimal count 152, and 21 rows per file at just
# under 5.9 KB. Re-derive them if the record's shape changes; do not raise one to make
# a new file pass.
MAX_VERDICT_ROWS = 48
MAX_VERDICT_BYTES = 12_000
MAX_DECIMAL_COUNT = 512
MAX_MANIFEST_ENTRIES = 32
# Manifest page-record counts, observed maxima se=76, math=14, imgs=1.
MAX_SELECTION_COUNT = 512
MAX_PAGE_IMAGES = 32
DOCUMENT_NAME = re.compile(r"[A-Za-z0-9_.-]{1,160}")

# key -> (value types, element types, max string, max elements, max serialised size)
VERDICT_SCHEMA = {
    "doc": ((str,), (), 80, 0, 96),
    "page": ((int,), (), 0, 0, 6),
    "kind": ((str,), (), 16, 0, 20),
    "shipped_engine": ((str, type(None)), (), 16, 0, 20),
    "page_status": ((str, type(None)), (), 16, 0, 20),
    "flags": ((list,), (str,), 48, 4, 100),
    "candidates": ((list,), (str,), 16, 6, 80),
    "decimals": ((dict,), (int,), 16, 6, 80),
}

# Campaign-approved, NOT the pipeline's vocabularies -- socr's EngineType and
# PageStatus are both larger. These are the values this measurement is allowed to
# record, which is a deliberately tighter set.
APPROVED_FLAGS = frozenset(
    {
        "native_table_structure_defective",
        "native_table_structure_failed",
        "native_table_unverifiable",
        "native_table_header_unattributed",
        "native_rotated_text_shredded",
    }
)
APPROVED_ENGINES = frozenset({"gemini", "native", "nougat", "qwen"})
APPROVED_KINDS = frozenset({"equation", "figure", "table"})
APPROVED_STATUSES = frozenset({"error", "success", "warning", "partial"})

# A named-prefix scan, deliberately not a general path parser: a regex loose enough to
# catch every absolute path also matches ordinary prose like "/api/tags".
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


def _decoded_strings(path):
    """Every string a JSON consumer would see, escapes resolved.

    Scanning raw file text is not enough: JSON may escape the solidus, so a value that
    decodes to an absolute path can carry no bare marker for a text scan to find.
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


# Widths the runner has truncated document names to across the committed campaigns.
# A FINITE set, deliberately: accepting any prefix lets one document impersonate
# twenty-one, which is exactly how the previous version of this check was defeated.
RUNNER_NAME_WIDTHS = (58, 60)


def _manifest_documents():
    """``{permitted spelling: canonical document id}`` for the manifest's documents."""
    spellings = {}
    for entry in json.loads(MANIFEST.read_text()):
        canonical = str(entry["name"]).removesuffix(".pdf")
        names = {str(entry.get(key, "")) for key in ("pdf", "name")}
        names |= {n.removesuffix(".pdf") for n in names}
        names |= {n[:width] for n in set(names) for width in RUNNER_NAME_WIDTHS}
        for name in names:
            if name:
                spellings.setdefault(name, canonical)
    return spellings


def _manifest_page_keys():
    """``(canonical document, page, kind)`` for every page the manifest selected."""
    keys = set()
    for entry in json.loads(MANIFEST.read_text()):
        canonical = str(entry["name"]).removesuffix(".pdf")
        for page in entry.get("pages", []):
            keys.add((canonical, page["page"], page["kind"]))
    return keys


def test_at_least_one_record_and_one_verdict_file_are_committed():
    """Otherwise every loop below would pass vacuously over an empty set."""
    assert VERDICT_FILES, "no *verdict*.json found in docs/log"
    assert RECORD_FILES, "no *lane-comparison* files found in docs/log"


def test_verdict_rows_carry_exactly_the_schema_keys():
    for path in VERDICT_FILES:
        rows = json.loads(path.read_text())
        assert rows, f"{path.name} is empty"
        for row in rows:
            assert set(row) == set(VERDICT_SCHEMA), (
                f"{path.name}: row keys {sorted(row)} != schema {sorted(VERDICT_SCHEMA)}"
            )


def test_verdict_rows_are_exactly_the_manifest_pages():
    """The check that leaves ``doc``/``page``/``kind`` nothing of their own to say.

    Bounding a field limits how big a payload can be; pinning the row set to the
    manifest removes the row as a place to put one. Every triple must be one the
    manifest already selected, none may repeat, and the count must match.
    """
    manifest_keys = _manifest_page_keys()
    documents = _manifest_documents()
    assert manifest_keys, "manifest selects no pages to anchor against"
    for path in VERDICT_FILES:
        rows = json.loads(path.read_text())
        seen = set()
        for row in rows:
            canonical = documents.get(row["doc"])
            assert canonical is not None, (
                f"{path.name}: doc {row['doc']!r} is not a permitted spelling of any "
                "manifest document"
            )
            triple = (canonical, row["page"], row["kind"])
            assert triple not in seen, f"{path.name}: duplicate row {triple}"
            seen.add(triple)
        assert seen == manifest_keys, (
            f"{path.name}: the rows are not the manifest's pages. "
            f"missing {sorted(manifest_keys - seen)[:3]}, extra {sorted(seen - manifest_keys)[:3]}"
        )
        assert len(rows) == len(manifest_keys), (
            f"{path.name} holds {len(rows)} rows for {len(manifest_keys)} selected pages"
        )


def test_every_verdict_string_comes_from_an_approved_vocabulary():
    """Bounds cap a payload's size; this removes anywhere to put one."""
    for path in VERDICT_FILES:
        for row in json.loads(path.read_text()):
            assert row["kind"] in APPROVED_KINDS, f"{path.name}: kind {row['kind']!r}"
            for key, vocabulary in (
                ("page_status", APPROVED_STATUSES),
                ("shipped_engine", APPROVED_ENGINES),
            ):
                value = row.get(key)
                assert value is None or value in vocabulary, (
                    f"{path.name}: {key} {value!r} is not an approved value"
                )
            for flag in row["flags"]:
                assert flag in APPROVED_FLAGS, f"{path.name}: unknown flag {flag!r}"
            for engine in list(row["candidates"]) + list(row["decimals"]):
                assert engine in APPROVED_ENGINES, f"{path.name}: unknown engine {engine!r}"


def test_integers_are_counts_not_a_payload_channel():
    """An unbounded integer carries as much as an unbounded string.

    A reviewer compressed a 1,675-byte page into 37 ``decimals`` values and passed
    every string-side check. Ranging them costs nothing and reduces that channel to
    roughly nine bits per count.
    """
    for path in VERDICT_FILES:
        for row in json.loads(path.read_text()):
            for engine, count in row["decimals"].items():
                assert isinstance(count, int) and not isinstance(count, bool), (
                    f"{path.name}: decimals[{engine}] is {type(count).__name__}"
                )
                assert 0 <= count <= MAX_DECIMAL_COUNT, (
                    f"{path.name}: decimals[{engine}] is {count}, outside "
                    f"[0, {MAX_DECIMAL_COUNT}] -- a count this size is a payload"
                )


def test_verdict_values_cannot_smuggle_prose_or_images():
    """Type, element type, string length, element count, serialised size."""
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


def test_verdict_files_are_bounded_as_files_not_only_per_value():
    """Per-value bounds are per value; a payload can be spread across rows."""
    for path in VERDICT_FILES:
        rows = json.loads(path.read_text())
        assert len(rows) <= MAX_VERDICT_ROWS, (
            f"{path.name} holds {len(rows)} rows against a cap of {MAX_VERDICT_ROWS}"
        )
        size = len(path.read_bytes())
        assert size <= MAX_VERDICT_BYTES, (
            f"{path.name} is {size} bytes against a cap of {MAX_VERDICT_BYTES}"
        )


def test_manifest_is_bounded_since_the_verdicts_anchor_to_it():
    """The anchor needs bounds of its own, or it becomes the way in.

    This does not make the manifest trustworthy -- it is mutable in the same commit as
    these tests, so what is proved is self-consistency, not provenance. It does stop
    the anchor being an unbounded free-text field.
    """
    entries = json.loads(MANIFEST.read_text())
    assert 0 < len(entries) <= MAX_MANIFEST_ENTRIES, (
        f"manifest holds {len(entries)} entries, cap {MAX_MANIFEST_ENTRIES}"
    )
    for entry in entries:
        assert set(entry) == {"pdf", "name", "pages"}, f"manifest keys {sorted(entry)}"
        for key in ("pdf", "name"):
            assert DOCUMENT_NAME.fullmatch(str(entry[key])), (
                f"manifest {key} {entry[key]!r} is not a plain document name"
            )
        assert isinstance(entry["pages"], list), "manifest pages must be a list"
        seen_pages = set()
        for page in entry["pages"]:
            assert set(page) == {"page", "kind", "se", "math", "imgs"}, (
                f"manifest page keys {sorted(page)}"
            )
            assert page["kind"] in APPROVED_KINDS, f"manifest kind {page['kind']!r}"
            for field, cap in (
                ("page", 2000),
                ("se", MAX_SELECTION_COUNT),
                ("math", MAX_SELECTION_COUNT),
                ("imgs", MAX_PAGE_IMAGES),
            ):
                value = page[field]
                assert isinstance(value, int) and not isinstance(value, bool), (
                    f"manifest {field} is {type(value).__name__}"
                )
                assert 0 <= value <= cap, f"manifest {field} is {value}, cap {cap}"
            assert page["page"] > 0, f"manifest page {page['page']}"
            key = (page["page"], page["kind"])
            assert key not in seen_pages, f"manifest repeats page {key}"
            seen_pages.add(key)


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
