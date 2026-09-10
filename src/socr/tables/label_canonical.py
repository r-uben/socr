"""#688 -- the one canonicalisation boundary for a page candidate's table labels.

#624a taught ``binding.parse_grid`` to decode HTML entities and strip leading
whitespace from a row-label cell, so the binder and
``judge.table_verdict.resolve_cell_refs`` both compare the real label. The
SHIPPED page text was untouched: ``witness.markdown`` is a substring of a page
body produced upstream of ``parse_grid``, so a consumer parsing the corpus
``.md`` as data still read ``&nbsp;&nbsp;&nbsp;&nbsp;Swiss francs``. A real bind
reported ``candidate_row_labels == ('Swiss francs',)`` while
``finalized_page_records`` + ``_phase_assemble`` wrote
``| &nbsp;&nbsp;Swiss francs | 600.0 |``.

Astra's ruling (2026-09-10): fix it at ONE shared boundary -- the point where a
proposed ``PageOutput.text`` becomes the candidate used for judging and
selection -- never in the final selector, which runs after the decisions have
already been made. Extraction, crop reconciliation, header repair and the
escalation lane all submit their resulting candidate through this same
boundary; a later text replacement is a new candidate and crosses it again.
Raw engine/crop bytes stay immutable provenance.

What the transform may do, and nothing else:

* **Label cells only.** Column 0 of a BODY row of a genuine markdown table.
  Header cells are column titles and ``parse_grid`` does not normalise them,
  so neither does this. Delimiters, row order, row count, column count and
  every value cell are byte-identical.
* **Decode, then strip leading whitespace** (including U+00A0), exactly
  ``decode_label_cell`` -- the same function ``binding`` uses -- so the
  shipped bytes and the binder's view of the label cannot diverge again.
* **Re-encode structural characters** that decoding introduced. A cell
  segment cannot contain a literal ``|``, ``\\n`` or ``\\r`` by construction
  (it is delimited by pipes and lives on one line), so any that appears
  after ``html.unescape`` came from an entity, and writing it out raw would
  manufacture a cell or a row -- a coordinate move, which this transform is
  forbidden to make. ``&#124;`` / ``&#10;`` / ``&#13;`` go back in their
  place; ``decode_label_cell`` still resolves them to the intended
  character, so the binder and the judge read what the model meant.
* **Idempotent.** ``canonicalize_table_labels`` is a fixed point after one
  application, which is what lets fresh processing and resume agree byte for
  byte.

#624b's font-evidence wrapped-label merge stays ``bind()``-internal and is not
touched here: it changes a ROW COUNT, and this boundary never does.
"""

from __future__ import annotations

import html
import re

from socr.tables.reconcile import table_body_row_indices

#: Leading presentation whitespace on a decoded label. ``\s`` already covers
#: U+00A0 for ``str`` patterns; it is spelled out so the intent survives a
#: future reader (the model encodes sub-row indentation as ``&nbsp;`` runs).
_LEADING_WS_RE = re.compile("^[\\s ]+")

#: Characters that ARE markdown table structure. Decoding must never emit one
#: raw into the page body -- see the module docstring.
_STRUCTURAL_ENTITIES = (("|", "&#124;"), ("\n", "&#10;"), ("\r", "&#13;"))

#: Markdown cell padding: the ASCII run around a cell's content. Deliberately
#: NOT ``str.strip``, which also eats the U+00A0 indentation this transform
#: exists to remove -- stripping it as padding would leave the raw line
#: unchanged and the bug alive for literal NBSP runs.
_PADDING_RE = re.compile(r"[ \t]*")


def decode_label_cell(text: str) -> str:
    """Decode HTML entities and strip leading whitespace (incl. U+00A0).

    The binder's view of a label cell (``binding._normalize_label_cell``) and
    the shipped bytes' view are the same function on purpose.
    """
    return _LEADING_WS_RE.sub("", html.unescape(text))


def canonicalize_label_cell(text: str) -> str:
    """``decode_label_cell`` for a cell that must go back into page markdown.

    Idempotent: a second application is a no-op, because the only characters
    re-encoded are the ones decoding would otherwise turn into structure.
    """
    decoded = decode_label_cell(text)
    for char, entity in _STRUCTURAL_ENTITIES:
        decoded = decoded.replace(char, entity)
    return decoded


def _canonicalize_row(line: str) -> tuple[str, bool]:
    """Rewrite *line*'s label cell in place; everything else is untouched."""
    first = line.find("|")
    if first < 0:
        return line, False
    second = line.find("|", first + 1)
    if second < 0:
        return line, False
    segment = line[first + 1 : second]
    lead = _PADDING_RE.match(segment).group(0)
    rest = segment[len(lead) :]
    trail = _PADDING_RE.match(rest[::-1]).group(0)
    content = rest[: len(rest) - len(trail)] if trail else rest
    canonical = canonicalize_label_cell(content)
    if canonical == content:
        return line, False
    return line[: first + 1] + lead + canonical + trail + line[second:], True


def canonicalize_table_labels(text: str) -> tuple[str, int]:
    """Canonicalise every table row-label cell in *text*.

    Returns the new text and the number of label cells that changed. Splitting
    on ``"\\n"`` rather than ``str.splitlines`` keeps the rejoin byte-exact for
    any line ending and for a trailing newline.
    """
    if not text or "|" not in text:
        return text, 0
    lines = text.split("\n")
    body = table_body_row_indices(lines)
    if not body:
        return text, 0
    changed = 0
    for idx in body:
        rewritten, did = _canonicalize_row(lines[idx])
        if did:
            lines[idx] = rewritten
            changed += 1
    if not changed:
        return text, 0
    return "\n".join(lines), changed


def canonicalize_candidate(output) -> int:
    """THE boundary. Canonicalise a proposed candidate's text in place.

    Duck-typed on ``.text`` so this module stays free of ``socr.core``
    imports. Returns the number of label cells changed (0 leaves the object
    byte-identical, which is what makes calling it twice free).
    """
    text = getattr(output, "text", "") or ""
    if not text:
        return 0
    canonical, changed = canonicalize_table_labels(text)
    if changed:
        output.text = canonical
    return changed
