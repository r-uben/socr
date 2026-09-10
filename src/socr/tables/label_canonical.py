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
  segment cannot contain a literal ``|`` or a line boundary by construction
  (it is delimited by pipes and lives on one line), so any that appears
  after ``html.unescape`` came from an entity, and writing it out raw would
  manufacture a cell or a row -- a coordinate move, which this transform is
  forbidden to make. **Every** boundary the real parsers honour counts, not
  just LF and CR: ``binding.parse_grid`` and
  ``reconcile._markdown_content_lines`` split with ``str.splitlines``, which
  also breaks on VT, FF, FS/GS/RS, NEL, U+2028 and U+2029. They are
  enumerated in ``_LINE_BOUNDARIES`` and each one is pinned by a test.
* **Round-tripping is verified, not assumed.** ``html.unescape`` implements
  the html5 replacement table, so a numeric reference to a C0 control or to
  the C1 range does NOT come back (``&#11;`` decodes to nothing, ``&#133;``
  to U+2026). A decoded label carrying such a character therefore has no
  serialised form, and the cell is left exactly as found -- fail closed, no
  coordinate move. Which characters survive is decided by asking
  ``html.unescape`` at import time, not by a hand-written list.
* **Idempotent, per cell, mechanically.** The serialisation chosen for a
  cell is accepted only if ``decode_label_cell`` maps it back to the decoded
  label it came from; a second application then recomputes the same choice
  and is a byte-for-byte no-op. That is also the invariant #688 needs: the
  binder's view of the shipped label equals the binder's view of the raw
  one. Nested escapes are the reason a plain "decode to a fixed point"
  would be wrong -- ``&amp;nbsp;X`` means the literal text ``&nbsp;X``, and
  decoding it twice would silently reinterpret it as indentation.

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

#: Every character ``str.splitlines`` recognises as a line boundary. The
#: parsers this transform must not disturb (``binding.parse_grid``,
#: ``reconcile._markdown_content_lines``) split with it, so emitting any of
#: these raw into a label cell splits the row in two.
_LINE_BOUNDARIES = (
    "\n",  # LF
    "\r",  # CR
    "\v",  # VT
    "\f",  # FF
    "\x1c",  # FS
    "\x1d",  # GS
    "\x1e",  # RS
    "\x85",  # NEL
    "\u2028",  # LINE SEPARATOR
    "\u2029",  # PARAGRAPH SEPARATOR
)

#: Characters that ARE markdown structure: the cell boundary plus every line
#: boundary above. Decoding must never emit one raw into the page body -- see
#: the module docstring.
_STRUCTURAL = ("|",) + _LINE_BOUNDARIES

#: The subset of ``_STRUCTURAL`` that a numeric character reference actually
#: round-trips, asked of ``html.unescape`` rather than assumed: html5 drops a
#: numeric reference to a C0 control and remaps the C1 range, so VT, FS, GS, RS
#: and NEL have no representation. A label needing one of them is left alone.
_ENTITY = {
    char: f"&#{ord(char)};" for char in _STRUCTURAL if html.unescape(f"&#{ord(char)};") == char
}

#: Structural characters with no serialised form. Their presence in a decoded
#: label makes the cell unrepresentable, so it is not rewritten.
_UNREPRESENTABLE = tuple(char for char in _STRUCTURAL if char not in _ENTITY)


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


def _serialise_label(decoded: str) -> str | None:
    """The page-markdown form of an already-decoded label, or ``None``.

    Two candidate serialisations are tried, cheapest first: leave literal
    ampersands alone, or escape every one of them as ``&amp;``. The first
    whose ``decode_label_cell`` is *exactly* ``decoded`` wins. That check is
    what makes the transform idempotent (a second pass recomputes the same
    choice from the same decoded label) and what keeps the binder's reading of
    the shipped cell equal to its reading of the raw cell. ``None`` means no
    faithful form exists and the caller must leave the cell untouched.
    """
    if any(char in decoded for char in _UNREPRESENTABLE):
        return None
    for escape_amp in (False, True):
        candidate = decoded.replace("&", "&amp;") if escape_amp else decoded
        for char, entity in _ENTITY.items():
            candidate = candidate.replace(char, entity)
        if decode_label_cell(candidate) == decoded:
            return candidate
    return None


def canonicalize_label_cell(text: str) -> str:
    """``decode_label_cell`` for a cell that must go back into page markdown.

    Idempotent, and a no-op when the decoded label has no faithful markdown
    form (see :func:`_serialise_label`).
    """
    serialised = _serialise_label(decode_label_cell(text))
    return text if serialised is None else serialised


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
