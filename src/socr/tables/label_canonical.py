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
* **An entity that spells active syntax is left alone.** Decoding is not
  meaning-preserving: ``&lt;b&gt;X&lt;/b&gt;`` decodes to a live CommonMark
  tag and ``&ast;important&ast;`` to italics, so a label's own punctuation
  would vanish from the rendered corpus. Any entity whose value is a single
  Markdown/HTML syntax character is restored with its ORIGINAL spelling; only
  the rest decodes. Nothing is ever newly encoded, so a literal ``&`` in
  ``R&D`` stays literal and a clean label is never churned. A LITERAL line
  boundary in the decoded remainder leaves the whole cell unchanged --
  serialising it would move a row.
* **Idempotent, per cell, mechanically.** The serialisation chosen for a
  cell is accepted only if ``decode_label_cell`` maps it back to the decoded
  label the raw cell reads as; a second application then recomputes the same
  choice and is a byte-for-byte no-op. That is also the invariant #688 needs: the
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


#: Characters that are ACTIVE Markdown or HTML syntax inside a table cell.
#: #688 round 3: decoding is not meaning-preserving. ``&lt;b&gt;X&lt;/b&gt;``
#: decodes to ``<b>X</b>``, which CommonMark then reads as a live tag, and
#: ``&ast;important&ast;`` decodes to ``*important*``, which the review
#: renderer emits as italics -- the label's own punctuation disappears from
#: the visible text. An entity that spells one of these is doing real work and
#: is kept exactly as written.
_ACTIVE_SYNTAX = frozenset("<>&*_`[]\\~|")

#: Block-level markers. Inert inside a GFM cell, but an entity spelling one is
#: still deliberate, and preserving it costs nothing: this set only ever
#: PREVENTS a rewrite, it never introduces an entity that was not there.
_LEADING_SYNTAX = frozenset("#>-+=")

#: The characters an entity is allowed to keep spelling.
_KEEP_ENCODED = _ACTIVE_SYNTAX | _LEADING_SYNTAX | frozenset(_STRUCTURAL)

#: A character reference, named or numeric, with or without its semicolon
#: (html5 accepts both). Used only to LOCATE candidates; what each one means
#: is decided by ``html.unescape``, never by this pattern.
_ENTITY_REF_RE = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});?")

#: Placeholder delimiter for a kept entity. NUL cannot occur in extracted page
#: text and is not whitespace, so it neither collides with content nor lets a
#: kept entity be mistaken for strippable indentation.
_KEEP_MARK = "\x00"


def _canonical_cell(text: str) -> str:
    """The page-markdown form of a label cell. Idempotent; may be *text*.

    One level of decoding, with the entities that spell active syntax held
    back. Concretely:

    * an entity whose value is a single character in ``_KEEP_ENCODED`` is
      restored VERBATIM -- same spelling, same bytes -- so ``&lt;``,
      ``&ast;``, ``&amp;`` and ``&#124;`` all survive untouched and the
      shipped label stays literal;
    * everything else decodes, which is the ticket's actual target: the
      ``&nbsp;`` runs the model emits as sub-row indentation;
    * a LITERAL line-boundary character in the decoded remainder makes the
      cell unrepresentable, and the cell is returned unchanged rather than
      rewritten around a row split this transform is forbidden to make.

    Nothing is ever newly encoded, so a literal ``&`` in ``R&D`` stays literal
    and a clean label is never churned. The result is verified by decoding it
    back: if it does not read as the raw cell reads, the raw cell is returned.
    """
    kept: list[str] = []

    def _hold(match: re.Match[str]) -> str:
        raw = match.group(0)
        value = html.unescape(raw)
        if len(value) == 1 and value in _KEEP_ENCODED:
            kept.append(raw)
            return f"{_KEEP_MARK}{len(kept) - 1}{_KEEP_MARK}"
        return raw

    held = _ENTITY_REF_RE.sub(_hold, text)
    decoded_rest = _LEADING_WS_RE.sub("", html.unescape(held))
    if any(char in decoded_rest for char in _LINE_BOUNDARIES):
        return text
    result = re.sub(
        rf"{_KEEP_MARK}(\d+){_KEEP_MARK}", lambda m: kept[int(m.group(1))], decoded_rest
    )
    if decode_label_cell(result) != decode_label_cell(text):
        return text
    return result


def canonicalize_label_cell(text: str) -> str:
    """The shipped form of a label cell. See :func:`_canonical_cell`."""
    return _canonical_cell(text)


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
