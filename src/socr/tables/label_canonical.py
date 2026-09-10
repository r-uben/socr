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

**The transform DELETES leading indentation and does nothing else.** Round 4
(2026-09-10) removed the decoder that used to run over the whole cell. Three
successive reviews found three rendering regressions in it, each from a
different direction -- a decoded ``&lt;b&gt;`` became a live tag, a decoded
``&ast;`` became emphasis, and decoding a leading ``&nbsp;`` next to a literal
``*`` flipped CommonMark delimiter flanking so the emphasis vanished and the
asterisks became visible. The ticket's target was never interior content; it
was the indentation run at the front of the cell. So:

* **Label cells only.** Column 0 of a BODY row of a genuine markdown table.
  Header cells are column titles and ``parse_grid`` does not normalise them,
  so neither does this. Delimiters, row order, row count, column count and
  every value cell are byte-identical.
* **Only a leading run is removed.** The run is the longest prefix of the
  cell's content made of literal whitespace characters and character
  references whose decoded value ``.isspace()``. References are located with
  ``html.unescape``'s OWN pattern and read by ``html.unescape`` itself, so a
  legacy/partial reference (``&#160``) and a longest-prefix named match
  (``&notit;`` -> ``¬it;``) are classified exactly as the decoder classifies
  them. The prefix is deleted, never rewritten, and the run stops at a
  LITERAL line boundary: that character already splits the row, so
  deleting it would merge two halves.
* **Nothing interior is decoded, held or re-encoded.** No character is ever
  introduced. The rest of the cell is copied byte for byte, so a literal
  ``&`` in ``R&D`` stays literal, ``&amp;nbsp;X`` keeps meaning the literal
  text ``&nbsp;X``, ``&lt;b&gt;`` stays inert, ``&#0000000042;`` stays a
  reference, and ``*&nbsp;x*`` keeps rendering as emphasis. A line boundary
  or a pipe cannot appear that was not already there, so the transform cannot
  move a cell or split a row (``_LINE_BOUNDARIES`` enumerates every boundary
  ``str.splitlines`` honours; each one is pinned by a test).
* **Idempotent by construction.** The scan stops at the first character that
  is neither whitespace nor a whitespace-valued reference, so a second pass
  finds no leading run and is a byte-for-byte no-op. No acceptance check is
  needed.
* **The binder still agrees.** ``decode_label_cell`` unescapes and then strips
  leading whitespace; the deleted prefix decodes to whitespace and would have
  been stripped anyway, so the binder's view of the shipped label equals its
  view of the raw one. That equality is #688's actual invariant.

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
_LEADING_WS_RE = re.compile("^[\\s\u00a0]+")

#: Every character ``str.splitlines`` recognises as a line boundary. The
#: parsers this transform must not disturb (``binding.parse_grid``,
#: ``reconcile._markdown_content_lines``) split with it. Nothing here is ever
#: emitted -- the transform only deletes -- and a test pins each one.
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

#: ``html.unescape``'s own character-reference pattern. Using the decoder's
#: pattern is the point: a stricter one silently disagrees with it (round 3
#: capped decimal references at seven digits, so ``&#0000000042;`` was not
#: recognised as a reference at all). The literal fallback is the same
#: expression, copied, in case a future CPython renames the private name.
_CHARREF = getattr(html, "_charref", None) or re.compile(
    r"&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)"
)

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
    """Drop *text*'s leading indentation run. Idempotent; may be *text*.

    The run is literal whitespace and character references whose decoded value
    is whitespace, in any mix. Everything from the first other character on is
    returned byte for byte -- see the module docstring for why nothing
    interior is decoded.
    """
    pos = 0
    end = len(text)
    while pos < end:
        char = text[pos]
        if char.isspace():
            # A LITERAL line boundary already splits this row for every
            # ``str.splitlines`` parser. Deleting it would MERGE the two
            # halves -- a coordinate move -- so the run stops here. An
            # ENCODED one is only text on this line and may go.
            if char in _LINE_BOUNDARIES:
                break
            pos += 1
            continue
        if char == "&":
            match = _CHARREF.match(text, pos)
            if match is not None and html.unescape(match.group(0)).isspace():
                pos = match.end()
                continue
        break
    return text[pos:]


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
