"""Structure-preserving HTML cleanup for VLM OCR output.

DeepSeek-OCR and GLM-OCR natively emit tables as HTML. The historical cleanup
deleted ALL tags with ``<[^>]+>``, which (a) flattened every table into a
run-on wordstream where adjacent cell digits fuse into fabricated numbers
('4.4' + '79.1' -> '4.479.1'), and (b) matched across newlines and through
non-tag text, deleting inequalities and literal tokens like ``<EOS>``.

This module replaces that with:

  1. :func:`convert_html_tables` — HTML ``<table>`` blocks become
     GitHub-markdown tables (colspan/rowspan-aware). A trailing table whose
     closing tag never arrived (truncated generation) is converted from
     whatever rows it managed to emit. THE INVARIANT IS: this function never
     deletes content — a block that does not parse as rows is left in place
     for the tag-stripping pass, and text after a truncated table's last row
     is re-emitted verbatim.
  2. :func:`strip_html_tags` — removes only allowlisted, tag-shaped HTML
     (``<name ...>`` with no newline inside). Inline formatting tags (b, i,
     em, span, ...) are removed with NO replacement so intra-word markup
     (``the <i>t</i>-statistic``) cannot grow spaces; structural tags (div,
     p, td, ...) become separators so two values can never fuse.
     ``<sup>x</sup>`` -> ``^x`` and ``<sub>t</sub>`` -> ``_t`` keep their
     semantics. Anything else between angle brackets — ``a < b``
     comparisons, ``<EOS>``, multi-line spans — is left untouched.
  3. :func:`clean_residual_html` — the two above plus HTML-entity decoding,
     in the right order. The single entry point engine cleaners should use.

Known limitation (documented, not silent): nested ``<table>`` inside a cell
ends the outer block early; the trailing outer rows degrade to separated
pipe fragments via the residual pass — structure mangled, content preserved.
"""

from __future__ import annotations

import html
import re

_FLAGS = re.DOTALL | re.IGNORECASE

_TABLE_OPEN_RE = re.compile(r"<table[^>]*>", re.IGNORECASE)
_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", _FLAGS)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", _FLAGS)
_TR_SPLIT_RE = re.compile(r"<tr[^>]*>", re.IGNORECASE)
_CELL_RE = re.compile(r"<(t[dh])([^>]*)>(.*?)</t[dh]>", _FLAGS)
_OPEN_CELL_RE = re.compile(r"\s*<t[dh][^>]*>([^<]*)\Z", _FLAGS)
_COLSPAN_RE = re.compile(r"colspan\s*=\s*[\"']?(\d+)", re.IGNORECASE)
_ROWSPAN_RE = re.compile(r"rowspan\s*=\s*[\"']?(\d+)", re.IGNORECASE)
_SUP_RE = re.compile(r"<sup>(.*?)</sup>", _FLAGS)
_SUB_RE = re.compile(r"<sub>(.*?)</sub>", _FLAGS)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
# Non-row markup tolerated between rows of one table.
_BETWEEN_ROWS_RE = re.compile(
    r"(?:\s|</?(?:thead|tbody|tfoot|colgroup)[^>\n]*>|<col[^>\n]*>"
    r"|<caption[^>]*>.*?</caption>)+",
    _FLAGS,
)

# A hallucinated colspan must never allocate unbounded padding cells.
_MAX_SPAN = 50

# Inline formatting tags: removed with NO replacement, because they commonly
# sit inside words (``the <i>t</i>-statistic``, ``<b>F</b>ederal``). Distinct
# values are essentially never separated by an inline boundary alone.
_INLINE_TAGS = "b|strong|i|em|u|s|strike|span|font|small|big|mark|del|ins|tt|code|a"
# Structural tags: replaced by a separator, because their boundaries DO
# separate distinct content (the digit-fusion failure mode).
_STRUCTURAL_TAGS = (
    "html|head|body|table|thead|tbody|tfoot|tr|td|th|caption|col|colgroup"
    "|p|div|center|hr|img|ul|ol|li|h1|h2|h3|h4|h5|h6|pre|blockquote"
    "|section|article|header|footer|figure|figcaption"
)
# Tag-shaped only: optional /, known name, optional attributes that contain
# no angle brackets and NO NEWLINE, optional self-close. ``[^<>\n]`` is what
# keeps a match from ever spanning lines or swallowing a second '<'.
_INLINE_TAG_RE = re.compile(rf"</?(?:{_INLINE_TAGS})(?:\s[^<>\n]*)?/?>", re.IGNORECASE)
_STRUCTURAL_TAG_RE = re.compile(rf"</?(?:{_STRUCTURAL_TAGS})(?:\s[^<>\n]*)?/?>", re.IGNORECASE)
# Residual cell/row boundaries on malformed fragments (a table the block
# converter could not parse). Replaced with separators, never with "".
_RESIDUAL_CELL_RE = re.compile(r"</?t[dh](?:\s[^<>\n]*)?>", re.IGNORECASE)
_RESIDUAL_ROW_RE = re.compile(r"</?tr(?:\s[^<>\n]*)?>", re.IGNORECASE)


def _convert_sup_sub(text: str) -> str:
    """``<sup>x</sup>`` -> ``^x``, ``<sub>t</sub>`` -> ``_t``.

    Subscripts must NOT become ``^`` (an OCR'd time index ``x<sub>t</sub>``
    rewritten as an exponent is a wrong-meaning transformation). An empty
    element becomes a single space — never "" — so ``4.4<sup> </sup>79.1``
    cannot fuse into a fabricated number.
    """
    text = _SUP_RE.sub(lambda m: f"^{m.group(1)}" if m.group(1).strip() else " ", text)
    text = _SUB_RE.sub(lambda m: f"_{m.group(1)}" if m.group(1).strip() else " ", text)
    return text


def _span(attrs: str, pattern: re.Pattern) -> int:
    m = pattern.search(attrs)
    if not m:
        return 1
    return min(max(int(m.group(1)), 1), _MAX_SPAN)


def _cell_text(raw: str) -> str:
    """Flatten one cell's inner HTML to a single markdown-safe line."""
    raw = _convert_sup_sub(raw)
    raw = _BR_RE.sub(" ", raw)
    raw = _INLINE_TAG_RE.sub("", raw)
    raw = _STRUCTURAL_TAG_RE.sub(" ", raw)
    raw = html.unescape(raw)
    # A literal pipe would shift every following cell in the markdown row.
    raw = raw.replace("|", "\\|")
    return " ".join(raw.split())


def _parsed_cell(m: re.Match) -> tuple[str, int, int]:
    """(text, colspan, rowspan) for one matched closed cell."""
    return (_cell_text(m.group(3)), _span(m.group(2), _COLSPAN_RE), _span(m.group(2), _ROWSPAN_RE))


def _row_cells(row_inner: str) -> list[tuple[str, int, int]]:
    """Parse one row's inner HTML into (text, colspan, rowspan) triples.

    Tolerates a trailing cell whose closing tag never arrived (truncated
    generation): its text is kept rather than dropped.
    """
    cells: list[tuple[str, int, int]] = []
    last_end = 0
    for m in _CELL_RE.finditer(row_inner):
        cells.append(_parsed_cell(m))
        last_end = m.end()
    open_cell = _OPEN_CELL_RE.match(row_inner, last_end)
    if open_cell and open_cell.group(1).strip():
        cells.append((_cell_text(open_cell.group(1)), 1, 1))
    return cells


def _parse_rows(fragment: str) -> list[list[tuple[str, int, int]]]:
    """All rows in a fragment. A missing mid-table ``</tr>`` would make the
    non-greedy row regex fuse two rows (values shifting under wrong headers),
    so each matched row body is re-split on interior ``<tr>`` markers."""
    rows: list[list[tuple[str, int, int]]] = []
    for m in _TR_RE.finditer(fragment):
        for part in _TR_SPLIT_RE.split(m.group(1)):
            cells = _row_cells(part)
            if cells:
                rows.append(cells)
    return rows


def _grid_from_rows(parsed: list[list[tuple[str, int, int]]]) -> list[list[str]]:
    """Expand colspan/rowspan into a rectangular grid of cell strings.

    colspan pads N-1 empty cells after the value; rowspan reserves the same
    column(s) in the following rows so later values are not shifted under the
    wrong header.
    """
    grid: list[list[str]] = []
    pending: dict[int, int] = {}  # col index -> rows still occupied from above
    for cells in parsed:
        out: list[str] = []
        col = 0
        i = 0
        max_pending = max(pending) if pending else -1
        while i < len(cells) or col <= max_pending:
            if pending.get(col, 0) > 0:
                out.append("")
                pending[col] -= 1
                if pending[col] == 0:
                    del pending[col]
                    max_pending = max(pending) if pending else -1
                col += 1
                continue
            if i < len(cells):
                text, cspan, rspan = cells[i]
                i += 1
                for k in range(cspan):
                    out.append(text if k == 0 else "")
                    if rspan > 1:
                        pending[col] = pending.get(col, 0) + (rspan - 1)
                        max_pending = max(max_pending, col)
                    col += 1
            else:
                out.append("")
                col += 1
        grid.append(out)
    return grid


def _rows_to_markdown(parsed: list[list[tuple[str, int, int]]]) -> str:
    grid = _grid_from_rows(parsed)
    if not grid:
        return ""
    width = max(len(r) for r in grid)
    lines: list[str] = []
    for idx, row in enumerate(grid):
        padded = row + [""] * (width - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if idx == 0:
            lines.append("|" + "|".join(["---"] * width) + "|")
    return "\n".join(lines)


def html_table_to_markdown(table_html: str) -> str:
    """One HTML table -> a GitHub-markdown table (or "" if nothing parses)."""
    return _rows_to_markdown(_parse_rows(table_html))


def _consume_truncated_table(tail: str) -> tuple[str, int] | None:
    """Parse a truncated (unclosed) table from the START of ``tail``.

    Consumes consecutive closed rows (plus tolerated between-row markup) and
    at most one trailing open row that runs to end-of-text. Returns
    ``(markdown, chars_consumed)``, or None when no rows parse — the caller
    must then leave the text unchanged. Content after the last parsed row is
    NEVER consumed, so trailing prose survives verbatim.
    """
    parsed: list[list[tuple[str, int, int]]] = []
    pos = 0
    while pos < len(tail):
        skip = _BETWEEN_ROWS_RE.match(tail, pos)
        if skip:
            pos = skip.end()
            continue
        row = _TR_RE.match(tail, pos)
        if row:
            for part in _TR_SPLIT_RE.split(row.group(1)):
                cells = _row_cells(part)
                if cells:
                    parsed.append(cells)
            pos = row.end()
            continue
        break
    # One trailing open row, only when it runs cleanly to end-of-text.
    open_row = re.compile(r"<tr[^>]*>(.*)\Z", _FLAGS).match(tail, pos)
    if open_row:
        inner = open_row.group(1)
        inner_cells: list[tuple[str, int, int]] = []
        last_end = 0
        for m in _CELL_RE.finditer(inner):
            inner_cells.append(_parsed_cell(m))
            last_end = m.end()
        rest = inner[last_end:]
        open_cell = _OPEN_CELL_RE.match(rest)
        if open_cell and open_cell.group(1).strip():
            inner_cells.append((_cell_text(open_cell.group(1)), 1, 1))
            rest = ""
        if inner_cells and not rest.strip():
            parsed.append(inner_cells)
            pos = len(tail)
    if not parsed:
        return None
    md = _rows_to_markdown(parsed)
    if not md:
        return None
    return md, pos


def convert_html_tables(text: str) -> str:
    """Replace HTML table blocks with their markdown equivalents.

    NEVER deletes content: a ``<table>`` block with no parseable rows is left
    in place (the residual tag-strip separates its content), and for a
    trailing unclosed table only the parsed rows are consumed — anything
    after them is re-emitted verbatim.
    """
    if not _TABLE_OPEN_RE.search(text):
        return text

    def _replace_closed(match: re.Match) -> str:
        md = html_table_to_markdown(match.group(1))
        # No parseable rows: leave the block for the residual tag pass, which
        # separates the content instead of deleting it.
        return f"\n{md}\n" if md else match.group(0)

    text = _TABLE_RE.sub(_replace_closed, text)

    # Trailing truncated table: convert only the LAST remaining '<table' and
    # only when actual rows follow it.
    last_open = None
    for m in _TABLE_OPEN_RE.finditer(text):
        last_open = m
    if last_open is None:
        return text
    tail = text[last_open.end() :]
    consumed = _consume_truncated_table(tail)
    if consumed is None:
        return text
    md, end = consumed
    return text[: last_open.start()] + f"\n{md}\n" + tail[end:]


def strip_html_tags(text: str) -> str:
    """Strip allowlisted HTML tags without ever fusing adjacent content or
    growing spaces inside words.

    ``<sup>/<sub>`` keep their content as ``^content``/``_content``; ``<br>``
    becomes a newline; residual cell/row tags become ``|``-separators /
    newlines; inline formatting tags vanish in place; remaining structural
    tags become a space. Unknown angle-bracket content is preserved.
    """
    text = _convert_sup_sub(text)
    text = _BR_RE.sub("\n", text)
    text = _RESIDUAL_ROW_RE.sub("\n", text)
    text = _RESIDUAL_CELL_RE.sub(" | ", text)
    text = _INLINE_TAG_RE.sub("", text)
    text = _STRUCTURAL_TAG_RE.sub(" ", text)
    return text


def _decode_entities(text: str) -> str:
    """Decode HTML entities AFTER tag stripping (a decoded ``&lt;`` must
    never be re-parsed as markup)."""
    if "&" not in text:
        return text
    return html.unescape(text)


def clean_residual_html(text: str) -> str:
    """Tables -> markdown, then safe tag strip, then entity decode.

    The single entry point for engine cleaners (normalizer, deepseek-vllm).
    """
    text = convert_html_tables(text)
    text = strip_html_tags(text)
    text = _decode_entities(text)
    # Tag replacement leaves doubled spaces mid-line; tidy WITHOUT touching
    # line-leading whitespace (indented code blocks, nested lists).
    text = re.sub(r"(?<=\S)[ \t]{2,}", " ", text)
    return text
