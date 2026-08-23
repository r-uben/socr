"""Reconcile whole-page table OCR (Pass A) against the crop-pass reading (Pass B).

The crop pass re-reads each table from a high-resolution crop and is treated as
authoritative. This module:

1. Finds the GitHub-markdown table blocks already present in the page OCR (Pass A).
2. Pairs them, in reading order, with the crop-pass tables (Pass B).
3. Does a structural, cell-by-cell diff — no similarity threshold, no magic
   number: any differing cell is a disagreement, and the exact (row, col, old,
   new) changes are reported (the same "name the misread" behaviour the judge
   validated).
4. On disagreement, patches B back into the page markdown — but only when B is a
   well-formed table whose column count matches A. If the table counts don't line
   up or B is malformed, it flags the page without editing (a sloppy auto-edit on
   a research corpus is worse than a flag).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PIPE_LINE = re.compile(r"^\s*\|?.*\|.*$")  # a markdown table row contains a pipe
_SEP_CELL = re.compile(r"^:?-{1,}:?$")  # ---, :---, ---:, :---: separator cell

# GH-95: the note attached to a disagreement that *could* have been auto-patched
# but was left flag-only because ``--auto-patch-tables`` is off. Named rather than
# inlined so the trust sidecar can identify patch-eligible pages by constant
# comparison instead of parsing prose out of the audit log.
PATCH_ELIGIBLE_NOTE = "eligible for patch; flag-only (enable --auto-patch-tables)"


@dataclass
class CellDiff:
    """One cell that differs between Pass A and Pass B."""

    row: int
    col: int
    page_value: str  # what the whole-page OCR said (Pass A)
    crop_value: str  # what the crop pass said (Pass B), now authoritative


@dataclass
class TableDisagreement:
    """A reconciled table where the two passes disagreed."""

    table_index: int  # reading-order index on the page
    source: str  # "ruled" | "booktabs" (locator that found it)
    action: str  # "patched" | "flagged"
    changed_cells: list[CellDiff] = field(default_factory=list)
    note: str = ""  # why flagged-only, if applicable

    def summary(self) -> str:
        if self.changed_cells:
            head = self.changed_cells[0]
            extra = f" (+{len(self.changed_cells) - 1} more)" if len(self.changed_cells) > 1 else ""
            return f"table {self.table_index}: '{head.page_value}' -> '{head.crop_value}'{extra}"
        return f"table {self.table_index}: {self.note or 'mismatch'}"


@dataclass
class TableReconcileResult:
    """Outcome of reconciling one page's tables."""

    text: str  # page markdown (patched if any)
    patched: bool = False
    disagreements: list[TableDisagreement] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.disagreements)


# --------------------------------------------------------------------------
# Markdown table parsing
# --------------------------------------------------------------------------


@dataclass
class _Block:
    start: int  # first line index (inclusive)
    end: int  # last line index (inclusive)
    grid: list[list[str]]  # parsed cells, separator row dropped


def find_table_blocks(markdown: str) -> list[_Block]:
    """Locate markdown table blocks: runs of >= 2 consecutive pipe-bearing lines.

    Two lines is the GitHub-markdown minimum (header + separator). Prose almost
    never produces consecutive pipe lines, so this is robust without a heuristic
    cutoff.
    """
    lines = markdown.splitlines()
    blocks: list[_Block] = []
    i = 0
    n = len(lines)
    while i < n:
        if _is_table_line(lines[i]):
            j = i
            while j < n and _is_table_line(lines[j]):
                j += 1
            if j - i >= 2:
                grid = _parse_grid(lines[i:j])
                if grid:
                    blocks.append(_Block(start=i, end=j - 1, grid=grid))
            i = j
        else:
            i += 1
    return blocks


#: GFM permits a delimiter cell with one hyphen, but this shipping-policy
#: predicate deliberately requires the conventional three. One- and two-dash
#: fragments are easy products of prose or truncated output; rejecting them
#: fails closed to the existing native/marker behavior. Keep this separate from
#: ``_SEP_CELL``, whose permissive grammar remains part of reconciliation.
_STRICT_SEPARATOR_MIN_HYPHENS = 3
_STRICT_SEP_CELL = re.compile(rf"^:?-{{{_STRICT_SEPARATOR_MIN_HYPHENS},}}:?$")

#: A fence opener: three or more backticks or tildes, optionally indented and
#: followed by an info string. Content inside a fence is a code sample, not
#: markdown structure.
_FENCE_LINE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

#: A closing fence may be longer than its opener, but never shorter, and may
#: carry only trailing whitespace -- not an opener-style info string.
_FENCE_CLOSER = re.compile(r"^\s{0,3}(`{3,}|~{3,})[ \t]*$")

#: GH-268 requires consistent column counts across every row of an authored
#: grid. Keep the policy named because accepting a ragged block here can either
#: displace native text (#259) or suppress the D3 fail-closed marker (#262).
STRICT_GRID_REQUIRES_UNIFORM_BODY = True


#: An HTML comment, possibly spanning lines. A grid inside one is commented-out
#: content, not a reading of the page.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

#: CommonMark's OTHER code construct: four spaces (or a tab) of indentation.
#: ``_PIPE_LINE`` allows leading whitespace and ``_split_row`` strips it, so
#: without this an indented code sample reads as table structure.
_INDENTED_CODE = re.compile(r"^(?: {4,}|\t)")

# GH-226: these commands describe LaTeX table structure, not mathematical
# notation.  If one survives inside a Markdown table, the span/rule it encoded
# has not been represented as GFM structure.  Keep the set deliberately narrow:
# ordinary cell math (``\\alpha``, ``\\frac``, ...) remains valid content.
_LATEX_TABLE_COMMAND = re.compile(r"\\(?:multicolumn|multirow|cline|hline)\b")

# GFM type-1 raw HTML blocks whose contents are literal/preformatted rather
# than Markdown. Keep this emission-only: ``_markdown_content_lines`` is also
# the provenance policy for GH-268's grid-selection decisions.
_RAW_LITERAL_BLOCK_OPEN = re.compile(
    r"^\s{0,3}<(?P<tag>pre|script|style|textarea)(?:[\s>]|$)", re.IGNORECASE
)
_INLINE_HTML_CODE = re.compile(r"<code(?:\s[^>]*)?>.*?</code\s*>", re.IGNORECASE)

TABLE_EMISSION_NONE = ""
TABLE_EMISSION_LATEX_LEAK = "table_latex_leak"
TABLE_EMISSION_WIDTH_MISMATCH = "table_width_mismatch"


def _strip_html_comments(markdown: str) -> str:
    """Blank out HTML comments, keeping line count so nothing shifts.

    An UNCLOSED ``<!--`` swallows the rest of the document, by the same
    fail-closed reasoning as an unclosed fence.
    """
    text = _HTML_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), markdown)
    head, sep, _ = text.partition("<!--")
    return head if sep else text


def _row_border(line: str) -> tuple[bool, bool]:
    """Whether a row is written with a leading / trailing pipe.

    A real table is written consistently by whatever emitted it, so its header
    and its separator share a border style. A prose sentence that happens to
    sit above a separator generally does not -- which is what separates
    ``noise | a`` from a header.
    """
    stripped = line.strip()
    return stripped.startswith("|"), stripped.endswith("|")


def _strip_fenced_regions(lines: list[str]) -> list[str]:
    """Blank out fenced code content so it is never read as markdown structure.

    A table inside a backtick or tilde fence is a code SAMPLE -- the model
    showing what a grid looks like, or echoing the prompt -- not a reading of
    the page. The fence lines themselves go too, so a grid opening immediately after a fence
    with no blank line between them cannot inherit the fence's row.

    An UNCLOSED fence swallows the rest of the document, on purpose: the
    document is malformed, and this predicate suppresses a fail-closed marker,
    so the honest answer under malformation is "no grid here".
    """
    out: list[str] = []
    fence: tuple[str, int] | None = None
    for line in lines:
        if fence is None:
            m = _FENCE_LINE.match(line)
            if m:
                run = m.group(1)
                fence = (run[0], len(run))
                out.append("")
                continue
            out.append(line)
        else:
            # CommonMark: the closer uses the same character, is at least as
            # long as the opener, and has no info string.
            m = _FENCE_CLOSER.match(line)
            if m:
                run = m.group(1)
                opener_char, opener_length = fence
                if run[0] == opener_char and len(run) >= opener_length:
                    fence = None
            out.append("")
    return out


def _markdown_content_lines(markdown: str) -> list[str]:
    """Markdown lines that may represent emitted page content.

    Code samples and comments can legitimately show malformed-table examples;
    they are not page readings and must never trip a shipping defect.
    """
    text = _strip_html_comments(markdown.replace("\r\n", "\n").replace("\r", "\n"))
    lines = _strip_fenced_regions(text.splitlines())
    return ["" if _INDENTED_CODE.match(line) else line for line in lines]


def _strip_emission_literal_blocks(lines: list[str]) -> list[str]:
    """Blank raw-HTML literal blocks without changing line positions."""
    out: list[str] = []
    tag: str | None = None
    for line in lines:
        if tag is None:
            match = _RAW_LITERAL_BLOCK_OPEN.match(line)
            if match is None:
                out.append(line)
                continue
            tag = match.group("tag").lower()
        out.append("")
        if re.search(rf"</{re.escape(tag)}\s*>", line, re.IGNORECASE):
            tag = None
    return out


def _strip_inline_code_spans(text: str) -> str:
    """Remove Markdown/HTML inline code before scanning for live commands."""
    chars = list(_INLINE_HTML_CODE.sub("", text))
    i = 0
    while i < len(chars):
        if chars[i] != "`":
            i += 1
            continue
        opener = i
        while i < len(chars) and chars[i] == "`":
            i += 1
        opener_len = i - opener
        cursor = i
        closer: tuple[int, int] | None = None
        while cursor < len(chars):
            if chars[cursor] != "`":
                cursor += 1
                continue
            run_start = cursor
            while cursor < len(chars) and chars[cursor] == "`":
                cursor += 1
            if cursor - run_start == opener_len:
                closer = (run_start, cursor)
                break
        if closer is None:
            continue
        for index in range(opener, closer[1]):
            chars[index] = " "
        i = closer[1]
    return "".join(chars)


def _contains_live_latex_table_command(cell: str) -> bool:
    """Whether a cell contains an unescaped, non-literal table command."""
    text = _strip_inline_code_spans(cell)
    for match in _LATEX_TABLE_COMMAND.finditer(text):
        backslashes = 0
        cursor = match.start() - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return True
    return False


def table_emission_defect(markdown: str | None) -> str:
    """Return a deterministic raw-row defect in an emitted Markdown table.

    Unlike :func:`find_table_blocks`, this check retains the delimiter row.
    That closes GH-226's blind spot where the header and every body row agree
    on a width but the delimiter alone is narrower or wider; dropping the
    delimiter makes the parsed grid look perfectly rectangular.

    A run qualifies only when its first two rows are an authored header and a
    strict GFM delimiter with matching border style.  This is the same
    provenance boundary used by the strict shipping predicate, and code,
    comments, prose outside table runs, and ambiguous multi-delimiter runs are
    ignored.  Generic ragged bodies remain owned by the existing grid-shape
    policy (including GH-259's keep-and-flag disposition).
    """
    if not markdown:
        return TABLE_EMISSION_NONE
    lines = _strip_emission_literal_blocks(_markdown_content_lines(markdown))
    i, n = 0, len(lines)
    while i < n:
        if not _is_table_line(lines[i]):
            i += 1
            continue
        j = i
        while j < n and _is_table_line(lines[j]):
            j += 1
        block = lines[i:j]
        rows = [_split_emission_row(line) for line in block]
        separator_indices = [idx for idx, cells in enumerate(rows) if _is_separator_row(cells)]
        if (
            len(rows) >= 2
            and separator_indices == [1]
            and _row_border(block[0]) == _row_border(block[1])
            and any(cell.strip() for cell in rows[0])
        ):
            if any(_contains_live_latex_table_command(cell) for row in rows for cell in row):
                return TABLE_EMISSION_LATEX_LEAK

            content_widths = {len(row) for idx, row in enumerate(rows) if idx != 1}
            if len(content_widths) == 1 and len(rows[1]) != next(iter(content_widths)):
                return TABLE_EMISSION_WIDTH_MISMATCH
        i = j
    return TABLE_EMISSION_NONE


def has_strict_table_grid(markdown: str) -> bool:
    """Whether ``markdown`` contains a real GitHub-markdown table.

    ``find_table_blocks`` asks only for two consecutive pipe-bearing lines that
    parse to a non-empty cell grid, and its docstring states the safety
    assumption rather than checking it: "Prose almost never produces
    consecutive pipe lines". Measured at ``origin/main``: prose with a pipe on
    two adjacent lines returns one block, and so does a header plus separator
    with no body row.

    That looseness is harmless where a false positive costs a wasted
    comparison. It is NOT harmless in the #262 D3 keep predicate, where "an
    attempt authored a grid" is what SUPPRESSES a fail-closed marker. A phantom
    grid there silences the marker on a page with nothing usable on it, which
    inverts the ticket's own intent: the marker must yield to a real
    extraction, never to a phantom one.

    WHAT THIS IS, STATED PLAINLY (round 5). It is a deliberately CONSERVATIVE
    INTERIM, not a sound decision procedure, and it is not claimed to be one.
    Round 4 claimed "a case nobody anticipated fails closed"; a reviewer
    falsified that within the hour with three inputs -- a grid inside an
    indented code block, a grid inside an HTML comment, and a prose line acting
    as a header. None of those are shape violations. Each contains a
    syntactically perfect GitHub-markdown table. They are PROVENANCE failures:
    the text is a real grid that is not a reading of the page. No markdown
    predicate can see that, because by markdown the input is correct.

    GH-268 makes this the policy predicate for the D3 and S1 keep decisions.
    ``flagged_model_page_output`` uses the same structural scan through
    ``has_authored_table_grid``, but follows #259's keep-and-flag ruling for a
    ragged body. The permissive reconciliation parser is intentionally not a
    shipping-policy oracle.

    Until then the operative rule is FAIL CLOSED: where the answer is unclear,
    return False and let the failed-table marker ship. Refusing a real grid
    costs the marker, which is exactly what ships today; accepting a phantom
    silently replaces a page with text that describes nothing. Those costs are
    not symmetric, so neither is this predicate.

    WHY THE POSITIVE SHAPE CHECK, AND NOT A LIST OF REJECTIONS. Rounds 1-3 each
    bounded this by enumerating what to refuse, and each list turned out to be
    incomplete -- the failure shape that took #259/#260 through three
    rejections. A denylist grown one reviewer at a time does not converge. But
    unlike that problem, this one is bounded: a GitHub-markdown table has a
    DEFINED shape, so the predicate asserts the shape and returns False for
    everything else. A case nobody anticipated fails CLOSED, which is the
    honest outcome here -- the marker is a true statement about the page and a
    phantom grid is not.

    The shape, at the start of a run of consecutive pipe-bearing lines that is
    not inside a code fence::

        rows[0]      header     N cells, N >= 1; not entirely blank
        rows[1]      separator  N cells, EVERY cell matching :?-{3,}:?
        rows[2:]     body       at least one CONTENT row; every row has N cells

    The separator is at index 1 specifically, never merely "somewhere"; the
    body starts strictly at index 2, so a separator can never also be counted
    as the body row after itself.

    ``find_table_blocks`` itself is deliberately UNCHANGED. Other callers
    depend on its looseness, and tightening it globally would be a silent
    behaviour change across the codebase. This predicate is the stricter,
    strict shipping-policy layer.
    """
    return _has_table_grid(markdown, require_uniform_body=STRICT_GRID_REQUIRES_UNIFORM_BODY)


def has_authored_table_grid(markdown: str) -> bool:
    """Whether model text contains a table reading for #259 to keep and flag.

    This uses the same positive markdown shape and provenance guards as
    ``has_strict_table_grid``: a real delimiter row, a nonblank header, a
    content-bearing body, consistent border style, and no code/comment grid.
    It differs only on body width. Under #259's owner ruling, a ragged table is
    still authored table content and must remain visible with its structural
    flag instead of being replaced by the already-distrusted native reading.

    D3 and S1 deliberately do not use this predicate: their fail-closed choice
    is between a strict grid and the existing marker/native behavior.
    """
    return _has_table_grid(markdown, require_uniform_body=False)


def _has_table_grid(markdown: str, *, require_uniform_body: bool) -> bool:
    """Scan markdown for the shared table shape under a caller's width policy."""
    # Order matters: comments first (they can contain fence markers), then
    # fences, then indented code (a fence's CONTENT may be indented, and is
    # already blanked by then). CRLF is normalised so a trailing \r cannot
    # make a separator cell fail to match.
    lines = _markdown_content_lines(markdown)
    i, n = 0, len(lines)
    while i < n:
        if not _is_table_line(lines[i]):
            i += 1
            continue
        j = i
        while j < n and _is_table_line(lines[j]):
            j += 1
        block = lines[i:j]
        if _run_contains_grid(
            [_split_row(line) for line in block],
            [_row_border(line) for line in block],
            require_uniform_body=require_uniform_body,
        ):
            return True
        i = j
    return False


def _run_contains_grid(
    rows: list[list[str]],
    borders: list[tuple[bool, bool]],
    *,
    require_uniform_body: bool,
) -> bool:
    """True iff the header/separator/body shape starts at the first row.

    FAIL-CLOSED RULE, and the reason it lives here rather than inside the
    per-offset check: a block containing MORE THAN ONE separator row is
    ambiguous about where its table starts, so it is refused outright. That is
    what closes the reviewer's ``noise | a`` phantom, whose block carries a
    doubled separator -- and note the remedy proposed with that finding (a
    header whose cells are "not all separators and not all blank") does NOT
    close it: ``['noise', 'a']`` is neither, so it would have been accepted.
    Measured before choosing this rule instead.
    """
    separators = sum(1 for cells in rows if _is_separator_row(cells))
    if separators != 1:
        return False
    # FAIL-CLOSED: the header is the run's FIRST line, not "some offset in it".
    # Round 4 scanned every offset so a genuine grid would not be lost to a
    # stray pipe-bearing sentence directly above it. That flexibility is
    # exactly the room the reviewer's prose-as-header phantom lived in, and the
    # cost of removing it is a marker on a real grid whose run happens to open
    # with prose and no blank line between -- which is today's behaviour, not a
    # new loss. A blank line still separates runs, so a grid after a paragraph
    # is unaffected.
    return _grid_starts_at(rows, 0, borders, require_uniform_body=require_uniform_body)


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(_STRICT_SEP_CELL.match(cell.strip()) for cell in cells)


def _grid_starts_at(
    rows: list[list[str]],
    i: int,
    borders: list[tuple[bool, bool]],
    *,
    require_uniform_body: bool,
) -> bool:
    """The shape, anchored at ``i`` (always 0 -- see ``_run_contains_grid``).

    Every exit is False; the only True is the end.
    """
    if i + 2 >= len(rows) + 1:  # need a header and a separator at minimum
        return False
    if i + 1 >= len(rows):
        return False
    header, separator = rows[i], rows[i + 1]
    width = len(header)
    if len(separator) != width:
        return False
    if not _is_separator_row(separator):
        return False
    # FAIL-CLOSED: the header must be written in the SAME border style as the
    # separator. A table is emitted consistently by whatever wrote it; a prose
    # line that happens to precede a separator generally is not. This is what
    # separates ``noise | a`` (no outer pipes) from the ``| --- | --- |`` under
    # it, while leaving the legitimate all-bordered and all-unbordered
    # spellings alone. A heuristic, deliberately: under a fail-closed interim
    # the cost of refusing a real grid is the marker, which is today's
    # behaviour, and the cost of accepting a phantom is silent content loss.
    if borders[i] != borders[i + 1]:
        return False
    # A header may not itself be a separator, nor be entirely blank.
    if _is_separator_row(header) or not any(cell.strip() for cell in header):
        return False
    body = rows[i + 2 :]
    if not body:
        return False  # header + separator and nothing under it

    def _is_body_row(cells: list[str], *, require_width: bool) -> bool:
        """A body row carries CONTENT, at header width when required.

        Empty punctuation and a second separator are structure, not a reading.
        The #259 authored-content policy permits any body width because both a
        narrower row and GH-276's wider-body shape are ragged tables to keep
        and flag; the strict D3/S1 policy still requires the header's width.
        """
        if require_width and len(cells) != width:
            return False
        if not any(cell.strip() for cell in cells):
            return False
        return not all(_STRICT_SEP_CELL.match(cell.strip()) for cell in cells)

    if require_uniform_body:
        # Per spec: every row under the separator belongs to the grid.
        return all(len(cells) == width for cells in body) and any(
            _is_body_row(cells, require_width=True) for cells in body
        )
    return any(_is_body_row(cells, require_width=False) for cells in body)


def _is_table_line(line: str) -> bool:
    return "|" in line and bool(_PIPE_LINE.match(line))


def _parse_grid(rows: list[str]) -> list[list[str]]:
    """Parse markdown rows into a cell grid, dropping the separator row."""
    grid: list[list[str]] = []
    for row in rows:
        cells = _split_row(row)
        if cells and all(_SEP_CELL.match(c.strip()) for c in cells if c.strip()):
            continue  # separator row (---|---)
        if cells:
            grid.append(cells)
    return grid


def _split_row(row: str) -> list[str]:
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _split_emission_row(row: str) -> list[str]:
    """Split a raw row without treating an escaped pipe as a cell boundary.

    The reconciliation parser predates GH-226 and deliberately remains
    unchanged. The final emission guard only needs cell counts, and must not
    newly reject valid GFM cells such as ``A \\| B``.
    """

    def _escaped(text: str, index: int) -> bool:
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        return backslashes % 2 == 1

    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not _escaped(s, len(s) - 1):
        s = s[:-1]

    cells: list[str] = []
    start = 0
    for index, char in enumerate(s):
        if char == "|" and not _escaped(s, index):
            cells.append(s[start:index].strip())
            start = index + 1
    cells.append(s[start:].strip())
    return cells


# --------------------------------------------------------------------------
# Diff
# --------------------------------------------------------------------------


def _norm(cell: str) -> str:
    """Normalise a cell for comparison: collapse whitespace, unify minus signs.

    Formatting-only differences (spacing, unicode minus vs hyphen) are not
    corruption; a changed digit or sign is.
    """
    s = " ".join(cell.split())
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    return s


def diff_grids(grid_a: list[list[str]], grid_b: list[list[str]]) -> list[CellDiff]:
    """Cell-by-cell diff. Shape differences surface as diffs on the off-grid cells."""
    diffs: list[CellDiff] = []
    rows = max(len(grid_a), len(grid_b))
    for r in range(rows):
        row_a = grid_a[r] if r < len(grid_a) else []
        row_b = grid_b[r] if r < len(grid_b) else []
        cols = max(len(row_a), len(row_b))
        for c in range(cols):
            va = row_a[c] if c < len(row_a) else ""
            vb = row_b[c] if c < len(row_b) else ""
            if _norm(va) != _norm(vb):
                diffs.append(CellDiff(row=r, col=c, page_value=va, crop_value=vb))
    return diffs


def _well_formed(grid: list[list[str]]) -> bool:
    """A patch-worthy crop grid: >= 2 rows and a consistent column count."""
    if len(grid) < 2:
        return False
    widths = {len(row) for row in grid}
    return len(widths) == 1 and next(iter(widths)) >= 1


def _col_count(grid: list[list[str]]) -> int:
    return max((len(row) for row in grid), default=0)


# --------------------------------------------------------------------------
# Reconcile
# --------------------------------------------------------------------------


def reconcile_page_tables(
    page_markdown: str,
    crop_tables: list[tuple[str, str]],
    *,
    auto_patch: bool = False,
) -> TableReconcileResult:
    """Reconcile a page's OCR tables against crop-pass readings.

    ``crop_tables`` is a list of ``(markdown, source)`` in reading order, where
    ``source`` is the locator tag ("ruled"/"booktabs") for reporting.

    ``auto_patch`` (default False = flag-only): when False, the page text is never
    modified — disagreements are reported but the corpus is left untouched. The
    crop reader's numeric fidelity is unproven, and a silent wrong patch to a
    research number (a model can keep table shape and still change 0.031->0.037)
    is worse than a missed correction. Opt in with ``--auto-patch-tables`` once the
    crop reader is trusted. When True, an eligible disagreement (well-formed crop,
    matching column count) is patched in; the rest are still flag-only.

    Returns the (possibly patched) page text plus per-table disagreements.
    """
    result = TableReconcileResult(text=page_markdown)
    if not crop_tables:
        return result

    blocks = find_table_blocks(page_markdown)

    if len(blocks) != len(crop_tables):
        # Counts don't line up: we cannot safely map crop tables to page blocks
        # (the engine may have emitted a table as plain text, or merged two).
        # Flag, never guess where to splice.
        result.notes.append(
            f"table count mismatch: {len(blocks)} in page OCR vs "
            f"{len(crop_tables)} located/cropped — flagged, not patched"
        )
        for idx, (_, source) in enumerate(crop_tables):
            result.disagreements.append(
                TableDisagreement(
                    table_index=idx,
                    source=source,
                    action="flagged",
                    note="count mismatch (no safe patch target)",
                )
            )
        return result

    # Counts match: pair by reading-order index. Patch from the bottom up so line
    # offsets of earlier blocks stay valid as we splice.
    lines = page_markdown.splitlines()
    patches: list[tuple[_Block, str]] = []  # (block, replacement) applied later
    for idx in range(len(blocks)):
        block = blocks[idx]
        crop_md, source = crop_tables[idx]
        crop_grid = _parse_grid(crop_md.splitlines())
        diffs = diff_grids(block.grid, crop_grid)
        if not diffs:
            continue  # passes agree — high confidence, nothing to do

        if _well_formed(crop_grid) and _col_count(crop_grid) == _col_count(block.grid):
            if auto_patch:
                patches.append((block, crop_md.strip()))
                result.disagreements.append(
                    TableDisagreement(
                        table_index=idx, source=source, action="patched", changed_cells=diffs
                    )
                )
            else:
                # Eligible to patch, but flag-only by default: surface the exact
                # changes for review without editing the corpus.
                result.disagreements.append(
                    TableDisagreement(
                        table_index=idx,
                        source=source,
                        action="flagged",
                        changed_cells=diffs,
                        note=PATCH_ELIGIBLE_NOTE,
                    )
                )
        else:
            result.disagreements.append(
                TableDisagreement(
                    table_index=idx,
                    source=source,
                    action="flagged",
                    changed_cells=diffs,
                    note="crop reading malformed or column count differs — not patched",
                )
            )

    if patches:
        for block, replacement in sorted(patches, key=lambda p: p[0].start, reverse=True):
            lines[block.start : block.end + 1] = replacement.splitlines()
        result.text = "\n".join(lines)
        result.patched = True

    return result
