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


# #262 round 3: a STRICT separator cell. ``_SEP_CELL`` above accepts a single
# dash (``|-|``), which ordinary prose and a truncated grid both reach by
# accident; three is the GitHub-markdown convention and is not something a
# sentence produces. Deliberately a SECOND pattern rather than a tightening of
# ``_SEP_CELL``: that one is read by ``_parse_grid`` on every caller's behalf.
_STRICT_SEP_CELL = re.compile(r"^:?-{3,}:?$")

#: #262 round 3: the minimum column count for a grid to count as authored. One
#: column is a list, not a table, and cannot carry the row/column binding the
#: D3 floor is arbitrating over.
_STRICT_MIN_COLUMNS = 2

#: A fence opener/closer: three or more backticks or tildes, optionally indented,
#: optionally followed by an info string. Content inside a fence is a code
#: sample, not markdown structure.
_FENCE_LINE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

#: #262 round 4, PENDING THE OWNER'S RULING -- do not silently flip this.
#:
#: The spec this predicate was built to requires every body row to carry the
#: header's cell count. That rejects a RAGGED but genuine grid, so such a page
#: ships the zero-content marker -- which contradicts #259's merged ruling to
#: FLAG AND KEEP a structurally defective grid (``kept_table_grid_defect``).
#: The reviewer recommends yielding to #259; the owner has not ruled.
#:
#: Isolated here as a single named switch so the decision is one line, not a
#: rewrite. False makes a body row count when it matches the header's width,
#: ignoring its ragged siblings; True (current, per spec) requires all of them.
STRICT_GRID_REQUIRES_UNIFORM_BODY = True


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
    fence: str | None = None
    for line in lines:
        m = _FENCE_LINE.match(line)
        if fence is None:
            if m:
                fence = m.group(1)[0]  # ` or ~
                out.append("")
                continue
            out.append(line)
        else:
            # Only a fence of the SAME character can close one (CommonMark).
            if m and m.group(1)[0] == fence:
                fence = None
            out.append("")
    return out


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

    WHY THIS IS A SHAPE CHECK AND NOT A LIST OF REJECTIONS. Rounds 1-3 each
    bounded this by enumerating what to refuse, and each list turned out to be
    incomplete -- the failure shape that took #259/#260 through three
    rejections. A denylist grown one reviewer at a time does not converge. But
    unlike that problem, this one is bounded: a GitHub-markdown table has a
    DEFINED shape, so the predicate asserts the shape and returns False for
    everything else. A case nobody anticipated fails CLOSED, which is the
    honest outcome here -- the marker is a true statement about the page and a
    phantom grid is not.

    The shape, at some offset ``i`` inside a run of consecutive pipe-bearing
    lines that is not inside a code fence::

        rows[i]      header     N cells, N >= 2
        rows[i + 1]  separator  N cells, EVERY cell matching :?-{3,}:?
        rows[i + 2:] body       at least one CONTENT row -- N cells, not blank,
                                and not itself a separator

    The separator is at ``i + 1`` specifically, never merely "somewhere"; the
    body starts strictly at ``i + 2``, so a separator can never also be counted
    as the body row after itself.

    The offset exists because a run may open with pipe-bearing prose before the
    real table starts. That is not laxity -- each candidate offset must satisfy
    the whole shape -- and it is what keeps a genuine grid from being lost to a
    stray sentence above it.

    ``find_table_blocks`` itself is deliberately UNCHANGED. Other callers
    depend on its looseness, and tightening it globally would be a silent
    behaviour change across the codebase. This is additive, with one caller.
    """
    lines = _strip_fenced_regions(markdown.splitlines())
    i, n = 0, len(lines)
    while i < n:
        if not _is_table_line(lines[i]):
            i += 1
            continue
        j = i
        while j < n and _is_table_line(lines[j]):
            j += 1
        if _run_contains_grid([_split_row(line) for line in lines[i:j]]):
            return True
        i = j
    return False


def _run_contains_grid(rows: list[list[str]]) -> bool:
    """True iff the header/separator/body shape holds at some offset in ``rows``."""
    return any(_grid_starts_at(rows, i) for i in range(len(rows)))


def _grid_starts_at(rows: list[list[str]], i: int) -> bool:
    """The shape, anchored at ``i``. Every exit is False; the only True is the end."""
    if i + 2 >= len(rows) + 1:  # need a header and a separator at minimum
        return False
    if i + 1 >= len(rows):
        return False
    header, separator = rows[i], rows[i + 1]
    width = len(header)
    if width < _STRICT_MIN_COLUMNS:
        return False
    if len(separator) != width:
        return False
    if not all(_STRICT_SEP_CELL.match(cell.strip()) for cell in separator):
        return False
    body = rows[i + 2 :]
    if not body:
        return False  # header + separator and nothing under it

    def _is_body_row(cells: list[str]) -> bool:
        """A body row carries CONTENT at the header's width.

        Three things a row of the right width can be and still not be content:
        empty (punctuation), a second separator (structure, not a reading --
        found by adversarial testing of this predicate, and the reason the body
        cannot simply be "the rows after index i+1"), or both.
        """
        if len(cells) != width:
            return False
        if not any(cell.strip() for cell in cells):
            return False
        return not all(_STRICT_SEP_CELL.match(cell.strip()) for cell in cells)

    if STRICT_GRID_REQUIRES_UNIFORM_BODY:
        # Per spec: every row under the separator belongs to the grid.
        return all(len(cells) == width for cells in body) and any(
            _is_body_row(cells) for cells in body
        )
    return any(_is_body_row(cells) for cells in body)


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
