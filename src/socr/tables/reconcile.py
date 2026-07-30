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
