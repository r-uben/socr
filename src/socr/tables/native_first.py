"""Native structured table as the first reader of a born-digital table page.

The deterministic gate is the existing native-table verifier (``VerifierState``
in ``native_verifier``): ``EXACT_PASS`` means the structured grid's numeric
tokens already match the PDF's own words, row for row, with no label-binding
failure and no row-count gap. That state is not a new threshold.

When the same check names a multiset mismatch that pins to one cell, the
caller sends a model only that cell. Anything that cannot be pinned — a
label-binding failure, a row-count gap, a structure defect, a numeric word
the rowizer dropped — fails closed. A lane-count warning with a clean value
guard does not name a cell; that page is ``DEFER`` and keeps the existing
whole-page route.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from socr.tables.native_verifier import (
    _MD_SEP_RE,
    VerifierState,
    _effective_native_rows_for_output,
    _normalize_numeric_token,
    _pair_output_to_native_rows,
    _parse_output_data_rows,
    _parse_output_row_cells,
    _verify_from_words,
    is_numeric_token,
)

SHIP = "ship"
CELLS = "cells"
REFUSE = "refuse"
DEFER = "defer"


@dataclass(frozen=True)
class FailingCell:
    """One grid cell whose numeric token disagrees with the text layer."""

    row_line: str
    cell_index: int
    grid_token: str
    native_token: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class NativeTablePlan:
    action: str
    cells: tuple[FailingCell, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class NativeTableFirstWork:
    """Planner output for one born-digital table page."""

    plan: NativeTablePlan
    #: When set, the upright rowizer produced this markdown and it must replace
    #: ``PageState.native_text`` before shipping or re-verifying cells.
    markdown: str | None = None
    structure_defective: bool | None = None
    header_unattributed: bool | None = None
    orphan_word_drops: tuple[dict, ...] = ()
    clear_ocr_enhancement: bool = True


@dataclass(frozen=True)
class RotatedNativeTableAttempt:
    """Upright re-read of a GH-147 refused rotated table page."""

    plan: NativeTablePlan
    markdown: str
    words: list
    structure_defective: bool
    header_unattributed: bool
    orphan_words: tuple[str, ...] = ()
    regions: tuple[tuple[object, str], ...] = ()
    orphan_drops: tuple[dict, ...] = ()


def upright_words_for_page(page) -> tuple[list, int]:
    """Word list in the upright frame ``rowize_from_words`` uses for *page*.

    Returns ``([], 0)`` when the page has no rotation or no words.
    """
    from socr.core.born_digital import upright_rotation_for
    from socr.tables.reconstruct import _rotate_word_bbox

    try:
        words = list(page.get_text("words"))
    except Exception:
        return [], 0
    if not words:
        return [], 0
    rotation = upright_rotation_for(page)
    if rotation == 0:
        return words, 0
    xs = [w[0] for w in words]
    ys = [w[1] for w in words]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    return [_rotate_word_bbox(w, cx, cy, rotation) for w in words], rotation


def attempt_rotated_native_table(page) -> RotatedNativeTableAttempt | None:
    """Rowize a rotated table page upright and run ``plan_native_table``.

    Returns None when the page is not rotated or the rowizer finds no grid.
    """
    from socr.core.born_digital import upright_rotation_for
    from socr.tables import structure_check
    from socr.tables.reconstruct import rowize_from_words

    rotation = upright_rotation_for(page)
    if rotation == 0:
        return None
    orphan_drops: list[dict] = []
    regions = rowize_from_words(page, orphan_drops=orphan_drops)
    if not regions:
        return None
    markdown = "\n\n".join(md for _rect, md in regions if (md or "").strip())
    if not markdown.strip():
        return None
    words, _ = upright_words_for_page(page)
    reports = structure_check.check_markdown(markdown)
    structure_defective = bool(
        structure_check.table_emission_defect(markdown)
        or structure_check.table_content_defect(markdown)
        or structure_check.structural_gate_fires(reports)
    )
    header_unattributed = False
    if not structure_defective:
        header_unattributed = (
            structure_check.table_output_defect(markdown, words)
            == structure_check.DEFECT_HEADER_UNATTRIBUTED
        )
    orphan_words = tuple(
        str(drop.get("word", ""))
        for drop in orphan_drops
        if isinstance(drop, dict) and drop.get("word")
    )
    plan = plan_native_table(
        words,
        markdown,
        structure_defective=structure_defective,
        header_unattributed=header_unattributed,
        orphan_words=list(orphan_words),
    )
    return RotatedNativeTableAttempt(
        plan=plan,
        markdown=markdown,
        words=words,
        structure_defective=structure_defective,
        header_unattributed=header_unattributed,
        orphan_words=orphan_words,
        regions=tuple(regions),
        orphan_drops=tuple(orphan_drops),
    )


def compose_upright_shipped_page(page, regions: list) -> str:
    """Interleave upright table regions with the page's surviving prose blocks."""
    from socr.core.born_digital import BornDigitalDetector

    return BornDigitalDetector().interleave_table_regions_into_page(page, list(regions))


def _markdown_table_tokens(table_markdown: str) -> set[str]:
    table_tokens: set[str] = set()
    for line in (table_markdown or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("| ---"):
            continue
        for cell in stripped.strip("|").split("|"):
            token = cell.strip()
            if token:
                table_tokens.add(token)
    return table_tokens


def retained_prose_lines_to_keep(retained: str, table_markdown: str) -> list[str]:
    """GH-147 prose lines ``splice_retained_prose_beside_table`` may prepend."""
    table_tokens = _markdown_table_tokens(table_markdown)
    kept: list[str] = []
    seen: set[str] = set()
    for line in (retained or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped in seen:
            continue
        if stripped in table_tokens:
            continue
        if stripped.replace(".", "").replace(",", "").isdigit():
            continue
        if len(stripped) <= 3 and stripped.isalpha() and stripped.isupper():
            continue
        words = stripped.split()
        if stripped.startswith("Table ") or len(words) >= 2 or len(stripped) >= 12:
            kept.append(stripped)
            seen.add(stripped)
    return kept


def splice_retained_prose_beside_table(
    retained: str,
    table_markdown: str,
    interleaved: str,
) -> str:
    """Prepend GH-147 prose lines that the upright grid does not already carry."""
    body = interleaved or ""
    extra = [
        line for line in retained_prose_lines_to_keep(retained, table_markdown) if line not in body
    ]
    if not extra:
        return body
    return "\n".join(extra + ["", body]).strip()


def retained_prose_survives(
    composed: str,
    retained: str,
    *,
    table_markdown: str = "",
) -> bool:
    """Whether every retained prose line the splice keeps is still in *composed*."""
    composed_text = composed or ""
    for line in retained_prose_lines_to_keep(retained, table_markdown):
        if line not in composed_text:
            return False
    return True


def plan_native_table(
    words: list,
    markdown: str,
    *,
    structure_defective: bool = False,
    header_unattributed: bool = False,
    unverifiable: bool = False,
    orphan_words: list[str] | None = None,
) -> NativeTablePlan:
    """Decide whether *markdown* may ship, needs cell reads, or must be refused.

    ``words`` is a PyMuPDF ``get_text("words")`` list for the same page.
    Blocking flags are the detector's existing structure verdicts, passed in
    so this function does not re-derive them.
    """
    if structure_defective:
        return NativeTablePlan(REFUSE, reason="structure_defective")
    if header_unattributed:
        return NativeTablePlan(REFUSE, reason="header_unattributed")
    if any(is_numeric_token(word) for word in (orphan_words or [])):
        # A number the rowizer dropped has no cell to put it back into.
        # Shipping the grid would drop it. The non-numeric orphan (a dagger,
        # ``n.a.``) stays an audit event, which is the GH-418 ruling.
        return NativeTablePlan(REFUSE, reason="numeric_orphan")
    if not any(_MD_SEP_RE.match(line) for line in (markdown or "").splitlines()):
        return NativeTablePlan(DEFER, reason="no_markdown_table")

    verdict = _verify_from_words(words or [], markdown or "", scope_label="native-first")
    if verdict.state == VerifierState.EXACT_PASS and not unverifiable:
        return NativeTablePlan(SHIP, reason="exact_pass")
    # A row-count gap makes the per-row pairing unreliable. Do not send a
    # model a cell chosen from that pairing, and do not ship the grid.
    if verdict.row_count_warn:
        return NativeTablePlan(REFUSE, reason="row_count")
    if verdict.hard_fail:
        if any(row.get("predicate") != "multiset_mismatch" for row in verdict.drifted_rows):
            return NativeTablePlan(REFUSE, reason="label_binding")
        cells = locate_mismatched_cells(words or [], markdown or "")
        if not cells:
            return NativeTablePlan(REFUSE, reason="unlocalizable_cells")
        return NativeTablePlan(CELLS, cells=tuple(cells), reason="cell_mismatch")
    if unverifiable:
        return NativeTablePlan(REFUSE, reason="region_unverifiable")
    return NativeTablePlan(DEFER, reason=verdict.state or VerifierState.AMBIGUOUS)


def locate_mismatched_cells(words: list, markdown: str) -> tuple[FailingCell, ...] | None:
    """Pin each paired-row multiset mismatch to one cell, or return None.

    None means at least one disagreement could not be placed without guessing
    a column. An empty tuple means every paired numeric cell already agrees.
    Alignment is left-to-right against the native words' own x order, and only
    when both sides have the same number of single-number cells. That equality
    is the verifier's own row pairing, not a new cutoff.
    """
    native_rows = _effective_native_rows_for_output(words, markdown, scope_label="native-first")
    pairs = _pair_output_to_native_rows(native_rows, _parse_output_data_rows(markdown))
    found: list[FailingCell] = []
    for (_row_idx, _count, row_text), native_tokens in pairs:
        numeric_cells: list[tuple[int, str]] = []
        for index, cell in enumerate(_parse_output_row_cells(row_text)):
            tokens = [tok for tok in re.split(r"\s+", cell) if tok and is_numeric_token(tok)]
            if not tokens:
                continue
            if len(tokens) != 1:
                return None
            numeric_cells.append((index, tokens[0]))
        native_sorted = sorted(native_tokens, key=lambda item: item[0])
        if len(numeric_cells) != len(native_sorted):
            return None
        for (cell_index, grid_token), (x_pos, native_token) in zip(numeric_cells, native_sorted):
            if _normalize_numeric_token(grid_token) == _normalize_numeric_token(native_token):
                continue
            bbox = _bbox_for_word(words, x_pos, native_token)
            if bbox is None:
                return None
            found.append(
                FailingCell(
                    row_line=row_text.strip(),
                    cell_index=cell_index,
                    grid_token=grid_token,
                    native_token=native_token,
                    bbox=bbox,
                )
            )
    return tuple(found)


def transcription_matches_native(heard: str, native_token: str, grid_token: str) -> bool:
    """True when the model read the text-layer number and not the bad grid token.

    Equality is the verifier's N2 normalization (``_normalize_numeric_token``):
    precision is kept and the token is not parsed as a float. The caller then
    writes the text-layer spelling, not the model's.
    """
    heard_norm = _normalize_numeric_token(heard.strip())
    native_norm = _normalize_numeric_token(native_token)
    grid_norm = _normalize_numeric_token(grid_token)
    return bool(heard_norm) and heard_norm == native_norm and heard_norm != grid_norm


def splice_cell_tokens(
    markdown: str,
    replacements: list[tuple[str, int, str, str]],
) -> str | None:
    """Replace named cell tokens, one pass per row, or return None.

    Each item is ``(row_line, cell_index, old, new)``. The row must occur
    once. The old token must occur once in that cell. Two edits of the same
    row are applied to the original cells together, so the second edit still
    finds the line the planner recorded.
    """
    if not replacements:
        return markdown
    lines = markdown.splitlines()
    grouped: dict[str, list[tuple[int, str, str]]] = {}
    for row_line, cell_index, old, new in replacements:
        grouped.setdefault(row_line.strip(), []).append((cell_index, old, new))
    for row_line, edits in grouped.items():
        hits = [index for index, line in enumerate(lines) if line.strip() == row_line]
        if len(hits) != 1:
            return None
        line_index = hits[0]
        cells = [cell.strip() for cell in lines[line_index].strip().strip("|").split("|")]
        seen: set[int] = set()
        for cell_index, old, new in edits:
            if cell_index in seen or cell_index < 0 or cell_index >= len(cells):
                return None
            seen.add(cell_index)
            if cells[cell_index].count(old) != 1:
                return None
            cells[cell_index] = cells[cell_index].replace(old, new, 1)
        lines[line_index] = "| " + " | ".join(cells) + " |"
    return "\n".join(lines)


def _bbox_for_word(
    words: list,
    x_pos: float,
    text: str,
) -> tuple[float, float, float, float] | None:
    """The word box whose x0 and text are the verifier's own token.

    The verifier stored ``x0`` off this same word list, so the match is
    identity, not a distance cutoff. Zero or several hits is not a cell.
    """
    hits: list[tuple[float, float, float, float]] = []
    for word in words:
        x0, y0, x1, y1, token, *_rest = word
        if token == text and x0 == x_pos:
            hits.append((x0, y0, x1, y1))
    if len(hits) != 1:
        return None
    return hits[0]
