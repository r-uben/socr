"""#1399: the stored old-OCR text layer as a witness for a model-read table on a scanned page.

On a page flagged ``prior_ocr_layer_suspect`` the layer is excluded from the evidence
(#1156/#1205) and tesseract on the table crops is the only witness. An audit of 30 rejected
scanned-table pages found that witness both too strict (it threw away 1,500 correct tokens: lost
signs and points) and too weak (it confirmed three real errors by repeating them: ``-.7``
shipped as ``-7``). The layer's number multiset inside the table crop held every printed number
on 10 of the 13 correctly read pages.

This module keeps the standing exclusion for everything else and uses the layer for ONE
purpose, this table check. It never serves as evidence for the page's prose, labels or any
other check. Four comparisons, all on numeric tokens with signs and decimal points verbatim:

A. forward: every table token of the reading is in the layer's tokens inside the located
   table crop(s), as a multiset.
B. mark conflict: a reading token no exact layer token consumed whose digits the layer or a
   tesseract rung holds with other marks. Tesseract may never rescue it.
C. format outlier, needs no witness: an integer cell in a row whose other numeric cells share one
   decimal count (a dropped decimal point looks like this).
D. reverse: layer tokens in the crop absent from the whole-page reading (a dropped number).

Position-blind by design (a multiset): two swapped cells pass all four. The strict row binding
downstream is what marks those.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from socr.tables.native_verifier import (
    _normalize_cell,
    _numeric_multiset_from_tokens,
    is_numeric_token,
)
from socr.tables.source_evidence import (
    TableTokens,
    _digit_key,
    _digit_keys_of_text,
    _evidence_table_blocks,
    _tokens_and_dash_led,
)
from socr.tables.structure_check import table_truncated

#: The OPEN OWNER RULING behind SUCCESS (#1399). ``False``: the old layer's agreement with the
#: reading is necessary but not enough for SUCCESS; the tesseract check (today's) must also pass,
#: so no page that fails today's check can newly ship SUCCESS, only WARNING with the table.
#: ``True``: exact agreement with the layer confirms a page alone (the 2026-11-18 ruling read
#: widely). The layer's multiset has no positions, so a table whose numbers are right and whose
#: labels are shifted agrees with it: the audit's misbound page a14 reaches SUCCESS under ``True``,
#: and strict row binding does not run on a suspect layer (``layer_can_judge_tables``). Origin: the
#: audit in ``docs/log/2026-11-21_1399_scanned_table_witness.md``. Not a threshold; a policy switch.
LAYER_CONFIRMS_ALONE: bool = False

#: Reasons, fixed text so a consumer can count them.
REASON_TRUNCATED = "the reading ends mid-table (cut off), so it cannot be checked as a table"
REASON_CHART = "the page is a chart or figure page and its table reading is not a table"


@dataclass(frozen=True)
class TableLayerWitness:
    """The stored layer's numeric tokens, split by where the word sits on the page.

    ``in_crop``: tokens of layer words whose centre lies inside a located table crop (the
    witness the table is compared with). ``whole``: every layer token, used only when no
    crop was located. ``dash_keys``: digit keys of layer words whose number is led by an en or
    em dash (they state no sign and support nothing, but their digits are veto evidence).
    ``crops_found``: a table crop was located (D needs it).
    """

    in_crop: Counter = field(default_factory=Counter)
    whole: Counter = field(default_factory=Counter)
    dash_keys: frozenset[str] = frozenset()
    crops_found: bool = False

    @property
    def has_numbers(self) -> bool:
        return bool(self.whole)


def table_layer_witness(page, boxes=None) -> TableLayerWitness | None:
    """The layer witness of *page*, or ``None`` when the page holds no layer words.

    ``boxes`` are the located table boxes (``locate_tables``); looked up here when omitted.
    A layer with no numeric token at all is not a witness either (``None``): there is nothing
    to compare the table with, and the caller keeps today's behaviour.
    """
    try:
        words = page.get_text("words")
    except Exception:  # noqa: BLE001 - an unreadable layer is no witness
        return None
    if not words:
        return None
    if boxes is None:
        try:
            from socr.tables.locate import locate_tables

            boxes = [b.bbox for b in locate_tables(page)]
        except Exception:  # noqa: BLE001 - no crop located, never a failure
            boxes = []
    in_crop: Counter = Counter()
    whole: Counter = Counter()
    dash_keys: set[str] = set()
    for x0, y0, x1, y1, text, *_ in words:
        numeric, _content, dash_led = _tokens_and_dash_led(str(text))
        if dash_led:
            dash_keys |= _digit_keys_of_text(str(text))
        if not numeric:
            continue
        whole.update(numeric)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if any(bx0 <= cx <= bx2 and by0 <= cy <= by2 for bx0, by0, bx2, by2 in boxes):
            in_crop.update(numeric)
    if not whole:
        return None
    return TableLayerWitness(
        in_crop=in_crop,
        whole=whole,
        dash_keys=frozenset(dash_keys),
        crops_found=bool(boxes),
    )


def page_is_chart(page) -> bool:
    """The existing chart detectors: vector or raster chart marks, or a figure caption in the
    page's own stored text (``has_chart_marks``, ``has_figure_caption``). Nothing new.

    A detector that raises says nothing about the page, so it is read as "not a chart": the
    checks that follow (A to D) still apply to the table.
    """
    from socr.figures.extractor import has_chart_marks
    from socr.figures.scanned_figures import has_figure_caption

    try:
        if has_chart_marks(page):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        return has_figure_caption(page.get_text() or "")
    except Exception:  # noqa: BLE001
        return False


def _multiset_missing(emitted: Counter, evidence: Counter) -> Counter:
    """The part of *emitted* the *evidence* multiset does not cover."""
    return Counter(
        {tok: n - evidence.get(tok, 0) for tok, n in emitted.items() if n > evidence.get(tok, 0)}
    )


def _by_key(tokens) -> dict[str, set[str]]:
    keyed: dict[str, set[str]] = {}
    for tok in tokens:
        keyed.setdefault(_digit_key(tok), set()).add(tok)
    return keyed


def mark_conflicts(
    leftover: Counter,
    layer_tokens: Counter,
    dash_keys: frozenset[str],
    rung_tokens: frozenset[str],
) -> list[str]:
    """B: tokens of *leftover* (reading tokens no exact layer token consumed) whose digits a
    witness holds with other marks. One entry per occurrence, sorted."""
    layer_by_key = _by_key(layer_tokens)
    rung_by_key = _by_key(rung_tokens)
    out: list[str] = []
    for tok, count in sorted(leftover.items()):
        key = _digit_key(tok)
        other_layer = layer_by_key.get(key, set()) - {tok}
        other_rung = rung_by_key.get(key, set()) - {tok}
        if other_layer or other_rung or key in dash_keys:
            out.extend([tok] * count)
    return out


def _decimals(token: str) -> int:
    return len(token.split(".", 1)[1]) if "." in token else 0


def format_outliers(markdown: str) -> list[str]:
    """C: integer cells in a row whose other numeric cells share one decimal count (>= 1).

    Needs no witness. A table's header rows (row 0 and the rows after it with an empty stub
    cell) are not data. A column whose numeric cells are ALL integers is a label, year, line
    number or count column and is excluded; that is derived from the table itself, no
    threshold. A row with a single numeric cell, or whose other cells differ in decimal count,
    flags nothing: there is no format to depart from.
    """
    outliers: list[str] = []
    for block in _evidence_table_blocks(markdown):
        data_rows: list[list[tuple[int, str]]] = []
        in_header = True
        for row_idx, row in enumerate(block.grid):
            if row_idx and in_header and row and row[0].strip():
                in_header = False
            if in_header:
                continue
            cells: list[tuple[int, str]] = []
            for col, raw in enumerate(row):
                cell = raw.strip()
                if not cell:
                    continue
                normalized = _normalize_cell(cell)
                if not is_numeric_token(normalized):
                    continue
                tokens = list(_numeric_multiset_from_tokens([normalized]).elements())
                if len(tokens) == 1:
                    cells.append((col, tokens[0]))
            data_rows.append(cells)
        decimal_cols: set[int] = {col for cells in data_rows for col, tok in cells if "." in tok}
        for cells in data_rows:
            counted = [(col, tok) for col, tok in cells if col in decimal_cols]
            for i, (_col, tok) in enumerate(counted):
                if "." in tok:
                    continue
                others = {_decimals(t) for j, (_c, t) in enumerate(counted) if j != i}
                if len(others) == 1 and next(iter(others)) >= 1:
                    outliers.append(tok)
    return sorted(outliers)


def page_text_numeric(markdown: str) -> Counter:
    """Numeric tokens of the whole page reading (table cells and prose alike)."""
    numeric, _content, _dash = _tokens_and_dash_led(markdown.replace("|", " "))
    return numeric


@dataclass(frozen=True)
class LayerVerdict:
    """The four comparisons for one reading. ``pregate`` names a refused reading, else ``""``."""

    pregate: str = ""
    a_missing: Counter = field(default_factory=Counter)
    b: tuple[str, ...] = ()
    c: tuple[str, ...] = ()
    d: Counter = field(default_factory=Counter)
    crops_found: bool = False

    @property
    def a_passes(self) -> bool:
        return not self.a_missing

    @property
    def disputed(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.b) | set(self.c)))


def judge_against_layer(
    markdown: str,
    tokens: TableTokens,
    witness: TableLayerWitness,
    *,
    tesseract_tokens: frozenset[str] = frozenset(),
    chart_page: bool = False,
) -> LayerVerdict:
    """Run pre-gates and A to D. *markdown* is the reading with HTML tables as markdown."""
    if table_truncated(markdown, None):
        return LayerVerdict(pregate=REASON_TRUNCATED)
    if chart_page:
        return LayerVerdict(pregate=REASON_CHART)
    layer = witness.in_crop if witness.crops_found else witness.whole
    a_missing = _multiset_missing(tokens.numeric, layer)
    b = mark_conflicts(a_missing, layer, witness.dash_keys, tesseract_tokens)
    c = format_outliers(markdown)
    d: Counter = Counter()
    if witness.crops_found:
        d = _multiset_missing(witness.in_crop, page_text_numeric(markdown))
    return LayerVerdict(
        a_missing=a_missing,
        b=tuple(b),
        c=tuple(c),
        d=d,
        crops_found=witness.crops_found,
    )


__all__ = [
    "LayerVerdict",
    "TableLayerWitness",
    "format_outliers",
    "judge_against_layer",
    "mark_conflicts",
    "page_is_chart",
    "page_text_numeric",
    "table_layer_witness",
]
