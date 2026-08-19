"""GH-212: header attribution proved from the drawn rule, not token content.

TR-3 (``native_verifier``) proves the page's numbers and the emitted table's
numbers are the same bag of values, and is blind by construction to whether each
number is still bound to the right column -- a destroyed header changes zero
numerals. On the ``EXACT_PASS`` path that blindness is the whole gate, so a table
with a wrecked header band ships at confidence 1.0 and its coefficients lose
their column.

Four previous predicates were reverted for the opposite failure: they recovered
header *role* from token CONTENT (or a numeral-count proxy for it) and so
returned HARD on byte-perfect correct tables carrying significance-star and
``n.a.`` rows -- see the parking docstring in ``structure_check``. A false HARD
rejects correct output; it is strictly worse than a miss.

This module does not classify rows. It cuts the page at the drawn rule
immediately above the numeric anchor row and owes only what lies above that cut.
Star and ``n.a.`` rows sit BELOW the midrule on a booktabs table, so the proof
never inspects their tokens and cannot be confused by them. That is geometry,
not vocabulary.

Every step abstains rather than guessing, and abstain is UNVERIFIABLE, never a
pass or a fail. Measured across 6,350 table-bearing pages of the reference
library: the hairline rule this needs is present essentially always (44,655 of
~56,600 rules are 0.0pt), a table page loses its topmost rule to
``_RULE_FLATNESS_PT`` on 1.6% of pages, and 11.1% carry a ``(1)``...``(2)``
column-number run -- which is why ``_fold`` exists.

Pure: no I/O, no model calls, no mutation of its inputs. The caller supplies the
rule list so this module never touches a ``fitz`` page.
"""

from __future__ import annotations

import logging
import re

from socr.tables.header_attribution import HeaderVerdict
from socr.tables.header_repair import (
    _MIN_DATA_NUMERIC_CELLS,
    _all_rows_by_y,
    _local_table_ys,
    _median_row_gap,
    _row_numeric_multiset,
)
from socr.tables.native_verifier import (
    _numeric_multiset_from_tokens,
    is_numeric_token,
    strip_presentation,
)
from socr.tables.reconstruct import _LANE_SNAP_MULT, _LANE_X_TOL_PT

logger = logging.getLogger(__name__)

#: Bracket and per-cent punctuation is presentation ON THE HEADER PATH ONLY.
#: ``strip_presentation`` deliberately keeps parentheses because they denote a
#: negative value where numbers are compared; a column header's ``(1)`` carries
#: no such meaning, and a unit that moved to the caption (``Share %`` -> ``Share``)
#: is standard booktabs practice. Folding these is punctuation, not a vocabulary:
#: no year lists, no ``n.a.`` lists, nothing content-specific.
_HEADER_PUNCT = str.maketrans("", "", "()[]")
_TRAILING_UNIT_RE = re.compile(r"[%‰]+$")


def _fold(tok: str) -> str:
    """Normalise one token for header membership. Never used on the numeric path."""
    folded = strip_presentation(tok).translate(_HEADER_PUNCT)
    return _TRAILING_UNIT_RE.sub("", folded).casefold()


def _anchor_candidates(rows_by_y: dict[int, list], grid: list[list[str]]) -> list[float]:
    """Native y-rows that UNIQUELY match an emitted row's numeric multiset.

    ``header_repair._best_anchor_y`` returns the FIRST match in ascending y and
    has no uniqueness test. That is unsafe here: two panels on one page routinely
    share a rounded standard-error row, and anchoring onto the wrong copy puts
    the cut on the wrong panel's midrule and HARDs a byte-perfect table. A row
    matching more than one native y is therefore dropped, never guessed at.

    Several are returned rather than one because an emitted row is not
    necessarily a DATA row: a second header tier of column indices (``(1) (2)
    (3)``) matches its own native row perfectly well. Such an anchor has only
    one rule above it and so yields no header band; the caller walks on to the
    next candidate rather than abstaining on the whole table.
    """
    out: list[float] = []
    for row in grid[1:]:
        out_ms = _numeric_multiset_from_tokens(row)
        if len(out_ms) < _MIN_DATA_NUMERIC_CELLS:
            continue
        matches = [y for y in sorted(rows_by_y) if _row_numeric_multiset(rows_by_y[y]) == out_ms]
        if len(matches) == 1 and float(matches[0]) not in out:
            out.append(float(matches[0]))
    return out


def _lane_centers(rows_by_y: dict[int, list], data_ys: list[int]) -> list[float]:
    """Cluster the x-positions of native numeric tokens into column lanes.

    Presentation-aware via ``is_numeric_token``, unlike the raw anchored pattern
    ``header_repair._derive_lane_centers`` uses: otherwise a leading-decimal
    column (``.034``) forms no lane, the owed set silently shrinks and a real
    header loss goes unseen. That is the GH-206 blind spot.
    """
    xs: list[float] = []
    for y in data_ys:
        row_words = rows_by_y.get(y, [])
        if len(_row_numeric_multiset(row_words)) < _MIN_DATA_NUMERIC_CELLS:
            continue
        xs += [w[0] for w in row_words if is_numeric_token(w[4])]
    if not xs:
        return []

    lanes: list[list[float]] = []
    for x in sorted(set(xs)):
        if lanes and x - lanes[-1][-1] <= _LANE_X_TOL_PT:
            lanes[-1].append(x)
        else:
            lanes.append([x])
    return [sum(g) / len(g) for g in lanes]


def _header_band(
    rules: list[tuple[float, float, float]],
    rows_by_y: dict[int, list],
    anchor_y: float,
    local_ys: list[int],
) -> tuple[float, float] | None:
    """The ``(top, cut)`` y-range that holds the header, or ``None`` to abstain.

    ``cut`` is the rule nearest above the anchor; ``top`` is the rule above that
    -- the toprule. There is deliberately NO fallback to the top of the table
    neighbourhood: ``_local_table_ys`` admits every y-group overlapping the
    anchor's x-extent +/-20pt, so on a full-width table that fallback would owe
    the caption, the running head and any prose numeral within snapping distance
    of a lane, none of which appear in the emitted header. Abstaining is the safe
    direction; over-owing rejects correct tables.
    """
    anchor_words = rows_by_y.get(round(anchor_y), [])
    if not anchor_words:
        return None
    x0 = min(w[0] for w in anchor_words)
    x1 = max(w[2] for w in anchor_words)

    # A toprule sits just ABOVE the table's topmost text row, so bounding rules
    # by ``min(local_ys)`` would discard the very rule that caps the header.
    # Extend the window upward by one local row gap -- derived from this page's
    # own row spacing, so it scales with the type size instead of assuming one.
    hi = max(local_ys)
    lo = min(local_ys) - _median_row_gap(sorted(local_ys))

    spanning = [r for r in rules if r[1] <= x0 and r[2] >= x1 and lo <= r[0] <= hi]
    above_anchor = [r[0] for r in spanning if r[0] < anchor_y]
    if not above_anchor:
        logger.debug("header_cut: no spanning rule above anchor y=%.1f", anchor_y)
        return None

    cut = max(above_anchor)
    higher = [y for y in above_anchor if y < cut]
    if not higher:
        logger.debug("header_cut: no rule above the cut at y=%.1f; abstaining", cut)
        return None
    return max(higher), cut


def _owed_tokens(
    rows_by_y: dict[int, list],
    lanes: list[float],
    top: float,
    cut: float,
) -> set[str]:
    """Folded tokens between the two rules that snap to a data lane.

    A word left of every lane is the stub label, not column metadata, and is not
    owed; a lane carrying no native word above the cut owes nothing at all.
    """
    margin = _LANE_X_TOL_PT * _LANE_SNAP_MULT
    owed: set[str] = set()
    for y, row_words in rows_by_y.items():
        if not top < y < cut:
            continue
        for w in row_words:
            if any(abs(centre - w[0]) <= margin for centre in lanes):
                folded = _fold(w[4])
                if folded:
                    owed.add(folded)
    return owed


def _above_cut_multisets(rows_by_y: dict[int, list], top: float, cut: float) -> list:
    """Numeric multisets of the native rows inside the header band."""
    return [
        _row_numeric_multiset(row_words)
        for y, row_words in rows_by_y.items()
        if top < y < cut and _row_numeric_multiset(row_words)
    ]


def _emitted_header_tokens(grid: list[list[str]], header_multisets: list) -> set[str]:
    """The emitted header's tokens: ``grid[0]``, plus one further tier if present.

    Markdown carries exactly one header row, but a native two-tier band
    (``Model A | Model B`` over ``(1) | (2)``) is faithfully emitted with its
    second tier in ``grid[1]``. Ignoring that would make the verdict depend on
    the model's flattening style rather than on content.

    A second tier is identified geometrically, not by vocabulary: ``grid[1]``
    counts as header when it carries too few numerals to be a data row, OR when
    the native row bearing that same numeric multiset lies inside the header
    band. Column-index rows (``(1) (2) (3)``) are numerically indistinguishable
    from data by count alone, so the band membership is what separates them.

    Bounded to ONE extra row, and only when ``grid[0]`` is itself non-blank.
    A blank ``grid[0]`` is the destroyed-header case: its tokens landing in body
    rows is precisely the defect (they bind as data, and the columns stay
    unnamed), so those rows must not be absorbed.
    """
    rows = [grid[0]]
    if any(cell.strip() for cell in grid[0]) and len(grid) > 1:
        nxt = grid[1]
        ms = _numeric_multiset_from_tokens(nxt)
        if len(ms) < _MIN_DATA_NUMERIC_CELLS or any(ms == m for m in header_multisets):
            rows.append(nxt)
    return {_fold(t) for row in rows for cell in row for t in cell.split() if t.strip()}


def header_cut_verdict(
    grid: list[list[str]],
    words: list | None,
    rules: list[tuple[float, float, float]] | None,
) -> HeaderVerdict:
    """HARD when native header content above the midrule is absent from the emitted header.

    ``grid`` is one emitted, separator-free table grid; ``words`` is
    ``page.get_text("words")`` for the source page; ``rules`` is
    ``locate._horizontal_rules(page)``, precomputed by the caller so this stays
    pure. Returns only HARD, OK or UNVERIFIABLE -- SOFT remains the advisory
    signal of ``header_attribution`` and is not produced here.
    """
    if not grid or not grid[0] or not words or not rules:
        return HeaderVerdict.UNVERIFIABLE

    rows_by_y = _all_rows_by_y(words)
    if not rows_by_y:
        return HeaderVerdict.UNVERIFIABLE

    resolved = None
    for anchor_y in _anchor_candidates(rows_by_y, grid):
        anchor_int = round(anchor_y)
        local_ys = _local_table_ys(rows_by_y, anchor_int)
        lanes = _lane_centers(rows_by_y, [y for y in local_ys if y >= anchor_int])
        if len(lanes) < 2:
            continue
        band = _header_band(rules, rows_by_y, anchor_y, local_ys)
        if band is not None:
            resolved = (lanes, band)
            break
    if resolved is None:
        return HeaderVerdict.UNVERIFIABLE
    lanes, band = resolved

    owed = _owed_tokens(rows_by_y, lanes, band[0], band[1])
    if not owed:
        return HeaderVerdict.OK

    emitted = _emitted_header_tokens(grid, _above_cut_multisets(rows_by_y, band[0], band[1]))
    return HeaderVerdict.OK if owed <= emitted else HeaderVerdict.HARD
