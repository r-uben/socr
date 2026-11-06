"""Proximity clustering of axis-aligned boxes on a grid (leaf module, no socr imports).

``cluster_box_indices(boxes, gap)`` partitions box indices into the connected
components of the relation "h_gap <= gap and v_gap <= gap" (closed ``<=``), the
exact test the old pairwise loops used. Groups are ordered by their smallest
member index and members are ascending, so callers reproduce the old output
byte for byte.

Algorithm: a uniform grid with cell side ``cell = gap / 2``.

* Same cell implies within gap. Two boxes that both meet one cell of side
  ``cell`` overlap that cell on each axis, so their separation on each axis is
  at most ``cell`` plus the rounding of ``floor(x / cell)``, which is
  ``cell / 2 + 2 * eps`` (see the magnitude guard) and less than ``gap``. They
  are unioned without a check.
* Reach is 3 cells. Two boxes within ``gap`` of each other have a separation
  of at most ``gap = 2`` cells per axis, which spans 2 cell indices, plus one
  cell of margin because ``floor(x / cell)`` is computed in floating point: a
  coordinate just below a cell edge can floor one cell lower than the exact
  quotient (x1 = 14.999999999999998, x0 = x1 + 30 = 45.0 with cell 15 are cells
  0 and 3). Only cells within +-3 are candidates; those pairs are checked
  exactly, skipped when already in one component, and stopped at the first hit
  per cell pair.
* Large boxes. A box meeting more cells than there are boxes (``n``) is not
  registered in the grid. It is checked exactly against every other box
  instead, which costs ``n`` and is cheaper than registering more than ``n``
  cells.
* Work budget. Failed cross-cell checks can repeat for the same box pair in
  many neighbouring cell pairs, and registering boxes can cost up to ``n`` cells
  each, so the grid has no O(n^2) bound on its own. Registration and exact
  checks are charged to ONE shared budget, the pairwise pair count
  ``B = n * (n - 1) // 2``. Before allocating anything, the grid sums the
  memberships it would register (the cell-count sum over non-large boxes, O(n),
  same int spans); if that exceeds ``B`` it returns None without registering. The
  exact checks may then use only ``B - memberships``; when they would exceed it,
  the grid result is abandoned. A grid attempt therefore never stores more
  memberships, and never does more registration-plus-check work, than one
  pairwise pass (``B``), and on abandonment the exact pairwise loop runs from
  scratch: registration plus exact checks total at most 2 pairwise passes
  (union-find and dictionary bookkeeping are not counted in that bound).
* Fallback to the exact pairwise loop, with no grid, when the grid's premises
  fail: ``gap`` or any coordinate non-finite; ``cell`` not a positive finite
  float (``gap <= 0``, or ``gap / 2`` underflowing to 0); any inverted box
  (``x1 < x0`` or ``y1 < y0``), which would vanish from the grid ranges; or any
  ``|coordinate| > cell * 2**50``. Magnitude guard: with ``|x| <= cell * 2**50``
  the quotient ``x / cell`` is at most 2**50, where (exactly at 2**50) the float
  spacing is 2**-2, so its rounding error is at most 2**-3 cell units (<= cell/4
  spacing). Two coordinates in one computed cell then differ by less than
  1 + 2 * 2**-3 cells = 1.25 cells, below ``gap = 2`` cells, so the unchecked
  same-cell union stays within gap.
  Beyond that magnitude the spacing exceeds a cell and distinct coordinates can
  share a cell. Cell spans are computed as Python ints, never ``len(range())``.

#1269: 52,325-drawing Tealbook pages, >60 s -> ~0.1 s.
"""

from __future__ import annotations

import math

Box = tuple[float, float, float, float]

# Precision bits kept below the cell size: |coordinate| <= cell * 2**_GRID_MAGNITUDE_BITS.
# Derived in the module docstring (float spacing at that magnitude is cell / 4 at most;
# same-cell spread < 1.25 cells < 2 = gap / cell).
_GRID_MAGNITUDE_BITS = 50


def _within_gap(a: Box, b: Box, gap: float) -> bool:
    """The exact proximity test (the single place a pair is compared)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    h_gap = max(0.0, bx0 - ax1) if ax1 < bx0 else max(0.0, ax0 - bx1)
    v_gap = max(0.0, by0 - ay1) if ay1 < by0 else max(0.0, ay0 - by1)
    return h_gap <= gap and v_gap <= gap


class _Components:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        parent = self.parent
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _pairwise(boxes: list[Box], gap: float) -> _Components:
    comps = _Components(len(boxes))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _within_gap(boxes[i], boxes[j], gap):
                comps.union(i, j)
    return comps


def _pairwise_budget(n: int) -> int:
    """The pairwise pass's pair count, ``n * (n - 1) // 2``: the grid's total work budget."""
    return n * (n - 1) // 2


def _grid_usable(boxes: list[Box], gap: float) -> bool:
    if not math.isfinite(gap) or gap <= 0:
        return False
    cell = gap / 2.0
    if not (cell > 0 and math.isfinite(cell)):
        return False
    limit = cell * 2.0**_GRID_MAGNITUDE_BITS
    if not math.isfinite(limit):
        return False
    for x0, y0, x1, y1 in boxes:
        if x1 < x0 or y1 < y0:
            return False
        for v in (x0, y0, x1, y1):
            if not math.isfinite(v) or abs(v) > limit:
                return False
    return True


def _grid(boxes: list[Box], gap: float) -> _Components | None:
    """Grid clustering, or None when registration plus exact checks would exceed the
    pairwise count."""
    n = len(boxes)
    cell = gap / 2.0
    reach = 3
    spans = []
    large: list[int] = []
    memberships = 0
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        cx0, cx1 = math.floor(x0 / cell), math.floor(x1 / cell)
        cy0, cy1 = math.floor(y0 / cell), math.floor(y1 / cell)
        count = (cx1 - cx0 + 1) * (cy1 - cy0 + 1)
        if count > n:
            large.append(i)
        else:
            spans.append((i, cx0, cx1, cy0, cy1))
            memberships += count
    # One shared budget: registration (memberships) and exact checks together.
    budget = _pairwise_budget(n)
    if memberships > budget:
        return None
    budget -= memberships
    checks = 0
    comps = _Components(n)
    cells: dict[tuple[int, int], list[int]] = {}
    for i, cx0, cx1, cy0, cy1 in spans:
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                cells.setdefault((cx, cy), []).append(i)
    for members in cells.values():
        first = members[0]
        for j in members[1:]:
            comps.union(first, j)
    for (cx, cy), a_members in cells.items():
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                if (dx, dy) <= (0, 0):
                    continue
                b_members = cells.get((cx + dx, cy + dy))
                if b_members is None or comps.find(a_members[0]) == comps.find(b_members[0]):
                    continue
                hit = False
                for a in a_members:
                    for b in b_members:
                        checks += 1
                        if checks > budget:
                            return None
                        if _within_gap(boxes[a], boxes[b], gap):
                            comps.union(a, b)
                            hit = True
                            break
                    if hit:
                        break
    for i in large:
        for j in range(n):
            if j != i and comps.find(i) != comps.find(j):
                checks += 1
                if checks > budget:
                    return None
                if _within_gap(boxes[i], boxes[j], gap):
                    comps.union(i, j)
    return comps


def cluster_box_indices(boxes: list[Box], gap: float) -> list[list[int]]:
    """Connected components of "within *gap*" over *boxes*; see module docstring."""
    n = len(boxes)
    if n == 0:
        return []
    comps = _grid(boxes, gap) if _grid_usable(boxes, gap) else None
    if comps is None:
        comps = _pairwise(boxes, gap)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(comps.find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])
