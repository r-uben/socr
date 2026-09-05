# TICKET-A2 — binder row-label repair (GH-331 / #418 / #146)

Frozen replay on `~/Data/socr/ladder-run2-2026-09-04` after centroid
membership. Numeric-free groups are not folded into other numeric-free
groups: on doc04 p3 the ``1t`` subscript is a different ``(block, line)``
from its parent, its box top sits inside the parent height (also true of
a short overlapping annotation), and the page's shorter-glyph class mixes
``1t`` with an on-line ``∗``, so no page-derived test separates them.

## doc01 p2-t0 (non-target, same shape-1 stub drop)

Recovered printed token: `2 YR, GSS` (matches `labels.json` `printed`).
Native no longer contradicts the model; `fresh#` 1 → 0 because the
contradiction item is gone, not because a disproof fired.

## Targets

doc02 p3-t0, doc02 p4-t0: `fresh#=0`. Native labels match printed
(`NY Treasury inst. forward rate`).

doc04 p3-t0: **does not clear.** Native token remains `1t 1t` (the fold
abstained). The recorded contradiction against the model ROTATED PCs
line is still present.
