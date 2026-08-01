"""Native-layer table row extraction for born-digital pages.

Recovers ``(hierarchy path, ordered values)`` for each row of a table directly
from the PDF's own text layer, with no model involved. Two consumers rely on it:
the GH-96 exactness metric (as ground truth) and the GH-96 escalation canary (as
the oracle a candidate's numbers are checked against).

Lives in ``tables`` rather than ``benchmark`` because it is a native-layer table
parser used by the production pipeline; the benchmark package re-exports it.
"""

from __future__ import annotations

import collections
import re
from collections.abc import Callable
from dataclasses import dataclass

from socr.benchmark.scorer import BenchmarkScorer
from socr.tables.native_verifier import strip_presentation

# A row whose label is only a hierarchy marker ("of which:") introduces children of
# the preceding labelled row; it carries no values of its own.
_MARKER_RE = re.compile(r"^\s*of\s+which\s*:?\s*$", re.IGNORECASE)

_FOOTNOTE_SUFFIX_RE = re.compile(r"(?<=[a-z\)])[0-9](?:[0-9,]{0,5})$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _is_value(cell: str) -> bool:
    """True when *cell* is a table value.

    Emphasis is stripped first (GH-103's lesson, in a second tokenizer):
    ``BenchmarkScorer._is_numeric_cell`` anchors its regex and does not tolerate
    markdown, so a bold cell like ``**43.2**`` reads as non-numeric. Engines emit
    section-total rows bold, so leaving this unstripped silently discards every
    parent row — it under-scored a candidate by 39 points before this was found.
    """
    return bool(cell) and BenchmarkScorer._is_numeric_cell(strip_presentation(cell))


# Footnote markers, in every spelling encountered. Each engine writes them
# differently and the PDF's native layer writes them differently again, so this is
# ONE generic rule rather than an accumulating list of special cases — that list
# reached five entries (bare digit, "$^1$", "^{1}", "<sup>1</sup>", "<sup>1,2</sup>")
# before it became clear the cases were endless.
#
# Every spelling is folded away entirely rather than to a bare digit: the marker is
# a reference to a note, never part of the row's identity.
_FOOTNOTE_MARKER_RE = re.compile(
    r"(?:"
    r"<sup>[\d,\s]{1,8}</sup>"  # HTML: <sup>1</sup>, <sup>1,2</sup>
    r"|\$?\^\{?[\d,\s]{1,8}\}?\$?"  # LaTeX/markdown: $^1$, ^{1}, ^1
    r"|[\u00b9\u00b2\u00b3\u2070-\u209f]+"  # unicode superscripts
    r")\s*$"
)

# A bare trailing digit, anchored to a preceding letter or bracket so that
# "Panel 1" and "Panel 2" stay distinct while "Income tax1" folds.


def normalize_label(label: str) -> str:
    """Fold a row label to a comparison key.

    Emphasis, footnote markers and punctuation are presentation; the words are the
    identity.

    Footnote markers are the hard part, because every producer spells them
    differently: the PDF's native layer emits a bare trailing digit, one engine emits
    LaTeX ("$^1$"), another emits HTML ("<sup>1,2</sup>"). Handling them one at a time
    produced a run of near-misses where two identical rows differed by a single
    character and simply never matched - a perfect page scored 85.1%, and one engine's
    aggregate was understated by 15 points. That mattered beyond reporting: the
    escalation accept rule runs on this metric, so an engine was being penalised for
    its footnote syntax.
    """
    text = label.replace("**", "").replace("*", "").replace("__", "").strip()
    text = text.lower()
    text = _FOOTNOTE_MARKER_RE.sub("", text).strip()
    text = _FOOTNOTE_SUFFIX_RE.sub("", text)
    return _NON_ALNUM_RE.sub("", text)


@dataclass(frozen=True)
class LabeledRow:
    """One data row: its hierarchy path and its ordered values.

    ``path`` is ``(parent, ..., label)``. The last element is the row's own label;
    everything before it is the enclosing hierarchy, used for reporting so a failure
    names the block it occurred in. ``values`` stays compacted (empty cells dropped)
    on both the ground-truth and markdown sides, as it always has - existing readers
    (``escalation_canary``, ``rows_establish_grid``) consume it as plain strings.

    #123 TICKET-B2: two parallel, optional position fields, each empty by default so
    a hand-built ``LabeledRow`` (as most tests below construct) degrades to the
    pre-B2 positional comparison rather than being silently mis-scored:

    - ``lanes`` (ground truth only): the page-level anonymous column-lane index for
      each entry in ``values``, assigned by ``_assign_lanes``. ``-1`` marks a
      lane-ambiguous value - a lone numeric with no supporting peer elsewhere on the
      page - which the scorer must not fold into the alignment.
    - ``columns`` (markdown only): the row's own emitted column position for each
      entry in ``values``, i.e. where the value actually sat in the predicted grid
      before compaction.
    """

    path: tuple[str, ...]
    values: tuple[str, ...]
    lanes: tuple[int, ...] = ()
    columns: tuple[int, ...] = ()

    @property
    def label(self) -> str:
        return self.path[-1] if self.path else ""

    @property
    def key(self) -> str:
        return normalize_label(self.label)

    @property
    def display(self) -> str:
        return " / ".join(self.path)


def native_rows_from_page(page) -> list[LabeledRow]:
    """Ground-truth rows from a born-digital page's native text layer.

    Scoped to the page's located table regions (``tables.locate.locate_tables``) —
    without that scoping the parser happily reads prose sentences that merely
    contain numbers ("rising to £61.7 billion (2.1 per cent of GDP)") as table rows.

    Within a region, words are grouped into their true text lines (fitz block/line
    indices rather than a y tolerance), the label is the leading non-numeric run and
    the values are the numeric tokens in x order. Hierarchy comes from x-indentation
    of the label, which is the only model-free parent signal available: a child row
    is indented relative to its parent.

    Note ``native_verifier._rows_by_y`` is deliberately NOT reused here — it keeps
    numeric tokens only, because it exists for lane counting.

    #123 TICKET-B2: once every region's rows are collected, ``_assign_lanes``
    clusters their value positions into anonymous column lanes **across the whole
    page**, not per region - ``BenchmarkScorer._markdown_table_cells`` flattens every
    pipe table on the page into one grid with no table boundaries, so a per-region
    lane space would not have anything to compare against on the markdown side.
    """
    from socr.tables.locate import locate_tables

    try:
        boxes = locate_tables(page)
    except Exception:  # locate_tables never raises, but stay defensive
        boxes = []
    if not boxes:
        return []

    raw_rows: list[tuple[tuple[str, ...], list[tuple[float, float, str]]]] = []
    for box in boxes:
        raw_rows.extend(_rows_in_region(page, box.bbox))
    return _assign_lanes(raw_rows)


def _assign_lanes(
    raw_rows: list[tuple[tuple[str, ...], list[tuple[float, float, str]]]],
) -> list[LabeledRow]:
    """Cluster every row's value positions into page-wide anonymous column lanes.

    #123 TICKET-B2. Parameter-free by construction:

    1. Cluster the value tokens' **right edges** (``x1``) and, separately, their
       **centres**. Financial columns are right-aligned and left edges drift with
       digit count, so right edges are the natural anchor - but a column of
       centred text disperses less around its centre, so both are tried and the
       partition with lower total within-lane dispersion wins (tie -> right edge,
       the more common case). This is a **per-partition** choice, not per-lane:
       per-lane is chicken-and-egg, since a lane's variance cannot be measured
       before the lane exists.
    2. Within a partition, the cut between two lanes is the largest *ratio jump*
       between consecutive gaps in the sorted anchor list - the point where
       "spread inside one column" ends and "space between columns" begins,
       derived from the page's own geometry rather than a point constant.
    3. A cluster only becomes a lane when it has support from **at least two
       distinct rows** - the same justification as the GH-113 grid rule: a lone
       numeral cannot demonstrate a column by itself. A value that snaps to no
       surviving lane is **lane-ambiguous**, marked ``-1`` and excluded from the
       scorer's alignment rather than guessed.
    """
    if not raw_rows:
        return []

    # token = (token_idx, row_idx, x0, x1, text), flattened in row/x order.
    flat: list[tuple[int, int, float, float, str]] = []
    token_idx = 0
    for row_idx, (_path, values) in enumerate(raw_rows):
        for x0, x1, text in values:
            flat.append((token_idx, row_idx, x0, x1, text))
            token_idx += 1

    if flat:
        right_clusters = _cluster_by_anchor(flat, _anchor_right)
        centre_clusters = _cluster_by_anchor(flat, _anchor_centre)
        right_dispersion = _total_dispersion(right_clusters, _anchor_right)
        centre_dispersion = _total_dispersion(centre_clusters, _anchor_centre)
        clusters = centre_clusters if centre_dispersion < right_dispersion else right_clusters

        lane_clusters = [c for c in clusters if len({tok[1] for tok in c}) >= 2]
        lane_of_token: dict[int, int] = {
            tok[0]: lane_idx for lane_idx, cluster in enumerate(lane_clusters) for tok in cluster
        }
    else:
        lane_of_token = {}

    rows: list[LabeledRow] = []
    idx = 0
    for path, values in raw_rows:
        lanes = tuple(lane_of_token.get(idx + i, -1) for i in range(len(values)))
        idx += len(values)
        rows.append(LabeledRow(path=path, values=tuple(v[2] for v in values), lanes=lanes))
    return rows


def _anchor_right(token: tuple[int, int, float, float, str]) -> float:
    return token[3]


def _anchor_centre(token: tuple[int, int, float, float, str]) -> float:
    return (token[2] + token[3]) / 2


def _cluster_by_anchor(
    tokens: list[tuple[int, int, float, float, str]],
    anchor: Callable[[tuple[int, int, float, float, str]], float],
) -> list[list[tuple[int, int, float, float, str]]]:
    """Partition *tokens* left to right, cutting at the Otsu threshold over gaps."""
    ordered = sorted(tokens, key=anchor)
    if len(ordered) <= 1:
        return [ordered] if ordered else []

    anchors = [anchor(t) for t in ordered]
    gaps = [anchors[i + 1] - anchors[i] for i in range(len(anchors) - 1)]
    cut = _gap_cut_threshold(gaps)

    clusters: list[list[tuple[int, int, float, float, str]]] = [[ordered[0]]]
    for i in range(1, len(ordered)):
        if gaps[i - 1] > cut:
            clusters.append([])
        clusters[-1].append(ordered[i])
    return clusters


def _gap_cut_threshold(gaps: list[float]) -> float:
    """The gap size above which two anchors belong to different lanes.

    #123 TICKET-B2 reopen (second fix). The previous rule picked "the largest
    ratio jump between consecutive **distinct** positive gap magnitudes" - a
    single distinguished gap, chosen without regard to how many tokens actually
    produced each magnitude. That formulation cannot work: there are two
    distinct noise floors in real geometry (the exact-zero gap between tokens
    sharing an anchor, and small non-zero jitter from digit widths or float
    rounding), and a single ratio-jump search cannot separate either from real
    column spacing. Folding zero into the search made *every* transition away
    from it read as an infinite ratio, so it always won regardless of whether
    the next value up was real spacing or itself still noise - it once picked a
    5pt digit-jitter gap as the elbow over a real 55pt spacing. Excluding zero
    lost the tokens-that-share-a-lane signal entirely, so on a regular grid
    where the only other magnitude present was itself corrupted by sub-pt
    float noise (``50.0`` vs ``50.000015`` pt, from glyph rendering at a
    different page offset), the "largest ratio jump" was that meaningless
    ``1.0000003`` between two magnitudes that are, in truth, the same lane
    boundary measured twice.

    The fix: stop choosing between a single pair of magnitudes and instead
    partition the **whole weighted gap list** - every gap, in the multiplicity
    it actually occurs, zero included - the way Otsu's method thresholds a
    bimodal histogram. For every threshold candidate (a value between two
    consecutive distinct magnitudes), split all gaps into "at or below" and
    "above" and score the split by its between-group variance
    (``_between_group_variance``); take the candidate that maximises it. This
    is still parameter-free - the cut is selected by an objective computed
    from the data, not a constant - but unlike a ratio jump it is a function
    of *how much evidence* separates two candidate lane-boundary magnitudes,
    not just their relative size. Two magnitudes that differ by a
    rendering-noise fraction of a point (``50.0`` vs ``50.000015``) carry
    almost no between-group variance if grouped apart, because their means are
    nearly identical either way; splitting the exact-zero floor from either of
    them carries enormous between-group variance, because the group means are
    ~50pt apart. Otsu's search finds that gap regardless of how many other
    magnitudes happen to be sitting near it, which a single max-ratio search
    cannot.

    With at most one distinct gap magnitude present there is nothing to
    partition; the cut sits at that magnitude's own midpoint against the
    implied zero floor (mirrors the previous rule's ``len(positive) == 1``
    case, generalised to also cover an all-zero gap list, where the returned
    cut of ``0.0`` correctly draws no boundary at all).
    """
    if not gaps:
        return 0.0
    candidates = sorted(set(gaps))
    if len(candidates) == 1:
        return candidates[0] / 2

    best_cut = candidates[0] / 2
    best_score = -1.0
    for i in range(len(candidates) - 1):
        threshold = candidates[i]
        low = [g for g in gaps if g <= threshold]
        high = [g for g in gaps if g > threshold]
        score = _between_group_variance(low, high)
        if score > best_score:
            best_score = score
            best_cut = (candidates[i] + candidates[i + 1]) / 2
    return best_cut


def _between_group_variance(low: list[float], high: list[float]) -> float:
    """Otsu's between-class variance for a two-way split of gap magnitudes.

    Weighted by how many gaps actually fall on each side, not merely by which
    distinct magnitudes are present - the property the previous ratio-jump
    rule lacked. Maximising this over all candidate splits is equivalent to
    minimising the summed within-group variance, and picks out the split
    where the two groups are both internally tight *and* far apart, rather
    than the split with the largest ratio between two arbitrary neighbours.
    """
    total = len(low) + len(high)
    weight_low = len(low) / total
    weight_high = len(high) / total
    mean_low = sum(low) / len(low)
    mean_high = sum(high) / len(high)
    return weight_low * weight_high * (mean_high - mean_low) ** 2


def _total_dispersion(
    clusters: list[list[tuple[int, int, float, float, str]]],
    anchor: Callable[[tuple[int, int, float, float, str]], float],
) -> float:
    """Sum of within-cluster variance across *clusters*, under *anchor*."""
    total = 0.0
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        values = [anchor(t) for t in cluster]
        mean = sum(values) / len(values)
        total += sum((v - mean) ** 2 for v in values) / len(values)
    return total


def rows_establish_grid(rows: list[LabeledRow]) -> bool:
    """GH-113: True when *rows* look like a table grid, not prose or chart labels.

    The row parser above will happily read two numeric lines of prose, or a
    chart's axis labels and legend, as if they were table rows — on the OBR
    reference document that fabricated rows from a cover date
    ("November=['2022']"), a fragment of "4 per cent" ("cent=['4']"), and a fan
    chart's axes and legend (page 54, scored 0.0% against ground truth that was
    never a table). ``rows`` alone cannot discriminate; the shape can: a grid
    needs at least two value columns, and at least two rows sharing that width.
    Both are minimums that follow from what a table is, not tuned cutoffs.

    Deliberately not "most rows share the modal width": a real +89-point
    recovery on the reference document has only 7 of 17 rows at its modal
    width, and a majority rule would have refused it.

    Shared by the GH-96 exactness metric (ground truth must be scorable) and the
    GH-113 escalation trigger (don't pay for a cloud call to compare an empty
    result against an empty result) — one predicate, not two copies drifting
    apart.
    """
    widths = collections.Counter(len(r.values) for r in rows)
    if not widths:
        return False
    modal_width, rows_at_modal = widths.most_common(1)[0]
    return modal_width >= 2 and rows_at_modal >= 2


def _rows_in_region(page, bbox) -> list[tuple[tuple[str, ...], list[tuple[float, float, str]]]]:
    """Labelled rows from the words falling inside one table bbox.

    Returns ``(path, values)`` pairs rather than finished ``LabeledRow`` objects —
    ``values`` carries each value token's ``(x0, x1, text)`` so the caller
    (``native_rows_from_page``) can cluster lanes across every region on the page
    before ``LabeledRow.lanes`` is assigned. #123 TICKET-B2.

    Two things this must get right, both learned from the OBR reference page:

    1. **Row grouping is by vertical overlap, not by fitz block/line.** A PDF table
       routinely emits the row label and its numbers as separate text objects, so
       block/line indices split one visual row in two. Words sharing a visual row
       have overlapping ``[y0, y1]`` intervals; words on different rows do not. That
       is parameter-free — no y tolerance to tune.

    2. **The label/value boundary is a column position, not "the first number".**
       "Growth Plan after 17 October reversals" contains a numeral inside the label.
       The boundary is the row's own **last non-numeric word** (below), never by
       clustering numeric x-positions to find it: an earlier attempt did that, and
       a numeral inside a label formed its own x-cluster and dragged the boundary
       left. Lane clustering (page-wide, over tokens already known to be values) is
       decoupled into ``_assign_lanes`` and runs strictly after this boundary is
       fixed.
    """
    x0_lim, y0_lim, x1_lim, y1_lim = bbox
    words = [
        (x0, y0, x1, y1, word)
        for x0, y0, x1, y1, word, *_ in page.get_text("words")
        if x0 >= x0_lim and x1 <= x1_lim and y0 >= y0_lim and y1 <= y1_lim
    ]
    if not words:
        return []

    # (1) cluster words into visual rows by y-interval overlap
    words.sort(key=lambda w: (w[1], w[0]))
    bands: list[list[tuple[float, float, float, float, str]]] = []
    band_y0 = band_y1 = None
    for word in words:
        wy0, wy1 = word[1], word[3]
        if band_y1 is None:
            overlaps = False
        else:
            # Any overlap at all is too weak a test. A wrapped label at tight
            # leading overlaps the line above by a descender, and merging the two
            # then interleaves their words by x-sort: "Central government net /
            # debt" became "Central net government debt". Require the shared span
            # to exceed half the shorter line's height - the definitional midpoint
            # for "same visual row", not a tuned value.
            shared = min(band_y1, wy1) - max(band_y0, wy0)
            shorter = min(band_y1 - band_y0, wy1 - wy0)
            overlaps = shorter > 0 and shared > shorter / 2
        if overlaps:
            bands[-1].append(word)
            band_y0 = min(band_y0, wy0)
            band_y1 = max(band_y1, wy1)
        else:
            bands.append([word])
            band_y0, band_y1 = wy0, wy1

    # (2) split each row at its last non-numeric word.
    #
    # Neither "everything before the first number" nor an x-clustered column
    # boundary survives real data: the first rule truncates
    # "Growth Plan after 17 October reversals" at the 17, and the second is
    # chicken-and-egg (that stray 17 forms its own x-cluster and drags the boundary
    # left). A table row is a label followed by values, so the last non-numeric word
    # ends the label. Parameter-free, and correct for both cases above.
    raw: list[tuple[float, str, list[tuple[float, float, str]]]] = []
    for band in bands:
        ordered = sorted(band, key=lambda w: w[0])
        last_text = -1
        for idx, word in enumerate(ordered):
            if not _is_value(word[4]):
                last_text = idx
        if last_text < 0:
            continue  # numbers only: an orphan row, no label to key on
        label = " ".join(w[4].strip() for w in ordered[: last_text + 1]).strip()
        values = [(w[0], w[2], w[4].strip()) for w in ordered[last_text + 1 :] if _is_value(w[4])]
        indent = ordered[0][0]
        if label and values and not _MARKER_RE.match(label):
            raw.append((indent, label, values))

    if not raw:
        return []

    raw = _drop_footnote_markers(raw, _superscript_tokens(page, bbox))

    # Parent = the nearest preceding row at a strictly smaller indent.
    rows: list[tuple[tuple[str, ...], list[tuple[float, float, str]]]] = []
    stack: list[tuple[float, str]] = []
    for indent, label, values in raw:
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = tuple(lbl for _i, lbl in stack) + (label,)
        rows.append((path, values))
        stack.append((indent, label))
    return rows


def _superscript_tokens(page, bbox) -> set[tuple[int, str]]:
    """Locate footnote superscripts inside *bbox*, keyed by (rounded x, text).

    A footnote marker is a separate word, so "Memo: UK oil and gas revenues5"
    leaves a bare ``5`` between the label and the first real value. The
    label/value split cannot see it — the token is numeric and follows the last
    non-numeric word — so it inflates the row's width and injects a spurious
    value into the ground truth.

    Font size separates them, but only *within a row*. Comparing against the whole
    region's modal size misfires badly: an entire "Memo:" row is often set smaller
    than the table body, and region-wide sizing then flags all of its real values
    as markers. A superscript is small relative to the text it is attached to, so
    the comparison is per-row.

    Deliberately not a width heuristic — trimming rows merely wider than the modal
    row silently deletes real leading values from legitimately wide rows.
    """
    x0_lim, y0_lim, x1_lim, y1_lim = bbox
    by_line: dict[tuple[int, int], list[tuple[float, float, str]]] = {}
    for bi, block in enumerate(page.get_text("dict").get("blocks", [])):
        for li, line in enumerate(block.get("lines", [])):
            for span in line.get("spans", []):
                sx0, sy0, sx1, sy1 = span.get("bbox", (0, 0, 0, 0))
                if sx0 < x0_lim or sx1 > x1_lim or sy0 < y0_lim or sy1 > y1_lim:
                    continue
                text = (span.get("text") or "").strip()
                if text:
                    by_line.setdefault((bi, li), []).append(
                        (round(span.get("size", 0.0), 1), sx0, text)
                    )

    markers: set[tuple[int, str]] = set()
    for spans in by_line.values():
        sizes = [s for s, _x, _t in spans]
        row_size = max(set(sizes), key=sizes.count)
        for size, x, text in spans:
            if size < row_size and _is_value(text):
                markers.add((round(x), text))
    return markers


def _drop_footnote_markers(
    raw: list[tuple[float, str, list[tuple[float, float, str]]]],
    superscripts: set[tuple[int, str]],
) -> list[tuple[float, str, list[tuple[float, float, str]]]]:
    """Remove superscript markers that the label/value split counted as values."""
    return [
        (
            indent,
            label,
            [(x0, x1, v) for x0, x1, v in values if (round(x0), v) not in superscripts],
        )
        for indent, label, values in raw
    ]
