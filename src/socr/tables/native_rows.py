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

    1. Cluster the value tokens' **right edges** (``x1``), their **centres**,
       and their **left edges** (``x0``). Financial columns are usually
       right-aligned and left edges drift with digit count, so right edges
       are the natural anchor - but a column of centred text disperses less
       around its centre, and a column of left-aligned text (labels, or
       fixed-width values) has an exactly constant left edge regardless of
       digit count, disperses not at all around it. All three are tried and
       the partition with lower total within-lane dispersion wins (tie ->
       right, then centre, then left, in that order - the more common cases
       first). This is a **per-partition** choice, not per-lane: per-lane is
       chicken-and-egg, since a lane's variance cannot be measured before the
       lane exists.
    2. Within a partition, lanes come from ``_cluster_by_anchor`` - a
       best-first, row-count-bounded split (see that function's docstring
       for how), derived from the page's own geometry rather than a point
       constant.
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
        left_clusters = _cluster_by_anchor(flat, _anchor_left)
        right_dispersion = _total_dispersion(right_clusters, _anchor_right)
        centre_dispersion = _total_dispersion(centre_clusters, _anchor_centre)
        left_dispersion = _total_dispersion(left_clusters, _anchor_left)
        # Tie-break on cluster count before anchor order: two partitions that
        # explain the data equally well (same total dispersion, most often
        # both exactly 0.0) are not equally good - the one with fewer
        # clusters is the more parsimonious explanation, and the one with
        # more clusters is by construction *never* a better fit, only ever a
        # finer one (splitting a cluster can't raise its own dispersion, so
        # over-splitting is free to "tie" without evidence). Anchor order
        # (right, then centre, then left) only breaks remaining ties.
        best = min(
            (right_dispersion, len(right_clusters), 0, right_clusters),
            (centre_dispersion, len(centre_clusters), 1, centre_clusters),
            (left_dispersion, len(left_clusters), 2, left_clusters),
        )
        clusters = best[3]

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


def _anchor_left(token: tuple[int, int, float, float, str]) -> float:
    return token[2]


def _cluster_by_anchor(
    tokens: list[tuple[int, int, float, float, str]],
    anchor: Callable[[tuple[int, int, float, float, str]], float],
) -> list[list[tuple[int, int, float, float, str]]]:
    """Partition *tokens* into lanes by best-first, row-count-bounded bisection.

    #123 TICKET-B2 reopen (real-page over-split), second attempt. Every earlier
    version of this function placed *one* cut value and treated it as globally
    decisive - most recently, "if an exact-zero gap is present, every positive
    magnitude above it is a lane boundary" (see git history for the superseded
    ``_gap_cut_threshold``). That branch was correct only on the fixtures it
    was built against, which all had a clean zero floor with no other noise on
    the winning anchor. Measured on the real reference document, 16 of 18
    scorable pages violate that precondition: right-aligned numbers of
    different digit counts (``"24.8"`` vs ``"177.0"``) render with sub-point
    kerning drift, so the *same* logical column produces both an exact-zero
    gap (two tokens that happen to land bit-identically) **and** a spread of
    distinct positive magnitudes under 1.2pt that are pure rendering noise -
    while the real column-to-column spacing is 20-70pt.

    A first attempt at this reopen replaced the single cut with recursive,
    row-support-validated bisection: split at the highest-``_between_group_
    variance`` boundary, accept it only if both sides keep >=2 distinct rows
    (GH-113's "a lone value cannot demonstrate a column" rule), recurse depth-
    first into whichever side validated. That measurably made the real corpus
    *worse* (page 13: 24 -> 56 lanes), because >=2-row support is too weak a
    bar once a table has 20+ rows: almost any split leaves >=2 rows on each
    side by pure chance (e.g. a 1-1.2pt kerning-scale sub-grid offset between
    a table's body and a footnote/memo block), so depth-first recursion never
    runs out of "validated" splits to make and fragments a single real column
    into many spurious ones.

    The fix keeps row-support validation (it correctly rejects a lone-value
    split) but adds the one bound implied by the data itself: **a page cannot
    have more real lanes than its widest row has values** - the same
    ``widest_row`` invariant ``tests/test_corpus_rescore_gate.py`` already
    uses as ground truth for "how many lanes should exist". That count is
    read directly off *tokens* (the largest number of tokens any single row
    contributes), not supplied externally, so it is not a tunable constant -
    it is a fact about this page's own rows. Combined with **best-first**
    ordering (search every open cluster at each step, apply the single
    highest-scoring valid split across all of them, not the first one found
    depth-first), the search always spends its bounded budget of splits on
    the largest, most evidence-backed boundaries first: real 20-70pt column
    gaps score far higher on ``_between_group_variance`` than any kerning-
    scale sub-grid offset, so the true boundaries are exhausted, and the cap
    is hit, before the search ever needs to look inside the noise band.

    Known scope limit, stated rather than hidden: this bound assumes the
    widest row on the page has no positionally-skipped column (the same
    assumption the corpus gate's own ``widest_row`` ground truth already
    makes). It also does not solve duplicated content: a page whose rows are
    themselves duplicated (the same row rendered twice, byte-identical) makes
    every noise-scale split look exactly as well-supported as a real one and
    does not raise the widest single row's value count, so the cap alone
    cannot separate real repeated structure from evidence manufactured by
    duplication - that is a content-identity question, not a position one,
    and out of this function's scope.
    """
    ordered = sorted(tokens, key=anchor)
    if len(ordered) <= 1:
        return [ordered] if ordered else []
    max_lanes = max(collections.Counter(tok[1] for tok in ordered).values())
    return _split_bounded(ordered, anchor, max_lanes)


def _split_bounded(
    ordered: list[tuple[int, int, float, float, str]],
    anchor: Callable[[tuple[int, int, float, float, str]], float],
    max_lanes: int,
) -> list[list[tuple[int, int, float, float, str]]]:
    """Best-first bisection: apply the single best valid split repeatedly.

    At each step, every current cluster proposes its own best row-supported
    split (or none); the highest-scoring proposal across *all* clusters is
    applied. Stops when no cluster has a valid split left, or when the
    cluster count reaches *max_lanes* - whichever comes first.
    """
    clusters = [ordered]
    while len(clusters) < max_lanes:
        proposal = _best_proposal(clusters, anchor)
        if proposal is None:
            break
        score, cluster_idx, left, right = proposal
        clusters[cluster_idx : cluster_idx + 1] = [left, right]
    return clusters


def _best_proposal(
    clusters: list[list[tuple[int, int, float, float, str]]],
    anchor: Callable[[tuple[int, int, float, float, str]], float],
) -> tuple[float, int, list, list] | None:
    """The highest-scoring valid split among all *clusters*, or ``None``."""
    best: tuple[float, int, list, list] | None = None
    for cluster_idx, cluster in enumerate(clusters):
        if len(cluster) <= 1:
            continue
        anchors = [anchor(t) for t in cluster]
        if anchors[0] == anchors[-1]:
            continue  # every anchor identical: nothing to split on
        for score, split_i in _split_candidates(anchors):
            left, right = cluster[:split_i], cluster[split_i:]
            if _row_support(left) >= 2 and _row_support(right) >= 2:
                if best is None or score > best[0]:
                    best = (score, cluster_idx, left, right)
                break  # this cluster's own best valid split; try the next cluster
    return best


def _split_candidates(anchors: list[float]) -> list[tuple[float, int]]:
    """Every split index with positive Otsu score, most evidence-backed first."""
    scored = []
    for i in range(1, len(anchors)):
        score = _between_group_variance(anchors[:i], anchors[i:])
        if score > 0.0:
            scored.append((score, i))
    scored.sort(key=lambda si: si[0], reverse=True)
    return scored


def _row_support(tokens: list[tuple[int, int, float, float, str]]) -> int:
    """Number of distinct source rows contributing a token to *tokens*."""
    return len({tok[1] for tok in tokens})


def _between_group_variance(low: list[float], high: list[float]) -> float:
    """Otsu's between-class variance for splitting *low* from *high*.

    Weighted by how many anchors actually fall on each side. Maximising this
    over all candidate splits is equivalent to minimising the summed
    within-group variance: it picks out the split where the two groups are
    both internally tight *and* far apart, rather than one chosen by the
    relative size of two arbitrary neighbouring magnitudes.
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

    # (1b) fold a wrapped label's continuation line into the row that carries
    # its values. #123 TICKET-B5.
    bands = _merge_continuation_bands(bands, _span_fonts(page, bbox))

    # (2) split each row at its last non-numeric word.
    #
    # Neither "everything before the first number" nor an x-clustered column
    # boundary survives real data: the first rule truncates
    # "Growth Plan after 17 October reversals" at the 17, and the second is
    # chicken-and-egg (that stray 17 forms its own x-cluster and drags the boundary
    # left). A table row is a label followed by values, so the last non-numeric word
    # ends the label. Parameter-free, and correct for both cases above.
    #
    # Ordered by _reading_order rather than plain x0 - #123 TICKET-B5. See that
    # function's docstring: a band can hold a genuine wrapped label folded onto
    # its value line by step (1)'s half-height rule, which needs top-line-then-
    # bottom-line order, not x0 order, or the two lines' words interleave
    # ("Central government net" / "debt" became "Central debt government net").
    raw: list[tuple[float, str, list[tuple[float, float, str]]]] = []
    for band in bands:
        ordered = _reading_order(band)
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


def _reading_order(
    band: list[tuple[float, float, float, float, str]],
) -> list[tuple[float, float, float, float, str]]:
    """Order a band's words for the label/value split in step (2). #123 TICKET-B5.

    Plain left-to-right (x0) order is correct for the vast majority of bands,
    which hold exactly one printed line - "one line" here means "whatever
    step (1)'s half-height grouping over-reached to include", not "words with
    equal y0"; two words on one printed line can legitimately have different
    y0 ("Nominal" and "GDP1" read left to right despite "GDP1"'s digit glyph
    starting 1.6pt lower than "Nominal"'s cap height, on page 61 of the
    reference corpus).

    A band that instead holds two GENUINE physical lines - a wrapped label at
    tight leading, merged into its value line below by step (1)'s half-height
    rule before this function ever runs - needs top-line-then-bottom-line
    order, not x0 order, or the two lines' words interleave ("Central
    government net" / "debt" becomes "Central debt government net").

    Told apart the same way the band-level merge above tells a continuation
    line from unrelated content: a genuine two-line split shares one left
    edge, unrelated content sharing a band by accident (an over-wide table
    region absorbing a chart legend) usually does not. Bisect the band by y0
    at every candidate boundary, keep only the bisections where BOTH halves'
    own leftmost word shares one left edge (nearest-point precision, as
    elsewhere in this module), and take the most confident one (the largest
    y0 gap at the boundary). No qualifying bisection - the ordinary case -
    falls back to plain x0 order, which is also the safe fallback for a wide
    band that accidentally absorbed unrelated content: it does not invent a
    line split that was never there.
    """
    if len(band) < 2:
        return list(band)
    by_y0 = sorted(band, key=lambda w: w[1])
    best: tuple[float, int] | None = None
    for i in range(1, len(by_y0)):
        gap = by_y0[i][1] - by_y0[i - 1][1]
        if gap <= 0:
            continue
        top, bottom = by_y0[:i], by_y0[i:]
        top_left = round(min(w[0] for w in top))
        bottom_left = round(min(w[0] for w in bottom))
        if top_left == bottom_left and (best is None or gap > best[0]):
            best = (gap, i)
    if best is None:
        return sorted(band, key=lambda w: w[0])
    split_i = best[1]
    top, bottom = by_y0[:split_i], by_y0[split_i:]
    return sorted(top, key=lambda w: w[0]) + sorted(bottom, key=lambda w: w[0])


def _span_fonts(page, bbox) -> list[tuple[tuple[float, float, float, float], str]]:
    """Font name per text span inside *bbox*, for ``_merge_continuation_bands``.

    ``page.get_text("words")`` (used everywhere else in this module) does not
    carry font identity, so this is a second, narrower read of the same
    region purely to recover it - nothing downstream of the band merge sees
    or depends on this data.
    """
    spans: list[tuple[tuple[float, float, float, float], str]] = []
    for block in page.get_text("dict", clip=bbox).get("blocks", ()):
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                spans.append((tuple(span["bbox"]), span["font"]))
    return spans


def _font_of(
    word: tuple[float, float, float, float, str],
    spans: list[tuple[tuple[float, float, float, float], str]],
) -> str | None:
    """The font of whichever span's box contains *word*'s centre point."""
    cx = (word[0] + word[2]) / 2
    cy = (word[1] + word[3]) / 2
    for (sx0, sy0, sx1, sy1), font in spans:
        if sx0 <= cx <= sx1 and sy0 <= cy <= sy1:
            return font
    return None


def _merge_continuation_bands(
    bands: list[list[tuple[float, float, float, float, str]]],
    spans: list[tuple[tuple[float, float, float, float], str]] = (),
) -> list[list[tuple[float, float, float, float, str]]]:
    """Fold a wrapped label's continuation line into the row that carries its
    values. #123 TICKET-B5.

    A row label genuinely split across two visual bands - "Central government
    net" / "debt" - has no numeric values of its own on its first line, so
    step (2) below would otherwise drop it as an orphan (a label with nothing
    to attach it to). A continuation line is identified structurally, from
    the page's own geometry, never a point constant:

    - **no values of its own** - the defining property of a continuation
      line, not a data row;
    - **the band immediately below it already carries values, on its own,
      before any merging** - single-hop only, deliberately. A first version
      chained right-to-left without this restriction, so any run of
      text-only bands sharing one left edge (a units note, a chart legend
      fragment, a blank line between real content - all common inside a
      located table bbox that is scoped generously) folded together and,
      wherever the chain eventually reached ANY numeric band regardless of
      distance, glued the whole run onto that band's values as if it were
      one row. On the real reference document this fabricated a phantom
      first row spanning half a page (measured on OBR page 48: a "£ billion"
      units note, an unrelated chart-legend fragment, and the table's real
      first data row all merged into one label). Anchoring the merge to a
      band that *already* has values, checked before any merge happens
      rather than accumulated through one, bounds the blast radius to
      exactly the shape TICKET-B5 asks for - a label wrapped across two
      lines - and nothing an unrelated run of text-only bands can reach;
    - **vertically adjacent to that band** - bounded by the *text-only*
      band's own text height (``bottom - top``), the same page-native metric
      already used above to decide whether two lines share one visual row. A
      real gap between two distinct table rows is about a full row height or
      more; the gap between a wrapped line and its own continuation is at
      most one line height, often negative (the two already overlap enough
      to share one band, just not enough to trigger the half-height merge
      above);
    - **shares the left edge** of the band below it - a continuation is a
      literal line wrap, not a new indent level. Compared at the nearest
      point precision already used for word geometry elsewhere in this module
      (``_superscript_tokens``'s ``round(x0)``), not exact float equality;
    - **is not a hierarchy marker** - "of which:" is a text-only band by the
      same definition above (no values of its own) and, on a table with
      indented children, regularly sits at the exact indent and gap of the
      child row beneath it. It already has its own identity
      (``_MARKER_RE``, checked below at the word level rather than after
      joining, since joining is what this function decides); a continuation
      line never does;
    - **shares the target band's font** - the geometric rule above (no
      values, adjacent, same left edge) turned out NOT to be enough: the
      reference corpus carries section headers ("Key fiscal determinants",
      "Financial and property sectors") set flush with the data row directly
      beneath them at the exact same indent and a gap tighter than one line
      height - geometrically identical to a genuine wrapped continuation, and
      wrongly folded into that row before this check existed (measured on
      OBR pages 59, 60, 62, 63, 66, 67: "Key fiscal determinants" absorbed
      into "Employment (million)", etc.). What actually distinguishes them is
      typographic, not geometric: a continuation is the same label carrying
      on, so it is set in the same font as the row it continues; a section
      header is conventionally set apart (bold/medium weight here) from the
      body text it introduces. Font identity is read from the page itself
      (``_span_fonts``/``_font_of``, a second pass over the same region,
      since ``page.get_text("words")`` used everywhere else in this module
      does not carry it) - not a named font, just an equality check against
      whatever the target band already uses. A word whose centre point falls
      in no span keeps this check permissive (``None == None`` still passes
      for two undetected bands) rather than blocking a real continuation on
      an extraction gap.

    Known scope limits, stated rather than hidden:

    - a label wrapped across **three or more** lines only has its innermost
      line reconstructed - the single-hop restriction above is deliberate,
      and TICKET-B5 asks for two lines;
    - a text-only header set in the SAME font as the row beneath it, at the
      same indent and closer than one line height, is still indistinguishable
      from a wrapped continuation. No fixture or corpus page in this plan
      exercises that shape.
    """
    if not bands:
        return bands

    def _band_font(band: list[tuple[float, float, float, float, str]]) -> str | None:
        for word in band:
            font = _font_of(word, spans)
            if font is not None:
                return font
        return None

    def _top(band: list[tuple[float, float, float, float, str]]) -> tuple[float, float]:
        top_y0 = min(w[1] for w in band)
        top_x0 = min(w[0] for w in band if w[1] == top_y0)
        return top_y0, top_x0

    def _text(band: list[tuple[float, float, float, float, str]]) -> str:
        ordered = sorted(band, key=lambda w: w[0])
        return " ".join(w[4].strip() for w in ordered).strip()

    tops = [_top(b) for b in bands]
    bottoms = [max(w[3] for w in b) for b in bands]
    has_value = [any(_is_value(w[4]) for w in b) for b in bands]
    is_marker = [bool(_MARKER_RE.match(_text(b))) for b in bands]
    fonts = [_band_font(b) for b in bands]

    merged: list[list[tuple[float, float, float, float, str]]] = []
    i = 0
    n = len(bands)
    while i < n:
        if (
            not has_value[i]
            and not is_marker[i]
            and i + 1 < n
            and has_value[i + 1]
            and fonts[i] == fonts[i + 1]
        ):
            top_y0, top_x0 = tops[i]
            height = bottoms[i] - top_y0
            next_top_y0, next_top_x0 = tops[i + 1]
            gap = next_top_y0 - bottoms[i]
            if gap <= height and round(next_top_x0) == round(top_x0):
                merged.append(bands[i] + bands[i + 1])
                i += 2
                continue
        merged.append(bands[i])
        i += 1
    return merged


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
