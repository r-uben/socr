"""Row corroboration — ordered numeric-row match against native baseline bands.

## Why this exists

TICKET-A1a (defect census, 2026-09-06). ``tables/binding.py:bind()`` convicts
individual cells by 1:1 geometric binding (row band AND lane), which is the
right oracle when a candidate has *already* been chosen as the page's
winner. It has nothing to say about *selection* itself: when the structure
guard abstains (``header_unattributed``) and the S1 floor discards every
grid candidate sight-unseen, nothing today checks whether a candidate's
rows actually reproduce the page before it is thrown away. That is exactly
the failure the defect census found nine times over on ECB statistical
pages: a candidate that reproduced the page numerically was floored anyway.

This module is the mechanical, row-level corroboration check the census
used to establish that reproduction: does each of a candidate table's
numeric body rows sit, in order, as a contiguous run of tokens on ONE
native printed line? That is weaker than ``bind()``'s cell-level oracle
(no lane/column claim is made here at all) and deliberately so — it exists
to let a *reproducing* candidate survive selection, not to convict cells.

## What "ordered" buys over a multiset check

The old ``_value_guard`` (``native_verifier.py``) compared numeric
multisets: a flattened table (values present, attached to the wrong row)
has an identical multiset to a correctly-shaped one and passes vacuously.
Requiring a candidate row's tokens to appear, IN ORDER, as one contiguous
run on a single native line rules that out — a flattened row's tokens are
scattered across several native lines (or reordered on one), so a
transposition fails this check even though the multiset still matches.

## Baseline bands: no ``round(y0)`` (GH-600)

``binding._assign_bands`` keys native words on ``round(word[1])``, which
GH-600 found splits one printed row in two when its words straddle a
half-point boundary (0.14 pt spread, y0 = 208.43 -> 208 beside 208.57 ->
209). This module does not reuse that keying: baseline bands are built by
clustering word y-centres with a tolerance derived from the REGION's own
median word height, so the fix does not depend on a page-independent
constant. ``binding.py`` is otherwise untouched — GH-600 remains open for
its own callers.

## Gate semantics

``corroborate_rows`` returns raw counts; it does not itself decide
pass/fail — ``RowCorroboration.clears`` does, against three module
constants:

- ``ROW_CORROBORATION_MIN``: the minimum *share* of a candidate's numeric
  body rows that must corroborate (``bound / total``), inclusive. Measured
  on the census's wrapped-label page (bulletin p2): 36/39.
- ``EXTRA_NUMBERS_MAX_SHARE``: the maximum share of a candidate's numeric
  tokens that are simply absent from the region's native words at all (the
  invented-digit signal — a fabricated row can corroborate zero rows AND
  still slip under a loose ``ROW_CORROBORATION_MIN`` if ``total`` is
  large; this second gate catches invented content the row check alone
  would not).
- ``SKIPPED_ROWS_MAX`` (round 3): the maximum count of native numeric bands
  a table block's matched rows may "skip over" without any candidate row
  even attempting them — a signature distinct from a garbled row (which
  still costs a share-gate unbound row and explains its own gap). A row
  DROPPED from the candidate entirely shrinks ``bound`` and ``total``
  together, so ``share`` alone reads 1.0 and cannot see it; this third gate
  does. See ``RowCorroboration.skipped_native_rows``.

``total == 0`` (the candidate has no numeric body rows to check) is an
ABSENCE of evidence, not a failure — ``clears`` is ``None`` (abstain), the
same fail-safe posture ``binding.py`` uses throughout. The same abstention
applies when the region has no native numeric evidence at all
(``native_numeric_rows == 0``): a region with no printed numbers proves
nothing about a candidate that has some (the extraction may have missed
the region, or mis-scoped it), so this is doubt, not a conviction.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass

from socr.tables.native_verifier import is_numeric_token, strip_presentation

#: A spec-number decoration token, e.g. "(1)", "(12)" — numeric by
#: ``is_numeric_token`` but header/footnote decoration, not a data value.
#: Mirrors ``binding._SPEC_NUMBER_RE``; kept local rather than imported since
#: it is a private name of that module.
_SPEC_NUMBER_RE = re.compile(r"^\(\d+\)$")

#: Minimum share of a candidate's numeric body rows that must corroborate
#: against a native baseline band, inclusive. Measured on the census's
#: wrapped-label page (ECB economic bulletin p2, 2026-09-06 fixture run):
#: 36 of 39 rows bound. See docs/log/2026-09-06_A1a-row-corroboration.md.
ROW_CORROBORATION_MIN: float = 36 / 39

#: Maximum share of a candidate's numeric tokens that may be entirely absent
#: from the region's native words (fabricated, not merely misplaced). Set
#: strictly between two measured anchors (see
#: docs/log/2026-09-06_A1a-row-corroboration.md): the six clean ECB fixture
#: candidates measure at most 0.0075 (bulletin p2's wrapped-label extras);
#: a known-wrong candidate (bulletin p3 qwen, a truncated table the census
#: flagged for value drift) measures 0.0714. 0.02 sits with headroom on
#: both sides of that gap — a wrong candidate must fail this gate even when
#: its row-share alone would pass.
EXTRA_NUMBERS_MAX_SHARE: float = 0.02

#: Maximum count of native numeric bands "skipped over" by a table block's
#: matched rows (a native band strictly between the first and last matched
#: band index that no candidate row bound to — the geometric signature of a
#: DROPPED candidate row: omitting a row shrinks ``bound`` and ``total``
#: together, so the share gate alone cannot see it). Set strictly between
#: two measured anchors (see docs/log/2026-09-06_A1a-row-corroboration.md,
#: round 3): the six clean ECB fixture candidates measure 0 skipped bands
#: each; the same six candidates with their first data row deleted measure
#: at least 1. ``SKIPPED_ROWS_MAX = 0`` sits at the only value strictly
#: between those two anchors given they are consecutive integers.
SKIPPED_ROWS_MAX: int = 0

#: Baseline-band clustering tolerance, as a fraction of the region's median
#: word height. Measured on the real A1a fixture (ECB economic bulletin
#: p127-129, page 1, whole-page region, 734 native words): median word
#: height 9.56 pt, median line pitch (distinct baseline y-centres) 7.7 pt.
#: A fraction of 0.5 gives a tolerance of 4.78 pt -- comfortably less than
#: the 7.7 pt line pitch (so adjacent printed rows still separate) while
#: wide enough to keep one line's own words, whose y-centres differ only by
#: sub-pixel rounding noise, in the same band. Named rather than inlined so
#: a future fixture that falsifies it changes one number, not a scattered
#: literal.
_ROW_BAND_TOLERANCE_FRACTION = 0.5


@dataclass(frozen=True)
class RowCorroboration:
    """Outcome of corroborating one region's candidate rows against native words.

    ``bound``: candidate numeric body rows whose ordered numeric run sits
    contiguously on at least one native baseline band.
    ``total``: candidate numeric body rows considered (header, delimiter,
    and value-less rows excluded).
    ``extra_numbers``: candidate numeric tokens (normalized) with no
    matching native occurrence in the region at all, one entry per excess
    occurrence (a native token, once matched, is consumed — repeating a
    legitimate value does not manufacture extras).
    ``candidate_numbers``: total candidate numeric tokens considered (the
    denominator for ``extra_share``); not itself one of the ticket's four
    named outputs but required to express ``EXTRA_NUMBERS_MAX_SHARE`` as a
    share rather than a bare count.
    ``native_numeric_rows``: native baseline bands, within the region, that
    carry at least one genuine (non-spec-number) numeric token.
    ``skipped_native_rows``: for each table block, the EXCESS width of every
    gap between two consecutive BOUND rows' matched bands, beyond what the
    unbound candidate rows sitting in that same gap can account for. A
    candidate row that is present but garbled (a mismatch: it attempts
    every remaining band and matches none) already costs one unbound row
    against the row-share gate and explains one gap band; ``skipped_native_rows``
    only rises when a gap is WIDER than the number of present-but-unbound
    rows explains it — the geometric signature of a row DROPPED from the
    candidate entirely (round 3 review), not merely garbled. Summed across
    every table block found in the markdown.
    ``unbound_rows``: per table block, the 0-based indices (within that
    block's own numeric body rows) of candidate rows that failed to bind —
    exposed for A1b's per-row surfacing.
    ``skipped_bands``: the native band y-centres of EVERY gap (not just the
    excess counted in ``skipped_native_rows`` — this may be a superset,
    since an ordinary mismatch still leaves its own band's index unclaimed
    even though it costs nothing extra), across all blocks, sorted top to
    bottom — exposed for A1b's per-row surfacing.
    """

    bound: int
    total: int
    extra_numbers: tuple[str, ...]
    candidate_numbers: int
    native_numeric_rows: int
    skipped_native_rows: int
    unbound_rows: tuple[tuple[int, ...], ...]
    skipped_bands: tuple[float, ...]

    @property
    def share(self) -> float | None:
        """``bound / total``, or ``None`` when there is nothing to corroborate."""
        if self.total == 0:
            return None
        return self.bound / self.total

    @property
    def extra_share(self) -> float | None:
        """``len(extra_numbers) / candidate_numbers``, or ``None`` if none were seen."""
        if self.candidate_numbers == 0:
            return None
        return len(self.extra_numbers) / self.candidate_numbers

    @property
    def clears(self) -> bool | None:
        """Whether this candidate corroborates the region well enough to survive.

        ``None`` is abstention (no evidence either way): no numeric body
        rows to check, or no native numeric evidence in the region at all.
        """
        if self.total == 0 or self.native_numeric_rows == 0:
            return None
        if self.share is None or self.share < ROW_CORROBORATION_MIN:
            return False
        extra_share = self.extra_share
        if extra_share is not None and extra_share > EXTRA_NUMBERS_MAX_SHARE:
            return False
        if self.skipped_native_rows > SKIPPED_ROWS_MAX:
            return False
        return True


@dataclass
class _NativeBand:
    tokens: tuple[str, ...]  # left-to-right normalized numeric tokens, spec-numbers excluded
    y_center: float  # the band's clustered word y-centre (mean), for skipped-band reporting


def _word_centroid_in_region(word: tuple, region: tuple[float, float, float, float]) -> bool:
    rx0, ry0, rx1, ry1 = region
    cx = (word[0] + word[2]) / 2.0
    cy = (word[1] + word[3]) / 2.0
    return rx0 <= cx <= rx1 and ry0 <= cy <= ry1


def words_in_region(words: list, region: tuple | None) -> list:
    """Filter *words* to those whose box-centroid falls inside *region*.

    Same centroid predicate as ``binding.words_in_region`` (GH-330 /
    GH-331); duplicated locally rather than imported since that name is
    private to ``binding.py``. ``region=None`` returns *words* unchanged.
    """
    if region is None:
        return words
    try:
        x0, y0, x1, y1 = (float(v) for v in region)
    except (TypeError, ValueError):
        return words  # a malformed region is an absence of scoping, not a conviction
    if not (x0 <= x1 and y0 <= y1):
        return words
    box = (x0, y0, x1, y1)
    return [w for w in words if _word_centroid_in_region(w, box)]


def _is_genuine_numeric(text: str) -> tuple[bool, str]:
    """Whether *text* is a genuine (non-spec-number) numeric token.

    Returns ``(is_numeric, normalized)``. Spec-number decoration such as
    ``(1)`` is numeric by ``is_numeric_token`` but is header/footnote
    decoration, not a data value — excluded here the same way
    ``binding._project_candidate_data_columns`` excludes it.
    """
    if not is_numeric_token(text):
        return False, ""
    normalized = strip_presentation(text)
    if _SPEC_NUMBER_RE.match(normalized):
        return False, ""
    return True, normalized


def baseline_bands(words: list) -> list[_NativeBand]:
    """Cluster *words* into ordered baseline bands (top to bottom).

    Clustering key is the word's y-centre, with a tolerance derived from
    the region's own median word height (``_ROW_BAND_TOLERANCE_FRACTION``)
    — never ``round(word_y0)`` (GH-600). A band's token list is its
    genuine numeric tokens, left to right by x0; a band with none is kept
    (it still occupies a line) but contributes nothing to matching.
    """
    if not words:
        return []
    heights = [w[3] - w[1] for w in words if w[3] > w[1]]
    median_height = statistics.median(heights) if heights else 0.0
    tolerance = median_height * _ROW_BAND_TOLERANCE_FRACTION

    centered = sorted(words, key=lambda w: (w[1] + w[3]) / 2.0)
    raw_bands: list[list[tuple]] = []
    band_y_sum = 0.0
    band_y_count = 0
    for word in centered:
        y_center = (word[1] + word[3]) / 2.0
        if raw_bands and abs(y_center - band_y_sum / band_y_count) <= tolerance:
            raw_bands[-1].append(word)
            band_y_sum += y_center
            band_y_count += 1
        else:
            raw_bands.append([word])
            band_y_sum = y_center
            band_y_count = 1

    bands: list[_NativeBand] = []
    for band_words in raw_bands:
        band_words_sorted = sorted(band_words, key=lambda w: w[0])
        tokens = []
        for word in band_words_sorted:
            is_numeric, normalized = _is_genuine_numeric(word[4])
            if is_numeric:
                tokens.append(normalized)
        y_center = statistics.mean((w[1] + w[3]) / 2.0 for w in band_words)
        bands.append(_NativeBand(tokens=tuple(tokens), y_center=y_center))
    return bands


#: A GFM separator/rule cell: optional leading/trailing ':' around one or
#: more '-'. Deliberately more permissive than ``binding._STRICT_SEP_CELL_RE``
#: (which requires >= 3 dashes): this module drops a separator row purely to
#: keep it out of the row/token stream, never to gate whether a table
#: "counts" — an overly strict match would let a genuine rule row leak
#: through as spurious row content instead.
_SEPARATOR_CELL_RE = re.compile(r"^:?-+:?$")


def split_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_separator_row(cells: list[str]) -> bool:
    return (
        bool(cells)
        and all(_SEPARATOR_CELL_RE.match(c) for c in cells if c)
        and any("-" in c for c in cells)
    )


def table_blocks(markdown: str) -> list[list[list[str]]]:
    """Locate every markdown table block: runs of >= 2 consecutive pipe lines.

    Unlike ``binding.parse_grid`` (which requires every row in a block —
    header and body alike — to share the separator's exact cell count, and
    gives up on the WHOLE block the first time a row does not), this keeps
    each row's own cells as extracted. A single ragged interstitial row (a
    spanning header continuation the model emitted with one cell short, for
    instance) is common, real model output — it costs this module nothing,
    since row corroboration checks each row's own ordered tokens
    independently and never needs column counts to agree across rows.
    Separator/rule rows are dropped (they carry no content), and so is the
    row IMMEDIATELY ABOVE each one -- the GFM leaf header line. A header can
    legitimately be pure digits (year columns: ``| Item | 2023 | 2022 |``),
    so it cannot be told apart from a data row by content; position relative
    to the separator is the only reliable signal, and this module makes only
    that one exclusion (a multi-row SPANNING header, several equal-width
    rows stacked above the leaf line, is a coarser structural claim than a
    row-level corroboration check needs — any of those rows that carries a
    genuine numeric token, such as a bare column-index row, still ends up
    scored as a "numeric body row" that will typically fail to bind, costing
    at most a small, page-bounded undercount of ``total``; see the A1a
    measurement log). Two tables written back-to-back with no blank line
    between them are read as one block; that merges their row streams but
    does not lose either table's rows, since scoring is per-row, not
    per-table-shape.
    """
    lines = markdown.splitlines()
    pipe_idxs = [i for i, ln in enumerate(lines) if "|" in ln and ln.strip()]
    blocks: list[list[list[str]]] = []
    i = 0
    while i < len(pipe_idxs):
        j = i
        while j + 1 < len(pipe_idxs) and pipe_idxs[j + 1] == pipe_idxs[j] + 1:
            j += 1
        run = pipe_idxs[i : j + 1]
        if len(run) >= 2:
            run_cells = [split_cells(lines[k]) for k in run]
            separator_positions = {
                pos for pos, cells in enumerate(run_cells) if is_separator_row(cells)
            }
            header_positions = {pos - 1 for pos in separator_positions if pos - 1 >= 0}
            rows = [
                cells
                for pos, cells in enumerate(run_cells)
                if pos not in separator_positions and pos not in header_positions
            ]
            if rows:
                blocks.append(rows)
        i = j + 1
    return blocks


def is_column_index_row(tokens: tuple[str, ...]) -> bool:
    """True when *tokens* is exactly the consecutive integers 1..K, K = len(tokens).

    Real ECB statistical pages print a bold column-index legend row (a
    superscript numeral key naming each column, e.g. a whole row reading
    ``**1** **2** **3** ... **10**`` — measured on the A1a bulletin-p127-129
    page-1/page-3 qwen fixtures) directly below the leaf header. Lexically
    this is indistinguishable from a genuine numeric data row — its stub
    cell is non-blank ("1"), so the empty-stub exclusion does not catch it
    — but it is a printed-table CONVENTION (a legend, not a measurement),
    and the native page prints the identical index line right where the
    real header sits, so an unexcluded index row can spuriously bind. The
    rule is deliberately structural (values are exactly 1..K in order), not
    lexical (it does not look at the cells' text) — a genuine data row
    whose values happen to start at 1 and count up by exactly one each
    column is not a realistic false positive for a statistical table.
    """
    if len(tokens) < 2:
        return False  # a lone "1" is an ordinary single-column value, not a legend
    try:
        values = [int(strip_presentation(tok).replace(",", "")) for tok in tokens]
    except ValueError:
        return False
    return values == list(range(1, len(values) + 1))


def numeric_body_rows(rows: list[list[str]]) -> list[tuple[str, ...]]:
    """Return each row's ordered genuine numeric tokens, anchored to a numeric label.

    Column 0 (the row's stub/label) is normally NOT numeric data (a code or
    a line item is not a value) and is excluded from the row's token
    sequence. But when the stub cell IS itself a genuine numeric token — a
    bare year or ordinal row label, e.g. ``2018`` / ``2019`` — it is
    PREPENDED to the row's data tokens as an anchor. Native baseline bands
    already include such a label as their own first token (``baseline_bands``
    does not exclude any word by position), so anchoring a numeric label
    ties a row's match to ITS OWN printed line, not to a same-shaped value
    run copied from a different line. Without this, swapping two rows'
    VALUE cells while leaving their (numeric) labels in place — a whole-row
    misattribution defect — passed silently: the value-only tuple for the
    swapped-in row still matched the OTHER row's native band, since nothing
    tied the match to the row's own label. See
    ``test_row_value_swap_between_numeric_labels_does_not_clear`` and
    docs/log/2026-09-06_A1a-row-corroboration.md.

    A row with zero genuine numeric tokens in its data columns — a
    value-less panel/section row, or a header/decoration row that happens
    to sit in the pipe run — is excluded; it is not a "numeric body row".
    So is a row whose stub cell (column 0) is blank: a real printed data
    line always carries SOME row label (a year, a line item, a repeated
    stub); an empty stub is the signature of a spanning-header remnant that
    sits below the separator -- a footnote-marker row
    (``| | 1 | 1 | 5 | 5 |``) or a column-index row (``| | 1 | 2 | 3 | ... |``)
    -- neither of which is a data row (measured on the A1a ECB fixtures).
    A row (anchored or not) whose full token sequence is exactly the
    consecutive integers 1..K is also excluded — see
    ``is_column_index_row``.
    """
    result: list[tuple[str, ...]] = []
    for row in rows:
        if not row or not row[0].strip():
            continue
        stub = row[0].strip()
        is_numeric_stub, stub_normalized = _is_genuine_numeric(stub)
        data_tokens: list[str] = []
        for cell in row[1:]:
            for token in re.split(r"\s+", cell.strip()):
                if not token:
                    continue
                is_numeric, normalized = _is_genuine_numeric(token)
                if is_numeric:
                    data_tokens.append(normalized)
        if not data_tokens:
            continue
        full_tokens = (stub_normalized, *data_tokens) if is_numeric_stub else tuple(data_tokens)
        if is_column_index_row(full_tokens):
            continue
        result.append(full_tokens)
    return result


def _contiguous_run(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    """True when *needle* appears, in order, as a contiguous run in *haystack*."""
    if not needle:
        return False
    n = len(needle)
    for start in range(len(haystack) - n + 1):
        if haystack[start : start + n] == needle:
            return True
    return False


def match_rows_monotonic(
    rows: list[tuple[str, ...]], native_row_token_lists: list[tuple[str, ...]]
) -> list[int | None]:
    """Match *rows* (one table block, candidate order) against native bands.

    Returns, per row, the matched native band index or ``None`` if unbound.

    A row binds only to a native band whose index is STRICTLY GREATER than
    the index the previous BOUND row in this block matched. Without this, a
    whole-row value swap between two rows can still bind both rows: each
    swapped-in value tuple is a genuine contiguous run on SOME native band,
    just the wrong one, and an unordered any-band search (the original A1a
    implementation) cannot tell "matched" from "matched out of order". The
    rule was originally non-decreasing (ties allowed), which let a
    DUPLICATED candidate row bind the same native band twice (round 3
    review, Astra): strictly increasing makes the second occurrence of a
    duplicated row unbound (``bound < total`` surfaces it), even though the
    duplicated values are still individually present on the page (the
    extras gate alone would not catch it — see
    ``test_duplicate_row_second_occurrence_unbound``). Monotonicity is reset
    per table block — two markdown table blocks in one call are independent
    legends over the same region and are not required to be in native order
    relative to each other.
    """
    matches: list[int | None] = []
    last_idx = -1
    for row_tokens in rows:
        matched_idx = None
        start = last_idx + 1
        for idx in range(max(start, 0), len(native_row_token_lists)):
            if _contiguous_run(row_tokens, native_row_token_lists[idx]):
                matched_idx = idx
                break
        matches.append(matched_idx)
        if matched_idx is not None:
            last_idx = matched_idx
    return matches


def corroborate_rows(
    words: list, markdown: str, region: tuple[float, float, float, float] | None
) -> RowCorroboration:
    """Corroborate *markdown*'s candidate rows against native baseline bands.

    *words* is a ``page.get_text("words")``-shaped list (or already
    region-scoped; scoping is idempotent). *region* is the table's own
    ``(x0, y0, x1, y1)`` extent — words outside it are dropped before
    banding (matches ``binding.bind``'s GH-330 scoping). *markdown* may
    contain more than one table block; every block found is scored
    together against the same region.

    Never raises: a markdown with no parseable table, or a region with no
    words, returns a ``RowCorroboration`` of all zeros (``clears`` is
    ``None`` — abstain, not a conviction).
    """
    region_words = words_in_region(words, region)
    native_bands = baseline_bands(region_words)
    native_row_token_lists = [band.tokens for band in native_bands if band.tokens]
    native_row_y_centers = [band.y_center for band in native_bands if band.tokens]
    native_numeric_rows = len(native_row_token_lists)

    native_counts: Counter = Counter()
    for tokens in native_row_token_lists:
        native_counts.update(tokens)

    blocks = table_blocks(markdown)
    candidate_rows: list[tuple[str, ...]] = []
    unbound_rows: list[tuple[int, ...]] = []
    all_gap_band_idxs: set[int] = set()
    skipped_native_rows = 0
    bound = 0
    for rows in blocks:
        block_rows = numeric_body_rows(rows)
        candidate_rows.extend(block_rows)
        matches = match_rows_monotonic(block_rows, native_row_token_lists)
        bound_pairs = [(pos, idx) for pos, idx in enumerate(matches) if idx is not None]
        bound += len(bound_pairs)
        unbound_rows.append(tuple(i for i, m in enumerate(matches) if m is None))
        # A native band strictly between two consecutive BOUND rows' matched
        # indices is a "gap". A gap explained by an intervening UNBOUND
        # candidate row (a row that was present, attempted every remaining
        # band, and matched none -- a mismatch, not an omission) is not a
        # drop: the row-share gate already penalizes it. Only the EXCESS
        # gap width beyond what the intervening (present, unbound) rows can
        # account for is a genuine dropped-row signature -- see
        # ``test_dropped_row_does_not_clear`` and
        # ``test_wrapped_label_page_36_of_39_clears`` (which has unbound
        # rows but zero excess, and must still clear).
        for (pos_a, band_a), (pos_b, band_b) in zip(bound_pairs, bound_pairs[1:]):
            gap_bands = band_b - band_a - 1
            gap_positions = pos_b - pos_a - 1  # intervening (unbound) candidate rows
            all_gap_band_idxs.update(range(band_a + 1, band_b))
            skipped_native_rows += max(0, gap_bands - gap_positions)

    total = len(candidate_rows)
    skipped_bands = tuple(sorted(native_row_y_centers[idx] for idx in all_gap_band_idxs))

    remaining = Counter(native_counts)
    extra_numbers: list[str] = []
    candidate_numbers = 0
    for row_tokens in candidate_rows:
        for token in row_tokens:
            candidate_numbers += 1
            if remaining.get(token, 0) > 0:
                remaining[token] -= 1
            else:
                extra_numbers.append(token)

    return RowCorroboration(
        bound=bound,
        total=total,
        extra_numbers=tuple(extra_numbers),
        candidate_numbers=candidate_numbers,
        native_numeric_rows=native_numeric_rows,
        skipped_native_rows=skipped_native_rows,
        unbound_rows=tuple(unbound_rows),
        skipped_bands=skipped_bands,
    )
