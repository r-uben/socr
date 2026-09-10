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

    GH-609 round 3: this is a PURE centroid point test, duplicated from
    ``binding.py`` back when that module used the same predicate (GH-330 /
    GH-331). It no longer matches: ``binding._words_in_region`` switched to
    a majority-overlap-area rule (GH-609) that provably rejects some words
    this centroid test still admits (a caption/label wider than the region
    that also dips deep enough for its centroid to land inside). This
    module's own predicate is UNCHANGED here -- fixing the drift between
    the two is #608's scope, not this one's. ``region=None`` returns
    *words* unchanged.
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


def cluster_band_words(words: list) -> list[list[tuple]]:
    """Cluster *words* into ordered baseline bands, keeping each band's WORDS.

    The clustering half of :func:`baseline_bands`, factored out (#652) so the
    prose-region partition (:func:`prose_region_words`) can reach the words a
    band is made of rather than only its numeric tokens. ``baseline_bands``
    calls this and then reduces each band to its tokens, so the two can never
    disagree about where a printed line begins.
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
    return raw_bands


def baseline_bands(words: list) -> list[_NativeBand]:
    """Cluster *words* into ordered baseline bands (top to bottom).

    Clustering key is the word's y-centre, with a tolerance derived from
    the region's own median word height (``_ROW_BAND_TOLERANCE_FRACTION``)
    — never ``round(word_y0)`` (GH-600). A band's token list is its
    genuine numeric tokens, left to right by x0; a band with none is kept
    (it still occupies a line) but contributes nothing to matching.
    """
    bands: list[_NativeBand] = []
    for band_words in cluster_band_words(words):
        band_words_sorted = sorted(band_words, key=lambda w: w[0])
        tokens = []
        for word in band_words_sorted:
            is_numeric, normalized = _is_genuine_numeric(word[4])
            if is_numeric:
                tokens.append(normalized)
        y_center = statistics.mean((w[1] + w[3]) / 2.0 for w in band_words)
        bands.append(_NativeBand(tokens=tuple(tokens), y_center=y_center))
    return bands


#: The fail-closed limit of TICKET-A1b's per-candidate ``ROW_SHAPE_MIN``
#: (``manifest._row_shape_reconciliation_ok``: the minimum numeric-token count
#: over a CANDIDATE's own numeric body rows). #649's page has no candidate to
#: derive it from -- the attempt emitted the table as column runs and authored
#: no markdown grid at all -- so the partition below falls back to the
#: strictest value the same formula can take: a band carrying ANY genuine
#: numeric token is table-shaped and is withheld. Measured on the ticket's own
#: fixture (Fed 1989-11-14 p3, 295 native words, 48 bands): every one of the
#: 16 swap-arrangement rows carries 1-2 genuine numeric tokens and every one of
#: the 32 prose/header bands carries 0, so this limit separates that page
#: exactly. It is a limit of an existing derivation, not a tuned threshold: no
#: value below 1 exists, and any value above it would ship printed numbers.
PROSE_BAND_MAX_NUMERIC_TOKENS: int = 0

_DIGIT_RE = re.compile(r"[0-9]")


def bears_printed_numeral(text: str) -> bool:
    """Whether *text* carries a printed digit of any form.

    #649 round 2 (Astra, 2026-09-10). The withholding decision must NOT reuse
    ``_is_genuine_numeric``: that predicate answers "is this token usable for
    numeric ROW MATCHING", and it deliberately says no to forms that are very
    much printed values -- a maturity date (``12/04/89``) is rejected outright,
    and ``(1)``-style decoration is excluded as a footnote marker. A band
    holding only ``12/04/89`` was therefore tagged prose and shipped verbatim
    under the unverified-scan banner, breaking the one promise that lane makes.
    The fixture tables' own maturity dates only vanished because a recognised
    amount happened to share their baseline.

    "Not useful for numeric row matching" is not "contains no printed value",
    so withholding asks the exhaustive question instead: does the token show a
    digit at all? Nothing about a digit's FORM can make it safe to ship off an
    unverified scan, which is why this looks for the digit rather than for a
    grammar of accepted numeric shapes -- there is no shape this could fail to
    enumerate.

    Deliberately NOT used for row matching, which still needs the narrower
    predicate: this one would count a page number and a footnote marker as
    table rows.
    """
    return bool(_DIGIT_RE.search(text or ""))


def partition_prose_bands(words: list, row_shape_min: int | None = None) -> list[tuple[bool, list]]:
    """*words* as ordered bands, each tagged ``(is_prose, band_words)``.

    The interleaved form of :func:`prose_region_words`, top of page to bottom.
    #649's caller rebuilds the page from this: it has to put the fail-closed
    marker where each withheld run actually sits, which the two flat lists
    cannot say. See :func:`prose_region_words` for what "prose" means here and
    for the disclosed cost of the default *row_shape_min*.

    Counts tokens by :func:`bears_printed_numeral`, never by
    ``_is_genuine_numeric`` -- see that function for why a row-matching
    predicate is the wrong instrument for a withholding decision.
    """
    if row_shape_min is None:
        row_shape_min = PROSE_BAND_MAX_NUMERIC_TOKENS + 1
    bands: list[tuple[bool, list]] = []
    for band in cluster_band_words(words):
        numeral_count = sum(1 for word in band if bears_printed_numeral(word[4]))
        ordered = sorted(band, key=lambda w: w[0])
        bands.append((numeral_count < row_shape_min, ordered))
    return bands


def corroboration_witness_words(words: list, row_shape_min: int | None = None) -> tuple[list, list]:
    """``(witness_words, unresolved_words)`` -- the prose a MODEL may be scored
    against, and the prose that is real page text but proves nothing.

    A page yields a witness ONLY when the shipping partition
    (:func:`partition_prose_bands`) finds no withheld numeric band anywhere on
    it. One withheld band and every band on the page is unresolved: there is
    then no witness, corroboration refuses, and #649's native recovery ships
    the page's own prose flagged with the numeric bands withheld. Model-prose
    salvage is disabled on exactly the pages where a table's extent is in
    question, which is what #652 asks for.

    #652 round 8 (Astra's ruling, 2026-09-10) replaced the geometric admission
    rule this function carried through rounds 2-7. FIVE successive variants of
    it were reproduced as fabrication paths, each on a real selection run and
    each shipping an invented sentence built from a table's own row labels:

    * the page-wide MEDIAN line advance as the walk's stopping step. Tightening
      an unrelated footnote block to 6pt pulled the median down, the table's
      own unchanged 12pt step became a "block break", and its label entered the
      witness. Text elsewhere must not redraw a table's extent.
    * the ANCHORS' OWN MEAN PITCH as that step. An average is not an upper
      bound on the individual steps inside one table: a units caption printed
      18pt under a row label, in a table whose rows average 12pt, stops the
      walk inside the table.
    * SEPARATION alone -- a run of unattributed bands with larger gaps around
      it. Separation proves a BLOCK exists, not that the block is prose. A
      table label wrapped over two lines is such a block, and so is a whole
      date table printed between two numeric rows.
    * a recognised numeric row on ONE side of the run. That proves a table is
      NEARBY. Two bank-name bands at the page edge, 6pt apart, with an amount
      24pt below them satisfied it.
    * a recognised numeric row on BOTH sides. Narrower, still not prose: a
      section heading or a wrapped header sits between two numeric sections
      just as readily as a paragraph does (250.0 / two bank-name bands / 300.0,
      the reviewer's reproduction at eccd394).
    * and the lever considered instead of this rewrite -- letting an ABSORBED
      band vouch for a side -- was reproduced as a fabrication path too: put a
      units caption at each end of that same geometry and the walk absorbs the
      captions, so the intervening label block becomes admissible again.

    The lesson is not that some sixth variant is waiting. Band-gap geometry
    measures where blocks BREAK; it cannot say what a block IS, and the
    printed page genuinely does not distinguish a two-line paragraph above a
    table from that table's wrapped header. Reliable model-prose salvage needs
    independent source evidence for the region AND for its transcription -- a
    source-verified prose region, or conservative matching against trusted
    source spans -- and that is separate work, not a threshold.

    Refusing costs the page no text. Since #649 the native layer's own prose
    ships flagged whether or not a model attempt corroborates, so the only
    thing discarded on a refusal is the model's WORDING. That the guard
    therefore accepts rarely is intended: #652 exists to stop unsupported model
    prose passing corroboration, not to maximise acceptance.

    The one page that still yields a witness is the pure-prose scan -- no
    printed numeral anywhere, so the shipping partition withholds nothing and
    there is no table whose extent could be in question. That case is asked
    through the shipping partition rather than through a second numeric
    detector, which is #652 round 5's finding: ``_is_genuine_numeric`` is a
    row-MATCHING predicate and deliberately rejects printed forms that are
    unmistakably values (a maturity date above all), so a table of institution
    names and dates carries no anchor at all while the shipping side correctly
    withholds every one of its bands. ``bears_printed_numeral``, via
    :func:`partition_prose_bands`, is the safety predicate and the right one to
    ask.

    Both lists are exhaustive: every word lands in exactly one of them. The
    caller subtracts unresolved tokens from the witness before scoring -- see
    ``manifest._prose_corroboration_ok`` -- so a third, silent case would let a
    band be neither evidence nor subtracted.
    """
    return witness_from_prose_partition(partition_prose_bands(words, row_shape_min))


def witness_from_prose_partition(bands: list) -> tuple[list, list]:
    """:func:`corroboration_witness_words` over an ALREADY-COMPUTED partition.

    Round 9 (Astra, 2026-09-10): the rule above is a statement about a PAGE
    ("no withheld numeric band anywhere"), so it is only sound when the
    partition it reads covers the page. ``manifest._prose_corroboration_ok``
    used to filter every word inside a detected table bbox away FIRST and
    partition what was left; a bbox that covered a scan's numeric bands but
    not its labels therefore deleted every digit before the check, the labels
    became a full witness, and a fabricated sentence shipped. A filtered
    region with no numerals is not a page with no numerals.

    Taking the partition as an argument lets the caller build ONE authoritative
    partition of the page's native words and hand the same object to both the
    corroboration decision and ``manifest.native_prose_floor_text``, so the
    two cannot be reading different populations of the same page.
    """
    if not bands:
        return [], []

    every_word = [word for _is_prose, band in bands for word in band]
    if any(not is_prose for is_prose, _band in bands):
        return [], every_word
    return every_word, []


def prose_region_words(words: list, row_shape_min: int | None = None) -> tuple[list, list]:
    """Split *words* into ``(prose_words, withheld_words)`` by baseline band.

    #649 / #652 (owner ruling, 2026-09-10): on a scanned page with NO detected
    table geometry there is no bbox to scope prose with, so the prose region is
    delimited by the page's own native baseline bands -- a band whose count of
    digit-bearing tokens (:func:`bears_printed_numeral`) is below
    *row_shape_min* is prose; every band at or above it is withheld, exactly
    the printed numeric content the D3 floor protects.

    *row_shape_min* defaults to ``PROSE_BAND_MAX_NUMERIC_TOKENS + 1`` (see that
    constant for the derivation and the measurement behind it).

    NOTHING ELSE is withheld, and that is a measured decision rather than an
    omission. A withheld table's own zero-token lines -- the header block
    above it, a wrapped row label inside it ("Bank for International" /
    "Settlements-" on the ticket's fixture) -- do ship as prose. The obvious
    fix, withholding every zero-token band inside the withheld bands' y-span,
    was tried and rejected: on a two-table page it swallows the entire
    paragraph printed BETWEEN the tables, and no threshold separates "wrapped
    row label" from "paragraph between two tables" (the identical trap
    ``manifest._row_shape_reconciliation_ok``'s docstring records for its own
    distance-anchored rounds). Shipping a bare label is the cheaper error of
    the two: it carries no printed value, so it cannot ship a wrong number,
    and the fail-closed marker sits right beside it saying the table was
    withheld. Losing a paragraph of policy text is the loss #649 exists to
    stop.

    The cost of the default *row_shape_min* runs the other way and is
    disclosed too: a prose line carrying any printed digit ("...has remained
    around 5-1/4 percent...", the one such line on the ticket's own fixture) is
    withheld with the table. That is the intended direction -- it is a printed
    value on a scan nothing verified -- and it is withheld, never silently
    dropped: #649's caller stamps a marker at every contiguous withheld run
    precisely so a line elided mid-paragraph is visible where it was elided.
    One fixture is not enough to calibrate anything looser; that calibration
    is the same follow-up ``manifest.PROSE_CORROBORATION_MIN`` waits on.

    Both lists are returned in PAGE READING ORDER (bands top to bottom, words
    left to right within a band), not in the input order of *words*: #649's
    caller ships the prose half as text, and the band order is the only order
    that reproduces the printed page.
    """
    prose: list = []
    withheld: list = []
    for is_prose, band in partition_prose_bands(words, row_shape_min):
        (prose if is_prose else withheld).extend(band)
    return prose, withheld


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


def table_shaped_native_row_count(words: list, row_shape_min: int) -> int:
    """Count of native baseline bands that look like a table row, by shape alone.

    Factored out of ``manifest._row_shape_reconciliation_ok`` (TICKET-A1b,
    #634) so TICKET-A2's truncation term (#645) can reuse the identical
    "table-shaped row" definition without a second implementation drifting
    from it. A band counts iff it has at least ``row_shape_min`` numeric
    tokens (a caller-supplied, per-candidate floor — see
    ``_row_shape_reconciliation_ok``'s own docstring for why that floor is
    derived from the candidate rather than a named constant) and is not the
    printed column-index legend row (``is_column_index_row``, a table
    convention, not data).
    """
    return sum(
        1
        for band in baseline_bands(words)
        if band.tokens
        and len(band.tokens) >= row_shape_min
        and not is_column_index_row(band.tokens)
    )


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
