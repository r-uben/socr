"""B1 / #591: the ``page_failed`` ending must keep native prose outside a
detected table, not drop it under the whole-page failure marker.

Before B1, ``_select_page_output_tagged``'s "nothing anywhere produced text"
branch shipped ``page_failed_marker(page_num)`` alone whenever every attempt
was empty -- even when ``p.native_text`` carried real prose the page never
needed OCR for (a question stem above a table, a policy directive paragraph).
B1 routes that ending through ``table_floor_text_for_source`` (GH-520's
four-condition coverage guard, now joined by ``_table_bbox_sane``'s two bbox
sanity checks) exactly as the ``structure_class`` floor and the
``TABLE_WITHHELD`` ending already do: guard satisfied -> splice the table
region(s) with the D3-style per-table marker and keep the rest; guard
violated -> the caller's own whole-page marker, unchanged from today.

Two test tiers:
  * synthetic ``PageState`` fixtures pin the guard-satisfied vs
    guard-violated difference directly (no PDF needed for most of them).
  * the two named census fixtures (ECB survey-2013 p1, Fed 1989-11-14 p3)
    are read read-only from ``~/Data/socr`` -- never copied into the repo --
    via ``BornDigitalDetector``, the same detector the pipeline runs, and
    measured for word recall against ``pdftotext``. Both are skipped if the
    fixture file is not present on the machine running the test.
"""

from __future__ import annotations

import collections
import re
import subprocess
from pathlib import Path

import pytest

from socr.core.manifest import (
    PROSE_CORROBORATION_MIN,
    SelectionProvenance,
    _prose_corroboration_ok,
    _select_page_output_tagged,
    _table_bbox_sane,
    page_failed_marker,
    table_floor_text_for_source,
)
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState

# ---------------------------------------------------------------------------
# Synthetic fixture text: prose before and after a single markdown table,
# built so the table's own bbox is easy to state exactly and every table row
# carries a genuine numeric token (row_corroboration's ``_is_genuine_numeric``
# recognizes plain digits, not the ``(1)``-style footnote markers).
# ---------------------------------------------------------------------------

PROSE_BEFORE = "Section four reports the survey responses collected this quarter."
PROSE_AFTER = "Respondents who selected 'other' were asked to specify a reason."
NATIVE_TABLE_MD = (
    "| Category | Jan | Apr |\n"
    "|---|---|---|\n"
    "| Decrease | 17 | 14 |\n"
    "| Unchanged | 78 | 73 |\n"
    "| Increase | 6 | 12 |\n"
)
NATIVE_TEXT_WITH_PROSE = f"{PROSE_BEFORE}\n\n{NATIVE_TABLE_MD}\n{PROSE_AFTER}\n"

# A word-tuple grid covering the fixture text above. pymupdf's own
# ``page.get_text("words")`` tuple shape is ``(x0, y0, x1, y1, text, block,
# line, word_no)`` -- reproduced by hand here rather than rendered through a
# real PDF, since only the geometry (which words land inside which region)
# matters to ``_table_bbox_sane`` and ``baseline_bands``, not real glyphs.
# Three horizontal bands: prose-before at y=100, five table rows at
# y=120..160 (one row per band, numeric tokens present), prose-after at
# y=180.
TABLE_BBOX = (50.0, 115.0, 250.0, 165.0)


def _word(x0: float, y0: float, text: str, line: int) -> tuple:
    return (x0, y0, x0 + len(text) * 5.0, y0 + 10.0, text, 0, line, 0)


def _words_for(prose_before: str, prose_after: str) -> list[tuple]:
    """Two prose bands above the table, three numeric table-row bands, three
    prose bands below -- so a bbox stretched to cover the whole page has MORE
    prose bands than numeric ones (6 vs 3), which is what B1's "too large"
    check must catch. One word per line keeps each token its own band."""
    words: list[tuple] = []
    before_tokens = prose_before.split()
    for i, tok in enumerate(before_tokens):
        y = 85.0 + (i % 2) * 10.0  # two prose-before bands
        words.append(_word(50.0, y, tok, i % 2))
    table_rows = [
        ["Decrease", "17", "14"],
        ["Unchanged", "78", "73"],
        ["Increase", "6", "12"],
    ]
    for row_i, row in enumerate(table_rows):
        y = 120.0 + row_i * 15.0
        for tok_i, tok in enumerate(row):
            words.append(_word(50.0 + tok_i * 40.0, y, tok, row_i + 10))
    after_tokens = prose_after.split()
    for i, tok in enumerate(after_tokens):
        y = 180.0 + (i % 3) * 10.0  # three prose-after bands
        words.append(_word(50.0, y, tok, i % 3 + 20))
    return words


NATIVE_WORDS = _words_for(PROSE_BEFORE, PROSE_AFTER)


def _no_text_marker_state(
    *,
    native_text: str = NATIVE_TEXT_WITH_PROSE,
    native_words: list[tuple] | None = None,
    detected_table_count: int = 1,
    detected_table_bboxes: list[tuple] | None = None,
    native_table_region_count: int = 1,
) -> PageState:
    """A page that reaches ``NO_TEXT_MARKER``: no attempts, no best_output, not
    structure-class (``has_tables=False`` keeps ``is_structure_class()``
    false), and -- load-bearing -- ``is_born_digital=False``.

    CONSILIUM-GATE finding, verified against ``_select_page_output_tagged``
    and every early return in ``born_digital._assess_page_signals``: the
    "nothing anywhere produced text" branch this ticket changed is reached
    ONLY when ``not (p.is_born_digital and p.native_text)`` -- every path
    through the preceding ``if p.is_born_digital and p.native_text:`` block
    (native-clean, native-fallback, the D3/TR-3 floor, the rotated-shredded
    floor, the structure-class floor, ...) returns unconditionally, so
    execution never falls through with native text intact. And in
    production ``is_born_digital=False`` always pairs with
    ``native_text=""`` -- every one of the 11 early returns in
    ``_assess_page_signals`` sets both together. So a REAL page can reach
    this branch only with EMPTY native text: B1's guard-routing fix is
    correct and behaves exactly as designed when this branch runs, but
    cannot currently change any real page's shipped bytes. ``is_born_digital
    =False`` here is the only way to exercise the branch with non-empty
    ``native_text`` at all; it does not claim this combination occurs on a
    real page. See ``docs/log/2026-09-07_B1-page-failed-marker-scope.md``."""
    ps = PageState(page_num=1)
    ps.is_born_digital = False
    ps.has_tables = False
    ps.native_text = native_text
    ps.native_words = NATIVE_WORDS if native_words is None else native_words
    ps.detected_table_count = detected_table_count
    ps.detected_table_bboxes = (
        [TABLE_BBOX] if detected_table_bboxes is None else detected_table_bboxes
    )
    ps.native_table_region_count = native_table_region_count
    return ps


def _tagged(ps: PageState):
    state = DocumentState.__new__(DocumentState)
    state.pages = {1: ps}
    return _select_page_output_tagged(state, 1)


# ---------------------------------------------------------------------------
# Core contract: guard satisfied keeps prose; guard violated (any of the
# four GH-520 conditions, or either new bbox check) ships the whole-page
# marker exactly as before B1.
# ---------------------------------------------------------------------------


def test_guard_satisfied_keeps_prose_around_the_table():
    ps = _no_text_marker_state()

    output, provenance = _tagged(ps)

    assert provenance is SelectionProvenance.NO_TEXT_MARKER
    assert PROSE_BEFORE in output.text
    assert PROSE_AFTER in output.text
    assert "[page 1 failed:" in output.text  # the D3-style per-table marker
    assert page_failed_marker(1) not in output.text  # not the whole-page marker
    assert "Decrease" not in output.text  # the table region itself is withheld
    assert output.status is PageStatus.ERROR
    assert output.audit_passed is False


def test_guard_violated_region_count_mismatch_ships_whole_page_marker():
    """GH-520 condition 3: the parser's own region count must match the
    detector's. A mismatch (here: the detector saw 2 tables, only 1 native
    region reconstructed) fails closed -- this is the #639 shape."""
    ps = _no_text_marker_state(detected_table_count=2)

    output, provenance = _tagged(ps)

    assert provenance is SelectionProvenance.NO_TEXT_MARKER
    assert output.text == page_failed_marker(1)
    assert PROSE_BEFORE not in output.text
    assert PROSE_AFTER not in output.text


def test_guard_violated_missing_bbox_ships_whole_page_marker():
    """GH-520 condition 2: fewer bboxes than detected tables."""
    ps = _no_text_marker_state(detected_table_count=1, detected_table_bboxes=[])

    output, provenance = _tagged(ps)

    assert output.text == page_failed_marker(1)


def test_guard_violated_no_markdown_block_ships_whole_page_marker():
    """GH-520 condition 4: the parser found no table block in the source
    text at all, even though the detector saw one."""
    ps = _no_text_marker_state(native_text=f"{PROSE_BEFORE}\n\n{PROSE_AFTER}\n")

    output, provenance = _tagged(ps)

    assert output.text == page_failed_marker(1)


def test_no_native_text_ships_whole_page_marker_unchanged():
    """The pre-B1 shape must be exactly preserved when there is nothing to
    splice around: empty native text still ships the plain marker."""
    ps = _no_text_marker_state(native_text="", native_words=[])

    output, provenance = _tagged(ps)

    assert output.text == page_failed_marker(1)
    assert provenance is SelectionProvenance.NO_TEXT_MARKER


# ---------------------------------------------------------------------------
# B1's two new bbox sanity checks (#639: a detected bbox can be structurally
# wrong -- the caption/units box, not the table body -- even when every
# GH-520 coverage count lines up).
# ---------------------------------------------------------------------------


def test_bbox_too_small_fails_closed():
    """The claimed table bbox is shrunk to a sliver that contains none of
    the table's own numeric rows: the box is not bounding the table."""
    sliver = (400.0, 400.0, 410.0, 410.0)  # nowhere near any word on the page
    ps = _no_text_marker_state(detected_table_bboxes=[sliver])

    assert _table_bbox_sane(ps) is False
    output, _ = _tagged(ps)
    assert output.text == page_failed_marker(1)


def test_bbox_too_large_fails_closed():
    """The claimed table bbox is stretched to also cover the prose bands
    above and below: the box swallowed paragraph text, not just the table."""
    everything = (0.0, 0.0, 900.0, 300.0)
    ps = _no_text_marker_state(detected_table_bboxes=[everything])

    assert _table_bbox_sane(ps) is False
    output, _ = _tagged(ps)
    assert output.text == page_failed_marker(1)


def test_bbox_exactly_the_table_is_sane():
    ps = _no_text_marker_state()
    assert _table_bbox_sane(ps) is True


# ---------------------------------------------------------------------------
# structure_class_floor_text and the TABLE_WITHHELD ending must keep their
# pre-B1 default (D3-style marker, not the caller's own): backward
# compatibility for table_floor_text_for_source's two existing callers.
# ---------------------------------------------------------------------------


def test_default_fallback_marker_is_still_the_d3_style_marker():
    ps = _no_text_marker_state()
    ps.d3_floor_png_ref = "![page-1](assets/page-1.png)"
    # No fallback_marker kwarg passed -- the structure_class_floor_text /
    # TABLE_WITHHELD call shape.
    out = table_floor_text_for_source(ps, 1, "not enough to satisfy the guard")
    assert out.startswith("[page 1 failed: unverifiable table")
    assert "assets/page-1.png" in out


# ---------------------------------------------------------------------------
# Real fixtures (read-only, external to the repo). Skipped when the fixture
# file is not present on this machine.
# ---------------------------------------------------------------------------

ECB_SURVEY_P1 = (
    Path.home()
    / "Data/socr/census-ecb-2026-09-06/in"
    / "ecb-surveys-2013-ecb.blssurvey2013q1.en-p29-31.pdf"
)
FED_1989_P3 = Path.home() / "Data/socr/fed-sample-2026-09-05/in/fed-1989-11-14-minutes.pdf"


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9%.\-]+", text.lower())


def _recall(expected_words: list[str], produced_text: str) -> float:
    if not expected_words:
        return 1.0
    expected = collections.Counter(expected_words)
    produced = collections.Counter(_word_tokens(produced_text))
    matched = sum(min(c, produced[w]) for w, c in expected.items())
    return matched / sum(expected.values())


def _words_outside_bbox(words: list[tuple], bbox: tuple[float, float, float, float]) -> list[str]:
    x0, y0, x1, y1 = bbox
    return [
        w[4]
        for w in words
        if not (w[0] >= x0 - 1 and w[1] >= y0 - 1 and w[2] <= x1 + 1 and w[3] <= y1 + 1)
    ]


@pytest.mark.skipif(not ECB_SURVEY_P1.exists(), reason="fixture not present on this machine")
def test_ecb_survey_2013_p1_guard_fails_closed_639():
    """Measured finding (#639): on this page ``find_tables()`` reports THREE
    bboxes (the survey-question's own table plus two mis-detected regions --
    almost certainly the chart-caption/units area, #639's exact shape) while
    only ONE native table region actually reconstructs. GH-520 condition 3
    (region count must equal detection count) fails, so B1's guard floors
    the whole page here -- the ticket's own escape clause. This test pins
    that measured behaviour with exact numbers rather than asserting the
    ≥0.9 recall the ticket's Done-when describes for the guard-satisfied
    case, which this real fixture does not reach."""
    fitz = pytest.importorskip("fitz")
    from socr.core.born_digital import BornDigitalDetector

    detector = BornDigitalDetector()
    pa = detector.detect_page(ECB_SURVEY_P1, 1)

    assert pa.detected_table_count == 3, pa.detected_table_count
    assert pa.native_table_region_count == 1, pa.native_table_region_count

    doc = fitz.open(str(ECB_SURVEY_P1))
    words = doc[0].get_text("words")
    doc.close()

    ps = PageState(page_num=1)
    ps.native_text = pa.native_text
    ps.native_words = words
    ps.detected_table_count = pa.detected_table_count
    ps.detected_table_bboxes = list(pa.detected_table_bboxes)
    ps.native_table_region_count = pa.native_table_region_count

    marker = page_failed_marker(1)
    out = table_floor_text_for_source(ps, 1, pa.native_text, fallback_marker=marker)

    assert out == marker  # guard fails closed: #639, not B1's mechanism


@pytest.mark.skipif(not ECB_SURVEY_P1.exists(), reason="fixture not present on this machine")
def test_ecb_survey_2013_p1_guard_satisfied_with_the_tables_own_bbox_achieves_recall():
    """The same page, with the detector restricted to only the table's own
    bbox (index 0 of the three -- the one ``find_table_blocks`` actually
    reconstructs), demonstrates B1's spliced-prose mechanism on REAL text:
    guard satisfied, ≥0.9 word recall of everything outside the table
    against the page's own words, table body withheld."""
    fitz = pytest.importorskip("fitz")
    from socr.core.born_digital import BornDigitalDetector

    detector = BornDigitalDetector()
    pa = detector.detect_page(ECB_SURVEY_P1, 1)
    table_bbox = pa.detected_table_bboxes[0]

    doc = fitz.open(str(ECB_SURVEY_P1))
    words = doc[0].get_text("words")
    doc.close()

    ps = PageState(page_num=1)
    ps.native_text = pa.native_text
    ps.native_words = words
    ps.detected_table_count = 1
    ps.detected_table_bboxes = [table_bbox]
    ps.native_table_region_count = 1

    marker = page_failed_marker(1)
    out = table_floor_text_for_source(ps, 1, pa.native_text, fallback_marker=marker)

    assert out != marker
    assert "Decrease considerably" not in out  # table body withheld

    expected = _word_tokens(" ".join(_words_outside_bbox(words, table_bbox)))
    recall = _recall(expected, out)
    assert recall >= 0.9, recall


@pytest.mark.skipif(not FED_1989_P3.exists(), reason="fixture not present on this machine")
def test_fed_1989_11_14_p3_has_no_native_text_to_recover():
    """Measured finding: on this page ``BornDigitalDetector`` returns
    ``is_born_digital=False`` and an EMPTY ``native_text`` -- an earlier,
    unrelated gate in ``_assess_page_signals`` (encoding-corruption
    detection) routes the page to OCR before the GH-520 table guard is ever
    consulted. B1's fix operates on ``p.native_text``, so it cannot recover
    this page's prose no matter which branch calls it -- there is nothing to
    splice. The real pipeline run for this fixture also does not reach the
    ``NO_TEXT_MARKER`` branch B1 changed (it reaches
    ``UNVERIFIABLE_TABLE_SCANNED``, now also guarded -- see
    ``test_fed_1989_11_14_p3_scanned_branch_guard_passes_but_no_table_block_to_splice``
    below). This test pins the empty-native-text finding, which holds
    independently of that branch question."""
    from socr.core.born_digital import BornDigitalDetector

    detector = BornDigitalDetector()
    pa = detector.detect_page(FED_1989_P3, 3)

    assert pa.is_born_digital is False
    assert pa.native_text == ""

    marker = page_failed_marker(3)
    ps = PageState(page_num=1)
    ps.native_text = pa.native_text
    ps.detected_table_count = pa.detected_table_count
    ps.detected_table_bboxes = list(pa.detected_table_bboxes)
    ps.native_table_region_count = pa.native_table_region_count
    out = table_floor_text_for_source(ps, 3, pa.native_text, fallback_marker=marker)

    assert out == marker  # nothing to splice: empty source_text short-circuits


# ---------------------------------------------------------------------------
# B1 (option 3, team-lead's post-CONSILIUM-GATE extension): the SAME
# fail-closed philosophy applied to ``UNVERIFIABLE_TABLE_SCANNED``, which
# splices ``best_output.text`` around a withheld table region with no
# coverage or corroboration guard at all. ``_prose_corroboration_ok`` gates
# that splice on the attempt's own outside-table vocabulary overlapping the
# page's real native words -- geometric/mechanical only, it never trusts the
# attempt's row/column structure.
# ---------------------------------------------------------------------------


def _scanned_table_state(
    *,
    attempt_text: str,
    native_words: list[tuple] | None = None,
    detected_table_bboxes: list[tuple] | None = None,
) -> PageState:
    """A page reaching ``UNVERIFIABLE_TABLE_SCANNED``: not born-digital, the
    source-evidence gate rejected the scanned table, and there is a failed
    OCR attempt to (maybe) splice prose from."""
    ps = PageState(page_num=1)
    ps.is_born_digital = False
    ps.has_tables = False
    ps.native_text = ""  # apply_born_digital never sets this when not born-digital
    ps.native_words = NATIVE_WORDS if native_words is None else native_words
    ps.detected_table_bboxes = [] if detected_table_bboxes is None else detected_table_bboxes
    ps.detected_table_count = 0  # measured on Fed p3: native detection found nothing
    ps.scanned_table_evidence_failed = True
    attempt = PageOutput(
        page_num=1,
        text=attempt_text,
        status=PageStatus.ERROR,
        engine="nougat",
        audit_passed=False,
        failure_mode=FailureMode.HALLUCINATION,
    )
    ps.attempts = [attempt]
    ps.best_output = attempt
    return ps


GENUINE_ATTEMPT_MD = (
    f"{PROSE_BEFORE}\n\n"
    "| Category | Jan | Apr |\n"
    "|---|---|---|\n"
    "| Decrease | 17 | 14 |\n"
    "| Unchanged | 78 | 73 |\n"
    "| Increase | 6 | 12 |\n"
    f"\n{PROSE_AFTER}\n"
)

FABRICATED_ATTEMPT_MD = (
    "The quorum ratified an entirely unrelated resolution about municipal "
    "bond covenants and pension fund exposure, none of which appears "
    "anywhere on this page.\n\n"
    "| Category | Jan | Apr |\n"
    "|---|---|---|\n"
    "| Decrease | 99 | 98 |\n"
    "|---|\n"
)


def test_prose_corroboration_guard_satisfied_keeps_prose():
    """The attempt's outside-table vocabulary is the page's own -- keep it."""
    ps = _scanned_table_state(attempt_text=GENUINE_ATTEMPT_MD)

    assert _prose_corroboration_ok(ps, GENUINE_ATTEMPT_MD) is True

    output, provenance = _tagged(ps)

    assert provenance is SelectionProvenance.UNVERIFIABLE_TABLE_SCANNED
    assert PROSE_BEFORE in output.text
    assert PROSE_AFTER in output.text
    assert "Decrease" not in output.text  # table region still withheld


def test_prose_corroboration_guard_violated_never_ships_the_invented_prose():
    """The attempt's vocabulary shares almost nothing with the page's real
    native words -- its prose must never be spliced around the withheld table.

    #649 changed what fills the gap, not what is refused. The attempt's
    invented sentences are still discarded; what ships in their place is the
    page's OWN trusted text layer, flagged, with every numeric band withheld
    behind the same marker. Losing the page's real prose was never part of
    refusing the model's."""
    ps = _scanned_table_state(attempt_text=FABRICATED_ATTEMPT_MD)

    assert _prose_corroboration_ok(ps, FABRICATED_ATTEMPT_MD) is False

    output, provenance = _tagged(ps)

    assert provenance is SelectionProvenance.UNVERIFIABLE_TABLE_SCANNED
    assert "[page 1 failed: unverifiable table — see image]" in output.text
    # The fabrication is gone.
    assert "quorum" not in output.text
    assert "municipal" not in output.text
    # The page's own prose is not. Asserted token by token: this fixture's
    # word grid round-robins each token onto its own baseline (see
    # ``_words_for``, built to exercise band COUNTS), so the recovered body
    # reproduces those synthetic bands rather than readable sentences.
    assert "survey" in output.text
    assert "responses" in output.text
    assert "Respondents" in output.text
    # And its numeric rows are still withheld.
    assert "78" not in output.text
    assert "| Decrease |" not in output.text


def test_prose_corroboration_guard_no_witness_fails_closed():
    """No native words cached (the orchestrator's caching gate never ran for
    this page, or the PDF had no real text layer) -- absence of a check is
    not corroboration; fail closed exactly like a violated guard."""
    ps = _scanned_table_state(attempt_text=GENUINE_ATTEMPT_MD, native_words=[])

    assert _prose_corroboration_ok(ps, GENUINE_ATTEMPT_MD) is False

    output, _ = _tagged(ps)
    assert output.text == "[page 1 failed: unverifiable table — see image]"


# ---------------------------------------------------------------------------
# #650: PROSE_CORROBORATION_MIN=0.5 was set with no low anchor -- neither
# census fixture (Fed p3 nougat, ECB survey p1) contained a fabrication, so
# both measured overlap 1.0. FABRICATED_ATTEMPT_MD above is a real
# low-anchor (measured ratio below), but its vocabulary shares almost
# nothing with the page at all -- an easy case for ANY positive floor to
# catch. The harder, more realistic case per the issue: a model that
# *paraphrases* the page's own prose (reusing a good share of its real
# vocabulary) while inventing sentences/claims the native layer does not
# contain at all -- the way a hallucinating OCR attempt actually behaves,
# not a wall of unrelated text. This fixture is built to land close to the
# 0.5 floor rather than far below it, so the assertion actually exercises
# the threshold rather than a case any cutoff would separate.
# ---------------------------------------------------------------------------

NEAR_FLOOR_FABRICATED_ATTEMPT_MD = (
    "This quarter, respondents in section four reported that the survey "
    "responses were quietly redirected to an undisclosed offshore account "
    "before regulators could specify a reason for the missing funds.\n\n"
    "| Category | Jan | Apr |\n"
    "|---|---|---|\n"
    "| Decrease | 41 | 39 |\n"
    "|---|\n"
)


def test_prose_corroboration_near_floor_fabrication_measured_ratios(monkeypatch):
    """#650: measure both fixtures' overlap ratios directly (not just the
    pass/fail verdict) and pin that the 0.5 floor is load-bearing -- with it
    monkeypatched to 0.0, the same fabricated attempt's verdict flips.

    Measured (via ``_PROSE_TOKEN_RE`` outside-table tokens, see manifest.py;
    ``_scanned_table_state``'s default ``detected_table_bboxes=[]`` means
    the table-row words also count as native vocabulary here, same as the
    other tests in this section):
      * genuine (``GENUINE_ATTEMPT_MD``): overlap ratio 0.947 -- passes 0.5.
      * fabricated, paraphrase-style (``NEAR_FLOOR_FABRICATED_ATTEMPT_MD``):
        overlap ratio 0.458 -- BELOW 0.5, but much closer to the floor than
        ``FABRICATED_ATTEMPT_MD``'s 0.05 above, despite reusing a
        substantial share of the page's real vocabulary (quarter, section,
        four, respondents, survey, responses, specify, reason all appear
        genuinely on the page), because it invents an entire unrelated
        claim (an offshore account, missing funds, regulators) the native
        layer does not contain. The floor separates this realistic,
        near-boundary case, not just the far-below-floor wall-of-noise case
        in ``test_prose_corroboration_guard_violated_ships_marker_only``.
    """
    from socr.core.manifest import _PROSE_TOKEN_RE

    ps = _scanned_table_state(attempt_text=NEAR_FLOOR_FABRICATED_ATTEMPT_MD)

    def _measured_ratio(attempt_text: str) -> float:
        bboxes = ps.detected_table_bboxes
        native_tokens: set[str] = set()
        for w in ps.native_words:
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if any(bx0 <= cx <= bx1 and by0 <= cy <= by1 for bx0, by0, bx1, by1 in bboxes):
                continue
            native_tokens.update(_PROSE_TOKEN_RE.findall(text.lower()))
        attempt_tokens = set(_PROSE_TOKEN_RE.findall(attempt_text.lower()))
        return len(attempt_tokens & native_tokens) / len(attempt_tokens)

    genuine_ratio = _measured_ratio(GENUINE_ATTEMPT_MD)
    fabricated_ratio = _measured_ratio(NEAR_FLOOR_FABRICATED_ATTEMPT_MD)

    assert genuine_ratio == pytest.approx(0.9474, abs=0.001)
    assert fabricated_ratio == pytest.approx(0.4583, abs=0.001)
    assert fabricated_ratio < PROSE_CORROBORATION_MIN < genuine_ratio

    # The floor separates the two cases at its real value...
    assert _prose_corroboration_ok(ps, GENUINE_ATTEMPT_MD) is True
    assert _prose_corroboration_ok(ps, NEAR_FLOOR_FABRICATED_ATTEMPT_MD) is False

    # ...and is load-bearing: monkeypatching it to 0.0 flips the fabricated
    # verdict (any nonzero overlap now clears the floor), proving the guard
    # actually depends on PROSE_CORROBORATION_MIN rather than always failing
    # closed for some unrelated reason.
    monkeypatch.setattr("socr.core.manifest.PROSE_CORROBORATION_MIN", 0.0)
    assert _prose_corroboration_ok(ps, NEAR_FLOOR_FABRICATED_ATTEMPT_MD) is True


@pytest.mark.skipif(not FED_1989_P3.exists(), reason="fixture not present on this machine")
def test_fed_1989_11_14_p3_scanned_branch_guard_passes_but_no_table_block_to_splice():
    """Real fixture, real cached attempt: measured finding for #591/B1.

    Fed 1989-11-14 p3 is a scanned page WITH a real native text layer (295
    pymupdf words; ``BornDigitalDetector`` still returns ``is_born_digital=
    False`` for it, so ``apply_born_digital`` never copies that text onto
    ``PageState.native_text`` -- see ``orchestrator.py``'s widened
    native-words caching, B1). The real cached OCR attempt for this page is
    nougat, ``failure_mode=hallucination``: it read the page's genuine
    vocabulary (bank names, dollar amounts, the FOMC directive paragraph)
    but emitted it with the swap-arrangement table's rows and columns
    reordered into one run per column rather than one row per line -- a
    STRUCTURAL defect the corroboration guard, being vocabulary-only by
    design, correctly does not catch: ``_prose_corroboration_ok`` measures
    True here (overlap 1.0, see the decision log's overlap table).

    ``splice_all_table_regions`` finds no markdown pipe-table syntax in
    nougat's raw text (it never emitted one) and returns ``None``, so no
    attempt can be spliced here regardless of the guard.

    #649: that used to end the page -- the D3 marker shipped ALONE and the
    three paragraphs of the FOMC domestic policy directive printed below the
    swap-arrangement table went with it. They now ship from the page's own
    text layer, flagged, with every numeric band withheld behind the same
    marker. Both halves are asserted: the directive comes back, and not one
    of the table's printed values does."""
    import fitz

    from socr.core.manifest import splice_all_table_regions

    doc = fitz.open(str(FED_1989_P3))
    words = doc[2].get_text("words")
    doc.close()

    nougat_text_path = (
        Path.home()
        / "Data/socr/census-591-recheck/out/fed-1989-11-14-minutes/cache/ef"
        / "ef6b822de4eb1e8546c0fa1d51be70b25e5f0200b4701462000c3c8773ca9a65.json"
    )
    if not nougat_text_path.exists():
        pytest.skip("cached nougat attempt not present on this machine")
    import json

    nougat_text = json.loads(nougat_text_path.read_text())["text"]

    ps = PageState(page_num=1)
    ps.is_born_digital = False
    ps.native_text = ""
    ps.native_words = words
    ps.detected_table_count = 0
    ps.detected_table_bboxes = []
    ps.scanned_table_evidence_failed = True
    attempt = PageOutput(
        page_num=3,
        text=nougat_text,
        status=PageStatus.ERROR,
        engine="nougat",
        audit_passed=False,
        failure_mode=FailureMode.HALLUCINATION,
    )
    ps.attempts = [attempt]
    ps.best_output = attempt

    assert _prose_corroboration_ok(ps, nougat_text) is True  # measured: overlap 1.0
    assert splice_all_table_regions(nougat_text, marker_line="[x]", png_ref="") is None

    state = DocumentState.__new__(DocumentState)
    state.pages = {3: ps}
    output, provenance = _select_page_output_tagged(state, 3)

    assert provenance is SelectionProvenance.UNVERIFIABLE_TABLE_SCANNED
    # The D3-style scanned-table marker still stamps the withheld table -- not
    # page_failed_marker (that's NO_TEXT_MARKER's marker, a different branch,
    # see the test above) -- and the page keeps its ERROR ending.
    assert "[page 3 failed: unverifiable table — see image]" in output.text
    assert output.status is PageStatus.ERROR
    assert output.audit_passed is False

    # #649's own loss, recovered: the directive's three paragraphs.
    assert "following domestic policy directive" in output.text
    assert "The information reviewed at this meeting suggests" in output.text
    assert "civilian unemployment rate" in output.text

    # And nothing from the withheld swap-arrangement table. Every printed
    # amount and every maturity date stays behind the marker.
    for withheld_value in ("250.0", "1,000.0", "6,000.0", "4,000.0", "1,250.0"):
        assert withheld_value not in output.text, withheld_value

    # Recorded where the corpus reads it, not only in the bytes.
    assert any("scanned_prose_recovered" in note for note in output.audit_notes)

    # And the page is no longer counted as marker-only: it ships content now.
    from socr.core.manifest import is_page_failed_marker

    assert is_page_failed_marker(output.text) is False
