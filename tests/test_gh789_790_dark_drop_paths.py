"""GH-789 / GH-790 -- two leftovers from PR #788's desk review.

**GH-789**: of the five ``rowize_from_word_list`` call sites,
``reconstruct.py``'s destroyed-token fallback (inside
``_reconstruct_table_regions_for_words``, reached only when the text-strategy
grid boundary-splits a native numeric token and is rejected -- GH-144) was the
one that never passed ``orphan_drops``. Its output feeds ``out.extend(rowized)``
directly, i.e. these ARE shipped table regions, so an orphan word dropped on
this path was a genuine, unmeasured table loss -- the exact residual GH-418
step 1 exists to make visible everywhere else.

**GH-790**: ``_last_extraction_grid_rejections`` (GH-195) has the identical gap
its sibling ``_last_extraction_orphan_drops`` (GH-418) was given a fix for --
reset only inside ``_assess_page``, so a harness calling ``extract_structured``
directly (the same standalone shape ``test_born_digital_aligned_runs.py``
already exercises) raised ``AttributeError`` the first time a rejection fired
without ever going through ``_assess_page`` first.

Both tests are written to fail with an ``AttributeError``/empty-list at the
pre-fix revision, not merely assert a changed value, per the reachability
requirement in the ticket.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from socr.core.born_digital import BornDigitalDetector  # noqa: E402
from socr.tables.reconstruct import reconstruct_table_regions  # noqa: E402


class _FakeRow:
    def __init__(self, bbox, cells):
        self.bbox = bbox
        self.cells = cells


class _FakeTable:
    def __init__(self, bbox, grid, rows):
        self.bbox = bbox
        self._grid = grid
        self.rows = rows

    def extract(self):
        return self._grid


class _FakeResult:
    def __init__(self, tables):
        self.tables = tables


def _destroyed_grid_with_sparse_orphan_row():
    """A 4-column (3-numeric-lane) table whose text-strategy grid boundary-
    splits ``"0.67"`` into ``"0"`` -- the GH-144 defect that triggers
    rejection and the word-geometry rowizer fallback at
    ``reconstruct.py:371`` (the ONE ``rowize_from_word_list`` call site GH-789
    is about).

    The final "Memo" row populates only ONE of the table's three numeric
    lanes (``A``) and carries an orphan word (``"n.a."``) 50pt right of lane
    ``C`` -- past the snap radius (``_LANE_X_TOL_PT * _LANE_SNAP_MULT`` =
    18pt) of every lane. Deliberately sparse: a row with >= 2 populated
    numeric lanes is a "data row" under GH-418 step 2 and the fallback
    rowizer CAPTURES a stray word into a trailing column instead of dropping
    it -- this fixture needs the drop, not the capture, so the row must stay
    below that gate (see ``rowize_from_word_list``'s own docstring).

    Returns ``(page, rejections, orphan_drops)`` set up but not yet run.
    """
    grid = [
        ["Firm", "A", "B", "C"],
        ["Alpha", "0", "0.61", "0.06"],  # destroyed: native "0.67" != raw cell "0"
        ["Beta", "0.85", "0.80", "0.05"],
        ["Gamma", "1.00", "0.94", "0.06"],
        ["Delta", "1.10", "1.06", "0.04"],
        ["Memo", "0.50", "", ""],
    ]

    def _row(y0, y1, cells):
        return _FakeRow((0.0, y0, 260.0, y1), cells)

    rows = [
        _row(
            10.0 * n,
            10.0 * (n + 1),
            [
                (0.0, 10.0 * n, 40.0, 10.0 * (n + 1)),
                (50.0, 10.0 * n, 90.0, 10.0 * (n + 1)),
                (100.0, 10.0 * n, 140.0, 10.0 * (n + 1)),
                (150.0, 10.0 * n, 190.0, 10.0 * (n + 1)),
            ],
        )
        for n in range(6)
    ]
    table = _FakeTable((0.0, 0.0, 260.0, 60.0), grid, rows)

    labels = ["Firm", "Alpha", "Beta", "Gamma", "Delta", "Memo"]
    lane_a = [None, "0.67", "0.85", "1.00", "1.10", "0.50"]
    lane_b = [None, "0.61", "0.80", "0.94", "1.06", None]
    lane_c = [None, "0.06", "0.05", "0.06", "0.04", None]

    words = []
    for n, (label, a, b, c) in enumerate(zip(labels, lane_a, lane_b, lane_c)):
        y0, y1 = 2.0 + 10.0 * n, 8.0 + 10.0 * n
        col0 = "Firm" if n == 0 else label
        words.append((0.0, y0, 30.0, y1, col0, 0, n, 0))
        header_a = "A" if n == 0 else a
        header_b = "B" if n == 0 else b
        header_c = "C" if n == 0 else c
        if header_a is not None:
            words.append((55.0, y0, 80.0, y1, header_a, 0, n, 1))
        if header_b is not None:
            words.append((105.0, y0, 130.0, y1, header_b, 0, n, 2))
        if header_c is not None:
            words.append((155.0, y0, 180.0, y1, header_c, 0, n, 3))
    # The orphan: 50pt right of lane C's centre (~167), 6th row (Memo, n=5).
    words.append((230.0, 52.0, 260.0, 58.0, "n.a.", 0, 5, 4))

    doc = fitz.open()
    page = doc.new_page()

    orig_get_text = page.get_text

    def _fake_get_text(*a, **k):
        if a and a[0] == "words":
            return words
        return orig_get_text(*a, **k)

    page.get_text = _fake_get_text

    def _fake_find_tables(*a, **k):
        # The text-strategy call (what reconstruct.py's fallback path makes)
        # sees the fixture's damaged table. A DEFAULT (lines-strategy) call --
        # made earlier in extract_structured's own detection pass, before it
        # ever reaches reconstruct_table_regions -- must see nothing, or that
        # earlier pass ships its own table_regions first and the code path
        # this fixture targets is never reached.
        if k.get("vertical_strategy") == "text":
            return _FakeResult([table])
        return _FakeResult([])

    page.find_tables = _fake_find_tables
    return doc, page


# ---------------------------------------------------------------------------
# GH-789
# ---------------------------------------------------------------------------


def test_destroyed_token_fallback_reports_its_own_orphan_drop():
    """The reachability case: the destroyed-token fallback IS reached (a
    rejection fires) and it DOES drop a word on that path -- both must be
    true for this test to say anything about the call site GH-789 fixes.
    """
    doc, page = _destroyed_grid_with_sparse_orphan_row()
    try:
        rejections: list[dict] = []
        orphan_drops: list[dict] = []
        out = reconstruct_table_regions(page, rejections=rejections, orphan_drops=orphan_drops)
    finally:
        doc.close()

    assert rejections, "setup: the text-strategy grid must actually be rejected"
    assert out, "setup: the fallback rowizer must actually ship a region"
    assert any(rec["word"] == "n.a." for rec in orphan_drops), (
        "the destroyed-token fallback's own rowize_from_word_list call dropped "
        f"'n.a.' with no event: {orphan_drops}"
    )


def test_orphan_drops_defaults_to_none_and_is_backward_compatible():
    """Every existing caller of ``reconstruct_table_regions`` that does not
    pass ``orphan_drops`` must keep working unchanged (GH-195's own
    ``rejections`` precedent for this call).
    """
    doc, page = _destroyed_grid_with_sparse_orphan_row()
    try:
        out = reconstruct_table_regions(page)
    finally:
        doc.close()
    assert out, "setup: the fallback rowizer must still ship a region without orphan_drops"


def test_no_spurious_orphan_events_on_a_clean_page():
    """Negative control: a page whose text-strategy grid is never rejected
    (no destroyed-token fallback reached) must not manufacture drop events.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 70), "prose only, no table on this page", fontsize=10)

    rejections: list[dict] = []
    orphan_drops: list[dict] = []
    try:
        reconstruct_table_regions(page, rejections=rejections, orphan_drops=orphan_drops)
    finally:
        doc.close()

    assert rejections == []
    assert orphan_drops == []


# ---------------------------------------------------------------------------
# GH-790
# ---------------------------------------------------------------------------


def test_extract_structured_standalone_does_not_raise_and_records_grid_rejections():
    """A harness calling ``extract_structured`` WITHOUT first going through
    ``_assess_page`` (the exact shape ``test_born_digital_aligned_runs.py``
    already exercises for other assertions) must not raise ``AttributeError``
    on ``_last_extraction_grid_rejections`` the first time a rejection fires
    -- the GH-195 sibling of the GH-418 gap ``_last_extraction_orphan_drops``
    was already fixed for in ``__init__``.
    """
    doc, page = _destroyed_grid_with_sparse_orphan_row()
    try:
        detector = BornDigitalDetector()
        # No _assess_page call anywhere above this -- __init__ alone must be
        # enough for the attribute to exist.
        detector.extract_structured(page)
        rejections = detector._last_extraction_grid_rejections
    finally:
        doc.close()

    assert rejections, (
        "extract_structured() must record the destroyed-token rejection even "
        "when called standalone, outside _assess_page"
    )


def test_extract_structured_standalone_no_drops_on_clean_page():
    """Negative control for the GH-790 fix: a standalone call on a page with
    no rejection leaves the list empty, not absent.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 70), "prose only, no table on this page", fontsize=10)
    try:
        detector = BornDigitalDetector()
        detector.extract_structured(page)
        assert detector._last_extraction_grid_rejections == []
    finally:
        doc.close()
