"""GH-418 step 2: capture the orphan word into the trailing column.

Step 1 (docs/log/2026-09-16_418-event.md) surfaced the drop -- no grid, column
count or markdown byte changed. This ticket changes the grid: an orphan word
(further than the snap radius from every lane) is now CAPTURED into GH-461's
trailing ``orphan_marginals`` column instead of silently deleted, **iff its
own row already populates >= 2 numeric lanes** -- the same predicate
``_looks_tabular`` (``reconstruct.py:978``) uses to call a row a "data row",
applied one step earlier inside ``_rowize_segment``. Rows below that gate (a
sparse row, a header row) still drop the word, and step 1's ``orphan_drops``
sideband still records it -- that residual is the panel ruling's own
requirement, not a gap (docs/log/2026-09-16_418-design.md, "Panel ruling").

Reused fixtures, per the ruling ("retarget, do not delete"):
  * ``_four_data_lanes`` from ``test_gh342_stub_promotion_runaway.py`` --
    every row there populates all 4 numeric lanes, so its markers are the
    "full data row" case this ticket captures.
  * the §3.4 lane-aligned-reference fixture, built directly against
    ``_rowize_segment`` (the same private entry point the design note's own
    measurement used) -- the worst case measured: a reference row whose year
    and page number already occupy two lanes, so it is a "data row" by the
    SAME predicate this ticket reuses, and its trailing prose gets captured
    rather than corrupted into a lane cell.

Every test here fails on the pre-step-2 code (the word stays dropped, or the
``elif orphan_drops is not None: orphan_drops.append(...)`` branch fires
unconditionally) -- confirmed by the mutation run in
``docs/log/2026-09-16_418-capture.md``, not merely assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from socr.tables import reconstruct as r  # noqa: E402
from socr.tables.reconstruct import rowize_from_word_list  # noqa: E402
from test_gh342_stub_promotion_runaway import _four_data_lanes  # noqa: E402


def _w(x: float, y: float, text: str, width: float = 26.0, h: float = 10.0) -> tuple:
    return (x, y, x + width, y + h, text, 0, 0, 0)


def _grid(regions: list) -> list[list[str]]:
    assert regions, "fixture must produce a table region"
    return [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in regions[0][1].splitlines()
        if line.lstrip().startswith("|") and "---" not in line
    ]


# ---------------------------------------------------------------------------
# 1. A full data row's qualifiers are captured, never into a lane or label.
# ---------------------------------------------------------------------------


def test_qualifiers_are_captured_on_a_full_data_row() -> None:
    """n.a., a dagger footnote, and a right-of-last-lane asterisk all appear
    in the emitted grid, in a column that is neither a numeric lane nor the
    row label -- on a row that already populates all 4 numeric lanes."""
    for marker in ("n.a.", "†"):
        grid = _grid(rowize_from_word_list(_four_data_lanes(markers=True, marker=marker)))
        marker_rows = [row for row in grid if marker in row]
        assert marker_rows, f"{marker!r} was dropped instead of captured: {grid}"
        for row in marker_rows:
            assert row[-1] == marker, f"{marker!r} landed outside the trailing column: {row}"
            # Never in a lane cell or the label -- the label and lane cells
            # are untouched by the marker.
            assert not row[0].strip().endswith(marker), f"{marker!r} corrupted the label: {row}"


def test_right_of_last_lane_asterisk_is_captured() -> None:
    words: list = []
    y = 100.0
    lanes = [80.0, 200.0, 320.0, 440.0]
    for row_i in range(4):
        words.append(_w(40.0, y, f"Row{row_i}"))
        for c, x in enumerate(lanes):
            words.append(_w(x, y, f"{row_i}{c}.5"))
        if row_i < 3:
            words.append(_w(500.0, y, "*"))  # 60pt right of the last lane (440)
        y += 16.0

    grid = _grid(rowize_from_word_list(words))
    marker_rows = [row for row in grid if "*" in row]
    assert len(marker_rows) == 3, f"expected 3 rows carrying the asterisk: {grid}"
    for row in marker_rows:
        assert row[-1] == "*", f"the asterisk did not land in its own trailing column: {row}"
    for row_i in range(4):
        for c in range(4):
            assert f"{row_i}{c}.5" in {tok for row in grid for cell in row for tok in cell.split()}


# ---------------------------------------------------------------------------
# 2. Residual: a row below the >= 2 numeric-lane gate still drops, still
#    emits the step-1 event.
# ---------------------------------------------------------------------------


def test_single_numeric_lane_row_still_drops_and_still_emits_the_event() -> None:
    """A row with only 1 populated numeric lane stays below the capture
    gate -- the orphan is still deleted, and step 1's ``orphan_drops`` still
    names it. The table as a whole still ships (3 of 4 rows are full data
    rows, clearing ``_looks_tabular``'s >= 0.5 majority)."""
    words: list = []
    lanes = [100.0, 220.0, 340.0, 460.0]
    y = 100.0
    for row_i in range(3):
        words.append(_w(36.0, y, f"Row{row_i}"))
        for c, x in enumerate(lanes):
            words.append(_w(x, y, f"{row_i}{c}.5"))
        y += 20.0
    # Sparse row: only ONE numeric lane populated, plus the orphan.
    words.append(_w(36.0, y, "Sparse"))
    words.append(_w(lanes[0], y, "9.9"))
    words.append(_w(160.0, y, "n.a."))

    drops: list[dict] = []
    regions = rowize_from_word_list(words, orphan_drops=drops)
    grid = _grid(regions)

    assert drops == [{"word": "n.a.", "x": 160.0, "y": 160.0}], drops
    sparse_row = [row for row in grid if row[0] == "Sparse"][0]
    assert "n.a." not in sparse_row, f"the residual drop must not be captured: {sparse_row}"


def test_header_row_with_no_numerics_still_drops_and_still_emits_the_event() -> None:
    """A row with ZERO numeric lanes (a header row inside the same segment,
    not `_prepend_header_band`'s separate multi-line-header path) is also
    below the >= 2 gate -- same residual, same event."""
    words: list = []
    lanes = [100.0, 220.0, 340.0, 460.0]
    y = 80.0
    words.append(_w(36.0, y, "Model"))
    for x, text in zip(lanes, ["Est", "SE", "Tval", "Pval"]):
        words.append(_w(x, y, text))
    words.append(_w(160.0, y, "(pct)"))
    y += 20.0
    for row_i in range(3):
        words.append(_w(36.0, y, f"Row{row_i}"))
        for c, x in enumerate(lanes):
            words.append(_w(x, y, f"{row_i}{c}.5"))
        y += 20.0

    drops: list[dict] = []
    regions = rowize_from_word_list(words, orphan_drops=drops)
    grid = _grid(regions)

    assert drops == [{"word": "(pct)", "x": 160.0, "y": 80.0}], drops
    header_row = grid[0]
    assert "(pct)" not in header_row, f"the residual drop must not be captured: {header_row}"


# ---------------------------------------------------------------------------
# 3. The §3.4 worst case: a lane-aligned reference row.
# ---------------------------------------------------------------------------


def test_lane_aligned_reference_row_recovers_prose_not_corruption() -> None:
    """Design note §3.4: a reference row engineered so its year and page
    number already occupy two lanes scores as a "data row" under the SAME
    ``>= 2`` predicate this ticket reuses -- there is no signal that
    separates it from a genuine table row (measured, not assumed). The
    fix's failure mode here is *visible misplacement* (prose recovered into
    the trailing column), not *silent deletion* or *value corruption*: no
    numeric cell is touched, and the verdict (this block ships as a table)
    is the SAME with and without the capture -- only the content differs.

    Fama's row uses the actual word that surfaced the real regression this
    ticket found (see ``test_a_word_that_would_look_like_a_running_head_is_
    refused_not_the_whole_row`` below): "Journal" makes the row's joined
    text match ``_RUNHEAD_RE``, so it is refused and recorded as a drop
    instead of captured -- "Finance" alone does not, and is captured
    normally. That refusal is exercised here inline rather than assumed.
    """
    words: list = []
    lanes = [130.0, 200.0, 240.0]
    y = 100.0
    refs = [
        ("Fama", "1992", "427", "12", "Journal", "Finance"),
        ("Smith", "1998", "512", "5", "Rev", "Econ"),
        ("Jones", "2001", "88", "9", "J", "Pol"),
    ]
    for author, year, page_no, vol, w1, w2 in refs:
        words.append(_w(60.0, y, author))
        words.append(_w(lanes[0], y, year))
        words.append(_w(lanes[1], y, page_no))
        words.append(_w(lanes[2], y, vol))
        words.append(_w(300.0, y, w1))
        words.append(_w(340.0, y, w2))
        y += 16.0

    rows_by_y: dict[int, list] = {}
    for w in words:
        rows_by_y.setdefault(round(w[1]), []).append(w)
    seg_ys = sorted(rows_by_y.keys())

    drops: list[dict] = []
    captured = r._rowize_segment(words, seg_ys, rows_by_y, orphan_drops=drops)
    baseline_verdict = r._rowize_segment(words, seg_ys, rows_by_y)  # orphan_drops=None

    assert captured is not None, "the row-shape ships as a table regardless of this ticket"
    assert baseline_verdict is not None, "capture must not change whether the block ships"
    assert captured[0] == baseline_verdict[0], (
        "wiring orphan_drops must not itself change the captured grid"
    )

    grid, *_ = captured
    assert len(grid) == len(refs), f"a row was lost: {grid}"
    for author, year, page_no, vol, w1, w2 in refs:
        row = [r for r in grid if r[0] == author][0]
        assert row[1:4] == [year, page_no, vol], f"a real value moved lanes: {row}"
        if author == "Fama":
            assert row[4] == "Finance", f"Finance should still be captured: {row}"
            assert {"word": "Journal", "x": 300.0, "y": 100.0} in drops, (
                f"Journal must be refused and recorded, not silently lost: {drops}"
            )
        else:
            assert row[4] == f"{w1} {w2}", f"prose not captured in its own column: {row}"
    assert len(drops) == 1, f"only the runhead-colliding word should be refused: {drops}"


def test_a_word_that_would_look_like_a_running_head_is_refused_not_the_whole_row() -> None:
    """The regression this ticket found: capturing a word that makes the
    row's joined text match ``_clean_grid``'s ``_RUNHEAD_RE`` used to get
    the ENTIRE row deleted by ``_clean_grid``'s leading-runhead stripper --
    not just the word, the row's real numeric values too. That is strictly
    worse than the pre-ticket behaviour (word-only drop) and the exact
    prose-page regression class #342's two prior attempts were rejected
    for. Routed through the FULL pipeline (``rowize_from_word_list``, which
    calls ``_clean_grid``), not ``_rowize_segment`` directly, because the
    loss only happens after ``_clean_grid`` runs.

    Five rows so the runhead-colliding row (Fama, row 0) is a LEADING row
    once "Smith" et al are stripped of anything -- the exact shape
    ``_clean_grid``'s ``while g and _is_runhead(g[0])`` peels.
    """
    words: list = []
    lanes = [130.0, 200.0, 240.0]
    y = 100.0
    refs = [
        ("Fama", "1992", "427", "12", "Journal", "Finance"),
        ("Smith", "1998", "512", "5", "Rev", "Econ"),
        ("Jones", "2001", "88", "9", "J", "Pol"),
        ("Lucas", "1976", "19", "1", "CarnegieR", "Series"),
        ("Sims", "1980", "1", "48", "Econometrica", "Vol"),
    ]
    for author, year, page_no, vol, w1, w2 in refs:
        words.append(_w(60.0, y, author))
        words.append(_w(lanes[0], y, year))
        words.append(_w(lanes[1], y, page_no))
        words.append(_w(lanes[2], y, vol))
        words.append(_w(300.0, y, w1))
        words.append(_w(340.0, y, w2))
        y += 16.0

    drops: list[dict] = []
    full_grid = _grid(rowize_from_word_list(words, orphan_drops=drops))
    # Drop the markdown header row (blank cells, emitted unconditionally by
    # `_grid_to_markdown`) -- it is not one of the reference rows.
    grid = [row for row in full_grid if row[0]]

    assert len(grid) == len(refs), (
        f"every reference row must survive -- a captured word must never "
        f"delete the whole row: {grid}"
    )
    present_authors = {row[0] for row in grid}
    assert present_authors == {a for a, *_ in refs}, f"a row was lost: {grid}"

    fama = [row for row in grid if row[0] == "Fama"][0]
    assert fama[1:4] == ["1992", "427", "12"], f"Fama's real values must survive intact: {fama}"
    assert "Finance" in fama[-1], f"the non-colliding word must still be captured: {fama}"
    assert "Journal" not in fama[-1], (
        f"the runhead-colliding word must be refused, not captured: {fama}"
    )

    sims = [row for row in grid if row[0] == "Sims"][0]
    assert sims[1:4] == ["1980", "1", "48"], f"Sims's real values must survive intact: {sims}"
    assert "Vol" in sims[-1], f"the non-colliding word must still be captured: {sims}"
    assert "Econometrica" not in sims[-1], (
        f"the runhead-colliding word must be refused, not captured: {sims}"
    )

    refused_words = {d["word"] for d in drops}
    assert refused_words == {"Journal", "Econometrica"}, (
        f"exactly the runhead-colliding words must be recorded as drops, "
        f"not silently disappear: {drops}"
    )


# ---------------------------------------------------------------------------
# 4. Pinned as a DIFFERENCE: same geometry, same data-lane count, same
#    numeric multiset, with and without markers.
# ---------------------------------------------------------------------------


def test_capture_pins_the_same_data_lane_count_and_numeric_multiset() -> None:
    without = _grid(rowize_from_word_list(_four_data_lanes(markers=False)))
    with_markers = _grid(rowize_from_word_list(_four_data_lanes(markers=True)))

    data_width = len(without[0])
    assert [row[:data_width] for row in with_markers] == without, (
        "the data-lane cells (label + numeric lanes) must be identical with "
        f"and without markers: {with_markers} != {without}"
    )

    def numeric_multiset(grid: list[list[str]]) -> list[str]:
        return sorted(
            tok
            for row in grid
            for cell in row[:data_width]
            for tok in cell.split()
            if reconstruct_numeric_token(tok)
        )

    def reconstruct_numeric_token(tok: str) -> bool:
        return bool(r._NUM_TOKEN_RE.match(tok) and r._NUMERIC_RE.search(tok))

    assert numeric_multiset(with_markers) == numeric_multiset(without), (
        "the marker must not change a single numeric value's lane"
    )
