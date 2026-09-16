"""GH-152 TICKET-A2: consume the gutter detector at the SECOND merging rung.

``rowize_from_word_list`` (fixed in ``test_gh152_column_aware_rowize.py``) is
only reached when the text-strategy grid is REJECTED for destroying a
numeric token. On a clean booktabs-style page with no ruling lines,
PyMuPDF's own unclipped ``page.find_tables(strategy="text")`` call
(``reconstruct_table_regions``) merges two side-by-side tables into ONE
grid that does not trip that rejection -- every token survives character-
intact, just glued into the wrong table -- so the rowizer fix is never
consulted. This is the "measured end-to-end no-op" the GH-152 plan's
TICKET-A1 retarget ruling names.

Fix: reuse the SAME gutter detector and label-column guard
(``_detect_column_gutter`` / ``_has_row_labels``) to clip
``page.find_tables()`` per band instead of calling it unclipped over the
whole page. Any doubt -- no gutter, one side has no label column, or either
band's clipped call comes back empty -- falls through to today's single
unclipped call, unchanged.

Per the GH-152 plan's TICKET-A1 ruling: "any content-loss claim must go
through extract_structured or process() -- never an isolated rung." The
headline and false-positive tests here therefore go through
``BornDigitalDetector().extract_structured()``, not through
``reconstruct_table_regions`` directly.

Hermetic: PDFs are built in-process with ``fitz`` (``insert_text``), no
corpus content, no provider.
"""

from __future__ import annotations

import re

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")


def _table_count(md: str) -> int:
    """Count separator ROWS (one per distinct markdown table), not
    occurrences of the substring within a row -- a single row like
    ``| --- | --- | --- |`` contains ``"| --- "`` once per column."""
    return sum(1 for line in md.splitlines() if re.fullmatch(r"\|(\s*-{3,}\s*\|)+", line.strip()))


CHAR_W = 6.0
FS = 10


def _assert_worktree_source():
    import socr

    assert "/socr-152/" in socr.__file__, (
        f"expected the socr-152 worktree's source, got {socr.__file__}"
    )


def _row_text(cells: list[tuple[float, str]], x0: float) -> str:
    """A single courier-spaced string whose cells land near their target x.

    One ``insert_text`` call per (band, row) -- not per word -- so
    ``get_text('words')`` groups a row's words under one ``(block_no,
    line_no)``, the grouping ``_median_word_gap`` (this ticket's gutter
    yardstick) and ``born_digital.py``'s own ``_median_word_space_width``
    both rely on. Word-by-word ``insert_text`` calls each land on a
    distinct PyMuPDF line, which starves that measurement -- an artefact of
    single-word insertion, not of anything a real (LaTeX-produced) PDF's
    content stream does.
    """
    parts: list[str] = []
    cursor_chars = 0
    for target_x, text in cells:
        target_chars = round((target_x - x0) / CHAR_W)
        pad = max(1, target_chars - cursor_chars) if parts else max(0, target_chars)
        parts.append(" " * pad + text)
        cursor_chars = target_chars + len(text)
    return "".join(parts)


def _two_tables_pdf(n_rows: int = 6) -> fitz.Document:
    """Booktabs-style (no ruling lines): left table 2 numeric lanes, right
    table 2 numeric lanes, offset far enough to be a genuine gutter."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y0, row_h = 100.0, 20.0
    left_x0, right_x0 = 60.0, 350.0
    page.insert_text(
        (left_x0, y0),
        _row_text([(60, "Label"), (140, "Mean"), (190, "SD")], left_x0),
        fontsize=FS,
        fontname="cour",
    )
    page.insert_text(
        (right_x0, y0),
        _row_text([(350, "Label"), (420, "Corr"), (460, "Guide")], right_x0),
        fontsize=FS,
        fontname="cour",
    )
    for i in range(n_rows):
        y = y0 + row_h * (i + 1)
        page.insert_text(
            (left_x0, y),
            _row_text([(60, f"LeftLab{i}"), (140, f"{i}.11"), (190, f"{i}.22")], left_x0),
            fontsize=FS,
            fontname="cour",
        )
        page.insert_text(
            (right_x0, y),
            _row_text([(350, f"RightLab{i}"), (420, f"{i}.33"), (460, f"{i}.44")], right_x0),
            fontsize=FS,
            fontname="cour",
        )
    return doc


def _wide_single_table_pdf(n_rows: int = 6) -> fitz.Document:
    """ONE table, no second label column -- a large label-to-value gap that
    must NOT be read as a two-table gutter (the false-positive this ticket's
    unit-level guard already defeats; here re-measured at the find_tables
    rung, where a wrongly-split rect could itself misattribute)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y0, row_h = 100.0, 20.0
    x0 = 60.0
    page.insert_text(
        (x0, y0),
        _row_text([(60, "Label"), (300, "Alpha"), (360, "Beta"), (420, "Gamma")], x0),
        fontsize=FS,
        fontname="cour",
    )
    for i in range(n_rows):
        y = y0 + row_h * (i + 1)
        page.insert_text(
            (x0, y),
            _row_text(
                [
                    (60, f"WideRow{i}"),
                    (300, f"{i}.10"),
                    (360, f"{i}.20"),
                    (420, f"{i}.30"),
                ],
                x0,
            ),
            fontsize=FS,
            fontname="cour",
        )
    return doc


class TestBandClippedFindTablesRestoresAttribution:
    """The headline end-to-end defect: through the installed package."""

    def test_main_reproduces_the_merge(self):
        """Sanity check the fixture on the pre-fix code path (gutter
        detection disabled) still merges two tables into one, as GH-152
        describes. If this stops reproducing, the fixture proves nothing."""
        from socr.core.born_digital import BornDigitalDetector
        from socr.tables import reconstruct

        _assert_worktree_source()
        doc = _two_tables_pdf()
        page = doc[0]

        monkey = pytest.MonkeyPatch()
        monkey.setattr(reconstruct, "_detect_column_gutter", lambda _words: None)
        try:
            md = BornDigitalDetector().extract_structured(page)
        finally:
            monkey.undo()

        # One merged 6-column grid, not two separate 3-column tables.
        assert _table_count(md) == 1, md
        assert "RightLab0" in md and "LeftLab0" in md

    def test_two_tables_emitted_separately_and_correctly_attributed(self):
        from socr.core.born_digital import BornDigitalDetector

        _assert_worktree_source()
        doc = _two_tables_pdf()
        page = doc[0]

        md = BornDigitalDetector().extract_structured(page)

        assert _table_count(md) == 2, f"expected two distinct tables:\n{md}"
        for i in range(6):
            assert f"LeftLab{i}" in md
            assert f"RightLab{i}" in md

        # No row-level misattribution: a LeftLab row's markdown line must
        # not also carry a value that belongs to a DIFFERENT right-table
        # row index.
        lines = [ln for ln in md.splitlines() if ln.strip().startswith("|") and "---" not in ln]
        for line in lines:
            if "LeftLab" not in line:
                continue
            idx = re.search(r"LeftLab(\d+)", line).group(1)
            for other in range(6):
                if str(other) == idx:
                    continue
                assert f"{other}.33" not in line, line
                assert f"{other}.44" not in line, line


class TestWideSingleTableFalsePositiveGuardHoldsAtThisRung:
    def test_wide_single_table_stays_one_region(self):
        from socr.core.born_digital import BornDigitalDetector

        _assert_worktree_source()
        doc = _wide_single_table_pdf()
        page = doc[0]

        md = BornDigitalDetector().extract_structured(page)
        assert _table_count(md) == 1, f"a genuine single table must not split:\n{md}"
        for i in range(6):
            assert f"WideRow{i}" in md
            assert f"{i}.10" in md and f"{i}.20" in md and f"{i}.30" in md


class TestSingleColumnByteIdentityAtThisRung:
    def test_wide_single_table_byte_identical_with_gutter_detection_forced_off(self):
        """Difference-pin, not an absolute string pin: the same page, run
        once with the gutter detector forced to always return None (today's
        pre-GH-152 behaviour) and once unpatched, must produce identical
        output for a page with no genuine second table."""
        from socr.core.born_digital import BornDigitalDetector
        from socr.tables import reconstruct

        _assert_worktree_source()

        def _render():
            doc = _wide_single_table_pdf()
            return BornDigitalDetector().extract_structured(doc[0])

        fixed_md = _render()

        monkey = pytest.MonkeyPatch()
        monkey.setattr(reconstruct, "_detect_column_gutter", lambda _words: None)
        try:
            disabled_md = _render()
        finally:
            monkey.undo()

        assert fixed_md == disabled_md
