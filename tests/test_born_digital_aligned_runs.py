"""Differential tests for GH-592 (C1): baseline-aligned two-column runs.

A tab-aligned two-column list (e.g. an attendee roster typeset as a "Mr."/
"Ms." column and a separate name column) previously native-extracted as one
column's lines in full, then the other's, because ``page.get_text("text")``
walks blocks in their original order with no notion that two x-disjoint
blocks form a single reading-order run. See
``docs/log/2026-09-06_C1-aligned-runs.md`` for the geometry measurement this
fix is based on (the real Fed 1989-11-14 minutes p.1 fixture, plus a
synthetic two-column-journal negative control).

The two fixtures below are built inline with ``fitz`` rather than shipped as
files, following this test module's existing convention
(``tests/test_born_digital.py``). No existing two-column-journal PDF fixture
was found anywhere in this repo (see the decision log) -- the negative
control here is synthesized to have the same "two blocks of lines, laid out
side by side" shape as the positive case, but with a gutter many times wider
than a single word space, which is the actual discriminator the fix relies
on.
"""

import fitz
import pytest

from socr.core.born_digital import (
    BornDigitalDetector,
    _assemble_prose_with_aligned_runs,
)


def _build_attendee_list_page() -> fitz.Page:
    """A prose page whose attendee roster is a genuine aligned two-column run.

    Built to mirror the real Fed 1989-11-14 minutes p.1 defect: a "Mr."/"Ms."
    label column and a name column, laid out as two separate text blocks
    (via ``insert_textbox``, one per column) with a horizontal gap on the
    order of one word space -- not a column gutter.
    """
    doc = fitz.open()
    page = doc.new_page()
    y0 = 72
    page.insert_text(
        (72, y0),
        "Minutes of the Federal Open Market Committee meeting held on",
        fontsize=10,
        fontname="helv",
    )
    page.insert_text((72, y0 + 14), "November 14, 1989", fontsize=10, fontname="helv")
    page.insert_text((72, y0 + 34), "PRESENT:", fontsize=10, fontname="helv")

    rows = [
        ("Mr.", "Greenspan, Chairman"),
        ("Mr.", "Corrigan, Vice Chairman"),
        ("Mr.", "Angell"),
        ("Mr.", "Black"),
        ("Ms.", "Seger"),
    ]
    left_x = 90
    label_w = fitz.get_text_length("Mr.", fontname="helv", fontsize=10)
    space_w = fitz.get_text_length(" ", fontname="helv", fontsize=10)
    gap = 1.2 * space_w  # comparable to one word space -- the defect's shape
    right_x = left_x + label_w + gap
    row_start_y = y0 + 50

    left_rect = fitz.Rect(left_x, row_start_y - 8, left_x + 40, row_start_y + 140)
    right_rect = fitz.Rect(right_x, row_start_y - 8, 560, row_start_y + 140)
    page.insert_textbox(
        left_rect, "\n".join(label for label, _ in rows), fontsize=10, fontname="helv"
    )
    page.insert_textbox(
        right_rect, "\n".join(name for _, name in rows), fontsize=10, fontname="helv"
    )
    return page


def _build_uniform_gap_bijection_page(gap_multiplier: float) -> fitz.Page:
    """A synthetic bijection of five rows with a precisely controlled gap.

    Unlike the two fixtures above, every left-column row here has the SAME
    text ("XXXX"), so every row's right edge is identical and the gap to the
    (also fixed-x) right column is constant across the whole run -- letting
    ``gap_multiplier`` pin an exact, reproducible multiple of the page's own
    measured word-space width, rather than an approximate one. Used to pin
    both sides of ``ALIGNED_RUN_GAP_MAX_WORD_SPACES`` directly against real
    measured body-text gutters (see the decision log): the narrowest
    real two-column journal gutter measured was 2.30x a word space
    (Fama-French 1997 JFE p.2), and the widest measured aligned-run gap was
    1.76x (Fed 1989 fixture) -- so 1.5x must merge and 2.3x must not.

    Right-column length variance (four short entries, one long outlier)
    deliberately mirrors the Fed fixture's honorific->name column shape
    (mostly bare names, one "Corrigan, Vice Chairman" outlier) rather than
    the near-uniform-length lines an earlier round used here -- those
    accidentally tripped RIGHT_BLOCK_FILL_TOLERANCE_WORD_WIDTHS /
    MEASURE_FILL_SHARE_MAX (a column of near-identical-length independent
    values reads as "wrapped body prose" to a purely width-based fill
    check), which is a fixture-realism gap, not a genuine positive case the
    guard is meant to reject -- see the log's "Round 4" notes.
    """
    doc = fitz.open()
    page = doc.new_page()
    y0 = 72
    page.insert_text(
        (72, y0),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
        fontname="helv",
    )

    label = "XXXX"
    left_lines = [label] * 5
    right_lines = [
        "Angellton",
        "Guffeyman",
        "Keehnston",
        "Segerwood",
        "Corrigan, Vice Chairman of the Committee",
    ]
    left_x = 90
    label_w = fitz.get_text_length(label, fontname="helv", fontsize=10)
    space_w = fitz.get_text_length(" ", fontname="helv", fontsize=10)
    gap = gap_multiplier * space_w
    right_x = left_x + label_w + gap
    row_start_y = y0 + 40

    left_rect = fitz.Rect(left_x, row_start_y - 8, left_x + 40, row_start_y + 140)
    right_rect = fitz.Rect(right_x, row_start_y - 8, right_x + 300, row_start_y + 140)
    page.insert_textbox(left_rect, "\n".join(left_lines), fontsize=10, fontname="helv")
    page.insert_textbox(right_rect, "\n".join(right_lines), fontsize=10, fontname="helv")
    return page


def _build_two_column_journal_page() -> fitz.Page:
    """A genuinely two-column prose page: full paragraphs, wide gutter.

    Shares the "two blocks of lines, side by side" shape with the attendee
    fixture above (so a naive fix that merges on shape alone would wrongly
    fire here too), but the gutter between columns is a page-layout gutter,
    not a tab stop -- many times wider than a word space. This must NEVER
    merge.
    """
    doc = fitz.open()
    page = doc.new_page()
    left_para = [
        "The committee reviewed economic conditions across several",
        "regions and noted continued moderation in price pressures",
        "alongside steady employment growth in most sectors of the",
        "economy, with particular strength in services activity.",
    ]
    right_para = [
        "Members discussed the outlook for monetary policy over the",
        "coming quarters, emphasizing the need for continued vigilance",
        "regarding inflation expectations while supporting the ongoing",
        "expansion through appropriately calibrated policy settings.",
    ]
    left_x, right_x = 72, 320
    y = 72
    for line in left_para:
        page.insert_text((left_x, y), line, fontsize=10, fontname="helv")
        y += 14
    y = 72
    for line in right_para:
        page.insert_text((right_x, y), line, fontsize=10, fontname="helv")
        y += 14
    return page


class TestAlignedRunMerge:
    def test_attendee_list_merges_label_and_name_columns(self):
        page = _build_attendee_list_page()
        out = BornDigitalDetector().extract_structured(page)

        assert "Mr. Greenspan, Chairman" in out
        assert "Mr. Corrigan, Vice Chairman" in out
        assert "Mr. Angell" in out
        assert "Mr. Black" in out
        assert "Ms. Seger" in out

        bare_honorifics = sum(1 for line in out.splitlines() if line.strip() in ("Mr.", "Ms."))
        assert bare_honorifics == 0, (
            f"expected zero bare honorific lines after merge, got {bare_honorifics}: {out!r}"
        )

    def test_attendee_list_assembler_returns_merged_rows_directly(self):
        # Narrower unit check on the assembler itself, independent of the
        # rest of extract_structured's routing (table detection, links).
        page = _build_attendee_list_page()
        assembled = _assemble_prose_with_aligned_runs(page)
        assert assembled is not None
        assert "Mr. Greenspan, Chairman" in assembled
        assert "Ms. Seger" in assembled


class TestGenuineTwoColumnNeverMerges:
    def test_wide_gutter_assembler_declines(self):
        page = _build_two_column_journal_page()
        assert _assemble_prose_with_aligned_runs(page) is None

    def test_wide_gutter_output_is_byte_identical_with_assembler_forced_off(self, monkeypatch):
        page = _build_two_column_journal_page()

        with_assembler = BornDigitalDetector().extract_structured(page)

        monkeypatch.setattr(
            "socr.core.born_digital._assemble_prose_with_aligned_runs",
            lambda _page: None,
        )
        without_assembler = BornDigitalDetector().extract_structured(page)

        assert with_assembler == without_assembler, (
            "a genuine two-column prose page must be byte-identical whether "
            "the aligned-run assembler runs or is forced off"
        )


def _build_astra_independent_columns_page() -> fitz.Page:
    """Astra's (Codex reviewer) false-merge repro for GH-592, verbatim shape.

    Two independent, unrelated prose sentences in adjacent narrow columns:
    a full two-row bijection, trimmed gaps (3pt and 9pt) comfortably under
    ``ALIGNED_RUN_GAP_MAX_WORD_SPACES`` x a 6pt word space, so the gap-and-
    -bijection checks alone accept this and previously produced
    "We stopped the trial. We approved the drug." -- every word preserved,
    meaning destroyed. This is what ``LABEL_COLUMN_WIDTH_SHARE`` exists to
    catch: neither column here is a narrow label for the other; both are
    close to full column width (ratio ~1.0, see the decision log).
    """
    doc = fitz.open()
    page = doc.new_page()
    left_rect = fitz.Rect(72, 100, 200, 200)
    right_rect = fitz.Rect(201, 100, 400, 200)
    page.insert_textbox(
        left_rect, "We stopped the trial.\nThe drug was unsafe.", fontsize=10, fontname="cour"
    )
    page.insert_textbox(
        right_rect, "We approved the drug.\nThe trial was sound.", fontsize=10, fontname="cour"
    )
    return page


class TestIndependentColumnsNeverMerge:
    """GH-592 Astra finding: gap + bijection alone is not enough evidence."""

    def test_astra_repro_assembler_declines(self):
        page = _build_astra_independent_columns_page()
        assert _assemble_prose_with_aligned_runs(page) is None

    def test_astra_repro_output_is_byte_identical_with_assembler_forced_off(self, monkeypatch):
        page = _build_astra_independent_columns_page()

        with_assembler = BornDigitalDetector().extract_structured(page)

        monkeypatch.setattr(
            "socr.core.born_digital._assemble_prose_with_aligned_runs",
            lambda _page: None,
        )
        without_assembler = BornDigitalDetector().extract_structured(page)

        assert with_assembler == without_assembler, (
            "two independent prose columns must never be merged into one "
            "line, even when gap and bijection alone would accept them"
        )


def _build_astra_residual_narrow_label_page() -> fitz.Page:
    """GH-592 Astra RESIDUAL finding: width asymmetry alone is not enough.

    A short repeated left label ("Note" x4) beside four INDEPENDENT,
    unrelated, very unequal-length sentences (short "Ask." next to a much
    longer one). This passes the bijection, the gap check, AND
    ``LABEL_COLUMN_WIDTH_SHARE`` (the left block genuinely is much narrower
    -- ratio ~0.16, well under 0.65 -- because "Note" really is short
    compared to these sentences) -- narrow-left/wide-right identifies "the
    left block is narrow", not "the left block is a genuine label column".

    Astra's literal rect coordinates (72,100,105,180) / (115,100,400,180)
    do not reproduce a merge here: PyMuPDF's real Courier glyph metrics
    leave the left "Note" box's own trimmed text ending well short of its
    box edge (Courier "Note" at 10pt is 24pt wide, ending at x=96, not the
    box's x=105), so the true trimmed gap to a right column starting at
    x=115 is 19pt (~3.17x this page's 6pt word space) -- already above
    ``ALIGNED_RUN_GAP_MAX_WORD_SPACES`` (2.0x) on its own, before either
    width check runs. The right column's start x is adjusted to 100 here
    (trimmed gap 4pt, ~0.67x word space) to actually reproduce the
    near-passing shape Astra intended and measure the residual honestly --
    see the decision log's "Round 4" section for the measured numbers this
    is based on.

    KNOWN, ACCEPTED RESIDUAL (see the log): this specific case's right-block
    fill share measures at exactly 0.50 -- not STRICTLY greater than
    ``MEASURE_FILL_SHARE_MAX`` (0.5) -- so it is NOT caught by the
    wrapped-body-prose guard either, and still merges. This is deliberate,
    not a bug: four short, wildly unequal, independent sentences do not
    reliably look like either "a label's value list" or "a wrapped
    paragraph" on pure geometry, and loosening the fill-share threshold to
    catch this one contrived case would put the Fed positive case at risk
    with essentially no real-world benefit (four-word attendee-style values
    of near-uniform length are a real production shape; four wildly
    unequal one-line non-sequiturs beside a repeated label are not a shape
    this detector has evidence any real document uses). The test below
    documents CURRENT behaviour so a future change to it is measurable.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(
        fitz.Rect(72, 100, 105, 180), "Note\nNote\nNote\nNote", fontsize=10, fontname="cour"
    )
    sentences = [
        "Fix the leak.",
        "Ping the vendor about invoices today.",
        "Ask.",
        "Review the quarterly compliance report drafts.",
    ]
    page.insert_textbox(
        fitz.Rect(100, 100, 400, 180), "\n".join(sentences), fontsize=10, fontname="cour"
    )
    return page


def _build_astra_residual_wrapped_paragraph_page() -> fitz.Page:
    """The real-body-text sibling of the residual above: this one MUST decline.

    Same left "Note" x4 label column and left/right geometry as the residual
    fixture, but the right column is one genuine wrapped paragraph (a real
    Fed-minutes sentence, hard-wrapped to exactly 4 lines) instead of four
    independent sentences. A wrapped paragraph's non-final lines each run up
    against the column's own measure (greedy line-wrap fills every line
    until the next word would overflow) -- fill share here measures 0.75,
    comfortably above ``MEASURE_FILL_SHARE_MAX`` (0.5), so this is caught
    even though its ``LABEL_COLUMN_WIDTH_SHARE`` (~0.067) would otherwise
    look like an even more convincing label/value pair than the residual
    case above.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(
        fitz.Rect(72, 100, 105, 240), "Note\nNote\nNote\nNote", fontsize=10, fontname="cour"
    )
    lines = [
        "The Manager of the System Open Market Account reported on",
        "developments in foreign exchange markets during the period and",
        "on System open market transactions in foreign currencies over it",
        "for the Committee.",
    ]
    page.insert_textbox(
        fitz.Rect(100, 100, 520, 240), "\n".join(lines), fontsize=10, fontname="cour"
    )
    return page


class TestNarrowLabelWrappedProseDiscriminator:
    """GH-592 Astra RESIDUAL finding: width asymmetry alone is not enough."""

    def test_astra_residual_still_merges_documented_known_gap(self):
        # KNOWN RESIDUAL, accepted -- see _build_astra_residual_narrow_label_page's
        # docstring and docs/log/2026-09-06_C1-aligned-runs.md's "Round 4"
        # section. If this assertion ever starts failing because the
        # assembler now declines, that is an IMPROVEMENT: update this test
        # to assert None and remove the residual note from the log.
        page = _build_astra_residual_narrow_label_page()
        assembled = _assemble_prose_with_aligned_runs(page)
        assert assembled is not None
        assert assembled == (
            "Note Fix the leak.\n"
            "Note Ping the vendor about invoices today.\n"
            "Note Ask.\n"
            "Note Review the quarterly compliance report drafts."
        )

    def test_wrapped_paragraph_beside_narrow_label_never_merges(self):
        page = _build_astra_residual_wrapped_paragraph_page()
        assert _assemble_prose_with_aligned_runs(page) is None


def _build_uniform_width_ratio_page(
    width_ratio: float, right_text: str = "one two threefour"
) -> fitz.Page:
    """A bijection with a precisely controlled left/right median-width ratio.

    Pins ``LABEL_COLUMN_WIDTH_SHARE`` directly: both columns use fixed,
    repeated text per row (so every row has the same width on each side),
    letting ``width_ratio`` target an exact left-width / right-width share.
    See the decision log: Fed fixture's true candidate-range ratio is 0.51
    (a genuine label column), Astra's false-merge repro is 1.0 (two
    independent prose columns) -- the threshold (0.65) sits in between.

    Four of the five right-column rows repeat ``right_text`` verbatim and
    the fifth is a much longer outlier -- the outlier does not move the
    MEDIAN width (still ``right_text``'s width, preserving the exact
    width-ratio target the caller asked for) but keeps the right block's
    fill share low, mirroring the Fed fixture's shape (mostly short bare
    names, one long "Corrigan, Vice Chairman" outlier). Five IDENTICAL
    lines (an earlier round of this fixture) read as maximally "filled" to
    RIGHT_BLOCK_FILL_TOLERANCE_WORD_WIDTHS / MEASURE_FILL_SHARE_MAX and
    would always trip that guard regardless of width_ratio -- a
    fixture-realism gap, not a genuine positive this guard should reject
    (see the log's "Round 4" notes).
    """
    doc = fitz.open()
    page = doc.new_page()
    y0 = 72
    page.insert_text(
        (72, y0),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
        fontname="helv",
    )

    right_w = fitz.get_text_length(right_text, fontname="helv", fontsize=10)
    x_w = fitz.get_text_length("X", fontname="helv", fontsize=10)
    n = max(1, round((width_ratio * right_w) / x_w))
    label = "X" * n
    left_lines = [label] * 5
    right_lines = [right_text] * 4 + [right_text + " and quite a bit more besides"]

    left_x = 90
    label_w = fitz.get_text_length(label, fontname="helv", fontsize=10)
    space_w = fitz.get_text_length(" ", fontname="helv", fontsize=10)
    gap = 1.2 * space_w
    right_x = left_x + label_w + gap
    row_start_y = y0 + 40

    left_rect = fitz.Rect(left_x, row_start_y - 8, left_x + label_w + 20, row_start_y + 140)
    right_rect = fitz.Rect(right_x, row_start_y - 8, right_x + 300, row_start_y + 140)
    page.insert_textbox(left_rect, "\n".join(left_lines), fontsize=10, fontname="helv")
    page.insert_textbox(right_rect, "\n".join(right_lines), fontsize=10, fontname="helv")
    return page


class TestGapThresholdBothSides:
    """Pin ALIGNED_RUN_GAP_MAX_WORD_SPACES against real measured gutters.

    See the decision log: the widest positive-case gap measured on the real
    Fed fixture was 1.76x a word space; the narrowest real two-column
    journal gutter measured (Fama-French 1997 JFE p.2) was 2.30x. The
    threshold (2.0x) sits in between with roughly 15% margin on each side --
    tight enough that both sides need an explicit regression pin.
    """

    def test_gap_at_1_5x_word_space_merges(self):
        page = _build_uniform_gap_bijection_page(1.5)
        assembled = _assemble_prose_with_aligned_runs(page)
        assert assembled is not None
        assert "XXXX Angellton" in assembled

    def test_gap_at_2_3x_word_space_does_not_merge(self):
        page = _build_uniform_gap_bijection_page(2.3)
        assert _assemble_prose_with_aligned_runs(page) is None


class TestWidthRatioThresholdBothSides:
    """Pin LABEL_COLUMN_WIDTH_SHARE against the Fed / Astra measured ratios."""

    def test_width_ratio_0_5_merges(self):
        page = _build_uniform_width_ratio_page(0.5)
        assert _assemble_prose_with_aligned_runs(page) is not None

    def test_width_ratio_0_8_does_not_merge(self):
        page = _build_uniform_width_ratio_page(0.8)
        assert _assemble_prose_with_aligned_runs(page) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
