"""GH-773 — HTML-entity decode gap in table-verification numeric-token gates.

A VLM can emit an HTML entity instead of the literal character
(``&minus;9.9`` instead of ``-9.9``, ``&#49;`` instead of ``1``). Several
numeric-token predicates in ``native_verifier.py``/``header_repair.py``/
``header_cut.py``/``witness.py`` tested the RAW markdown text against
``_NUM_TOKEN_RE``/``_NUMERIC_RE`` without decoding first, so an
entity-encoded token silently failed the numeric-token test — not flagged
as a mismatch, just invisible to the comparison. See
docs/log/2026-09-16_773-census.md (the defect catalogue) and
docs/log/2026-09-16_773-fix.md (the fix + this file's mutation results).

Each test below is written at the REAL CALLER named by the census's ranked
fix order, not at a helper, and is constructed so it demonstrably fails
without its corresponding fix (mutation results in the decision log).
Fixtures are entirely synthetic (no corpus content, no provider).
"""

from __future__ import annotations

from pathlib import Path

import fitz

# --------------------------------------------------------------------------
# Shared fixture builders
# --------------------------------------------------------------------------


def _fitz_page_with_numeric_rows(rows: list[list[tuple[float, str]]]) -> fitz.Page:
    doc = fitz.open()
    page = doc.new_page(width=700, height=900)
    for row_idx, cells in enumerate(rows):
        y = 100.0 + row_idx * 30
        for x, word in cells:
            page.insert_text((x, y), word, fontsize=9)
    return page


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    lines = ["| " + " | ".join(header) + " |", sep]
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(lines)


def _two_ruled_boxes_pdf(tmp_path: Path, top_values: list[str], bottom_values: list[str]) -> Path:
    """Two ruled tables, non-overlapping x-spans (mirrors test_table_witness.py)."""
    doc = fitz.open()
    page = doc.new_page()
    top_cols = [100, 220]
    bottom_cols = [340, 460]
    top_rows = [100 + i * 22 for i in range(2)]
    bottom_rows = [320 + i * 22 for i in range(2)]
    for cols, rows, values in (
        (top_cols, top_rows, top_values),
        (bottom_cols, bottom_rows, bottom_values),
    ):
        x0, x1 = cols[0], cols[-1] + 80
        it = iter(values)
        for y in rows:
            for x in cols:
                page.insert_text((x + 4, y + 12), next(it), fontsize=9)
        for yy in rows:
            page.draw_line((x0, yy), (x1, yy))
        for xx in cols + [x1]:
            page.draw_line((xx, rows[0]), (xx, rows[-1]))
    pdf_path = tmp_path / "doc.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


# --------------------------------------------------------------------------
# Rank 1 (native_verifier.py:1431 + :435, ``_value_guard``) — S1
# --------------------------------------------------------------------------


def test_rank1_value_guard_entity_fabrication_hard_fails_like_plain() -> None:
    """The load-bearing test: an entity-encoded fabricated value must
    hard-fail exactly like its plain-ASCII equivalent, not evade detection
    as a downgraded row-count warning.

    Pre-fix, an all-entity data row (``&#52;.40 &#53;.50 &minus;9.9``) has
    ZERO tokens recognized as numeric, so the row silently drops out of
    ``_parse_all_data_rows``'s numeric-row count. That desyncs the output
    row count from the native row count, which routes the multiset mismatch
    through the row-count-discrepancy AMBIGUOUS path (ships flagged, not
    hard-fail) instead of the clean-alignment CERTAIN_FAIL path — case D
    (entity, wrong) silently ships as a warning instead of hard-failing like
    case B (plain, wrong).
    """
    from socr.tables.native_verifier import _value_guard

    page = _fitz_page_with_numeric_rows(
        [
            [(100.0, "1.10"), (160.0, "2.20"), (220.0, "3.30")],
            [(100.0, "4.40"), (160.0, "5.50"), (220.0, "6.60")],
            [(100.0, "7.70"), (160.0, "8.80"), (220.0, "9.90")],
        ]
    )
    words = page.get_text("words")

    def _rows(row2_a: str, row2_b: str, row2_c: str) -> str:
        return _md_table(
            ["a", "b", "c"],
            [
                ["1.10", "2.20", "3.30"],
                [row2_a, row2_b, row2_c],
                ["7.70", "8.80", "9.90"],
            ],
        )

    case_a = _rows("4.40", "5.50", "6.60")  # plain, correct
    case_b = _rows("4.40", "5.50", "-9.9")  # plain, wrong
    case_c = _rows("&#52;.40", "&#53;.50", "&#54;.60")  # entity, correct
    case_d = _rows("&#52;.40", "&#53;.50", "&minus;9.9")  # entity, wrong (fabricated)

    hard_a, _, _, warn_a = _value_guard(words, case_a, "page")
    hard_b, _, _, warn_b = _value_guard(words, case_b, "page")
    hard_c, _, _, warn_c = _value_guard(words, case_c, "page")
    hard_d, _, _, warn_d = _value_guard(words, case_d, "page")

    assert hard_a is False and warn_a is None, "negative control: plain correct must pass clean"
    assert hard_b is True, "plain wrong value must hard-fail"
    assert hard_c is False and warn_c is None, (
        "entity-encoded CORRECT value must pass exactly like plain (no false positive)"
    )
    assert hard_d is True, (
        "entity-encoded WRONG value must hard-fail exactly like plain wrong (case D == case B); "
        "silently shipping this as a warning is the GH-773 S1 evasion"
    )


# --------------------------------------------------------------------------
# Rank 2 (native_verifier.py, ``_parse_all_data_rows`` -> label-binding) — S2
# --------------------------------------------------------------------------


def test_rank2_interleaved_label_binding_entity_hard_fails_like_plain() -> None:
    """TR-4a interleaved name/value-offset shape must hard-fail regardless of
    whether the data-row values are entity-encoded.

    Pre-fix, an all-entity data-without-label row has an empty numeric
    multiset AND an empty label, so it is dropped entirely by
    ``_parse_all_data_rows`` (neither a numeric row nor a label-only row).
    With both data rows dropped, only the two label-only rows remain in the
    timeline -- zero adjacent (numeric, label-only) pairs are found and the
    label-binding predicate never fires.
    """
    from socr.tables.native_verifier import verify_native_table

    gap = 60.0
    native_rows = [
        [(150.0, "5.1"), (150.0 + gap, "3.2")],
        [(150.0, "2.7"), (150.0 + gap, "4.1")],
    ]
    page = _fitz_page_with_numeric_rows(native_rows)

    def _interleaved(v1: str, v2: str, v3: str, v4: str) -> str:
        return _md_table(
            ["Name", "col_A", "col_B"],
            [
                ["", v1, v2],
                ["Alpha", "", ""],
                ["", v3, v4],
                ["Beta", "", ""],
            ],
        )

    plain = _interleaved("5.1", "3.2", "2.7", "4.1")
    entity = _interleaved("&#53;.1", "&#51;.2", "&#50;.7", "&#52;.1")

    result_plain = verify_native_table(page, plain)
    result_entity = verify_native_table(page, entity)

    assert result_plain.hard_fail is True, "baseline: plain interleaved pattern must hard-fail"
    assert "label_binding" in result_plain.reason
    assert result_entity.hard_fail is True, (
        "entity-encoded interleaved pattern must hard-fail exactly like plain "
        "(GH-773 S2): dropping the row from the numeric-row count instead of "
        "decoding it makes the label-binding predicate never see the data rows"
    )
    assert "label_binding" in result_entity.reason


# --------------------------------------------------------------------------
# Rank 3 (witness.py:392 via ``_numeric_tokens_from_text``) — S3
# --------------------------------------------------------------------------


def test_rank3_witness_block_swap_detected_when_entity_encoded(tmp_path: Path) -> None:
    """A geometry/content swap between two table blocks must be caught by
    pairing corroboration even when the model emits entity-encoded minus
    signs.

    Pre-fix, ``_numeric_tokens_from_text`` cannot decode ``&minus;NNN``, so
    every block's output multiset is empty, every overlap score is 0, and
    the strict-majority contradiction test never fires -- the swap ships
    silently as ``LOCATED`` under the wrong (swapped) index pairing.
    """
    from socr.tables.witness import WitnessStatus, prepare_table_witnesses

    top_values = ["-111", "-222", "-333", "-444"]
    bottom_values = ["-555", "-666", "-777", "-888"]
    pdf_path = _two_ruled_boxes_pdf(tmp_path, top_values, bottom_values)

    # Correctly-ordered case (negative control): first block carries the
    # TOP box's numbers, second block the BOTTOM box's -> no contradiction.
    md_correct = (
        "| a | b |\n| --- | --- |\n| &minus;111 | &minus;222 |\n| &minus;333 | &minus;444 |\n"
        "\nprose between the two tables\n\n"
        "| c | d |\n| --- | --- |\n| &minus;555 | &minus;666 |\n| &minus;777 | &minus;888 |\n"
    )
    with prepare_table_witnesses(pdf_path, page_num=1, markdown=md_correct) as witnesses:
        assert len(witnesses) == 2
        for w in witnesses:
            assert w.status is WitnessStatus.LOCATED, "correctly-ordered content must ship LOCATED"

    # Swapped case: first block actually carries the BOTTOM box's numbers.
    md_swapped = (
        "| c | d |\n| --- | --- |\n| &minus;555 | &minus;666 |\n| &minus;777 | &minus;888 |\n"
        "\nprose between the two tables\n\n"
        "| a | b |\n| --- | --- |\n| &minus;111 | &minus;222 |\n| &minus;333 | &minus;444 |\n"
    )
    with prepare_table_witnesses(pdf_path, page_num=1, markdown=md_swapped) as witnesses:
        assert len(witnesses) == 2
        for w in witnesses:
            assert w.status is WitnessStatus.AMBIGUOUS, (
                "entity-encoded swap must be detected exactly like a plain-ASCII "
                "swap (GH-773 S3), not silently ship under the wrong pairing"
            )
            assert "corroboration" in w.note


# --------------------------------------------------------------------------
# Rank 4 (header_cut.py:238, ``_emitted_header_tokens``) — one-hop wrapper
# --------------------------------------------------------------------------


def test_rank4_emitted_header_tokens_does_not_absorb_entity_data_row() -> None:
    """A genuine data row (>= 3 numeric cells) must never be absorbed into
    the emitted-header token set, even when its cells are entity-encoded.

    Pre-fix, the entity cells in ``grid[1]`` fail the raw numeric-token
    test, so the row LOOKS like "too few numerals to be a data row" and is
    incorrectly folded into the header token set (fixed transitively by the
    native_verifier.py sink -- no direct edit needed in header_cut.py).
    """
    from socr.tables.header_cut import _emitted_header_tokens

    grid_entity = [
        ["Model A", "", "", ""],
        ["&#49;.0", "&#50;.0", "&#51;.0", "&#52;.0"],
    ]
    grid_plain = [
        ["Model A", "", "", ""],
        ["1.0", "2.0", "3.0", "4.0"],
    ]

    tokens_entity = _emitted_header_tokens(grid_entity, [])
    tokens_plain = _emitted_header_tokens(grid_plain, [])

    assert tokens_plain == {"model", "a"}, "negative control: plain data row not absorbed"
    assert tokens_entity == {"model", "a"}, (
        "entity-encoded data row must not be absorbed into the header token "
        "set either (GH-773): row 1 has 4 numeric cells and is genuine data"
    )


# --------------------------------------------------------------------------
# Rank 5 (header_repair.py:60, ``detect_header_column_collapse``) — S4
# --------------------------------------------------------------------------


def test_rank5_detect_header_column_collapse_sees_entity_data_row() -> None:
    from socr.tables.header_repair import detect_header_column_collapse

    grid_plain = [
        ["Currency", "-23% to -14% ... +23% or more", "Appreciation"],
        ["Euro1", "1", "2", "16", "49", "23", "8", "1"],
    ]
    grid_entity = [
        ["Currency", "-23% to -14% ... +23% or more", "Appreciation"],
        ["Euro1", "&#49;", "&#50;", "&#49;&#54;", "&#52;&#57;", "&#50;&#51;", "&#56;", "&#49;"],
    ]

    collapsed_plain, hdr_cols_plain, expected_plain = detect_header_column_collapse(grid_plain)
    collapsed_entity, hdr_cols_entity, expected_entity = detect_header_column_collapse(grid_entity)

    assert (collapsed_plain, hdr_cols_plain, expected_plain) == (True, 3, 8)
    assert (collapsed_entity, hdr_cols_entity, expected_entity) == (True, 3, 8), (
        "an entity-encoded data row must still be recognized as a data row and "
        "detect the same collapse the plain row does (GH-773 S4); pre-fix each "
        "entity cell counts as zero numeric cells and the collapse goes undetected"
    )


# --------------------------------------------------------------------------
# Rank 6 (header_repair.py:209 + :896, header_cut.py:89) — S5
# --------------------------------------------------------------------------


def test_rank6_first_data_row_idx_finds_entity_data_row() -> None:
    """header_repair.py's own inline reimplementation (:896)."""
    from socr.tables.header_repair import _first_data_row_idx

    grid_plain = [["h1", "h2", "h3", "h4"], ["1.0", "2.0", "3.0", "4.0"]]
    grid_entity = [["h1", "h2", "h3", "h4"], ["&#49;.0", "&#50;.0", "&#51;.0", "&#52;.0"]]

    assert _first_data_row_idx(grid_plain, expected_cols=4) == 1
    assert _first_data_row_idx(grid_entity, expected_cols=4) == 1, (
        "entity-encoded data row must be found at the same index as its plain "
        "equivalent (GH-773); pre-fix it counts zero numeric cells and the "
        "function falls through to len(grid) (no data row found)"
    )


def test_rank6_best_anchor_y_matches_entity_grid_row_to_native() -> None:
    """Sink-transitive fix (header_repair.py:209, ``_best_anchor_y``)."""
    from socr.tables.header_repair import _best_anchor_y

    rows_by_y = {
        100: [
            (100.0, 100.0, 110.0, 109.0, "1.0", 0, 0, 0),
            (160.0, 100.0, 170.0, 109.0, "2.0", 0, 0, 0),
            (220.0, 100.0, 230.0, 109.0, "3.0", 0, 0, 0),
        ]
    }
    grid_plain = [["h1", "h2", "h3"], ["1.0", "2.0", "3.0"]]
    grid_entity = [["h1", "h2", "h3"], ["&#49;.0", "&#50;.0", "&#51;.0"]]

    assert _best_anchor_y(rows_by_y, grid_plain) == 100.0
    assert _best_anchor_y(rows_by_y, grid_entity) == 100.0, (
        "an entity-encoded emitted row must still anchor to its native "
        "counterpart (GH-773); pre-fix its multiset is empty and no anchor is found"
    )


def test_rank6_header_cut_anchor_candidates_matches_entity_grid_row() -> None:
    """Sink-transitive fix (header_cut.py:89, ``_anchor_candidates``)."""
    from socr.tables.header_cut import _anchor_candidates

    rows_by_y = {
        100: [
            (100.0, 100.0, 110.0, 109.0, "1.0", 0, 0, 0),
            (160.0, 100.0, 170.0, 109.0, "2.0", 0, 0, 0),
            (220.0, 100.0, 230.0, 109.0, "3.0", 0, 0, 0),
        ]
    }
    grid_plain = [["h1", "h2", "h3"], ["1.0", "2.0", "3.0"]]
    grid_entity = [["h1", "h2", "h3"], ["&#49;.0", "&#50;.0", "&#51;.0"]]

    assert _anchor_candidates(rows_by_y, grid_plain) == [100.0]
    assert _anchor_candidates(rows_by_y, grid_entity) == [100.0], (
        "entity-encoded emitted row must anchor exactly like plain (GH-773)"
    )


# --------------------------------------------------------------------------
# Rank 7 (header_repair.py:148, guard inside
# ``_repair_too_narrow_spanning_header``) — S9, shipped with rank 6
# --------------------------------------------------------------------------


def _narrow_spanning_grid(v1: str, v2: str) -> list[list[str]]:
    """A body-data-shaped grid: geometrically identical to the GH-276 narrow
    spanning-header shape, EXCEPT the secondary band (row 1) actually carries
    two numeric values rather than group labels -- the case the guard exists
    to reject."""
    return [
        ["", "Group", "", ""],
        ["", v1, "", v2],
        ["", "(1)", "(2)", "(3)", "(4)"],
        ["row", "10.0", "11.0", "12.0", "13.0"],
    ]


def test_rank7_spanning_header_guard_refuses_entity_body_data_row(monkeypatch) -> None:
    """A body-data-shaped secondary band must abort the repair (never get
    restructured into a spanning header) whether its cells are plain or
    entity-encoded.

    Full-function coverage note: with ``_native_label_lane`` in its real,
    unmocked state, this fixture returns ``None`` regardless of whether this
    guard fires or not -- a separate, unranked gap in ``_native_label_lane``
    (matches a label against native words without decoding) independently
    aborts the repair one step later, masking this guard's own effect
    (confirmed by reverting ONLY the guard: docs/log/2026-09-16_773-fix.md).
    ``_native_label_lane`` is therefore stubbed here to a value that clears
    the later arithmetic checks, isolating the ONE behaviour this ticket's
    diff changed: whether the guard recognizes an entity-encoded numeric
    secondary band and refuses to restructure it as a header.
    """
    import socr.tables.header_repair as header_repair

    monkeypatch.setattr(header_repair, "_native_label_lane", lambda grid, label, words: 2)

    plain = header_repair._repair_too_narrow_spanning_header(
        _narrow_spanning_grid("1.5", "2.0"), []
    )
    entity = header_repair._repair_too_narrow_spanning_header(
        _narrow_spanning_grid("&#49;.5", "&#50;.0"), []
    )

    assert plain is None, "baseline: a numeric secondary band must abort the repair"
    assert entity is None, (
        "an entity-encoded numeric secondary band must abort the repair exactly "
        "like plain (GH-773 S9); pre-fix the guard cannot see the entity values "
        "are numeric and the body-data row gets restructured as a header instead"
    )


# --------------------------------------------------------------------------
# Rank 8 (native_verifier.py, ``_output_header_numeric_tokens``) — S8
# --------------------------------------------------------------------------


def test_rank8_output_header_numeric_tokens_counts_entity_form() -> None:
    from socr.tables.native_verifier import _output_header_numeric_tokens

    text_plain = "| 1 | 2 | 3 |\n| --- | --- | --- |\n| 1.0 | 2.0 | 3.0 |\n"
    text_entity = "| &#49; | &#50; | &#51; |\n| --- | --- | --- |\n| 1.0 | 2.0 | 3.0 |\n"

    assert _output_header_numeric_tokens(text_plain) == frozenset({"1", "2", "3"})
    assert _output_header_numeric_tokens(text_entity) == frozenset({"1", "2", "3"}), (
        "an entity-encoded header spec-number must be counted exactly like "
        "its plain form (GH-773 S8), not silently dropped from the set"
    )


# --------------------------------------------------------------------------
# Negative controls
# --------------------------------------------------------------------------


def test_negative_control_genuinely_non_numeric_cell_stays_non_numeric() -> None:
    """Decoding must never manufacture a false numeric-token match."""
    from socr.tables.native_verifier import is_numeric_token

    assert is_numeric_token("Panel A.") is False
    assert is_numeric_token("&amp;nbsp;Panel A.") is False
    assert is_numeric_token("n.a.") is False


def test_negative_control_native_trailing_dash_token_is_pinned() -> None:
    """GH-773 was raised alongside a warning that ``_normalize_cell``'s
    trailing-dash strip (``'1990-'`` -> ``'1990'``) is NOT inert on native
    tokens the way HTML-entity decoding is. This ticket's fix decodes
    entities ONLY (``html.unescape``, no dash strip) at every numeric-token
    predicate touched, so a native year-range token like ``1990-`` must be
    classified identically before and after -- pinning the DIFFERENCE (none)
    rather than an absolute value.
    """
    from socr.tables.native_verifier import is_numeric_token

    # '1990-' has no HTML entity to decode, so html.unescape is a no-op on
    # it either way; the trailing dash makes it fail _NUM_TOKEN_RE (which is
    # anchored and does not allow a trailing '-'), so it is not a numeric
    # token both before and after this ticket's fix.
    assert is_numeric_token("1990-") is False
    assert is_numeric_token("1990") is True


def test_negative_control_plain_well_formed_table_is_unaffected() -> None:
    """Baseline: a clean plain-ASCII table with no entities must still pass
    with no drift and no warning after this ticket's changes."""
    from socr.tables.native_verifier import verify_native_table

    page = _fitz_page_with_numeric_rows(
        [
            [(100.0, "1.10"), (160.0, "2.20")],
            [(100.0, "3.30"), (160.0, "4.40")],
        ]
    )
    text = _md_table(["a", "b"], [["1.10", "2.20"], ["3.30", "4.40"]])
    result = verify_native_table(page, text)
    assert result.hard_fail is False
    assert result.reason == ""
