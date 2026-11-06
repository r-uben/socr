"""#1218: geometric classes of layer numbers, and the dropped-numbers multiset."""

from socr.tables.layer_body import (
    BODY,
    FOOTNOTE,
    FURNITURE,
    MARGIN,
    body_text,
    classify_layer_numbers,
    drop_reference_from_page,
    slot_predicate,
    superscript_boxes,
)
from socr.tables.reading_agreement import dropped_numbers

RECT = (0.0, 0.0, 612.0, 792.0)


def line(texts, y, h=10.0, x=50.0, step=60.0, heights=None):
    """Upright words of one line: (x0, y0, x1, y1, text)."""
    out = []
    for i, t in enumerate(texts):
        hh = heights[i] if heights else h
        y1 = y + h
        out.append((x + i * step, y1 - hh, x + i * step + 40.0, y1, t))
    return out


def classify(words, **kw):
    return classify_layer_numbers(words, RECT, kw.pop("furniture", None), **kw)


def classes(words, **kw):
    return {n.written: n.cls for n in classify(words, **kw)}


def test_body_positive():
    got = classes(line(["Inflation", "was", "3.2", "percent", "and", "4.7"], 100))
    assert got["3.2"] == BODY and got["4.7"] == BODY


def column(values, x=40.0, y0=100.0, pitch=30.0):
    return [(x, y0 + i * pitch, x + 20.0, y0 + i * pitch + 10.0, v) for i, v in enumerate(values)]


def test_footnote_by_height_ratio_without_span_data():
    words = line(["Rates", "rose", "sharply", "3"], 100, heights=[10, 10, 10, 6])
    assert classes(words)["3"] == FOOTNOTE
    assert classes(line(["Rates", "rose", "sharply", "3"], 100))["3"] == BODY
    assert classes(line(["rose", "3"], 100, heights=[10, 6]))["3"] == BODY


def test_furniture_predicate_slot_and_folio():
    head = line(["Annual", "Report", "2020"], 20)
    repeated = {(str(w[4]).casefold(), round(w[0], 1), round(w[1], 1)) for w in head}
    folio_word = (300.0, 700.0, 320.0, 710.0, "17")
    pred = slot_predicate(repeated, {(300.0, 700.0)})
    body = line(["x", "5.5"], 200)
    got = classes(head + body + [folio_word], furniture=pred)
    assert got["2020"] == FURNITURE and got["17"] == FURNITURE
    assert got["5.5"] == BODY
    assert classes(head + body + [folio_word])["2020"] == BODY
    # a non-number at a folio slot is not furniture by that rule
    assert not pred((300.0, 700.0, 320.0, 710.0, "Page"))


def sideways(texts, x, y0=100.0, step=60.0, w=10.0):
    """A line reading bottom-to-top on a rotated page: boxes taller than wide."""
    return [(x, y0 + i * step, x + w, y0 + i * step + 50.0, t) for i, t in enumerate(texts)]


def test_margin_on_rotated_page_positive_and_negative():
    words = sideways(["Inflation", "outlook", "table", "1.5", "2.5"], 100.0)
    words.append((20.0, 400.0, 60.0, 410.0, "I-31"))  # upright label on a sideways page
    got = classify(words, rotation=90)
    assert {n.cls for n in got if n.written in ("1.5", "2.5")} == {BODY}
    assert [n.cls for n in got if n.cls == MARGIN] == [MARGIN]
    upright = line(["Inflation", "outlook", "table", "I-31"], 100)
    assert {n.cls for n in classify(upright)} == {BODY}


def test_rotation_decides_when_words_cannot():
    # short sideways words: no multi-character evidence; /Rotate 90 says rotated
    words = [(100.0, 100.0 + 12 * i, 110.0, 110.0 + 12 * i, str(i + 1)) for i in range(4)]
    assert all(n.cls == BODY for n in classify(words, rotation=90))


def test_body_text_is_the_label_row_tails_only():
    words = line(["Rates", "rose", "3.20", "to", "4.70"], 100)  # prose: numbers not a tail
    words += line(["Core", "inflation", "2.1", "(0.014)", "3.3"], 130)
    assert body_text(classify(words)) == "2.1 (0.014) 3.3"


def tails(words, **kw):
    return [n.written for n in classify(words, **kw) if n.row is not None]


def test_label_row_positive():
    assert tails(line(["Unemployment", "rate", "5.1", "4.9", "4.8"], 100)) == ["5.1", "4.9", "4.8"]


def test_label_row_only_the_tail_numbers_are_candidates():
    # digits glued to the label word are never candidates
    got = tails(line(["rate1", "65.9", "64.9", "64.7"], 100))
    assert got == ["65.9", "64.9", "64.7"]
    assert tails(line(["Greenbook1", "R2", "10.5", "12.5", "9.5"], 100)) == ["10.5", "12.5", "9.5"]


def test_label_row_stars_and_trailing_punctuation_are_tail():
    got = tails(line(["Effect", "of", "x", "0.5**", "1.2*", "3.4,"], 100))
    assert len(got) == 3


def test_label_row_negative_prose_with_numbers_inside():
    assert (
        tails(line(["Inflation", "was", "3.2", "percent", "and", "4.7", "in", "2020"], 100)) == []
    )
    # a tail shorter than the minimum is not a row
    assert tails(line(["Total", "3.2", "4.7"], 100)) == []


def test_label_row_negative_formula_line():
    assert tails(line(["y", "=", "0.5", "1.5", "2.5", "4.5"], 100)) == []
    assert tails(line(["Growth", "+", "2", "7", "3", "9"], 100)) == []


def test_a_constant_step_tail_with_no_chart_evidence_is_a_data_row():
    """Astra, PR #1277: ``China 10 20 30`` is a table row whose values step evenly."""
    assert tails(line(["China", "10", "20", "30"], 100)) == ["10", "20", "30"]
    assert tails(line(["Percent", "0.75", "0.50", "0.25", "0.00"], 100)) == [
        "0.75",
        "0.50",
        "0.25",
        "0.00",
    ]


def test_a_constant_step_tail_at_a_chart_box_is_an_axis_even_without_signs():
    row = line(["Percent", "0.75", "0.50", "0.25", "0.00"], 100)
    box = [(40.0, 20.0, 400.0, 90.0)]  # the plot, its tick labels just below
    assert tails(row, excluded_boxes=box) == []
    # difference: the same row with the plot far away is a data row
    assert len(tails(row, excluded_boxes=[(40.0, 400.0, 400.0, 500.0)])) == 4
    # a non-constant tail is a row beside the same box
    assert (
        len(tails(line(["Percent", "0.75", "0.50", "0.30", "0.00"], 100), excluded_boxes=box)) == 4
    )


def test_a_constant_step_run_of_calendar_years_is_a_header_or_axis():
    assert tails(line(["Year", "1996", "1998", "2000", "2002"], 100)) == []
    # a year with its footnote callout run on (2008 + 1)
    assert tails(line(["Item", "20071", "20081", "20091"], 100)) == []
    # difference: a non-constant run of years, or constant-step numbers that are not years
    assert len(tails(line(["Year", "1996", "1998", "2001", "2002"], 100))) == 4
    assert len(tails(line(["Count", "1000", "1100", "1200"], 100))) == 3


def test_label_row_needs_a_label_with_two_letters():
    assert tails(line(["a", "3.2", "4.7", "5.1"], 100)) == []
    assert tails(line(["3.2", "4.7", "5.1", "6.0"], 100)) == []


def test_label_row_ignores_furniture_and_superscript_words():
    words = line(["Rate", "3.2", "4.7", "5.2", "17"], 100)
    pred = slot_predicate(set(), {(50.0 + 4 * 60.0, round(words[4][1], 1))})
    # the folio at the end is furniture: removed before the split, tail stays the 3 numbers
    assert tails(words, furniture=pred) == ["3.2", "4.7", "5.2"]


def test_dropped_identical_is_zero():
    rec = dropped_numbers("3.2 4.7\n1,200", "Inflation 3.2 and 4.7, with 1,200 units")
    assert rec.dropped == () and rec.body_count == 3


def test_dropped_one_deleted_number():
    assert dropped_numbers("3.2 4.7 9.9", "Inflation 3.2 and 9.9").dropped == ("4.7",)


def test_dropped_is_a_multiset():
    assert dropped_numbers("5 5", "one 5 only").dropped == ("5",)


def test_one_space_thousands_group_never_satisfies_two_layer_numbers():
    """Mutation M27b: a claimed group's parts are consumed. Here the plain ``234`` takes one
    of the two ``234`` tokens, so two ``1,234`` cannot both be found in two groups."""
    two_groups = "1 234 and 1 234"
    assert dropped_numbers("234 1,234 1,234", two_groups).dropped == ("1,234",)
    # a single group cannot hide two layer numbers either
    assert dropped_numbers("1,234 1,234", "total 1 234").dropped == ("1,234",)
    # the control: a group for each, and nothing else owed
    assert dropped_numbers("1,234 1,234", two_groups).dropped == ()


def test_number_only_inside_latex_counts_as_present():
    rec = dropped_numbers("0.75 12.5", "the rate is $\\rho = 0.75$ and $12.5\\%$")
    assert rec.dropped == ()


def test_math_glue_is_split():
    # exponent, subscript, equation tag and a power: digits stand alone
    rec = dropped_numbers("2 14 15", "$R^2$ and $y \\tag{14}$ and $10^{15}$")
    assert rec.dropped == ()


def test_doi_only_in_a_link_url_counts_as_present():
    ref = "10.1257/aer.20180123"
    assert (
        dropped_numbers(ref, "[Smith (2020)](https://doi.org/10.1257/aer.20180123)").dropped == ()
    )
    assert dropped_numbers(ref, "[Smith (2020)](https://example.org/x)").dropped != ()


def test_a_faithful_body_number_equal_to_a_furniture_number_is_never_dropped():
    """The reading side is not reduced by running heads or folios, so the count is a true
    lower bound: a faithful body number cannot be reported dropped (Astra #1218 P1-A)."""
    # Heston p22: the row ``Day 4`` is faithful; the folio ``4`` is not in the reference
    assert dropped_numbers("4 1.06 1.07", "Day 4 1.06 1.07").dropped == ()
    # a faithful ``| Year | 2020 | 5.1 | 4.9 |`` against a running head ``Year 2020``
    assert dropped_numbers("2020 5.1 4.9", "| Year | 2020 | 5.1 | 4.9 |").dropped == ()
    # a head wrapped over two reading lines changes nothing (Astra P1-B: no invented drop)
    assert dropped_numbers("5.1", "January 25,\n2018 Page 36\nrow 5.1").dropped == ()


def test_a_transcribed_running_head_may_absorb_a_dropped_body_copy():
    """The documented limitation of the lower bound: the head's number stands in for a
    dropped body number of the same value. It can hide a drop, never invent one."""
    reading = "January 25, 2018 Page 36 of 40\nrow 5.1"
    assert dropped_numbers("2018 5.1", reading).dropped == ()
    # and a real drop of another value is still found
    assert dropped_numbers("2018 5.1 7.7", reading).dropped == ("7.7",)


def at(items, y=100.0, h=8.5):
    """Words at explicit x positions: items are (text, x0, x1)."""
    return [(x0, y, x1, y + h, t) for t, x0, x1 in items]


def fragment_row():
    # Otonello p7: the decimal point is missing, the two halves sit 2.4 pt apart, cells ~60 pt
    return at(
        [
            ("Mean", 37.6, 58.6),
            ("0", 155.1, 159.3),
            ("004", 161.7, 174.4),
            ("0", 237.5, 241.7),
            ("263", 244.1, 256.8),
            ("5", 321.8, 326.0),
            ("788", 328.4, 341.1),
        ]
    )


def test_decimal_fragments_in_a_tail_are_joined():
    assert tails(fragment_row()) == ["0.004", "0.263", "5.788"]


def test_decimal_fragment_with_a_minus_star_and_colon():
    row = at(
        [
            ("PLow", 54.1, 75.9),
            ("−24", 105.5, 120.6),
            ("9∗", 121.6, 128.7),
            ("−12", 161.1, 176.2),
            ("5∗", 177.2, 184.2),
            ("−7", 216.6, 227.5),
            ("2∗", 228.5, 235.6),
        ]
    )
    # intra-pair gap 1.0 pt against a largest gap of 32.4 pt: ratio 0.03, inside the cut
    assert tails(row) == ["−24.9", "−12.5", "−7.2"]
    # difference: the same row with 2.4 pt intra-pair gaps (ratio 0.07, above the derived
    # DECIMAL_GAP_FRACTION) is NOT joined
    wide = at(
        [
            ("PLow", 54.1, 75.9),
            ("−24", 105.5, 120.6),
            ("9∗", 123.0, 130.1),
            ("−12", 161.1, 176.2),
            ("5∗", 178.6, 185.6),
            ("−7", 216.6, 227.5),
            ("2∗", 229.9, 237.0),
        ]
    )
    assert "−24.9" not in tails(wide)
    colon = at(
        [
            ("month", 118.0, 146.0),
            ("91", 206.1, 216.9),
            (":3", 217.0, 225.4),
            ("0", 295.0, 300.0),
            (":52", 300.1, 314.2),
            ("5", 383.9, 388.0),
            (":7", 388.1, 397.7),
        ]
    )
    assert tails(colon) == ["91.3", "0.52", "5.7"]


def test_one_close_pair_or_evenly_spaced_integers_are_not_joined():
    one_pair = at(
        [
            ("Total", 10.0, 40.0),
            ("12", 100.0, 110.0),
            ("345", 112.4, 125.0),
            ("7", 200.0, 205.0),
            ("8", 300.0, 305.0),
        ]
    )
    assert tails(one_pair) == ["12", "345", "7", "8"]
    even = at(
        [("Count", 10.0, 40.0), ("12", 100.0, 110.0), ("345", 140.0, 155.0), ("77", 190.0, 200.0)]
    )
    assert tails(even) == ["12", "345", "77"]


def test_a_lost_minus_is_never_matched_by_the_unsigned_number():
    assert dropped_numbers("-0.04", "the value is 0.04").dropped == ("-0.04",)
    assert dropped_numbers("0.04", "the value is -0.04").dropped == ("0.04",)


def test_zero_comma_is_a_decimal_comma_but_a_thousands_comma_is_not():
    row = line(["GDP", "real", "0,332", "0,184", "0,055"], 100)
    assert tails(row) == ["0.332", "0.184", "0.055"]
    assert tails(line(["Count", "real", "1,332", "2,184", "5,055"], 100)) == [
        "1,332",
        "2,184",
        "5,055",
    ]
    assert tails(line(["GDP", "real", "-0,238", "0,341", "0,197"], 100))[0] == "-0.238"


def test_words_in_an_excluded_box_are_ignored():
    row = line(["Unemployment", "rate", "5.1", "4.9", "4.8"], 100)
    assert len(tails(row)) == 3
    assert tails(row, excluded_boxes=[(0.0, 90.0, 600.0, 120.0)]) == []
    assert len(tails(row, excluded_boxes=[(0.0, 300.0, 600.0, 400.0)])) == 3


def test_span_flag_superscript_leaves_the_row_tail():
    words = line(["Rate", "3.2", "4.7", "5.2", "7"], 100)  # the 7 is as tall as the rest
    flagged = [(w[0], w[1], w[2], w[3]) for w in words[-1:]]
    boxes = superscript_boxes(
        {"blocks": [{"lines": [{"spans": [{"flags": 1, "bbox": flagged[0]}]}]}]}
    )
    assert tails(words, superscripts=boxes) == ["3.2", "4.7", "5.2"]
    assert tails(words, superscripts=[]) == ["3.2", "4.7", "5.2", "7"]


def _page_with_row(table_bbox=None, image=False):
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    for i, t in enumerate(["Unemployment", "rate", "5.1", "4.9", "4.8"]):
        page.insert_text((40 + 60 * i, 100), t, fontsize=10)
    if image:
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 4, 4), False)
        pix.clear_with(200)
        page.insert_image(fitz.Rect(30, 80, 380, 110), pixmap=pix)
    return page


def test_adapter_reads_a_label_row_from_a_page():
    page = _page_with_row()
    ref = drop_reference_from_page(page, repeated_slots=frozenset(), folio_slots=frozenset())
    assert ref.text == "5.1 4.9 4.8"


def test_adapter_excludes_a_row_under_an_image_box():
    kw = {"repeated_slots": frozenset(), "folio_slots": frozenset()}
    # difference: the same row, with and without an image drawn over it
    assert drop_reference_from_page(_page_with_row(), **kw).text == "5.1 4.9 4.8"
    assert drop_reference_from_page(_page_with_row(image=True), **kw).text == ""


def test_adapter_image_box_shares_the_words_space_on_a_rotated_page():
    page = _page_with_row(image=True)
    page.set_rotation(90)
    kw = {"repeated_slots": frozenset(), "folio_slots": frozenset()}
    # get_image_info() and get_text("words") are both unrotated: the box still covers the row
    assert drop_reference_from_page(page, **kw).text == ""
    bare = _page_with_row()
    bare.set_rotation(90)
    assert drop_reference_from_page(bare, **kw).text == "5.1 4.9 4.8"


def test_a_decimal_comma_read_back_as_written_is_not_a_drop():
    """Astra P2: the layer's ``0,332`` is repaired to ``0.332``; so is the reading's."""
    row = line(["Korea", "0,332", "0,471", "0,913"], 100)
    reference = body_text(classify(row))
    assert reference == "0.332 0.471 0.913"
    assert dropped_numbers(reference, "| Korea | 0,332 | 0,471 | 0,913 |").dropped == ()
    assert dropped_numbers(reference, "Korea 0,332 0,471 0,913").dropped == ()
    # difference: a different value still drops, and a thousands comma is not a decimal
    assert dropped_numbers(reference, "| Korea | 0,332 | 0,471 | 0,914 |").dropped == ("0.913",)
    assert dropped_numbers("1,332 2 3", "| Count | 1,332 | 2 | 3 |").dropped == ()
    assert dropped_numbers("1,332 2 3", "| Count | 1.332 | 2 | 3 |").dropped == ("1,332",)


def test_space_thousands_is_an_alternative_writing_never_a_second_claim():
    reference = "1,234 2,345 3,456"
    assert dropped_numbers(reference, "| Total | 1 234 | 2 345 | 3 456 |").dropped == ()
    assert dropped_numbers(reference, "Total 1 234, 2 345 and 3 456").dropped == ()
    # a different value still drops
    assert dropped_numbers(reference, "| Total | 1 235 | 2 345 | 3 456 |").dropped == ("1,234",)
    # the parts of ``1 234`` are claimed once: a layer ``234`` takes the reading's ``234``
    # first, so the layer's ``1,234`` is not also satisfied by it
    assert dropped_numbers("1,234 234 5", "| Total | 1 234 | 5 |").dropped == ("1,234",)
    # two reading cells ``1`` and ``234`` in separate cells are not ``1 234``
    assert dropped_numbers("1,234 7 8", "| Total | 1 | 234 | 7 | 8 |").dropped == ("1,234",)


def test_bracket_for_minus_is_an_alternative_writing_never_a_second_claim():
    reference = "(1.2) (3.4) (5.6)"
    assert dropped_numbers(reference, "| Net | -1.2 | -3.4 | -5.6 |").dropped == ()
    assert dropped_numbers(reference, "| Net | \u22121.2 | \u22123.4 | \u22125.6 |").dropped == ()
    # a different value, or the sign lost, still drops
    assert dropped_numbers(reference, "| Net | -1.2 | -3.5 | -5.6 |").dropped == ("(3.4)",)
    assert dropped_numbers(reference, "| Net | 1.2 | 3.4 | 5.6 |").dropped == (
        "(1.2)",
        "(3.4)",
        "(5.6)",
    )
    # one reading ``-1.2`` cannot stand for two layer ``(1.2)``
    assert dropped_numbers("(1.2) (1.2) 7", "| Net | -1.2 | 7 |").dropped == ("(1.2)",)


def test_known_gap_prose_repeating_a_dropped_rows_values_hides_it():
    """DOCUMENTED RECALL GAP (#1277 P0-1, Astra), pinned so it is not mistaken for coverage.

    The comparison is a multiset over the WHOLE reading, not row by row. Source row
    ``China 32.7 32.2 32.9``; a reading that drops the China row but whose prose repeats the
    three values reports nothing dropped. A row-local comparison was built and measured
    (+5 false alarms, 0 real drops gained on the corpus) and not shipped. If this test starts
    failing, the semantics changed: update the decision log with the new measurement.
    """
    reference = body_text(classify(line(["China", "32.7", "32.2", "32.9"], 100)))
    assert reference == "32.7 32.2 32.9"
    faithful = "| China | 32.7 | 32.2 | 32.9 |"
    prose = "Growth was 32.7, 32.2 and 32.9 in China."
    assert dropped_numbers(reference, faithful).dropped == ()
    # the gap: the row is gone, the prose answers for it
    assert dropped_numbers(reference, "| Korea | 1.5 |\n\n" + prose).dropped == ()
    # difference: without the prose the same reading is caught
    assert dropped_numbers(reference, "| Korea | 1.5 |").dropped == ("32.7", "32.2", "32.9")
