"""#1185: a model-read page is CHECKED only if its numbers occur in the page's own text layer.

Pins the comparator (``tables.reading_agreement``), the clause that turns a disagreement
into a mark (``verdict.text_layer_numbers_clause``) and the seam that demotes the page
(``manifest._apply_verdict_marks``), including the resume permanence of the demotion.
The motivating case is 2018 Tealbook p125, where a reading shipped ``-8`` for a printed
``-.8`` and the page judge called it faithful.
"""

from __future__ import annotations

from pathlib import Path

import socr
from socr.core.marks import Check, CheckResult
from socr.core.manifest import _apply_verdict_marks
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import PageState
from socr.core.verdict import (
    DAMAGED_LAYER_AGREES_REASON,
    DAMAGED_LAYER_PREFIX,
    NO_NUMBER_REASON,
    NO_OWN_TEXT_REASON,
    text_layer_numbers_clause,
)
from socr.pipeline.orchestrator import _resume_verdict_is_permanently_unverifiable
from socr.tables.reading_agreement import (
    PROVENANCE_OWN_CHARACTERS,
    compare_numbers,
    describe_disputes,
    numeric_tokens,
)

import pytest

TEALBOOK_LAYER = "Korea\n4.0\n2.6\n5.7\n-.8\n4.0\n3.3"
TEALBOOK_READING = "| Korea | 4.0 | 2.6 | 5.7 | -8 | 4.0 | 3.3 |"
TEALBOOK_READING_OK = "| Korea | 4.0 | 2.6 | 5.7 | -.8 | 4.0 | 3.3 |"


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _page(layer: str = TEALBOOK_LAYER, *, born_digital: bool = True) -> PageState:
    p = PageState(page_num=1)
    p.own_layer_text = layer
    p.is_born_digital = born_digital
    return p


def _qwen(text: str) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen")


# --- comparator -------------------------------------------------------------------------


def test_tealbook_minus_dropped_is_disputed_and_named() -> None:
    """A reading that ships -8 for a layer's -.8 is disputed, with the counterpart named."""
    record = compare_numbers(TEALBOOK_LAYER, TEALBOOK_READING, provenance=PROVENANCE_OWN_CHARACTERS)
    assert record.disputed == (("-.8", "-8"),)
    assert record.agreed is False
    assert describe_disputes(record) == "text layer says -.8, reading says -8"


def test_faithful_minus_point_eight_agrees() -> None:
    """Shipping -.8 or -0.8 for a layer's -.8 is agreement: the two spellings are one value."""
    for reading in (
        "| Korea | 4.0 | 2.6 | 5.7 | -.8 | 4.0 | 3.3 |",
        "| Korea | 4.0 | 2.6 | 5.7 | -0.8 | 4.0 | 3.3 |",
    ):
        record = compare_numbers(TEALBOOK_LAYER, reading, provenance=PROVENANCE_OWN_CHARACTERS)
        assert record.disputed == ()
        assert record.agreed is True


def test_reading_order_does_not_create_disputes() -> None:
    """The same multiset in a different stream order is not a dispute."""
    record = compare_numbers(
        "a 1.5 b 2.5\nc 3.5", "c 3.5 a 1.5 b 2.5", provenance=PROVENANCE_OWN_CHARACTERS
    )
    assert record.disputed == ()


def test_one_inserted_number_is_the_only_dispute() -> None:
    """One extra number in the reading is disputed alone; the numbers after it do not cascade."""
    record = compare_numbers(
        "1.1 2.2 3.3 4.4 5.5",
        "1.1 2.2 9.9 3.3 4.4 5.5",
        provenance=PROVENANCE_OWN_CHARACTERS,
    )
    assert record.disputed == ((None, "9.9"),)


def test_shipping_a_number_more_often_than_the_layer_is_one_dispute() -> None:
    """A surplus copy of a number the layer prints once is disputed exactly once."""
    record = compare_numbers("2.1 3.0", "2.1 2.1 3.0", provenance=PROVENANCE_OWN_CHARACTERS)
    assert len(record.disputed) == 1


def test_no_numbers_shipped_is_not_agreement() -> None:
    """Agreement over nothing is vacuous: compared == 0 must not read as agreed."""
    record = compare_numbers("1.5", "no numbers here", provenance=PROVENANCE_OWN_CHARACTERS)
    assert record.agreed is False
    assert record.compared == 0


def test_table_bracket_is_part_of_the_token() -> None:
    """In a table row, (0.014) is a standard error and does not match a bare 0.014."""
    record = compare_numbers("0.014", "| x | (0.014) |", provenance=PROVENANCE_OWN_CHARACTERS)
    assert record.disputed


def test_running_text_bracket_only_difference_is_matched() -> None:
    """In running text, a citation (2017) matches a layer's bare 2017."""
    record = compare_numbers(
        "see Kilian 2017 here", "see Kilian (2017) here", provenance=PROVENANCE_OWN_CHARACTERS
    )
    assert record.disputed == ()


def test_running_text_year_with_run_on_callout_matches() -> None:
    """A year whose footnote callout the layer ran on (2009.9 for 2009.) is not a dispute."""
    record = compare_numbers(
        "half of 2009.9 We drop", "half of 2009. We drop", provenance=PROVENANCE_OWN_CHARACTERS
    )
    assert record.disputed == ()


def test_dropped_minus_is_caught_when_the_layer_detaches_it() -> None:
    """A minus the layer sets apart with a space is joined, so dropping it is disputed."""
    record = compare_numbers(
        "Constant 0.001 − 0.077",
        "| Constant | 0.001 | 0.077 |",
        provenance=PROVENANCE_OWN_CHARACTERS,
    )
    assert record.disputed == (("−0.077", "0.077"),)


def test_a_spaced_ascii_hyphen_mid_line_is_a_range_not_a_sign() -> None:
    """``5 - 8`` in the layer must not manufacture ``-8`` that would clear a reading's ``-8``."""
    record = compare_numbers(
        "rose 5 - 8 percent", "rose -8 percent", provenance=PROVENANCE_OWN_CHARACTERS
    )
    assert [read for _ref, read in record.disputed] == ["-8"]


def test_a_hyphenated_label_does_not_manufacture_a_negative() -> None:
    """``Table A-8`` and ``t-1`` in the layer are labels: they must not clear ``-8`` / ``-1``."""
    record = compare_numbers(
        "Table A-8 lag t-1\nKorea -.8", "| Korea | -8 |", provenance=PROVENANCE_OWN_CHARACTERS
    )
    assert [read for _ref, read in record.disputed] == ["-8"]


def test_a_footnoted_cell_keeps_its_number() -> None:
    """``142^a`` is the cell 142 with a footnote letter, not a power: 142 is still checked."""
    record = compare_numbers(
        "Korea 198a", "| Korea | $142^a$ |", provenance=PROVENANCE_OWN_CHARACTERS
    )
    assert [read for _ref, read in record.disputed] == ["142"]


def test_currency_does_not_open_a_maths_span() -> None:
    """``-$8 billion to $1.2 billion`` keeps both amounts; a dropped point is caught."""
    record = compare_numbers(
        "Losses went from -$.8 billion to $1.2 billion.",
        "Losses went from -$8 billion to $1.2 billion.",
        provenance=PROVENANCE_OWN_CHARACTERS,
    )
    assert [read for _ref, read in record.disputed] == ["-8"]


def test_a_model_transcribed_quote_is_checked_and_socr_notes_are_not() -> None:
    """A quote the model read is part of the reading; socr's own quoted notes are not."""
    from socr.core.verdict import reading_numbers_text

    text = (
        "> **Unresolved chart placement on page 3** region 1, 2\n"
        "> continuation 7\n\n"
        "## Page 3\n\n"
        "> Growth was -8 percent.\n"
    )
    assert numeric_tokens(reading_numbers_text(text, 3), reading=True) == (("-8", "-8"),)


def test_a_numberless_reading_names_how_many_numbers_its_layer_prints() -> None:
    """Group (a): the layer prints numbers the reading lacks; the mark says how many."""
    from socr.core.verdict import NO_NUMBER_REASON, text_layer_numbers_clause

    p = PageState(page_num=1)
    p.is_born_digital = True
    p.own_layer_text = "Growth was 2.1 and 3.4 in 2017."
    some = text_layer_numbers_clause(
        PageOutput(page_num=1, text="Growth was steady.", status=PageStatus.SUCCESS, engine="qwen"),
        p,
    )
    assert some.result is CheckResult.UNKNOWN
    assert "the page's text layer prints 3" in some.reason and "#1218" in some.reason
    p.own_layer_text = "Growth was steady."
    none = text_layer_numbers_clause(
        PageOutput(page_num=1, text="Growth was steady.", status=PageStatus.SUCCESS, engine="qwen"),
        p,
    )
    assert none.reason == NO_NUMBER_REASON


def test_reading_mathematics_outside_tables_is_dropped_but_table_cell_kept() -> None:
    """Prose mathematics is dropped from the reading; a math table cell is kept."""
    assert numeric_tokens(r"we have $1 - \alpha$ and $x_{t+1}$", reading=True) == ()
    assert numeric_tokens(r"| a | $-0.04^{***}$ |", reading=True) == (("-0.04", "-0.04"),)


def test_lone_number_in_prose_mathematics_is_kept() -> None:
    """A math expression that is a lone number in running text is still a number."""
    assert ("-0.80", "-0.80") in numeric_tokens(r"the slope is $-0.80$ here", reading=True)


def test_interval_is_not_a_number() -> None:
    """(0,1) and [0,1] are intervals, not numbers."""
    assert numeric_tokens("on (0,1) and [0,1]") == ()


def _disputes(layer: str, reading: str) -> tuple:
    return compare_numbers(layer, reading, provenance=PROVENANCE_OWN_CHARACTERS).disputed


@pytest.mark.parametrize(
    ("layer", "reading"),
    [
        ("Cost was $20,000 in total.", "Cost was $2,000 in total."),
        ("Cost was $200,000 in total.", "Cost was $2,000 in total."),
        ("Output 19,500 units.", "Output 1,950 units."),
        ("Price 1,950.75 here.", "Price 1,950 here."),
        ("Rate 2015.8% here.", "Rate 2015% here."),
        ("In 20000 cases", "In 2000 cases"),
    ],
)
def test_the_year_callout_relation_needs_bare_numbers_on_both_sides(layer, reading) -> None:
    """Argon P0-2: the relation compared NORMALISED values, so a 10x or 100x error or a lost
    decimal passed as "a year with its callout". The same shape written bare still matches
    (``test_running_text_year_with_run_on_callout_matches``)."""
    assert _disputes(layer, reading)


def test_a_label_run_into_its_digit_is_split_on_both_sides() -> None:
    """Argon P0-3, its exact example: ``Q1..Q4`` and ``A-8`` split on the layer side only
    left free ``1..4`` and ``8`` for a dropped point (``.8`` read ``8``) to match."""
    layer = "Table A-8\nQ1 Q2 Q3 Q4\nKorea .8 .4"
    reading = "Table A-8\n| | Q1 | Q2 | Q3 | Q4 |\n| Korea | 8 | 4 |"
    assert _disputes(layer, reading) == ((".8", "8"), (".4", "4"))
    faithful = "Table A-8\n| | Q1 | Q2 | Q3 | Q4 |\n| Korea | .8 | .4 |"
    assert _disputes(layer, faithful) == ()


@pytest.mark.parametrize(
    "reading",
    [
        "<table><tr><td>Korea</td><td>$1.4 \\times 10$</td><td>142</td></tr></table>",
        "\\begin{tabular}{lll} Korea & $1.4 \\times 10$ & 142 \\end{tabular}",
        "| Korea | $1.4^1$ | 142 |",
        "| Korea | 1.4^{1} | 142 |",
    ],
)
def test_a_table_cell_is_compared_however_the_table_is_written(reading: str) -> None:
    """Argon P1-6: an HTML table's maths cells (tags stripped before the maths step), a
    ``tabular`` environment (dropped as maths) and a cell with a numeric callout (dropped as
    a power) were never compared. Layer ``1.3``, reading ``1.4``: named in each writing."""
    assert (("1.3", "1.4") in _disputes("Korea 1.3 10 142", reading)) or (
        _disputes("Korea 1.3 142", reading) == (("1.3", "1.4"),)
    )


@pytest.mark.parametrize("cell", ["$143^1$", "143^{1}", "143^1"])
def test_an_integer_cell_with_a_numeric_callout_keeps_its_number(cell: str) -> None:
    """Argon P1-6.3, its own case: an INTEGER cell with a numeric footnote was dropped
    whole as a power, so a wrong ``143`` for a printed ``142`` was never compared. (A
    decimal base keeps its number by another path, which is why the decimal rows above
    could not show it -- found by the mutant that removed this rule and survived them.)"""
    assert _disputes("Korea 142 7", f"| Korea | {cell} | 7 |") == (("142", "143"),)


@pytest.mark.parametrize("cell", ["$2 \\times 10^6$", "$9.60 \\times 10^4$"])
def test_a_power_of_ten_in_a_cell_stays_a_power(cell: str) -> None:
    """The narrowing measured on run 23319732: a layer prints ``2×106``; keeping the base
    ``10`` of ``10^6`` as the cell's number disputed it falsely."""
    assert _disputes("Obs. 2×106 9.60×104", f"| Obs. | {cell} |") == ()


#: A label a reading writes in maths with a command or a superscript is not claimed yet:
#: round 4 measured both claim designs on run 23319732 and each made false disputes (#1220).
_LATEX_LABEL_OPEN = pytest.mark.xfail(
    strict=True, reason="#1220: labels written in maths ($\\beta_1$, $R^2$) are not claimed"
)


@pytest.mark.parametrize(
    ("layer", "reading"),
    [
        ("β1 β2\nKorea .8 .1", "| | β1 | β2 |\n| Korea | .8 | 1 |"),
        ("σ2 σ3\nKorea .2 .3", "| | σ2 | σ3 |\n| Korea | 2 | .3 |"),
        ("1Q 2Q\nKorea .1 .2", "| | 1Q | 2Q |\n| Korea | 1 | 2 |"),
        ("1st 2nd\nKorea .1 .2", "| | 1st | 2nd |\n| Korea | 1 | 2 |"),
        ("Col.1 Col.2\nKorea .1 .2", "| | Col.1 | Col.2 |\n| Korea | 1 | 2 |"),
        ("Q1 Q2\nKorea .1 .2", "| | $Q_1$ | $Q_2$ |\n| Korea | 1 | 2 |"),
        # Round 4 (Fable, round-3 review): the writings readings actually use. The maths
        # writings stay open (#1220): both ways of claiming them measured false disputes.
        pytest.param(
            *("β1 β2\nKorea .8 .1", "| | $\\beta_1$ | $\\beta_2$ |\n| Korea | .8 | 1 |"),
            marks=_LATEX_LABEL_OPEN,
        ),
        pytest.param(
            *("σ2 σ3\nKorea .2 .3", "| | $\\sigma^2$ | $\\sigma^3$ |\n| Korea | 2 | .3 |"),
            marks=_LATEX_LABEL_OPEN,
        ),
        pytest.param(
            *("𝜔1\nKorea .1", "| | $\\omega_{1}$ |\n| Korea | 1 |"), marks=_LATEX_LABEL_OPEN
        ),
        pytest.param(
            *("xt-1\nKorea .1", "| | $x_{t-1}$ |\n| Korea | 1 |"), marks=_LATEX_LABEL_OPEN
        ),
        pytest.param(*("R2\nKorea .2", "| | $R^2$ |\n| Korea | 2 |"), marks=_LATEX_LABEL_OPEN),
        ("In Q1, growth was .1 percent.", "In Q1, growth was 1 percent."),
        ("Q4, Q1\nKorea .4 .1", "| | Q4, Q1 |\n| Korea | 4 | .1 |"),
        ("25th 75th\nRate 2.5 2.75", "| | 25th | 75th |\n| Rate | 25 | 2.75 |"),
        ("5th 95th\nRate .5 .95", "| | 5th | 95th |\n| Rate | 5 | .95 |"),
    ],
)
def test_every_label_writing_claims_the_digit_its_layer_frees(layer: str, reading: str) -> None:
    """Round-2 reviews (Fable P1-1, Argon P1-1b): the label rule knew only ASCII letters
    before a digit, so ``β1``, ``1Q``, ``1st``, ``Col.1`` and ``$Q_1$`` headers left the
    layer's digit free for a dropped point to match."""
    assert _disputes(layer, reading)


@pytest.mark.parametrize(
    ("layer", "cell"),
    [
        ("Korea 4.0 -.8", r"$\approx-8$"),
        ("Korea 4.0 .8", r"$\sim8$"),
        ("Korea 4.0 0.9", r"$\pm0.8$"),
        ("Korea 4.0 0.01", r"$\le0.05$"),
        ("Korea 4.0 .8", r"{\bf8}"),
    ],
)
def test_a_value_after_a_latex_command_is_never_a_label(layer: str, cell: str) -> None:
    """Argon P1-1a (round 2): the command's last letter made the value a label digit, which
    is never disputed -- a wrong number became undisputable."""
    assert _disputes(layer, f"| Korea | 4.0 | {cell} |")


@pytest.mark.parametrize(
    "reading",
    [
        "\\begin{tabular}{lll}\n\\hline\nKorea & $1.4 \\times 10$ & 142 \\\\\n\\hline\n\\end{tabular}",
        "\\begin{tabular}{lll}\n\\toprule\nKorea & $1.4$ & $142^1$ \\\\\n\\bottomrule\n\\end{tabular}",
        "\\begin{tabular}{@{}lcc@{}}\nKorea &\n$1.4$ & 142 \\\\\n\\end{tabular}",
    ],
)
def test_a_multi_line_tabular_row_stays_a_table_row(reading: str) -> None:
    """Argon P1-2 (round 2): a row after ``\\hline`` on its own line lost its leading ``|``,
    so its maths cells were dropped as prose. Layer ``1.3``, reading ``1.4``: disputed."""
    assert "1.4" in [read for _ref, read in _disputes("Korea 1.3 10 142", reading)]


def test_a_formula_in_a_table_cell_is_not_the_cells_number() -> None:
    """Round 3, measured: 2018 Tealbook p111 tabulates policy rules; the layer prints their
    maths as glyphs it cannot read, so the formula's coefficient ``0.5`` was a false dispute
    once its rows were read as table rows. An assignment of one number is still a value."""
    rule = r"| Taylor (1993) rule | $R_t = r^{LR} + \pi_t + 0.5 (\pi_t - \pi^{LR}) + 0.5 ygap_t$ |"
    assert _disputes("Taylor (1993) rule [math]", rule) == ()
    value = r"| Base | $\rho^{m} = 0.75$ |"
    assert _disputes("Base ρm = 0.70", value) == (("0.70", "0.75"),)


def test_only_a_short_integer_after_a_short_name_is_a_label() -> None:
    """Argon P1-1a (round 2): a label is never disputed, so the tag must never fall on a
    value -- a comma, decimal or percent number run into letters (``USD2,000``, ``ab2.5``,
    ``x15%``) or one after a long word (``Korea-8``) is not a label."""
    from socr.tables.reading_agreement import _numeric_tokens

    tokens = _numeric_tokens(
        "| USD2,000 | ab2.5 | x15% | Korea-8 | Q1 | $2 \\times 10^{6}$ |", True
    )
    assert [written for _value, written, label in tokens if label] == ["1"]


@pytest.mark.parametrize(
    "reading",
    [
        "Korea -.8 2.8\n\n> **Note:** The rate was 9.31 in 2019.",
        "Korea -.8 2.8\n\n> [1] The rate was 9.31 in 2019.",
        "> **Table 3 Growth**\n>\n> | Korea | -8 | 2.8 |\n> | Rate | 9.31 | |",
    ],
)
def test_a_model_quote_with_a_bold_or_bracket_opener_is_checked(reading: str) -> None:
    """socr bot, PR #1219 decision: the openers ``> **`` and ``> [`` dropped a model's own
    quote whole, including a quoted table under a bold title shipping a wrong number. Only
    socr's exact openers are skipped now; the layer prints ``[1]`` and ``Table 3`` so only
    the wrong numbers are disputed."""
    layer = "Table 3 Growth Korea -.8 2.8 [1] Note: The rate was 0.31 in 2019. Rate 0.31"
    p = _page(layer)
    clause = text_layer_numbers_clause(_qwen(reading), p)
    assert clause is not None and clause.result is CheckResult.FAIL, clause
    assert "9.31" in clause.reason


@pytest.mark.parametrize(
    "note",
    [
        "> **Chart region on page 1:** values 9.9 7.7",
        "> **Chart grids on page 1:** 9.9",
        "> **Unresolved chart placement on page 1:** 9.9",
        "> *Figure description (page 1): 9.9 percent*",
        "> [table unverified: socr could not check this table against the page; 9.9]",
        "> [table disputed: a second reading of the page (Tesseract) read 9.9; check]",
    ],
)
def test_socrs_own_quote_openers_are_still_skipped(note: str) -> None:
    """The other face: every opener socr itself emits stays out of the reading."""
    assert text_layer_numbers_clause(_qwen(f"{note}\n\n{TEALBOOK_READING_OK}"), _page()) is None


def test_a_percent_written_in_latex_is_one_number_in_a_cell() -> None:
    r"""socr bot, round-3 review (a round-3 regression): ``$\pi^* = 9\%$`` was dropped whole
    as a formula, so a wrong target was silent; ``2\,\%`` must still agree."""
    layer = "Target 2 % 1.5"
    assert ("2", "9%") in _disputes(layer, r"| target | $\pi^* = 9\%$ | 1.5 |")
    assert _disputes(layer, r"| target | $\pi^* = 2\,\%$ | 1.5 |") == ()


_CELL_ROW = "| target | {} | 1.5 |"
#: prose keeps only a lone number from its mathematics, so no equation rows here
_PROSE_ROW = "Target {} and 1.5"


@pytest.mark.parametrize(
    ("row", "cell"),
    [
        (_CELL_ROW, r"$\pi^* = {v}\,\%$"),
        (_CELL_ROW, r"$\pi^* = {v} \%$"),
        (_CELL_ROW, r"$\pi^* = {v}~\%$"),
        (_CELL_ROW, r"{v}~\%"),
        (_CELL_ROW, r"${v} \%$"),
        (_PROSE_ROW, r"${v} \%$"),
        (_PROSE_ROW, r"${v}~\%$"),
        (_PROSE_ROW, r"{v}~\%"),
    ],
)
def test_a_spaced_latex_percent_cell_is_counted(row: str, cell: str) -> None:
    r"""#1227: a percent spaced from its number (``\,``, a plain space, a tie ``~``) is the
    number, in a table cell and in prose. Agreement alone cannot tell a counted cell from a
    dropped one, so the right value must add one compared number and a planted wrong one
    must be caught."""
    layer = "Target 2 % 1.5"
    baseline = compare_numbers(layer, row.format(""), provenance=PROVENANCE_OWN_CHARACTERS)
    right = row.format(cell.format(v=2))
    record = compare_numbers(layer, right, provenance=PROVENANCE_OWN_CHARACTERS)
    assert record.compared == baseline.compared + 1
    assert text_layer_numbers_clause(_qwen(right), _page(layer)) is None
    wrong = row.format(cell.format(v=9))
    assert [read for _ref, read in _disputes(layer, wrong)] == ["9%"]
    clause = text_layer_numbers_clause(_qwen(wrong), _page(layer))
    assert clause is not None and clause.result is CheckResult.FAIL, clause


@pytest.mark.parametrize(
    ("layer", "spaced", "unspaced"),
    [
        ("x 9 1.5", r"| x | (9 \%) | 1.5 |", r"| x | (9\%) | 1.5 |"),
        ("x 9 1.5", r"| x | $(9~\%)$ | 1.5 |", r"| x | $(9\%)$ | 1.5 |"),
        ("x 9 1.5", r"| x | $\pi^* = (9 \%)$ | 1.5 |", r"| x | $\pi^* = (9\%)$ | 1.5 |"),
        ("in 1995.5 the", r"in 1995 \% the", r"in 1995\% the"),
    ],
)
def test_a_spaced_latex_percent_reads_as_the_unspaced_one(
    layer: str, spaced: str, unspaced: str
) -> None:
    r"""Argon, PR #1229 review: spacing removed only inside the equation check left ``9 %``
    to split later, so a bracket was lost (``(9 \%)`` agreed with a bare 9) and a year's
    percent passed the callout guard. The spaced writing must compare exactly as ``9\%``
    does, and here that is a dispute."""
    record = compare_numbers(layer, spaced, provenance=PROVENANCE_OWN_CHARACTERS)
    assert record == compare_numbers(layer, unspaced, provenance=PROVENANCE_OWN_CHARACTERS)
    assert record.disputed
    clause = text_layer_numbers_clause(_qwen(spaced), _page(layer))
    assert clause is not None and clause.result is CheckResult.FAIL, clause


def test_an_escaped_tilde_before_a_percent_stays_unparsed() -> None:
    r"""Argon, PR #1229 review: ``\~`` is an accent, not a tie; ``3\~\%`` must drop out (fail
    closed), not read as 3."""
    layer = "x 3 1.5"
    baseline = compare_numbers(layer, "| x |  | 1.5 |", provenance=PROVENANCE_OWN_CHARACTERS)
    record = compare_numbers(layer, r"| x | 3\~\% | 1.5 |", provenance=PROVENANCE_OWN_CHARACTERS)
    assert record.compared == baseline.compared


def test_a_number_in_markdown_italics_is_the_number() -> None:
    """socr bot, round-1 review: ``_12.5_`` lost its integer part as a subscript, a false
    dispute on a right number and a wrong ``_13.5_`` passing on a page that prints ``.5``."""
    assert _disputes("growth 12.5", "growth _12.5_") == ()
    assert [read for _ref, read in _disputes("0.5 12.5 -.8 2.8", "| _13.5_ | -.8 | 2.8 |")] == [
        "13.5"
    ]


def test_unknown_provenance_is_rejected() -> None:
    """compare_numbers refuses a provenance it does not know."""
    with pytest.raises(ValueError):
        compare_numbers("1.5", "1.5", provenance="hearsay")


# --- clause -----------------------------------------------------------------------------


def test_clause_is_none_when_numbers_agree() -> None:
    """A reading whose numbers occur in the layer raises no clause."""
    agreeing = "| Korea | 4.0 | 2.6 | 5.7 | -.8 | 4.0 | 3.3 |"
    assert text_layer_numbers_clause(_qwen(agreeing), _page()) is None


def test_clause_fails_and_names_the_dispute_on_a_born_digital_layer() -> None:
    """A disagreement on a born-digital layer is a FAIL that quotes both numbers."""
    clause = text_layer_numbers_clause(_qwen(TEALBOOK_READING), _page())
    assert clause is not None
    assert clause.check is Check.TEXT_LAYER_NUMBERS
    assert clause.result is CheckResult.FAIL
    assert "text layer says -.8, reading says -8" in clause.reason


def test_damaged_layer_turns_the_disagreement_into_unknown() -> None:
    """A layer the detector calls damaged may be at fault: UNKNOWN, still naming the dispute."""
    clause = text_layer_numbers_clause(_qwen(TEALBOOK_READING), _page(born_digital=False))
    assert clause is not None
    assert clause.result is CheckResult.UNKNOWN
    assert clause.reason.startswith(DAMAGED_LAYER_PREFIX)
    assert "reading says -8" in clause.reason


def test_minus_as_digit_hits_turns_the_disagreement_into_unknown() -> None:
    """A minus-as-digit finding on a born-digital layer also makes the layer suspect."""
    p = _page()
    p.minus_as_digit_hits = 1
    clause = text_layer_numbers_clause(_qwen(TEALBOOK_READING), p)
    assert clause is not None
    assert clause.result is CheckResult.UNKNOWN


@pytest.mark.parametrize("flag", ["refused", "minus_as_digit_hits", "control_byte_scan_failed"])
def test_agreement_with_a_refused_or_damaged_layer_is_not_a_check(flag: str) -> None:
    """Round 2 (Argon P0-4, overruling the round-1 asymmetry): the same agreeing reading is
    CHECKED on a clean born-digital layer and UNKNOWN on one the detector refused or found
    damaged -- the layer may be an earlier OCR run's that the analysis could not name."""
    agreeing = "| Korea | 4.0 | 2.6 | 5.7 | -.8 | 4.0 | 3.3 |"
    clean = _page()
    suspect = _page(born_digital=flag != "refused")
    if flag != "refused":
        setattr(suspect, flag, 1 if flag.endswith("hits") else True)
    assert text_layer_numbers_clause(_qwen(agreeing), clean) is None
    clause = text_layer_numbers_clause(_qwen(agreeing), suspect)
    assert (clause.result, clause.reason) == (CheckResult.UNKNOWN, DAMAGED_LAYER_AGREES_REASON)


def test_a_cosmetic_layer_checks_an_agreement_and_softens_a_dispute() -> None:
    """#136's cosmetic class (lost spaces, fused words) carries the page's own digits, so an
    agreement is a CHECK; a dispute may be a fused pair hiding the number, so UNKNOWN."""
    agreeing = "| Korea | 4.0 | 2.6 | 5.7 | -.8 | 4.0 | 3.3 |"
    cosmetic = _page()
    cosmetic.has_encoding_hygiene_suspect = True
    assert text_layer_numbers_clause(_qwen(agreeing), cosmetic) is None
    clean = text_layer_numbers_clause(_qwen(TEALBOOK_READING), _page())
    soft = text_layer_numbers_clause(_qwen(TEALBOOK_READING), cosmetic)
    assert (clean.result, soft.result) == (CheckResult.FAIL, CheckResult.UNKNOWN)
    assert soft.reason == DAMAGED_LAYER_PREFIX + clean.reason


def test_no_own_text_layer_is_unknown() -> None:
    """A page with no text layer of its own cannot confirm its numbers: UNKNOWN."""
    clause = text_layer_numbers_clause(_qwen(TEALBOOK_READING), _page(layer=""))
    assert clause is not None
    assert clause.result is CheckResult.UNKNOWN
    assert clause.reason == NO_OWN_TEXT_REASON


def test_reading_without_a_number_is_unknown() -> None:
    """A reading that ships no number gives vacuous agreement, so it is UNKNOWN; the page's
    layer prints numbers here, so the mark says how many (group (a), #1218)."""
    from socr.core.verdict import NO_NUMBER_LAYER_HAS_SOME_REASON

    clause = text_layer_numbers_clause(_qwen("A reading with no figures at all."), _page())
    assert clause is not None
    assert clause.result is CheckResult.UNKNOWN
    assert clause.reason == NO_NUMBER_LAYER_HAS_SOME_REASON.format(count=6)
    assert clause.reason != NO_NUMBER_REASON


def test_prior_ocr_page_is_left_to_the_prior_ocr_clause() -> None:
    """#1156: a prior-OCR layer is not this clause's to judge."""
    p = _page()
    p.prior_ocr_layer_suspect = True
    assert text_layer_numbers_clause(_qwen(TEALBOOK_READING), p) is None


@pytest.mark.parametrize("engine", ["native", "chart_asset"])
def test_non_model_engines_are_not_compared(engine: str) -> None:
    """A native lane or a chart picture is not a model reading of the page's characters."""
    output = PageOutput(page_num=1, text=TEALBOOK_READING, status=PageStatus.SUCCESS, engine=engine)
    assert text_layer_numbers_clause(output, _page()) is None


def test_socrs_own_quote_lines_are_not_the_reading() -> None:
    """socr's blockquote notes carry socr's numbers (99, 1), which must not be compared."""
    output = _qwen("> [page 1 unchecked: chart_region=FAIL; scope=page; reason=x 99]\n\nKorea -.8")
    assert text_layer_numbers_clause(output, _page(layer="Korea -.8")) is None


# --- seam -------------------------------------------------------------------------------


def test_disagreeing_success_page_is_demoted_and_marked() -> None:
    """A SUCCESS page whose reading disagrees with its layer is WARNING, marked and explained."""
    out, _verdict = _apply_verdict_marks(_qwen(TEALBOOK_READING), _page(), 1)
    assert out.status is PageStatus.WARNING
    assert out.failure_mode is FailureMode.NUMBERS_DISAGREE_WITH_TEXT_LAYER
    assert out.text.startswith(
        "> [page 1 unchecked: text_layer_numbers=FAIL; scope=page; "
        "reason=text layer says -.8, reading says -8]"
    )


def test_applying_the_seam_twice_is_idempotent() -> None:
    """Re-running the seam over its own output changes nothing and never falls back."""
    p = _page()
    first_out, first_verdict = _apply_verdict_marks(_qwen(TEALBOOK_READING), p, 1)
    second_out, second_verdict = _apply_verdict_marks(first_out, p, 1)
    assert second_out.text == first_out.text
    assert second_out.status is first_out.status
    assert second_out.failure_mode is first_out.failure_mode
    assert second_verdict.clauses == first_verdict.clauses
    assert all(c.check is not Check.UNKNOWN_CHECK for c in second_verdict.clauses)


def test_unknown_case_demotes_with_numbers_unchecked() -> None:
    """With no own text layer the page is demoted with the NUMBERS_UNCHECKED mode."""
    out, _verdict = _apply_verdict_marks(_qwen(TEALBOOK_READING), _page(layer=""), 1)
    assert out.status is PageStatus.WARNING
    assert out.failure_mode is FailureMode.NUMBERS_UNCHECKED


def test_disagreement_is_permanent_across_resume() -> None:
    """The demotion's verdict is one a resume may skip-and-keep rather than re-OCR forever."""
    out, verdict = _apply_verdict_marks(_qwen(TEALBOOK_READING), _page(), 1)
    assert _resume_verdict_is_permanently_unverifiable(verdict.to_dict(), out.failure_mode.value)


def test_unchecked_is_permanent_across_resume() -> None:
    """The UNKNOWN demotion is equally permanent across resume."""
    out, verdict = _apply_verdict_marks(_qwen(TEALBOOK_READING), _page(layer=""), 1)
    assert _resume_verdict_is_permanently_unverifiable(verdict.to_dict(), out.failure_mode.value)


@pytest.mark.parametrize("engine", ["qwen", "native"])
def test_a_prior_ocr_page_is_never_checked_by_either_clause(engine: str) -> None:
    """#1185 x #1156 (round 5, after #1217 merged): on a page whose layer is an earlier OCR
    run, this clause stands aside and #1156's marks the page. Difference: the same reading on
    a page WITHOUT the flag is FAILed by this clause (a model reading) or raises nothing (a
    native one); with the flag, only #1156's clause is there and the page is not CHECKED."""
    # A DISAGREEING reading (socr bot, round-5 note): with an agreeing one this clause
    # raises nothing anyway, so the test could not show that it stands aside.
    output = PageOutput(page_num=1, text=TEALBOOK_READING, status=PageStatus.SUCCESS, engine=engine)
    flagged = _page()
    flagged.prior_ocr_layer_suspect = True
    shipped, verdict = _apply_verdict_marks(output, flagged, 1)
    checks = [c.check for c in verdict.clauses]
    assert checks == [Check.PRIOR_OCR_LAYER], checks
    assert shipped.status is not PageStatus.SUCCESS
    _clean_shipped, clean_verdict = _apply_verdict_marks(output, _page(), 1)
    expected = [Check.TEXT_LAYER_NUMBERS] if engine == "qwen" else []
    assert [c.check for c in clean_verdict.clauses] == expected
