"""GH-369: a chart-asset page must not ship axis tick scales as body prose.

The lane saves the chart as an image and records "data values not
transcribed", then shipped ``ps.native_text`` whole -- on the reported page
(FOMC Minutes 17-18 March 2015, p24) 52% of non-empty lines were a lone
number. A column of ``2 4 6 ... 18`` in a document about economic projections
is indistinguishable from a series of values downstream, and the page carried
a clean SUCCESS with no flag.

The residue is FENCED, never dropped: the repo's rule is that a wrong or
dropped number is worse than a missing one, so every input line survives in
the output. Interleaving is deliberately lost when the residue moves to the
trailing fence, so the page is not byte-reconstructible -- the guarantee is
that no line is dropped, not that the original order survives. There is no
share-of-page
threshold in this path -- each line is judged on its own content, so nothing
here depends on a ratio measured on one document.
"""

from __future__ import annotations

from socr.figures.extractor import (
    fence_chart_axis_residue,
    split_chart_axis_residue,
)

# The reported page's shape: caption + axis label + tick scale + word ticks.
_REPORTED_PAGE = "\n".join(
    [
        "Figure 4. Uncertainty and risks in economic projections",
        "Uncertainty about GDP growth",
        "Number of participants",
        "2",
        "4",
        "6",
        "8",
        "10",
        "Lower",
        "Broadly",
        "similar",
        "Higher",
        "March projections",
        "December projections",
    ]
)


class TestNothingIsLost:
    def test_every_input_line_survives_somewhere(self) -> None:
        """The load-bearing guarantee. Suppression was the other candidate fix
        and was rejected precisely because it drops numbers."""
        body, residue = split_chart_axis_residue(_REPORTED_PAGE)
        recovered = sorted(body.splitlines() + residue)
        assert recovered == sorted(_REPORTED_PAGE.splitlines())

    def test_fenced_output_still_contains_every_original_line(self) -> None:
        fenced = fence_chart_axis_residue(_REPORTED_PAGE)
        for line in _REPORTED_PAGE.splitlines():
            assert line in fenced, f"line vanished from the fenced output: {line!r}"


class TestTickScaleLeavesTheBody:
    def test_bare_numbers_are_not_body_prose(self) -> None:
        body, residue = split_chart_axis_residue(_REPORTED_PAGE)
        assert residue == ["2", "4", "6", "8", "10"]
        for number in residue:
            assert number not in body.splitlines()

    def test_caption_and_word_ticks_stay_in_the_body(self) -> None:
        """Word-shaped labels are ordinary prose -- they do not read as data,
        and a classifier aggressive enough to take them would take sentences."""
        body, _residue = split_chart_axis_residue(_REPORTED_PAGE)
        for kept in (
            "Figure 4. Uncertainty and risks in economic projections",
            "Number of participants",
            "Lower",
            "Broadly",
            "March projections",
        ):
            assert kept in body.splitlines()


class TestNonChartPagesAreUntouched:
    def test_prose_without_bare_numbers_is_byte_identical(self) -> None:
        """The 24 non-chart pages of the same document score 0%. Anything but
        an exact passthrough here would churn every page in the corpus."""
        prose = "The Committee agreed that the pace of 3 percent growth\nwould continue."
        assert fence_chart_axis_residue(prose) == prose

    def test_number_inside_a_sentence_is_not_residue(self) -> None:
        """Only a line that is ENTIRELY a number counts. An inline figure is
        data and must never be fenced away from its sentence."""
        _body, residue = split_chart_axis_residue("unemployment fell to 4.9 in March")
        assert residue == []

    def test_empty_input_is_unchanged(self) -> None:
        assert fence_chart_axis_residue("") == ""


class TestNumericShapes:
    def test_signed_decimal_and_percent_tick_labels_are_residue(self) -> None:
        """Axis scales are not always bare integers."""
        _body, residue = split_chart_axis_residue("-0.5\n2.5\n25%\n+3")
        assert residue == ["-0.5", "2.5", "25%", "+3"]

    def test_a_lone_word_is_never_residue(self) -> None:
        _body, residue = split_chart_axis_residue("Higher\nsimilar")
        assert residue == []


class TestFenceIsMachineDistinguishable:
    def test_residue_sits_inside_a_marked_fence(self) -> None:
        """The whole point: a downstream consumer must be able to tell axis
        furniture from data without re-deriving the classification."""
        fenced = fence_chart_axis_residue(_REPORTED_PAGE)
        assert "socr:chart-axis-residue" in fenced
        assert "socr:end-chart-axis-residue" in fenced

        head, _, tail = fenced.partition("socr:chart-axis-residue")
        for number in ("2", "4", "6", "8", "10"):
            assert f"\n{number}\n" in tail
        assert "Figure 4. Uncertainty and risks in economic projections" in head


class TestALoneNumberIsContentNotFurniture:
    """cubic P2 on PR #383. The first draft fenced ANY line that was only a
    number, so a genuine standalone value on a chart page -- a year, a headline
    stat, a footnote marker -- vanished from the visible prose into the trailing
    comment. That is the silent content change the fencing design exists to
    avoid, arriving by a different door.

    An axis scale is a SEQUENCE of ticks, so only a run of consecutive
    bare-number lines is furniture. There is still no page-level ratio anywhere.
    """

    def test_a_lone_year_stays_visible(self) -> None:
        body, residue = split_chart_axis_residue("Published in\n2025\nby the Board")
        assert residue == []
        assert "2025" in body.splitlines()

    def test_a_lone_headline_stat_stays_visible(self) -> None:
        body, residue = split_chart_axis_residue("unemployment\n4.9\nin March")
        assert residue == []
        assert "4.9" in body.splitlines()

    def test_a_lone_percent_fact_stays_visible(self) -> None:
        body, residue = split_chart_axis_residue("share of respondents\n25%\nsaid yes")
        assert residue == []
        assert "25%" in body.splitlines()

    def test_a_run_of_ticks_is_still_fenced(self) -> None:
        """The control: the fix must not disarm the actual bug."""
        _body, residue = split_chart_axis_residue("caption\n2\n4\ntail")
        assert residue == ["2", "4"]

    def test_a_lone_number_leaves_the_page_completely_unfenced(self) -> None:
        """End to end: a chart page whose only numeric line is content gains no
        fence at all, so its rendered output is unchanged."""
        page = "Figure 1. Outlook\nPublished 2025\n2025\nSee notes"
        assert fence_chart_axis_residue(page) == page

    def test_two_separated_singles_are_not_a_run(self) -> None:
        """Adjacency is what makes a scale. Two lone numbers with prose between
        them are two facts, not a two-tick axis."""
        _body, residue = split_chart_axis_residue("in 2024\n2024\ngrowth was\n4.9\npercent")
        assert residue == []
