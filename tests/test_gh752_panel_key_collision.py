"""#752: two panels sharing a label must not silently collapse in the
ground-truth scorer.

`_reader_readings` and `_model_readings` used to key their output by panel
label -- a plain `dict[str, ...]`. A real release measured the cost: two
panels sharing a heading collapsed onto one key, three of four panels on a
page were overwritten, and 120 cells vanished from the report with no trace
(not even `no_ground_truth`). Both sites are exercised here with synthetic
fixtures only -- this repo is public and the SEP corpus is copyrighted, so no
corpus content appears below.
"""

from __future__ import annotations

import socr.figures.score_sep_ground_truth as scorer
from socr.figures.chart_reader import (
    INTEGER,
    PRESENT,
    Cell,
    Frame,
    PageReading,
    PanelReading,
    SeriesReading,
    YCalibration,
)
from socr.figures.score_sep_ground_truth import (
    EXACT,
    _model_readings,
    _reader_readings,
    _score_side,
)
from socr.figures.sep_ground_truth import GroundTruthCell, GroundTruthPanel, ReleaseTable

_CAL = YCalibration(points_per_unit=1.0, zero_y=0.0, pairs=(), residual=0.0, checked_ticks=2)
_FRAME = Frame(baseline=0.0, x0=0.0, x1=1.0, tick_ys=(0.0, 1.0))


def _cell(bin_label: str, count: int) -> Cell:
    return Cell(
        bin_label=bin_label,
        status=INTEGER,
        count=count,
        interval=(count - 0.5, count + 0.5),
        detail="",
    )


def _series(name: str, counts: dict[str, int]) -> SeriesReading:
    return SeriesReading(
        name=name,
        style="solid",
        presence=PRESENT,
        cells=tuple(_cell(b, c) for b, c in counts.items()),
    )


def _panel(region_index: int, label: str, counts: dict[str, int]) -> PanelReading:
    return PanelReading(
        page_num=1,
        region_index=region_index,
        label=label,
        bins=(),
        series=(_series("December projections", counts),),
        calibration=_CAL,
        frame=_FRAME,
    )


# ---------------------------------------------------------------------------
# _reader_readings
# ---------------------------------------------------------------------------


def test_reader_readings_keeps_two_panels_that_share_a_label(monkeypatch, tmp_path) -> None:
    """The #752 shape: two `PanelReading`s from the same page share a label.
    A dict keyed by label would collapse them into one entry, discarding the
    first panel's cells entirely."""
    page_reading = PageReading(
        page_num=1,
        panels={
            0: _panel(0, "2020", {"0.13|0.37": 1}),
            1: _panel(1, "2020", {"0.13|0.37": 2}),
        },
    )
    monkeypatch.setattr(scorer, "open_pdf", lambda _path: _FakeDoc())
    monkeypatch.setattr(scorer, "chart_region_bboxes", lambda _page: [])
    monkeypatch.setattr(scorer, "read_chart_page", lambda *a, **k: page_reading)

    readings = _reader_readings(tmp_path / "sep-20201216-p09.pdf")

    assert [label for label, _by_series in readings] == ["2020", "2020"]
    counts = [by_series["December projections"]["0.13|0.37"] for _label, by_series in readings]
    assert sorted(counts) == [1, 2]


class _FakePage:
    pass


class _FakeDoc:
    def __getitem__(self, _idx: int) -> _FakePage:
        return _FakePage()

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# _model_readings
# ---------------------------------------------------------------------------

_TWO_PANELS_SAME_HEADING_MD = """### 2020
| Percent range | December projections |
| :--- | :---: |
| 0.13-0.37 | 1 |

### 2020
| Percent range | December projections |
| :--- | :---: |
| 0.13-0.37 | 2 |
"""


def test_model_readings_keeps_two_grids_under_the_same_heading(tmp_path) -> None:
    (tmp_path / "doc.md").write_text(_TWO_PANELS_SAME_HEADING_MD)

    readings = _model_readings(tmp_path, "doc")

    assert [label for label, _by_series in readings] == ["2020", "2020"]
    counts = [by_series["December projections"]["0.13|0.37"] for _label, by_series in readings]
    assert sorted(counts) == [1, 2]


# ---------------------------------------------------------------------------
# _score_side -- pin the DIFFERENCE a collision used to cause
# ---------------------------------------------------------------------------


def _truth() -> ReleaseTable:
    return ReleaseTable(
        release_date="20201216",
        panels={
            "2020": GroundTruthPanel(
                year_label="2020",
                columns={"December projections": (GroundTruthCell("0.13-0.37", 1),)},
            )
        },
    )


def test_score_side_scores_every_panel_even_when_labels_collide() -> None:
    """Same total input either way -- only the labels differ between the two
    panels. Distinct labels ("2020", "2021") were never at risk; colliding
    labels ("2020", "2020") are the #752 shape. Both must score the same
    number of cells: the collision must cost nothing."""
    distinct = [
        ("2020", {"December projections": {"0.13|0.37": 1}}),
        ("2021", {"December projections": {"0.13|0.37": 2}}),
    ]
    colliding = [
        ("2020", {"December projections": {"0.13|0.37": 1}}),
        ("2020", {"December projections": {"0.13|0.37": 2}}),
    ]
    rt = ReleaseTable(
        release_date="20201216",
        panels={
            "2020": GroundTruthPanel(
                year_label="2020",
                columns={"December projections": (GroundTruthCell("0.13-0.37", 1),)},
            ),
            "2021": GroundTruthPanel(
                year_label="2021",
                columns={"December projections": (GroundTruthCell("0.13-0.37", 2),)},
            ),
        },
    )

    distinct_scores = _score_side("doc", distinct, rt)
    colliding_scores = _score_side("doc", colliding, rt)

    # Both panels, in both cases, contribute one cell each -- nothing is
    # dropped because two panels happen to share a label.
    assert len(distinct_scores) == 2
    assert len(colliding_scores) == 2
    # Distinct labels: each panel matches its own truth panel, both exact.
    assert [s.outcome for s in distinct_scores] == [EXACT, EXACT]
    # Colliding labels: both panels match the SAME truth panel ("2020"), so
    # the second panel's 2 != truth's 1 reads as a genuine miscount -- a real,
    # visible outcome, not a vanished cell.
    assert colliding_scores[0].outcome == EXACT
    assert colliding_scores[1].truth == 1
    assert colliding_scores[1].value == 2
