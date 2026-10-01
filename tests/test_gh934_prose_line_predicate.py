"""GH-934: a line-level prose/caption predicate for the stub-first header band.

A header row whose first word is the row stub ("Country | Est | SE | R2") is
absorbed into the header band (GH-925). The rejected GH-925 attempt also
absorbed captions and footnote lines. The predicate here
(``_is_prose_like_row``) is applied to every row past the point where main's
all-snap walk stops, at the single shared walk ``_header_band_ys``.

Every test is a DIFFERENCE pin: the same page is rowized twice, changing only
the one thing under test (a clause, or the whole stub exemption), and the two
outputs are asserted to differ (or to be identical, for the controls).
"""

from __future__ import annotations

import pytest

from socr.tables import reconstruct
from socr.tables.reconstruct import rowize_from_word_list

LANES = (250.0, 330.0, 410.0)
DATA_SIZE = 9.0


class Page:
    """Word builder: every row gets its own (block, line) so word keys are unique."""

    def __init__(self) -> None:
        self.words: list = []
        self.sizes: dict = {}
        self._line = 0

    def row(self, y: float, items: list[tuple[float, str, float]], size: float = DATA_SIZE) -> None:
        """items: (x0, text, width)."""
        self._line += 1
        for i, (x, text, width) in enumerate(items):
            self.words.append((x, y, x + width, y + 10, text, 0, self._line, i))
            self.sizes[(0, self._line, i)] = size

    def prose(self, y: float, n: int = 12) -> None:
        """A body-text line with ordinary word spacing: sets the page word space."""
        self.row(y, [(60.0 + i * 34.5, f"word{i}", 30.0) for i in range(n)])

    def data(self, y0: float = 140.0) -> None:
        for i in range(4):
            y = y0 + 16.0 * i
            self.row(
                y,
                [
                    (60.0, f"Nation{'abcd'[i]}", 26.0),
                    (250.0, f"{i}.11", 26.0),
                    (330.0, f"{i}.22", 26.0),
                    (410.0, f"0.{i}5", 26.0),
                ],
            )


def _run(page: Page, *, sizes: bool = True) -> list[list[str]]:
    regions = rowize_from_word_list(page.words, word_sizes=page.sizes if sizes else None)
    assert regions, "fixture must reconstruct"
    return [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in regions[0][1].splitlines()
        if line.lstrip().startswith("|") and "---" not in line
    ]


STUB_HEADER = [
    (60.0, "Country", 40.0),
    (250.0, "Est", 26.0),
    (330.0, "SE", 26.0),
    (410.0, "R2", 26.0),
]
HEADER_ROW = ["Country", "Est", "SE", "R2"]


def _page(*, above=None, above_size: float = DATA_SIZE, header=STUB_HEADER, y_above=86.0) -> Page:
    p = Page()
    if above:
        p.row(y_above, above, size=above_size)
    if header:
        p.row(100.0, header)
    p.data()
    p.prose(300.0)
    p.prose(312.0)
    return p


# A caption whose every word snaps to a lane and is set as ONE run (4pt gaps, wide words).
ALL_SNAP_CAPTION = [
    (250.0, "Descriptive", 76.0),
    (330.0, "statistics", 76.0),
    (410.0, "summary", 76.0),
]
# A prose line: label-region words, the last one snapping to the first lane. One run.
PROSE_LINE = [(250.0 - 34.0 * k, f"text{k}", 30.0) for k in range(4, -1, -1)]


def _flat(rows) -> list[str]:
    return [c for r in rows for c in r]


# --------------------------------------------------------------------- recovery


def test_stub_first_header_is_recovered(monkeypatch) -> None:
    """Difference pin: only the stub exemption differs from main's rule."""
    after = _run(_page())
    monkeypatch.setattr(reconstruct, "_stub_row_eligible", lambda *a, **k: False)
    before = _run(_page())
    assert after[0] == HEADER_ROW
    assert "Country" not in _flat(before)
    assert after[1:] == before[1:] or len(before) == len(after) - 1


def test_data_rows_identical_with_and_without_stub_header() -> None:
    assert _run(_page())[1:] == _run(_page(header=None))[1:]


# --------------------------------------------------------------------- rejection


def test_caption_above_band_is_rejected_single_run_clause(monkeypatch) -> None:
    """Difference pin on the single-run clause: a label-region caption over the band."""
    page = lambda: _page(above=PROSE_LINE)  # noqa: E731
    with_pred = _run(page())
    assert "text0" not in _flat(with_pred) and with_pred[0] == HEADER_ROW
    monkeypatch.setattr(reconstruct, "_is_prose_like_row", lambda *a, **k: False)
    without = _run(page())
    assert "text0" in _flat(without)


def test_all_snap_caption_above_stub_header_is_rejected_by_reach(monkeypatch) -> None:
    """The Kalemli shape. Main's own all-snap rule would admit this caption once the
    stub row has been recovered; only testing EVERY row past main's stop rejects it."""
    page = lambda: _page(above=ALL_SNAP_CAPTION)  # noqa: E731
    with_pred = _run(page())
    # the caption is not absorbed, and (lane-shaped row rejected above the stub row)
    # the stub recovery is discarded: main's behaviour
    assert "Descriptive" not in _flat(with_pred) and "Country" not in _flat(with_pred)
    monkeypatch.setattr(reconstruct, "_is_prose_like_row", lambda *a, **k: False)
    assert "Descriptive" in _flat(_run(page()))


def test_prose_line_with_wide_sentence_space_is_rejected_by_size_clause(monkeypatch) -> None:
    """A footnote whose sentence space exceeds the run threshold survives the
    single-run clause; its smaller font is the only thing that rejects it."""
    gapped = [(250.0 - 60.0 * k, f"foot{k}", 30.0) for k in range(4, -1, -1)]  # 30pt gaps
    page = lambda sizes=True: _page(above=gapped, above_size=8.0)  # noqa: E731
    assert "foot0" not in _flat(_run(page())) and _run(page())[0] == HEADER_ROW
    # no size information: the clause is inert, the line is absorbed
    assert "foot0" in _flat(_run(page(), sizes=False))
    # the single-run clause alone does not reject it
    monkeypatch.setattr(reconstruct, "_median_word_size", lambda *a, **k: None)
    assert "foot0" in _flat(_run(page()))


# --------------------------------------------------------------------- controls


def test_plain_header_without_stub_is_unchanged(monkeypatch) -> None:
    plain = STUB_HEADER[1:]
    base = _run(_page(header=plain))
    monkeypatch.setattr(reconstruct, "_is_prose_like_row", lambda *a, **k: False)
    monkeypatch.setattr(reconstruct, "_stub_row_eligible", lambda *a, **k: False)
    assert _run(_page(header=plain)) == base
    assert base[0] == ["", "Est", "SE", "R2"]


def test_main_absorption_is_not_changed_by_the_predicate(monkeypatch) -> None:
    """Rows main's all-snap walk absorbs (no stub word anywhere) are untouched, even
    when single-run: the predicate only applies past main's stopping point."""
    upper = [(250.0, "Panel", 76.0), (330.0, "A", 76.0), (410.0, "results", 76.0)]
    page = lambda: _page(header=STUB_HEADER[1:], above=upper)  # noqa: E731
    base = _run(page())
    assert any("Panel" in c for c in _flat(base))  # collapsed into the header cells
    monkeypatch.setattr(reconstruct, "_is_prose_like_row", lambda *a, **k: True)
    assert _run(page()) == base


# --------------------------------------------------------------------- exposures


def test_one_word_caption_in_the_data_font_size_is_absorbed() -> None:
    """KNOWN EXPOSURE (GH-934 log): a single-word row has no gap to measure and shares
    the data font size, so neither clause rejects it. Pinned so a change is deliberate."""
    rows = _run(_page(above=[(250.0, "Notes", 26.0)]))
    assert "Notes" in _flat(rows)


def test_spanning_header_above_stub_header_falls_back_to_main(monkeypatch) -> None:
    """The Ayivodji 43 shape: a group-spanning lane-shaped row above a stub header.
    Keeping only the lower row would ship a PARTIAL header the gate cannot see, so the
    whole stub recovery is discarded: the output equals main's (stub exemption off).
    KNOWN LOSS: the stub header is not recovered on such a page (the gate DEFERs it)."""
    spanning = [(250.0, "Panel", 76.0), (330.0, "A:", 76.0), (410.0, "Estimates", 76.0)]
    after = _run(_page(above=spanning))
    assert "Country" not in _flat(after) and "Panel" not in _flat(after)
    monkeypatch.setattr(reconstruct, "_stub_row_eligible", lambda *a, **k: False)
    assert _run(_page(above=spanning)) == after  # identical to main's behaviour
    monkeypatch.undo()
    # difference: without the fallback clause the partial header (stub row only) ships
    monkeypatch.setattr(reconstruct, "_is_lane_shaped_row", lambda *a, **k: False)
    assert _run(_page(above=spanning))[0] == HEADER_ROW


@pytest.mark.parametrize("fn", ["_header_band_ys", "_is_prose_like_row", "_stub_row_eligible"])
def test_single_walk_site(fn) -> None:
    """Both sites call ONE walk: neither re-implements the header loop."""
    import inspect

    for site in (reconstruct._extend_scope_for_header, reconstruct._prepend_header_band):
        src = inspect.getsource(site)
        assert ("_header_band_ys" in src) and (fn == "_header_band_ys" or fn not in src)


def test_label_only_row_above_the_band_is_not_absorbed() -> None:
    """A stub word alone (no lane-snapping word) is not a header row."""
    rows = _run(_page(above=[(60.0, "Country", 40.0)]))
    assert rows[0] == HEADER_ROW
    assert _flat(rows).count("Country") == 1


# ------------------------------------------------- the other site (scope extension)


def _extend_y0(page: Page, monkeypatch=None, *, sizes: bool = True) -> float:
    import fitz

    tight = fitz.Rect(240.0, 140.0, 440.0, 200.0)  # the numeric rows' bbox
    return reconstruct._extend_scope_for_header(tight, page.words, page.sizes if sizes else None).y0


def test_scope_extension_recovers_stub_header_and_rejects_captions(monkeypatch) -> None:
    assert _extend_y0(_page(above=PROSE_LINE)) == 100.0  # header kept, caption not
    assert _extend_y0(_page(above=ALL_SNAP_CAPTION)) == 140.0  # fallback to main
    monkeypatch.setattr(reconstruct, "_is_prose_like_row", lambda *a, **k: False)
    assert _extend_y0(_page(above=PROSE_LINE)) == 86.0  # difference: the caption is absorbed
    assert _extend_y0(_page(above=ALL_SNAP_CAPTION)) == 86.0
    monkeypatch.setattr(reconstruct, "_stub_row_eligible", lambda *a, **k: False)
    assert _extend_y0(_page()) == 140.0  # main's rule: the stub header is not reached
