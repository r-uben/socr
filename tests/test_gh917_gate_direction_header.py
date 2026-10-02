"""GH-917 (PR A): two DEFER-only predicates in the native-first ship gate.

``foreign_direction`` (P7): the grid carries a source word whose text-line direction
differs from the table's own words. ``header_band_missing``: a numeric-free source row
above the first data row, over distinct table lanes, that the grid omits.

Each test pins a DIFFERENCE (the same page twice, one thing changed, and the gate switched
off to prove the exact-pass it overrides exists). Words are synthetic tuples in the shape of
``page.get_text("words")``; the PDF fixtures are drawn in-test. Nothing reads a corpus file
and nothing needs a provider.

The rotated quarantine (#918) is untouched; these predicates run before it.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest
from native_table_fixtures import (
    CHAR_W,
    COL_XS,
    PITCH,
    ROWS,
    Y0,
    dense_pdf,
    forecast_pdf,
    native_first_config,
    no_prose_in_header,  # noqa: F401  (fixture, used through pytestmark)
    place,
)
from test_gh916_native_ship_gate import _base, _md, _predicates, _word, _words

from socr.core.config import PipelineConfig  # noqa: F401  (kept for parity with the gh916 file)
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables import native_first as nf
from socr.tables import ship_gate
from socr.tables.native_first import DEFER, SHIP, plan_native_table
from socr.tables.ship_gate import LineDirections, line_directions_for_page

#: These synthetic pages carry no prose, see ``no_prose_in_header``; GH-936 has its own file.
pytestmark = pytest.mark.usefixtures("no_prose_in_header")

HORIZONTAL = (1.0, 0.0)
UP = (0.0, -1.0)


def _dirs(words, *, overrides: dict | None = None, default=HORIZONTAL) -> LineDirections:
    """A complete direction map for *words*; ``overrides`` re-points chosen (block, line) keys."""
    mapping = {(w[5], w[6]): default for w in words}
    mapping.update(overrides or {})
    return LineDirections(dirs=mapping)


def _stamp(text: str, *, line: int = 99, x: float = 560.0, y: float = 700.0) -> tuple:
    """A source word away from the table, on its own line (so its direction is its own)."""
    return (x, y, x + CHAR_W * len(text), y + 9.0, text, 0, line, 0)


def _output_blocks_of(markdown: str):
    return ship_gate._output_blocks(markdown)


def _gate_off(words, markdown, line_dirs):
    with patch.object(nf, "native_ship_gate", return_value=()):
        return plan_native_table(words, markdown, line_dirs=line_dirs)


# ------------------------------------------------------------ P7 foreign_direction


class TestForeignDirection:
    def test_difference_pin_foreign_word_defers_and_same_direction_ships(self) -> None:
        words, md = _base()
        # "GDP" is a row label, so its stamp is carried by the grid's cells. Same word, same
        # place; only the stamp's line direction changes.
        words = words + [_stamp("GDP")]
        same = plan_native_table(words, md, line_dirs=_dirs(words))
        foreign = plan_native_table(words, md, line_dirs=_dirs(words, overrides={(0, 99): UP}))
        off = _gate_off(words, md, _dirs(words, overrides={(0, 99): UP}))
        assert same.action == SHIP and same.faults == ()
        assert off.action == SHIP, "the exact-pass the gate overrides must exist"
        assert foreign.action == DEFER
        assert _predicates(foreign) == {ship_gate.FOREIGN_DIRECTION}
        assert foreign.reason == f"{ship_gate.SHIP_GATE_REASON_PREFIX}:foreign_direction"

    def test_direction_not_vocabulary_is_what_fires(self) -> None:
        # The same foreign line, but the word is not in the grid: it is not carried.
        words, md = _base()
        words = words + [_stamp("ZZQQX")]
        plan = plan_native_table(words, md, line_dirs=_dirs(words, overrides={(0, 99): UP}))
        assert plan.action == SHIP and plan.faults == ()

    def test_a_table_whose_every_word_is_vertical_does_not_fire(self) -> None:
        words, md = _base()
        plan = plan_native_table(words, md, line_dirs=_dirs(words, default=UP))
        assert plan.action == SHIP and plan.faults == ()

    def test_jitter_within_tolerance_does_not_fire_and_a_real_skew_does(self) -> None:
        words, md = _base()
        words = words + [_stamp("GDP")]

        def stamped(angle_deg: float):
            a = math.radians(angle_deg)
            return plan_native_table(
                words,
                md,
                line_dirs=_dirs(words, overrides={(0, 99): (math.cos(a), math.sin(a))}),
            )

        # measured same-table deviation is exactly 0 (float32 dir noise ~1e-7 rad)
        for jitter in (0.0, 1e-7, 1e-6, 5e-4):  # degrees: up to ~1e-5 rad
            assert stamped(jitter).action == SHIP, jitter
            assert stamped(-jitter).action == SHIP, -jitter
        # a slightly rotated stamp is a different text, whatever its width: 0.01 deg and up
        for skew in (0.01, 0.5, 4.0, 10.0, 45.0, 90.0, 180.0):
            plan = stamped(skew)
            assert plan.action == DEFER and _predicates(plan) == {ship_gate.FOREIGN_DIRECTION}

    def test_a_narrow_table_still_fires_on_a_10_and_a_90_degree_foreign_line(self) -> None:
        # A table barely wider than the lane snap: the old width-derived tolerance reached
        # ~20 degrees here and let a 10 degree line through. The stamp sits INSIDE the
        # table's width so it does not widen it.
        md = "| a | n1 | n2 |\n| --- | --- | --- |\n| x | 1.5 | 2.5 |\n| y | 3.5 | 4.5 |"
        blocks = ship_gate._output_blocks(md)
        words = [
            _word(40.0, 100.0, "a", 0, 0),
            _word(62.0, 100.0, "n1", 0, 1),
            _word(84.0, 100.0, "n2", 0, 2),
            _word(40.0, 114.0, "x", 1, 0),
            _word(62.0, 114.0, "1.5", 1, 1),
            _word(84.0, 114.0, "2.5", 1, 2),
            _word(40.0, 128.0, "y", 2, 0),
            _word(62.0, 128.0, "3.5", 2, 1),
            _word(84.0, 128.0, "4.5", 2, 2),
            _word(50.0, 700.0, "n1", 99, 0),  # a foreign copy of a carried word
        ]
        # narrow enough that snap/extent tolerance would exceed 10 degrees
        extent = max(w[2] for w in words) - min(w[0] for w in words)
        assert extent < ship_gate._SNAP_PT / math.tan(math.radians(10.0))

        def run(angle_deg: float | None):
            over = {}
            if angle_deg is not None:
                a = math.radians(angle_deg)
                over = {(0, 99): (math.cos(a), math.sin(a))}
            return ship_gate.foreign_direction_faults(words, blocks, _dirs(words, overrides=over))

        assert run(None) == []
        for angle in (10.0, 90.0):
            faults = run(angle)
            assert [f["predicate"] for f in faults] == [ship_gate.FOREIGN_DIRECTION], angle

    def test_upside_down_is_a_different_direction(self) -> None:
        # A direction is a vector, not an axis: 180 degrees apart is foreign.
        words, md = _base()
        words = words + [_stamp("GDP")]
        plan = plan_native_table(
            words, md, line_dirs=_dirs(words, overrides={(0, 99): (-1.0, 0.0)})
        )
        assert _predicates(plan) == {ship_gate.FOREIGN_DIRECTION}

    def test_tolerance_is_the_measured_constant_and_independent_of_width(self) -> None:
        assert ship_gate._SAME_TEXT_DIRECTION_TOL_RAD == 1e-5
        assert not hasattr(ship_gate, "_direction_tolerance")

    def test_a_tie_between_two_directions_defers(self) -> None:
        # Half the rows horizontal, half vertical: no "modal" direction, still foreign.
        words, md = _base()
        half = {(0, line): UP for line in range(0, 4)}
        plan = plan_native_table(words, md, line_dirs=_dirs(words, overrides=half))
        assert plan.action == DEFER and _predicates(plan) == {ship_gate.FOREIGN_DIRECTION}

    def test_membership_is_substring_presence_a_documented_over_defer(self) -> None:
        # ``_CellText.has_word`` is substring presence, not occurrence attribution: a foreign
        # "0.2" printed elsewhere on the page is "carried" because "0.253" contains it, even
        # though no grid cell came from that word. Conservative by design (a false DEFER costs
        # one model read); pinned so a change to it is deliberate.
        words, md = _base()
        words = words + [_stamp("0.2")]
        plan = plan_native_table(words, md, line_dirs=_dirs(words, overrides={(0, 99): UP}))
        assert plan.action == DEFER and _predicates(plan) == {ship_gate.FOREIGN_DIRECTION}

    def test_membership_and_direction_are_per_output_block(self) -> None:
        # Block 1 is entirely vertical text, block 2 entirely horizontal, with disjoint
        # vocabularies. Each block is single-direction, so nothing fires; a page-wide
        # direction (or page-wide membership) would see two directions.
        head_a = ["name", "cd", "ac", "bd", "ab"]
        rows_a = [[f"{c}{c}", "1.234", "0.123", "2.310", "3.412"] for c in "ab"] + [
            ["cd", "0.141", "1.232", "2.044", "3.021"]
        ]
        head_b = ["xyz", "wvu", "tsr", "pon", "mlk"]
        rows_b = [[f"{c}z", "5.678", "6.789", "7.895", "8.967"] for c in "wvu"]
        words_a = _words([head_a] + rows_a)
        words_b = [
            (w[0], w[1] + 300.0, w[2], w[3] + 300.0, w[4], 1, w[6], w[7])
            for w in _words([head_b] + rows_b)
        ]
        words = words_a + words_b
        md = _md(head_a, rows_a) + "\n\n" + _md(head_b, rows_b)
        overrides = {(0, line): UP for line in range(0, 4)}
        both = plan_native_table(words, md, line_dirs=_dirs(words, overrides=overrides))
        assert ship_gate.FOREIGN_DIRECTION not in _predicates(both), both.faults
        # The same two blocks with block 2's own words disagreeing DO fire, once.
        clash = {**overrides, (1, 1): UP}
        faults = ship_gate.foreign_direction_faults(
            words, ship_gate._output_blocks(md), _dirs(words, overrides=clash)
        )
        assert [f["predicate"] for f in faults] == [ship_gate.FOREIGN_DIRECTION]


# ----------------------------------------------- P7 failure contract (Astra, priority one)


class TestDirectionFailureContract:
    """``line_dirs`` is required. Only the named sentinel skips P7; every failure DEFERs."""

    def _words_md(self):
        words, md = _base()
        return words + [_stamp("GDP")], md

    def test_line_dirs_cannot_be_omitted(self) -> None:
        words, md = self._words_md()
        with pytest.raises(TypeError):
            plan_native_table(words, md)
        with pytest.raises(TypeError):
            ship_gate.native_ship_gate(words, md)

    def test_none_is_a_fault_not_a_skip_and_only_the_sentinel_skips(self) -> None:
        words, md = self._words_md()
        stamped = _dirs(words, overrides={(0, 99): UP})
        none = plan_native_table(words, md, line_dirs=None)
        assert none.action == DEFER and _predicates(none) == {ship_gate.DIRECTION_UNAVAILABLE}
        skipped = plan_native_table(words, md, line_dirs=LineDirections.unchecked_for_tests())
        assert skipped.action == SHIP, "the sentinel skips P7 even when a foreign word exists"
        assert plan_native_table(words, md, line_dirs=stamped).action == DEFER

    @pytest.mark.parametrize(
        "bad",
        [
            LineDirections(fault="RuntimeError: boom"),
            LineDirections(dirs={}),
            LineDirections(),
            {},
            {(0, 0): HORIZONTAL},  # not a LineDirections at all
        ],
        ids=["extraction-failed", "empty-map", "default-empty", "plain-empty-dict", "wrong-type"],
    )
    def test_any_unusable_map_defers_with_a_recorded_fault_never_ships(self, bad) -> None:
        words, md = self._words_md()
        plan = plan_native_table(words, md, line_dirs=bad)
        assert plan.action == DEFER
        assert _predicates(plan) == {ship_gate.DIRECTION_UNAVAILABLE}
        assert plan.faults and plan.faults[0]["detail"]
        assert _gate_off(words, md, bad).action == SHIP, "the exact-pass it overrides exists"

    def test_an_empty_map_defers_even_when_no_word_is_carried(self) -> None:
        # The empty map is a failure of its own, not just "every carried word lacks a key":
        # a page with words and no line map at all must never read as "nothing foreign".
        words, _md_unused = _base()
        unrelated = _output_blocks_of("| xxx | yyy |\n| --- | --- |\n| zzz | www |")
        faults = ship_gate.foreign_direction_faults(words, unrelated, LineDirections())
        assert [f["predicate"] for f in faults] == [ship_gate.DIRECTION_UNAVAILABLE]
        # the same blocks with a populated map: nothing carried, nothing to say
        assert ship_gate.foreign_direction_faults(words, unrelated, _dirs(words)) == []

    def test_a_carried_word_without_a_key_defers(self) -> None:
        words, md = self._words_md()
        full = _dirs(words)
        short = LineDirections(dirs={k: v for k, v in full.dirs.items() if k != (0, 3)})
        plan = plan_native_table(words, md, line_dirs=short)
        assert plan.action == DEFER and _predicates(plan) == {ship_gate.DIRECTION_UNAVAILABLE}
        assert plan_native_table(words, md, line_dirs=full).action == SHIP

    def test_a_word_the_grid_does_not_carry_may_lack_a_key(self) -> None:
        words, md = _base()
        words = words + [_stamp("ZZQQX")]
        mapping = _dirs(words).dirs
        del mapping[(0, 99)]
        plan = plan_native_table(words, md, line_dirs=LineDirections(dirs=mapping))
        assert plan.action == SHIP and plan.faults == ()

    def test_words_without_block_and_line_indices_defer(self) -> None:
        words, md = _base()
        bare = [w[:5] for w in words]
        plan = plan_native_table(bare, md, line_dirs=_dirs(words))
        assert plan.action == DEFER and _predicates(plan) == {ship_gate.DIRECTION_UNAVAILABLE}

    @pytest.mark.parametrize("vec", [(0.0, 0.0), (float("nan"), 1.0), None, "ab"])
    def test_an_unusable_direction_value_defers(self, vec) -> None:
        words, md = _base()
        plan = plan_native_table(words, md, line_dirs=_dirs(words, overrides={(0, 2): vec}))
        assert plan.action == DEFER and _predicates(plan) == {ship_gate.DIRECTION_UNAVAILABLE}

    def test_extraction_failure_is_returned_not_raised(self) -> None:
        class Boom:
            def get_text(self, *a, **k):
                raise RuntimeError("no dict")

        got = line_directions_for_page(Boom())
        assert got.fault.startswith("RuntimeError") and got.dirs == {}


# ------------------------------------------- extraction aligned to word keys (image blocks)


def _image_block_pdf(path: Path) -> None:
    """Images before and after the title split PyMuPDF's dict blocks; one vertical line."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), False)
    pix.clear_with(200)
    page.insert_image(fitz.Rect(72, 20, 100, 48), pixmap=pix)
    page.insert_text((72, 60), "Title text line", fontsize=10)
    page.insert_image(fitz.Rect(300, 20, 328, 48), pixmap=pix)
    for ri in range(4):
        for ci, text in enumerate(["GDP", "0.253", "0.179"]):
            page.insert_text((90 + ci * 90, 100 + ri * 22), text, fontsize=9)
    page.insert_text((560, 300), "MARGIN", fontsize=9, rotate=90)
    doc.save(str(path))
    doc.close()


class TestDirectionKeysAlignWithWords:
    def test_every_word_has_a_key_and_its_own_lines_direction(self, tmp_path: Path) -> None:
        pdf = tmp_path / "img.pdf"
        _image_block_pdf(pdf)
        with fitz.open(pdf) as doc:
            page = doc[0]
            words = list(page.get_text("words"))
            got = line_directions_for_page(page)
            # the fixture is only worth anything if the two segmentations really differ
            assert any(b["type"] == 1 for b in page.get_text("dict")["blocks"])
        assert got.fault == ""
        assert words and all((w[5], w[6]) in got.dirs for w in words)
        for w in words:
            want = UP if w[4] == "MARGIN" else HORIZONTAL
            assert got.dirs[(w[5], w[6])] == pytest.approx(want), w[4]

    def test_the_foreign_margin_word_is_what_the_gate_sees(self, tmp_path: Path) -> None:
        pdf = tmp_path / "img.pdf"
        _image_block_pdf(pdf)
        with fitz.open(pdf) as doc:
            words = list(doc[0].get_text("words"))
            dirs = line_directions_for_page(doc[0])
        md = "| h | a | b |\n| --- | --- | --- |\n| GDP | 0.253 | 0.179 |\n| MARGIN | 1 | 2 |"
        faults = ship_gate.foreign_direction_faults(words, ship_gate._output_blocks(md), dirs)
        assert [f["predicate"] for f in faults] == [ship_gate.FOREIGN_DIRECTION]
        assert "MARGIN" in faults[0]["detail"]
        # drop the margin word's text from the grid: it is no longer carried
        md2 = "| h | a | b |\n| --- | --- | --- |\n| GDP | 0.253 | 0.179 |"
        assert ship_gate.foreign_direction_faults(words, ship_gate._output_blocks(md2), dirs) == []


# -------------------------------------------------------------------- PDF-level wiring


def _stampeddense_pdf(path: Path, *, stamp: bool, rotate: int = 90) -> None:
    """``_dense_pdf`` plus a "GDP" margin stamp drawn at *rotate* (a grid label, so carried)."""
    dense_pdf(path)
    if stamp:
        doc = fitz.open(path)
        doc[0].insert_text((580, 400), "GDP", fontsize=9, fontname="helv", rotate=rotate)
        doc.save(str(path), incremental=True, encryption=0)
        doc.close()


def _upright_plan(tmp_path: Path, name: str, *, stamp: bool, rotate: int = 90, plan_ctx=None):
    """Analyze normally, then plan under *plan_ctx* (a context manager, or None)."""
    import contextlib

    pdf = tmp_path / f"{name}.pdf"
    _stampeddense_pdf(pdf, stamp=stamp, rotate=rotate)
    pipeline = UnifiedPipeline(native_first_config())
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    pipeline._phase_analyze(state)
    with patch.object(pipeline, "_available_engines_for_agentic", return_value=[]):
        with plan_ctx or contextlib.nullcontext():
            work = pipeline._plan_native_table_first(state, 1, state.pages[1])
    return work, state


class TestUprightEmitSite:
    def test_a_foreign_stamp_defers_through_the_orchestrator_and_is_recorded(
        self, tmp_path: Path
    ) -> None:
        clean_work, clean_state = _upright_plan(tmp_path, "clean", stamp=False)
        same_work, same_state = _upright_plan(tmp_path, "same", stamp=True, rotate=0)
        work, state = _upright_plan(tmp_path, "foreign", stamp=True, rotate=90)
        # same page text, the stamp's direction is the only thing that differs
        assert clean_work is not None and clean_work.plan.action == SHIP
        assert same_work is not None and same_work.plan.action == SHIP
        assert work is None, "DEFER leaves the page on route_page"
        assert [e.kind for e in clean_state.events].count(ship_gate.SHIP_GATE_KIND) == 0
        assert [e.kind for e in same_state.events].count(ship_gate.SHIP_GATE_KIND) == 0
        events = [e for e in state.events if e.kind == ship_gate.SHIP_GATE_KIND]
        assert len(events) == 1
        assert events[0].data["predicates"] == [ship_gate.FOREIGN_DIRECTION]

    def test_a_direction_extraction_failure_defers_it_does_not_refuse_or_ship(
        self, tmp_path: Path
    ) -> None:
        real = fitz.Page.get_text

        def _fail_dict(self, option="text", *args, **kwargs):
            if option == "dict":
                raise RuntimeError("dict extraction broke")
            return real(self, option, *args, **kwargs)

        work, state = _upright_plan(
            tmp_path,
            "fail",
            stamp=False,
            plan_ctx=patch.object(fitz.Page, "get_text", _fail_dict),
        )
        # the words were read fine, so this is NOT the "text layer unreadable" REFUSE
        assert work is None
        events = [e for e in state.events if e.kind == ship_gate.SHIP_GATE_KIND]
        assert len(events) == 1
        assert events[0].data["predicates"] == [ship_gate.DIRECTION_UNAVAILABLE]
        assert "dict extraction broke" in events[0].data["faults"][0]["detail"]


class TestRotatedEmitSite:
    @staticmethod
    def _attempt(tmp_path: Path, name: str, *, stamp_rotate: int | None):
        pdf = tmp_path / f"{name}.pdf"
        forecast_pdf(pdf, 90)
        if stamp_rotate is not None:
            doc = fitz.open(pdf)
            page = doc[0]
            page.insert_text(
                place(500, 380, 90, 612, 792),
                "GDP",
                fontsize=9,
                fontname="helv",
                rotate=stamp_rotate,
            )
            doc.save(str(pdf), incremental=True, encryption=0)
            doc.close()
        with fitz.open(pdf) as doc:
            return nf.attempt_rotated_native_table(doc[0])

    def test_a_foreign_word_changes_the_reason_from_quarantine_to_the_gate(
        self, tmp_path: Path
    ) -> None:
        clean = self._attempt(tmp_path, "clean", stamp_rotate=None)
        same = self._attempt(tmp_path, "same", stamp_rotate=90)
        foreign = self._attempt(tmp_path, "foreign", stamp_rotate=0)
        assert clean is not None and same is not None and foreign is not None
        # a gate-clean rotated page still hits the #918 quarantine; that is untouched
        assert clean.plan.reason == nf.ROTATED_SHIP_QUARANTINED
        assert same.plan.reason == nf.ROTATED_SHIP_QUARANTINED
        assert foreign.plan.reason.startswith(ship_gate.SHIP_GATE_REASON_PREFIX)
        assert _predicates(foreign.plan) == {ship_gate.FOREIGN_DIRECTION}


# ---------------------------------------------------------- header_band_missing

# A header row of five-character words over the four numeric lanes, with the stub at the
# left of the first lane. Width keeps every x-centre within one lane snap of its lane.
BAND = ["Country", "Alpha", "Gamma", "Delta", "Omega"]
BAND_GRID_HEAD = ["Country", "", "", "", ""]  # a grid that kept only the stub


def _band_case(*, band_y: float | None, band=BAND, rows=ROWS):
    """Data rows start at ``Y0 + PITCH``; the band row is at ``band_y`` (None: no band)."""
    words = _words(rows, y_start=Y0 + PITCH)
    if band_y is not None:
        words += [
            _word(COL_XS[i], band_y, text, 40 + 0, 50 + i) for i, text in enumerate(band) if text
        ]
    return words


def _band_plan(words, head):
    return plan_native_table(words, _md(head, ROWS), line_dirs=_dirs(words))


class TestHeaderBandMissing:
    FIRST = Y0 + PITCH  # first data row

    def test_difference_pin_dropped_band_defers_and_kept_band_ships(self) -> None:
        words = _band_case(band_y=self.FIRST - 2 * PITCH)
        dropped = _band_plan(words, BAND_GRID_HEAD)
        kept = _band_plan(words, BAND)
        off = _gate_off(words, _md(BAND_GRID_HEAD, ROWS), _dirs(words))
        assert off.action == SHIP, "the exact-pass the gate overrides must exist"
        assert dropped.action == DEFER and _predicates(dropped) == {ship_gate.HEADER_BAND_MISSING}
        assert kept.action == SHIP and kept.faults == ()

    def test_no_header_row_on_the_page_cannot_fire(self) -> None:
        words = _band_case(band_y=None)
        plan = _band_plan(words, BAND_GRID_HEAD)
        assert plan.action == SHIP and plan.faults == ()

    def test_two_lane_row_is_not_a_header_band(self) -> None:
        words = _band_case(
            band_y=self.FIRST - 2 * PITCH, band=["Country", "Alpha", "Gamma", "", ""]
        )
        assert ship_gate._MIN_LANES_PER_ROW == 3
        assert _band_plan(words, BAND_GRID_HEAD).action == SHIP

    def test_three_lanes_is_the_minimum(self) -> None:
        words = _band_case(
            band_y=self.FIRST - 2 * PITCH, band=["Country", "Alpha", "Gamma", "Delta", ""]
        )
        assert _predicates(_band_plan(words, BAND_GRID_HEAD)) == {ship_gate.HEADER_BAND_MISSING}

    def test_two_words_over_one_lane_is_not_a_header_band(self) -> None:
        words = _band_case(band_y=self.FIRST - 2 * PITCH)
        # a second word crossing the Alpha lane: five region words over four lanes
        words.append(_word(COL_XS[1] + 3.0, self.FIRST - 2 * PITCH, "Zeta", 40, 60))
        assert _band_plan(words, BAND_GRID_HEAD).action == SHIP

    def test_a_word_between_lanes_is_not_a_header_band(self) -> None:
        words = _band_case(band_y=self.FIRST - 2 * PITCH)
        # centre of this word falls between two lanes (45 pt from both)
        words.append(_word(COL_XS[1] + 45.0 - CHAR_W * 2, self.FIRST - 2 * PITCH, "Mid", 40, 60))
        assert _band_plan(words, BAND_GRID_HEAD).action == SHIP

    def test_a_row_carrying_a_number_is_not_a_header_band(self) -> None:
        band = ["Country", "Alpha", "Gamma", "Delta", "Omega"]
        words = _band_case(band_y=self.FIRST - 2 * PITCH, band=band)
        words.append(_word(COL_XS[0] + 20, self.FIRST - 2 * PITCH, "2004", 40, 61))
        assert _band_plan(words, BAND_GRID_HEAD).action == SHIP

    def test_reach_is_five_row_pitches_and_no_further(self) -> None:
        reach = ship_gate._PANEL_GAP_ROWS
        assert reach == 5
        inside = _band_case(band_y=self.FIRST - reach * PITCH)
        outside = _band_case(band_y=self.FIRST - (reach + 1) * PITCH)
        assert _predicates(_band_plan(inside, BAND_GRID_HEAD)) == {ship_gate.HEADER_BAND_MISSING}
        assert _band_plan(outside, BAND_GRID_HEAD).action == SHIP

    def test_a_row_below_the_first_data_row_is_not_a_header_band(self) -> None:
        words = _band_case(band_y=self.FIRST + 0.5 * PITCH)
        assert ship_gate.HEADER_BAND_MISSING not in _predicates(_band_plan(words, BAND_GRID_HEAD))


# --------------------------------------------- the CELL-REPAIR caller (Astra round 2)


def _repair_run(
    tmp_path: Path,
    tag: str,
    providers: list,
    *,
    bad_dirs=None,
    bad_from_call: int = 2,
    stamp: bool = False,
    corrupt: bool = True,
):
    """``process()`` on the dense table whose first cell the grid got wrong (CELLS plan).

    ``bad_dirs(real_map)`` replaces the direction map from the *bad_from_call*-th call of
    ``line_directions_for_page`` on: call 1 is the plan, call 2 the repair re-plan.
    ``stamp`` draws a rotated "GDP" word, which only the re-plan's gate can see (a CELLS
    plan never reaches the gate). ``corrupt=False`` is an ordinary EXACT_PASS page.
    """
    from socr.core.born_digital import BornDigitalDetector
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision, ProviderAttempt

    pdf = tmp_path / f"{tag}.pdf"
    _stampeddense_pdf(pdf, stamp=stamp, rotate=90)
    pipeline = UnifiedPipeline(native_first_config())
    if corrupt:
        real_detector = BornDigitalDetector()

        class _Corrupt(BornDigitalDetector):
            def detect(self, path):
                assessment = real_detector.detect(path)
                page = assessment.pages[0]
                assert "0.253" in (page.native_text or "")
                page.native_text = page.native_text.replace("0.253", "9.999", 1)
                return assessment

        pipeline.bd_detector = _Corrupt()
    routes: list[int] = []
    transcribed: list[int] = []

    def _route(page_num, ladder, run_provider, judge, **kwargs):
        routes.append(page_num)
        out = PageOutput(
            page_num=page_num,
            text="model table",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        prof = ladder[0]
        att = ProviderAttempt(
            engine=prof.engine,
            output=out,
            cost_usd=0.0,
            accepted=True,
            reason="test",
            provider_id=prof.id,
            model=prof.model,
            backend=prof.backend,
        )
        return PageDecision(page_num=page_num, final_output=out, attempts=[att], accepted=True)

    def _transcribe(_crop):
        transcribed.append(1)
        return "0.253"

    calls = {"n": 0}
    real_dirs = ship_gate.line_directions_for_page

    def _dirs_by_call(page):
        calls["n"] += 1
        got = real_dirs(page)
        if bad_dirs is not None and calls["n"] >= bad_from_call:
            return bad_dirs(got)
        return got

    out_dir = tmp_path / f"out_{tag}"
    with (
        patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=providers),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_transcribe_cell_token", side_effect=_transcribe),
        patch.object(ship_gate, "line_directions_for_page", _dirs_by_call),
    ):
        result = pipeline.process(pdf, out_dir)
    import json

    sidecar = json.loads(next(out_dir.rglob("pages/00001.json")).read_text(encoding="utf-8"))
    return result, sidecar, routes, transcribed, calls["n"]


def _kinds(sidecar) -> list[str]:
    return [e["kind"] for e in sidecar["audit_events"]]


_REPAIR_FAILURES = {
    "extraction-failed": (
        lambda real: LineDirections(fault="RuntimeError: dict extraction broke"),
        ship_gate.DIRECTION_UNAVAILABLE,
    ),
    "empty-map": (lambda real: LineDirections(), ship_gate.DIRECTION_UNAVAILABLE),
    "missing-keys": (
        lambda real: LineDirections(
            dirs={k: v for i, (k, v) in enumerate(sorted(real.dirs.items())) if i % 2}
        ),
        ship_gate.DIRECTION_UNAVAILABLE,
    ),
}


class TestCellRepairKeepsTheGateDeferOnly:
    """A CELLS plan whose repaired grid the gate DEFERs must reach route_page, recorded."""

    @pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "none"])
    @pytest.mark.parametrize("mode", list(_REPAIR_FAILURES) + ["foreign-stamp"])
    def test_post_repair_gate_fault_is_deferred_recorded_and_not_refused(
        self, tmp_path: Path, providers: list, mode: str
    ) -> None:
        if mode == "foreign-stamp":
            bad, predicate, stamp = None, ship_gate.FOREIGN_DIRECTION, True
        else:
            bad, predicate = _REPAIR_FAILURES[mode]
            stamp = False
        # the same page when the repair succeeds and the gate is clean: SHIPs
        _r0, clean, clean_routes, clean_tx, _n = _repair_run(tmp_path, "clean", providers)
        assert clean_routes == [] and len(clean_tx) == 1
        assert "native_table_cell_repaired" in _kinds(clean)
        assert ship_gate.SHIP_GATE_KIND not in _kinds(clean)

        _r, side, routes, tx, ncalls = _repair_run(
            tmp_path, f"def_{mode}", providers, bad_dirs=bad, stamp=stamp
        )
        assert ncalls >= 2, "the repair re-plan must have asked for directions"
        assert len(tx) == 1, "the repair did run"
        kinds = _kinds(side)
        assert kinds.count(ship_gate.SHIP_GATE_KIND) == 1
        event = next(e for e in side["audit_events"] if e["kind"] == ship_gate.SHIP_GATE_KIND)
        assert event["data"]["predicates"] == [predicate]
        # DEFER, not REFUSE: no unresolved-cell event and no repaired grid shipped.
        assert "native_table_cell_unresolved" not in kinds
        assert "native_table_cell_repaired" not in kinds
        assert "9.999" not in (side.get("markdown") or "")
        if providers:
            # Routed to the model: no refuse state and no failure marker.
            assert routes == [1], "the page is routed to the model"
            assert side.get("native_table_structure_failed") in (None, False)
            assert side.get("native_table_unverifiable") in (None, False)
            assert side["status"] != "error"
        else:
            # No provider: the page takes the SAME native-text fallback any plan-time
            # DEFER takes with no provider (WARNING / MODEL_UNAVAILABLE). That fallback
            # sets native_table_structure_failed on table pages, so it is not asserted
            # absent here; test_a_repair_time_deferral_matches_a_plan_time_deferral
            # pins that the two paths agree (cubic P3 on 1205d2b).
            assert routes == []

    @pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "none"])
    def test_a_repair_time_deferral_matches_a_plan_time_deferral(
        self, tmp_path: Path, providers: list
    ) -> None:
        # Same stamped page. (a) intact grid: P7 DEFERs at plan time. (b) corrupted grid:
        # CELLS, repaired, then P7 DEFERs at the re-plan. Routing and refuse state agree.
        _ra, plan_side, plan_routes, plan_tx, _ = _repair_run(
            tmp_path, "planlevel", providers, stamp=True, corrupt=False
        )
        _rb, rep_side, rep_routes, rep_tx, _ = _repair_run(
            tmp_path, "repairlevel", providers, stamp=True, corrupt=True
        )
        assert plan_tx == [] and len(rep_tx) == 1
        assert plan_routes == rep_routes
        for key in ("status", "native_table_structure_failed", "native_table_unverifiable"):
            assert plan_side.get(key) == rep_side.get(key), key
        assert _kinds(plan_side).count(ship_gate.SHIP_GATE_KIND) == 1
        assert _kinds(rep_side).count(ship_gate.SHIP_GATE_KIND) == 1
