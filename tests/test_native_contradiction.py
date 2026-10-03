"""A shipped table the PDF's own text layer contradicts is WITHHELD, not shipped unverified.

Two layers, both hermetic (no ollama, no provider; the corpus is copyrighted so every
fixture is a synthetic PDF drawn here):

* ``socr.tables.native_contradiction``: the three checks, each pinned as a DIFFERENCE --
  the same table with and without the one fault (a dropped minus, a row shift, a wrong
  number) -- never as an absolute outcome read off one machine.
* the pipeline: ``process()`` on the same page with the table text identical except for
  the fault. Both end at the table ladder with no rung configured (UNVERIFIED); only the
  contradicted one is withheld, and it surfaces on the page, the document status, the
  metadata note and the CLI summary.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.manifest import _winning_page_output  # noqa: E402
from socr.core.providers import PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.table_counts import count_page_tables  # noqa: E402
from socr.judge.table_verdict import (  # noqa: E402
    REASON_NATIVE_CONTRADICTION,
    TABLE_LADDER_WITHHELD_KIND,
)
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402
from socr.tables import native_contradiction as nc  # noqa: E402
from socr.tables.locate import locate_tables  # noqa: E402

# A coefficient table of 14 rows x 4 values (56 numbers): large enough that ONE wrong cell is
# inside ``EXTRA_NUMBERS_MAX_SHARE``, the existing tolerance under which the text layer is
# taken to carry the table at all. The minus is printed as an ASCII hyphen (the base-14 fonts
# a synthetic PDF can use carry no U+2212); the Unicode forms are pinned on a stand-in page
# below.
LABELS = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Omicron", "Sigma", "Omega",
]  # fmt: skip
NCOLS = 4
#: Every value distinct, so a value identifies its cell.
VALUES = {
    label: tuple(f"{0.1 + (i * NCOLS + j + 1) * 0.0137:.3f}" for j in range(NCOLS))
    for i, label in enumerate(LABELS)
}
#: Cells printed with a minus.
NEGATIVE = {("Alpha", 0), ("Beta", 3), ("Epsilon", 1), ("Omicron", 2)}
ROWS = {
    label: tuple(("-" if (label, j) in NEGATIVE else "") + v for j, v in enumerate(vals))
    for label, vals in VALUES.items()
}
#: The last cell is a four-digit figure the page prints without a thousands separator.
ROWS["Omega"] = (*ROWS["Omega"][:-1], "1234")
PROSE = "The estimates below are reported for the full sample."
XS = (72.0, 170.0, 240.0, 310.0, 380.0)
TOP = 140.0
STEP = 16.0
A_NEGATIVE = ROWS["Alpha"][0]  # "-0.114"
AN_UNSIGNED = ROWS["Gamma"][0]  # printed without a minus
LAST_VALUE = ROWS["Omega"][-1]


def _md(rows: dict[str, tuple[str, ...]], order: list[str] | None = None) -> str:
    """The table as a model would emit it; ``order`` overrides the LABEL printed per row."""
    labels = order or list(rows)
    lines = [
        "| Label | " + " | ".join(f"b{j}" for j in range(NCOLS)) + " |",
        "| --- | " + " | ".join("---" for _ in range(NCOLS)) + " |",
    ]
    for label, (_key, cells) in zip(labels, rows.items()):
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


CORRECT_MD = _md(ROWS)
MINUS_DROPPED_MD = _md({k: tuple(c.lstrip("-") for c in cells) for k, cells in ROWS.items()})
#: Same values, every row bound to the next row's label: a row shift.
SHIFTED_MD = _md(ROWS, order=LABELS[1:] + LABELS[:1])
#: One cell wrong: the last value replaced by one the page does not print.
WRONG_NUMBER_MD = CORRECT_MD.replace(LAST_VALUE, "9871")
#: The same figure typeset with a thousands separator: not a different number.
THOUSANDS_MD = CORRECT_MD.replace(LAST_VALUE, "1,234")
#: Beta's label kept on a row of its own, its values moved to an unlabelled row beneath it.
DETACHED_MD = CORRECT_MD.replace(
    "| Beta | " + " | ".join(ROWS["Beta"]) + " |",
    "| Beta |  |  |  |  |\n|  | " + " | ".join(ROWS["Beta"]) + " |",
)


def _page_text(table_md: str) -> str:
    return f"{PROSE}\n\n{table_md}"


def _draw_table(page, *, invisible: bool = False, footnote: str = "") -> None:
    rect = fitz.Rect(60, TOP - 28, XS[-1] + 60, TOP + STEP * len(LABELS) + 4)
    page.draw_rect(rect, color=(0, 0, 0), width=1.0)
    for i in range(len(LABELS) + 1):  # a ruled grid, so the locator places the table
        y = TOP - 18 + i * STEP
        page.draw_line(fitz.Point(rect.x0, y), fitz.Point(rect.x1, y), width=0.8)
    for x in XS[1:]:
        page.draw_line(fitz.Point(x - 8, rect.y0), fitz.Point(x - 8, rect.y1), width=0.8)
    page.insert_text((72, 80), PROSE, fontsize=10, fontname="helv")
    mode = 3 if invisible else 0
    for x, text in zip(XS, ("Label", *(f"b{j}" for j in range(NCOLS)))):
        page.insert_text((x, TOP - 12), text, fontsize=10, fontname="helv", render_mode=mode)
    for i, label in enumerate(LABELS):
        y = TOP + i * STEP
        for x, text in zip(XS, (label, *ROWS[label])):
            if footnote and text != label:
                text += footnote  # a footnote mark glued to the printed value
            page.insert_text((x, y), text, fontsize=10, fontname="helv", render_mode=mode)


def _pdf(
    tmp_path: Path, name: str = "doc.pdf", *, invisible: bool = False, footnote: str = ""
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    doc = fitz.open()
    _draw_table(doc.new_page(width=612, height=792), invisible=invisible, footnote=footnote)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def page(tmp_path: Path):
    doc = fitz.open(str(_pdf(tmp_path)))
    yield doc[0]
    doc.close()


def _region(page):
    boxes = locate_tables(page)
    assert len(boxes) == 1, "fixture premise: the ruled box is located as one table"
    return boxes[0].bbox


def _kinds(page, md: str, *, with_region: bool = True) -> list[str]:
    regions = [_region(page)] if with_region else None
    found = nc.contradictions_for_tables(page, _page_text(md), [md], regions)[0]
    return sorted({c.kind for c in found})


# ---------------------------------------------------------------------------
# The three checks, as differences
# ---------------------------------------------------------------------------


class TestSign:
    def test_dropped_minus_is_the_only_difference(self, page) -> None:
        assert _kinds(page, CORRECT_MD) == []
        assert _kinds(page, MINUS_DROPPED_MD) == [nc.SIGN]

    def test_invented_minus_is_the_reverse(self, page) -> None:
        invented = CORRECT_MD.replace(f"| {AN_UNSIGNED} |", f"| -{AN_UNSIGNED} |")
        assert invented != CORRECT_MD
        # The page prints the number unsigned. A single invented minus is also a number the
        # page does not print, so the absent-number check agrees; the sign check names it.
        assert nc.SIGN in _kinds(page, invented)

    def test_a_sibling_tables_copy_of_the_value_is_not_this_tables_fault(self) -> None:
        """The page prints -0.18 twice; one table keeps the minus, its sibling drops it.
        Only the table that carries the unsigned copy is convicted."""
        native = [("0.18", "neg"), ("0.18", "neg")]
        signed = ("0.18", "neg", 0)
        unsigned = ("0.18", "pos", 1)
        page_numbers = [signed, unsigned]
        assert nc.sign_contradictions(native, page_numbers, [signed])[0] == nc.CLEAR
        outcome, found = nc.sign_contradictions(native, page_numbers, [unsigned])
        assert outcome == nc.CONTRADICTED and [c.kind for c in found] == [nc.SIGN]

    def test_runs_without_a_located_region(self, page) -> None:
        assert _kinds(page, MINUS_DROPPED_MD, with_region=False) == [nc.SIGN]

    def test_range_and_label_digits_are_not_signed_numbers(self) -> None:
        class P:
            def get_text(self, _mode):
                chars = lambda s: [{"c": c} for c in s]  # noqa: E731
                span = lambda s: {"chars": chars(s), "font": "F", "size": 10.0}  # noqa: E731
                line = lambda s: {"spans": [span(s)]}  # noqa: E731
                return {"blocks": [{"lines": [line("1990-2000 SIEM50 -3.5 −2.5 Mean – 4.5")]}]}

        got = nc.native_signed_numbers(P())
        assert ("1990", "pos") in got and ("2000", "pos") in got
        assert ("50", "pos") not in got  # a digit inside a label
        assert ("3.5", "neg") in got
        assert ("2.5", "neg") in got  # U+2212
        assert ("4.5", "neg") in got  # the sign-space form (#930)

    def test_an_undecodable_glyph_and_a_minus_drawn_as_a_two_are_signs(self) -> None:
        class P:
            def get_text(self, _mode):
                def span(text, font="F", size=10.0):
                    return {"chars": [{"c": c} for c in text], "font": font, "size": size}

                return {
                    "blocks": [
                        {
                            "lines": [
                                {"spans": [span("\x01"), span("0.18")]},  # #990
                                {
                                    "spans": [span("2", font="Sym"), span("0.29", font="Txt")]
                                },  # #913
                                {"spans": [span("2.50", font="Txt")]},  # an ordinary 2.50
                            ]
                        }
                    ]
                }

        assert sorted(nc.native_signed_numbers(P())) == [
            ("0.18", "neg"),
            ("0.29", "neg"),
            ("2.50", "pos"),
        ]


class TestRowShift:
    def test_a_permuted_label_column_is_the_only_difference(self, page) -> None:
        assert _kinds(page, CORRECT_MD) == []
        assert nc.ROW_SHIFT in _kinds(page, SHIFTED_MD)

    def test_values_on_an_unlabelled_row_are_a_shift(self, page) -> None:
        assert nc.ROW_SHIFT in _kinds(page, DETACHED_MD)

    def test_needs_a_located_region(self, page) -> None:
        assert nc.ROW_SHIFT not in _kinds(page, SHIFTED_MD, with_region=False)


class TestNumberAbsent:
    def test_one_wrong_cell_is_the_only_difference(self, page) -> None:
        assert _kinds(page, CORRECT_MD) == []
        assert _kinds(page, WRONG_NUMBER_MD) == [nc.NUMBER_ABSENT]


class TestAbstention:
    """No usable text layer: the checks say nothing, in either direction."""

    def test_an_invisible_ocr_layer_proves_nothing(self, tmp_path: Path) -> None:
        doc = fitz.open(str(_pdf(tmp_path, "ocr.pdf", invisible=True)))
        try:
            page = doc[0]
            assert not nc.text_layer_is_visible(page)
            for md in (MINUS_DROPPED_MD, SHIFTED_MD, WRONG_NUMBER_MD):
                assert nc.contradictions_for_tables(page, _page_text(md), [md], [None]) == [[]]
        finally:
            doc.close()

    def test_a_table_the_layer_does_not_carry_is_not_convicted(self, page) -> None:
        other = (
            "| Label | b0 |\n| --- | --- |\n| Alpha | 7.71 |\n| Beta | 8.82 |\n| Gamma | 6.63 |\n"
        )
        assert nc.contradictions_for_tables(page, _page_text(other), [other]) == [[]]

    def test_a_failure_abstains(self) -> None:
        assert nc.contradictions_for_tables(object(), "x", ["y"]) == [[]]


# ---------------------------------------------------------------------------
# The pipeline: the same page, the table text differing only by the fault
# ---------------------------------------------------------------------------


def _route(text: str):
    def _fake(page_num, ladder, run_provider, judge, **kwargs):
        from socr.pipeline.agentic import PageDecision, ProviderAttempt

        out = PageOutput(
            page_num=page_num,
            text=text,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        prof = ladder[0]
        att = ProviderAttempt(
            engine=prof.engine,
            output=out,
            cost_usd=prof.cost_per_page_usd,
            accepted=True,
            reason="ok",
            provider_id=prof.id,
            model=prof.model,
            backend=prof.backend,
        )
        return PageDecision(page_num=page_num, final_output=out, attempts=[att], accepted=True)

    return _fake


def _process(tmp_path: Path, name: str, md: str, *, ladder: bool = True):
    pdf = _pdf(tmp_path / name)
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=False,
            write_manifest=False,
            table_judge_ladder=ladder,
        )
    )
    captured: dict = {}
    original = pipeline._phase_assemble

    def _spy(state, out_dir):
        captured["state"] = state
        return original(state, out_dir)

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route(_page_text(md)))
        )
        stack.enter_context(
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            )
        )
        stack.enter_context(patch.object(pipeline, "_resolve_judge_model", return_value=""))
        stack.enter_context(patch.object(pipeline, "_plan_native_table_first", return_value=None))
        # No rung configured: every table ends the ladder UNVERIFIED, so the only thing that
        # can differ between the two runs is the contradiction check.
        stack.enter_context(patch.object(pipeline, "_build_table_judge_rungs", return_value=[]))
        stack.enter_context(patch.object(pipeline, "_phase_assemble", side_effect=_spy))
        result = pipeline.process(pdf, tmp_path / name / "out")
    return pipeline, result, captured["state"], pdf, tmp_path / name / "out"


class TestWithholdThroughTheWholePipeline:
    def test_a_dropped_minus_withholds_and_a_correct_table_stays_unverified(
        self, tmp_path: Path, capsys
    ) -> None:
        _p, ok_result, ok_state, _pdf_ok, _ = _process(tmp_path, "ok", CORRECT_MD)
        pipe, bad_result, bad_state, pdf, out_dir = _process(tmp_path, "bad", MINUS_DROPPED_MD)

        # Page disposition: the SAME ladder end, UNVERIFIED, becomes WITHHELD only on the fault.
        assert ok_state.pages[1].table_ladder_disposition is FailureMode.TABLE_UNVERIFIED
        assert bad_state.pages[1].table_ladder_disposition is FailureMode.TABLE_WITHHELD

        # The shipped page: the unverified table's bytes ship; the contradicted table's do not,
        # and the page keeps its prose.
        shipped_ok = _winning_page_output(ok_state, 1)
        shipped_bad = _winning_page_output(bad_state, 1)
        assert shipped_ok.failure_mode is FailureMode.NONE or shipped_ok.failure_mode is (
            FailureMode.TABLE_UNVERIFIED
        )
        assert ROWS["Gamma"][1] in shipped_ok.text
        assert shipped_bad.failure_mode is FailureMode.TABLE_WITHHELD
        assert shipped_bad.status is PageStatus.ERROR
        assert (
            ROWS["Gamma"][1] not in shipped_bad.text
            and A_NEGATIVE.lstrip("-") not in shipped_bad.text
        )
        assert "failed: unverifiable table" in shipped_bad.text

        # The event names the contradiction kind.
        events = [e for e in bad_state.events if e.kind == TABLE_LADDER_WITHHELD_KIND]
        assert len(events) == 1
        assert events[0].data["reason"] == REASON_NATIVE_CONTRADICTION
        assert [c["kind"] for c in events[0].data["contradictions"]] == [nc.SIGN] * len(
            events[0].data["contradictions"]
        )
        assert not [e for e in ok_state.events if e.kind == TABLE_LADDER_WITHHELD_KIND]

        # The #993 metric: withheld, not unverified.
        counts_ok = count_page_tables(
            shipped_ok.text,
            shipped_ok.status.value,
            shipped_ok.failure_mode.value,
            trust_reasons=["table_ladder_unverified"],
        )
        counts_bad = count_page_tables(
            shipped_bad.text,
            shipped_bad.status.value,
            shipped_bad.failure_mode.value,
            trust_reasons=["table_ladder_unverified", "table_ladder_withheld"],
        )
        assert (counts_ok.unverified_text, counts_ok.withheld) == (1, 0)
        assert (counts_bad.unverified_text, counts_bad.withheld) == (0, 1)
        assert counts_bad.shipped_text == 0

        # Document status, metadata and the CLI, each naming WHY.
        assert ok_result.status == DocumentStatus.AUDIT_FAILED
        assert "table_unverified" in (ok_result.error or "")
        assert "contradicts" not in (ok_result.error or "")
        assert bad_result.status in (DocumentStatus.AUDIT_FAILED, DocumentStatus.ERROR)
        assert "table_withheld" in (bad_result.error or "")
        assert "contradicts" in (bad_result.error or "")
        assert "sign" in (bad_result.error or "")
        meta = json.loads((out_dir / pdf.stem / "metadata.json").read_text(encoding="utf-8"))
        assert "contradicts" in meta["error"] and "table_withheld" in meta["error"]
        capsys.readouterr()
        pipe._print_summary(bad_result, bad_state)
        printed = capsys.readouterr().out
        assert "WITHHELD" in printed and "contradicts" in printed
        assert "blind cell transcription" not in printed

    def test_a_row_shift_withholds_the_same_way(self, tmp_path: Path) -> None:
        _p, _r, ok_state, *_ = _process(tmp_path, "ok", CORRECT_MD)
        _p, _r, bad_state, *_ = _process(tmp_path, "bad", SHIFTED_MD)
        assert ok_state.pages[1].table_ladder_disposition is FailureMode.TABLE_UNVERIFIED
        assert bad_state.pages[1].table_ladder_disposition is FailureMode.TABLE_WITHHELD
        events = [e for e in bad_state.events if e.kind == TABLE_LADDER_WITHHELD_KIND]
        assert {c["kind"] for c in events[0].data["contradictions"]} == {nc.ROW_SHIFT}

    def test_a_wrong_number_withholds_the_same_way(self, tmp_path: Path) -> None:
        _p, _r, bad_state, *_ = _process(tmp_path, "bad", WRONG_NUMBER_MD)
        assert bad_state.pages[1].table_ladder_disposition is FailureMode.TABLE_WITHHELD
        events = [e for e in bad_state.events if e.kind == TABLE_LADDER_WITHHELD_KIND]
        assert {c["kind"] for c in events[0].data["contradictions"]} == {nc.NUMBER_ABSENT}

    def test_ladder_off_leaves_the_table_alone(self, tmp_path: Path) -> None:
        """The check lives inside the ladder gate: with the ladder off, the contradicted
        table ships exactly as it did before."""
        _p, _r, state, *_ = _process(tmp_path, "off", MINUS_DROPPED_MD, ladder=False)
        assert state.pages[1].table_ladder_disposition is None
        assert not [e for e in state.events if e.kind == TABLE_LADDER_WITHHELD_KIND]


class TestScope:
    """Withhold only: nothing the ladder already decided is touched."""

    def _state(self, tmp_path: Path, md: str):
        from socr.core.document import DocumentHandle
        from socr.core.state import DocumentState

        pdf = _pdf(tmp_path)
        state = DocumentState(handle=DocumentHandle.from_path(pdf))
        out = PageOutput(
            page_num=1,
            text=_page_text(md),
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        state.pages[1].attempts.append(out)
        state.pages[1].best_output = out
        return state, out

    def _pipeline(self) -> UnifiedPipeline:
        return UnifiedPipeline(
            PipelineConfig(
                primary_engine=EngineType.QWEN, enabled_engines=[EngineType.QWEN], quiet=True
            )
        )

    def test_an_accepted_table_is_never_withheld(self, tmp_path: Path) -> None:
        from socr.core.audit_log import AuditEvent
        from socr.judge.table_verdict import TABLE_LADDER_ACCEPTED_KIND

        state, out = self._state(tmp_path, MINUS_DROPPED_MD)
        state.events.append(
            AuditEvent(page_num=1, kind=TABLE_LADDER_ACCEPTED_KIND, data={"table_id": "p1-t0"})
        )
        self._pipeline()._withhold_contradicted_unverified_tables(state, 1, state.pages[1], out)
        assert state.pages[1].table_ladder_disposition is None

    def test_the_same_page_unverified_is_withheld(self, tmp_path: Path) -> None:
        from socr.core.audit_log import AuditEvent
        from socr.judge.table_verdict import TABLE_LADDER_UNVERIFIED_KIND

        state, out = self._state(tmp_path, MINUS_DROPPED_MD)
        state.events.append(
            AuditEvent(page_num=1, kind=TABLE_LADDER_UNVERIFIED_KIND, data={"table_id": "p1-t0"})
        )
        self._pipeline()._withhold_contradicted_unverified_tables(state, 1, state.pages[1], out)
        assert state.pages[1].table_ladder_disposition is FailureMode.TABLE_WITHHELD

    def test_a_rejected_page_keeps_its_stronger_verdict(self, tmp_path: Path) -> None:
        state, out = self._state(tmp_path, MINUS_DROPPED_MD)
        state.pages[1].table_ladder_disposition = FailureMode.TABLE_REJECTED
        self._pipeline()._withhold_contradicted_unverified_tables(state, 1, state.pages[1], out)
        assert state.pages[1].table_ladder_disposition is FailureMode.TABLE_REJECTED


class TestFalsePositiveShapes:
    """Each shape an earlier version convicted a correct table on."""

    def test_an_accounting_negative_equals_a_minus(self) -> None:
        assert nc.scan_numbers("(0.12)") == [("0.12", "paren")]
        bracketed = [("0.12", "paren")]
        minus = ("0.12", "neg", 0)
        plain = ("0.12", "pos", 0)
        # the page prints (0.12); the table prints -0.12 or 0.12: neither is a dropped or an
        # invented minus
        assert nc.sign_contradictions(bracketed, [minus], [minus])[0] == nc.CLEAR
        assert nc.sign_contradictions(bracketed, [plain], [plain])[0] == nc.CLEAR
        # and the reverse direction
        brk = ("0.12", "paren", 0)
        assert nc.sign_contradictions([("0.12", "neg")], [brk], [brk])[0] == nc.CLEAR
        # control: a real dropped minus is still convicted
        assert nc.sign_contradictions([("0.12", "neg")], [plain], [plain])[0] == nc.CONTRADICTED

    def test_a_range_is_not_a_minus_however_it_is_spaced(self) -> None:
        tight = nc.scan_numbers("1\u20132")
        loose = nc.scan_numbers("1 \u2013 2")
        assert tight == loose == [("1", "pos"), ("2", "pos")]
        assert nc.scan_numbers("0.45 -0.07") == [("0.45", "pos"), ("0.07", "neg")]
        assert nc.sign_contradictions(loose, [("2", "pos", 0)], [("2", "pos", 0)])[0] == nc.CLEAR

    def test_a_thousands_separator_is_not_a_different_number(self, page) -> None:
        assert nc.scan_numbers("1,234") == nc.scan_numbers("1234") == [("1234", "pos")]
        assert _kinds(page, CORRECT_MD) == []
        assert _kinds(page, THOUSANDS_MD) == []
        assert _kinds(page, WRONG_NUMBER_MD) == [nc.NUMBER_ABSENT]

    def test_footnote_marks_do_not_hide_a_printed_value(self, tmp_path: Path) -> None:
        for mark in ("\u2020", "*", "\u00b9"):
            assert nc._printed_abs_value(f"0.45{mark}") == "0.45"
        doc = fitz.open(str(_pdf(tmp_path, footnote="*")))  # base-14 fonts carry no dagger
        try:
            page = doc[0]
            assert _kinds(page, CORRECT_MD) == []
            # the shift is still found: the marked values were not dropped from row matching
            assert nc.ROW_SHIFT in _kinds(page, SHIFTED_MD)
            # a table printing stars the page does not is not a different number
            starred = CORRECT_MD.replace(ROWS["Gamma"][1], ROWS["Gamma"][1] + "**")
            assert _kinds(page, starred) == []
        finally:
            doc.close()

    def test_identical_rows_abstain_instead_of_convicting(self, page) -> None:
        twin = ROWS["Delta"]
        duplicated = {k: (twin if k in ("Beta", "Gamma", "Delta") else v) for k, v in ROWS.items()}
        assert nc.ROW_SHIFT in _kinds(page, SHIFTED_MD)
        assert nc.ROW_SHIFT not in _kinds(page, _md(duplicated))


class TestEveryRemovedTableIsReported:
    """Withholding is page-granular, so a sibling of a contradicted table loses its bytes too.
    It must say so in the events and in the #993 count, not read as shipped."""

    SIBLING = f"| Label | b0 |\n| --- | --- |\n| Gamma | {ROWS['Gamma'][0]} |\n"

    @staticmethod
    def _pipeline() -> UnifiedPipeline:
        return UnifiedPipeline(
            PipelineConfig(
                primary_engine=EngineType.QWEN, enabled_engines=[EngineType.QWEN], quiet=True
            )
        )

    def _run(self, tmp_path: Path, prior: str | None, table_md: str = MINUS_DROPPED_MD):
        from socr.core.audit_log import AuditEvent
        from socr.core.document import DocumentHandle
        from socr.core.state import DocumentState

        state = DocumentState(handle=DocumentHandle.from_path(_pdf(tmp_path)))
        out = PageOutput(
            page_num=1,
            text=_page_text(table_md + "\n" + self.SIBLING),
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        state.pages[1].attempts.append(out)
        state.pages[1].best_output = out
        if prior:
            state.events.append(AuditEvent(page_num=1, kind=prior, data={"table_id": "p1-t1"}))
        self._pipeline()._withhold_contradicted_unverified_tables(state, 1, state.pages[1], out)
        return state

    @pytest.mark.parametrize("prior", ["table_ladder_accepted", "table_ladder_unverified", None])
    def test_the_uncontradicted_sibling_gets_its_own_withheld_record(
        self, tmp_path: Path, prior
    ) -> None:
        from socr.judge.table_verdict import REASON_SIBLING_OF_CONTRADICTED

        state = self._run(tmp_path, prior)
        events = {
            e.data["table_id"]: e.data for e in state.events if e.kind == TABLE_LADDER_WITHHELD_KIND
        }
        assert set(events) == {"p1-t0", "p1-t1"}
        assert events["p1-t0"]["reason"] == REASON_NATIVE_CONTRADICTION
        assert events["p1-t1"]["reason"] == REASON_SIBLING_OF_CONTRADICTED
        assert events["p1-t1"]["contradicted_tables"] == ["p1-t0"]
        assert events["p1-t1"]["prior_terminal"] == (prior or "none")

    def test_a_clean_page_files_no_sibling_record(self, tmp_path: Path) -> None:
        state = self._run(tmp_path, "table_ladder_accepted", table_md=CORRECT_MD)
        assert not [e for e in state.events if e.kind == TABLE_LADDER_WITHHELD_KIND]

    def test_the_metric_counts_each_removed_table_not_one_per_page(self, tmp_path: Path) -> None:
        from socr.core.table_counts import withheld_table_events

        state = self._run(tmp_path, "table_ladder_accepted")
        events = [
            {"kind": e.kind, "page_num": e.page_num, "data": e.data or {}} for e in state.events
        ]
        per_page = withheld_table_events(events)
        assert per_page == {1: 2}
        # a whole-page floor ships ONE marker for the two removed tables
        floor = "[page 1 failed: unverifiable table — see image]"
        with_events = count_page_tables(
            floor, "error", "table_withheld", withheld_events=per_page[1]
        )
        markers_only = count_page_tables(floor, "error", "table_withheld")
        assert (with_events.withheld, markers_only.withheld) == (2, 1)
        assert with_events.shipped_text == with_events.unverified_text == 0


class TestBracketedMinus:
    """A bracket with NO sign inside is the accounting convention and matches either sign; a
    bracket that states its minus is negative and matches nothing else."""

    def test_the_scanner_tells_the_two_brackets_apart(self) -> None:
        assert nc.scan_numbers("(-0.12)") == [("0.12", "bracket_neg")]
        assert nc.scan_numbers("(0.12)") == [("0.12", "paren")]

    def test_a_stated_minus_in_a_bracket_is_not_matched_by_an_unsigned_bracket(self) -> None:
        stated = ("0.12", "bracket_neg")
        bare = ("0.12", "paren", 0)
        # the page prints (-0.12), the table prints (0.12)
        assert nc.sign_contradictions([stated], [bare], [bare]) == (
            nc.CONTRADICTED,
            [nc.Contradiction(nc.SIGN, "0.12: the page prints a minus the table dropped")],
        )
        # the reverse: the table states a minus the page's bracket does not
        table_stated = ("0.12", "bracket_neg", 0)
        outcome, found = nc.sign_contradictions([("0.12", "paren")], [table_stated], [table_stated])
        assert outcome == nc.CONTRADICTED and [c.kind for c in found] == [nc.SIGN]

    def test_controls_stay_clear(self) -> None:
        bare = ("0.12", "paren", 0)
        assert nc.sign_contradictions([("0.12", "paren")], [bare], [bare])[0] == nc.CLEAR
        minus = ("0.12", "neg", 0)
        assert nc.sign_contradictions([("0.12", "paren")], [minus], [minus])[0] == nc.CLEAR
        stated = ("0.12", "bracket_neg", 0)
        assert nc.sign_contradictions([("0.12", "bracket_neg")], [stated], [stated])[0] == nc.CLEAR


class TestRowShiftNeedsTheSameCoverageAsAbsentNumbers:
    def _page_with_one_row(self, tmp_path: Path):
        path = tmp_path / "one_row.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        for x, text in zip(XS, ("Alpha", *ROWS["Alpha"])):
            page.insert_text((x, TOP), text, fontsize=10, fontname="helv")
        doc.save(str(path))
        doc.close()
        return fitz.open(str(path))

    def test_a_layer_that_prints_one_row_cannot_convict_the_table(self, page, tmp_path) -> None:
        whole = (0.0, 0.0, 612.0, 792.0)
        words = page.get_text("words")
        # full layer: the permuted labels are found
        assert nc.row_shift_contradictions(words, SHIFTED_MD, whole, page)[0] == nc.CONTRADICTED
        # the same table against a page that prints only Alpha's row: that one row matches and
        # its label is bound elsewhere, but the layer does not carry the table
        one = self._page_with_one_row(tmp_path)
        try:
            assert nc.row_shift_contradictions(
                one[0].get_text("words"), SHIFTED_MD, whole, one[0]
            ) == (nc.NO_EVIDENCE, [])
            assert nc.contradictions_for_tables(
                one[0], _page_text(SHIFTED_MD), [SHIFTED_MD], [whole]
            ) == [[]]
        finally:
            one.close()


class TestRegionsArePairedByContent:
    """locate_tables orders boxes by position, the markdown by emission."""

    TOP_MD = "| L | a | b |\n| --- | --- | --- |\n| r1 | 11.11 | 22.22 |\n| r2 | 33.33 | 44.44 |\n"
    LOW_MD = "| L | a | b |\n| --- | --- | --- |\n| s1 | 55.55 | 66.66 |\n| s2 | 77.77 | 88.88 |\n"
    TOP_BOX = (60.0, 80.0, 300.0, 140.0)
    LOW_BOX = (60.0, 300.0, 300.0, 360.0)

    def _page(self, tmp_path: Path):
        path = tmp_path / "two.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        for y, rows in (
            (100.0, ("11.11 22.22", "33.33 44.44")),
            (320.0, ("55.55 66.66", "77.77 88.88")),
        ):
            for k, line in enumerate(rows):
                page.insert_text((72, y + 16 * k), line, fontsize=10, fontname="helv")
        doc.save(str(path))
        doc.close()
        return fitz.open(str(path))

    def test_blocks_in_a_different_order_than_the_boxes_get_their_own_box(self, tmp_path) -> None:
        doc = self._page(tmp_path)
        try:
            boxes = [self.TOP_BOX, self.LOW_BOX]  # locator order: top first
            # emitted lower table first: index pairing would swap the regions
            got = nc.pair_regions(doc[0], [self.LOW_MD, self.TOP_MD], boxes)
            assert got == [self.LOW_BOX, self.TOP_BOX]
            assert nc.pair_regions(doc[0], [self.TOP_MD, self.LOW_MD], boxes) == boxes
        finally:
            doc.close()

    def test_an_ambiguous_pairing_abstains(self, tmp_path) -> None:
        doc = self._page(tmp_path)
        try:
            boxes = [self.TOP_BOX, self.LOW_BOX]
            # two blocks that both claim the top box, and a block no box prints
            assert nc.pair_regions(doc[0], [self.TOP_MD, self.TOP_MD], boxes) == [None, None]
            stray = self.TOP_MD.replace("11.11", "99.99").replace("22.22", "98.98")
            stray = stray.replace("33.33", "97.97").replace("44.44", "96.96")
            assert nc.pair_regions(doc[0], [stray, self.LOW_MD], boxes) == [None, self.LOW_BOX]
            assert nc.pair_regions(doc[0], [self.TOP_MD], boxes) == [None]  # counts differ
        finally:
            doc.close()


class TestCandidateRowsOnly:
    def test_a_year_in_a_decorative_header_row_is_not_a_claimed_number(self, page) -> None:
        # a blank-stub row of years above the grid is not a numeric body row
        decorated = CORRECT_MD.replace(
            "| --- | --- | --- | --- | --- |\n",
            "| --- | --- | --- | --- | --- |\n|  | 1901 | 1902 | 1903 | 1904 |\n",
            1,
        )
        assert decorated != CORRECT_MD
        assert _kinds(page, decorated) == []
        assert nc._candidate_values(decorated) == nc._candidate_values(CORRECT_MD)


class TestMalformedSidecarEvents:
    def _doc(self, tmp_path: Path, events) -> Path:
        pages = tmp_path / "pages"
        pages.mkdir(parents=True)
        (tmp_path / "metadata.json").write_text(json.dumps({"pages": 1}))
        (pages / "00001.json").write_text(
            json.dumps(
                {
                    "page_num": 1,
                    "status": "error",
                    "failure_mode": "table_withheld",
                    "winning_output": {"text": "[page 1 failed: unverifiable table — see image]"},
                    "audit_events": events,
                }
            )
        )
        return tmp_path

    def test_unreadable_event_data_makes_the_count_unknown_not_a_crash(self, tmp_path) -> None:
        from socr.core.table_counts import count_from_sidecars

        good = [{"kind": "table_ladder_withheld", "data": {"table_id": "p1-t0"}}]
        assert count_from_sidecars(self._doc(tmp_path / "good", good)).withheld == 1
        # absent or null events are a legacy sidecar: the markers alone count
        assert count_from_sidecars(self._doc(tmp_path / "none", None)).withheld == 1
        for bad in (
            [{"kind": "table_ladder_withheld", "data": "oops"}],
            ["not an object"],
            "not a list",
            {},
            "",
            0,
        ):
            assert (
                count_from_sidecars(self._doc(tmp_path / f"bad{abs(hash(str(bad)))}", bad)) is None
            )
