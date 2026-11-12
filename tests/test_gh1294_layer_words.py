"""GH-1294: the words a page's own text layer owes a shipped page, and the ones it lacks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
from test_gh1156_part_a_interim import _FixedJudge
from test_gh1218_drops_clause import _page_rows, _rotated_pdf

from socr.core.marks import Check, CheckResult
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.verdict import (
    LAYER_CHECK_MODES,
    PERMANENTLY_UNRESOLVABLE_MODES,
    page_missing_words,
    silent_success_clauses,
    text_layer_words_clause,
)
from socr.tables.layer_words import (
    WordReference,
    missing_words,
    reference_from_words,
)


def _w(x0, y0, text, block=0, line=0, wno=0, width=40.0, height=10.0):
    return (x0, y0, x0 + width, y0 + height, text, block, line, wno)


def _never(_word) -> bool:
    return False


class TestReference:
    def test_repeated_text_at_the_edge_is_not_owed_but_inside_the_body_it_is(self) -> None:
        stamp = _w(72, 20, "Authorized", block=0)
        heading = _w(72, 300, "contribution", block=2)
        words = [
            stamp,
            _w(72, 100, "Inflation", block=1),
            heading,
            _w(72, 500, "Outlook", block=3),
        ]
        ref = reference_from_words(words, lambda w: w in (stamp, heading))
        assert ref.text.split() == ["inflation", "contribution", "outlook"]

    def test_a_page_number_above_a_running_head_does_not_pull_it_into_the_body(self) -> None:
        folio = _w(72, 10, "12", block=0)
        head = _w(72, 25, "Monetary", block=0, line=1)
        words = [folio, head, _w(72, 100, "Inflation", block=1)]
        # The folio is not a repeated slot (its digits change on every page).
        ref = reference_from_words(words, lambda w: w is head)
        assert ref.text.split() == ["inflation"]

    def test_repeated_text_in_a_sideways_block_is_not_owed(self) -> None:
        stamp = _w(10, 300, "Downloaded", block=5)
        words = [_w(72, 100, "Inflation", block=1), stamp, _w(72, 500, "Outlook", block=2)]
        ref = reference_from_words(words, lambda w: w is stamp, sideways_blocks={5})
        assert ref.text.split() == ["inflation", "outlook"]

    def test_a_line_break_hyphen_joins_the_halves(self) -> None:
        words = [
            _w(72, 100, "Home-", block=1, line=0),
            _w(72, 112, "builders", block=1, line=1),
            _w(72, 124, "long-", block=1, line=2),
            _w(72, 136, "Term", block=1, line=3),
        ]
        ref = reference_from_words(words, _never)
        # A capitalised next line is a new word, not the hyphenated word's second half.
        assert ref.text.split() == ["homebuilders", "long", "term"]

    def test_a_line_break_hyphen_joins_across_blocks(self) -> None:
        words = [_w(72, 100, "construc-", block=5), _w(72, 112, "tion", block=6)]
        assert reference_from_words(words, _never).text == "construction"

    def test_words_inside_a_figure_box_are_kept_apart(self) -> None:
        words = [_w(72, 100, "Inflation", block=1), _w(300, 300, "Exports", block=2)]
        ref = reference_from_words(words, _never, figure_boxes=[(250, 250, 400, 400)])
        assert ref == WordReference("inflation", "exports")


class TestMissing:
    def test_a_dropped_qualifier_is_named(self) -> None:
        ref = WordReference("percent seasonally adjusted annual rate")
        assert missing_words(ref, "percent annual rate", figure_shipped=False) == [
            "seasonally",
            "adjusted",
        ]

    def test_repeats_count(self) -> None:
        ref = WordReference("projected projected growth")
        assert missing_words(ref, "projected growth", figure_shipped=False) == ["projected"]

    def test_hyphenated_and_fused_and_ligature_forms_are_found(self) -> None:
        ref = WordReference("longerterm homebuilders onthe infation frst")
        shipped = "the longer-term outlook; Homebuilders on the inflation first"
        assert missing_words(ref, shipped, figure_shipped=False) == []

    def test_a_split_the_page_kept_as_printed_is_the_joined_word(self) -> None:
        ref = WordReference("inflation inflation")
        assert (
            missing_words(ref, "Alpha inflation\nthe infla-\ntion rate", figure_shipped=False) == []
        )

    def test_a_split_shipped_with_a_space_counts_even_when_the_word_recurs(self) -> None:
        ref = WordReference("headwinds headwinds")
        assert missing_words(ref, "strong headwinds; head winds", figure_shipped=False) == []
        assert missing_words(ref, "strong headwinds; winds", figure_shipped=False) == ["headwinds"]

    def test_figure_words_are_owed_only_without_a_picture(self) -> None:
        ref = WordReference("inflation", "exports")
        assert missing_words(ref, "Inflation", figure_shipped=True) == []
        assert missing_words(ref, "Inflation", figure_shipped=False) == ["exports"]

    def test_math_letters_are_not_words(self) -> None:
        ref = WordReference("growth", "")
        assert (
            missing_words(
                WordReference("\U0001d466\U0001d466\U0001d45b\U0001d45b growth"),
                "growth",
                figure_shipped=False,
            )
            == []
        )
        assert ref.text == "growth"

    def test_short_words_are_not_compared(self) -> None:
        ref = WordReference("the of and growth")
        assert missing_words(ref, "growth", figure_shipped=False) == []


def test_page_reference_excuses_a_stamp_repeated_at_the_top_of_every_page(tmp_path) -> None:
    from socr.core.born_digital import BornDigitalDetector

    path = tmp_path / "doc.pdf"
    pdf = pymupdf.open()
    for body in ("Inflation remained elevated", "Payrolls increased strongly"):
        page = pdf.new_page()
        page.insert_text((72, 40), "Authorized for public release")
        page.insert_text((72, 200), body)
    pdf.save(path)

    detector = BornDigitalDetector()
    with pymupdf.open(path) as doc:
        detector._read_document_furniture(doc)
        a = detector._own_layer_word_reference(doc[0])
    assert a == ("inflation remained elevated", "")


# --- the clause --------------------------------------------------------------------------

LAYER_WORDS = "real output rose percent seasonally adjusted annual rate"
FULL = "Real output rose 2.4 percent (seasonally adjusted annual rate)."
SHORT = "Real output rose 2.4 percent (annual rate)."


def _p(**over):
    base = dict(
        own_layer_word_reference=LAYER_WORDS,
        own_layer_figure_words="",
        is_born_digital=True,
        prior_ocr_layer_suspect=False,
        has_unrecovered_symbol_glyphs=False,
        table_ladder_disposition=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _out(text: str, engine: str = "native", mode=FailureMode.NONE) -> PageOutput:
    return PageOutput(
        page_num=1, text=text, status=PageStatus.SUCCESS, engine=engine, failure_mode=mode
    )


@pytest.mark.parametrize("engine", ["native", "qwen", "native+math"])
def test_a_lost_qualifier_fails_and_names_the_words_on_every_engine(engine: str) -> None:
    clause = text_layer_words_clause(_out(SHORT, engine), _p())
    assert clause is not None and clause.check is Check.TEXT_LAYER_WORDS
    assert clause.result is CheckResult.FAIL
    assert clause.reason.endswith(
        "2 word(s) this page's own text layer prints are missing from it: seasonally, adjusted"
    )
    assert text_layer_words_clause(_out(FULL, engine), _p()) is None


def test_the_same_page_with_and_without_the_words_differs_only_in_the_clause() -> None:
    got = {t: [c.check for c in silent_success_clauses(_out(t), _p())] for t in (FULL, SHORT)}
    assert got == {FULL: [], SHORT: [Check.TEXT_LAYER_WORDS]}


def test_a_damaged_layer_names_the_words_as_unknown() -> None:
    for over in ({"is_born_digital": False}, {"has_unrecovered_symbol_glyphs": True}):
        clause = text_layer_words_clause(_out(SHORT), _p(**over))
        assert clause is not None and clause.result is CheckResult.UNKNOWN, over
        assert "seasonally, adjusted" in clause.reason


@pytest.mark.parametrize(
    ("over", "out"),
    [
        pytest.param({"own_layer_word_reference": ""}, _out(SHORT), id="empty reference"),
        pytest.param({"prior_ocr_layer_suspect": True}, _out(SHORT), id="earlier OCR layer"),
        pytest.param({}, _out("[page 1 failed: unreadable — see image]"), id="failed marker"),
        pytest.param({}, _out(SHORT, mode=FailureMode.TABLE_WITHHELD), id="socr withheld a table"),
        pytest.param({}, _out("   "), id="no text"),
    ],
)
def test_the_check_does_not_apply(over, out) -> None:
    assert page_missing_words(out, _p(**over)) is None
    assert text_layer_words_clause(out, _p(**over)) is None


def test_socr_s_own_marks_do_not_count_as_the_page_s_words() -> None:
    marked = "> [page 1 unchecked: seasonally adjusted]\n\n" + SHORT
    assert page_missing_words(_out(marked), _p()) == ("seasonally", "adjusted")


def test_chart_words_are_owed_only_when_no_picture_ships() -> None:
    p = _p(own_layer_word_reference="real output rose", own_layer_figure_words="exports")
    assert page_missing_words(_out("Real output rose"), p) == ("exports",)
    assert page_missing_words(_out("Real output rose ![chart](figures/p1.png)"), p) == ()


def test_the_mode_is_permanent_and_guarded_with_the_number_modes() -> None:
    assert FailureMode.WORDS_MISSING_FROM_PAGE in PERMANENTLY_UNRESOLVABLE_MODES
    assert LAYER_CHECK_MODES[FailureMode.WORDS_MISSING_FROM_PAGE] is Check.TEXT_LAYER_WORDS


def test_the_seam_demotes_the_page_and_writes_the_mark() -> None:
    from socr.core.manifest import apply_verdict_marks

    arms = {t: apply_verdict_marks(_out(t), _p(), 1) for t in (FULL, SHORT)}
    (full, full_v), (short, short_v) = arms[FULL], arms[SHORT]
    assert full_v.clauses == ()
    assert (full.status, full.failure_mode) == (PageStatus.SUCCESS, FailureMode.NONE)
    assert (short.status, short.failure_mode) == (
        PageStatus.WARNING,
        FailureMode.WORDS_MISSING_FROM_PAGE,
    )
    assert "text_layer_words=FAIL" in short.text and "text_layer_words" not in full.text
    assert [c.check for c in short_v.clauses] == [Check.TEXT_LAYER_WORDS]


# --- a run, end to end -------------------------------------------------------------------


class _Reader:
    """The stub model reading: each page's rows, with the first row's label left out when
    ``lose_label`` (its numbers kept, so only a word differs)."""

    name = "qwen"

    def __init__(self, lose_label: bool, pages_read: list[int]) -> None:
        self._lose = lose_label
        self._pages_read = pages_read

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        self._pages_read.extend(page_nums)
        outs = []
        for n in page_nums:
            rows = _page_rows(n - 1)
            if self._lose:
                rows[0] = rows[0].split(" ", 1)[1]
            outs.append(
                PageOutput(
                    page_num=n, text="\n".join(rows), status=PageStatus.SUCCESS, engine="qwen"
                )
            )
        return outs


def _run(out_dir: Path, pdf: Path, monkeypatch, *, lose_label: bool, pages_read, ladder: bool):
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch
    from socr.pipeline.orchestrator import UnifiedPipeline

    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _Reader(lose_label, pages_read))
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=True,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=True,
                describe_figure_crops=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL] if ladder else []
        pipe._build_page_judge = lambda state: _FixedJudge(True)
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        return pipe.process(pdf, output_dir=out_dir)


def _first_page(out: Path) -> tuple[dict, str]:
    pages = next(out.rglob("pages"))
    return json.loads((pages / "00001.json").read_text()), (pages / "00001.md").read_text()


@pytest.mark.parametrize("ladder", [True, False], ids=["provider", "no provider"])
def test_one_lost_label_changes_the_page_only_when_a_reading_loses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ladder: bool
) -> None:
    arms = {}
    reads = {}
    for lose in (False, True):
        read: list[int] = []
        out = tmp_path / f"out{int(lose)}"
        _run(
            out,
            _rotated_pdf(tmp_path / f"p{int(lose)}.pdf"),
            monkeypatch,
            lose_label=lose,
            pages_read=read,
            ladder=ladder,
        )
        reads[lose] = read
        sidecar, md = _first_page(out)
        words = [
            c
            for c in (sidecar.get("verdict") or {}).get("clauses") or []
            if c["check"] == Check.TEXT_LAYER_WORDS.value
        ]
        arms[lose] = (sidecar["status"], sidecar["failure_mode"], words, "text_layer_words" in md)
    kept, lost = arms[False], arms[True]
    if not ladder:
        assert not reads[True], "fixture premise: with no provider nothing reads the page"
        assert kept == lost
        return
    assert 1 in reads[True], "fixture premise: with a provider the model reads the page"
    assert kept[2] == [] and not kept[3]
    assert lost[:2] == ("warning", "words_missing_from_page")
    [clause] = lost[2]
    assert clause["result"] == "FAIL" and clause["reason"].endswith(": korea")
    assert lost[3]


def test_the_check_records_a_pass_as_well_as_a_fail() -> None:
    from socr.core.verdict import page_word_check_record

    assert page_word_check_record(_out(FULL), _p()) == {"compared": 8, "missing": 0}
    assert page_word_check_record(_out(SHORT), _p()) == {"compared": 8, "missing": 2}
    assert page_word_check_record(_out(SHORT), _p(own_layer_word_reference="")) is None


def test_the_run_writes_the_record_on_the_page(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "out"
    _run(
        out,
        _rotated_pdf(tmp_path / "p.pdf"),
        monkeypatch,
        lose_label=False,
        pages_read=[],
        ladder=True,
    )
    record = _first_page(out)[0]["text_layer_words"]
    assert record["missing"] == 0 and record["compared"] > 0
