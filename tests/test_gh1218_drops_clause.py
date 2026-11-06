"""#1218: a model reading that DROPS numbers the page prints in table rows is marked.

The detection itself (which layer numbers are owed) is ``tables.layer_body`` and
``tables.reading_agreement.dropped_numbers``; this file pins the WIRING: the
``text_layer_drops`` check, its failure mode, mark, event, document note, resume rule and
``socr-refinalize-own-text``.

Per ``CLAUDE.md`` the run-level tests pin DIFFERENCES: one page run twice in one process,
changing only whether the reading omits a row, with the provider ladder in both states.
"""

from __future__ import annotations

import io
import json
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from rich.console import Console

import socr
from socr.core import verdict as verdict_module
from socr.core.manifest import apply_verdict_marks
from socr.core.marks import Check, CheckResult
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.verdict import (
    NUMBER_CHECK_MODES,
    PERMANENTLY_UNRESOLVABLE_MODES,
    TEXT_LAYER_DROPS_REASON,
    page_dropped_numbers,
    reading_numbers_text,
    silent_success_clauses,
    text_layer_drops_clause,
    with_text_layer_numbers_clause,
)
from socr.pipeline import orchestrator as orchestrator_module
from socr.pipeline.orchestrator import UnifiedPipeline

from test_gh1156_part_a_interim import _FixedJudge
from test_gh1185_text_layer_numbers_e2e import (
    _comparable,
    _crash_after_page_one,
    _doc_dir,
    _page,
)


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# --- the clause, on its own -------------------------------------------------------------

REFERENCE = "31.7 31.2 31.9 31.4\n32.7 32.2 32.9 32.4"
LAYER = "Korea 31.7 31.2 31.9 31.4\nChina 32.7 32.2 32.9 32.4"
FULL = "Korea 31.7 31.2 31.9 31.4\nChina 32.7 32.2 32.9 32.4"
MISSING_ROW = "Korea 31.7 31.2 31.9 31.4\n"


def _p(**over):
    base = dict(
        own_layer_text=LAYER,
        own_layer_drop_reference=REFERENCE,
        is_born_digital=True,
        prior_ocr_layer_suspect=False,
        control_byte_digit_hits=0,
        control_byte_scan_failed=False,
        minus_as_digit_hits=0,
        minus_as_digit_scan_failed=False,
        has_unrecovered_symbol_glyphs=False,
        has_encoding_hygiene_suspect=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _out(text: str, engine: str = "qwen") -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine=engine)


def test_a_dropped_row_fails_and_names_every_number() -> None:
    clause = text_layer_drops_clause(_out(MISSING_ROW), _p())
    assert clause is not None and clause.result is CheckResult.FAIL
    assert clause.check is Check.TEXT_LAYER_DROPS
    assert clause.reason == TEXT_LAYER_DROPS_REASON.format(count=4)
    # COUNT ONLY on the page: no dropped number is printed (#1218 ruling)
    assert not any(n in clause.reason for n in ("32.7", "32.2", "32.9", "32.4"))
    assert page_dropped_numbers(_out(MISSING_ROW), _p()) == ("32.7", "32.2", "32.9", "32.4")
    assert text_layer_drops_clause(_out(FULL), _p()) is None


def test_the_same_page_with_and_without_the_row_differs_only_in_the_clause() -> None:
    got = {
        name: [c.check for c in silent_success_clauses(_out(text), _p())]
        for name, text in (("full", FULL), ("short", MISSING_ROW))
    }
    assert got["full"] == []
    assert got["short"] == [Check.TEXT_LAYER_DROPS]


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({"own_layer_text": ""}, id="no own layer"),
        pytest.param({"own_layer_drop_reference": ""}, id="empty reference is not checked"),
        pytest.param({"own_layer_drop_reference": "  \n"}, id="blank reference"),
        pytest.param({"is_born_digital": False}, id="refused layer"),
        pytest.param({"minus_as_digit_hits": 1}, id="damaged: minus as digit"),
        pytest.param({"control_byte_scan_failed": True}, id="damaged: scan failed"),
        pytest.param({"has_unrecovered_symbol_glyphs": True}, id="damaged: symbol glyphs"),
        pytest.param({"prior_ocr_layer_suspect": True}, id="earlier OCR layer never votes"),
    ],
)
def test_the_clause_emits_nothing_when_the_layer_cannot_witness(over) -> None:
    assert text_layer_drops_clause(_out(MISSING_ROW), _p(**over)) is None
    # and nothing at all, not an UNKNOWN: the pair differs exactly by the clause
    assert text_layer_drops_clause(_out(MISSING_ROW), _p()) is not None


@pytest.mark.parametrize("engine", ["native", "", "blank"])
def test_a_native_or_empty_engine_gets_no_clause(engine: str) -> None:
    assert text_layer_drops_clause(_out(MISSING_ROW, engine), _p()) is None
    assert text_layer_drops_clause(_out(MISSING_ROW, "qwen"), _p()) is not None


def test_a_failed_page_marker_gets_no_clause() -> None:
    marker = "[page 1 failed: model returned nothing]"
    assert text_layer_drops_clause(_out(marker), _p()) is None


def test_the_cosmetic_flag_does_not_suppress_it() -> None:
    # #1185's dispute is softened by this flag; an omission is not (orchestrator decision).
    plain = text_layer_drops_clause(_out(MISSING_ROW), _p())
    cosmetic = text_layer_drops_clause(_out(MISSING_ROW), _p(has_encoding_hygiene_suspect=True))
    assert plain is not None and cosmetic == plain


_WITHHELD_BODY = "Prose.\n\n[page 1 failed: unverifiable table \u2014 see image]"


@pytest.mark.parametrize("status", [PageStatus.WARNING, PageStatus.ERROR])
@pytest.mark.parametrize(
    "mode",
    [FailureMode.TABLE_WITHHELD, FailureMode.TABLE_WITHHELD_UNCONFIRMED],
)
def test_a_table_socr_withheld_gets_no_drop_clause(status, mode) -> None:
    """Merge gate (Fable P2): the numbers of a withheld table are missing BY DESIGN, so the
    page must not say the model reading omits them. Pinned as a difference: the same page
    and layer, with and without the withhold."""
    plain = PageOutput(page_num=1, text=MISSING_ROW, status=status, engine="qwen")
    held = PageOutput(page_num=1, text=MISSING_ROW, status=status, engine="qwen", failure_mode=mode)
    assert text_layer_drops_clause(plain, _p()) is not None, "premise: the page drops a row"
    assert text_layer_drops_clause(held, _p()) is None
    assert page_dropped_numbers(held, _p()) == ()


def test_a_withheld_marker_in_the_body_or_a_ladder_disposition_gets_no_drop_clause() -> None:
    marker = _out(MISSING_ROW + "\n\n" + _WITHHELD_BODY)
    assert text_layer_drops_clause(_out(MISSING_ROW), _p()) is not None
    assert text_layer_drops_clause(marker, _p()) is None
    disposed = _p(table_ladder_disposition=FailureMode.TABLE_WITHHELD)
    assert text_layer_drops_clause(_out(MISSING_ROW), disposed) is None
    # a disposition that is not the withhold does not suppress it
    other = _p(table_ladder_disposition=FailureMode.TABLE_UNVERIFIED)
    assert text_layer_drops_clause(_out(MISSING_ROW), other) is not None


def test_a_furniture_number_in_the_reading_never_creates_a_drop() -> None:
    """No reading-side furniture subtraction: a faithful row whose number equals a running
    head's or a folio's is not marked, and a transcribed head MAY hide a real drop (the
    documented limitation of the lower bound)."""
    p = _p(own_layer_drop_reference="2020 5.1 4.9")
    assert text_layer_drops_clause(_out("| Year | 2020 | 5.1 | 4.9 |"), p) is None
    # the body 2020 is dropped but a transcribed head holds a 2020: hidden, by design
    assert text_layer_drops_clause(_out("Year 2020\nrow 5.1 4.9"), p) is None
    # the same page with a real drop of another value is marked
    assert text_layer_drops_clause(_out("Year 2020\nrow 5.1"), p) is not None


def test_the_mark_is_not_a_reading_number() -> None:
    """The mark names the dropped numbers; the next pass must not count them as shipped."""
    out, verdict = apply_verdict_marks(_out(MISSING_ROW), _p(), 1)
    assert out.text.startswith("> [page 1 unchecked: text_layer_drops=FAIL; scope=page; ")
    assert "at least 4 number(s)" in out.text.splitlines()[0]
    assert "32.7" not in out.text
    assert reading_numbers_text(out.text, 1).strip() == MISSING_ROW.strip()
    again, _ = apply_verdict_marks(out, _p(), 1, published=verdict)
    assert again.text == out.text and again.status is PageStatus.WARNING
    assert out.failure_mode is FailureMode.NUMBERS_DROPPED_FROM_READING


def test_the_clause_is_emitted_last_and_the_numbers_slot_is_unchanged() -> None:
    p = _p()
    # a reading that BOTH ships a disputed number and drops a row
    text = "Korea 31.7 31.2 31.9 31.5\n"
    clauses = silent_success_clauses(_out(text), p)
    assert [c.check for c in clauses] == [Check.TEXT_LAYER_NUMBERS, Check.TEXT_LAYER_DROPS]
    # #1185's count-of-clauses-ahead slot ignores the new clause
    published = verdict_module.PageVerdict(page_num=1, clauses=(clauses[1],))
    extended = with_text_layer_numbers_clause(published, clauses[0], _out(text), p, frozenset())
    assert [c.check for c in extended.clauses] == [Check.TEXT_LAYER_NUMBERS, Check.TEXT_LAYER_DROPS]
    # the FAIL of the dispute outranks: the page records #1185's mode, not the drop's
    out, _ = apply_verdict_marks(_out(text), p, 1)
    assert out.failure_mode is FailureMode.NUMBERS_DISAGREE_WITH_TEXT_LAYER


def test_the_mode_is_permanent_and_guarded_with_the_other_two() -> None:
    assert FailureMode.NUMBERS_DROPPED_FROM_READING in PERMANENTLY_UNRESOLVABLE_MODES
    assert NUMBER_CHECK_MODES[FailureMode.NUMBERS_DROPPED_FROM_READING] is Check.TEXT_LAYER_DROPS
    assert set(NUMBER_CHECK_MODES) >= {
        FailureMode.NUMBERS_DISAGREE_WITH_TEXT_LAYER,
        FailureMode.NUMBERS_UNCHECKED,
    }


def test_a_stale_reference_is_not_a_clause_for_a_mock_assessment() -> None:
    """A MagicMock assessment answers every getattr: it must read as "not checked"."""
    assert text_layer_drops_clause(_out(MISSING_ROW), mock.MagicMock()) is None


# --- a run, end to end ------------------------------------------------------------------

_LABELS = (
    "Korea China Mexico Brazil Chile Peru Spain Italy Japan Egypt Ghana Kenya "
    "Nepal Qatar Yemen Sudan Libya Malta Haiti Cuba Fiji Laos Togo Mali"
).split()
ROWS_PER_PAGE = 12


def _page_rows(page_index: int) -> list[str]:
    """Rows whose numbers and labels are distinct on every page, so none is furniture, and
    numerous enough that the page is a table the native lane refuses (rotated)."""
    base = page_index * ROWS_PER_PAGE
    return [
        f"{_LABELS[base + j]} {page_index + 3}{j}.7 {page_index + 3}{j}.2 "
        f"{page_index + 3}{j}.9 {page_index + 3}{j}.4"
        for j in range(ROWS_PER_PAGE)
    ]


def _rotated_pdf(path: Path) -> Path:
    """Rotated born-digital table rows (the native table lane refuses them, so a model
    reads the page); the layer is the page's own."""
    import fitz

    doc = fitz.open()
    for k in range(2):
        page = doc.new_page(width=612, height=792)
        for i, row in enumerate(_page_rows(k)):
            page.insert_text((100 + i * 20, 700), row, fontsize=9, rotate=90)
    doc.save(path)
    doc.close()
    return path


class _Reader:
    """The stub model reading: each page's rows, minus the last row when ``drop``."""

    name = "qwen"

    def __init__(self, drop: bool, pages_read: list[int], extra: str = "") -> None:
        self._drop = drop
        self._pages_read = pages_read
        self._extra = extra

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        self._pages_read.extend(page_nums)
        outs = []
        for n in page_nums:
            rows = _page_rows(n - 1)
            rows = rows[:-1] if self._drop else rows
            outs.append(
                PageOutput(
                    page_num=n,
                    text="\n".join([*rows, *([self._extra] if self._extra else [])]),
                    status=PageStatus.SUCCESS,
                    engine="qwen",
                )
            )
        return outs


#: A table with a header and delimiter and only placeholder cells: the exact final body fails
#: ``table_content_defect``, so a run records a ``table_structure_failed`` event (site
#: ``final_body``) and an "invalid final table emission" document note AFTER its own
#: ``numbers_dropped`` ones, and the page keeps its text and both clauses.
_EMPTY_TABLE = "| A | B |\n| --- | --- |\n| - | - |"


def _run(
    out_dir: Path,
    pdf: Path,
    monkeypatch,
    *,
    drop: bool,
    pages_read: list[int],
    ladder=True,
    late_defect: bool = False,
    extra: str = "",
    quiet: bool = True,
):
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _Reader(drop, pages_read, extra))
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=quiet,
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
        if late_defect:
            embed = pipe._describe_and_embed_figures

            def with_empty_table(state, result, output_dir, text):
                return embed(state, result, output_dir, text).replace(
                    "## Page 2", _EMPTY_TABLE + "\n\n## Page 2", 1
                )

            pipe._describe_and_embed_figures = with_empty_table
        return pipe.process(pdf, output_dir=out_dir)


def _drop_clauses(sidecar: dict) -> list[dict]:
    return [
        c
        for c in (sidecar.get("verdict") or {}).get("clauses") or []
        if c["check"] == Check.TEXT_LAYER_DROPS.value
    ]


def _levels(out: Path, result) -> dict:
    from socr.core.result import contract_status_for

    sidecar = _page(out)[0]
    doc = json.loads((_doc_dir(out) / "metadata.json").read_text())
    root = json.loads((out / "metadata.json").read_text())
    [entry] = root["files"].values()
    audit_path = _doc_dir(out) / "audit_log.json"
    counts = json.loads(audit_path.read_text())["counts"] if audit_path.is_file() else {}
    return {
        "page": (sidecar["status"], sidecar["failure_mode"]),
        "clause": [c["result"] for c in _drop_clauses(sidecar)],
        "mark": "text_layer_drops=FAIL" in _page(out)[1],
        "document": result.status.value,
        "metadata": doc["status"],
        "root index": entry["status"],
        "cli": contract_status_for(result).value,
        "error note": "omits numbers" in (result.error or ""),
        "audit event": counts.get("numbers_dropped_from_reading", 0),
    }


@pytest.mark.parametrize("ladder", [True, False], ids=["provider", "no provider"])
def test_one_dropped_row_changes_every_level_and_only_with_a_model_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ladder: bool
) -> None:
    levels = {}
    reads = {}
    for drop in (False, True):
        out = tmp_path / f"out{int(drop)}"
        read: list[int] = []
        result = _run(
            out,
            _rotated_pdf(tmp_path / f"p{int(drop)}.pdf"),
            monkeypatch,
            drop=drop,
            pages_read=read,
            ladder=ladder,
        )
        reads[drop] = read
        levels[drop] = _levels(out, result)
    kept, dropped = levels[False], levels[True]
    if ladder:
        assert reads[True], "fixture premise: with a provider the model must read the page"
    else:
        assert not reads[True], "fixture premise: with no provider nothing reads the page"
    if reads[True]:
        # a model read the page: the row is the whole difference, at every level
        assert dropped["page"] == ("warning", "numbers_dropped_from_reading")
        assert dropped["clause"] == ["FAIL"] and kept["clause"] == []
        assert {k for k in kept if kept[k] != dropped[k]} == set(kept), (kept, dropped)
        assert (kept["document"], dropped["document"]) == ("success", "audit_failed")
        for level in ("metadata", "root index", "cli"):
            assert (kept[level], dropped[level]) == ("completed", "partial"), level
        assert dropped["audit event"] == 2 and kept["audit event"] == 0
    else:
        # nothing read the page, so the reading cannot drop anything: the arms agree
        assert kept == dropped
        assert dropped["clause"] == []


def test_no_dropped_number_is_printed_on_the_page_and_the_audit_event_lists_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two arms, one row apart: the dropped row's numbers are absent from the shipped page
    in the drop arm (only the count-only mark differs), and sit in the audit event's data."""
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    arms = {}
    for drop in (False, True):
        out = tmp_path / f"out{int(drop)}"
        _run(out, pdf, monkeypatch, drop=drop, pages_read=[])
        arms[drop] = out
    dropped_row = _page_rows(0)[-1].split()[1:]
    kept_md, drop_md = _page(arms[False])[1], _page(arms[True])[1]
    assert all(n in kept_md for n in dropped_row), "fixture premise: the full reading lacks them"
    assert not any(n in drop_md for n in dropped_row), drop_md
    assert drop_md.startswith("> [page 1 unchecked: text_layer_drops=FAIL; ")
    doc_md = lambda out: next(p for p in _doc_dir(out).glob("*.md")).read_text()  # noqa: E731
    assert not any(n in doc_md(arms[True]) for n in dropped_row)
    events = json.loads((_doc_dir(arms[True]) / "audit_log.json").read_text())["events"]
    [event] = [
        e for e in events if e["kind"] == "numbers_dropped_from_reading" and e["page_num"] == 1
    ]
    assert event["data"] == {"page": 1, "dropped": dropped_row}


def test_the_dropped_page_is_kept_on_resume_and_an_error_one_is_reprocessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    _run(out, pdf, monkeypatch, drop=True, pages_read=[])
    before = _page(out)
    assert _drop_clauses(before[0]), "fixture premise: page 1 was not marked"
    _crash_after_page_one(out)
    resume: list[int] = []
    _run(out, pdf, monkeypatch, drop=True, pages_read=resume)
    assert 2 in resume and 1 not in resume, resume
    assert _page(out) == before

    # The same sidecar and mode with a status that is neither SUCCESS nor WARNING is not a
    # page this run demoted (the guard of #1185's modes): reprocess.
    side = _doc_dir(out) / "pages" / "00001.json"
    data = json.loads(side.read_text())
    data["winning_output"]["status"] = PageStatus.ERROR.value
    side.write_text(json.dumps(data))
    _crash_after_page_one(out)
    again: list[int] = []
    _run(out, pdf, monkeypatch, drop=True, pages_read=again)
    assert 1 in again, again


# --- socr-refinalize-own-text -----------------------------------------------------------


def _without_the_clause(m: pytest.MonkeyPatch) -> None:
    m.setattr(verdict_module, "text_layer_drops_clause", lambda output, p: None)


def test_refinalize_equals_a_run_of_this_code_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from socr.pipeline.refinalize_own_text import refinalize_document

    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    fresh, old = tmp_path / "fresh", tmp_path / "old"
    _run(fresh, pdf, monkeypatch, drop=True, pages_read=[])
    with monkeypatch.context() as m:
        _without_the_clause(m)
        _run(old, pdf, monkeypatch, drop=True, pages_read=[])
    assert _drop_clauses(_page(old)[0]) == [], "fixture premise: the old run had the clause"
    assert _drop_clauses(_page(fresh)[0]), "fixture premise: a fresh run does not mark"

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    shutil.copy(pdf, pdf_dir / pdf.name)
    changes = refinalize_document(_doc_dir(old), pdf_dir)
    assert {c.page_num for c in changes} == {1, 2}
    assert {c.drops for c in changes} == {"FAIL"}

    for n in (1, 2):
        (re_side, re_body), (fr_side, fr_body) = _page(old, n), _page(fresh, n)
        assert re_body == fr_body
        assert _comparable(re_side) == _comparable(fr_side)
    doc_md = lambda out: next(p for p in _doc_dir(out).glob("*.md")).read_text()  # noqa: E731
    assert doc_md(old) == doc_md(fresh)
    meta = lambda out: json.loads((_doc_dir(out) / "metadata.json").read_text())  # noqa: E731
    assert {k: meta(old).get(k) for k in ("status", "error")} == {
        k: meta(fresh).get(k) for k in ("status", "error")
    }
    audit = lambda out: json.loads((_doc_dir(out) / "audit_log.json").read_text())  # noqa: E731
    assert audit(old)["counts"] == audit(fresh)["counts"]
    assert audit(old)["events"] == audit(fresh)["events"], "event data differs"

    snapshot = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
    assert refinalize_document(_doc_dir(old), pdf_dir) == []
    assert {p: p.read_bytes() for p in old.rglob("*") if p.is_file()} == snapshot


def test_refinalize_on_a_run_of_this_code_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from socr.pipeline.refinalize_own_text import refinalize_document

    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    for drop in (True, False):
        out = tmp_path / f"fresh{int(drop)}"
        _run(out, pdf, monkeypatch, drop=drop, pages_read=[])
        pdf_dir = tmp_path / f"pdfs{int(drop)}"
        pdf_dir.mkdir()
        shutil.copy(pdf, pdf_dir / pdf.name)
        snapshot = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
        assert refinalize_document(_doc_dir(out), pdf_dir) == []
        assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == snapshot


def test_refinalize_adds_the_clause_to_a_page_that_already_has_the_numbers_clause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run finished with #1185 and before #1218 must still receive the drop clause
    (Fable A4): keyed on its own check, not on ``text_layer_numbers``."""
    from socr.pipeline.refinalize_own_text import refinalize_page

    p = _p()
    with mock.patch.object(verdict_module, "text_layer_drops_clause", lambda o, p: None):
        out_old, verdict_old = apply_verdict_marks(_out("Korea 31.7 31.2 31.9 31.5\n"), p, 1)
    assert [c.check for c in verdict_old.clauses] == [Check.TEXT_LAYER_NUMBERS]
    sidecar = {
        "page_num": 1,
        "terminal": True,
        "winning_output": out_old.to_dict(),
        "verdict": verdict_old.to_dict(),
    }
    analysis = SimpleNamespace(
        own_layer_text=LAYER,
        own_layer_drop_reference=REFERENCE,
        is_born_digital=True,
        control_byte_digit_hits=0,
        control_byte_scan_failed=False,
        minus_as_digit_hits=0,
        minus_as_digit_scan_failed=False,
        has_unrecovered_symbol_glyphs=False,
        has_encoding_hygiene_suspect=False,
        invisible_text_over_raster=False,
        invisible_text_scan_failed=False,
        prior_ocr_layer=False,
        prior_ocr_layer_scan_failed=False,
    )
    done = refinalize_page(sidecar, analysis)
    assert done is not None
    _out2, verdict = done
    assert [c.check for c in verdict.clauses] == [Check.TEXT_LAYER_NUMBERS, Check.TEXT_LAYER_DROPS]
    # a second pass over what it wrote adds nothing
    sidecar2 = {**sidecar, "winning_output": _out2.to_dict(), "verdict": verdict.to_dict()}
    assert refinalize_page(sidecar2, analysis) is None


# --- parity of a re-finalized or resumed run with a fresh one ---------------------------


def _artifacts(out: Path) -> dict:
    """Everything a reader of a run looks at, minus what the clock writes."""
    doc = _doc_dir(out)
    meta = json.loads((doc / "metadata.json").read_text())
    root = json.loads((out / "metadata.json").read_text())
    [entry] = root["files"].values()
    pages = sorted((doc / "pages").glob("[0-9]*.json"))
    return {
        "audit_log": json.loads((doc / "audit_log.json").read_text()),
        "document": {k: meta.get(k) for k in ("status", "error", "tables")},
        "root index": {k: entry.get(k) for k in ("status", "error", "tables")},
        "sidecars": {p.name: _comparable(json.loads(p.read_text())) for p in pages},
        "fragments": {p.name: p.with_suffix(".md").read_text() for p in pages},
        "document md": next(p for p in doc.glob("*.md")).read_text(),
    }


def _old_run_dir(tmp_path: Path, pdf: Path, monkeypatch) -> Path:
    """A run made before the clause existed, on the same fixture with the late defect."""
    old = tmp_path / "old"
    with monkeypatch.context() as m:
        _without_the_clause(m)
        _run(old, pdf, monkeypatch, drop=True, pages_read=[], late_defect=True)
    return old


def _pdf_dir(tmp_path: Path, pdf: Path) -> Path:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    shutil.copy(pdf, pdf_dir / pdf.name)
    return pdf_dir


def _first_difference(got: dict, want: dict) -> list[str]:
    return sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))


def test_the_fixture_has_a_competing_late_event_and_a_second_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Premise of the parity tests: a fresh run records events and a note AFTER ours."""
    out = tmp_path / "fresh"
    _run(
        out,
        _rotated_pdf(tmp_path / "doc.pdf"),
        monkeypatch,
        drop=True,
        pages_read=[],
        late_defect=True,
    )
    art = _artifacts(out)
    page_one = [e["kind"] for e in art["audit_log"]["events"] if e["page_num"] == 1]
    assert page_one.index("numbers_dropped_from_reading") < page_one.index("table_structure_failed")
    sidecar_kinds = [e["kind"] for e in art["sidecars"]["00001.json"]["audit_events"]]
    assert sidecar_kinds.index("numbers_dropped_from_reading") < sidecar_kinds.index(
        "table_structure_failed"
    )
    error = art["document"]["error"]
    assert error.index("omits numbers") < error.index("invalid final table emission")


def test_refinalize_matches_a_fresh_run_event_for_event_and_note_for_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from socr.pipeline.refinalize_own_text import refinalize_document

    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    fresh = tmp_path / "fresh"
    _run(fresh, pdf, monkeypatch, drop=True, pages_read=[], late_defect=True)
    old = _old_run_dir(tmp_path, pdf, monkeypatch)
    assert "numbers_dropped_from_reading" not in json.dumps(_artifacts(old)["audit_log"])
    refinalize_document(_doc_dir(old), _pdf_dir(tmp_path, pdf))
    got, want = _artifacts(old), _artifacts(fresh)
    assert _first_difference(got, want) == [], _first_difference(got, want)
    for key in want:
        assert got[key] == want[key], key


def test_an_interrupted_pass_is_repaired_on_the_next_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sidecar carries the clause and lost its event; the document's event has no data.
    The next pass restores both from the analysis, and the result is a fresh run's."""
    from socr.pipeline.refinalize_own_text import refinalize_document

    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    fresh = tmp_path / "fresh"
    _run(fresh, pdf, monkeypatch, drop=True, pages_read=[], late_defect=True)
    old = _old_run_dir(tmp_path, pdf, monkeypatch)
    pdf_dir = _pdf_dir(tmp_path, pdf)
    refinalize_document(_doc_dir(old), pdf_dir)
    assert _artifacts(old) == _artifacts(fresh), "fixture premise: the first pass is not whole"

    kind = "numbers_dropped_from_reading"
    for n in (1, 2):
        side = _doc_dir(old) / "pages" / f"{n:05d}.json"
        data = json.loads(side.read_text())
        data["audit_events"] = [e for e in data["audit_events"] if e["kind"] != kind]
        side.write_text(json.dumps(data, indent=2))
    audit_path = _doc_dir(old) / "audit_log.json"
    audit = json.loads(audit_path.read_text())
    for e in audit["events"]:
        if e["kind"] == kind:
            e["data"] = {}
    audit_path.write_text(json.dumps(audit, indent=2))
    assert _artifacts(old) != _artifacts(fresh), "fixture premise: nothing was broken"

    refinalize_document(_doc_dir(old), pdf_dir)
    assert _first_difference(_artifacts(old), _artifacts(fresh)) == []
    assert _artifacts(old) == _artifacts(fresh)
    snapshot = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
    assert refinalize_document(_doc_dir(old), pdf_dir) == []
    assert {p: p.read_bytes() for p in old.rglob("*") if p.is_file()} == snapshot


def test_a_resumed_run_keeps_the_complete_record_of_a_fresh_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    fresh, resumed = tmp_path / "fresh", tmp_path / "resumed"
    _run(fresh, pdf, monkeypatch, drop=True, pages_read=[], late_defect=True)
    _run(resumed, pdf, monkeypatch, drop=True, pages_read=[], late_defect=True)
    _crash_after_page_one(resumed)
    again: list[int] = []
    _run(resumed, pdf, monkeypatch, drop=True, pages_read=again, late_defect=True)
    # page 1 ends ERROR (invalid final emission), so a resume re-reads it: that is the
    # resume path with the most to lose, and the record must come out the same.
    assert 1 in again and 2 in again, again
    got, want = _artifacts(resumed), _artifacts(fresh)
    # the re-read page 1 again carries its clause and its list, as the fresh run's does
    assert (
        got["sidecars"]["00001.json"]["audit_events"]
        == want["sidecars"]["00001.json"]["audit_events"]
    )
    assert any(
        e["kind"] == "numbers_dropped_from_reading" and e["data"]["dropped"]
        for e in got["sidecars"]["00001.json"]["audit_events"]
    )
    assert _first_difference(got, want) == [], _first_difference(got, want)


# --- the strings that say where the numbers are ------------------------------------------


def test_the_document_note_and_cli_line_do_not_claim_the_numbers_are_on_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from socr.core.verdict import numbers_dropped_cli_line, numbers_dropped_note

    for text in (numbers_dropped_note([1, 2]), numbers_dropped_cli_line([1, 2])):
        assert "named on" not in text
        assert "audit log" in text and "count on each page" in text
    # the run writes that same note, and no dropped value
    out = tmp_path / "out"
    printed = io.StringIO()
    # not quiet: the summary lines the CLI shows are printed by the orchestrator's console
    monkeypatch.setattr(
        orchestrator_module, "console", Console(file=printed, width=10_000, force_terminal=False)
    )
    result = _run(
        out, _rotated_pdf(tmp_path / "doc.pdf"), monkeypatch, drop=True, pages_read=[], quiet=False
    )
    assert numbers_dropped_note([1, 2]) in (result.error or "")
    dropped = _page_rows(0)[-1].split()[1:] + _page_rows(1)[-1].split()[1:]
    assert not any(n in (result.error or "") for n in dropped)
    # and the run PRINTS the shared line, for the pages that dropped, not its own wording
    assert numbers_dropped_cli_line([1, 2]) in " ".join(printed.getvalue().split())
    # difference: the run that drops nothing prints no such line
    quiet_out, kept = tmp_path / "kept", io.StringIO()
    monkeypatch.setattr(
        orchestrator_module, "console", Console(file=kept, width=10_000, force_terminal=False)
    )
    _run(
        quiet_out,
        _rotated_pdf(tmp_path / "doc2.pdf"),
        monkeypatch,
        drop=False,
        pages_read=[],
        quiet=False,
    )
    assert "omit numbers the page's own text layer prints" not in " ".join(kept.getvalue().split())


#: The helpers ``process()``'s figure phase calls after the drops event is recorded. Their
#: bodies are part of "what a run records after" and are scanned with the assemble block.
_FIGURE_PHASE_HELPERS = (
    "_drop_chart_region_duplicates",
    "_describe_and_embed_figures",
    "_describe_crop_refs",
    "_record_figure_recoverable_labels",
)


def _kinds_in(text: str) -> set[str]:
    import re

    kinds: set[str] = set()
    for m in re.finditer(r"\bkind=(.*?)(?:engine=|detail=|data=|\))", text, re.S):
        kinds |= set(re.findall(r'"([a-z0-9_]+)"', m.group(1)))
    return kinds


def _method_source(source: str, name: str) -> str:
    head = f"    def {name}("
    assert source.count(head) == 1, name
    start = source.index(head)
    nxt = source.find("\n    def ", start + 1)
    return source[start : nxt if nxt != -1 else len(source)]


def test_the_after_lists_cover_what_the_orchestrator_appends_later() -> None:
    """``EVENT_KINDS_AFTER_NUMBERS_DROPPED`` is a copy of what the orchestrator records after
    our event; a kind added there and not here would put re-finalized events in the wrong
    place. Pinned BOTH ways over every ``kind=`` after the drops event: the rest of the
    assemble block and ``process()``'s figure phase (including the helpers it calls and the
    final-body ``table_structure_failed`` site). A new later kind fails the first assertion,
    a stale constant entry the second."""
    from socr.core.verdict import (
        EVENT_KINDS_AFTER_NUMBERS_DROPPED,
        FINAL_BODY_EVENT_SITE,
        NUMBERS_DROPPED_KIND,
    )

    source = Path(socr.__file__).with_name("pipeline").joinpath("orchestrator.py").read_text()
    start = source.index("for n in numbers_dropped_pages:")
    assert source.count("for n in numbers_dropped_pages:") == 1
    end = source.index("    def _document_handle_for(", start)
    appended = _kinds_in(source[start:end])
    for helper in _FIGURE_PHASE_HELPERS:
        appended |= _kinds_in(_method_source(source, helper))
    appended.discard(NUMBERS_DROPPED_KIND)
    assert appended, "the scan found nothing"
    # table_structure_failed is recorded earlier too; only its final_body site follows,
    # and that site is pinned separately below.
    appended.discard("table_structure_failed")
    listed = set(EVENT_KINDS_AFTER_NUMBERS_DROPPED)
    assert appended <= listed, (
        f"appended later, missing from the constant: {sorted(appended - listed)}"
    )
    assert listed <= appended, f"in the constant, never appended later: {sorted(listed - appended)}"
    assert f'"site": "{FINAL_BODY_EVENT_SITE}"' in source[start:end]


# --- evidence is never erased; existing entries converge (Astra P0-A, P0-B) --------------

_DROPPED_KIND = "numbers_dropped_from_reading"
_DISAGREE_KIND = "numbers_disagree_with_text_layer"


def _tree(root: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _fresh_and_pdf_dir(tmp_path: Path, monkeypatch, **run):
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    fresh = tmp_path / "fresh"
    _run(fresh, pdf, monkeypatch, pages_read=[], **run)
    return fresh, _pdf_dir(tmp_path, pdf)


def _reassessment(monkeypatch, how: str) -> None:
    """The next assessment of every page: no reference at all (a failed ``get_text`` reads
    so), or a reference with two more numbers the reading lacks (a different count)."""
    from socr.core.born_digital import BornDigitalDetector

    original = BornDigitalDetector._own_layer_drop_reference
    if how == "failed":
        monkeypatch.setattr(BornDigitalDetector, "_own_layer_drop_reference", lambda s, p: "")
    else:
        monkeypatch.setattr(
            BornDigitalDetector,
            "_own_layer_drop_reference",
            lambda s, p: original(s, p) + "\n998.1 998.2",
        )


@pytest.mark.parametrize("how", ["failed", "different count"])
def test_a_persisted_list_survives_a_reassessment_that_cannot_confirm_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    from socr.pipeline.refinalize_own_text import refinalize_document

    fresh, pdf_dir = _fresh_and_pdf_dir(tmp_path, monkeypatch, drop=True)
    before = _tree(fresh)
    _reassessment(monkeypatch, how)
    skipped: Counter = Counter()
    assert refinalize_document(_doc_dir(fresh), pdf_dir, skipped=skipped) == []
    assert _tree(fresh) == before, "a persisted list or any other byte changed"
    assert not skipped, skipped


@pytest.mark.parametrize("how", ["failed", "different count"])
def test_a_list_that_cannot_be_confirmed_is_not_written_and_is_surfaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    from socr.pipeline.refinalize_own_text import LOG_NAME, refinalize_document

    fresh, pdf_dir = _fresh_and_pdf_dir(tmp_path, monkeypatch, drop=True)
    doc = _doc_dir(fresh)
    for n in (1, 2):  # the lists are gone from the sidecars and the document's audit log
        side = doc / "pages" / f"{n:05d}.json"
        data = json.loads(side.read_text())
        data["audit_events"] = [e for e in data["audit_events"] if e["kind"] != _DROPPED_KIND]
        side.write_text(json.dumps(data, indent=2))
    audit = json.loads((doc / "audit_log.json").read_text())
    audit["events"] = [e for e in audit["events"] if e["kind"] != _DROPPED_KIND]
    (doc / "audit_log.json").write_text(json.dumps(audit, indent=2))
    before = _tree(fresh)

    _reassessment(monkeypatch, how)
    skipped: Counter = Counter()
    refinalize_document(doc, pdf_dir, skipped=skipped)
    after = _tree(fresh)
    changed = {p.name for p in after if after[p] != before.get(p)}
    assert changed == {LOG_NAME}, changed  # nothing invented, nothing else touched
    log = json.loads((doc / LOG_NAME).read_text())
    assert sorted(log["unwritten_drops_events"]) == ["1", "2"], log
    assert len(skipped) == 2 and all("numbers_dropped event" in k for k in skipped), skipped
    reasons = " ".join(log["unwritten_drops_events"].values())
    assert ("found none" if how == "failed" else "found") in reasons
    # A later pass that CAN confirm them writes them and clears the record.
    monkeypatch.undo()
    refinalize_document(doc, pdf_dir)
    assert "unwritten_drops_events" not in json.loads((doc / LOG_NAME).read_text())
    events = json.loads((doc / "audit_log.json").read_text())["events"]
    assert [e["data"]["dropped"] for e in events if e["kind"] == _DROPPED_KIND]


def test_an_artifact_in_the_earlier_tools_order_converges_to_a_fresh_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page with a #1185 dispute, no drops, and a final-body event: the earlier tool put
    its event after that one and its note after the final-emission note, in the sidecar, the
    document's audit log and both metadata files. The tool puts all four back."""
    from socr.pipeline.refinalize_own_text import refinalize_document

    fresh, pdf_dir = _fresh_and_pdf_dir(
        tmp_path, monkeypatch, drop=False, late_defect=True, extra="Total 999.5"
    )
    doc = _doc_dir(fresh)
    art = _artifacts(fresh)
    kinds = [e["kind"] for e in art["audit_log"]["events"] if e["page_num"] == 1]
    assert _DISAGREE_KIND in kinds and _DROPPED_KIND not in kinds, "fixture premise"
    assert kinds.index(_DISAGREE_KIND) < kinds.index("table_structure_failed")
    notes = art["document"]["error"]
    assert notes.index("does not hold") < notes.index("invalid final table emission")

    def move_after(events: list[dict], page_num: int | None) -> list[dict]:
        late = next(e for e in events if e["kind"] == "table_structure_failed")
        moved = next(
            e
            for e in events
            if e["kind"] == _DISAGREE_KIND and e.get("page_num", page_num) == page_num
        )
        rest = [e for e in events if e is not moved]
        at = next(i for i, e in enumerate(rest) if e is late)
        return [*rest[: at + 1], moved, *rest[at + 1 :]]

    side = doc / "pages" / "00001.json"
    data = json.loads(side.read_text())
    data["audit_events"] = move_after(data["audit_events"], None)
    side.write_text(json.dumps(data, indent=2))
    audit_path = doc / "audit_log.json"
    audit = json.loads(audit_path.read_text())
    audit["events"] = move_after(audit["events"], 1)
    audit_path.write_text(json.dumps(audit, indent=2))
    for path in (doc / "metadata.json", fresh / "metadata.json"):
        meta = json.loads(path.read_text())
        record = meta if "error" in meta else next(iter(meta["files"].values()))
        parts = record["error"].split("; ")
        disagree = next(p for p in parts if "does not hold" in p)
        parts.remove(disagree)
        record["error"] = "; ".join([*parts, disagree])
        path.write_text(json.dumps(meta, indent=2))
    assert _artifacts(fresh) != art, "fixture premise: nothing was reordered"

    refinalize_document(doc, pdf_dir)
    assert _artifacts(fresh) == art
    assert json.loads(audit_path.read_text())["events"] == art["audit_log"]["events"]
    snapshot = _tree(fresh)
    assert refinalize_document(doc, pdf_dir) == []
    assert _tree(fresh) == snapshot


def test_a_page_with_no_drops_is_untouched_by_the_drop_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from socr.pipeline.refinalize_own_text import refinalize_document

    fresh, pdf_dir = _fresh_and_pdf_dir(tmp_path, monkeypatch, drop=False)
    before = _tree(fresh)
    assert refinalize_document(_doc_dir(fresh), pdf_dir) == []
    assert _tree(fresh) == before


@pytest.mark.parametrize("late_defect", [False, True], ids=["plain", "late final-body defect"])
def test_a_faithful_page_with_a_number_dispute_is_byte_preserved_by_refinalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, late_defect: bool
) -> None:
    """Astra's guard: no drops (the reading holds every row), but a #1185 dispute from a
    number the layer lacks. The drop machinery must leave the run exactly as it wrote it."""
    from socr.pipeline.refinalize_own_text import refinalize_document

    fresh, pdf_dir = _fresh_and_pdf_dir(
        tmp_path, monkeypatch, drop=False, late_defect=late_defect, extra="Total 999.5"
    )
    side = json.loads((_doc_dir(fresh) / "pages" / "00001.json").read_text())
    checks = {c["check"] for c in side["verdict"]["clauses"]}
    assert "text_layer_numbers" in checks and "text_layer_drops" not in checks, checks
    before = _tree(fresh)
    skipped: Counter = Counter()
    assert refinalize_document(_doc_dir(fresh), pdf_dir, skipped=skipped) == []
    assert _tree(fresh) == before and not skipped


# --- resume must not erase a persisted list (merge gate, Astra P0-3) ---------------------


def _drops_events(out: Path) -> dict[int, list[dict]]:
    """Page number -> its ``numbers_dropped`` events, from the document audit log."""
    events = json.loads((_doc_dir(out) / "audit_log.json").read_text())["events"]
    got: dict[int, list[dict]] = {}
    for e in events:
        if e["kind"] == _DROPPED_KIND:
            got.setdefault(e["page_num"], []).append(e["data"])
    return got


def _resume_page_one(
    base: Path, dest: Path, pdf: Path, monkeypatch, *, failed_reassessment: bool, prepare=None
) -> tuple[list[int], list[int]]:
    """Copy a finished run, drop its document metadata and page 2 (a crash between pages),
    resume it, and return (pages restored by the page ledger, pages re-read)."""
    from socr.pipeline import orchestrator as orch

    shutil.copytree(base, dest)
    _crash_after_page_one(dest)
    for md in dest.rglob("metadata.json"):
        md.unlink()
    if prepare is not None:
        prepare(dest)
    restored: list[int] = []
    original = UnifiedPipeline._restore_terminal_page_state

    def spy(self, state, page_num, page_out, output_dir):
        restored.append(page_num)
        return original(self, state, page_num, page_out, output_dir)

    reread: list[int] = []
    with monkeypatch.context() as m:
        m.setattr(UnifiedPipeline, "_restore_terminal_page_state", spy)
        if failed_reassessment:
            _reassessment(m, "failed")
        _run(dest, pdf, m, drop=True, pages_read=reread)
    assert orch  # the module under test is the one imported above
    return restored, reread


def test_a_resumed_page_keeps_its_dropped_list_when_the_reassessment_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    first = tmp_path / "first"
    _run(first, pdf, monkeypatch, drop=True, pages_read=[])
    original = _drops_events(first)[1]
    assert original and original[0]["dropped"], "fixture premise: page 1 lists its numbers"

    arms = {}
    for name, failed in (("normal", False), ("failed", True)):
        dest = tmp_path / name
        restored, reread = _resume_page_one(
            first, dest, pdf, monkeypatch, failed_reassessment=failed
        )
        # the page was RESTORED by the page ledger, not reprocessed
        assert restored == [1], (name, restored)
        assert 1 not in reread and 2 in reread, (name, reread)
        arms[name] = dest

    for name, dest in arms.items():
        # the dropped page's audit event keeps the original list, never an empty one
        assert _drops_events(dest)[1] == original, name
        sidecar = json.loads((_doc_dir(dest) / "pages" / "00001.json").read_text())
        assert [
            e["data"] for e in sidecar["audit_events"] if e["kind"] == _DROPPED_KIND
        ] == original, name
        assert _drop_clauses(sidecar), name
    # and page 1's artifacts are byte-identical to the first run's, with or without the failure
    for name in ("00001.json", "00001.md"):
        want = (_doc_dir(first) / "pages" / name).read_bytes()
        for arm, dest in arms.items():
            assert (_doc_dir(dest) / "pages" / name).read_bytes() == want, (arm, name)
    # the difference being pinned: the failed reassessment changes nothing about page 1
    assert _drops_events(arms["failed"])[1] == _drops_events(arms["normal"])[1]


def test_a_list_that_cannot_be_confirmed_on_resume_is_recorded_as_unresolved_never_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Merge gate (mutation M19): no persisted list and a reassessment that finds nothing.
    The event says so (``unresolved``, with the reason) and carries NO ``dropped`` key -- an
    empty list would read as "nothing was dropped" under a FAIL clause."""
    import logging

    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    first = tmp_path / "first"
    _run(first, pdf, monkeypatch, drop=True, pages_read=[])

    def blank_the_persisted_list(dest: Path) -> None:
        side = _doc_dir(dest) / "pages" / "00001.json"
        data = json.loads(side.read_text())
        data["audit_events"] = [e for e in data["audit_events"] if e["kind"] != _DROPPED_KIND]
        side.write_text(json.dumps(data, indent=2))

    dest = tmp_path / "resumed"
    with caplog.at_level(logging.WARNING):
        restored, _ = _resume_page_one(
            first,
            dest,
            pdf,
            monkeypatch,
            failed_reassessment=True,
            prepare=blank_the_persisted_list,
        )
    assert restored == [1], restored
    assert any(
        "no dropped-number list could be confirmed" in r.getMessage() for r in caplog.records
    )
    sidecar = json.loads((_doc_dir(dest) / "pages" / "00001.json").read_text())
    assert _drop_clauses(sidecar), "premise: the restored verdict still carries the FAIL clause"
    events = {
        "sidecar": [e["data"] for e in sidecar["audit_events"] if e["kind"] == _DROPPED_KIND],
        "audit log": _drops_events(dest).get(1, []),
    }
    for where, data in events.items():
        assert len(data) == 1, (where, data)
        assert "dropped" not in data[0], (where, data)
        assert data[0]["page"] == 1 and data[0]["unresolved"], (where, data)
