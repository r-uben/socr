"""#1048: a scanned table socr could not check ships, marked unverified.

Owner ruling (2026-10-08): a table the evidence gate COULD NOT CHECK ships its text with a
visible mark; a table the evidence CONTRADICTS stays on the fail-closed floor. Every pin below
is a DIFFERENCE between runs in the same process that change one thing (what the witness
reads), never an absolute outcome measured locally (CLAUDE.md, #257).

Hermetic: ``_available_engines_for_agentic`` patched, ``_resolve_judge_model`` -> "", the OCR
witness is an injected function (no tesseract), the model is a stub engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from socr.core import manifest
from socr.core.config import EngineType, PipelineConfig
from socr.core.manifest import (
    DISPUTED_TABLE_MARK_RE,
    PAGE_UNCHECKED_MARK_RE,
    UNVERIFIED_TABLE_MARK,
    UNVERIFIED_TABLE_MARK_RE,
    disputed_table_mark,
)
from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.core.state import PageState
from socr.pipeline import orchestrator as orch
from socr.pipeline.agentic import AcceptDecision, SourceEvidenceTableJudge
from socr.pipeline.orchestrator import UnifiedPipeline

_PROSE = "Results are reported in the table below."
_TABLE = "| Item | Value |\n| --- | --- |\n| alpha | 41.5 |\n| beta | 17.2 |\n"
_MODEL_TEXT = f"{_PROSE}\n\n{_TABLE}"

#: What the witness reads. SILENT: nothing, so the gate cannot check. CONTRADICTS: a real
#: reading that lacks the model's numbers. SUPPORTS: a reading carrying them.
SILENT = ""
CONTRADICTS = "Item Value alpha 99.9 beta 88.8"
SUPPORTS = "Item Value alpha 41.5 beta 17.2"


def _strip_marks(text: str) -> str:
    """Remove the page-level and table-level marks (and their blank line) from a body."""
    out = []
    for line in text.split("\n"):
        if UNVERIFIED_TABLE_MARK_RE.match(line) or DISPUTED_TABLE_MARK_RE.match(line):
            out.append(None)
        elif PAGE_UNCHECKED_MARK_RE.match(line):
            out.append(None)
        else:
            out.append(line)
    cleaned = []
    skip_blank = False
    for line in out:
        if line is None:
            skip_blank = True
            continue
        if skip_blank and line == "":
            skip_blank = False
            continue
        skip_blank = False
        cleaned.append(line)
    return "\n".join(cleaned)


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_QWEN_LOCAL]
    )
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")


# ---------------------------------------------------------------------------
# The finalizer guard (unit)
# ---------------------------------------------------------------------------


def _out(text: str, **kw) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.WARNING, engine="qwen", **kw)


def _unverified_page() -> PageState:
    p = PageState(page_num=1)
    p.table_ladder_disposition = FailureMode.TABLE_UNVERIFIED
    return p


def test_mark_goes_directly_above_each_table_and_nothing_else_changes() -> None:
    two = f"{_PROSE}\n\n{_TABLE}\nmiddle prose\n\n{_TABLE}"
    out = manifest._apply_unverified_table_mark(_out(two), _unverified_page())
    assert out.text.count(UNVERIFIED_TABLE_MARK) == 2
    assert len(UNVERIFIED_TABLE_MARK_RE.findall(out.text)) == 2
    assert out.text.replace(f"{UNVERIFIED_TABLE_MARK}\n\n", "") == two
    assert f"{UNVERIFIED_TABLE_MARK}\n\n| Item" in out.text


def test_mark_is_idempotent_for_a_restored_page() -> None:
    once = manifest._apply_unverified_table_mark(_out(_MODEL_TEXT), _unverified_page())
    twice = manifest._apply_unverified_table_mark(once, _unverified_page())
    assert twice.text == once.text and once.text.count(UNVERIFIED_TABLE_MARK) == 1


def test_mark_needs_an_unverified_page_and_a_table() -> None:
    """Difference: the same text on a page that is not unverified is untouched."""
    plain = PageState(page_num=1)
    verified = _out(_MODEL_TEXT)
    assert manifest._apply_unverified_table_mark(verified, plain) is verified
    prose = _out(_PROSE)
    assert manifest._apply_unverified_table_mark(prose, _unverified_page()) is prose
    flagged = _out(_MODEL_TEXT, table_unverifiable="no local content evidence")
    assert UNVERIFIED_TABLE_MARK in manifest._apply_unverified_table_mark(flagged, plain).text


def test_disputed_mark_names_values_links_image_and_is_idempotent() -> None:
    p = PageState(page_num=1)
    p.d3_floor_png_ref = "![Failed table page 1](figures/failed_table_p1.png)"
    flagged = _out(_MODEL_TEXT, table_disputed="41.5, 17.2")
    once = manifest._apply_unverified_table_mark(flagged, p)
    assert disputed_table_mark("41.5, 17.2", p.d3_floor_png_ref) in once.text
    assert (
        "(figures/failed_table_p1.png)" in once.text
        and "disagrees with 2 values: 41.5, 17.2" in once.text
    )
    # No cap: all of a long list are named.
    many = ", ".join(f"{n}.5" for n in range(1, 13))
    assert many in disputed_table_mark(many, "") and "12 values" in disputed_table_mark(many, "")
    assert manifest._apply_unverified_table_mark(once, p).text == once.text
    # No rendered image: the mark still ships, without a link.
    bare = manifest._apply_unverified_table_mark(flagged, PageState(page_num=1))
    assert "](" not in bare.text.split("| Item")[0]


def test_mark_is_not_a_page_failure_marker() -> None:
    marked = manifest._apply_unverified_table_mark(_out(_TABLE), _unverified_page())
    assert not manifest.is_page_failed_marker(marked.text)


def test_output_field_round_trips_and_is_omitted_when_empty() -> None:
    assert "table_unverifiable" not in _out(_TABLE).to_dict()
    flagged = _out(_TABLE, table_unverifiable="why")
    assert PageOutput.from_dict(flagged.to_dict()).table_unverifiable == "why"


# ---------------------------------------------------------------------------
# End to end through process()
# ---------------------------------------------------------------------------


def _scan_pdf(path: Path) -> Path:
    """One page that is a bare image: no text layer, so the scanned lane owns it."""
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200), False)
    pix.set_rect(pix.irect, (235, 235, 235))
    page.insert_image(page.rect, pixmap=pix)
    doc.save(path)
    doc.close()
    return path


class _Engine:
    """Stub rung: returns ``texts[engine_type]`` and records that it was called."""

    def __init__(self, engine_type, texts: dict, calls: list) -> None:
        self.name = engine_type.value
        self._type = engine_type
        self._texts = texts
        self._calls = calls

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        self._calls.append(self._type)
        text = self._texts.get(self._type, _MODEL_TEXT)
        return [
            PageOutput(page_num=n, text=text, status=PageStatus.SUCCESS, engine=self.name)
            for n in page_nums
        ]


class _InnerAccepts:
    def assess(self, output, provider):
        return AcceptDecision(accept=True, reason="inner ok", confidence=0.9)


def _run(
    tmp_path,
    monkeypatch,
    tag,
    *,
    witness,
    out_dir=None,
    pdf=None,
    profiles=None,
    texts=None,
    calls=None,
):
    profiles = profiles or [PROFILE_QWEN_LOCAL]
    texts = texts or {}
    calls = calls if calls is not None else []
    pdf = pdf or _scan_pdf(tmp_path / f"{tag}.pdf")
    out = out_dir or tmp_path / f"out-{tag}"
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _Engine(engine_type, texts, calls))
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[p.engine for p in profiles],
                native_first=True,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                table_judge_ladder=False,
                detect_equations=False,
                save_figures=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: list(profiles)
        docs: dict = {}

        def _judge(state):
            docs["doc"] = fitz.open(str(pdf))
            return SourceEvidenceTableJudge(
                inner=_InnerAccepts(),
                get_fitz_page=lambda n: docs["doc"][n - 1],
                record_event=state.events.append,
                ocr_image_fn=lambda _pix: witness,
                native_trusted=lambda _n: False,
            )

        pipe._build_page_judge = _judge
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        result = pipe.process(pdf, output_dir=out)
    side = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    frag = next(iter(out.rglob("pages/00001.md"))).read_text()
    # A document whose only page failed closed writes no stitched .md at all.
    finals = [p for p in out.rglob("*.md") if "pages" not in p.relative_to(out).parts]
    final = finals[0].read_text() if finals else ""
    logs = list(out.rglob("audit_log.json"))
    events = json.loads(logs[0].read_text())["events"] if logs else []
    return result, side, frag, final, events


def test_could_not_check_and_disputed_ship_each_with_their_mark(tmp_path, monkeypatch) -> None:
    sil, sil_side, sil_frag, sil_final, sil_ev = _run(tmp_path, monkeypatch, "s", witness=SILENT)
    con, con_side, con_frag, con_final, _ = _run(tmp_path, monkeypatch, "c", witness=CONTRADICTS)
    sup, sup_side, sup_frag, sup_final, _ = _run(tmp_path, monkeypatch, "v", witness=SUPPORTS)

    # Could not check: the model's table ships, marked, in the fragment AND the final .md.
    assert _TABLE.rstrip() in sil_frag and _TABLE.rstrip() in sil_final
    assert sil_frag.count(UNVERIFIED_TABLE_MARK) == 1
    assert sil_final.count(UNVERIFIED_TABLE_MARK) == 1
    assert f"{UNVERIFIED_TABLE_MARK}\n\n| Item" in sil_final
    assert sil_side["failure_mode"] == FailureMode.TABLE_UNVERIFIED.value
    assert sil_side["status"] == PageStatus.WARNING.value
    assert sil_side["table_ladder_disposition"] == FailureMode.TABLE_UNVERIFIED.value
    assert sil.status is not DocumentStatus.SUCCESS
    meta = json.loads((tmp_path / "out-s" / "s" / "metadata.json").read_text())
    assert meta["tables"]["unverified_text"] == 1 and meta["tables"]["verified_text"] == 0
    trust = json.loads(next(iter((tmp_path / "out-s").rglob("tables_trust.json"))).read_text())
    assert 1 in trust["untrusted_pages"]
    assert any(e["kind"] == "source_evidence_table_reject" for e in sil_ev)

    # Disputed (owner ruling 2026-10-12): the table ships with the STRONGER mark, the
    # disputed values the check named, and a link to the page image the floor renders.
    assert _TABLE.rstrip() in con_final and _TABLE.rstrip() in con_frag
    assert UNVERIFIED_TABLE_MARK not in con_final
    assert len(DISPUTED_TABLE_MARK_RE.findall(con_final)) == 1
    assert len(DISPUTED_TABLE_MARK_RE.findall(con_frag)) == 1
    assert "41.5" in con_final.split("| Item")[0] and "17.2" in con_final.split("| Item")[0]
    assert "[page image](figures/failed_table_p1.png)" in con_final
    assert "failed: unverifiable table" not in con_final
    assert con_side["failure_mode"] == FailureMode.TABLE_REJECTED.value
    assert con_side["status"] == PageStatus.WARNING.value
    assert con.status is not DocumentStatus.SUCCESS
    # Same page, same table: the two marks differ and neither page is the floor.
    assert sil_final != con_final

    # Verified: the table, no mark, and the unverified page differs from it ONLY by the mark.
    assert UNVERIFIED_TABLE_MARK not in sup_final and UNVERIFIED_TABLE_MARK not in sup_frag
    assert _TABLE.rstrip() in sup_final
    assert _strip_marks(sil_final) == sup_final
    assert _strip_marks(sil_frag) == sup_frag
    # The promoted page (and only it) opens with the page-level mark, below the page header.
    assert PAGE_UNCHECKED_MARK_RE.search(sil_frag.split("\n", 1)[0]) is not None
    assert not PAGE_UNCHECKED_MARK_RE.search(sup_frag)
    assert sil_final.split("## Page 1\n\n", 1)[1].startswith("> [page unchecked: ")
    assert sup_final == sup_final.replace("> [page unchecked", "")
    assert PAGE_UNCHECKED_MARK_RE.match(con_frag) and "(figures/failed_table_p1.png)" in con_frag
    assert sup_side["failure_mode"] != FailureMode.TABLE_UNVERIFIED.value


def test_disputed_mark_survives_a_resume_with_no_provider_call(tmp_path, monkeypatch) -> None:
    """P2-5: the disputed (TABLE_REJECTED) page IS restored from its terminal sidecar, so this
    exercises restoration; a TABLE_UNVERIFIED page is always re-run and could not."""
    pdf = _scan_pdf(tmp_path / "r.pdf")
    out = tmp_path / "out-r"
    first: list = []
    second: list = []
    _run(tmp_path, monkeypatch, "r", witness=CONTRADICTS, out_dir=out, pdf=pdf, calls=first)
    _res, side, frag, final, _ = _run(
        tmp_path, monkeypatch, "r", witness=CONTRADICTS, out_dir=out, pdf=pdf, calls=second
    )
    assert first and second == [], "the resumed run must not call a provider"
    for body in (frag, final):
        assert len(DISPUTED_TABLE_MARK_RE.findall(body)) == 1
        assert body.count("> [page unchecked:") == 1
        assert "[page image](figures/failed_table_p1.png)" in body
    assert side["failure_mode"] == FailureMode.TABLE_REJECTED.value
    assert side["unchecked_table_kind"] == "disputed"


def test_unverified_page_is_rerun_and_still_marked_once(tmp_path, monkeypatch) -> None:
    pdf = _scan_pdf(tmp_path / "u.pdf")
    out = tmp_path / "out-u"
    _run(tmp_path, monkeypatch, "u", witness=SILENT, out_dir=out, pdf=pdf)
    _res, side, frag, final, _ = _run(
        tmp_path, monkeypatch, "u", witness=SILENT, out_dir=out, pdf=pdf
    )
    assert frag.count(UNVERIFIED_TABLE_MARK) == 1 and final.count(UNVERIFIED_TABLE_MARK) == 1
    assert side["failure_mode"] == FailureMode.TABLE_UNVERIFIED.value


# ---------------------------------------------------------------------------
# Review fixes: every table form, the page's state (not the candidate) decides, page mark
# ---------------------------------------------------------------------------

_HTML_TABLE = "<table>\n<tr><td>alpha</td><td>41.5</td></tr>\n</table>\n"


def _promoted_page(kind="disputed", values="41.5, 17.2") -> PageState:
    p = PageState(page_num=1)
    p.unchecked_table_kind = kind
    p.unchecked_table_values = values
    p.d3_floor_png_ref = "![Failed table page 1](figures/failed_table_p1.png)"
    return p


def test_html_and_pipe_tables_are_both_marked_on_a_promoted_page() -> None:
    text = f"{_PROSE}\n\n{_TABLE}\n{_HTML_TABLE}\nafter\n"
    out = manifest._apply_unverified_table_mark(_out(text), _promoted_page())
    assert len(DISPUTED_TABLE_MARK_RE.findall(out.text)) == 2
    html_at = out.text.index("<table>")
    assert DISPUTED_TABLE_MARK_RE.search(out.text[:html_at].rsplit("\n\n", 2)[-2])
    assert manifest._apply_unverified_table_mark(out, _promoted_page()).text == out.text


def test_html_table_alone_is_marked_for_a_could_not_check_page() -> None:
    out = manifest._apply_unverified_table_mark(
        _out(f"{_PROSE}\n\n{_HTML_TABLE}"), _promoted_page("unverified", "")
    )
    assert out.text.count(UNVERIFIED_TABLE_MARK) == 1


def test_mark_follows_the_page_when_the_replacement_carries_no_field() -> None:
    """P0-2: a later replacement of the incumbent has neither field; the page state decides."""
    replacement = _out(_MODEL_TEXT)  # no table_disputed / table_unverifiable
    assert replacement.table_disputed == "" and replacement.table_unverifiable == ""
    marked = manifest._apply_unverified_table_mark(replacement, _promoted_page())
    assert DISPUTED_TABLE_MARK_RE.search(marked.text) and "41.5, 17.2" in marked.text
    # Difference: the same replacement on a page that was never promoted is untouched.
    assert manifest._apply_unverified_table_mark(replacement, PageState(page_num=1)) is replacement


def test_promoted_page_opens_with_the_page_mark_and_link() -> None:
    out = manifest._apply_unverified_table_mark(_out(_MODEL_TEXT), _promoted_page())
    first = out.text.split("\n", 1)[0]
    assert PAGE_UNCHECKED_MARK_RE.match(first)
    assert "[page image](figures/failed_table_p1.png)" in first
    bare = _promoted_page()
    bare.d3_floor_png_ref = ""
    assert (
        "]("
        not in manifest._apply_unverified_table_mark(_out(_MODEL_TEXT), bare).text.split("\n", 1)[0]
    )
    # A page that only has the ladder's TABLE_UNVERIFIED (never promoted) gets no page mark.
    ladder = manifest._apply_unverified_table_mark(_out(_MODEL_TEXT), _unverified_page())
    assert not PAGE_UNCHECKED_MARK_RE.search(ladder.text)


class _RejudgeAccepts:
    def __init__(self, accept: bool) -> None:
        self.accept = accept

    def assess(self, output, provider):
        return AcceptDecision(accept=self.accept, reason="stub")


def _rejudge_setup(tmp_path):
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "source page")
    doc.save(pdf)
    doc.close()
    state = DocumentState(DocumentHandle(pdf))
    bo = PageOutput(
        page_num=1,
        text=_MODEL_TEXT,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        table_disputed="41.5, 17.2",
    )
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.attempts.append(bo)
    ps.best_output = bo
    ps.unchecked_table_kind = "disputed"
    ps.unchecked_table_values = "41.5, 17.2"
    ps.table_ladder_disposition = FailureMode.TABLE_REJECTED
    ps.d3_floor_png_ref = "![Failed table page 1](figures/failed_table_p1.png)"
    return state, ps, bo


@pytest.mark.parametrize("passes", [True, False])
def test_crop_rejudge_clears_or_keeps_the_unchecked_claims(tmp_path, passes) -> None:
    """P1-4: a patch that passes the source-evidence check retires the claims about the old
    reading; a refused patch keeps everything, link included."""
    state, ps, bo = _rejudge_setup(tmp_path)
    pipe = UnifiedPipeline(PipelineConfig(quiet=True, judge_backend="heuristic"))
    pipe._rejudge_crop_patched_page(
        state, 1, ps, bo, "old text", _RejudgeAccepts(passes), PROFILE_QWEN_LOCAL
    )
    if passes:
        assert (ps.unchecked_table_kind, bo.table_disputed) == ("", "")
        assert ps.table_ladder_disposition is None
        assert manifest._apply_unverified_table_mark(bo, ps) is bo
    else:
        assert ps.unchecked_table_kind == "disputed" and bo.table_disputed
        assert ps.table_ladder_disposition is FailureMode.TABLE_REJECTED
        assert "figures/failed_table_p1.png" in ps.d3_floor_png_ref


# ---------------------------------------------------------------------------
# Escalation is kept: the mark is applied only where the floor would be
# ---------------------------------------------------------------------------

_DISPUTED_TABLE = "| Item | Value |\n| --- | --- |\n| alpha | 99.9 |\n| beta | 88.8 |\n"
_TWO_RUNGS = [PROFILE_QWEN_LOCAL, PROFILE_GEMINI]


def test_a_later_rung_that_verifies_ships_unmarked_and_is_called(tmp_path, monkeypatch) -> None:
    """Difference against a one-rung ladder: same witness, same first-rung table (disputed).
    Only the second rung exists or not."""
    texts = {
        EngineType.QWEN: f"{_PROSE}\n\n{_DISPUTED_TABLE}",
        EngineType.GEMINI: _MODEL_TEXT,
    }
    one_calls: list = []
    two_calls: list = []
    _, one_side, one_frag, _, _ = _run(
        tmp_path, monkeypatch, "one", witness=SUPPORTS, texts=texts, calls=one_calls
    )
    _, two_side, two_frag, two_final, _ = _run(
        tmp_path,
        monkeypatch,
        "two",
        witness=SUPPORTS,
        profiles=_TWO_RUNGS,
        texts=texts,
        calls=two_calls,
    )

    assert one_calls == [EngineType.QWEN]
    assert EngineType.GEMINI in two_calls, "rung 2 must be called when rung 1 is disputed"
    # One rung: ships rung 1's table, marked disputed. Two rungs: rung 2's verified text, no mark.
    assert len(DISPUTED_TABLE_MARK_RE.findall(one_frag)) == 1 and "99.9" in one_frag
    assert not DISPUTED_TABLE_MARK_RE.search(two_frag)
    assert UNVERIFIED_TABLE_MARK not in two_frag and UNVERIFIED_TABLE_MARK not in two_final
    assert _TABLE.rstrip() in two_frag and "99.9" not in two_frag
    assert two_side["failure_mode"] != FailureMode.TABLE_REJECTED.value
    assert two_side["status"] == PageStatus.SUCCESS.value


def test_both_rungs_disputed_ships_the_last_candidate_marked(tmp_path, monkeypatch) -> None:
    other = "| Item | Value |\n| --- | --- |\n| alpha | 77.7 |\n| beta | 66.6 |\n"
    texts = {
        EngineType.QWEN: f"{_PROSE}\n\n{_DISPUTED_TABLE}",
        EngineType.GEMINI: f"{_PROSE}\n\n{other}",
    }
    calls: list = []
    _, side, frag, final, _ = _run(
        tmp_path,
        monkeypatch,
        "both",
        witness=SUPPORTS,
        profiles=_TWO_RUNGS,
        texts=texts,
        calls=calls,
    )
    assert calls == [EngineType.QWEN, EngineType.GEMINI]
    # The documented choice: the LAST (most escalated) rung that produced a table.
    assert "77.7" in frag and "99.9" not in frag and "77.7" in final
    assert len(DISPUTED_TABLE_MARK_RE.findall(frag)) == 1
    assert "77.7" in frag.split("| Item")[0]  # the mark names the shipped candidate's values
    assert side["failure_mode"] == FailureMode.TABLE_REJECTED.value


def test_page_state_round_trips_through_the_sidecar(tmp_path) -> None:
    """The mark follows the page's state, so a restored page must get that state back."""
    import json
    from unittest.mock import patch

    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    out_dir = tmp_path / "out"
    pipe = UnifiedPipeline(
        PipelineConfig(
            quiet=True,
            judge_backend="heuristic",
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
        )
    )
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
        resumed = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.unchecked_table_kind = "disputed"
    ps.unchecked_table_values = "41.5, 17.2"
    ps.table_ladder_disposition = FailureMode.TABLE_REJECTED
    bo = _out(_MODEL_TEXT, audit_passed=True)
    ps.attempts.append(bo)
    ps.best_output = bo
    sidecar = pipe._flush_page_sidecar(state, 1, out_dir, terminal=True)
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["unchecked_table_kind"] == "disputed"
    pipe._restore_terminal_page_state(resumed, 1, bo, out_dir)
    rps = resumed.pages[1]
    assert (rps.unchecked_table_kind, rps.unchecked_table_values) == ("disputed", "41.5, 17.2")


# ---------------------------------------------------------------------------
# With #1099 (born-digital pages are checked against their own text layer) and #1100 (blank pages)
# ---------------------------------------------------------------------------


def test_born_digital_rejected_candidate_fields_never_mark_the_page() -> None:
    """A born-digital page's rejected candidate carries the doubt fields now; it is not
    promoted, so it must not be marked. Difference: the same candidate on a scan is."""
    flagged = _out(_MODEL_TEXT, table_disputed="41.5", table_unverifiable="why")
    digital = PageState(page_num=1)
    digital.is_born_digital = True
    scan = PageState(page_num=1)
    scan.is_born_digital = False
    assert manifest._apply_unverified_table_mark(flagged, digital) is flagged
    assert DISPUTED_TABLE_MARK_RE.search(manifest._apply_unverified_table_mark(flagged, scan).text)


def test_a_born_digital_table_that_passes_carries_no_mark() -> None:
    digital = PageState(page_num=1)
    digital.is_born_digital = True
    passing = _out(_MODEL_TEXT)
    assert manifest._apply_unverified_table_mark(passing, digital) is passing


def test_a_blank_page_never_reaches_a_model_and_is_never_marked(tmp_path, monkeypatch) -> None:
    pdf = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    calls: list = []
    _res, side, frag, final, _ = _run(
        tmp_path, monkeypatch, "blank", witness=CONTRADICTS, pdf=pdf, calls=calls
    )
    assert calls == []
    assert "blank page" in frag
    assert "[table " not in frag and "[page unchecked" not in frag
    assert side.get("unchecked_table_kind", "") == ""


# ---------------------------------------------------------------------------
# Round 2: HTML retries, partial crop acceptance, the linked asset
# ---------------------------------------------------------------------------

_HTML_OK = "<table><tr><td>alpha</td><td>41.5</td></tr><tr><td>beta</td><td>17.2</td></tr></table>"
_HTML_BAD = "<table>\nnot rows at all\n</table>"


def _html_retry(tmp_path, monkeypatch, tag, html_table):
    texts = {
        EngineType.QWEN: f"{_PROSE}\n\n{_DISPUTED_TABLE}",
        EngineType.GEMINI: f"{_PROSE}\n\n{html_table}\n",
    }
    calls: list = []
    out = _run(
        tmp_path,
        monkeypatch,
        tag,
        witness=SUPPORTS,
        profiles=_TWO_RUNGS,
        texts=texts,
        calls=calls,
    )
    return calls, out


def test_html_retry_is_checked_not_waved_through(tmp_path, monkeypatch) -> None:
    """Rung 1 disputed pipe table; rung 2 re-emits as HTML. Before: rung 2 skipped the check,
    was accepted, and shipped unmarked. Pinned as a difference between three HTML bodies."""
    ok_calls, (_, ok_side, ok_frag, _, _) = _html_retry(tmp_path, monkeypatch, "hok", _HTML_OK)
    bad_calls, (_, bad_side, bad_frag, _, _) = _html_retry(tmp_path, monkeypatch, "hbad", _HTML_BAD)
    wrong = _HTML_OK.replace("41.5", "99.9")
    dis_calls, (_, dis_side, dis_frag, _, _) = _html_retry(tmp_path, monkeypatch, "hdis", wrong)

    assert EngineType.GEMINI in ok_calls
    # Verified HTML: accepted by the check, unmarked, shipped as emitted.
    assert "<table>" in ok_frag and "alpha" in ok_frag
    assert "[table " not in ok_frag and "[page unchecked" not in ok_frag
    assert ok_side["status"] == PageStatus.SUCCESS.value
    # Disputed HTML: promoted, every table marked, page mark first.
    assert "99.9" in dis_frag and DISPUTED_TABLE_MARK_RE.search(dis_frag)
    assert dis_frag.index("> [table disputed") < dis_frag.index("<table>")
    assert dis_side["unchecked_table_kind"] == "disputed"
    # Unparseable HTML: could-not-check, shipped marked unverified at the ladder end.
    assert "not rows at all" in bad_frag
    assert bad_frag.index(UNVERIFIED_TABLE_MARK) < bad_frag.index("<table>")
    assert bad_side["unchecked_table_kind"] == "unverified"
    assert "> [page unchecked:" in bad_frag
    assert bad_calls == [EngineType.QWEN, EngineType.GEMINI]


def _mk(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_crop_acceptance_keeps_state_while_an_unparseable_html_table_remains(tmp_path) -> None:
    state, ps, bo = _rejudge_setup(tmp_path)
    pipe = UnifiedPipeline(PipelineConfig(quiet=True, judge_backend="heuristic"))
    bo.text = f"{_PROSE}\n\n{_TABLE}\n{_HTML_BAD}\n"
    pipe._rejudge_crop_patched_page(
        state, 1, ps, bo, "old text", _RejudgeAccepts(True), PROFILE_QWEN_LOCAL
    )
    assert ps.unchecked_table_kind == "disputed"
    assert ps.table_ladder_disposition is FailureMode.TABLE_REJECTED
    assert "figures/failed_table_p1.png" in ps.d3_floor_png_ref
    assert bo.table_disputed
    # Difference: the same acceptance with no HTML left clears everything.
    state2, ps2, bo2 = _rejudge_setup(_mk(tmp_path / "b"))
    pipe._rejudge_crop_patched_page(
        state2, 1, ps2, bo2, "old text", _RejudgeAccepts(True), PROFILE_QWEN_LOCAL
    )
    assert ps2.unchecked_table_kind == "" and ps2.table_ladder_disposition is None


def test_the_linked_page_image_is_a_discovered_asset_and_replays(tmp_path) -> None:
    from socr.core.manifest import _local_page_asset_refs

    doc_dir = tmp_path / "doc"
    (doc_dir / "figures").mkdir(parents=True)
    (doc_dir / "figures" / "failed_table_p1.png").write_bytes(b"png-bytes")
    page = manifest._apply_unverified_table_mark(_out(_MODEL_TEXT), _promoted_page())
    refs = _local_page_asset_refs(page.text, doc_dir)
    assert [r.logical_path for r in refs] == ["figures/failed_table_p1.png"]
    # Replay: copy_page_assets relocates every recorded asset into a fresh directory.
    from socr.core.manifest import Manifest, ManifestEntry, copy_page_assets

    entry = ManifestEntry.__new__(ManifestEntry)
    entry.assets = refs
    mf = Manifest.__new__(Manifest)
    mf.page_count = 1
    mf.entries = {1: entry}
    dest = tmp_path / "fresh"
    copied = copy_page_assets(mf, doc_dir, dest)
    assert (dest / "figures" / "failed_table_p1.png").read_bytes() == b"png-bytes"
    assert copied
