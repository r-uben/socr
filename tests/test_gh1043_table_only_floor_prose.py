"""#1043: a page whose every model reading was rejected only for a table keeps its prose.

The whole-page table floor used to drop correct prose with the table. When every reading was
refused for a table alone and one reading's prose (outside its table blocks) matches the
page's own text layer, the prose ships WARNING and each table is a withheld-table marker.

Hermetic: ``_available_engines_for_agentic`` patched, ``_resolve_judge_model`` -> "". Every
behavioural pin is a DIFFERENCE between two runs that change one thing, never an absolute
outcome measured locally (CLAUDE.md, #257).
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from socr.core import manifest
from socr.core.config import EngineType, PipelineConfig
from socr.core.manifest import (
    PageEnding,
    PagePrimaryReason,
    SelectionProvenance,
    _select_page_output_tagged,
)
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import (
    JUDGE_OUTCOME_COMPLETED,
    JUDGE_OUTCOME_TIMEOUT,
    DocumentStatus,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState, PageState
from socr.core.table_counts import count_page_tables
from socr.pipeline import orchestrator as orch
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.prose_corroboration import (
    PROSE_CORROBORATION_MIN_TOKENS,
    corroborate_prose,
    rejection_is_table_only,
)

_PROSE_LINES = [
    "The estimated effect of the policy change on output is reported below",
    "Standard errors are clustered by industry and year in every column here",
    "We conclude that the response of prices differs across the two regimes",
    "Robustness checks using alternative samples give very similar results",
]
_TABLE = "| Variable | Mean |\n| --- | --- |\n| Output | 12.5 |\n| Prices | 13.5 |"
_TABLE_REASON = "Table 1 is missing the 'Mean' column; The last row of the table is malformed"


#: Forsythe p28's real qwen rejection (#1047 review): it names the data tables AND the figure
#: axes, legends and page number, so it is mixed and must floor.
_P28_REASON = (
    "The transcription completely omits the data tables located at the top of Figure 17 and "
    "Figure 18.; The transcription omits the axis labels, legends, and x-axis tick labels for "
    "the charts in Figure 17 and Figure 18.; The transcription omits the page number '335'."
)


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_QWEN_LOCAL]
    )
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _words(prose_lines, table_rows=("Output 12.5", "Prices 13.5")):
    """fitz-style word tuples: one baseline band per line, prose first then numeric rows."""
    out = []
    y = 50.0
    for line in list(prose_lines) + list(table_rows):
        x = 40.0
        for w in line.split():
            out.append((x, y, x + 8 * len(w), y + 10, w, 0, 0, 0))
            x += 8 * len(w) + 4
        y += 20.0
    return out


def _reading(prose_lines=_PROSE_LINES, table=_TABLE, engine="qwen", reason=_TABLE_REASON, **kw):
    return PageOutput(
        page_num=1,
        text="\n\n".join(list(prose_lines) + [table]) if table else "\n\n".join(prose_lines),
        status=PageStatus.SUCCESS,
        engine=engine,
        audit_passed=False,
        judge_reason=reason,
        judge_outcome=kw.pop("judge_outcome", JUDGE_OUTCOME_COMPLETED),
        **kw,
    )


def _state(attempts, *, words=True, layer_prose=_PROSE_LINES):
    from socr.core.document import DocumentHandle

    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = " ".join(layer_prose)
    p.needs_ocr_enhancement = True
    p.invisible_text_over_raster = True
    p.native_words = _words(layer_prose) if words else []
    p.attempts = list(attempts)
    p.best_output = attempts[0] if attempts else None
    state.pages[1] = p
    return state


def _shipped(state):
    return _select_page_output_tagged(state, 1)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("table_structure_failed: grid_shape", True),
        ("source_evidence_table: numeric tokens unsupported by page evidence", True),
        (_TABLE_REASON, True),
        ("Table 1 is missing a column; Missing text in the first paragraph", False),
        ("The transcription omits the axis labels of Figure 3", False),
        ("The table is malformed and the figure axis labels are missing", False),
        ("Table 1 is missing a column; the footnote under the table is dropped", False),
        (_P28_REASON, False),
        ("", False),
        ("judge raised: page judge timeout", False),
    ],
)
def test_table_only_rejection_reason(reason: str, expected: bool) -> None:
    assert rejection_is_table_only(reason) is expected


def test_corroboration_difference_same_reading_other_layer() -> None:
    text = "\n\n".join(_PROSE_LINES + [_TABLE])
    same = corroborate_prose(text, _words(_PROSE_LINES))
    other = corroborate_prose(
        text, _words(["Entirely unrelated words about a different matter altogether " * 2] * 4)
    )
    assert same.passed and same.mismatches == 0
    assert not other.passed and other.mismatches > 0
    assert same.prose_tokens >= PROSE_CORROBORATION_MIN_TOKENS


# ---------------------------------------------------------------------------
# #1047 review (Fable): every counterexample that shipped under the bag-of-words guard
# ---------------------------------------------------------------------------

_CE_PROSE = [
    "The estimated effect of the policy change on aggregate output is reported in the table below",
    "Standard errors are clustered by industry and by year in every specification we consider here",
    "We find that the response of consumer prices differs markedly across the two monetary regimes",
    "Under the first regime the pass through of import costs to retail prices was rapid and complete",
    "Under the second regime the same shock produced a smaller and considerably more delayed response",
    "In 1987 the central bank raised its policy rate by 2.5 percentage points over three meetings",
    "Output fell by 4.2 percent in the following year while unemployment rose to 7.8 percent",
    "Robustness checks using alternative samples and estimators give very similar results throughout",
    "We conclude that the regime shift and not the size of the shock explains the different outcomes",
    "The remaining sections of the paper discuss implications for the conduct of monetary policy",
]
_CE_TABLE_ROWS = ["Output 12.5 3.1", "Prices 13.5 2.7", "Employment 9.4 1.8", "Investment 21.0 5.6"]
_CE_TABLE_MD = (
    "| Variable | Mean | SD |\n| --- | --- | --- |\n| Output | 12.5 | 3.1 |\n"
    "| Prices | 13.5 | 2.7 |\n| Employment | 9.4 | 1.8 |\n| Investment | 21.0 | 5.6 |"
)


def _ce_words(prose=_CE_PROSE, extra=()):
    out, y = [], 50.0
    for line in [*prose, *_CE_TABLE_ROWS, *extra]:
        x = 40.0
        for w in line.split():
            out.append((x, y, x + 6 * len(w), y + 10, w, 0, 0, 0))
            x += 6 * len(w) + 4
        y += 14.0
    return out


def _ce(prose, layer=None):
    return corroborate_prose("\n\n".join([*prose, _CE_TABLE_MD]), layer or _ce_words())


def _edit(index, old, new):
    lines = list(_CE_PROSE)
    assert old in lines[index]
    lines[index] = lines[index].replace(old, new)
    return lines


def test_ce_faithful_reading_corroborates() -> None:
    assert _ce(_CE_PROSE).passed


@pytest.mark.parametrize(
    "prose",
    [
        pytest.param(
            [ln for i, ln in enumerate(_CE_PROSE) if i not in (5, 6)], id="drops-numeric-sentences"
        ),
        pytest.param(_CE_PROSE[:7] + _CE_PROSE[8:], id="drops-one-sentence"),
        pytest.param(
            [ln for i, ln in enumerate(_CE_PROSE) if i not in (3, 5, 6, 7)], id="drops-four"
        ),
        pytest.param(_edit(3, "rapid and complete", "slow and incomplete"), id="meaning-flip"),
        pytest.param(_edit(6, "fell", "rose"), id="fell-rose"),
        pytest.param(_edit(8, "and not the size", "and the size"), id="negation-deleted"),
        pytest.param(_edit(6, "fell by 4.2", "fell by -4.2"), id="sign-inserted"),
        pytest.param(_edit(6, "fell by 4.2", "fell by −4.2"), id="unicode-minus-inserted"),
        pytest.param(_edit(6, "4.2 percent", "4.8 percent"), id="decimal-4.2-4.8"),
        pytest.param(_edit(5, "2.5 percentage", "2.7 percentage"), id="decimal-2.5-2.7"),
        pytest.param(_edit(6, "7.8 percent", "7.1 percent"), id="decimal-7.8-7.1"),
        pytest.param(list(reversed(_CE_PROSE)), id="paragraphs-reversed"),
        pytest.param([*_CE_PROSE, "Investment 12.0 5.6"], id="table-row-leaked-misread"),
        pytest.param([*_CE_PROSE, "Investment 21.0 5.6"], id="table-row-leaked-true"),
    ],
)
def test_ce_every_fable_counterexample_floors(prose) -> None:
    c = _ce(prose)
    assert not c.passed and c.mismatches > 0


def test_ce_year_swap_to_a_year_printed_elsewhere_floors() -> None:
    layer = _ce_words(extra=["Friedman 1978 Journal of Political Economy"])
    assert not _ce(_edit(5, "In 1987", "In 1978"), layer).passed


def test_ce_number_words_changed_floors() -> None:
    base = [
        ln.replace(
            "2.5 percentage points over three meetings", "two and a half points over three meetings"
        )
        for ln in _CE_PROSE
    ]
    layer = _ce_words(base)
    flipped = list(base)
    flipped[5] = flipped[5].replace("two and a half", "four and a half").replace("three", "two")
    assert _ce(base, layer).passed
    assert not _ce(flipped, layer).passed


def test_ce_two_column_row_wise_read_floors() -> None:
    col_a, col_b = _CE_PROSE[:5], _CE_PROSE[5:]
    mixed = []
    for a, b in zip(col_a, col_b, strict=True):
        wa, wb = a.split(), b.split()
        mixed.append(" ".join(wa[: len(wa) // 2] + wb[: len(wb) // 2]))
        mixed.append(" ".join(wa[len(wa) // 2 :] + wb[len(wb) // 2 :]))
    assert not _ce(mixed).passed


def test_ce_other_page_sharing_a_running_header_floors() -> None:
    other = [_CE_PROSE[0]] + [
        "Monetary aggregates grew slowly during the sample period while velocity declined"
    ] * 9
    assert not _ce(other).passed


def test_ce_layer_missing_a_line_the_reading_has_floors() -> None:
    layer = _ce_words([ln for i, ln in enumerate(_CE_PROSE) if i != 8])
    assert not _ce(_CE_PROSE, layer).passed


def test_noise_class_is_only_case_punctuation_and_alphabetic_joins() -> None:
    noisy = list(_CE_PROSE)
    noisy[0] = (
        "THE Estimated effect, of the policy change on aggregate output is reported in the table below."
    )
    noisy[1] = noisy[1].replace("specification", "specifi-cation")
    c = _ce(noisy)
    assert c.passed and c.noise_edits >= 1
    # widening the class to "any word" is exactly what the counterexamples above forbid
    assert not _ce(_edit(0, "estimated", "estimate")).passed


# ---------------------------------------------------------------------------
# Selection: the difference between corroborated and uncorroborated prose
# ---------------------------------------------------------------------------


def test_corroborated_prose_ships_with_withheld_marker() -> None:
    out, prov = _shipped(_state([_reading()]))
    assert prov is SelectionProvenance.TABLE_WITHHELD_PROSE_CORROBORATED
    assert out.failure_mode is FailureMode.TABLE_WITHHELD_PROSE_CORROBORATED
    assert out.status is PageStatus.WARNING
    assert out.audit_passed is False, "audit_passed is the selection flag; demote via status"
    assert "estimated effect of the policy change" in out.text
    assert "12.5" not in out.text and "| Variable" not in out.text
    assert out.text.count("failed: unverifiable table") == 1
    assert not manifest.is_page_failed_marker(out.text)
    disp = manifest.provenance_to_disposition(prov)
    assert disp.ending is PageEnding.MODEL_OUTPUT
    assert disp.primary_reason is PagePrimaryReason.TABLE_WITHHELD_PROSE_CORROBORATED
    assert any("ordered_match tokens=" in n for n in out.audit_notes)


def test_withheld_table_counts_in_the_993_metric() -> None:
    out, _ = _shipped(_state([_reading()]))
    counts = count_page_tables(out.text, out.status.value, out.failure_mode.value)
    assert counts.withheld == 1
    assert counts.shipped_text == 0 and counts.verified_text == 0


def test_difference_uncorroborated_prose_keeps_the_floor() -> None:
    other = ["Quarterly dividends were ratified by the committee without dissent"] * 4
    good, good_prov = _shipped(_state([_reading()]))
    bad, bad_prov = _shipped(_state([_reading(prose_lines=other)]))
    assert good_prov is SelectionProvenance.TABLE_WITHHELD_PROSE_CORROBORATED
    assert bad_prov is SelectionProvenance.INVISIBLE_SCAN_UNREAD
    assert bad.failure_mode is FailureMode.INVISIBLE_SCAN_UNREAD
    assert manifest.is_page_failed_marker(bad.text) and "dividends" not in bad.text


def test_difference_non_table_rejection_keeps_the_floor() -> None:
    mixed = _TABLE_REASON + "; Missing text in the first paragraph"
    ok, ok_prov = _shipped(_state([_reading()]))
    bad, bad_prov = _shipped(_state([_reading(reason=mixed)]))
    assert ok_prov is SelectionProvenance.TABLE_WITHHELD_PROSE_CORROBORATED
    assert bad_prov is SelectionProvenance.INVISIBLE_SCAN_UNREAD
    assert manifest.is_page_failed_marker(bad.text)


@pytest.mark.parametrize(
    "reason",
    ["The table is malformed and the figure axis labels are missing", _P28_REASON],
)
def test_difference_mixed_clause_keeps_the_floor(reason: str) -> None:
    ok, ok_prov = _shipped(_state([_reading()]))
    bad, bad_prov = _shipped(_state([_reading(reason=reason)]))
    assert ok_prov is SelectionProvenance.TABLE_WITHHELD_PROSE_CORROBORATED
    assert bad_prov is SelectionProvenance.INVISIBLE_SCAN_UNREAD
    assert manifest.is_page_failed_marker(bad.text)


def test_model_authored_image_refs_do_not_ship_but_the_floor_image_does() -> None:
    invented = list(_PROSE_LINES)
    invented[1] = invented[1] + "\n\n![fig](figures/invented_by_model.png)"
    state = _state([_reading(prose_lines=invented)])
    state.pages[1].invisible_scan_png_ref = "![scan](figures/invisible_scan_page_p1.png)"
    out, prov = _shipped(state)
    assert prov is SelectionProvenance.TABLE_WITHHELD_PROSE_CORROBORATED
    assert "invented_by_model" not in out.text
    assert out.text.count("invisible_scan_page_p1.png") == 1


def test_non_ascii_numeral_absent_from_the_layer_is_vetoed() -> None:
    """An ASCII-only tokeniser dropped Arabic-Indic digits, so the veto never saw them."""
    lines = [*_PROSE_LINES[:3], "Robustness checks give very similar results in ٣١ cases"]
    c = corroborate_prose("\n\n".join([*lines, _TABLE]), _words(_PROSE_LINES))
    assert c.mismatches >= 1 and not c.passed
    _, prov = _shipped(_state([_reading(prose_lines=lines)]))
    assert prov is SelectionProvenance.INVISIBLE_SCAN_UNREAD


def test_banner_claims_corroboration_not_word_for_word() -> None:
    out, _ = _shipped(_state([_reading()]))
    first = out.text.splitlines()[0]
    assert "word-for-word" not in out.text
    assert "corroborated by this page's text layer" in first
    assert "ordered match" in first and "tokens" in first and "similarity" not in first


def test_one_non_table_rejected_reading_blocks_even_a_corroborated_one() -> None:
    other = _reading(engine="gemini", reason="The figure axis labels are missing")
    _, prov = _shipped(_state([_reading(), other]))
    assert prov is SelectionProvenance.INVISIBLE_SCAN_UNREAD


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: setattr(a, "judge_outcome", JUDGE_OUTCOME_TIMEOUT),
        lambda a: setattr(a, "judge_outcome", ""),
        lambda a: setattr(a, "judge_reason", ""),
    ],
    ids=["timeout", "never-judged", "no-reason"],
)
def test_missing_verdict_is_not_a_table_rejection(mutate) -> None:
    reading = _reading()
    mutate(reading)
    _, prov = _shipped(_state([reading]))
    assert prov is SelectionProvenance.INVISIBLE_SCAN_UNREAD


def test_no_text_layer_keeps_the_floor() -> None:
    _, prov = _shipped(_state([_reading()], words=False))
    assert prov is SelectionProvenance.INVISIBLE_SCAN_UNREAD


def test_reading_without_a_table_keeps_the_floor() -> None:
    _, prov = _shipped(_state([_reading(table="")]))
    assert prov is SelectionProvenance.INVISIBLE_SCAN_UNREAD


def test_numeral_absent_from_the_layer_keeps_the_floor() -> None:
    lines = [*_PROSE_LINES[:3], "Robustness checks give very similar results in 1987 as well"]
    _, prov = _shipped(_state([_reading(prose_lines=lines)]))
    assert prov is SelectionProvenance.INVISIBLE_SCAN_UNREAD


def test_accepted_reading_is_untouched() -> None:
    accepted = _reading(reason="")
    accepted.audit_passed = True
    out, prov = _shipped(_state([accepted]))
    assert prov is SelectionProvenance.PASSING_BEST_OUTPUT
    assert out.failure_mode is not FailureMode.TABLE_WITHHELD_PROSE_CORROBORATED


def test_native_only_attempts_never_corroborate_themselves() -> None:
    _, prov = _shipped(_state([_reading(engine="native")]))
    assert prov is not SelectionProvenance.TABLE_WITHHELD_PROSE_CORROBORATED


# ---------------------------------------------------------------------------
# End to end through process()
# ---------------------------------------------------------------------------


def _scan_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200), False)
    pix.set_rect(pix.irect, (235, 235, 235))
    page.insert_image(page.rect, pixmap=pix)
    y = 72
    for line in [*_PROSE_LINES, "Output 12.5", "Prices 13.5"]:
        page.insert_text((72, y), line, fontname="helv", fontsize=10, render_mode=3)
        y += 14
    doc.save(path)
    doc.close()
    return path


class _Engine:
    name = "qwen"

    def __init__(self, prose) -> None:
        self.prose = prose

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        text = "\n\n".join(list(self.prose) + [_TABLE])
        return [
            PageOutput(page_num=n, text=text, status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


class _Judge:
    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=False, reason=_TABLE_REASON)


def _run(tmp_path, monkeypatch, tag, prose):
    pdf = _scan_pdf(tmp_path / f"{tag}.pdf")
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _Engine(prose))
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=True,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._build_page_judge = lambda state: _Judge()
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        result = pipe.process(pdf, output_dir=tmp_path / f"out-{tag}")
    out = tmp_path / f"out-{tag}"
    side = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    text = next(iter(out.rglob("pages/00001.md"))).read_text()
    audit = json.loads(next(iter(out.rglob("audit_log.json"))).read_text())
    return result, side, text, audit["events"]


def test_e2e_corroborated_vs_uncorroborated_same_scan(tmp_path, monkeypatch) -> None:
    other = ["Quarterly dividends were ratified by the committee without dissent"] * 4
    good, good_side, good_text, good_events = _run(tmp_path, monkeypatch, "good", _PROSE_LINES)
    bad, bad_side, bad_text, bad_events = _run(tmp_path, monkeypatch, "bad", other)
    good_kinds = {e["kind"] for e in good_events}
    bad_kinds = {e["kind"] for e in bad_events}
    (ev,) = [e for e in good_events if e["kind"] == "table_withheld_prose_corroborated"]
    assert isinstance(ev["data"]["matched_tokens"], int) and ev["data"]["matched_tokens"] >= 17

    assert good_side["failure_mode"] == FailureMode.TABLE_WITHHELD_PROSE_CORROBORATED.value
    assert good_side["status"] == "warning" and good_side["audit_passed"] is False
    assert "estimated effect of the policy change" in good_text
    assert "failed: unverifiable table" in good_text and "12.5" not in good_text
    assert "table_withheld_prose_corroborated" in good_kinds
    assert good.status is not DocumentStatus.SUCCESS
    assert "table_withheld_prose_corroborated" in (good.error or "")
    assert good_side["disposition"]["primary_reason"] == "table_withheld_prose_corroborated"

    # The uncorroborated twin floors exactly as before and carries none of the new surface.
    assert bad_side["failure_mode"] != FailureMode.TABLE_WITHHELD_PROSE_CORROBORATED.value
    assert "dividends" not in bad_text
    assert "table_withheld_prose_corroborated" not in bad_kinds
