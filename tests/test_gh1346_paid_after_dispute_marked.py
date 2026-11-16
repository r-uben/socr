"""#1346 (owner ruling 2026-11-16): a paid reading taken after a witness dispute is ALWAYS marked.

"A page whose shipped reading came from a paid reader that was called after the pixel witness
disputed an earlier reader's table always carries a warning. It never ships as unmarked SUCCESS,
even when the paid reading passes every check."

Hermetic per CLAUDE.md: engines, ladder and page judge are pinned, and every claim is a DIFFERENCE
between two arms run in one process (only whether the earlier rung's rejection was a witness
dispute differs), never an absolute outcome measured on this machine.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_gh1185_text_layer_numbers_e2e import _doc_dir, _page, _prose, _rotated_pdf

import socr
from socr.benchmark.scoreboard import _is_marked
from socr.core import verdict as verdict_module
from socr.core.marks import Check, CheckResult
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.verdict import PAID_AFTER_DISPUTE_REASON, PERMANENTLY_UNRESOLVABLE_MODES

#: What the witness says when it refutes the earlier reader's table, and what a rejection that
#: is NOT a dispute says (no witness existed: ``blocks_paid_escalation`` is false for it).
WITNESS_DISPUTE = "source_evidence_table: numeric tokens unsupported by page evidence: ['4.0']"
NO_DISPUTE = "source_evidence_table: [no_witness_backend] no classical OCR witness read this page"

PAID_BODY = (
    "Korea grew 4.0 2.6 5.7 4.0. China grew 6.9 7.0 6.6 6.4 7.1. Mexico grew 2.3 1.1 2.5 2.4."
)


#: An engine label that is not a model reading, so the dispute clause is the page's only clause.
_NOT_A_MODEL = "native"


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


class _Reader:
    """A reader for both rungs: the free one and the paid one read the same words."""

    def __init__(self, name: str, calls: list | None = None) -> None:
        self.name = name
        self.calls = calls

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        if self.calls is not None:
            self.calls.append((self.name, tuple(page_nums)))
        return [
            PageOutput(
                page_num=n,
                text=f"{_prose(n)}. {PAID_BODY}",
                status=PageStatus.SUCCESS,
                engine=self.name,
            )
            for n in page_nums
        ]


class _LadderJudge:
    """Rejects the free rung with *free_reason*, accepts the paid one."""

    def __init__(self, free_reason: str) -> None:
        self._free_reason = free_reason

    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        if provider.engine.value == "gemini":
            return AcceptDecision(accept=True, reason="faithful")
        return AcceptDecision(accept=False, reason=self._free_reason)


def _run(
    out_dir: Path,
    pdf: Path,
    monkeypatch,
    *,
    free_reason: str,
    calls: list | None = None,
    reprocess: bool = False,
    lane_forced: bool = False,
):
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch
    from socr.pipeline.orchestrator import UnifiedPipeline

    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _Reader(engine_type.value, calls))
        if lane_forced:
            # The table-escalation lane's two decisions, forced: whether to re-read the page
            # and whether the re-read wins. What they decide is not under test, what the lane
            # does to the flag of the reading it promotes is.
            from socr.tables import escalation_decision as ed

            m.setattr(
                orch.UnifiedPipeline, "_table_page_needs_escalation", lambda self, *a, **k: True
            )
            m.setattr(
                ed,
                "decide_escalation",
                lambda page, inc, cand: ed.EscalationDecision(
                    accepted=True, gate="exactness", reason="forced"
                ),
            )
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN, EngineType.GEMINI],
                native_first=True,
                write_manifest=True,
                reprocess=reprocess,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=True,
                describe_figure_crops=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL, PROFILE_GEMINI]
        pipe._build_page_judge = lambda state: _LadderJudge(free_reason)
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        return pipe.process(pdf, output_dir=out_dir)


def _clauses(sidecar: dict, check: str) -> list[dict]:
    return [c for c in sidecar["verdict"]["clauses"] if c["check"] == check]


@pytest.fixture
def arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    disputed, control = tmp_path / "disputed", tmp_path / "control"
    _run(disputed, pdf, monkeypatch, free_reason=WITNESS_DISPUTE)
    _run(control, pdf, monkeypatch, free_reason=NO_DISPUTE)
    return pdf, disputed, control


def test_the_paid_reading_after_a_dispute_ships_its_text_marked_and_the_control_does_not(
    arms,
) -> None:
    _pdf, disputed, control = arms
    for n in (1, 2):
        d_side, d_body = _page(disputed, n)
        c_side, c_body = _page(control, n)
        # both arms shipped the paid reading
        assert d_side["winning_output"]["engine"] == c_side["winning_output"]["engine"] == "gemini"
        # the dispute arm: WARNING, text kept, the clause, the banner
        assert d_side["status"] == "warning"
        assert d_side["winning_output"]["failure_mode"] == (
            FailureMode.PAID_READING_AFTER_DISPUTE.value
        )
        assert d_side["winning_output"]["paid_after_witness_dispute"] is True
        [clause] = _clauses(d_side, Check.READING_DISPUTE.value)
        assert (clause["result"], clause["scope"], clause["reason"]) == (
            CheckResult.UNKNOWN.value,
            "page",
            PAID_AFTER_DISPUTE_REASON,
        )
        assert _is_marked(d_body)
        assert f"> [page {n} unchecked: reading_dispute=UNKNOWN; scope=page; reason=" in d_body
        assert PAID_BODY in d_body  # the text is kept, not withheld
        # the control arm: no clause, no flag, and it is the same page minus the mark
        assert _clauses(c_side, Check.READING_DISPUTE.value) == []
        assert "paid_after_witness_dispute" not in c_side["winning_output"]
        assert "reading_dispute" not in c_body
        assert PAID_BODY in c_body
        assert d_side["status"] != c_side["status"]


def test_the_document_rollup_reflects_the_marked_page(arms) -> None:
    _pdf, disputed, control = arms

    def doc_status(out: Path) -> str:
        return json.loads((_doc_dir(out) / "metadata.json").read_text())["status"]

    assert doc_status(disputed) != "completed"
    assert doc_status(control) != doc_status(disputed)


def _pages_bytes(out: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in (_doc_dir(out) / "pages").glob("*")}


def test_a_resumed_run_restores_the_marked_page_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The PER-PAGE resume gate restores the marked page: no reader is called, same bytes.

    ``reprocess=True`` defeats the document-level skip, so the gate under test is the one that
    reads each page's sidecar. The first run's calls are the control: they prove the readers
    would have been called had the gate not restored the page.
    """
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    out = tmp_path / "out"
    first_calls: list = []
    _run(out, pdf, monkeypatch, free_reason=WITNESS_DISPUTE, calls=first_calls)
    before = _pages_bytes(out)
    second_calls: list = []
    _run(out, pdf, monkeypatch, free_reason=WITNESS_DISPUTE, calls=second_calls, reprocess=True)
    assert [c for c in first_calls if c[0] == "gemini"], "the first run paid the reader"
    assert second_calls == []
    assert _pages_bytes(out) == before


def test_the_resume_gate_reprocesses_an_error_page_that_carries_the_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mode is restored on the strength of a WARNING demotion only.

    A sidecar carrying the mode with any other status was not demoted by this run, so doubt
    reprocesses it. Pinned as a difference: the same sidecar with its status left WARNING is
    restored (no calls), with it edited to ERROR the page is read again.
    """
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    out = tmp_path / "out"
    _run(out, pdf, monkeypatch, free_reason=WITNESS_DISPUTE)
    side_path = _doc_dir(out) / "pages" / "00001.json"
    side = json.loads(side_path.read_text())
    assert side["winning_output"]["failure_mode"] == FailureMode.PAID_READING_AFTER_DISPUTE.value
    kept_calls: list = []
    _run(out, pdf, monkeypatch, free_reason=WITNESS_DISPUTE, calls=kept_calls, reprocess=True)
    side["winning_output"]["status"] = PageStatus.ERROR.value
    side_path.write_text(json.dumps(side))
    edited_calls: list = []
    _run(out, pdf, monkeypatch, free_reason=WITNESS_DISPUTE, calls=edited_calls, reprocess=True)
    assert kept_calls == []
    assert 1 in [n for _name, pages in edited_calls for n in pages]


def test_the_escalation_lane_marks_the_paid_reading_it_promotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The table-escalation lane re-reads with a paid reader AFTER ``route_page`` and promotes
    that fresh output, which ``route_page`` never stamped. After a witness dispute the promoted
    reading is marked too; the control (no dispute) is not."""
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    arms = {}
    for name, reason in (("disputed", WITNESS_DISPUTE), ("control", NO_DISPUTE)):
        out = tmp_path / name
        _run(out, pdf, monkeypatch, free_reason=reason, lane_forced=True)
        arms[name] = out
    d_side, d_body = _page(arms["disputed"], 1)
    c_side, c_body = _page(arms["control"], 1)
    # the lane promoted the paid reading in both arms
    assert d_side["winning_output"]["escalated_from"] == c_side["winning_output"]["escalated_from"]
    assert d_side["winning_output"]["escalated_from"]
    assert d_side["winning_output"]["engine"] == c_side["winning_output"]["engine"] == "gemini"
    assert d_side["winning_output"]["paid_after_witness_dispute"] is True
    assert "paid_after_witness_dispute" not in c_side["winning_output"]
    assert len(_clauses(d_side, Check.READING_DISPUTE.value)) == 1
    assert _clauses(c_side, Check.READING_DISPUTE.value) == []
    assert d_side["status"] != c_side["status"]
    assert _is_marked(d_body) and "reading_dispute" not in c_body
    assert PAID_BODY in d_body


def test_the_mode_is_one_a_resume_restores() -> None:
    assert FailureMode.PAID_READING_AFTER_DISPUTE in PERMANENTLY_UNRESOLVABLE_MODES


# --- the seam: sidecar round trip and re-finalize ---------------------------------------


def _output(**kw) -> PageOutput:
    base = dict(
        page_num=1,
        text="Korea grew 4.0 2.6.",
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=True,
        judge_accepted=True,
        paid_after_witness_dispute=True,
    )
    base.update(kw)
    return PageOutput(**base)


def _p():
    return SimpleNamespace(
        own_layer_text="",
        own_layer_drop_reference="",
        own_layer_word_reference="",
        own_layer_figure_words="",
        is_born_digital=False,
        prior_ocr_layer_suspect=False,
        has_unrecovered_symbol_glyphs=False,
        table_ladder_disposition=None,
        control_bytes_replaced=None,
        row_binding={},
        hidden_footnote_words="",
    )


def test_the_flag_survives_the_sidecar_round_trip() -> None:
    out = _output()
    assert PageOutput.from_dict(out.to_dict()).paid_after_witness_dispute is True
    # omit-when-unset: a page without it keeps the bytes (and the fingerprint) it always had
    assert "paid_after_witness_dispute" not in _output(paid_after_witness_dispute=False).to_dict()
    assert (
        PageOutput.from_dict(
            _output(paid_after_witness_dispute=False).to_dict()
        ).paid_after_witness_dispute
        is False
    )


def test_refinalizing_a_restored_page_writes_the_same_banner_and_is_idempotent() -> None:
    from socr.core.manifest import apply_verdict_marks
    from socr.core.verdict import PageVerdict

    first, verdict = apply_verdict_marks(_output(engine=_NOT_A_MODEL), _p(), 1)
    assert first.status is PageStatus.WARNING
    assert first.failure_mode is FailureMode.PAID_READING_AFTER_DISPUTE
    assert first.text.startswith("> [page 1 unchecked: reading_dispute=UNKNOWN; scope=page;")
    # restored from the sidecar: the typed field and the published verdict
    restored = PageOutput.from_dict(first.to_dict())
    published = PageVerdict.from_dict(verdict.to_dict())
    second, verdict2 = apply_verdict_marks(restored, _p(), 1, published=published)
    assert second.text == first.text
    assert verdict2.clauses == verdict.clauses
    # and from the flag alone, with no published verdict (the verdict is derived again)
    third, _ = apply_verdict_marks(restored, _p(), 1)
    assert third.text == first.text


# --- the clause adds to the page's other marks and never replaces one ----------------------


def _check_set(verdict) -> set[Check]:
    return {c.check for c in verdict.clauses}


def test_the_clause_leaves_the_numbers_mode_and_clause_in_place() -> None:
    """A true scan has no layer to check the reading's numbers against. Flag on versus off:
    the same failure mode, and the clause set grows by exactly the dispute clause."""
    from socr.core.manifest import apply_verdict_marks

    on, on_verdict = apply_verdict_marks(_output(), _p(), 1)
    off, off_verdict = apply_verdict_marks(_output(paid_after_witness_dispute=False), _p(), 1)
    assert Check.TEXT_LAYER_NUMBERS in _check_set(off_verdict)
    assert on.status is off.status is PageStatus.WARNING
    assert on.failure_mode is off.failure_mode
    assert _check_set(on_verdict) == _check_set(off_verdict) | {Check.READING_DISPUTE}
    # re-finalizing the demoted page keeps the same mode (the mode chooser is stable)
    again, _ = apply_verdict_marks(PageOutput.from_dict(on.to_dict()), _p(), 1)
    assert again.failure_mode is on.failure_mode


def test_a_stale_dispute_only_mode_follows_the_full_clause_set() -> None:
    """A page already demoted with the dispute mode, then given another clause, takes the mode a
    fresh run gives that clause set (the ``_VERDICT_DERIVED_MODES`` row)."""
    from socr.core.manifest import apply_verdict_marks

    fresh, _ = apply_verdict_marks(_output(), _p(), 1)
    stale = _output(status=PageStatus.WARNING, failure_mode=FailureMode.PAID_READING_AFTER_DISPUTE)
    out, _ = apply_verdict_marks(stale, _p(), 1)
    assert out.failure_mode is fresh.failure_mode
    assert fresh.failure_mode is not FailureMode.PAID_READING_AFTER_DISPUTE


def test_a_recorded_mode_without_its_flag_still_names_its_clause() -> None:
    """The mode alone, flag absent (the ``_MODE_CLAUSES`` row): the clause is derived from the
    mode, and the fail-closed fallback does not add an ``unknown`` clause beside it."""
    from socr.core.manifest import apply_verdict_marks

    bare = _output(
        engine=_NOT_A_MODEL,
        status=PageStatus.WARNING,
        failure_mode=FailureMode.PAID_READING_AFTER_DISPUTE,
        paid_after_witness_dispute=False,
    )
    _out, verdict = apply_verdict_marks(bare, _p(), 1)
    assert _check_set(verdict) == {Check.READING_DISPUTE}


def test_the_dispute_clause_does_not_hide_the_judge_only_clause_offline() -> None:
    """``with_judge_only_clause`` (the re-finalize tools) keeps #1251's clause beside the dispute
    clause instead of dropping it as 'another clause already marks the page'."""
    from socr.core.manifest import apply_verdict_marks
    from socr.core.marks import SCOPE_PAGE
    from socr.core.verdict import JUDGE_ONLY_REASON, VerdictClause, with_judge_only_clause

    out, verdict = apply_verdict_marks(_output(engine=_NOT_A_MODEL), _p(), 1)
    dispute = next(c for c in verdict.clauses if c.check is Check.READING_DISPUTE)
    judge_only = VerdictClause(Check.PAGE_JUDGE, CheckResult.UNKNOWN, SCOPE_PAGE, JUDGE_ONLY_REASON)
    both = replace(verdict, clauses=(judge_only, dispute))
    kept = with_judge_only_clause(both, out, None)
    assert set(kept.clauses) == {judge_only, dispute}


def test_an_error_page_stays_error_and_gains_no_text() -> None:
    from socr.core.manifest import apply_verdict_marks

    marker = "[page 1 failed: unverifiable table -- see image]\n\n"
    err = _output(
        status=PageStatus.ERROR,
        text=marker,
        failure_mode=FailureMode.WITNESS_DISAGREES,
        audit_passed=False,
    )
    plain = replace(err, paid_after_witness_dispute=False)
    out, verdict = apply_verdict_marks(err, _p(), 1)
    ref, ref_verdict = apply_verdict_marks(plain, _p(), 1)
    assert (out.status, out.failure_mode, out.text) == (ref.status, ref.failure_mode, ref.text)
    assert out.status is PageStatus.ERROR and out.text == marker
    assert verdict.clauses == ref_verdict.clauses


def test_another_warning_keeps_its_ending_and_gains_the_clause() -> None:
    from socr.core.manifest import apply_verdict_marks

    warned = _output(status=PageStatus.WARNING, failure_mode=FailureMode.TABLE_UNVERIFIED)
    plain = replace(warned, paid_after_witness_dispute=False)
    out, verdict = apply_verdict_marks(warned, _p(), 1)
    ref, ref_verdict = apply_verdict_marks(plain, _p(), 1)
    assert (out.status, out.failure_mode) == (ref.status, ref.failure_mode)
    assert [c for c in verdict.clauses if c.check is not Check.READING_DISPUTE] == list(
        ref_verdict.clauses
    )
    assert [c.check for c in verdict.clauses if c.check is Check.READING_DISPUTE] == [
        Check.READING_DISPUTE
    ]


def test_the_scoreboard_counts_the_banner_as_a_mark() -> None:
    from socr.core.manifest import apply_verdict_marks

    marked, _ = apply_verdict_marks(_output(), _p(), 1)
    plain, _ = apply_verdict_marks(_output(paid_after_witness_dispute=False), _p(), 1)
    assert _is_marked(marked.text)
    # the same page without the flag carries no reading_dispute line
    assert "reading_dispute" in marked.text and "reading_dispute" not in plain.text
    # and the banner alone is a mark: strip every other line and it still counts
    only = next(line for line in marked.text.split("\n") if "reading_dispute" in line)
    assert _is_marked(only + "\n\nbody 1.0")


def test_refinalize_tool_leaves_a_marked_page_alone(arms, tmp_path: Path) -> None:
    from socr.pipeline.refinalize_own_text import refinalize_document

    pdf, disputed, _control = arms
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    shutil.copy(pdf, pdf_dir / pdf.name)
    snapshot = {p: p.read_bytes() for p in disputed.rglob("*") if p.is_file()}
    assert refinalize_document(_doc_dir(disputed), pdf_dir) == []
    assert {p: p.read_bytes() for p in disputed.rglob("*") if p.is_file()} == snapshot
    assert verdict_module  # the module under test is the one imported at the top
