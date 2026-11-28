"""Hermetic regression tests for measured paid-call leaks and cache scoping.

Pins four fixes derived from the Fed full-run cost audit:
  1. ``_escalate_table_page`` (GH-96 / #1394) must NEVER double-call a paid
     engine after ``route_page`` already ran a paid attempt on that page, and
     must NEVER record phantom ``page_cost_usd`` when Gemini is skipped by the
     quota/402 circuit or a non-run ``MODEL_UNAVAILABLE`` stub.
  2. ``route_page`` / ``process()`` must skip paid rungs when
     ``CircuitBreakerPageJudge`` is open (paired difference: open vs closed).
  3. ``UnifiedPipeline._JUDGE_MODEL_CANDIDATES`` must contain no forbidden
     ``qwen3-vl:8b`` (or any ``:8b`` Qwen) model (``AGENTS.md`` compliance).
  4. ``_socr_source_digest`` ignores non-reading packages (``benchmark/``,
     ``refinalize/``, ``ui/``, ``devtools/``, ``pipeline/refinalize_*.py``)
     while still changing on reading-path edits.

All tests are 100% synthetic and hermetic (no live Ollama, no network, no API).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

import socr
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.judge.judge import JudgeVerdict
from socr.judge.ollama_judge import OllamaVisionJudge
from socr.pipeline import agentic as agentic_mod
from socr.pipeline import orchestrator as orch
from socr.pipeline.agentic import (
    AcceptDecision,
    CircuitBreakerPageJudge,
    route_page,
)
from socr.pipeline.orchestrator import UnifiedPipeline

_TABLE_ROWS = [
    ("Total effect of decisions", ["42.8", "30.5", "2.6"]),
    ("September energy package", ["43.2", "26.8", "3.7"]),
    ("Energy price guarantee", ["24.8", "26.8", "3.7"]),
    ("Energy bill relief scheme", ["18.4", "9.1", "5.5"]),
]

_SHIFTED_MD = (
    "| Category | c1 | c2 | c3 |\n"
    "| --- | --- | --- | --- |\n"
    "| Total effect of decisions | 42.8 | 30.5 | 2.6 |\n"
    "| September energy package |  |  |  |\n"
    "| Energy price guarantee | 43.2 | 26.8 | 3.7 |\n"
    "| Energy bill relief scheme | 24.8 | 26.8 | 3.7 |\n"
)

_PERFECT_MD = (
    "| Category | c1 | c2 | c3 |\n"
    "| --- | --- | --- | --- |\n"
    "| Total effect of decisions | 42.8 | 30.5 | 2.6 |\n"
    "| September energy package | 43.2 | 26.8 | 3.7 |\n"
    "| Energy price guarantee | 24.8 | 26.8 | 3.7 |\n"
    "| Energy bill relief scheme | 18.4 | 9.1 | 5.5 |\n"
)


def test_loaded_socr_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _table_pdf(tmp_path: Path, *, pages: int = 1, name: str = "table.pdf") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    doc = fitz.open()
    for _ in range(pages):
        pg = doc.new_page()
        y = 200.0
        for label, values in _TABLE_ROWS:
            pg.insert_text((60.0, y), label, fontsize=9)
            for x, v in zip((300.0, 360.0, 420.0), values, strict=True):
                pg.insert_text((x, y), v, fontsize=9)
            y += 18.0
        pg.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
        pg.draw_line(fitz.Point(50, y), fitz.Point(470, y))
    doc.save(path)
    doc.close()
    return path


def _prose_pdf(tmp_path: Path, *, pages: int = 2, name: str = "prose.pdf") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    doc = fitz.open()
    for idx in range(pages):
        pg = doc.new_page()
        y = 80.0
        for _ in range(12):
            pg.insert_text(
                (60.0, y),
                f"Estimated policy coefficient 0.0{idx + 1} significant in sample",
                fontsize=9,
            )
            y += 16.0
    doc.save(path)
    doc.close()
    return path


def _sidecars(out_dir: Path) -> dict[int, dict]:
    return {int(p.stem): json.loads(p.read_text()) for p in out_dir.rglob("pages/*.json")}


# ---------------------------------------------------------------------------
# 1. No double paid call in _escalate_table_page + no phantom cost on 402 circuit
# ---------------------------------------------------------------------------


def _run_process_for_escalation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    qwen_accepted: bool,
) -> tuple[list[tuple[str, int]], dict[int, dict]]:
    """Drive real ``process()`` on a born-digital table page with [QWEN, GEMINI].

    * ``qwen_accepted=False`` -> ``route_page`` rejects Qwen and runs Gemini once;
      ``_escalate_table_page`` must NOT call Gemini a second time.
    * ``qwen_accepted=True`` -> ``route_page`` accepts Qwen (no paid call yet);
      ``_escalate_table_page`` runs Gemini once on the disagreeing table.
    """
    pdf = _table_pdf(tmp_path, pages=1)
    out_dir = tmp_path / "out"
    calls: list[tuple[str, int]] = []

    cfg = PipelineConfig(
        agentic=True,
        quiet=True,
        primary_engine=EngineType.QWEN,
        local_engine=EngineType.QWEN,
        enabled_engines=[EngineType.QWEN, EngineType.GEMINI],
        escalate_ambiguous_tables=True,
        dual_pass_tables=False,
        table_judge_ladder=False,
        write_manifest=False,
    )
    pipe = UnifiedPipeline(cfg)

    detect = pipe.bd_detector.detect

    def _force_ocr(path: Path):
        assessment = detect(path)
        for p in assessment.pages:
            p.needs_ocr_enhancement = True
        return assessment

    pipe.bd_detector.detect = _force_ocr
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL, PROFILE_GEMINI]
    pipe._resolve_judge_model = lambda *a, **k: ""
    pipe._resolve_crop_vlm_model = lambda: None

    def _fake_run_engine(state, page_nums, native_pages, engine_type, phase, profile=None, **kw):
        outputs = []
        for pn in page_nums:
            calls.append((engine_type.value, pn))
            outputs.append(
                PageOutput(
                    page_num=pn,
                    # Both engines return the shifted table so _score_table_signal is True
                    # and _escalate_table_page is reached after route_page!
                    text=_SHIFTED_MD,
                    status=PageStatus.SUCCESS,
                    engine=engine_type.value,
                )
            )
        return outputs

    pipe._run_engine_on_pages = _fake_run_engine

    class _StubJudge:
        def assess(self, output: PageOutput, provider) -> AcceptDecision:
            if provider.engine == EngineType.QWEN:
                return AcceptDecision(
                    accept=qwen_accepted,
                    reason="faithful" if qwen_accepted else "shifted rows",
                )
            return AcceptDecision(accept=True, reason="faithful")

    pipe._build_page_judge = lambda state: _StubJudge()

    pipe.process(pdf, output_dir=out_dir)
    return calls, _sidecars(out_dir)


def test_process_never_double_calls_gemini_in_table_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paired difference through real ``process()``:

    * When ``route_page`` ALREADY ran Gemini (``qwen_accepted=False``),
      ``_escalate_table_page`` skips the redundant paid re-call -> exactly 1
      Gemini call and 1x ``page_cost_usd``.
    * Control (``qwen_accepted=True``): ``route_page`` ran only free Qwen ->
      ``_escalate_table_page`` DOES fire Gemini once -> 1 Gemini call.
    """
    after_gemini_calls, after_gemini_sc = _run_process_for_escalation(
        tmp_path / "after_gemini", monkeypatch, qwen_accepted=False
    )
    after_qwen_calls, after_qwen_sc = _run_process_for_escalation(
        tmp_path / "after_qwen", monkeypatch, qwen_accepted=True
    )

    gemini_after_gemini = [c for c in after_gemini_calls if c[0] == "gemini"]
    gemini_after_qwen = [c for c in after_qwen_calls if c[0] == "gemini"]

    assert len(gemini_after_gemini) == 1, (
        f"expected 1 Gemini call (no GH-96 double call), got {after_gemini_calls}"
    )
    assert len(gemini_after_qwen) == 1, (
        f"control: GH-96 escalation after free Qwen must still fire once, got {after_qwen_calls}"
    )
    assert after_gemini_sc[1]["page_cost_usd"] == pytest.approx(PROFILE_GEMINI.cost_per_page_usd)


def test_escalate_table_page_records_zero_phantom_cost_when_quota_circuit_open(
    tmp_path: Path,
) -> None:
    """Neither an open ``_gemini_quota_circuit_open`` nor a non-run
    ``MODEL_UNAVAILABLE`` stub may record phantom ``page_cost_usd``, whereas a
    real executed ``CLI_ERROR`` attempt still bills (paired difference)."""
    pdf = _table_pdf(tmp_path, pages=1)

    # Arm A: pipe._gemini_quota_circuit_open is already True -> no provider call, $0 cost.
    state_a = DocumentState(DocumentHandle(pdf))
    state_a.paid_disabled = {"reason": "quota_exhausted", "detail": "402", "pages_skipped": 0}
    ps_a = state_a.pages[1]
    bo_a = PageOutput(page_num=1, text=_SHIFTED_MD, status=PageStatus.SUCCESS, engine="qwen")
    ps_a.attempts.append(bo_a)
    ps_a.best_output = bo_a

    pipe_a = UnifiedPipeline(PipelineConfig(quiet=True, escalate_ambiguous_tables=True))
    pipe_a._gemini_quota_circuit_open = True
    called_a: list[int] = []

    def _must_not_run(profile, page_num):
        called_a.append(page_num)
        return PageOutput(
            page_num=page_num, text=_PERFECT_MD, status=PageStatus.SUCCESS, engine="gemini"
        )

    degraded, out = pipe_a._escalate_table_page(
        state_a, 1, ps_a, bo_a, PROFILE_GEMINI, _must_not_run, pdf, needs_escalation=True
    )
    assert degraded is False
    assert out is bo_a
    assert called_a == []
    assert state_a.total_cost == 0.0
    assert ps_a.page_cost_usd == 0.0
    assert state_a.paid_disabled["pages_skipped"] == 1

    # Arm B: provider returns circuit-blocked PageOutput (MODEL_UNAVAILABLE + empty text) -> $0 cost.
    state_b = DocumentState(DocumentHandle(pdf))
    ps_b = state_b.pages[1]
    bo_b = PageOutput(page_num=1, text=_SHIFTED_MD, status=PageStatus.SUCCESS, engine="qwen")
    ps_b.attempts.append(bo_b)
    ps_b.best_output = bo_b
    pipe_b = UnifiedPipeline(PipelineConfig(quiet=True, escalate_ambiguous_tables=True))

    def _returns_circuit_blocked(profile, page_num):
        return pipe_b._gemini_circuit_blocked_output(page_num, profile.engine.value)

    pipe_b._escalate_table_page(
        state_b, 1, ps_b, bo_b, PROFILE_GEMINI, _returns_circuit_blocked, pdf, needs_escalation=True
    )
    assert state_b.total_cost == 0.0
    assert ps_b.page_cost_usd == 0.0

    # Arm C (control): real executed CLI_ERROR -> still billed!
    state_c = DocumentState(DocumentHandle(pdf))
    ps_c = state_c.pages[1]
    bo_c = PageOutput(page_num=1, text=_SHIFTED_MD, status=PageStatus.SUCCESS, engine="qwen")
    ps_c.attempts.append(bo_c)
    ps_c.best_output = bo_c
    pipe_c = UnifiedPipeline(PipelineConfig(quiet=True, escalate_ambiguous_tables=True))

    def _returns_cli_error(profile, page_num):
        return PageOutput(
            page_num=page_num,
            text="",
            status=PageStatus.ERROR,
            engine=profile.engine.value,
            failure_mode=FailureMode.CLI_ERROR,
            error="CLI exited 1",
        )

    pipe_c._escalate_table_page(
        state_c, 1, ps_c, bo_c, PROFILE_GEMINI, _returns_cli_error, pdf, needs_escalation=True
    )
    assert state_c.total_cost == pytest.approx(PROFILE_GEMINI.cost_per_page_usd)


# ---------------------------------------------------------------------------
# 2. route_page / process() skips paid Gemini when CircuitBreakerPageJudge is open
# ---------------------------------------------------------------------------


def test_route_page_skips_paid_rung_when_circuit_breaker_judge_is_open() -> None:
    """Paired difference on ``route_page``: identical ladder and failing free rung,
    changing ONLY whether the nested ``CircuitBreakerPageJudge`` is open."""

    class _RejectLeaf:
        def assess(self, output: PageOutput, provider) -> AcceptDecision:
            return AcceptDecision(accept=False, reason="unfaithful")

    class _OuterWrap:
        def __init__(self, inner) -> None:
            self._inner = inner

        def assess(self, output: PageOutput, provider) -> AcceptDecision:
            return self._inner.assess(output, provider)

    def _run_arm(*, circuit_open: bool) -> tuple[list[str], float]:
        called: list[str] = []

        def _run_provider(prof, pn: int) -> PageOutput:
            called.append(prof.engine.value)
            return PageOutput(
                page_num=pn,
                text="Estimated table text " * 20,
                status=PageStatus.SUCCESS,
                engine=prof.engine.value,
            )

        judge = _OuterWrap(CircuitBreakerPageJudge(_RejectLeaf(), lambda: circuit_open))
        decision = route_page(
            1,
            [PROFILE_QWEN_LOCAL, PROFILE_GEMINI],
            _run_provider,
            judge,
        )
        return called, decision.total_cost_usd

    open_calls, open_cost = _run_arm(circuit_open=True)
    closed_calls, closed_cost = _run_arm(circuit_open=False)

    assert open_calls == ["qwen"]
    assert open_cost == 0.0
    assert closed_calls == ["qwen", "gemini"]
    assert closed_cost == pytest.approx(PROFILE_GEMINI.cost_per_page_usd)


def test_process_skips_paid_gemini_once_judge_circuit_breaker_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real ``process()`` paired difference (open vs closed judge circuit):

    * Wedged arm: judge hangs on p1 -> ``_judge_circuit_breaker`` probes and
      opens ``CircuitBreakerPageJudge`` -> paid Gemini is skipped on BOTH p1
      and p2 (0 Gemini calls, $0 paid cost).
    * Closed control arm: judge answers quickly and rejects Qwen -> paid Gemini
      runs on both pages (2 Gemini calls).
    """
    monkeypatch.setattr(agentic_mod, "DEFAULT_PROVIDER_TIMEOUTS", {e: 0.15 for e in EngineType})

    def _run_arm(sub: str, *, wedged: bool) -> tuple[list[tuple[str, int]], dict[int, dict]]:
        arm_dir = tmp_path / sub
        arm_dir.mkdir(parents=True, exist_ok=True)
        pdf = _prose_pdf(arm_dir, pages=2)
        out_dir = arm_dir / "out"
        calls: list[tuple[str, int]] = []

        def _fake_judge(self, image_path, text, *a, **k):
            if wedged:
                time.sleep(0.8)
                raise AssertionError("deadline should have fired first")
            return JudgeVerdict(faithful=False, issues=["missing row"], confidence=0.2)

        monkeypatch.setattr(OllamaVisionJudge, "judge", _fake_judge)
        monkeypatch.setattr(
            "socr.pipeline.orchestrator.probe_model_generation",
            lambda *a, **k: (False, "timed out") if wedged else (True, ""),
        )

        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN, EngineType.GEMINI],
                judge_backend="vlm",
                write_manifest=False,
                table_judge_ladder=False,
                escalate_ambiguous_tables=False,
            )
        )
        detect = pipe.bd_detector.detect

        def _needs_ocr(path: Path):
            assessment = detect(path)
            for p in assessment.pages:
                p.needs_ocr_enhancement = True
            return assessment

        pipe.bd_detector.detect = _needs_ocr
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL, PROFILE_GEMINI]
        pipe._resolve_judge_model = lambda *a, **k: "qwen3.8:27b"
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._make_page_renderer = lambda state: lambda pn: "img"
        pipe._probe_backend_idle = lambda: True

        def _fake_run(state, nums, nat, eng, phase, profile=None, **kw):
            for pn in nums:
                calls.append((eng.value, pn))
            return [
                PageOutput(
                    page_num=pn,
                    text=f"Estimated policy coefficient text for page {pn} " * 8,
                    status=PageStatus.SUCCESS,
                    engine=eng.value,
                )
                for pn in nums
            ]

        pipe._run_engine_on_pages = _fake_run
        pipe.process(pdf, output_dir=out_dir)
        return calls, _sidecars(out_dir)

    wedged_calls, wedged_sc = _run_arm("wedged", wedged=True)
    closed_calls, closed_sc = _run_arm("closed", wedged=False)

    wedged_gemini = [c for c in wedged_calls if c[0] == "gemini"]
    closed_gemini = [c for c in closed_calls if c[0] == "gemini"]

    assert wedged_gemini == [], f"open judge breaker must skip paid Gemini, got {wedged_calls}"
    assert sum(sc.get("page_cost_usd", 0.0) for sc in wedged_sc.values()) == 0.0
    assert len(closed_gemini) == 2, f"closed control must escalate to Gemini, got {closed_calls}"
    assert sum(sc.get("page_cost_usd", 0.0) for sc in closed_sc.values()) > 0.0


# ---------------------------------------------------------------------------
# 3. No forbidden qwen3-vl:8b in _JUDGE_MODEL_CANDIDATES
# ---------------------------------------------------------------------------


def test_judge_model_candidates_contain_no_forbidden_qwen_8b() -> None:
    candidates = UnifiedPipeline._JUDGE_MODEL_CANDIDATES
    assert candidates, "_JUDGE_MODEL_CANDIDATES must not be empty"
    for model in candidates:
        lower = model.lower()
        assert "qwen3-vl:8b" not in lower
        assert not re.search(r"qwen.*:8b\b", lower), f"forbidden :8b Qwen model: {model!r}"
        assert lower != "qwen3-vl:30b", "thinking qwen3-vl:30b is forbidden"


# ---------------------------------------------------------------------------
# 4. _socr_source_digest ignores non-reading packages while tracking reading edits
# ---------------------------------------------------------------------------


def test_socr_source_digest_ignores_non_reading_packages_and_tracks_reading_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = tmp_path / "socr"
    files = {
        "core/verdict.py": "VERDICT = 1\n",
        "engines/qwen.py": "MODEL = 'qwen3-vl:30b-a3b-instruct'\n",
        "tables/second_reader.py": "STRICT = True\n",
        "pipeline/orchestrator.py": "ORCH = 1\n",
        "pipeline/refinalize_own_text.py": "REFIN = 1\n",
        "benchmark/runner.py": "BENCH = 1\n",
        "refinalize/offline.py": "OFFLINE = 1\n",
        "ui/dashboard.py": "UI = 1\n",
        "devtools/regen.py": "DEV = 1\n",
    }
    for rel, content in files.items():
        p = pkg / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    (pkg / "__init__.py").write_text("__version__ = '0.0.0'\n")

    def _live_digest() -> str:
        orch._SOURCE_DIGEST_CACHE = None
        with patch.object(socr, "__file__", str(pkg / "__init__.py")):
            return orch._socr_source_digest()

    try:
        baseline = _live_digest()

        # Editing ANY non-reading package/module must NOT invalidate the digest.
        for non_reading in (
            "benchmark/runner.py",
            "refinalize/offline.py",
            "ui/dashboard.py",
            "devtools/regen.py",
            "pipeline/refinalize_own_text.py",
        ):
            (pkg / non_reading).write_text("# edited non-reading file\nX = 999\n")
            assert _live_digest() == baseline, f"{non_reading} must not change _socr_source_digest"

        # Editing a reading-path module MUST change the digest.
        for reading in (
            "core/verdict.py",
            "engines/qwen.py",
            "tables/second_reader.py",
            "pipeline/orchestrator.py",
        ):
            before = _live_digest()
            (pkg / reading).write_text((pkg / reading).read_text() + "# reading edit\n")
            after = _live_digest()
            assert after != before, (
                f"reading-path edit to {reading} must change _socr_source_digest"
            )
    finally:
        orch._SOURCE_DIGEST_CACHE = None
