"""#1013: re-judge a page-judge-TIMEOUT candidate on resume instead of re-OCRing.

Every pin is a DIFFERENCE between two runs on ONE output directory that differ in exactly
one thing (CLAUDE.md, #257): the verdict the judge gives in run 2, or one identity field of
the kept candidate. No absolute outcome measured locally is pinned.

Hermetic: ``_available_engines_for_agentic`` is patched (CI has no provider), the judge is
injected through ``_build_page_judge``, ``_resolve_judge_model`` returns "", and the OCR
engine call is replaced by a counter so "no OCR call in run 2" is a measured fact.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.born_digital import DocumentAssessment, PageAssessment  # noqa: E402
from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.providers import PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.judge.judge import JudgeVerdict, PageJudgeTimeoutError  # noqa: E402
from socr.pipeline.agentic import (  # noqa: E402
    REJUDGE_EVENT_KINDS,
    AcceptDecision,
    rejudge_candidate,
)
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

_MODEL_TEXT = "model read of the page: the estimate is 0.42 with n = 117 observations"
_OTHER_TEXT = "a DIFFERENT model read produced by the ladder in run 2, estimate 0.43"
_NATIVE_TEXT = "native layer of the page with garbled symbols and enough words to count"


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_QWEN_LOCAL]
    )
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")


class _Judge:
    """Judge whose verdict is chosen per run: ``timeout``, ``accept`` or ``reject``."""

    def __init__(self) -> None:
        self.mode = "timeout"
        self.calls: list[str] = []

    def assess(self, output, provider):
        self.calls.append(output.text)
        if self.mode == "timeout":
            raise PageJudgeTimeoutError("page judge timeout (test)")
        if self.mode == "accept":  # a completed VLM verdict
            return AcceptDecision(
                accept=True, reason="faithful", raw_verdict=JudgeVerdict(faithful=True)
            )
        if self.mode == "accept_heuristic":  # accept=True, no VLM verdict behind it
            return AcceptDecision(accept=True, reason="heuristics passed")
        if self.mode == "accept_label_unverified":
            # SourceEvidenceTableJudge annotates the judged output in place.
            output.table_label_unverified = "row labels not witnessed"
            return AcceptDecision(accept=True, reason="x", raw_verdict=JudgeVerdict(faithful=True))
        if self.mode == "accept_mutating":  # the verifier rewrote the text it was shown
            if self.calls[-1] == _MODEL_TEXT:
                output.text = output.text + " (rewritten)"
            return AcceptDecision(accept=True, reason="x", raw_verdict=JudgeVerdict(faithful=True))
        if self.mode == "accept_with_outcome":  # a missing verdict wearing a verdict's shape
            return AcceptDecision(
                accept=True,
                reason="x",
                raw_verdict=JudgeVerdict(faithful=True),
                judge_outcome="page_judge_verifier_error",
            )
        return AcceptDecision(accept=False, reason=self.mode)


class _Harness:
    def __init__(self, tmp_path: Path, **config) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.tmp = tmp_path
        self.pdf = tmp_path / "doc.pdf"
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "born-digital page " * 8)
        doc.save(str(self.pdf))
        doc.close()
        self.out = tmp_path / "out"
        self.judge = _Judge()
        self.ocr_calls: list[int] = []
        self.ocr_text = _MODEL_TEXT
        self.config = config
        self.runs = 0
        self.judge_identity = "vlm-test"
        self.judge_backend = config.pop("judge_backend", "heuristic")

    def run(self, *, real_builder: bool = False):
        # A recorded AUDIT_FAILED document is skipped at the root index, so the real
        # second run is `--reprocess`; that flag is excluded from the fingerprint.
        reprocess = self.runs > 0
        self.runs += 1
        pipeline = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                judge_backend=self.judge_backend,
                enabled_engines=[EngineType.GEMINI],
                primary_engine=EngineType.DEEPSEEK,
                save_figures=False,
                dual_pass_tables=False,
                detect_equations=False,
                recover_clean_equations=False,
                table_judge_ladder=False,
                quiet=True,
                write_manifest=False,
                reprocess=reprocess,
                **self.config,
            )
        )
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = DocumentAssessment(
            path=self.pdf,
            pages=[
                PageAssessment(
                    page_num=1,
                    is_born_digital=True,
                    native_text=_NATIVE_TEXT,
                    confidence=0.9,
                    needs_ocr_enhancement=True,
                )
            ],
        )

        def _engine(state, pages, *a, **kw):
            self.ocr_calls.extend(pages)
            return [
                PageOutput(
                    page_num=p,
                    text=self.ocr_text,
                    status=PageStatus.SUCCESS,
                    engine="qwen",
                    audit_passed=True,
                )
                for p in pages
            ]

        build = (
            contextlib.nullcontext()
            if real_builder
            else patch.object(pipeline, "_build_page_judge", side_effect=self._build_judge)
        )
        with patch.object(pipeline, "_run_engine_on_pages", side_effect=_engine), build:
            result = pipeline.process(self.pdf, self.out)
        return result, pipeline

    def _build_judge(self, state):
        # What the real builder records: the judge that ACTUALLY runs.
        state.agentic_judge_model = self.judge_identity
        return self.judge

    def sidecar_path(self) -> Path:
        return next((self.out / "doc" / "pages").glob("*.json"))

    def sidecar(self) -> dict:
        return json.loads(self.sidecar_path().read_text(encoding="utf-8"))

    def shipped(self) -> str:
        return next((self.out / "doc" / "pages").glob("*.md")).read_text(encoding="utf-8")


def _run1_times_out(h: _Harness) -> None:
    h.judge.mode = "timeout"
    result, _ = h.run()
    assert h.ocr_calls == [1], "run 1 must OCR the page once"
    assert h.sidecar()["failure_mode"] == FailureMode.NATIVE_UNTRUSTED_JUDGE_TIMEOUT.value
    assert "judge_timeout_candidate" in h.sidecar()
    h.ocr_calls.clear()
    h.judge.calls.clear()


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert (
        Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")
    ), socr.__file__


def test_a_accepting_rejudge_ships_the_candidate_without_ocr(tmp_path) -> None:
    h = _Harness(tmp_path)
    _run1_times_out(h)
    assert _NATIVE_TEXT in h.shipped()

    h.judge.mode = "accept"
    h.ocr_text = _OTHER_TEXT
    result, pipeline = h.run()

    assert h.ocr_calls == [], "an accepting re-judge must not OCR the page"
    assert h.judge.calls == [_MODEL_TEXT], "the judge saw the kept bytes, once"
    assert _MODEL_TEXT in h.shipped()
    assert _OTHER_TEXT not in h.shipped()
    sidecar = h.sidecar()
    assert sidecar["failure_mode"] != FailureMode.NATIVE_UNTRUSTED_JUDGE_TIMEOUT.value
    assert sidecar["winning_output"]["text"] == _MODEL_TEXT
    kinds = [e["kind"] for e in sidecar["audit_events"]]
    assert kinds.count("rejudge_accepted") == 1


def test_b_rejecting_rejudge_runs_the_ladder_and_does_not_ship_the_candidate(tmp_path) -> None:
    accept_h = _Harness(tmp_path / "acc")
    reject_h = _Harness(tmp_path / "rej")
    for h in (accept_h, reject_h):
        _run1_times_out(h)
        h.ocr_text = _OTHER_TEXT
    accept_h.judge.mode = "accept"
    reject_h.judge.mode = "reject"
    accept_h.run()
    reject_h.run()

    # Differ in exactly the verdict: accept skips OCR, reject runs it.
    assert accept_h.ocr_calls == []
    assert reject_h.ocr_calls == [1]
    assert _MODEL_TEXT not in reject_h.shipped()
    # The judge refused the ladder's candidate too (same mode), so a completed rejection
    # is NOT the timeout ending.
    assert reject_h.sidecar()["failure_mode"] != FailureMode.NATIVE_UNTRUSTED_JUDGE_TIMEOUT.value
    kinds = [e["kind"] for e in reject_h.sidecar()["audit_events"]]
    assert kinds.count("rejudge_rejected") == 1


def test_second_timeout_runs_the_ladder_not_native_on_a_bare_timeout(tmp_path) -> None:
    h = _Harness(tmp_path)
    _run1_times_out(h)
    h.judge.mode = "timeout"
    h.run()
    # The ladder ran (one OCR call) after the single bounded re-judge attempt.
    assert h.ocr_calls == [1]
    assert h.judge.calls.count(_MODEL_TEXT) == 2  # 1 re-judge + 1 ladder candidate
    kinds = [e["kind"] for e in h.sidecar()["audit_events"]]
    assert kinds.count("rejudge_timeout") == 1


def test_rejudge_attempts_is_bounded_by_config(tmp_path) -> None:
    counts = {}
    for n in (0, 1, 3):
        h = _Harness(tmp_path / f"n{n}", rejudge_attempts=n)
        _run1_times_out(h)
        h.run()
        counts[n] = h.judge.calls.count(_MODEL_TEXT) - 1  # minus the ladder's own call
    assert counts == {0: 0, 1: 1, 3: 3}


def test_c_changed_fingerprint_or_tampered_bytes_means_no_reuse(tmp_path) -> None:
    def _fresh(tag: str) -> _Harness:
        h = _Harness(tmp_path / tag)
        _run1_times_out(h)
        h.judge.mode = "accept"
        h.ocr_text = _OTHER_TEXT
        return h

    control = _fresh("control")
    control.run()

    stale_fp = _fresh("fp")
    side = stale_fp.sidecar()
    side["run_fingerprint"] = "0" * 64
    stale_fp.sidecar_path().write_text(json.dumps(side), encoding="utf-8")
    stale_fp.run()

    tampered = _fresh("bytes")
    side = tampered.sidecar()
    side["judge_timeout_candidate"]["candidate"]["text"] = _MODEL_TEXT + " (edited)"
    tampered.sidecar_path().write_text(json.dumps(side), encoding="utf-8")
    tampered.run()

    wrong_input = _fresh("input")
    side = wrong_input.sidecar()
    side["input_checksum"] = "deadbeef"
    wrong_input.sidecar_path().write_text(json.dumps(side), encoding="utf-8")
    wrong_input.run()

    assert control.ocr_calls == []
    for h in (stale_fp, tampered, wrong_input):
        assert h.ocr_calls == [1]
        assert _OTHER_TEXT in h.shipped()
        assert h.judge.calls == [_OTHER_TEXT], "only the ladder's own candidate was judged"


def test_unaffected_page_sidecar_has_no_kept_candidate(tmp_path) -> None:
    h = _Harness(tmp_path)
    h.judge.mode = "accept"
    h.run()
    assert "judge_timeout_candidate" not in h.sidecar()
    assert not any(k in json.dumps(h.sidecar()) for k in REJUDGE_EVENT_KINDS)


def test_rejudge_candidate_stops_on_first_non_timeout_answer() -> None:
    class _J:
        def __init__(self, script):
            self.script, self.n = list(script), 0

        def assess(self, output, provider):
            self.n += 1
            step = self.script.pop(0)
            if step == "timeout":
                raise PageJudgeTimeoutError("t")
            return AcceptDecision(accept=step == "accept", reason=step)

    def make():
        return PageOutput(page_num=1, text="x y z", status=PageStatus.WARNING)

    j = _J(["timeout", "reject", "accept"])
    assert rejudge_candidate(make, PROFILE_QWEN_LOCAL, j, attempts=5)[0] == "rejected"
    assert j.n == 2


def _timed_out_then(tmp_path, tag, mode, *, edit=None, identity=None):
    """Run 1 times out; optionally edit the sidecar; run 2 uses ``mode``."""
    h = _Harness(tmp_path / tag)
    _run1_times_out(h)
    if edit is not None:
        side = h.sidecar()
        edit(side)
        h.sidecar_path().write_text(json.dumps(side), encoding="utf-8")
    h.judge.mode = mode
    h.ocr_text = _OTHER_TEXT
    if identity is not None:
        h.judge_identity = identity
    h.run()
    return h


def _kinds(h):
    return [e["kind"] for e in h.sidecar()["audit_events"]]


def test_item1_only_a_completed_vlm_acceptance_ships(tmp_path) -> None:
    control = _timed_out_then(tmp_path, "ok", "accept")
    assert control.ocr_calls == [] and "rejudge_accepted" in _kinds(control)
    for mode in ("accept_heuristic", "accept_with_outcome"):
        h = _timed_out_then(tmp_path, mode, mode)
        assert h.ocr_calls == [1], mode
        assert _MODEL_TEXT not in h.shipped(), mode
        assert "rejudge_accepted" not in _kinds(h), mode
        assert "rejudge_error" in _kinds(h), mode


def test_item1_a_degraded_or_different_judge_does_not_ship(tmp_path) -> None:
    for tag, identity in (("heur", "heuristic"), ("other", "some-other-vlm")):
        h = _timed_out_then(tmp_path, tag, "accept", identity=identity)
        assert h.ocr_calls == [1], tag
        assert h.judge.calls == [_OTHER_TEXT], "the kept bytes were never shown to it"
        assert "rejudge_error" in _kinds(h)


def test_item2_judge_construction_failure_falls_through(tmp_path, monkeypatch) -> None:
    """Same fingerprint (resolved model unchanged), but the VLM judge cannot be built."""
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "m")
    h = _Harness(tmp_path, judge_backend="vlm")
    _run1_times_out(h)
    fp1 = h.sidecar()["run_fingerprint"]
    h.ocr_text = _OTHER_TEXT

    def boom(*a, **kw):
        raise RuntimeError("cannot construct the judge")

    # The REAL _build_page_judge runs and degrades to heuristics.
    with patch("socr.judge.ollama_judge.OllamaVisionJudge", side_effect=boom):
        h.run(real_builder=True)
    assert h.sidecar()["run_fingerprint"] == fp1, "reuse would have been admitted"
    assert h.ocr_calls == [1]
    assert _MODEL_TEXT not in h.shipped()
    assert "rejudge_accepted" not in _kinds(h)


def test_item3_serialised_engine_is_not_trusted(tmp_path) -> None:
    def spoof(side):
        side["judge_timeout_candidate"]["candidate"]["engine"] = "chart_asset"

    h = _timed_out_then(tmp_path, "spoof", "accept", edit=spoof)
    assert h.ocr_calls == []  # still the same profile: it is re-judged and accepted
    assert h.sidecar()["winning_output"]["engine"] != "chart_asset"
    assert h.sidecar()["winning_output"]["engine"] == "qwen"

    def wrong_model(side):
        side["judge_timeout_candidate"]["candidate"]["provider_model"] = "not-this-model"

    h2 = _timed_out_then(tmp_path, "model", "accept", edit=wrong_model)
    assert h2.ocr_calls == [1]

    def unknown_provider(side):
        side["judge_timeout_candidate"]["candidate"]["provider_id"] = "no-such-rung"

    h3 = _timed_out_then(tmp_path, "prov", "accept", edit=unknown_provider)
    assert h3.ocr_calls == [1]


def test_item4_shipped_bytes_hash_equals_judged_hash(tmp_path) -> None:
    from socr.core.page_credential import sha256_text

    h = _timed_out_then(tmp_path, "hash", "accept")
    assert h.ocr_calls == []
    shipped = h.sidecar()["winning_output"]["text"]
    assert [sha256_text(t) for t in h.judge.calls] == [sha256_text(shipped)]

    # A judge chain that rewrites the text it was shown has not judged the kept bytes.
    m = _timed_out_then(tmp_path, "mut", "accept_mutating")
    assert m.ocr_calls == [1]
    assert "rejudge_accepted" not in _kinds(m)
    assert "(rewritten)" not in m.shipped()


def test_fields_set_by_verification_survive_to_the_ship(tmp_path) -> None:
    """Label disclosure set on the judged snapshot must reach the shipped page (WARNING)."""
    plain = _timed_out_then(tmp_path, "plain", "accept")
    flagged = _timed_out_then(tmp_path, "flagged", "accept_label_unverified")
    for h in (plain, flagged):
        assert h.ocr_calls == [] and "rejudge_accepted" in _kinds(h)
        assert _MODEL_TEXT in h.shipped()
    # Differ in exactly the verification annotation.
    assert plain.sidecar()["winning_output"]["status"] == "success"
    assert not plain.sidecar()["winning_output"].get("table_label_unverified")
    won = flagged.sidecar()["winning_output"]
    assert won["table_label_unverified"] == "row labels not witnessed"
    assert won["status"] == "warning"
