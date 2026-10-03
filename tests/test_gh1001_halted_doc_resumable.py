"""GH-1001: a document that ended in a PARTIAL_SAVE_VLM_TIMEOUT halt is not skippable.

``_resume_skippable`` skips a PARTIAL document with an unchanged fingerprint because
re-running "cannot improve" it. A halt is transient, so a plain re-run must reach the
per-page ledger, which reuses only the pages that finished.

Pinned as a DIFFERENCE in one process: the same halted output directory is re-run once as
recorded (latch present) and once with the latch stripped from the root index, which is
the shape an index had before the fix. No locally measured status tuple is pinned.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from test_gh995_post_halt_not_terminal import PAGES, NATIVE, _decision, _pdf
from test_pp2_agentic_fuse import EngineType, _make_bd_assessment, _make_config, _make_pipeline

from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, FailureMode

HALT_PAGE = 3  # p1 (OCR) and p2 (native) finish; p3 times out; p4 is never reached


def _run(pdf: Path, out: Path, *, halt: bool, providers=(PROFILE_QWEN_LOCAL,), pipeline=None):
    pipeline = pipeline or _make_pipeline(
        _make_config(agentic=True, enabled_engines=[EngineType.QWEN])
    )
    pipeline.bd_detector = MagicMock()
    pipeline.bd_detector.detect.return_value = _make_bd_assessment(PAGES, born_digital_pages=NATIVE)
    routed: list[int] = []
    ledger_hits: list[int] = []
    real_load = pipeline._load_terminal_page

    def _spy_load(state, page_num, output_dir):
        hit = real_load(state, page_num, output_dir)
        if hit is not None:
            ledger_hits.append(page_num)
        return hit

    pipeline._load_terminal_page = _spy_load

    def _route(page_num, ladder, run_provider, judge, **kwargs):
        routed.append(page_num)
        return _decision(page_num, ladder, timeout=halt and page_num == HALT_PAGE)

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=list(providers)),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=not halt),
    ):
        result = pipeline.process(pdf, out)
    return result, routed, ledger_hits


def _strip_latch(out: Path) -> None:
    index = out / "metadata.json"
    data = json.loads(index.read_text())
    stripped = 0
    for entry in data["files"].values():
        if entry.pop("halt_retry_pending", None) is not None:
            stripped += 1
    assert stripped == 1, "the halted run must record the latch"
    index.write_text(json.dumps(data))


def test_plain_rerun_of_halted_doc_resumes_via_ledger_but_unlatched_is_skipped(
    tmp_path: Path,
) -> None:
    pdf = _pdf(tmp_path)
    halted = tmp_path / "halted"
    first, routed, _ = _run(pdf, halted, halt=True)
    assert first.error and "PARTIAL_SAVE_VLM_TIMEOUT" in first.error
    assert routed == [1, HALT_PAGE]  # the halt really cut the loop after p3

    prefix = tmp_path / "prefix"
    shutil.copytree(halted, prefix)
    _strip_latch(prefix)

    # Without the latch (the pre-fix index shape): skipped whole, nothing reprocessed.
    skipped, routed_prefix, hits_prefix = _run(pdf, prefix, halt=False)
    assert skipped.status is DocumentStatus.SKIPPED
    assert routed_prefix == [] and hits_prefix == []

    # With it: the pages after the halt are processed, the finished ones are ledger hits.
    resumed, routed_fixed, hits_fixed = _run(pdf, halted, halt=False)
    assert resumed.status is not DocumentStatus.SKIPPED
    assert HALT_PAGE in routed_fixed
    assert {1, 2} <= set(hits_fixed)
    assert HALT_PAGE not in hits_fixed and 4 not in hits_fixed


def _latch(out: Path) -> bool:
    entries = json.loads((out / "metadata.json").read_text())["files"]
    assert len(entries) == 1
    return next(iter(entries.values())).get("halt_retry_pending") is True


def _entry(out: Path) -> dict:
    return next(iter(json.loads((out / "metadata.json").read_text())["files"].values()))


def test_latch_follows_the_outcome_not_the_halt_event(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path)
    out = tmp_path / "out"

    _run(pdf, out, halt=True)
    assert _latch(out)  # 1. halt run

    # 2. providerless re-run: no new halt, but p3/p4 are still unprocessed.
    providerless, routed, _ = _run(pdf, out, halt=False, providers=())
    assert routed == []
    assert providerless.status is not DocumentStatus.SKIPPED
    assert _latch(out)
    again, routed_again, _ = _run(pdf, out, halt=False, providers=())
    assert again.status is not DocumentStatus.SKIPPED  # not skippable: it ran again
    assert routed_again == []
    assert _latch(out)

    # The native page past the halt is taken by the NATIVE lane, not the OCR no-provider
    # branch; it was not processed in any run, so it must stay flagged and non-terminal.
    sidecars = {
        int(f.stem): json.loads(f.read_text()) for f in (out / "doc" / "pages").glob("*.json")
    }
    assert sidecars[4]["terminal"] is not True
    assert sidecars[4]["failure_mode"] == FailureMode.PAGE_NOT_PROCESSED_AFTER_HALT.value
    assert sidecars[2]["terminal"] is True  # finished before the halt: restored, not flagged

    # 3. recovered re-run processes the pages and clears the latch.
    recovered, routed_ok, _ = _run(pdf, out, halt=False)
    assert recovered.status is not DocumentStatus.SKIPPED
    assert HALT_PAGE in routed_ok
    assert "halt_retry_pending" not in _entry(out)

    # 4. a further plain re-run is skipped as usual.
    final, routed_final, _ = _run(pdf, out, halt=False)
    assert final.status is DocumentStatus.SKIPPED
    assert routed_final == []


def test_prior_latch_is_not_inherited_by_a_fresh_output_dir(tmp_path: Path) -> None:
    """Same pipeline instance, same input, fresh output dir: nothing carries over."""
    pdf = _pdf(tmp_path)
    pipeline = _make_pipeline(_make_config(agentic=True, enabled_engines=[EngineType.QWEN]))

    _run(pdf, tmp_path / "a", halt=True, pipeline=pipeline)
    assert _latch(tmp_path / "a")
    # A second run in "a" READS the latch, which is what a cross-run cache would keep.
    _run(pdf, tmp_path / "a", halt=False, providers=(), pipeline=pipeline)
    assert _latch(tmp_path / "a")

    fresh = tmp_path / "b"
    _run(pdf, fresh, halt=False, providers=(), pipeline=pipeline)
    assert "halt_retry_pending" not in _entry(fresh)
    sidecars = (fresh / "doc" / "pages").glob("*.json")
    modes = {json.loads(f.read_text())["failure_mode"] for f in sidecars}
    assert FailureMode.PAGE_NOT_PROCESSED_AFTER_HALT.value not in modes
