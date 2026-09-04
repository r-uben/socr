"""VI-B1: per-page exclusive stage wall-clock.

Measurement only. Proves (a) exclusive keys sum to ``timings_s.total`` within
1 ms, (b) resume skip is identical with and without ``timings_s``, (c) the
final ``.md`` is byte-identical with timings on and off.

Hermetic: CI has no ollama. Every ``process()`` path patches
``_available_engines_for_agentic`` -> ``[PROFILE_QWEN_LOCAL]`` and
``_resolve_judge_model`` -> ``""``. Pin a DIFFERENCE, never a value.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.cli import format_timings_summary_line
from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.cache import BlobStore
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.manifest import (
    PAGE_TIMING_EXCLUSIVE_KEYS,
    build_manifest,
    exclusive_timings_sum,
    rollup_page_timings,
)
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline, _PageStageClock


_OCR_TEXT = (
    "This document presents an analysis of market dynamics across several European "
    "economies during the post-pandemic recovery period. We examine monetary policy "
    "transmission mechanisms and their effects on inflation expectations output gaps "
    "and financial stability indicators. Our empirical framework builds on vector "
    "autoregressive models with sign restrictions estimated using Bayesian methods "
    "on quarterly macroeconomic data spanning the period from 2019 to 2024. The "
    "results suggest that unconventional monetary policy tools had asymmetric "
    "effects across core and peripheral economies in the sample studied here."
)


def _make_config(**overrides) -> PipelineConfig:
    kwargs = dict(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        save_figures=False,
        write_manifest=True,
        dual_pass_tables=False,
        detect_equations=False,
        recover_clean_equations=False,
        table_judge_ladder=False,
    )
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


def _make_pipeline(config: PipelineConfig | None = None) -> UnifiedPipeline:
    return UnifiedPipeline(config or _make_config())


def _real_pdf(tmp_path: Path, page_count: int = 1) -> Path:
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    for i in range(page_count):
        doc.new_page().insert_text((72, 72), f"page {i + 1} text " * 10)
    doc.save(str(path))
    doc.close()
    return path


def _bd_assessment(pdf_path: Path, page_count: int, born_digital: set[int]) -> DocumentAssessment:
    pages = []
    for i in range(1, page_count + 1):
        is_bd = i in born_digital
        pages.append(
            PageAssessment(
                page_num=i,
                is_born_digital=is_bd,
                native_text=f"native text for page {i} " * 8 if is_bd else "",
                confidence=0.9,
            )
        )
    return DocumentAssessment(path=pdf_path, pages=pages)


def _route_page_returning(text: str, engine: str = "qwen"):
    def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
        from socr.pipeline.agentic import PageDecision, ProviderAttempt

        out = PageOutput(
            page_num=page_num,
            text=text,
            status=PageStatus.SUCCESS,
            engine=engine,
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

    return _fake_route


def _process(pipeline: UnifiedPipeline, pdf_path: Path, out_dir: Path) -> object:
    with (
        patch(
            "socr.pipeline.orchestrator.route_page",
            side_effect=_route_page_returning(_OCR_TEXT),
        ),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        return pipeline.process(pdf_path, out_dir)


def _sidecar(out_dir: Path, page_num: int = 1) -> dict:
    path = next(out_dir.rglob(f"pages/{page_num:05d}.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _fragment(out_dir: Path, page_num: int = 1) -> str:
    path = next(out_dir.rglob(f"pages/{page_num:05d}.md"))
    return path.read_text(encoding="utf-8")


def _assert_exclusive_sum(timings: dict) -> None:
    assert abs(exclusive_timings_sum(timings) - float(timings["total"])) <= 0.001
    for key in PAGE_TIMING_EXCLUSIVE_KEYS:
        assert key in timings
        assert timings[key] >= 0.0


class _NullClock:
    """Timings-off clock: same control flow, no recorded seconds."""

    def __init__(self, now=None) -> None:
        pass

    @contextmanager
    def span(self, key: str):
        yield

    def add_exclusive(self, key: str, seconds: float) -> None:
        return None

    def finalize(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# (a) exclusive keys sum to total
# ---------------------------------------------------------------------------


def test_nested_clock_subtracts_children_from_parent() -> None:
    ticks = [0.0]

    def now() -> float:
        return ticks[0]

    clock = _PageStageClock(now=now)
    with clock.span("route"):
        ticks[0] += 0.05
        with clock.span("extract"):
            ticks[0] += 1.20
    with clock.span("tables"):
        ticks[0] += 0.10
        with clock.span("ladder"):
            ticks[0] += 0.40
            with clock.span("adjudication"):
                ticks[0] += 0.25
    with clock.span("flush"):
        ticks[0] += 0.02
    got = clock.finalize()

    assert got["extract"] == pytest.approx(1.20)
    assert got["route"] == pytest.approx(0.05)
    assert got["adjudication"] == pytest.approx(0.25)
    assert got["ladder"] == pytest.approx(0.40)
    assert got["tables"] == pytest.approx(0.10)
    assert got["flush"] == pytest.approx(0.02)
    _assert_exclusive_sum(got)
    assert got["total"] == pytest.approx(2.02)


def test_total_is_independent_page_wall_not_the_stage_sum() -> None:
    """A constructed total=sum(stages) cannot show unattributed time.

    Advance the fake clock between spans: ``total`` must move with the wall,
    not with the exclusive keys. That is the 8 min/page owner: a gap the
    stages did not name.
    """
    ticks = [0.0]

    def now() -> float:
        return ticks[0]

    clock = _PageStageClock(now=now)
    with clock.span("extract"):
        ticks[0] += 1.0
    ticks[0] += 0.05  # unattributed
    got = clock.finalize()

    assert got["extract"] == pytest.approx(1.0)
    assert got["total"] == pytest.approx(1.05)
    assert exclusive_timings_sum(got) == pytest.approx(1.0)
    assert abs(exclusive_timings_sum(got) - got["total"]) > 0.001


def test_process_sidecar_exclusive_keys_sum_to_total(tmp_path: Path) -> None:
    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _make_pipeline()
    pipeline.bd_detector = MagicMock()
    pipeline.bd_detector.detect.return_value = _bd_assessment(pdf_path, 1, born_digital=set())

    _process(pipeline, pdf_path, out_dir)

    meta = _sidecar(out_dir)
    assert "timings_s" in meta
    _assert_exclusive_sum(meta["timings_s"])
    # OCR path: extract is nested under route, so route is the exclusive remainder.
    assert meta["timings_s"]["extract"] >= 0.0
    assert meta["timings_s"]["route"] >= 0.0
    assert meta["timings_s"]["flush"] >= 0.0
    assert "timings_s" not in json.dumps(meta.get("audit_events") or [])


def test_native_page_records_extract_not_route(tmp_path: Path) -> None:
    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _make_pipeline()
    pipeline.bd_detector = MagicMock()
    pipeline.bd_detector.detect.return_value = _bd_assessment(pdf_path, 1, born_digital={1})

    _process(pipeline, pdf_path, out_dir)

    meta = _sidecar(out_dir)
    _assert_exclusive_sum(meta["timings_s"])
    # Trusted native copies already-extracted text; it is not an OCR route.
    assert meta["timings_s"]["extract"] >= 0.0


# ---------------------------------------------------------------------------
# (b) resume skip identical with and without timings_s
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda meta: meta, id="unmodified"),
        pytest.param(lambda meta: {**meta, "timings_s": None}, id="timings_explicitly_null"),
        pytest.param(
            lambda meta: {k: v for k, v in meta.items() if k != "timings_s"},
            id="timings_key_absent_old_sidecar",
        ),
    ],
)
def test_resume_decision_is_identical_regardless_of_timings_s(tmp_path: Path, mutate) -> None:
    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent

    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    accepted = PageOutput(
        page_num=1,
        text="accepted body text long enough to look like real content here.",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    ps.attempts.append(accepted)
    ps.best_output = accepted
    ps.timings_s = _PageStageClock().finalize()

    pipeline._flush_page_fragment(state, 1, accepted.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

    sidecar_path = next(out_dir.rglob("pages/00001.json"))
    baseline_meta = json.loads(sidecar_path.read_text())
    mutated_meta = mutate(baseline_meta)
    sidecar_path.write_text(json.dumps(mutated_meta))

    fresh_state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    resumed = pipeline._load_terminal_page(fresh_state, 1, out_dir)

    reference_state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    sidecar_path.write_text(json.dumps(baseline_meta))
    reference = pipeline._load_terminal_page(reference_state, 1, out_dir)

    assert (resumed is None) == (reference is None), (
        f"timings_s presence/shape changed the resume verdict: "
        f"mutated={resumed!r} baseline={reference!r}"
    )
    if resumed is not None and reference is not None:
        assert resumed.text == reference.text
        assert resumed.status == reference.status
        assert resumed.audit_passed == reference.audit_passed


def test_restore_round_trips_timings_s(tmp_path: Path) -> None:
    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent

    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    accepted = PageOutput(
        page_num=1,
        text="accepted body text long enough to look like real content here.",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    ps.attempts.append(accepted)
    ps.best_output = accepted
    recorded = {
        "route": 0.1,
        "extract": 1.2,
        "tables": 0.3,
        "ladder": 0.4,
        "adjudication": 0.05,
        "figures": 0.0,
        "equations": 0.0,
        "flush": 0.02,
        "total": 2.07,
    }
    ps.timings_s = recorded
    pipeline._flush_page_fragment(state, 1, accepted.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

    restored_state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    page_out = pipeline._load_terminal_page(restored_state, 1, out_dir)
    assert page_out is not None
    pipeline._restore_terminal_page_state(restored_state, 1, page_out, out_dir)
    assert restored_state.pages[1].timings_s["extract"] == pytest.approx(1.2)
    assert restored_state.pages[1].timings_s["total"] == pytest.approx(2.07)

    # A field not restored is dropped on the assemble re-flush.
    pipeline._flush_page_sidecar(restored_state, 1, out_dir, terminal=True)
    again = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
    assert again["timings_s"]["extract"] == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# (c) final .md byte-identical with timings on and off
# ---------------------------------------------------------------------------


def test_final_md_is_byte_identical_with_timings_on_and_off(tmp_path: Path) -> None:
    pdf_path = _real_pdf(tmp_path)

    on_dir = tmp_path / "on"
    off_dir = tmp_path / "off"
    pipeline_on = _make_pipeline()
    pipeline_on.bd_detector = MagicMock()
    pipeline_on.bd_detector.detect.return_value = _bd_assessment(pdf_path, 1, born_digital=set())
    result_on = _process(pipeline_on, pdf_path, on_dir)

    pipeline_off = _make_pipeline()
    pipeline_off.bd_detector = MagicMock()
    pipeline_off.bd_detector.detect.return_value = _bd_assessment(pdf_path, 1, born_digital=set())
    with patch("socr.pipeline.orchestrator._PageStageClock", _NullClock):
        result_off = _process(pipeline_off, pdf_path, off_dir)

    assert result_on.markdown == result_off.markdown
    assert "timings_s" in _sidecar(on_dir)
    assert "timings_s" not in _sidecar(off_dir)
    frag = _fragment(on_dir)
    assert "timings_s" not in frag
    assert "adjudication=" not in frag


def test_timings_never_enter_fingerprint_or_canonical_texts(tmp_path: Path) -> None:
    pdf_path = _real_pdf(tmp_path)
    pipeline = _make_pipeline()
    before = pipeline._run_fingerprint()

    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    state.pages[1].timings_s = _PageStageClock().finalize()
    after = pipeline._run_fingerprint()
    assert before == after

    from socr.core.manifest import canonical_page_texts

    state.pages[1].best_output = PageOutput(
        page_num=1,
        text="body text only",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    texts = canonical_page_texts(state)
    joined = "\n".join(texts)
    assert "timings_s" not in joined
    assert "route=" not in joined


def test_manifest_rolls_up_page_timings(tmp_path: Path) -> None:
    pdf_path = _real_pdf(tmp_path, page_count=2)
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    for n in (1, 2):
        ps = state.pages[n]
        ps.best_output = PageOutput(
            page_num=n,
            text=f"page {n} body",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        ps.attempts = [ps.best_output]
        ps.timings_s = {
            "route": 0.1 * n,
            "extract": 1.0 * n,
            "tables": 0.0,
            "ladder": 0.0,
            "adjudication": 0.0,
            "figures": 0.0,
            "equations": 0.0,
            "flush": 0.01,
            "total": 0.1 * n + 1.0 * n + 0.01,
        }

    blobs = BlobStore(tmp_path / "blobs")
    manifest = build_manifest(state, blobs)
    rolled = manifest.timings_s
    assert rolled is not None
    assert rolled["extract"] == pytest.approx(3.0)
    assert rolled["route"] == pytest.approx(0.3)
    assert rolled["total"] == pytest.approx(
        sum(ps.timings_s["total"] for ps in state.pages.values())
    )
    assert rollup_page_timings(state) == rolled

    payload = manifest.to_dict()
    assert "timings_s" in payload
    reloaded = type(manifest).from_dict(payload)
    assert reloaded.timings_s["extract"] == pytest.approx(3.0)

    line = format_timings_summary_line(rolled)
    assert line.startswith(" | ")
    assert "extract=" in line
    assert "total=" in line


def test_old_manifest_without_timings_still_loads() -> None:
    from socr.core.manifest import Manifest

    old = {
        "schema_version": "1",
        "pdf_filename": "doc.pdf",
        "pdf_file_hash": "h",
        "page_count": 1,
        "render_dpi": 200,
        "entries": {
            "1": {
                "page_num": 1,
                "blob_ref": "deadbeef",
                "fingerprint": {
                    "pdf_file_hash": "h",
                    "page_num": 1,
                    "render_dpi": 200,
                    "engine": "native",
                    "model_version": "",
                    "image_hash": "",
                    "prompt_hash": "",
                },
                "journal": [],
            }
        },
    }
    loaded = Manifest.from_dict(old)
    assert loaded.timings_s is None
