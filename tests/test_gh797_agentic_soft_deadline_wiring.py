"""GH-797: pin the per-provider CLI soft deadline at the SHIPPING boundary.

GH-172's fix bounds the child OCR CLI by the SAME per-provider soft deadline
``route_page``'s wrapper timeout uses, instead of leaving it at the whole-document
``config.timeout`` (1800s). Two hops carry it:

1. ``_phase_agentic``'s ``run_provider`` closure passes
   ``subprocess_timeout_sec=provider_timeout.get(profile.engine)`` into
   ``_run_engine_on_pages`` (``orchestrator.py`` ~8551-8561);
2. ``_run_engine_on_pages`` forwards it as ``subprocess_timeout=`` to
   ``engine.process_pages`` (~2065).

``tests/test_gh172_cli_subprocess_timeout.py`` drives ``BaseEngine.process_pages``
with an explicit override and ``subprocess.run`` stubbed. That pins
``engines/base.py`` *honours* the keyword; it does not pin either hop above.
Deleting the ``subprocess_timeout_sec=`` argument from ``run_provider`` left the
whole suite green while a wedged CLI again survived to the document timeout after
``route_page`` had abandoned its wrapper thread (issue #797, leftover from #796).

**Pin a DIFFERENCE, not a value.** Per CLAUDE.md, provider-dependent machinery does
not fire in CI, so an absolute number measured on one machine is not a safe pin (a
pinned tuple reverted PR #253; see #257). Every test here runs the same production
path two or three times in one process, changing ONLY the configured per-provider
timeout, and asserts the value observed downstream TRACKS it. With either hop
deleted, every leg observes ``None`` -- the observations collapse to one value and
the test fails.

**Hermetic.** No ollama, no network, no subprocess, no real engine: the provider
ladder, the judge model, the crop-VLM probe and ``get_engine`` are all patched, and
the spy engine never renders or spawns anything. Nothing here asserts a page status,
``audit_passed`` or a document status, so the CI-vs-workstation provider divergence
cannot reach these assertions.

**Out of scope** (owned by open #172): ``route_page`` / ``_escalate_table_page`` /
``_read_with_deadline`` are still ThreadPool-based, and no child CLI is executed.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.providers import PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.pipeline.agentic import AcceptDecision, DEFAULT_PROVIDER_TIMEOUTS  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

# Arbitrary synthetic sentinels -- NOT tuned thresholds and not calibrated from any
# measurement. Their only requirements are that they are distinct from each other
# and from the calibrated default for this engine (asserted in ``_setup_sentinels``),
# so that "the observed deadline tracks the configured one" is falsifiable.
_SOFT_DEADLINE_A = 11.0
_SOFT_DEADLINE_B = 23.0

_ENGINE = EngineType.QWEN
_DEFAULT_DEADLINE = DEFAULT_PROVIDER_TIMEOUTS[_ENGINE]


def _setup_sentinels() -> None:
    """Setup guard: the three legs must be mutually distinguishable at all."""
    assert len({_SOFT_DEADLINE_A, _SOFT_DEADLINE_B, _DEFAULT_DEADLINE}) == 3, (
        "setup: the two synthetic sentinels must differ from each other and from "
        f"the calibrated default {_DEFAULT_DEADLINE!r}, or the legs cannot be told apart"
    )


class _AcceptingJudge:
    """Accepts the first rung, so exactly one provider call happens per page."""

    def assess(self, output: Any, provider: Any) -> AcceptDecision:
        return AcceptDecision(accept=True, reason="stub accepts all")


class _SpyEngine:
    """Records the ``subprocess_timeout`` each ``process_pages`` call receives.

    Never renders, never spawns: this is the seam the real CLI sits behind, so
    stubbing it keeps the test hermetic while still exercising both production hops.
    """

    name = "qwen"

    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    def is_available(self) -> bool:
        return True

    def process_pages(
        self,
        pdf_path: Any,
        page_nums: list[int],
        config: Any,
        dpi: Any,
        subprocess_timeout: float | None = None,
        **_kwargs: Any,
    ) -> list[PageOutput]:
        self.timeouts.append(subprocess_timeout)
        return [
            PageOutput(page_num=n, text=f"ocr text {n}", status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


class _StubHandle:
    path = pathlib.Path("/nonexistent/doc.pdf")


class _StubPage:
    native_text = ""
    has_tables = False
    native_table_structure_failed = False


class _StubState:
    def __init__(self) -> None:
        self.handle = _StubHandle()
        self.pages = {1: _StubPage()}


def _fixture_pdf(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    """A real, on-disk born-digital page with genuine inserted text."""
    pdf = tmp_path / f"{name}.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 80
    for _ in range(14):
        page.insert_text((60, y), "Estimated coefficient 0.082 significant", fontsize=9)
        y += 16
    doc.save(str(pdf))
    doc.close()
    return pdf


def _hermetic_pipeline(spy: _SpyEngine, monkeypatch: pytest.MonkeyPatch) -> UnifiedPipeline:
    """An agentic pipeline with every provider-dependent probe pinned.

    CI has no ollama and no provider: without these patches the ladder is empty and
    ``_phase_agentic`` bails before routing (so the spy would never be called and the
    test would pass vacuously), and ``_phase_judge_hard_pages`` would build a real
    ``OllamaVisionJudge`` and POST to it regardless of ``judge_backend``.

    ``primary_engine`` is pinned for the same reason: the default ``AUTO`` makes
    ``process()`` call ``resolve_auto_engine()``, which instantiates engines from the
    registry directly (NOT through the patched ``get_engine``) and shells out to the
    ``ollama`` CLI to probe them. That probe is the one live external dependency this
    file would otherwise keep -- measured at ~6s per ``process()`` call against an
    unreachable ollama host, and it is exactly the kind of ambient provider state
    CLAUDE.md warns diverges between this machine and CI.
    """
    from socr.pipeline import orchestrator as orch

    monkeypatch.setattr(orch, "get_engine", lambda engine_type: spy)

    pipe = UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            quiet=True,
            primary_engine=_ENGINE,
            local_engine=_ENGINE,
            enabled_engines=[_ENGINE],
            write_manifest=False,
            judge_backend="heuristic",
            dual_pass_tables=False,
            detect_equations=False,
            save_figures=False,
        )
    )
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
    pipe._build_page_judge = lambda state: _AcceptingJudge()
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._resolve_judge_model = lambda *a, **k: ""

    # A clean born-digital prose page takes the trusted-native bypass and never
    # reaches the ladder, which would make every leg observe nothing at all. Force
    # the page to need OCR so ``run_provider`` actually runs.
    _detect = pipe.bd_detector.detect

    def _needs_ocr(path):
        assessment = _detect(path)
        assessment.pages[0].needs_ocr_enhancement = True
        return assessment

    pipe.bd_detector.detect = _needs_ocr
    return pipe


def _deadlines_reaching_the_engine(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    leg: str,
    provider_timeout: dict[EngineType, float] | None,
) -> list[float | None]:
    """Run the real agentic loop once and report what the engine seam was handed.

    ``provider_timeout=None`` leaves ``agentic_provider_timeout`` unset, which is the
    shipping default: ``_phase_agentic`` then falls back to
    ``DEFAULT_PROVIDER_TIMEOUTS``.
    """
    spy = _SpyEngine()
    pipe = _hermetic_pipeline(spy, monkeypatch)
    if provider_timeout is not None:
        # Not a declared ``PipelineConfig`` field -- the orchestrator reads it with
        # ``getattr(self.config, "agentic_provider_timeout", None)``, so this is the
        # only way to exercise the override arm (see the module note in the test).
        pipe.config.agentic_provider_timeout = provider_timeout

    pipe.process(_fixture_pdf(tmp_path, leg), output_dir=tmp_path / f"out-{leg}")

    assert spy.timeouts, f"leg {leg}: the agentic loop never reached the engine"
    return spy.timeouts


def test_the_configured_soft_deadline_tracks_through_to_the_engine(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard: three legs of the REAL agentic loop, differing only in the
    configured per-provider timeout, must hand the CLI seam three different bounds.

    Delete ``subprocess_timeout_sec=`` from ``run_provider`` (or the
    ``subprocess_timeout=`` forward in ``_run_engine_on_pages``) and all three legs
    observe ``None``: the three observations collapse to one and this fails.
    """
    _setup_sentinels()

    seen_a = _deadlines_reaching_the_engine(tmp_path, monkeypatch, "a", {_ENGINE: _SOFT_DEADLINE_A})
    seen_b = _deadlines_reaching_the_engine(tmp_path, monkeypatch, "b", {_ENGINE: _SOFT_DEADLINE_B})
    seen_default = _deadlines_reaching_the_engine(tmp_path, monkeypatch, "default", None)

    # The DIFFERENCE, stated without pinning any absolute outcome of the run: change
    # only the configured deadline and the bound the CLI seam receives changes with it.
    observed = {seen_a[0], seen_b[0], seen_default[0]}
    assert len(observed) == 3, (
        "the per-provider soft deadline does not reach the CLI seam: three legs that "
        f"differ only in the configured timeout produced {observed!r}. A single-valued "
        "set (typically {None}) means the wiring between _phase_agentic and "
        "engine.process_pages has been dropped"
    )

    # ...and each leg tracks its OWN configuration, not merely some other leg's.
    assert set(seen_a) == {_SOFT_DEADLINE_A}
    assert set(seen_b) == {_SOFT_DEADLINE_B}
    # The unconfigured leg falls back to the calibrated default for this engine,
    # sourced from the registry rather than repeated as a literal here.
    assert set(seen_default) == {_DEFAULT_DEADLINE}


def test_the_agentic_loop_passes_the_deadline_into_the_engine_runner(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Localises the first hop: ``run_provider`` -> ``_run_engine_on_pages``.

    Same DIFFERENCE, observed one level higher. If this fails while the end-to-end
    test above also fails, the closure argument is the hop that was dropped; if only
    the one above fails, the ``subprocess_timeout=`` forward inside
    ``_run_engine_on_pages`` is.
    """
    _setup_sentinels()

    def _observe(leg: str, provider_timeout: dict[EngineType, float] | None) -> list[Any]:
        seen: list[Any] = []
        spy = _SpyEngine()
        pipe = _hermetic_pipeline(spy, monkeypatch)
        if provider_timeout is not None:
            pipe.config.agentic_provider_timeout = provider_timeout

        def _spy_runner(state, nums, nat, eng, phase, profile=None, **kwargs):
            seen.append(kwargs.get("subprocess_timeout_sec"))
            return [
                PageOutput(page_num=p, text=f"text {p}", status=PageStatus.SUCCESS, engine="qwen")
                for p in nums
            ]

        pipe._run_engine_on_pages = _spy_runner
        pipe.process(_fixture_pdf(tmp_path, leg), output_dir=tmp_path / f"out-{leg}")
        assert seen, f"leg {leg}: the agentic loop never called the engine runner"
        return seen

    seen_a = _observe("runner-a", {_ENGINE: _SOFT_DEADLINE_A})
    seen_b = _observe("runner-b", {_ENGINE: _SOFT_DEADLINE_B})
    seen_default = _observe("runner-default", None)

    observed = {seen_a[0], seen_b[0], seen_default[0]}
    assert len(observed) == 3, (
        "_phase_agentic's run_provider closure is not passing subprocess_timeout_sec "
        f"down: three differently-configured legs produced {observed!r}"
    )
    assert set(seen_a) == {_SOFT_DEADLINE_A}
    assert set(seen_b) == {_SOFT_DEADLINE_B}
    assert set(seen_default) == {_DEFAULT_DEADLINE}


def test_a_caller_that_passes_no_deadline_leaves_the_engine_unbounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: the bound comes from the ARGUMENT, not from ambient config.

    The phase-major (non-agentic) callers pass no ``subprocess_timeout_sec``, and must
    keep the pre-GH-172 whole-document behaviour. Without this leg, a hypothetical
    "always send the default" implementation would satisfy the tests above while the
    agentic wiring itself did nothing.
    """
    from socr.pipeline import orchestrator as orch

    spy = _SpyEngine()
    monkeypatch.setattr(orch, "get_engine", lambda engine_type: spy)
    pipe = orch.UnifiedPipeline(PipelineConfig(quiet=True))

    pipe._run_engine_on_pages(_StubState(), [1], [], _ENGINE, "local")
    pipe._run_engine_on_pages(
        _StubState(),
        [1],
        [],
        _ENGINE,
        "agentic",
        profile=PROFILE_QWEN_LOCAL,
        subprocess_timeout_sec=_SOFT_DEADLINE_A,
    )

    unbounded, bounded = spy.timeouts
    assert unbounded is None, "a caller that passes no deadline must not acquire one"
    assert bounded == _SOFT_DEADLINE_A
    assert unbounded != bounded
