"""Number-free figure-crop descriptions (optional enrichment, ON by default).

Contract pinned here:

* a description never contains a digit or a spelled-out number (the validator is the only
  gate; one retry; then the description is DROPPED, never shipped);
* only genuine figure CROPS are described -- by asset kind, never by pixel size. Whole-page
  images (``chart_page_N``, scanned figure pages, failed-table pages) are never described;
* it is non-authoritative: labelled in the text, recorded as events, and it cannot move page
  status, document status or the metadata record;
* off by ``--no-figure-descriptions`` and under ``--native-only``; a figure-free document is
  byte-identical;
* resume reproduces the same bytes without calling the model again.

Hermetic: the model call is stubbed (``_figure_description_ask``; conftest already stubs it
to "absent" for every other test), ``_available_engines_for_agentic`` and
``_resolve_judge_model`` are patched in the end-to-end tests. Outcomes are compared as
DIFFERENCES between two runs in one process that change one thing (CLAUDE.md, #257).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from ocr_output_contract import assemble_pages, split_native_pages

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.state import DocumentState
from socr.figures import crop_descriptions as cd
from socr.pipeline.orchestrator import UnifiedPipeline

CLEAN = "A line chart of a series against time, comparing named groups."
WITH_DIGIT = "A line chart of the 2008 series against time."
WITH_WORD = "A line chart with three series against time."


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "values in 2008",
        "a 5% change",
        "series 1 and series B2",
        "½ of the sample",  # vulgar fraction
        "٣ series",  # Arabic-Indic digit
        "three series",
        "Two panels",
        "a fifty percent drop",
        "hundreds of points",
        "double the width",
        "half-yearly",
    ],
)
def test_validator_rejects_digits_and_spelled_numbers(text: str) -> None:
    assert cd.find_number_tokens(text), text


@pytest.mark.parametrize(
    "text",
    [
        CLEAN,
        "One of the groups is shaded.",  # 'one' is a pronoun, deliberately allowed
        "Axes show time and the policy rate for the euro area.",
    ],
)
def test_validator_accepts_prose(text: str) -> None:
    assert cd.find_number_tokens(text) == []


@pytest.mark.parametrize("text", ["has | pipe", "has `code`", "has ![x](y)", "a\\b", "", None])
def test_clean_description_refuses_markup_that_could_break_the_page(text) -> None:
    assert cd.clean_description(text) is None


def test_clean_description_flattens_to_one_line() -> None:
    assert cd.clean_description("a line\n\nchart  of   x") == "a line chart of x"


# ---------------------------------------------------------------------------
# Retry policy: pass, digit then retry pass, digit twice then dropped
# ---------------------------------------------------------------------------


def _ask_from(answers: list):
    calls: list[str] = []

    def ask(prompt: str):
        calls.append(prompt)
        return answers[len(calls) - 1] if len(calls) <= len(answers) else answers[-1]

    return ask, calls


def test_first_answer_passes() -> None:
    ask, calls = _ask_from([CLEAN])
    out = cd.describe_number_free(ask)
    assert (out.text, out.retried, out.status) == (CLEAN, False, "described")
    assert len(calls) == 1


def test_a_digit_then_a_clean_retry_ships_the_retry() -> None:
    ask, calls = _ask_from([WITH_DIGIT, CLEAN])
    out = cd.describe_number_free(ask)
    assert (out.text, out.retried, out.status) == (CLEAN, True, "described")
    assert len(calls) == 2
    assert "2" in calls[1] and cd.RETRY_SUFFIX.split("{")[0].strip() in calls[1]


def test_a_digit_twice_is_dropped_and_the_text_never_returned() -> None:
    ask, calls = _ask_from([WITH_DIGIT, WITH_WORD])
    out = cd.describe_number_free(ask)
    assert out.text is None and out.status == "dropped" and out.retried
    assert out.reason == "number_in_description"
    assert len(calls) == 2, "exactly one retry"
    assert "three" in out.violations


def test_model_unavailable_is_dropped_without_a_retry() -> None:
    ask, calls = _ask_from([None])
    out = cd.describe_number_free(ask)
    assert out.text is None and out.reason == "model_unavailable" and not out.retried
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Asset kind: crops only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target,expected",
    [
        ("figures/chart_region_p3_1.png", True),
        ("figures/figure_12_page9.png", True),
        ("chart_region_p10_2.png", True),
        ("figures/chart_page_9.png", False),
        ("figures/scanned_figure_page_4.png", False),
        ("figures/failed_table_p6.png", False),
        ("figures/page_image_7.png", False),
        ("figures/anything_else.png", False),
        ("https://example.org/figure_1_page1.png.exe", False),
    ],
)
def test_only_crop_assets_are_describable(target: str, expected: bool) -> None:
    assert cd.is_crop_asset(target) is expected


def test_insert_never_calls_the_model_for_page_sized_assets() -> None:
    body = "Text\n\n![Chart page 4](figures/chart_page_4.png)\n\n![x](figures/scanned_figure_page_4.png)"
    calls: list[str] = []
    out = cd.insert_descriptions(body, lambda t: calls.append(t) or CLEAN)
    assert out == body and calls == []


def test_insert_places_the_description_directly_under_the_crop_ref() -> None:
    body = "Before\n![Chart region 1](figures/chart_region_p1_1.png)\nAfter"
    out = cd.insert_descriptions(body, lambda t: CLEAN)
    assert out == (
        "Before\n![Chart region 1](figures/chart_region_p1_1.png)\n\n"
        f"{cd.DESCRIPTION_PREFIX} {CLEAN}\n\nAfter"
    )
    assert "model-generated, non-authoritative gist, no values" in out


def test_insert_is_idempotent_and_does_not_recall_the_model() -> None:
    body = "![c](figures/chart_region_p1_1.png)"
    once = cd.insert_descriptions(body, lambda t: CLEAN)
    calls: list[str] = []
    twice = cd.insert_descriptions(once, lambda t: calls.append(t) or "DIFFERENT")
    assert twice == once and calls == []


def test_insert_ignores_refs_in_code_fences_and_table_rows() -> None:
    body = "```\n![c](figures/chart_region_p1_1.png)\n```\n| ![c](figures/chart_region_p1_2.png) |"
    calls: list[str] = []
    assert cd.insert_descriptions(body, lambda t: calls.append(t) or CLEAN) == body
    assert calls == []


def test_insert_leaves_a_body_without_refs_byte_identical() -> None:
    body = "Plain page.\n\n| a | b |\n| - | - |\n| 1 | 2 |"
    assert cd.insert_descriptions(body, lambda t: pytest.fail("called")) is body


def test_a_ref_whose_description_is_dropped_is_left_untouched() -> None:
    body = "![c](figures/chart_region_p1_1.png)\nnext"
    assert cd.insert_descriptions(body, lambda t: None) == body


# ---------------------------------------------------------------------------
# Pipeline: _describe_crop_refs with a stubbed describer
# ---------------------------------------------------------------------------


def _handle(pdf: Path, pages: int) -> DocumentHandle:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        return DocumentHandle(path=pdf, page_count=pages)


def _pipeline(**overrides) -> UnifiedPipeline:
    cfg = dict(
        primary_engine=EngineType.QWEN,
        local_engine=EngineType.QWEN,
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        write_manifest=False,
        judge_backend="heuristic",
    )
    cfg.update(overrides)
    return UnifiedPipeline(PipelineConfig(**cfg))


class _Model:
    """Stub of the model call; records every (image name, prompt)."""

    def __init__(self, answers):
        self.answers = answers
        self.calls: list[tuple[str, str]] = []
        self.model = "stub-model"

    def install(self, monkeypatch) -> None:
        def factory(pipeline_self):
            def ask(path, prompt):
                self.calls.append((Path(path).name, prompt))
                a = self.answers
                return a(path, prompt) if callable(a) else a.pop(0) if len(a) > 1 else a[0]

            ask.model = self.model
            return ask

        monkeypatch.setattr(UnifiedPipeline, "_figure_description_ask", factory)


def _doc(tmp_path: Path, *, with_assets=("chart_region_p1_1.png",)):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    pipe = _pipeline()
    state = DocumentState(handle=_handle(pdf, 2))
    doc_dir, figures = pipe._doc_and_figures_dir(pdf, tmp_path / "out")
    figures.mkdir(parents=True)
    for name in with_assets:
        (figures / name).write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode())
    return pipe, state, tmp_path / "out", doc_dir, pdf


def _text(*refs: str) -> str:
    body1 = "Page one prose.\n\n" + "\n\n".join(f"![r](figures/{r})" for r in refs)
    return assemble_pages([body1, "Page two prose."], page_numbers=[1, 2])


def _events(state, kind: str):
    return [e for e in state.events if e.kind == kind]


def test_described_event_text_and_marker(tmp_path, monkeypatch) -> None:
    pipe, state, out, _doc_dir, _ = _doc(tmp_path)
    _Model([CLEAN]).install(monkeypatch)
    res = pipe._describe_crop_refs(state, out, _text("chart_region_p1_1.png"))
    page1, page2 = split_native_pages(res)
    assert f"{cd.DESCRIPTION_PREFIX} {CLEAN}" in page1
    assert page2 == "Page two prose."
    ev = _events(state, "figure_description_described")
    assert len(ev) == 1 and ev[0].page_num == 1
    assert not _events(state, "figure_description_retried")
    assert not _events(state, "figure_description_dropped")


def test_digit_then_retry_records_retried_and_described(tmp_path, monkeypatch) -> None:
    pipe, state, out, *_ = _doc(tmp_path)
    model = _Model([WITH_DIGIT, CLEAN])
    model.install(monkeypatch)
    res = pipe._describe_crop_refs(state, out, _text("chart_region_p1_1.png"))
    assert CLEAN in res and WITH_DIGIT not in res
    assert len(model.calls) == 2
    assert len(_events(state, "figure_description_retried")) == 1
    assert len(_events(state, "figure_description_described")) == 1
    assert not _events(state, "figure_description_dropped")


def test_digit_twice_is_dropped_and_no_digit_ever_ships(tmp_path, monkeypatch) -> None:
    pipe, state, out, *_ = _doc(tmp_path)
    text = _text("chart_region_p1_1.png")
    model = _Model([WITH_DIGIT, WITH_WORD])
    model.install(monkeypatch)
    res = pipe._describe_crop_refs(state, out, text)
    assert res == text, "a dropped description leaves the text byte-identical"
    assert len(model.calls) == 2
    (ev,) = _events(state, "figure_description_dropped")
    assert ev.data["reason"] == "number_in_description"
    # The event names the offending TOKENS, never the rejected sentence.
    assert WITH_DIGIT not in json.dumps(ev.data) and WITH_WORD not in json.dumps(ev.data)
    assert len(_events(state, "figure_description_retried")) == 1


def test_shipped_description_lines_never_contain_a_number(tmp_path, monkeypatch) -> None:
    pipe, state, out, *_ = _doc(
        tmp_path, with_assets=("chart_region_p1_1.png", "chart_region_p1_2.png")
    )
    answers = {"chart_region_p1_1.png": [CLEAN], "chart_region_p1_2.png": [WITH_DIGIT, WITH_WORD]}

    def per_image(path, prompt):
        return answers[Path(path).name].pop(0)

    _Model(per_image).install(monkeypatch)
    res = pipe._describe_crop_refs(
        state, out, _text("chart_region_p1_1.png", "chart_region_p1_2.png")
    )
    lines = [ln for ln in res.split("\n") if ln.startswith(cd.DESCRIPTION_PREFIX)]
    assert len(lines) == 1
    assert cd.find_number_tokens(lines[0].removeprefix(cd.DESCRIPTION_PREFIX)) == []


def test_page_sized_assets_never_reach_the_model(tmp_path, monkeypatch) -> None:
    pipe, state, out, *_ = _doc(
        tmp_path,
        with_assets=("chart_page_1.png", "scanned_figure_page_1.png", "failed_table_p1.png"),
    )
    model = _Model([CLEAN])
    model.install(monkeypatch)
    text = _text("chart_page_1.png", "scanned_figure_page_1.png", "failed_table_p1.png")
    assert pipe._describe_crop_refs(state, out, text) == text
    assert model.calls == []
    assert not any(e.kind.startswith("figure_description") for e in state.events)


def test_figure_free_document_is_byte_identical_and_never_asks(tmp_path, monkeypatch) -> None:
    pipe, state, out, *_ = _doc(tmp_path)
    model = _Model([CLEAN])
    model.install(monkeypatch)
    text = assemble_pages(
        ["No figure here.", "| a | b |\n| - | - |\n| x | y |"], page_numbers=[1, 2]
    )
    assert pipe._describe_crop_refs(state, out, text) is text
    assert model.calls == []


def test_cache_reproduces_the_same_bytes_without_calling_the_model(tmp_path, monkeypatch) -> None:
    pipe, state, out, doc_dir, _ = _doc(tmp_path)
    model = _Model([CLEAN])
    model.install(monkeypatch)
    text = _text("chart_region_p1_1.png")
    first = pipe._describe_crop_refs(state, out, text)
    assert (doc_dir / "figures" / "figure_descriptions.json").exists()
    n = len(model.calls)

    # A regenerated page (resume re-ran the figure phase over a fresh body):
    second = pipe._describe_crop_refs(state, out, text)
    assert second == first and len(model.calls) == n

    # A page restored from its terminal fragment already carries the description:
    third = pipe._describe_crop_refs(state, out, first)
    assert third == first and len(model.calls) == n


def test_a_dropped_outcome_is_cached_but_model_unavailable_is_not(tmp_path, monkeypatch) -> None:
    pipe, state, out, doc_dir, _ = _doc(tmp_path)
    text = _text("chart_region_p1_1.png")
    absent = _Model([None])
    absent.install(monkeypatch)
    pipe._describe_crop_refs(state, out, text)
    cache_file = doc_dir / "figures" / "figure_descriptions.json"
    assert not cache_file.exists(), "an unreachable model must not be remembered as a verdict"

    # Later run, model up: the figure IS described.
    _Model([CLEAN]).install(monkeypatch)
    assert CLEAN in pipe._describe_crop_refs(state, out, text)


def test_a_tampered_cache_entry_with_a_digit_is_not_shipped(tmp_path, monkeypatch) -> None:
    pipe, state, out, doc_dir, _ = _doc(tmp_path)
    text = _text("chart_region_p1_1.png")
    _Model([CLEAN]).install(monkeypatch)
    pipe._describe_crop_refs(state, out, text)
    cache_file = doc_dir / "figures" / "figure_descriptions.json"
    cache = json.loads(cache_file.read_text())
    for entry in cache.values():
        entry["text"] = WITH_DIGIT
    cache_file.write_text(json.dumps(cache))
    res = pipe._describe_crop_refs(state, out, text)
    assert res == text


@pytest.mark.parametrize(
    "overrides,enabled",
    [
        ({}, True),
        ({"describe_figure_crops": False}, False),
        ({"native_only": True}, False),
        ({"strict_local": True}, True),  # local model: strict-local needs no extra branch
    ],
)
def test_gate(overrides, enabled) -> None:
    assert _pipeline(**overrides)._crop_descriptions_enabled() is enabled


def test_the_fingerprint_moves_with_the_effective_state() -> None:
    from socr.pipeline import orchestrator as orch

    def fp(**o):
        orch._SOURCE_DIGEST_CACHE = None
        with patch.object(orch, "_socr_source_digest", lambda: "pinned"):
            return _pipeline(**o)._run_fingerprint()

    on, off, native = fp(), fp(describe_figure_crops=False), fp(native_only=True)
    assert on != off
    assert fp(native_only=True, describe_figure_crops=False) != on
    # native-only never describes, so the flag cannot matter there.
    assert native == fp(native_only=True, describe_figure_crops=False)


# ---------------------------------------------------------------------------
# End to end through process(): enabled vs disabled differ only by the description line
# ---------------------------------------------------------------------------


def _mixed_pdf(tmp_path: Path) -> Path:
    import test_gh189_mixed_chart_preservation as g

    return g._make_mixed_chart_table_pdf(tmp_path)


class _Engine:
    name = "qwen"

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        from socr.core.result import PageOutput, PageStatus

        import test_gh189_mixed_chart_preservation as g

        return [
            PageOutput(page_num=n, text=g.WINNER_TEXT, status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


class _Judge:
    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=True, reason="")


def _run_process(tmp_path: Path, monkeypatch, tag: str, model: _Model, **overrides):
    from socr.pipeline import orchestrator as orch

    (tmp_path / tag).mkdir()
    pdf = _mixed_pdf(tmp_path / tag)
    out = tmp_path / f"out-{tag}"
    model.install(monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _Engine())
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
                table_judge_ladder=False,
                **overrides,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._build_page_judge = lambda state: _Judge()
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        result = pipe.process(pdf, output_dir=out)
    final = next(iter(out.rglob("mixed_chart_table.md")))
    frag = next(iter(out.rglob("pages/00001.md"))).read_text()
    side = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    meta = json.loads(next(iter(out.rglob("metadata.json"))).read_text())
    return pipe, result, final.read_text(), frag, side, meta, out


def _strip_descriptions(text: str) -> str:
    return re.sub(rf"\n\n{re.escape(cd.DESCRIPTION_PREFIX)}[^\n]*", "", text)


def test_e2e_enabled_vs_disabled_differ_only_by_description_lines(tmp_path, monkeypatch) -> None:
    on_model, off_model = _Model([CLEAN]), _Model([CLEAN])
    _, res_on, md_on, frag_on, side_on, meta_on, _ = _run_process(
        tmp_path, monkeypatch, "on", on_model
    )
    _, res_off, md_off, frag_off, side_off, meta_off, _ = _run_process(
        tmp_path, monkeypatch, "off", off_model, describe_figure_crops=False
    )
    assert off_model.calls == [], "disabled by flag: the model is never called"
    assert len(on_model.calls) == 2, "two chart regions on the page, one call each"

    # The descriptions are in the final .md AND in the authoritative fragment.
    assert md_on.count(cd.DESCRIPTION_PREFIX) == 2 and frag_on.count(cd.DESCRIPTION_PREFIX) == 2
    assert cd.DESCRIPTION_PREFIX not in md_off and cd.DESCRIPTION_PREFIX not in frag_off
    # Everything else is byte-identical.
    assert _strip_descriptions(md_on) == md_off
    assert _strip_descriptions(frag_on) == frag_off

    # Non-authoritative: status, audit verdict and table counts do not move.
    assert res_on.status == res_off.status
    assert res_on.audit_passed == res_off.audit_passed
    assert side_on["status"] == side_off["status"]
    assert meta_on["files"] and next(iter(meta_on["files"].values())).get("status") == next(
        iter(meta_off["files"].values())
    ).get("status")
    assert side_on.get("table_counts") == side_off.get("table_counts")


def test_e2e_native_only_produces_no_descriptions(tmp_path, monkeypatch) -> None:
    model = _Model([CLEAN])
    _, _, md, frag, *_ = _run_process(tmp_path, monkeypatch, "native", model, native_only=True)
    assert model.calls == []
    assert cd.DESCRIPTION_PREFIX not in md and cd.DESCRIPTION_PREFIX not in frag


def test_e2e_a_dropped_description_leaves_the_output_identical_to_disabled(
    tmp_path, monkeypatch
) -> None:
    always_digit = _Model([WITH_DIGIT, WITH_WORD])
    always_digit.answers = lambda path, prompt: WITH_DIGIT
    _, _, md_drop, frag_drop, *_ = _run_process(tmp_path, monkeypatch, "drop", always_digit)
    _, _, md_off, frag_off, *_ = _run_process(
        tmp_path, monkeypatch, "off2", _Model([CLEAN]), describe_figure_crops=False
    )
    assert md_drop == md_off and frag_drop == frag_off
    assert len(always_digit.calls) == 4, "two figures, each asked twice"


def test_e2e_resume_reproduces_the_descriptions_without_the_model(tmp_path, monkeypatch) -> None:
    first = _Model([CLEAN])
    _, _, md1, _, _, _, out = _run_process(tmp_path, monkeypatch, "res", first)
    assert md1.count(cd.DESCRIPTION_PREFIX) == 2

    # Second run over the SAME output dir with a model that would answer differently.
    second = _Model(["A bar chart of categories, comparing groups."])
    second.install(monkeypatch)
    from socr.pipeline import orchestrator as orch

    pdf = tmp_path / "res" / "mixed_chart_table.pdf"
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _Engine())
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
                table_judge_ladder=False,
                reprocess=True,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._build_page_judge = lambda state: _Judge()
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        pipe.process(pdf, output_dir=out)
    md2 = next(iter(out.rglob("mixed_chart_table.md")))
    assert second.calls == [], "the cache answers; the model is not consulted again"
    assert md2.read_text().count(CLEAN) == 2
