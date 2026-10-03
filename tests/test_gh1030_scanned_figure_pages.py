"""#1030: a scanned page whose invisible layer names a figure ships the page image beside its text.

Measurement (``docs/log/2026-10-03_scanned-figures.md``): a caption line in the invisible layer
fired on 17 of 98 raster pages, all of them figure pages, none of the 29 table pages or 50 prose
pages. This file pins the behaviour as DIFFERENCES between two runs in one process that change
one thing (the caption line in the layer), never an absolute outcome measured on one machine
(CLAUDE.md, #257): provider-dependent machinery does not fire in CI.

Hermetic: ``_available_engines_for_agentic`` patched, ``_resolve_judge_model`` -> "".
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import fitz
import pytest

from socr.core import manifest
from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import PageState
from socr.figures.scanned_figures import (
    MIN_SPELLED_RUN,
    SPELLED_FENCE_CLOSE,
    SPELLED_FENCE_OPEN,
    fence_spelled_runs,
    has_figure_caption,
)
from socr.pipeline import orchestrator as orch
from socr.pipeline.orchestrator import UnifiedPipeline

_PROSE = "The estimated effect of the policy change on output is reported below. "
_OCR_MARK = "MODEL-READING-MARK"
_CAPTION = "FIGURE 5.--Market 5-Series BC, Parameter Set I."
_AXIS_TITLE = "Price" + "ABCDEFGH"[: max(0, MIN_SPELLED_RUN - 5)]  # >= MIN_SPELLED_RUN characters


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_QWEN_LOCAL]
    )
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "FIGURE 5.-Market 5-Series BC",
        "Figure 7.--Market 7-Series",
        "Fig. 3: Power under mean correction",
        "FIGURE 2",
        "  Figure 12 — Annual returns",
        "FIGURE A1.--Pilot 1.",
        "Figure 2a: When an analyst",
    ],
)
def test_caption_line_is_a_caption(line: str) -> None:
    assert has_figure_caption(f"some prose\n{line}\nmore prose")


@pytest.mark.parametrize(
    "text",
    [
        "As Figure 5 shows, prices converge.",
        "see\nFigure 5 shows the series",  # a wrapped reference starts the line but continues
        "TABLE I\nBinomial test",
        "",
    ],
)
def test_reference_or_table_is_not_a_caption(text: str) -> None:
    assert not has_figure_caption(text)


# ---------------------------------------------------------------------------
# Fence: separated, never dropped
# ---------------------------------------------------------------------------


def test_fence_keeps_every_character_and_leaves_short_runs_alone() -> None:
    run = "\n".join("Price" + "X" * (MIN_SPELLED_RUN - 5))
    text = f"Intro line\n{run}\nclosing line"
    fenced, n = fence_spelled_runs(text)
    assert n == MIN_SPELLED_RUN
    assert SPELLED_FENCE_OPEN in fenced and SPELLED_FENCE_CLOSE in fenced
    assert fenced.startswith("Intro line\nclosing line")
    inside = fenced.split(SPELLED_FENCE_OPEN, 1)[1].split(SPELLED_FENCE_CLOSE, 1)[0]
    assert "".join(inside.split("\n")[2:]).replace(" ", "") == "P" + "rice" + "X" * (
        MIN_SPELLED_RUN - 5
    )

    short = "\n".join("a" * (MIN_SPELLED_RUN - 1))
    assert fence_spelled_runs(f"xx\n{short}\nyy") == (f"xx\n{short}\nyy", 0)


# ---------------------------------------------------------------------------
# Guard (unit): same output, one flag differs
# ---------------------------------------------------------------------------


def _out(text: str, engine: str = "qwen") -> PageOutput:
    return PageOutput(
        page_num=1, text=text, status=PageStatus.SUCCESS, engine=engine, audit_passed=True
    )


def _p(ref: str = "", **kw) -> PageState:
    p = PageState(page_num=1)
    p.scanned_figure_png_ref = ref
    for k, v in kw.items():
        setattr(p, k, v)
    return p


REF = "![Scanned figure page 1](figures/scanned_figure_page_p1.png)"


def test_guard_difference_flag_on_vs_off() -> None:
    out = _out("Body text.\n\n[Figure 5: caption kept as text]")
    off = manifest._apply_scanned_figure_guard(out, _p(""))
    on = manifest._apply_scanned_figure_guard(out, _p(REF))
    assert off is out
    assert on.text == out.text + "\n\n" + REF, "text is untouched, the ref is appended"
    assert manifest._apply_scanned_figure_guard(on, _p(REF)) is on, "idempotent"
    assert on.status is out.status and on.audit_passed is out.audit_passed


@pytest.mark.parametrize(
    "floor", ["d3_floor_png_ref", "rotated_shred_png_ref", "invisible_scan_png_ref"]
)
def test_guard_abstains_when_the_page_already_ships_its_image(floor: str) -> None:
    out = _out("Body text.")
    page = _p(REF, **{floor: "![x](figures/x.png)"})
    assert manifest._apply_scanned_figure_guard(out, page) is out


def test_guard_abstains_on_a_failure_marker_and_on_empty_text() -> None:
    marker = _out("[page 1 failed: no usable OCR output]")
    assert manifest.is_page_failed_marker(marker.text)
    assert manifest._apply_scanned_figure_guard(marker, _p(REF)) is marker
    empty = _out("   ")
    assert manifest._apply_scanned_figure_guard(empty, _p(REF)) is empty


def test_guard_fences_only_when_the_layer_itself_ships() -> None:
    spelled = "\n".join("Price" + "X" * (MIN_SPELLED_RUN - 5))
    body = f"Caption\n{spelled}"
    layer = manifest._apply_scanned_figure_guard(_out(body, engine="native"), _p(REF))
    model = manifest._apply_scanned_figure_guard(_out(body, engine="qwen"), _p(REF))
    assert SPELLED_FENCE_OPEN in layer.text
    assert SPELLED_FENCE_OPEN not in model.text and spelled in model.text, (
        "a model reading is never rewritten"
    )
    assert layer.text.endswith(REF) and model.text.endswith(REF)


def test_guard_render_failure_demotes_status_only() -> None:
    out = _out("Body text.")
    failed = manifest._apply_scanned_figure_guard(out, _p("", scanned_figure_render_failed=True))
    assert failed.status is PageStatus.WARNING
    assert failed.text == out.text and failed.audit_passed is out.audit_passed
    assert replace(out).status is PageStatus.SUCCESS


# ---------------------------------------------------------------------------
# End to end through process(): the same scan with and without a caption in its layer
# ---------------------------------------------------------------------------


def _scan_pdf(path: Path, *, caption: bool, spelled: bool = False) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200), False)
    pix.set_rect(pix.irect, (235, 235, 235))
    page.insert_image(page.rect, pixmap=pix)
    y = 72
    for _ in range(8):
        page.insert_text((72, y), _PROSE * 2, fontname="helv", fontsize=10, render_mode=3)
        y += 14
    if spelled:
        for ch in _AXIS_TITLE:
            page.insert_text((300, y), ch, fontname="helv", fontsize=10, render_mode=3)
            y += 12
    if caption:
        page.insert_text((72, y + 14), _CAPTION, fontname="helv", fontsize=10, render_mode=3)
    doc.save(path)
    doc.close()
    return path


class _Engine:
    name = "qwen"

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        return [
            PageOutput(page_num=n, text=_OCR_MARK, status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


class _Judge:
    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=True, reason="")


def _run(
    tmp_path, monkeypatch, tag, *, caption, providers, spelled=False, save_figures=True, lane=True
):
    pdf = _scan_pdf(tmp_path / f"{tag}.pdf", caption=caption, spelled=spelled)
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _Engine())
        if not lane:
            m.setattr(UnifiedPipeline, "_agentic_scanned_figure_page", lambda self, *a, **k: None)
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
                save_figures=save_figures,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL] if providers else []
        pipe._build_page_judge = lambda state: _Judge()
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        result = pipe.process(pdf, output_dir=tmp_path / f"out-{tag}")
    out = tmp_path / f"out-{tag}"
    side = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    text = next(iter(out.rglob("pages/00001.md"))).read_text()
    final = next(iter(out.rglob("*.md")))
    audit = json.loads(next(iter(out.rglob("audit_log.json"))).read_text())
    meta = next(
        iter(json.loads(next(iter(out.rglob("metadata.json"))).read_text())["files"].values())
    )
    return result, side, text, audit["events"], meta, out, final


def _images(text: str) -> list[str]:
    import re

    return re.findall(r"!\[[^\]]*\]\(([^)]*)\)", text)


@pytest.mark.parametrize("providers", [True, False], ids=["provider", "no-provider"])
@pytest.mark.parametrize("save_figures", [True, False], ids=["save-figures", "no-save-figures"])
def test_e2e_caption_page_gets_the_image_and_prose_page_does_not(
    tmp_path, monkeypatch, providers, save_figures
) -> None:
    fig = _run(
        tmp_path, monkeypatch, "fig", caption=True, providers=providers, save_figures=save_figures
    )
    pro = _run(
        tmp_path, monkeypatch, "pro", caption=False, providers=providers, save_figures=save_figures
    )
    off = _run(
        tmp_path,
        monkeypatch,
        "off",
        caption=True,
        providers=providers,
        save_figures=save_figures,
        lane=False,
    )
    _, fig_side, fig_text, fig_events, fig_meta, fig_out, fig_final = fig
    _, pro_side, pro_text, pro_events, pro_meta, _, _ = pro

    # The difference: the figure page carries exactly one page image; the prose page none of it.
    fig_imgs = [i for i in _images(fig_text) if "scanned_figure_page" in i]
    assert len(fig_imgs) == 1, fig_imgs
    assert not [i for i in _images(pro_text) if "scanned_figure_page" in i]
    assert (next(iter(fig_out.rglob("figures"))) / Path(fig_imgs[0]).name).is_file()
    assert len(_images(fig_text)) == len(set(_images(fig_text))), "no duplicate image refs"

    # Nothing is taken out of the page: the same scan with the lane switched off ships the same
    # text, minus the ref. (With a model reading the text is the reading; without one it is the
    # layer, and only a spelled run would be fenced -- there is none in this fixture.)
    # The one other difference is the doc-level extractor's full-page "Figure N" block, which a
    # scan that is not classified scanned gets under --save-figures: the lane REPLACES it with its
    # own ref instead of shipping the same raster twice.
    import re

    stripped = fig_text.replace(f"![Scanned figure page 1]({fig_imgs[0]})", "").rstrip()
    off_text = re.sub(
        r"\n*\*\*Figure \d+\*\* \(page \d+\)\n\n!\[Figure \d+\]\([^)]*\)", "", off[2]
    ).rstrip()
    assert stripped == off_text
    assert not [i for i in _images(off[2]) if "scanned_figure_page" in i]
    assert not [i for i in _images(fig_text) if "/figure_" in i], "the raster is not shipped twice"

    # Surfaced at every level, and only for the figure page.
    assert "scanned_figure_asset" in {e["kind"] for e in fig_events}
    assert "scanned_figure_asset" not in {e["kind"] for e in pro_events}
    assert Path(fig_imgs[0]).name in fig_side["scanned_figure_png_ref"]
    assert not pro_side.get("scanned_figure_png_ref")
    assert "scanned figure page" in (fig_meta.get("error") or "")
    assert "scanned figure page" not in (pro_meta.get("error") or "")
    assert fig_imgs[0] in fig_final.read_text(), "the stitched document carries it too"
    # Status is not changed by the asset.
    assert fig_side["status"] == pro_side["status"]


def test_e2e_no_provider_layer_ships_with_its_spelled_axis_title_fenced(
    tmp_path, monkeypatch
) -> None:
    on = _run(tmp_path, monkeypatch, "on", caption=True, spelled=True, providers=False)
    off = _run(tmp_path, monkeypatch, "off", caption=False, spelled=True, providers=False)
    on_text, off_text = on[2], off[2]
    # Same layer, one thing changed (the caption): only the caption page fences the spelled run.
    assert SPELLED_FENCE_OPEN in on_text and SPELLED_FENCE_OPEN not in off_text
    fenced = on_text.split(SPELLED_FENCE_OPEN, 1)[1].split(SPELLED_FENCE_CLOSE, 1)[0]
    kept = [ln for ln in fenced.split("\n")[2:] if ln.strip()]
    assert "".join(ch.strip() for ch in kept) == _AXIS_TITLE, "every character is kept, in order"
    assert _AXIS_TITLE not in on_text.replace(fenced, "")


def test_e2e_resume_does_not_duplicate_the_image(tmp_path, monkeypatch) -> None:
    pdf = _scan_pdf(tmp_path / "r.pdf", caption=True)
    texts = []
    for _ in range(2):
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
                    save_figures=True,
                )
            )
            pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
            pipe._build_page_judge = lambda state: _Judge()
            pipe._resolve_crop_vlm_model = lambda: None
            pipe._resolve_judge_model = lambda *a, **k: ""
            pipe.process(pdf, output_dir=tmp_path / "out-r")
        texts.append(next(iter((tmp_path / "out-r").rglob("pages/00001.md"))).read_text())
    for t in texts:
        assert len([i for i in _images(t) if "scanned_figure_page" in i]) == 1
