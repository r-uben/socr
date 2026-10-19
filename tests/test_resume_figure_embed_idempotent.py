"""A page restored from the resume ledger must not have its figure block embedded twice.

``_describe_and_embed_figures`` appended each page's figure blocks UNCONDITIONALLY. A page
restored from the per-page ledger arrives at assemble with those blocks already in its body
-- the run that first embedded them wrote them into ``pages/NNN.md``, and that fragment is
what assemble reads back -- so every partial resume added a SECOND copy of each block, and
of the description attached to it.

Pre-existing, and measured that way: the test below drives the ORDINARY terminal-SUCCESS
resume gate, with no #1156 page and nothing from #1192 involved. #1192 only widens the set
of pages that reach this code.

This module also holds the shared run/crash/resume harness, which
``test_gh1192_resume_accepted_prior_ocr.py`` imports.

Hermetic per ``CLAUDE.md``: the engine registry, the agentic ladder, the page judge and the
judge model are all pinned, so nothing here reaches ollama or a provider.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import socr
from socr.core.result import FailureMode, PageStatus
from socr.pipeline.orchestrator import UnifiedPipeline

from test_gh1156_part_a_interim import _FixedJudge, _png, _prior_ocr_pdf

__all__ = [
    "_CountingEngine",
    "_FixedJudge",
    "_clean_pdf",
    "_crash_after_page_one",
    "_page_one",
    "_png",
    "_prior_ocr_pdf",
    "_run",
]


def test_loaded_source_is_this_checkout() -> None:
    """The mutation canary, in the house shape: a mutant copy that quietly imports the REAL
    tree proves nothing, and ``"socr" in socr.__file__`` is true of every tree there is."""
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


class _CountingEngine:
    """The stub reading engine, counting the PAGES it is asked to read."""

    name = "qwen"

    def __init__(self, pages_read: list[int]) -> None:
        self._pages_read = pages_read

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        from socr.core.result import PageOutput

        self._pages_read.extend(page_nums)
        return [
            PageOutput(
                page_num=n,
                text="A MODEL READING OF THE PAGE",
                status=PageStatus.SUCCESS,
                engine="qwen",
            )
            for n in page_nums
        ]


def _run(
    out_dir: Path,
    pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    accept: bool,
    pages_read: list[int],
    render_dpi: int | None = None,
):
    """``process()`` over ``pdf`` into ``out_dir``, recording every page a model read."""
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _CountingEngine(pages_read))
        kwargs: dict = dict(
            agentic=True,
            quiet=True,
            # Pinned per CLAUDE.md #841: with ``primary_engine`` left AUTO, ``process()``
            # probes the registry and the outcome depends on which models this machine has.
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            native_first=True,
            write_manifest=False,
            judge_backend="heuristic",
            dual_pass_tables=False,
            detect_equations=False,
            save_figures=True,
            describe_figure_crops=False,
        )
        if render_dpi is not None:
            kwargs["render_dpi"] = render_dpi
        pipe = UnifiedPipeline(PipelineConfig(**kwargs))
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._build_page_judge = lambda state: _FixedJudge(accept)
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        return pipe.process(pdf, output_dir=out_dir)


def _crash_after_page_one(out_dir: Path) -> None:
    """Leave on disk exactly what a crash between page 1's flush and page 2's would.

    Page 1's terminal sidecar and fragment are left untouched; page 2's never existed, and
    neither did the assembled ``.md``, which assemble had not reached. Removing the ``.md``
    is what makes the DOC-level gate reprocess (``RootIndex.is_completed`` requires the
    recorded ``output_path`` to exist), and reprocessing is what puts the per-page ledger
    -- the gate under test -- back in the path. Without this the second run reports SKIPPED
    on the doc-level fast path and never consults the ledger at all, which is how a first
    draft of this test "passed" while proving nothing.
    """
    for page_two in out_dir.rglob("pages/00002.*"):
        page_two.unlink()
    for md in out_dir.rglob("*.md"):
        if "pages" not in md.parts:
            md.unlink()


def _page_one(out_dir: Path) -> tuple[dict, str]:
    sidecar = json.loads(next(iter(out_dir.rglob("pages/00001.json"))).read_text())
    body = next(iter(out_dir.rglob("pages/00001.md"))).read_text()
    return sidecar, body


def _clean_pdf(path: Path, *, pages: int = 2) -> Path:
    """A born-digital document the #1156 detector does NOT flag: text visible, no raster
    painted over it, and the image small enough to be a figure rather than a page scan."""
    import fitz

    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=612, height=792)
        for row in range(4):
            page.insert_text(
                (72, 120 + row * 24),
                f"{'w' * (i + 1)} Ordinary visible prose line {row} of a born-digital page.",
                fontsize=10,
            )
        page.insert_image(fitz.Rect(72, 300, 272, 460), stream=_png(200, 160))
    doc.save(path)
    doc.close()
    return path


def _promote_page_one_to_plain_success(sidecar_path: Path) -> None:
    """Make page 1 a plain ACCEPTED SUCCESS, leaving its FRAGMENT exactly as the run wrote
    it, so it resumes through the ORDINARY terminal-SUCCESS gate and nothing from #1192 is
    involved in reaching this code."""
    meta = json.loads(sidecar_path.read_text())
    meta["winning_output"] |= {
        "status": PageStatus.SUCCESS.value,
        "audit_passed": True,
        "engine": "qwen",
        "failure_mode": FailureMode.NONE.value,
    }
    meta["verdict"] = {"page_num": 1, "clauses": []}
    meta.pop("prior_ocr_layer_suspect", None)
    meta.pop("prior_ocr_layer_text", None)
    sidecar_path.write_text(json.dumps(meta))


def _resume_watching_the_ledger(
    out: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[int, bool]:
    """Resume, recording which pages the ledger GRANTED a skip to.

    r-uben's nit: ``assert 1 not in pages_read`` cannot fail on this fixture, because its
    pages take a native lane and no model reads them on either run. The ledger's own
    decision is the observable that can.
    """
    granted: dict[int, bool] = {}
    real = UnifiedPipeline._load_terminal_page

    def spy(self, state, page_num, output_dir):
        got = real(self, state, page_num, output_dir)
        granted[page_num] = got is not None
        return got

    monkeypatch.setattr(UnifiedPipeline, "_load_terminal_page", spy)
    _run(out, pdf, monkeypatch, accept=True, pages_read=[])
    return granted


def test_a_restored_pages_figure_block_is_embedded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_describe_and_embed_figures`` must be idempotent over its OWN output.

    A page restored from the per-page ledger arrives at assemble with its figure blocks
    already in its body -- the previous run wrote them into the fragment, and the fragment
    is what assemble reads back. The append was unconditional, so every partial resume
    added a second copy of each block to every restored page carrying a figure.

    NOT a defect of #1192: measured first on the ORDINARY terminal-SUCCESS gate, with no
    flagged page involved, which is the arm this test drives. #1192 only widens the set of
    pages that reach it -- every accepted earlier-OCR page, ~141,000 on the Fed corpus --
    which is why it is fixed here rather than filed.
    """
    import fitz

    path = tmp_path / "clean.pdf"
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=612, height=792)
        for row in range(4):
            page.insert_text(
                (72, 120 + row * 24),
                f"{'w' * (i + 1)} Ordinary visible prose line {row} of a born-digital page.",
                fontsize=10,
            )
        # SMALL, so it is a figure rather than a page-covering raster.
        page.insert_image(fitz.Rect(72, 300, 272, 460), stream=_png(200, 160))
    doc.save(path)
    doc.close()

    out = tmp_path / "out"
    _run(out, path, monkeypatch, accept=True, pages_read=[])
    sidecar_path = next(iter(out.rglob("pages/00001.json")))
    _side, before = _page_one(out)
    assert before.count("![Figure") == 1, (
        f"fixture premise: page 1 shipped {before.count('![Figure')} figure blocks, not 1"
    )

    _promote_page_one_to_plain_success(sidecar_path)
    _crash_after_page_one(out)
    granted = _resume_watching_the_ledger(out, path, monkeypatch)
    _side_after, after = _page_one(out)

    assert granted.get(1) is True, (
        "the ledger did not restore page 1, so it was re-read and this shows nothing"
    )
    assert after.count("![Figure") == before.count("![Figure"), (
        "the figure block was embedded a second time on a restored page:\n"
        f"before={before!r}\nafter={after!r}"
    )


def _two_figure_pdf(path: Path) -> Path:
    """Page 1 carries TWO figures; page 2 is the page a crash loses."""
    import fitz

    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=612, height=792)
        for row in range(4):
            page.insert_text(
                (72, 120 + row * 24),
                f"{'w' * (i + 1)} Ordinary visible prose line {row} of a born-digital page.",
                fontsize=10,
            )
        page.insert_image(fitz.Rect(72, 300, 220, 420), stream=_png(180, 150))
        page.insert_image(fitz.Rect(320, 300, 468, 420), stream=_png(150, 180))
    doc.save(path)
    doc.close()
    return path


def test_a_figure_missing_from_a_restored_fragment_is_re_embedded_without_duplicating_the_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The filter is per FIGURE, not per page -- r-uben's review of #1210, should-fix 1.

    The comment on the guard claims "filtered per figure rather than per page, so a page
    that gained a NEW figure still gets that one embedded beside the ones already there".
    Nothing pinned that: collapsing the guard to a per-PAGE skip
    (``if any(ref in already for fig in page_figs): continue``) left all 644 tests in the
    30 resume/figure/#1156/#1145 files passing, and DROPPED the second figure entirely --
    silent loss, worse than the duplicate the guard was added to stop.

    How a real run reaches this state: ``_drop_chart_region_duplicates`` suppresses an
    extracted figure only while its chart crop survives, so a crop that renders on run 1
    and fails on the resume leaves the restored page one block short while the other is
    already there. Reproduced here by cutting one block out of the fragment, which is that
    state on disk without needing the render to fail.

    Measured by r-uben at ``process()``, refs for (figure 1, figure 2):
    main ``3bcc43d`` (2, 1); this PR (1, 1); the per-page mutant (1, **0**).
    """
    path = _two_figure_pdf(tmp_path / "two.pdf")
    out = tmp_path / "out"
    _run(out, path, monkeypatch, accept=True, pages_read=[])

    frag_path = next(iter(out.rglob("pages/00001.md")))
    sidecar_path = next(iter(out.rglob("pages/00001.json")))
    before = frag_path.read_text()
    refs = ("figures/figure_1_page1.png", "figures/figure_2_page1.png")
    assert all(before.count(f"]({ref})") == 1 for ref in refs), (
        "fixture premise: page 1 did not ship exactly one block per figure, so neither arm "
        f"of this test means anything. refs = {[before.count(f']({r})') for r in refs]}"
    )

    # Cut the SECOND figure's block out of the fragment, leaving the first in place.
    kept, dropped = refs
    blocks = before.split("\n\n")
    surviving = [b for b in blocks if f"]({dropped})" not in b]
    assert len(surviving) == len(blocks) - 1, (
        f"expected to cut exactly one block, cut {len(blocks) - len(surviving)}"
    )
    frag_path.write_text("\n\n".join(surviving))

    _promote_page_one_to_plain_success(sidecar_path)
    _crash_after_page_one(out)
    granted = _resume_watching_the_ledger(out, path, monkeypatch)
    after = frag_path.read_text()

    assert granted.get(1) is True, (
        "the ledger did not restore page 1, so the figure phase never saw a restored body"
    )
    assert after.count(f"]({dropped})") == 1, (
        f"the figure missing from the restored fragment was not re-embedded ({dropped}): a "
        "per-PAGE skip drops it entirely, which is silent content loss\n"
        f"after={after!r}"
    )
    assert after.count(f"]({kept})") == 1, (
        f"the figure already in the restored fragment was embedded twice ({kept})\nafter={after!r}"
    )
