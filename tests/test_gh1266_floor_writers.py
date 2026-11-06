"""#1266 round 6 (PR #1273): socr's own post-selection writers keep a floor page's body.

``finalized_page_records(state, final_text)`` runs a saved-body pass: a floor page keeps the
saved bytes only if they are a body a floor certified. Assemble then adds lines after
selection (``_preserve_chart_regions``: a chart-region crop or the "Unresolved chart
placement" blockquote; ``_describe_crop_refs``: the number-free description under the crop),
and the saved pass shipped the recomputed floor instead, so the crop and the description left
the fragment, the .md and the sidecar while the audit events still said they were placed.

The rule is provenance, not line shape: a writer that changes a floor page's body re-certifies
that page (``recertify_floor_body``); the saved pass compares against the certified bodies, so
saved bytes ship only if socr's own writers produced them. Every test pins an identity or a
difference, never a locally measured status tuple.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gh1144_prose_survives_the_table_floor import (  # noqa: E402
    _GridEngine,
    _p77_like,
    _pipeline,
)

from socr.core.manifest import table_floor_digest_of  # noqa: E402

#: The audit-event kind a refused saved body records (pinned as a string, so a rename fails).
SAVED_FLOOR_BODY_REJECTED = "floor_saved_body_rejected"

DESCRIPTION = "A bar chart comparing growth across periods."
CHART_REF = "![chart region 1](figures/chart_region_p1_1.png)"
UNRESOLVED = "> **Unresolved chart placement on page 1**"
DESCRIPTION_LINE = "Figure description"


def _chart_floor_pdf(path: Path, *, anchored: bool, raster: bool = False) -> Path:
    """A born-digital page with two table boxes (the #1144 fixture) and a bar chart below them.

    ``anchored``: a prose line below the chart lets the chart region be placed by a source
    anchor. Without it the region has no anchor in the accepted body, so the writer injects
    the "Unresolved chart placement" blockquote above the crop. ``raster``: also a raster
    image the figure extractor finds (a non-chart figure)."""
    fitz = pytest.importorskip("fitz")
    _p77_like(path)
    doc = fitz.open(str(path))
    page = doc[0]
    base = 470
    page.draw_line(fitz.Point(100, base), fitz.Point(100, base - 150), width=2)
    page.draw_line(fitz.Point(100, base), fitz.Point(400, base), width=2)
    for i in range(6):
        height = 30 + 15 * i
        page.draw_rect(
            fitz.Rect(120 + i * 45, base - height, 150 + i * 45, base),
            color=(0, 0, 0),
            fill=(0.3, 0.3, 0.6),
        )
    if anchored:
        page.insert_text((72, 600), "Below the chart the staff noted that risks remained balanced.")
    if raster:
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 80), False)
        pix.set_rect(pix.irect, (200, 30, 30))
        png = path.with_name("blob.png")
        pix.save(str(png))
        page.insert_image(fitz.Rect(400, 620, 520, 700), filename=str(png))
    out = path.with_name("chart_floor.pdf")
    doc.save(str(out))
    doc.close()
    return out


class _Vision:
    """A stand-in for the ``--describe-figures`` vision engine: no model, a fixed caption."""

    name = "stub-vision"

    def __init__(self, caption: str) -> None:
        self.caption = caption

    def describe_figure(self, image, context=None):
        from types import SimpleNamespace

        return SimpleNamespace(description=self.caption, figure_type="chart")

    def close(self) -> None:
        pass


class _Calls(dict):
    """What the spies saw on the LAST ``process()`` of a ``_run``."""


def _run(
    tmp_path: Path,
    monkeypatch,
    *,
    anchored: bool,
    describe: bool = True,
    again=None,
    resume: bool = False,
    raster: bool = False,
    save_figures: bool = False,
    describe_figures: bool = False,
    caption: str = "A caption without any figures.",
):
    """Hermetic ``process()`` runs; returns ``(runs, calls)``, one ``_read`` per run.

    The ladder, the judge model and the crop model are pinned by ``_pipeline``; the crop
    describer and the ``--describe-figures`` vision engine are stubs. ``again=1`` runs a second
    time over the same output directory; with ``resume`` the root ``metadata.json`` indexes are
    removed first, so the document is not SKIPPED and the page-level resume really runs
    (``calls`` proves it)."""
    from socr.pipeline import orchestrator as orch

    work = tmp_path / "w"
    work.mkdir(parents=True, exist_ok=True)
    pdf = _chart_floor_pdf(work / "doc.pdf", anchored=anchored, raster=raster)
    monkeypatch.setattr(orch, "get_engine", lambda engine_type: _GridEngine())
    out = work / "out"
    runs = []
    calls = _Calls(load_terminal=0, load_restored=0, assemble=0)
    for n in range(1 if again is None else 1 + again):
        if n and resume:
            for index in list(out.rglob("metadata.json")):
                index.unlink()
        calls.update(load_terminal=0, load_restored=0, assemble=0)
        pipe = _pipeline()
        pipe.config.save_figures = save_figures
        pipe.config.describe_figures = describe_figures
        pipe.config.describe_figure_crops = describe
        pipe._get_vision_engine = lambda caption=caption: _Vision(caption)
        if describe:

            def ask(path, prompt):
                return DESCRIPTION

            ask.model = "stub-describer"
            pipe._figure_description_ask = lambda ask=ask: ask
        real_load = orch.UnifiedPipeline._load_terminal_page
        real_assemble = orch.UnifiedPipeline._phase_assemble

        def load(self, *a, _real=real_load, **k):
            calls["load_terminal"] += 1
            got = _real(self, *a, **k)
            calls["load_restored"] += got is not None
            return got

        def assemble(self, *a, _real=real_assemble, **k):
            calls["assemble"] += 1
            return _real(self, *a, **k)

        monkeypatch.setattr(orch.UnifiedPipeline, "_load_terminal_page", load)
        monkeypatch.setattr(orch.UnifiedPipeline, "_phase_assemble", assemble)
        pipe.process(pdf, output_dir=out)
        monkeypatch.setattr(orch.UnifiedPipeline, "_load_terminal_page", real_load)
        monkeypatch.setattr(orch.UnifiedPipeline, "_phase_assemble", real_assemble)
        runs.append(_read(out))
    return runs, calls


def _read(out: Path):
    frag = next(iter(out.rglob("pages/00001.md"))).read_text()
    meta = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    mds = [p for p in out.rglob("*.md") if "pages" not in p.parts]
    assert mds, "premise: a final .md was written"
    winning = dict(meta.get("winning_output") or {})
    winning["_events"] = meta.get("audit_events") or []
    winning["_figure_refs"] = meta.get("figure_refs") or []
    return frag, mds[0].read_text(), winning, out


def _kinds(winning) -> list[str]:
    return [e.get("kind") for e in winning["_events"]]


@pytest.mark.parametrize("anchored", [True, False], ids=["anchored", "unresolved_placement"])
def test_the_writers_additions_survive_the_saved_pass(
    tmp_path: Path, monkeypatch, anchored: bool
) -> None:
    """The crop, the description and (unplaced) the blockquote ship in the fragment, the .md
    and the sidecar, with the credential bound to those exact bytes. Fails on fd0ba04c, where
    the saved pass shipped the recomputed floor and dropped all three."""
    [(frag, md, winning, _out)], _ = _run(tmp_path, monkeypatch, anchored=anchored)

    text = winning.get("text") or ""
    assert winning.get("table_floor_body") is True, "premise: a floor page"
    for label, body in (("fragment", frag), ("final .md", md), ("sidecar", text)):
        assert CHART_REF in body, f"{label} lost the chart crop"
        assert DESCRIPTION_LINE in body and DESCRIPTION in body, f"{label} lost the description"
        assert (UNRESOLVED in body) is (not anchored), f"{label}: unresolved blockquote differs"
    assert text == frag, "the sidecar and the fragment must carry the same bytes"
    assert frag.strip() in md
    assert winning.get("table_floor_digest") == table_floor_digest_of(text), (
        "the credential must stay bound to the shipped bytes"
    )


def test_the_describer_only_adds_its_own_lines(tmp_path: Path, monkeypatch) -> None:
    """DIFFERENCE: the same page with the describer off ships the crop and no description,
    and with it on ships the same body plus the description (the writer's output, nothing
    else)."""
    [(off_frag, *_)], _ = _run(tmp_path / "off", monkeypatch, anchored=True, describe=False)
    [(on_frag, *_)], _ = _run(tmp_path / "on", monkeypatch, anchored=True, describe=True)
    assert CHART_REF in off_frag and DESCRIPTION_LINE not in off_frag
    assert CHART_REF in on_frag and DESCRIPTION_LINE in on_frag
    dropped = [ln for ln in on_frag.splitlines() if DESCRIPTION_LINE not in ln]
    assert [ln for ln in dropped if ln.strip()] == [
        ln for ln in off_frag.splitlines() if ln.strip()
    ]


_RESUME_ARMS = [
    pytest.param(True, False, id="anchored"),
    pytest.param(False, False, id="unresolved_placement"),
    pytest.param(True, True, id="save_figures"),
]


@pytest.mark.parametrize(("anchored", "save_figures"), _RESUME_ARMS)
def test_a_resumed_run_is_byte_identical(
    tmp_path: Path, monkeypatch, anchored: bool, save_figures: bool
) -> None:
    """The page-level resume (the root indexes removed, ``pages/*`` kept, so the document is
    not SKIPPED) restores the terminal page and reassembles it; fragment, .md and sidecar are
    byte-identical to run 1. The spies prove the resume really ran: a SKIP would pass the
    comparison vacuously."""
    (first, second), calls = _run(
        tmp_path,
        monkeypatch,
        anchored=anchored,
        again=1,
        resume=True,
        raster=save_figures,
        save_figures=save_figures,
    )
    # Run 2 is not the document-level SKIP: the page ledger gate was consulted and assemble ran.
    # (A floor page here is non-SUCCESS -- the page judge timed out -- so the gate reprocesses
    # it rather than restoring it; the comparison below is of what run 2 writes on the same
    # output directory, and ``load_restored`` is recorded in ``calls`` for anyone who changes
    # that.)
    assert calls["load_terminal"] >= 1, calls
    assert calls["assemble"] >= 1, calls
    assert CHART_REF in first[0] and DESCRIPTION in first[0], "premise: run 1 shipped them"
    if save_figures:
        assert "![Figure" in first[0], "premise: run 1 embedded the extracted figure"
    assert second[0] == first[0], "fragment changed on resume"
    assert second[1] == first[1], "final .md changed on resume"
    assert second[2].get("text") == first[2].get("text"), "sidecar text changed on resume"
    assert second[2].get("table_floor_digest") == first[2].get("table_floor_digest")
    assert second[0].count("![Figure") == first[0].count("![Figure"), "a block was duplicated"
    assert "floor_saved_body_rejected" not in _kinds(second[2])


_MATRIX = [
    pytest.param(s, d, c, id=f"save{int(s)}-desc{int(d)}-crop{int(c)}")
    for s in (False, True)
    for d in (False, True)
    for c in (False, True)
]


@pytest.mark.parametrize(("save_figures", "describe_figures", "crop"), _MATRIX)
def test_no_writer_combination_loses_what_the_events_claim(
    tmp_path: Path, monkeypatch, save_figures: bool, describe_figures: bool, crop: bool
) -> None:
    """Guard against an uncovered post-selection writer: in every config combination, on a
    floor page with a chart region and a raster figure, nothing is rejected, everything the
    events and ``figure_refs`` claim is in the fragment, the .md and the sidecar, and the
    fragment equals the sidecar text."""
    [(frag, md, winning, _out)], _ = _run(
        tmp_path,
        monkeypatch,
        anchored=True,
        raster=True,
        save_figures=save_figures,
        describe_figures=describe_figures,
        describe=crop,
    )
    text = winning.get("text") or ""
    assert winning.get("table_floor_body") is True, "premise: a floor page"
    kinds = _kinds(winning)
    assert "floor_saved_body_rejected" not in kinds
    assert "chart_region_preserved" in kinds, "premise: the chart region was placed"
    bodies = {"fragment": frag, "final .md": md, "sidecar": text}
    for label, body in bodies.items():
        assert CHART_REF in body, f"{label} lost the chart crop the events claim"
        if "figure_description_described" in kinds:
            assert DESCRIPTION in body, f"{label} lost the description the events claim"
    if crop:
        assert "figure_description_described" in kinds, "premise: the crop was described"
    claimed = [r["image_path"] for r in winning["_figure_refs"] if r.get("image_path")]
    if save_figures or describe_figures:
        assert claimed, "premise: the extracted figure is claimed in figure_refs"
    for image_path in claimed:
        for label, body in bodies.items():
            assert Path(image_path).name in body, f"{label} lost {image_path}"
    assert text == frag, "the sidecar and the fragment must carry the same bytes"
    assert frag.strip() in md
    assert winning.get("table_floor_digest") == table_floor_digest_of(text)


DIGIT_CAPTION = "Real GDP rose 2.6 percent against a forecast of 4.1 percent."


def test_a_digit_bearing_caption_never_ships_on_a_floor_page(tmp_path: Path, monkeypatch) -> None:
    """``--describe-figures`` captions have no number check; on a floor page one over a
    withheld table box would ship its numbers under the floor credential. The caption is not
    written into the body, the sidecar's ``figure_refs`` or the metadata; the image and the
    ``**Figure N**`` header are, and a withheld-caption event is recorded without the text."""
    [(frag, md, winning, out)], _ = _run(
        tmp_path,
        monkeypatch,
        anchored=True,
        raster=True,
        describe_figures=True,
        caption=DIGIT_CAPTION,
    )
    kinds = _kinds(winning)
    assert "figure_caption_withheld_on_floor" in kinds
    assert "floor_saved_body_rejected" not in kinds
    for label, body in (("fragment", frag), ("final .md", md), ("sidecar", winning["text"])):
        assert "2.6" not in body and "4.1" not in body, f"{label} shipped the caption numbers"
        assert "![Figure" in body and "**Figure 1** (page 1)" in body, f"{label}: image or header"
    assert all("2.6" not in (r.get("description") or "") for r in winning["_figure_refs"])
    for meta_file in out.rglob("metadata.json"):
        assert DIGIT_CAPTION not in meta_file.read_text()
    event = next(e for e in winning["_events"] if e["kind"] == "figure_caption_withheld_on_floor")
    assert "2.6" not in json.dumps(event)


def test_the_same_caption_ships_on_a_page_that_is_not_a_floor(tmp_path: Path, monkeypatch) -> None:
    """DIFFERENCE, one thing changed: the page holds no certified floor body. The identical
    caption is written as before (non-floor output is unchanged), and the floor arm of the same
    run withholds it."""
    fitz = pytest.importorskip("fitz")
    from socr.pipeline import orchestrator as orch

    work = tmp_path / "plain"
    work.mkdir()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 80), "A plain page of prose with no table at all, only a picture.")
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 80), False)
    pix.set_rect(pix.irect, (200, 30, 30))
    png = work / "blob.png"
    pix.save(str(png))
    page.insert_image(fitz.Rect(400, 620, 520, 700), filename=str(png))
    pdf = work / "plain.pdf"
    doc.save(str(pdf))
    doc.close()
    monkeypatch.setattr(orch, "get_engine", lambda engine_type: _GridEngine())
    pipe = _pipeline()
    pipe.config.save_figures = True
    pipe.config.describe_figures = True
    pipe._get_vision_engine = lambda: _Vision(DIGIT_CAPTION)
    out = work / "out"
    pipe.process(pdf, output_dir=out)
    plain_frag = next(iter(out.rglob("pages/00001.md"))).read_text()
    plain_meta = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    assert (plain_meta.get("winning_output") or {}).get("table_floor_body") is not True
    assert DIGIT_CAPTION in plain_frag, "the non-floor page lost its caption"

    [(floor_frag, *_)], _ = _run(
        tmp_path / "floor",
        monkeypatch,
        anchored=True,
        raster=True,
        describe_figures=True,
        caption=DIGIT_CAPTION,
    )
    assert DIGIT_CAPTION not in floor_frag, "the floor page withheld it"


# ---------------------------------------------------------------------------
# the saved pass: only socr's own writers certify a changed body
# ---------------------------------------------------------------------------


def _floor_state(tmp_path: Path):
    from test_gh1266_d3_regional_floor import _d3
    from test_gh1266_floor_geometry import GRID_BOX, _build, _draw_p10

    state, ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="fw.pdf")
    _d3(ps)
    return state, ps


@pytest.mark.parametrize(
    "line",
    [
        "[Loose 7.7 8.8](https://example.org)",
        "![Loose 7.7 8.8](figures/failed_table_p1.png)",
        "![Loose](https://example.org/x.png)",
        "[a plain link](https://example.org)",
        "Loose 7.7 8.8",
    ],
    ids=["digit_link", "digit_alt", "foreign_image_target", "plain_link", "loose_row"],
)
def test_saved_text_a_writer_did_not_produce_never_ships(tmp_path: Path, line: str) -> None:
    """Appended content that no socr writer produced is refused on a floor page and the
    refusal is recorded, once, however often finalize repeats. Fails on fd0ba04c: the link and
    image lines compared equal to the certified body and shipped with a re-bound digest, and
    no event was recorded."""
    from socr.core.manifest import _select_and_finalize_page

    state, ps = _floor_state(tmp_path)
    fresh = _select_and_finalize_page(state, 1).output
    assert fresh.table_floor_body, "premise: a certified floor body"

    control = _select_and_finalize_page(state, 1, saved_text=fresh.text).output
    assert control.text == fresh.text
    assert not [e for e in state.events if e.kind == SAVED_FLOOR_BODY_REJECTED], (
        "the floor's own saved body is not a rejection"
    )

    altered = fresh.text + "\n\n" + line + "\n"
    for _ in range(3):
        shipped = _select_and_finalize_page(state, 1, saved_text=altered).output
        assert line not in shipped.text, shipped.text
        assert shipped.text == fresh.text
        assert shipped.table_floor_digest == table_floor_digest_of(fresh.text)
    events = [e for e in state.events if e.kind == SAVED_FLOOR_BODY_REJECTED]
    assert len(events) == 1 and events[0].page_num == 1
    assert "7.7" not in json.dumps(events[0].data), "the event must not carry the rejected text"


def test_only_a_writer_that_recertifies_lets_a_changed_body_ship(tmp_path: Path) -> None:
    """DIFFERENCE, one thing changed: the same appended chart ref ships after
    ``recertify_floor_body`` and not before; a page holding no certificate cannot be
    re-certified, and a writer whose input was already altered certifies nothing."""
    from socr.core.manifest import _select_and_finalize_page, recertify_floor_body

    state, ps = _floor_state(tmp_path)
    fresh = _select_and_finalize_page(state, 1).output
    written = fresh.text + "\n\n" + CHART_REF + "\n"

    refused = _select_and_finalize_page(state, 1, saved_text=written).output
    assert CHART_REF not in refused.text, "premise: nothing certified the writer's body yet"

    assert recertify_floor_body(ps, fresh.text, written, 1) is True
    kept = _select_and_finalize_page(state, 1, saved_text=written).output
    assert kept.text == written
    assert kept.table_floor_body is True
    assert kept.table_floor_digest == table_floor_digest_of(written)

    # a fresh finalize in between (the sidecar flush) recomputes the floor; the writer's
    # body still counts because its base is still the floor
    _select_and_finalize_page(state, 1)
    assert _select_and_finalize_page(state, 1, saved_text=written).output.text == written

    # a second writer starting from the first writer's body chains
    described = written + "\n> *Figure description:* " + DESCRIPTION + "\n"
    assert recertify_floor_body(ps, written, described, 1) is True
    assert _select_and_finalize_page(state, 1, saved_text=described).output.text == described

    # a writer whose input was altered by something else certifies nothing
    altered = described + "\nLoose 7.7 8.8\n"
    assert recertify_floor_body(ps, altered, altered + CHART_REF, 1) is False
    assert _select_and_finalize_page(state, 1, saved_text=altered).output.text != altered

    # a page that holds no certificate cannot be re-certified
    (tmp_path / "bare").mkdir()
    bare_state, bare = _floor_state(tmp_path / "bare")
    assert not bare.certified_floor_text
    assert recertify_floor_body(bare, fresh.text, written, 1) is False
    shipped = _select_and_finalize_page(bare_state, 1, saved_text=written).output
    assert CHART_REF not in shipped.text


def test_a_changed_floor_drops_the_writers_stale_body(tmp_path: Path) -> None:
    """The writer's body counts only while its base is still the page's certified floor."""
    from socr.core.manifest import _select_and_finalize_page, recertify_floor_body

    state, ps = _floor_state(tmp_path)
    fresh = _select_and_finalize_page(state, 1).output
    written = fresh.text + "\n\n" + CHART_REF + "\n"
    assert recertify_floor_body(ps, fresh.text, written, 1)
    ps.certified_floor_text = "A different floor body about other things entirely."
    shipped = _select_and_finalize_page(state, 1, saved_text=written).output
    assert CHART_REF not in shipped.text
