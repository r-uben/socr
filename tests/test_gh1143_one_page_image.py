"""#1143: a withheld-table marker must not re-embed the page image per table.

Forney 1988 p22 (``scratch/routing-baseline/review-pages/out/d_forney_p22``,
main 25daab9f) withholds twelve runs of numeric bands on one scanned page, and
``native_prose_floor_text`` appended ``marker + "\\n\\n" + png_ref`` to each of
them -- so the same full-page render shipped twelve times. Owner, reviewing
that output: "I see the figure repeated all the time."

Every leg here is a DIFFERENCE between two runs of the same code in one
process, changing only the number of withheld runs on the page. The marker
count must track the runs (nothing may lose its marker -- that is #649 round
2's reason for the per-run marker); the EMBED count must not.

The one embed sits under the banner, not at the first withheld run: this branch
has no table geometry, so no run is "the table", and run 1 is routinely a lone
prose line carrying a year. Each marker LINKS that embed instead.
"""

from __future__ import annotations

import json

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.manifest import (
    SCANNED_PROSE_RECOVERED_FLAG,
    native_prose_floor_text,
    page_image_link,
)
from socr.core.state import PageState

fitz = pytest.importorskip("fitz")

MARKER = "[page 1 failed: unverifiable table — see image]"
PNG_REF = "![Failed table page 1](figures/failed_table_p1.png)"
PNG_LINK = "[page image](figures/failed_table_p1.png)"

_PROSE_ABOVE = [
    "With Ms. Seger dissenting, the Federal Reserve Bank of New York",
    "was authorized and directed, until otherwise directed by the Committee,",
    "to execute transactions in the System Account in accordance with the",
    "following domestic policy directive:",
]
#: Three withheld runs: the wrapped label lines carry no printed value, so they
#: ship as prose and split the numeric bands into three contiguous runs. This
#: is the geometry ``tests/test_gh649_scanned_prose_recovery.py`` already pins
#: at three markers -- the same page, now carrying a rendered page image.
_ROWS_THREE_RUNS = [
    "Austrian National Bank 250.0",
    "Bank for International",
    "Settlements-",
    "Swiss francs 600.0",
    "Other authorized",
    "European currencies 1,250.0",
]
#: One withheld run: three adjacent numeric bands, nothing splitting them.
_ROWS_ONE_RUN = [
    "Austrian National Bank 250.0",
    "National Bank of Belgium 1,000.0",
    "German Federal Bank 6,000.0",
]
_PROSE_BELOW = [
    "The information reviewed at this meeting suggests continuing expansion",
    "in economic activity, though at a somewhat slower pace than earlier.",
    "Total nonfarm payroll employment increased appreciably in October.",
]
_WITHHELD_AMOUNTS = ("250.0", "600.0", "1,250.0", "1,000.0", "6,000.0")

PAGE_TWO = "A second page of clean prose, so the document assembles a stem markdown file."


def _words(lines: list[str]) -> list[tuple]:
    """Native words on real baselines, one printed line per entry."""
    words: list[tuple] = []
    for offset, line in enumerate(lines):
        top = offset * 12.0
        for word_idx, tok in enumerate(line.split()):
            left = word_idx * 14.0
            words.append((left, top, left + 12.0, top + 9.0, tok, 0, 0, 0))
    return words


def _page(rows: list[str]) -> PageState:
    """A page reaching the scanned-table floor with NO table geometry."""
    ps = PageState(page_num=1)
    ps.is_born_digital = False
    ps.native_text = ""
    ps.native_words = _words(_PROSE_ABOVE + rows + _PROSE_BELOW)
    ps.detected_table_count = 0
    ps.detected_table_bboxes = []
    ps.scanned_table_evidence_failed = True
    return ps


def _recovered(rows: list[str], *, png_ref: str = PNG_REF) -> str:
    text = native_prose_floor_text(_page(rows), 1, marker_line=MARKER, png_ref=png_ref)
    assert text is not None, "the recovery lane must be reached for this leg to mean anything"
    return text


class TestTheEmbedDoesNotGrowWithTheWithheldRuns:
    def test_three_runs_carry_three_markers_and_one_embed(self) -> None:
        """The ticket, as a difference. Only the band geometry changes between
        the two arms; the marker count follows it and the embed count does
        not."""
        three = _recovered(_ROWS_THREE_RUNS)
        one = _recovered(_ROWS_ONE_RUN)

        assert three.count(MARKER) == 3
        assert one.count(MARKER) == 1
        # The whole point: three times the markers, the same one picture.
        assert three.count(PNG_REF) == one.count(PNG_REF) == 1

    def test_no_withheld_run_loses_its_marker(self) -> None:
        """#649 round 2's reason for the per-run marker is untouched: a run
        elided mid-page is still marked where it happened, and every printed
        value on the page is still withheld."""
        three = _recovered(_ROWS_THREE_RUNS)

        assert three.startswith(SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1))
        for amount in _WITHHELD_AMOUNTS:
            assert amount not in three, amount
        # The bare wrapped labels still ship, with a marker beside each run.
        assert "Bank for International" in three
        assert "Other authorized" in three

    def test_every_marker_links_the_one_file_the_embed_records(self) -> None:
        """What replaces the duplicated embed is a plain link, one per marker,
        pointing at the single file the embed on this same page records -- so
        the links still resolve after a replay into another directory, where
        manifest asset discovery collects embeds only."""
        three = _recovered(_ROWS_THREE_RUNS)
        one = _recovered(_ROWS_ONE_RUN)

        assert three.count(PNG_LINK) == three.count(MARKER) == 3
        assert one.count(PNG_LINK) == one.count(MARKER) == 1
        # The embed comes first, in document order, so no link precedes it.
        assert three.index(PNG_REF) < three.index(PNG_LINK)

    def test_the_one_embed_sits_under_the_banner(self) -> None:
        """A whole-page render is page-level, like the banner. Nothing of the
        page's own text comes between them, and the banner is still line one so
        ``_is_restored_prose_recovery`` and ``is_page_failed_marker`` read what
        they always read."""
        three = _recovered(_ROWS_THREE_RUNS)
        banner = SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1)

        assert three.startswith(f"{banner}\n\n{PNG_REF}\n\n")
        # Not at the first withheld run, which here is a table row band but in
        # general is whatever band happened to print a digit first.
        assert three.index(PNG_REF) < three.index(MARKER)

    def test_the_marker_line_itself_is_unchanged(self) -> None:
        """``socr.core.table_counts`` and ``_PAGE_FAILED_ANY_RE`` match the
        marker LINE. Every run must still present it byte-identically, or a
        withheld table stops being counted as withheld."""
        from socr.core.table_counts import _WITHHELD_MARKER_RE

        three = _recovered(_ROWS_THREE_RUNS)
        marked = [line for line in three.splitlines() if _WITHHELD_MARKER_RE.search(line)]

        assert len(marked) == 3
        assert {line for line in marked} == {MARKER}

    def test_a_page_with_no_rendered_image_still_ships_bare_markers(self) -> None:
        """The render can fail (``_render_d3_floor_png`` returns ""). Nothing
        is linked then, and the markers are unaffected."""
        three = _recovered(_ROWS_THREE_RUNS, png_ref="")
        with_image = _recovered(_ROWS_THREE_RUNS)

        assert three.count(MARKER) == with_image.count(MARKER) == 3
        assert "page image](" not in three
        assert "![" not in three

    def test_an_unparseable_ref_still_ships_once_and_leaves_the_markers_bare(self) -> None:
        """``_render_d3_floor_png`` returns either "" or a well-formed embed, so
        this is defensive -- but the outcome must still be lossless: whatever was
        handed in is visible ONCE, and no marker carries a link that would be
        broken."""
        odd = "figures/failed_table_p1.png"
        three = _recovered(_ROWS_THREE_RUNS, png_ref=odd)

        assert three.count(MARKER) == 3
        assert three.count(odd) == 1
        assert "page image](" not in three


class TestPageImageLink:
    def test_it_reads_the_path_out_of_the_rendered_embed(self) -> None:
        assert page_image_link(PNG_REF) == PNG_LINK

    @pytest.mark.parametrize("ref", ["", "figures/failed_table_p1.png", "[not an embed](x.png)"])
    def test_it_abstains_on_anything_that_is_not_an_embed(self, ref: str) -> None:
        """Abstention is what lets the caller fall back to its bare marker
        rather than ship a broken link."""
        assert page_image_link(ref) == ""


# ----------------------------------------------------------------------------
# The same difference, through ``process()``.
# ----------------------------------------------------------------------------

MODEL_TABLE = "| Institution | Amount |\n| --- | --- |\n| Austrian National Bank | 999.9 |\n"


def _scan_pdf(path: Path, rows: list[str]) -> Path:
    """One page shaped like the ticket's, plus a clean second page.

    The second page is why a ``<stem>.md`` exists at all: page one ships no
    verified table, and a one-page document would leave the byte-identity legs
    with nothing to compare.
    """
    doc = fitz.open()
    page = doc.new_page()
    y = 72.0
    for line in _PROSE_ABOVE + rows + _PROSE_BELOW:
        page.insert_text((72, y), line, fontsize=10)
        y += 12.0
    second = doc.new_page()
    second.insert_text((72, 72), PAGE_TWO, fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def _run_process(pdf_path: Path, out_dir: Path):
    """Hermetic: no provider, no judge, no ollama probe, no native-first lane.

    Page one is classified as a scan at ``BornDigitalDetector._assess_page``
    -- the one ambient verdict this lane needs and the only thing a real scan
    with an old OCR layer supplies that ``fitz`` will not synthesise. Its
    ``native_words`` still come from the page's own text layer, unpatched,
    which is the half the recovery actually reads.
    """
    from socr.core.born_digital import BornDigitalDetector
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import FailureMode, PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision, ProviderAttempt
    from socr.pipeline.orchestrator import UnifiedPipeline

    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=True,
        enabled_engines=[EngineType.QWEN],
        local_engine=EngineType.QWEN,
        # Argon's review, 2026-10-14: ``qwen_backend`` defaults to "auto", and with
        # ``VLLM_BASE_URL`` set in the environment the ladder probes an OpenAI
        # server instead of the patched ``probe_ollama_idle``. Pin it, per
        # CLAUDE.md's no-provider trap.
        qwen_backend="ollama",
        quiet=True,
        save_figures=True,
        write_manifest=True,
        table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md
    )
    pipeline = UnifiedPipeline(config)
    unpatched_assess = BornDigitalDetector._assess_page

    def _page_one_is_a_scan(self, page, page_num):
        assessment = unpatched_assess(self, page, page_num)
        return replace(assessment, is_born_digital=False) if page_num == 1 else assessment

    def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
        """Page one: a model table the source-evidence gate rejected."""
        if page_num != 1:
            raise AssertionError(
                "page 2 is born-digital prose and must never reach the model ladder; "
                "if it does, this fixture is no longer the shape it claims to be"
            )
        out = PageOutput(
            page_num=page_num,
            text=MODEL_TABLE,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=False,
        )
        out.failure_mode = FailureMode.HALLUCINATION
        attempt = ProviderAttempt(
            engine=EngineType.QWEN,
            output=out,
            cost_usd=0.0,
            accepted=False,
            reason="source_evidence_table: no witness corroborated this table",
            provider_id=PROFILE_QWEN_LOCAL.id,
            model=PROFILE_QWEN_LOCAL.model,
            backend=PROFILE_QWEN_LOCAL.backend,
        )
        return PageDecision(page_num=page_num, final_output=out, attempts=[attempt], accepted=False)

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_plan_native_table_first", return_value=None),
        patch("socr.pipeline.orchestrator.route_page", _fake_route),
        patch.object(BornDigitalDetector, "_assess_page", _page_one_is_a_scan),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
    ):
        pipeline.process(pdf_path, out_dir)
    return pipeline, out_dir


def _surfaces(pdf_path: Path, out_dir: Path) -> dict[str, str]:
    """Every place page one's bytes land, so one of them cannot drift alone."""
    from socr.core.cache import BlobStore
    from socr.core.manifest import Manifest, replay

    manifest_path = next(out_dir.rglob("manifest.json"))
    manifest = Manifest.load(manifest_path)
    store = BlobStore(manifest_path.parent / "cache")
    sidecar = json.loads(next(out_dir.rglob("pages/00001.json")).read_text(encoding="utf-8"))
    return {
        "final_md": next(out_dir.rglob(f"{pdf_path.stem}.md")).read_text(encoding="utf-8"),
        "fragment": next(out_dir.rglob("pages/00001.md")).read_text(encoding="utf-8"),
        "sidecar": sidecar["winning_output"]["text"],
        "manifest_blob": store.get_page(manifest.entries[1].blob_ref).text,
        "replay": replay(manifest, store),
    }


class TestThroughProcess:
    @pytest.fixture(scope="class")
    def arms(self, tmp_path_factory) -> dict:
        """Both arms of the difference, run in this one process."""
        root = tmp_path_factory.mktemp("gh1143")
        out = {}
        for label, rows in (("three", _ROWS_THREE_RUNS), ("one", _ROWS_ONE_RUN)):
            pdf = _scan_pdf(root / f"doc_{label}.pdf", rows)
            pipeline, out_dir = _run_process(pdf, root / f"out_{label}")
            out[label] = {
                "pdf": pdf,
                "out_dir": out_dir,
                "pipeline": pipeline,
                "surfaces": _surfaces(pdf, out_dir),
            }
        return out

    def test_the_lane_was_actually_reached(self, arms: dict) -> None:
        """Without this the counts below would all be zero and every leg
        vacuous. Nothing here depends on a provider: the ladder, the judge and
        the page's scan verdict are all patched."""
        for label in ("three", "one"):
            fragment = arms[label]["surfaces"]["fragment"]
            # #1185: finalization prepends the text_layer_numbers mark (the model reading ships
            # no number while the page's own layer prints some, so it is MARKED and the mark
            # counts them); the recovery banner follows it.
            unchecked_mark = (
                "> [page 1 unchecked: text_layer_numbers=UNKNOWN; scope=page; reason=the reading "
                "ships no number, while the page's text layer prints "
            )
            banner = SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1)
            assert fragment.startswith(unchecked_mark)
            assert fragment.split("\n\n", 1)[1].startswith(banner)
            assert MARKER in fragment

    def test_every_surface_shows_three_markers_and_one_image(self, arms: dict) -> None:
        """The acceptance criterion, at each place the page's bytes land, as a
        difference against the one-run arm."""
        for name, three in arms["three"]["surfaces"].items():
            one = arms["one"]["surfaces"][name]
            assert three.count(MARKER) == 3, name
            assert one.count(MARKER) == 1, name
            assert three.count(PNG_REF) == 1, name
            assert one.count(PNG_REF) == 1, name
            assert three.count(PNG_LINK) == 3, name
            assert one.count(PNG_LINK) == 1, name

    def test_no_printed_value_ships_on_either_arm(self, arms: dict) -> None:
        """The floor is unchanged. Dropping a duplicated picture must not let
        a withheld number through anywhere."""
        for label in ("three", "one"):
            for name, text in arms[label]["surfaces"].items():
                for amount in _WITHHELD_AMOUNTS:
                    assert amount not in text, f"{label}/{name}: {amount}"

    def test_the_one_embed_is_the_recorded_manifest_asset(self, arms: dict) -> None:
        """A single embed is only safe if it is what manifest asset discovery
        records, because the links lean on it: ``_local_page_asset_refs`` scans
        ``![...]`` tokens and nothing else, so a page with links alone would
        lose the PNG on a replay into another directory.

        Argon's review, 2026-10-14: the earlier version of this leg only checked
        that ``figures/failed_table_p1.png`` was on disk. ``_render_d3_floor_png``
        writes that file before this lane runs, so the leg passed even with zero
        embeds in the text -- it was pinning the renderer, not the reference.
        """
        from socr.core.manifest import Manifest

        for label in ("three", "one"):
            out_dir = arms[label]["out_dir"]
            manifest = Manifest.load(next(out_dir.rglob("manifest.json")))
            recorded = [
                asset.logical_path
                for asset in manifest.entries[1].assets
                if asset.logical_path.endswith("failed_table_p1.png")
            ]
            assert recorded == ["figures/failed_table_p1.png"], (label, recorded)
            pngs = list(out_dir.rglob("figures/failed_table_p1.png"))
            assert len(pngs) == 1 and pngs[0].exists(), label

    def test_the_stitch_is_byte_identical_to_the_assembled_markdown(self, arms: dict) -> None:
        from socr.core.document import DocumentHandle
        from socr.core.state import DocumentState

        for label in ("three", "one"):
            arm = arms[label]
            state = DocumentState(handle=DocumentHandle.from_path(arm["pdf"]))
            stitched = arm["pipeline"]._stitch_fragments(state, arm["out_dir"])
            assert stitched == arm["surfaces"]["final_md"], label

    def test_the_restored_recovery_is_returned_verbatim(self, arms: dict) -> None:
        """The OTHER resume shape, and the one the second-run leg below does
        NOT reach.

        Argon's review, 2026-10-14: page one keeps ERROR status, so
        ``_load_terminal_page`` never restores it and a second ``process()``
        rebuilds the body from freshly extracted ``native_words``. The path where
        ``_is_restored_prose_recovery`` actually fires is finalization over a
        page whose words are GONE and whose ``best_output`` came back from the
        sidecar -- the #649 round 2 resume loss. Pinned here as a difference
        against the live arm: the frozen bytes must come back unchanged, so the
        single embed cannot multiply on the next run.
        """
        from socr.core.document import DocumentHandle
        from socr.core.manifest import finalized_page_records
        from socr.core.result import PageOutput
        from socr.core.state import DocumentState, PageState

        for label in ("three", "one"):
            shipped = arms[label]["surfaces"]["sidecar"]
            sidecar = json.loads(
                next(arms[label]["out_dir"].rglob("pages/00001.json")).read_text(encoding="utf-8")
            )
            restored = PageOutput.from_dict(sidecar["winning_output"])
            assert restored.scanned_prose_recovered is True, label

            page = PageState(page_num=1)
            page.is_born_digital = False
            page.native_words = []  # the words are gone on the next run
            page.detected_table_count = 0
            page.detected_table_bboxes = []
            page.scanned_table_evidence_failed = True
            page.d3_floor_png_ref = PNG_REF
            page.attempts = [restored]
            page.best_output = restored

            with patch.object(DocumentHandle, "__post_init__", lambda self: None):
                handle = DocumentHandle(path=arms[label]["pdf"], page_count=1)
            state = DocumentState(handle=handle)
            state.pages[1] = page

            assert finalized_page_records(state)[0].output.text == shipped, label

    def test_a_second_run_over_the_same_output_is_byte_identical(self, arms: dict) -> None:
        """Resume, the other half. Page one keeps ERROR status, so the per-page
        ledger gate does not skip it and the recovery is rebuilt from scratch --
        which is also a path that could re-embed on the next run."""
        for label in ("three", "one"):
            arm = arms[label]
            _run_process(arm["pdf"], arm["out_dir"])
            again = _surfaces(arm["pdf"], arm["out_dir"])
            assert again == arm["surfaces"], label
