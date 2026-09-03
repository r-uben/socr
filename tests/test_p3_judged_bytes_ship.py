"""P3 (GH-513 follow-up, docs/log/2026-09-01_conceptual-revision.md): shipped
bytes are judged bytes.

Before this ticket, ``_phase_agentic`` MUTATED the accepted output's text
after ``route_page`` returned (``repair_table_headers_on_page``,
orchestrator.py ~3592-3624) and then re-ran a string-only recheck on the
mutated text (``table_output_defect``, ~3637-3662, ``post_route_recheck``).
So the text a real page could ship as SUCCESS was not necessarily the text
the judge accepted — the judge saw one candidate, the reader got another.

After the fix, header repair happens exactly once, INSIDE
``NativeTableVerifierJudge`` (``_maybe_repair_collapsed_headers``), BEFORE
the judge's verdict, and there is no second post-route mutation site. This
test proves the invariant end to end through the real pipeline: whatever
text the judge accepted is byte-identical to the provisional in-loop flush,
the ``pages/00001.md`` fragment on disk, the authoritative
``_rewrite_all_fragments`` rewrite, and page 1 of the final stitched
document.

Hermetic (CLAUDE.md house rules):
  - ``_available_engines_for_agentic`` is patched to a single local profile so
    the ladder never depends on an installed provider.
  - ``_resolve_judge_model`` is patched to "" so no code path can construct a
    real ``OllamaVisionJudge`` and POST to it.
  - Backend probes (``probe_ollama_idle``, ``probe_openai_server_idle``,
    ``_probe_backend_idle``) are patched so no installed provider or Ollama
    state is consulted.
  - ``judge_backend="heuristic"`` avoids the VLM judge entirely; the
    deterministic ``NativeTableVerifierJudge`` (real, undstubbed) is what
    actually decides this page.
  - ``dual_pass_tables=False`` and ``escalate_ambiguous_tables=False`` turn
    off the PP-3 in-loop crop reread and the GH-96 escalation lane — both are
    unrelated post-judge text transforms this test must not entangle with the
    one thing under test (P3's post-route mutation). P5 gives those their own
    coverage in tests/test_p5_reread_on_signal.py.
  - The OCR call itself (``UnifiedPipeline._run_engine_on_pages``) is
    replaced with a stub returning one controlled candidate reading — no
    engine subprocess, no network.
  - The page is forced through the OCR ladder (rather than the born-digital
    native-fallback lane) by patching ``_is_agentic_trusted_native`` to
    False, so the real ``route_page`` / ``NativeTableVerifierJudge`` chain
    actually runs on it.

This deliberately does NOT parametrize over an empty provider list: with no
provider, ``route_page`` cannot produce a judge acceptance at all, so there is
nothing to pin bytes against. That is P5 escalation-lane territory, not P3.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from ocr_output_contract import assemble_pages, split_native_pages  # noqa: E402

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.providers import PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.pipeline.agentic import NativeTableVerifierJudge  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture: a real fitz page whose header row is too-narrow / spanning, so the
# in-judge repair (``_maybe_repair_collapsed_headers``) genuinely CHANGES the
# candidate text before the verdict. Mirrors
# tests/test_native_table_verifier.py::test_repairs_too_narrow_spanning_header_before_judging.
# ---------------------------------------------------------------------------


def _build_fixture_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=700, height=900)
    tokens: list[tuple[float, float, str]] = [
        (80.0, 150.0, "Near"),
        (80.0, 175.0, "outcome"),
        (80.0, 315.0, "Far"),
        (80.0, 335.0, "outcome"),
    ]
    data_xs = [150.0, 215.0, 280.0, 345.0]
    for x, ordinal in zip(data_xs, ["(1)", "(2)", "(3)", "(4)"]):
        tokens.append((110.0, x, ordinal))
    for x, value in zip(data_xs, ["-4.8", None, "-4.1", "-0.2"]):
        if value is not None:
            tokens.append((140.0, x, value))
    for x, value in zip(data_xs, ["0.1", "0.2", "0.3", "0.4"]):
        tokens.append((170.0, x, value))
    for y, x, word in tokens:
        page.insert_text((x, y), word, fontsize=9)
    pdf_path = tmp_path / "doc.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# The OCR candidate: a caption sentence (keeps the heuristic garbage-ratio
# check comfortably under threshold -- a table-only blob of "|" characters
# alone trips it) followed by a too-narrow/spanning header row that
# ``_maybe_repair_collapsed_headers`` expands into distinct header columns.
_CANDIDATE_TEXT = (
    "Table 2 reports estimated treatment effects for the near and far outcome "
    "measures used throughout this regression analysis of the intervention.\n\n"
    "| | Dependent variable: | | |\n"
    "| --- | --- | --- | --- |\n"
    "| | Near outcome | | Far outcome |\n"
    "| | (1) | (2) | (3) | (4) |\n"
    "| Signal | -4.8 | | -4.1 | -0.2 |\n"
    "| Control | 0.1 | 0.2 | 0.3 | 0.4 |"
)


def _run_pipeline(tmp_path: Path):
    pdf_path = _build_fixture_pdf(tmp_path)
    out_dir = tmp_path / "out"

    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        save_figures=False,
        write_manifest=False,
        native_first=True,
        dual_pass_tables=False,
        escalate_ambiguous_tables=False,
        table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md
    )
    pipeline = UnifiedPipeline(config)

    def _stub_run_engine_on_pages(
        state, page_nums, enhancement_pages, engine_type, label, profile=None
    ):
        return [
            PageOutput(
                page_num=page_nums[0],
                text=_CANDIDATE_TEXT,
                status=PageStatus.SUCCESS,
                engine="qwen",
                confidence=0.9,
            )
        ]

    captured_accepted_text: dict[str, str] = {}
    original_assess = NativeTableVerifierJudge.assess

    def _spy_assess(self, output, provider):
        decision = original_assess(self, output, provider)
        if decision.accept:
            captured_accepted_text["text"] = output.text
        return decision

    flushed_bodies: list[tuple[int, str]] = []
    original_flush = UnifiedPipeline._flush_page_fragment

    def _spy_flush(self, state, page_num, body, output_dir, **kwargs):
        flushed_bodies.append((page_num, body))
        return original_flush(self, state, page_num, body, output_dir, **kwargs)

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_is_agentic_trusted_native", return_value=False),
        patch.object(pipeline, "_probe_backend_idle", return_value=True),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
        patch("socr.pipeline.orchestrator.probe_openai_server_idle", return_value=True),
        patch.object(
            UnifiedPipeline,
            "_run_engine_on_pages",
            autospec=True,
            side_effect=lambda self, *a, **k: _stub_run_engine_on_pages(*a, **k),
        ),
        patch.object(NativeTableVerifierJudge, "assess", _spy_assess),
        patch.object(UnifiedPipeline, "_flush_page_fragment", _spy_flush),
    ):
        result = pipeline.process(pdf_path, out_dir)

    return result, out_dir, captured_accepted_text, flushed_bodies


class TestJudgedBytesShip:
    def test_accepted_text_is_captured_and_nontrivial(self, tmp_path: Path) -> None:
        """Setup precondition: the judge must actually accept a candidate, and
        repair must actually have changed it from the raw OCR candidate --
        otherwise this test proves nothing about the post-route mutation."""
        _result, _out_dir, captured, _flushed = _run_pipeline(tmp_path)
        assert "text" in captured, "the judge never accepted a candidate on this page"
        assert captured["text"] != _CANDIDATE_TEXT, (
            "the in-judge header repair must have changed the candidate for "
            "this test to distinguish judged bytes from raw OCR bytes"
        )

    def test_provisional_flush_matches_accepted_text(self, tmp_path: Path) -> None:
        _result, _out_dir, captured, flushed = _run_pipeline(tmp_path)
        p1_flushes = [body for page_num, body in flushed if page_num == 1]
        assert p1_flushes, "page 1 was never flushed"
        for body in p1_flushes:
            assert body == captured["text"], (
                "every provisional in-loop flush body must equal the judge-accepted text exactly"
            )

    def test_page_fragment_on_disk_matches_accepted_text(self, tmp_path: Path) -> None:
        _result, out_dir, captured, _flushed = _run_pipeline(tmp_path)
        fragment_candidates = list(out_dir.rglob("pages/00001.md"))
        assert fragment_candidates, f"no pages/00001.md found under {out_dir}"
        fragment_text = fragment_candidates[0].read_text(encoding="utf-8")
        assert fragment_text == captured["text"], (
            "the pages/00001.md fragment after authoritative rewrite must equal "
            "the judge-accepted text exactly"
        )

    def test_final_markdown_page_body_equals_accepted_text_exactly(self, tmp_path: Path) -> None:
        _result, out_dir, captured, _flushed = _run_pipeline(tmp_path)
        md_candidates = [
            p for p in out_dir.rglob("*.md") if "pages" not in p.parts and p.name != "README.md"
        ]
        assert md_candidates, f"no final assembled markdown found under {out_dir}"
        final_markdown = md_candidates[0].read_text(encoding="utf-8")

        pages = split_native_pages(final_markdown)
        assert len(pages) >= 1
        assert pages[0] == captured["text"], (
            "page 1 of the final stitched document must be BYTE-IDENTICAL to "
            "the text the judge accepted -- not merely contain it"
        )

        # The fragment-stitch result (assemble_pages over the same per-page
        # bodies) must equal the saved final markdown.
        restitched = assemble_pages(pages)
        assert restitched == final_markdown

    def test_no_post_route_defect_event_appears_for_this_page(self, tmp_path: Path) -> None:
        """The retired ``post_route_recheck`` site must not fire even
        transiently while both fixes are being landed together (P3 deletes
        it outright)."""
        _result, out_dir, _captured, _flushed = _run_pipeline(tmp_path)
        audit_log_candidates = list(out_dir.rglob("audit_log.json"))
        assert audit_log_candidates, f"no audit_log.json found under {out_dir}"
        audit_log = json.loads(audit_log_candidates[0].read_text(encoding="utf-8"))
        events = [e for e in audit_log.get("events", []) if e.get("page_num") == 1]
        post_route_sites = [
            e for e in events if e.get("data", {}).get("site") == "post_route_recheck"
        ]
        assert post_route_sites == []
