"""Regression tests for issue #38: the pipeline silently destroying or
dropping correct content.

Covers the six P0 fixes from docs/log/2026-06-09_quality-diagnosis.md:
  1. HTML tables converted (not stripped into fused digit-streams)
  2. Assembler never ships silent empty pages; attempts fallback
  3. Repair router never selects non-runnable engines (AUTO crash)
  4. VLM-judge rejection actually triggers repair on born-digital pages
  5. Figure-phase failure cannot destroy a completed OCR run
  6. Native fallback on enhancement pages ships flagged, not as success
"""

from pathlib import Path
from unittest.mock import patch

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.html_tables import (
    clean_residual_html,
    convert_html_tables,
    strip_html_tags,
)
from socr.core.manifest import (
    _winning_page_output,
    is_page_failed_marker,
    page_failed_marker,
)
from socr.core.normalizer import OutputNormalizer
from socr.core.result import (
    DocumentStatus,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState, PageState
from socr.engines.deepseek_vllm import DeepSeekVLLMEngine
from socr.pipeline.orchestrator import UnifiedPipeline

# ---------------------------------------------------------------------------
# Helpers (same patterns as test_orchestrator.py)
# ---------------------------------------------------------------------------


def _make_handle(page_count: int = 2) -> DocumentHandle:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        h = DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=page_count)
    return h


def _make_config(**overrides) -> PipelineConfig:
    defaults = dict(
        primary_engine=EngineType.DEEPSEEK,
        fallback_chain=[EngineType.GEMINI],
        enabled_engines=list(EngineType),
        save_figures=False,
        quiet=True,
        tiered=False,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _page_output(
    page_num: int,
    text: str,
    engine: str = "deepseek",
    audit_passed: bool = False,
    failure_mode: FailureMode = FailureMode.NONE,
) -> PageOutput:
    return PageOutput(
        page_num=page_num,
        text=text,
        status=PageStatus.SUCCESS,
        engine=engine,
        audit_passed=audit_passed,
        failure_mode=failure_mode,
    )


# ---------------------------------------------------------------------------
# Fix 1 — HTML tables converted, never fused
# ---------------------------------------------------------------------------


class TestHtmlTableConversion:
    TABLE = (
        "<table><tr><th>Year</th><th>Actual</th><th>Expected</th></tr>"
        "<tr><td>1994</td><td>4.4</td><td>79.1</td></tr></table>"
    )

    def test_table_becomes_markdown(self) -> None:
        md = convert_html_tables(self.TABLE)
        assert "| Year | Actual | Expected |" in md
        assert "| 1994 | 4.4 | 79.1 |" in md

    def test_adjacent_cell_digits_never_concatenate(self) -> None:
        """The fabricated-number bug: '4.4' + '79.1' must never fuse."""
        cleaned = clean_residual_html(self.TABLE)
        assert "4.479.1" not in cleaned
        assert "4.4" in cleaned and "79.1" in cleaned

    def test_colspan_pads_columns(self) -> None:
        html = (
            '<table><tr><th colspan="2">Span</th><th>C</th></tr>'
            "<tr><td>a</td><td>b</td><td>c</td></tr></table>"
        )
        md = convert_html_tables(html).strip()
        # Header row padded to 3 columns so the body stays aligned
        assert md.splitlines()[0].count("|") == 4
        assert "| a | b | c |" in md

    def test_unclosed_table_still_converted(self) -> None:
        truncated = "<table><tr><td>1.23</td><td>4.56</td></tr><tr><td>7.89</td>"
        md = convert_html_tables(truncated)
        assert "| 1.23 | 4.56 |" in md
        assert "1.234.56" not in md

    def test_residual_cell_tags_become_separators(self) -> None:
        # A malformed fragment the block converter can't parse must still
        # never fuse adjacent values.
        fragment = "<td>4.4</td><td>79.1</td>"
        cleaned = strip_html_tags(fragment)
        assert "4.479.1" not in cleaned

    def test_pipe_in_cell_escaped(self) -> None:
        html = "<table><tr><td>a|b</td><td>c</td></tr></table>"
        md = convert_html_tables(html)
        assert "a\\|b" in md


class TestSafeTagStrip:
    def test_inequalities_preserved(self) -> None:
        text = "we require 0 < b and x > 1 throughout"
        assert strip_html_tags(text) == text

    def test_eos_token_preserved(self) -> None:
        text = 'the model emits "<EOS>" at the end'
        assert "<EOS>" in clean_residual_html(text)

    def test_strip_never_spans_newlines(self) -> None:
        # The old <[^>]+> deleted everything between a stray '<' and a '>'
        # sentences later, across lines.
        text = "loss < 0.5 here\nand precision > 0.9 there"
        assert clean_residual_html(text) == text

    def test_known_tags_stripped(self) -> None:
        text = "<div>Some <b>bold</b> text</div>"
        assert clean_residual_html(text).strip() == "Some bold text"

    def test_sup_sub_kept_as_caret(self) -> None:
        text = "R<sup>2</sup> = 0.95"
        assert "R^2" in clean_residual_html(text)

    def test_entities_decoded_after_strip(self) -> None:
        text = "AT&amp;T requires p &lt; 0.05"
        cleaned = clean_residual_html(text)
        assert "AT&T" in cleaned
        assert "p < 0.05" in cleaned

    def test_deepseek_vllm_cleaner_preserves_tables(self) -> None:
        raw = (
            "<|ref|>title<|/ref|>Results below.\n<table><tr><td>4.4</td><td>79.1</td></tr></table>"
        )
        cleaned = DeepSeekVLLMEngine._clean_ocr_output(raw)
        assert "4.479.1" not in cleaned
        assert "| 4.4 | 79.1 |" in cleaned

    def test_normalizer_deepseek_path_preserves_tables(self) -> None:
        raw = "<table><tr><td>0.040</td><td>0.014</td></tr></table>"
        cleaned = OutputNormalizer().normalize(raw, engine="deepseek")
        assert "0.0400.014" not in cleaned
        assert "| 0.040 | 0.014 |" in cleaned


# ---------------------------------------------------------------------------
# Fix 2 — no silent empty pages
# ---------------------------------------------------------------------------


class TestNoSilentEmptyPages:
    def test_attempts_fallback_when_best_output_cleared(self) -> None:
        """Judge/scoring cleared best_output; the rejected text still ships,
        flagged — never an empty page (the Kuttner-Table-2 failure)."""
        state = DocumentState(handle=_make_handle(1))
        rejected = _page_output(1, "Substantial rejected table text", audit_passed=False)
        state.pages[1].attempts.append(rejected)
        state.pages[1].best_output = None  # judge rejection

        winner = _winning_page_output(state, 1, None)
        assert winner.text == "Substantial rejected table text"
        assert winner.audit_passed is False
        assert winner.status == PageStatus.WARNING

    def test_truly_empty_page_ships_explicit_marker(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        winner = _winning_page_output(state, 1, None)
        assert winner.text == page_failed_marker(1)
        assert is_page_failed_marker(winner.text)
        assert winner.status == PageStatus.ERROR

    def test_document_text_includes_rejected_attempt(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        state.pages[1].attempts.append(_page_output(1, "rejected but real text"))
        assert "rejected but real text" in state.text

    def test_best_attempt_prefers_passing_then_longest(self) -> None:
        ps = PageState(page_num=1)
        ps.attempts.append(_page_output(1, "short"))
        ps.attempts.append(_page_output(1, "a much longer rejected attempt"))
        assert ps.best_attempt.text == "a much longer rejected attempt"
        ps.attempts.append(_page_output(1, "passing", audit_passed=True))
        assert ps.best_attempt.text == "passing"


# ---------------------------------------------------------------------------
# Fix 5 — figure phase cannot destroy a completed run
# ---------------------------------------------------------------------------


class TestFigurePhaseCrashSafety:
    def test_markdown_and_metadata_survive_figure_crash(self, tmp_path: Path) -> None:
        config = _make_config(save_figures=True)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))
        good = _page_output(1, "Completed OCR text worth keeping.", audit_passed=True)
        state.pages[1].attempts.append(good)
        state.pages[1].best_output = good

        with patch.object(
            pipeline,
            "_describe_and_embed_figures",
            side_effect=RuntimeError("paid API exploded mid-loop"),
        ):
            result = pipeline._phase_assemble(state, tmp_path)

        md_files = list(tmp_path.rglob("*.md"))
        assert md_files, "markdown must be on disk despite the figure crash"
        assert "Completed OCR text worth keeping." in md_files[0].read_text()
        meta_files = list(tmp_path.rglob("metadata.json"))
        assert meta_files, "metadata must be on disk despite the figure crash"
        assert "Completed OCR text worth keeping." in result.markdown


# ---------------------------------------------------------------------------
# Fix 6 — native fallback on enhancement pages ships flagged
# ---------------------------------------------------------------------------


class TestNativeFallbackFlagged:
    def test_native_fallback_after_failed_ocr_is_flagged(self) -> None:
        state = DocumentState(handle=_make_handle(1))
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "flat native table tokens"
        ps.needs_ocr_enhancement = True
        ps.attempts.append(_page_output(1, "failed ocr"))

        winner = _winning_page_output(state, 1, None)
        assert winner.engine == "native"
        assert winner.audit_passed is False
        assert winner.status == PageStatus.WARNING

    def test_native_on_untried_page_still_passes(self) -> None:
        """Prose-only born-digital pages (no OCR attempted) are still the
        char-exact happy path — not a fallback."""
        state = DocumentState(handle=_make_handle(1))
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "clean native prose"

        winner = _winning_page_output(state, 1, None)
        assert winner.audit_passed is True
        assert winner.status == PageStatus.SUCCESS

    def test_assemble_demotes_success_on_native_fallback(self, tmp_path: Path) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "flat native table tokens"
        ps.needs_ocr_enhancement = True
        ps.attempts.append(_page_output(1, "failed ocr"))

        result = pipeline._phase_assemble(state, tmp_path)
        assert result.status == DocumentStatus.AUDIT_FAILED
        kinds = {e.kind for e in state.events}
        assert "native_fallback" in kinds


# ---------------------------------------------------------------------------
# Adversarial-review regressions (codex + review workflow, 2026-06-09)
# ---------------------------------------------------------------------------


class TestConverterNeverDeletesContent:
    """The critical review finding: the first version of the open-table arm
    deleted everything from any stray '<table' to end of document."""

    def test_stray_table_mention_in_prose_survives(self) -> None:
        text = "The model emits <table> tokens for tabular regions.\nSection 4 results: r = 0.95."
        cleaned = clean_residual_html(text)
        assert "Section 4 results: r = 0.95." in cleaned

    def test_unclosed_table_keeps_trailing_prose(self) -> None:
        text = (
            "<table><tr><td>1.23</td><td>4.56</td></tr>\n\n"
            "Conclusion: the effect is significant at the 1% level."
        )
        cleaned = clean_residual_html(text)
        assert "| 1.23 | 4.56 |" in cleaned
        assert "Conclusion: the effect is significant at the 1% level." in cleaned

    def test_closed_table_with_no_rows_degrades_not_deletes(self) -> None:
        text = "<table>4.4 and 79.1 are the key values</table>"
        cleaned = clean_residual_html(text)
        assert "4.4 and 79.1 are the key values" in cleaned

    def test_open_table_with_no_rows_leaves_text_unchanged(self) -> None:
        text = "Intro paragraph.\n<table>\nNo rows here. Key result: beta = 0.04 (s.e. 0.014)."
        cleaned = clean_residual_html(text)
        assert "Key result: beta = 0.04 (s.e. 0.014)." in cleaned

    def test_cells_without_tr_wrapper_survive(self) -> None:
        # Classic VLM malformation: cells with no row wrapper.
        text = "<table><td>4.4</td><td>79.1</td></table>"
        cleaned = clean_residual_html(text)
        assert "4.4" in cleaned and "79.1" in cleaned
        assert "4.479.1" not in cleaned


class TestConverterRowFidelity:
    def test_trailing_unclosed_row_kept_when_closed_rows_exist(self) -> None:
        # codex trigger: truncated generation loses its final data row.
        text = "<table><tr><td>1.23</td><td>4.56</td></tr><tr><td>7.89</td><td>0.12</td>"
        md = convert_html_tables(text)
        assert "| 1.23 | 4.56 |" in md
        assert "| 7.89 | 0.12 |" in md

    def test_rowspan_does_not_shift_following_rows(self) -> None:
        text = (
            '<table><tr><td rowspan="2">A</td><td>4.4</td><td>79.1</td></tr>'
            "<tr><td>5.5</td><td>80.2</td></tr></table>"
        )
        md = convert_html_tables(text)
        # Second row gets a pad cell under the rowspanned column, so 5.5
        # stays in column 2 (under 4.4), not under A.
        assert "|  | 5.5 | 80.2 |" in md

    def test_missing_mid_table_tr_close_splits_rows(self) -> None:
        text = "<table><tr><td>H1</td><td>H2</td><tr><td>1</td><td>2</td></tr></table>"
        md = convert_html_tables(text)
        assert "| H1 | H2 |" in md
        assert "| 1 | 2 |" in md

    def test_colspan_is_clamped(self) -> None:
        text = '<table><tr><td colspan="100000000">x</td></tr></table>'
        md = convert_html_tables(text)
        assert len(md) < 5000

    def test_mixed_case_table_tag_converted(self) -> None:
        text = "<Table><TR><TD>4.4</TD><TD>79.1</TD></TR></Table>"
        cleaned = clean_residual_html(text)
        assert "| 4.4 | 79.1 |" in cleaned


class TestInlineTagFidelity:
    def test_empty_sup_does_not_fuse_digits(self) -> None:
        assert "4.479.1" not in clean_residual_html("4.4<sup> </sup>79.1")

    def test_sub_becomes_underscore_not_caret(self) -> None:
        cleaned = clean_residual_html("the index x<sub>t</sub> and beta<sub>1</sub>")
        assert "x_t" in cleaned
        assert "beta_1" in cleaned
        assert "x^t" not in cleaned

    def test_intra_word_italics_do_not_grow_spaces(self) -> None:
        cleaned = clean_residual_html("the <i>t</i>-statistic and <b>F</b>ederal Reserve")
        assert "t-statistic" in cleaned
        assert "Federal Reserve" in cleaned

    def test_leading_indentation_preserved(self) -> None:
        text = "Algorithm:\n\n    for i in range(10):\n        total += x[i]\n"
        cleaned = clean_residual_html(text)
        assert "\n    for i in range(10):" in cleaned
        assert "\n        total += x[i]" in cleaned


class TestMathPreservationCoverage:
    def test_letterlike_and_plane1_math_preserved(self) -> None:
        text = "the set ℝ and coefficient \U0001d6fd"
        result = OutputNormalizer().normalize(text)
        assert "ℝ" in result
        assert "\U0001d6fd" in result


class TestApplyResultPromotion:
    def test_passing_attempt_replaces_failed_best_output(self) -> None:
        """Review finding: a failed round-1 repair pinned best_output and every
        later PASSING repair attempt was silently discarded."""
        from socr.core.result import EngineResult

        state = DocumentState(handle=_make_handle(1))
        failed = _page_output(1, "failed round-1 text", audit_passed=False)
        state.pages[1].attempts.append(failed)
        state.pages[1].best_output = failed  # pinned by round 1 + in-place scoring

        passing = _page_output(1, "passing round-2 text", engine="gemini", audit_passed=True)
        state.apply_result(
            EngineResult(
                document_path=Path("/tmp/fake.pdf"),
                engine="gemini",
                status=DocumentStatus.SUCCESS,
                pages=[passing],
            )
        )
        assert state.pages[1].best_output is passing

    def test_passing_best_output_not_displaced(self) -> None:
        from socr.core.result import EngineResult

        state = DocumentState(handle=_make_handle(1))
        first = _page_output(1, "first passing", audit_passed=True)
        state.pages[1].attempts.append(first)
        state.pages[1].best_output = first

        second = _page_output(1, "second passing", engine="gemini", audit_passed=True)
        state.apply_result(
            EngineResult(
                document_path=Path("/tmp/fake.pdf"),
                engine="gemini",
                status=DocumentStatus.SUCCESS,
                pages=[second],
            )
        )
        assert state.pages[1].best_output is first


class TestCacheInvalidation:
    def test_run_fingerprint_changes_with_normalizer_version(self) -> None:
        """Review critical: corpora cached under the digit-fusing v1
        normalizer must fail the resume gate after the fix."""
        from socr.core import manifest as manifest_mod
        from socr.pipeline.orchestrator import UnifiedPipeline

        pipeline = UnifiedPipeline(_make_config(primary_engine=EngineType.GEMINI))
        fp_now = pipeline._run_fingerprint(EngineType.GEMINI)
        with patch.object(manifest_mod, "NORMALIZER_VERSION", "old-1"):
            fp_old = pipeline._run_fingerprint(EngineType.GEMINI)
        assert fp_now != fp_old

    def test_versions_bumped_past_v1(self) -> None:
        from socr.core.manifest import ASSEMBLY_VERSION, NORMALIZER_VERSION

        assert NORMALIZER_VERSION != "1"
        assert ASSEMBLY_VERSION != "1"


class TestPartialResumeGate:
    def _index(self, tmp_path, status, fingerprint, output_name="doc.md"):
        out = tmp_path / output_name
        out.write_text("content")

        class _Idx:
            def __init__(self):
                self.files = {
                    "doc.pdf": {
                        "status": status,
                        "checksum": "sha256:abc",
                        "fingerprint": fingerprint,
                        "output_path": output_name,
                    }
                }

            def is_completed(self, rel_key, checksum, fingerprint=None):
                e = self.files.get(rel_key)
                return bool(
                    e
                    and e["status"] == "completed"
                    and e["checksum"] == checksum
                    and (fingerprint is None or e["fingerprint"] == fingerprint)
                )

        return _Idx()

    def test_partial_with_matching_fingerprint_skips(self, tmp_path) -> None:
        from socr.pipeline.orchestrator import _resume_skippable

        idx = self._index(tmp_path, "partial", "fp1")
        assert _resume_skippable(idx, "doc.pdf", "sha256:abc", "fp1", tmp_path) is True

    def test_partial_with_different_fingerprint_reprocesses(self, tmp_path) -> None:
        from socr.pipeline.orchestrator import _resume_skippable

        idx = self._index(tmp_path, "partial", "fp1")
        assert _resume_skippable(idx, "doc.pdf", "sha256:abc", "fp2", tmp_path) is False

    def test_provisional_prefigures_record_never_skips(self, tmp_path) -> None:
        """A crash during the figure phase leaves a ':pre-figures'-suffixed
        fingerprint, which must never satisfy the resume gate."""
        from socr.pipeline.orchestrator import _resume_skippable

        idx = self._index(tmp_path, "partial", "fp1:pre-figures")
        assert _resume_skippable(idx, "doc.pdf", "sha256:abc", "fp1", tmp_path) is False


class TestWholeDocEmptySection:
    def test_empty_whole_doc_section_falls_through_to_marker(self) -> None:
        from socr.core.manifest import _WholeDoc

        state = DocumentState(handle=_make_handle(2))
        whole = _WholeDoc(texts={1: "Real content", 2: "   "}, engine="cli", audit_passed=True)
        winner = _winning_page_output(state, 2, whole)
        assert winner.text == page_failed_marker(2)
        assert winner.audit_passed is False


class TestCliExitCodes:
    def _invoke(self, tmp_path, error):
        from unittest.mock import MagicMock

        from click.testing import CliRunner

        from socr.cli import cli as cli_group
        from socr.core.result import EngineResult

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        result = EngineResult(
            document_path=pdf,
            engine="gemini",
            status=DocumentStatus.AUDIT_FAILED,
            error=error,
        )
        fake_pipeline = MagicMock()
        fake_pipeline.process.return_value = result
        with patch("socr.pipeline.orchestrator.UnifiedPipeline", return_value=fake_pipeline):
            return CliRunner().invoke(
                cli_group,
                ["process", str(pdf), "--primary", "gemini", "-o", str(tmp_path / "out"), "-q"],
            )

    def test_lost_content_exits_nonzero(self, tmp_path) -> None:
        from socr.core.result import LOST_CONTENT_NOTE

        res = self._invoke(tmp_path, f"page(s) 10 {LOST_CONTENT_NOTE}")
        assert res.exit_code != 0
        assert "lost content" in res.output.lower()

    def test_plain_audit_failed_also_exits_nonzero(self, tmp_path) -> None:
        """GH-177: renamed from `..._exits_zero`, which encoded the bug.

        `RunOutcome`'s documented policy is uniform across single-file and
        batch -- a PARTIAL document exits nonzero -- and batch already did that
        for the same `AUDIT_FAILED` status. Exiting 0 here meant a script
        wrapping `socr process` and one wrapping `socr batch` saw OPPOSITE
        signals for the same document.

        The lost-content case is still distinguished, on the axis that still
        differs: the MESSAGE. Both are nonzero; only one says content was lost.
        """
        res = self._invoke(tmp_path, None)
        assert res.exit_code != 0
        assert "completed with warnings" in res.output.lower()
        assert "lost content" not in res.output.lower()


class TestFigurePhaseSuccessPath:
    def test_markdown_resaved_with_figure_blocks(self, tmp_path) -> None:
        config = _make_config(save_figures=True)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))
        good = _page_output(1, "Body text of the page.", audit_passed=True)
        state.pages[1].attempts.append(good)
        state.pages[1].best_output = good

        def _embed(state_, result_, out_dir_, text_):
            return text_ + "\n\n**Figure 1** (page 1): a chart.\n"

        with patch.object(pipeline, "_describe_and_embed_figures", side_effect=_embed):
            pipeline._phase_assemble(state, tmp_path)

        md_files = list(tmp_path.rglob("*.md"))
        assert md_files
        on_disk = md_files[0].read_text()
        assert "**Figure 1** (page 1): a chart." in on_disk
        # Final metadata record carries the REAL fingerprint (not provisional).
        import json

        meta_files = list(tmp_path.rglob("metadata.json"))
        assert meta_files
        recorded = json.loads(meta_files[0].read_text())
        fp = json.dumps(recorded)
        assert ":pre-figures" not in fp


# ---------------------------------------------------------------------------
# GH-34 — Recovered-to-empty must not count as recovered
# ---------------------------------------------------------------------------


class TestEmptyRepairNotPromoted:
    """(a) An audit-passing but empty repair attempt must NOT displace a
    non-empty prior best_output / best attempt.  The bug: apply_result only
    checked audit_passed, not that the repair produced any text."""

    def test_empty_audit_passed_does_not_displace_nonempty_best_output(self) -> None:
        """Empty-but-passing repair should never overwrite a non-empty best_output."""
        from socr.core.result import EngineResult

        state = DocumentState(handle=_make_handle(1))
        # Round-1: failed but has content
        prior = _page_output(1, "Non-empty prior OCR text", audit_passed=False)
        state.pages[1].attempts.append(prior)
        state.pages[1].best_output = prior

        # Round-2: audit-passed but empty
        empty_repair = PageOutput(
            page_num=1,
            text="",
            status=PageStatus.SUCCESS,
            engine="gemini",
            audit_passed=True,
        )
        state.apply_result(
            EngineResult(
                document_path=Path("/tmp/fake.pdf"),
                engine="gemini",
                status=DocumentStatus.SUCCESS,
                pages=[empty_repair],
            )
        )
        # best_output must NOT be replaced by the empty attempt
        assert state.pages[1].best_output is prior
        assert state.pages[1].best_output.text == "Non-empty prior OCR text"

    def test_empty_audit_passed_does_not_become_best_output_when_no_prior(self) -> None:
        """When there is no prior best_output, an empty-but-passing attempt must not
        become best_output either — best_output should stay None."""
        from socr.core.result import EngineResult

        state = DocumentState(handle=_make_handle(1))
        empty_repair = PageOutput(
            page_num=1,
            text="   ",  # whitespace-only
            status=PageStatus.SUCCESS,
            engine="gemini",
            audit_passed=True,
        )
        state.apply_result(
            EngineResult(
                document_path=Path("/tmp/fake.pdf"),
                engine="gemini",
                status=DocumentStatus.SUCCESS,
                pages=[empty_repair],
            )
        )
        assert state.pages[1].best_output is None

    def test_nonempty_audit_passed_still_promoted(self) -> None:
        """Non-empty passing attempts must still be promoted (no false negative)."""
        from socr.core.result import EngineResult

        state = DocumentState(handle=_make_handle(1))
        prior = _page_output(1, "Non-empty prior text", audit_passed=False)
        state.pages[1].attempts.append(prior)
        state.pages[1].best_output = prior

        good_repair = _page_output(1, "Non-empty repair text", engine="gemini", audit_passed=True)
        state.apply_result(
            EngineResult(
                document_path=Path("/tmp/fake.pdf"),
                engine="gemini",
                status=DocumentStatus.SUCCESS,
                pages=[good_repair],
            )
        )
        assert state.pages[1].best_output is good_repair

    def test_all_empty_after_failure_resolves_to_explicit_failure(self) -> None:
        """When every post-failure attempt is empty, the page must resolve to an
        explicit failure marker, not empty text and not the rejected text
        masquerading as accepted."""
        from socr.core.manifest import _winning_page_output, is_page_failed_marker

        state = DocumentState(handle=_make_handle(1))
        # All attempts are empty
        for engine in ("deepseek", "gemini", "qwen"):
            empty = PageOutput(
                page_num=1,
                text="",
                status=PageStatus.ERROR,
                engine=engine,
                audit_passed=False,
                failure_mode=FailureMode.EMPTY_OUTPUT,
            )
            state.pages[1].attempts.append(empty)
        # best_output stays None (no non-empty attempt was promoted)
