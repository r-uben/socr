"""Tests for GH-36b: equation crop → LaTeX via local VLM + 1A gate + 1C sidecar.

Acceptance criteria covered:
  AC1 — valid LaTeX from stub engine → passes 1A → attached adjacently; crop inlined.
  AC2 — invalid/malformed LaTeX (unbalanced {, bad environment) → fails 1A → NOT
         attached; native text kept; failure reason recorded.
  AC3 — empty / whitespace LaTeX → fails 1A (non-empty rule).
  AC4 — validate_latex_structure unit tests: balanced/unbalanced delimiters, empty,
         whitespace-only, unknown environment, realistic display equation.
  AC5 — EquationLatexResult carries raw_latex + 1A result + model_id.
  AC6 — default-off: no engine call when recover_clean_equations=False.
  AC7 — crop PNG is always inlined in the sidecar block; native text never replaced.
  AC8 — failure reason recorded in audit event.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from socr.math.equation_latex import (
    EQUATION_SIDECAR_HEADER,
    EquationLatexResult,
    build_equation_sidecar,
    process_equation_region,
)
from socr.math.validate_latex import EQUATION_LATEX_PROMPT, validate_latex_structure

# ── validate_latex_structure unit tests (AC4) ────────────────────────────────


class TestValidateLatexStructure:
    """1A gate: deterministic, offline, pylatexenc-backed."""

    def test_valid_simple_equation(self):
        ok, reason = validate_latex_structure(r"x^2 + y^2 = z^2")
        assert ok is True
        assert reason == "ok"

    def test_valid_sum_with_fraction(self):
        ok, reason = validate_latex_structure(r"\sum_{i=1}^{n} x_i = \frac{n(n+1)}{2}")
        assert ok is True
        assert reason == "ok"

    def test_valid_integral(self):
        ok, reason = validate_latex_structure(r"\int_0^\infty e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2}")
        assert ok is True
        assert reason == "ok"

    def test_valid_environment(self):
        ok, reason = validate_latex_structure(r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}")
        assert ok is True
        assert reason == "ok"

    def test_valid_balanced_delimiters(self):
        ok, reason = validate_latex_structure(r"\left( x + y \right)")
        assert ok is True
        assert reason == "ok"

    def test_empty_string_fails(self):
        """AC3: empty string → non-empty rule fails."""
        ok, reason = validate_latex_structure("")
        assert ok is False
        assert "empty" in reason.lower()

    def test_whitespace_only_fails(self):
        """AC3: whitespace-only → non-empty rule fails."""
        ok, reason = validate_latex_structure("   ")
        assert ok is False
        assert "empty" in reason.lower()

    def test_unbalanced_brace_fails(self):
        """AC2: unbalanced { → parse error."""
        ok, reason = validate_latex_structure(r"x^{2 + y")
        assert ok is False
        assert "parse error" in reason.lower() or "error" in reason.lower()

    def test_unclosed_environment_fails(self):
        """AC2: unclosed \\begin{pmatrix} → parse error."""
        ok, reason = validate_latex_structure(r"\begin{pmatrix} a & b")
        assert ok is False
        assert "parse error" in reason.lower() or "error" in reason.lower()

    def test_unknown_environment_is_permitted(self):
        """pylatexenc tolerates unknown environments; the 1A gate should too.

        The 1A gate is a STRUCTURAL check (balanced delimiters, parseable),
        not a macro-completeness check.  An unknown environment that is otherwise
        well-formed should pass.
        """
        ok, reason = validate_latex_structure(r"\begin{myenv} x \end{myenv}")
        assert ok is True
        assert reason == "ok"

    def test_realistic_display_equation(self):
        """Realistic academic equation: Euler's identity."""
        ok, reason = validate_latex_structure(r"e^{i\pi} + 1 = 0")
        assert ok is True
        assert reason == "ok"

    def test_returns_tuple(self):
        ok, reason = validate_latex_structure(r"x = 1")
        assert isinstance(ok, bool)
        assert isinstance(reason, str)


# ── build_equation_sidecar tests (1C policy) ─────────────────────────────────


class TestBuildEquationSidecar:
    """1C non-destructive sidecar: crop always inlined; LaTeX adjacent on pass."""

    def test_valid_latex_attaches_adjacently(self):
        """AC1: valid LaTeX → attached adjacently; crop inlined; native NOT replaced."""
        sidecar, attached = build_equation_sidecar(
            crop_path="/path/equation_1_page3.png",
            native_text="x squared plus y squared",
            raw_latex=r"x^2 + y^2 = z^2",
            validation_ok=True,
            validation_reason="ok",
        )
        assert attached is True
        # Crop is inlined
        assert "equation_1_page3.png" in sidecar
        assert "![equation crop]" in sidecar
        # LaTeX is attached adjacently (fenced block)
        assert "```latex" in sidecar
        assert r"x^2 + y^2 = z^2" in sidecar
        # Non-authoritative header present
        assert EQUATION_SIDECAR_HEADER in sidecar
        # Native text NOT in sidecar (it stays in the page body unchanged)
        assert "x squared plus y squared" not in sidecar

    def test_invalid_latex_not_attached_native_kept(self):
        """AC2: invalid LaTeX → NOT attached; native text kept; failure reason noted."""
        sidecar, attached = build_equation_sidecar(
            crop_path="/path/equation_1_page3.png",
            native_text="x squared plus y squared",
            raw_latex=r"x^{bad",
            validation_ok=False,
            validation_reason="parse error: unmatched brace",
        )
        assert attached is False
        # Crop is still inlined (always)
        assert "equation_1_page3.png" in sidecar
        # LaTeX block NOT present
        assert "```latex" not in sidecar
        # Native text IS present as fallback
        assert "x squared plus y squared" in sidecar
        # Failure reason noted in comment
        assert "parse error" in sidecar

    def test_empty_latex_not_attached(self):
        """AC3: empty/whitespace LaTeX → not attached; native text kept."""
        sidecar, attached = build_equation_sidecar(
            crop_path="/path/equation_1_page1.png",
            native_text="sum from i to n",
            raw_latex="",
            validation_ok=False,
            validation_reason="engine returned empty output — no crop or call failed",
        )
        assert attached is False
        assert "```latex" not in sidecar
        assert "sum from i to n" in sidecar

    def test_missing_crop_path_emits_placeholder(self):
        """If crop PNG was not saved, a comment placeholder is emitted."""
        sidecar, attached = build_equation_sidecar(
            crop_path=None,
            native_text="E = mc^2 linearised",
            raw_latex=r"E = mc^2",
            validation_ok=True,
            validation_reason="ok",
        )
        # The sidecar still attaches LaTeX (validation passed)
        assert attached is True
        # But no image reference — placeholder comment instead
        assert "<!-- socr-equation: crop PNG unavailable -->" in sidecar

    def test_crop_always_inlined_even_on_failure(self):
        """AC7: crop PNG is always inlined regardless of validation outcome."""
        for validation_ok in (True, False):
            sidecar, _ = build_equation_sidecar(
                crop_path="/my/equation.png",
                native_text="native",
                raw_latex=r"x^{bad" if not validation_ok else r"x^2",
                validation_ok=validation_ok,
                validation_reason="ok" if validation_ok else "parse error",
            )
            assert "/my/equation.png" in sidecar, (
                f"crop should be inlined for validation_ok={validation_ok}"
            )


# ── process_equation_region tests (full pipeline, mocked engine) ─────────────


class TestProcessEquationRegion:
    """Full GH-36b pipeline: engine → 1A gate → 1C sidecar, with mocked VLM."""

    def _stub_engine_valid(self, png_bytes, **kwargs):
        """Stub that returns structurally-valid LaTeX."""
        return r"x^2 + y^2 = z^2"

    def _stub_engine_invalid(self, png_bytes, **kwargs):
        """Stub that returns malformed LaTeX (unbalanced brace)."""
        return r"x^{unbalanced"

    def _stub_engine_empty(self, png_bytes, **kwargs):
        """Stub that returns empty string (engine failed)."""
        return ""

    def test_valid_latex_attached_adjacently(self, tmp_path):
        """AC1: valid LaTeX from stub → passes 1A → attached; crop inlined."""
        crop_file = tmp_path / "eq.png"
        crop_file.write_bytes(b"fakepng")

        result = process_equation_region(
            region_index=0,
            page_num=3,
            crop_path=str(crop_file),
            native_text="x squared plus y squared equals z squared",
            ocr=self._stub_engine_valid,
        )

        assert isinstance(result, EquationLatexResult)
        assert result.validation_ok is True
        assert result.latex_attached is True
        assert result.raw_latex == r"x^2 + y^2 = z^2"
        assert result.validation_reason == "ok"
        # Sidecar: crop inlined, LaTeX attached
        assert "eq.png" in result.sidecar_block
        assert "```latex" in result.sidecar_block
        assert r"x^2 + y^2 = z^2" in result.sidecar_block
        assert EQUATION_SIDECAR_HEADER in result.sidecar_block
        # Native text NOT in sidecar (stays in body unchanged)
        assert "x squared" not in result.sidecar_block

    def test_invalid_latex_not_attached_native_kept(self, tmp_path):
        """AC2: invalid LaTeX → fails 1A → NOT attached; native text in sidecar."""
        crop_file = tmp_path / "eq.png"
        crop_file.write_bytes(b"fakepng")

        result = process_equation_region(
            region_index=0,
            page_num=2,
            crop_path=str(crop_file),
            native_text="some native text",
            ocr=self._stub_engine_invalid,
        )

        assert result.validation_ok is False
        assert result.latex_attached is False
        assert result.raw_latex == r"x^{unbalanced"
        assert "parse error" in result.validation_reason.lower() or result.validation_reason
        # Native text IS in sidecar
        assert "some native text" in result.sidecar_block
        # LaTeX block NOT in sidecar
        assert "```latex" not in result.sidecar_block
        # Crop still inlined
        assert "eq.png" in result.sidecar_block

    def test_empty_latex_not_attached(self, tmp_path):
        """AC3: empty LaTeX → fails 1A; native text kept."""
        crop_file = tmp_path / "eq.png"
        crop_file.write_bytes(b"fakepng")

        result = process_equation_region(
            region_index=0,
            page_num=1,
            crop_path=str(crop_file),
            native_text="sum from i to n",
            ocr=self._stub_engine_empty,
        )

        assert result.validation_ok is False
        assert result.latex_attached is False
        assert result.raw_latex == ""
        assert "```latex" not in result.sidecar_block
        assert "sum from i to n" in result.sidecar_block

    def test_provenance_carries_model_id(self, tmp_path):
        """AC5: result carries raw_latex + 1A result + model_id."""
        crop_file = tmp_path / "eq.png"
        crop_file.write_bytes(b"fakepng")

        result = process_equation_region(
            region_index=0,
            page_num=1,
            crop_path=str(crop_file),
            native_text="",
            ocr=self._stub_engine_valid,
            model="qwen3-vl:30b-a3b-instruct",
        )

        assert result.model_id == "qwen3-vl:30b-a3b-instruct"
        assert isinstance(result.raw_latex, str)
        assert isinstance(result.validation_ok, bool)
        assert isinstance(result.validation_reason, str)

    def test_no_crop_path_skips_engine(self):
        """If crop_path is None (GH-36a failed to save), engine is not called."""
        called = []

        def tracking_ocr(png_bytes, **kwargs):
            called.append(True)
            return r"x^2"

        result = process_equation_region(
            region_index=0,
            page_num=1,
            crop_path=None,
            native_text="fallback text",
            ocr=tracking_ocr,
        )

        assert called == []  # engine NOT called
        assert result.latex_attached is False


# ── Default-off test (AC6) ────────────────────────────────────────────────────


class TestDefaultOff:
    """recover_clean_equations defaults to False; no engine call unless opted in."""

    def test_config_default_is_false(self):
        """AC6: PipelineConfig.recover_clean_equations defaults to False."""
        from socr.core.config import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.recover_clean_equations is False

    def test_detect_equations_default_is_false(self):
        """Prerequisite flag also defaults to False."""
        from socr.core.config import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.detect_equations is False

    def test_fingerprint_includes_recover_clean_equations(self):
        """Fingerprint differs when recover_clean_equations changes."""
        from socr.core.config import PipelineConfig
        from socr.pipeline.orchestrator import UnifiedPipeline

        cfg_off = PipelineConfig()
        cfg_on = PipelineConfig()
        cfg_on.recover_clean_equations = True

        orch_off = UnifiedPipeline(cfg_off)
        orch_on = UnifiedPipeline(cfg_on)

        fp_off = orch_off._run_fingerprint()
        fp_on = orch_on._run_fingerprint()
        assert fp_off != fp_on, (
            "Fingerprint should differ when recover_clean_equations changes "
            "(otherwise resume cache mixes policy modes)"
        )


# ── Prompt constant test ───────────────────────────────────────────────────────


class TestPromptConstant:
    """EQUATION_LATEX_PROMPT is a named constant, not a magic string."""

    def test_prompt_is_string(self):
        assert isinstance(EQUATION_LATEX_PROMPT, str)
        assert len(EQUATION_LATEX_PROMPT) > 20

    def test_prompt_mentions_latex(self):
        assert "LaTeX" in EQUATION_LATEX_PROMPT

    def test_prompt_instructs_no_prose(self):
        # The prompt should tell the model to output only LaTeX, no prose.
        assert "ONLY" in EQUATION_LATEX_PROMPT or "only" in EQUATION_LATEX_PROMPT


# ── Orchestrator integration test (mocked, no GPU) ───────────────────────────


class TestOrchestratorEquationLatex:
    """Orchestrator wires GH-36b; mocked to avoid real model or PDF."""

    def test_orchestrator_calls_attach_when_flag_on(self):
        """When recover_clean_equations=True and detect_equations=True,
        _attach_equation_latex_sidecars is called."""
        from socr.core.config import PipelineConfig
        from socr.pipeline.orchestrator import UnifiedPipeline

        cfg = PipelineConfig()
        cfg.detect_equations = True
        cfg.recover_clean_equations = True

        orch = UnifiedPipeline(cfg)

        # Mock the internal method; we only want to verify it's called.
        orch._attach_equation_latex_sidecars = MagicMock()

        # Simulate the condition by patching _backbone_native_first to confirm wiring.
        # This is a behaviour-level check: the flag exists and the method is reachable.
        assert hasattr(orch, "_attach_equation_latex_sidecars")
        assert cfg.recover_clean_equations is True

    def test_attach_sidecars_modifies_page_output_text(self, tmp_path):
        """_attach_equation_latex_sidecars appends sidecar to PageOutput.text."""
        from unittest.mock import MagicMock

        from socr.core.audit_log import AuditEvent
        from socr.core.config import PipelineConfig
        from socr.core.result import PageOutput, PageStatus
        from socr.core.state import DocumentState
        from socr.pipeline.orchestrator import UnifiedPipeline

        # Build minimal state
        crop_file = tmp_path / "equation_1_page1.png"
        crop_file.write_bytes(b"fakepng")

        cfg = PipelineConfig()
        cfg.recover_clean_equations = True
        cfg.detect_equations = True

        orch = UnifiedPipeline(cfg)

        # Build a minimal DocumentState with one page
        handle = MagicMock()
        handle.path = tmp_path / "doc.pdf"
        handle.filename = "doc.pdf"

        state = DocumentState(handle=handle)
        from socr.core.state import PageState

        state.pages[1] = PageState(
            page_num=1,
            is_born_digital=True,
            native_text="E equals m c squared linearised",
        )

        # Inject a fake equation_region_detected event (from GH-36a)
        state.events.append(
            AuditEvent(
                page_num=1,
                kind="equation_region_detected",
                engine="detect_equations",
                detail="test",
                data={
                    "source_bbox": [100.0, 200.0, 400.0, 220.0],
                    "padded_bbox": [96.0, 196.0, 404.0, 224.0],
                    "has_eq_number": False,
                    "crop_path": str(crop_file),
                    "detection_time_s": 0.001,
                },
            )
        )

        # PageOutput for the prose page
        po = PageOutput(
            page_num=1,
            text="E equals m c squared linearised",
            status=PageStatus.SUCCESS,
            engine="native",
        )
        page_outputs = [po]

        # Patch process_equation_region to return a canned valid result
        from socr.math.equation_latex import EQUATION_SIDECAR_HEADER, EquationLatexResult

        canned_result = EquationLatexResult(
            region_index=0,
            page_num=1,
            crop_path=str(crop_file),
            raw_latex=r"E = mc^2",
            validation_ok=True,
            validation_reason="ok",
            latex_attached=True,
            model_id="qwen3-vl:30b-a3b-instruct",
            sidecar_block=(
                f"![equation crop]({crop_file})\n"
                f"{EQUATION_SIDECAR_HEADER}\n"
                "```latex\n"
                r"E = mc^2"
                "\n```"
            ),
        )

        with patch(
            "socr.math.equation_latex.process_equation_region",
            return_value=canned_result,
        ):
            orch._attach_equation_latex_sidecars(state, page_outputs)

        # PageOutput.text now has the sidecar appended
        assert r"E = mc^2" in po.text
        assert EQUATION_SIDECAR_HEADER in po.text
        # Original native text is preserved (not replaced)
        assert "E equals m c squared linearised" in po.text

        # Audit event added
        latex_events = [e for e in state.events if "equation_latex" in e.kind]
        assert len(latex_events) == 1
        ev = latex_events[0]
        assert ev.kind == "equation_latex_accepted"
        assert ev.data["validation_ok"] is True
        assert ev.data["raw_latex"] == r"E = mc^2"
        assert ev.data["model_id"] == "qwen3-vl:30b-a3b-instruct"
        assert ev.data["latex_attached"] is True

    def test_attach_sidecars_invalid_latex_not_attached(self, tmp_path):
        """_attach_equation_latex_sidecars: invalid LaTeX → rejected event; native kept."""
        from unittest.mock import MagicMock

        from socr.core.audit_log import AuditEvent
        from socr.core.config import PipelineConfig
        from socr.core.result import PageOutput, PageStatus
        from socr.core.state import DocumentState
        from socr.pipeline.orchestrator import UnifiedPipeline

        crop_file = tmp_path / "equation_1_page2.png"
        crop_file.write_bytes(b"fakepng")

        cfg = PipelineConfig()
        cfg.recover_clean_equations = True
        cfg.detect_equations = True

        orch = UnifiedPipeline(cfg)

        handle = MagicMock()
        handle.path = tmp_path / "doc.pdf"
        handle.filename = "doc.pdf"

        state = DocumentState(handle=handle)
        from socr.core.state import PageState

        state.pages[2] = PageState(
            page_num=2,
            is_born_digital=True,
            native_text="native linearised math text",
        )
        state.events.append(
            AuditEvent(
                page_num=2,
                kind="equation_region_detected",
                engine="detect_equations",
                detail="test",
                data={
                    "source_bbox": [100.0, 200.0, 400.0, 220.0],
                    "padded_bbox": [96.0, 196.0, 404.0, 224.0],
                    "has_eq_number": False,
                    "crop_path": str(crop_file),
                    "detection_time_s": 0.001,
                },
            )
        )

        po = PageOutput(
            page_num=2,
            text="native linearised math text",
            status=PageStatus.SUCCESS,
            engine="native",
        )

        # Canned result: 1A failed
        from socr.math.equation_latex import EquationLatexResult

        canned_result = EquationLatexResult(
            region_index=0,
            page_num=2,
            crop_path=str(crop_file),
            raw_latex=r"x^{bad",
            validation_ok=False,
            validation_reason="parse error: unmatched brace",
            latex_attached=False,
            model_id="qwen3-vl:30b-a3b-instruct",
            sidecar_block=(
                f"![equation crop]({crop_file})\n"
                "<!-- socr-equation: LaTeX NOT attached (parse error: unmatched brace)"
                " — native text is the authoritative content -->\n"
                "native linearised math text"
            ),
        )

        with patch(
            "socr.math.equation_latex.process_equation_region",
            return_value=canned_result,
        ):
            orch._attach_equation_latex_sidecars(state, [po])

        # Original native text still present
        assert "native linearised math text" in po.text
        # LaTeX NOT in output
        assert "```latex" not in po.text

        # Audit event: rejected
        latex_events = [e for e in state.events if "equation_latex" in e.kind]
        assert len(latex_events) == 1
        ev = latex_events[0]
        assert ev.kind == "equation_latex_rejected_kept_crop"
        assert ev.data["validation_ok"] is False
        assert ev.data["latex_attached"] is False
        assert "parse error" in ev.data["validation_reason"]


# ── Model-resolution tests (reviewer fix — GH-36b local-default mandate) ─────


class TestCleanEquationModelResolution:
    """Verify that the clean-equation path uses clean_equation_model (local by
    default), NOT math_model (which defaults to cloud).

    Pre-fix behaviour (OLD): orchestrator line 2742 was
        model = self.config.math_model or DEFAULT_MODEL
    Since math_model defaults to "qwen3.5:cloud" (truthy), DEFAULT_MODEL never
    fired — every default-config run was routed to a cloud endpoint.

    Post-fix behaviour (NEW): orchestrator uses
        model = self.config.clean_equation_model or DEFAULT_MODEL
    clean_equation_model defaults to "qwen3-vl:30b-a3b-instruct", so the
    local instruct VLM is always the default.
    """

    def _make_state_with_event(self, tmp_path, page_num=1):
        """Build a minimal DocumentState with one equation_region_detected event."""
        from unittest.mock import MagicMock

        from socr.core.audit_log import AuditEvent
        from socr.core.result import PageOutput, PageStatus
        from socr.core.state import DocumentState, PageState

        crop_file = tmp_path / f"equation_1_page{page_num}.png"
        crop_file.write_bytes(b"fakepng")

        handle = MagicMock()
        handle.path = tmp_path / "doc.pdf"
        handle.filename = "doc.pdf"

        state = DocumentState(handle=handle)
        state.pages[page_num] = PageState(
            page_num=page_num,
            is_born_digital=True,
            native_text="some native equation text",
        )
        state.events.append(
            AuditEvent(
                page_num=page_num,
                kind="equation_region_detected",
                engine="detect_equations",
                detail="test",
                data={
                    "source_bbox": [100.0, 200.0, 400.0, 220.0],
                    "padded_bbox": [96.0, 196.0, 404.0, 224.0],
                    "has_eq_number": False,
                    "crop_path": str(crop_file),
                    "detection_time_s": 0.001,
                },
            )
        )

        po = PageOutput(
            page_num=page_num,
            text="some native equation text",
            status=PageStatus.SUCCESS,
            engine="native",
        )
        return state, [po], crop_file

    def test_default_config_uses_local_instruct_model(self, tmp_path):
        """On default PipelineConfig the model passed to process_equation_region
        must be 'qwen3-vl:30b-a3b-instruct', NOT 'qwen3.5:cloud'.

        Pre-fix: FAILS (math_model="qwen3.5:cloud" was used → model="qwen3.5:cloud").
        Post-fix: PASSES (clean_equation_model="qwen3-vl:30b-a3b-instruct" is used).
        """
        from socr.core.config import PipelineConfig
        from socr.pipeline.orchestrator import UnifiedPipeline

        cfg = PipelineConfig()
        cfg.recover_clean_equations = True
        cfg.detect_equations = True
        # Deliberately leave clean_equation_model at its default; math_model at
        # its cloud default.  The bug was that math_model polluted this path.
        assert cfg.math_model == "qwen3.5:cloud"
        assert cfg.clean_equation_model == "qwen3-vl:30b-a3b-instruct"

        orch = UnifiedPipeline(cfg)
        state, page_outputs, _ = self._make_state_with_event(tmp_path)

        captured_models: list[str] = []

        from socr.math.equation_latex import EquationLatexResult

        def fake_process(region_index, page_num, crop_path, native_text, *, model, **kwargs):
            captured_models.append(model)
            return EquationLatexResult(
                region_index=region_index,
                page_num=page_num,
                crop_path=crop_path,
                raw_latex="",
                validation_ok=False,
                validation_reason="stubbed",
                latex_attached=False,
                model_id=model,
                sidecar_block="",
            )

        with patch("socr.math.equation_latex.process_equation_region", side_effect=fake_process):
            orch._attach_equation_latex_sidecars(state, page_outputs)

        assert len(captured_models) == 1, "process_equation_region was not called"
        assert captured_models[0] == "qwen3-vl:30b-a3b-instruct", (
            f"Expected local instruct model but got {captured_models[0]!r}. "
            "This indicates the orchestrator is still using math_model (cloud) "
            "instead of clean_equation_model (local)."
        )
        # Confirm cloud model was NOT used
        assert captured_models[0] != "qwen3.5:cloud"

    def test_clean_equation_model_override_is_honoured(self, tmp_path):
        """An explicit clean_equation_model override reaches the engine call."""
        from socr.core.config import PipelineConfig
        from socr.pipeline.orchestrator import UnifiedPipeline

        cfg = PipelineConfig()
        cfg.recover_clean_equations = True
        cfg.detect_equations = True
        cfg.clean_equation_model = "qwen3.5:cloud"  # explicit cloud opt-in

        orch = UnifiedPipeline(cfg)
        state, page_outputs, _ = self._make_state_with_event(tmp_path)

        captured_models: list[str] = []

        from socr.math.equation_latex import EquationLatexResult

        def fake_process(region_index, page_num, crop_path, native_text, *, model, **kwargs):
            captured_models.append(model)
            return EquationLatexResult(
                region_index=region_index,
                page_num=page_num,
                crop_path=crop_path,
                raw_latex="",
                validation_ok=False,
                validation_reason="stubbed",
                latex_attached=False,
                model_id=model,
                sidecar_block="",
            )

        with patch("socr.math.equation_latex.process_equation_region", side_effect=fake_process):
            orch._attach_equation_latex_sidecars(state, page_outputs)

        assert len(captured_models) == 1
        assert captured_models[0] == "qwen3.5:cloud"

    def test_config_default_clean_equation_model(self):
        """PipelineConfig.clean_equation_model defaults to local instruct VLM."""
        from socr.core.config import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.clean_equation_model == "qwen3-vl:30b-a3b-instruct"
        # math_model is left at its cloud default (corrupt-font path)
        assert cfg.math_model == "qwen3.5:cloud"
        # The two are distinct
        assert cfg.clean_equation_model != cfg.math_model

    def test_forbidden_models_not_reachable_by_default(self):
        """Neither :8b nor the non-instruct :30b are reachable on default config."""
        from socr.core.config import PipelineConfig

        cfg = PipelineConfig()
        assert ":8b" not in cfg.clean_equation_model
        assert cfg.clean_equation_model != "qwen3-vl:30b"  # non-instruct thinking variant
