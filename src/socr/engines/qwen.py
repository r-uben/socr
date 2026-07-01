"""Qwen-VL OCR engine adapter.

CLI: qwen-ocr process <path> -o <dir> [--backend auto|ollama|vllm|api]
     [--prefer cost|quality|speed] [--dpi N] [-w N] [-q] [--verbose] [--reprocess]
Click group — requires explicit 'process' subcommand. See ../ocr/qwen-ocr-cli.

Qwen-VL is the strongest *open* OCR family on social-science / historical documents
(socOCRbench ~0.47-0.58 vs GLM 0.37, DeepSeek 0.09), so it leads socr's local tier.

Availability here reflects the **local ollama backend** (qwen3-vl), matching how the
glm/deepseek adapters gate on their Ollama models — that is the free, local role socr
uses this engine for. The standalone CLI's vllm/api backends are reachable by setting
``config.qwen_backend`` explicitly; the CLI's ``--backend auto`` then picks what is live.

Backend/model resolution is centralised in ``resolve_qwen_intent`` below so there is
exactly ONE place that maps (backend, model-pin) → (resolved-backend, resolved-model).
"""

import logging
import os
from pathlib import Path

from socr.core.config import PipelineConfig
from socr.core.ollama_utils import check_ollama_model as _check_ollama_model
from socr.engines.base import BaseEngine

logger = logging.getLogger(__name__)

# Local tier model. Validated 2026-06-13 on the owner's hard pages
# ([[reference-local-ocr-benchmark-jun2026]]): the A3B MoE (~3B active/token) reconstructs
# dense multi-column tables with exact digits AND recovers signs native extraction corrupts,
# at ~1-2 min/page on Metal. MUST be the INSTRUCT (non-thinking) build: the default
# `qwen3-vl:30b` thinking build runs away on dense 11-column tables (5000+ thinking tokens,
# never emits) and neither `think:false` nor `/no_think` suppress it on Ollama 0.30.8. The
# dense `qwen3-vl:8b` collapses dense tables and is slow — do not use it as the local tier.
OLLAMA_MODEL = "qwen3-vl:30b-a3b-instruct"

# Backends that resolve to the local Ollama layer.  All other values ("vllm", "api")
# route to non-local infrastructure and must not have their model strings rewritten.
_LOCAL_BACKENDS: frozenset[str] = frozenset({"auto", "ollama"})


def resolve_qwen_intent(config: PipelineConfig) -> tuple[str, str]:
    """Return (resolved_backend, resolved_model) for the qwen-ocr-cli invocation.

    Rules (in priority order):
    1. Explicit user pin (``config.qwen_model_pinned``): pass model through unchanged.
    2. Non-local backend ("vllm", "api") without a pin: request ``config.qwen_vllm_model``
       (the exact model the OpenAI-compatible server is serving), falling back to
       ``config.qwen_model`` only if ``qwen_vllm_model`` is blank. The CLI's own default
       must never be relied on here: a mismatched model name makes the server 404.
    3. Local/auto backend ("auto", "ollama") without a pin: use ``OLLAMA_MODEL``,
       the validated instruct MoE.  A cloud model string must never reach a local backend
       via this path.

    Returns a ``(backend, model)`` tuple.  The backend is always the value from config
    (callers pass it through to ``--backend``).  Only the model may be rewritten.
    """
    backend = config.qwen_backend
    model = config.qwen_model

    if config.qwen_model_pinned:
        # Explicit user override — honour it verbatim, regardless of backend.
        resolved_model = model
    elif backend in _LOCAL_BACKENDS:
        # Local/auto path: always use the validated instruct MoE unless the user pinned
        # something else.  This prevents a cloud model string (e.g. qwen3.5:cloud) from
        # silently landing on the local Ollama runtime.
        resolved_model = OLLAMA_MODEL
    else:
        # vllm / api: the qwen-ocr CLI must request the EXACT model the vLLM server
        # is serving. Use config.qwen_vllm_model (e.g. Qwen/Qwen3-VL-30B-A3B-Instruct);
        # without it the CLI sends its own default model name and the server 404s.
        resolved_model = config.qwen_vllm_model or model

    logger.info(
        "[qwen] resolved backend=%r model=%r (pinned=%s)",
        backend,
        resolved_model,
        config.qwen_model_pinned,
    )
    return backend, resolved_model


class QwenEngine(BaseEngine):
    """Adapter for qwen-ocr-cli."""

    @property
    def name(self) -> str:
        return "qwen"

    @property
    def cli_command(self) -> str:
        return "qwen-ocr"

    def is_available(self) -> bool:
        """CLI installed AND a usable backend.

        Local path: the instruct MoE (OLLAMA_MODEL) must be pulled in Ollama.
        vLLM path (e.g. HPC, where Ollama is forbidden on server GPUs): when
        ``VLLM_BASE_URL`` is set the qwen-ocr CLI serves via vLLM, so the local
        Ollama model is NOT required. Mirrors qwen-ocr's own backend gate.
        """
        if not super().is_available():
            return False
        if os.environ.get("VLLM_BASE_URL"):
            return True
        error = _check_ollama_model(OLLAMA_MODEL)
        if error:
            logger.debug(f"[{self.name}] {error}")
            return False
        return True

    def process_document(self, pdf_path, output_dir, config):
        """Process document, pre-checking the Ollama model for the local backend."""
        if config.qwen_backend in ("auto", "ollama"):
            error = _check_ollama_model(OLLAMA_MODEL)
            if error:
                from socr.core.result import DocumentStatus, EngineResult, FailureMode

                logger.error(f"[{self.name}] {error}")
                return EngineResult(
                    document_path=pdf_path,
                    engine=self.name,
                    status=DocumentStatus.ERROR,
                    failure_mode=FailureMode.MODEL_UNAVAILABLE,
                    error=error,
                    processing_time=0.0,
                )
        return super().process_document(pdf_path, output_dir, config)

    def _build_command(
        self,
        pdf_path: Path,
        output_dir: Path,
        config: PipelineConfig,
    ) -> list[str]:
        backend, model = resolve_qwen_intent(config)
        cmd = [
            self.cli_command,
            "process",
            str(pdf_path),
            "-o",
            str(output_dir),
            "--backend",
            backend,
            "--dpi",
            str(config.render_dpi),
        ]
        if model:
            cmd.extend(["--model", model])
        if config.workers > 1:
            cmd.extend(["-w", str(config.workers)])
        if config.quiet:
            cmd.append("-q")
        if config.verbose:
            cmd.append("--verbose")
        if config.reprocess:
            cmd.append("--reprocess")
        return cmd
