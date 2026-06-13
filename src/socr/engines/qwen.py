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
"""

import logging
from pathlib import Path

from socr.core.config import PipelineConfig
from socr.engines.base import BaseEngine
from socr.engines.deepseek import _check_ollama_model

logger = logging.getLogger(__name__)

# Local tier model. Validated 2026-06-13 on the owner's hard pages
# ([[reference-local-ocr-benchmark-jun2026]]): the A3B MoE (~3B active/token) reconstructs
# dense multi-column tables with exact digits AND recovers signs native extraction corrupts,
# at ~1-2 min/page on Metal. MUST be the INSTRUCT (non-thinking) build: the default
# `qwen3-vl:30b` thinking build runs away on dense 11-column tables (5000+ thinking tokens,
# never emits) and neither `think:false` nor `/no_think` suppress it on Ollama 0.30.8. The
# dense `qwen3-vl:8b` collapses dense tables and is slow — do not use it as the local tier.
OLLAMA_MODEL = "qwen3-vl:30b-a3b-instruct"


class QwenEngine(BaseEngine):
    """Adapter for qwen-ocr-cli."""

    @property
    def name(self) -> str:
        return "qwen"

    @property
    def cli_command(self) -> str:
        return "qwen-ocr"

    def is_available(self) -> bool:
        """CLI installed AND the local instruct MoE (OLLAMA_MODEL) is pulled."""
        if not super().is_available():
            return False
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
        cmd = [
            self.cli_command,
            "process",
            str(pdf_path),
            "-o",
            str(output_dir),
            "--backend",
            config.qwen_backend,
            "--dpi",
            str(config.render_dpi),
        ]
        # Local (ollama) backend must not be handed a cloud model string. When the user
        # left qwen_model at its cloud default, substitute the validated local instruct MoE;
        # an explicit --qwen-model (e.g. qwen3-vl:8b offline) is still respected.
        model = config.qwen_model
        if config.qwen_backend == "ollama" and model == "qwen3.5:cloud":
            model = OLLAMA_MODEL
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
