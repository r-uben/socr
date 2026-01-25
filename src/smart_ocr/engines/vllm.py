"""vLLM vision engine adapter for figure description.

Uses vLLM's OpenAI-compatible API to run vision models like Qwen2-VL or InternVL2
for figure/chart description. This is a figures-only engine (no OCR).

Usage:
    1. Start vLLM server with a vision model:
       vllm serve Qwen/Qwen2-VL-7B-Instruct --dtype auto --api-key token-abc123

    2. Set environment variables (or configure in YAML):
       export VLLM_BASE_URL=http://localhost:8000/v1
       export VLLM_API_KEY=token-abc123

    3. Run smart-ocr with vLLM for figures:
       smart-ocr process paper.pdf --figures-engine vllm

Supported models (via vLLM):
    - Qwen/Qwen2-VL-7B-Instruct (default)
    - Qwen/Qwen2-VL-72B-Instruct
    - OpenGVLab/InternVL2-8B
    - OpenGVLab/InternVL2-26B
"""

import base64
import io
import time

import httpx
from PIL import Image

from smart_ocr.core.config import VLLMConfig
from smart_ocr.core.result import FigureResult, PageResult, PageStatus
from smart_ocr.engines.base import BaseEngine, EngineCapabilities


class VLLMEngine(BaseEngine):
    """Adapter for vLLM vision models via OpenAI-compatible API.

    This engine is specialized for figure description and does not support OCR.
    It uses vision-language models like Qwen2-VL or InternVL2 served via vLLM.
    """

    def __init__(self, config: VLLMConfig | None = None) -> None:
        super().__init__()
        self.config = config or VLLMConfig()
        self._client: httpx.Client | None = None

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="vllm",
            supports_pdf=False,  # Not an OCR engine
            supports_images=False,  # Not for OCR, only figures
            supports_batch=False,
            supports_figures=True,  # Primary purpose
            is_local=True,  # Typically runs locally
            cost_per_page=0.0,  # Free (self-hosted)
            best_for=["figures", "charts", "diagrams", "vision"],
        )

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.config.timeout, connect=30.0),
            )
        return self._client

    def initialize(self) -> bool:
        """Check if vLLM server is available."""
        if self._initialized:
            return True

        try:
            client = self._get_client()
            # Check models endpoint to verify server is running
            response = client.get("/models")
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                # Check if our target model is available
                model_ids = [m.get("id", "") for m in models]
                if self.config.model in model_ids or any(
                    self.config.model.lower() in m.lower() for m in model_ids
                ):
                    self._initialized = True
                    return True
                # If model check fails but server responds, still mark as available
                # (model might be loading or using different naming)
                if models:
                    self._initialized = True
                    return True
            return False
        except Exception:
            return False

    def process_image(self, image: Image.Image, page_num: int = 1) -> PageResult:
        """Process a single image - NOT SUPPORTED for vLLM engine.

        vLLM engine is specialized for figure description only.
        Use deepseek, nougat, mistral, or gemini for OCR.
        """
        return self._create_error_result(
            page_num,
            "vLLM engine is for figure description only, not OCR. "
            "Use --primary nougat/deepseek/mistral/gemini for OCR.",
        )

    def describe_figure(
        self,
        image: Image.Image,
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureResult:
        """Describe a figure using vLLM vision model."""
        if not self._initialized and not self.initialize():
            return FigureResult(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description=f"vLLM server not available at {self.config.base_url}",
            )

        start_time = time.time()

        try:
            # Convert image to base64
            buffered = io.BytesIO()
            # Ensure RGB mode for JPEG
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Build prompt
            prompt = self._build_figure_prompt(figure_type, context)

            # Build OpenAI-compatible request with vision
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            client = self._get_client()
            response = client.post(
                "/chat/completions",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                },
            )

            if response.status_code != 200:
                error_text = response.text[:200]
                return FigureResult(
                    figure_num=0,
                    page_num=0,
                    figure_type=figure_type,
                    description=f"vLLM API error ({response.status_code}): {error_text}",
                    engine=self.name,
                )

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return FigureResult(
                    figure_num=0,
                    page_num=0,
                    figure_type=figure_type,
                    description="vLLM returned empty response",
                    engine=self.name,
                )

            description = choices[0].get("message", {}).get("content", "").strip()

            if not description or len(description) < 10:
                description = "Unable to generate meaningful description"

            # Detect figure type from description
            detected_type = self._detect_figure_type(description, figure_type)

            processing_time = time.time() - start_time

            return FigureResult(
                figure_num=0,
                page_num=0,
                figure_type=detected_type,
                description=description,
                engine=self.name,
            )

        except httpx.TimeoutException:
            return FigureResult(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description=f"vLLM request timed out after {self.config.timeout}s",
                engine=self.name,
            )
        except Exception as e:
            return FigureResult(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description=f"vLLM error: {type(e).__name__}: {e}",
                engine=self.name,
            )

    def _build_figure_prompt(self, figure_type: str, context: str) -> str:
        """Build the prompt for figure description."""
        base_prompt = (
            "Describe this figure in detail. What does the chart, graph, table, or diagram show? "
            "Explain the axes, data, key findings, and any notable patterns or trends. "
            "Be specific about numbers, labels, and relationships shown."
        )

        if figure_type and figure_type != "unknown":
            base_prompt = f"This appears to be a {figure_type}. {base_prompt}"

        if context:
            base_prompt += f"\n\nContext from surrounding text: {context[:500]}"

        return base_prompt

    def _detect_figure_type(self, description: str, default: str) -> str:
        """Detect figure type from description."""
        desc_lower = description.lower()
        type_keywords = {
            "chart": ["bar chart", "pie chart", "chart"],
            "graph": ["line graph", "scatter plot", "graph", "plot"],
            "table": ["table", "tabular"],
            "diagram": ["diagram", "flowchart", "schematic", "architecture"],
            "map": ["map", "geographic", "spatial"],
            "photo": ["photo", "photograph", "image"],
        }

        for fig_type, keywords in type_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                return fig_type

        return default

    def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        self.close()
