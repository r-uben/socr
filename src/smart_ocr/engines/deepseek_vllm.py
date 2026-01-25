"""DeepSeek-OCR via vLLM engine adapter for HPC mode.

Uses vLLM's OpenAI-compatible API to run DeepSeek-OCR model for text extraction.
This is a full OCR engine (unlike VLLMEngine which is figures-only).

Usage:
    1. Start vLLM server with DeepSeek-OCR model:
       vllm serve deepseek-ai/DeepSeek-OCR --dtype auto --api-key token-abc123

    2. Set environment variables (or configure in YAML):
       export VLLM_BASE_URL=http://localhost:8000/v1
       export VLLM_API_KEY=token-abc123

    3. Run smart-ocr in HPC mode:
       smart-ocr process paper.pdf --hpc --vllm-url http://localhost:8000/v1

Supported models (via vLLM):
    - deepseek-ai/DeepSeek-OCR (default)
    - Any vision-capable model that can perform OCR
"""

import base64
import io
import time

import httpx
from PIL import Image

from smart_ocr.core.config import DeepSeekVLLMConfig
from smart_ocr.core.result import FigureResult, PageResult, PageStatus
from smart_ocr.engines.base import BaseEngine, EngineCapabilities


class DeepSeekVLLMEngine(BaseEngine):
    """Adapter for DeepSeek-OCR via vLLM OpenAI-compatible API.

    This engine performs full OCR using the DeepSeek-OCR model served via vLLM.
    It also supports figure description, making it versatile for HPC mode.
    """

    def __init__(self, config: DeepSeekVLLMConfig | None = None) -> None:
        super().__init__()
        self.config = config or DeepSeekVLLMConfig()
        self._client: httpx.Client | None = None

    @property
    def name(self) -> str:
        return "deepseek-vllm"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="deepseek-vllm",
            supports_pdf=False,  # Processes images, not PDF directly
            supports_images=True,  # Full OCR support
            supports_batch=False,
            supports_figures=True,  # Can describe figures too
            is_local=True,  # Runs on local vLLM server (typically HPC)
            cost_per_page=0.0,  # Free (self-hosted)
            best_for=["academic", "scientific", "equations", "hpc"],
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
        """Check if vLLM server is available with DeepSeek-OCR model."""
        if self._initialized:
            return True

        try:
            client = self._get_client()
            # Check models endpoint to verify server is running
            response = client.get("/models")
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                model_ids = [m.get("id", "") for m in models]

                # Check if our target model is available
                if self.config.model in model_ids or any(
                    self.config.model.lower() in m.lower() for m in model_ids
                ):
                    self._initialized = True
                    return True

                # If model check fails but server responds, mark as available
                # (model might be loading or using different naming)
                if models:
                    self._initialized = True
                    return True
            return False
        except Exception:
            return False

    def process_image(self, image: Image.Image, page_num: int = 1) -> PageResult:
        """Process a single image and extract text using DeepSeek-OCR."""
        if not self._initialized and not self.initialize():
            return self._create_error_result(
                page_num,
                f"vLLM server not available at {self.config.base_url}",
            )

        start_time = time.time()

        try:
            # Convert image to base64
            buffered = io.BytesIO()
            # Ensure RGB mode for JPEG
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(buffered, format="JPEG", quality=90)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Build OCR prompt
            prompt = self._build_ocr_prompt()

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

            processing_time = time.time() - start_time

            if response.status_code != 200:
                error_text = response.text[:200]
                return self._create_error_result(
                    page_num,
                    f"vLLM API error ({response.status_code}): {error_text}",
                )

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return self._create_error_result(
                    page_num,
                    "vLLM returned empty response",
                )

            text = choices[0].get("message", {}).get("content", "").strip()

            if not text or len(text) < 10:
                return self._create_error_result(
                    page_num,
                    "OCR produced empty or minimal output",
                )

            return self._create_success_result(
                page_num=page_num,
                text=text,
                confidence=0.85,  # Reasonable default for DeepSeek-OCR
                processing_time=processing_time,
                cost=0.0,
            )

        except httpx.TimeoutException:
            return self._create_error_result(
                page_num,
                f"vLLM request timed out after {self.config.timeout}s",
            )
        except Exception as e:
            return self._create_error_result(
                page_num,
                f"vLLM error: {type(e).__name__}: {e}",
            )

    def describe_figure(
        self,
        image: Image.Image,
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureResult:
        """Describe a figure using DeepSeek vision capabilities."""
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
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Build figure description prompt
            prompt = self._build_figure_prompt(figure_type, context)

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
                    "max_tokens": 1024,  # Shorter for figure descriptions
                    "temperature": 0.1,
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

    def _build_ocr_prompt(self) -> str:
        """Build the prompt for OCR text extraction."""
        return (
            "Extract all text from this document image. "
            "Preserve the original structure including paragraphs, headings, lists, and tables. "
            "For mathematical equations, use LaTeX notation (e.g., $E = mc^2$ for inline, "
            "$$\\int_0^\\infty f(x) dx$$ for display equations). "
            "For tables, use markdown table format. "
            "Maintain the reading order and hierarchical structure of the content."
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
            "equation": ["equation", "formula", "mathematical"],
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
