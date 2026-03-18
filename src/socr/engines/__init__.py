"""OCR engine adapters."""

from socr.engines.base import BaseEngine, BaseHTTPEngine
from socr.engines.deepseek import DeepSeekEngine
from socr.engines.gemini import GeminiEngine
from socr.engines.gemini_api import GeminiAPIEngine
from socr.engines.marker import MarkerEngine
from socr.engines.mistral import MistralEngine
from socr.engines.nougat import NougatEngine

__all__ = [
    "BaseEngine",
    "BaseHTTPEngine",
    "DeepSeekEngine",
    "GeminiAPIEngine",
    "GeminiEngine",
    "MarkerEngine",
    "MistralEngine",
    "NougatEngine",
]
