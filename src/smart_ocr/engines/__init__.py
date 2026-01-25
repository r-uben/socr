"""OCR engine adapters."""

from smart_ocr.engines.base import BaseEngine, EngineCapabilities
from smart_ocr.engines.deepseek import DeepSeekEngine
from smart_ocr.engines.deepseek_vllm import DeepSeekVLLMEngine
from smart_ocr.engines.gemini import GeminiEngine
from smart_ocr.engines.mistral import MistralEngine
from smart_ocr.engines.nougat import NougatEngine
from smart_ocr.engines.vllm import VLLMEngine
from smart_ocr.engines.vllm_manager import (
    ServerConfig,
    VLLMServerManager,
    detect_gpu_setup,
    get_gpu_memory_gb,
)

__all__ = [
    "BaseEngine",
    "EngineCapabilities",
    "NougatEngine",
    "DeepSeekEngine",
    "DeepSeekVLLMEngine",
    "MistralEngine",
    "GeminiEngine",
    "VLLMEngine",
    "VLLMServerManager",
    "ServerConfig",
    "detect_gpu_setup",
    "get_gpu_memory_gb",
]
