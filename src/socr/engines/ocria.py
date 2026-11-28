"""Lightweight OCRia-1 vision engine for clean HTML table transcription.

Calls ``llamaindex/ocria-1:bf16`` via either:
  1. Ollama HTTP ``/api/chat`` (respecting ``OLLAMA_HOST`` via
     :func:`socr.tables.extract.resolve_ollama_host`, with ``think: False``
     and ``temperature: 0``), or
  2. An OpenAI-compatible vLLM endpoint when ``base_url`` or the
     ``OCRIA_BASE_URL`` environment variable is configured (e.g., Bocconi
     H200 cluster runs).

Deliberately kept outside :class:`socr.core.config.EngineType` and the CLI
engine registry so it integrates cleanly alongside ``socr.engines`` without
perturbing exhaustive ``EngineType`` / default-ladder invariants.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

from socr.core.ollama_utils import (
    call_with_total_deadline,
    host_reachable,
    ollama_endpoint,
    raise_for_status_redacted,
    safe_host_label,
)
from socr.tables.extract import resolve_ollama_host

logger = logging.getLogger(__name__)

DEFAULT_OCRIA_MODEL = "llamaindex/ocria-1:bf16"
#: Model-card maximum generation budget for OCRia-1 dense tables.
DEFAULT_OCRIA_MAX_TOKENS = 24576
#: Conservative per-call HTTP timeout for local Ollama / vLLM table crops.
DEFAULT_OCRIA_TIMEOUT_SEC = 300.0
#: Liveness probe timeout in seconds.
DEFAULT_OCRIA_PROBE_TIMEOUT_SEC = 10.0

#: Environment variable selecting an OpenAI-compatible vLLM endpoint (e.g.
#: ``http://h200-node:8000/v1``). When unset or blank, Ollama ``/api/chat`` is used.
OCRIA_BASE_URL_ENV = "OCRIA_BASE_URL"
OCRIA_API_KEY_ENV = "OCRIA_API_KEY"
OCRIA_MODEL_ENV = "OCRIA_MODEL"

#: Table-focused transcription prompt that emits clean HTML ``<table>`` blocks
#: and strictly forbids converting charts or graphs into synthetic tables.
OCRIA_TABLE_NO_CHART_PROMPT = """\
Transcribe this image in markdown. Be as accurate as possible in your transcription; \
do not describe or summarize.

Tables:
- Output tables in clean HTML <table>...</table> markup whenever tabular data appears.
- Preserve every row, column, minus sign, decimal point, footnote callout, and spanned \
header faithfully.
- NEVER truncate or shorten tabular data (no ellipses or "[...continues...]").
- Do not use <table> for non-tabular page layout.

Charts, Graphs, and Figures:
- IMPORTANT: NEVER convert a chart or graph into a table and NEVER write data values \
read off a chart.
- If the image contains a chart or graph, output only ![chart: title] using the printed \
title and do not transcribe axis tick labels.

Page furniture:
- Output any page header inside <page_header>...</page_header>, page footer inside \
<page_footer>...</page_footer>, and printed page number inside <page_number>...</page_number>.

Output the transcription cleanly without extra commentary."""


@dataclass
class OCRiaConfig:
    """Configuration for :class:`OCRiaEngine`."""

    model: str = ""
    ollama_host: str | None = None
    base_url: str = ""
    api_key: str = ""
    timeout: float = DEFAULT_OCRIA_TIMEOUT_SEC
    max_tokens: int = DEFAULT_OCRIA_MAX_TOKENS
    temperature: float = 0.0
    prompt: str = OCRIA_TABLE_NO_CHART_PROMPT

    def __post_init__(self) -> None:
        if not self.model:
            self.model = os.environ.get(OCRIA_MODEL_ENV, "").strip() or DEFAULT_OCRIA_MODEL
        if not self.base_url:
            self.base_url = os.environ.get(OCRIA_BASE_URL_ENV, "").strip()
        if not self.api_key:
            self.api_key = os.environ.get(OCRIA_API_KEY_ENV, "").strip()


def _normalize_openai_chat_url(base_url: str) -> str:
    """Build the OpenAI-compatible ``/chat/completions`` URL from *base_url*."""
    trimmed = base_url.strip().rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return f"{trimmed}/chat/completions"


def _normalize_openai_models_url(base_url: str) -> str:
    """Build the OpenAI-compatible ``/models`` URL from *base_url*."""
    trimmed = base_url.strip().rstrip("/")
    if trimmed.endswith("/chat/completions"):
        trimmed = trimmed[: -len("/chat/completions")]
    return f"{trimmed}/models"


def _encode_image_bytes(image: Path | str | bytes) -> tuple[str, str]:
    """Return ``(base64_ascii, mime_type)`` for a path or raw image bytes."""
    if isinstance(image, (bytes, bytearray)):
        raw = bytes(image)
        if raw[:2] == b"\xff\xd8":
            mime = "image/jpeg"
        elif raw.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            mime = "image/webp"
        elif raw[:6] in (b"GIF87a", b"GIF89a"):
            mime = "image/gif"
        else:
            raise ValueError("Unsupported raw image format")
    else:
        path = Path(image)
        raw = path.read_bytes()
        suffix = path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        elif suffix == ".webp":
            mime = "image/webp"
        elif suffix == ".gif":
            mime = "image/gif"
        else:
            mime = "image/png"
    return base64.b64encode(raw).decode("ascii"), mime


class OCRiaEngine:
    """HTTP vision reader for OCRia-1 over Ollama ``/api/chat`` or OpenAI vLLM."""

    def __init__(
        self,
        config: OCRiaConfig | None = None,
        *,
        model: str | None = None,
        ollama_host: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        prompt: str | None = None,
    ) -> None:
        if config is None:
            config = OCRiaConfig(
                model=model or "",
                ollama_host=ollama_host,
                base_url=base_url or "",
                api_key=api_key or "",
                timeout=DEFAULT_OCRIA_TIMEOUT_SEC if timeout is None else timeout,
                max_tokens=DEFAULT_OCRIA_MAX_TOKENS if max_tokens is None else max_tokens,
                temperature=0.0 if temperature is None else temperature,
                prompt=OCRIA_TABLE_NO_CHART_PROMPT if prompt is None else prompt,
            )
        self.config = config

    @property
    def name(self) -> str:
        return "ocria"

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def backend(self) -> str:
        """Active transport: ``"vllm"`` when ``base_url`` is set, else ``"ollama"``."""
        return "vllm" if self.config.base_url else "ollama"

    def resolved_ollama_host(self) -> str:
        """Resolved Ollama host honouring explicit config, then ``OLLAMA_HOST``."""
        return resolve_ollama_host(self.config.ollama_host).rstrip("/")

    def endpoint_url(self) -> str:
        """Resolved inference URL (credentials redacted for display/logging)."""
        if self.backend == "vllm":
            return safe_host_label(_normalize_openai_chat_url(self.config.base_url))
        url, _ = ollama_endpoint(self.resolved_ollama_host(), "/api/chat", strip_slash=True)
        return safe_host_label(url)

    def is_available(self, *, timeout: float = DEFAULT_OCRIA_PROBE_TIMEOUT_SEC) -> bool:
        """Return True iff the configured OCRia endpoint is reachable and serves the model."""
        if self.backend == "vllm":
            headers: dict[str, str] = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            try:
                resp = httpx.get(
                    _normalize_openai_models_url(self.config.base_url),
                    headers=headers,
                    timeout=timeout,
                )
                if resp.status_code != 200:
                    return False
                data = resp.json()
                models = data.get("data") or []
                return any(
                    isinstance(model, dict) and model.get("id") == self.config.model
                    for model in models
                )
            except Exception:
                return False

        host = self.resolved_ollama_host()
        if not host_reachable(host):
            return False
        url, extra = ollama_endpoint(host, "/api/tags", strip_slash=True)
        try:
            resp = httpx.get(url, **extra, timeout=timeout)
            raise_for_status_redacted(resp)
            models = resp.json().get("models") or []
            wanted = self.config.model
            return any(
                (m.get("name") or "") == wanted or (m.get("model") or "") == wanted
                for m in models
                if isinstance(m, dict)
            )
        except Exception:
            return False

    def read_image(
        self,
        image: Path | str | bytes,
        *,
        prompt: str | None = None,
    ) -> tuple[str, str | None]:
        """Transcribe one page or table crop image.

        Returns ``(raw_text, finish_reason)``. ``finish_reason == "length"``
        indicates truncation so downstream callers can fail closed.
        """
        effective_prompt = self.config.prompt if prompt is None else prompt
        image_b64, mime = _encode_image_bytes(image)

        if self.backend == "vllm":
            return self._post_vllm(image_b64, mime, effective_prompt)
        return self._post_ollama(image_b64, effective_prompt)

    def _post_ollama(self, image_b64: str, prompt: str) -> tuple[str, str | None]:
        host = self.resolved_ollama_host()
        url, extra = ollama_endpoint(host, "/api/chat", strip_slash=True)
        payload = {
            "model": self.config.model,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "", "images": [image_b64]},
            ],
        }
        resp = call_with_total_deadline(
            lambda: httpx.post(url, **extra, json=payload, timeout=self.config.timeout),
            self.config.timeout,
            label=f"ocria:ollama:{self.config.model}",
        )
        raise_for_status_redacted(resp)
        data = resp.json()
        msg = data.get("message") or {}
        content = msg.get("content")
        done_reason = data.get("done_reason")
        return (content if isinstance(content, str) else ""), (
            str(done_reason) if done_reason is not None else None
        )

    def _post_vllm(self, image_b64: str, mime: str, prompt: str) -> tuple[str, str | None]:
        url = _normalize_openai_chat_url(self.config.base_url)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                        }
                    ],
                },
            ],
        }
        resp = call_with_total_deadline(
            lambda: httpx.post(url, headers=headers, json=payload, timeout=self.config.timeout),
            self.config.timeout,
            label=f"ocria:vllm:{self.config.model}",
        )
        raise_for_status_redacted(resp)
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OCRia vLLM endpoint returned no choices")
        content = (choices[0].get("message") or {}).get("content")
        finish_reason = choices[0].get("finish_reason")
        return (content if isinstance(content, str) else ""), (
            str(finish_reason) if finish_reason is not None else None
        )

    def __call__(self, crop_path: Path | str) -> tuple[str, str | None]:
        return self.read_image(crop_path)
