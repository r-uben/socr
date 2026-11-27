"""Gemini vision API for figure descriptions.

NOT used for OCR — OCR goes through gemini-ocr-cli.
This module is only used by the orchestrator's figure pass to describe
extracted figure images via the Gemini generateContent API.
"""

import base64
import io
import os
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from PIL import Image

from socr.core.ollama_utils import (
    TotalDeadlineExceeded,
    call_with_total_deadline,
    ollama_endpoint,
    raise_for_status_redacted,
    safe_host_label,
)
from socr.core.config import DEFAULT_GEMINI_MODEL
from socr.core.providers import gemini_usage_cost_usd
from socr.core.result import FigureInfo
from socr.engines._figure_prompt import (
    CAPTION_MARKER as _CAPTION_MARKER,  # noqa: F401  re-exported for tests
)
from socr.engines._figure_prompt import (
    build_figure_prompt as _build_figure_prompt,
)
from socr.engines._figure_prompt import (
    wrap_caption as _wrap_caption,
)


@dataclass
class GeminiAPIConfig:
    """Configuration for the Gemini vision API."""

    api_key: str = ""
    model: str = DEFAULT_GEMINI_MODEL
    timeout: float = 120.0
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get(
                "GOOGLE_API_KEY", ""
            )


class GeminiAPIEngine:
    """Gemini vision API for figure descriptions only.

    Used by the orchestrator's figure pass. NOT an OCR engine.
    """

    def __init__(self, config: GeminiAPIConfig | None = None) -> None:
        self.config = config or GeminiAPIConfig()
        self._client: httpx.Client | None = None
        #: Metered cost of the LAST ``describe_figure`` call (#1405): ``0.0`` when
        #: nothing billable was sent or the API refused the request, the priced
        #: token usage of a 200 response, and ``None`` when a request went out
        #: and its usage is not known (transport error after sending, or a 200
        #: without ``usageMetadata``). ``None`` means unknown and is recorded as
        #: unknown; it is never read as zero.
        self.last_call_cost_usd: float | None = 0.0
        self._initialized: bool = False

    @property
    def name(self) -> str:
        return "gemini-api"

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.config.timeout, connect=30.0),
            )
        return self._client

    def _build_url(self) -> str:
        return (
            f"{self.config.base_url}/models/{self.config.model}"
            f":generateContent?key={self.config.api_key}"
        )

    def initialize(self) -> bool:
        """Check API key availability and basic connectivity."""
        if self._initialized:
            return True
        if not self.config.api_key:
            return False
        try:
            client = self._get_client()
            url = f"{self.config.base_url}/models?key={self.config.api_key}"
            response = client.get(url, timeout=15.0)
            if response.status_code == 200:
                self._initialized = True
                return True
            return False
        except Exception:
            return False

    def describe_figure(
        self,
        image: Image.Image,
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureInfo:
        """Describe a figure image using the Gemini vision API."""
        self.last_call_cost_usd = 0.0
        if not self._initialized and not self.initialize():
            return FigureInfo(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description="Gemini API not available",
            )

        try:
            img_base64 = _image_to_base64(image)
            prompt = _build_figure_prompt(figure_type, context)

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": img_base64,
                                }
                            },
                            {"text": prompt},
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 1024,
                    "temperature": 0.1,
                },
            }

            client = self._get_client()
            # From here a request is on the wire: its cost is unknown until the
            # response says otherwise.
            self.last_call_cost_usd = None
            response = client.post(self._build_url(), json=payload)

            if response.status_code != 200:
                # A refused request generated no tokens.
                self.last_call_cost_usd = 0.0
                return FigureInfo(
                    figure_num=0,
                    page_num=0,
                    figure_type=figure_type,
                    description=f"Gemini API error ({response.status_code})",
                    engine=self.name,
                )

            body = response.json()
            self.last_call_cost_usd = _usage_cost_usd(body)
            raw = _extract_text(body) or "Unable to generate description"
            detected_type = _detect_figure_type(raw, figure_type)
            # Only wrap genuine model output — not the fallback error string.
            description = _wrap_caption(raw) if raw != "Unable to generate description" else raw

            return FigureInfo(
                figure_num=0,
                page_num=0,
                figure_type=detected_type,
                description=description,
                engine=self.name,
            )

        except Exception as e:
            return FigureInfo(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description=f"Gemini API error: {type(e).__name__}: {e}",
                engine=self.name,
            )

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        self.close()


#: Total deadline for ONE ``OllamaFigureEngine`` HTTP call. Named, not changed:
#: ``ask`` returns ``None`` on ANY failure, so a caller that must tell a timeout
#: from a refusal has to compare its own elapsed time against this (#1050 step 2,
#: round 2 item C -- the swallowed exception meant real timeouts never armed the
#: cascade halt).
FIGURE_ASK_TIMEOUT_SEC: float = 120.0

#: Environment override for the context window of ONE Ollama figure call.
FIGURE_NUM_CTX_ENV = "SOCR_FIGURE_NUM_CTX"

#: Context window (tokens) requested for ONE Ollama figure call (#1405).
#:
#: Without ``options.num_ctx`` the daemon's own default applies, and the daemon
#: sizes the loaded model for that window. On a daemon whose default was 262144
#: the caption model (``qwen3-vl:30b-a3b-instruct``) occupied about 42 GB; at
#: this value it occupied about 21 GB (measured on a 64 GB Apple-silicon host,
#: 2026-11-24). A figure call is one image, the fixed prompt and at most the
#: 500 characters of page context ``_get_page_context`` passes, so the request
#: never approaches the daemon default. 32768 is the window socr's vLLM recipe
#: serves the same model with (``--max-model-len 32768``), so both local
#: backends give the caption model the same room. Override per deployment with
#: ``SOCR_FIGURE_NUM_CTX``.
FIGURE_NUM_CTX_DEFAULT: int = 32768


def resolve_figure_num_ctx(num_ctx: int | None = None) -> int:
    """Context window for a figure call: argument, then env var, then the default.

    A value that is not a positive integer is ignored rather than raised on: the
    caller is building a request on the caption path, and a malformed override
    must not cost the document its figures.
    """
    for candidate in (num_ctx, os.environ.get(FIGURE_NUM_CTX_ENV)):
        if candidate is None:
            continue
        try:
            value = int(str(candidate).strip())
        except ValueError:
            continue
        if value > 0:
            return value
    return FIGURE_NUM_CTX_DEFAULT


class OllamaFigureEngine:
    """Local-first figure description engine using Ollama.

    Uses qwen3-vl:30b-a3b-instruct (non-thinking MoE) via Ollama's /api/chat
    endpoint. Intended as the primary figure-description tier, with GeminiAPIEngine
    as fallback on empty result or error.
    """

    def __init__(
        self,
        model: str = "qwen3-vl:30b-a3b-instruct",
        host: str | None = None,
        num_ctx: int | None = None,
    ) -> None:
        # #1405: the host used to default to a literal localhost URL, so a
        # deployment whose ``OLLAMA_HOST`` points elsewhere had its table judge
        # there and its figure captions on localhost. Resolved the way every
        # other Ollama caller resolves it (GH-222). Imported here, not at module
        # top: ``socr.tables.extract`` pulls in the crop-reader stack.
        from socr.tables.extract import resolve_ollama_host

        self.model = model
        self.host = resolve_ollama_host(host).rstrip("/")
        self.num_ctx = resolve_figure_num_ctx(num_ctx)
        self._last_available = False

    @property
    def name(self) -> str:
        return "ollama-figure"

    def is_available(self) -> bool:
        """Return True if Ollama is reachable and this model is in its tag list."""
        try:
            url, extra = ollama_endpoint(self.host, "/api/tags")
            resp = call_with_total_deadline(
                lambda: httpx.get(url, **extra, timeout=3.0),
                3.0,
                label=f"ollama figure {safe_host_label(self.host)}/api/tags",
            )
            if resp.status_code != 200:
                return self._remember_availability(False)
            data = resp.json()
            models = data.get("models", [])
            return self._remember_availability(any(m.get("name", "") == self.model for m in models))
        except TotalDeadlineExceeded:
            # GH-968: a deadline overrun, or a fail-fast because an earlier probe
            # is still draining, says nothing about the daemon (the same way a
            # timeout is never "unavailable" in ``probe_failure_reason``). Report
            # the last DEFINITIVE answer rather than a fresh False.
            return self._last_available
        except Exception:
            return self._remember_availability(False)

    def _remember_availability(self, value: bool) -> bool:
        self._last_available = value
        return value

    def describe_figure(
        self,
        image: Image.Image,
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureInfo:
        """Describe a figure image using Ollama's /api/chat endpoint."""
        try:
            buf = io.BytesIO()
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(buf, format="JPEG", quality=90)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            prompt = _build_figure_prompt(figure_type, context)
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64],
                    }
                ],
                "stream": False,
                "options": {"num_ctx": self.num_ctx},
            }
            url, extra = ollama_endpoint(self.host, "/api/chat")
            resp = call_with_total_deadline(
                lambda: httpx.post(url, **extra, json=payload, timeout=FIGURE_ASK_TIMEOUT_SEC),
                FIGURE_ASK_TIMEOUT_SEC,
                label=f"ollama figure {safe_host_label(self.host)}/api/chat ({self.model})",
            )
            raise_for_status_redacted(resp)
            raw = resp.json()["message"]["content"].strip()

            detected_type = _detect_figure_type(raw, figure_type)
            return FigureInfo(
                figure_num=0,
                page_num=0,
                figure_type=detected_type,
                description=_wrap_caption(raw),
                engine=self.name,
            )
        except Exception as e:
            return FigureInfo(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description=f"Ollama figure error: {type(e).__name__}: {e}",
                engine=self.name,
            )

    def ask(self, image: Image.Image, prompt: str) -> str | None:
        """Send *prompt* with *image*; return the RAW answer, or ``None`` on failure.

        One exception to "any failure": ``TotalDeadlineExceeded`` PROPAGATES
        (#1050 step 2). A fail-fast from the outstanding-call guard is the only
        signal a caller has that the backend is wedged rather than merely
        unhelpful, and folding it into ``None`` hid that from every caller.
        Callers that must never raise -- the number-free figure describer --
        catch it themselves.

        Unlike ``describe_figure`` this neither wraps the text in the caption marker nor
        folds an error into the description string: the number-free describer must be able
        to tell "the model said nothing usable" from prose, and does its own labelling.
        """
        try:
            buf = io.BytesIO()
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(buf, format="JPEG", quality=90)
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [base64.b64encode(buf.getvalue()).decode()],
                    }
                ],
                "stream": False,
                "options": {"num_ctx": self.num_ctx},
            }
            url, extra = ollama_endpoint(self.host, "/api/chat")
            resp = call_with_total_deadline(
                lambda: httpx.post(url, **extra, json=payload, timeout=FIGURE_ASK_TIMEOUT_SEC),
                FIGURE_ASK_TIMEOUT_SEC,
                label=f"ollama figure {safe_host_label(self.host)}/api/chat ({self.model})",
            )
            raise_for_status_redacted(resp)
            return resp.json()["message"]["content"].strip() or None
        except TotalDeadlineExceeded:
            # #1050 step 2 round 4 (Argon #4): a fail-fast from ``_OUTSTANDING``
            # raises this in under a millisecond, and folding it into the
            # ``None`` below hid a wedge from every caller. Re-raised so a
            # caller that must tell a timeout from a refusal can.
            raise
        except Exception:
            return None

    def close(self) -> None:
        pass  # stateless HTTP — nothing to close


#: Opens the visible mark a figure carries when its paid caption call was
#: refused (#1405). Number-free and free of Markdown control characters on
#: purpose: a figure on a certified-floor page keeps its caption only if the
#: caption passes ``crop_descriptions``' number-free check, and a mark that
#: failed it would be blanked there -- a refusal with no trace on the page.
PAID_CAPTION_REFUSED_PREFIX = "[figure description withheld:"


def paid_caption_refused_mark(reason: str) -> str:
    """The description a figure ships with when its paid caption was refused."""
    return f"{PAID_CAPTION_REFUSED_PREFIX} {reason}]"


class BudgetedPaidFigureEngine:
    """A paid caption engine behind an admission gate and a cost recorder (#1405).

    ``GeminiAPIEngine`` calls a paid API directly. It used to be handed out bare,
    so every caption call was outside ``--max-paid-pages`` and ``--cost-budget``
    and left nothing in the run's cost journal. This wrapper is the only form
    in which the orchestrator hands the paid engine out:

    - ``admit()`` runs before EVERY call and returns ``None`` to allow it or a
      short reason to refuse it. A refused call sends nothing and returns the
      refusal mark as the description, so the figure shows why it has no caption.
    - ``record(cost)`` runs after every call that was allowed, with the engine's
      metered cost (``None`` = unknown).
    """

    def __init__(
        self,
        inner: "GeminiAPIEngine",
        admit: Callable[[], str | None],
        record: Callable[[float | None], None],
    ) -> None:
        self._inner = inner
        self._admit = admit
        self._record = record

    @property
    def name(self) -> str:
        return self._inner.name

    def describe_figure(
        self,
        image: Image.Image,
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureInfo:
        refusal = self._admit()
        if refusal:
            return FigureInfo(
                figure_num=0,
                page_num=0,
                figure_type=figure_type,
                description=paid_caption_refused_mark(refusal),
                engine=self.name,
            )
        try:
            return self._inner.describe_figure(image, figure_type=figure_type, context=context)
        finally:
            self._record(getattr(self._inner, "last_call_cost_usd", None))

    def close(self) -> None:
        self._inner.close()


class LocalFirstFigureEngine:
    """Composite engine: try the local engine first, fall back to Gemini per-call.

    Used by the orchestrator's _get_vision_engine() when Ollama is available.
    On each describe_figure call:
    1. Call OllamaFigureEngine.
    2. If the result has a non-empty description that is not an error prefix,
       return it.
    3. Otherwise fall back to GeminiAPIEngine (if provided).
    """

    _OLLAMA_ERROR_PREFIX = "Ollama figure error:"

    def __init__(
        self,
        ollama: "OllamaFigureEngine",
        gemini: "GeminiAPIEngine | BudgetedPaidFigureEngine | None" = None,
        is_failure: Callable[[FigureInfo], bool] | None = None,
    ) -> None:
        # ``ollama`` is whichever LOCAL engine runs first; the name predates the
        # vLLM tier (#1405) and is kept because callers pass it positionally.
        # ``is_failure`` replaces the Ollama error-prefix test for a local engine
        # whose failure strings differ.
        self._ollama = ollama
        self._gemini = gemini
        self._is_failure = is_failure

    @property
    def name(self) -> str:
        return "local-first-figure"

    def describe_figure(
        self,
        image: Image.Image,
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureInfo:
        """Describe via Ollama; fall back to Gemini on empty or error."""
        info = self._ollama.describe_figure(image, figure_type=figure_type, context=context)
        if self._is_failure is not None:
            failed = self._is_failure(info)
        else:
            failed = not info.description or info.description.startswith(self._OLLAMA_ERROR_PREFIX)
        if not failed:
            return info

        # Ollama returned empty or an error — try Gemini
        if self._gemini is not None:
            return self._gemini.describe_figure(image, figure_type=figure_type, context=context)

        # No Gemini fallback available — return what Ollama gave us
        return info

    def close(self) -> None:
        self._ollama.close()
        if self._gemini is not None:
            self._gemini.close()


def _detect_figure_type(description: str, default: str) -> str:
    desc_lower = description.lower()
    for fig_type, keywords in {
        "chart": ["bar chart", "pie chart", "chart"],
        "graph": ["line graph", "scatter plot", "graph", "plot"],
        "table": ["table", "tabular"],
        "diagram": ["diagram", "flowchart", "schematic", "architecture"],
        "map": ["map", "geographic", "spatial"],
        "equation": ["equation", "formula", "mathematical"],
    }.items():
        if any(kw in desc_lower for kw in keywords):
            return fig_type
    return default


def _image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _usage_cost_usd(data: dict) -> float | None:
    """Cost of a 200 ``generateContent`` response from its ``usageMetadata``.

    ``None`` when the response does not report a prompt token count: the call
    ran and was billed, and a missing figure is unknown spend, not zero spend.
    """
    usage = data.get("usageMetadata") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("promptTokenCount")
    if not isinstance(prompt, int) or isinstance(prompt, bool):
        return None
    billed_as_output = 0
    for key in ("candidatesTokenCount", "thoughtsTokenCount"):
        value = usage.get(key, 0)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        billed_as_output += value
    return gemini_usage_cost_usd(prompt, billed_as_output)


def _extract_text(data: dict) -> str:
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return ""
    return parts[0].get("text", "").strip()
