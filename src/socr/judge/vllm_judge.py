"""Local-VLM judge backend (vLLM / any OpenAI-compatible vision server).

GH-873: ``OllamaVisionJudge`` is the only VLM judge socr can build, and its
model candidates are probed over the Ollama HTTP API. On a box with no Ollama
daemon -- the Bocconi HPC nodes, where vLLM serves the OCR model and Ollama is
not installed at all -- ``_resolve_judge_model`` can never resolve, and
``_build_page_judge`` degrades to the heuristic judge on every run. The vision
model is already loaded and serving in the same job; it simply could not be
reached, because the OCR engine has vLLM settings (``--qwen-backend vllm``,
``--qwen-vllm-url``, ``--qwen-vllm-model``) and the judge has none.

This module is that missing pair. It mirrors :mod:`socr.judge.ollama_judge`
deliberately -- same two-method surface (``is_available`` / ``judge``), same
prompt, same verdict parser, same killable process boundary -- so the judge
ladder, the fingerprint and the timeout classification machinery treat the two
backends identically and nothing downstream has to know which one ran.

The wire format is the only real difference. Ollama takes ``/api/generate``
with a bare base64 string in ``images``; an OpenAI-compatible server takes
``/v1/chat/completions`` with the image as a ``data:`` URI inside the message
content. Both are asked for temperature 0 and a JSON object.

Locality: an operator who passes ``--judge-vllm-url`` has named a specific
server, exactly as ``--judge-model`` names a specific model. Neither is
probed against the cloud candidate ladder, and neither is treated as a cloud
identity -- a vLLM endpoint is whatever host the operator pointed it at, and
socr has no way to price it. The URL is therefore honoured under
``--strict-local``; pointing it at a metered remote endpoint is an operator
decision socr does not second-guess, the same way it does not second-guess
``--qwen-vllm-url``.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

from socr.core.killable import CallSpec, run_killable
from socr.judge.judge import JudgeVerdict, load_judge_prompt, parse_verdict

#: No default model. Unlike Ollama there is no candidate ladder to probe: an
#: OpenAI-compatible server serves whatever it was launched with, so the
#: operator must name it. An empty model is "not configured", never a guess.
DEFAULT_MODEL = ""

#: Matches ``PipelineConfig.qwen_vllm_url`` so a single-server job can point
#: both at the same place without repeating it.
DEFAULT_URL = "http://localhost:8000/v1"


def _image_media_type(path: Path) -> str:
    """Media type for the ``data:`` URI, from the suffix.

    The page renderer writes PNG, but the judge is handed whatever path the
    caller rendered, so a JPEG must not be announced as a PNG -- some servers
    decode strictly on the declared type.
    """
    suffix = path.suffix.casefold()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _post_chat(
    base_url: str,
    model: str,
    prompt: str,
    image_data_uri: str,
    timeout: float,
) -> str:
    """The ONLY part of ``judge()`` that crosses the killable boundary.

    Same contract as ``ollama_judge._post_generate`` and for the same reason
    (GH-172): a top-level function taking plain picklable arguments, never a
    bound method, so ``run_killable`` can execute it in a fresh ``spawn``-ed
    child and kill it if the peer wedges. The ``httpx`` timeout here is
    defence-in-depth only -- it is per-chunk, so a peer that trickles bytes
    into an open response defeats it; the caller's ``run_killable`` wall-clock
    deadline is what actually bounds this call.
    """
    resp = httpx.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                }
            ],
            # Judging should be as stable as we can make it, same as the
            # Ollama backend.
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    choices = payload.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content") or ""


class VLLMVisionJudge:
    """Judge backed by an OpenAI-compatible vision server (vLLM, SGLang, …)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._prompt = load_judge_prompt()

    def is_available(self) -> bool:
        """True if the server is up and THIS EXACT model id is being served.

        Matched on the full id, for the reason #133 gives for the Ollama
        backend: a family-prefix match let a server satisfy a request for a
        model it had never loaded, and the failure then surfaced at judge time
        as a 404 rather than as unavailability. An unset model is never
        available -- there is nothing to match against, and guessing the
        served model would reintroduce exactly that bug.
        """
        if not self.model:
            return False
        try:
            resp = httpx.get(f"{self.base_url}/models", timeout=5.0)
            resp.raise_for_status()
            served = {m.get("id", "") for m in resp.json().get("data", [])}
            return self.model in served
        except (httpx.HTTPError, ValueError):
            return False

    def judge(self, image_path: Path, ocr_text: str) -> JudgeVerdict:
        """Judge one page. The model call runs behind a killable boundary."""
        path = Path(image_path)
        image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        data_uri = f"data:{_image_media_type(path)};base64,{image_b64}"
        prompt = f"{self._prompt}\n\n---\nCANDIDATE TRANSCRIPTION:\n\n{ocr_text}"
        spec = CallSpec(
            func="socr.judge.vllm_judge:_post_chat",
            args=(self.base_url, self.model, prompt, data_uri, self.timeout),
        )
        raw = run_killable(spec, timeout=self.timeout)
        return parse_verdict(raw)
