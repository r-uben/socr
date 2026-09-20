"""Crop-pass table extractor (Pass B).

Crops each located table to a high-resolution PNG and re-reads it with a
table-specialised VLM pass. Reusing the judge's direct image -> Ollama path
(``/api/generate`` with a base64 image) rather than the CLI engines, because we
want to hand the model *just the table crop*, not a whole rendered page.

The VLM call is injected (``TableReader`` protocol) so the cropping/orchestration
logic is testable without a model. ``OllamaTableReader`` is the default backend,
mirroring ``OllamaVisionJudge``.
"""

from __future__ import annotations

import base64
import concurrent.futures
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from socr.core.killable import CallSpec, run_killable
from socr.tables.locate import TableBox

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "table_extract.md"

# Crops cover a small fraction of a page, so a moderate render DPI keeps small
# table digits/parens crisp while staying fast. 400 dpi was measured to push a
# single dense-table crop read on qwen3-vl:30b-a3b-instruct past the per-crop
# wall-clock deadline (GH-56), which forced crop timeouts and disabled the
# Tier-2 crop-repair fallback. 250 dpi is ~3x faster per read (a full page reads
# in ~120 s at 200 dpi) and remains legible for dense numerals, so crop reads fit
# the deadline and the fallback can actually fire. Rendering setting, not a model
# threshold.
DEFAULT_CROP_DPI = 250
# Padding (PDF points) around a located bbox so a rule or edge digit is never
# clipped by an off-by-a-pixel boundary.
#
# PUBLIC (GH-367): the constrained cell transcriber must crop with the SAME
# padding as the table witnesses it is adjudicating, or it is not looking at
# like for like. Sharing one constant is what makes that guarantee checkable;
# a second copy would let the two drift silently.
CROP_PADDING_PT = 6.0

# Wall-clock deadline (seconds) applied per-crop in addition to the httpx I/O
# timeout. Crops are small (fraction of a page), so a crop reread at
# qwen3-vl:30b-a3b-instruct should complete well under 120 s. The ThreadPoolExecutor
# guard (mirroring agentic.py route_page) guarantees the call cannot hold the
# pipeline hostage even when the httpx read-timeout does not fire (e.g. a wedged
# Ollama socket that never closes the response stream).
#
# Residual: closing the httpx client does NOT abort Ollama server-side generation
# (stream:false). The GPU continues until the model finishes, so strict-local mode
# stays serial even after a crop is abandoned. Ollama-side cancellation is out of
# scope; callers that need hard GPU preemption must restart the Ollama process.
#
# Basis: the httpx OllamaTableReader.timeout default is 120 s; we give the
# ThreadPoolExecutor a headroom multiplier of 2.0 so the I/O timeout can fire
# first in normal wedge scenarios and dense multi-table pages (GH-56) get modest
# extra room before the wall-clock guard trips. Callers may pass a different
# deadline.
_CROP_WALL_CLOCK_MULTIPLIER = 2.0
# Minimum deadline, regardless of multiplier outcome, to avoid rounding to 0.
_CROP_DEADLINE_FLOOR_S = 30.0


def crop_wall_clock_deadline(reader_timeout_s: float) -> float:
    """Return the per-crop ThreadPoolExecutor wall-clock deadline in seconds.

    Derived from the reader's httpx timeout: give the OS I/O timeout a chance to
    fire first (multiplier > 1), but never less than the floor. The result is not
    a magic constant — it tracks the configured reader timeout so both guards
    scale together when models or hardware change.
    """
    return max(_CROP_DEADLINE_FLOOR_S, reader_timeout_s * _CROP_WALL_CLOCK_MULTIPLIER)


DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# What a liveness probe is allowed to swallow. ``httpx.InvalidURL`` is NOT an
# ``httpx.HTTPError`` — it is raised while building the request, before any
# transport runs — so a malformed host (a typo in ``OLLAMA_HOST``, an IPv6
# literal with no unambiguous reading) used to escape the probe and propagate
# out of the cascade-halt guard. That is a failure path: it must answer "not
# idle", never lose the document.
_PROBE_ERRORS = (httpx.HTTPError, httpx.InvalidURL, OSError)

# Backends served over an OpenAI-compatible HTTP API rather than by an Ollama
# daemon. Single source of truth: ``make_table_reader`` picks the crop reader
# with it, and the cascade-halt probe picks which server to ask with it, so the
# two cannot drift into asking different machines about the same run.
OPENAI_COMPATIBLE_BACKENDS: frozenset[str] = frozenset({"vllm", "sglang", "openai", "api"})


def _bracket_bare_ipv6(candidate: str) -> str:
    """Bracket an unbracketed IPv6 literal in *candidate* (scheme optional).

    ``OLLAMA_HOST=::1`` is a spelling users and shell exports actually produce,
    and RFC 3986 requires the brackets: without them ``http://::1`` is not a
    URL, ``urlsplit`` cannot find a port in it, and httpx cannot connect to it.

    A host token is read as a bare IPv6 literal when it holds more than one
    colon and does not already start with ``[`` — one colon is a port
    (``gpu-node:9999``), more than one cannot be. That also brackets the
    ambiguous ``::1:11434``, which has no unambiguous reading: RFC 3986 says a
    port after an IPv6 literal needs the brackets, so this is treated as an
    address, not as address-plus-port.
    """
    scheme, sep, rest = candidate.partition("://")
    if not sep:
        scheme, rest = "http", candidate
    hostpart, slash, tail = rest.partition("/")
    if not hostpart.startswith("[") and hostpart.count(":") > 1:
        hostpart = f"[{hostpart}]"
    return f"{scheme}://{hostpart}{slash}{tail}" if slash else f"{scheme}://{hostpart}"


def resolve_ollama_host(host: str | None = None) -> str:
    """The Ollama base URL this deployment actually uses (GH-222).

    Resolution order, most specific first: an explicit argument, then the
    ``OLLAMA_HOST`` environment variable, then the localhost default. The env
    var is the one the Ollama daemon and its official client already read, so
    a deployment that has pointed those at a remote or non-default host has
    already said where its backend lives — socr should not need to be told a
    second time, and must not silently disagree.

    Accepts the daemon's own bare ``host`` / ``host:port`` spelling as well as a
    full URL; a value with no scheme is read as ``http://``, and a bare IPv6
    literal is bracketed. A blank or whitespace-only env var is treated as unset
    rather than as an empty host.

    The port is filled in when the value omits it, because ``OLLAMA_HOST`` is
    very commonly spelled bare (``127.0.0.1``, ``gpu-node``) and the daemon
    itself supplies 11434 in that case.  Reading it literally would resolve to
    ``http://127.0.0.1`` — port 80 — and turn honouring the variable into a NEW
    way to probe the wrong place, which is the defect this function exists to
    remove rather than relocate.

    A value this cannot parse is returned unchanged rather than raising: the
    caller is a liveness probe on a failure path, and a malformed host must
    produce a failed probe, never an exception that loses the document.
    """
    import os
    from urllib.parse import urlsplit, urlunsplit

    candidate = (host or os.environ.get("OLLAMA_HOST") or "").strip()
    if not candidate:
        return DEFAULT_OLLAMA_HOST
    candidate = _bracket_bare_ipv6(candidate)
    try:
        parts = urlsplit(candidate)
        has_port = parts.port is not None
    except ValueError:  # malformed host or port — leave the value exactly as given
        logger.warning("GH-222: cannot parse backend host %r; using it verbatim", candidate)
        return candidate
    if not has_port and parts.hostname:
        default_port = urlsplit(DEFAULT_OLLAMA_HOST).port
        parts = parts._replace(netloc=f"{parts.netloc}:{default_port}")
        candidate = urlunsplit(parts)
    return candidate


def _default_canary_model() -> str:
    """The model a canary probes when the caller does not name one.

    ``probe_ollama_idle``/``probe_openai_server_idle`` are called from the
    orchestrator's ``_probe_backend_idle`` with no ``model`` argument — that
    call site has no reader instance to read ``.model`` off of, only a host.
    Falling back to ``PROFILE_QWEN_LOCAL.model`` is not a guess: it is the
    repo's one supported local VLM (CLAUDE.md), the same tag every other local
    OCR call in this codebase already targets. A reader-driven caller
    (``_probe_reader_idle``, below) always passes the reader's own model and
    never hits this default.
    """
    from socr.core.providers import PROFILE_QWEN_LOCAL

    return PROFILE_QWEN_LOCAL.model


# GH-221 review: every vision call in this codebase sends an ``images``/``image_url``
# payload (judge/ollama_judge.py, judge/table_rung_ollama.py, math/equation_latex.py,
# engines/gemini_api.py) -- and so does TableCropExtractor, the workload this canary
# guards. A text-only probe exercises a different code path than the one that wedges,
# so it must send an image too: a probe that answers healthy for the exact failure it
# exists to detect is the same failure class GH-221 was filed to close. This is the
# smallest legal PNG (1x1, transparent) -- a decoded, fixed image so the canary payload
# never depends on disk state or an actual table crop.
_CANARY_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _ollama_generation_canary(host: str, model: str, timeout: float) -> bool:
    """GH-221: the only thing that tells "HTTP alive" from "GPU available".

    ``/api/tags`` answers instantly regardless of what the model is doing —
    the issue measured it returning 200 OK in 0.05-0.14s while the same model
    was mid-generation at 100% GPU. Ollama serialises ``/api/generate`` calls
    per model: a request sent while a prior one is still running QUEUES behind
    it rather than returning, so a single-token request (``num_predict: 1``)
    answers almost instantly on a genuinely idle backend and blocks for the
    full *timeout* on a wedged one — which is exactly the distinction
    ``/api/tags`` cannot make.

    Carries ``images`` because the workload this guards (``TableCropExtractor``)
    is a vision call, not a text one: a probe must exercise the code path it
    guards, or a hang localised to image handling could pass a text-only canary
    while the vision path stays wedged.
    """
    try:
        resp = httpx.post(
            f"{host.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": "ok",
                "images": [_CANARY_IMAGE_B64],
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return True
    except _PROBE_ERRORS:
        return False


def _openai_generation_canary(base_url: str, model: str, timeout: float) -> bool:
    """The OpenAI-compatible (vLLM/SGLang) sibling of ``_ollama_generation_canary``.

    Same reasoning: a chat-completion request with ``max_tokens: 1`` queues
    behind an in-flight generation on the same server rather than returning,
    so it distinguishes "the HTTP layer answers" from "the GPU can serve a new
    request within *timeout*". Carries an ``image_url`` message part for the
    same reason as the Ollama canary: the workload it guards is a vision call.
    """
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json={
                "model": model,
                "max_tokens": 1,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "ok"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{_CANARY_IMAGE_B64}"},
                            },
                        ],
                    }
                ],
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return True
    except _PROBE_ERRORS:
        return False


def probe_openai_server_idle(
    base_url: str,
    timeout: float = 5.0,
    *,
    model: str | None = None,
    generation_timeout: float | None = None,
) -> bool:
    """Return True if an OpenAI-compatible VLM server (vLLM/SGLang) can serve now.

    GH-222: the Ollama probe asks ``/api/tags``, an endpoint a vLLM server does
    not serve, so pointing it at one reports a healthy machine dead. The
    OpenAI-compatible equivalent precondition is ``/models`` under the same
    ``/v1`` base URL the crop reader already talks to.

    GH-221: ``/models`` is an HTTP-layer liveness ping, NOT a check that the
    GPU is free — a server mid-generation answers it too. Once that
    precondition passes, a minimal generation call (``_openai_generation_canary``)
    is the only thing that tells the two apart. ``generation_timeout`` defaults
    to ``_CROP_DEADLINE_FLOOR_S``, the wall-clock floor this pipeline already
    budgets for a normal crop read — no new number invented for this probe.
    """
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        resp.raise_for_status()
    except _PROBE_ERRORS:
        return False
    return _openai_generation_canary(
        base_url,
        model or _default_canary_model(),
        generation_timeout if generation_timeout is not None else _CROP_DEADLINE_FLOOR_S,
    )


def probe_ollama_idle(
    host: str | None = None,
    timeout: float = 5.0,
    *,
    model: str | None = None,
    generation_timeout: float | None = None,
) -> bool:
    """Return True if the Ollama backend can actually serve a request now.

    Used as a cascade guard after a crop timeout: if the backend is wedged or
    unreachable, the pipeline should not fire additional VLM calls into it.

    GH-221: an earlier version of this function was a lightweight ``/api/tags``
    ping — it did NOT check whether a generation was still running
    server-side, only that the HTTP layer answered. The issue measured that
    ping returning 200 OK in 0.05-0.14s while the model was mid-generation at
    100% GPU, so the cascade-halt guard never armed for the exact failure it
    exists to catch. ``/api/tags`` is now only the cheap precondition — an
    unreachable host still fails fast — and a minimal generation request
    (``_ollama_generation_canary``) is the evidence that actually gates the
    return value. ``generation_timeout`` defaults to ``_CROP_DEADLINE_FLOOR_S``,
    the wall-clock floor this pipeline already budgets for a normal crop read;
    no new threshold is invented for this probe.

    GH-222: ``host`` used to default to a hardcoded ``http://localhost:11434``,
    and the cascade call site passed nothing. On any deployment without a local
    Ollama daemon — vLLM, HPC, a remote Ollama host — this returned False
    unconditionally, forever, on a perfectly healthy machine, and a single
    timeout anywhere in the ladder truncated the document with a
    ``PARTIAL_SAVE_VLM_TIMEOUT`` that named a cause which never happened.
    ``None`` now means "resolve it" rather than "assume localhost".
    """
    resolved = resolve_ollama_host(host)
    try:
        resp = httpx.get(f"{resolved.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except _PROBE_ERRORS:
        return False
    return _ollama_generation_canary(
        resolved,
        model or _default_canary_model(),
        generation_timeout if generation_timeout is not None else _CROP_DEADLINE_FLOOR_S,
    )


def load_table_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


class TableReader(Protocol):
    """Anything that turns a table-crop image into Markdown."""

    def read(self, image_path: Path) -> str: ...


@dataclass
class CropTable:
    """One crop-pass result, in reading order."""

    markdown: str
    source: str  # locator tag from TableBox ("ruled" | "booktabs")
    bbox: tuple[float, float, float, float]


def _ollama_read_crop(host: str, model: str, prompt: str, image_b64: str, timeout: float) -> str:
    """The ONLY part of ``OllamaTableReader.read`` that crosses the killable
    boundary (GH-798).

    Top-level and picklable (plain str/float args, never a bound method or
    closure) so a fresh ``spawn``-ed child can resolve it via ``CallSpec`` --
    mirrors ``judge/ollama_judge.py:_post_generate`` (GH-172 site 1), the same
    pattern applied to the crop-reread caller ``_read_with_deadline``
    (``TableCropExtractor``) already wraps in its own wall-clock
    ``ThreadPoolExecutor`` deadline. That outer wrapper only ABANDONS a
    wedged thread (GH-172), so it cannot stop a peer that trickles bytes
    faster than ``httpx``'s per-chunk read timeout
    (``docs/log/2026-09-17_172-design.md``) -- the ``httpx`` timeout below is
    defence-in-depth only; the caller's ``run_killable`` deadline is what
    actually bounds that case, by killing the process making the call.
    """
    resp = httpx.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0},  # transcription must be deterministic
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return _clean_markdown(resp.json().get("response", ""))


def _vllm_read_crop(
    base_url: str, model: str, api_key: str, prompt: str, image_b64: str, timeout: float
) -> str:
    """The killable-boundary counterpart of ``_ollama_read_crop``, for
    ``VllmTableReader.read`` (GH-798). Same reasoning; see that function's
    docstring."""
    resp = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "temperature": 0,  # transcription must be deterministic
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    choices = resp.json().get("choices") or [{}]
    return _clean_markdown(choices[0].get("message", {}).get("content", ""))


class OllamaTableReader:
    """Default crop reader: a local/cloud Ollama vision model via /api/generate."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._prompt = load_table_prompt()

    def read(self, image_path: Path) -> str:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        spec = CallSpec(
            func="socr.tables.extract:_ollama_read_crop",
            args=(self.host, self.model, self._prompt, image_b64, self.timeout),
        )
        return run_killable(spec, timeout=self.timeout)


class VllmTableReader:
    """Crop reader for an OpenAI-compatible server (vLLM / SGLang).

    Used on HPC, where Ollama/llama.cpp are forbidden on server GPUs and the
    VLM is served by vLLM. Talks ``/v1/chat/completions`` with the crop image as
    a base64 data URL (OpenAI multimodal message). Mirrors ``OllamaTableReader``
    so it drops into the same ``TableReader`` slot; exposes ``.timeout`` so the
    crop wall-clock deadline scales the same way.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        timeout: float = 120.0,
        api_key: str = "EMPTY",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.host = base_url  # for cascade-probe parity with OllamaTableReader
        self._api_key = api_key
        self._prompt = load_table_prompt()

    def read(self, image_path: Path) -> str:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        spec = CallSpec(
            func="socr.tables.extract:_vllm_read_crop",
            args=(self.base_url, self.model, self._api_key, self._prompt, image_b64, self.timeout),
        )
        return run_killable(spec, timeout=self.timeout)


def make_table_reader(
    *,
    backend: str,
    model: str,
    timeout: float = 120.0,
    ollama_host: str | None = None,
    vllm_url: str = "http://localhost:8000/v1",
) -> TableReader:
    """Build the crop reader for *backend* ("vllm"/"sglang" -> OpenAI server, else Ollama).

    Keeps Ollama as the default local path; selects the vLLM/OpenAI reader for
    server/HPC backends. Single place that maps backend -> reader so both crop
    construction sites stay consistent.
    """
    if backend in OPENAI_COMPATIBLE_BACKENDS:
        return VllmTableReader(model=model, base_url=vllm_url, timeout=timeout)
    return OllamaTableReader(model=model, host=resolve_ollama_host(ollama_host), timeout=timeout)


def _probe_reader_idle(reader: object) -> bool:
    """Liveness ping for whichever server *reader* talks to (GH-222/GH-221).

    ``VllmTableReader`` and ``OllamaTableReader`` both expose ``.host``, but the
    two servers answer different endpoints, so the endpoint has to follow the
    reader type rather than the host string. ``.model`` is passed through too
    (GH-221) so the functional canary asks about the SAME model this reader
    was reading crops with, rather than falling back to the generic default.
    """
    host = getattr(reader, "host", None) or DEFAULT_OLLAMA_HOST
    model = getattr(reader, "model", None)
    if isinstance(reader, VllmTableReader):
        return probe_openai_server_idle(host, model=model)
    return probe_ollama_idle(host, model=model)


class TableCropExtractor:
    """Render table crops and read them with an injected ``TableReader``."""

    def __init__(self, reader: TableReader, crop_dpi: int = DEFAULT_CROP_DPI) -> None:
        self._reader = reader
        self._crop_dpi = crop_dpi

    def extract(
        self,
        pdf_path: Path,
        page_num: int,
        boxes: list[TableBox],
        *,
        deadline: float | None = None,
        cascade_probe: bool = True,
    ) -> list[CropTable]:
        """Crop each box on ``page_num`` (1-indexed) and read it. Never raises.

        Each VLM call is wrapped in a ThreadPoolExecutor wall-clock guard (mirroring
        agentic.py ``route_page``). On ``TimeoutError`` the future is abandoned and a
        ``CropTimeout`` sentinel is injected into the output list so callers can
        record an audit event and apply the cascade guard without re-running
        detection. ``deadline`` defaults to ``crop_wall_clock_deadline(reader.timeout)``
        when the reader exposes a ``timeout`` attribute; otherwise 180 s.

        cascade_probe — when True (default), after any crop timeout this method
        pings the Ollama backend; if it is unreachable, all remaining crops are
        skipped and ``_backend_degraded`` is set on self. A future PP-2 document-
        level halt can test this attribute to abort the whole document.

        GH-166: a failed crop/read (non-timeout) now returns a TYPED SENTINEL --
        a ``CropTable`` with empty markdown and a ``_failed`` reason
        (``render_failed`` / ``read_error`` / ``empty_response`` /
        ``backend_degraded``) -- rather than no entry at all. The orchestrator
        turns each into a ``dualpass_crop_failed`` audit event, which is a
        distrust kind, so a reread that verified nothing cannot leave the
        incumbent table looking verified. It previously dropped that table
        rather than aborting the page — the reconciler then sees a count mismatch
        and flags rather than patches, which is the safe outcome.
        """
        from socr.core.pdf import open_pdf

        # Resolve the per-crop wall-clock deadline.
        if deadline is None:
            reader_timeout = getattr(self._reader, "timeout", None)
            deadline = (
                crop_wall_clock_deadline(reader_timeout) if reader_timeout is not None else 180.0
            )

        out: list[CropTable] = []
        try:
            doc = open_pdf(pdf_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("dual-pass: cannot open %s (%s)", pdf_path, exc)
            return out
        try:
            page = doc[page_num - 1]
            page_rect = page.rect
            for i, box in enumerate(boxes):
                if getattr(self, "_backend_degraded", False):
                    # Cascade guard: a prior timeout left the GPU in an unknown
                    # state — don't fire more VLM calls into the wedged backend.
                    logger.warning(
                        "dual-pass: backend degraded; skipping remaining crops on p%d",
                        page_num,
                    )
                    # GH-166 review (P1): the skip must leave a TRACE. Breaking
                    # with no sentinel meant a page skipped entirely after a
                    # prior timeout produced an empty `raw_crops`, so the
                    # orchestrator had nothing to iterate and emitted no
                    # distrust -- the skipped page looked verified, which is
                    # this ticket's defect one level up.
                    out.extend(self._failed_crop(b, "backend_degraded") for b in boxes[i:])
                    break
                img_path = self._render_crop(page, box, page_rect)
                if img_path is None:
                    out.append(self._failed_crop(box, "render_failed"))
                    continue
                try:
                    md = self._read_with_deadline(img_path, deadline, page_num)
                except _CropTimeoutError:
                    # Timeout: emit a sentinel so the orchestrator can log an audit
                    # event and apply the cascade guard.
                    out.append(
                        CropTable(
                            markdown="",
                            source=box.source,
                            bbox=box.bbox,
                        )
                    )
                    out[-1]._timed_out = True  # type: ignore[attr-defined]
                    # Mark backend degraded UNCONDITIONALLY after any crop timeout.
                    # /api/tags may answer while /api/generate is still running on
                    # the GPU, so using the probe as the degradation condition is
                    # unsound. We degrade first, then probe only to enrich the log.
                    self._backend_degraded = True
                    if cascade_probe:
                        # GH-222: ask the reader's OWN server, with the endpoint
                        # that server actually serves. ``VllmTableReader.host``
                        # is its ``/v1`` base URL, and /api/tags is not there —
                        # probing it reported "backend down" on every healthy
                        # HPC run. The result only enriches this log line
                        # (degradation above is unconditional), but a log line
                        # that names a hardware failure that never happened is
                        # exactly what #222 was filed about.
                        idle = _probe_reader_idle(self._reader)
                        logger.warning(
                            "dual-pass: crop timeout on p%d — backend marked degraded "
                            "(probe idle=%s)",
                            page_num,
                            idle,
                        )
                    img_path.unlink(missing_ok=True)
                    continue
                except Exception as exc:
                    logger.warning("dual-pass: crop read failed p%d (%s)", page_num, exc)
                    img_path.unlink(missing_ok=True)
                    out.append(self._failed_crop(box, "read_error"))
                    continue
                img_path.unlink(missing_ok=True)
                if md.strip():
                    out.append(CropTable(markdown=md, source=box.source, bbox=box.bbox))
                else:
                    out.append(self._failed_crop(box, "empty_response"))
        finally:
            doc.close()
        return out

    def _read_with_deadline(self, img_path: Path, deadline: float, page_num: int) -> str:
        """Submit ``reader.read`` to a single-worker executor; raise ``_CropTimeoutError``
        if the wall-clock deadline expires.

        Mirrors the ``ThreadPoolExecutor`` pattern in ``agentic.py route_page``:
        abandon the future with ``wait=False`` so the pipeline is not blocked by
        a stalled thread.

        The abandoned thread is NOT a daemon (GH-172) -- see the note at the
        ``ex.shutdown(wait=False)`` below, and the matching one in
        ``route_page``. It keeps the process alive until it unblocks, and on a
        wedged socket nothing guarantees that it does.
        """
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = ex.submit(self._reader.read, img_path)
        try:
            return future.result(timeout=deadline)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.warning(
                "dual-pass: crop VLM call timed out after %.1f s on p%d — releasing",
                deadline,
                page_num,
            )
            raise _CropTimeoutError(deadline, page_num)
        finally:
            # wait=False: release the executor without blocking. On the success
            # path this reclaims the idle worker immediately; on timeout the
            # stalled thread (blocked on httpx response) is NOT reaped — it keeps
            # running until it unblocks. ThreadPoolExecutor workers are NOT daemon
            # threads; they keep the process alive if it tries to exit while a
            # worker is blocked.
            #
            # GH-172 measured the exit path rather than assuming it: the process
            # exits when the worker returns, and nothing short of that releases
            # it (`threading._shutdown` joins on locks captured at thread START,
            # so re-flagging the thread as daemonic afterwards changes nothing).
            #
            # How long that is depends on WHY the deadline fired, and the two
            # cases are not alike:
            #
            #  - a merely slow call unblocks at ``self._reader``'s own httpx
            #    timeout, so the residual delay is bounded;
            #  - a WEDGED socket does not. That is the case the header comment
            #    at the top of this file describes, and the reason this
            #    wall-clock deadline exists at all: the httpx read-timeout does
            #    not fire when the server never closes the response stream. The
            #    wait is then unbounded, and it is the case #172 is about.
            #
            # An earlier revision of this comment claimed the httpx timeout
            # bounds it in general. It does not, and saying so here would have
            # sent the #172 fix looking in the wrong place.
            ex.shutdown(wait=False)

    def _failed_crop(self, box, reason: str):
        """A typed sentinel for a crop that was located but produced nothing.

        GH-166. Render errors, reader exceptions and empty responses each did a
        bare ``continue``, so a page whose crops ALL failed returned an empty
        list -- indistinguishable from a page with no crops to read. The
        incumbent table then looked verified because the check that would have
        contradicted it left no trace.

        Mirrors the existing ``_timed_out`` sentinel rather than inventing a
        second mechanism: same ``CropTable`` shape, empty markdown, one marker
        attribute the orchestrator reads.
        """
        crop = CropTable(markdown="", source=box.source, bbox=box.bbox)
        crop._failed = reason  # type: ignore[attr-defined]
        return crop

    def _render_crop(self, page, box: TableBox, page_rect) -> Path | None:
        import fitz
        from PIL import Image

        from socr.core.born_digital import upright_rotation_for

        x0, y0, x1, y1 = box.bbox
        clip = fitz.Rect(
            max(page_rect.x0, x0 - CROP_PADDING_PT),
            max(page_rect.y0, y0 - CROP_PADDING_PT),
            min(page_rect.x1, x1 + CROP_PADDING_PT),
            min(page_rect.y1, y1 + CROP_PADDING_PT),
        )
        # GH-304b: derive clip-local rotation; keep bbox and clip in page space, rotate only raster pixels.
        rotation = upright_rotation_for(page, clip=clip)
        mat = fitz.Matrix(self._crop_dpi / 72, self._crop_dpi / 72)
        if rotation != 0:
            mat.prerotate(rotation)
        try:
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("dual-pass: crop render failed (%s)", exc)
            return None
        fd, name = tempfile.mkstemp(prefix="socr_tablecrop_", suffix=".png")
        path = Path(name)
        try:
            import os

            os.close(fd)
            img.save(path)
        except Exception:  # pragma: no cover - defensive
            path.unlink(missing_ok=True)
            return None
        return path


class _CropTimeoutError(Exception):
    """Internal sentinel: a single crop VLM call exceeded its wall-clock deadline."""

    def __init__(self, deadline: float, page_num: int) -> None:
        super().__init__(f"crop reread timed out after {deadline:.1f}s on p{page_num}")
        self.deadline = deadline
        self.page_num = page_num


def _clean_markdown(text: str) -> str:
    """Strip code fences / stray prose, keep the markdown table lines.

    The prompt asks for a bare table, but small models sometimes wrap it in a
    ```` ```markdown ```` fence or add a lead-in line. Keep the contiguous run of
    pipe-bearing lines.
    """
    lines = text.strip().splitlines()
    cleaned = [ln for ln in lines if not ln.strip().startswith("```")]
    table_lines = [ln for ln in cleaned if "|" in ln]
    return "\n".join(table_lines).strip() if table_lines else ""
