"""GH-88: vLLM/OpenAI crop reader + backend-selection factory (HPC support).

Hermetic: mocks the HTTP layer; no vLLM/Ollama/GPU.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

from socr.engines.qwen import QwenEngine
from socr.tables.extract import (
    OllamaTableReader,
    VllmTableReader,
    _vllm_read_crop,
    make_table_reader,
)


class TestQwenAvailabilityVllm:
    """GH-88: with VLLM_BASE_URL set, qwen is available without the Ollama model."""

    def test_vllm_url_makes_qwen_available_without_ollama(self, monkeypatch):
        monkeypatch.setattr("socr.engines.base.BaseEngine.is_available", lambda self: True)
        # Ollama model is MISSING — must be bypassed when VLLM_BASE_URL is set.
        monkeypatch.setattr("socr.engines.qwen._check_ollama_model", lambda m: "not pulled")
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        assert QwenEngine().is_available() is True

    def test_no_vllm_url_still_requires_ollama(self, monkeypatch):
        monkeypatch.setattr("socr.engines.base.BaseEngine.is_available", lambda self: True)
        monkeypatch.setattr("socr.engines.qwen._check_ollama_model", lambda m: "not pulled")
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        assert QwenEngine().is_available() is False


class TestMakeTableReader:
    def test_vllm_backend_returns_vllm_reader(self):
        r = make_table_reader(backend="vllm", model="Qwen/Qwen3-VL-30B-A3B-Instruct")
        assert isinstance(r, VllmTableReader)
        assert r.model == "Qwen/Qwen3-VL-30B-A3B-Instruct"

    def test_sglang_and_api_also_route_to_openai_reader(self):
        assert isinstance(make_table_reader(backend="sglang", model="m"), VllmTableReader)
        assert isinstance(make_table_reader(backend="api", model="m"), VllmTableReader)

    def test_ollama_and_auto_return_ollama_reader(self):
        assert isinstance(make_table_reader(backend="ollama", model="m"), OllamaTableReader)
        assert isinstance(make_table_reader(backend="auto", model="m"), OllamaTableReader)

    def test_vllm_url_threaded_through(self):
        r = make_table_reader(
            backend="vllm", model="m", vllm_url="http://node07:8000/v1", timeout=300.0
        )
        assert r.base_url == "http://node07:8000/v1"
        assert r.timeout == 300.0  # so the crop wall-clock deadline scales correctly


class TestVllmTableReaderRead:
    """GH-798: ``VllmTableReader.read`` now crosses a ``run_killable`` process
    boundary, so a patch on this process's ``httpx.post`` cannot reach the
    spawned child that actually makes the call (the same reason
    ``judge/ollama_judge.py``'s ``_post_generate`` is tested directly rather
    than through ``OllamaVisionJudge.judge()``). These tests exercise
    ``_vllm_read_crop`` -- the one function that crosses the boundary -- in
    process instead; the boundary itself is proven generically by
    ``run_killable``'s own tests and by the trickle test in
    ``test_gh798_crop_reader_killable.py``.
    """

    def test_posts_openai_multimodal_and_parses_choice(self):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json = lambda: {
                "choices": [{"message": {"content": "| a | b |\n| --- | --- |\n| 1 | 2 |"}}]
            }
            return resp

        with patch("socr.tables.extract.httpx.post", side_effect=fake_post):
            out = _vllm_read_crop(
                "http://h:8000/v1",
                "Qwen/Qwen3-VL-30B-A3B-Instruct",
                "EMPTY",
                "prompt",
                "aGk=",
                120.0,
            )

        assert out.strip().startswith("| a | b |")
        # Hits the OpenAI chat endpoint, not Ollama's /api/generate.
        assert captured["url"] == "http://h:8000/v1/chat/completions"
        content = captured["json"]["messages"][0]["content"]
        kinds = {part["type"] for part in content}
        assert kinds == {"text", "image_url"}
        img = next(p for p in content if p["type"] == "image_url")
        assert img["image_url"]["url"].startswith("data:image/png;base64,")
        assert captured["json"]["temperature"] == 0

    def test_empty_choices_does_not_crash(self):
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: {"choices": []}
        with patch("socr.tables.extract.httpx.post", return_value=resp):
            assert _vllm_read_crop("http://h:8000/v1", "m", "EMPTY", "prompt", "aGk=", 120.0) == ""


class TestVllmTableReaderReadWiring:
    """GH-848: ``VllmTableReader.read``'s OWN body is otherwise unexercised --
    the tests above call ``_vllm_read_crop`` directly, so a revert to an
    in-process ``httpx.post``, a scrambled arg order, or a wrong ``CallSpec``
    func string all stay green. Patches ``run_killable`` itself and pins the
    ``CallSpec`` it is handed.
    """

    def test_read_wires_call_spec_to_vllm_read_crop(self, tmp_path, monkeypatch):
        png_bytes = b"\x89PNG\r\n\x1a\nfake-but-distinct-bytes-848"
        crop = tmp_path / "crop.png"
        crop.write_bytes(png_bytes)

        captured = {}

        def fake_run_killable(spec, timeout):
            captured["spec"] = spec
            captured["timeout"] = timeout
            return "| stub |"

        monkeypatch.setattr("socr.tables.extract.run_killable", fake_run_killable)

        reader = VllmTableReader(
            model="Qwen/Qwen3-VL-30B-A3B-Instruct",
            base_url="http://h:8000/v1",
            timeout=45.0,
            api_key="sk-test",
        )
        out = reader.read(crop)

        assert out == "| stub |"
        spec = captured["spec"]
        assert spec.func == "socr.tables.extract:_vllm_read_crop"
        assert len(spec.args) == 6
        base_url, model, api_key, prompt, image_b64, timeout = spec.args
        assert base_url == "http://h:8000/v1"
        assert model == "Qwen/Qwen3-VL-30B-A3B-Instruct"
        assert api_key == "sk-test"
        assert isinstance(prompt, str) and prompt  # the loaded table prompt
        assert timeout == 45.0
        assert base64.b64decode(image_b64) == png_bytes
