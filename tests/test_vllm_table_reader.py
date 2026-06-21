"""GH-88: vLLM/OpenAI crop reader + backend-selection factory (HPC support).

Hermetic: mocks the HTTP layer; no vLLM/Ollama/GPU.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from socr.tables.extract import (
    OllamaTableReader,
    VllmTableReader,
    make_table_reader,
)


def _png(tmp_path: Path) -> Path:
    p = tmp_path / "crop.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    return p


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
    def test_posts_openai_multimodal_and_parses_choice(self, tmp_path):
        reader = VllmTableReader(
            model="Qwen/Qwen3-VL-30B-A3B-Instruct", base_url="http://h:8000/v1"
        )
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
            out = reader.read(_png(tmp_path))

        assert out.strip().startswith("| a | b |")
        # Hits the OpenAI chat endpoint, not Ollama's /api/generate.
        assert captured["url"] == "http://h:8000/v1/chat/completions"
        content = captured["json"]["messages"][0]["content"]
        kinds = {part["type"] for part in content}
        assert kinds == {"text", "image_url"}
        img = next(p for p in content if p["type"] == "image_url")
        assert img["image_url"]["url"].startswith("data:image/png;base64,")
        assert captured["json"]["temperature"] == 0

    def test_empty_choices_does_not_crash(self, tmp_path):
        reader = VllmTableReader(model="m")
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: {"choices": []}
        with patch("socr.tables.extract.httpx.post", return_value=resp):
            assert reader.read(_png(tmp_path)) == ""
