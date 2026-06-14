"""Unit tests for _check_ollama_model — the Ollama model availability probe."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from socr.core.ollama_utils import check_ollama_model as _check_ollama_model


def _make_result(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["ollama", "list"], returncode=returncode, stdout=stdout, stderr="")


_OLLAMA_LIST_HEADER = "NAME                              ID              SIZE    MODIFIED\n"

_TYPICAL_OUTPUT = (
    _OLLAMA_LIST_HEADER
    + "qwen3-vl:30b-a3b-instruct         c871fc73fabc    19 GB   21 hours ago\n"
    + "qwen3-vl:8b                       901cae732162    6.1 GB  10 days ago\n"
    + "deepseek-ocr:latest               abc123456789    4.2 GB  3 days ago\n"
    + "glm-ocr:latest                    def987654321    3.8 GB  5 days ago\n"
)


def test_model_found_with_full_tag():
    """Model matched by full name including tag — the bug this test guards."""
    with patch("subprocess.run", return_value=_make_result(_TYPICAL_OUTPUT)):
        assert _check_ollama_model("qwen3-vl:30b-a3b-instruct") is None


def test_model_found_short_tag():
    with patch("subprocess.run", return_value=_make_result(_TYPICAL_OUTPUT)):
        assert _check_ollama_model("qwen3-vl:8b") is None


def test_model_found_latest_tag():
    with patch("subprocess.run", return_value=_make_result(_TYPICAL_OUTPUT)):
        assert _check_ollama_model("deepseek-ocr:latest") is None


def test_model_not_found():
    with patch("subprocess.run", return_value=_make_result(_TYPICAL_OUTPUT)):
        err = _check_ollama_model("nonexistent:7b")
        assert err is not None
        assert "nonexistent:7b" in err
        assert "ollama pull" in err


def test_ollama_not_running_nonzero_exit():
    with patch("subprocess.run", return_value=_make_result("", returncode=1)):
        err = _check_ollama_model("qwen3-vl:30b-a3b-instruct")
        assert err is not None
        assert "not running" in err.lower() or "not installed" in err.lower()


def test_ollama_binary_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        err = _check_ollama_model("qwen3-vl:30b-a3b-instruct")
        assert err is not None
        assert "not installed" in err.lower()


def test_ollama_timeout():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ollama", timeout=10)):
        err = _check_ollama_model("qwen3-vl:30b-a3b-instruct")
        assert err is not None
        assert "timeout" in err.lower()


def test_empty_list():
    """Empty model list — nothing available."""
    with patch("subprocess.run", return_value=_make_result(_OLLAMA_LIST_HEADER)):
        err = _check_ollama_model("qwen3-vl:30b-a3b-instruct")
        assert err is not None
