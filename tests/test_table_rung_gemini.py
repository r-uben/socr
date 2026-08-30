"""GH-353 TICKET-A3: CLI2 rung — gemini CLI table judge invoker.

All tests patch the module-local `_run_gemini_cli` subprocess seam — never
`PATH`/`shutil.which` — so nothing here shells out to a real `gemini`
binary. A guard fixture asserts `subprocess.run` is never reached
unpatched.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.config import PipelineConfig
from socr.judge.table_verdict import Finding, FindingCode, RungResult
from socr.judge.table_rung_gemini import (
    RUNG_ID,
    build_gemini_argv,
    judge_table_gemini,
    make_gemini_rung,
)

PASS_STDOUT = json.dumps({"verdict": "PASS", "confidence": "high", "findings": []})

FAIL_STDOUT = json.dumps(
    {
        "verdict": "FAIL",
        "confidence": "low",
        "findings": [
            {
                "code": "MISSING_VALUE",
                "where": "row 2, col Total",
                "detail": "1,204 present in the crop, absent from the output",
            }
        ],
    }
)


def _config(**overrides) -> PipelineConfig:
    return PipelineConfig(**overrides)


def _completed(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["gemini"], returncode=returncode, stdout=stdout, stderr=stderr
    )


@pytest.fixture(autouse=True)
def _network_must_not_run():
    """No test may reach a real subprocess; only the patched seam runs."""
    with patch(
        "socr.judge.table_rung_gemini.subprocess.run",
        side_effect=AssertionError("real subprocess.run must never be called in tests"),
    ):
        yield


# --------------------------------------------------------------------------
# argv shape
# --------------------------------------------------------------------------


def test_build_gemini_argv_pins_exact_shape(tmp_path: Path):
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"fake-png")
    prompt = "JUDGE THIS TABLE"

    argv = build_gemini_argv("gemini", crop, prompt)

    assert argv == [
        "gemini",
        "--skip-trust",
        "--approval-mode",
        "plan",
        "--include-directories",
        str(crop.parent),
        "-p",
        f"Image crop: @{crop}\n\n{prompt}",
    ]


def test_build_gemini_argv_uses_configured_binary(tmp_path: Path):
    crop = tmp_path / "crop.png"
    argv = build_gemini_argv("/opt/custom/gemini-cli", crop, "x")
    assert argv[0] == "/opt/custom/gemini-cli"


# --------------------------------------------------------------------------
# happy path: S1 answers routed through A1's parser
# --------------------------------------------------------------------------


def test_judge_table_gemini_pass_is_s1_and_s2(tmp_path: Path):
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"fake-png")
    config = _config()

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        return_value=_completed(PASS_STDOUT),
    ) as mock_run:
        result = judge_table_gemini(crop, "| a |\n|---|\n| 1 |", None, config)

    mock_run.assert_called_once()
    argv, timeout_sec = mock_run.call_args[0]
    assert argv[0] == config.table_judge_rung2_binary
    assert timeout_sec == config.table_judge_timeout_sec
    assert isinstance(result, RungResult)
    assert result.rung == RUNG_ID
    assert result.ok is True
    assert result.verdict is not None
    assert result.verdict.passed is True


def test_judge_table_gemini_fail_is_s1_not_s2(tmp_path: Path):
    crop = tmp_path / "crop.png"
    config = _config()

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        return_value=_completed(FAIL_STDOUT),
    ):
        result = judge_table_gemini(crop, "| a |\n|---|\n| 1 |", None, config)

    assert result.ok is True
    assert result.verdict.passed is False
    assert result.verdict.findings[0].code == FindingCode.MISSING_VALUE


def test_prior_findings_reach_the_prompt(tmp_path: Path):
    crop = tmp_path / "crop.png"
    config = _config()
    prior = [Finding(code=FindingCode.WRONG_BINDING, where="row 1", detail="shifted")]

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        return_value=_completed(PASS_STDOUT),
    ) as mock_run:
        judge_table_gemini(crop, "| a |\n|---|\n| 1 |", prior, config)

    argv, _ = mock_run.call_args[0]
    prompt_arg = argv[argv.index("-p") + 1]
    assert "WRONG_BINDING" in prompt_arg
    assert "shifted" in prompt_arg


def test_first_look_prompt_has_no_prior_findings_leak(tmp_path: Path):
    crop = tmp_path / "crop.png"
    config = _config()

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        return_value=_completed(PASS_STDOUT),
    ) as mock_run:
        judge_table_gemini(crop, "| a |\n|---|\n| 1 |", None, config)

    argv, _ = mock_run.call_args[0]
    prompt_arg = argv[argv.index("-p") + 1]
    assert "no prior findings" in prompt_arg


# --------------------------------------------------------------------------
# S1 failures: never an exception, never a synthesized verdict
# --------------------------------------------------------------------------


def test_timeout_is_s1_failure_without_sleeping(tmp_path: Path):
    crop = tmp_path / "crop.png"
    config = _config(table_judge_timeout_sec=5.0)

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        side_effect=subprocess.TimeoutExpired(cmd=["gemini"], timeout=5.0),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert result.verdict is None
    assert "timed out" in result.error


def test_missing_binary_is_s1_failure(tmp_path: Path):
    crop = tmp_path / "crop.png"
    config = _config(table_judge_rung2_binary="not-a-real-gemini-binary")

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        side_effect=FileNotFoundError("no such file"),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert "not found" in result.error


def test_nonzero_exit_is_s1_failure(tmp_path: Path):
    crop = tmp_path / "crop.png"
    config = _config()

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        return_value=_completed("", returncode=1, stderr="quota exceeded"),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert "quota exceeded" in result.error


def test_transport_oserror_is_s1_failure(tmp_path: Path):
    crop = tmp_path / "crop.png"
    config = _config()

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        side_effect=OSError("broken pipe"),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert "transport error" in result.error


def test_garbage_stdout_is_s1_failure_not_exception(tmp_path: Path):
    crop = tmp_path / "crop.png"
    config = _config()

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        return_value=_completed("I cannot see the image."),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert result.verdict is None


def test_fenced_json_stdout_is_not_s1_failure(tmp_path: Path):
    """gemini CLI fences routinely — A1 strips fences, this rung must not re-reject."""
    crop = tmp_path / "crop.png"
    config = _config()
    fenced = f"```json\n{PASS_STDOUT}\n```"

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        return_value=_completed(fenced),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is True
    assert result.verdict.passed is True


# --------------------------------------------------------------------------
# RungCallable factory
# --------------------------------------------------------------------------


def test_make_gemini_rung_matches_rung_callable_shape(tmp_path: Path):
    crop = tmp_path / "crop.png"
    config = _config()
    rung = make_gemini_rung(config)

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        return_value=_completed(PASS_STDOUT),
    ):
        result = rung(crop, "md", None)

    assert isinstance(result, RungResult)
    assert result.ok is True


def test_make_gemini_rung_binds_config_not_closure_over_call_site(tmp_path: Path):
    crop = tmp_path / "crop.png"
    fast_config = _config(table_judge_rung2_binary="fast-gemini")
    rung = make_gemini_rung(fast_config)

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        return_value=_completed(PASS_STDOUT),
    ) as mock_run:
        rung(crop, "md", None)

    argv, _ = mock_run.call_args[0]
    assert argv[0] == "fast-gemini"
