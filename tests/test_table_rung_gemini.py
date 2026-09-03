"""GH-353 TICKET-A3: CLI2 rung — Gemini-family table judge invoker (via `agy`).

All tests patch the module-local `_run_gemini_cli` subprocess seam — never
`PATH`/`shutil.which` — so nothing here shells out to a real `agy`/`gemini`
binary. A guard fixture asserts `subprocess.run` is never reached
unpatched.
"""

from __future__ import annotations

import errno
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.config import PipelineConfig
from socr.judge.table_rung_gemini import (
    RUNG_ID,
    build_gemini_argv,
    judge_table_gemini,
    make_gemini_rung,
)
from socr.judge.table_verdict import Finding, FindingCode, RungResult

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

    argv = build_gemini_argv("agy", crop, prompt)

    assert argv == [
        "agy",
        "-p",
        f"Image crop: @{crop}\n\n{prompt}",
        "--add-dir",
        str(crop.parent),
    ]
    assert "--approval-mode" not in argv
    assert "--skip-trust" not in argv
    assert "--include-directories" not in argv


def test_build_gemini_argv_uses_configured_binary(tmp_path: Path):
    crop = tmp_path / "crop.png"
    argv = build_gemini_argv("/opt/custom/agy", crop, "x")
    assert argv[0] == "/opt/custom/agy"


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
    argv, timeout_sec, cwd = mock_run.call_args[0]
    assert argv[0] == config.table_judge_rung2_binary
    assert timeout_sec == config.table_judge_timeout_sec
    assert cwd == crop.parent
    assert isinstance(result, RungResult)
    assert result.rung == RUNG_ID
    assert result.ok is True
    assert result.verdict is not None
    assert result.verdict.passed is True
    assert result.unavailable is False


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
    assert result.unavailable is False


def test_prior_findings_do_not_reach_the_prompt(tmp_path: Path):
    """GH-359 ruling 4: even a caller that still passes findings must not
    leak them into the judge prompt."""
    crop = tmp_path / "crop.png"
    config = _config()
    prior = [Finding(code=FindingCode.WRONG_BINDING, where="row 1", detail="shifted-payload")]

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        return_value=_completed(PASS_STDOUT),
    ) as mock_run:
        judge_table_gemini(crop, "| a |\n|---|\n| 1 |", prior, config)

    argv, _, _ = mock_run.call_args[0]
    prompt_arg = argv[argv.index("-p") + 1]
    assert "shifted-payload" not in prompt_arg
    assert "independently" in prompt_arg.lower()


def test_first_look_prompt_is_independent(tmp_path: Path):
    crop = tmp_path / "crop.png"
    config = _config()

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        return_value=_completed(PASS_STDOUT),
    ) as mock_run:
        judge_table_gemini(crop, "| a |\n|---|\n| 1 |", None, config)

    argv, _, _ = mock_run.call_args[0]
    prompt_arg = argv[argv.index("-p") + 1]
    assert "independently" in prompt_arg.lower()
    assert "not given" in prompt_arg.lower()


def test_subprocess_cwd_is_the_crop_scratch_dir_not_the_repo(tmp_path: Path):
    """The CLI must never see this repo's checkout as its workspace root —
    otherwise it loads GEMINI.md/.gemini/ context and MCP servers, breaking
    the "crop + markdown, nothing else" judge-input isolation."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    crop = scratch / "crop.png"
    crop.write_bytes(b"fake-png")
    config = _config()

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        return_value=_completed(PASS_STDOUT),
    ) as mock_run:
        judge_table_gemini(crop, "md", None, config)

    call = mock_run.call_args
    cwd = call.kwargs.get("cwd", call.args[2] if len(call.args) > 2 else None)
    assert cwd == scratch
    assert cwd != Path.cwd()


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
    assert result.unavailable is True


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
    assert result.unavailable is True


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
    assert result.unavailable is True


def test_transport_oserror_is_s1_failure(tmp_path: Path):
    crop = tmp_path / "crop.png"
    config = _config()

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        side_effect=OSError(errno.ENOEXEC, "exec format error"),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert "transport error" in result.error
    assert result.unavailable is True


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
    assert result.unavailable is False


# ---------------------------------------------------------------------------
# P1 prep item 1 (docs/log/2026-09-02_gh359-ladder-terminals-design.md,
# "Panel and synthesis"): the rung-unavailable retry latch needs a bit that
# distinguishes "the configured rung could not be reached at all" (transport
# failure, missing binary, timeout, non-zero exit / quota refusal -- latch
# should fire once this reaches the gate) from "the rung answered but the
# answer was unusable" (malformed/garbage output -- content-shaped, must NOT
# latch: a rung that is up and simply returning junk will return junk again
# when retried, so latching on it does not converge and papers over a real
# content problem).
#
# ``RungResult.unavailable`` carries that bit. Cold review round 2 narrowed
# the nonzero-exit cause: an exit code alone is not an outage. It is one when
# the CLI fails its own health handshake (``gemini_rung_reachable`` -- the
# SAME check the resume gate uses) or when stderr names an external refusal
# (quota, auth, host down). A usage/configuration error from a healthy CLI is
# deterministic and does not latch.
# ---------------------------------------------------------------------------


def test_timeout_is_unavailable(tmp_path: Path):
    crop = tmp_path / "crop.png"
    config = _config(table_judge_timeout_sec=5.0)

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        side_effect=subprocess.TimeoutExpired(cmd=["gemini"], timeout=5.0),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert result.unavailable is True


def test_missing_binary_is_unavailable(tmp_path: Path):
    crop = tmp_path / "crop.png"
    config = _config(table_judge_rung2_binary="not-a-real-gemini-binary")

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        side_effect=FileNotFoundError("no such file"),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert result.unavailable is True


def test_nonzero_exit_including_quota_refusal_is_unavailable(tmp_path: Path):
    crop = tmp_path / "crop.png"
    config = _config()

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        return_value=_completed("", returncode=1, stderr="quota exceeded"),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert result.unavailable is True


def test_usage_error_from_a_healthy_cli_is_not_unavailable(tmp_path: Path):
    """Cold review round 2, finding 2. The CLI is installed and passes its
    health handshake, and it exited nonzero on a usage/configuration error
    that names no external refusal. That is deterministic: it ends the table
    UNVERIFIED and must NOT latch, or every resume repeats the same bad call."""
    crop = tmp_path / "crop.png"
    config = _config()

    with (
        patch(
            "socr.judge.table_rung_gemini._run_gemini_cli",
            return_value=_completed("", returncode=2, stderr="unknown flag: --nope"),
        ),
        patch("socr.judge.table_rung_gemini.gemini_rung_reachable", return_value=True),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert result.unavailable is False


def test_usage_error_from_an_unreachable_cli_is_unavailable(tmp_path: Path):
    """The control for the test above, isolating the one variable: identical
    nonzero exit and identical stderr, but the health handshake now fails, so
    the CLI itself is not usable and the outage classification stands."""
    crop = tmp_path / "crop.png"
    config = _config()

    with (
        patch(
            "socr.judge.table_rung_gemini._run_gemini_cli",
            return_value=_completed("", returncode=2, stderr="unknown flag: --nope"),
        ),
        patch("socr.judge.table_rung_gemini.gemini_rung_reachable", return_value=False),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert result.unavailable is True


def test_quota_refusal_from_a_healthy_cli_is_still_unavailable(tmp_path: Path):
    """A healthy binary can still be refused externally. The stderr signature
    is what tells the two apart, so quota keeps latching even though the
    health handshake passes."""
    crop = tmp_path / "crop.png"
    config = _config()

    with (
        patch(
            "socr.judge.table_rung_gemini._run_gemini_cli",
            return_value=_completed("", returncode=1, stderr="Error: quota exceeded for project"),
        ),
        patch("socr.judge.table_rung_gemini.gemini_rung_reachable", return_value=True),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert result.unavailable is True


def test_a_refusal_past_the_audit_excerpt_still_latches(tmp_path: Path):
    """Cold review round 3. The 500-character excerpt kept for the audit trail
    is not the classification input. A quota message printed after that cutoff
    was discarded before it was ever classified, so recovery was never retried."""
    crop = tmp_path / "crop.png"
    config = _config()

    with (
        patch(
            "socr.judge.table_rung_gemini._run_gemini_cli",
            return_value=_completed("", returncode=1, stderr="x" * 3000 + " quota exceeded"),
        ),
        patch("socr.judge.table_rung_gemini.gemini_rung_reachable", return_value=True),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.unavailable is True
    assert result.refusal is True


def test_a_refusal_printed_on_stdout_still_latches(tmp_path: Path):
    """A CLI is free to print its refusal on either stream."""
    crop = tmp_path / "crop.png"
    config = _config()

    with (
        patch(
            "socr.judge.table_rung_gemini._run_gemini_cli",
            return_value=_completed("RESOURCE_EXHAUSTED: quota", returncode=1, stderr=""),
        ),
        patch("socr.judge.table_rung_gemini.gemini_rung_reachable", return_value=True),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.unavailable is True
    assert result.refusal is True


def test_a_local_permission_error_is_not_read_as_a_refusal(tmp_path: Path):
    """Cold review round 3: loose markers used to classify ordinary local
    errors as outages, which is the direction that re-judges forever."""
    crop = tmp_path / "crop.png"
    config = _config()

    with (
        patch(
            "socr.judge.table_rung_gemini._run_gemini_cli",
            return_value=_completed(
                "", returncode=1, stderr="cannot open /home/u/.config: permission denied"
            ),
        ),
        patch("socr.judge.table_rung_gemini.gemini_rung_reachable", return_value=True),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.unavailable is False
    assert result.refusal is False


class TestGeminiRungReachable:
    """Cold review round 2, finding 1: the resume gate's reachability notion.

    The health handshake runs through its own module-local seam
    (``_run_health_check``), so this module's no-real-subprocess guard still
    holds. The seam's real subprocess behaviour is exercised end to end by
    ``tests/test_p1_ladder_retry_latch.py``.
    """

    def test_missing_binary_is_not_reachable(self) -> None:
        from socr.judge.table_rung_gemini import gemini_rung_reachable

        assert gemini_rung_reachable("socr-no-such-binary-anywhere") is False

    def test_installed_but_failing_health_check_is_not_reachable(self) -> None:
        """The whole point of the finding: presence on PATH is not health."""
        from socr.judge.table_rung_gemini import gemini_rung_reachable

        with (
            patch("socr.judge.table_rung_gemini.shutil.which", return_value="/opt/bin/agy"),
            patch(
                "socr.judge.table_rung_gemini._run_health_check",
                return_value=_completed("", returncode=1, stderr="not configured"),
            ),
        ):
            assert gemini_rung_reachable("agy") is False

    def test_installed_and_healthy_is_reachable(self) -> None:
        from socr.judge.table_rung_gemini import gemini_rung_reachable

        with (
            patch("socr.judge.table_rung_gemini.shutil.which", return_value="/opt/bin/agy"),
            patch(
                "socr.judge.table_rung_gemini._run_health_check",
                return_value=_completed("agy 1.2.3", returncode=0),
            ),
        ):
            assert gemini_rung_reachable("agy") is True

    def test_a_health_check_that_cannot_spawn_is_not_reachable(self) -> None:
        from socr.judge.table_rung_gemini import gemini_rung_reachable

        with (
            patch("socr.judge.table_rung_gemini.shutil.which", return_value="/opt/bin/agy"),
            patch(
                "socr.judge.table_rung_gemini._run_health_check",
                side_effect=OSError("exec format error"),
            ),
        ):
            assert gemini_rung_reachable("agy") is False


@pytest.mark.parametrize(
    "err,unavailable",
    [
        # The ENVIRONMENT is wrong: install, rebuild or chmod the binary and the
        # identical call works. Worth remembering and retrying.
        (errno.ENOEXEC, True),
        (errno.EACCES, True),
        (errno.EPERM, True),
        # THIS CALL is wrong. An oversized argv (an enormous prompt) raises
        # E2BIG on every attempt, so latching it re-runs the ladder forever to
        # reproduce the same rejection (cold review round 3).
        (errno.E2BIG, False),
        (errno.EPIPE, False),
        (errno.EINVAL, False),
    ],
    ids=lambda v: errno.errorcode.get(v, str(v)) if isinstance(v, int) else str(v),
)
def test_spawn_oserror_is_classified_by_errno(tmp_path: Path, err: int, unavailable: bool):
    crop = tmp_path / "crop.png"
    config = _config()

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        side_effect=OSError(err, "spawn failed"),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert result.unavailable is unavailable


def test_an_unattributable_spawn_error_does_not_latch(tmp_path: Path):
    """No errno at all: nothing identifies an external cause. The terminal is
    still UNVERIFIED, so no content is trusted; the conservative choice is not
    to re-run the ladder on every resume for a cause we cannot name."""
    crop = tmp_path / "crop.png"
    config = _config()

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        side_effect=OSError("broken pipe"),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert result.unavailable is False


def test_garbage_stdout_is_not_unavailable(tmp_path: Path):
    """The CLI answered with something unusable -- content-shaped, not a rung
    outage. Retrying immediately will hit the same junk; this must not latch
    a retry the same way a transport failure does."""
    crop = tmp_path / "crop.png"
    config = _config()

    with patch(
        "socr.judge.table_rung_gemini._run_gemini_cli",
        return_value=_completed("I cannot see the image."),
    ):
        result = judge_table_gemini(crop, "md", None, config)

    assert result.ok is False
    assert result.unavailable is False


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

    argv, _, cwd = mock_run.call_args[0]
    assert argv[0] == "fast-gemini"
    assert cwd == crop.parent
