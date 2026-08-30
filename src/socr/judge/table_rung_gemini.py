"""GH-353 TICKET-A3: CLI2 rung — gemini CLI table judge invoker.

Rung 2 of the ladder: a per-crop subprocess call to the `gemini` CLI (the
Google One AI Pro headless surface), the off-family, non-qwen, non-ollama
judge (design: `docs/log/2026-08-30_table-judge-ladder.md`). `GeminiEngine`
(`socr.engines.gemini`) wraps a *document-level* OCR CLI (`gemini-ocr`) and
is not reusable here — this module owns its own per-crop subprocess
invocation of the actual `gemini` agent CLI.

Image handoff: the crop is a temp file owned by the B0 witness module (this
module must never delete it — cleanup is the caller's context manager).
Headless gemini CLI (`-p/--prompt`) has no dedicated "attach image" flag;
its documented mechanism for pulling a local file into a headless turn is an
`@<path>` reference inside the prompt text, which the CLI resolves and loads
(binary/image content included) before sending the turn to the model.
`--include-directories` grants read access to the crop's parent directory
(a scratch/temp dir, not necessarily under the CLI's default trusted
workspace); `--skip-trust` avoids an interactive workspace-trust prompt that
would otherwise hang the subprocess past its timeout; `--approval-mode plan`
keeps the call read-only (the judge never needs write/execute tools).

S1 classification ("the judge answered") is TICKET-A1's job
(`rung_result_from_output` / `parse_table_verdict`). This module's only
responsibility is: build argv, run it under a bounded timeout, and turn
every non-answer outcome (timeout, missing binary, non-zero exit, transport
OSError) into `RungResult(ok=False, ...)` — never an exception, never a
synthesized verdict.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from socr.core.config import PipelineConfig
from socr.judge.table_prompt import build_table_judge_prompt
from socr.judge.table_verdict import Finding, RungResult, rung_result_from_output

#: Rung identifier recorded on every `RungResult` this module produces.
RUNG_ID = "gemini"

#: Bytes of stderr kept in the error message on a non-zero exit — enough for
#: a human to see the cause in the audit trail without unbounded growth.
_STDERR_ERROR_CHARS = 500


def _run_gemini_cli(argv: list[str], timeout_sec: float) -> subprocess.CompletedProcess[str]:
    """Module-local subprocess seam.

    Tests patch THIS function (not `PATH`, not `shutil.which`) so the argv
    and timeout handling are exercised without a real `gemini` binary.
    """
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )


def _findings_as_mappings(findings: list[Finding] | None) -> list[dict[str, str]] | None:
    """Adapt A1's `Finding` dataclasses to A0's duck-typed mapping slot."""
    if not findings:
        return None
    return [{"code": f.code.value, "where": f.where, "detail": f.detail} for f in findings]


def build_gemini_argv(binary: str, crop_path: Path, prompt: str) -> list[str]:
    """Build the gemini CLI argv for one headless table-judge call.

    The crop is attached via an `@<path>` reference inside the prompt text
    (gemini CLI's file-inclusion syntax) so the model sees the actual image,
    not just its path as a string.
    """
    full_prompt = f"Image crop: @{crop_path}\n\n{prompt}"
    return [
        binary,
        "--skip-trust",
        "--approval-mode",
        "plan",
        "--include-directories",
        str(crop_path.parent),
        "-p",
        full_prompt,
    ]


def judge_table_gemini(
    crop_path: Path,
    markdown: str,
    prior_findings: list[Finding] | None,
    config: PipelineConfig,
) -> RungResult:
    """CLI2 rung: judge one table crop against its emitted markdown.

    Matches the `RungCallable` shape from `socr.judge.table_verdict` plus an
    explicit `config` — `make_gemini_rung(config)` below closes over
    `config` to produce the exact `RungCallable` the ladder (A4) / gate (B1)
    inject. Every failure mode (missing binary, timeout, non-zero exit,
    unparseable output) returns `RungResult(ok=False, ...)`; this function
    never raises for judge-side failures.
    """
    binary = config.table_judge_rung2_binary
    timeout_sec = config.table_judge_timeout_sec
    prompt = build_table_judge_prompt(
        markdown,
        prior_findings=_findings_as_mappings(prior_findings),
    )
    argv = build_gemini_argv(binary, crop_path, prompt)

    start = time.monotonic()
    try:
        completed = _run_gemini_cli(argv, timeout_sec)
    except FileNotFoundError as exc:
        return RungResult(
            rung=RUNG_ID,
            ok=False,
            latency_sec=time.monotonic() - start,
            error=f"gemini binary not found: {exc}",
        )
    except subprocess.TimeoutExpired:
        return RungResult(
            rung=RUNG_ID,
            ok=False,
            latency_sec=time.monotonic() - start,
            error=f"gemini CLI timed out after {timeout_sec}s",
        )
    except OSError as exc:
        return RungResult(
            rung=RUNG_ID,
            ok=False,
            latency_sec=time.monotonic() - start,
            error=f"gemini CLI transport error: {exc}",
        )
    latency_sec = time.monotonic() - start

    if completed.returncode != 0:
        stderr_tail = (completed.stderr or "")[:_STDERR_ERROR_CHARS]
        return RungResult(
            rung=RUNG_ID,
            ok=False,
            latency_sec=latency_sec,
            error=f"gemini CLI exited {completed.returncode}: {stderr_tail}",
        )

    return rung_result_from_output(RUNG_ID, completed.stdout, latency_sec)


def make_gemini_rung(config: PipelineConfig):
    """Bind `config` to produce the `RungCallable` the ladder/gate inject.

    Returned closure matches `socr.judge.table_verdict.RungCallable`
    exactly: `(crop_path, markdown, prior_findings) -> RungResult`.
    """

    def _rung(
        crop_path: Path,
        markdown: str,
        prior_findings: list[Finding] | None,
    ) -> RungResult:
        return judge_table_gemini(crop_path, markdown, prior_findings, config)

    return _rung
