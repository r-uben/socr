"""GH-353 TICKET-A3: CLI2 rung — Gemini-family table judge invoker.

Rung 2 of the ladder: a per-crop subprocess call to a Gemini-family headless
CLI, the off-family, non-qwen, non-ollama judge (design:
`docs/log/2026-08-30_table-judge-ladder.md`). `GeminiEngine`
(`socr.engines.gemini`) wraps a *document-level* OCR CLI (`gemini-ocr`) and
is not reusable here — this module owns its own per-crop subprocess
invocation.

**Binary: `agy` (Antigravity CLI), not the bare `gemini` CLI.** The design
originally specified the `gemini` CLI; the pre-merge B1 live smoke
(2026-08-30) found it can no longer authenticate headlessly on this
machine — Google retired the "Gemini Code Assist for individuals" free tier
that backed it (`IneligibleTierError`, migration message points at
Antigravity). `agy` reaches the same model family through a live, working
headless surface (smoke: schema-perfect unfenced JSON, all six decoy
defects caught — see `docs/log/2026-08-30_gh353-ticket-a3.md`). `RUNG_ID`
stays `"gemini"` — it names the judge model *family* for the audit trail,
not the literal binary; `config.table_judge_rung2_binary` (default `"agy"`)
is what actually runs.

Image handoff: the crop is a temp file owned by the B0 witness module (this
module must never delete it — cleanup is the caller's context manager). Both
`gemini` and `agy` headless print modes (`-p`) lack a dedicated "attach
image" flag; the shared mechanism for pulling a local file into a headless
turn is an `@<path>` reference inside the prompt text, which the CLI
resolves and loads (binary/image content included) before sending the turn
to the model — proven for `agy` by the live smoke above. `--add-dir` grants
read access to the crop's parent directory (a scratch/temp dir, not
necessarily under the CLI's default workspace); `agy` has no `--skip-trust`
/ `--include-directories` equivalent to `gemini`'s, so those flags are
dropped for this binary. The subprocess `cwd` is pinned to the crop's parent
directory (a scratch dir) rather than left as this repo's checkout —
`agy`, like `gemini`, reads ambient context from cwd, and running with this
repo's checkout as cwd would leak repo context into the judge call, breaking
the design's "crop + markdown, nothing else" judge-input isolation.

S1 classification ("the judge answered") is TICKET-A1's job
(`rung_result_from_output` / `parse_table_verdict`). This module's only
responsibility is: build argv, run it under a bounded timeout, and turn
every non-answer outcome (timeout, missing binary, non-zero exit, transport
OSError) into `RungResult(ok=False, ...)` — never an exception, never a
synthesized verdict.

**Known gap:** `agy` has no per-call model-selection flag (the machine-known
`agy-set-model` route is a dead symlink); it answers with whichever model
its encrypted local state currently has active. Rung-2 model identity is
therefore UNCONFIRMED per-call — weaker provenance than rung 1's
config-pinned `table_judge_rung1_model`. The fingerprint records the binary
name only (`table_judge_rung2_binary`), which is accurate to what this
module actually controls; a future `agy` model-pinning fix should tighten
this.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path

from socr.core.config import PipelineConfig
from socr.judge.table_prompt import build_table_judge_prompt
from socr.judge.table_verdict import (
    RUNG_KIND_GEMINI,
    Finding,
    RungResult,
    classify_spawn_oserror,
    output_reads_as_refusal,
    rung_result_from_output,
)

#: Rung identifier recorded on every `RungResult` this module produces —
#: names the judge model FAMILY (gemini), not the literal binary (`agy`).
RUNG_ID = RUNG_KIND_GEMINI

logger = logging.getLogger(__name__)

#: Bytes of stderr kept in the error message on a non-zero exit — enough for
#: a human to see the cause in the audit trail without unbounded growth.
_STDERR_ERROR_CHARS = 500


def _run_gemini_cli(
    argv: list[str], timeout_sec: float, cwd: Path
) -> subprocess.CompletedProcess[str]:
    """Module-local subprocess seam.

    Tests patch THIS function (not `PATH`, not `shutil.which`) so the argv,
    cwd, and timeout handling are exercised without a real binary. `cwd` is
    the crop's (scratch) parent directory — see the module docstring for why
    the CLI must not run with this repo's checkout as its workspace root.
    """
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
        cwd=cwd,
    )


#: Cold review round 2, finding 1. The health handshake: the cheapest
#: invocation that proves the CLI itself runs, with no model call, no prompt
#: and no quota spend. Both ``gemini`` and ``agy`` accept it.
_HEALTH_ARGV_TAIL = ("--version",)

#: Bounded separately from the judge timeout: this is a version print, and a
#: resume gate must not block on it.
_HEALTH_TIMEOUT_SEC = 10.0


def gemini_rung_reachable(binary: str, timeout: float = _HEALTH_TIMEOUT_SEC) -> bool:
    """Whether rung 2 could be attempted right now: installed AND it runs.

    Cold review round 2, finding 1. ``shutil.which`` alone was the resume
    gate's whole test, and it answers True for a CLI that is present but
    broken -- unconfigured, half-installed, or shadowed by something that is
    not the CLI at all. The gate then refuses the document skip, the ladder
    runs, the rung fails the same way it failed last time, and the latch is
    re-set: an outage that never ends re-pays the full ladder on every resume.

    So reachability is: on PATH, AND a trivial no-model invocation succeeds.
    This makes no model call and spends no quota, which also bounds what it
    can see -- a CLI whose credentials or quota are exhausted still prints its
    version. That residue is deliberate and is the ONE retry per resume the
    latch is for; what this closes is the CLI that will never work at all.
    """
    if shutil.which(binary) is None:
        return False
    try:
        completed = _run_health_check([binary, *_HEALTH_ARGV_TAIL], timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("table judge rung 2 health check for %r failed: %s", binary, exc)
        return False
    return completed.returncode == 0


def _run_health_check(argv: list[str], timeout_sec: float) -> subprocess.CompletedProcess[str]:
    """Module-local subprocess seam for the health handshake.

    Separate from ``_run_gemini_cli``: that one is pinned to the crop's
    scratch directory so the judge call cannot read ambient repo context,
    while this one takes no input at all and needs no workspace. Tests patch
    THIS function, so the reachability rule is exercised without a real CLI.
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
    """Build the `agy` argv for one headless table-judge call.

    The crop is attached via an `@<path>` reference inside the prompt text
    (proven for `agy` print mode by the 2026-08-30 live smoke) so the model
    sees the actual image, not just its path as a string. `--add-dir` grants
    read access to the crop's (scratch) parent directory — `agy`'s flag
    surface has no `--skip-trust`/`--include-directories` equivalent.
    """
    full_prompt = f"Image crop: @{crop_path}\n\n{prompt}"
    return [
        binary,
        "-p",
        full_prompt,
        "--add-dir",
        str(crop_path.parent),
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
        completed = _run_gemini_cli(argv, timeout_sec, crop_path.parent)
    except FileNotFoundError as exc:
        return RungResult(
            rung=RUNG_ID,
            ok=False,
            latency_sec=time.monotonic() - start,
            error=f"gemini rung binary {binary!r} not found: {exc}",
            unavailable=True,
        )
    except subprocess.TimeoutExpired:
        return RungResult(
            rung=RUNG_ID,
            ok=False,
            latency_sec=time.monotonic() - start,
            error=f"gemini rung ({binary}) timed out after {timeout_sec}s",
            unavailable=True,
        )
    except OSError as exc:
        # Cold review round 3: for the remaining spawn errors the ERRNO decides.
        # ENOEXEC/EACCES/EPERM describe the ENVIRONMENT -- rebuild or chmod the
        # binary and the identical call works, so they are outages, like the
        # missing-binary case above. Every other errno describes THIS call:
        # E2BIG (an argv the kernel will not accept, which an oversized prompt
        # produces) is reproduced exactly by every retry, so it must not latch.
        return RungResult(
            rung=RUNG_ID,
            ok=False,
            latency_sec=time.monotonic() - start,
            error=f"gemini rung ({binary}) transport error: {exc}",
            unavailable=classify_spawn_oserror(exc),
        )
    latency_sec = time.monotonic() - start

    if completed.returncode != 0:
        # Cold review rounds 2 and 3: a nonzero exit is not automatically an
        # outage. It is one when the CLI names an external refusal, or when the
        # CLI itself cannot be reached at all. A usage or configuration error
        # is deterministic: it ends the table UNVERIFIED without latching, so
        # resume does not repeat it forever.
        #
        # Classification reads the FULL captured output, both streams. The
        # 500-char excerpt below is the audit trail's, and using it to classify
        # discarded any refusal printed past that cutoff -- a false negative
        # that settles the table permanently.
        refusal = output_reads_as_refusal(completed.stderr or "", completed.stdout or "")
        unavailable = refusal or not gemini_rung_reachable(binary)
        stderr_tail = (completed.stderr or "")[:_STDERR_ERROR_CHARS]
        return RungResult(
            rung=RUNG_ID,
            ok=False,
            latency_sec=latency_sec,
            error=f"gemini rung ({binary}) exited {completed.returncode}: {stderr_tail}",
            unavailable=unavailable,
            refusal=refusal,
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

    # Cold review round 5: see the matching note in ``table_rung_ollama``. The
    # executing identity is the BINARY, which is what this rung actually
    # controls -- ``agy`` has no per-call model selector, so the model is
    # unconfirmed and must not be claimed here.
    _rung.rung_kind = RUNG_KIND_GEMINI
    _rung.rung_id = RUNG_ID
    _rung.executing = config.table_judge_rung2_binary
    return _rung
