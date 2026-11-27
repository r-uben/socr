"""#1257 release-rule mutant check: every guard of the release rule is removed in a COPY, its named test must FAIL.

CLAUDE.md guard rule.  Loaded: ``src``, ``tests`` and ``pyproject.toml`` are copied outside the
repo and ``socr.__file__`` is asserted to lie inside the copy.  Applied: each anchor is counted
UNCAPPED in the copied file and must occur exactly once, or the run aborts.  Load-bearing: the
named tests must fail on the mutant (and pass on the unmutated copy, run first).  Right suite: the
tests named are the ones that own the mutated function.

Run from the repo root::

    ~/venvs/socr/bin/python docs/log/2026-11-14_1257-release-mutants.py

Nothing is read from the command line.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = str(Path.home() / "venvs/socr/bin/python")
RULE = "tests/test_gh1257_release_rule.py"
ORCH = "src/socr/pipeline/orchestrator.py"
VERDICT = "src/socr/core/verdict.py"
MANIFEST = "src/socr/core/manifest.py"

# (name, file, [(anchor, replacement)], [test selectors])
MUTANTS: list[tuple[str, str, list[tuple[str, str]], list[str]]] = [
    (
        "release without independent evidence (the page's mark always goes)",
        ORCH,
        [
            (
                "            if _held and not self._row_binding_release(state, page_num, ps, out):\n",
                "            if False:\n",
            )
        ],
        [f"{RULE}::test_the_mark_goes_only_with_independent_page_evidence"],
    ),
    (
        "a native page released on the words and numbers checks alone (circular)",
        VERDICT,
        [
            (
                "    if model_read:\n        return independent_page_evidence(entries, text)\n",
                "    if True:\n        return independent_page_evidence(entries, text)\n",
            )
        ],
        [f"{RULE}::test_a_native_page_is_released_only_on_ink_coverage"],
    ),
    (
        "a native page released with a FAIL or UNKNOWN entry on it",
        VERDICT,
        [
            (
                "    checked = verify_pass_digests(entries, text)\n"
                "    if any(e.result in (TraceResult.FAIL, TraceResult.UNKNOWN) for e in checked):\n"
                "        return False\n"
                "    return any(\n",
                "    checked = verify_pass_digests(entries, text)\n    return any(\n",
            )
        ],
        [f"{RULE}::test_a_native_page_is_released_only_on_ink_coverage"],
    ),
    (
        "a native page released on a native_coverage PASS at any scope",
        VERDICT,
        [
            (
                "        e.check == Check.NATIVE_COVERAGE.value\n        and e.scope == SCOPE_PAGE\n",
                "        e.check == Check.NATIVE_COVERAGE.value\n",
            )
        ],
        [f"{RULE}::test_a_native_page_is_released_only_on_ink_coverage"],
    ),
    (
        "a native page released on a native_coverage PASS not computed on the shipped text",
        VERDICT,
        [
            (
                "    checked = verify_pass_digests(entries, text)\n"
                "    if any(e.result in (TraceResult.FAIL, TraceResult.UNKNOWN) for e in checked):\n"
                "        return False\n"
                "    return any(\n",
                "    checked = list(entries)\n"
                "    if any(e.result in (TraceResult.FAIL, TraceResult.UNKNOWN) for e in checked):\n"
                "        return False\n"
                "    return any(\n",
            )
        ],
        [f"{RULE}::test_a_native_page_is_released_only_on_ink_coverage"],
    ),
    (
        "a held table marked with the unsettled wording instead of the true one",
        MANIFEST,
        [("                if _held\n", "                if False\n")],
        [f"{RULE}::test_the_mark_goes_only_with_independent_page_evidence"],
    ),
]


def run_pytest(copy: Path, selectors: list[str]) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(copy / "src"), "HOME": str(Path.home()), "PATH": "/usr/bin:/bin"}
    return subprocess.run(
        [PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *selectors],
        cwd=copy,
        env=env,
        capture_output=True,
        text=True,
    )


def fresh_copy(root: Path) -> Path:
    copy = Path(tempfile.mkdtemp(dir=root))
    for name in ("src", "tests"):
        shutil.copytree(REPO / name, copy / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO / "pyproject.toml", copy / "pyproject.toml")
    probe = subprocess.run(
        [PY, "-c", "import socr; print(socr.__file__)"],
        cwd=copy,
        env={"PYTHONPATH": str(copy / "src"), "HOME": str(Path.home()), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    loaded = probe.stdout.strip()
    assert loaded.startswith(str(copy)), f"socr loads from {loaded}, not from the copy {copy}"
    return copy


def main() -> int:
    outside = Path(tempfile.mkdtemp(prefix="socr-1257-mutants-"))
    assert not str(outside).startswith(str(REPO))
    failures: list[str] = []
    try:
        base = fresh_copy(outside)
        every = sorted({sel for _n, _f, _e, sels in MUTANTS for sel in sels})
        clean = run_pytest(base, every)
        print(f"unmutated copy: {'PASS' if clean.returncode == 0 else 'FAIL'}")
        if clean.returncode != 0:
            print(clean.stdout[-2000:])
            return 1
        for name, src_file, edits, selectors in MUTANTS:
            copy = fresh_copy(outside)
            target = copy / src_file
            text = target.read_text(encoding="utf-8")
            for anchor, replacement in edits:
                count = text.count(anchor)  # uncapped
                assert count == 1, f"{name}: anchor occurs {count} times: {anchor[:70]!r}"
                text = text.replace(anchor, replacement)
            target.write_text(text, encoding="utf-8")
            result = run_pytest(copy, selectors)
            killed = result.returncode != 0
            tail = [ln for ln in result.stdout.splitlines() if ln.startswith("FAILED")][:1]
            print(f"{'KILLED  ' if killed else 'SURVIVED'} {name}  {tail[0][:110] if tail else ''}")
            if not killed:
                failures.append(name)
            shutil.rmtree(copy, ignore_errors=True)
    finally:
        shutil.rmtree(outside, ignore_errors=True)
    print(f"{len(MUTANTS) - len(failures)}/{len(MUTANTS)} mutants killed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
