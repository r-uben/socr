#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""TICKET-V1 — mechanically verify every triage citation. NO MODEL RUNS HERE.

This is the one place in the sweep where fabrication would otherwise survive: an
LLM adjudicator asked to check another LLM's `file:line` evidence will mark a
plausible-looking citation verified without ever opening the file. So
`evidence_verified` is computed here, by git, and no model may write it.

Per verdict:
  * every citation must resolve at main_sha: the blob exists, the line exists,
    and `snippet` is genuinely a substring of THAT line;
  * an ALREADY-FIXED verdict additionally needs all three proofs from CONTRACT.md
    - fixed_by_commit exists and is an ancestor of main_sha,
    - `git log -L<line>,<line>:<path>` shows that commit actually touched the
      cited region (an ancestor commit that never went near the code proves
      nothing),
    - acceptance_criteria enumerated, and a reproducer with real output.

Exit status:
  0  all ALREADY-FIXED verdicts carry all three proofs
  1  at least one ALREADY-FIXED passed without them  (the dangerous case)
  2  usage / input error
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PLAN = Path(__file__).resolve().parent.parent
REPO = "/Users/rubenffuertes/repos/.worktrees/socr-night-base"
MAIN_SHA = json.loads((PLAN / "baseline.json").read_text())["main_sha"]

# A verdict with no citations is only defensible when it is not a claim about code.
CITATION_EXEMPT = {"ROADMAP-UMBRELLA", "BLOCKED-PROPOSAL", "CHORE-PLAN", "DEFERRED"}


def git(*args: str) -> tuple[int, str]:
    p = subprocess.run(
        ["git", "-C", REPO, *args], capture_output=True, text=True, timeout=60
    )
    return p.returncode, p.stdout


def check_citation(c: dict) -> dict:
    """Resolve one {path,line,snippet} at MAIN_SHA. Never trusts the model's text."""
    out = {"citation": c, "ok": False, "reason": None}
    path, line, snippet = c.get("path"), c.get("line"), c.get("snippet")
    if not path or not isinstance(line, int) or not snippet:
        out["reason"] = "malformed citation (needs path, integer line, snippet)"
        return out
    rc, blob = git("show", f"{MAIN_SHA}:{path}")
    if rc != 0:
        out["reason"] = f"path does not exist at main_sha: {path}"
        return out
    lines = blob.splitlines()
    if not (1 <= line <= len(lines)):
        out["reason"] = f"line {line} out of range (file has {len(lines)} lines)"
        return out
    actual = lines[line - 1]
    if snippet.strip() not in actual:
        out["reason"] = (
            f"snippet not on line {line}. claimed={snippet.strip()!r} actual={actual.strip()!r}"
        )
        return out
    out["ok"] = True
    out["actual_line"] = actual.strip()
    return out


def check_already_fixed(v: dict, cites: list[dict]) -> tuple[bool, list[str]]:
    """The three proofs. A citation proves code exists; it never proves a fix."""
    fails: list[str] = []
    sha = v.get("fixed_by_commit")

    if not sha:
        fails.append("no fixed_by_commit")
    else:
        rc, _ = git("cat-file", "-e", f"{sha}^{{commit}}")
        if rc != 0:
            fails.append(f"fixed_by_commit {sha} is not a commit in this repo")
        else:
            rc, _ = git("merge-base", "--is-ancestor", sha, MAIN_SHA)
            if rc != 0:
                fails.append(f"fixed_by_commit {sha} is NOT an ancestor of main_sha")
            else:
                # ancestry alone is cheap to satisfy: prove it touched cited code
                touched = False
                for c in cites:
                    rc, log = git(
                        "log",
                        f"-L{c['line']},{c['line']}:{c['path']}",
                        "--format=%H",
                        MAIN_SHA,
                    )
                    if rc == 0 and sha[:8] in log:
                        touched = True
                        break
                if not touched:
                    fails.append(
                        f"{sha[:8]} is an ancestor but never touched any cited line range"
                    )

    ac = v.get("acceptance_criteria") or []
    if not ac:
        fails.append("acceptance_criteria not enumerated")
    elif not all(isinstance(a, dict) and a.get("status") for a in ac):
        fails.append("acceptance_criteria present but not each marked met/unmet")
    elif any(str(a.get("status", "")).lower().startswith("unmet") for a in ac):
        fails.append("an acceptance criterion is marked unmet — cannot be ALREADY-FIXED")

    r = v.get("reproducer")
    if not r:
        fails.append("no reproducer")
    elif isinstance(r, dict) and not (r.get("output") or r.get("result")):
        fails.append("reproducer recorded without its actual output")

    return (not fails), fails


def main() -> int:
    batch_files: dict[str, list[Path]] = {}
    for p in sorted((PLAN / "triage").glob("batch-*/*.json")):
        batch_files.setdefault(p.parent.name, []).append(p)
    if not batch_files:
        print("V1: no triage files found", file=sys.stderr)
        return 2

    (PLAN / "triage" / "verified").mkdir(parents=True, exist_ok=True)
    bad_already_fixed = 0
    grand = {"checked": 0, "verified": 0, "void": 0}

    for batch, files in sorted(batch_files.items()):
        rows = []
        for f in files:
            vendor = f.stem
            try:
                doc = json.loads(f.read_text())
            except json.JSONDecodeError as e:
                rows.append(
                    {
                        "vendor": vendor,
                        "issue": None,
                        "verdict": "MISSING",
                        "evidence_verified": False,
                        "reason": f"output file is not valid JSON: {e}",
                    }
                )
                continue

            for v in doc.get("verdicts", []):
                grand["checked"] += 1
                verdict = v.get("verdict")
                cites = v.get("citations") or []
                results = [check_citation(c) for c in cites]
                bad = [r for r in results if not r["ok"]]

                ok, reasons = True, []
                if bad:
                    ok = False
                    reasons += [r["reason"] for r in bad]
                if not cites and verdict not in CITATION_EXEMPT:
                    ok = False
                    reasons.append(f"{verdict} with zero citations")

                if verdict == "ALREADY-FIXED":
                    af_ok, af_fails = check_already_fixed(v, [c for c in cites])
                    if not af_ok:
                        ok = False
                        reasons += af_fails
                        bad_already_fixed += 1

                grand["verified" if ok else "void"] += 1
                rows.append(
                    {
                        "vendor": vendor,
                        "issue": v.get("issue"),
                        "verdict": verdict,
                        # forced by CONTRACT: unverifiable evidence => NEEDS-MEASUREMENT
                        "effective_verdict": verdict if ok else "NEEDS-MEASUREMENT",
                        "evidence_verified": ok,
                        "reason": None if ok else "; ".join(reasons),
                        "citations_checked": len(cites),
                        "citations_failed": len(bad),
                        "citation_detail": results,
                        "confidence": v.get("confidence"),
                        "rationale": v.get("rationale"),
                        "duplicate_of": v.get("duplicate_of"),
                        "fixed_by_commit": v.get("fixed_by_commit"),
                        "cluster_consistency": v.get("cluster_consistency"),
                    }
                )

        out = PLAN / "triage" / "verified" / f"{batch}.json"
        out.write_text(
            json.dumps(
                {
                    "batch": batch,
                    "main_sha": MAIN_SHA,
                    "vendors": [f.stem for f in files],
                    "rows": rows,
                },
                indent=1,
            )
        )
        v_ok = sum(1 for r in rows if r.get("evidence_verified"))
        print(f"{batch}: {len(rows):3d} verdicts, {v_ok:3d} evidence_verified -> {out.name}")

    print(
        f"\nTOTAL checked={grand['checked']} verified={grand['verified']} void={grand['void']}"
    )
    if bad_already_fixed:
        print(
            f"FAIL: {bad_already_fixed} ALREADY-FIXED verdict(s) lacked the three proofs",
            file=sys.stderr,
        )
        return 1
    print("OK: every ALREADY-FIXED verdict carried all three proofs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
