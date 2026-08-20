#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""TICKET-D0 — turn adjudicated dispositions into a staged tracker-action manifest.

Nothing here writes to GitHub. It produces actions/tracker_actions.json, which the
review board then tries to refute and bin/apply_tracker_actions.sh later executes —
but only for rows the board marked APPROVED.

Three guards, all mechanical:
  * a `close` action is emitted ONLY if the disposition is ALREADY-FIXED and the
    machine check confirmed all three proofs. A close is irreversible-ish and
    public; "the adjudicator said so" is not sufficient.
  * every action is stamped with the issue's snapshot updated_at/body_hash. If a
    human touches the issue while we work, the apply script skips it.
  * action_id is deterministic (issue + kind + main_sha), so re-running D0 cannot
    produce duplicate work, and the marker comment makes apply idempotent.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PLAN = Path(__file__).resolve().parent.parent
BASE = json.loads((PLAN / "baseline.json").read_text())
SHA = BASE["main_sha"]
SNAP = {i["number"]: i for i in json.loads((PLAN / "issues_snapshot.json").read_text())["issues"]}

KIND_BY_ACTION = {"close": "close", "comment-correction": "comment", "file-new": "new"}


def verified_already_fixed(issue: int) -> tuple[bool, str]:
    """Did ANY triage seat carry an ALREADY-FIXED that survived the machine check?"""
    for f in (PLAN / "triage" / "verified").glob("batch-*.json"):
        for r in json.loads(f.read_text())["rows"]:
            if r.get("issue") == issue and r.get("verdict") == "ALREADY-FIXED":
                if r.get("evidence_verified"):
                    return True, r.get("fixed_by_commit") or ""
                return False, f"machine check rejected it: {r.get('reason')}"
    return False, "no ALREADY-FIXED verdict from any seat"


def main() -> int:
    actions, refused = [], []
    for vf in sorted((PLAN / "verdicts").glob("batch-*.json")):
        doc = json.loads(vf.read_text())
        for r in doc["rows"]:
            issue = r["issue"]
            act = (r.get("action_recommended") or "none").strip()
            if act in ("none", "fix"):
                continue  # 'fix' is stream E's business, not the tracker's
            if issue not in SNAP:
                refused.append({"issue": issue, "why": "not in the pinned snapshot"})
                continue

            kind = KIND_BY_ACTION.get(act)
            if kind is None:
                refused.append({"issue": issue, "why": f"unknown action {act!r}"})
                continue

            if kind == "close":
                ok, why = verified_already_fixed(issue)
                if not ok or r.get("final_disposition") != "ALREADY-FIXED":
                    refused.append(
                        {
                            "issue": issue,
                            "why": f"close refused at D0 — {why}; "
                            f"disposition={r.get('final_disposition')}",
                        }
                    )
                    continue

            comment = r.get("proposed_comment")
            if not comment:
                refused.append({"issue": issue, "why": f"{act} with no proposed_comment"})
                continue

            aid = hashlib.sha256(f"{issue}|{kind}|{SHA}".encode()).hexdigest()[:12]
            actions.append(
                {
                    "action_id": aid,
                    "issue": issue,
                    "kind": kind,
                    "disposition": r["final_disposition"],
                    "batch": doc["batch"],
                    "adjudicator": doc.get("adjudicator"),
                    "agreement": r.get("agreement"),
                    "snapshot_updated_at": SNAP[issue]["updated_at"],
                    "snapshot_body_hash": SNAP[issue]["body_hash"],
                    "evidence": r.get("citations", []),
                    "basis": r.get("basis"),
                    "risk_if_wrong": r.get("risk_if_wrong"),
                    "comment": comment,
                    "url": SNAP[issue]["url"],
                }
            )

    (PLAN / "actions").mkdir(exist_ok=True)
    (PLAN / "actions" / "tracker_actions.json").write_text(
        json.dumps(
            {"main_sha": SHA, "count": len(actions), "actions": actions, "refused": refused},
            indent=1,
        )
    )
    print(f"staged {len(actions)} action(s); refused {len(refused)}")
    for a in actions:
        print(f"  {a['action_id']}  #{a['issue']:<4} {a['kind']:<7} {a['disposition']}")
    for x in refused:
        print(f"  REFUSED #{x['issue']}: {x['why']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
