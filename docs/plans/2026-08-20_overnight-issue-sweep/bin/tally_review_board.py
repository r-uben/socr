#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""TICKET-DR — tally the review board. Both reviewers approve, or the action is held.

No weighing, no averaging, no "the rejection was weak". Two APPROVE votes from two
different vendors, neither of which triaged or adjudicated that issue, or the
action goes to the owner's morning decision queue with both readings recorded.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

PLAN = Path(__file__).resolve().parent.parent
ACTIONS = json.loads((PLAN / "actions" / "tracker_actions.json").read_text())["actions"]

# who may not review what — enforced here, not trusted to the prompt
TRIAGERS = {
    "batch-1": {"grok", "claude", "ollama-minimax"},
    "batch-2": {"ollama-deepseek", "gemini-flash"},
    "batch-3": {"kimi", "ollama-minimax"},
    "batch-4": {"grok", "gemini-pro"},
}
ADJUDICATOR = {"batch-1": "kimi", "batch-2": "grok", "batch-3": "gemini-pro", "batch-4": "deepseek"}
FAMILY = {"gemini-pro": "google", "gemini-flash": "google"}


def family(v: str) -> str:
    return FAMILY.get(v, v)


def main() -> int:
    votes = defaultdict(list)
    for f in sorted((PLAN / "actions" / "review").glob("*.json")):
        d = json.loads(f.read_text())
        votes[d["action_id"]].append(d)

    decisions, held, approved = {}, [], []
    for a in ACTIONS:
        aid, batch = a["action_id"], a["batch"]
        vs = votes.get(aid, [])
        reasons = []

        # independence, checked mechanically
        excluded = {family(x) for x in TRIAGERS.get(batch, set())} | {
            family(ADJUDICATOR.get(batch, ""))
        }
        conflicted = [v["reviewer"] for v in vs if family(v["reviewer"]) in excluded]
        if conflicted:
            reasons.append(f"reviewer(s) {conflicted} triaged or adjudicated this batch")

        distinct = {family(v["reviewer"]) for v in vs}
        if len(vs) < 2:
            reasons.append(f"only {len(vs)} reviewer(s) reported")
        elif len(distinct) < 2:
            reasons.append(f"both reviewers are the same vendor family: {distinct}")

        rejects = [v for v in vs if v.get("vote") != "APPROVE"]
        if rejects:
            reasons.append(
                "rejected by " + ", ".join(f"{v['reviewer']} ({v.get('reason')})" for v in rejects)
            )

        # a reviewer with no access proof cannot approve
        noproof = [
            v["reviewer"]
            for v in vs
            if v.get("vote") == "APPROVE" and not (v.get("access_proof") or {}).get("output")
        ]
        if noproof:
            reasons.append(f"approving reviewer(s) {noproof} produced no access proof")

        decision = "APPROVED" if not reasons else "HELD-FOR-OWNER"
        (approved if decision == "APPROVED" else held).append(a)
        decisions[aid] = {
            "action_id": aid,
            "issue": a["issue"],
            "kind": a["kind"],
            "decision": decision,
            "why_held": reasons or None,
            "readings": {
                v["reviewer"]: {
                    "vote": v.get("vote"),
                    "reason": v.get("reason"),
                    "refutation_attempted": v.get("refutation_attempted"),
                    "unsatisfied_criteria": v.get("unsatisfied_criteria"),
                }
                for v in vs
            },
        }

    (PLAN / "actions" / "decisions.json").write_text(json.dumps(decisions, indent=1))
    print(f"APPROVED {len(approved)}   HELD-FOR-OWNER {len(held)}\n")
    for aid, d in decisions.items():
        mark = "OK  " if d["decision"] == "APPROVED" else "HELD"
        vs = "  ".join(f"{r}={v['vote']}" for r, v in d["readings"].items())
        print(f"{mark} {aid} #{d['issue']:<4} {d['kind']:<7} {vs}")
        if d["why_held"]:
            for r in d["why_held"]:
                print(f"       - {r}")
    return 0


raise SystemExit(main())
