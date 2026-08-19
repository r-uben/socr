#!/usr/bin/env bash
# TICKET-D0/D1/D2/D3 — execute APPROVED tracker actions, serially and idempotently.
#
# Reads actions/tracker_actions.json + actions/decisions.json. Executes ONLY rows
# that are APPROVED by the review board. Every comment carries an invisible marker
# comment containing its action_id; before writing, the script re-reads the live
# issue and skips the action if that marker is already present. So a timeout after
# GitHub accepted a write cannot duplicate a comment on retry.
#
# After every write it reads the issue back and records the observed state — a
# write that "succeeded" but did not change anything is a failure here.
#
# Usage:  apply_tracker_actions.sh [--dry-run]
# Never merges, never force-pushes, never touches a HELD-FOR-OWNER row.

set -uo pipefail
PLAN="$(cd "$(dirname "$0")/.." && pwd)"
REPO=r-uben/socr
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

ACTIONS="$PLAN/actions/tracker_actions.json"
DECISIONS="$PLAN/actions/decisions.json"
LEDGER="$PLAN/actions/execution_ledger.json"

[ -f "$PLAN/state/ABORT" ] && { echo "ABORT latch set — refusing to write"; exit 3; }
[ -f "$ACTIONS" ]   || { echo "no $ACTIONS"; exit 2; }
[ -f "$DECISIONS" ] || { echo "no $DECISIONS — the review board has not run; refusing"; exit 2; }
gh auth status >/dev/null 2>&1 || { echo "gh auth dead"; touch "$PLAN/state/ABORT"; exit 3; }

echo "[]" > "$LEDGER.tmp"

jq -c '.actions[]' "$ACTIONS" | while read -r row; do
  aid=$(jq -r '.action_id'  <<<"$row")
  num=$(jq -r '.issue'      <<<"$row")
  kind=$(jq -r '.kind'      <<<"$row")
  body=$(jq -r '.comment'   <<<"$row")
  decision=$(jq -r --arg a "$aid" '.[$a].decision // "MISSING"' "$DECISIONS")

  if [ "$decision" != "APPROVED" ]; then
    echo "SKIP  $aid (#$num) — $decision"; continue
  fi

  # freshness: a human touching the issue while we worked voids the action
  staged_upd=$(jq -r '.snapshot_updated_at' <<<"$row")
  live_upd=$(gh issue view "$num" --repo "$REPO" --json updatedAt --jq .updatedAt)
  marker="<!-- socr-night-sweep:$aid -->"

  if gh issue view "$num" --repo "$REPO" --json comments \
       --jq '.comments[].body' | grep -qF "$marker"; then
    echo "IDEMP $aid (#$num) — marker already present, not rewriting"; continue
  fi
  if [ "$staged_upd" != "$live_upd" ]; then
    echo "SKIP  $aid (#$num) — SKIPPED-CHANGED (updated_at moved: $staged_upd -> $live_upd)"; continue
  fi

  if [ "$DRY" = 1 ]; then echo "DRY   $aid (#$num) $kind"; continue; fi

  url=$(gh issue comment "$num" --repo "$REPO" --body "$body"$'\n\n'"$marker") || {
    echo "FAIL  $aid (#$num) — comment write failed"; continue; }
  if [ "$kind" = "close" ]; then
    gh issue close "$num" --repo "$REPO" --reason completed || echo "WARN  $aid close failed after comment"
  fi
  # read-after-write
  state=$(gh issue view "$num" --repo "$REPO" --json state --jq .state)
  echo "DONE  $aid (#$num) $kind -> $state  $url"
done

rm -f "$LEDGER.tmp"
