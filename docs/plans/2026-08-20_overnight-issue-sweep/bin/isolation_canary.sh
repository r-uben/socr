#!/usr/bin/env bash
# isolation_canary.sh — prove that `import socr` resolves to THIS worktree's
# src/socr, not the main checkout's, under the caller's ambient environment.
#
# socr is installed editable against /Users/rubenffuertes/repos/tools/socr, so a
# git worktree does NOT isolate the code under test. Every test run in this sweep
# must be preceded by:
#
#     export PYTHONPATH=<worktree>/src
#     bash bin/isolation_canary.sh [worktree]
#
# Exit 0  : socr.__file__ is inside <worktree>/src and <worktree> is not the
#           protected main checkout.
# Exit 1  : anything else. A test result obtained without a passing canary is void.
#
# The canary deliberately does NOT set PYTHONPATH itself — it inspects the
# environment the caller will actually run pytest in.

set -uo pipefail

MAIN_CHECKOUT="/Users/rubenffuertes/repos/tools/socr"
PY="${SOCR_PY:-$HOME/venvs/socr/bin/python}"

fail() { printf 'CANARY FAIL: %s\n' "$1" >&2; exit 1; }

# --read-only: prove the tree, not the interpreter.
# Some vendor subagents run in a sandbox where the socr venv is unreachable, so a
# triager or reviewer that never executes Python cannot pass the full canary. What
# matters for a READ-ONLY agent is different anyway: that it is reading the right
# worktree at the right SHA and not the main checkout. Requiring an interpreter it
# does not need would push honest agents into reporting a failure they cannot fix —
# or, worse, into skipping the check. Anyone who RUNS TESTS still needs full mode.
READONLY=0
if [ "${1:-}" = "--read-only" ]; then READONLY=1; shift; fi

root="${1:-$(git rev-parse --show-toplevel 2>/dev/null)}"
[ -n "$root" ] || fail "no worktree given and cwd is not inside a git worktree"
root="$(cd "$root" 2>/dev/null && pwd -P)" || fail "worktree path does not exist: $1"

main_real="$(cd "$MAIN_CHECKOUT" 2>/dev/null && pwd -P || echo "$MAIN_CHECKOUT")"
[ "$root" != "$main_real" ] || fail "refusing to run in the main checkout ($main_real) — it is owned by another session (CONTRACT fact 2)"

[ -d "$root/src/socr" ] || fail "no src/socr under $root — not a socr worktree"

if [ "$READONLY" = 1 ]; then
  head_sha="$(git -C "$root" rev-parse HEAD 2>/dev/null)" || fail "not a git worktree: $root"
  want="$(grep -o '"main_sha": *"[0-9a-f]*"' "$(dirname "$0")/../baseline.json" 2>/dev/null | grep -o '[0-9a-f]\{40\}')"
  if [ -n "$want" ] && [ "$head_sha" != "$want" ]; then
    fail "worktree HEAD $head_sha != pinned main_sha $want"
  fi
  echo "CANARY OK (read-only mode): worktree = $root"
  echo "           HEAD          = $head_sha"
  echo "           note: interpreter NOT checked; this proves the tree, not import isolation."
  echo "           Anyone running tests must use full mode instead."
  exit 0
fi

[ -x "$PY" ] || fail "interpreter not found: $PY
         if you are a READ-ONLY agent that never runs Python, re-run as:
           bash $0 --read-only $root"

resolved="$("$PY" -c 'import socr,os;print(os.path.realpath(socr.__file__))' 2>&1)" \
  || fail "import socr failed: $resolved"

case "$resolved" in
  "$root"/src/socr/*)
    echo "CANARY OK: socr.__file__ = $resolved"
    echo "           worktree      = $root"
    exit 0
    ;;
  *)
    fail "socr resolves OUTSIDE this worktree
         got      : $resolved
         expected : $root/src/socr/...
         fix      : export PYTHONPATH=$root/src   (and re-run)"
    ;;
esac
