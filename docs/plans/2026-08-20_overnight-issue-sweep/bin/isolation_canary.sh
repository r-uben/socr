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

root="${1:-$(git rev-parse --show-toplevel 2>/dev/null)}"
[ -n "$root" ] || fail "no worktree given and cwd is not inside a git worktree"
root="$(cd "$root" 2>/dev/null && pwd -P)" || fail "worktree path does not exist: $1"

main_real="$(cd "$MAIN_CHECKOUT" 2>/dev/null && pwd -P || echo "$MAIN_CHECKOUT")"
[ "$root" != "$main_real" ] || fail "refusing to run in the main checkout ($main_real) — it is owned by another session (CONTRACT fact 2)"

[ -d "$root/src/socr" ] || fail "no src/socr under $root — not a socr worktree"
[ -x "$PY" ] || fail "interpreter not found: $PY"

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
