# GH-910: check Ollama models over HTTP instead of the CLI

## Problem

`check_ollama_model` ran `["ollama", "list"]`. On macOS `/usr/local/bin/ollama`
is a symlink into `/Applications/Ollama.app`; against an unreachable
`OLLAMA_HOST` the CLI tries to start a server, which launches Ollama.app and
steals window focus (about 20 times in 25 minutes during a hermetic suite run
with `OLLAMA_HOST=http://127.0.0.1:9`).

## Change

`src/socr/core/ollama_utils.py`:

- `check_ollama_model` now does `GET {resolve_ollama_host()}/api/tags`.
  `OLLAMA_HOST` is honoured through the same `resolve_ollama_host` as the rest
  of socr (imported lazily, as the other callers do).
- Order: `host_reachable(host)` first (spawn-free, DNS-bounded). Unreachable
  returns "Ollama is not running or not installed" with no HTTP call. Then the
  GET.
- Signature and return convention unchanged (`None` = pulled, string = absent
  or unreachable).
- Error strings: unreachable / connection error keep the old
  "Ollama is not running or not installed"; timeout keeps
  "Ollama did not respond (timeout)"; a non-200 or unparsable body appends a
  detail to the "not running" text. The "Ollama is not installed (ollama command
  not found)" string is gone: the binary is no longer consulted, and a missing
  binary is indistinguishable from a down server over HTTP (and irrelevant to
  socr).

### Matching semantics (unchanged)

The old code took the first column of `ollama list` and tested
`model_name in names`: exact string equality, tag included. There was NO
`:latest` defaulting and NO prefix matching (the brief expected both; neither
existed). The new code builds the set of `models[].name` plus `models[].model`
and does the same exact test. `tests/test_check_ollama_model.py::_MATCH_TABLE`
(12 cases, x2 for `name`/`model` keys) pins this. The same table was also run
against the OLD implementation (`git show HEAD:...` with `subprocess.run`
stubbed to a formatted `ollama list`): 12/12 agreed.

### Bounding

httpx's `timeout=` is per-read inactivity, so a peer that trickles bytes never
trips it. Choice: run `httpx.get` in a daemon thread and `join(timeout)`,
abandoning an overrun. Reasons: identical to the `_resolve_within` approach
already in this module; a hard total bound; no process spawned; a daemon
thread cannot keep the process alive. Rejected: streaming with a per-chunk
deadline check plus size cap (overshoots by up to one read timeout, and a peer
that sends nothing after headers still relies on the per-read timeout); a
subprocess/`run_killable` (spawns a process per check, the thing this ticket
removes). Cost: an abandoned request keeps its thread until httpx's own
per-read timeout ends it; checks run once per engine init, not per page.

`TAGS_CHECK_TIMEOUT_SEC = 10.0`: the old subprocess timeout, kept.

## Call sites that exec the ollama binary

`grep -rn '"ollama"' src/`, `subprocess.*`, `shutil.which` across `src/`:

- `src/socr/core/ollama_utils.py` `check_ollama_model` : the only one.
  Replaced. Reached through `_check_ollama_model` in `engines/qwen.py`,
  `engines/deepseek.py`, `engines/glm.py` (`initialize`/`is_available`), and so
  also through `resolve_auto_engine` (the #841 probe floor), which now spawns
  nothing.
- Other subprocess sites are not the `ollama` binary and are unchanged:
  `engines/base.py` (`<engine cli> --version`, e.g. `deepseek-ocr`, `glm-ocr`,
  `qwen-ocr`, per-engine CLIs), `engines/vllm_manager.py` (vllm server),
  `judge/table_rung_gemini.py` (gemini CLI), `devtools/regenerate_p6_prechange.py`
  (git).

## Tests

- `tests/test_check_ollama_model.py` rewritten for HTTP (stubs
  `host_reachable` and `httpx.get`; no subprocess, no sockets): match table,
  present / absent, unreachable makes no HTTP call, HTTP error, connection
  error, timeout, malformed JSON (4 shapes), trickling peer hits the total
  deadline, and a guard that monkeypatches `subprocess.run/Popen/call/
  check_call/check_output` and `os.system` to raise.
- No other test stubbed `subprocess.run` for this function (others patch
  `_check_ollama_model` itself). No test execs the real ollama CLI.
- Run with `OLLAMA_HOST=http://127.0.0.1:9` plus a temporary autouse
  `subprocess.Popen` guard raising on any `ollama` argv (not committed): see
  the commit report for counts.

## Mutation checks (copies of src+tests+pyproject in /tmp, socr.__file__ canary asserted inside the copy, anchor count == 1 asserted)

- (a) subprocess call restored: `test_never_spawns_a_process` fails (1 failed).
- (b) `host_reachable` pre-check dropped: `test_unreachable_makes_no_http_call`
  fails (1 failed).
- (c) matching broken: prefix match, 8 failures in `test_matching_rules`;
  `:latest` defaulting added, 4 failures in `test_matching_rules`.
