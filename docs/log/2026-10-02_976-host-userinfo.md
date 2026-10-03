# GH-976: resolve_ollama_host mangled URL userinfo

## Cause
`_bracket_bare_ipv6` (in `src/socr/tables/extract.py`, where `resolve_ollama_host`
lives, not `core/ollama_utils.py`) treats a host token with more than one colon
as a bare IPv6 literal. The `:` in `user:pass@` counted, so any userinfo host
with an explicit port, or with an IPv6 host, was bracketed whole:
`http://u:p@h:9` became `http://[u:p@h:9]` (unparseable; returned verbatim, so
requests failed). The `GH-222` warning also logged that raw value, credentials
included.

## Fix
Split userinfo off at the last `@` before the colon count, bracket only the host
part, restore the userinfo untouched. The unparseable-host warning now logs
`safe_host_label(candidate)`. A grep of `src` for other log/raise sites that
print a host found none unrouted (all others already use `safe_host_label`).

## Forms fixed (all previously garbled)
`u:p@h:9`, `http://u:p@h:9`, `http://u:p@h:9/`, `http://u:p@[::1]`,
`http://u:p@[::1]:9`, `u:p@[::1]`, `u:p@::1`, `http://u:p@::1`, `https://u:p@::1/x`,
`http://u:p%40w@h:9`. Userinfo forms without a port or IPv6 already worked and are
pinned. Every non-userinfo form is pinned byte-identical (including the odd but
pre-existing `::1:11434 -> http://[::1:11434]`).

## Tests
`tests/test_gh976_host_userinfo.py`: pinned outputs, "userinfo twin resolves like
the plain host", env-var path equals explicit path, warning leaks no credentials.
Mutations (external copy, `socr.__file__` canary, uncapped anchor count asserted):
no userinfo split, split on first `@`, drop userinfo on restore, leaky warning.
Each fails at least one test.

Scope note: touched `tables/extract.py` (small), not orchestrator/agentic.

## Review round (Astra on #977): the "no other unrouted host logging" claim was wrong

Corrected. Two leaks remained:
1. `judge/table_rung_ollama.py` `ollama_rung_reachable` logged the raw `resolved`
   host and raw `exc` at debug. Now `safe_host_label(resolved)` and
   `redact_credentials(str(exc))`.
2. `httpx.Response.raise_for_status()` writes `... for url 'http://u:p@h/api'`
   into the `HTTPStatusError` message (verified; connect/DNS errors do not embed
   the URL). That text reaches `str(exc)` loggers and audit events downstream
   (`pipeline/agentic.py:362,382`, `orchestrator.py:~3135,~4240,~4492,~8695,
   ~11082,~15641`, `source_evidence`, ...). Rather than patch each consumer
   (orchestrator is under edit by #974), the source is fixed: every Ollama/vLLM
   `raise_for_status()` call now goes through
   `core.ollama_utils.raise_for_status_redacted`, which re-raises the same
   `HTTPStatusError` type with a redacted message and `from None`.
   Sites switched: `tables/extract.py` (6), `core/ollama_utils.py` (1),
   `engines/gemini_api.py` (1), `judge/table_rung_ollama.py` (2),
   `judge/vllm_judge.py` (2), `judge/ollama_judge.py` (1).

### Audit: logger/print/audit calls interpolating a host, URL or exception from an Ollama call
- Host: all `label=`/log sites in `extract.py`, `equation_latex.py`, `recover.py`,
  `gemini_api.py`, `table_rung_ollama.py` use `safe_host_label` (checked by grep
  for `host|resolved|base_url` inside logger/raise/print calls: none raw).
- Exceptions logged or stored via `str(exc)`: the consumers above. They receive
  either non-URL text (httpx connect/timeout errors) or the now-redacted
  `HTTPStatusError`. `table_rung_ollama.py:96` additionally redacts at the site.
- Not covered: urllib (`URLError`) paths carry no URL in their text; an exception
  from a non-socr library embedding a URL in a different shape would not be caught.

### Added
`redact_credentials` (greedy to the last `@` before the first `/`, so a raw `@`
in the password is handled), `raise_for_status_redacted`. Tests: raw `@` and IPv6
zone-id resolution pins; failed request to a userinfo host leaves no credentials
in captured logs (both a 404 and a ConnectError that embeds the URL).
Mutants (external copy, canary, uncapped anchor count): unredacted status error,
no-op redact, raw host in log, raw exc in log: all killed. `rung uses plain
raise_for_status` survives and is an equivalent mutant (the log site redacts on
its own); the status-error text itself is pinned by the helper test.

Process note: a stale script of another agent in the shared scratchpad
(`e2.py`) was run by mistake and rewrote the `call_with_total_deadline` region of
`ollama_utils.py` in worktree agent-a995e88e5bffc276f before aborting. It
re-applied that script's own `new` text; the owner should `git diff` that file.

## Review round 3 (Astra on 0488083): two more leaks

1. **Regex character class.** `[^/\s'"]*@` let a password holding `'` (which httpx
   keeps) pass through. `redact_credentials` is now structural: per `scheme://`,
   the authority runs to the first `/` (bounded by the next scheme or the end of
   the text) and everything up to the LAST `@` in it is dropped. Pinned for
   `' " % ! $ ; space @ : & ( * ~ #`, several URLs in one message, the path
   boundary (a later `ops@example.com` survives), and the no-path case (errs
   towards removing).
2. **httpx logs `request.url` at INFO** (`HTTP Request: POST http://user:pass@...`),
   before any helper runs. Credentials are now kept out of every Ollama request
   URL: `split_userinfo` -> `ollama_endpoint(host, path) -> (url, httpx.BasicAuth|None)`
   for httpx and `urllib_auth_headers(host)` for urllib. `resolve_ollama_host`
   output is unchanged (it still carries userinfo); no request URL is built from
   it directly any more. Sites converted: `core/ollama_utils.py` (`_get_tags`,
   `probe_generate`), `tables/extract.py` (generation canary, `probe_ollama_idle`,
   `_ollama_read_crop`), `judge/ollama_judge.py` (`_post_generate`),
   `judge/table_rung_ollama.py` (rung probe, `_post_chat`), `engines/gemini_api.py`
   (tags, chat), `math/recover.py` and `math/equation_latex.py` (urllib; these
   previously failed outright on a userinfo URL). Not converted: vLLM
   (`base_url`, `/chat/completions`) which takes its own API key and is not an
   Ollama path. Percent-encoded userinfo is decoded (as httpx does) before it is
   sent as Basic auth.

Pin: real loopback `ThreadingHTTPServer`, real `httpx` (conftest's global
`httpx.get` stub is undone with `httpx._api.get`), DEBUG capture; asserts httpx
did log a request, the server got the exact `Authorization: Basic ...`, the path
holds no userinfo, and neither user, password fragments nor the base64 token
appear in any captured record. Same for the two urllib paths. Mutants (external
copy, canary, uncapped anchor count): first-`@`, no slash boundary, first-URL-only,
endpoint keeps userinfo, endpoint drops auth, urllib no header, ollama_judge and
rung sites bypassing the helper, unredacted status error, raw host in log: all
killed (the slash-boundary one needed a new test).

## Review round 4 (Astra on a46f6389)

1. **Credentials followed cross-origin redirects.** `math/recover.py` and
   `math/equation_latex.py` passed Authorization via `Request(headers=...)`, which
   urllib re-sends on a 301/302/303. Now `req.add_unredirected_header(...)`.
   Test: loopback A answers 302 to loopback B; B must see no Authorization (and
   the redirect must have been followed). Mutants (`add_header`, `headers=` kwarg
   in both modules): killed.
2. **Credential-free URLs were not byte-identical.** `ollama_endpoint` stripped
   trailing slashes at sites that never did. It now takes `strip_slash` (default
   False = `host + path`; True = `host.rstrip('/') + path`). Stripping sites, as
   on origin/main: `extract.py` generation canary and `probe_ollama_idle`,
   `table_rung_ollama.py` rung probe and `_post_chat`. Everything else keeps the
   host as given. Pin: against a real loopback server with a trailing-slash host,
   each of 11 sites must request the exact raw request-target the original code
   did (`/api/tags` vs `//api/tags`; raw `requestline`, because http.server
   collapses a leading `//` in `self.path`), plus a parametrised unit pin of the
   helper. Mutants (always strip, never strip, rung site loses flag, canary site
   loses flag): killed. The rung `_post_chat` is stubbed by conftest, so the test
   loads a pristine copy of the module from source.
