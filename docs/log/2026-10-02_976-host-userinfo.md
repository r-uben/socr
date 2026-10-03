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
