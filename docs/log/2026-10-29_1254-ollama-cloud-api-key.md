# #1254: Ollama cloud models by API key, no daemon

## Why

On a cluster batch job, socr's table judge could not reach its rung 1 (`glm-5.3-flash:cloud`).
socr talks to a local Ollama daemon, which relays to the cloud, and a compute node has no
daemon. So the cluster recipe turned the ladder off, and failed native tables shipped as
marks (#1253).

Bocconi compute nodes do have internet: `ollama.com` answered in 0.14 s from `cnode04`, job
723497, 2026-10-27.

## Scope

Narrowed on 2026-10-28 by owner decision:
- (a) Ollama cloud by API key;
- (c) preflight names each rung's route.

The paid Gemini API backend (b) moved to the Mac mini repair cascade (follow-up to #1255),
where the subscription CLIs (`agy`, `codex`, `claude -p`, ...) and the cost order live.

## Design

- **Only cloud-tagged models move.** `route_ollama(host, model)` sends a model ending in
  `:cloud` or `-cloud` to `https://ollama.com` under its cloud name, and only when
  `OLLAMA_API_KEY` is set. Every local model, and every call without a key, stays on
  `OLLAMA_HOST` exactly as before.
  - The rejected alternative was setting `OLLAMA_HOST=https://ollama.com`. It would also send
    local models (Qwen on a laptop) to the cloud, and break them.
- **Cloud names.** On 2026-10-29, `GET https://ollama.com/api/tags` listed the judge's models
  without a suffix: `glm-5.3-flash`, `kimi-k2.6`. So the suffix is stripped on the cloud
  route, and the rung id keeps the configured name.
- **One seam each.** Rung 1, the blind-cell adjudicator and the cell transcriber all post
  through `table_rung_ollama._post_chat`, and probe through `ollama_rung_reachable`. Routing
  lives in those two functions.
  - The Bearer header lives in `ollama_endpoint` and `urllib_auth_headers`, the only places
    socr builds Ollama requests, so any later caller that routes to the cloud is
    authenticated the same way.
- **The key goes to one host.** `_bearer_for` sends it to `https://ollama.com` and nothing
  else. That excludes `http://ollama.com`, look-alike hosts, and hosts carrying GH-976
  userinfo.
  - The key is read from the environment at call time, never logged, and does not appear in
    errors: httpx errors carry the URL, not headers. A test pins this.
- **Preflight** reports `rung 1: <model> via the ollama cloud API (OLLAMA_API_KEY)` or
  `... via the ollama daemon at <host>`, plus `rung 2: <binary> CLI`, on success and on
  failure.

## Known limits

- The cloud's `/api/tags` catalog is public, so reachability is True with a key even when the
  key is wrong. A bad key shows up at the first call, as a 401 classified as a refusal
  (`unavailable` and `refusal`), so the run's per-run breaker trips. It does not loop.
- Not verified live: whether the direct `/api/chat` accepts `"images"` and `"format": "json"`
  the way a daemon does. Ollama's docs do not say. One live call with a real key on a crop
  must confirm it before a production run.

## Tests

`tests/test_gh1254_ollama_cloud_api_key.py`, 24 tests. Each behaviour is pinned as a
difference between a run with the key and a run without it. httpx is faked, and the suite's
autouse no-daemon stub is lifted only by restoring the real `_post_chat`.

## Mutants

Each ran in an external copy of `src`, `tests` and `pyproject.toml`, with `socr.__file__`
inside the copy and `count(anchor) == 1`:

| mutant | failing tests |
| --- | --- |
| no routing in `_post_chat` | 1 |
| no routing in the reachability probe | 1 |
| no Bearer header | 2 |
| key sent to any host | 6 |
| `-cloud` suffix not recognised | 1 |
| preflight blind to the cloud route | 1 |
