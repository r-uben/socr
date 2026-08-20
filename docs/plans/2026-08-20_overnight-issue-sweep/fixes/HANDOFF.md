# HANDOFF — Stream E code owner, night of 2026-08-20

Written by the code owner who carried the stack. For the successor code owner and for
the owner in the morning. Everything here is either measured or a stated opinion, and
which one it is, is marked.

---

## State of the four PRs

| PR | Branch | Head | Status |
|---|---|---|---|
| #251 | (merged) | `dc0f13f` on main | MERGED by the lead (squash) |
| #252 | `fix/225-phantom-image-urls` | `3a01cb5` | round 3, all findings addressed, based on **main**, real CI |
| #254 | `fix/195-197-198-destruction-check` | `83d23ee` | round 2 APPROVED by a fresh reviewer, plus one comment-only follow-up |
| #253 | `fix/205-tr3-auditevent` | `83924e9` | **untouched — round-2 findings NOT addressed** |
| #255 | `fix/222-probe-interface` | `a157f76` | **untouched — round-2 findings NOT addressed** |

The stack order is `main → #252 → #253 → #254 → #255`. **#253 and #255 were branched
before `#252@3a01cb5` and `#254@83d23ee` and do not contain them.** I never rebased and
never force-pushed; the lead did the one rebase this stack needed. Someone has to carry
the new heads up before #253/#255 are meaningful.

---

## What #253 still needs (my words, from `fixes/E3-review.gpt.json`)

The PR surfaces the TR-3 geometry hard-fail as an audit event on every affected page.
That part is right and reproduces. Three things are wrong with it:

1. **Blocking — I reused an existing event kind that already means something else.**
   `table_region_unverifiable` is emitted at assemble for D3 fail-closed pages, where it
   means "the OCR ladder also failed, we shipped a failed-table marker and routed the
   region to the image-asset lane". My analyze-time emission means only "TR-3 fired
   here, detection only, nothing acted on it". A consumer reading `tables_trust.json`
   cannot tell those apart, and a D3 page now carries the kind twice.

   My reasoning at the time — that reusing the kind got me `TABLE_DISTRUST_KINDS`
   membership and the audit-log ordering for free, with one meaning regardless of mode —
   was real but it traded a *consumer-visible* ambiguity for *implementer* convenience.
   That is the wrong trade in a file whose whole purpose is telling a downstream reader
   what to distrust. Give the detection its own kind and add it to
   `TABLE_DISTRUST_KINDS` and to the `rank` dict in `audit_log.py` explicitly.

   **Trap:** doing that will move `tests/test_native_only_table_status_gh211.py`'s
   exhaustive `reasons` assertion again. I already widened it once (that edit was
   independently judged legitimate — see below); changing the kind changes it a second
   time. Read that test before you touch the kind, not after.

2. **I hardcoded `62`/`245`/`25.3%` into runtime source comments.** #205 itself
   disclaims either sample as "the" rate, and this repo forbids magic numbers. They read
   as calibration when they are a single unreplicated measurement. Take them out of
   `src/` entirely — the issue is the right place for them.

3. **My scope guard is too narrow to do its job.** `test_surfacing_does_not_key_page_status_scope_guard`
   only drives `_phase_analyze`. Its stated purpose is to fail if this ever starts keying
   status or routing on the TR-3 flag — but status is decided in `_phase_assemble` and
   routing in `_phase_agentic`, neither of which it touches. As written it cannot catch
   the creep it exists to prevent. Widen it to `process()`/assemble, and remember CI has
   no provider, so patch `_available_engines_for_agentic`.

**Do not re-litigate** the one existing test I edited in that PR
(`test_native_only_table_status_gh211.py`, widening the `reasons` list). I flagged it
myself as the risky-looking kind of edit and an independent reviewer judged it
legitimate: the added entry is true, nothing was relaxed, every other assertion is
unchanged.

---

## What #255 still needs (my words, from `fixes/E6-review.gpt.json`)

1. **Blocking — the fix does not reach the path that matters.** I made
   `resolve_ollama_host()` correct and put the probe behind a `backend_probe` seam, and
   then wired neither to the real CLI path. A run configured for vLLM still probes Ollama
   at localhost, because `qwen_vllm_url` is never consulted and the CLI never assigns
   `backend_probe`. That is the entire point of #222, and my PR body's claim to have
   fixed defect 1 is true only of the helper, not of the deployment. The host resolver
   itself is fine and tested; it just needs connecting.

2. **Major — bare IPv6.** `OLLAMA_HOST=::1` resolves to `http://::1`, which is invalid.
   Bracket the host when it contains a colon and is not already bracketed. Note the
   interaction with the port-defaulting logic I added: `urlsplit("http://::1").port`
   raises, and I currently swallow that and return the value as-is, so the bug is
   silent. Fix both together and test `::1`, `[::1]`, `[::1]:11434`.

**Context you will want:** I deliberately did NOT touch `_had_timeout` in that PR —
neither its scan across all attempts nor its bare-substring `"timeout"` match. Those are
#222's defects 2 and 3, they belong with #221/#227, and
`test_trigger_predicate_unchanged` pins the predicate so a well-meaning edit there fails
loudly. Leave it pinned.

---

## Traps I hit that are not in any JSON

- **`audit_passed` is the winner-SELECTION flag, not a page-quality flag.**
  `manifest.py::_winning_page_output` returns `best_output` *only while it is True*;
  otherwise a born-digital page falls through to the native branch and ships flattened
  native text under a fresh SUCCESS. Setting it False to "flag" a page therefore
  **deletes that page's extracted content**. This shipped in #252 round 1 and destroyed a
  real table. To demote without losing content: set `status` and `failure_mode` on the
  winner, and give the document-level signal its own term in `pages_ok`. There are now
  five such terms; follow the pattern.

- **`status` is safe to demote, including to ERROR.** I initially refused #225's "fail
  the page" criterion because I assumed ERROR would delete the text. It does not:
  `failed_pages` is derived from a page-failure *marker in the body text*
  (`orchestrator.py`, `is_page_failed_marker`), not from status. Verified.

- **A test that names a symbol your fix adds is vacuous, and it is easy to do by
  accident.** I did it three times in one night — a `rejections=` keyword, a new
  `PageState` attribute, a new helper method — each time producing a baseline
  `TypeError`/`AttributeError` instead of a behavioural failure. Two of them turned
  baseline-*passing* reverse-regression guards into baseline failures, which is worse
  than useless. Cheap defence: use `getattr(obj, "new_attr", default)` in guards, call
  functions with their base-compatible signature, and always read *why* the baseline
  failed, never just the count.

- **Reproduce the reviewer's counterexample before you fix it, and reproduce your own
  before you believe it.** A reviewer's structural observation can be exactly right while
  the conclusion drawn from it is wrong (#147: the gate really is inside `if has_tables:`,
  but rotated prose-only pages extract perfectly, so there was nothing to fix). And my own
  first probe of that produced a convincing red result that turned out to be my fixture
  running rotated text off the page edge. One bad fixture looks identical to one real bug.

- **The isolation canary earns its keep.** It caught me once, mid-run, with `PYTHONPATH`
  unexported — `socr` was resolving to the *main checkout*, not the worktree. Every Bash
  call starts a fresh shell; export it in the same command as the thing you are running.

- **CI does not run on stacked PRs.** `ci.yml` triggers only on `push`/`pull_request`
  into `main`, so #253/#254/#255 report NO-CHECKS by construction. Only #252 (based on
  main) is actually exercised. The local full suite is the owner's only verification for
  the others — say so in the PR body rather than letting NO-CHECKS read as "fine".

- **`uvx ruff@0.16.0 format --check .`, never the venv ruff.** It caught reflows on three
  separate commits that the venv ruff called clean.

---

## Known gap I am leaving open, described precisely

`reconstruct.py::_row_union_bbox` (added in #254 round 2) unions the table's raw row
rects. `_numeric_row_bbox`, which it stands in for on the `None` path, additionally
unions the bbox of every word whose centre falls inside a row rect, because PyMuPDF's row
bbox is a layout rect rather than a glyph-tight one. Since the destruction check tests a
token's **centre** against the scope, a numeric token whose centre lands between two row
rects or past a row's right edge is skipped on the `None` path but checked on the normal
one.

Narrow, non-blocking (its reviewer said so), and real. The fix: after unioning the row
rects, union in the bbox of every word whose centre falls inside any row rect — which
preserves #197's overrun exclusion, because a word below the last row is still outside
every row rect. Needs a fixture with a token centred between two rows, and a full suite.
I left it rather than land a behavioural change to a just-approved detector at the end of
a long run.

---

## Things I refused, and why, so nobody quietly reverses them

- **#147 (E4) — refused as does-not-reproduce.** Measured: a rotated prose-only page
  extracts 4/4 lines in correct reading order, byte-identical to unrotated. The rotated
  *table* case is already refused at `main_sha`. Implementing it would have required
  deleting a passing test that deliberately asserts the opposite. Send it to the owner as
  a design question about remedy (2), not as a fix.
- **#221+#227 (E7) — skipped.** The gate I was given said skip if the timeout stub cannot
  carry backend identity without #159; it *can*, in three lines (`ProviderAttempt` already
  declares the fields, `prof` is in scope, the two sibling branches already populate them).
  I skipped for a stronger reason: the actual fix is a functional canary that distinguishes
  a wedged GPU from a healthy one, and that is untestable here — CI has no provider, so the
  one component whose correctness matters would ship exercised only by a stub. Both failure
  directions destroy output. Give it a design pass, not an implementation ticket.
- **Significance stars in `_destroyed_numeric_tokens` (#254).** `***` does carry meaning,
  but this predicate decides whether to **discard the whole grid**. Rejecting a table over
  one dropped presentation mark inverts the cardinal rule, and the rowizer rebuild would
  not restore the star anyway. Upheld by the lead and by an independent reviewer. It wants
  its own detector with its own remedy — flag the page, keep the grid — as a separate
  issue.

---

## Housekeeping

- `socr-night-e` clean on `fix/195-197-198-destruction-check` at `83d23ee`.
- `socr-night-base` clean at `53b0637` (0 porcelain entries). It is the run's reference
  tree — I dirtied it once by leaving a copied test behind and reported it late; check
  `git status --porcelain` there explicitly rather than trusting an `echo`.
- Temporary worktrees I created for base runs (`socr-e5base` at `83924e9`, `socr-mainbase`
  at `dc0f13f`) are both removed; `git worktree list` shows nothing stray.
- Nothing merged by me. Nothing force-pushed by me, at any point.
