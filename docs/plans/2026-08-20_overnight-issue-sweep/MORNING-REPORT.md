# Morning report — overnight autonomous issue sweep

**Run:** 2026-08-20, 01:20 → 05:10. Orchestrator: Claude (Opus 5), unattended.
**Baseline:** `main_sha = 53b0637b928c486e9ff3023fa9527905fec017b2`
**Drift since:** none — `origin/main` is still at `53b0637`.

---

## The short version

- **Nothing was closed.** 62 issues open at the start, 62 open now.
- **6 correction comments posted**, every issue verified still open afterwards.
- **2 actions held for you**, including the run's only close.
- **5 PRs open, none merged**: #251–#255. **#251 is the only one ready.** A late GPT review round (§6b) returned `REQUEST-CHANGES` on #252, #253, #254 and #255 — every one with a blocking defect.
- **Zero fabricated citations** across 192 machine-checked verdicts.

The single most valuable thing that happened tonight was a fix **not** being
written. See "The #147 story" — it is the whole justification for the gates, and it
is the first thing worth your time.

---

## 1. Baseline

| | |
|---|---|
| `main_sha` | `53b0637` — "fix(glyph): gate on the Monotype H\<number\> namespace, not table membership" |
| Live drift | **none**; `origin/main` unmoved for the whole run |
| Open issues at snapshot | 62 |
| Open issues now | **62** |
| `gh` identity | `r-uben`, alive throughout; abort latch never set |
| Reference worktree | `/Users/rubenffuertes/repos/.worktrees/socr-night-base`, clean at `main_sha` |

The main checkout at `/Users/rubenffuertes/repos/tools/socr` was never written to.

---

## 2. Coverage reconciliation

**62 snapshot issues → 62 final dispositions. No issue missing, none invented.**

| disposition | count |
|---|---|
| FIX-CANDIDATE | 32 |
| PARTIALLY-IMPLEMENTED | 9 |
| ROADMAP-UMBRELLA | 6 |
| BLOCKED-PROPOSAL | 5 |
| CHORE-PLAN | 5 |
| NEEDS-MEASUREMENT | 3 |
| ALREADY-FIXED | 1 (held, not executed — and now contested) |
| DEFERRED | 1 |

### The finding that reframes the backlog

**The premise of this sweep was wrong.** The plan assumed much of the backlog was
stale after #243/#246/#247/#250. Across 192 machine-checked verdicts, seats claimed
`ALREADY-FIXED` **five times**. The mechanical check rejected **four** of them, and
the review board rejected the fifth. Not one survived.

This backlog is not stale. It is 32 live fix candidates deep. Overnight triage was
the wrong tool for the job it was pointed at — but it did establish that, with
evidence, which is worth having.

| # | disposition | agreement | tracker action | tonight | title |
|---|---|---|---|---|---|
| 39 | ROADMAP-UMBRELLA | UNANIMOUS | — |  | P1: route engines by measured quality-per-dollar (benchmar |
| 49 | ROADMAP-UMBRELLA | UNANIMOUS | — |  | General extraction method: single-pass VLM + free native v |
| 56 | ROADMAP-UMBRELLA | UNANIMOUS | — |  | CE OCR is not solved: prioritize reliable tables and figur |
| 64 | FIX-CANDIDATE | UNANIMOUS | — |  | Audit-flag tabular-looking born-digital pages that fall to |
| 114 | BLOCKED-PROPOSAL | UNANIMOUS | — |  | proposal: socr escalate — post-hoc escalation pass so loca |
| 127 | ROADMAP-UMBRELLA | SPLIT | comment posted |  | feat(born-digital): native path discards heading, emphasis |
| 139 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(cli): --no-audit is silently inert on the default agen |
| 140 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(born-digital): math-font pages trusted as native with  |
| 142 | CHORE-PLAN | SPLIT | — |  | chore(cli): audit every flag against the agentic path — tw |
| 144 | PARTIALLY-IMPLEMENTED | UNANIMOUS | comment posted |  | bug(tables): word-geometry rowizer drops numeric values —  |
| 146 | PARTIALLY-IMPLEMENTED | UNANIMOUS | comment posted |  | bug(tables): first data row is emitted as the table header |
| 147 | ALREADY-FIXED | SPLIT | close HELD | no fix — does not reproduce | bug(tables): landscape pages are rowized along the wrong a |
| 150 | PARTIALLY-IMPLEMENTED | UNANIMOUS | — |  | bug(figures): figures are extracted as tables — the two wo |
| 151 | PARTIALLY-IMPLEMENTED | UNANIMOUS | comment HELD |  | bug(tables): a table can ship at 100% word recall with its |
| 152 | DEFERRED | SPLIT | — |  | bug(tables): two side-by-side tables are merged into one r |
| 154 | FIX-CANDIDATE | SINGLE-VENDOR | — |  | bug(routing): --max-cost-per-page does not constrain qwen- |
| 155 | ROADMAP-UMBRELLA | SPLIT | — |  | chore(arch): split pipeline/orchestrator.py god-module (~5 |
| 156 | CHORE-PLAN | UNANIMOUS | — |  | chore(docs): TODO.md / TICKETS.md drift vs closed GitHub i |
| 157 | PARTIALLY-IMPLEMENTED | SPLIT | — |  | bug(equations): --recover-clean-equations can skip pages w |
| 158 | PARTIALLY-IMPLEMENTED | SPLIT | — |  | chore(audit): populate model_version in fingerprints / def |
| 159 | FIX-CANDIDATE | SINGLE-VENDOR | — |  | bug(routing): ProviderProfile identity discarded — qwen-cl |
| 160 | FIX-CANDIDATE | SINGLE-VENDOR | — |  | bug(routing): post-route table escalation ignores --max-co |
| 161 | FIX-CANDIDATE | SINGLE-VENDOR | — | PR #251 | bug(routing): resume ledger treats judge-rejected SUCCESS  |
| 162 | FIX-CANDIDATE | SPLIT | — |  | bug(audit): table verifier exceptions fail open into the a |
| 163 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(tables): any OCR text-layer word defers the scanned so |
| 164 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(equations): rejected recovery appends the entire page  |
| 165 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(equations): PUA-only math pages skip recovery routing  |
| 166 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(tables): all-failed crop rereads look clean; crop time |
| 167 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(figures): any embedded raster routes a trusted-native  |
| 168 | PARTIALLY-IMPLEMENTED | SINGLE-VENDOR | — |  | bug(cli): --config/--profile values are silently dropped o |
| 169 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(audit): agentic manifests drop judge rejection reasons |
| 170 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(audit): replay validates page blobs but not figure/cha |
| 171 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(audit): terminal page sidecars finalized before figure |
| 172 | FIX-CANDIDATE | SINGLE-VENDOR | — |  | bug(routing): soft timeouts abandon non-daemon workers tha |
| 174 | ROADMAP-UMBRELLA | SPLIT | — |  | chore(arch): make agentic the only first-class path; quara |
| 175 | CHORE-PLAN | SPLIT | — |  | chore(arch): break inverted package layering (tables↔bench |
| 176 | CHORE-PLAN | SPLIT | — |  | chore(arch): restore dumb DocumentState blackboard + one a |
| 177 | FIX-CANDIDATE | SINGLE-VENDOR | — |  | bug(cli): single-file exit codes disagree with RunOutcome  |
| 178 | CHORE-PLAN | UNANIMOUS | — |  | chore(arch): ADR — stay Python; optional native kernels on |
| 181 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(figures): recursive find() in _cluster_drawings blows  |
| 189 | BLOCKED-PROPOSAL | SPLIT | — |  | bug(pipeline): chart silently dropped on a mixed chart+tab |
| 190 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(tables): an all-empty but structurally-valid table pas |
| 195 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(tables): GH-144's text-strategy grid rejection ships a |
| 197 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(tables): GH-144 destruction check still uses table.bbo |
| 198 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(tables): GH-144 destruction check skips decorated nume |
| 202 | BLOCKED-PROPOSAL | UNANIMOUS | — |  | proposal: measure Mistral OCR 4.1 before any routing chang |
| 203 | BLOCKED-PROPOSAL | UNANIMOUS | — |  | proposal: consume Mistral OCR 4 block labels (blocked) |
| 205 | FIX-CANDIDATE | SPLIT | — | PR #253 | bug(tables): TR-3 geometry hard-fail is detected on 62/245 |
| 213 | FIX-CANDIDATE | SPLIT | — |  | bug(tables): book indexes are routed to table reconstructi |
| 215 | PARTIALLY-IMPLEMENTED | SPLIT | comment posted |  | bug(tables): header-attribution reject term is parked — de |
| 219 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(equations): Palatino/Pazo math fonts miss `_MATH_FONT_ |
| 220 | PARTIALLY-IMPLEMENTED | UNANIMOUS | comment posted |  | feat(review): side-by-side page-image ↔ extracted-markdown |
| 221 | FIX-CANDIDATE | SINGLE-VENDOR | — |  | bug(routing): cascade-halt cannot fire — probe_ollama_idle |
| 222 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(routing): probe_ollama_idle is hardcoded to localhost  |
| 223 | BLOCKED-PROPOSAL | SPLIT | — |  | bug(structure): heading loss is not native-lane-specific — |
| 225 | FIX-CANDIDATE | UNANIMOUS | — | PR #252 | bug(fabrication): VLM invents image URLs and they ship as  |
| 226 | FIX-CANDIDATE | UNANIMOUS | — |  | bug(tables): LaTeX \multicolumn leaks into markdown header |
| 227 | FIX-CANDIDATE | SINGLE-VENDOR | — |  | bug(routing): cascade-halt fires on a page that timed out  |
| 238 | FIX-CANDIDATE | UNANIMOUS | — |  | Caption lane: fingerprint records the configured model, no |
| 245 | NEEDS-MEASUREMENT | SPLIT | — |  | bug(agentic): when the header-attribution term abstains, E |
| 248 | NEEDS-MEASUREMENT | SPLIT | — |  | bug(tables): a corrupt text layer makes prose pages look l |
| 249 | NEEDS-MEASUREMENT | UNANIMOUS | comment posted |  | bug(agentic): chart pages are graded against a phantom tab |
---

## 3. Autonomous mutations — staged vs approved vs executed

| | count |
|---|---|
| Staged by D0 | 8 |
| Approved by the review board | 6 |
| **Executed** | **6** (all comments; issues stay open) |
| Held for you | 2 |
| **Issues closed** | **0** |

Every executed write was read back afterwards and confirmed `OPEN`. Re-running the
apply script is a no-op — each comment carries a hidden `action_id` marker and the
script skips any action whose marker is already present.

| issue | comment |
|---|---|
| #144 | https://github.com/r-uben/socr/issues/144#issuecomment-5349673861 |
| #146 | https://github.com/r-uben/socr/issues/146#issuecomment-5349674143 |
| #215 | https://github.com/r-uben/socr/issues/215#issuecomment-5349674438 |
| #127 | https://github.com/r-uben/socr/issues/127#issuecomment-5349674695 |
| #220 | https://github.com/r-uben/socr/issues/220#issuecomment-5349674950 |
| #249 | https://github.com/r-uben/socr/issues/249#issuecomment-5349675188 |

**Reopen targets: none.** Nothing was closed, so there is nothing to reopen. If you
disagree with a comment, it is a comment — edit or delete it, no state to unwind.

---

## 4. Held for you

### #147 — the run's only close. **Do not close it. Do not fix it either.**

Both reviewers' readings, verbatim in `actions/review/04d07063d61a.*.json`:

- `ollama-deepseek` — **APPROVE**: commit `13033a3` is a real ancestor of `main_sha`
  and satisfies the criteria it checked.
- `gemini-pro` — **REJECT**: "The rotation check is nested inside `if has_tables:`,
  meaning rotated pages without tables bypass the gate entirely and ship as trusted
  native text."

One rejection is enough, so the close was held. Then the code owner tried to *fix*
what the reviewer found, and could not — because it is not a defect. See below.

**Your decision:** #147's closing Note says "a page whose dominant text direction is
not horizontal must not ship as trusted native text in its current form. That part
is not a design question." Read literally, that demands refusing rotated **prose**
pages too. The measurement says rotated prose extracts byte-perfectly. Acting on
the Note literally would route every rotated prose page to a paid VLM to reproduce
output that is already correct. Either the Note should be narrowed to table pages
and #147 closed, or the Note stands and there is real work to do. **That is a
design call and it is yours.**

### #151 — correction comment held

`ollama-deepseek` — **REJECT**: "the comment's central claim 'the p26 evidence shape
passes the gate' is false: the verbatim p26 markdown is ragged (row_widths
8,6,7,7,7,7,7) and fires `structural_gate_fires`, so the comment mischaracterizes
the p26 case as 'rectangular' and overstates the gap."
`gemini-pro` — APPROVE.

**Ask:** the disposition (PARTIALLY-IMPLEMENTED, stays open) is probably right; only
the p26 characterisation is disputed. One line from you settles whether to post a
corrected version.

---

## 5. The #147 story — why the gates paid for themselves

Worth four minutes, because it is the same failure this repo hit with #249 and
#250, caught three times in one night by three different mechanisms.

1. **Two triage seats** (grok, ollama-minimax) proved `ALREADY-FIXED` against a real
   ancestor commit whose subject is literally "fix(147): refuse the native table
   lane on rotated pages". Clean, exact citations.
2. **Two other seats** (claude, kimi) read the same code as PARTIALLY-IMPLEMENTED.
3. **The adjudicator broke the 2-2 tie toward closing** and staged the close.
4. **The review board refused it** — one reviewer found the `has_tables` nesting.
5. **The code owner then tried to fix that** — and its first measurement *agreed
   with the reviewer*: on a rotated prose page, 0 of 4 lines survived, every line
   cut mid-word. A textbook silent-content-loss defect. It was one step from
   writing the fix.
6. **It re-derived the fixture instead of trusting the red result.** The clipping
   was its own bug: rotated text runs upward from its insertion point, so the test
   content ran off the top of the page and was clipped by the PDF, not by socr.
   Re-placed at y=720, extraction is byte-identical to the unrotated page.

So: a plausible mechanism, a reviewer who had already asserted it, and a first
measurement that confirmed it — and it was still wrong. The thing that caught it
was re-deriving the fixture rather than trusting the first red result.

The code owner also declined to delete
`test_rotated_prose_only_page_does_not_set_needs_ocr_enhancement`, which asserts
the current behaviour is deliberate. Deleting a passing test to make a new one pass
is exactly the #250 shape.

Full measurement, including the positive control on a rotated ruled-grid table
page: `fixes/E4-result.json`.

---

## 6. PR decision queue

**None merged. All five are proposals.** They are a **stack** — each is based on
its predecessor, not on `main`.

| PR | issue | head | base | CI | local suite | reviewer | stack pos |
|---|---|---|---|---|---|---|---|
| [#251](https://github.com/r-uben/socr/pull/251) | #161 resume ledger trusts judge-rejected pages | `b562acc` | `main` | **test + typecheck PASS** | 1830 passed, 3 xfailed | gemini-pro **APPROVE** | 1 (bottom) |
| [#252](https://github.com/r-uben/socr/pull/252) | #225 fabricated image URLs ship under SUCCESS | `4200171` | `fix/161-…` | **NO-CHECKS** | 1835 passed, 3 xfailed | grok, in flight at cutoff | 2 |
| [#253](https://github.com/r-uben/socr/pull/253) | #205 TR-3 hard-fail surfaced (surfacing only) | `83f6ac8` | `fix/225-…` | **NO-CHECKS** | 1838 passed, 3 xfailed | not dispatched | 3 |
| [#254](https://github.com/r-uben/socr/pull/254) | #195+#197+#198 destruction check, one PR | `163ffcb` | `fix/205-…` | **NO-CHECKS** | 1846 passed, 3 xfailed | not dispatched | 4 |
| [#255](https://github.com/r-uben/socr/pull/255) | #222 probe extracted behind an interface | `2fe7355` | `fix/195-…` | **NO-CHECKS** | see result file | not dispatched | 5 (top) |

### #252 — a content-loss defect was found in the fix, and then fixed

Its independent reviewer (grok) returned **REQUEST-CHANGES**, and one finding is a
cardinal-rule violation introduced *by the fix*:

- **blocking, `manifest.py:314`** — demoting `audit_passed=False` makes assemble
  throw away the cleaned OCR page and ship native SUCCESS instead. On #225's own
  document class (born-digital) that is **silent content loss caused by the fix
  meant to prevent it**.
- **blocking, `manifest.py:355`** — no end-to-end assertion that a legitimate image
  ref survives assemble. The reviewer reports `legitimate_urls_still_ship = false`.
  The three reverse tests only inspect the gate predicate, not the assembled output.
- major, `normalizer.py:66` — a provenanced URL in CommonMark title syntax
  (`![Chart](https://…/c.png "Official chart")`) is stripped as fabricated.
- major, `orchestrator.py:5104` — #225 asks to **fail** the page; the PR demotes
  SUCCESS→WARNING and the born-digital path then restamps SUCCESS, so the signal
  can vanish entirely.

The reviewer independently reproduced the baseline failure, so it read the code
carefully.

**All four findings were fixed in a second round**, pushed at the very end of the
run: head moved `4200171` → **`4243212`** (normal push, no force, no rebase), with a
[reply comment](https://github.com/r-uben/socr/pull/252#issuecomment-5350071385) on
the PR. What the code owner did is worth knowing, because it changes how much you
need to re-check:

- It **reproduced the content loss before fixing it**. At `4200171`, on a
  born-digital OBR-shaped page: the sanitizer produced `WARNING / audit_passed=False`
  with the fabricated `imgur` ref gone and table value `177.0` intact — but what
  *assemble actually shipped* was `native / SUCCESS / audit_passed=True` with
  **`177.0` gone**. The reviewer was right.
- **Root cause**: `audit_passed` is the *winner-selection* flag, not a page flag.
  `_winning_page_output` returns `best_output` only while it is True, so flipping it
  made a born-digital page fall through to the native branch and discard the cleaned
  OCR page. The fix stops flipping it; the page carries its own demotion
  (`status → WARNING`, `failure_mode → HALLUCINATION`) and stays the winner.
- Reverse regressions now assert through `canonical_page_texts` — the actual source
  of the saved `.md` — not on the `PageOutput` the gate mutated. **The guard was
  verified real** by re-introducing the flipped flag and watching the new test fail,
  then reverting.
- It caught a vacuity trap **in its own new tests**: it had written the guards as
  `state.pages[1].fabricated_image_refs == 0`, which raises `AttributeError` at
  `main_sha` — turning three baseline-passing guards into baseline failures for the
  wrong reason. Changed to `getattr(..., 0)` so they stay evaluable at both revisions.

Refreshed non-vacuity at `main_sha`: 4 failed / 4 passed, all four failures
behavioural (`'i.imgur.com' not in …`), no `AttributeError`/`ImportError`. On the
branch: 8 passed. Full suite 1838 passed, 3 xfailed. Lint clean.

**One stated disagreement, which is yours to settle:** #225 asks to *fail* the page.
The code owner keeps page status at `WARNING` and makes the *document*
`AUDIT_FAILED` (via a new `fabricated_ref_pages` term), arguing that after redaction
the page content is correct — the invented pointer is gone, everything real is
intact — so emitting a page-failure marker would delete a genuine table to punish a
pointer already removed, inverting the cardinal rule. The run no longer looks clean;
the content survives. I think that argument is right, but it is a deliberate
departure from the issue's literal wording and it is flagged, not hidden.

**Caveat 1: round 2 was not re-reviewed.** `4243212` has had no independent reviewer.

**Caveat 2 — the fix is NOT in the three PRs above it.** #253, #254 and #255 were
branched off `4200171`, before the fix. Verified just now:

    fix/205-tr3-auditevent           MISSING the fix
    fix/195-197-198-destruction-check MISSING the fix
    fix/222-probe-interface           MISSING the fix

Nothing was rebased or force-pushed, deliberately. **So whoever merges must carry
`4243212` up the stack, or the content-loss defect ships in the descendants.**
Merging strictly bottom-up (#251, then #252, then #253…) handles this automatically,
because each merge retargets the next PR's base. Merging out of order does not.

**#252 is the base of #253 and #254, so all three inherit this.** Merging bottom-up
means #251 first, then **stopping** until #252 is fixed.

### Non-vacuity proof

Each PR body carries the new test run **against `main_sha`** (FAILS, on a
behavioural assertion, not an ImportError) and **on the branch** (PASSES), with
both raw outputs pasted. #251's was independently reproduced by its reviewer, who
confirmed the reverse regression is also guarded — a page that is SUCCESS *and*
`audit_passed=True` must still be skipped on resume, or the fix would silently
destroy resume for everyone.

### Read this before merging: **stacked PRs get no CI**

`ci.yml` triggers on `pull_request: branches: [main]`, which filters on the **base**
branch. Only #251 is CI-verified. #252 and #253 ran **no test job at all** — the
`NO-CHECKS` above is literal.

The stack itself is right: #161, #205 and #225 all write
`orchestrator.py::_phase_agentic`, and parallel branches would collide inside a
900-line function and could silently undo each other. Unstacking to buy a green
tick would trade a verified risk for an unverified one.

**Cheapest fix: merge bottom-up.** Merge #251; #252's base auto-retargets to `main`
and CI runs on it for real; repeat. Alternatively drop the `branches: [main]` filter
from `ci.yml`'s `pull_request:` trigger — one line, and every future stacked PR
self-verifies. Detail in `logs/2026-08-20_stacked-prs-get-no-ci.md`.

### What each PR deliberately does *not* do

- **#253 (#205)** is **surfacing only** — it emits the AuditEvent and nothing else.
  It does not key page status, document status, escalation or routing on the TR-3
  hard-fail, and there is a test asserting it does not. The issue forbids acting on
  the signal before its 62-page set is hand-judged: a 25% firing rate is not a 25%
  defect rate, and routing on it unattended could delete good tables. **#205 must
  stay open** — this closes step 1 of 3.
- **#252 (#225)** leaves `OutputNormalizer.strip_phantom_images`' absolute-URL
  passthrough alone. Two existing tests assert that contract; changing it would
  have meant editing tests that assert current behaviour to make a new one pass.
  The new provenance gate sits beside it instead.

---

## 6b. GPT review round — added after the main run, and it changes the picture

GPT's quota returned and the owner asked for the missing reviewers. Four independent
GPT reviewers were dispatched, one per PR, each seeing only its own PR and each
prompted to refute rather than confirm.

**All four returned `REQUEST-CHANGES`. Every one found at least one blocking defect.**
All four confirmed non-vacuity, all four reproduced the baseline failure themselves,
and all four left the reference worktree clean — so these are grounded readings, not
drive-by rejections. Each posted its findings as a comment on its PR.

| PR | verdict | blocking finding |
|---|---|---|
| #252 | **REQUEST-CHANGES** | The phase-major document sweep happens after overall status is computed, so a document whose only defect is a fabricated image ref remains DocumentSta |
| #253 | **REQUEST-CHANGES** | The reused event kind no longer has one meaning: it previously denoted a D3 fail-closed page routed to an image asset, but now also denotes a detectio |
| #254 | **REQUEST-CHANGES** | The new None-scope early return deletes a reachable real-detection path based on an invalid inference. |
| #255 | **REQUEST-CHANGES** | The default path still probes Ollama localhost for a configured vLLM backend, so the issue's named non-Ollama false-halt remains unless callers know t |

The three worst, in the order I would read them:

- **#254 — a real detection path was deleted.** `reconstruct.py:178`: the new
  None-scope early return removes a *reachable* real-detection path on an invalid
  inference. This was the single edit I flagged to the reviewer as the most dangerous
  in the stack, and it did not survive contact. #254 also fails #195's explicit
  page-status and document-status requirement.
- **#252 — the document sweep runs too late.** `orchestrator.py:5505`: the phase-major
  sweep happens *after* overall status is computed, so a document whose only defect is
  a fabricated ref can still finish clean. Round 2 fixed the content loss; it did not
  fix the surfacing. GPT also calls the "warn, don't fail" argument a false dichotomy.
- **#255 — the named defect is still live on the default path.** `orchestrator.py:5145`:
  a configured vLLM backend still gets probed at Ollama's localhost, which is exactly
  the non-Ollama case #222 is about. The host resolver improved; the wiring did not
  reach the default path.

**#253** is the mildest but still blocking: reusing the `table_region_unverifiable`
event kind gives it two meanings, and its scope guard only exercises `_phase_analyze`,
so it could not catch scope creep in `process`/`assemble`.

**What this means for the merge order.** Nothing in the stack is ready. #251 remains
the only PR that is both CI-verified and approved by its reviewer — it is still
mergeable on its own. Everything above it needs another round.

**What this means about the night.** My earlier assessment said to assume the
unreviewed PRs contained problems comparable to the one reviewed PR. That was right,
and if anything understated: the base rate is now five reviews, five substantive
rejections, zero clean passes. The implementation seat outran the review seat all
night, and the review seat is where the value was.

---

## 7. Deferred measurements — what needs the corpus

A local corpus exists at `~/data/fiscal-ballast` (302 PDFs); the specific cited runs
are not on disk. Nothing below can be settled without a run.

| issue | what needs measuring | command |
|---|---|---|
| #205 | hand-judge the 62 TR-3 hard-fail pages before any routing change | `socr <pdf> --agentic` over the table corpus, then `socr review <doc_dir>` |
| #144 step 3 | re-measure p17/42/43 destruction loss at `53b0637` | rerun the GH-144 corpus set |
| #245 | header-attribution defect rate — the new `table_header_unverifiable` event makes this cheap now | count the event across a corpus run |
| #249 | 49-vs-1 row grading on chart pages; needs a hand pass before any routing change | `socr review` on the chart pages |
| #248, #189 | corpus measurement / hand judgement | as above |
| #147 | rotated-page prevalence across the 407-paper library, *if* you keep the Note's literal reading | scan for non-horizontal `dominant_text_direction` |

`socr review` (#220) is the right instrument for four of these and is 4/5
implemented — the missing piece is a filter for "only pages a given gate fired on",
which is exactly what would make these passes fast.

---

## 8. Failures, skips, and things that went wrong

- **E4 (#147)** — no branch, no PR. `DOES-NOT-REPRODUCE`. Correct outcome, §5.
- **E5 (#195+#197+#198)** — landed as **PR #254**, 1846 passed locally, no blockers
  reported by the code owner, not independently reviewed.
- **E6 (#222)** — landed as **PR #255**, no blockers reported by the code owner, not
  independently reviewed.
- **E7 (#221+#227)** — **SKIPPED**, and the reasoning is better than the gate I set.
  The gate said skip if the timeout stub cannot carry backend identity without #159.
  The code owner found that premise is **false** — `ProviderAttempt` and
  `ProviderProfile` already carry `provider_id`/`model`/`backend`, `prof` is in scope
  at the timeout site, and the two sibling branches already populate all three; the
  timeout branch is simply the one never updated. Three lines, no #159 needed.
  It skipped anyway, for a stronger reason: the substantive fix is a routing
  redesign around a 1-token functional canary, and *the canary is the fix* — it can
  only be validated against a real backend in both wedged and healthy states, and CI
  has no provider at all. Correct call. Detail in `fixes/E7-result.json`.
- **#252's blocking findings** — addressed in a second round at the very end
  (`4243212`), including an independent reproduction of the content loss. Not
  re-reviewed. See §6. E7 was expected to be
  **SKIPPED**: two independent seats found the combined fix depends on #159's
  attempt-identity work, and #227 warns that fixing #221's probe alone makes
  behaviour *worse*. It remains the correct thing to skip.
- **Vendors:** grok and kimi produced nothing for ~35 minutes and were declared
  MISSING; replacement seats were dispatched. Both then delivered, and grok's
  citations were the cleanest in the run (39/39 exact). The replacements became
  useful third seats. No coverage was lost, but the wall-clock cap was tuned for
  the wrong failure mode — slow is not dead.
- **gemini-pro on batch-2 triage** never delivered; `gemini-flash` replaced it.
- **cursor** was tested and works (`cursor-agent -p --force`), but a review-board
  seat exceeded a 10-minute call budget and was re-seated on kimi. It stayed unused.
- **GPT** was at zero quota all night, per your note. It never entered the run. It
  was the sharpest critic on coverage at planning time and is the obvious second
  opinion on the #147 design question this morning.
- **`state/ABORT` was never set.** No aborted work.

### Three bugs in my own tooling, found by the agents

1. **`zsh` silently corrupts `git show $SHA:path`** when the revision argument is
   unquoted — the `:s` history modifier fires and returns a *different blob* with no
   error. This produced a wrong grounding token in my own table; an agent's correct
   value looked like a mismatch. Always `git show "${SHA}:path"`.
2. **My first fabrication classifier had a false positive.** It flagged two
   citations as invented; both were real docstrings the model had closed with a
   trailing quote the source line leaves open. It now separates
   EXACT / DRIFT / PARTIAL / FABRICATED, re-tested against adversarial fixtures.
   Final count across the whole run: **0 fabricated citations**.
3. **A verdict was voided over a schema key** — a seat wrote `{"met": true}` where
   the validator wanted `{"status": "met"}`. Fixed to accept either. The substance
   then surfaced immediately: that same verdict marks one of #220's five acceptance
   criteria UNMET, so it was correctly rejected as a close anyway.

### A correction to CONTRACT fact 1

The editable-install trap is **narrower than the contract states**. `pyproject.toml`
sets `[tool.pytest.ini_options] pythonpath = ["src"]`, resolved against rootdir, so
any pytest run whose targets are inside a worktree is *already* isolated. The trap
still bites the `socr` CLI, `python -c` probes and reproducer scripts — which is
what you reach for when building a repro — so the mandate is unchanged, but the
stated reason was wrong. Proof in `logs/2026-08-20_A1-sentinel-transcript.md`.


---

## 9. Human actions — ranked, shortest first

1. **Merge or close #251** (2 min). It is the only CI-verified PR, its reviewer
   reproduced the baseline failure independently, and merging it un-blocks CI for
   #252 automatically.
2. **Decide #151's comment** (2 min). Disposition is fine; one reviewer disputes the
   p26 characterisation. Say post-corrected or drop.
3. **One line in `ci.yml`** (2 min, optional). Drop `branches: [main]` from the
   `pull_request:` trigger so stacked PRs get CI. Pays for itself immediately.
4. **Merge only #251 for now** (2 min). It is the only PR with both CI and an
   approving reviewer. Everything above it was rejected in §6b.
5. **Read #252's round-2 fix** (10 min). A reviewer found the original caused silent
   content loss on born-digital pages; the code owner reproduced it, root-caused it
   to `audit_passed` being the winner-selection flag, and fixed it — but round 2 is
   unreviewed. It also declines part of the issue's literal wording (page stays
   WARNING, document becomes AUDIT_FAILED) on cardinal-rule grounds. Confirm you
   agree. See §6.
5. **Read the four GPT reviews** (20 min) — §6b. They are now reviewed, and all four
   were rejected with blocking findings. Start with #254's deleted detection path.
7. **Settle #147** (15 min, design). Narrow the closing Note to table pages and
   close it, or keep the literal reading and accept the work. The measurement in
   `fixes/E4-result.json` is the input. Worth a GPT second opinion now that its
   quota is back.
8. **Decide what to do with 32 live fix candidates** (30 min). That is the real
   output of the night. The backlog is not stale; it is deep. `fixes/queue.json`
   lists the 28 that were adjudicated FIX-CANDIDATE but deliberately not attempted,
   because authoring fix scope at 04:00 is how #250 happened.

---

## 10. Honest assessment of this run

**What worked.** Every gate that was argued for in the panel critique caught
something real:

- machine-checked citations caught 4 of 5 `ALREADY-FIXED` claims;
- the refute-first review board caught the 5th and stopped the only close;
- the non-vacuity requirement is what made the code owner measure #147 instead of
  implementing it;
- cluster co-adjudication caught a *unanimous* verdict on #249 that was
  category-wrong in the same way the owner's first two drafts were.

**What did not.** The plan was built to close stale issues, and there were none.
Four hours of five-vendor triage produced 6 comments and 0 closes. That is the
correct answer to the question asked, but it is a small yield for the machinery,
and the machinery's cost was mostly paid before the first verdict.

**What I would change.** Point this at the 32 fix candidates, not at the backlog's
staleness — the evidence apparatus is worth far more supervising *fixes* than
supervising *closes*. And fix the CI trigger first; a stack that cannot be verified
is a stack you have to review by hand.

**What I'd distrust in my own report.** #253, #254 and #255 have had **no
independent review at all** — only the code owner's own testing. Of the two PRs that
*were* reviewed, one was approved and one was found to cause silent content loss
that its author had not seen — and the round-2 fix for that is itself unreviewed. On that base rate, assume the three unreviewed PRs
contain comparable problems until someone looks. **The review seat, not the
implementation seat, was the scarce resource tonight, and I allocated it badly:**
I let the code owner run five tickets deep while only two PRs got a reviewer. The
stack should have been capped at whatever I could get reviewed.

---

## Where everything lives

| what | path |
|---|---|
| Per-issue final dispositions | `verdicts/batch-{1..4}.json` |
| Machine-checked evidence | `triage/verified/batch-{1..4}.json` |
| Raw triage, 11 vendor seats | `triage/batch-*/​*.json` |
| Staged actions + both reviewer readings | `actions/tracker_actions.json`, `actions/decisions.json`, `actions/review/` |
| Fix results and measurements | `fixes/*.json` |
| The #147 measurement | `fixes/E4-result.json` |
| CI/stacking problem | `logs/2026-08-20_stacked-prs-get-no-ci.md` |
| Canary proof + editable-install correction | `logs/2026-08-20_A1-sentinel-transcript.md` |
| Verifier self-test | `logs/v1-selftest/RESULT.md` |
| Tracker write log | `logs/2026-08-20_tracker-apply.log` |

Re-runnable, no model required:

    ./bin/verify_citations.py        # re-verify every citation at main_sha
    ./bin/tally_review_board.py      # re-tally the board, independence enforced
    ./bin/apply_tracker_actions.sh --dry-run
