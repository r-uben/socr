# #826 — absent vs. refused at the fail-closed floor (design note)

Read-only design pass. No `src/` or `tests/` change. Worktree `/tmp/wt-826d` off `origin/main`
@ `392a8df`; every line reference below is that tree. Predicate behaviour in §1.4 was executed
against that source (`PYTHONPATH=/tmp/wt-826d/src`, `socr.__file__` asserted inside it).

---

## 1. What exists today

### 1.1 Where the floor is raised

One site. `src/socr/core/manifest.py:3563-3592`, inside `_select_page_output_tagged`'s
structure-class branch:

```python
timed_out = page_judge_timeout_attempt(p) is not None
text_table_declined = not timed_out and structure_class_text_table_declined(p)
floor_text = structure_class_floor_text(p, page_num)
return PageOutput(
    page_num=page_num,
    text=floor_text,
    status=PageStatus.ERROR,
    engine="native",
    audit_passed=False,
    failure_mode=(
        FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED
        if not (timed_out or text_table_declined)
        else (...)
    ),
), (SelectionProvenance.STRUCTURE_CLASS_FLOOR if not (timed_out or text_table_declined) else ...)
```

The audit event the issue quotes is emitted downstream from the bucket this tag feeds,
`src/socr/pipeline/orchestrator.py:13799-13810`:

```text
kind="structure_class_ladder_exhausted_floor",
engine="native",
detail=(
    "every usable grid candidate was refused/absent; marker plus "
    "page image was selected, and the native geometry grid was "
    "withheld (fail-closed floor)"
),
```

### 1.2 The predicate that decides it

`structure_class_floor_applies` (`manifest.py:2091-2116`) is two conjuncts:

```python
if not _reaches_structure_class_branch(p):
    return False
return structure_class_grid_winner(p) is None
```

`structure_class_grid_winner` is `None` when the strict pool is empty and A1b's row
corroboration found nothing. The strict pool is built from `_grid_authored_attempt`
(`manifest.py:887-911`), whose structural half is `_grid_shaped_attempt`
(`manifest.py:811-841`).

### 1.3 The exact line where absent becomes refused

**`src/socr/core/manifest.py:837-840`**, inside `_grid_shaped_attempt`:

```python
    text = (out.text or "").strip()
    if not text or is_page_failed_marker(text):
        return False
    if out.status is PageStatus.ERROR or out.failure_mode is FailureMode.HALLUCINATION:
        return False
```

Line 838 rejects an attempt because it has **no text** — the shape of every absence. Line 840
rejects an attempt because a witness **measured it as fabricated** — a verdict. Both return the
same `False`, into the same empty pool, into the same floor, under the same failure mode. There
is no third return value and no caller that could tell them apart.

The same collapse is stated, deliberately, one function up. `_grid_authored_attempt`'s own
docstring (`manifest.py:903-906`) says the quiet part:

> an empty ``rejection_class`` is indistinguishable from "no judge ever ran" and must default to
> distrust, not to "authored a grid, ship it."

That default is defensible for a *judged* candidate. It is applied identically to a candidate
that never existed.

### 1.4 Why an absent candidate still reaches the floor at all

`_reaches_structure_class_branch` ends (`manifest.py:997`):

```python
    return any(not (a.engine or "").startswith("native") for a in p.attempts)
```

This is the R3 gate: "at least one non-native model rung ran". Every absence path in
`agentic.route_page` appends a `ProviderAttempt` built by `_error_output`
(`src/socr/pipeline/agentic.py:137-140`):

```python
def _error_output(page_num: int, msg: str) -> PageOutput:
    return PageOutput(
        page_num=page_num, text="", status=PageStatus.ERROR, error=msg, audit_passed=False
    )
```

`engine` defaults to `""` (`src/socr/core/result.py:344`), and `"".startswith("native")` is
False — so **a rung that timed out satisfies the gate that asks whether a rung ran.** Measured
on this tree:

```
timed_out  grid_shaped False   authored False
halluc     grid_shaped False
budget     grid_shaped False
engine default repr: ''   R3 gate contribution: True
failure_mode on timed-out attempt: FailureMode.NONE
```

Three inputs — a provider timeout, a measured hallucination, a budget skip — are one value to
every consumer.

### 1.5 The second lane the issue reports (p51)

`page_judge_timeout_attempt` (`manifest.py:1829-1850`) already reads a **typed** field,
`judge_outcome == JUDGE_OUTCOME_TIMEOUT`, and #713 gave that case its own failure mode, its own
`SelectionProvenance`, its own audit kind and its own CLI line. On that lane the absent/refused
distinction is **already made — and the page is discarded anyway.** `STRUCTURE_CLASS_PAGE_JUDGE_TIMEOUT_FLOOR` (`manifest.py:2860`) maps to the same `FAIL_CLOSED_MARKER` ending as
`STRUCTURE_CLASS_FLOOR` (`manifest.py:2929-2946`).

This is the sharpest fact in the ticket. Done-when item 1 (distinguish absence in the sidecar)
is already solved once, in #713, and it bought nothing, because nothing branches on it. Item 2
(absence retries or degrades) is the whole issue. Re-solving item 1 on the provider lane without
item 2 reproduces #713's outcome exactly.

### 1.6 Why page 19 kept its prose and pages 17/20/21/22/24/25/27/32 did not

Not the absent/refused axis at all. `structure_class_floor_text` (`manifest.py:2119`) delegates
to `table_floor_text_for_source` (`manifest.py:2269-2329`), whose GH-520 coverage guard requires
**four** things before any prose ships (`manifest.py:2307-2326`): `detected_table_count > 0`,
`len(detected_bboxes) == count`, `native_table_region_count == count`, `len(find_table_blocks(text))
== count`, plus `_table_bbox_sane`. Page 19 satisfied them. A borderless table seen only by the
lane-cooccupancy pass contributes no bbox and is not counted, so `detected_count == 0` is common,
and zero fails the first condition — the whole page floors.

There is a second recovery in this repo that does **not** need table geometry:
`native_prose_floor_text` (`manifest.py:2566-2600`) partitions the page's own native baseline
bands and withholds only the numeric bands. Its sole call site is `manifest.py:3169` — the
`UNVERIFIABLE_TABLE_SCANNED` / D3 branch. **It is not wired into the structure-class floor.**

So Done-when item 3 is a separable defect with a separable fix, and it is the one that recovers
bytes on the first run. Bundling it into the absent/refused decision is what would make this
ticket unshippable.

### 1.7 Where the absence reason currently goes

`orchestrator.py:8816-8818`:

```text
att.output.skip_reason = (
    att.reason if not att.accepted and not att.output.text else ""
)  # B3
```

`att.reason` is free text set at four sites in `agentic.route_page`: `"budget exceeded"`
(`agentic.py:238`), `"provider timeout"` (`agentic.py:298`), `"provider raised"`
(`agentic.py:323`), `"all providers failed"` (`agentic.py:158`). `skip_reason` is read at
exactly two places, both of them reporting: a journal string (`manifest.py:4522-4527`) and a
sidecar copy (`orchestrator.py:11838`). **No selection branch reads it.** The information exists
and is thrown away at the one place it would matter.

### 1.8 The `re-run the page` dead end (#728), and the precedent that closes it

A floored page is ERROR, so the per-page ledger gate refuses to restore it (`_load_terminal_page`,
`orchestrator.py:11329`, requires status exactly SUCCESS) — the page *would* be re-read. It never
gets the chance: the **document** gate skips first (`orchestrator.py:900-924` → `_resume_skippable`,
`orchestrator.py:540-549`).

That gate already takes two "a pending retry refuses the skip" hooks:
`equation_lane_retry_blocks` and `table_judge_retry_blocks` (`orchestrator.py:547-548`, documented
at `orchestrator.py:560-580`). A third latch of identical shape is the precedent-following fix, and
it is the mechanism that would make "absence retries" mean anything across runs.

---

## 2. Is the distinction representable today?

**Partly — and the missing half is on the provider lane, not the judge lane.**

Typed signals that already exist and already carry "no measurement was taken":

| signal | where | carries |
|---|---|---|
| `JUDGE_OUTCOME_TIMEOUT` | `manifest.py:1832`, read by `page_judge_timeout_attempt` | judge never answered |
| `FailureMode.PAGE_JUDGE_TIMEOUT` | `result.py` | page-level verdict missing |
| `FailureMode.TABLE_UNVERIFIED` vs `TABLE_REJECTED` | `result.py:99-106` | ladder could not adjudicate vs. ladder refused |
| `FailureMode.NO_WITNESS_BACKEND` | `result.py:144-152` | no witness existed to read the pixels |
| `FailureMode.TIMEOUT`, `MODEL_UNAVAILABLE`, `EMPTY_OUTPUT`, `API_ERROR` | `result.py:49-53` | the enum members for provider-side absence |
| `rejection_class` / `REJECTION_AMBIGUOUS_DEFERRED` | `result.py:380`, `manifest.py` | how a refusal was reached |

The vocabulary is not missing. What is missing is that **`agentic.route_page` never writes it.**
`_error_output` (`agentic.py:137-140`) sets no `failure_mode`, so every provider-side absence
arrives at selection as `FailureMode.NONE` + `PageStatus.ERROR` + empty text — the one shape
`_grid_shaped_attempt:838` cannot read. `FailureMode.TIMEOUT` is set only by the engine
subprocess wrappers (`engines/base.py:208`, `:301`; `engines/deepseek_vllm.py:175`), never by the
ladder's own deadline.

So: a new enum member is **not** required to distinguish provider absence. Setting the existing
ones at the four `agentic.py` absence sites, and reading them at `manifest.py:837`, is sufficient.
A new member is only required if the design wants one tag that means "this page reached the floor
with no measurement at all" as opposed to four differently-caused ones — a floor-level
`SelectionProvenance` member (`STRUCTURE_CLASS_CANDIDATE_ABSENT_FLOOR`) plus a `FailureMode`
sibling, which is the shape #713 and #714 both already used for this same floor.

---

## 3. Every way a candidate can be absent, and every way it can be refused

### Absent (no measurement taken)

| # | cause | site | what the code produces today |
|---|---|---|---|
| A1 | provider deadline fired | `agentic.py:290-312` | `_error_output`, text `""`, status ERROR, **failure_mode NONE**, `skip_reason="provider timeout"`; `reason` contains "timeout" so the halt probe can arm |
| A2 | provider raised (transport, 5xx, decode) | `agentic.py:313-327` | same shape, `skip_reason="provider raised"`, no "timeout" substring → halt probe **not** armed |
| A3 | budget exhausted before the rung ran | `agentic.py:232-244` | same shape, `skip_reason="budget exceeded"` |
| A4 | no rung produced anything at all | `agentic.py:152-158` | same shape, `reason="all providers failed"`, `engine=AUTO` |
| A5 | engine/model unavailable on this host | ladder never offers the profile | the profile is absent from `p.attempts`; if no non-native rung remains, `_reaches_structure_class_branch:997` is False and the page never reaches this floor |
| A6 | backend wedged, document halted | `orchestrator.py:9015-9040` | `partial_save_vlm_timeout` audit event + doc-level halt; **the halted page is still flushed through selection and floored** |
| A7 | page judge timed out | typed, `manifest.py:1829` | `STRUCTURE_CLASS_PAGE_JUDGE_TIMEOUT_FLOOR` — distinguished, still discarded (§1.5) |
| A8 | table-judge ladder produced no parseable verdict | `FailureMode.TABLE_UNVERIFIED`, `result.py:99` | distinguished at the table level; keeps bytes there |
| A9 | no OCR witness installed | `FailureMode.NO_WITNESS_BACKEND` | distinguished, own CLI line, still fails closed |
| A10 | empty doubt set / adjudicator declined | documented at `result.py:107-119` | explicitly **not** treated as evidence; leaves `TABLE_UNVERIFIED`, keeps bytes |

A1–A4 and A6 are the issue's case and are **all one indistinguishable value** at
`manifest.py:837-840`. A7–A10 are already typed; A7 and A9 still discard.

### Refused (a verdict was reached)

| # | cause | what the code produces |
|---|---|---|
| R1 | page judge completed and rejected | attempt not `audit_passed`, `rejection_class` empty or `REJECTION_AMBIGUOUS_DEFERRED`; pooled or not at `manifest.py:909-910` |
| R2 | native table verifier CERTAIN_FAIL (numeric multiset / label binding) | `audit_passed=False`, empty `rejection_class` → distrusted by the allowlist |
| R3 | source-evidence witness contradicted the numbers | `FailureMode.HALLUCINATION` → `manifest.py:840` |
| R4 | structural gate: not a strict grid | fails `has_strict_table_grid` inside `_grid_shaped_attempt` |
| R5 | table emission invalid | `FailureMode.TABLE_EMISSION_INVALID` |
| R6 | table judge ladder corroborated a FAIL | `FailureMode.TABLE_REJECTED` |
| R7 | reading truncated mid-emission | `_truncated_grid_reading_ids` |
| R8 | text-table declined a route it was never eligible for | `STRUCTURE_CLASS_TEXT_TABLE_FLOOR` — #714 already split this out, and its own comment insists this is **not** a refusal |

Note R8: the repo has already twice refused to let "not eligible" be called "refused". #826 asks
for the third instance of a pattern this codebase has established.

---

## 4. The fork

Both options set the typed failure mode at the four `agentic.py` absence sites and give the
floor its own tag and message — that part is common, deterministic, and not in dispute (it fixes
Done-when items 1 and 4 and puts no model in the routing path). The fork is **what absence then
does.**

### Option A — Absence latches a retry; the run still ships the marker

Introduce one absence-aware floor tag. When it fires, write a per-document retry latch in the
record and refuse the document-level resume skip, mirroring `table_judge_retry_blocks`
exactly (`orchestrator.py:547-580`). The next invocation re-opens the document, the per-page
ledger restores every page that finished, and only the absent pages are re-read — cheap, because
absence costs nothing to detect. Nothing unverified ever ships. Cost: on the run where the
backend wedged, the page still delivers the marker, so the loss the issue measured is not
recovered *in that run* — it is recovered by the next one. Cost: it only works if #728 is closed
in the same change, because today the second invocation skips the document whole. Risk: a third
latch on a gate that already carries two, and a wedged backend that stays wedged produces an
un-skippable document that re-reads forever unless the latch is bounded — the existing latches
bound themselves on capability ("is a rung reachable now"), and an absence latch has no equally
crisp bound.

### Option B — Absence degrades in-run to the prose floor

On absence (never on refusal), route the floor text through `native_prose_floor_text`
(`manifest.py:2566`) instead of the geometry-gated `table_floor_text_for_source` — the recovery
that already exists, already ships on the D3 lane, and already withholds every numeric band while
keeping prose. The page keeps its prose under a distinct absence failure mode and stays ERROR.
Cost: **this ships bytes nothing examined**, which needs its argument stated, not assumed. The
argument available is that the withheld region is exactly the numeric content the absent candidate
could have spoken to, and the band partition — not the parser being audited — decides the
boundary; the prose was never what the ladder was adjudicating. Risk: it is a strictly larger
blast radius than A, because it changes what bytes leave the pipeline rather than when they are
re-read. Second risk, and it is the one to check first: **B may be inert on the issue's own
pages.** `native_prose_floor_text` returns `None` whenever it cannot prove what it would ship; if
it returns `None` on the MPR pages, B delivers the same 105-byte marker as today and only A
recovers anything. That is measurable before choosing, and §5 is how.

Not proposed: an LLM deciding absent-vs-refused. Every signal needed is already a typed field on
a dataclass; a model in this path would put non-determinism inside the gate whose whole job is to
make the same bytes produce the same output.

Deliberately out of scope here: §1.6 (Done-when item 3, wiring the prose floor into this branch
unconditionally). It is a real defect, it is the one that recovers bytes today, and it is
independent of this fork — note that Option B is a *conditional* version of the same wiring, which
is why the two must not be decided in one breath.

---

## 5. What must be measured before choosing

**Measurement 1 — does Option B recover anything at all?** Replay the eight floored MPR pages
(17, 20, 21, 22, 24, 25, 27, 32) through `native_prose_floor_text` off their persisted page
state and count how many return a non-`None` body, and how many native characters that body
carries. If the answer is zero pages, B is inert on the corpus that motivated the ticket and the
fork collapses to A.

**Control for measurement 1:** run the identical call on **page 19** — the page that already
ships prose through the *other* guard. A recovery function that returns a body on the eight
defective pages has not been shown to be about absence until it also returns the *same* body on
the page that is already correct. If it returns `None` on page 19, the function is not doing what
this design claims and B is disqualified regardless of the eight.

**Measurement 2 — is absence actually the discriminator?** The issue's own repeat run is the
right instrument and it is already half-built: 16 pages failed loaded, 6 failed idle, the idle
set a strict subset. Instrument the ten load-only pages and classify each by which of A1–A6
produced its empty attempt. The claim under test is that **every** load-only failure carries a
provider-absence reason and **no** always-failing page does.

**Control for measurement 2:** the six pages that fail under *both* conditions. If those also
carry a provider-absence reason, then absence does not separate load-induced loss from structural
loss, the "same failed set under different load" acceptance criterion cannot be met by this
change alone, and the ticket's scope is wrong. A signal seen on the ten has not been shown to be
about load until it is seen *not* to fire on the six.

**Measurement 3 — is the latch bounded?** Run Option A's latch against a backend that stays
unresponsive and count invocations before the document settles. An unbounded latch turns a wedged
GPU into an infinite re-read loop, which is a worse failure than the one being fixed.

---

## 6. Acceptance-criteria readiness

| Done-when | implementable as written? |
|---|---|
| 1. absence distinguished in the sidecar | **Yes**, and cheaply — set the existing `FailureMode` members at `agentic.py:158/238/298/323`, read at `manifest.py:837`. Needs tightening in one respect: #713 already did this for the judge lane and the page still died, so the criterion should say *distinguished **and branched on***, or it can be satisfied vacuously. |
| 2. absence retries or degrades, does not discard | **This is the fork.** Not implementable until A-or-B is chosen. "Retries" and "degrades" are different tickets with different blast radii and the criterion currently permits either. |
| 3. degrade the way page 19 does | **Implementable, and independent of the fork** (§1.6) — but not as written: page 19's path is the GH-520 geometry guard, which structurally cannot fire on a page with `detected_table_count == 0`. "The way page 19 does" is unachievable on the pages that need it; the achievable behaviour is `native_prose_floor_text`. Reword to name the outcome (prose kept, numeric bands withheld) rather than page 19's route. |
| 4. replace `refused/absent` with a message naming which happened | **Yes**, follows from item 1 mechanically. |
| 5. `re-run the page` works or is removed | **Yes** — the page-level ledger already refuses to restore an ERROR page; the block is the document gate, and `_resume_skippable` already takes two latches of exactly the needed shape (`orchestrator.py:547-580`). Bounding the latch (§5, measurement 3) is the open part. |
| 6. (added in comment) same failed set under different load | **Not implementable as an acceptance test yet.** It is a property of the whole pipeline under a condition the test cannot control, and measurement 2's control is what decides whether this change can deliver it at all. It belongs as a measurement, not a checkbox. |

---

## The question the owner has to answer

**When no measurement of a page was ever taken, does socr keep the page's prose in that run under
an absence tag (Option B, shipping bytes nothing examined), or ship the marker and latch the
document for a re-read on the next invocation (Option A, no unverified byte, loss deferred rather
than recovered)?**
