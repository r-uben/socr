# #200 shipping spec — the structural escalation gate, ratified

Date: 2026-08-15
Status: **SUPERSEDED IN PART — the header-attribution sections did not survive implementation.**

> **Read this first (added 2026-08-15, after implementation).** The escalation-gate half of
> this spec shipped as specified: #200's plumbing with B1's own predicate
> (`ragged OR detached_label_rows`). The **header-attribution half did not**. Four
> implementations were attempted; each either abstained on the 4-of-4 header-loss case this
> spec exists to catch, or returned `HARD` on byte-perfect correct tables carrying
> significance-star or `n.a.` rows. The reject term is parked and tracked as **#215**.
>
> Specific sections now known to be wrong, kept for the record rather than edited:
> the reuse of the existing geometry chain to find a header band (it cannot recognise
> `1997/2002` or `(1)(2)(3)` bands); exact-multiset anchoring as a unique table locator on a
> multi-table page; whitespace-token equality as the comparison (it false-rejects
> presentation-equivalent headers such as an en-dash vs a hyphen); and treating one surviving
> token as preservation of a whole header cell.
>
> The measurement named as OPEN DECISION 1 remains the right next step, and is still unmade.

Inputs: `2026-08-15_tr3-hand-judgement.md` (the measurement), `2026-08-14_gh151-b1-escalation-decision.md`,
`2026-08-14_gh151-b1-predicate-design.md`, the 2026-08-15 consilium panel
(`~/.local/share/consilium/runs/2026/08/20260815T003453Z-3777/`), and two independent model proposals.

**Process note:** the adjudication brief said three proposals; the payload carried **two**. This
document adjudicates the two that arrived. Nothing below rests on a third. (The third was
discarded by the harness for omitting one required schema field, not for its content.)

Every file:line below was read in the working tree at `docs/200-tr3-hand-judgement`
(= `main` + one docs commit) unless explicitly marked *branch*.

---

## 0. The state of the tree — read this before anything else

The single most consequential fact neither proposal foregrounded enough:

**`detached_label_rows` and `structural_gate_fires` do not exist on `main`.**

- `src/socr/tables/structure_check.py` on main exposes exactly two findings:
  `FINDING_RAGGED` (:55) and `FINDING_ORPHAN_ROWS` (:56). `check_grid` (:95),
  `check_markdown` (:136). `grep -rn "detached_label_rows" src/socr/` returns **nothing**.
- `structural_gate_fires` exists only at *branch* `feat/151-b1-structural-gate:structure_check.py:210`.
- `git rev-list --count main..feat/151-b1-structural-gate` = **3**;
  `git rev-list --count feat/151-b1-structural-gate..main` = **7**. The branch is 3 ahead, 7 behind.

So the B1 predicate this spec ships **cannot be evaluated on main today**. Proposal 1 flagged the
divergence from a GitHub compare; proposal 2 read the branch diff. Both are right, and the ordering
consequence is non-negotiable:

> **Step 0 of implementation is rebasing `feat/151-b1-structural-gate` onto current `main` and
> landing #200's plumbing unchanged.** Everything in §2–§6 is built on top of it. Do not re-derive
> `detached_label_rows` in a new module; do not cherry-pick pieces of the branch.

---

## 1. Where the two proposals agree (recorded once, not relitigated)

1. **The predicate is a disjunction, not a swap.** Keep B1's own `ragged OR detached_label_rows`;
   TR-3 (`vr.hard_fail`) stays an independent disjunct; header attribution is a third term.
   Grounded in the measured non-overlap (`2026-08-14_gh151-b1-predicate-design.md` §2.4: 35/66 overlap,
   27 TR-3 pages the gate misses entirely) and in the hand judgement's finding that TR-3 is blind to
   3 of 5 observed defect classes.
2. **The winner-side hole is real.** `NativeTableVerifierJudge.assess` (`agentic.py:508-615`) returns
   `AcceptDecision(accept=True, reason="native_table_verifier: EXACT_PASS", confidence=1.0)` at
   `agentic.py:608-612` **without calling `self._inner.assess`**, on the sole strength of
   `native_verifier.py:1054-1058` (`if not result.hard_fail and not result.warn:` → `EXACT_PASS`),
   whose inputs are entirely numeric. Verified by reading both files. A model table with every numeral
   correct and a destroyed header ships at confidence 1.0 today.
3. **Header attribution needs geometry, not markdown alone.** An empty header cell over an
   implied/spacer column is legitimate; only the native word layer says whether a header was *owed*.
4. **The post-route mutation is a second hole.** `orchestrator.py:2534` assigns
   `ps.best_output = decision.final_output`, then `:2536-2568` mutates `ps.best_output.text` with
   `repair_table_headers_on_page` **after** the judge accepted. Verified. The shipped text is not the
   text any gate saw.
5. **`--native-only`: record and surface, never reroute.** Settled ruling, both concur.
6. **Fail-closed at the top rung = the existing D3 floor** (`manifest.py:308-333`), not a new floor.

## 1b. Two panel wordings both proposals correctly narrowed

The panel's "grep finds zero header checks in `native_verifier.py` and no structural check of any
kind on `agentic.py`'s accept path" is **too broad on the letter, right on the substance**:

- `header_repair.py:322-326` — `_header_is_faithful` requires every data-lane header cell
  (`cols 1..expected_cols-1`) be non-empty. That **is** a header check.
- But it runs only inside `repair_collapsed_header`, which returns `None` at `header_repair.py:361-363`
  unless `detect_header_column_collapse` already fired. It is a *repair-side guard*, not an
  *acceptance gate*. A destroyed-but-not-collapsed header never reaches it.

The hole stands. Ship the fix; drop the "zero header checks" phrasing from the PR body.

---

## 2. The escalation predicate (SHIP)

One source of truth, in `src/socr/tables/structure_check.py` (which already owns
`structural_gate_fires` post-#200). Pure, no I/O.

```python
DEFECT_NONE = ""
DEFECT_GRID_SHAPE = "grid_shape"
DEFECT_HEADER_UNATTRIBUTED = "header_unattributed"


def table_output_defect(output_md: str, words: list | None) -> str:
    """Empty string when the candidate is structurally sound, else a defect key.

    `words` is page.get_text("words"), or None (scan / no native layer).
    """
    if structural_gate_fires(check_markdown(output_md)):  # UNCHANGED #200 predicate
        return DEFECT_GRID_SHAPE  # ragged OR detached_label_rows
    if words:
        for block in find_table_blocks(output_md):
            if header_attribution(block.grid, words) is HeaderVerdict.HARD:
                return DEFECT_HEADER_UNATTRIBUTED
    return DEFECT_NONE
```

The escalation predicate at the judge is then, explicitly, a **disjunction**:

```python
reject = vr.hard_fail or table_output_defect(output.text, words) != DEFECT_NONE
```

- `vr.hard_fail` is TR-3 and is **already** a reject at `agentic.py:553-576`. It stays. It is not
  nested with, and does not subsume, the shape or header terms.
- `orphan_rows` is excluded, as `structural_gate_fires` already documents (*branch*
  `structure_check.py:210-223`). No majority vote, no empirical threshold, no new constant.
- **Do not narrow B1 to the Q_num/Q_body qualifiers in this change.** Proposal 2's reason is verified:
  those qualifiers rest on `is_numeric_token` → `_NUM_TOKEN_RE` (`reconstruct.py:78`,
  `^[\(\[]?-?\d[\d.,]*[\)\]%]?$`). I ran it:
  `.034` → `False`, `***` → `False`, `∗∗` (U+2217) → `False`, while `0.034`/`-1.2`/`(0.03)`/`12%` → `True`.
  Narrowing today would silently drop leading-decimal standard errors — the #206/#207 gap.

**Adjudication:** proposal 2's formulation ships. Proposal 1's equivalent expression
(`b1 or vr.hard_fail or hdr == FAIL`) is the same predicate; proposal 2's is the one with a named
call site and a defect key that can be carried into an audit event and a `FailureMode`.

---

## 3. The header-attribution check (SHIP proposal 2's construction, with proposal 1's abstain rigour)

### The conflict, resolved by evidence

- **Proposal 1** specifies a new `check_header_attribution(page, candidate_markdown)` requiring a
  trustworthy source *table bbox*, a monotone injective emitted-column→native-x-interval map, a
  leaf-header rule, and header-token conservation.
- **Proposal 2** specifies extracting the geometry chain that **already exists** inside
  `repair_collapsed_header` (`header_repair.py:365-400`) into a reusable `native_header_row(grid, words)`.

**Resolved for proposal 2, on two grounds, not on vote:**

1. *Minimalism ladder rung 2 (already in this codebase).* Every step proposal 1 designs is already
   written and in production: `_best_anchor_y` (`header_repair.py:95-108`), `_local_table_ys` (:118),
   `_derive_lane_centers` (:246-268), `_header_ys` (:198-243), `_assign_words_to_lanes` (:271-291),
   `_merge_multiline_header_rows` (:293-307). Reuse, do not re-implement.
2. *Proposal 1's own trust boundary defeats it.* Its design needs a source table bbox from
   `locate.py`, whose module docstring states at `locate.py:17-19` that "fully borderless
   (whitespace-only) tables have no geometric anchor and are out of scope", and `locate.py:131-139`
   records that stacked tables over-merge into one band. On a booktabs-heavy econ corpus that is a
   large, silent `UNVERIFIABLE` surface. Proposal 2's anchor needs no bbox at all.

Proposal 1 contributes the part that **must** be kept: its spanning-header falsifier, and the
insistence that inability to establish attribution is an explicit abstain, never a pass.

### The specification

New module `src/socr/tables/header_attribution.py`.

**Inputs: both.** The emitted block grid (markdown) **and** `page.get_text("words")`.

**Refactor first, then add.** Extract `header_repair.py:365-400` verbatim into

```python
def native_header_row(grid, words) -> list[str] | None:
    # rows_by_y   = _all_rows_by_y(words)                          header_repair.py:365
    # anchor_y    = _best_anchor_y(rows_by_y, grid)                :369   None -> abstain
    # local_ys / split_threshold / data_ys                         :374-378
    # lane_centers= _derive_lane_centers(rows_by_y, data_ys)       :379   <2   -> abstain
    # hdr_ys      = _header_ys(...)                                :386   []   -> abstain
    # rows = [_assign_words_to_lanes(rows_by_y[y], lane_centers, data_start_x) for y in hdr_ys]
    # return _merge_multiline_header_rows(rows)                    :402
```

and have `repair_collapsed_header` call it, so there is **one** geometry chain. The extraction must
**skip** the `if not collapsed or not words: return None` guard at `header_repair.py:361-363` —
attribution must run on tables whose widths already agree, which is exactly the destroyed-header case.

**Verdict, tri-state:**

- **`HARD`** (rejects) — some lane *i* has `native_header[i].strip()` non-empty (the page provably
  carries header words over that data lane) and **none** of that cell's whitespace-split tokens
  appears anywhere in `grid[0]`. Header content was **lost**, not merely moved. Shift- and
  skew-tolerant by construction: membership is tested against the whole emitted header row, never a
  column index.
- **`SOFT`** (records, does **not** reject) — tokens present in `grid[0]` but under a different
  column index than lane *i*. Evaluated only when `len(grid[0]) == len(lane_centers) + 1`. Emits an
  `AuditEvent` and nothing else. **See OPEN DECISION 1.**
- **`UNVERIFIABLE`** (abstains) — any abstain above. Emits `AuditEvent(kind="table_header_unverifiable")`,
  sets no gating flag, rejects nothing.

**Why this is sound, and why it is not a threshold:** `_best_anchor_y` (`header_repair.py:95-108`)
returns a `y` only when a native row's numeric multiset **exactly equals** an emitted data row's.
No fuzzy match, no tolerance. If the table cannot be located on the page that way, we do not judge it.

**No new constants.** Reuses `_LANE_X_TOL_PT` (`reconstruct.py:86`), `_LANE_SNAP_MULT` (:611),
`_SPLIT_GAP_MULT` (:601), `_SPLIT_GAP_MIN_PT` (:605), `_MIN_DATA_NUMERIC_CELLS`
(`header_repair.py:48`) — all already named and documented.

**Proposal 1's spanning-header falsifier, adjudicated.** A single merged `Results` cell spanning every
numeric lane with no leaf header row: under this design `_assign_words_to_lanes` puts that word into
*one* lane, so lanes 2..n have **empty** native header cells and `HARD` cannot fire. That is a
**miss**, not a false pass — the failure direction the corpus rule permits. It must be a shipped
negative-control test asserting "not `HARD`", and it must be named as a documented limit in the PR body.
Proposal 1's naive-intersection worry is real but applies to Gemini's original formulation, not to this one.

**The index/figure cases from the hand judgement are handled by abstain, deliberately.** A headerless
two-column index (Nagel p157, Woodford p799) has no header band; `_header_ys` returns `[]`
(`header_repair.py:198-243`) and the check abstains rather than buying a model call. This contains the
cost of the upstream table-detector defect; it does not fix it (see §7).

---

## 4. Where the winner-side check runs

**PRIMARY SITE — `src/socr/pipeline/agentic.py:508-615`, `NativeTableVerifierJudge.assess`.**
The only place a defect can still cause an **escalation** rather than a post-hoc demotion, because
this return value is what `route_page` reads to decide whether to try the next rung.

**Do not patch only the `EXACT_PASS` branch.** Three accept paths exist and all three must funnel
through one gate:

| line | path |
|---|---|
| `agentic.py:592` | `return self._inner.assess(...)` — `vr.warn` / AMBIGUOUS |
| `agentic.py:608-612` | the `EXACT_PASS` accept, `confidence=1.0`, inner judge never called |
| `agentic.py:615` | `return self._inner.assess(...)` — no issue detected |

Restructure `assess` so each produces a local `decision`, then:

```python
def _apply_structural_gate(self, decision, output, fitz_page, page_num, vr):
    if not decision.accept:
        return decision
    words = fitz_page.get_text("words") if fitz_page is not None else None
    defect = table_output_defect(output.text, words)
    if not defect:
        return decision
    self._emit_event(
        page_num=page_num,
        kind="table_structure_failed",
        engine=output.engine or "",
        detail=defect,
        data={"defect": defect, "verifier_state": vr.state},
    )
    return AcceptDecision(accept=False, reason=f"table_structure_failed: {defect}", confidence=0.0)
```

- Run it **after** `_maybe_repair_collapsed_headers` (`agentic.py:523`) so the deterministic repair
  gets its chance first: a repaired header is a pass, not an escalation.
- Reuse the `fitz_page` already fetched at `agentic.py:521`. No second PDF open.
- `is_table_page` short-circuits at `agentic.py:515-516`, so prose pages cost nothing.
- **Verifier exceptions** (`agentic.py:524-530`) currently delegate to the inner judge. Proposal 1 is
  right that this must not become a heuristic pass: on exception, set `words = None` and still run the
  grid-shape term (string-only, free), then delegate. Never accept a shape-defective candidate because
  geometry raised.

**SECOND SITE — `orchestrator.py:2536-2568`.** After `ps.best_output = decision.final_output`
(`:2534`), `repair_table_headers_on_page` mutates `ps.best_output.text` (`:2545-2547`). Re-run
**the grid-shape term only** (string-only, no geometry, free) immediately after that mutation. If it
now fires, **demote in place** — `PageStatus.WARNING`, `audit_passed=False`,
`FailureMode.NATIVE_TABLE_STRUCTURE_FAILED`, plus the audit event. Do **not** reroute: routing is
finished by that point. Guard against a double `AuditEvent` for one page, or the CLI counts double.

**THIRD — defence in depth.** `manifest.py:279-281` returns `p.best_output` immediately when
`p.best_output.audit_passed`. Add an assertion there that an accepted output carrying an unresolved
structural/header flag takes the fail-closed path instead of being frozen as success (proposal 1's
contribution; cheap, and it catches any future fourth mutation site).

**What happens on rejection is already built.** `route_page` advances to the next rung; on ladder
exhaustion `orchestrator.py:2593-2596` (`if not decision.accepted and self._page_has_tables(...)`)
sets `ps.native_table_structure_failed = True`, which is the D3 floor's first conjunct
(`manifest.py:311`). Verified.

---

## 5. `--native-only` — record and surface, never reroute

**Compute** in `BornDigitalDetector._assess_page_signals`, in the existing non-rotated/`has_tables`
branch where #200 already calls `structure_check.check_markdown`. The detector already holds the fitz
page, so `page.get_text("words")` is free. Add a second `PageAssessment` field beside #200's
`native_table_structure_defective`:

```python
native_table_header_unattributed: bool = False  # HARD verdict only
```

Stamped once at the moment of the evidence, never re-derived downstream — the discipline the existing
docstring already states.

**Propagate** in `DocumentState.apply_born_digital` (`state.py:175-191`), beside the existing
`native_table_unverifiable` propagation at `state.py:190-191`.

**MUST NOT enter `needs_repair`.** #200's `state.py` carries an explicit comment that
`native_table_structure_defective` is deliberately absent from that property, because including it
would force a repair pass under `--native-only`. The header flag is the same class, gets the same
treatment and the same comment. Settled; not relitigated here.

**Surface at all four levels** (the no-silent-loss rule), riding the plumbing #200 already wired for
the sibling flag:

| level | site (main line numbers) |
|---|---|
| page status | `orchestrator.py:911-918` (`prose_pages` ship) and `:2447-2456` (`elif is_native:` Tier-1 ship) — both currently hardcode `PageStatus.SUCCESS, audit_passed=True`. Take the flag as a second disjunct of the `defective` local: `WARNING`, `audit_passed=False`, `FailureMode.NATIVE_TABLE_STRUCTURE_FAILED`. |
| document status | the `native_fallback_pages` list around `orchestrator.py:4674-4690`; document → `AUDIT_FAILED` |
| metadata | sidecar write `orchestrator.py:4282-4287`, restore `:4471-4475`, so the verdict survives resume |
| CLI | run summary + `tables_trust.json`. `TABLE_DISTRUST_KINDS` already contains `table_structure_failed` (#200), so a header defect surfaces with **no new kind**. |

**Verified, and this is #211 exactly:** `_is_trusted_native_without_ocr` (`orchestrator.py:1186-1210`)
returns `True` for **table** pages under `--native-only` (`:1204-1209`), so those pages land in
`prose_pages` (`orchestrator.py:793`) and ship at `PageStatus.SUCCESS, audit_passed=True`
(`orchestrator.py:911-918`). That is the one default-reachable path on which a table like the four
hand-judged ones ships labelled clean.

**Net effect:** zero additional attempts, zero model calls, `len(ps.attempts)` unchanged, page ships
its native text flagged. Assert that explicitly.

---

## 6. Fail-closed at the top rung

Extend the **existing** D3 floor. Do not invent a second one.

Today `manifest.py:310-314` requires

```python
p.native_table_structure_failed
and getattr(p, "native_table_unverifiable", False)
and bool(p.attempts)
```

and then ships marker + PNG ref, `PageStatus.ERROR`, `audit_passed=False`,
`FailureMode.NATIVE_TABLE_STRUCTURE_FAILED` (`manifest.py:315-333`).

**A header-only defect satisfies the first conjunct** (set at `orchestrator.py:2593-2596` on ladder
exhaustion) **but not the second** — TR-3 is blind to header loss by construction — so today it falls
through to `native_is_fallback` at `manifest.py:340-350` and **ships the native table text at
`WARNING`**. Verified by reading both branches. That is precisely the plausible-but-wrong artifact the
D3 panel verdict exists to refuse.

**Change, `manifest.py:310-314`:**

```python
if (
    p.native_table_structure_failed
    and (
        getattr(p, "native_table_unverifiable", False)
        or getattr(p, "native_table_header_unattributed", False)
    )
    and bool(p.attempts)
):
```

Everything downstream unchanged: same marker `[page N failed: unverifiable table — see image]`, same
`d3_floor_png_ref`, same `ERROR` / `audit_passed=False` / `NATIVE_TABLE_STRUCTURE_FAILED`.

Two consequential widenings that must land in the same commit or the floor degrades:

- `orchestrator.py:2615-2626` renders the floor PNG only `if getattr(ps, "native_table_unverifiable", False)`.
  It must render on the header flag too, else the floor is a bare marker with no image — still
  fail-closed (per the `manifest.py:315-322` comment) but a human then cannot see the table.
- `orchestrator.py:4674-4690` splits D3-floor pages from `native_fallback_pages` on the **same
  conjunction**. Widen it identically or a header-floor page is double-counted in both lists.

**So the top-rung answer is: no table text ships at all.** Explicit marker + full-page PNG,
`PageStatus.ERROR`, `audit_passed=False`, counted at document level, greppable. A missing number,
loudly. If PNG rendering fails, the marker alone ships — never plausible table text.

---

## 7. OPEN DECISIONS — for the owner, not invented closure

### OPEN DECISION 1 (BLOCKING, pre-merge) — is `SOFT` where the damage actually lives?

The two proposals disagree and **the code cannot settle it.**

- Proposal 1 makes "token present but in the wrong emitted column" a **`FAIL`**.
- Proposal 2 makes it **`SOFT`** (record only), because establishing the column map soundly needs
  machinery it declined to build.

The hand judgement records "header band destroyed / detached from columns" on **4/4** pages
(`2026-08-15_tr3-hand-judgement.md:52`) but does **not** distinguish *lost* from *mis-columned*. If
those four are mostly mis-columned, the shipped `HARD` term catches almost none of the defect it was
added for, and the third term is theatre.

**The test, and it must run before the PR is called done:** re-open the four damaged pages already
staged outside the repo at `~/.local/share/socr/tr3-judge/`
(`2026-08-15_tr3-hand-judgement.md:20-25`) and classify each: header tokens **absent** from the
emitted table block, or **present but mis-columned**. Four pages, no new tooling, no new corpus access.

- ≤2 mis-columned → ship as specified, `SOFT` stays advisory.
- ≥3 mis-columned → `SOFT` is promoted to a reject term, the `len(grid[0]) == len(lane_centers) + 1`
  precondition becomes load-bearing, and §3 must be revised before merge.

Promoting `SOFT` without this measurement would be a threshold invented from nothing. Leaving it
advisory without the measurement would be shipping an unmeasured claim of coverage. Either way the
four pages must be looked at.

### OPEN DECISION 2 (not blocking) — the table detector is upstream and still wrong

3 of 7 TR-3 firings were not tables (`2026-08-15_tr3-hand-judgement.md:69-83`): two book indexes and a
figure. This spec **abstains** on them rather than escalating, which contains the cost but does not fix
the defect. Open question 3 of the hand judgement stands and should stay a **separate ticket**, not be
folded in here.

---

## 8. Risks (carry these into the PR body)

1. **The firing rate on model output is completely unmeasured.** Every number in
   `2026-08-14_gh151-b1-predicate-design.md` (26.9% of pages, the 35/66 and 27-page non-overlap) was
   measured on **native** markdown. The winner-side gate runs on **VLM** markdown — a different
   population with different failure shapes. Before merge, run `table_output_defect` over the accepted
   outputs of an existing agentic corpus run and report the rate. If it exceeds the native rate, the
   ladder is being paid for repeatedly.
2. **Lane derivation inherits the #206/#207 notation gap, and I confirmed it by running the regex.**
   `_derive_lane_centers` (`header_repair.py:246-268`) counts numeric cells via `_NUM_TOKEN_RE`
   (`reconstruct.py:78`). Measured: `.034` → `False`, `***` → `False`, `∗∗` (U+2217) → `False`. A two-
   or three-column table typeset in leading-decimal notation derives fewer than 2 lanes and **silently
   abstains** — on exactly the econometrics notation this corpus is full of. The check could be
   near-inert and look clean. Hence: `table_header_unverifiable` must be **counted and reported**, and
   its firing rate — not its precision — is the number to watch.
3. **Widening the D3 floor increases the number of pages that ship no table text.** Correct trade for
   this corpus, but it shows up as more `ERROR` pages in run summaries. Say so in the PR body so it is
   not mistaken for a new bug.
4. **The hand judgement is a selected sample.** It supports "when a native table page fires TR-3, the
   table is broken". It does **not** measure a defect base rate over all 245 native table pages
   (`2026-08-15_tr3-hand-judgement.md:113-122`). Nothing in this spec may be justified by an unmeasured
   base rate.
5. **The spanning-header case is a documented miss**, not a caught defect (§3). Do not claim otherwise.
6. **#200 is 7 commits behind main.** Rebase before implementing; blindly merging the old branch risks
   reverting intervening table-integrity work.

---

## 9. Acceptance tests (all hermetic — no ollama, no provider, no corpus PDF)

Fixtures follow the existing in-process synthetic-fitz pattern in
`tests/test_header_repair.py` (`fitz.open()` + `page.insert_text(...)`) — generated look-alikes only.

**Header attribution (`tests/test_header_attribution.py`)**

1. `test_missing_header_band_is_hard` — header words present natively over every data lane; candidate's
   header row all-blank. `header_attribution` → `HARD`; `table_output_defect` → `"header_unattributed"`.
2. `test_spacer_column_without_native_header_is_not_hard` — one data lane has **no** word above it and
   the candidate's header cell for that lane is empty. Verdict **not** `HARD`. Gemini's named falsifier,
   shipped as a negative control.
3. `test_spanning_merged_header_is_not_hard` — one merged non-empty cell over all lanes, no leaf header
   row, all body numbers correct. Verdict **not** `HARD`. Pins the documented miss in §3.
4. `test_headerless_two_column_index_abstains` — short label lines with a right-hand run of page numbers,
   no header band (the Nagel p157 / Woodford p799 shape). `_header_ys` → `[]`, verdict `UNVERIFIABLE`,
   `table_output_defect` → `""`, no reject.
5. `test_misplaced_header_token_is_soft_not_hard` — tokens present in `grid[0]` but shifted one column
   against the geometry lanes. Verdict `SOFT`, `table_output_defect` → `""`, audit event recorded.
   Pins the documented limit of the reject term. **Revise if OPEN DECISION 1 resolves ≥3.**
6. `test_leading_decimal_table_abstains_and_is_counted` — a table typeset `.034` throughout. Assert
   `UNVERIFIABLE` **and** that the `table_header_unverifiable` event is emitted (risk 2 made visible).

**Winner-side gate (`tests/test_agentic.py`)**

7. `test_exact_pass_with_destroyed_header_is_rejected` — construct `NativeTableVerifierJudge` directly
   (no `process()`, no ladder) with a stub inner judge that always returns `accept=True`, a
   `get_fitz_page` returning the synthetic page, `is_table_page` → `True`, and a `PageOutput` that is
   numerically perfect and header-destroyed. Assert the verifier reaches `EXACT_PASS`
   (`native_verifier.py:1054-1058`) **and** that `assess` returns `accept=False` with reason starting
   `"table_structure_failed"`. The direct regression test for the `agentic.py:595-612` hole.
8. `test_structural_gate_covers_all_three_accept_paths` — parametrised over `vr.warn` / `EXACT_PASS` /
   no-issue (`agentic.py:592`, `:608`, `:615`), same stub-accepting inner judge, same defective text;
   all three return `accept=False`. Guards against patching only the `EXACT_PASS` branch.
9. `test_verifier_exception_still_runs_shape_term` — geometry raises; a `ragged` candidate must still
   be rejected, not accepted by delegation.
10. **Disjunction controls** — (a) `ragged`/`detached_label_rows` candidate with a clean numeric
    multiset must reject; (b) rectangular candidate with a per-row multiset mismatch but intact headers
    must reject. Proves neither term subsumes the other.

**Pipeline (`tests/test_structural_gate_b1_gh151.py`)**

11. `test_native_only_records_header_defect_without_rerouting` — `process()` on a generated born-digital
    table PDF with `native_only`. **MUST patch `_available_engines_for_agentic` to return
    `[PROFILE_QWEN_LOCAL]`** per the CI rule, even though the ladder should never be consulted. Assert:
    page `WARNING`, `audit_passed=False`, `failure_mode=NATIVE_TABLE_STRUCTURE_FAILED`, an `AuditEvent`
    of kind `table_structure_failed`, page present in the document-level native-fallback list,
    `tables_trust` marks it distrusted, document status `AUDIT_FAILED`, and `len(ps.attempts) == 1`
    (proof of no reroute).
12. `test_post_route_header_repair_recheck` — an accepted output that the GH-56 repair at
    `orchestrator.py:2536-2568` mutates into a ragged grid; page demoted in place (`WARNING`,
    `audit_passed=False`) and **exactly one** audit event recorded for the page.

**Fail-closed (`tests/test_manifest_agentic.py`)**

13. `test_d3_floor_fires_on_header_defect` — `PageState` with `native_table_structure_failed=True`,
    `native_table_unverifiable=False`, `native_table_header_unattributed=True`, non-empty `attempts`.
    `_winning_page_output` returns the marker, `PageStatus.ERROR`, `audit_passed=False` — no table text
    ships. Paired negative: with the header flag `False`, the existing `WARNING` native-fallback
    behaviour at `manifest.py:340-350` is unchanged.
14. `test_d3_floor_without_png_ships_marker_alone` — forced render failure still fail-closed.

**Regression guards**

15. Sidecar round-trip: `native_table_header_unattributed` survives write
    (`orchestrator.py:4282-4287`) and restore (`:4471-4475`).
16. The existing golden `_rewrite_all_fragments` / whole-doc byte-identity tests stay green with the
    gate present and not firing.
17. `~/venvs/socr/bin/pytest tests/ -q` full suite green, and the blocking lint gate run exactly as CI
    does: `uvx ruff@0.16.0 format --check .` (**not** `~/venvs/socr/bin/ruff`).

---

## 10. Implementation order

0. Rebase `feat/151-b1-structural-gate` onto current `main`; land #200's plumbing unchanged (§0).
1. Run OPEN DECISION 1's four-page classification. Revise §3 if it resolves ≥3.
2. Extract `native_header_row` from `repair_collapsed_header`; prove `repair_collapsed_header` behaviour
   is unchanged by the existing `tests/test_header_repair.py`.
3. Add `header_attribution.py` + `table_output_defect`; tests 1-6, 10.
4. Wire the winner-side gate at `agentic.py:508-615`; tests 7-9.
5. Wire the post-route recheck at `orchestrator.py:2536-2568` and the `manifest.py:279-281` assertion;
   test 12.
6. Wire the native-side flag + the four surfaces; test 11.
7. Widen the D3 floor (`manifest.py:310-314`), the PNG trigger (`orchestrator.py:2615-2626`), and the
   double-count exclusion (`orchestrator.py:4674-4690`); tests 13-14.
8. Measure the VLM-side firing rate (risk 1). Full suite + format gate. Wait for CI green before merge.

Serially, in the one checkout. No worktree — the editable install resolves to the main tree and a
worktree produces false green.
