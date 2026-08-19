# 2026-08-18 — Corpus triage of the 64 open issues

Adjudication of a two-model (Grok / Kimi) triage against
`/tmp/socr-triage/corpus-profile.md`: born-digital economics papers with a
scanned minority, regression tables as the payload, equations secondary,
figures nice-to-have but never silently lost. Verdicts were re-derived from
source at `1132b70`; every tie in the dossier was broken by reading the code.

Buckets: **5 BLOCKS · 15 DEGRADES · 9 ALREADY_FIXED · 35 POSTPONE**.

---

## BLOCKS — fix before the corpus run

| # | Defect | What it does to an econ paper | Size |
|---|---|---|---|
| **163** | `verify_scanned_table` defers the fail-closed raster gate whenever *any* non-empty word exists on the page, with no trust check on that text layer (`src/socr/tables/source_evidence.py:77-84,308-314`). | A JSTOR-era scan with a baked-in corrupt OCR layer skips source-evidence verification entirely and falls through to the native verifier, which compares the model's table against the same junk. A **hallucinated** coefficient table ships as SUCCESS. Worst failure class for a citation corpus. | S/M |
| **212** | `EXACT_PASS` returns `accept=True, confidence=1.0` with no header-attribution term; the only gate on that path is grid *shape* (`src/socr/pipeline/agentic.py:603-625`; `src/socr/tables/structure_check.py:234-268`). | The measured 4-of-4 defect: every numeral correct, header band destroyed, so coefficients cannot be attributed to a column. Ships SUCCESS on the exact booktabs shape this corpus is made of. Broadest exposure of anything here. | L |
| **225** | `strip_phantom_images` returns `False` (never strip) for any `http(s)://` or `data:` image path (`src/socr/core/normalizer.py:227-228`), and no URL validation exists anywhere in the pipeline. | A Qwen-invented `imgur` link ships inside page markdown under SUCCESS with zero audit signal. Fabricated content presented as extracted content. | S |
| **150** | TICKET-C1 (self-described "the actual production fix") is still TODO: `rowize_from_words_chart_aware` — the only placeholder emitter — runs only `if not table_regions` (`src/socr/core/born_digital.py:1328-1350`). | A mixed chart + table page produces a `find_tables()` hit on the axis ticks, so the chart placeholder is never emitted. Figure vanishes from the markdown with no marker. Profile explicitly forbids silent figure loss. | M |
| **189** | `_render_chart_region_pngs` sits inside the `if not decision.accepted and self._page_has_tables` branch (`src/socr/pipeline/orchestrator.py:2833-2847`). | Same silent figure loss on the accept side: when a ladder rung is accepted on a mixed chart+table page, the chart PNG is never rendered — no page status, no audit event, no CLI trace. Fix alongside #150. | S/M |

Fix order: **163 → 212 → 225 → 150 → 189.** #163 is cheapest and has the worst
per-occurrence failure; #212 has the widest blast radius but needs a sound
predicate first (see #215 below — do not simply unpark the old one).

---

## DEGRADES — ship, but know these

| # | What degrades | Workaround / warning for the owner |
|---|---|---|
| 127 | `detect_native_structure_loss` is defined at `born_digital.py:301` with **no caller anywhere in `src/`**. Native prose pages flatten headings and lists and nothing records it. | Text tokens all survive (DOIs, reference strings included). Markdown structure for downstream search is degraded, silently. Accept for this week. |
| 223 | Same class on the VLM lane: headings emitted as plain lines, same absent consumer. | Same — markup, not content. |
| 160 | `_resolve_table_escalation_provider(available)` is handed the **tier-filtered but not cost-filtered** list; `max_cost_per_page` is applied only when building `ladder` (`orchestrator.py:1688-1704,2322-2332`). The docstring's claim that `--max-cost-per-page` suppresses the lane "for free" is false. | Default `max_cost_per_page=0` (no cap) so it does not fire on a default run. **Do not rely on `--max-cost-per-page` as a spend fence — use `--strict-local`.** |
| 161 | Resume ledger skips a page iff `status == SUCCESS` (`orchestrator.py:4692-4693`); an all-reject best-effort page carries SUCCESS with `audit_passed=False` (`manifest.py:462`), so it is treated as terminal. | The document still demotes to AUDIT_FAILED on resume (flags are restored at `orchestrator.py:4718+`), so it is loud. **But `--reprocess` will not actually re-OCR that page** — delete the `pages/NNN.json` sidecar by hand. |
| 162 | Both verifier judges catch every exception and delegate to an accepting inner judge (`agentic.py:418-424,526-537`). | Every accept path, exception included, now passes through `_apply_structural_gate` (shape-only at minimum) and a raise is logged. Residual fail-open needs a rare crash on a well-formed grid. |
| 166 | Non-timeout crop failures drop with no sentinel; `dualpass_crop_timeout` is absent from `TABLE_DISTRUST_KINDS` (`tables/extract.py:246-248,309-312`; `core/tables_trust.py:49-89`). | `tables_trust.json` can report *trusted* after incomplete verification. `auto_patch_tables` is default False so no text is rewritten. **Treat tables_trust as advisory, not as a clearance.** |
| 172 | Soft timeouts abandon non-daemon `ThreadPoolExecutor` workers (`agentic.py:213-246`); the stale comment claims daemon threads. | A wedged VLM can keep the CLI alive after timeout/halt. Visible hang, not corrupt output. Ctrl-C and resume. |
| 177 | `socr process` exits 0 on AUDIT_FAILED (`cli.py:453-463`), while batch exits 1. | **Do not gate a corpus script on the single-file exit code** — read `status` from the metadata instead. |
| 198 | `_destroyed_numeric_tokens` filters candidates with the raw anchored `_NUM_TOKEN_RE` (`reconstruct.py:78,377`), blind to `0.67***`, U+2212 minus, and `£43.2`; the presentation-stripping `is_numeric_token` exists but is not used there. | A split decorated coefficient escapes the GH-144 rejection. **Not** on the default path: `_is_trusted_native_without_ocr` returns `not self._page_has_tables(...)` (`orchestrator.py:1395`), so table pages leave the native lane. Reachable only via `--native-only` or flagged native fallback. Do not use `--native-only` on this corpus. |
| 217 | `count_digit_corruption` is slash-digit only (`born_digital.py:216-227`), so a 2-for-minus forgery is invisible to the hygiene gates. | The B1 structural gate fires on the measured page and `native_table_structure_defective` joins the demotion union, so the corrupt text ships as a loud WARNING. Residual silent hole is prose negatives. Scale: 8 pages in 1 of 40 probe papers. |
| 219 | `_MATH_FONT_RE` excludes PazoMath / URWPalladioL (`born_digital.py:35-41`). | `mathpazo` display math flattens under SUCCESS. Equations are secondary and the output is visibly mangled, not plausibly wrong. |
| 221 | The cascade-halt probe is `GET /api/tags` (`tables/extract.py:77-91`), which returns 200 on a wedged GPU, so the halt latch at `orchestrator.py:2894-2897` never fires. | **The wedge protection does not work.** Every remaining page burns the full provider timeout. Failures are loud and resumable, but a wedged GPU costs hours of wall clock on a corpus run — watch the run, do not fire-and-forget. |
| 222 | `probe_ollama_idle()` is called with its `localhost:11434` default regardless of the actual backend (`orchestrator.py:2894-2895`). | Inverse of #221: running against HPC/vLLM with **no local ollama daemon**, one timeout false-halts the rest of the document as `PARTIAL_SAVE_VLM_TIMEOUT`. Loud and resumable, but it truncates. Keep a local daemon reachable, or expect truncation. |
| 227 | `any("timeout" in (att.reason or "") for att in decision.attempts)` spans *superseded* attempts (`orchestrator.py:2894`). | A page that timed out on rung 1 and then succeeded on rung 2 still arms the halt. Latent behind #221 on a local setup; **composes badly with #222 on HPC**. One-line fix: inspect only the winning attempt. |
| 226 | Residual collapse cases: consistent-pipe-count collapse and no residual-LaTeX guard. | The observed 3-cell `\multicolumn` vs 8-column body case is now caught by `header_repair` (`tables/header_repair.py:51-77`) plus the chain-wide structural gate. Residual is narrower and partially surfaced. |

---

## POSTPONE

39 (design epic, no live defect) · 49 (proposal; the actionable core shipped in
`300c31b`) · 56 (Consensus-Economics corpus, not this one) · 64 (warning-only
audit for 2-column borderless tables; values char-exact) · 114 (unbuilt CLI
ergonomics) · 139 (`--no-audit` inert on agentic — verified fail-safe: all
`audit_enabled` gates sit in the non-agentic branch, so you get *more* auditing
than asked) · 140 (native math flattening; equations secondary) · 142 (flag
inventory chore) · 152 (side-by-side merge lives in the native rowizer; table
pages leave that lane) · 154 (qwen-cloud $0.00 defeats a cost cap, but the rung
executes the local backend) · 155 (orchestrator split; refactor) · 156 (docs
hygiene) · 157 (equation skip behind default-off flags) · 159 (qwen-cloud rung
re-runs local qwen; provenance, not content) · 164 (equation-recovery
duplication, opt-in) · 165 (PUA math detection, opt-in) · 167
(`has_chart_marks` over-fires; prose retained plus a loud PNG) · 168
(`--config`/`--profile` load path, not CLI runs) · 169 (`skip_reason` audit gap)
· 170 (replay ignores figure assets) · 171 (sidecar `figure_refs` provenance;
the shipped `.md` is correct) · 174 (legacy quarantine) · 175 (import layering)
· 176 (`DocumentState.text` hygiene) · 178 (ADR) · 181 (recursive `find()`;
worst measured page is 932 drawings, under the edge) · 190 (all-empty-grid hole
latent; offending model builds removed) · 195 (surfacing-only; the GH-144
rejection is lossless) · **197** (`numeric_scope is None` falls back to
`fitz.Rect(table.bbox)` at `reconstruct.py:169`, but the failure direction is a
*logged false reject* into the lossless word-geometry rowizer — content is
preserved) · 202 / 203 (Mistral bake-off, off the default ladder) · 213 (book
back-matter, absent from this corpus) · **215** (the header-attribution reject
term is parked *deliberately*: four documented implementations returned HARD on
byte-perfect tables carrying star and `n.a.` rows — unparking it this week would
reject correct output. The live ship hole is #212; fixing it needs a new sound
predicate, not the old one) · 235 (`page_fingerprint` `ensure_ascii` divergence;
write-only metadata) · 238 (caption engine identity; `describe_figures`
default-off).

## ALREADY_FIXED

144 (`_destroyed_numeric_tokens` + lossless rowizer, `d645b24`) · 146 (empty
header instead of promoting a data row, plus header-band recovery, `e1d5d91`) ·
147 (rotated table pages refuse the native lane and route to OCR,
`born_digital.py:1040-1054`, `13033a3`) · 151 (`structural_gate_fires` — ragged
or detached-label — on every accepting path, `structure_check.py:210-223`,
`11093ca`) · **158** (`_engine_determinants` folds resolved model/backend/task
for primary *and* every secondary into `_run_fingerprint`, plus per-attempt
`provider_id`/`model`/`backend`; `orchestrator.py:292-380,2731-2737`) · 173
(the four output-changing knobs are lane-gated into the fingerprint,
`orchestrator.py:420-466`, `2d47d8f`) · **205** (TR-3 surfaced by audit event on
every verdict branch and `_apply_structural_gate` runs on every accepting path,
`agentic.py:539-625`; hand-judgement shipped as `62066d1`) · 220 (`socr review`
+ `src/socr/review/html.py`, `1216f2e` / PR #224) · **231** (Grok's POSTPONE was
stale — the fix is committed on this branch as `1132b70`;
`QwenEngine.resolved_model_version` now returns the resolved model, not the
config sentinel).

---

## Confidence

**Verified by reading source at `1132b70`:** every BLOCKS entry, every tie-break
(127, 158, 160, 161, 197, 198, 205, 215, 221, 222, 223, 227, 231), and eight
spot-checks of unanimous calls — 139, 144, 147, 151, 173, 220, 225, plus the
`_is_trusted_native_without_ocr` routing claim at `orchestrator.py:1395` that
three POSTPONE verdicts (152, 198, 217) all lean on. **No spot-check failed; no
re-bucketing was required from them.** The only verdict that moved against both
models' original position is #231, and only because the fix landed as a commit
between the two triage passes.

**Not verified — carried on the models' evidence:**

- **#217's scale claim** (8 pages in 1 of 40 probe papers) comes from the issue's
  own measurement table; not re-measured here.
- **#181's "worst measured page is 932 drawings"** is an inherited measurement.
  If a corpus paper carries a denser vector figure, the recursive `find()` can
  hit `RecursionError`, which is swallowed to `False`.
- **#190's latency** rests on the claim that the offending `qwen3.5:27b` builds
  were removed. Unconfirmed against the local model store.
- **#215's four failed predicate implementations** are documented in the
  `structure_check.py:234-268` docstring and the referenced design logs; the
  failures themselves were not reproduced. This matters: it is the reason #212
  is sized **L** rather than a one-line unpark.
- **#163's exploitability** is confirmed at the code level (the defer is
  unconditional on any non-empty word). Whether a *specific* scanned paper in
  the corpus carries a junk text layer bad enough to pass the native verifier
  was not tested on real PDFs — that test is worth running before the batch.
