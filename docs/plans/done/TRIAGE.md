# Issue triage

Done 2026-10-08 against the rule in `GOAL.md`. Nothing has been closed or labelled yet.

**Rule.** KEEP only if the issue names a real paper and page where a wrong or missing output
was actually seen. Everything else is PARK: no named page, a failure seen only in a synthetic
test or a code review, or a proposal, measurement, docs or test ticket.

**Protected.** #749, #826 and #962 are never closed or labelled, whatever their verdict.
#962 is an open pull request (`cursor/typesafe-table-gate-0f59`), not an issue, so it has no
row below.

**Counts.** 103 open issues: 31 KEEP, 72 PARK.

**How.** Four Sonnet readers read every issue body and comment. Each KEEP carries a verbatim
quote naming the paper or page, checked mechanically against that issue's own text. PARKs whose
text mentions a page number were spot-checked by hand.

## KEEP — grouped by paper

| # | Title | Verdict | Reason |
|---|---|---|---|
| #917 | bug(tables): rotated native-first SHIP absorbs margin running heads into cells and splits dates (Barrot p15 ships SUCCESS) | KEEP | Barrot-Sauvagnat p15: rotated SHIP glues margin running head into cells, splits dates |
| #151 | bug(tables): a table can ship at 100% word recall with its structure destroyed — recall is not a sufficient gate | KEEP | Bauer-Pflueger-Sunderam p26 Table 3: 100% word recall but grid structure destroyed, shipped as success. |
| #953 | bug(tables): beckmann p79 ships a truncated grid with geometry present (not the #949 hole) | KEEP | beckmann p79 ships only ~7 of 39 table lines in the grid, rest loose lines |
| #317 | arch: the native lane ships unwitnessed — no model ever sees a trusted-native page | KEEP | Bianchi & Sosa-Padilla p29, p60, p15: tables torn, headers flattened, equations shredded, shipped native |
| #945 | bug(tables): header_band_missing still silent on residual Boukus 39 | KEEP | Boukus p39: header content lost on shipped grid and header_band_missing does not fire. |
| #219 | bug(equations): Palatino/Pazo math fonts miss `_MATH_FONT_RE`, so display equations ship flattened with no flag and no recovery | KEEP | BHL_2026 p6: display equations flattened byte-identical from native text, reported Success |
| #988 | bug(tables): table_truncated row-shortfall counts SE/t-stat rows on the native side but not the candidate side | KEEP | Coca-Cola 2021 p74: row-shortfall counts furniture bands, complete table flagged truncated |
| #1044 | bug(tables): catch invented / wrongly-bound values that #1042 now lets through | KEEP | Coca-Cola 2018 p60: invented value passes the gate; 2019 p62 and p59 also wrong. |
| #1045 | bug(tables): catch column / structure shortfalls that #1042 now lets through | KEEP | Coca-Cola 2020 p63, 2019 p51, 2020 p76: incomplete answers pass after #1042 |
| #331 | bug(tables): native reconstruction loses row labels at scale — the stub column is dropped, and it poisons every drift comparison | KEEP | Cochrane-Piazzesi p12 Table 3: every row label dropped from native table reconstruction. |
| #56 | CE OCR is not solved: prioritize reliable tables and figures | KEEP | CE 202401 p4: qwen table ragged, native fallback ships fully collapsed table; p24 audit failed |
| #957 | bug(tables): word_split_across_cells abstains on multi-line wrap (cook_kazinnik p21 still ships) | KEEP | cook_kazinnik p21: true mid-word split across cells still ships, predicate abstains on multi-line wrap. |
| #639 | bug(tables): detected_table_bboxes does not cover the numeric table body on ECB annex pages | KEEP | ECB bulletin p127-129 fixture pages: detected table bbox covers only 2-5% of numeric rows |
| #696 | measure(tables): spanning column headers lose the whole table on 9 of 30 ECB census pages | KEEP | ECB 2018 blssurvey p37-39: spanning-header tables replaced by unverifiable-table marker |
| #223 | bug(structure): heading loss is not native-lane-specific — the VLM lane ships 36 unrepresented headings | KEEP | EFO Nov 2022 p42: box title heading emitted as plain line by qwen lane. |
| #749 | bug(fail-closed): a born-digital two-column chart page is failed entirely — 2,777 chars of prose discarded over a table the pipeline itself calls not_scorable | KEEP (protected) | Fed MPR March 2011 PDF p32 and p51: whole page failed to 102 B, prose lost. |
| #804 | bug(figures): has_chart_marks False on a page with four line charts and 1,935 vector drawings | KEEP | Fed MPR March 2011 PDF p32: has_chart_marks False despite four line charts and 1,935 drawings. |
| #826 | Fail-closed floor treats an absent candidate as a refused one, discarding pages on timeout | KEEP (protected) | MPR March 2011 p32: page discarded to marker on VLM timeout under load |
| #734 | bug(charts): on the Fed SEP projections PDF the five dot-plot panels detect as ONE chart region with no vector frame found, and the model's own counts ship flagged instead of the reader's | KEEP | Fed SEP Dec 2020 p9: filled dot-plot grid shipped model counts contradicting geometry; March 2022 undercounts. |
| #737 | proposal(charts): expected per-series totals belong in verify_panel's caller hook, never in reconciliation | KEEP | Fed SEP sep-20220316 p09: model undercounts dot-plot series, sums 14/14/15 instead of 16. |
| #635 | proposal(figures): read chart crops into data tables, gated by a caller-supplied plausibility check | KEEP | FOMC Minutes Sept 2018 p20: dot plot chart tables ship with every data row empty. |
| #994 | bug(tables): a table the detector misses is flattened to prose and the page ships SUCCESS | KEEP | Forsythe p8, Barrot p29, bybee p30: ruled tables flattened to prose, page ships SUCCESS |
| #1043 | bug(floor): a table-only rejection floors the whole page and drops correct prose | KEEP | Forsythe-Lundholm 1990 p25: table-only rejection floored page, correct prose dropped. |
| #152 | bug(tables): two side-by-side tables are merged into one region and flattened | KEEP | Haim p31: side-by-side tables A5/A6 merged and flattened, 54 words missing |
| #150 | bug(figures): figures are extracted as tables — the two worst pages in the corpus are charts | KEEP | Heston-Korajczyk-Sadka 2010 p10: chart axis ticks shipped as a table, figure content lost. |
| #49 | General extraction method: single-pass VLM + free native verification + agentic-on-signal | KEEP | Nakamura & Steinsson p42: rowizer loses 49 of 152 values, shipped SUCCESS |
| #144 | bug(tables): word-geometry rowizer drops numeric values — 49 of 152 on one page, shipped as SUCCESS | KEEP | Nakamura & Steinsson p42: rowizer drops 49 of 152 numeric values, ships SUCCESS |
| #146 | bug(tables): first data row is emitted as the table header, and the real header band is excluded | KEEP | Nakamura & Steinsson p13 Table I: first data row emitted as header, real header lost |
| #901 | bug(tables): rotated table pages ship no table: verification assumes upright geometry (11/11 sampled) | KEEP | Nakamura-Steinsson 2018 p42: rotated table withheld, correct candidate rejected by value guard. |
| #215 | bug(tables): header-attribution reject term is parked — destroyed header bands ship undetected | KEEP | Romer&Romer p21 and Menzly-Ozbas p20: destroyed header band in shipped tables |
| #213 | bug(tables): book indexes are routed to table reconstruction | KEEP | Woodford p798: book back-matter index routed into table reconstruction. |

## PARK

| # | Title | Verdict | Reason |
|---|---|---|---|
| #39 | P1: route engines by measured quality-per-dollar (benchmark fix, calibrated per-page-type ladders, uncapped escalation, gate calibration) | PARK | Routing proposal and measurement plan; no failing real page named. |
| #114 | proposal: socr escalate — post-hoc escalation pass so local GPU and cloud egress can be different machines | PARK | Proposal for a post-hoc escalate command; no failing page named. |
| #127 | feat(born-digital): native path discards heading, emphasis, list, and link structure | PARK | Feature request; corpus-level sample counts, no specific document page with observed loss. |
| #155 | chore(arch): split pipeline/orchestrator.py god-module (~5.5k LOC) | PARK | Refactor/chore of orchestrator module; no failing page. |
| #165 | bug(equations): PUA-only math pages skip recovery routing and can suppress unrecovered audits | PARK | Code-reading finding on PUA math detection; no real page named. |
| #172 | bug(routing): soft timeouts abandon non-daemon workers that can prevent CLI exit | PARK | Hang risk shown by stub tests; no real document page named. |
| #176 | chore(arch): restore dumb DocumentState blackboard + one authoritative page-text selector | PARK | Architecture refactor chore; no observed failing page. |
| #202 | proposal: measure Mistral OCR 4.1 before any routing change | PARK | Measurement proposal; no observed failing page. |
| #203 | proposal: consume Mistral OCR 4 block labels (blocked) | PARK | Blocked design placeholder for Mistral block labels; no failing page. |
| #220 | feat(review): side-by-side page-image ↔ extracted-markdown viewer for hand judgement | PARK | Feature request for a review viewer; no failing page. |
| #291 | chore(naming): retire the d3_ prefix — it names a design-menu option, not a behaviour | PARK | Rename chore; no failing page. |
| #328 | chore(measure): firing-rate validation of structure-restore across the corpus | PARK | Measurement chore on firing rates; no specific document page named. |
| #338 | docs(log): #333 measurement on main is not a decision of record | PARK | Docs/measurement-log critique; Nakamura p42 named only as fixture, no output failure seen. |
| #343 | bug(tests): GH-331 suite pins stay green when the production line is reverted | PARK | Test-pin ticket; reverted-line guards, no observed failing page. |
| #393 | feat(tables): deterministic rotation-frame lift for binding-clamped tables | PARK | Design proposal for rotation lift; no real page with observed failure named. |
| #471 | test(born_digital): extract_structured pin for GH-341 span-path geometry bind (deferred from #465) | PARK | Test-only pin deferred; fixtures are synthetic. |
| #496 | bug(audit): a re-run without the figure phase erases the sidecar's figure metadata | PARK | Synthetic one-page reproducer; premise later shown invalid. |
| #548 | arch(pipeline): DEMOTED_NATIVE exit — assign each demotion trigger to N or F (#544 leftover) | PARK | Architecture ticket with corpus-level trigger counts; no specific page named. |
| #603 | bug(tables): math-font header row keeps only the PUA subscripts as its native label — needs the math-font lane, not the rowizer | PARK | Anonymous ladder fixture doc04, no real document named. |
| #608 | bug(tables): binder centroid membership diverges from verifier/reconstruct top-left (leftover from #602) | PARK | Refactor of region-membership predicates; no observed failing page. |
| #609 | bug(tables): VI-A2 centroid membership unpins TL-in/centroid-out drop + y-straddle caption admit (leftover from #602) | PARK | Code-review predicate hole shown on synthetic geometry; no real page. |
| #619 | bug(tables): VI-C2a next-boundary gap uses first baseline of a heuristic merge (leftover from #616) | PARK | Review finding with constructed geometry; no real page. |
| #633 | bug(native): C1 aligned-run assembler reindexes past type-1 blocks so logo pages never merge (leftover from #631) | PARK | Code-review finding; only a synthetic fixture proposed, no real page named. |
| #638 | bug(benchmark): corroborate_rows has no replay_binding / bit-identity pin (leftover from #630) | PARK | Test-pin ticket; no observed failing page. |
| #643 | bug(tables): row-shape reconciliation wrongly keeps a complete narrow-table candidate fail-closed when numeric footnotes share the page | PARK | Synthetic reproduction only; no real narrow-table page observed failing. |
| #653 | bug(figures): a page-sized raster chart dense with data labels reads as a scan under the E1 density rule | PARK | Seen only on a synthetic reviewer-built page; no real fixture exists. |
| #676 | docs(plans): TICKETS C2a Do still teaches superseded ordinal_origin rule-thickness merge (leftover from #667) | PARK | Docs drift in plan files; no failing page. |
| #684 | docs(census): reconcile Fed page-sum 5,507 vs 2,690+2,765 (#636 item 5 leftover) | PARK | Docs reconciliation of census page counts; no failing page. |
| #685 | docs(census): name the six ECB fixture pages for EXTRA_NUMBERS_MAX_SHARE / A1c Done-when (#636 item 7 leftover) | PARK | Docs leftover to name unnamed fixture pages; no observed failure. |
| #694 | docs(plans): STATUS still over-closes #658 and leaves D3 "#659 in review" (leftover from #693) | PARK | Docs STATUS correction; no failing page. |
| #700 | bug(fail-closed): on a two-column page, left-column prose is withheld behind the right column's table marker | PARK | Generic two-column limitation; only synthetic fixtures and test shapes, no specific real page failure. |
| #701 | bug(extract): a PyMuPDF crash or timeout ends the page with no fallback reader, so readable PDFs are recorded as unreadable | PARK | Three Fed PDFs failed whole-document, but no specific page named. |
| #702 | docs(log): D3 Fed re-measure Class A recall numbers contradict the addendum (leftover from #698) | PARK | Docs log reconciliation of recall numbers; no new failing page. |
| #707 | measure(fail-closed): scanned native-prose fallback retention and fidelity across the six Fed minutes | PARK | Measurement plan; no observed failure on a named page. |
| #717 | chore(tests): conftest patches shutil.which globally, so any call-time probe for an external binary reports it missing | PARK | Test-infrastructure chore; no failing page. |
| #738 | proposal(charts): may a panel whose model grid is unverifiable ship the reader's own numbers? | PARK | Proposal; SEP documents named without specific pages. |
| #742 | proposal(charts): carry a monotonic key on reconciliation events instead of depending on stream order | PARK | Proposal; failure only in constructed reversed-sidecar test. |
| #755 | docs(chart-data): STATUS summary still says only Stage 0; CORRECTION overgeneralizes Fed URL (leftover from #754) | PARK | Docs STATUS wording fix; no failing page. |
| #767 | chore(tables): #690 soft leftovers after spacer entity-decode fix | PARK | Review leftovers and doc drift; no failing page. |
| #780 | bug(tables): the rowizer density floor is page-sized, so a narrow band is refused and re-merged | PARK | Synthetic fixture only; real page explicitly unverified. |
| #824 | bug(tables): row_corroboration uses a third region-membership predicate, with no uncertainty channel | PARK | Code-reading finding about a membership predicate; no observed failing page. |
| #827 | proposal(routing): score a routing-miss rate — how often a page is kept native that should not have been | PARK | Metric proposal; #317 pages cited only as context. |
| #829 | proposal(tables): select the two-column split gutter by row-label evidence, or by carrying table-region detection through to scanned pages | PARK | Design proposal with unmeasured hazards; only repo test fixture named. |
| #853 | bug(tables): equal httpx/run_killable budgets still cascade on peer ReadTimeout (#852 leftover) | PARK | Timeout budget code-reading finding; no real document page named. |
| #863 | measure(tables): GH-600's banding change reaches winner selection via _binding_evidence_for_witness, unmeasured | PARK | Measurement proposal; no observed failing page. |
| #868 | bug(tables): the lane-REUSE filter in has_numeric_columns is defeated by token density, not by alignment | PARK | Measures alternative clustering; shipped output not wrong, later shown none of the pages ship. |
| #898 | feat(cli): --reprocess should resume only non-terminal pages from the per-page ledger | PARK | CLI design question on reprocess; no failing page. |
| #919 | bug(robustness): a long-lived pipeline hangs (100% CPU) after ~317 documents; the same document alone takes 5 s | PARK | Hang shown to be state-dependent, not the document; not reproduced on the paper. |
| #930 | bug(tables): negative values shipped as '– 0.48' (sign, space, number) can be parsed as positive | PARK | Fama-French 1997 named but no specific page; corpus-level cell count. |
| #952 | bug(tables): geometryless_block misses a wide source row dropped between multi-anchor geometry-None pairs | PARK | Hypothetical truncate mode; absent from census, no named page. |
| #967 | docs: OUTPUT.md Soft leftovers from #965 — gated figures/equations + §2 vocab | PARK | Docs vocabulary fix; no failing page. |
| #979 | fix(tables): extend PageLadderBudget to pre-gate native CELLS repair | PARK | Budget coverage follow-up; no failing page. |
| #980 | fix(tables): reject NaN/inf in PageLadderBudget total_sec | PARK | Malformed-config validation leftover; no failing page. |
| #981 | fix(ollama): probe_model_generation unreachable reason must use safe_host_label | PARK | Credential-leak code review finding; no document page involved. |
| #982 | fix(ollama): probe_failure_reason should redact credentials in str(exc) | PARK | Credential redaction hardening; no observed failing page. |
| #999 | bug(tables): flattened-table scan fail-opens on drawings/get_text blow-up so a miss can still ship SUCCESS | PARK | Hypothetical fail-open in exception handler; no real page observed. |
| #1003 | fix(1001): _prior_halt_latch swallows read failures and can drop the latch (leftover from #1002) | PARK | Code-reading finding on latch read failure; no real page failure observed. |
| #1008 | feat(tables): split-page lane — native grid for verified table regions, model read for garbled prose | PARK | Feature proposal; named Fama pages show no wrong output. |
| #1009 | test(library): hermetic #972 — pin symlink missing_text/documents refuses before any index write | PARK | Hermetic test-coverage ticket; no failing page. |
| #1010 | fix(library): align _journal_rolls_forward_staged with recover_promotion roll-forward refuses | PARK | Library dry-run predicate alignment; no OCR page failure. |
| #1014 | fix(assemble): derive judge_timeout_native_pages from the shipped failure_mode, not PageState | PARK | Refactor of bucket derivation; no observed failing page. |
| #1015 | test(#1004): pin the shipped outcome when a later COMPLETED refusal supersedes TIMEOUT | PARK | Test-pin ticket; no failing page. |
| #1017 | perf: the OCR model and the page judge may swap on the GPU every page (~2.5 min/page) | PARK | Perf ticket; slowness only, no wrong or missing output on a page. |
| #1024 | fix(tables): text_layer_is_visible should use span.get("chars") like born_digital | PARK | Defensive-access code review finding; no real page named. |
| #1025 | test(tables): invisible-OCR abstention pin should pass a real region | PARK | Test-hardening ticket; no observed failing page. |
| #1026 | test(tables): pin that reason=native_contradiction / sibling survives sidecar resume | PARK | Test-pin ticket; no failing page. |
| #1029 | chore(figures): align invisible_scan PNG render gate with the floor (leftover from #1028) | PARK | Chore: render predicate drift; no real page output wrong. |
| #1035 | bug(figures): crop descriptions — strip query before resolve; read only after bbox gate | PARK | Code-review findings on crop description paths; no real page failure. |
| #1036 | chore(figures): number-free the legacy --describe-figures path (or retire it) | PARK | Chore on legacy flag; no specific document page named. |
| #1040 | bug(judge): tolerance clause in judge_page.md excuses tail/head loss; no truncated_tail/head case measured (#1039 leftover) | PARK | Prompt hole shown only with injected synthetic defects; no real failing page. |
| #1041 | bug(tables): crop readers send no reply cap and ignore finish_reason/done_reason=length, so a cut table reading can ship as complete (#1039 follow-up) | PARK | Code-reading finding; cut reply shipping not observed on a page. |
| #1046 | Soft(tables): _has_data_row treats overview year lines as table data | PARK | Review soft finding, fail-closed; explicitly not seen on real docs. |
