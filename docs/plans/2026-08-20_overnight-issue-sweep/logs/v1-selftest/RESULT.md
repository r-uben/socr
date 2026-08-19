# V1 verifier self-test — proof the checker discriminates

Run 2026-08-20 against `logs/v1-selftest/fixture.json`, seven hand-built verdicts
covering every way a citation can be wrong. Exit status was **1** (correct: two
ALREADY-FIXED verdicts lacked the three proofs).

| issue | case | evidence_verified | reason |
|---|---|---|---|
| 9001 | genuinely true citation | True | — |
| 9002 | snippet off by one operator (`>=` vs `>`) | False | snippet not on line 55. claimed='return len(doc) >= threshold' actual='return len(doc) > threshold' |
| 9003 | line number past EOF | False | line 99999 out of range (file has 108 lines) |
| 9004 | fabricated file path | False | path does not exist at main_sha: src/socr/core/imaginary_module.py |
| 9005 | real ancestor commit that DID touch the cited line, but no acceptance criteria and no reproducer | False | acceptance_criteria not enumerated; no reproducer |
| 9006 | fabricated `fixed_by_commit` SHA | False | fixed_by_commit deadbeefdeadbeefdeadbeefdeadbeefdeadbeef is not a commit in this repo |
| 9007 | ROADMAP-UMBRELLA with zero citations | True | citation-exempt by design |

Case 9002 is the one that matters most: a citation that looks right, names a real
file and a real line, and is wrong by one character. A model reviewer waves it
through; git does not.

Case 9005 is the second: ancestry and a touching commit are NOT enough for
ALREADY-FIXED. Without enumerated acceptance criteria and a reproducer that ran,
the verdict is forced to NEEDS-MEASUREMENT.

## Second self-test, after the classifier was extended (same run, later)

The first real triage output made the flat pass/fail useless: 9 of 12 verdicts in
one batch failed the strict check, and the failures were indistinguishable from
each other. Two were labelled FABRICATED. Both turned out to be **real docstrings
with a stray trailing quote** — the model had closed a docstring the source line
leaves open. Calling that fabrication would have slandered an honest agent and
thrown away a correct verdict.

So `check_citation` now returns four classes. `evidence_verified` stays exactly as
CONTRACT specifies (true only for EXACT); the class is extra information for the
adjudicator, not a relaxation.

| class | meaning | evidence_verified |
|---|---|---|
| `EXACT` | snippet is on the cited line | true |
| `DRIFT` | snippet is in the file, different line | false |
| `PARTIAL` | substantial prefix present; transcription noise | false |
| `FABRICATED` | nothing resembling it anywhere in the file | false |

Re-tested with two added adversarial cases:

| issue | case | class |
|---|---|---|
| 9008 | a plausible invented line placed in a real file | `FABRICATED` |
| 9009 | a real line copied from a *different* file into this one | `FABRICATED` |

Both are caught. The weakened matching does not open a door for invention: the
prefix fallback has a 25-character floor, so `return len(doc) >= threshold` cited
against `return len(doc) > threshold` is still `FABRICATED`, not excused as noise.

**Result on the real corpus: 0 fabricated citations across all four batches.**
No dispatched agent invented evidence. The failures are line drift, which the
adjudicator can weigh.
