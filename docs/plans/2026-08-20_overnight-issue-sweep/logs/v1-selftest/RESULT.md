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
