# TICKET-A1 — isolation canary sentinel transcript

Generated 2026-08-20 by the night orchestrator. Baseline `main_sha` = 53b0637b928c486e9ff3023fa9527905fec017b2.
Worktree under test: `/Users/rubenffuertes/repos/.worktrees/socr-night-base` (detached at main_sha).

Purpose: prove the canary is not decorative — that with `PYTHONPATH=<worktree>/src`
pytest genuinely loads **this worktree's** source, by breaking a sentinel symbol here
and watching a green test go red, then restoring it.

## Step 0 — canary states (three conditions)

### 0a. Base worktree, PYTHONPATH unset → must exit 1 (the editable-install trap, live)
```
CANARY FAIL: socr resolves OUTSIDE this worktree
         got      : /Users/rubenffuertes/repos/tools/socr/src/socr/__init__.py
         expected : /Users/rubenffuertes/repos/.worktrees/socr-night-base/src/socr/...
         fix      : export PYTHONPATH=/Users/rubenffuertes/repos/.worktrees/socr-night-base/src   (and re-run)
exit=1
```

### 0b. Base worktree, PYTHONPATH set → must exit 0
```
CANARY OK: socr.__file__ = /Users/rubenffuertes/repos/.worktrees/socr-night-base/src/socr/__init__.py
           worktree      = /Users/rubenffuertes/repos/.worktrees/socr-night-base
exit=0
```

### 0c. Main checkout (owned by another session) → must exit 1
```
CANARY FAIL: refusing to run in the main checkout (/Users/rubenffuertes/repos/tools/socr) — it is owned by another session (CONTRACT fact 2)
exit=1
```

## Step 1 — baseline: tests/test_chunker.py green in the worktree
```
16 passed, 5 warnings in 0.79s
```

## Step 2 — break a sentinel IN THE WORKTREE ONLY

Edit applied to `/Users/rubenffuertes/repos/.worktrees/socr-night-base/src/socr/core/chunker.py`:
```diff
@@ -52,7 +52,7 @@ class PDFChunker:
             threshold = self.max_pages_per_chunk
 
         with fitz.open(pdf_path) as doc:
-            return len(doc) > threshold
+            return len(doc) >= threshold  # SENTINEL-BREAK
 
     def chunk(self, pdf_path: Path, output_dir: Path) -> list[PDFChunk]:
         """Split *pdf_path* into chunk PDFs written to *output_dir*.
```

Confirm the main checkout was NOT touched (its working tree must show no diff for this file):
```
$ git -C /Users/rubenffuertes/repos/tools/socr diff --stat -- src/socr/core/chunker.py
(no output above = main checkout untouched)
```

## Step 3 — the same test now FAILS, proving pytest loaded the worktree source
```
tests/test_chunker.py:49: AssertionError
tests/test_chunker.py:56: AssertionError
FAILED tests/test_chunker.py::TestNeedsChunking::test_exact_threshold_does_not_need_chunking
FAILED tests/test_chunker.py::TestNeedsChunking::test_custom_threshold - Asse...
2 failed, 14 passed, 5 warnings in 1.89s
```

## Step 4 — the trap itself: same broken worktree, PYTHONPATH unset → tests pass anyway

This is the void-every-result failure mode. The source is broken *in this worktree*,
yet the suite is green, because `import socr` resolved to the main checkout.
```
2 failed, 14 passed, 5 warnings in 0.82s
```

### Why it still failed — a correction to CONTRACT fact 1, found while proving it

Step 4 was expected to go GREEN (broken worktree source made invisible by the
editable install). It went red instead. Cause, verified mechanically:

```
$ grep -n "pythonpath" pyproject.toml
98:pythonpath = ["src"]
```

`pytest` resolves `pythonpath = ["src"]` **relative to rootdir**, and rootdir is
discovered by walking up from the test arguments to the nearest `pyproject.toml`.
So any pytest run whose targets live inside a worktree already prepends that
worktree's `src` — isolation holds even from `/tmp` with absolute paths:

```
$ cd /tmp && env -u PYTHONPATH pytest /Users/rubenffuertes/repos/.worktrees/socr-night-base/tests/test_chunker.py -q
2 failed, 14 passed, 5 warnings in 0.45s
```

The trap is therefore **narrower but real**. It bites everything that is not
pytest-with-rootdir-inside-the-worktree:

```
$ cd /Users/rubenffuertes/repos/.worktrees/socr-night-base && env -u PYTHONPATH python -c 'import socr; print(socr.__file__)'
/Users/rubenffuertes/repos/tools/socr/src/socr/__init__.py
```

i.e. the `socr` console script, ad-hoc `python -c` probes, reproducer scripts and
any pytest invocation rooted outside the worktree all silently execute **main
checkout** code. Those are exactly the tools a triager reaches for when building a
reproducer, so the mandate in CONTRACT fact 1 stands unchanged — `export
PYTHONPATH=<worktree>/src` plus a passing canary before any measurement. Only the
stated reason changes: pytest is already safe, everything else is not.

## Step 5 — restore the sentinel, worktree clean, tests green again
```
$ git -C /Users/rubenffuertes/repos/.worktrees/socr-night-base status --porcelain
(no output = restored to main_sha exactly)
$ git -C /Users/rubenffuertes/repos/.worktrees/socr-night-base rev-parse HEAD
53b0637b928c486e9ff3023fa9527905fec017b2
$ PYTHONPATH=/Users/rubenffuertes/repos/.worktrees/socr-night-base/src pytest tests/test_chunker.py -q
16 passed, 5 warnings in 0.73s
```

**Result: canary proven in all three states, sentinel break observed and reverted,
worktree byte-identical to `main_sha`.**
