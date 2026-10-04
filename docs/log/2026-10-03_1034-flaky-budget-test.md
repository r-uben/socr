# GH-1034: flaky budget test under CPU load

`test_budget_exhaustion_demotes_page_and_document_through_process` slept `SLOW` of real
time per rung against a `SLOW/3` budget, so scheduler latency decided the outcome. #1011
fixed the sibling tests with a virtual clock; this one was missed.

Change (tests only, `tests/test_gh974_review_pins.py`): `_KeyedRung` advances the shared
`_VirtualClock` instead of `time.sleep`; `_process` installs `_virtual_budget_clock()` and
builds the rungs from it. The transcriber test shares `_KeyedRung`/`_process` and gets the
same fix.

Evidence:
- Stress, 4x `yes` load, single test run 30 times: before 20/30, after 30/30.
- Mutant (external copy of src+tests, `socr.__file__` canary, anchor
  `if self.remaining() <= 0:` count 1, replaced by `if False:`): 5 of 8 tests in the file
  fail, including the target test (document status SUCCESS instead of AUDIT_FAILED).
- File: 8 passed.
