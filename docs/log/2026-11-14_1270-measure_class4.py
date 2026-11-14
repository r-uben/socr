"""#1270 B4 class 4 measurement: error + 'hallucination' pages of a cached run, projected.

Reads sidecars only (no OCR, no PDFs). Usage:
    measure_class4 <unpacked_run_dir>
The projection applies the gate's own rule (``verifiable`` False in the page's
``source_evidence_table_reject`` event -> ``table_unverified_witness``, else
``witness_disagrees``) and checks that the page keeps status ``error`` and a mode in
``WITNESS_REJECT_MODES`` (the set every control-flow site reads). Status is read from the
sidecar and is not recomputed: the change is a label, proved unchanged by the tests.
"""

import collections
import glob
import json
import sys

from socr.core.result import WITNESS_REJECT_MODES, FailureMode

root = sys.argv[1]
count = collections.Counter()
for f in glob.glob(root + "/*/out/*/*/pages/*.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get("status") != "error" or d.get("failure_mode") != "hallucination":
        continue
    reject = [
        e for e in d.get("audit_events") or [] if e.get("kind") == "source_evidence_table_reject"
    ]
    if not reject:
        count["no_reject_event"] += 1
        continue
    verifiable = bool(reject[-1].get("data", {}).get("verifiable"))
    new = FailureMode.WITNESS_DISAGREES if verifiable else FailureMode.TABLE_UNVERIFIED_WITNESS
    assert new in WITNESS_REJECT_MODES
    count[new.value] += 1
    count["status_error_kept"] += 1
print(
    dict(count),
    "labelled hallucination before:",
    sum(
        v
        for k, v in count.items()
        if k in {"witness_disagrees", "table_unverified_witness", "no_reject_event"}
    ),
)
