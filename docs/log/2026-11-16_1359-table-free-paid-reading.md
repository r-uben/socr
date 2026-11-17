# #1359: a faithful table-free paid reading may lift the floor

Condition (agentic.paid_reading_keeps_disputed_table_tokens): a paid reading taken after a
witness dispute, accepted by the page judge, with no table, lifts the #1124 floor iff every
numeric token of every earlier disputed table (disputed ones included) and every label the
witness did not dispute is present in it (set containment).

## Measurement (tesseract through build_scanned_evidence, text layer excluded, 2026-11-16)

Chart pages. The qwen tables of p053 p067 p093 were never saved. Known only: the tokens the
witness disputed. All are present in the real Hydrogen reading.
- p053 disputed 0, 0.0, -6: present.  p067 disputed -1.5 -0.5 0.0 1.0 2.0: present.
- p093 disputed 10, 0: present.
p027 p028 p051: real qwen tables (chart-title grids), no numeric tokens. Hydrogen readings are
the offline fairread/hydrogen files, not pipeline runs. p028 10/10 and p051 20/20 labels
present; p027 table has no checkable token (vacuous).

Genuine tables, qwen table with its lines stripped from its own reading as the table-free reading:
p007 20 numeric tokens, 8+ missing; p009 134, missing; p044 34, missing; p063 185, missing;
p097 162, missing; p034 285 (witness supported 0 of them). Condition false on all six.

Why not the starting candidate (witness-supported tokens only): p034's witness supports none
of 285 tokens, so the obligation is empty and a table-free reading would lift. Requiring all
numeric tokens closes that.
Why set not multiset: chart readings list a tick once, a table repeats it per row; the genuine
controls fail under both.

Residual: a label-only table whose labels were all disputed carries no obligation.
