# 2026-10-12 — #1060 default Gemini model and resume fingerprint (GH-1086 item 3)

No repo-level CHANGELOG exists; this note is the decision record for operators.

## What changed (#1060)

- ``DEFAULT_GEMINI_MODEL`` is ``gemini-3.8-flash`` (was ``gemini-3-flash-preview`` in four modules).
- ``socr process --gemini-model <id>`` sets ``PipelineConfig.gemini_model`` for one run and is part of the cloud rung fingerprint when Gemini is enabled.

## Resume / cache behaviour (intentional, not a regression)

- The run fingerprint includes ``gemini_model`` (and related cloud-rung fields). Flipping the default to 3.8 invalidates cached Gemini OCR output recorded under the old default: a re-run under the new default will not resume-match and will re-escalate through the ladder.
- Runs whose metadata still record ``gemini-3-flash-preview`` likewise will not resume-match a fresh run on 3.8 unless you pass ``--gemini-model gemini-3-flash-preview`` (or keep the old fingerprint inputs). That is by design: model identity is part of what "same run" means.

## Pricing note

- ``profile_by_model`` maps ``RETIRED_GEMINI_MODELS`` to ``PROFILE_GEMINI`` so historical judge/OCR names are not priced at $0. The rate is a floor for cost aggregation, not a reconstruction of Google's invoice (see comment at the fallback in ``providers.py``).
