# Blind cell transcription

You are shown ONE image crop of a table and a list of cell references.

Your only job: read what is printed in each named cell, in the image, and
report it verbatim.

{{CELL_REF_GRAMMAR}}

Rules:

- Report exactly what the image shows for that cell — the printed
  characters, including a minus sign, decimal separator, currency symbol,
  percent sign, or parentheses. Do not reformat, round, or normalise.
- If you CAN see the cell and it is blank — the cell exists in the image and
  there is no mark in it — return the empty string `""` for it. That is a
  reading: you looked, and there was nothing printed.
- If you CANNOT see the cell at all — it is cut off, obscured, rotated out of
  view, illegible, or you cannot locate it — return `null` for it. `null` is
  NOT the same answer as `""`, and the difference matters: `""` says the cell
  is empty, `null` says you did not read it. Never guess, and never infer a
  value from a neighbouring cell, a column total, or the surrounding text.
  Answering `null` is always safe and is always better than a guess.
- You have not been told what anyone else thinks these cells contain, and
  there is no answer to agree with. Report only what you can read.

Answer with STRICT JSON and nothing else — no prose, no code fence, no
explanation. The object must have exactly one key per requested reference,
spelled exactly as it was requested, and no other keys. Its shape:

```
{"<ref>": "<text as printed>", "<ref>": "", "<ref>": null}
```

Replace each `<ref>` with one of the references you were asked for, and
`<text as printed>` with the characters you read at that position. The three
values above stand for the three cases, in the order the rules give them: a
cell you read, a cell you saw to be blank, and a cell you could not read.
