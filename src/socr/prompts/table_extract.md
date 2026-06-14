You are transcribing a SINGLE table from a high-resolution crop of one table in an
academic document (economics, finance, statistics). The image contains only the
table, possibly with its caption and notes.

Transcribe the table, and only the table, into a GitHub-flavored Markdown table.

Read every value directly from the image. This is going into a permanent research
corpus, so accuracy of the numbers is the whole point:

- Reproduce every digit, decimal point, minus sign, parenthesis, asterisk, and
  thousands separator exactly as shown. Do not round, reformat, or "tidy" values.
- Keep standard-error / t-statistic values that sit below a coefficient in their
  own cell, in the same column, on their own row — do not merge them into the
  coefficient cell.
- Preserve the row and column structure you see. One Markdown column per visual
  column. Empty cells stay empty.
- When a table has paired columns (e.g. two consecutive year columns such as 2026
  and 2027 for each variable), ALL rows — including summary rows (Consensus, High,
  Low, Std Dev, Mean) — must use the same number of columns as the data rows. Do
  not flatten or merge columns for summary rows. If a summary row spans fewer
  visual columns than the header, insert blank cells to maintain alignment.
- Significance stars (*, **, ***) belong with the value they annotate.
- Do not invent rows, columns, or values. If a cell is genuinely blank, leave it
  blank. If a value is unreadable, transcribe what you can see rather than guessing
  a plausible number.

Output ONLY the Markdown table — a header row, a separator row, then the body
rows. No prose, no explanation, no code fence, nothing before or after the table.
If the image does not actually contain a table, output nothing.
