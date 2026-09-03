Cell references use ONE canonical, value-free grammar. It is defined here, in
this file, and spliced into every prompt that mentions a cell — the reader
prompt and the blind-transcription prompt — so the coordinate a judge names,
the coordinate a blind reader is asked for, and the coordinate socr resolves
against the emitted markdown are always the same physical cell.

A reference is PHYSICAL. It names a position in the printed grid, and says
nothing about what that position holds:

- `C<k>` — the k-th column of the table, counting from the LEFTMOST column,
  which is always column number one. That holds whatever the leftmost column
  contains: a name, a number, a date, or nothing at all. Never skip a column
  for any reason, and never let what a column contains change how columns are
  counted.
- `R<n>C<k>` — the cell at column `<k>` of the n-th body row, counting body
  rows from the first row BELOW the header.
- `H<n>C<k>` — the cell at column `<k>` of the n-th header row. Most tables
  have a single header row, so `n` is usually one; a table whose header spans
  two levels numbers the upper row first and the lower row second.
- A reference carries coordinates and nothing else. Never put a cell's
  contents into a reference.
