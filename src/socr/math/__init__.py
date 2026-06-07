"""Math recovery: image-OCR of font-corrupted equation regions to LaTeX.

Some born-digital PDFs typeset math with a broken font/ToUnicode map, so the text
layer decodes operators and delimiters as Latin-1 mojibake ('=' -> '¼', '(' -> 'ð').
The prose is fine; only the math is corrupt, and it is unrecoverable from the text
layer. This package renders just the affected equation regions and reads them back
as LaTeX with a local vision model, leaving the native prose untouched.
"""
