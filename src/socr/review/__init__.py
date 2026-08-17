"""Hand-judgement review instrument: side-by-side page image vs extracted markdown.

See :mod:`socr.review.html` for the generator. GH-220.
"""

from socr.review.html import PageRecord, ReviewReport, build_review_html, collect_pages

__all__ = ["PageRecord", "ReviewReport", "build_review_html", "collect_pages"]
