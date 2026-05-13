"""Normalized HTML comparison helper for snapshot tests - Claude Generated.

Used by P-β renderer-migration tests. Tolerates whitespace runs,
attribute spacing, and tag-name case differences.
"""
from __future__ import annotations

import re


_TAG_NAME_RE = re.compile(r"(<\s*/?)(\w+)")


def normalize_html(html: str) -> str:
    """Return a normalized form for byte-tolerant HTML comparison.

    Steps:
    1. Strip + collapse all whitespace runs to a single space.
    2. Normalize attribute spacing (``foo = "x"`` → ``foo="x"``).
    3. Lowercase tag-names (HTML-spec case-insensitive).
    """
    if not html:
        return ""
    text = re.sub(r"\s+", " ", html.strip())
    text = re.sub(r"\s*=\s*", "=", text)
    text = _TAG_NAME_RE.sub(lambda m: m.group(1) + m.group(2).lower(), text)
    return text


def assert_html_equal(actual: str, expected: str) -> None:
    """Raise AssertionError if normalized forms differ.

    Error-message shows the first divergent slice for quick diagnosis.
    """
    a = normalize_html(actual)
    e = normalize_html(expected)
    if a == e:
        return
    # Find divergence offset
    offset = next(
        (i for i in range(min(len(a), len(e))) if a[i] != e[i]),
        min(len(a), len(e)),
    )
    start = max(0, offset - 40)
    raise AssertionError(
        "HTML mismatch at offset {}:\n"
        "  actual ...{}...\n"
        "  expect ...{}...".format(
            offset, a[start : offset + 80], e[start : offset + 80]
        )
    )
