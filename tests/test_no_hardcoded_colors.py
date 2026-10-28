"""No hard-coded colours in the Qt GUI outside the theme tokens. - Claude Generated

Every colour in ``src/ui`` comes from ``src/ui/styles.py`` (``COLORS`` /
``DARK_COLORS``, the main stylesheet and its ``tone``/``box``/``variant``
rules), so both themes stay consistent and a theme switch reaches every
widget. A literal colour in a widget file is how the GUI drifted before: a
white text field or a cyan button that looked right in one theme and wrong in
the other.

The scan covers hex literals, ``rgb()``/``rgba()``, ``QColor`` built from
numbers or a string literal, ``Qt.GlobalColor`` and CSS colour names in
``color:``/``background:`` declarations. Comment lines are skipped. A colour
that must stay literal goes into ``ALLOWED`` with its reason.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

UI = Path(__file__).resolve().parent.parent / "src" / "ui"

#: The token definitions themselves.
EXEMPT_FILES = {"styles.py"}

#: (file relative to src/ui, exact source text, reason)
ALLOWED = [
    (
        "webcam_capture_dialog.py",
        "QColor(0, 0, 0, 120)",
        "Dims the camera photo outside the crop rectangle; it sits on an image, "
        "not on a themed surface, and must darken the photo in either theme.",
    ),
]

PATTERNS = [
    ("hex", re.compile(r"(?<![\w&])#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?(?:[0-9a-fA-F]{2})?\b")),
    ("rgb()", re.compile(r"\brgba?\(\s*\d")),
    ("QColor literal", re.compile(r"QColor\(\s*(?:\d|['\"])")),
    ("Qt.GlobalColor", re.compile(r"Qt\.GlobalColor\.\w+")),
    (
        "CSS colour name",
        re.compile(
            r"(?:color|background(?:-color)?)\s*:\s*"
            r"(?:white|black|red|green|blue|gray|grey|orange|yellow|purple|navy|darkred|darkgreen)\b",
            re.I,
        ),
    ),
]


def find_hardcoded_colors(root: Path = UI):
    """Yield (relative path, line number, kind, line) for every hit."""
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        if rel in EXEMPT_FILES:
            continue
        allowed = [text for f, text, _ in ALLOWED if f == rel]
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            checked = line
            for text in allowed:
                checked = checked.replace(text, "")
            for kind, pat in PATTERNS:
                if pat.search(checked):
                    yield rel, no, kind, line.strip()


class NoHardcodedColorsTest(unittest.TestCase):
    def test_ui_files_use_theme_tokens(self):
        hits = [f"{rel}:{no} [{kind}] {line}" for rel, no, kind, line in find_hardcoded_colors()]
        self.assertEqual(hits, [], "literal colours in src/ui:\n" + "\n".join(hits))

    def test_every_allowed_entry_still_exists_and_has_a_reason(self):
        for rel, text, reason in ALLOWED:
            self.assertTrue(reason.strip(), rel)
            self.assertIn(text, (UI / rel).read_text(encoding="utf-8"), f"stale entry: {rel}")

    def test_scan_catches_the_usual_forms(self):
        import tempfile

        samples = [
            'w.setStyleSheet("color: #ff9800;")',
            'w.setStyleSheet("background: rgba(33, 150, 243, 0.1)")',
            'item.setForeground(QColor("#2e7d32"))',
            "item.setForeground(QColor(0, 128, 0))",
            "hint.setForeground(Qt.GlobalColor.darkRed)",
            'label.setStyleSheet("color: gray;")',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.py").write_text("\n".join(samples) + "\n", encoding="utf-8")
            lines_hit = {no for _rel, no, _kind, _line in find_hardcoded_colors(root)}
        self.assertEqual(lines_hit, set(range(1, len(samples) + 1)))


if __name__ == "__main__":
    unittest.main()
