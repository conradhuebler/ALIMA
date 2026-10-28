"""Style sheets in src/ must be valid QSS (scripts/lint_qss.py).

Qt drops malformed declarations (or the whole rule) without a warning. A line
``#padding: 12px 8px;`` in the pipeline tab's tab rule kept ``min-width: 100px``
from the same rule inactive for months; cleaning the line up made the vertical
tabs 84 px wider overnight. Claude Generated.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("lint_qss", REPO / "scripts" / "lint_qss.py")
lint = importlib.util.module_from_spec(_spec)
sys.modules["lint_qss"] = lint  # dataclasses look the module up by name
_spec.loader.exec_module(lint)  # type: ignore[union-attr]


def _levels(text, **kw):
    return [(f.level, f.message) for f in lint.lint_qss_text(text, **kw)]


def _errors(text, **kw):
    return [m for lvl, m in _levels(text, **kw) if lvl == "error"]


class RejectsWhatQtDropsTest(unittest.TestCase):
    def test_the_original_pipeline_tab_rule_is_caught(self):
        """The exact lines that went wrong."""
        qss = """
            QTabBar::tab {
                background: #f5f5f5;
                border: 1px solid #ddd;
                #padding: 12px 8px; # do not set padding to avoid increasing tab height/width
                margin-bottom: 2px;
                min-width: 100px;
            }
        """
        errors = _errors(qss)
        self.assertTrue(errors)
        self.assertTrue(any("#padding" in e or "comment-like" in e for e in errors))

    def test_hash_and_double_slash_comment_lines(self):
        self.assertTrue(_errors("QLabel { # a note\n color: red; }"))
        self.assertTrue(_errors("QLabel { // a note\n color: red; }"))

    def test_declaration_without_colon(self):
        self.assertTrue(_errors("QLabel { color red; }"))

    def test_missing_semicolon_between_declarations(self):
        self.assertTrue(any("missing ';'" in e for e in _errors(
            "QLabel { color: red\n background: blue; }")))

    def test_unbalanced_braces_and_text_after_the_last_rule(self):
        self.assertTrue(any("unclosed" in e for e in _errors("QLabel { color: red;")))
        self.assertTrue(any("unmatched" in e for e in _errors("QLabel { color: red; } }")))
        self.assertTrue(_errors("QLabel { color: red; } stray text"))

    def test_unbalanced_parenthesis_and_quote(self):
        self.assertTrue(_errors("QLabel { background: qlineargradient(x1:0, y1:0; }"))
        self.assertTrue(_errors('QLabel { font-family: "Segoe; }'))

    def test_nested_braces(self):
        self.assertTrue(any("nested" in e for e in _errors("QLabel { QPushButton { color: red; } }")))

    def test_unknown_property_is_only_a_warning(self):
        levels = _levels("QLabel { cursor: pointer; }")
        self.assertEqual([lvl for lvl, _ in levels], ["warning"])

    def test_bare_declaration_lists_are_checked_too(self):
        self.assertEqual(_errors("color: red; background: blue;", as_declarations=True), [])
        self.assertTrue(_errors("color: red; #oops: 1;", as_declarations=True))


class AcceptsValidQssTest(unittest.TestCase):
    def test_a_normal_rule_set(self):
        qss = """
            /* comment with { braces } and a: colon; */
            QPushButton { background-color: #aabbcc; border: 1px solid #000; padding: 4px 8px; }
            QPushButton:hover:!selected { background: #112233; }
            QTabWidget::tab-bar { alignment: left; }
            #panelStrip { border-bottom: 1px solid #ddd; }
        """
        self.assertEqual(_levels(qss), [])

    def test_gradients_spanning_lines_are_not_a_missing_semicolon(self):
        qss = """
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #111111, stop:0.5 #222222, stop:1 #333333);
                border-radius: 5px;
            }
        """
        self.assertEqual(_levels(qss), [])

    def test_qt_specific_and_property_prefixes(self):
        self.assertEqual(_levels("QLabel { -qt-background-role: base; qproperty-flat: true; }"), [])


class SourceExtractionTest(unittest.TestCase):
    def _lint(self, source):
        return lint.lint_source(source, "t.py")

    def test_fstring_placeholders_and_doubled_braces(self):
        src = '''
c = {"x": "#fff"}
css = f"""
    QFrame {{
        background: {c['x']};
        border: 1px solid {c['x']};
    }}
"""
'''
        self.assertEqual(self._lint(src), [])

    def test_format_template_with_doubled_braces(self):
        src = 'css = "QFrame {{ background: {bg}; }}".format(bg="#fff")\n'
        self.assertEqual(self._lint(src), [])

    def test_set_stylesheet_literal_is_checked_and_reports_the_line(self):
        src = "w.setStyleSheet('''\n  QLabel {\n    #bad: 1;\n  }\n''')\n"
        findings = self._lint(src)
        self.assertTrue(findings and findings[0].level == "error")
        self.assertGreaterEqual(findings[0].line, 1)

    def test_set_stylesheet_with_bare_declarations(self):
        self.assertEqual(self._lint('w.setStyleSheet("color: red; font-weight: bold;")\n'), [])
        self.assertTrue(self._lint('w.setStyleSheet("color red")\n'))

    def test_stylesheet_text_assigned_to_a_constant_is_found(self):
        src = 'STYLE = """\nQFrame {\n  # bad\n  color: red;\n}\n"""\n'
        self.assertTrue(self._lint(src))

    def test_docstrings_with_json_examples_are_ignored(self):
        src = 'def f():\n    """Returns {"hits": [...], "a": 1}\n    QFrame {x}"""\n'
        self.assertEqual(self._lint(src), [])

    def test_web_css_without_a_qt_selector_is_ignored(self):
        src = 'page = "<style>:root { --doc-bg: #fff; } body { margin: 0; }</style>"\n'
        self.assertEqual(self._lint(src), [])

    def test_non_literal_arguments_are_skipped_without_error(self):
        self.assertEqual(self._lint("w.setStyleSheet(build_css())\nw.setStyleSheet(css)\n"), [])


class RepositoryIsCleanTest(unittest.TestCase):
    def test_no_style_sheet_in_src_is_malformed(self):
        findings = lint.lint_paths([REPO / "src"])
        self.assertEqual([str(f) for f in findings], [])

    def test_the_script_runs_as_a_command(self):
        self.assertEqual(lint.main([str(REPO / "src" / "ui" / "styles.py")]), 0)


if __name__ == "__main__":
    unittest.main()
