"""Die Workflow-Liste auf der Erschließungsseite.

Vorher war der Standard-Workflow eine Auswahlliste von Dateinamen —
"alima_v51_105 (v5.1.105)". Sie sagte weder, was der Workflow tut, noch wo er
liegt, und es gab keinen Weg, von dort hineinzusehen. Die Workflows sind aber
der Gegenstand der Erschließung.

Geprüft wird hier die Liste selbst: Herkunft der Zeilen, Beschränkung der
Beschreibungszeile, Unterscheidbarkeit gleichnamiger Dateien, und dass die
Auswahl den Stern nicht mitzieht. - Claude Generated
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ui.settings.workflow_list import (
    CLASSIC_STEM,
    SUMMARY_LIMIT,
    WorkflowEntry,
    summarise,
    workflow_entries,
)


def _qapp(keep: list):
    from PyQt6.QtWidgets import QApplication

    if not keep:
        keep.append(QApplication.instance() or QApplication(["alima-tests"]))
    return keep[0]


class _Wf:
    """Was ``load_workflow`` zurückgibt, soweit die Liste es anfasst."""

    def __init__(self, version="1.0", description="", steps=1, name="X"):
        self.name = name
        self.version = version
        self.description = description
        self.steps = [object()] * steps


class SummaryLineIsBoundedTest(unittest.TestCase):
    """Die längste Beschreibung im Bestand hat 877 Zeichen (``alima_v51_105``).

    Als umbrechende Zeile wäre sie das höchste Element der Seite — genau das,
    was auf der Modellseite gerade entfernt wurde.
    """

    def test_it_names_version_and_step_count(self):
        self.assertEqual(summarise(_Wf(version="5.1", steps=9)), "v5.1 · 9 Schritte")

    def test_one_step_is_singular(self):
        self.assertEqual(summarise(_Wf(steps=1)), "v1.0 · 1 Schritt")

    def test_a_long_description_is_cut_after_the_first_sentence(self):
        wf = _Wf(description="Kurz gesagt. " + "Weiteres Geschwafel. " * 60)
        line = summarise(wf)
        self.assertLess(len(line), SUMMARY_LIMIT + 40, line)
        self.assertTrue(line.endswith("Kurz gesagt."), line)

    def test_a_description_without_a_sentence_end_is_still_cut(self):
        line = summarise(_Wf(description="wortwortwort " * 60))
        self.assertLess(len(line), SUMMARY_LIMIT + 40, line)
        self.assertTrue(line.endswith("…"), line)

    def test_a_first_sentence_beyond_the_limit_is_not_used_as_the_cut(self):
        """Sonst wäre der "erste Satz" die ganze Zeile."""
        wf = _Wf(description="x" * 400 + ". Danach noch mehr.")
        line = summarise(wf)
        self.assertLess(len(line), SUMMARY_LIMIT + 40, len(line))

    def test_no_description_leaves_just_the_head(self):
        self.assertEqual(summarise(_Wf(description="", steps=3)), "v1.0 · 3 Schritte")


class EntriesComeFromTheWorkflowFilesTest(unittest.TestCase):
    """Was in der Liste steht, steht so in den YAML-Dateien."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.base = Path(self._dir.name)
        (self.base / "tiefe.yaml").write_text(
            'name: Tiefenrecherche\n'
            'version: "1.2"\n'
            'description: Sucht breit und prueft nach.\n'
            'steps:\n'
            '  - id: one\n'
            '    type: llm_agent\n',
            encoding="utf-8",
        )
        (self.base / "alt.yaml").write_text(
            "name: Alt\npipeline:\n  - schritt\n", encoding="utf-8"
        )

    def tearDown(self):
        self._dir.cleanup()

    def _entries(self, files=None, **kw):
        files = files if files is not None else [
            self.base / "alt.yaml", self.base / "tiefe.yaml"
        ]
        with patch(
            "src.core.agents.workflow_loader.discover_workflow_files",
            return_value=files,
        ):
            return workflow_entries(**kw)

    def test_the_name_comes_from_the_yaml_not_the_file(self):
        entry = next(e for e in self._entries() if e.stem == "tiefe")
        self.assertEqual(entry.name, "Tiefenrecherche")
        self.assertEqual(entry.version, "1.2")
        self.assertEqual(entry.steps, 1)
        self.assertIn("Sucht breit und prueft nach.", entry.summary)

    def test_a_non_v4_file_is_skipped(self):
        self.assertNotIn("alt", [e.stem for e in self._entries()])

    def test_a_broken_file_does_not_stop_the_rest(self):
        broken = self.base / "kaputt.yaml"
        broken.write_text("name: [unclosed\n", encoding="utf-8")
        stems = [e.stem for e in self._entries(
            files=[broken, self.base / "tiefe.yaml"]
        )]
        self.assertNotIn("kaputt", stems)
        self.assertIn("tiefe", stems)

    def test_the_classic_pipeline_leads_the_list(self):
        """Sie läuft, wenn nichts Agentisches gewählt ist; fehlte sie, läse
        sich die Liste, als gäbe es die Möglichkeit nicht."""
        entries = self._entries()
        self.assertEqual(entries[0].stem, CLASSIC_STEM)
        self.assertEqual(entries[0].version, "")
        self.assertTrue(entries[0].editable)

    def test_it_can_be_asked_for_the_files_alone(self):
        stems = [e.stem for e in self._entries(include_classic=False)]
        self.assertNotIn(CLASSIC_STEM, stems)

    def test_a_shipped_workflow_is_marked_as_such(self):
        """Die Ablage entscheidet, was Bearbeiten tut: der Editor legt für eine
        Projektdatei eine Kopie im Benutzerverzeichnis an."""
        entry = next(e for e in self._entries() if e.stem == "tiefe")
        self.assertEqual(entry.location, "Projekt")

    def test_a_workflow_in_the_user_directory_is_marked_as_own(self):
        with patch(
            "src.ui.settings.workflow_list._user_dir", return_value=self.base
        ):
            entry = next(e for e in self._entries() if e.stem == "tiefe")
        self.assertEqual(entry.location, "Benutzer")


class TheListShowsWhatItWasGivenTest(unittest.TestCase):
    _kept: list = []

    def _list(self, entries, default=""):
        from src.ui.settings import WorkflowList

        _qapp(self._kept)
        widget = WorkflowList()
        widget.refresh(entries, default_stem=default)
        self._kept.append(widget)
        return widget

    @staticmethod
    def _entry(stem, name, **kw):
        return WorkflowEntry(stem=stem, name=name, version="1.0", steps=2,
                             location="Projekt", summary=f"zu {stem}", **kw)

    def test_the_default_carries_the_star_and_nothing_else_does(self):
        widget = self._list(
            [self._entry("a", "Eins"), self._entry("b", "Zwei")], default="b"
        )
        self.assertEqual([r[0] for r in widget.row_texts()], ["", "⭐"])

    def test_two_files_with_the_same_name_stay_distinguishable(self):
        """Der Name ist Anzeige, der Dateiname der Schlüssel."""
        widget = self._list(
            [self._entry("a", "Gleich"), self._entry("b", "Gleich")]
        )
        self.assertEqual(
            [r[1] for r in widget.row_texts()], ["Gleich [a]", "Gleich [b]"]
        )

    def test_distinct_names_are_not_suffixed(self):
        widget = self._list([self._entry("a", "Eins"), self._entry("b", "Zwei")])
        self.assertEqual([r[1] for r in widget.row_texts()], ["Eins", "Zwei"])

    def test_selecting_a_row_does_not_move_the_star(self):
        """Sonst könnte man nicht hineinsehen, ohne den Standard zu ändern."""
        widget = self._list(
            [self._entry("a", "Eins"), self._entry("b", "Zwei")], default="a"
        )
        widget.select("b")
        self.assertEqual(widget.default_stem(), "a")
        self.assertEqual([r[0] for r in widget.row_texts()], ["⭐", ""])

    def test_the_star_moves_only_on_the_button_and_reports_it(self):
        widget = self._list(
            [self._entry("a", "Eins"), self._entry("b", "Zwei")], default="a"
        )
        seen = []
        widget.default_changed.connect(seen.append)
        widget.select("b")
        widget._make_default()
        self.assertEqual(seen, ["b"])
        self.assertEqual(widget.default_stem(), "b")
        self.assertEqual([r[0] for r in widget.row_texts()], ["", "⭐"])

    def test_the_summary_of_the_selected_row_is_shown(self):
        widget = self._list([self._entry("a", "Eins"), self._entry("b", "Zwei")])
        widget.select("b")
        self.assertEqual(widget.summary.text(), "zu b")

    def test_a_non_editable_row_disables_the_edit_button(self):
        widget = self._list([
            self._entry("a", "Eins", editable=False), self._entry("b", "Zwei")
        ])
        widget.select("a")
        self.assertFalse(widget.edit_button.isEnabled())
        widget.select("b")
        self.assertTrue(widget.edit_button.isEnabled())

    def test_the_default_button_is_dead_on_the_current_default(self):
        widget = self._list(
            [self._entry("a", "Eins"), self._entry("b", "Zwei")], default="a"
        )
        widget.select("a")
        self.assertFalse(widget.default_button.isEnabled())
        widget.select("b")
        self.assertTrue(widget.default_button.isEnabled())

    def test_many_workflows_scroll_inside_the_list_not_the_page(self):
        """13 Workflows sind im Bestand; die Seite soll davon nicht wachsen."""
        from src.ui.settings import WorkflowList

        few = self._list([self._entry(str(i), f"W{i}") for i in range(3)])
        many = self._list([self._entry(str(i), f"W{i}") for i in range(30)])
        self.assertGreater(many.list.maximumHeight(), few.list.maximumHeight())
        self.assertLess(
            many.list.maximumHeight(),
            few.list.maximumHeight()
            + WorkflowList.MAX_VISIBLE_ROWS * many.list.sizeHintForRow(0),
        )

    def test_a_refresh_keeps_the_selection(self):
        entries = [self._entry("a", "Eins"), self._entry("b", "Zwei")]
        widget = self._list(entries, default="a")
        widget.select("b")
        widget.refresh(entries, default_stem="a")
        self.assertEqual(widget.selected_stem(), "b")


if __name__ == "__main__":
    unittest.main()
