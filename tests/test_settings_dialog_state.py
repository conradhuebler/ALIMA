"""The settings dialog must not lose configuration it has no widget for.

``_get_config_from_ui`` used to *rebuild* ``DatabaseConfig``, ``SystemConfig`` and
``UIConfig`` from scratch, so every field without a widget had to be hand-carried
in the constructor call. Five were not, and every "💾 Save & Close" silently reset
them:

* ``SystemConfig.url_fetch_allowlist`` — the SSRF exception list (``net_guard``)
* ``SystemConfig.url_fetch_max_bytes``
* ``UIConfig.ui_language`` — read by ``config_manager`` → ``i18n.set_language``
* ``UIConfig.ptheta_banner_seen`` — the one-time banner flag
* ``DatabaseConfig.gnd_local_path`` — the local GND authority DB

The dialog now mutates the ``config_to_edit`` copy instead, so a field without a
widget is simply left alone. These tests name each field individually, so a
regression says which one. - Claude Generated
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


def _qapp(keep: list):
    from PyQt6.QtWidgets import QApplication

    if not keep:
        keep.append(QApplication.instance() or QApplication(["alima-tests"]))
    return keep[0]


class _DialogHarness:
    """Builds a real ComprehensiveSettingsDialog over a config with non-default
    values everywhere, with the two sub-tabs stubbed out. - Claude Generated"""

    _kept: list = []

    def configure(self, cfg):
        """Hook: a subclass narrows the config the dialog is built on."""

    def _dialog(self):
        from src.ui.comprehensive_settings_dialog import ComprehensiveSettingsDialog
        from src.utils.config_models import AlimaConfig

        cfg = AlimaConfig()
        cfg.system_config.url_fetch_allowlist = ["intranet.example"]
        cfg.system_config.url_fetch_max_bytes = 4242
        cfg.ui_config.ui_language = "en"
        cfg.ui_config.ptheta_banner_seen = True
        cfg.database_config.gnd_local_path = "/srv/alima/gnd_local.db"
        # Fields that DO have a widget, set away from their defaults so the test
        # also proves the dialog still writes what it owns.
        cfg.system_config.autosave_dir = "/srv/alima/results"
        cfg.database_config.sqlite_path = "/srv/alima/knowledge.db"
        self.configure(cfg)

        _qapp(self._kept)
        cm = MagicMock()
        cm.load_config.return_value = cfg
        cm.config_file = "/x/config.json"
        cm.get_config_info.return_value = {"os": "Linux", "config_file": "/x/config.json"}
        # The two sub-tabs are real QWidgets (addTab needs one) with the handful
        # of methods the dialog calls on them.
        from PyQt6.QtCore import pyqtSignal
        from PyQt6.QtWidgets import QWidget

        class _StubTab(QWidget):
            config_changed = pyqtSignal()
            task_preferences_changed = pyqtSignal()

            def __init__(self, *a, **kw):
                super().__init__()

            def load(self, _config):
                pass

            def apply_to(self, _config):
                pass

            def _update_config_from_ui(self):
                pass

            def _refresh_effective_defaults(self):
                pass

        with patch(
            "src.ui.comprehensive_settings_dialog.ConfigManager", return_value=cm
        ), patch(
            "src.ui.comprehensive_settings_dialog.UnifiedProviderTab", _StubTab
        ), patch(
            "src.ui.comprehensive_settings_dialog.PluginSettingsTab", _StubTab
        ):
            dlg = ComprehensiveSettingsDialog()
        self._kept.append(dlg)
        return dlg


class SettingsDialogPreservesUnwidgetedFieldsTest(_DialogHarness, unittest.TestCase):
    """Each of the five fields, named individually so a regression says which."""

    def _saved(self):
        dlg = self._dialog()
        return dlg._get_config_from_ui()

    def test_url_fetch_allowlist_survives(self):
        self.assertEqual(
            self._saved().system_config.url_fetch_allowlist, ["intranet.example"]
        )

    def test_url_fetch_max_bytes_survives(self):
        self.assertEqual(self._saved().system_config.url_fetch_max_bytes, 4242)

    def test_ui_language_survives(self):
        self.assertEqual(self._saved().ui_config.ui_language, "en")

    def test_ptheta_banner_seen_survives(self):
        self.assertTrue(self._saved().ui_config.ptheta_banner_seen)

    def test_gnd_local_path_survives(self):
        self.assertEqual(
            self._saved().database_config.gnd_local_path, "/srv/alima/gnd_local.db"
        )

    def test_widgeted_fields_are_still_written(self):
        """The counter-check: preservation must not mean "writes nothing"."""
        dlg = self._dialog()
        dlg.autosave_dir.setText("/tmp/other")
        dlg.sqlite_path.setText("/tmp/other.db")
        cfg = dlg._get_config_from_ui()
        self.assertEqual(cfg.system_config.autosave_dir, "/tmp/other")
        self.assertEqual(cfg.database_config.sqlite_path, "/tmp/other.db")


class DatabaseResetCompletesTest(_DialogHarness, unittest.TestCase):
    """"↺ Reset Database Settings" ran `setText` on a QComboBox and died mid-reset.

    The AttributeError landed after username/password were cleared but before
    charset, SSL, timeout and auto-create were touched, and the "Reset Complete"
    box never appeared — so the dialog was left in a half-reset state with no
    indication. - Claude Generated
    """

    def test_reset_completes_and_sets_every_field(self):
        from unittest.mock import patch

        dlg = self._dialog()
        dlg.mysql_username.setText("someone")
        dlg.mysql_charset.setCurrentText("latin1")
        dlg.connection_timeout.setValue(99)

        from PyQt6.QtWidgets import QMessageBox

        with patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
        ), patch.object(QMessageBox, "information") as done:
            dlg._reset_database_to_defaults()

        self.assertEqual(dlg.mysql_username.text(), "")
        self.assertEqual(dlg.mysql_charset.currentText(), "utf8mb4")
        self.assertEqual(dlg.connection_timeout.value(), 30)
        done.assert_called_once()


class CancelActuallyCancelsTest(_DialogHarness, unittest.TestCase):
    """The dialog advertises a Unit of Work; two paths broke it.

    * "🔧 Test Connection" called _get_config_from_ui — which pulls in the plugin
      tab and the provider tab — and assigned the result to the process-wide
      ConfigManager singleton. One click published every unsaved edit into the
      running application, and Cancel could not take it back.
    * Provider add/edit/remove wrote straight to disk, so a provider added and
      then cancelled was saved anyway.

    - Claude Generated
    """

    def test_test_connection_does_not_publish_the_editing_copy(self):
        from unittest.mock import patch

        dlg = self._dialog()
        dlg.sqlite_path.setText("/tmp/probe.db")

        with patch(
            "src.ui.comprehensive_settings_dialog.DatabaseTestWorker"
        ) as worker:
            dlg._test_database_connection()

        # The manager's cached config was never assigned to.
        self.assertFalse(dlg.config_manager._config.__class__.__name__ == "AlimaConfig"
                         and dlg.config_manager._config is dlg.config_to_edit)
        # …and the probe carried the widget values, not the saved ones.
        probe = worker.call_args[0][1]
        self.assertEqual(probe.sqlite_path, "/tmp/probe.db")

    def test_test_connection_probes_only_database_settings(self):
        """It must not reach into the plugin/provider tabs to build its probe."""
        from unittest.mock import patch

        dlg = self._dialog()
        with patch.object(dlg, "_get_config_from_ui") as harvest, patch(
            "src.ui.comprehensive_settings_dialog.DatabaseTestWorker"
        ):
            dlg._test_database_connection()
        harvest.assert_not_called()



class ProviderEditsReachDiskAndCancelRollsThemBackTest(unittest.TestCase):
    """Providers persist on change; Cancel undoes it.

    Both halves are load-bearing and pull in opposite directions:

    * **Persist on change**, because the model list is fetched through
      ``LlmService``, whose client map (``llm_service.py:945``) and provider
      lookup (``:960``) both read the *saved* config. A provider that exists only
      in the dialog's editing copy has no client and no config entry, so its model
      combo stays empty — which is exactly what buffering it caused.
    * **Roll back on Cancel**, because otherwise a provider added and then
      cancelled stays on disk.

    Writing the whole editing copy would also publish unsaved plugin/chat edits
    into the ConfigManager singleton, so only ``providers`` is written.
    - Claude Generated
    """

    def setUp(self):
        import json
        import tempfile
        from pathlib import Path

        from src.utils.config_manager import ConfigManager

        ConfigManager.reset()
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"
        self.tmp.write_text(json.dumps({
            "unified_config": {"providers": [
                {"name": "ollama", "provider_type": "ollama", "enabled": True}
            ]},
            "system_config": {"autosave_dir": "/vorher/results"},
        }))
        self.cm = ConfigManager()
        self.cm.config_file = self.tmp

    def tearDown(self):
        from src.utils.config_manager import ConfigManager

        ConfigManager.reset()

    def _tab(self):
        from unittest.mock import patch

        from src.ui.unified_provider_tab import UnifiedProviderTab

        from copy import deepcopy

        # Wired like ComprehensiveSettingsDialog does it: the tab edits a DEEPCOPY,
        # never the object ConfigManager caches. Handing it the cached object
        # instead makes the isolation in _persist_providers untestable — both
        # paths then mutate the same instance. - Claude Generated
        editing = deepcopy(self.cm.load_config())
        tab = UnifiedProviderTab.__new__(UnifiedProviderTab)
        tab.logger = __import__("logging").getLogger("test")
        tab.unified_config = editing.unified_config
        tab.config = editing
        tab.config_manager = self.cm
        tab._providers_at_open = deepcopy(list(editing.unified_config.providers))
        return tab

    def _on_disk(self):
        import json

        return [p["name"] for p in
                json.loads(self.tmp.read_text())["unified_config"]["providers"]]

    def test_a_new_provider_reaches_disk_immediately(self):
        from src.utils.config_models import UnifiedProvider

        tab = self._tab()
        tab.unified_config.providers.append(
            UnifiedProvider(name="Mistral", provider_type="openai_compatible",
                            enabled=True, base_url="https://api.mistral.ai/v1")
        )
        tab._persist_providers()
        self.assertIn("Mistral", self._on_disk())

    def test_the_saved_config_is_what_the_model_lookup_reads(self):
        """llm_service:960 resolves the provider from load_config()."""
        from src.utils.config_models import UnifiedProvider

        tab = self._tab()
        tab.unified_config.providers.append(
            UnifiedProvider(name="Mistral", provider_type="openai_compatible",
                            enabled=True, base_url="https://api.mistral.ai/v1")
        )
        tab._persist_providers()
        names = [p.name for p in self.cm.load_config().unified_config.providers]
        self.assertIn("Mistral", names)

    def test_add_provider_persists_without_a_separate_save(self):
        """The mutation site itself, not just the helper it should call."""
        from unittest.mock import MagicMock, patch

        from PyQt6.QtWidgets import QDialog

        from src.utils.config_models import UnifiedProvider

        tab = self._tab()
        tab.provider_table = MagicMock()
        tab._load_configuration = lambda: None
        tab.config_changed = MagicMock()

        dlg = MagicMock()
        dlg.exec.return_value = QDialog.DialogCode.Accepted
        dlg.get_provider_data.return_value = UnifiedProvider(
            name="Mistral", provider_type="openai_compatible", enabled=True,
            base_url="https://api.mistral.ai/v1",
        )
        with patch("src.ui.unified_provider_tab.ProviderEditDialog", return_value=dlg):
            tab._add_provider()

        self.assertIn("Mistral", self._on_disk())

    def test_cancel_restores_the_provider_list(self):
        from src.utils.config_models import UnifiedProvider

        tab = self._tab()
        tab.unified_config.providers.append(
            UnifiedProvider(name="Weg-damit", provider_type="ollama", enabled=True)
        )
        tab._persist_providers()
        self.assertIn("Weg-damit", self._on_disk())

        self.assertTrue(tab.restore_providers_at_open())
        self.assertNotIn("Weg-damit", self._on_disk())
        self.assertEqual(self._on_disk(), ["ollama"])

    def test_cancel_without_changes_writes_nothing(self):
        tab = self._tab()
        before = self.tmp.read_text()
        self.assertFalse(tab.restore_providers_at_open())
        self.assertEqual(self.tmp.read_text(), before)

    def test_only_providers_are_written(self):
        """Not the whole editing copy: unsaved edits elsewhere must not leak.

        ``autosave_dir`` on purpose. ``contact_email`` looks like the obvious
        probe and is useless here: ``save_config`` runs ``derive_input_mirrors``,
        which recomputes it from the DOI plugin instances, so it reads back
        unchanged whatever the caller passed. - Claude Generated
        """
        from src.utils.config_models import UnifiedProvider

        tab = self._tab()
        tab.config.system_config.autosave_dir = "/ungespeichert/results"
        tab.unified_config.providers.append(
            UnifiedProvider(name="Neu", provider_type="ollama", enabled=True)
        )
        tab._persist_providers()
        import json

        on_disk = json.loads(self.tmp.read_text())
        self.assertEqual(
            on_disk["system_config"]["autosave_dir"], "/vorher/results"
        )
        self.assertIn("Neu", self._on_disk())


class SaveConfigIsCalledWithAConfigTest(unittest.TestCase):
    """`save_config()` with no argument is a TypeError, swallowed by an except.

    Two call sites did it, so "🎯 Als Standardeinstellung speichern" in the
    pipeline config dialog and the CLI's provider-preference learning both
    reported a save error every time and never worked. - Claude Generated
    """

    def test_no_call_site_omits_the_config(self):
        import re
        from pathlib import Path

        offenders = []
        for path in Path("src").rglob("*.py"):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if re.search(r"config_manager\.save_config\(\s*\)", line):
                    offenders.append(f"{path}:{i}")
        self.assertEqual(offenders, [])


class ChatUsesTheConfiguredIterationLimitTest(unittest.TestCase):
    """CLI/GUI parity: the GUI hardcoded max_iterations=30.

    headless_agent:156 reads ChatConfig.max_iterations, so the setting applied to
    CLI and webapp but not to the GUI chat. - Claude Generated
    """

    def test_the_gui_does_not_hardcode_the_limit(self):
        import inspect

        from src.ui import _chat_panel_chat_agent as mod

        lines = [
            ln.strip()
            for ln in inspect.getsource(mod).splitlines()
            if "max_iterations" in ln and not ln.lstrip().startswith("#")
        ]
        self.assertTrue(lines, "keine max_iterations-Zuweisung gefunden")
        self.assertNotIn("max_iterations=30,", lines)
        self.assertTrue(
            any("getattr(chat_config" in ln for ln in lines),
            f"max_iterations kommt nicht aus der Konfiguration: {lines}",
        )


class EveryWidgetTheSavePathTouchesExistsTest(_DialogHarness, unittest.TestCase):
    """Moving a control between tabs must not lose it.

    The tabs are grouped by the job at hand (Modelle / Datenquellen /
    Erschließung / Programm), assembled from per-group builders. Load and save
    address the widgets by attribute, so a group builder that is written but
    never composed into a tab — or a widget dropped while splitting a group —
    surfaces here as a missing attribute rather than as an AttributeError in
    front of the operator. - Claude Generated
    """

    def _referenced_attributes(self):
        import inspect
        import re

        from src.ui.comprehensive_settings_dialog import ComprehensiveSettingsDialog

        src = (
            inspect.getsource(ComprehensiveSettingsDialog._load_current_settings)
            + inspect.getsource(ComprehensiveSettingsDialog._get_config_from_ui)
        )
        return sorted(set(re.findall(r"self\.([a-z_][a-z0-9_]*)", src)))

    def test_no_referenced_widget_is_missing(self):
        dlg = self._dialog()
        names = self._referenced_attributes()
        self.assertGreater(len(names), 20, "Attributsuche hat offenbar nichts gefunden")
        missing = [n for n in names if not hasattr(dlg, n)]
        self.assertEqual(missing, [], f"nicht eingehängt: {missing}")

    def test_the_five_pages_are_present(self):
        dlg = self._dialog()
        titles = [dlg.shell.nav.item(i).text() for i in range(dlg.shell.nav.count())]
        for expected in ("Sprachmodelle", "Erschließung und Workflows",
                         "Datenquellen", "Daten und Speicher", "Anwendung"):
            self.assertTrue(
                any(expected in t for t in titles), f"{expected} fehlt in {titles}"
            )

    def test_the_dialog_frame_has_no_tabs_of_its_own(self):
        """The whole point of the sidebar: one navigation level.

        Sub-widgets brought in from elsewhere may still carry tabs of their own;
        what must not come back is a tab bar for the dialog's own pages.
        """
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QTabWidget

        dlg = self._dialog()
        self.assertFalse(hasattr(dlg, "tab_widget"))
        direct = dlg.shell.findChildren(
            QTabWidget, options=Qt.FindChildOption.FindDirectChildrenOnly
        )
        self.assertEqual(direct, [])

    def test_every_page_is_represented_in_the_search_index(self):
        """A page nobody can search is a page nobody finds."""
        dlg = self._dialog()
        pages = set(dlg.index.pages())
        for expected in ("Sprachmodelle", "Erschließung und Workflows",
                         "Datenquellen", "Daten und Speicher", "Anwendung"):
            self.assertIn(expected, pages)

    def test_the_footer_names_the_configuration_file(self):
        dlg = self._dialog()
        self.assertIn("config.json", dlg.shell.footer.text())

    def test_a_full_save_still_runs(self):
        """End to end over the regrouped dialog."""
        cfg = self._dialog()._get_config_from_ui()
        self.assertEqual(cfg.system_config.url_fetch_allowlist, ["intranet.example"])
        self.assertEqual(cfg.ui_config.ui_language, "en")


class NoConfigScopeFictionTest(unittest.TestCase):
    """There is one config file; nothing may offer a choice of "scope".

    The dialog had a "Save Configuration To" group with Project/User/System
    checkboxes and five CLI subcommands took --scope, all of it handed to a
    save_config parameter that was never read — the CLI even printed "saved to
    user scope" for a write that always went to the same file. - Claude Generated
    """

    def test_save_config_takes_no_scope(self):
        import inspect

        from src.utils.config_manager import ConfigManager

        params = inspect.signature(ConfigManager.save_config).parameters
        self.assertNotIn("scope", params)

    def test_config_info_reports_one_file(self):
        from src.utils.config_manager import ConfigManager

        info = ConfigManager().get_config_info()
        self.assertEqual(set(info), {"os", "config_file"})

    def test_no_cli_subcommand_offers_a_scope(self):
        from src.cli.main import create_argument_parser

        found = []

        def walk(parser, path=""):
            for action in parser._actions:
                if "--scope" in getattr(action, "option_strings", []):
                    found.append(path)
                choices = getattr(action, "choices", None)
                if isinstance(choices, dict):
                    for name, sub in choices.items():
                        walk(sub, f"{path} {name}".strip())

        walk(create_argument_parser())
        self.assertEqual(found, [])


class WorkflowsAreEditableWhereTheyAreChosenTest(_DialogHarness, unittest.TestCase):
    """Die Erschließungsseite trug eine Auswahlliste von Dateinamen.

    Die Workflows sind der Gegenstand der Erschließung: sie stehen jetzt mit
    ihrem eigenen ``name:`` in einer Liste, der Standard ist mit ⭐ markiert, und
    ein Doppelklick öffnet denselben Workflow-Editor, den auch das Menü öffnet.
    Gespeichert wird weiterhin der Dateiname (``SystemConfig.default_workflow``)
    — der Name ist Anzeige, kein Schlüssel. - Claude Generated
    """

    def setUp(self):
        import tempfile
        from pathlib import Path

        self._dir = tempfile.TemporaryDirectory()
        base = Path(self._dir.name)
        (base / "tiefe.yaml").write_text(
            'name: Tiefenrecherche\n'
            'version: "1.2"\n'
            'description: |\n'
            '  Sucht breit\n'
            '  und prueft nach.\n'
            'steps:\n'
            '  - id: one\n'
            '    type: llm_agent\n'
            '  - id: two\n'
            '    type: llm_agent\n',
            encoding="utf-8",
        )
        # v3: kein "steps:" — wird übersprungen, nicht angezeigt.
        (base / "alt.yaml").write_text(
            'name: Alt\npipeline:\n  - schritt\n', encoding="utf-8"
        )
        self._files = [base / "alt.yaml", base / "tiefe.yaml"]

    def tearDown(self):
        self._dir.cleanup()

    def _dlg(self, default="tiefe"):
        with patch(
            "src.core.agents.workflow_loader.discover_workflow_files",
            return_value=self._files,
        ), patch(
            "src.ui.settings.workflow_list.discover_workflow_files",
            return_value=self._files,
            create=True,
        ):
            dlg = self._dialog()
            dlg._refresh_workflow_list(default_stem=default)
        return dlg

    def test_the_page_lists_the_workflows_by_their_own_name(self):
        rows = self._dlg().workflow_list.row_texts()
        names = [r[1] for r in rows]
        self.assertIn("Tiefenrecherche", names)
        for name in names:
            self.assertNotIn("tiefe", name)

    def test_the_default_is_marked_with_a_star(self):
        rows = self._dlg().workflow_list.row_texts()
        starred = [r[1] for r in rows if r[0] == "⭐"]
        self.assertEqual(starred, ["Tiefenrecherche"])

    def test_the_row_names_version_steps_and_where_it_lives(self):
        rows = {r[1]: r for r in self._dlg().workflow_list.row_texts()}
        _, _, version, steps, location = rows["Tiefenrecherche"]
        self.assertEqual(version, "v1.2")
        self.assertEqual(steps, "2")
        self.assertEqual(location, "Projekt")

    def test_the_summary_follows_the_selection(self):
        dlg = self._dlg()
        dlg.workflow_list.select("tiefe")
        dlg.workflow_list._on_selection()
        agentic = dlg.workflow_list.summary.text()
        self.assertIn("Sucht breit und prueft nach.", agentic)

        dlg.workflow_list.select("__classic__")
        dlg.workflow_list._on_selection()
        self.assertNotEqual(dlg.workflow_list.summary.text(), agentic)

    def test_the_saved_value_is_the_file_stem(self):
        dlg = self._dlg()
        self.assertEqual(
            dlg._get_config_from_ui().system_config.default_workflow, "tiefe"
        )

    def test_what_is_saved_is_the_star_not_the_selection(self):
        """Hineinsehen darf den Standard nicht verschieben."""
        dlg = self._dlg(default="tiefe")
        dlg.workflow_list.select("__classic__")
        self.assertEqual(dlg.workflow_list.selected_stem(), "__classic__")
        self.assertEqual(
            dlg._get_config_from_ui().system_config.default_workflow, "tiefe"
        )

    def test_the_configured_default_is_where_the_star_starts(self):
        """Ohne eigenes ``_refresh_workflow_list``: der Dialog liest es selbst
        aus ``SystemConfig.default_workflow``."""
        self.configure = lambda cfg: setattr(
            cfg.system_config, "default_workflow", "tiefe"
        )
        with patch(
            "src.core.agents.workflow_loader.discover_workflow_files",
            return_value=self._files,
        ):
            dlg = self._dialog()
        self.assertEqual(dlg.workflow_list.default_stem(), "tiefe")
        starred = [r[1] for r in dlg.workflow_list.row_texts() if r[0] == "⭐"]
        self.assertEqual(starred, ["Tiefenrecherche"])

    def test_the_star_can_be_moved_and_is_then_saved(self):
        dlg = self._dlg()
        dlg.workflow_list.select("__classic__")
        dlg.workflow_list._make_default()
        self.assertEqual(
            dlg._get_config_from_ui().system_config.default_workflow, "__classic__"
        )

    def _double_click(self, dlg, row_name):
        """Double-click a row with both real editors replaced by recorders.

        Both are patched, never only one: a modal ``exec()`` that slips through
        hangs the suite instead of failing it. - Claude Generated
        """
        opened = []

        class _FakeEditor:
            saved = MagicMock()

            def __init__(self, _parent, initial_workflow=None):
                opened.append(initial_workflow)

            def exec(self):
                return 0

        names = [r[1] for r in dlg.workflow_list.row_texts()]
        item = dlg.workflow_list.list.topLevelItem(names.index(row_name))
        with patch(
            "src.ui.workflow_editor_dialog.WorkflowEditorDialog", _FakeEditor
        ), patch(
            "src.ui.pipeline_config_dialog.PipelineConfigDialog"
        ):
            dlg.workflow_list._on_double_click(item)
        return opened

    def test_a_double_click_opens_the_editor_on_that_workflow(self):
        """Das ist "mit bearbeitbar": derselbe Editor, den auch das Menü öffnet."""
        dlg = self._dlg()
        self.assertEqual(self._double_click(dlg, "Tiefenrecherche"), ["tiefe"])

    def test_the_classic_pipeline_is_listed_with_its_own_editor(self):
        """Sie hat keine YAML, aber dieselbe Frage: welches Modell macht was."""
        dlg = self._dlg()
        names = [r[1] for r in dlg.workflow_list.row_texts()]
        classic = next(n for n in names if "Klassische Pipeline" in n)

        # Nicht der Workflow-Editor: der bekäme hier nichts zu öffnen.
        self.assertEqual(self._double_click(dlg, classic), [])

        dlg.workflow_list.select("__classic__")
        dlg.workflow_list._on_selection()
        self.assertTrue(dlg.workflow_list.edit_button.isEnabled())

    def test_the_classic_row_opens_the_pipeline_step_dialog(self):
        """Und nicht den YAML-Editor: für die klassische Pipeline gibt es keine
        Datei, er hätte nichts zu öffnen."""
        dlg = self._dlg()
        dlg.alima_manager = MagicMock()
        pipeline_opened, yaml_opened = [], []

        class _FakePipelineDialog:
            config_saved = MagicMock()

            def __init__(self, **kw):
                pipeline_opened.append(kw)

            def exec(self):
                return 0

        class _FakeYamlEditor:
            saved = MagicMock()

            def __init__(self, _parent, initial_workflow=None):
                yaml_opened.append(initial_workflow)

            def exec(self):
                return 0

        with patch(
            "src.ui.pipeline_config_dialog.PipelineConfigDialog", _FakePipelineDialog
        ), patch(
            "src.ui.workflow_editor_dialog.WorkflowEditorDialog", _FakeYamlEditor
        ):
            dlg._edit_workflow("__classic__")
        self.assertEqual(yaml_opened, [])
        self.assertEqual(len(pipeline_opened), 1)
        self.assertIs(
            pipeline_opened[0]["llm_service"], dlg.alima_manager.llm_service
        )

    def test_without_a_running_application_it_says_so_instead_of_half_working(self):
        dlg = self._dlg()
        dlg.alima_manager = None
        with patch(
            "src.ui.comprehensive_settings_dialog.QMessageBox"
        ) as message, patch(
            "src.ui.pipeline_config_dialog.PipelineConfigDialog"
        ) as never, patch(
            "src.ui.workflow_editor_dialog.WorkflowEditorDialog"
        ) as never_yaml:
            dlg._edit_workflow("__classic__")
        self.assertTrue(message.information.called)
        self.assertFalse(never.called)
        self.assertFalse(never_yaml.called)

    def test_what_the_pipeline_dialog_wrote_is_not_overwritten_on_save(self):
        """Es schreibt selbst auf die Platte, dieser Dialog bearbeitet eine
        Kopie — ohne Übernahme setzte "Speichern" es zurück."""
        dlg = self._dlg()
        dlg.config_to_edit.unified_config.pipeline_default_provider = "alt"
        dlg.config_to_edit.unified_config.pipeline_default_model = "alt-m"

        saved = dlg.config_manager.load_config()
        saved.unified_config.pipeline_default_provider = "neu"
        saved.unified_config.pipeline_default_model = "neu-m"

        dlg._adopt_pipeline_defaults()
        result = dlg._get_config_from_ui().unified_config
        self.assertEqual(result.pipeline_default_provider, "neu")
        self.assertEqual(result.pipeline_default_model, "neu-m")

    def test_the_new_button_opens_an_empty_editor(self):
        dlg = self._dlg()
        opened = []

        class _FakeEditor:
            saved = MagicMock()

            def __init__(self, _parent, initial_workflow=None):
                opened.append(initial_workflow)

            def exec(self):
                return 0

        with patch(
            "src.ui.workflow_editor_dialog.WorkflowEditorDialog", _FakeEditor
        ), patch(
            "src.ui.pipeline_config_dialog.PipelineConfigDialog"
        ):
            dlg.workflow_list.new_button.click()
        self.assertEqual(opened, [None])

    def test_a_non_v4_file_is_skipped_quietly(self):
        stems = [
            e.stem for e in self._dlg().workflow_list._entries
        ]
        self.assertNotIn("alt", stems)
        self.assertIn("tiefe", stems)


class NoWorkflowStepNamesItsOwnModelTest(unittest.TestCase):
    """Der Dialog behauptete das Gegenteil, und die Behauptung war falsch.

    Die Modelltabelle trug den Hinweis, ein agentischer Schritt könne in seiner
    YAML ein eigenes Modell nennen. Weder ``LLMAgentStep._llm_params`` noch
    ``ReflectionStep`` lesen ``provider``/``model`` aus dem ``llm:``-Block; beide
    nehmen sie aus dem Lauf-Kontext. Dieser Test hält fest, dass der Hinweis erst
    zurückkommt, wenn der Code ihn einlöst. - Claude Generated
    """

    def test_llm_agent_step_takes_provider_and_model_from_the_context(self):
        from src.core.agents.steps.llm_agent_step import LLMAgentStep

        class _Ctx:
            provider = "aus-dem-kontext"
            model = "m-kontext"

        params = LLMAgentStep._llm_params(
            None,
            {"llm": {"provider": "aus-der-yaml", "model": "m-yaml"}},
            _Ctx(),
        )
        self.assertEqual(params["provider"], "aus-dem-kontext")
        self.assertEqual(params["model"], "m-kontext")

    def test_no_settings_surface_promises_a_per_step_model(self):
        """Die Behauptung darf auch nicht an anderer Stelle wieder auftauchen."""
        import pathlib

        surfaces = sorted(pathlib.Path("src/ui/settings").glob("*.py"))
        surfaces.append(pathlib.Path("src/ui/comprehensive_settings_dialog.py"))
        for module in surfaces:
            text = module.read_text(encoding="utf-8")
            for claim in ("llm.provider", "llm.model", "Modell je Schritt",
                          "eigenes Modell nennen", "WORKFLOW_NOTE"):
                self.assertNotIn(claim, text, f"{module.name}: {claim}")


if __name__ == "__main__":
    unittest.main()
