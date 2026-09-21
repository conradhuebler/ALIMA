"""The table that answers "which model runs, and why".

It sits first on the page because it is a consequence, not an input: four
settings can decide the model, and which one applied used to be invisible.
"Ich stelle X ein und bekomme Y" had no single cause and no surface that could
answer it.

It never probes the network on open. Four providers would be four calls before
the dialog is usable, and a dead endpoint hangs it. - Claude Generated
"""

from __future__ import annotations

import unittest

from src.utils.config_models import UnifiedProvider, UnifiedProviderConfig


def _qapp(keep: list):
    from PyQt6.QtWidgets import QApplication

    if not keep:
        keep.append(QApplication.instance() or QApplication(["alima-tests"]))
    return keep[0]


def _config(**kw) -> UnifiedProviderConfig:
    cfg = UnifiedProviderConfig(providers=[
        UnifiedProvider(name="A", provider_type="ollama", enabled=True,
                        preferred_model="a1"),
        UnifiedProvider(name="B", provider_type="ollama", enabled=True,
                        preferred_model="b1"),
    ])
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class ResolutionViewTest(unittest.TestCase):
    _kept: list = []

    def _view(self, cfg):
        from src.ui.settings import ResolutionView

        _qapp(self._kept)
        view = ResolutionView()
        view.refresh(cfg)
        self._kept.append(view)
        return view

    def test_there_is_one_row_per_axis(self):
        view = self._view(_config())
        self.assertEqual(len(view.row_texts()), 4)
        labels = [r[0] for r in view.row_texts()]
        for expected in ("Allgemein", "Pipeline", "Agentisch", "Bild"):
            self.assertTrue(any(expected in l for l in labels), labels)

    def test_it_names_the_model_and_the_source(self):
        view = self._view(_config(preferred_provider="A", preferred_model="a1"))
        general = view.row_texts()[0]
        self.assertIn("A", general[1])
        self.assertIn("a1", general[1])
        self.assertIn("Allgemeiner Standard", general[2])

    def test_an_inherited_row_says_so(self):
        """The point: you can see that this scope did not choose for itself."""
        view = self._view(_config(preferred_provider="A", preferred_model="a1"))
        pipeline = view.row_texts()[1]
        self.assertIn("geerbt", pipeline[2])

    def test_an_own_choice_is_not_marked_inherited(self):
        view = self._view(_config(
            preferred_provider="A", preferred_model="a1",
            pipeline_default_provider="B", pipeline_default_model="b1",
        ))
        pipeline = view.row_texts()[1]
        self.assertIn("B", pipeline[1])
        self.assertNotIn("geerbt", pipeline[2])

    def test_the_vision_row_follows_the_vision_default(self):
        view = self._view(_config(
            preferred_provider="A", preferred_model="a1",
            vision_default_provider="B", vision_default_model="b-vision",
        ))
        vision = view.row_texts()[3]
        self.assertIn("b-vision", vision[1])
        self.assertIn("Vision", vision[2])

    def test_nothing_configured_shows_a_dash_not_a_guess(self):
        view = self._view(UnifiedProviderConfig(providers=[]))
        for row in view.row_texts():
            self.assertEqual(row[1], "—")

    def test_the_workflow_override_is_stated(self):
        view = self._view(_config())
        self.assertIn("YAML", view.note.text())

    def test_refresh_replaces_instead_of_appending(self):
        view = self._view(_config())
        view.refresh(_config())
        self.assertEqual(len(view.row_texts()), 4)

    def test_a_broken_config_does_not_raise(self):
        """The table must never be the thing that stops the dialog opening."""
        class _Broken:
            def explain_default_provider_model(self, **kw):
                raise RuntimeError("kaputt")

        view = self._view(_Broken())
        self.assertEqual(view.row_texts(), [])


class TheTableIsAlsoTheControlTest(unittest.TestCase):
    """Showing the four values and offering the same four as pickers below would
    double them. Each row carries its own picker instead. - Claude Generated"""

    _kept: list = []

    def _view(self, cfg):
        from src.ui.settings import ResolutionView

        _qapp(self._kept)
        view = ResolutionView()
        view.refresh(cfg)
        self._kept.append(view)
        return view

    def test_every_axis_has_a_picker(self):
        view = self._view(_config())
        self.assertEqual(
            sorted(view._selectors), ["agentic", "general", "pipeline", "vision"]
        )

    def test_vision_is_reachable(self):
        """It was introduced as a setting without any control at all."""
        self.assertIn("vision", self._view(_config())._selectors)

    def test_the_general_default_has_no_empty_option(self):
        """Everything falls back to it, so it must stay concrete.

        An empty option here would let the operator unset the one value that has
        nothing behind it.
        """
        from src.ui.settings.resolution_view import ResolutionView

        self.assertIsNone(ResolutionView.EMPTY_LABELS["general"])
        for scope in ("pipeline", "agentic", "vision"):
            self.assertIsNotNone(ResolutionView.EMPTY_LABELS[scope], scope)

    def test_a_pick_reports_its_scope(self):
        view = self._view(_config())
        seen = []
        view.selection_changed.connect(lambda s, p, m: seen.append((s, p, m)))
        view._on_pick("vision", "B", "b-vision")
        self.assertEqual(seen, [("vision", "B", "b-vision")])

    def test_programmatic_population_does_not_report_a_pick(self):
        """Otherwise loading the dialog would look like an edit and mark it dirty."""
        view = self._view(_config())
        seen = []
        view.selection_changed.connect(lambda *a: seen.append(a))
        view._loading = True
        view._on_pick("vision", "B", "b-vision")
        self.assertEqual(seen, [])


class ItDoesNotTouchTheNetworkTest(unittest.TestCase):
    """Four probes before the dialog is usable, and a dead endpoint hangs it."""

    def test_refresh_makes_no_reachability_call(self):
        import inspect

        from src.ui.settings import resolution_view

        src = inspect.getsource(resolution_view)
        for forbidden in ("is_provider_reachable", "get_available_models",
                          "ProviderDetectionService"):
            self.assertNotIn(forbidden, src)


if __name__ == "__main__":
    unittest.main()
