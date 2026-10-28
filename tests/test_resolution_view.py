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
        self.assertEqual(len(view.row_texts()), 3)
        labels = [r[0] for r in view.row_texts()]
        for expected in ("Allgemein", "Pipeline", "Agentisch", "Bild"):
            self.assertTrue(any(expected in l for l in labels), labels)

    def test_it_names_the_model_and_the_source(self):
        view = self._view(_config(preferred_provider="A", preferred_model="a1"))
        general = view.row_texts()[0]
        self.assertEqual(general[1], "A")
        self.assertEqual(general[2], "a1")
        self.assertIn("Allgemeiner Standard", general[3])

    def test_an_inherited_row_says_so(self):
        """The point: you can see that this scope did not choose for itself."""
        view = self._view(_config(preferred_provider="A", preferred_model="a1"))
        self.assertIn("geerbt", view.row_texts()[1][3])

    def test_an_own_choice_is_not_marked_inherited(self):
        view = self._view(_config(
            preferred_provider="A", preferred_model="a1",
            pipeline_default_provider="B", pipeline_default_model="b1",
        ))
        pipeline = view.row_texts()[1]
        self.assertEqual(pipeline[1], "B")
        self.assertNotIn("geerbt", pipeline[3])

    def test_the_vision_row_follows_the_vision_default(self):
        view = self._view(_config(
            preferred_provider="A", preferred_model="a1",
            vision_default_provider="B", vision_default_model="b-vision",
        ))
        vision = view.row_texts()[2]
        self.assertEqual(vision[2], "b-vision")
        self.assertIn("Vision", vision[3])

    def test_nothing_configured_shows_a_dash_not_a_guess(self):
        view = self._view(UnifiedProviderConfig(providers=[]))
        for row in view.row_texts():
            self.assertEqual(row[1], "—")
            self.assertEqual(row[2], "")

    def test_refresh_replaces_instead_of_appending(self):
        view = self._view(_config())
        view.refresh(_config())
        self.assertEqual(len(view.row_texts()), 3)

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
            sorted(view._selectors), ["general", "pipeline", "vision"]
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
        for scope in ("pipeline", "vision"):
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

    def test_the_model_column_width_does_not_depend_on_the_model_name(self):
        """The "Modell" column used ResizeToContents - correct behaviour for
        that mode, but it meant the whole table visibly resized every time a
        differently-sized model name was selected (e.g. "a1" vs. a long
        Ollama tag like "deepseek-v4-flash:cloud"). Now fixed-width with a
        tooltip for the full name, like the rest of this view."""
        from src.utils.config_models import UnifiedProvider, UnifiedProviderConfig

        def _cfg(model):
            c = UnifiedProviderConfig(providers=[
                UnifiedProvider(name="A", provider_type="ollama", enabled=True,
                                preferred_model=model),
            ])
            c.preferred_provider = "A"
            c.preferred_model = model
            return c

        view = self._view(_cfg("m1"))
        widths_short = [view.table.columnWidth(c) for c in range(5)]

        long_model = "a-very-much-longer-model-name-than-before:latest-cloud-edition-xl"
        view.refresh(_cfg(long_model))
        widths_long = [view.table.columnWidth(c) for c in range(5)]

        self.assertEqual(widths_short, widths_long)
        self.assertEqual(view.table.item(0, 2).toolTip(), long_model)

    def test_the_model_picker_is_not_editable(self):
        """An editable QComboBox sizes itself from its current text, not
        from the full item list - picking a model with a different name
        length visibly resized the column every time. A plain dropdown's
        size hint comes from the widest item in the list, which does not
        change when the selection does; the exact configured model is shown
        regardless, in the separate "Modell" text column."""
        view = self._view(_config(preferred_provider="A", preferred_model="a1"))
        for scope, selector in view._selectors.items():
            self.assertFalse(selector.model_combo.isEditable(), scope)


class ItFitsAndItsPickersAreUsableTest(unittest.TestCase):
    """Defects found by actually using it. - Claude Generated"""

    _kept: list = []

    def _view(self, cfg):
        from src.ui.settings import ResolutionView

        _qapp(self._kept)
        view = ResolutionView()
        view.refresh(cfg)
        self._kept.append(view)
        return view

    def test_the_provider_combo_is_populated(self):
        """It came up empty while the model combo worked.

        The pickers are created on first refresh, so a caller that pushed the
        provider list in beforehand set it on nothing. The table derives the list
        itself now.
        """
        view = self._view(_config())
        for scope, selector in view._selectors.items():
            self.assertGreater(
                selector.provider_combo.count(), 0, f"{scope} ohne Anbieter"
            )

    def test_only_the_general_scope_lacks_an_empty_entry(self):
        view = self._view(_config())
        counts = {s: sel.provider_combo.count() for s, sel in view._selectors.items()}
        self.assertEqual(counts["general"], 2)
        for scope in ("pipeline", "vision"):
            self.assertEqual(counts[scope], 3, scope)

    def test_rows_are_uniform_and_fit_their_picker(self):
        """resizeRowsToContents measured the text cells and ignored the combo.

        The model names differ wildly in length on purpose: with similar ones the
        text-measured heights come out equal by accident and the test passes
        without holding anything. On the real config they were 35/35/35/26 with a
        25 px combo squeezed into the last row.
        """
        cfg = _config(
            preferred_provider="A",
            preferred_model="ein-ausgesprochen-langer-modellname-der-umbricht-2603",
            pipeline_default_provider="B", pipeline_default_model="m",
            vision_default_provider="A", vision_default_model="x",
        )
        view = self._view(cfg)
        heights = [view.table.rowHeight(r) for r in range(view.table.rowCount())]
        self.assertEqual(len(set(heights)), 1, f"ungleiche Zeilen: {heights}")
        for r in range(view.table.rowCount()):
            widget = view.table.cellWidget(r, 4)
            self.assertGreaterEqual(heights[r], widget.sizeHint().height())

    def test_the_table_is_tall_enough_for_its_pickers(self):
        """Its maximum was a little *below* what the pickers needed, so the last
        one was clipped.

        Asserted as the margin the calculation guarantees, not as a pixel count:
        the real clipping depends on the widget style, which is not loaded in a
        headless run — a number measured here would pass while the styled dialog
        still clips. What is checkable is that every row leaves room around its
        picker and that the table covers all rows plus its header.
        """
        view = self._view(_config())
        self.assertEqual(view.table.minimumHeight(), view.table.maximumHeight())

        for r in range(view.table.rowCount()):
            widget = view.table.cellWidget(r, 4)
            self.assertGreaterEqual(
                view.table.rowHeight(r), widget.sizeHint().height() + 6,
                f"Zeile {r} lässt dem Auswahlfeld keinen Rand",
            )

        rows = sum(view.table.rowHeight(r) for r in range(view.table.rowCount()))
        header = view.table.horizontalHeader()
        self.assertGreaterEqual(
            view.table.maximumHeight(),
            rows + max(header.height(), header.sizeHint().height()),
        )


class ItDoesNotTouchTheNetworkTest(unittest.TestCase):
    """Four probes before the dialog is usable, and a dead endpoint hangs it."""

    def test_refresh_makes_no_reachability_call(self):
        import inspect

        from src.ui.settings import resolution_view

        src = inspect.getsource(resolution_view)
        for forbidden in ("is_provider_reachable", "get_available_models",
                          "ProviderDetectionService"):
            self.assertNotIn(forbidden, src)


class NothingOnThisViewStretchesTest(unittest.TestCase):
    """Die Notizzeile unter der Tabelle war das einzige Element, das beim
    Vergrößern des Dialogs mitwuchs — in leeren Raum, auf einer Seite, deren
    Inhalt feste Höhe hat. Sie ist weg, und nichts tritt an ihre Stelle.
    - Claude Generated
    """

    _kept: list = []

    def _view(self):
        from src.ui.settings import ResolutionView

        _qapp(self._kept)
        view = ResolutionView()
        view.refresh(_config())
        self._kept.append(view)
        return view

    def test_the_table_is_the_only_widget_in_the_box(self):
        view = self._view()
        layout = view.layout()
        widgets = [layout.itemAt(i).widget() for i in range(layout.count())]
        self.assertEqual(widgets, [view.table])

    def test_the_view_does_not_claim_height_it_cannot_fill(self):
        from PyQt6.QtWidgets import QSizePolicy

        self.assertEqual(
            self._view().sizePolicy().verticalPolicy(), QSizePolicy.Policy.Maximum
        )

    def test_extra_space_in_the_parent_does_not_reach_the_view(self):
        """Der eigentliche Befund, gemessen statt behauptet.

        Die Tabelle allein sagt nichts: sie hat feste Höhe und wächst auch dann
        nicht, wenn die Gruppe darum herum wächst. Gemessen wird deshalb die
        Gruppe in einem Layout mit 300 px übrigem Platz.
        """
        from PyQt6.QtWidgets import QVBoxLayout, QWidget

        view = self._view()
        host = QWidget()
        box = QVBoxLayout(host)
        box.addWidget(view)
        self._kept.append(host)

        wanted = view.sizeHint().height()
        host.resize(700, wanted + 300)
        box.activate()
        self.assertLessEqual(view.height(), wanted)


if __name__ == "__main__":
    unittest.main()
