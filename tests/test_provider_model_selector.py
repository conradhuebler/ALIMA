"""Import smoke test for the shared ProviderModelSelector widget.

Claude Generated (provider-system cleanup, Phase 5).

Headless offscreen Qt in this environment segfaults intermittently when real
QWidgets are *instantiated* (the broader suite mocks widgets for the same
reason). So this only imports the module and checks its public surface — enough
to catch syntax/import regressions. Behavioral/visual verification of the widget
is done by running the GUI (see docs / operator verification).
"""
from __future__ import annotations

import inspect
import unittest


class TestProviderModelSelectorImport(unittest.TestCase):
    def test_module_imports_and_exposes_public_api(self):
        from src.ui.provider_model_selector import ProviderModelSelector

        # Signal + the methods host surfaces rely on must exist.
        self.assertTrue(hasattr(ProviderModelSelector, "selectionChanged"))
        for method in ("set_providers", "load_providers", "set_selection",
                       "get_selection", "set_decorations", "is_model_valid",
                       "refresh_models"):
            self.assertTrue(callable(getattr(ProviderModelSelector, method, None)),
                            f"ProviderModelSelector.{method} missing")

    def test_get_selection_signature_is_stable(self):
        from src.ui.provider_model_selector import ProviderModelSelector
        sig = inspect.signature(ProviderModelSelector.set_selection)
        self.assertEqual(list(sig.parameters)[1:3], ["provider", "model"])


if __name__ == "__main__":
    unittest.main()
