"""LlmService connects providers on first use, not at construction.

``lazy_initialization=True`` was stored and ignored: every configured provider
was connected in the constructor, and the SDK imports (google.genai, anthropic,
openai, ollama) alone took about 5 s of GUI start. Claude Generated.
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock

from src.llm.llm_service import LlmService, ProviderState, _LazyClients

SENTINEL = ProviderState.LAZY_UNINITIALIZED


def _clients(loader=None):
    calls = []

    def default(key):
        calls.append(key)
        clients[key] = f"client-{key}"

    clients = _LazyClients(loader or default)
    return clients, calls


class LazyClientsMapTest(unittest.TestCase):
    def test_registering_and_looking_up_names_does_not_connect(self):
        clients, calls = _clients()
        clients["a"] = SENTINEL
        clients["b"] = SENTINEL
        self.assertIn("a", clients)
        self.assertEqual(sorted(clients.keys()), ["a", "b"])
        self.assertIs(clients.peek("a"), SENTINEL)
        self.assertEqual(calls, [])

    def test_indexing_connects_once(self):
        clients, calls = _clients()
        clients["a"] = SENTINEL
        self.assertEqual(clients["a"], "client-a")
        self.assertEqual(clients["a"], "client-a")
        self.assertEqual(calls, ["a"])

    def test_a_failing_provider_is_absent_afterwards_like_in_eager_mode(self):
        def failing(key):
            pass  # initializer logged an error and stored nothing

        clients, _ = _clients(failing)
        clients["a"] = SENTINEL
        self.assertFalse(clients.resolve("a"))
        self.assertNotIn("a", clients)
        with self.assertRaises(KeyError):
            clients["a"]

    def test_a_raising_loader_also_leaves_no_placeholder(self):
        def boom(key):
            raise RuntimeError("x")

        clients, _ = _clients(boom)
        clients["a"] = SENTINEL
        with self.assertRaises(RuntimeError):
            clients.resolve("a")
        self.assertNotIn("a", clients)

    def test_clear_does_not_connect(self):
        clients, calls = _clients()
        clients["a"] = SENTINEL
        clients.clear()
        self.assertEqual((len(clients), calls), (0, []))

    def test_concurrent_first_access_connects_exactly_once(self):
        calls = []

        def slow(key):
            calls.append(key)
            time.sleep(0.1)
            clients[key] = object()

        clients = _LazyClients(slow)
        clients["a"] = SENTINEL
        seen = []
        threads = [threading.Thread(target=lambda: seen.append(clients["a"])) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(calls, ["a"])
        self.assertEqual(len({id(x) for x in seen}), 1)

    def test_an_already_connected_client_is_returned_without_the_loader(self):
        clients, calls = _clients()
        clients["a"] = "real"
        self.assertEqual(clients["a"], "real")
        self.assertEqual(calls, [])


class ServiceSetupTest(unittest.TestCase):
    def _svc(self, lazy):
        svc = LlmService.__new__(LlmService)
        svc.lazy_initialization = lazy
        svc.logger = MagicMock()
        svc.supported_providers = {"p1": {}, "p2": {}}
        svc.alima_config = MagicMock()
        svc.clients = _LazyClients(MagicMock())
        svc.initialize_providers = MagicMock()
        return svc

    def test_lazy_mode_registers_placeholders_and_connects_nothing(self):
        svc = self._svc(lazy=True)
        svc._setup_clients(None)
        svc.initialize_providers.assert_not_called()
        self.assertEqual({k: svc.clients.peek(k) for k in svc.clients.keys()},
                         {"p1": SENTINEL, "p2": SENTINEL})

    def test_eager_mode_connects_as_before(self):
        svc = self._svc(lazy=False)
        svc._setup_clients(None)
        svc.initialize_providers.assert_called_once_with(None)

    def test_ensure_provider_initialized_goes_through_the_lazy_map(self):
        svc = self._svc(lazy=True)
        svc._map_provider_name = lambda p: p

        def init(key):
            svc.clients[key] = "client"

        svc.clients = _LazyClients(init)
        svc.clients["p1"] = SENTINEL
        self.assertTrue(svc._ensure_provider_initialized("p1"))
        self.assertEqual(svc.clients.peek("p1"), "client")

    def test_ensure_provider_initialized_reports_failure(self):
        svc = self._svc(lazy=True)
        svc._map_provider_name = lambda p: p
        svc.clients = _LazyClients(lambda key: None)
        svc.clients["p1"] = SENTINEL
        self.assertFalse(svc._ensure_provider_initialized("p1"))


class ConstructionIntegrationTest(unittest.TestCase):
    """Die echte Konstruktion mit lazy_initialization=True importiert keine SDKs."""

    def test_construction_registers_without_connecting(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from src.utils.config_manager import ConfigManager
        from src.utils.config_models import UnifiedProvider

        with tempfile.TemporaryDirectory() as tmp:
            manager = ConfigManager()
            original = manager.config_file
            original_config = getattr(manager, "config", None)
            manager.config_file = Path(tmp) / "config.json"
            try:
                cfg = manager.load_config(force_reload=True)
                cfg.unified_config.providers = [
                    UnifiedProvider(name="a", provider_type="anthropic", api_key="k" * 20),
                    UnifiedProvider(name="g", provider_type="gemini", api_key="k" * 20),
                ]
                with patch.object(LlmService, "_initialize_single_provider") as init:
                    svc = LlmService(config_manager=manager, lazy_initialization=True)
                init.assert_not_called()
                self.assertEqual(sorted(svc.clients.keys()), ["a", "g"])
                self.assertTrue(all(svc.clients.peek(k) is SENTINEL for k in ("a", "g")))
            finally:
                manager.config_file = original
                manager.config = original_config  # Singleton nicht verändert hinterlassen


if __name__ == "__main__":
    unittest.main()
