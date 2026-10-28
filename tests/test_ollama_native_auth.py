"""Native Ollama client: the API key reaches the server as "Bearer <key>".

A raw key in the Authorization header is rejected by Ollama Cloud / proxies
with 401, while the OpenAI-compatible path (which sends Bearer) worked.

Claude Generated.
"""
from __future__ import annotations

import logging
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock


class _Capture(BaseHTTPRequestHandler):
    seen = []

    def do_GET(self):
        _Capture.seen.append(self.headers.get("Authorization"))
        body = b'{"models": []}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class TestOllamaNativeBearer(unittest.TestCase):
    def setUp(self):
        _Capture.seen = []
        self.server = HTTPServer(("127.0.0.1", 0), _Capture)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _init_and_list(self, api_key):
        from src.llm.llm_service import LlmService

        svc = MagicMock(spec=LlmService)
        svc.clients = {}
        svc.logger = logging.getLogger("test_ollama_auth")
        cfg = SimpleNamespace(
            base_url=f"http://127.0.0.1:{self.port}", api_key=api_key, enabled=True
        )
        LlmService._init_ollama_native_provider(
            svc, "cloud", None, None, {"config": cfg}
        )
        svc.clients["cloud"].list()
        return _Capture.seen[-1]

    def test_raw_key_gets_bearer_prefix(self):
        self.assertEqual(self._init_and_list("abc123"), "Bearer abc123")

    def test_existing_prefix_not_doubled(self):
        self.assertEqual(self._init_and_list("Bearer abc123"), "Bearer abc123")

    def test_no_key_sends_no_authorization(self):
        self.assertIsNone(self._init_and_list(""))

    def test_setup_validator_sends_bearer(self):
        from src.utils.setup_utils import OllamaConnectionValidator

        OllamaConnectionValidator.test_native("127.0.0.1", self.port, "abc123")
        self.assertEqual(_Capture.seen[-1], "Bearer abc123")


if __name__ == "__main__":
    unittest.main()


class TestOllamaCloudUrl(unittest.TestCase):
    """host=ollama.com + Dialog-Vorgaben darf nie auf http://…:11434 landen."""

    def _norm(self, url):
        from src.llm.llm_service import LlmService

        return LlmService._normalize_ollama_cloud_url(url)

    def test_cloud_forms_become_https_without_port(self):
        for u in ("http://ollama.com:11434", "ollama.com", "https://ollama.com:11434",
                  "http://www.ollama.com", "https://ollama.com"):
            self.assertEqual(self._norm(u), "https://ollama.com", u)

    def test_other_hosts_untouched(self):
        for u in ("http://localhost:11434", "http://gpu.uni.de:11434",
                  "https://notollama.com", "http://127.0.0.1:11434"):
            self.assertEqual(self._norm(u), u, u)

    def test_init_with_dialog_defaults_builds_https_client(self):
        import logging
        from src.llm.llm_service import LlmService

        svc = MagicMock(spec=LlmService)
        svc.clients = {}
        svc.logger = logging.getLogger("t")
        cfg = SimpleNamespace(base_url="", host="ollama.com", port=11434,
                              use_ssl=False, api_key="k", enabled=True)
        LlmService._init_ollama_native_provider(svc, "c", None, None, {"config": cfg})
        client = svc.clients["c"]._client
        self.assertEqual(str(client.base_url), "https://ollama.com")
        self.assertEqual(client.headers["authorization"], "Bearer k")
