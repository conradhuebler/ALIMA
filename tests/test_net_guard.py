"""Tests for src/utils/net_guard.py (two-posture URL validation + guarded fetch)
- Claude Generated."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.utils.net_guard import (
    GuardedResponse,
    assert_public_http_url,
    check_operator_url,
    fetch_guarded,
    require_http_url,
)


class CheckOperatorUrlTest(unittest.TestCase):
    def test_empty_is_fine(self):
        self.assertEqual(check_operator_url(""), [])
        self.assertEqual(check_operator_url(None), [])

    def test_https_public_no_warnings(self):
        self.assertEqual(check_operator_url("https://lobid.org/resources"), [])

    def test_intranet_https_not_flagged(self):
        # University catalogs legitimately live on private addresses.
        self.assertEqual(check_operator_url("https://katalog.intern:8080/api"), [])
        self.assertEqual(check_operator_url("https://192.168.1.10/proxy"), [])

    def test_plain_http_warns(self):
        warnings = check_operator_url("http://katalog.intern/api")
        self.assertTrue(any("unverschlüsselt" in w for w in warnings))

    def test_bad_scheme_warns(self):
        warnings = check_operator_url("file:///etc/passwd")
        self.assertTrue(any("Schema" in w for w in warnings))


class RequireHttpUrlTest(unittest.TestCase):
    def test_accepts_http_and_https(self):
        self.assertEqual(require_http_url("https://a.example/x"), "https://a.example/x")
        self.assertEqual(require_http_url("http://192.168.0.1/x"), "http://192.168.0.1/x")

    def test_rejects_other_schemes_and_no_host(self):
        for bad in ("file:///etc/passwd", "ftp://x/y", "not-a-url", "", "https://"):
            with self.assertRaises(ValueError, msg=bad):
                require_http_url(bad)


def _fake_getaddrinfo(ip):
    return lambda *a, **kw: [(2, 1, 6, "", (ip, 443))]


class AssertPublicHttpUrlTest(unittest.TestCase):
    def test_rejects_bad_scheme(self):
        with self.assertRaises(ValueError):
            assert_public_http_url("file:///etc/passwd")

    def test_public_address_passes(self):
        with patch("socket.getaddrinfo", _fake_getaddrinfo("93.184.216.34")):
            assert_public_http_url("https://example.org/x")

    def test_private_loopback_linklocal_metadata_rejected(self):
        for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1"):
            with patch("socket.getaddrinfo", _fake_getaddrinfo(ip)):
                with self.assertRaises(ValueError, msg=ip):
                    assert_public_http_url("https://evil.example/x")

    def test_allowlist_bypasses_address_check(self):
        with patch("socket.getaddrinfo", _fake_getaddrinfo("127.0.0.1")):
            assert_public_http_url("https://katalog.intern/x", allowlist=["katalog.intern"])
        # but never the scheme check
        with self.assertRaises(ValueError):
            assert_public_http_url("file://katalog.intern/x", allowlist=["katalog.intern"])

    def test_unresolvable_host_rejected(self):
        import socket as socket_mod

        def boom(*a, **kw):
            raise socket_mod.gaierror("nope")

        with patch("socket.getaddrinfo", boom):
            with self.assertRaises(ValueError):
                assert_public_http_url("https://doesnotexist.invalid/")


def _mock_response(status=200, headers=None, chunks=(b"hello",)):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.iter_content = lambda chunk_size: iter(chunks)
    return resp


class FetchGuardedTest(unittest.TestCase):
    def setUp(self):
        patcher = patch("socket.getaddrinfo", _fake_getaddrinfo("93.184.216.34"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_simple_fetch(self):
        with patch("requests.get", return_value=_mock_response()) as rg:
            out = fetch_guarded("https://example.org/x")
        self.assertIsInstance(out, GuardedResponse)
        self.assertEqual(out.content, b"hello")
        self.assertEqual(out.status_code, 200)
        self.assertFalse(rg.call_args.kwargs["allow_redirects"])

    def test_size_cap_enforced(self):
        big = _mock_response(chunks=(b"x" * 100, b"y" * 100))
        with patch("requests.get", return_value=big):
            with self.assertRaises(ValueError):
                fetch_guarded("https://example.org/big", max_bytes=150)

    def test_redirect_hops_are_guarded(self):
        redirect = _mock_response(status=302, headers={"Location": "https://internal.example/"})
        with patch("requests.get", return_value=redirect):
            # every hop re-resolves; make the redirect target resolve private
            with patch(
                "socket.getaddrinfo",
                lambda host, *a, **kw: [(2, 1, 6, "", ("10.0.0.1" if "internal" in host else "93.184.216.34", 443))],
            ):
                with self.assertRaises(ValueError):
                    fetch_guarded("https://example.org/start")

    def test_too_many_redirects(self):
        redirect = _mock_response(status=301, headers={"Location": "https://example.org/next"})
        with patch("requests.get", return_value=redirect):
            with self.assertRaises(RuntimeError):
                fetch_guarded("https://example.org/loop", max_redirects=3)

    def test_raise_for_status(self):
        with patch("requests.get", return_value=_mock_response(status=404)):
            out = fetch_guarded("https://example.org/missing")
        with self.assertRaises(RuntimeError):
            out.raise_for_status()


if __name__ == "__main__":
    unittest.main()
