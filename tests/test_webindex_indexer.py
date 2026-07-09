"""Tests for the website crawler/indexer - Claude Generated.

Netzfrei: ``fetch_func`` is injected so no HTTP happens; ``pdf_extractor.extract_text``
is patched for the PDF case; the LLM keyword extractor uses a fake service.
"""

import os
import tempfile
import unittest
from unittest import mock

try:
    from src.utils.lookups.webindex import indexer
    from src.utils.lookups.webindex.store import WebIndexStore
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "webindex", "site")
BASE = "https://test.local/"


class _Resp:
    def __init__(self, content=b"", content_type="text/html", status=200):
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.status_code = status


def _fixture_bytes(name):
    with open(os.path.join(_FIXTURES, name), "rb") as f:
        return f.read()


def _make_fetch(serves):
    """serves: {url: (bytes|filename, content_type, status)} — filename str reads
    a fixture file."""
    def fetch(url, *, user_agent, timeout):
        if url not in serves:
            raise RuntimeError(f"404 not found: {url}")
        body, ct, status = serves[url]
        if isinstance(body, str):
            body = _fixture_bytes(body)
        return _Resp(body, ct, status)
    return fetch


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class CrawlSiteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.store = WebIndexStore({"db_path": os.path.join(self._tmp, "w.db")},
                                   "webindex_idx_test")

    def tearDown(self):
        try:
            self.store.close()
        except Exception:
            pass

    def _serves(self):
        return {
            BASE: ("index.html", "text/html", 200),
            BASE + "kontakt": ("kontakt.html", "text/html", 200),
            BASE + "team": ("team.html", "text/html", 200),
        }

    def test_crawl_indexes_root_and_kontakt_skips_short_team(self):
        res = indexer.crawl_site(
            self.store, base_url=BASE, max_depth=1, max_pages=20,
            min_chars=50, fetch_func=_make_fetch(self._serves()),
        )
        self.assertIn(BASE, res["indexed_urls"])
        self.assertIn(BASE + "kontakt", res["indexed_urls"])
        # team.html text is < 50 chars → skipped (not indexed), but still visited.
        self.assertNotIn(BASE + "team", res["indexed_urls"])
        self.assertEqual(res["pages_skipped"], 1)

    def test_external_url_not_crawled(self):
        res = indexer.crawl_site(
            self.store, base_url=BASE, max_depth=1, max_pages=20, min_chars=50,
            fetch_func=_make_fetch(self._serves()),
        )
        self.assertFalse(any("other.example.de" in u for u in res["indexed_urls"]))
        self.assertFalse(any("other.example.de" in u for u in res["errors"]))

    def test_max_depth_zero_only_root(self):
        res = indexer.crawl_site(
            self.store, base_url=BASE, max_depth=0, max_pages=20, min_chars=50,
            fetch_func=_make_fetch(self._serves()),
        )
        self.assertEqual(res["indexed_urls"], [BASE])
        self.assertEqual(res["visited"], 1)

    def test_include_exclude_regex(self):
        # Seed is always crawled; include/exclude scope discovered children.
        res = indexer.crawl_site(
            self.store, base_url=BASE, max_depth=1, max_pages=20, min_chars=50,
            include_re=r"kontakt", fetch_func=_make_fetch(self._serves()),
        )
        # root (seed) + kontakt indexed; team excluded (no match).
        self.assertIn(BASE, res["indexed_urls"])
        self.assertIn(BASE + "kontakt", res["indexed_urls"])
        self.assertNotIn(BASE + "team", res["indexed_urls"])
        res2 = indexer.crawl_site(
            self.store, base_url=BASE, max_depth=1, max_pages=20, min_chars=50,
            exclude_re=r"kontakt", fetch_func=_make_fetch(self._serves()),
        )
        # kontakt is the only indexable child; excluding it leaves just the seed.
        self.assertIn(BASE, res2["indexed_urls"])
        self.assertNotIn(BASE + "kontakt", res2["indexed_urls"])

    def test_meta_and_heading_keywords_indexed(self):
        indexer.crawl_site(self.store, base_url=BASE, max_depth=0, max_pages=5,
                           min_chars=50, fetch_func=_make_fetch(self._serves()))
        kw = {k["keyword"] for k in self.store.list_keywords(200)}
        # from meta keywords
        self.assertIn("fernleihe", kw)
        self.assertIn("katalog", kw)
        self.assertIn("öffnungszeiten", kw)
        # from <title>/<h1> headings
        self.assertIn("universitätsbibliothek — startseite", kw)
        self.assertIn("willkommen bei der universitätsbibliothek", kw)

    def test_llm_keywords_added_with_boost(self):
        def extractor(text, max_keywords):
            return ["Erwerbung", "Lizenzierung"]

        res = indexer.crawl_site(
            self.store, base_url=BASE, max_depth=0, max_pages=5, min_chars=50,
            fetch_func=_make_fetch(self._serves()),
            keyword_extractor=extractor, max_keywords=5,
        )
        self.assertEqual(res["pages_indexed"], 1)
        # LLM keywords present + ranking reflects the 1.5 boost.
        hits = self.store.get_pages_for_keywords(["erwerbung"], 5)
        self.assertEqual(len(hits), 1)
        self.assertGreaterEqual(hits[0]["score"], 1.5)

    def test_llm_failure_falls_back_to_meta_only(self):
        def broken_extractor(text, max_keywords):
            raise RuntimeError("boom")

        res = indexer.crawl_site(
            self.store, base_url=BASE, max_depth=0, max_pages=5, min_chars=50,
            fetch_func=_make_fetch(self._serves()),
            keyword_extractor=broken_extractor,
        )
        # crawl still succeeds with meta keywords; no crash.
        self.assertEqual(res["pages_indexed"], 1)
        self.assertTrue(any(k["keyword"] == "fernleihe"
                            for k in self.store.list_keywords(200)))

    def test_keyword_extractor_none_means_meta_only(self):
        res = indexer.crawl_site(
            self.store, base_url=BASE, max_depth=0, max_pages=5, min_chars=50,
            fetch_func=_make_fetch(self._serves()),
            keyword_extractor=None,
        )
        self.assertEqual(res["pages_indexed"], 1)
        self.assertIn("fernleihe", {k["keyword"] for k in self.store.list_keywords(200)})

    def test_should_stop_cancels_crawl(self):
        stop = {"v": False}

        def should_stop():
            return stop["v"]

        # Flip the flag after the first page is fetched by counting via fetch.
        counter = {"n": 0}
        base_fetch = _make_fetch(self._serves())

        def counting_fetch(url, *, user_agent, timeout):
            counter["n"] += 1
            if counter["n"] >= 1:
                stop["v"] = True
            return base_fetch(url, user_agent=user_agent, timeout=timeout)

        res = indexer.crawl_site(
            self.store, base_url=BASE, max_depth=1, max_pages=20, min_chars=50,
            fetch_func=counting_fetch, should_stop=should_stop,
        )
        # The seed is fetched, then should_stop breaks the loop before children.
        self.assertEqual(res["pages_indexed"], 1)
        self.assertEqual(res["visited"], 1)

    def test_pdf_page_indexed_without_child_links(self):
        serves = self._serves()
        serves[BASE + "broschuere.pdf"] = (b"%PDF-1.4 fake", "application/pdf", 200)
        with mock.patch("src.utils.pdf_extractor.extract_text") as fake_pdf:
            fake_pdf.return_value = {
                "text": "Eine Broschüre über Fernleihe und Katalogdienste der Bibliothek.",
                "pages": 2, "quality": "good", "source": "pypdf", "truncated": False,
                "chars": 60,
            }
            res = indexer.crawl_site(
                self.store, base_url=BASE, max_depth=1, max_pages=20, min_chars=50,
                fetch_func=_make_fetch(serves),
            )
        self.assertIn(BASE + "broschuere.pdf", res["indexed_urls"])
        page = self.store.get_page(BASE + "broschuere.pdf")
        self.assertIsNotNone(page)
        self.assertIn("Broschüre", page["text"])

    def test_dry_run_does_not_store(self):
        res = indexer.crawl_site(
            self.store, base_url=BASE, max_depth=1, max_pages=20, min_chars=50,
            fetch_func=_make_fetch(self._serves()), dry_run=True,
        )
        # URLs discovered + reported, but the DB stays empty.
        self.assertTrue(res["indexed_urls"])  # discovered URLs reported
        self.assertEqual(self.store.stats()["pages"], 0)
        self.assertEqual(self.store.stats()["keywords"], 0)

    def test_fetch_error_recorded_not_fatal(self):
        serves = {BASE: ("index.html", "text/html", 200)}  # children → 404
        res = indexer.crawl_site(
            self.store, base_url=BASE, max_depth=1, max_pages=20, min_chars=50,
            fetch_func=_make_fetch(serves),
        )
        self.assertEqual(res["pages_indexed"], 1)  # root still indexed
        self.assertTrue(res["errors"])  # child fetches failed


if __name__ == "__main__":  # pragma: no cover
    unittest.main()