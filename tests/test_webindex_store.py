"""Tests for the WebIndexStore (own SQLite DB, LocalGndStore pattern) - Claude Generated.

Netzfrei: the store is pure DB I/O. Uses temp SQLite files (never the production
config). No UnifiedKnowledgeManager involvement, so no reset() needed.
"""

import os
import tempfile
import unittest

try:
    from src.utils.lookups.webindex.store import (
        WebIndexStore, _normalize_keyword,
    )
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc


def _store(tmpdir, label="t"):
    settings = {"db_path": os.path.join(tmpdir, "webindex.db")}
    return WebIndexStore(settings, f"webindex_test_{label}")


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class WebIndexStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.store = _store(self._tmp)

    def tearDown(self):
        try:
            self.store.close()
        except Exception:
            pass

    # --- normalisation -------------------------------------------------- #
    def test_normalize_keyword_strips_and_lowercases(self):
        self.assertEqual(_normalize_keyword("  Fernleihe "), "fernleihe")
        self.assertEqual(_normalize_keyword('"Öffnungszeiten,"'), "öffnungszeiten")
        self.assertEqual(_normalize_keyword("a  b"), "a b")
        self.assertEqual(_normalize_keyword("  "), "")

    # --- pages ---------------------------------------------------------- #
    def test_upsert_and_get_page(self):
        self.store.upsert_page(
            url="https://x/a", base_url="https://x/", title="A",
            text="hello world", http_status=200, content_type="text/html",
            text_truncated=False,
        )
        page = self.store.get_page("https://x/a")
        self.assertIsNotNone(page)
        self.assertEqual(page["title"], "A")
        self.assertEqual(page["text"], "hello world")
        self.assertFalse(page["text_truncated"])

    def test_upsert_replaces_on_recrawl(self):
        self.store.upsert_page(url="https://x/a", base_url="https://x/", title="A",
                               text="v1", http_status=200, content_type="text/html",
                               text_truncated=False)
        self.store.upsert_page(url="https://x/a", base_url="https://x/", title="A2",
                               text="v2", http_status=200, content_type="text/html",
                               text_truncated=True)
        page = self.store.get_page("https://x/a")
        self.assertEqual(page["title"], "A2")
        self.assertEqual(page["text"], "v2")
        self.assertTrue(page["text_truncated"])

    def test_get_page_text_missing(self):
        self.assertIsNone(self.store.get_page_text("https://nope/"))

    # --- keyword sync --------------------------------------------------- #
    def test_set_page_keywords_populates_central_catalogue(self):
        self.store.upsert_page(url="https://x/a", base_url="https://x/", title="A",
                               text="t", http_status=200, content_type="text/html",
                               text_truncated=False)
        self.store.set_page_keywords("https://x/a", [
            ("Fernleihe", 1.0, "meta"), ("Katalog", 1.0, "meta"),
        ])
        kw = {k["keyword"]: k for k in self.store.list_keywords()}
        self.assertIn("fernleihe", kw)
        self.assertIn("katalog", kw)
        # display form preserved
        self.assertEqual(kw["fernleihe"]["display"], "Fernleihe")

    def test_set_page_keywords_reindexes_sync(self):
        """Re-indexing a page replaces its links wholesale (the sync step)."""
        url = "https://x/a"
        self.store.upsert_page(url=url, base_url="https://x/", title="A", text="t",
                               http_status=200, content_type="text/html",
                               text_truncated=False)
        self.store.set_page_keywords(url, [("Fernleihe", 1.0, "meta"), ("Katalog", 1.0, "meta")])
        self.assertEqual(len(self.store.get_pages_for_keywords(["katalog"], 5)), 1)
        # Re-index with a completely different keyword set.
        self.store.set_page_keywords(url, [("Auskunft", 1.0, "meta")])
        # old link gone for this page:
        self.assertEqual(self.store.get_pages_for_keywords(["katalog"], 5), [])
        self.assertEqual(self.store.get_pages_for_keywords(["auskunft"], 5)[0]["url"], url)

    # --- ranking -------------------------------------------------------- #
    def test_ranking_score_and_match_count(self):
        """Page matching more keywords (higher weight sum) ranks first."""
        for url, kws in [
            ("https://x/a", [("Fernleihe", 1.0, "meta"), ("Katalog", 1.0, "meta")]),
            ("https://x/b", [("Fernleihe", 1.0, "meta")]),
        ]:
            self.store.upsert_page(url=url, base_url="https://x/", title=url[-1],
                                   text="t", http_status=200, content_type="text/html",
                                   text_truncated=False)
            self.store.set_page_keywords(url, kws)
        hits = self.store.get_pages_for_keywords(["fernleihe", "katalog"], max_results=5)
        self.assertEqual([h["url"] for h in hits], ["https://x/a", "https://x/b"])
        self.assertGreater(hits[0]["score"], hits[1]["score"])
        self.assertEqual(hits[0]["matched_count"], 2)
        self.assertEqual(hits[1]["matched_count"], 1)

    def test_llm_weight_boosts_ranking(self):
        """LLM-sourced keywords (weight 1.5) outrank meta keywords (weight 1.0)."""
        self.store.upsert_page(url="https://x/meta", base_url="https://x/", title="m",
                               text="t", http_status=200, content_type="text/html",
                               text_truncated=False)
        self.store.set_page_keywords("https://x/meta", [("Fernleihe", 1.0, "meta")])
        self.store.upsert_page(url="https://x/llm", base_url="https://x/", title="l",
                               text="t", http_status=200, content_type="text/html",
                               text_truncated=False)
        self.store.set_page_keywords("https://x/llm", [("Fernleihe", 1.5, "llm")])
        hits = self.store.get_pages_for_keywords(["fernleihe"], max_results=5)
        self.assertEqual(hits[0]["url"], "https://x/llm")
        self.assertGreater(hits[0]["score"], hits[1]["score"])

    def test_no_match_returns_empty(self):
        self.store.upsert_page(url="https://x/a", base_url="https://x/", title="A",
                               text="t", http_status=200, content_type="text/html",
                               text_truncated=False)
        self.store.set_page_keywords("https://x/a", [("Fernleihe", 1.0, "meta")])
        self.assertEqual(self.store.get_pages_for_keywords(["nichtvorhanden"], 5), [])

    # --- listing / stats ------------------------------------------------ #
    def test_list_keywords_contains_filter(self):
        self.store.upsert_page(url="https://x/a", base_url="https://x/", title="A",
                               text="t", http_status=200, content_type="text/html",
                               text_truncated=False)
        self.store.set_page_keywords("https://x/a", [("Fernleihe", 1.0, "meta"),
                                                     ("Katalog", 1.0, "meta")])
        rows = self.store.list_keywords(limit=100, contains="fern")
        self.assertEqual([r["keyword"] for r in rows], ["fernleihe"])

    def test_stats_counts(self):
        self.store.upsert_page(url="https://x/a", base_url="https://x/", title="A",
                               text="t", http_status=200, content_type="text/html",
                               text_truncated=False)
        self.store.set_page_keywords("https://x/a", [("Fernleihe", 1.0, "meta"),
                                                     ("Katalog", 1.0, "meta")])
        s = self.store.stats()
        self.assertEqual(s["pages"], 1)
        self.assertEqual(s["keywords"], 2)
        self.assertEqual(s["page_keywords"], 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()