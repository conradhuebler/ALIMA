"""Plugin-owned website/URL index store - Claude Generated.

Holds three tables in its **own** SQLite database (``webindex.db``), physically
separate from the main knowledge DB — mirroring the ``LocalGndStore`` pattern
(``src/core/search/providers/gnd_local/store.py``): a fresh ``DatabaseConfig``
pinned to SQLite + a dedicated ``DatabaseManager`` with its own ``connection_name``
so per-thread QtSql connections never collide with the main DB.

The store is the single source of truth for the website-RAG chatbot:

* ``pages``      — every crawled URL + its cached extracted text (the "live-fetch or
                   cache" cache), title, fetch metadata.
* ``keywords``   — the **central**, synchronised keyword catalogue (one row per
                   normalised keyword, with its original display form).
* ``page_keywords`` — the inverted index linking each page to its keywords with a
                   weight + source (``meta`` / ``heading`` / ``llm``). Re-indexing a
                   page replaces its links wholesale ("synchronisation").

Retrieval is keyword-based (no embeddings): a query's terms are matched against
``page_keywords`` and pages are ranked by summed keyword weight, tie-broken by the
number of distinct matched keywords.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.database_manager import DatabaseManager
from src.utils.config_models import DatabaseConfig


def _normalize_keyword(keyword: str) -> str:
    """Normalise a keyword for the central catalogue + matching - Claude Generated.

    Lowercase, collapse internal whitespace, strip surrounding punctuation/quotes.
    Keeps umlauts/unicode intact (we match on surface form, not stems).
    """
    if not keyword:
        return ""
    k = str(keyword).strip().lower()
    k = re.sub(r"\s+", " ", k)
    k = k.strip(" \t\",.;:!?«»\"'()[]{}")
    return k


def resolve_webindex_path(settings: Optional[Dict[str, Any]]) -> str:
    """Path of the webindex DB - Claude Generated.

    An explicit ``db_path`` setting wins; otherwise a sibling of the main SQLite DB
    (so a temp/main path yields an isolated sibling, and the production
    ``~/.config/alima/alima_knowledge.db`` yields ``~/.config/alima/webindex.db``).
    """
    explicit = ""
    if settings:
        explicit = str(settings.get("db_path") or "").strip()
    if explicit:
        return explicit
    try:
        from src.utils.config_manager import ConfigManager

        sqlite_path = ConfigManager().load_config().database_config.sqlite_path or ""
    except Exception:
        sqlite_path = ""
    base_dir = os.path.dirname(sqlite_path) or "."
    return os.path.join(base_dir, "webindex.db")


class WebIndexStore:
    """Owns the ``pages`` / ``keywords`` / ``page_keywords`` tables in a dedicated
    SQLite file. Qt-free except for the ``DatabaseManager`` it delegates to (which
    spins up a minimal ``QCoreApplication`` in headless mode itself). - Claude
    Generated"""

    def __init__(self, settings: Optional[Dict[str, Any]], connection_name: str):
        self.logger = logging.getLogger(__name__)
        self.settings = dict(settings or {})
        self.path = resolve_webindex_path(self.settings)
        cfg = DatabaseConfig(db_type="sqlite")
        cfg.sqlite_path = self.path
        # Distinct connection_name → per-thread connection names never collide with
        # the main knowledge DB's (see MEMORY.md QSqlDatabase thread-safety note).
        self.db_manager = DatabaseManager(cfg, connection_name)
        self._init_schema()

    # --- schema ----------------------------------------------------------- #
    def _init_schema(self) -> None:
        self.db_manager.get_connection()
        db_type = self.db_manager.get_db_type()
        dialect = self.db_manager.get_dialect()
        self.db_manager.execute_query(f"""
            CREATE TABLE IF NOT EXISTS pages (
                url {dialect.varchar_type(2048)} PRIMARY KEY,
                base_url {dialect.varchar_type(2048)},
                title {dialect.text_type(db_type)},
                text {dialect.text_type(db_type)},
                fetched_at {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP,
                http_status INTEGER,
                content_type {dialect.varchar_type(255)},
                text_truncated INTEGER DEFAULT 0,
                updated_at {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db_manager.execute_query(f"""
            CREATE TABLE IF NOT EXISTS keywords (
                keyword {dialect.varchar_type(256)} PRIMARY KEY,
                display {dialect.varchar_type(256)} NOT NULL,
                created_at {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP
            )
        """)
        pk_page_keywords = dialect.primary_key_def(
            db_type, ["page_url", "keyword"],
            key_lengths={"page_url": 2048, "keyword": 256},
        )
        self.db_manager.execute_query(f"""
            CREATE TABLE IF NOT EXISTS page_keywords (
                page_url {dialect.varchar_type(2048)} NOT NULL,
                keyword {dialect.varchar_type(256)} NOT NULL,
                weight REAL DEFAULT 1.0,
                source {dialect.varchar_type(16)} DEFAULT 'meta',
                {pk_page_keywords}
            )
        """)
        # Retrieval is "which pages match keyword X" → index the keyword column.
        self.db_manager.execute_query(
            "CREATE INDEX IF NOT EXISTS idx_pk_keyword ON page_keywords(keyword)"
        )
        self.db_manager.execute_query(
            "CREATE INDEX IF NOT EXISTS idx_pk_page ON page_keywords(page_url)"
        )

    # --- writes ----------------------------------------------------------- #
    def upsert_page(
        self,
        *,
        url: str,
        base_url: str,
        title: str,
        text: str,
        http_status: Optional[int],
        content_type: str,
        text_truncated: bool,
    ) -> None:
        """Insert/replace a page row (full re-fetch on re-crawl). - Claude Generated"""
        self.db_manager.execute_query(
            """
            INSERT OR REPLACE INTO pages
            (url, base_url, title, text, fetched_at, http_status, content_type,
             text_truncated, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [url, base_url, title, text, http_status, content_type, 1 if text_truncated else 0],
        )

    def set_page_keywords(self, url: str, links: Sequence[Tuple[str, float, str]]) -> None:
        """Replace a page's keyword links wholesale (the "synchronisation" step).

        ``links`` is an iterable of ``(keyword_display, weight, source)`` tuples.
        Each keyword is normalised + upserted into the central ``keywords`` table,
        then linked to the page. Old links for this page are deleted first so a
        re-index reflects the current page content. - Claude Generated
        """
        self.db_manager.begin_transaction()
        try:
            self.db_manager.execute_query(
                "DELETE FROM page_keywords WHERE page_url = ?", [url]
            )
            for display, weight, source in links:
                norm = _normalize_keyword(display)
                if not norm:
                    continue
                self.db_manager.execute_query(
                    "INSERT OR IGNORE INTO keywords (keyword, display) VALUES (?, ?)",
                    [norm, str(display).strip() or norm],
                )
                self.db_manager.execute_query(
                    """
                    INSERT OR REPLACE INTO page_keywords
                    (page_url, keyword, weight, source) VALUES (?, ?, ?, ?)
                    """,
                    [url, norm, float(weight), str(source)],
                )
            self.db_manager.commit_transaction()
        except Exception:
            try:
                self.db_manager.rollback_transaction()
            except Exception:
                pass
            raise

    # --- reads ------------------------------------------------------------ #
    def get_page(self, url: str) -> Optional[Dict[str, Any]]:
        row = self.db_manager.fetch_one("SELECT * FROM pages WHERE url = ?", [url])
        if row:
            row["text_truncated"] = bool(row.get("text_truncated"))
        return row

    def get_page_text(self, url: str) -> Optional[str]:
        row = self.db_manager.fetch_one("SELECT text FROM pages WHERE url = ?", [url])
        return (row or {}).get("text")

    def get_pages_for_keywords(
        self, keywords: Sequence[str], max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Rank pages by summed keyword weight over the matched terms.

        Tie-break: number of distinct matched keywords (more matches = more
        topical). Returns at most ``max_results`` rows with ``url``, ``title``,
        ``text`` (full cached text), ``base_url``, ``score`` and the list of
        matched keywords (display form). - Claude Generated
        """
        norms = []
        seen = set()
        for k in keywords:
            n = _normalize_keyword(k)
            if n and n not in seen:
                seen.add(n)
                norms.append(n)
        if not norms:
            return []
        placeholders = ", ".join(["?"] * len(norms))
        rows = self.db_manager.fetch_all(
            f"SELECT page_url, keyword, weight FROM page_keywords "
            f"WHERE keyword IN ({placeholders})",
            norms,
        )
        if not rows:
            return []
        agg: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            url = r["page_url"]
            entry = agg.setdefault(url, {"matched": [], "score": 0.0})
            entry["matched"].append(r["keyword"])
            try:
                entry["score"] += float(r["weight"] or 0.0)
            except (TypeError, ValueError):
                pass
        ranked = sorted(
            agg.items(),
            key=lambda kv: (kv[1]["score"], len(set(kv[1]["matched"]))),
            reverse=True,
        )[: max(0, int(max_results or 0))]
        out: List[Dict[str, Any]] = []
        for url, entry in ranked:
            page = self.get_page(url) or {}
            out.append({
                "url": url,
                "title": page.get("title") or "",
                "text": page.get("text") or "",
                "base_url": page.get("base_url") or "",
                "score": round(entry["score"], 4),
                "matched_count": len(set(entry["matched"])),
                "matched_keywords": sorted(set(entry["matched"])),
            })
        return out

    def list_keywords(self, limit: int = 200, contains: str = "") -> List[Dict[str, Any]]:
        """Browse the central keyword catalogue (display form + per-keyword page
        count). ``contains`` filters case-insensitively. - Claude Generated"""
        limit = max(0, int(limit or 0))
        if contains:
            like = f"%{_normalize_keyword(contains)}%"
            rows = self.db_manager.fetch_all(
                """
                SELECT k.keyword, k.display, COUNT(pk.page_url) AS page_count
                FROM keywords k LEFT JOIN page_keywords pk ON pk.keyword = k.keyword
                WHERE k.keyword LIKE ?
                GROUP BY k.keyword, k.display
                ORDER BY page_count DESC, k.keyword
                LIMIT ?
                """,
                [like, limit],
            )
        else:
            rows = self.db_manager.fetch_all(
                """
                SELECT k.keyword, k.display, COUNT(pk.page_url) AS page_count
                FROM keywords k LEFT JOIN page_keywords pk ON pk.keyword = k.keyword
                GROUP BY k.keyword, k.display
                ORDER BY page_count DESC, k.keyword
                LIMIT ?
                """,
                [limit],
            )
        return rows

    def stats(self) -> Dict[str, Any]:
        pages = int(self.db_manager.fetch_scalar("SELECT COUNT(*) FROM pages") or 0)
        keywords = int(self.db_manager.fetch_scalar("SELECT COUNT(*) FROM keywords") or 0)
        links = int(self.db_manager.fetch_scalar("SELECT COUNT(*) FROM page_keywords") or 0)
        last = self.db_manager.fetch_scalar("SELECT MAX(fetched_at) FROM pages")
        return {
            "db_path": self.path,
            "pages": pages,
            "keywords": keywords,
            "page_keywords": links,
            "last_fetched": last,
        }

    def close(self) -> None:
        try:
            self.db_manager.close_connection()
        except Exception:
            pass