"""
RVK MarcXML dump index for GND-based RVK lookup.

Builds a local SQLite index from the official RVK MarcXML dump and supports
direct GND-ID -> RVK candidate lookup before label-based RVK API fallback.
"""

from __future__ import annotations

import gzip
import html
import logging
import os
import re
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests

logger = logging.getLogger("rvk_marc_index")


class RvkMarcIndex:
    """Build and query a local GND -> RVK index from the official RVK MarcXML dump."""

    DOWNLOAD_PAGE_URL = "https://rvk.uni-regensburg.de/regensburger-verbundklassifikation-online/rvk-download"
    UPDATE_CHECK_INTERVAL_DAYS = 7

    def __init__(self, data_dir: Optional[Path] = None, timeout: int = 20):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.setdefault("User-Agent", "ALIMA/rvk-marc-index")
        self.data_dir = data_dir or self._default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "rvk_gnd_index.sqlite3"

    @staticmethod
    def _default_data_dir() -> Path:
        from ..config_manager import ConfigManager

        config_manager = ConfigManager()
        return config_manager.config_file.parent / "rvk"

    @staticmethod
    def _strip_namespace(tag: str) -> str:
        return tag.rsplit("}", 1)[-1] if "}" in tag else tag

    @staticmethod
    def _normalize_text(value: str) -> str:
        clean = html.unescape(str(value or "")).strip()
        clean = re.sub(r"\s+", " ", clean)
        return clean

    @staticmethod
    def normalize_gnd_id(value: str) -> str:
        clean = str(value or "").strip()
        match = re.search(r"\(DE-588\)\s*([0-9X-]+)", clean)
        if match:
            return match.group(1)
        match = re.search(r"([0-9X-]{6,})", clean)
        if match:
            return match.group(1)
        return clean

    @staticmethod
    def _branch_family(notation: str) -> str:
        match = re.search(r"[A-Z]", notation or "")
        return match.group(0) if match else ""

    def _resolve_current_dump(self) -> Tuple[str, str]:
        response = self.session.get(self.DOWNLOAD_PAGE_URL, timeout=self.timeout)
        response.raise_for_status()
        html_text = response.text
        match = re.search(
            r'href="([^"]+)"[^>]*>\s*aktueller\s+MarcXML-Abzug\s*</a>\s*\(([^)]+)\)',
            html_text,
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError("Could not resolve current RVK MarcXML dump URL")
        return urljoin(self.DOWNLOAD_PAGE_URL, match.group(1)), match.group(2).strip()

    def _download_dump(self, dump_url: str, release: str, progress_callback=None) -> Path:
        dump_path = self.data_dir / f"rvko_marcxml_{release}.xml.gz"
        if dump_path.exists():
            return dump_path

        if progress_callback:
            progress_callback(f"  ↳ Lade RVK MarcXML-Dump {release} herunter...\n")

        with self.session.get(dump_url, timeout=self.timeout, stream=True) as response:
            response.raise_for_status()
            with open(dump_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)

        return dump_path

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gnd_rvk_map (
                gnd_id TEXT NOT NULL,
                notation TEXT NOT NULL,
                label TEXT,
                ancestor_path TEXT,
                register_term TEXT,
                field_tag TEXT,
                branch_family TEXT,
                PRIMARY KEY (gnd_id, notation, register_term, field_tag)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gnd_rvk_map_gnd ON gnd_rvk_map(gnd_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gnd_rvk_map_notation ON gnd_rvk_map(notation)")
        conn.commit()

    def _read_meta(self) -> Dict[str, str]:
        if not self.db_path.exists():
            return {}
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
            conn.close()
            return {str(key): str(value) for key, value in rows}
        except Exception:
            return {}

    def _write_meta_value(self, key: str, value: str) -> None:
        if not self.db_path.exists():
            return
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            [key, value],
        )
        conn.commit()
        conn.close()

    def _should_check_updates(self, meta: Dict[str, str]) -> bool:
        last_checked = meta.get("last_release_check")
        if not last_checked:
            return True
        try:
            checked_at = datetime.fromisoformat(last_checked)
        except ValueError:
            return True
        return datetime.utcnow() - checked_at >= timedelta(days=self.UPDATE_CHECK_INTERVAL_DAYS)

    def _parse_record(self, record: ET.Element) -> Tuple[str, str, str, List[Tuple[str, str, str]]]:
        notation = ""
        label = ""
        hierarchy_labels: List[str] = []
        gnd_rows: List[Tuple[str, str, str]] = []

        for datafield in record:
            if self._strip_namespace(datafield.tag) != "datafield":
                continue
            tag = datafield.attrib.get("tag", "")

            if tag == "153":
                for subfield in datafield:
                    if self._strip_namespace(subfield.tag) != "subfield":
                        continue
                    code = subfield.attrib.get("code", "")
                    value = self._normalize_text(subfield.text or "")
                    if code == "a" and not notation:
                        notation = value
                    elif code == "j" and not label:
                        label = value
                    elif code == "h" and value:
                        hierarchy_labels.append(value)
                continue

            if tag not in {"700", "710", "711", "730", "750", "751"}:
                continue

            gnd_id = ""
            register_term = ""
            source = ""
            for subfield in datafield:
                if self._strip_namespace(subfield.tag) != "subfield":
                    continue
                code = subfield.attrib.get("code", "")
                value = self._normalize_text(subfield.text or "")
                if code == "0":
                    gnd_id = self.normalize_gnd_id(value)
                elif code == "a":
                    register_term = value
                elif code == "2":
                    source = value.lower()

            if gnd_id and register_term and source == "gnd":
                gnd_rows.append((gnd_id, register_term, tag))

        ancestor_path = " > ".join(hierarchy_labels)
        return notation, label, ancestor_path, gnd_rows

    def _build_index(self, dump_path: Path, release: str, progress_callback=None) -> None:
        fd, temp_path = tempfile.mkstemp(prefix="rvk_gnd_index_", suffix=".sqlite3", dir=str(self.data_dir))
        os.close(fd)
        temp_db = Path(temp_path)
        try:
            conn = sqlite3.connect(temp_db)
            self._create_schema(conn)

            batch: List[Tuple[str, str, str, str, str, str, str]] = []
            records_processed = 0
            mapped_rows = 0

            with gzip.open(dump_path, "rb") as fh:
                for _, elem in ET.iterparse(fh, events=("end",)):
                    if self._strip_namespace(elem.tag) != "record":
                        continue

                    notation, label, ancestor_path, gnd_rows = self._parse_record(elem)
                    if notation and gnd_rows:
                        branch_family = self._branch_family(notation)
                        for gnd_id, register_term, field_tag in gnd_rows:
                            batch.append((
                                gnd_id,
                                notation,
                                label,
                                ancestor_path,
                                register_term,
                                field_tag,
                                branch_family,
                            ))
                            mapped_rows += 1

                    records_processed += 1
                    if len(batch) >= 5000:
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO gnd_rvk_map
                            (gnd_id, notation, label, ancestor_path, register_term, field_tag, branch_family)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            batch,
                        )
                        conn.commit()
                        batch.clear()

                    if progress_callback and records_processed % 20000 == 0:
                        progress_callback(
                            f"  ↳ RVK MarcXML-Index: {records_processed} Datensaetze verarbeitet\n"
                        )

                    elem.clear()

            if batch:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO gnd_rvk_map
                    (gnd_id, notation, label, ancestor_path, register_term, field_tag, branch_family)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()

            conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [
                    ("release", release),
                    ("built_at", datetime.utcnow().isoformat()),
                    ("dump_path", str(dump_path)),
                    ("mapped_rows", str(mapped_rows)),
                    ("records_processed", str(records_processed)),
                ],
            )
            conn.commit()
            conn.close()
            temp_db.replace(self.db_path)
        finally:
            if temp_db.exists():
                temp_db.unlink(missing_ok=True)

    def ensure_index(self, force_rebuild: bool = False, progress_callback=None) -> bool:
        meta = self._read_meta()
        if self.db_path.exists() and not force_rebuild and not self._should_check_updates(meta):
            return True

        try:
            dump_url, release = self._resolve_current_dump()
            now_iso = datetime.utcnow().isoformat()

            if self.db_path.exists() and not force_rebuild:
                local_release = meta.get("release")
                self._write_meta_value("last_release_check", now_iso)
                if local_release == release:
                    if progress_callback:
                        progress_callback(f"  ℹ️ RVK-GND-Index aktuell ({release})\n")
                    return True
                if progress_callback:
                    progress_callback(f"  ↻ Neue RVK-Dump-Version erkannt: {local_release or 'unbekannt'} → {release}\n")

            dump_path = self._download_dump(dump_url, release, progress_callback=progress_callback)
            if progress_callback:
                progress_callback(f"  ↳ Baue lokalen RVK-GND-Index aus Dump {release}...\n")
            self._build_index(dump_path, release, progress_callback=progress_callback)
            self._write_meta_value("last_release_check", now_iso)
            if progress_callback:
                progress_callback("  ✅ RVK-GND-Index bereit\n")
            return True
        except Exception as exc:
            logger.warning(f"RVK MarcXML index setup failed: {exc}")
            if self.db_path.exists() and not force_rebuild:
                if progress_callback:
                    progress_callback(f"  ⚠️ RVK-GND-Index-Update fehlgeschlagen, nutze lokalen Index weiter: {exc}\n")
                return True
            if progress_callback:
                progress_callback(f"  ⚠️ RVK-GND-Index konnte nicht aufgebaut werden: {exc}\n")
            return False

    def lookup_by_gnd_keywords(
        self,
        keyword_entries: List[Dict[str, str]],
        max_results_per_keyword: int = 6,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        if not keyword_entries:
            return []
        if not self.ensure_index(progress_callback=progress_callback):
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        results: List[Dict[str, Any]] = []

        try:
            for entry in keyword_entries:
                keyword = self._normalize_text(entry.get("keyword", ""))
                gnd_id = self.normalize_gnd_id(entry.get("gnd_id", ""))
                if not keyword or not gnd_id:
                    continue

                rows = conn.execute(
                    """
                    SELECT
                        notation,
                        label,
                        ancestor_path,
                        field_tag,
                        branch_family,
                        GROUP_CONCAT(DISTINCT register_term) AS register_terms,
                        COUNT(*) AS term_hits
                    FROM gnd_rvk_map
                    WHERE gnd_id = ?
                    GROUP BY notation, label, ancestor_path, field_tag, branch_family
                    """,
                    [gnd_id],
                ).fetchall()

                if not rows:
                    continue

                classifications = []
                for row in rows:
                    register_terms = [term for term in str(row["register_terms"] or "").split(",") if term]
                    exact_match = any(term.casefold() == keyword.casefold() for term in register_terms)
                    contains_match = any(keyword.casefold() in term.casefold() for term in register_terms)
                    field_bonus = {
                        "750": 6,
                        "751": 5,
                        "730": 4,
                        "710": 4,
                        "711": 4,
                        "700": 3,
                    }.get(str(row["field_tag"]), 1)
                    specificity_bonus = len(re.sub(r"[^A-Z0-9]", "", str(row["notation"] or "")))
                    score = field_bonus + specificity_bonus + int(row["term_hits"] or 0)
                    if exact_match:
                        score += 8
                    elif contains_match:
                        score += 4

                    classifications.append({
                        "dk": self._normalize_text(row["notation"]),
                        "type": "RVK",
                        "classification_type": "RVK",
                        "count": int(row["term_hits"] or 1),
                        "titles": [],
                        "matched_keywords": [keyword],
                        "source": "rvk_gnd_index",
                        "label": self._normalize_text(row["label"]),
                        "ancestor_path": self._normalize_text(row["ancestor_path"]),
                        "register": register_terms,
                        "score": score,
                        "branch_family": self._normalize_text(row["branch_family"]),
                        "rvk_validation_status": "standard",
                        "validation_message": "",
                        "gnd_id": gnd_id,
                    })

                classifications.sort(key=lambda item: (-item["score"], item["dk"]))
                results.append({
                    "keyword": keyword,
                    "source": "rvk_gnd_index",
                    "search_time_ms": 0.0,
                    "classifications": classifications[:max_results_per_keyword],
                })

            return results
        finally:
            conn.close()
