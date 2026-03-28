"""
RVK API Client for ALIMA.

Provides authority-backed RVK candidate lookup via the official RVK API.
Used as a fallback when local catalog search yields no RVK suggestions.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger("rvk_api_client")


class RvkApiClient:
    """Small client for the official RVK API."""

    BASE_URL = "https://rvk.uni-regensburg.de/api/json"

    def __init__(self, timeout: int = 8, session: Optional[requests.Session] = None):
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "ALIMA/rvk-api-client")

    @staticmethod
    def normalize_keyword(keyword: str) -> str:
        """Strip GND metadata and normalize whitespace for RVK term search."""
        if not keyword:
            return ""
        clean = re.sub(r"\s*\(GND-ID:\s*[^)]+\)\s*$", "", str(keyword)).strip()
        clean = html.unescape(clean)
        clean = re.sub(r"\s+", " ", clean)
        return clean

    @staticmethod
    def _normalize_text(text: str) -> str:
        clean = html.unescape(str(text or "")).strip().casefold()
        clean = re.sub(r"\s+", " ", clean)
        return clean

    @staticmethod
    def normalize_notation(code: str) -> str:
        """Canonicalize RVK notation spacing for comparison."""
        clean = html.unescape(str(code or "")).strip().upper()
        clean = re.sub(r"\s+", " ", clean)
        match = re.match(r"^([A-Z]{1,4})\s*([0-9].*)$", clean)
        if match:
            clean = f"{match.group(1)} {match.group(2).strip()}"
        return clean

    def _get_json(self, path: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{path}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()

        text = response.text.strip()
        if not text.startswith("{"):
            return {}
        return response.json()

    @staticmethod
    def _ensure_nodes(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        nodes = payload.get("node")
        if isinstance(nodes, list):
            return [node for node in nodes if isinstance(node, dict)]
        if isinstance(nodes, dict):
            return [nodes]
        return []

    def _node_matches_keyword(self, keyword: str, node: Dict[str, Any]) -> bool:
        """Reject noisy substring matches such as author names for 'OSZE'."""
        normalized_keyword = self._normalize_text(keyword)
        if not normalized_keyword:
            return False

        texts = [self._normalize_text(node.get("benennung", ""))]
        texts.extend(self._normalize_text(item) for item in node.get("register", []) or [])

        pattern = re.compile(rf"(?<!\w){re.escape(normalized_keyword)}(?!\w)")
        for text in texts:
            if not text:
                continue
            if pattern.search(text) or normalized_keyword == text:
                return True
        return False

    def get_ancestors(self, notation: str) -> List[Dict[str, str]]:
        payload = self._get_json(f"ancestors/{quote(self.normalize_notation(notation), safe='')}")
        node = payload.get("node")
        if not isinstance(node, dict):
            return []

        def _flatten(item: Dict[str, Any]) -> List[Dict[str, str]]:
            chain: List[Dict[str, str]] = []
            ancestor = item.get("ancestor")
            if isinstance(ancestor, dict) and isinstance(ancestor.get("node"), dict):
                chain.extend(_flatten(ancestor["node"]))
            notation_value = str(item.get("notation", "")).strip()
            label_value = html.unescape(str(item.get("benennung", "")).strip())
            if notation_value or label_value:
                chain.append({
                    "notation": notation_value,
                    "label": label_value,
                })
            return chain

        return _flatten(node)

    @staticmethod
    def _branch_family(ancestors: List[Dict[str, str]]) -> str:
        for item in ancestors:
            notation = item.get("notation", "")
            match = re.search(r"[A-Z]", notation)
            if match:
                return match.group(0)
        return ""

    def validate_notation(self, notation: str) -> Dict[str, Any]:
        """Validate a concrete RVK notation against the official RVK API."""
        normalized = self.normalize_notation(notation)
        if not normalized:
            return {
                "status": "artifact",
                "notation": "",
                "label": "",
                "register": [],
                "ancestor_path": "",
                "branch_family": "",
                "message": "Empty RVK notation",
            }

        try:
            payload = self._get_json(f"node/{quote(normalized, safe='')}")
        except Exception as exc:
            return {
                "status": "validation_error",
                "notation": normalized,
                "label": "",
                "register": [],
                "ancestor_path": "",
                "branch_family": "",
                "message": str(exc),
            }

        nodes = self._ensure_nodes(payload)
        if not nodes:
            return {
                "status": "not_found",
                "notation": normalized,
                "label": "",
                "register": [],
                "ancestor_path": "",
                "branch_family": "",
                "message": payload.get("error-message", "Notation Not Found"),
            }

        node = nodes[0]
        canonical = self.normalize_notation(node.get("notation", normalized))
        label = html.unescape(str(node.get("benennung", "")).strip())
        register = [
            html.unescape(str(item).strip())
            for item in (node.get("register") or [])
            if str(item).strip()
        ]
        ancestors = self.get_ancestors(canonical)
        branch_path = " > ".join(
            item["label"] for item in ancestors if item.get("label")
        )
        return {
            "status": "standard",
            "notation": canonical,
            "label": label,
            "register": register,
            "ancestor_path": branch_path,
            "branch_family": self._branch_family(ancestors),
            "message": "",
        }

    def search_keyword(self, keyword: str, max_results: int = 8) -> List[Dict[str, Any]]:
        """Search authority-backed RVK candidates for a keyword."""
        search_term = self.normalize_keyword(keyword)
        if not search_term:
            return []

        try:
            payload = self._get_json(f"nodes/{quote(search_term, safe='')}")
        except Exception as exc:
            logger.warning(f"RVK nodes search failed for '{search_term}': {exc}")
            return []

        candidates = []
        for node in self._ensure_nodes(payload):
            notation = self.normalize_notation(node.get("notation", ""))
            if not notation or " - " in notation:
                continue
            if not self._node_matches_keyword(search_term, node):
                continue

            label = html.unescape(str(node.get("benennung", "")).strip())
            register = [
                html.unescape(str(item).strip())
                for item in (node.get("register") or [])
                if str(item).strip()
            ]
            ancestors = self.get_ancestors(notation)
            branch_path = " > ".join(
                item["label"] for item in ancestors if item.get("label")
            )

            score = 1
            normalized_label = self._normalize_text(label)
            normalized_keyword = self._normalize_text(search_term)
            if normalized_keyword and normalized_keyword == normalized_label:
                score += 5
            elif normalized_keyword and normalized_keyword in normalized_label:
                score += 3
            if any(normalized_keyword == self._normalize_text(item) for item in register):
                score += 2

            candidates.append({
                "notation": notation,
                "label": label,
                "register": register,
                "ancestor_path": branch_path,
                "branch_family": self._branch_family(ancestors),
                "score": score,
            })

        candidates.sort(key=lambda item: (-item["score"], item["notation"]))
        return candidates[:max_results]
