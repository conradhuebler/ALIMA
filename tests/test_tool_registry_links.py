"""MCP tool layer pre-formats canonical GND/SWB URLs for the chat agent.

Claude Generated. The chat agent must receive ready URLs (so it never builds one
from a bare id and never confuses a GND id with a catalog RSN).
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.mcp.tool_registry import ToolRegistry


class TestGndEntryDict(unittest.TestCase):
    def test_url_and_swb_url_present(self):
        d = ToolRegistry._gnd_entry_dict(
            "4047979-1", "Quantenchemie", "desc", ["Syn"], ["540"], "106192760"
        )
        self.assertEqual(d["url"], "https://d-nb.info/gnd/4047979-1")
        self.assertEqual(
            d["swb_url"],
            "https://swb.bsz-bw.de/DB=2.104/PPNSET?PPN=106192760&INDEXSET=21",
        )

    def test_no_ppn_omits_swb_url(self):
        d = ToolRegistry._gnd_entry_dict("4047979-1", "Q", "", [], [], "")
        self.assertNotIn("swb_url", d)
        self.assertEqual(d["url"], "https://d-nb.info/gnd/4047979-1")

    def test_malformed_gnd_id_omits_url(self):
        d = ToolRegistry._gnd_entry_dict("not-a-gnd", "Q", "", [], [], "")
        self.assertNotIn("url", d)


class TestSerializeSuggesterResults(unittest.TestCase):
    def test_gnd_urls_added_from_gnd_ids_set(self):
        results = {
            "quantenchemie": {
                "Quantenchemie": {"gnd_ids": {"4047979-1"}, "count": 3,
                                  "classifications": {"ddc": {"540"}}}
            }
        }
        out = ToolRegistry._serialize_suggester_results(results)
        row = out["quantenchemie"]["Quantenchemie"]
        self.assertEqual(row["gnd_urls"], ["https://d-nb.info/gnd/4047979-1"])
        # nested classification sets are serialized to lists too
        self.assertEqual(row["classifications"], {"ddc": ["540"]})

    def test_no_gnd_ids_no_gnd_urls(self):
        out = ToolRegistry._serialize_suggester_results(
            {"t": {"K": {"count": 1, "classifications": {"ddc": {"540"}}}}}
        )
        self.assertNotIn("gnd_urls", out["t"]["K"])


class TestHandleSearchGnd(unittest.TestCase):
    def test_handler_includes_url(self):
        reg = ToolRegistry()
        km = MagicMock()
        km.search_local_gnd.return_value = [
            SimpleNamespace(
                gnd_id="4047979-1", title="Quantenchemie", description="d",
                synonyms=[], ddcs=["540"], ppn="106192760",
            )
        ]
        reg._get_knowledge_manager = MagicMock(return_value=km)
        payload = json.loads(reg._handle_search_gnd("Quantenchemie"))
        entry = payload["entries"][0]
        self.assertEqual(entry["url"], "https://d-nb.info/gnd/4047979-1")
        self.assertTrue(entry["swb_url"].endswith("PPN=106192760&INDEXSET=21"))


if __name__ == "__main__":
    unittest.main()
