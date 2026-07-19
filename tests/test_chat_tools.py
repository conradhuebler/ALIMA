"""Tests for src/ui/chat_tools (P-δ.2). Claude Generated.

Covers 8 cases per plan:
 1. registry name-collision raises ValueError
 2. available_for filter — populated vs empty SharedContext
 3. list_available_data with Cadmium-like SharedContext
 4. validate_gnd_term — session pool hit
 5. validate_gnd_term — MCP fallback hit
 6. MCP adapter blocks store_search_result when no_cache_writes=True
 7. MCP adapter allows store_search_result when no_cache_writes=False
 8. ChatToolRegistry — AgentLoop contract (get_tool_schemas + execute shape)
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.agents.shared_context import SharedContext
from src.ui.chat_tools import (
    BaseChatTool,
    ChatToolRegistry,
    build_chat_toolset,
)
from src.ui.chat_tools.alima import (
    GetDkClassificationsTool,
    GetKeywordsTool,
    ValidateGndTermTool,
)
from src.ui.chat_tools.generic import (
    GetExtraTool,
    ListAvailableDataTool,
)
from src.ui.chat_tools.mcp_adapter import build_mcp_chat_tools


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _cadmium_shared_context() -> SharedContext:
    """Inline-construct (not file-fixture) per plan decision.

    Loosely Cadmium-themed: 3 GND entries, 2 DK codes, 1 keyword chain.
    """
    ctx = SharedContext(
        abstract="Cadmium und seine toxikologischen Eigenschaften.",
        initial_keywords=["Cadmium"],
        extracted_keywords=["Cadmium", "Schwermetall", "Toxikologie"],
        gnd_entries=[
            {"title": "Cadmium", "gnd_id": "4007249-3", "classifications": {"ddc": ["546.48"]}},
            {"title": "Schwermetall", "gnd_id": "4054086-9", "classifications": {"ddc": ["546.3"]}},
            {"title": "Toxikologie", "gnd_id": "4060451-7", "classifications": {"ddc": ["615.9"]}},
        ],
        selected_keywords=[
            {"keyword": "Cadmium", "gnd_id": "4007249-3"},
        ],
        keyword_chains=[
            {"chain": ["Cadmium", "Schwermetall"], "reason": "Hyperonym"},
        ],
        dk_classifications=[
            {"code": "546.48", "title": "Cadmium", "confidence": 0.95,
             "reasoning": "primary element"},
            {"code": "615.9", "title": "Toxikologie", "confidence": 0.78,
             "reasoning": "health impact"},
        ],
        dk_search_results=[
            {"dk": "546.48", "titles": ["Cadmium-Verbindungen"], "count": 12},
            {"dk": "615.9", "titles": ["Schwermetall-Toxizität"], "count": 7},
        ],
    )
    ctx.step_results = {
        "extraction": {"keywords": ["Cadmium", "Schwermetall"]},
        "chunk_0": {"response": "Cadmium ist ein Schwermetall.", "keywords": ["Cadmium"]},
        "chunk_1": {"response": "Toxische Wirkung auf Nieren.", "keywords": ["Niere"]},
    }
    ctx.extra = {"final_keywords": [{"keyword": "Cadmium", "gnd_id": "4007249-3"}]}
    return ctx


def _empty_session() -> SimpleNamespace:
    return SimpleNamespace(last_shared_context=None, messages=[])


def _cadmium_session() -> SimpleNamespace:
    return SimpleNamespace(
        last_shared_context=_cadmium_shared_context(),
        messages=[
            {"role": "user", "content": "Hallo"},
            {"role": "assistant", "content": "Hi"},
        ],
    )


class _DummyTool(BaseChatTool):
    name = "dummy"
    description = "noop"
    parameters_schema = {"type": "object", "properties": {}}

    def execute(self, session, **_):
        return json.dumps({"ok": True})


def _mcp_with_store_and_search() -> MagicMock:
    """Mock MCP registry with two tool defs: search_gnd + store_search_result."""

    search_def = SimpleNamespace(
        name="search_gnd",
        description="Search GND.",
        parameters={"type": "object", "properties": {"term": {"type": "string"}}},
    )
    store_def = SimpleNamespace(
        name="store_search_result",
        description="Persist search result (write).",
        parameters={"type": "object", "properties": {}},
    )
    registry = MagicMock()
    registry._tools = {"search_gnd": search_def, "store_search_result": store_def}
    registry.get_tool_names.return_value = ["search_gnd", "store_search_result"]
    return registry


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


class TestChatToolRegistry(unittest.TestCase):

    # 1
    def test_registry_collision_raises(self):
        reg = ChatToolRegistry(session=_empty_session())
        reg.add(_DummyTool())
        with self.assertRaises(ValueError):
            reg.add(_DummyTool())

    # 8
    def test_chat_tool_registry_agentloop_contract(self):
        session = _cadmium_session()
        reg = ChatToolRegistry(session=session)
        reg.add(GetKeywordsTool())

        schemas = reg.get_tool_schemas(None)
        self.assertEqual(len(schemas), 1)
        for required_key in ("name", "description", "parameters"):
            self.assertIn(required_key, schemas[0])

        raw = reg.execute("get_keywords", {"kind": "initial"})
        self.assertIsInstance(raw, str)
        parsed = json.loads(raw)
        self.assertEqual(parsed["kind"], "initial")
        self.assertEqual(parsed["keywords"], ["Cadmium"])

        # Unknown tool returns JSON error rather than raising
        err = json.loads(reg.execute("does_not_exist", {}))
        self.assertIn("error", err)


class TestAvailableForFilter(unittest.TestCase):

    # 2
    def test_available_for_filter(self):
        full = _cadmium_session()
        empty = _empty_session()

        # GetExtraTool: populated extra -> available; empty -> not
        self.assertTrue(GetExtraTool().available_for(full))
        self.assertFalse(GetExtraTool().available_for(empty))

        # GetDkClassificationsTool: same gating
        self.assertTrue(GetDkClassificationsTool().available_for(full))
        self.assertFalse(GetDkClassificationsTool().available_for(empty))

        # ListAvailableDataTool is always available so the agent can probe
        # pipeline state (see commit 1eec641 — "always available" replaced
        # the previous "gated on shared_context" check).
        self.assertTrue(ListAvailableDataTool().available_for(full))
        self.assertTrue(ListAvailableDataTool().available_for(empty))


class TestListAvailableData(unittest.TestCase):

    # 3
    def test_list_available_data_with_cadmium_state(self):
        session = _cadmium_session()
        raw = ListAvailableDataTool().execute(session)
        payload = json.loads(raw)
        self.assertIn("dk_classifications", payload["slots_populated"])
        self.assertIn("gnd_entries", payload["slots_populated"])
        # gnd_entries has 3 items, dk_classifications has 2
        self.assertEqual(payload["counts"]["gnd_entries"], 3)
        self.assertEqual(payload["counts"]["dk_classifications"], 2)


class TestValidateGndTerm(unittest.TestCase):

    # 4
    def test_validate_gnd_term_session_hit(self):
        tool = ValidateGndTermTool(mcp_registry=None)
        raw = tool.execute(_cadmium_session(), term="Cadmium")
        parsed = json.loads(raw)
        self.assertTrue(parsed["verified"])
        self.assertEqual(parsed["gnd_id"], "4007249-3")
        self.assertEqual(parsed["source"], "session")

    # 5
    def test_validate_gnd_term_mcp_fallback(self):
        mcp = MagicMock()
        mcp.execute.return_value = json.dumps(
            {"results": [{"title": "Krypton", "gnd_id": "4165800-9"}]}
        )
        tool = ValidateGndTermTool(mcp_registry=mcp)
        # Term not in session pool -> hits MCP
        raw = tool.execute(_cadmium_session(), term="Krypton")
        parsed = json.loads(raw)
        self.assertTrue(parsed["verified"])
        self.assertEqual(parsed["gnd_id"], "4165800-9")
        self.assertEqual(parsed["source"], "mcp")
        mcp.execute.assert_called_once_with(
            "search_gnd", {"term": "Krypton", "min_results": 1}
        )


class TestMcpAdapterFiltering(unittest.TestCase):

    # 6
    def test_mcp_adapter_blocks_store_when_no_cache_writes(self):
        registry = _mcp_with_store_and_search()
        tools = build_mcp_chat_tools(no_cache_writes=True, mcp_registry=registry)
        names = {t.name for t in tools}
        self.assertIn("search_gnd", names)
        self.assertNotIn("store_search_result", names)

    # 7
    def test_mcp_adapter_allows_store_when_writes_enabled(self):
        registry = _mcp_with_store_and_search()
        tools = build_mcp_chat_tools(no_cache_writes=False, mcp_registry=registry)
        names = {t.name for t in tools}
        self.assertIn("search_gnd", names)
        self.assertIn("store_search_result", names)


class TestBuildChatToolset(unittest.TestCase):
    """Bonus integration smoke for build_chat_toolset (not part of 8-case
    contract but cheap to add for confidence)."""

    def test_build_chat_toolset_assembles_layers(self):
        registry = _mcp_with_store_and_search()
        chat_config = SimpleNamespace(no_cache_writes=True)
        reg = build_chat_toolset(
            _cadmium_session(), chat_config=chat_config, mcp_registry=registry
        )
        names = set(reg.available_tools())
        # Generic + ALIMA layer (subset that is available_for full session)
        self.assertIn("list_available_data", names)
        self.assertIn("get_keywords", names)
        self.assertIn("get_dk_classifications", names)
        # MCP layer present, write filtered
        self.assertIn("search_gnd", names)
        self.assertNotIn("store_search_result", names)


if __name__ == "__main__":
    unittest.main()
