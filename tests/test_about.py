"""Project self-description: payload shape, tool wiring, and drift guard.

The facts live in ``src/core/about.py`` and are repeated for humans in
``README.md`` and ``CLAUDE.md``. Three copies drift, so the publication fields
are pinned against the README here — a citation that quietly disagrees with
itself is worse than none, because the agent states it with full confidence.

Claude Generated.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.core.about import PUBLICATION, about_payload

ROOT = Path(__file__).resolve().parent.parent


class TestAboutPayload(unittest.TestCase):
    def setUp(self):
        self.payload = about_payload()

    def test_carries_the_fields_the_tool_promises(self):
        for key in ("name", "expansion", "summary", "status", "institution",
                    "capabilities", "pipeline_modes", "publication",
                    "repository", "license", "contributors",
                    "acknowledgements", "see_also"):
            self.assertIn(key, self.payload)

    def test_names_both_pipeline_modes(self):
        self.assertEqual(set(self.payload["pipeline_modes"]), {"classic", "agentic"})

    def test_see_also_points_at_tools_that_exist(self):
        """A pointer to a tool name that does not exist sends the agent
        chasing something it cannot call."""
        from src.mcp.tool_registry import ToolRegistry

        reg = ToolRegistry()
        reg.register_all_tools()
        names = set(reg.get_tool_names())
        for tool in self.payload["see_also"]:
            self.assertIn(tool, names)

    def test_status_does_not_oversell(self):
        """The maturity caveat is the point of the field — a status that
        claims completeness would be worse than no status."""
        self.assertIn("In Entwicklung", self.payload["status"])

    def test_is_json_serialisable(self):
        json.loads(json.dumps(self.payload, ensure_ascii=False))

    def test_returns_copies_not_the_module_state(self):
        """A caller mutating the payload must not corrupt the source of truth."""
        self.payload["publication"]["doi"] = "10.0000/nope"
        self.payload["capabilities"].clear()
        self.payload["pipeline_modes"].clear()
        self.payload["contributors"].clear()
        self.payload["see_also"].clear()
        fresh = about_payload()
        self.assertEqual(fresh["publication"]["doi"], "10.1515/bfp-2026-0014")
        for key in ("capabilities", "pipeline_modes", "contributors", "see_also"):
            self.assertTrue(fresh[key], f"{key} was emptied through the payload")

    def test_citation_contains_the_verifiable_parts(self):
        citation = PUBLICATION["citation"]
        for part in (PUBLICATION["journal"], PUBLICATION["volume"],
                     PUBLICATION["year"], PUBLICATION["pages"], PUBLICATION["doi"]):
            self.assertIn(part, citation)


class TestReadmeAgrees(unittest.TestCase):
    """The README is the human-facing copy of the same facts."""

    def setUp(self):
        self.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.claude_md = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    def test_readme_quotes_the_same_publication(self):
        for part in (PUBLICATION["doi"], PUBLICATION["journal"],
                     PUBLICATION["pages"], PUBLICATION["year"]):
            self.assertIn(part, self.readme, f"README missing {part!r}")

    def test_readme_agrees_on_licence_and_contributors(self):
        from src.core.about import CONTRIBUTORS, LICENSE

        self.assertIn(LICENSE, self.readme)
        for person in CONTRIBUTORS:
            self.assertIn(person, self.readme)

    def test_claude_md_quotes_the_doi(self):
        self.assertIn(PUBLICATION["doi"], self.claude_md)

    def test_expansion_matches_the_readme_title(self):
        from src.core.about import EXPANSION

        self.assertIn(EXPANSION, self.readme.splitlines()[0])
        self.assertIn(EXPANSION, self.claude_md)


class TestPromptsRouteSelfQuestions(unittest.TestCase):
    """The tool description alone did not get the agent there.

    Measured against LLMachine/nemotron-3.5 before this rule existed: "which
    sources are active?" and "how reliable are your suggestions?" produced NO
    tool call and an answer as the base model ("developed by researchers at
    NVIDIA"). Both tiers therefore carry the routing rule explicitly.
    """

    def _prompts(self):
        from src.core.chat_prompts import build_system_prompt

        return {
            "full": build_system_prompt(mode="general", compact=False),
            "compact": build_system_prompt(mode="general", compact=True),
        }

    def test_both_tiers_route_self_questions_to_the_tool(self):
        for tier, prompt in self._prompts().items():
            with self.subTest(tier=tier):
                self.assertIn("about_alima", prompt)
                self.assertIn("list_plugins", prompt)

    def test_both_tiers_forbid_answering_as_the_base_model(self):
        for tier, prompt in self._prompts().items():
            with self.subTest(tier=tier):
                self.assertIn("Basismodell", prompt)


class TestAboutTool(unittest.TestCase):
    def _registry(self):
        from src.mcp.tool_registry import ToolRegistry

        return ToolRegistry()

    def test_tool_is_registered_and_returns_the_payload(self):
        reg = self._registry()
        reg.register_all_tools()
        self.assertIn("about_alima", reg.get_tool_names())
        data = json.loads(reg.execute("about_alima", {}))
        self.assertEqual(data["expansion"], about_payload()["expansion"])
        self.assertEqual(data["publication"]["doi"], PUBLICATION["doi"])

    def test_answerable_without_config_or_database(self):
        """'What is ALIMA?' must work even when nothing else is reachable."""
        from src.mcp.tool_registry import ToolRegistry

        reg = ToolRegistry.__new__(ToolRegistry)
        reg._config_manager = MagicMock(
            load_config=MagicMock(side_effect=RuntimeError("no config"))
        )
        data = json.loads(ToolRegistry._handle_about_alima(reg))
        self.assertEqual(data["name"], "ALIMA")

    def test_schema_takes_no_arguments(self):
        from src.mcp import tool_schemas

        self.assertEqual(tool_schemas.ABOUT_ALIMA.parameters.get("properties"), {})


if __name__ == "__main__":
    unittest.main()
