"""PromptService.get_prompt_set_overview: which set wins which model key - Claude Generated

The overview drives the quick-select buttons in the Abstract tab's Prompt
sub-tab: only runtime-reachable sets get a button, and the set winning the
``default`` key is marked as the pipeline standard. The winner logic must match
``_build_model_index`` (last assignment wins), so one test pins the overview
against the real ``get_prompt_config`` resolution.
"""

import json
import logging
import os
import tempfile
import unittest

from src.llm.prompt_service import PromptService

logging.disable(logging.CRITICAL)


def _pset(prompt, models):
    return [prompt, "sys", "0.7", "0.1", models, "0"]


class TestPromptSetOverview(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        config = {
            "keywords": {
                "fields": ["abstract"],
                "required": ["abstract"],
                "prompts": [
                    _pset("legacy default", ["default"]),
                    _pset("qwen prompt", ["qwen2.5:14b", "cogito:14b"]),
                    _pset("active default", ["default"]),
                ],
            }
        }
        self.path = os.path.join(self.tmpdir.name, "prompts.json")
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        self.service = PromptService(self.path, logger=logging.getLogger("test"))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_later_default_shadows_earlier(self):
        overview = self.service.get_prompt_set_overview("keywords")
        self.assertEqual(
            [info["live_models"] for info in overview],
            [[], ["cogito:14b", "qwen2.5:14b"], ["default"]],
        )
        self.assertEqual(
            [info["wins_default"] for info in overview], [False, False, True]
        )

    def test_overview_matches_runtime_resolution(self):
        overview = self.service.get_prompt_set_overview("keywords")
        default_winner = next(i for i in overview if i["wins_default"])
        resolved = self.service.get_prompt_config("keywords", "unbekanntes-modell")
        self.assertEqual(resolved.prompt, "active default")
        self.assertEqual(default_winner["index"], 2)

        resolved_exact = self.service.get_prompt_config("keywords", "cogito:14b")
        self.assertEqual(resolved_exact.prompt, "qwen prompt")

    def test_unknown_task_returns_empty(self):
        self.assertEqual(self.service.get_prompt_set_overview("gibtsnicht"), [])


if __name__ == "__main__":
    unittest.main()
