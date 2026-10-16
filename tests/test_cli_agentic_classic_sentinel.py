"""``alima pipeline --agentic`` without ``--workflow`` used to confuse
itself when SystemConfig.default_workflow is "__classic__".

PipelineConfig.create_from_provider_preferences() sets
pipeline_config.workflow_name from SystemConfig.default_workflow
unconditionally. That default can legitimately be the literal string
"__classic__" — the sentinel src/ui/settings/workflow_list.py::CLASSIC_STEM
writes when the operator stars the classic, non-agentic pipeline as the
default in Settings. handle_pipeline then force-sets enable_agentic_mode=True
for --agentic without checking that, so a run with --agentic and no explicit
--workflow would try to load a workflow literally named "__classic__" (no
such file exists), and the error message named that internal sentinel
directly — something the operator never typed and has no reason to
recognize. Confirmed by the actual resolution chain:
_pipeline_agentic.py::_resolve_workflow_path -> find_workflow_file(name) ->
None -> "Agentic workflow '__classic__' not found. Use `alima workflows
list`...".

agentic_mode_needs_an_explicit_workflow() catches the contradiction before
that point and fails with a message that tells the operator what to do
instead. - Claude Generated
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.cli.commands.pipeline_cmd import agentic_mode_needs_an_explicit_workflow


def _args(**kw):
    return SimpleNamespace(**kw)


def _config(workflow_name):
    return SimpleNamespace(workflow_name=workflow_name)


class AgenticModeNeedsAnExplicitWorkflowTest(unittest.TestCase):
    def test_classic_default_with_no_override_is_a_contradiction(self):
        self.assertTrue(
            agentic_mode_needs_an_explicit_workflow(_config("__classic__"), _args())
        )

    def test_a_real_default_workflow_is_fine(self):
        self.assertFalse(
            agentic_mode_needs_an_explicit_workflow(_config("alima_v51"), _args())
        )

    def test_explicit_workflow_name_already_overrode_the_classic_default(self):
        """By the time this runs, --workflow has already been written onto
        pipeline_config.workflow_name (see handle_pipeline) - so this checks
        the post-override state, same as the real call site does."""
        self.assertFalse(
            agentic_mode_needs_an_explicit_workflow(_config("alima_v51"), _args(workflow="alima_v51"))
        )

    def test_custom_workflow_path_also_resolves_the_contradiction(self):
        self.assertFalse(
            agentic_mode_needs_an_explicit_workflow(
                _config("__classic__"), _args(custom_workflow="/tmp/my.yaml")
            )
        )


if __name__ == "__main__":
    unittest.main()
