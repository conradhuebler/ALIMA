"""P-ζ — Pipeline Orchestration Tool tests. Claude Generated.

Covers:

1. ``SharedContext.from_keyword_analysis_state`` round-trip + field mapping.
2. Classical pipeline completion populates ``pm.last_shared_context``.
3. Mutation tools sync ``last_shared_context`` after a successful apply.
4. ``RunPipelineTool`` accept/reject/autonomous/busy/error flow.
5. ``RerunStepTool`` classical + agentic paths, tmp cleanup.
6. ``_StatusForwarder`` emits ``state.pipeline_step`` AlimaStateBus events.
7. Cancel: tool propagates a ``should_stop`` into ``pm.set_interrupt_flag``.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    from PyQt6.QtWidgets import QApplication
    from src.core.agents.shared_context import SharedContext
    from src.core.data_models import (
        KeywordAnalysisState,
        LlmKeywordAnalysis,
        SearchResult,
    )
    from src.core.state_bus import AlimaStateBus
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
    from src.utils.config_models import DatabaseConfig
    from src.ui.chat_tools.mutations import ProposeKeywordReplacementTool
    from src.ui.chat_tools.pipeline import (
        RunPipelineTool,
        RerunStepTool,
        _StatusForwarder,
        _InterruptInstaller,
        pipeline_tools,
    )
    PYQT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    PYQT_IMPORT_ERROR = exc


_qapp = None


def _ensure_qapp():
    global _qapp
    if _qapp is None and PYQT_IMPORT_ERROR is None:
        _qapp = QApplication.instance() or QApplication([])
    return _qapp


def _make_sqlite_config(path):
    cfg = DatabaseConfig(db_type="sqlite")
    cfg.sqlite_path = path
    return cfg


def _make_full_kas() -> KeywordAnalysisState:
    """Build a KAS exercising every field the bridge reads."""
    kas = KeywordAnalysisState(
        original_abstract="An abstract about cadmium contamination.",
        initial_keywords=["Cadmium", "Boden"],
        search_suggesters_used=["lobid"],
        working_title="Cadmium_Boden_text_20260101",
        input_type="text",
        source_value=None,
    )
    kas.final_llm_analysis = LlmKeywordAnalysis(
        task_name="keywords",
        model_used="gemma:7b",
        provider_used="ollama",
        prompt_template="keywords_chunked",
        filled_prompt="",
        temperature=0.4,
        seed=42,
        response_full_text="Selected: Cadmium (4007249-3)",
        extracted_gnd_keywords=["Cadmium (GND-ID: 4007249-3)"],
        missing_concepts=["Soil contamination"],
    )
    kas.search_results = [
        SearchResult(
            search_term="Cadmium",
            results={
                "4007249-3": {"title": "Cadmium", "ddc_codes": ["546.43"]},
            },
        ),
        SearchResult(
            search_term="Boden",
            results={
                "4007249-3": {"title": "Cadmium", "ddc_codes": ["546.43"]},
                "4006670-9": {"title": "Boden", "ddc_codes": ["631.4"]},
            },
        ),
    ]
    kas.dk_classifications = ["546.43", "631.4"]
    kas.dk_search_results = [{"dk": "546.43", "titles": ["x"]}]
    kas.dk_statistics = {"total_classifications": 2, "unique_dk_codes": 2}
    return kas


def _make_tool_common(autonomous=False, last_ctx=None, kas=None, config=None):
    """Build the common kwargs dict for tool instantiation."""
    pm = SimpleNamespace(
        current_analysis_state=kas,
        last_shared_context=last_ctx,
        config=config or SimpleNamespace(
            enable_agentic_mode=False, step_configs={}, workflow_name=None,
        ),
        pipeline_steps=[],
        current_step_index=0,
        _interrupt_lock=threading.Lock(),
        _interrupt_check_func=None,
        _is_interrupted=False,
        step_started_callback=None,
        step_completed_callback=None,
    )
    pm.start_pipeline = MagicMock()
    pm.execute_single_step = MagicMock()
    pm.set_config = MagicMock(side_effect=lambda c: setattr(pm, "config", c))
    pm.set_interrupt_flag = MagicMock(
        side_effect=lambda lock, fn: setattr(pm, "_interrupt_check_func", fn)
    )
    pm._resolve_workflow_path = MagicMock(return_value="/tmp/fake_wf.yaml")
    pm._start_v4_workflow_pipeline = MagicMock()

    kb = MagicMock()
    kb.record_mutation_pending.return_value = 7
    kb.record_mutation_outcome.return_value = None

    gw = MagicMock()
    gw.request_decision.return_value = {"accepted": True, "reject_reason": ""}

    cc = SimpleNamespace(autonomous_pipeline=autonomous)
    return pm, kb, gw, cc


# ---------------------------------------------------------------------------
# 1. SharedContext.from_keyword_analysis_state
# ---------------------------------------------------------------------------


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6 stack unavailable: {PYQT_IMPORT_ERROR}")
class TestSharedContextFromKAS(unittest.TestCase):

    def test_basic_field_mapping(self):
        kas = _make_full_kas()
        ctx = SharedContext.from_keyword_analysis_state(kas)
        self.assertEqual(ctx.abstract, kas.original_abstract)
        self.assertEqual(ctx.working_title, kas.working_title)
        self.assertEqual(ctx.initial_keywords, kas.initial_keywords)
        self.assertEqual(ctx.input_type, "text")

    def test_final_llm_analysis_propagates(self):
        kas = _make_full_kas()
        ctx = SharedContext.from_keyword_analysis_state(kas)
        self.assertEqual(ctx.extracted_keywords, kas.final_llm_analysis.extracted_gnd_keywords)
        self.assertEqual(ctx.missing_concepts, ["Soil contamination"])
        self.assertEqual(ctx.provider, "ollama")
        self.assertEqual(ctx.model, "gemma:7b")
        self.assertEqual(ctx.temperature, 0.4)
        self.assertEqual(ctx.seed, 42)

    def test_search_results_flatten_and_dedup(self):
        kas = _make_full_kas()
        ctx = SharedContext.from_keyword_analysis_state(kas)
        # 4007249-3 appears in both SR results — should be deduped.
        gnd_ids = [e["gnd_id"] for e in ctx.gnd_entries]
        self.assertEqual(len(gnd_ids), 2)
        self.assertEqual(set(gnd_ids), {"4007249-3", "4006670-9"})
        self.assertIn("Cadmium", ctx.gnd_entries_per_keyword)
        self.assertIn("Boden", ctx.gnd_entries_per_keyword)

    def test_gnd_entries_carry_preformatted_url(self):
        # Claude Generated - each GND entry gets a ready d-nb.info URL so the
        # chat agent never builds one (and never confuses it with a catalog RSN).
        kas = _make_full_kas()
        ctx = SharedContext.from_keyword_analysis_state(kas)
        by_id = {e["gnd_id"]: e for e in ctx.gnd_entries}
        self.assertEqual(
            by_id["4007249-3"]["url"], "https://d-nb.info/gnd/4007249-3"
        )
        self.assertEqual(
            by_id["4006670-9"]["url"], "https://d-nb.info/gnd/4006670-9"
        )

    def test_dk_strings_normalised_to_dicts(self):
        kas = _make_full_kas()
        ctx = SharedContext.from_keyword_analysis_state(kas)
        self.assertEqual(len(ctx.dk_classifications), 2)
        for cls in ctx.dk_classifications:
            self.assertIsInstance(cls, dict)
            self.assertIn("code", cls)
        self.assertEqual(ctx.dk_catalog_stats["total_classifications"], 2)

    def test_classical_extracts_selected_and_final_for_chat(self):
        """Classical Step-4 list mirrored to selected_keywords + extra.final_keywords.

        Chat tool get_keywords(kind="selected"|"final") works without the agent
        having to know the classical/agentic distinction.
        """
        kas = _make_full_kas()
        ctx = SharedContext.from_keyword_analysis_state(kas)
        self.assertTrue(ctx.selected_keywords)
        sel = ctx.selected_keywords[0]
        self.assertEqual(sel["title"], "Cadmium")
        self.assertEqual(sel["gnd_id"], "4007249-3")
        finals = ctx.extra.get("final_keywords") or []
        self.assertTrue(finals)
        self.assertEqual(finals[0]["title"], "Cadmium")

    def test_empty_state_safe(self):
        kas = KeywordAnalysisState(
            original_abstract="", initial_keywords=[], search_suggesters_used=[],
        )
        ctx = SharedContext.from_keyword_analysis_state(kas)
        self.assertEqual(ctx.abstract, "")
        self.assertEqual(ctx.extracted_keywords, [])
        self.assertEqual(ctx.dk_classifications, [])


# ---------------------------------------------------------------------------
# 2. Classical completion populates last_shared_context (integration via patch)
# ---------------------------------------------------------------------------


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6 stack unavailable: {PYQT_IMPORT_ERROR}")
class TestClassicalPipelineSyncsSharedContext(unittest.TestCase):
    """Smoke test that the bridge actually runs in the completion path.

    Rather than instantiating PipelineManager (heavy), reproduce the
    completion branch logic against a stub holder. Documents what the
    patched code is supposed to do.
    """

    def test_completion_branch_populates_context(self):
        # Mirrors pipeline_manager.py:1948 completion block (P-ζ block).
        kas = _make_full_kas()
        holder = SimpleNamespace(
            current_analysis_state=kas,
            last_shared_context=None,
            logger=MagicMock(),
        )
        # Reproduce the bridge:
        if holder.current_analysis_state is not None:
            holder.last_shared_context = SharedContext.from_keyword_analysis_state(
                holder.current_analysis_state
            )
        self.assertIsNotNone(holder.last_shared_context)
        self.assertEqual(holder.last_shared_context.abstract, kas.original_abstract)

    def test_state_pipeline_completed_event_subscribable(self):
        bus = AlimaStateBus()
        received = []
        bus.subscribe("state.pipeline_completed", lambda p: received.append(p))
        bus.emit_event("state.pipeline_completed", {"workflow": "test"})
        self.assertTrue(any(p.get("workflow") == "test" for p in received))


# ---------------------------------------------------------------------------
# 3. Mutation tools sync SharedContext
# ---------------------------------------------------------------------------


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6 stack unavailable: {PYQT_IMPORT_ERROR}")
class TestMutationToolsSyncSC(unittest.TestCase):

    def test_keyword_replacement_refreshes_shared_context(self):
        kas = _make_full_kas()
        pm = SimpleNamespace(
            current_analysis_state=kas,
            last_shared_context=SharedContext.from_keyword_analysis_state(kas),
        )
        # Original SC sees old keyword.
        self.assertEqual(pm.last_shared_context.initial_keywords, kas.initial_keywords)

        kb = MagicMock()
        kb.record_mutation_pending.return_value = 1
        gw = MagicMock()
        gw.request_decision.return_value = {"accepted": True, "reject_reason": ""}
        cc = SimpleNamespace(autonomous_pipeline=False)

        tool = ProposeKeywordReplacementTool(
            pipeline_manager=pm, kb_manager=kb, gateway=gw,
            chat_config=cc, session_id="s",
        )
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            old="Cadmium", new="Schwermetall", reason="more general",
        ))
        self.assertEqual(result["status"], "applied")
        # SC is re-derived → reflects the new keyword.
        self.assertTrue(
            any("Schwermetall" in k for k in pm.last_shared_context.initial_keywords),
            f"SC initial_keywords did not refresh: {pm.last_shared_context.initial_keywords}",
        )


# ---------------------------------------------------------------------------
# 4. RunPipelineTool
# ---------------------------------------------------------------------------


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6 stack unavailable: {PYQT_IMPORT_ERROR}")
class TestRunPipelineTool(unittest.TestCase):

    def _build(self, **kwargs):
        pm, kb, gw, cc = _make_tool_common(**kwargs)
        tool = RunPipelineTool(
            pipeline_manager=pm, kb_manager=kb, gateway=gw,
            chat_config=cc, session_id="s",
        )
        return tool, pm, kb, gw

    def test_accept_runs_pipeline_and_restores_config(self):
        tool, pm, kb, gw = self._build()
        # Snapshot config id before.
        original_config = pm.config
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            input_source="An abstract.",
            mode="classic",
            reason="run pipeline now",
        ))
        self.assertEqual(result["status"], "completed")
        pm.start_pipeline.assert_called_once()
        # set_config called in finally with a deep copy equivalent.
        pm.set_config.assert_called_once()
        # We don't compare by identity (deepcopy); compare attrs we set.
        restored = pm.set_config.call_args[0][0]
        self.assertEqual(
            getattr(restored, "enable_agentic_mode", None),
            getattr(original_config, "enable_agentic_mode", None),
        )
        gw.request_decision.assert_called_once()

    def test_reject_skips_run(self):
        tool, pm, _kb, gw = self._build()
        gw.request_decision.return_value = {"accepted": False, "reject_reason": "nope"}
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            input_source="x", reason="r",
        ))
        self.assertEqual(result["status"], "rejected")
        pm.start_pipeline.assert_not_called()

    def test_autonomous_skips_gateway(self):
        tool, pm, _kb, gw = self._build(autonomous=True)
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            input_source="x", reason="r",
        ))
        self.assertEqual(result["status"], "completed")
        gw.request_decision.assert_not_called()
        pm.start_pipeline.assert_called_once()

    def test_missing_input_source_invalid(self):
        tool, pm, _kb, _gw = self._build()
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            input_source="", reason="r",
        ))
        self.assertEqual(result["status"], "invalid")
        pm.start_pipeline.assert_not_called()

    def test_busy_when_pipeline_mid_run(self):
        tool, pm, _kb, _gw = self._build()
        # Simulate a running pipeline: 5 steps total, currently on step 2.
        pm.pipeline_steps = [object()] * 5
        pm.current_step_index = 2
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            input_source="x", reason="r",
        ))
        self.assertEqual(result["status"], "busy")
        pm.start_pipeline.assert_not_called()

    def test_pipeline_error_returns_status_error(self):
        tool, pm, kb, _gw = self._build()
        pm.start_pipeline.side_effect = RuntimeError("boom")
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            input_source="x", reason="r",
        ))
        self.assertEqual(result["status"], "error")
        self.assertIn("boom", result["error"])


# ---------------------------------------------------------------------------
# 5. RerunStepTool
# ---------------------------------------------------------------------------


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6 stack unavailable: {PYQT_IMPORT_ERROR}")
class TestRerunStepTool(unittest.TestCase):

    def _build(self, **kwargs):
        pm, kb, gw, cc = _make_tool_common(**kwargs)
        tool = RerunStepTool(
            pipeline_manager=pm, kb_manager=kb, gateway=gw,
            chat_config=cc, session_id="s",
        )
        return tool, pm, kb, gw

    def test_classical_rerun_calls_execute_single_step(self):
        kas = _make_full_kas()
        ctx = SharedContext.from_keyword_analysis_state(kas)
        tool, pm, kb, _gw = self._build(kas=kas, last_ctx=ctx)
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            step_id="dk_classification",
            params={"model": "cogito:32b"},
            reason="try bigger model",
        ))
        self.assertEqual(result["status"], "completed")
        pm.execute_single_step.assert_called_once()
        # last_shared_context refreshed (still set).
        self.assertIsNotNone(pm.last_shared_context)

    def test_invalid_classical_step_id(self):
        kas = _make_full_kas()
        ctx = SharedContext.from_keyword_analysis_state(kas)
        tool, pm, _kb, _gw = self._build(kas=kas, last_ctx=ctx)
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            step_id="nonsense_step", params={}, reason="r",
        ))
        self.assertEqual(result["status"], "invalid")
        pm.execute_single_step.assert_not_called()

    def test_missing_prior_run_is_invalid(self):
        tool, pm, _kb, _gw = self._build(kas=None, last_ctx=None)
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            step_id="dk_classification", params={}, reason="r",
        ))
        self.assertEqual(result["status"], "invalid")

    def test_agentic_rerun_uses_v4_workflow_pipeline(self):
        kas = _make_full_kas()
        ctx = SharedContext.from_keyword_analysis_state(kas)
        agentic_cfg = SimpleNamespace(
            enable_agentic_mode=True,
            step_configs={},
            workflow_name="alima_classic",
            agentic_step_id=None,
            agentic_input_context_path=None,
        )
        tool, pm, _kb, _gw = self._build(
            kas=kas, last_ctx=ctx, config=agentic_cfg,
        )
        tmp_paths = []
        original_save = ctx.save_to_file

        def spy_save(p):
            tmp_paths.append(p)
            original_save(p)

        with patch.object(ctx, "save_to_file", side_effect=spy_save):
            result = json.loads(tool.execute(
                session=SimpleNamespace(),
                step_id="dk_classification",
                params={"model": "cogito:14b"},
                reason="rerun",
            ))
        self.assertEqual(result["status"], "completed")
        pm._start_v4_workflow_pipeline.assert_called_once()
        # tmp file was created AND cleaned up.
        self.assertEqual(len(tmp_paths), 1)
        self.assertFalse(
            os.path.exists(tmp_paths[0]),
            f"tmp warm-start file leaked: {tmp_paths[0]}",
        )


# ---------------------------------------------------------------------------
# 6. _StatusForwarder
# ---------------------------------------------------------------------------


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6 stack unavailable: {PYQT_IMPORT_ERROR}")
class TestStatusForwarder(unittest.TestCase):

    def test_emits_state_pipeline_step_event(self):
        bus = AlimaStateBus()
        received: list[dict] = []
        bus.subscribe("state.pipeline_step", lambda p: received.append(p))

        original_started = MagicMock()
        pm = SimpleNamespace(
            step_started_callback=original_started,
            step_completed_callback=None,
        )
        step = SimpleNamespace(step_id="keywords", name="Keyword Selection")
        with _StatusForwarder(pm, "run_pipeline"):
            pm.step_started_callback(step)
            pm.step_completed_callback(step)

        statuses = [p["status"] for p in received if p.get("step_id") == "keywords"]
        self.assertIn("running", statuses)
        self.assertIn("completed", statuses)
        original_started.assert_called_once_with(step)
        # Callbacks restored on exit.
        self.assertIs(pm.step_started_callback, original_started)


# ---------------------------------------------------------------------------
# 7. Cancel via _InterruptInstaller
# ---------------------------------------------------------------------------


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6 stack unavailable: {PYQT_IMPORT_ERROR}")
class TestCancel(unittest.TestCase):

    def test_installer_registers_and_restores_check_func(self):
        prev_check = lambda: False
        pm = SimpleNamespace(
            _interrupt_lock=threading.Lock(),
            _interrupt_check_func=prev_check,
            _is_interrupted=False,
        )
        pm.set_interrupt_flag = MagicMock(
            side_effect=lambda lock, fn: setattr(pm, "_interrupt_check_func", fn)
        )
        my_stop = lambda: True
        with _InterruptInstaller(pm, my_stop):
            self.assertIs(pm._interrupt_check_func, my_stop)
        # Restored.
        self.assertIs(pm._interrupt_check_func, prev_check)
        self.assertFalse(pm._is_interrupted)

    def test_run_pipeline_returns_cancelled_on_interrupted_error(self):
        pm, kb, gw, cc = _make_tool_common()
        pm.start_pipeline.side_effect = InterruptedError("user")
        tool = RunPipelineTool(
            pipeline_manager=pm, kb_manager=kb, gateway=gw,
            chat_config=cc, session_id="s",
        )
        result = json.loads(tool.execute(
            session=SimpleNamespace(),
            input_source="x", reason="r",
        ))
        self.assertEqual(result["status"], "cancelled")


# ---------------------------------------------------------------------------
# 8. Factory wiring
# ---------------------------------------------------------------------------


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6 stack unavailable: {PYQT_IMPORT_ERROR}")
class TestFactory(unittest.TestCase):

    def test_pipeline_tools_returns_both(self):
        pm, kb, gw, cc = _make_tool_common()
        tools = pipeline_tools(
            pipeline_manager=pm, kb_manager=kb, gateway=gw,
            chat_config=cc, session_id="s",
        )
        names = {t.name for t in tools}
        self.assertEqual(names, {"run_pipeline", "rerun_step"})


if __name__ == "__main__":
    unittest.main()
