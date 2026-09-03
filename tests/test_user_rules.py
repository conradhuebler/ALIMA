"""Persönliche Zusatzregeln — store, scoping, rendering, injection, exchange.

Claude Generated.

Two properties matter more than the rest and are pinned with a mutation probe
(assert the null case, then assert the mutated case *differs*), because a gate
test that a trivial fixture also passes is worth nothing:

* with no active rule every prompt is byte-identical to before the feature;
* a rejected confirmation writes nothing at all.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from src.core.user_rules import (
    RULES_BLOCK_HEADING,
    RuleStore,
    UserRule,
    append_rules_block,
    next_rule_id,
    render_rules_block,
    select_rules,
)


class _LiftLogDisable:
    """Several test modules call ``logging.disable(CRITICAL)`` at import, which
    would swallow the warnings these tests assert on. Lift it for the duration,
    as ``test_classification_rank.py`` / ``test_lookup_plugins.py`` do.
    - Claude Generated"""

    def setUp(self):
        super().setUp()
        import logging

        self._prev_disable = logging.root.manager.disable
        logging.disable(logging.NOTSET)

    def tearDown(self):
        import logging

        logging.disable(self._prev_disable)
        super().tearDown()


def _rule(**kwargs) -> UserRule:
    base = dict(id="r-1", text="Kernschlagworte sind zwei bis fünf.")
    base.update(kwargs)
    return UserRule(**base)


# ----------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------


class RuleStoreTest(_LiftLogDisable, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "rules.yaml"
        self.store = RuleStore(self.path)

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def test_missing_file_is_an_empty_rule_set(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.store.load(), [])

    def test_roundtrip_preserves_every_field(self):
        stored = self.store.add(
            "Formschlagwörter gehören nicht in core_keywords.",
            applies_when="bei Lehrbüchern",
            workflows=["alima_v51*"],
            steps=["selection*"],
            origin={"source": "chat", "note": "aus Lauf X", "author": "Testfall"},
        )
        self.assertIsNotNone(stored)

        (loaded,) = RuleStore(self.path).load()
        self.assertEqual(loaded.id, stored.id)
        self.assertEqual(loaded.text, "Formschlagwörter gehören nicht in core_keywords.")
        self.assertEqual(loaded.applies_when, "bei Lehrbüchern")
        self.assertEqual(loaded.workflows, ["alima_v51*"])
        self.assertEqual(loaded.steps, ["selection*"])
        self.assertTrue(loaded.enabled)
        self.assertEqual(loaded.origin["source"], "chat")
        self.assertEqual(loaded.origin["note"], "aus Lauf X")
        # `created` is stamped by the store, not by the caller.
        self.assertIn("created", loaded.origin)

    def test_ids_do_not_collide(self):
        first = self.store.add("Regel eins.")
        second = self.store.add("Regel zwei.")
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(len({r.id for r in self.store.load()}), 2)

    def test_next_rule_id_avoids_taken_ids(self):
        taken = [_rule(id=next_rule_id([]))]
        self.assertNotEqual(next_rule_id(taken), taken[0].id)

    def test_set_enabled_and_remove(self):
        rule = self.store.add("Regel.")
        self.assertTrue(self.store.set_enabled(rule.id, False))
        self.assertFalse(self.store.load()[0].enabled)
        self.assertTrue(self.store.remove(rule.id))
        self.assertEqual(self.store.load(), [])
        # A second removal reports failure rather than pretending success.
        self.assertFalse(self.store.remove(rule.id))

    def test_update_replaces_by_id(self):
        rule = self.store.add("Alt.")
        rule.text = "Neu."
        self.assertTrue(self.store.update(rule))
        self.assertEqual(self.store.load()[0].text, "Neu.")

    def test_malformed_yaml_yields_no_rules_and_does_not_raise(self):
        self.path.write_text("rules: [unclosed\n", encoding="utf-8")
        with self.assertLogs("src.core.user_rules", level="WARNING"):
            self.assertEqual(self.store.load(), [])

    def test_rules_key_of_wrong_type_is_ignored(self):
        self.path.write_text("version: 1\nrules: 'not a list'\n", encoding="utf-8")
        with self.assertLogs("src.core.user_rules", level="WARNING"):
            self.assertEqual(self.store.load(), [])

    def test_entry_without_text_is_skipped(self):
        self.path.write_text(
            "version: 1\nrules:\n  - id: r-1\n    text: ''\n  - id: r-2\n    text: 'Gilt.'\n",
            encoding="utf-8",
        )
        loaded = self.store.load()
        self.assertEqual([r.id for r in loaded], ["r-2"])

    def test_write_is_atomic_no_leftover_temp_files(self):
        self.store.add("Regel.")
        names = sorted(p.name for p in Path(self._tmp.name).iterdir())
        self.assertEqual(names, ["rules.yaml"])


# ----------------------------------------------------------------------
# Scoping
# ----------------------------------------------------------------------


class SelectRulesTest(unittest.TestCase):
    def test_disabled_rules_never_apply(self):
        rules = [_rule(enabled=False)]
        self.assertEqual(select_rules(rules, workflow="alima_v51", step="selection"), [])

    def test_star_matches_everything(self):
        rules = [_rule()]
        self.assertEqual(len(select_rules(rules, workflow="anything", step="anystep")), 1)

    def test_step_glob(self):
        rules = [_rule(steps=["selection*"])]
        self.assertEqual(len(select_rules(rules, workflow="w", step="selection")), 1)
        self.assertEqual(len(select_rules(rules, workflow="w", step="selection_chunks")), 1)
        self.assertEqual(len(select_rules(rules, workflow="w", step="extraction")), 0)

    def test_workflow_glob(self):
        rules = [_rule(workflows=["alima_v51*"])]
        self.assertEqual(len(select_rules(rules, workflow="alima_v51_105", step="s")), 1)
        self.assertEqual(len(select_rules(rules, workflow="alima_classic", step="s")), 0)

    def test_named_scope_does_not_leak_into_an_unknown_step(self):
        # An empty step id means "we could not identify the prompt". A rule
        # scoped to a named step must stay out of it.
        rules = [_rule(steps=["selection"])]
        self.assertEqual(select_rules(rules, workflow="w", step=""), [])
        self.assertEqual(len(select_rules([_rule()], workflow="", step="")), 1)

    def test_empty_text_never_applies(self):
        self.assertEqual(select_rules([_rule(text="   ")], workflow="w", step="s"), [])


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


class RenderTest(unittest.TestCase):
    def test_no_rules_renders_nothing(self):
        self.assertEqual(render_rules_block([]), "")

    def test_block_carries_heading_and_format_guard(self):
        block = render_rules_block([_rule()])
        self.assertIn(RULES_BLOCK_HEADING, block)
        self.assertIn("Antwortformat", block)
        self.assertIn("Kernschlagworte sind zwei bis fünf.", block)

    def test_condition_becomes_a_leading_clause(self):
        block = render_rules_block([_rule(applies_when="bei Überblickswerken")])
        self.assertIn("- Nur bei Überblickswerken: Kernschlagworte", block)

    def test_braces_in_a_rule_survive_verbatim(self):
        # The block is appended after the prompt's own {name} rendering, so a
        # brace must never be treated as a placeholder.
        block = render_rules_block([_rule(text="Schreibe {abstract} nicht ab.")])
        self.assertIn("{abstract}", block)

    def test_append_is_a_no_op_without_a_block(self):
        self.assertEqual(append_rules_block("SYSTEM", ""), "SYSTEM")

    def test_append_puts_the_block_last(self):
        out = append_rules_block("SYSTEM", render_rules_block([_rule()]))
        self.assertTrue(out.startswith("SYSTEM"))
        self.assertIn(RULES_BLOCK_HEADING, out)


# ----------------------------------------------------------------------
# Injection — agentic steps
# ----------------------------------------------------------------------


class _Ctx:
    """Minimal stand-in for SharedContext."""

    def __init__(self, workflow: str = "alima_v51"):
        self.workflow_name = workflow
        self.applied_user_rules = []
        self.prompt_service = None
        self.model = ""


class PromptResolverInjectionTest(unittest.TestCase):
    """The null case must be byte-identical; the mutated case must differ."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "rules.yaml"
        import src.core.user_rules as ur

        self._orig_default = ur.default_rules_path
        ur.default_rules_path = lambda: self.path

    def tearDown(self):
        import src.core.user_rules as ur

        ur.default_rules_path = self._orig_default
        self._tmp.cleanup()

    def _resolve(self):
        from src.core.agents.prompt_resolver import resolve_prompts

        ctx = _Ctx()
        system, user, _ = resolve_prompts(
            raw_cfg={"id": "selection", "system_prompt": "BASIS", "user_prompt": "FRAGE"},
            resolved_inputs={},
            context=ctx,
            step_id="selection",
        )
        return system, user, ctx

    def test_without_rules_the_prompt_is_unchanged(self):
        system, user, ctx = self._resolve()
        self.assertEqual(system, "BASIS")
        self.assertEqual(user, "FRAGE")
        self.assertEqual(ctx.applied_user_rules, [])

    def test_an_active_rule_changes_the_prompt(self):
        # Mutation probe for the test above: same call, one rule active.
        RuleStore(self.path).add("Kernschlagworte sind zwei bis fünf.", steps=["selection*"])
        system, user, ctx = self._resolve()
        self.assertNotEqual(system, "BASIS")
        self.assertTrue(system.startswith("BASIS"))
        self.assertIn(RULES_BLOCK_HEADING, system)
        self.assertEqual(user, "FRAGE", "the user prompt must stay untouched")
        self.assertEqual([e["text"] for e in ctx.applied_user_rules],
                         ["Kernschlagworte sind zwei bis fünf."])

    def test_a_rule_for_another_step_stays_out(self):
        RuleStore(self.path).add("Nur bei der Klassifikation.", steps=["classification"])
        system, _, ctx = self._resolve()
        self.assertEqual(system, "BASIS")
        self.assertEqual(ctx.applied_user_rules, [])

    def test_a_disabled_rule_stays_out(self):
        store = RuleStore(self.path)
        rule = store.add("Gilt gleich nicht mehr.")
        store.set_enabled(rule.id, False)
        system, _, _ = self._resolve()
        self.assertEqual(system, "BASIS")

    def test_applied_rules_are_not_duplicated_across_steps(self):
        RuleStore(self.path).add("Gilt überall.")
        from src.core.agents.prompt_resolver import resolve_prompts

        ctx = _Ctx()
        for step in ("extraction", "selection", "classification"):
            resolve_prompts(
                raw_cfg={"id": step, "system_prompt": "BASIS"},
                resolved_inputs={},
                context=ctx,
                step_id=step,
            )
        self.assertEqual(len(ctx.applied_user_rules), 1)


# ----------------------------------------------------------------------
# Injection — chat prompt
# ----------------------------------------------------------------------


class ChatPromptInjectionTest(unittest.TestCase):
    def test_without_rules_the_chat_prompt_is_unchanged(self):
        from src.core.chat_prompts import build_system_prompt

        self.assertEqual(
            build_system_prompt(mode="general", user_rules=""),
            build_system_prompt(mode="general"),
        )

    def test_with_rules_the_block_is_appended_last(self):
        from src.core.chat_prompts import build_system_prompt

        block = render_rules_block([_rule()])
        with_rules = build_system_prompt(mode="general", user_rules=block)
        without = build_system_prompt(mode="general")
        self.assertNotEqual(with_rules, without)
        self.assertTrue(with_rules.startswith(without))
        self.assertIn(RULES_BLOCK_HEADING, with_rules)

    def test_the_chat_ruleset_teaches_the_agent_to_offer_a_rule(self):
        # Without this passage the model never proposes anything, and the whole
        # "recognise and ask" behaviour is dead.
        from src.core.chat_prompts import SHARED_RULES, SHARED_RULES_COMPACT

        self.assertIn("propose_rule", SHARED_RULES)
        self.assertIn("propose_rule", SHARED_RULES_COMPACT)


# ----------------------------------------------------------------------
# Injection — MetaAgent planner
# ----------------------------------------------------------------------


class MetaAgentInjectionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "rules.yaml"
        import src.core.user_rules as ur

        self._orig_default = ur.default_rules_path
        ur.default_rules_path = lambda: self.path

    def tearDown(self):
        import src.core.user_rules as ur

        ur.default_rules_path = self._orig_default
        self._tmp.cleanup()

    def test_planner_prompt_unchanged_without_rules(self):
        from src.core.agents.meta_agent import MetaAgent

        ctx = _Ctx()
        self.assertEqual(MetaAgent._append_user_rules("PLAN", ctx, "planner"), "PLAN")

    def test_planner_prompt_gets_a_planner_scoped_rule(self):
        from src.core.agents.meta_agent import MetaAgent

        RuleStore(self.path).add("Erst suchen, dann auswählen.", steps=["planner"])
        ctx = _Ctx()
        out = MetaAgent._append_user_rules("PLAN", ctx, "planner")
        self.assertNotEqual(out, "PLAN")
        self.assertIn("Erst suchen, dann auswählen.", out)
        self.assertEqual(len(ctx.applied_user_rules), 1)


# ----------------------------------------------------------------------
# Chat tools — the confirmation must actually gate the write
# ----------------------------------------------------------------------


class _Gateway:
    """Fake ProposalGateway with a fixed answer."""

    def __init__(self, accepted: bool, reason: str = ""):
        self._answer = {"accepted": accepted, "reject_reason": reason}
        self.calls = []

    def request_decision(self, audit_id, tool_name, payload, timeout_ms=0):
        self.calls.append((tool_name, payload))
        return dict(self._answer)


class RuleToolsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.store = RuleStore(Path(self._tmp.name) / "rules.yaml")
        self.chat_config = Mock(autonomous_pipeline=False, rule_author="")

    def tearDown(self):
        self._tmp.cleanup()

    def _tool(self, cls, gateway=None):
        return cls(
            gateway=gateway,
            chat_config=self.chat_config,
            session_id="sess-1",
            kb_manager=None,
            store=self.store,
        )

    def test_rejected_proposal_writes_nothing(self):
        from src.ui.chat_tools.rules import ProposeRuleTool

        gateway = _Gateway(accepted=False, reason="user_declined")
        out = json.loads(self._tool(ProposeRuleTool, gateway).execute(None, text="Regel."))
        self.assertEqual(out["status"], "rejected")
        self.assertEqual(self.store.load(), [])

    def test_accepted_proposal_writes_exactly_one_rule_with_chat_provenance(self):
        from src.ui.chat_tools.rules import ProposeRuleTool

        gateway = _Gateway(accepted=True)
        out = json.loads(
            self._tool(ProposeRuleTool, gateway).execute(
                None,
                text="Formschlagwörter gehören nicht in core_keywords.",
                applies_when="bei Lehrbüchern",
                steps=["selection*"],
                reason="aus dem Gespräch über Lauf X",
            )
        )
        self.assertEqual(out["status"], "saved")
        (rule,) = self.store.load()
        self.assertEqual(rule.origin["source"], "chat")
        self.assertEqual(rule.origin["session_id"], "sess-1")
        self.assertEqual(rule.origin["note"], "aus dem Gespräch über Lauf X")
        self.assertEqual(rule.applies_when, "bei Lehrbüchern")
        self.assertEqual(rule.steps, ["selection*"])
        self.assertTrue(rule.enabled)
        # The stored rule comes back so the agent can repeat it verbatim.
        self.assertEqual(out["rule"]["id"], rule.id)

    def test_without_a_gateway_nothing_is_written(self):
        # `_ask_user` fails safe to rejected. If that ever flips, an unconfirmed
        # rule would silently start shaping every run.
        from src.ui.chat_tools.rules import ProposeRuleTool

        out = json.loads(self._tool(ProposeRuleTool, None).execute(None, text="Regel."))
        self.assertEqual(out["status"], "rejected")
        self.assertEqual(self.store.load(), [])

    def test_non_interactive_rejection_says_where_to_go_instead(self):
        from src.ui.chat_tools.rules import ProposeRuleTool

        gateway = _Gateway(accepted=False, reason="non_interactive")
        out = json.loads(self._tool(ProposeRuleTool, gateway).execute(None, text="Regel."))
        self.assertIn("GUI", out["message"])

    def test_the_bubble_payload_shows_wording_condition_and_scope(self):
        from src.ui.chat_tools.rules import ProposeRuleTool

        gateway = _Gateway(accepted=False)
        self._tool(ProposeRuleTool, gateway).execute(
            None, text="Regeltext.", applies_when="bei X", steps=["selection"]
        )
        (_, payload) = gateway.calls[0]
        self.assertEqual(payload["text"], "Regeltext.")
        self.assertEqual(payload["applies_when"], "bei X")
        self.assertIn("selection", payload["scope"])

    def test_an_exact_duplicate_is_not_proposed_again(self):
        from src.ui.chat_tools.rules import ProposeRuleTool

        self.store.add("Lehrbücher gehören in Feld 1131.")
        gateway = _Gateway(accepted=True)
        out = json.loads(
            self._tool(ProposeRuleTool, gateway).execute(
                None, text="  lehrbücher   gehören in feld 1131.  "
            )
        )
        self.assertEqual(out["status"], "exists")
        self.assertEqual(gateway.calls, [], "must not ask about a rule that exists")
        self.assertEqual(len(self.store.load()), 1)

    def test_delete_requires_confirmation(self):
        from src.ui.chat_tools.rules import DeleteRuleTool

        rule = self.store.add("Bleibt.")
        out = json.loads(
            self._tool(DeleteRuleTool, _Gateway(accepted=False)).execute(None, rule_id=rule.id)
        )
        self.assertEqual(out["status"], "rejected")
        self.assertEqual(len(self.store.load()), 1)

        out = json.loads(
            self._tool(DeleteRuleTool, _Gateway(accepted=True)).execute(None, rule_id=rule.id)
        )
        self.assertEqual(out["status"], "deleted")
        self.assertEqual(self.store.load(), [])

    def test_toggle_needs_no_confirmation(self):
        from src.ui.chat_tools.rules import SetRuleEnabledTool

        rule = self.store.add("Regel.")
        out = json.loads(
            self._tool(SetRuleEnabledTool, None).execute(None, rule_id=rule.id, enabled=False)
        )
        self.assertEqual(out["status"], "ok")
        self.assertFalse(self.store.load()[0].enabled)

    def test_list_rules_reports_the_file_it_read(self):
        from src.ui.chat_tools.rules import ListRulesTool

        self.store.add("Regel.")
        out = json.loads(self._tool(ListRulesTool, None).execute(None))
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["file"], str(self.store.path))

    def test_autonomous_mode_does_not_skip_the_question(self):
        """A rule is asked about even in autonomous mode.

        The base class treats ``autonomous_pipeline`` as "skip the y/N", which
        fits a keyword replacement: it changes the run the user just started.
        A rule changes every future run instead. With the shortcut in place and
        autonomous mode on, six rules landed on the operator's machine in
        fifteen minutes without one question — a duplicate and a feature wish
        among them.
        """
        from src.ui.chat_tools.rules import ProposeRuleTool

        self.chat_config.autonomous_pipeline = True

        # No gateway → still refused, not silently stored.
        out = json.loads(self._tool(ProposeRuleTool, None).execute(None, text="Regel."))
        self.assertEqual(out["status"], "rejected")
        self.assertEqual(self.store.load(), [])

        # Gateway present → the question is actually put.
        gateway = _Gateway(accepted=False)
        self._tool(ProposeRuleTool, gateway).execute(None, text="Regel.")
        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual(self.store.load(), [])

    def test_deleting_also_asks_in_autonomous_mode(self):
        from src.ui.chat_tools.rules import DeleteRuleTool

        self.chat_config.autonomous_pipeline = True
        rule = self.store.add("Bleibt.")
        gateway = _Gateway(accepted=False)
        out = json.loads(self._tool(DeleteRuleTool, gateway).execute(None, rule_id=rule.id))
        self.assertEqual(out["status"], "rejected")
        self.assertEqual(len(self.store.load()), 1)


# ----------------------------------------------------------------------
# Export / import
# ----------------------------------------------------------------------


class ExchangeTest(_LiftLogDisable, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.store = RuleStore(self.dir / "rules.yaml")

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()

    def test_export_keeps_provenance_verbatim(self):
        self.store.add(
            "Regel.",
            origin={"source": "chat", "author": "Kollegin", "note": "warum"},
        )
        target = self.dir / "share.yaml"
        ok, count = self.store.export(target)
        self.assertTrue(ok)
        self.assertEqual(count, 1)

        (exported,) = RuleStore(target).load()
        self.assertEqual(exported.origin["source"], "chat")
        self.assertEqual(exported.origin["author"], "Kollegin")
        self.assertEqual(exported.origin["note"], "warum")

    def test_export_can_be_narrowed(self):
        keep = self.store.add("Behalten.")
        other = self.store.add("Auch da.")
        self.store.set_enabled(other.id, False)

        ok, count = self.store.export(self.dir / "a.yaml", ids=[keep.id])
        self.assertTrue(ok)
        self.assertEqual(count, 1)

        ok, count = self.store.export(self.dir / "b.yaml", enabled_only=True)
        self.assertTrue(ok)
        self.assertEqual(count, 1)

    def test_import_keeps_origin_and_records_where_it_came_from(self):
        self.store.add("Regel.", origin={"source": "chat", "author": "Kollegin"})
        share = self.dir / "share.yaml"
        self.store.export(share)

        target = RuleStore(self.dir / "other.yaml")
        ok, added = target.import_file(share)
        self.assertTrue(ok)
        (imported,) = added
        self.assertEqual(imported.origin["source"], "chat")
        self.assertEqual(imported.origin["author"], "Kollegin")
        self.assertEqual(imported.origin["imported_from"], "share.yaml")
        self.assertIn("imported_at", imported.origin)

    def test_imported_rules_land_inactive_unless_asked(self):
        self.store.add("Regel.")
        share = self.dir / "share.yaml"
        self.store.export(share)

        target = RuleStore(self.dir / "off.yaml")
        _, added = target.import_file(share)
        self.assertFalse(added[0].enabled)

        target2 = RuleStore(self.dir / "on.yaml")
        _, added2 = target2.import_file(share, activate=True)
        self.assertTrue(added2[0].enabled)

    def test_id_collision_gets_a_new_id_and_keeps_the_old_one(self):
        original = self.store.add("Regel.")
        share = self.dir / "share.yaml"
        self.store.export(share)

        ok, added = self.store.import_file(share)
        self.assertTrue(ok)
        (imported,) = added
        self.assertNotEqual(imported.id, original.id)
        self.assertEqual(imported.origin["original_id"], original.id)
        self.assertEqual(len(self.store.load()), 2)

    def test_importing_a_broken_file_reports_failure(self):
        broken = self.dir / "broken.yaml"
        broken.write_text("rules: [unclosed\n", encoding="utf-8")
        with self.assertLogs("src.core.user_rules", level="WARNING"):
            ok, added = self.store.import_file(broken)
        self.assertFalse(ok)
        self.assertEqual(added, [])


# ----------------------------------------------------------------------
# Result provenance
# ----------------------------------------------------------------------


class AppliedRulesInResultTest(unittest.TestCase):
    def test_shared_context_carries_the_rules_into_the_analysis_state(self):
        from src.core.agents.shared_context import SharedContext

        ctx = SharedContext(abstract="Text")
        ctx.applied_user_rules = [{"id": "r-1", "text": "Regel."}]
        state = ctx.to_keyword_analysis_state()
        self.assertEqual(state.applied_rules, [{"id": "r-1", "text": "Regel."}])

    def test_a_run_without_rules_records_an_empty_list(self):
        from src.core.agents.shared_context import SharedContext

        state = SharedContext(abstract="Text").to_keyword_analysis_state()
        self.assertEqual(state.applied_rules, [])


if __name__ == "__main__":
    unittest.main()


class StoreIsolationTest(unittest.TestCase):
    """The suite must never read the operator's own rules.

    Real rules are appended to every agentic prompt, so machine-local state
    would change what unrelated tests assert on — which is exactly how this
    surfaced: six real rules broke ``test_e2e_smoke``.
    """

    def test_env_override_wins_over_the_config_dir(self):
        import os
        from src.core.user_rules import RULES_PATH_ENV, default_rules_path

        self.assertIn(RULES_PATH_ENV, os.environ, "test bootstrap must set the override")
        self.assertEqual(str(default_rules_path()), os.environ[RULES_PATH_ENV])

    def test_the_default_store_is_empty_during_the_suite(self):
        self.assertEqual(RuleStore().load(), [])


class ReflectionRulesGateTest(unittest.TestCase):
    """The reflection turn is the last LLM turn — so it is where a rule that
    asks for something *at the end* of the run can still be carried out.

    The agentic workflows end in deterministic steps (`rvk_guard`,
    `dk_postprocess`), so without this gate such a rule reaches every earlier
    prompt and can act in none of them.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "rules.yaml"
        import src.core.user_rules as ur

        self._orig_default = ur.default_rules_path
        ur.default_rules_path = lambda: self.path

    def tearDown(self):
        import src.core.user_rules as ur

        ur.default_rules_path = self._orig_default
        self._tmp.cleanup()

    def test_without_rules_the_gate_is_absent(self):
        from src.core.agents.steps.reflection_step import _user_rules_values

        self.assertEqual(
            _user_rules_values(_Ctx()), {"user_rules_gate": "", "user_rules": ""}
        )

    def test_with_a_rule_the_gate_names_it_and_the_final_output_field(self):
        from src.core.agents.steps.reflection_step import _user_rules_values

        RuleStore(self.path).add("Am Ende den Katalogeintrag erzeugen.")
        values = _user_rules_values(_Ctx())
        self.assertIn("Am Ende den Katalogeintrag erzeugen.", values["user_rules_gate"])
        self.assertIn("final_output", values["user_rules_gate"])

    def test_every_step_type_still_resolves_to_a_step_class(self):
        """@register_step must sit on the class, not on a helper below it.

        Inserting a module-level function directly above ``class ReflectionStep``
        silently moved the decorator onto that function, so
        ``get_step_class("reflection")`` returned it and every agentic run died
        with "unexpected keyword argument 'config'". The whole suite stayed
        green — nothing checked what the registry actually holds.
        """
        from src.core.agents.registry import STEP_REGISTRY, get_step_class
        from src.core.agents.steps.base_step import BaseStep

        self.assertIn("reflection", STEP_REGISTRY)
        for name in STEP_REGISTRY:
            cls = get_step_class(name)
            self.assertTrue(
                isinstance(cls, type) and issubclass(cls, BaseStep),
                f"step type '{name}' resolves to {cls!r}, not a BaseStep subclass",
            )

    def test_the_json_contract_offers_final_output(self):
        from src.core.agents.steps.reflection_step import (
            DEFAULT_REFLECTION_SYSTEM_PROMPT,
        )

        self.assertIn("final_output", DEFAULT_REFLECTION_SYSTEM_PROMPT)
        self.assertIn("{user_rules_gate}", DEFAULT_REFLECTION_SYSTEM_PROMPT)

    def test_produced_output_is_kept_on_the_context(self):
        from src.core.agents.meta_agent import MetaAgent

        ctx = _Ctx()
        ctx.extra = {}
        MetaAgent._capture_rule_output(ctx, {"final_output": "5550 Cadmium"})
        self.assertEqual(ctx.extra["rule_output"], "5550 Cadmium")

    def test_an_empty_final_output_leaves_the_context_alone(self):
        from src.core.agents.meta_agent import MetaAgent

        ctx = _Ctx()
        ctx.extra = {}
        MetaAgent._capture_rule_output(ctx, {"final_output": "   "})
        MetaAgent._capture_rule_output(ctx, {})
        self.assertEqual(ctx.extra, {})

    def test_the_output_reaches_the_saved_result(self):
        from src.core.agents.shared_context import SharedContext

        ctx = SharedContext(abstract="Text")
        ctx.extra = {"rule_output": "5550 Cadmium\n5550 $ADE-105"}
        state = ctx.to_keyword_analysis_state()
        self.assertEqual(state.rule_output, "5550 Cadmium\n5550 $ADE-105")

    def test_a_run_without_a_rule_output_stays_empty(self):
        from src.core.agents.shared_context import SharedContext

        self.assertEqual(SharedContext(abstract="Text").to_keyword_analysis_state().rule_output, "")
