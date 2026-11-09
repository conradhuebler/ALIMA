"""geo_prior: lokale Zuordnungen als Eingabe der Klassifikation (alima_v51_105) - Claude Generated.

Der Schritt ``geo_prior`` schlägt die finalen Schlagwörter vor ``classification``
in der lokalen Sachdatenbank nach (Plugin ``geo_sachdb``) und gibt jede
gefundene Notation mit Bezeichnung, Schlagwörtern und der Zahl der
Kernschlagwörter als Block ``extra.geo_prompt_text`` an die Klassifikation.
Geprüft werden: Block (Reihenfolge, alle Kandidaten, ohne PPN, ohne Kern, ohne
Plugin, ohne Treffer), Quellenübersicht, Verdrahtung, die Bedingung der
Klassifikation, der Prompt (mit lokalen Daten, mit beidem, ohne lokale Daten
unverändert), bedingte Abschnitte im Renderer, Reflexions- und Planerzustand,
ein Lauf über den MetaAgent vom DK-Schritt bis zum WinIBW-Eintrag mit leerem
Katalog, Herkunft in DK-Karte und CLI, Speichern und Laden. Kein Netz, kein
echtes Modell; Datenbank und Daten erfunden.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import os
import re
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.agents.conditional_engine import ConditionalEngine
from src.core.agents.deterministic_functions import geo_sachdb_prior
from src.core.agents.registry import TOOL_FN_REGISTRY
from src.core.agents.shared_context import GEO_PRIOR_EXTRA_KEY, SharedContext
from src.core.data_models import AgentResponse
from src.core.search import source_health, source_usage
from src.utils.config_models import AlimaConfig
from src.utils.lookups.geo_sachdb.provider import GeoSachdbLookup

_spec = importlib.util.spec_from_file_location(
    "geo_lookup_fixture_prior", Path(__file__).resolve().parent / "test_geo_sachdb_lookup.py"
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
_HOME = {}

#: Zusätzlich zur Datenbank der Plugin-Tests: eine Notation, die über zwei
#: Kernschlagwörter erreicht wird (DK 999.5), ein allgemeines Schlagwort
#: „Geschichte“, das zwei Notationen anzieht, und „Testmineral“.
EXTRA_CLASSIFICATIONS = [
    (20, "DK", "999.5", "Testmineralogie", "121212121"),
    (21, "DK", "999.9", "Testgeschichte", None),
]
EXTRA_HEADINGS = [
    (20, "Testmineral", "Sachschlagwort", "131313131", "|s|Testmineral !131313131!", None),
    (21, "Geschichte", "Sachschlagwort", "141414141", "|s|Geschichte !141414141!", None),
]
EXTRA_LINKS = [(1, 20, 1, 0), (20, 20, 0, 0), (21, 2, 0, 0), (21, 21, 1, 0)]

KEYWORDS = [{"keyword": "Testgestein", "gnd_id": "1"}, {"keyword": "Testmineral", "gnd_id": "2"},
            {"keyword": "Lagerstätte", "gnd_id": "3"}, {"keyword": "Musterland", "gnd_id": "4"},
            {"keyword": "Geschichte", "gnd_id": "5"}, {"keyword": "Unbekanntwort"}]
CORE = [{"keyword": "Testgestein", "gnd_id": "1"}, {"keyword": "Testmineral", "gnd_id": "2"}]
TITLE = "Lokale Zuordnungen aus der Testsachdatenbank (kuratiert)"

#: Der Block für KEYWORDS/CORE: Kern zuerst, dann Gesamtzahl, dann Notation.
EXPECTED_BLOCK = [
    f"{TITLE}:",
    "Kuratierte Zuordnung von Schlagwörtern zu Notationen in der lokalen Datenbank. "
    "Vorschläge, keine Übernahmeliste: allgemeine Schlagwörter (z. B. „Geschichte“, "
    "„Thermodynamik“) ziehen Notationen an, die zum Werk nicht passen.",
    "- DK 999.5 | Testmineralogie | über Testgestein, Testmineral | Kernschlagwörter 2 von 2",
    "- DK 999.1 | Testgesteine | über Testgestein | Kernschlagwörter 1 von 1",
    "- DK 999.21 | Lagerstaetten (Test) | über Lagerstätte, Geschichte | Kernschlagwörter 0 von 2",
    "- DK 999.3 | Musterland | über Musterland | Kernschlagwörter 0 von 1",
    "- DK 999.31 | Doppelte ID | über Musterland | Kernschlagwörter 0 von 1",
    "- DK 999.9 | Testgeschichte | über Geschichte | Kernschlagwörter 0 von 1",
]
ORDER = ["999.5", "999.1", "999.21", "999.3", "999.31", "999.9"]


def setUpModule():
    _HOME["old"] = os.environ.get("HOME")
    _HOME["tmp"] = tempfile.mkdtemp()
    os.environ["HOME"] = _HOME["tmp"]


def tearDownModule():
    if _HOME.get("old") is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = _HOME["old"]
    shutil.rmtree(_HOME["tmp"], ignore_errors=True)


def build_db(path: Path) -> None:
    """Die Datenbank der Plugin-Tests plus die Zeilen oben."""
    _fixture.build_synthetic_db(path)
    conn = sqlite3.connect(path)
    conn.executemany("INSERT INTO classification VALUES (?,?,?,?,?)", EXTRA_CLASSIFICATIONS)
    conn.executemany("INSERT INTO heading VALUES (?,?,?,?,?,?)", EXTRA_HEADINGS)
    conn.executemany("INSERT INTO heading_classification VALUES (?,?,?,?)", EXTRA_LINKS)
    conn.commit()
    conn.close()


def _workflow(name="alima_v51_105"):
    from src.core.agents.workflow_loader import find_workflow_file, load_workflow

    return load_workflow(find_workflow_file(name), strict=True)


def _step(step_id):
    return {s.id: s for s in _workflow().steps}[step_id]


class _DbCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "geo.db"
        build_db(self.db)
        self.lookup = GeoSachdbLookup(**_fixture.settings(self.db))

    def tearDown(self):
        self.tmp.cleanup()

    def _build_lookup(self, config, lookup_id):
        """build_lookup stand-in: only geo_sachdb is configured."""
        return self.lookup if lookup_id == "geo_sachdb" else None

    def prior(self, keywords=KEYWORDS, core=CORE, context=None, config=None):
        with patch("src.utils.lookups.build_lookup", side_effect=self._build_lookup):
            return geo_sachdb_prior(keywords=keywords, core_keywords=core, context=context,
                                    config={"title": TITLE, **(config or {})})


# ── der Block ────────────────────────────────────────────────────────────────

class PriorBlock(_DbCase):
    def test_core_keywords_first_then_all_keywords_then_notation(self):
        out = self.prior()
        self.assertEqual(out["status"], "ok")
        self.assertEqual([c["code"] for c in out["candidates"]], ORDER)
        by_code = {c["code"]: c for c in out["candidates"]}
        self.assertEqual(by_code["999.5"]["via_core_keywords"], ["Testgestein", "Testmineral"])
        self.assertEqual(by_code["999.21"]["via_core_keywords"], [])
        self.assertEqual(by_code["999.21"]["via_keywords"], ["Lagerstätte", "Geschichte"])

    def test_the_block_lists_every_candidate(self):
        self.assertEqual(self.prior()["prompt_text"].splitlines(), EXPECTED_BLOCK)

    def test_no_ppn_and_no_winibw_line_in_the_block(self):
        text = self.prior()["prompt_text"]
        self.assertNotIn("!", text)
        self.assertNotIn("6700", text)
        self.assertNotIn("5550", text)

    def test_there_is_no_limit_on_the_candidates(self):
        # 40 Notationen an einem Schlagwort: alle stehen im Block.
        conn = sqlite3.connect(self.db)
        for i in range(40):
            conn.execute("INSERT INTO classification VALUES (?,?,?,?,?)",
                         (100 + i, "DK", f"777.{i:02d}", f"Massenklasse {i}", None))
            conn.execute("INSERT INTO heading_classification VALUES (?,?,?,?)", (20, 100 + i, 2 + i, 0))
        conn.commit()
        conn.close()
        self.lookup = GeoSachdbLookup(**_fixture.settings(self.db))
        out = self.prior()
        self.assertEqual(len(out["candidates"]), 46)
        lines = [line for line in out["prompt_text"].splitlines() if line.startswith("- DK ")]
        self.assertEqual(len(lines), 46)
        self.assertIn("- DK 777.39 | Massenklasse 39 | über Testmineral | Kernschlagwörter 1 von 1", lines)

    def test_without_core_keywords_the_order_is_by_keyword_count(self):
        out = self.prior(core=[])
        self.assertEqual([c["code"] for c in out["candidates"]],
                         ["999.21", "999.5", "999.1", "999.3", "999.31", "999.9"])
        lines = out["prompt_text"].splitlines()
        self.assertEqual(lines[2], "Kein Kern bestimmt: Reihenfolge nach Zahl der Schlagwörter.")
        self.assertEqual(lines[3], "- DK 999.21 | Lagerstaetten (Test) | über Lagerstätte, Geschichte | Schlagwörter 2")
        self.assertNotIn("Kernschlagwörter", out["prompt_text"])

    def test_inputs_fall_back_to_the_context(self):
        ctx = SharedContext(abstract="x")
        ctx.extra["final_keywords"] = list(KEYWORDS)
        ctx.extra["core_keywords"] = list(CORE)
        out = self.prior(keywords=None, core=None, context=ctx)
        self.assertEqual([c["code"] for c in out["candidates"]], ORDER)

    def test_assigned_notations_on_the_context_are_not_looked_at(self):
        ctx = SharedContext(abstract="x")
        ctx.dk_classifications = [{"code": "DK 999.5", "rank": "core"}]
        out = self.prior(context=ctx)
        self.assertEqual(out["notation_matches"], [])
        self.assertFalse(any(c["assigned"] for c in out["candidates"]))

    def test_without_plugin_the_block_is_empty(self):
        with patch("src.utils.lookups.resolve._load_config_best_effort", return_value=AlimaConfig()):
            with source_health.run_scope():
                out = geo_sachdb_prior(keywords=KEYWORDS, core_keywords=CORE, config={})
                rows = source_usage.rows()
        self.assertEqual(out["status"], "disabled")
        self.assertEqual((out["prompt_text"], out["candidates"]), ("", []))
        self.assertEqual(rows, [])

    def test_without_a_local_notation_the_block_is_empty(self):
        # „Gestein“ hat einen lokalen Satz ohne Notation, „Unbekanntwort“ keinen.
        out = self.prior(keywords=[{"keyword": "Gestein"}, {"keyword": "Unbekanntwort"}])
        self.assertEqual(out["status"], "ok")
        self.assertEqual((out["prompt_text"], out["candidates"]), ("", []))
        self.assertEqual(self.prior(keywords=[], core=[])["prompt_text"], "")


class PriorIsCounted(_DbCase):
    def test_the_asked_keywords_are_counted_under_the_lookup(self):
        with source_health.run_scope():
            self.prior()
            rows = {row["source"]: row for row in source_usage.rows()}
        row = rows["geo_sachdb"]
        self.assertEqual((row["requests"], row["hits"], row["errors"], row["network"], row["cache"]),
                         (6, 5, 0, 0, 0))
        self.assertIn("6 Anfragen", row["text"])
        self.assertIn("5 Treffer-Schlagwörter", row["text"])

    def test_a_failing_lookup_counts_as_errors(self):
        broken = Mock(spec=["lookup_terms", "match_notation", "is_available"])
        broken.is_available.return_value = True
        broken.lookup_terms.return_value = {"error": "kaputt"}
        with patch("src.utils.lookups.build_lookup", return_value=broken):
            with source_health.run_scope():
                out = geo_sachdb_prior(keywords=KEYWORDS[:2], core_keywords=CORE, config={})
                rows = {row["source"]: row for row in source_usage.rows()}
        self.assertEqual((out["status"], out["prompt_text"]), ("error", ""))
        self.assertEqual((rows["geo_sachdb"]["requests"], rows["geo_sachdb"]["errors"]), (2, 2))


# ── Verdrahtung ──────────────────────────────────────────────────────────────

class Wiring(unittest.TestCase):
    def test_the_step_sits_between_dk_collect_and_classification(self):
        steps = _workflow().steps
        ids = [s.id for s in steps]
        self.assertEqual(ids.index("geo_prior"), ids.index("dk_collect") + 1)
        self.assertEqual(ids.index("classification"), ids.index("geo_prior") + 1)
        by_id = {s.id: s for s in steps}
        prior = by_id["geo_prior"]
        self.assertEqual(prior.depends_on, ["dk_collect"])
        self.assertIn("geo_prior", by_id["classification"].depends_on)
        self.assertEqual(prior.raw["function"], "geo_sachdb_prior")
        self.assertIn("geo_sachdb_prior", TOOL_FN_REGISTRY)
        self.assertEqual(prior.outputs, {"extra.geo_sachdb_prior": "result",
                                         "extra.geo_prompt_text": "result.prompt_text"})
        self.assertEqual(prior.inputs, {"keywords": "${extra.final_keywords}",
                                        "core_keywords": "${extra.core_keywords}"})
        cfg = prior.raw["config"]
        self.assertEqual((cfg["lookup"], cfg["scheme"], cfg["min_score"], cfg["kinds"]),
                         ("geo_sachdb", "DK", 90, []))
        self.assertEqual(GEO_PRIOR_EXTRA_KEY, "geo_sachdb_prior")

    def test_classification_takes_the_block(self):
        cls = _step("classification")
        self.assertEqual(cls.inputs["local_notations"], "${extra.geo_prompt_text}")
        self.assertEqual(cls.inputs["keywords"], "${extra.dk_prompt_text}")

    def test_the_general_workflow_is_untouched(self):
        steps = {s.id: s for s in _workflow("alima_v51").steps}
        self.assertNotIn("geo_prior", steps)
        self.assertNotIn("local_notations", steps["classification"].inputs)

    def test_planner_and_reflection_name_the_step(self):
        meta = _workflow().meta_agent
        chain = " ".join(meta["planning"]["rules"].split())
        self.assertIn("dk_collect → geo_prior → classification", chain)
        reflection = " ".join(meta["reflection"]["rules"].split())
        self.assertIn("geo_prior→lokale Zuordnungen", reflection)
        self.assertIn("KEIN Grund für 'rerun_dk_collect'", reflection)
        enabled = [s for s in _workflow().steps if s.enabled]
        self.assertEqual(len(enabled), 15)  # + publication_form
        self.assertEqual(meta["max_cycles"], 23)


# ── Bedingung der Klassifikation ─────────────────────────────────────────────

class ClassificationGate(unittest.TestCase):
    OLD_CONDITION = "${extra.dk_prompt_text} != ''"

    def _runs(self, **extra):
        ctx = SharedContext(abstract="x")
        ctx.extra.update(extra)
        return ConditionalEngine.evaluate(_step("classification").condition, ctx)

    def test_empty_catalogue_with_local_notations_runs(self):
        self.assertTrue(self._runs(dk_prompt_text="", geo_prompt_text="Lokale Zuordnungen …"))
        ctx = SharedContext(abstract="x")
        ctx.extra.update(dk_prompt_text="", geo_prompt_text="Lokale Zuordnungen …")
        self.assertFalse(ConditionalEngine.evaluate(self.OLD_CONDITION, ctx))  # vorher: übersprungen

    def test_catalogue_alone_runs_as_before(self):
        self.assertTrue(self._runs(dk_prompt_text="DK: 999.1", geo_prompt_text=""))
        self.assertTrue(self._runs(dk_prompt_text="DK: 999.1"))

    def test_nothing_at_all_skips(self):
        self.assertFalse(self._runs(dk_prompt_text="", geo_prompt_text=""))

    def test_a_missing_local_block_counts_as_empty(self):
        # geo_prior nicht gelaufen (abgeschaltet): nicht auf „nichts“ klassifizieren.
        self.assertFalse(self._runs(dk_prompt_text=""))


# ── bedingte Abschnitte im Renderer ──────────────────────────────────────────

class ConditionalSections(unittest.TestCase):
    def _render(self, template, values):
        from src.core.agents.prompt_resolver import _render

        return _render(template, values)

    def test_a_section_stays_with_a_value_and_goes_without(self):
        tpl = "A{?x}\n\n[{x}]{/x}B"
        self.assertEqual(self._render(tpl, {"x": "wert"}), "A\n\n[wert]B")
        for empty in ("", "  \n", None, [], {}):
            self.assertEqual(self._render(tpl, {"x": empty}), "AB", repr(empty))
        self.assertEqual(self._render(tpl, {}), "AB")

    def test_values_are_not_scanned_for_sections(self):
        self.assertEqual(self._render("{x}", {"x": "{?y}geheim{/y}"}), "{?y}geheim{/y}")

    def test_a_template_without_sections_is_unchanged(self):
        tpl = '{\n  "classifications": [{"code": "DK 616.89"}]\n}\n{abstract} {unbekannt}'
        self.assertEqual(self._render(tpl, {"abstract": "x"}),
                         '{\n  "classifications": [{"code": "DK 616.89"}]\n}\nx {unbekannt}')


# ── der Prompt der Klassifikation ────────────────────────────────────────────

CATALOGUE = "DK: 999.1 (Häufigkeit: 3) | Beispieltitel: Ein Testbuch"
LOCAL = "\n".join(EXPECTED_BLOCK)
PARAGRAPH = "**Lokale Zuordnungen aus der Freiberger Sachdatenbank (UB Freiberg 105):**"


def _prompts(catalogue, local, *, set_local=True,
             abstract="Erfundener Abstract über Testgestein {in Klammern}."):
    """System and user prompt of the classification step, rendered like a run."""
    from src.core.agents.context_path import resolve_value
    from src.core.agents.steps.llm_agent_step import LLMAgentStep

    cfg = _step("classification")
    ctx = SharedContext(abstract=abstract)
    ctx.extra.update(dk_prompt_text=catalogue, final_keywords=list(KEYWORDS), core_keywords=list(CORE))
    if set_local:
        ctx.extra["geo_prompt_text"] = local
    inputs = {key: resolve_value(value, ctx) for key, value in cfg.inputs.items()}
    system, user, _ = LLMAgentStep(cfg)._resolve_prompts(cfg.raw, inputs, ctx, {})
    return system, user


class ClassificationPrompt(unittest.TestCase):
    def test_local_only_brings_block_and_paragraph(self):
        system, user = _prompts("", LOCAL)
        self.assertIn(PARAGRAPH, system)
        self.assertIn("zusätzliche Evidenz neben dem Bibliotheksbestand", system)
        self.assertTrue(user.endswith("\n\n" + LOCAL + "\n"), user[-300:])
        for text in (system, user):
            self.assertNotIn("{?", text)
            self.assertNotIn("{/", text)

    def test_catalogue_and_local_bring_both(self):
        system, user = _prompts(CATALOGUE, LOCAL)
        self.assertIn(CATALOGUE, user)
        self.assertIn(LOCAL, user)
        self.assertLess(user.index(CATALOGUE), user.index(LOCAL))
        self.assertIn(PARAGRAPH, system)

    def test_without_local_data_the_prompt_is_the_one_from_before(self):
        """Characterization: a run without local notations gets the prompt of
        the workflow before geo_prior. The step's own markup is the only
        difference, and it leaves no trace: no paragraph, no empty block, no
        extra line at either junction."""
        empty = _prompts(CATALOGUE, "")
        missing = _prompts(CATALOGUE, "", set_local=False)
        self.assertEqual(empty, missing)
        system, user = empty
        for text in (system, user):
            self.assertNotIn("Lokale Zuordnungen", text)
            self.assertNotIn("{?", text)
            self.assertNotIn("{/", text)
        self.assertIn("(kein Tool-Call).\n\n**Regelwerk klassifikatorische Sacherschließung (DK):**", system)
        final = json.dumps(KEYWORDS, ensure_ascii=False)
        self.assertTrue(user.endswith(f"übergeben):**\n{final}\n"), user[-200:])
        # Ohne die beiden Abschnitte ist die Vorlage die alte: genau zwei
        # Abschnitte, beide an den Nahtstellen oben.
        raw = _step("classification").raw
        sections = re.findall(r"\{\?local_notations\}.*?\{/local_notations\}",
                              raw["system_prompt"] + raw["user_prompt"], re.S)
        self.assertEqual(len(sections), 2)
        self.assertIn("(kein Tool-Call).\n\n{?local_notations}", raw["system_prompt"])
        self.assertIn("{/local_notations}**Regelwerk klassifikatorische", raw["system_prompt"])
        self.assertTrue(raw["user_prompt"].endswith(
            "{final_keywords}{?local_notations}\n\n{local_notations}{/local_notations}\n"))


# ── Lauf durch den Workflow ──────────────────────────────────────────────────

PICKED = [{"code": "DK 999.5", "type": "DK", "rank": "core"},
          {"code": "DK 999.3", "type": "DK", "rank": "additional"}]


class _Llm:
    """Stub-LLM: antwortet je Rolle und merkt sich die Prompts."""

    def __init__(self):
        self.calls = []
        self.service = Mock()
        self.service.generate_with_tools.side_effect = self._answer

    def _answer(self, provider="", model="", messages=None, **_kw):
        system = next((m.get("content", "") for m in messages or [] if m.get("role") == "system"), "")
        user = next((m.get("content", "") for m in messages or [] if m.get("role") == "user"), "")
        if "MetaAgent Planer" in system:
            role, content = "planner", "{}"  # → regelbasierter Planer
        elif "Quality Agent" in system:
            role, content = "reflection", json.dumps({"status": "continue", "gaps": [],
                                                      "action": "continue", "reason": "ok"})
        elif "Klassifikations-Experte" in system:
            role, content = "classification", json.dumps({"classifications": PICKED, "analyse": "x"})
        else:
            role, content = "other", "{}"
        self.calls.append((role, system, user))
        return AgentResponse(content=content)

    def users(self, role):
        return [user for r, _s, user in self.calls if r == role]


def _context_after_dk_collect(catalogue_text="", dk_entries=()):
    """The context as dk_collect leaves it; earlier steps marked as run."""
    ctx = SharedContext(abstract="Erfundener Abstract über Testgestein und Testmineral.")
    ctx.provider, ctx.model = "stub", "stub-model"
    ctx.working_title = "Testtitel"
    ctx.extra.update(final_keywords=list(KEYWORDS), core_keywords=list(CORE),
                     dk_prompt_text=catalogue_text, dk_entries=list(dk_entries))
    ctx.execution_history = [
        {"cycle": i, "step": step, "success": True, "duration": 0.0, "error": None}
        for i, step in enumerate(["extraction", "search", "selection_chunks", "selection",
                                  "verify_keywords", "dk_collect"], 1)
    ]
    return ctx


class ThroughTheWorkflow(_DbCase):
    def _run_step(self, ctx, step_id, llm):
        from src.core.agents.workflow_executor import WorkflowExecutor

        registry = Mock()
        registry.get_tool_schemas.return_value = []
        executor = WorkflowExecutor(llm_service=llm.service, tool_registry=registry)
        with patch("src.utils.lookups.build_lookup", side_effect=self._build_lookup):
            return executor.run(_workflow(), ctx, only_step=step_id)

    def test_empty_catalogue_classification_runs_with_the_local_block(self):
        llm, ctx = _Llm(), _context_after_dk_collect()
        self.assertTrue(self._run_step(ctx, "geo_prior", llm).success)
        report = self._run_step(ctx, "classification", llm)
        self.assertTrue(report.success, report.error)
        self.assertNotIn("skipped", report.step_results[0].data or {})
        users = llm.users("classification")
        self.assertEqual(len(users), 1)
        self.assertIn("Lokale Zuordnungen aus der Freiberger Sachdatenbank (kuratiert):", users[0])
        self.assertIn("- DK 999.5 | Testmineralogie | über Testgestein, Testmineral | "
                      "Kernschlagwörter 2 von 2", users[0])
        self.assertEqual([c["code"] for c in ctx.dk_classifications], ["DK 999.5", "DK 999.3"])

    def test_catalogue_and_local_data_reach_the_prompt_together(self):
        llm, ctx = _Llm(), _context_after_dk_collect(CATALOGUE)
        self._run_step(ctx, "geo_prior", llm)
        self._run_step(ctx, "classification", llm)
        user = llm.users("classification")[0]
        self.assertIn(CATALOGUE, user)
        self.assertIn("- DK 999.1 | Testgesteine | über Testgestein | Kernschlagwörter 1 von 1", user)

    def test_without_plugin_and_without_catalogue_nothing_runs(self):
        llm, ctx = _Llm(), _context_after_dk_collect()
        from src.core.agents.workflow_executor import WorkflowExecutor

        executor = WorkflowExecutor(llm_service=llm.service, tool_registry=Mock())
        with patch("src.utils.lookups.resolve._load_config_best_effort", return_value=AlimaConfig()):
            self.assertTrue(executor.run(_workflow(), ctx, only_step="geo_prior").success)
            report = executor.run(_workflow(), ctx, only_step="classification")
        self.assertEqual(ctx.extra["geo_prompt_text"], "")
        self.assertTrue(report.step_results[0].data.get("skipped"))
        self.assertEqual(llm.users("classification"), [])

    def test_with_catalogue_and_no_plugin_the_prompt_is_unchanged(self):
        llm, ctx = _Llm(), _context_after_dk_collect(CATALOGUE)
        with patch("src.utils.lookups.resolve._load_config_best_effort", return_value=AlimaConfig()):
            from src.core.agents.workflow_executor import WorkflowExecutor

            registry = Mock()
            registry.get_tool_schemas.return_value = []
            executor = WorkflowExecutor(llm_service=llm.service, tool_registry=registry)
            executor.run(_workflow(), ctx, only_step="geo_prior")
            executor.run(_workflow(), ctx, only_step="classification")
        system, user = next((s, u) for r, s, u in llm.calls if r == "classification")
        expected = _prompts(CATALOGUE, "", set_local=False, abstract=ctx.abstract)
        self.assertEqual((system, user), expected)
        self.assertNotIn("Lokale Zuordnungen", system + user)


class EndToEnd(_DbCase):
    """MetaAgent from the step after dk_collect to the end, catalogue empty:
    geo_prior → classification (picks two local notations) → rvk_guard →
    dk_postprocess → geo_candidates → shelf_* → WinIBW entry."""

    def _run(self):
        from src.core.agents.meta_agent import MetaAgent

        llm, ctx = _Llm(), _context_after_dk_collect()
        registry = Mock()
        registry.get_tool_schemas.return_value = []
        workflow = _workflow()
        agent = MetaAgent(llm_service=llm.service, tool_registry=registry)
        with patch("src.utils.lookups.build_lookup", side_effect=self._build_lookup):
            report = agent.run(workflow, ctx, meta_config=workflow.meta_agent)
        return llm, ctx, report

    def test_the_run_classifies_from_local_notations_and_enters_their_ppn(self):
        from src.utils.pipeline_formatters import geo_sachdb_proposals
        from src.utils.winibw_record import build_winibw_record

        llm, ctx, report = self._run()
        self.assertTrue(report.success, report.error)
        tail = [h["step"] for h in ctx.execution_history][6:]
        self.assertEqual(tail, ["geo_prior", "classification", "rvk_guard", "dk_postprocess",
                                "publication_form", "geo_candidates", "shelf_candidates",
                                "shelf_select", "shelf_guard"])
        self.assertEqual(len(llm.users("classification")), 1)
        self.assertEqual([c["code"] for c in ctx.dk_classifications], ["DK 999.5", "DK 999.3"])

        # dk_postprocess mit leerer Katalogdatenlage: nur die vergebenen Notationen.
        self.assertEqual([r["dk"] for r in ctx.dk_search_results_flattened], ["DK 999.5", "DK 999.3"])
        self.assertTrue(all(r["titles"] == [] for r in ctx.dk_search_results_flattened))

        # geo_candidates sieht die vergebenen Notationen.
        assigned = {c["code"]: c["assigned"] for c in ctx.extra["geo_sachdb"]["candidates"]}
        self.assertTrue(assigned["999.5"] and assigned["999.3"])
        self.assertFalse(assigned["999.1"])

        state = ctx.to_keyword_analysis_state()
        lines = build_winibw_record(state)["lines"]
        self.assertEqual([line for line in lines if line.startswith("6700")],
                         ["6700 !121212121!", "6700 !666666666!"])
        proposals = [p["line"] for p in geo_sachdb_proposals(state.geo_sachdb)]
        self.assertNotIn("6700 !121212121!", proposals)
        self.assertNotIn("6700 !666666666!", proposals)
        self.assertIn("6700 DK 999.21", proposals)

        # Herkunft im Ergebnis, Quellenübersicht, Reflexion und Planer.
        prior = {c["code"]: c for c in state.geo_prior["candidates"]}
        self.assertFalse(prior["999.5"]["in_catalog"])
        usage = {row["source"]: row for row in state.source_usage}
        self.assertEqual((usage["geo_sachdb"]["requests"], usage["geo_sachdb"]["hits"]), (6, 5))
        self.assertTrue(any("- Lokale Zuordnungen: 6 Notationen\n" in u for u in llm.users("reflection")))
        self.assertTrue(any("- Lokale Zuordnungen: 6 Notationen\n" in u for u in llm.users("planner")))


# ── Reflexion und Planer ─────────────────────────────────────────────────────

class StateLine(unittest.TestCase):
    def _ctx(self, prior):
        ctx = SharedContext(abstract="x")
        if prior is not None:
            ctx.extra[GEO_PRIOR_EXTRA_KEY] = prior
        return ctx

    def test_the_line_counts_the_local_notations(self):
        from src.core.agents.steps.reflection_step import local_notations_line

        cands = [{"code": str(i)} for i in range(6)]
        self.assertEqual(local_notations_line(self._ctx({"status": "ok", "candidates": cands})),
                         "- Lokale Zuordnungen: 6 Notationen\n")
        self.assertEqual(local_notations_line(self._ctx({"status": "ok", "candidates": cands[:1]})),
                         "- Lokale Zuordnungen: 1 Notation\n")
        for prior in (None, {"status": "ok", "candidates": []}, {"status": "disabled", "candidates": cands}):
            self.assertEqual(local_notations_line(self._ctx(prior)), "", prior)

    def _reflection_user_prompt(self, ctx):
        from src.core.agents.steps.reflection_step import ReflectionStep
        from src.core.agents.workflow_loader import StepConfig

        llm = _Llm()
        step = ReflectionStep(StepConfig(id="reflection", type="reflection", raw={"id": "reflection"}),
                              llm_service=llm.service, tool_registry=Mock())
        step.run(ctx)
        return llm.calls[-1][2]

    def test_the_reflection_sees_the_line_only_with_local_notations(self):
        with_prior = self._ctx({"status": "ok", "candidates": [{"code": "1"}, {"code": "2"}]})
        text = self._reflection_user_prompt(with_prior)
        self.assertIn("- Katalog-Notationen (unique): 0\n- Lokale Zuordnungen: 2 Notationen\n"
                      "- Fehlende Konzepte:", text)
        text = self._reflection_user_prompt(self._ctx(None))
        self.assertIn("- Katalog-Notationen (unique): 0\n- Fehlende Konzepte:", text)
        self.assertNotIn("Lokale Zuordnungen", text)


# ── Herkunft in Karte und CLI ────────────────────────────────────────────────

def _state(prior_payload=None, notations=("DK 999.5", "DK 999.1", "DK 888.8"), dk_entries=()):
    ctx = SharedContext(abstract="x")
    ctx.working_title = "Testtitel"
    ctx.extra["final_keywords"] = list(KEYWORDS)
    ctx.extra["dk_entries"] = list(dk_entries)
    if prior_payload is not None:
        ctx.extra[GEO_PRIOR_EXTRA_KEY] = prior_payload
    ctx.dk_classifications = [{"code": code, "type": "DK", "rank": "core" if i == 0 else "additional"}
                              for i, code in enumerate(notations)]
    return ctx.to_keyword_analysis_state()


#: DK 999.1 im Katalog; DDC 999.5 hat dieselben Ziffern wie die lokale DK
#: 999.5 und darf sie nicht zu „Katalog und lokale Datenbank“ machen.
CATALOGUE_ENTRIES = [{"dk": "999.1", "type": "DK", "classification_type": "DK", "titles": ["Ein Testbuch"],
                      "count": 1},
                     {"dk": "999.5", "type": "DDC", "classification_type": "DDC", "titles": ["y"], "count": 1},
                     {"dk": "QX 100", "type": "RVK", "classification_type": "RVK", "titles": ["x"], "count": 1}]


class OriginIsShown(_DbCase):
    def setUp(self):
        super().setUp()
        self.payload = self.prior()

    def _card(self, state):
        from src.utils.pipeline_formatters import PipelineResultFormatter

        return PipelineResultFormatter.format_dk_classifications_card_html(state)

    def test_the_state_says_whether_the_catalogue_had_it_too(self):
        prior = {c["code"]: c for c in _state(self.payload, dk_entries=CATALOGUE_ENTRIES).geo_prior["candidates"]}
        self.assertEqual(len(prior), 6)
        self.assertTrue(prior["999.1"]["in_catalog"])
        self.assertFalse(prior["999.5"]["in_catalog"])
        self.assertEqual(prior["999.5"]["via_core_keywords"], ["Testgestein", "Testmineral"])
        self.assertEqual(set(prior["999.5"]), {"notation", "system", "code", "label", "via_keywords",
                                               "via_core_keywords", "in_catalog"})
        self.assertEqual(_state({**self.payload, "status": "disabled"}).geo_prior, {})
        self.assertEqual(_state(None).geo_prior, {})

    def test_the_card_marks_local_and_catalogue_origin(self):
        html, plain = self._card(_state(self.payload, dk_entries=CATALOGUE_ENTRIES))
        rows = re.findall(r'<div class="classification-entry">(.*?)</div></div>', html, re.S)
        by_code = {re.search(r'entry__code">([^<]+)<', row).group(1): row for row in rows}
        self.assertIn('classification-badge--local">lokale Datenbank<', by_code["DK 999.5"])
        self.assertIn("über Testgestein, Testmineral", by_code["DK 999.5"])
        self.assertIn('classification-badge--local">Katalog und lokale Datenbank<', by_code["DK 999.1"])
        self.assertIn("über Testgestein", by_code["DK 999.1"])
        self.assertNotIn("classification-badge--local", html.split("DK 888.8")[1])
        self.assertEqual(plain, "DK 999.5 (Kern; lokale Datenbank), DK 999.1 (Katalog und lokale Datenbank), "
                                "DK 888.8")

    def test_without_local_data_the_card_is_unchanged(self):
        without = self._card(_state(None))
        self.assertNotIn("classification-badge--local", without[0])
        self.assertNotIn("lokale Datenbank", without[0])
        self.assertEqual(without[1], "DK 999.5 (Kern), DK 999.1, DK 888.8")
        # Lokale Daten, die keine vergebene Notation treffen, ändern nichts.
        other = {"status": "ok", "candidates": [{"code": "555.5", "notation": "DK 555.5", "system": "DK",
                                                 "via_keywords": ["X"], "via_core_keywords": []}]}
        self.assertEqual(self._card(_state(other)), without)

    def test_the_cli_marks_the_origin(self):
        from src.cli.formatters.protocol_formatters import format_step_dk_classification

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            format_step_dk_classification(_state(self.payload, dk_entries=CATALOGUE_ENTRIES))
        out = buffer.getvalue().splitlines()
        self.assertIn("  * DK 999.5 · lokale Datenbank · über Testgestein, Testmineral", out)
        self.assertIn("  * DK 999.1 · Katalog und lokale Datenbank · über Testgestein", out)
        self.assertIn("  * DK 888.8", out)

    def test_the_cli_is_unchanged_without_local_data(self):
        from src.cli.formatters.protocol_formatters import format_step_dk_classification

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            format_step_dk_classification(_state(None))
        out = buffer.getvalue().splitlines()
        start = out.index("DK Classifications (3 assigned):")
        self.assertEqual(out[start + 1:start + 4], ["  * DK 999.5", "  * DK 999.1", "  * DK 888.8"])
        self.assertNotIn("lokale Datenbank", buffer.getvalue())


class OriginIsSaved(_DbCase):
    def setUp(self):
        super().setUp()
        self.payload = self.prior()
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "state.json")

    def tearDown(self):
        self.dir.cleanup()
        super().tearDown()

    def _load(self):
        from src.utils.pipeline_persistence import PipelineJsonManager

        return PipelineJsonManager.load_analysis_state(self.path)

    def test_state_file_and_export_file_keep_it(self):
        from src.utils.pipeline_persistence import PipelineJsonManager, export_analysis_state_to_file

        state = _state(self.payload, dk_entries=CATALOGUE_ENTRIES)
        PipelineJsonManager.save_analysis_state(state, self.path)
        self.assertEqual(self._load().geo_prior, state.geo_prior)
        export_analysis_state_to_file(state, self.path, validate_rvk=False)
        self.assertEqual(json.loads(Path(self.path).read_text(encoding="utf-8"))["results"]["geo_prior"],
                         state.geo_prior)
        self.assertEqual(self._load().geo_prior, state.geo_prior)

    def test_old_files_without_the_field_load_empty(self):
        from src.utils.pipeline_persistence import export_analysis_state_to_file
        from src.utils.pipeline_formatters import PipelineResultFormatter

        data = dataclasses.asdict(_state(None))
        data.pop("geo_prior")
        Path(self.path).write_text(json.dumps(data), encoding="utf-8")
        state = self._load()
        self.assertEqual(state.geo_prior, {})
        self.assertNotIn("lokale Datenbank", PipelineResultFormatter.format_dk_classifications_card_html(state)[0])

        export_analysis_state_to_file(_state(None), self.path, validate_rvk=False)
        data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        data["results"].pop("geo_prior")
        Path(self.path).write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self._load().geo_prior, {})


if __name__ == "__main__":
    unittest.main()
