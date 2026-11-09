"""Publikationsform aus dem Eingabetext (alima_v51_105, Schritt publication_form) - Claude Generated.

Geprüft werden: die Regeltabelle (nur Formen aus FORM_NOTATIONS), Positivfälle
(Zusammenfassung einer Dissertation, Titelei deutsch und englisch, Diplom-/
Masterarbeit, Selbstbezug), Negativfälle (Wort „Dissertation“ ohne
Selbstbeschreibung, zitierte fremde Arbeit, Plural, Bachelorarbeit,
Habilitation, Promotionsordnung), Reichweite und Reihenfolge der Regeln, die
Textstelle, das Einsetzen ins Ergebnis, der Workflow-Schritt (ohne Treffer
unverändert), die Aufstellung (mit und ohne Treffer gleich), ein Lauf über den
MetaAgent bis zum WinIBW-Eintrag, Anzeige in DK-Karte und CLI, Speichern und
Laden. Kein Netz, kein echtes Modell; alle Texte und Daten erfunden.
"""

from __future__ import annotations

import copy
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

from src.core.agents.deterministic_functions import publication_form, shelf_candidates
from src.core.agents.registry import TOOL_FN_REGISTRY
from src.core.agents.shared_context import PUBLICATION_FORM_EXTRA_KEY, SharedContext
from src.core.data_models import AgentResponse
from src.utils.classification_systems import FORM_NOTATIONS, without_form_notations
from src.utils.lookups.geo_sachdb.provider import GeoSachdbLookup
from src.utils.lookups.local_scheme import LocalSchemeLookup
from src.utils.publication_form import (
    EVIDENCE_MAX,
    FORMS,
    HEAD_CHARS,
    ORIGIN_PUBLICATION_FORM,
    RULES,
    apply_publication_form,
    detect_publication_form,
)

_spec = importlib.util.spec_from_file_location(
    "geo_lookup_fixture_form", Path(__file__).resolve().parent / "test_geo_sachdb_lookup.py"
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
_HOME = {}


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


def _workflow(name="alima_v51_105"):
    from src.core.agents.workflow_loader import find_workflow_file, load_workflow

    return load_workflow(find_workflow_file(name), strict=True)


def _step(step_id):
    return {s.id: s for s in _workflow().steps}[step_id]


# ── erfundene Texte ──────────────────────────────────────────────────────────

SUMMARY = ("Zusammenfassung der Ergebnisse der Dissertation „Verhalten von Testgestein im Musterland“ "
           "von Herrn M. Sc. Max Muster\n\nDie Arbeit untersucht das Verhalten von Testgestein unter "
           "Druck und beschreibt die Lagerstätten im Musterland.")
NEUTRAL = ("Das Buch untersucht das Verhalten von Testgestein unter Druck und beschreibt die "
           "Lagerstätten im Musterland.")

POSITIVE = {
    "summary": (SUMMARY, "dissertation", "zusammenfassung_dissertation",
                "Zusammenfassung der Ergebnisse der Dissertation"),
    "summary_short": ("Zusammenfassung der Dissertation\n\nTestgestein im Musterland wird beschrieben.",
                      "dissertation", "zusammenfassung_dissertation", "Zusammenfassung der Dissertation"),
    "titelei_genehmigt": (
        "Verhalten von Testgestein\n\nVon der Fakultät für Geowissenschaften\nder Technischen "
        "Universität Musterstadt\ngenehmigte\nDISSERTATION\nzur Erlangung des akademischen Grades\n"
        "Doktor-Ingenieur\n(Dr.-Ing.)\nvorgelegt\nvon Dipl.-Ing. Max Muster",
        "dissertation", "titelei_genehmigte_dissertation", "genehmigte DISSERTATION zur Erlangung des"),
    "titelei_vorgelegt": (
        "Testgesteine im Wandel\nDissertation\nzur Erlangung des Doktorgrades der Naturwissenschaften "
        "(Dr. rer. nat.)\nvorgelegt von\nErika Muster\naus Musterstadt",
        "dissertation", "titelei_doktorgrad_vorgelegt",
        "zur Erlangung des Doktorgrades der Naturwissenschaften (Dr. rer. nat.) vorgelegt von"),
    "thesis_en": (
        "Mineral phases of test rock\n\nA thesis submitted in partial fulfilment of the requirements "
        "for the degree of Doctor of Philosophy\n\nby Jane Doe",
        "dissertation", "thesis_degree_doctor",
        "thesis submitted in partial fulfilment of the requirements for the degree of Doctor"),
    "titel_von": (
        "Kurzbericht. Die Dissertation „Testgestein unter Druck“ von Frau Dipl.-Ing. Erika Muster "
        "beschreibt neue Lagerstätten.",
        "dissertation", "dissertation_titel_von",
        "Dissertation „Testgestein unter Druck“ von Frau"),
    "vorliegende": (
        "Abstract. Im Rahmen der vorliegenden Dissertation wurde das Verhalten von Testgestein untersucht.",
        "dissertation", "vorliegende_dissertation", "vorliegenden Dissertation"),
    "this_dissertation": ("In this dissertation, test rock is studied under pressure.",
                          "dissertation", "this_dissertation", "this dissertation"),
    "master_titelei": (
        "Testgesteine\nMasterarbeit\nzur Erlangung des akademischen Grades Master of Science (M. Sc.)\n"
        "eingereicht von\nMax Muster",
        "diplom_master", "titelei_diplom_master_vorgelegt",
        "zur Erlangung des akademischen Grades Master of Science (M. Sc.) eingereicht von"),
    "diplom_summary": ("Kurzfassung der Diplomarbeit\n\nUntersucht wird Testgestein.",
                       "diplom_master", "zusammenfassung_diplom_master", "Kurzfassung der Diplomarbeit"),
    "master_en": ("In this master's thesis, test rock is mapped.",
                  "diplom_master", "this_master_thesis", "this master's thesis"),
}

NEGATIVE = {
    "anfertigung": "Das Buch hilft bei der Anfertigung von Dissertationen und gibt Tipps zur Zeitplanung.",
    "auswertung": "Wir werten 200 Dissertationen zum Thema Testgestein aus und vergleichen ihre Methoden.",
    "zitat": "Die Dissertation von Müller (2019) zeigt, dass Testgestein unter Druck zerfällt. Wir prüfen "
             "diese These an neuen Proben.",
    "basiert_auf": "Der Beitrag basiert auf der Dissertation „Testgestein im Wandel“ von Frau M. Sc. "
                   "Erika Muster und ergänzt neue Messungen.",
    "based_on": "This paper is based on a thesis submitted for the degree of Doctor of Philosophy at "
                "Test University.",
    "promotionsordnung": "Die Promotionsordnung regelt das Verfahren zur Erlangung des akademischen "
                         "Grades Doktor-Ingenieur. Die Dissertation ist beim Promotionsausschuss "
                         "einzureichen.",
    "bachelor": "Testgestein\nBachelorarbeit\nzur Erlangung des akademischen Grades Bachelor of Science\n"
                "vorgelegt von Max Muster",
    "habilitation_venia": "Habilitationsschrift zur Erlangung der Venia legendi vorgelegt von Dr. Max Muster",
    "habilitation_grad": "Habilitationsschrift zur Erlangung des akademischen Grades Dr. habil. "
                         "vorgelegt von Dr. Max Muster",
    "genitiv": "Die Betreuung einer Dissertation erfordert Zeit. Der Ratgeber zeigt, wie man eine "
               "Dissertation plant.",
    "fremde_zusammenfassung_spaeter": "In seiner Dissertation untersuchte Müller das Testgestein; die "
                                      "Zusammenfassung der Dissertation erschien 2019.",
    "plagiat": "Die Doktorarbeit von Herrn Muster wurde wegen Plagiaten aberkannt, berichtet die Zeitung.",
    "plural_am_anfang": "Zusammenfassung der Ergebnisse der Dissertationen des Jahrgangs 2019.",
    "englisch_plural": "Many dissertations address test rock; we review forty of them.",
    "leer": "",
}


# ── Regeltabelle ─────────────────────────────────────────────────────────────

class RuleTable(unittest.TestCase):
    def test_every_form_notation_is_a_registered_form_notation(self):
        for form, spec in FORMS.items():
            system, code = spec["notation"]
            self.assertIn(code, FORM_NOTATIONS.get(system, {}), form)
            self.assertEqual(spec["form_keyword"], "Hochschulschrift", form)

    def test_no_form_without_a_form_notation(self):
        self.assertEqual(set(FORMS), {"dissertation", "diplom_master"})
        self.assertEqual({rule.form for rule in RULES}, set(FORMS))
        self.assertEqual(len({rule.name for rule in RULES}), len(RULES))
        self.assertTrue(all(rule.scope in ("start", "head", "anywhere") for rule in RULES))


# ── Erkennung ────────────────────────────────────────────────────────────────

class Positive(unittest.TestCase):
    def test_positive_cases(self):
        for name, (text, form, rule, evidence) in POSITIVE.items():
            with self.subTest(name):
                found = detect_publication_form(text)
                self.assertIsNotNone(found, name)
                self.assertEqual((found["form"], found["rule"], found["evidence"]), (form, rule, evidence))
                self.assertEqual(found["notation"], FORMS[form]["notation"])
                self.assertEqual(found["form_keyword"], "Hochschulschrift")
                self.assertEqual(found["label"], FORMS[form]["label"])

    def test_dissertation_and_master_notations(self):
        self.assertEqual(detect_publication_form(SUMMARY)["notation"], ("DK", "378.245"))
        self.assertEqual(detect_publication_form(POSITIVE["master_titelei"][0])["notation"], ("DK", "378.244"))


class Negative(unittest.TestCase):
    def test_negative_cases(self):
        for name, text in NEGATIVE.items():
            with self.subTest(name):
                self.assertIsNone(detect_publication_form(text), name)
        self.assertIsNone(detect_publication_form(None))
        self.assertIsNone(detect_publication_form(NEUTRAL))


class ScopeOrderEvidence(unittest.TestCase):
    def test_a_title_page_formula_far_into_the_text_does_not_count(self):
        titelei = "genehmigte Dissertation zur Erlangung des akademischen Grades Doktor-Ingenieur"
        self.assertIsNotNone(detect_publication_form(titelei))
        padded = ("Testgestein wird beschrieben. " * 30)[:HEAD_CHARS + 50] + " " + titelei
        self.assertIsNone(detect_publication_form(padded))

    def test_a_summary_phrase_counts_only_at_the_start(self):
        self.assertIsNotNone(detect_publication_form("„Zusammenfassung der Dissertation“: Testgestein"))
        self.assertIsNone(detect_publication_form("Testgestein. Zusammenfassung der Dissertation"))

    def test_the_first_rule_of_the_table_wins_and_there_is_one_result(self):
        # Titelei einer Masterarbeit vor einem Selbstbezug „vorliegende Dissertation“.
        text = (POSITIVE["master_titelei"][0] + "\n\nDie vorliegende Dissertation …")
        found = detect_publication_form(text)
        self.assertEqual((found["form"], found["rule"]), ("diplom_master", "titelei_diplom_master_vorgelegt"))
        both = SUMMARY + " Die vorliegende Dissertation …"
        self.assertEqual(detect_publication_form(both)["rule"], "zusammenfassung_dissertation")
        order = [rule.name for rule in RULES]
        self.assertLess(order.index("titelei_diplom_master_vorgelegt"), order.index("vorliegende_dissertation"))

    def test_evidence_is_short_and_single_spaced(self):
        long_title = "Testgestein " * 20
        found = detect_publication_form(f"Die Dissertation „{long_title.strip()}“ von Herrn M. Sc. Max Muster")
        self.assertEqual(found["rule"], "dissertation_titel_von")
        self.assertLessEqual(len(found["evidence"]), EVIDENCE_MAX)
        self.assertTrue(found["evidence"].endswith("…"))
        found = detect_publication_form(POSITIVE["titelei_genehmigt"][0])
        self.assertNotIn("\n", found["evidence"])
        self.assertNotIn("  ", found["evidence"])


# ── ins Ergebnis ─────────────────────────────────────────────────────────────

PICKED = [{"code": "DK 999.3", "type": "DK", "rank": "core"},
          {"code": "DK 999.21", "type": "DK", "rank": "additional"}]


class Apply(unittest.TestCase):
    def setUp(self):
        self.detection = detect_publication_form(SUMMARY)

    def test_form_keyword_added_and_notation_first(self):
        keywords, classes = apply_publication_form(self.detection, [{"keyword": "Lehrbuch", "gnd_id": "x"}],
                                                   copy.deepcopy(PICKED))
        self.assertEqual(keywords, [{"keyword": "Lehrbuch", "gnd_id": "x"}, {"keyword": "Hochschulschrift"}])
        self.assertEqual([c["code"] for c in classes], ["DK 378.245", "DK 999.3", "DK 999.21"])
        self.assertEqual({k: classes[0][k] for k in ("code", "type", "rank", "origin")},
                         {"code": "DK 378.245", "type": "DK", "rank": "additional",
                          "origin": ORIGIN_PUBLICATION_FORM})
        self.assertIn("Form (aus dem Eingabetext)", classes[0]["reason"])
        self.assertIn(self.detection["evidence"], classes[0]["reason"])

    def test_form_keyword_is_not_added_twice(self):
        for present in ({"keyword": "Hochschulschrift", "gnd_id": "x"}, {"keyword": "hochschulschrift"},
                        "Hochschulschrift (GND-ID: x)"):
            with self.subTest(present):
                keywords, _ = apply_publication_form(self.detection, [present], [])
                self.assertEqual(keywords, [present])

    def test_an_assigned_notation_is_only_moved_to_the_front(self):
        assigned = {"code": "378.245", "type": "DK", "rank": "core"}
        given = copy.deepcopy(PICKED) + [assigned]
        _, classes = apply_publication_form(self.detection, [], given)
        self.assertEqual(len(classes), 3)
        self.assertIs(classes[0], assigned)
        self.assertEqual(assigned, {"code": "378.245", "type": "DK", "rank": "core"})
        self.assertEqual(classes[1:], PICKED)
        _, strings = apply_publication_form(self.detection, [], ["DK 999.3", "DK 378.245"])
        self.assertEqual(strings, ["DK 378.245", "DK 999.3"])

    def test_inputs_are_not_mutated_and_no_detection_returns_them(self):
        keywords, classes = [{"keyword": "Lehrbuch"}], copy.deepcopy(PICKED)
        before = copy.deepcopy((keywords, classes))
        apply_publication_form(self.detection, keywords, classes)
        self.assertEqual((keywords, classes), before)
        out = apply_publication_form(None, keywords, classes)
        self.assertIs(out[0], keywords)
        self.assertIs(out[1], classes)


# ── Workflow-Schritt ─────────────────────────────────────────────────────────

class Wiring(unittest.TestCase):
    def test_the_step_sits_after_dk_postprocess_and_before_geo_candidates(self):
        steps = _workflow().steps
        ids = [s.id for s in steps]
        self.assertEqual(ids.index("publication_form"), ids.index("dk_postprocess") + 1)
        self.assertEqual(ids.index("geo_candidates"), ids.index("publication_form") + 1)
        step = _step("publication_form")
        self.assertTrue(step.enabled)
        self.assertEqual(step.depends_on, ["dk_postprocess"])
        self.assertIn("publication_form", _step("geo_candidates").depends_on)
        self.assertEqual(step.raw["function"], "publication_form")
        self.assertIn("publication_form", TOOL_FN_REGISTRY)
        self.assertEqual(step.inputs, {"abstract": "${abstract}", "form_keywords": "${extra.form_keywords}",
                                       "dk_classifications": "${dk_classifications}"})
        self.assertEqual(step.outputs, {"extra.publication_form": "result.publication_form",
                                        "extra.form_keywords": "result.form_keywords",
                                        "dk_classifications": "result.dk_classifications"})
        self.assertEqual(PUBLICATION_FORM_EXTRA_KEY, "publication_form")

    def test_shelf_select_reads_the_list_without_form_notations(self):
        self.assertEqual(_step("shelf_select").inputs["dk_codes"], "${extra.shelf.classifications}")

    def test_planner_and_reflection_name_the_step(self):
        meta = _workflow().meta_agent
        planning = " ".join(meta["planning"]["rules"].split())
        self.assertIn("dk_postprocess → publication_form → geo_candidates", planning)
        reflection = " ".join(meta["reflection"]["rules"].split())
        self.assertIn("publication_form→Formschlagwort und Formnotation", reflection)
        self.assertIn("ist gewollt und KEIN Mangel", reflection)

    def test_the_step_is_pending_like_any_mandatory_step(self):
        from src.core.agents.meta_agent import MetaAgent

        ctx = SharedContext(abstract="x")
        ctx.execution_history = [{"step": s} for s in (
            "extraction", "search", "selection_chunks", "selection", "verify_keywords", "dk_collect",
            "geo_prior", "classification", "rvk_guard", "dk_postprocess")]
        self.assertEqual(MetaAgent._pending_step(_workflow(), ctx), "publication_form")
        self.assertEqual(MetaAgent._unmet_dependency(_workflow(), ctx, "geo_candidates"), "publication_form")

    def test_the_general_workflow_has_no_such_step(self):
        self.assertNotIn("publication_form", [s.id for s in _workflow("alima_v51").steps])


def _context(abstract, form_keywords=None, classes=None, set_form=True):
    ctx = SharedContext(abstract=abstract)
    ctx.dk_classifications = copy.deepcopy(PICKED if classes is None else classes)
    ctx.extra["final_keywords"] = [{"keyword": "Testgestein", "gnd_id": "1"}]
    if set_form:
        ctx.extra["form_keywords"] = copy.deepcopy(form_keywords or [])
    return ctx


def _run_step(ctx):
    from src.core.agents.steps.deterministic_step import DeterministicStep

    result = DeterministicStep(_step("publication_form")).execute(ctx)
    assert result.success, result.error
    return ctx


class Step(unittest.TestCase):
    def test_without_a_hit_nothing_changes(self):
        for set_form in (True, False):
            with self.subTest(form_keywords_present=set_form):
                ctx = _context(NEUTRAL, [{"keyword": "Lehrbuch", "gnd_id": "x"}], set_form=set_form)
                untouched = _context(NEUTRAL, [{"keyword": "Lehrbuch", "gnd_id": "x"}], set_form=set_form)
                extra_before = copy.deepcopy(ctx.extra)
                classes_before = ctx.dk_classifications
                _run_step(ctx)
                self.assertEqual(ctx.extra, extra_before)
                self.assertNotIn("publication_form", ctx.extra)
                self.assertIs(ctx.dk_classifications, classes_before)
                self.assertEqual(ctx.dk_classifications, PICKED)
                self.assertEqual(dataclasses.asdict(ctx.to_keyword_analysis_state()) | {"timestamp": ""},
                                 dataclasses.asdict(untouched.to_keyword_analysis_state()) | {"timestamp": ""})

    def test_with_a_hit_keyword_notation_and_detection(self):
        ctx = _run_step(_context(SUMMARY, [{"keyword": "Lehrbuch", "gnd_id": "x"}]))
        self.assertEqual(ctx.extra["form_keywords"],
                         [{"keyword": "Lehrbuch", "gnd_id": "x"}, {"keyword": "Hochschulschrift"}])
        self.assertEqual([c["code"] for c in ctx.dk_classifications], ["DK 378.245", "DK 999.3", "DK 999.21"])
        found = ctx.extra["publication_form"]
        self.assertEqual((found["form"], found["notation"], found["rule"]),
                         ("dissertation", ["DK", "378.245"], "zusammenfassung_dissertation"))
        json.dumps(found)  # speicherbar
        state = ctx.to_keyword_analysis_state()
        self.assertEqual(state.form_keywords, ["Lehrbuch (GND-ID: x)", "Hochschulschrift"])
        self.assertEqual(state.publication_form, {
            "form": "dissertation", "label": "Dissertation", "system": "DK", "code": "378.245",
            "form_keyword": "Hochschulschrift", "evidence": "Zusammenfassung der Ergebnisse der Dissertation",
            "rule": "zusammenfassung_dissertation"})
        self.assertEqual(state.classification_entries[0],
                         {"system": "DK", "code": "378.245", "display": "DK 378.245", "rank": "additional"})

    def test_without_form_keywords_from_the_selection(self):
        ctx = _run_step(_context(SUMMARY, set_form=False))
        self.assertEqual(ctx.extra["form_keywords"], [{"keyword": "Hochschulschrift"}])

    def test_the_function_reads_the_context_when_inputs_are_missing(self):
        ctx = _context(SUMMARY)
        out = publication_form(context=ctx)
        self.assertEqual(out["dk_classifications"][0]["code"], "DK 378.245")
        none = publication_form(context=_context(NEUTRAL))
        self.assertIsNone(none["publication_form"])


# ── Aufstellung ──────────────────────────────────────────────────────────────

SCHEME = """
version: 1
scheme:
  id: test_form_scheme
  label: "Testsystematik"
  base_system: DK
  source: "https://example.invalid/systematik"
  retrieved: 2026-11-09
entries:
  - notation: "999"
    label: "Testklasse"
    location: "OG"
  - notation: "999.3"
    label: "Musterland"
    location: "OG"
  - notation: "378"
    label: "Hochschulwesen"
    location: "EG"
  - notation: "378.24"
    label: "Hochschulabschluesse"
    location: "EG"
"""


class _SchemeCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "test_form_scheme.yaml"
        path.write_text(SCHEME, encoding="utf-8")
        self.scheme = LocalSchemeLookup(vocabulary=str(path))

    def tearDown(self):
        self._tmp.cleanup()


class ShelfIsUnchanged(_SchemeCase):
    KEYWORDS = [{"keyword": "Testgestein"}]

    def _shelf(self, classes):
        with patch("src.utils.lookups.build_lookup", return_value=self.scheme):
            return shelf_candidates(notations=classes, keywords=self.KEYWORDS, config={"lookup": "local_scheme"})

    def test_same_proposals_with_and_without_the_form_notation(self):
        detection = detect_publication_form(SUMMARY)
        cases = {
            "ranked": copy.deepcopy(PICKED),
            "unranked": [{"code": "DK 999.3", "type": "DK"}, {"code": "DK 999.21", "type": "DK"}],
            "assigned_as_core": copy.deepcopy(PICKED) + [{"code": "DK 378.245", "type": "DK", "rank": "core"}],
        }
        for name, classes in cases.items():
            with self.subTest(name):
                _, with_form = apply_publication_form(detection, [], classes)
                self.assertEqual(with_form[0]["code"], "DK 378.245")
                before, after = self._shelf(classes), self._shelf(with_form)
                self.assertTrue(after["candidates"], name)
                for key in ("candidates", "unplaced", "prompt_text", "classifications"):
                    self.assertEqual(after[key], before[key], f"{name}: {key}")
                self.assertNotIn("Hochschulwesen", after["prompt_text"])
                self.assertEqual(after["classifications"], without_form_notations(classes))

    def test_without_form_notations_the_list_is_the_input(self):
        result = self._shelf(copy.deepcopy(PICKED))
        self.assertEqual(result["classifications"], PICKED)
        empty = shelf_candidates(notations=copy.deepcopy(PICKED), config={"lookup": "nicht_da"})
        self.assertEqual(empty["classifications"], PICKED)

    def test_without_form_notations_keeps_order_and_entries(self):
        mixed = ["DK 999.3", {"code": "378.244", "type": "DK"}, {"code": "DK 061.3"},
                 {"code": "RVK QX 100", "type": "RVK"}, {"code": "999.21", "system": "DK"}]
        self.assertEqual(without_form_notations(mixed),
                         ["DK 999.3", {"code": "RVK QX 100", "type": "RVK"}, {"code": "999.21", "system": "DK"}])


# ── Lauf durch den Workflow (MetaAgent) ──────────────────────────────────────

FORM_PPN = "373737373"


def _build_db(path: Path, with_form_notation: bool) -> None:
    _fixture.build_synthetic_db(path)
    if with_form_notation:
        conn = sqlite3.connect(path)
        conn.execute("INSERT INTO classification VALUES (?,?,?,?,?)",
                     (30, "DK", "378.245", "Hochschulschriften (Test)", FORM_PPN))
        conn.commit()
        conn.close()


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
        elif "Aufstellung im Freihandbestand" in system:
            role, content = "shelf", json.dumps({"shelf": {"notation": "999.3", "reason": "x"}})
        else:
            role, content = "other", "{}"
        self.calls.append((role, system, user))
        return AgentResponse(content=content)

    def prompts(self, role):
        return [(system, user) for r, system, user in self.calls if r == role]


class EndToEnd(_SchemeCase):
    """MetaAgent from the step after dk_collect to the end: classification →
    rvk_guard → dk_postprocess → publication_form → geo_candidates → shelf_*
    → WinIBW entry."""

    def _run(self, abstract, *, with_form_notation=True, detect=True):
        from src.core.agents.meta_agent import MetaAgent

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Path(tmp.name) / "geo.db"
        _build_db(db, with_form_notation)
        geo = GeoSachdbLookup(**_fixture.settings(db))
        lookups = {"geo_sachdb": geo, "local_scheme": self.scheme}

        ctx = SharedContext(abstract=abstract)
        ctx.provider, ctx.model = "stub", "stub-model"
        ctx.working_title = "Testtitel"
        ctx.extra.update(final_keywords=[{"keyword": "Testgestein", "gnd_id": "1"},
                                         {"keyword": "Musterland", "gnd_id": "4"}],
                         core_keywords=[{"keyword": "Testgestein", "gnd_id": "1"}],
                         form_keywords=[],
                         dk_prompt_text="DK: 999.3 (Häufigkeit: 3) | Beispieltitel: Ein Testbuch",
                         dk_entries=[])
        ctx.execution_history = [
            {"cycle": i, "step": step, "success": True, "duration": 0.0, "error": None}
            for i, step in enumerate(["extraction", "search", "selection_chunks", "selection",
                                      "verify_keywords", "dk_collect"], 1)
        ]
        llm = _Llm()
        registry = Mock()
        registry.get_tool_schemas.return_value = []
        workflow = _workflow()
        agent = MetaAgent(llm_service=llm.service, tool_registry=registry)
        patches = [patch("src.utils.lookups.build_lookup", side_effect=lambda _c, lid: lookups.get(lid))]
        if not detect:
            patches.append(patch("src.utils.publication_form.detect_publication_form", return_value=None))
        for item in patches:
            item.start()
        try:
            report = agent.run(workflow, ctx, meta_config=workflow.meta_agent)
        finally:
            for item in patches:
                item.stop()
        self.assertTrue(report.success, report.error)
        return llm, ctx

    def test_a_dissertation_gets_1131_and_6700_with_the_local_ppn(self):
        from src.utils.winibw_record import build_winibw_record

        llm, ctx = self._run(SUMMARY)
        tail = [h["step"] for h in ctx.execution_history][6:]
        self.assertEqual(tail.index("publication_form"), tail.index("dk_postprocess") + 1)
        self.assertLess(tail.index("publication_form"), tail.index("geo_candidates"))
        self.assertEqual([c["code"] for c in ctx.dk_classifications], ["DK 378.245", "DK 999.3", "DK 999.21"])
        lines = build_winibw_record(ctx.to_keyword_analysis_state())["lines"]
        self.assertIn("1131 Hochschulschrift", lines)
        self.assertEqual([line for line in lines if line.startswith("6700")],
                         [f"6700 !{FORM_PPN}!", "6700 !666666666!", "6700 DK 999.21"])

    def test_without_a_local_entry_the_notation_stays_text(self):
        from src.utils.winibw_record import build_winibw_record

        _llm, ctx = self._run(SUMMARY, with_form_notation=False)
        lines = build_winibw_record(ctx.to_keyword_analysis_state())["lines"]
        self.assertIn("1131 Hochschulschrift", lines)
        self.assertEqual([line for line in lines if line.startswith("6700")][0], "6700 DK 378.245")

    def test_the_shelf_steps_are_the_same_with_and_without_the_hit(self):
        with_hit_llm, with_hit = self._run(SUMMARY)
        without_llm, without = self._run(SUMMARY, detect=False)
        self.assertEqual(without.dk_classifications, PICKED)
        self.assertNotIn("publication_form", without.extra)
        self.assertTrue(with_hit.extra["shelf"]["candidates"])
        for key in ("candidates", "unplaced", "prompt_text", "classifications"):
            self.assertEqual(with_hit.extra["shelf"][key], without.extra["shelf"][key], key)
        self.assertEqual(with_hit.extra["shelf_final"], without.extra["shelf_final"])
        self.assertEqual(with_hit_llm.prompts("shelf"), without_llm.prompts("shelf"))
        self.assertEqual(len(with_hit_llm.prompts("shelf")), 1)
        self.assertNotIn("378.245", with_hit_llm.prompts("shelf")[0][1])

    def test_without_a_hit_the_shelf_prompt_is_the_one_from_before(self):
        # Vorher las shelf_select ${dk_classifications}; ohne Formnotation ist
        # extra.shelf.classifications dieselbe Liste, der Prompt also gleich.
        llm, ctx = self._run(NEUTRAL)
        self.assertNotIn("publication_form", ctx.extra)
        user = llm.prompts("shelf")[0][1]
        self.assertIn("**Vergebene Klassifikationen:**\n" + json.dumps(PICKED, ensure_ascii=False) + "\n", user)


# ── Anzeige ──────────────────────────────────────────────────────────────────

def _state(abstract):
    ctx = _context(abstract)
    ctx.working_title = "Testtitel"
    return _run_step(ctx).to_keyword_analysis_state()


class Display(unittest.TestCase):
    def _card(self, state):
        from src.utils.pipeline_formatters import PipelineResultFormatter

        return PipelineResultFormatter.format_dk_classifications_card_html(state)

    def test_the_card_names_the_origin_with_the_passage_in_the_tooltip(self):
        html, plain = self._card(_state(SUMMARY))
        rows = re.findall(r'<div class="classification-entry">(.*?)</div></div>', html, re.S)
        by_code = {re.search(r'entry__code">([^<]+)<', row).group(1): row for row in rows}
        row = by_code["DK 378.245"]
        self.assertIn('title="Textstelle: „Zusammenfassung der Ergebnisse der Dissertation“"', row)
        self.assertIn(">Form: Dissertation (aus dem Eingabetext)</span>", row)
        self.assertNotIn("aus dem Eingabetext", by_code["DK 999.3"] + by_code["DK 999.21"])
        self.assertIn("DK 378.245 (Form: Dissertation (aus dem Eingabetext), Textstelle "
                      "„Zusammenfassung der Ergebnisse der Dissertation“)", plain)

    def test_without_a_hit_the_card_is_unchanged(self):
        state = _state(NEUTRAL)
        self.assertEqual(state.publication_form, {})
        reference = copy.copy(state)
        del reference.__dict__["publication_form"]  # wie ein Zustand vor dem Feld
        card = self._card(state)
        self.assertEqual(card, self._card(reference))
        self.assertNotIn("Eingabetext", card[0] + card[1])
        # Eine Erkennung, deren Notation nicht vergeben ist, ändert nichts.
        other = copy.copy(state)
        other.publication_form = dict(_state(SUMMARY).publication_form)
        self.assertEqual(self._card(other), card)

    def _cli(self, state):
        from src.cli.formatters.protocol_formatters import format_step_dk_classification

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            format_step_dk_classification(state)
        return buffer.getvalue().splitlines()

    def test_the_cli_names_the_origin_and_the_passage(self):
        out = self._cli(_state(SUMMARY))
        self.assertIn("  * DK 378.245 · Form: Dissertation (aus dem Eingabetext), Textstelle "
                      "„Zusammenfassung der Ergebnisse der Dissertation“", out)
        self.assertIn("  * DK 999.3", out)
        self.assertIn("1131 Hochschulschrift", out)

    def test_the_cli_is_unchanged_without_a_hit(self):
        out = self._cli(_state(NEUTRAL))
        start = out.index("DK Classifications (2 assigned):")
        self.assertEqual(out[start + 1:start + 3], ["  * DK 999.3", "  * DK 999.21"])
        self.assertNotIn("Eingabetext", "\n".join(out))


# ── Speichern und Laden ──────────────────────────────────────────────────────

class Saved(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "state.json")

    def tearDown(self):
        self.dir.cleanup()

    def _load(self):
        from src.utils.pipeline_persistence import PipelineJsonManager

        return PipelineJsonManager.load_analysis_state(self.path)

    def test_state_file_and_export_file_keep_it(self):
        from src.utils.pipeline_persistence import PipelineJsonManager, export_analysis_state_to_file

        state = _state(SUMMARY)
        PipelineJsonManager.save_analysis_state(state, self.path)
        self.assertEqual(self._load().publication_form, state.publication_form)
        export_analysis_state_to_file(state, self.path, validate_rvk=False)
        results = json.loads(Path(self.path).read_text(encoding="utf-8"))["results"]
        self.assertEqual(results["publication_form"], state.publication_form)
        self.assertIn("1131 Hochschulschrift", results["winibw"]["lines"])
        loaded = self._load()
        self.assertEqual(loaded.publication_form, state.publication_form)
        from src.utils.pipeline_formatters import PipelineResultFormatter

        self.assertIn("aus dem Eingabetext", PipelineResultFormatter.format_dk_classifications_card_html(loaded)[0])

    def test_old_files_without_the_field_load_empty(self):
        from src.utils.pipeline_persistence import export_analysis_state_to_file

        data = dataclasses.asdict(_state(NEUTRAL))
        data.pop("publication_form")
        Path(self.path).write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self._load().publication_form, {})

        export_analysis_state_to_file(_state(NEUTRAL), self.path, validate_rvk=False)
        data = json.loads(Path(self.path).read_text(encoding="utf-8"))
        self.assertEqual(data["results"]["publication_form"], {})
        data["results"].pop("publication_form")
        Path(self.path).write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self._load().publication_form, {})


if __name__ == "__main__":
    unittest.main()
