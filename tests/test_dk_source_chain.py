"""Claude Generated - DK sources in list order with fallback per keyword.

Pinned here (Nov 9, 2026):

A. ``factory.dk_candidates``: active instances whose type declares
   CLASSIFICATION and whose DK switch is on, in the order of the plugin list;
   a catalog instance that never stored ``dk_enabled`` counts only when no
   other instance qualifies.
B. Characterisation: the configuration of the UB Freiberg (catalog, finc,
   sru active; finc ``dk_enabled`` off; sru without the key) yields the bare
   Libero extractor, as the fixed chain did; and for every configuration with
   at most one DK switch on and no stored catalog switch, the source is the
   one the fixed chain picked (oracle = the old chain, re-implemented here).
C. ``describe_dk_source`` names the same chain as ``resolve_dk_extractor`` for
   several hundred configurations.
D. ``FallbackDkExtractor``: which keywords go on to the next source (down,
   cache-only miss, no institution, no extractor, connection failure, raised)
   and which do not ("no hits"), one note per hand-over, ``answered_by``.
E. The DK step reports the chain: ``dk_source_by_keyword``, ``dk_fallbacks``,
   one line per hand-over.
F. New installations list finc before catalog; an existing config.json keeps
   its order on load and save.

No test reaches the network: extractors are fakes or are built without being
asked, HOME is a temporary directory, the raw-cache switch is patched.
"""

from __future__ import annotations

import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from src.core.search import source_health
from src.core.search.dk_fallback import DkCandidate, FallbackDkExtractor
from src.core.search.factory import (
    NoDkSourceError,
    describe_dk_source,
    dk_candidates,
    resolve_dk_extractor,
)
from src.core.search.provider import SearchCapability
from src.core.search.registry import PROVIDER_REGISTRY, register_provider
from src.utils.config_models import PluginInstanceConfig


def _inst(pid, *, iid=None, enabled=True, label="", **settings):
    return PluginInstanceConfig(
        instance_id=iid or pid, category="search_provider", provider_id=pid,
        label=label, enabled=enabled, settings=dict(settings),
    )


class _Cfg:
    def __init__(self, *instances):
        self.plugins = list(instances)

    def enabled_instances_for(self, category):
        return [i for i in self.plugins if i.category == category and i.enabled]


class _Quiet(unittest.TestCase):
    """Temporary HOME; the raw-cache switch is not read from a config file."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = patch.dict(os.environ, {"HOME": tmp.name})
        home.start()
        self.addCleanup(home.stop)
        p = patch("src.core.search.factory._global_response_cache_enabled", return_value=False)
        p.start()
        self.addCleanup(p.stop)
        notice = patch.object(source_health, "_emit_notice", lambda _t: None)
        notice.start()
        self.addCleanup(notice.stop)


def _chain_of(ext):
    """Source ids of what resolve_dk_extractor returned."""
    if isinstance(ext, FallbackDkExtractor):
        return ext.source_ids
    return [ext.source_id]


# ===========================================================================
# A. dk_candidates
# ===========================================================================
class TestCandidates(_Quiet):
    def ids(self, *instances):
        return [i.instance_id for i in dk_candidates(_Cfg(*instances))]

    def test_switches_in_list_order(self):
        self.assertEqual(self.ids(_inst("sru", preset="dnb", dk_enabled=True),
                                  _inst("finc", base_url="https://f", dk_enabled=True)),
                         ["sru", "finc"])
        self.assertEqual(self.ids(_inst("finc", base_url="https://f", dk_enabled=True),
                                  _inst("sru", preset="dnb", dk_enabled=True)),
                         ["finc", "sru"])

    def test_catalog_without_stored_switch_only_when_nothing_else(self):
        self.assertEqual(self.ids(_inst("catalog"), _inst("finc", base_url="https://f")),
                         ["catalog"])
        self.assertEqual(self.ids(_inst("catalog"),
                                  _inst("finc", base_url="https://f", dk_enabled=True)),
                         ["finc"])

    def test_stored_catalog_switch_is_taken_literally(self):
        self.assertEqual(self.ids(_inst("catalog", dk_enabled=True),
                                  _inst("finc", base_url="https://f", dk_enabled=True)),
                         ["catalog", "finc"])
        self.assertEqual(self.ids(_inst("catalog", dk_enabled=False)), [])
        self.assertEqual(self.ids(_inst("catalog", dk_enabled=False),
                                  _inst("finc", base_url="https://f", dk_enabled=True)),
                         ["finc"])

    def test_only_the_first_catalog_without_switch(self):
        self.assertEqual(self.ids(_inst("catalog", iid="cat-a"), _inst("catalog", iid="cat-b")),
                         ["cat-a"])

    def test_off_and_types_without_classification_are_not_candidates(self):
        self.assertEqual(self.ids(_inst("finc", enabled=False, base_url="https://f",
                                        dk_enabled=True), _inst("lobid"), _inst("kvk"),
                                  _inst("catalog")), ["catalog"])

    def test_cache_only_counts_as_active(self):
        self.assertEqual(self.ids(_inst("finc", base_url="https://f", dk_enabled=True,
                                        cache_only=True)), ["finc"])

    def test_unreadable_config(self):
        class Broken:
            def enabled_instances_for(self, _c):
                raise RuntimeError("no config")

        self.assertEqual(dk_candidates(Broken()), [])
        self.assertEqual(dk_candidates(None), [])


# ===========================================================================
# B. Characterisation
# ===========================================================================
def _freiberg_config():
    """The plugin list of the UB Freiberg as stored before Nov 9 (values invented)."""
    return _Cfg(
        _inst("lobid", label="Lobid (GND/DNB)"),
        _inst("swb", label="SWB (BSZ)"),
        _inst("catalog", label="Katalog (Libero)", token="T",
              catalog_search_url="https://libero.example/search",
              catalog_web_record_url="https://opac.example/Record"),
        _inst("finc", label="finc (VuFind)", base_url="https://finc.example/proxy",
              dk_enabled=False),
        _inst("sru", label="SRU / MARC-XML", preset="k10plus"),
        _inst("gnd_local", label="Lokale GND-DB"),
        _inst("kvk", label="KVK (Verbundkataloge)"),
    )


class TestFreibergConfiguration(_Quiet):
    def test_yields_the_bare_libero_extractor(self):
        cfg = _freiberg_config()
        ext = resolve_dk_extractor(config=cfg)
        self.assertEqual(type(ext).__name__, "BiblioClient")
        self.assertEqual(ext.source_id, "catalog")
        self.assertEqual(ext.token, "T")
        self.assertEqual([i.instance_id for i in dk_candidates(cfg)], ["catalog"])
        info = describe_dk_source(cfg)
        self.assertEqual(info["text"], "DK-Quelle: Katalog (Libero)")
        self.assertFalse(info["fallback"])

    def test_switching_finc_on_drops_a_never_stored_libero_switch(self):
        cfg = _freiberg_config()
        cfg.plugins[3].settings["dk_enabled"] = True
        self.assertEqual(type(resolve_dk_extractor(config=cfg)).__name__, "FincCatalogClient")

    def test_both_switches_on_chain_in_list_order(self):
        cfg = _freiberg_config()
        cfg.plugins[2].settings["dk_enabled"] = True
        cfg.plugins[3].settings["dk_enabled"] = True
        ext = resolve_dk_extractor(config=cfg)
        self.assertEqual(ext.source_ids, ["catalog", "finc"])
        self.assertEqual(describe_dk_source(cfg)["text"],
                         "DK-Quellen: Katalog (Libero), dann finc (VuFind) (Rückfall)")


# --- the fixed chain as it was before Nov 9, as an oracle -------------------
def _old_pick(cfg):
    """Instance id the fixed chain chose: finc with dk_enabled (available) →
    custom CLASSIFICATION plugin → sru with dk_enabled (available) → catalog;
    ``None`` = the registry default."""
    by_id = {}
    for inst in cfg.enabled_instances_for("search_provider"):
        by_id.setdefault(inst.provider_id, inst)
    finc = by_id.get("finc")
    if finc and finc.settings.get("dk_enabled") and finc.settings.get("base_url"):
        return finc.instance_id
    custom = by_id.get(_CUSTOM)
    if custom:
        return custom.instance_id
    sru = by_id.get("sru")
    if sru and sru.settings.get("dk_enabled") and (sru.settings.get("preset")
                                                    or sru.settings.get("base_url")):
        return sru.instance_id
    cat = by_id.get("catalog")
    return cat.instance_id if cat else None


_CUSTOM = "mylib_dk_chain"


class _Marker:
    def __init__(self, source_id):
        self.source_id = source_id

    def extract_dk_classifications_for_keywords(self, keywords, max_results=50):
        return []


def _register_custom(test):
    @register_provider
    class _Mine:
        id = _CUSTOM
        label = "Eigene DK"
        capabilities = {SearchCapability.CLASSIFICATION}

        def __init__(self, **cfg):
            pass

        def is_available(self, cfg=None):
            return True

        def dk_extractor(self, **_):
            return _Marker(_CUSTOM)

    test.addCleanup(PROVIDER_REGISTRY.pop, _CUSTOM, None)


def _configs():
    """Several hundred plugin lists: order, catalog switch (absent / on / off)
    and state, finc and sru (off / on / on without endpoint), a custom plugin."""
    cat_states = [None, True, False]
    finc_states = ["off", "on", "on_no_url"]
    sru_states = ["off", "on", "on_no_endpoint"]
    for order in itertools.permutations(["catalog", "finc", "sru"]):
        for cat_dk, cat_on, finc, sru, custom in itertools.product(
                cat_states, (True, False), finc_states, sru_states, (False, True)):
            items = []
            for pid in order:
                if pid == "catalog":
                    s = {"token": "T"} if cat_dk is None else {"token": "T", "dk_enabled": cat_dk}
                    items.append(_inst("catalog", enabled=cat_on, label="Katalog (Libero)", **s))
                elif pid == "finc":
                    s = {} if finc == "off" else {"dk_enabled": True}
                    if finc != "on_no_url":
                        s["base_url"] = "https://finc.example/proxy"
                    items.append(_inst("finc", label="finc (VuFind)", **s))
                else:
                    s = {} if sru == "off" else {"dk_enabled": True}
                    if sru != "on_no_endpoint":
                        s["preset"] = "dnb"
                    items.append(_inst("sru", label="SRU", **s))
            if custom:
                items.append(_inst(_CUSTOM, label="Eigene DK"))
            name = f"{order}/cat={cat_dk},{cat_on}/finc={finc}/sru={sru}/custom={custom}"
            yield name, _Cfg(*items), dict(cat_dk=cat_dk, finc=finc, sru=sru, custom=custom)


class TestUnchangedWhereOneSwitchWasOn(_Quiet):
    """For each config whose catalog never stored the switch and that has at
    most one other DK source switched on, the chain is the one source the
    fixed chain chose (or the registry default where it chose that)."""

    def test_against_the_old_chain(self):
        _register_custom(self)
        checked = 0
        for name, cfg, facts in _configs():
            switched = sum([facts["finc"] != "off", facts["sru"] != "off", facts["custom"]])
            if facts["cat_dk"] is not None or switched > 1:
                continue
            ext = resolve_dk_extractor(config=cfg)
            self.assertNotIsInstance(ext, FallbackDkExtractor, name)
            old = _old_pick(cfg)
            if old is None:
                self.assertEqual(type(ext).__name__, "BiblioClient", name)
                self.assertIsNone(describe_dk_source(cfg)["instance_id"], name)
            else:
                self.assertEqual(ext.source_id, old, name)
                if old == "catalog":  # the instance, not the registry default
                    self.assertEqual(ext.token, "T", name)
            checked += 1
        self.assertGreater(checked, 50)


# ===========================================================================
# C. describe_dk_source == resolve_dk_extractor
# ===========================================================================
class TestDescriptionMatchesResolver(_Quiet):
    def test_every_config(self):
        _register_custom(self)
        count = 0
        for name, cfg, _facts in _configs():
            info = describe_dk_source(cfg)
            try:
                ext = resolve_dk_extractor(config=cfg)
            except NoDkSourceError:
                self.assertIsNone(info["provider_id"], name)
                self.assertEqual(info["chain"], [], name)
                count += 1
                continue
            described = [c["instance_id"] for c in info["chain"]]
            if not described:
                # registry default: no active catalog instance
                self.assertEqual(info["provider_id"], "catalog", name)
                self.assertIsNone(info["instance_id"], name)
                self.assertEqual(type(ext).__name__, "BiblioClient", name)
            else:
                self.assertEqual(described, _chain_of(ext), name)
                self.assertEqual(info["fallback"], len(described) > 1, name)
                candidates = [i.instance_id for i in dk_candidates(cfg)]
                # the chain keeps the list order of the candidates
                if all(d in candidates for d in described):
                    self.assertEqual(described, [c for c in candidates if c in described], name)
            count += 1
        self.assertGreater(count, 600)

    def test_texts(self):
        self.assertEqual(describe_dk_source(_Cfg(_inst("catalog", dk_enabled=False)))["text"],
                         "DK-Quelle: keine (bei keiner aktiven Instanz ist der DK-Schalter an)")
        self.assertIn("eingebauten Standardwerten", describe_dk_source(_Cfg())["text"])

    def test_no_source_is_an_initialisation_error_of_the_dk_step(self):
        from src.utils.pipeline_utils import PipelineStepExecutor

        executor = PipelineStepExecutor(alima_manager=None, cache_manager=None, logger=None,
                                        config_manager=None)
        cm = MagicMock()
        cm.return_value.load_config.return_value = _Cfg(_inst("catalog", dk_enabled=False))
        with patch("src.utils.config_manager.ConfigManager", cm):
            result = executor.execute_notation_search(["K (GND-ID: 4000000-0)"],
                                                      rvk_enabled=False)
        self.assertTrue(result["statistics"]["initialization_failed"])
        self.assertIn("Keine DK-Quelle", result["statistics"]["error"])


# ===========================================================================
# D. FallbackDkExtractor
# ===========================================================================
class FakeDk:
    """A DK extractor with scripted outcomes per keyword."""

    def __init__(self, sid, answers=(), *, down=(), miss=(), not_set=(), raises=(),
                 connection=(), cache_only=False, down_from=None):
        self.source_id = sid
        self.cache_only = cache_only
        self.answers = set(answers)
        self.down, self.miss, self.not_set = set(down), set(miss), set(not_set)
        self.raises, self.connection = set(raises), set(connection)
        self.down_from = down_from
        self.calls = []
        self.last_source_down = []
        self.last_cache_misses = []
        self.last_not_configured = []

    def extract_dk_classifications_for_keywords(self, keywords, max_results=50, force_update=False):
        self.calls.append(list(keywords))
        self.last_source_down, self.last_cache_misses, self.last_not_configured = [], [], []
        out = []
        for kw in keywords:
            if kw in self.raises:
                raise ValueError("kaputt")
            if kw == self.down_from:
                for _ in range(source_health.FAILURE_THRESHOLD):
                    source_health.record_failure(
                        self.source_id, requests.exceptions.ConnectionError("x"))
                continue
            if kw in self.down or source_health.is_down(self.source_id):
                self.last_source_down.append(kw)
                continue
            if kw in self.miss:
                self.last_cache_misses.append(kw)
                continue
            if kw in self.not_set:
                self.last_not_configured.append(kw)
                continue
            if kw in self.connection:
                source_health.record_failure(self.source_id,
                                             requests.exceptions.ConnectionError("x"))
                continue
            if kw in self.answers:
                out.append({"keyword": kw, "source": self.source_id,
                            "classifications": [{"dk": f"{self.source_id}-{kw}", "type": "DK",
                                                 "count": 1, "titles": ["T"]}]})
        return out


def _chain(*fakes):
    return FallbackDkExtractor([
        DkCandidate(source_id=f.source_id if f else f"none{i}",
                    label=(f.source_id if f else f"none{i}").upper(), extractor=f,
                    cache_only=bool(f and f.cache_only))
        for i, f in enumerate(fakes)
    ])


def _codes(result):
    return {e["keyword"]: e["classifications"][0]["dk"] for e in result}


class TestFallbackRule(unittest.TestCase):
    def setUp(self):
        notice = patch.object(source_health, "_emit_notice", lambda _t: None)
        notice.start()
        self.addCleanup(notice.stop)

    def test_first_source_answers_everything_second_is_not_asked(self):
        a, b = FakeDk("a", {"k1", "k2"}), FakeDk("b", {"k1", "k2"})
        out = _chain(a, b).extract_dk_classifications_for_keywords(["k1", "k2"])
        self.assertEqual(_codes(out), {"k1": "a-k1", "k2": "a-k2"})
        self.assertEqual(b.calls, [])

    def test_no_hits_is_an_answer(self):
        a, b = FakeDk("a", {"k1"}), FakeDk("b", {"k1", "k2"})
        ext = _chain(a, b)
        out = ext.extract_dk_classifications_for_keywords(["k1", "k2"])
        self.assertEqual(_codes(out), {"k1": "a-k1"})
        self.assertEqual(b.calls, [])
        self.assertEqual(ext.answered_by, {"k1": "a", "k2": "a"})
        self.assertEqual(ext.fallback_notes(), [])

    def test_reasons_that_hand_on(self):
        cases = {
            "down": dict(down={"k"}),
            "cache_miss": dict(miss={"k"}, cache_only=True),
            "not_set": dict(not_set={"k"}),
            "raises": dict(raises={"k"}),
            "connection": dict(connection={"k"}),
        }
        for name, kwargs in cases.items():
            source_health.begin_run()
            a, b = FakeDk("a", **kwargs), FakeDk("b", {"k"})
            ext = _chain(a, b)
            out = ext.extract_dk_classifications_for_keywords(["k"])
            self.assertEqual(_codes(out), {"k": "b-k"}, name)
            self.assertEqual(ext.answered_by, {"k": "b"}, name)

    def test_a_cache_miss_outside_cache_only_mode_is_no_hit(self):
        a, b = FakeDk("a", miss={"k"}, cache_only=False), FakeDk("b", {"k"})
        self.assertEqual(_codes(_chain(a, b).extract_dk_classifications_for_keywords(["k"])), {})

    def test_extractor_that_could_not_be_built(self):
        b = FakeDk("b", {"k1", "k2"})
        ext = _chain(None, b)
        self.assertEqual(_codes(ext.extract_dk_classifications_for_keywords(["k1", "k2"])),
                         {"k1": "b-k1", "k2": "b-k2"})
        self.assertEqual(ext.handoffs()[0]["reasons"], {"init_failed": 2})

    def test_open_circuit_breaker_hands_on(self):
        a, b = FakeDk("a", {"k"}), FakeDk("b", {"k"})
        a.get_circuit_breaker_status = lambda: {"open": True}
        self.assertEqual(_codes(_chain(a, b).extract_dk_classifications_for_keywords(["k"])),
                         {"k": "b-k"})
        self.assertEqual(a.calls, [])

    def test_open_keywords_go_on_collected_and_one_note_per_hand_over(self):
        kws = [f"k{i}" for i in range(6)]
        a = FakeDk("a", {"k0"}, down={"k1", "k2", "k3"})
        b = FakeDk("b", {"k1", "k2"}, miss={"k3"}, cache_only=True)
        c = FakeDk("c", {"k3"})
        ext = _chain(a, b, c)
        out = []
        for kw in kws:  # the DK step asks one keyword per call
            out += ext.extract_dk_classifications_for_keywords([kw])
        self.assertEqual(_codes(out), {"k0": "a-k0", "k1": "b-k1", "k2": "b-k2", "k3": "c-k3"})
        self.assertEqual(ext.answered_by, {"k0": "a", "k1": "b", "k2": "b", "k3": "c",
                                           "k4": "a", "k5": "a"})
        self.assertEqual(ext.handoffs(), [
            {"from": "a", "to": "b", "keywords": 3, "reasons": {"source_down": 3}},
            {"from": "b", "to": "c", "keywords": 1, "reasons": {"cache_miss": 1}},
        ])
        self.assertEqual(ext.fallback_notes(), [
            "A: 3 Schlagwörter nicht beantwortet (gesperrt), weiter mit B",
            "B: 1 Schlagwort nicht beantwortet (nicht im Cache), weiter mit C",
        ])

    def test_batch_call_hands_the_open_keywords_on_in_order(self):
        a = FakeDk("a", {"k2"}, down={"k1", "k3"})
        b = FakeDk("b", {"k1", "k2", "k3"})
        out = _chain(a, b).extract_dk_classifications_for_keywords(["k1", "k2", "k3"])
        self.assertEqual([e["keyword"] for e in out], ["k1", "k2", "k3"])
        self.assertEqual(_codes(out)["k2"], "a-k2")  # not merged with b
        self.assertEqual(b.calls, [["k1"], ["k3"]])

    def test_source_going_down_mid_run(self):
        a = FakeDk("a", {"k1", "k2", "k3", "k4"}, down_from="k2")
        b = FakeDk("b", {"k1", "k2", "k3", "k4"})
        ext = _chain(a, b)
        out = []
        for kw in ["k1", "k2", "k3", "k4"]:
            out += ext.extract_dk_classifications_for_keywords([kw])
        self.assertEqual(_codes(out), {"k1": "a-k1", "k2": "b-k2", "k3": "b-k3", "k4": "b-k4"})
        self.assertEqual(ext.handoffs()[0]["reasons"], {"connection": 1, "source_down": 2})
        self.assertEqual(len(ext.fallback_notes()), 1)

    def test_keywords_no_source_answered_carry_the_last_reason(self):
        a = FakeDk("a", down={"k1", "k2", "k3"})
        b = FakeDk("b", down={"k1"}, miss={"k2"}, not_set={"k3"}, cache_only=True)
        ext = _chain(a, b)
        self.assertEqual(ext.extract_dk_classifications_for_keywords(["k1", "k2", "k3"]), [])
        self.assertEqual(ext.last_source_down, ["k1"])
        self.assertEqual(ext.last_cache_misses, ["k2"])
        self.assertEqual(ext.last_not_configured, ["k3"])
        self.assertEqual(ext.answered_by, {})

    def test_identity(self):
        ext = _chain(FakeDk("a", cache_only=True), FakeDk("b", cache_only=False))
        self.assertEqual(ext.source_id, "a")
        self.assertFalse(ext.cache_only)
        self.assertTrue(_chain(FakeDk("a", cache_only=True),
                               FakeDk("b", cache_only=True)).cache_only)
        with self.assertRaises(ValueError):
            FallbackDkExtractor([DkCandidate("a", "A", extractor=FakeDk("a"))])


class TestResolverBuildsTheChain(_Quiet):
    def test_one_source_is_its_own_extractor(self):
        for inst, name in ((_inst("finc", base_url="https://f", dk_enabled=True),
                            "FincCatalogClient"),
                           (_inst("sru", preset="dnb", dk_enabled=True), "MarcXmlClient")):
            ext = resolve_dk_extractor(config=_Cfg(_inst("catalog"), inst))
            self.assertEqual(type(ext).__name__, name)

    def test_unavailable_source_is_left_out(self):
        ext = resolve_dk_extractor(config=_Cfg(
            _inst("finc", dk_enabled=True), _inst("sru", preset="dnb", dk_enabled=True)))
        self.assertEqual(type(ext).__name__, "MarcXmlClient")

    def test_extractor_build_failure_keeps_the_source_in_the_chain(self):
        from src.core.search.providers.finc.provider import FincProvider

        cfg = _Cfg(_inst("finc", base_url="https://f", dk_enabled=True),
                   _inst("sru", preset="dnb", dk_enabled=True))
        with patch.object(FincProvider, "dk_extractor", side_effect=RuntimeError("nope")):
            ext = resolve_dk_extractor(config=cfg)
        self.assertEqual(ext.source_ids, ["finc", "sru"])
        self.assertIsNone(ext.candidates[0].extractor)
        self.assertIn("nope", ext.candidates[0].init_error)

    def test_single_source_build_failure_raises_as_before(self):
        from src.core.search.providers.finc.provider import FincProvider

        cfg = _Cfg(_inst("finc", base_url="https://f", dk_enabled=True))
        with patch.object(FincProvider, "dk_extractor", side_effect=RuntimeError("nope")):
            with self.assertRaises(RuntimeError):
                resolve_dk_extractor(config=cfg)


# ===========================================================================
# E. The DK step reports the chain
# ===========================================================================
class TestDkStepReport(unittest.TestCase):
    KEYWORDS = [f"K{i} (GND-ID: 400000{i}-{i})" for i in range(5)]

    def setUp(self):
        notice = patch.object(source_health, "_emit_notice", lambda _t: None)
        notice.start()
        self.addCleanup(notice.stop)

    def _run(self, ext):
        from src.utils.pipeline_utils import PipelineStepExecutor

        lines = []
        executor = PipelineStepExecutor(alima_manager=None, cache_manager=None, logger=None,
                                        config_manager=None)
        with patch("src.core.search.factory.resolve_dk_extractor", return_value=ext), \
                patch("src.utils.config_manager.ConfigManager", MagicMock()):
            result = executor.execute_notation_search(
                list(self.KEYWORDS), stream_callback=lambda m, _s: lines.append(m),
                rvk_enabled=False)
        return lines, result

    def test_statistics_and_one_line_per_hand_over(self):
        a = FakeDk("catalog", {"K0"}, down={"K1", "K2", "K3"})
        b = FakeDk("finc", {"K1", "K2"}, miss={"K3"}, cache_only=True)
        ext = FallbackDkExtractor([
            DkCandidate("catalog", "Katalog (Libero)", "catalog", extractor=a),
            DkCandidate("finc", "finc (VuFind)", "finc", extractor=b, cache_only=True),
        ])
        lines, result = self._run(ext)
        stats = result["statistics"]
        self.assertEqual(stats["dk_sources"], ["catalog", "finc"])
        self.assertEqual(stats["dk_source_by_keyword"],
                         {"K0": "catalog", "K1": "finc", "K2": "finc", "K4": "catalog"})
        self.assertEqual(stats["dk_fallbacks"],
                         [{"from": "catalog", "to": "finc", "keywords": 3,
                           "reasons": {"source_down": 3}}])
        notes = [m for m in lines if m.startswith("↪")]
        self.assertEqual(notes, ["↪ Katalog (Libero): 3 Schlagwörter nicht beantwortet "
                                 "(gesperrt), weiter mit finc (VuFind)\n"])
        self.assertEqual(stats["cache_only_misses"], ["K3"])
        self.assertIn("DK-Quellen: Katalog (Libero), dann finc (VuFind) (Rückfall)\n", lines)
        self.assertTrue(any("K1: 1 Klassifikationen gefunden (finc (VuFind))" in m for m in lines))
        self.assertTrue(any(m.startswith("ℹ️ finc (VuFind): 1 von 5") for m in lines), lines)

    def test_single_extractor_report_unchanged(self):
        a = FakeDk("catalog", {"K0"})
        _lines, result = self._run(a)
        for key in ("dk_sources", "dk_source_by_keyword", "dk_fallbacks"):
            self.assertNotIn(key, result["statistics"])


# ===========================================================================
# F. Order of new installations, existing order kept
# ===========================================================================
class TestOrder(_Quiet):
    def test_new_installation(self):
        from src.utils.plugin_migration import synthesize_search_instances

        instances = synthesize_search_instances({})
        self.assertEqual([i.instance_id for i in instances],
                         ["lobid", "swb", "finc", "catalog", "sru", "gnd_local", "kvk"])
        cfg = _Cfg(*instances)
        self.assertEqual([i.instance_id for i in dk_candidates(cfg)], ["catalog"])
        self.assertEqual(type(resolve_dk_extractor(config=cfg)).__name__, "BiblioClient")
        self.assertEqual(describe_dk_source(cfg)["text"], "DK-Quelle: Katalog (Libero)")

    def test_existing_config_keeps_its_order_and_settings(self):
        from src.utils.config_manager import ConfigManager

        plugins = [
            {"instance_id": pid, "category": "search_provider", "provider_id": pid,
             "label": label, "enabled": True, "usage_hint": "", "settings": settings}
            for pid, label, settings in (
                ("lobid", "Lobid (GND/DNB)", {}),
                ("swb", "SWB (BSZ)", {}),
                ("catalog", "Katalog (Libero)", {"token": "T"}),
                ("finc", "finc (VuFind)", {"base_url": "https://finc.example/proxy",
                                           "dk_enabled": False}),
                ("sru", "SRU / MARC-XML", {"preset": "k10plus"}),
                ("gnd_local", "Lokale GND-DB", {}),
                ("kvk", "KVK (Verbundkataloge)", {}),
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "unified_config": {"providers": [], "gemini_api_key": "K"},
                "plugins": plugins,
            }))
            ConfigManager.reset()
            try:
                cm = ConfigManager()
                cm.config_file = path
                loaded = cm.load_config()
                ext = resolve_dk_extractor(config=loaded)
                cm.save_config(loaded)
                first = path.read_text()
                ConfigManager.reset()
                cm = ConfigManager()
                cm.config_file = path
                cm.save_config(cm.load_config())
                second = path.read_text()
            finally:
                ConfigManager.reset()
        saved = [p for p in json.loads(first)["plugins"] if p["category"] == "search_provider"]
        self.assertEqual(saved, plugins)
        self.assertEqual(first, second)
        self.assertEqual(type(ext).__name__, "BiblioClient")


# ===========================================================================
# G. The plugin tab: catalog switch, line above the list
# ===========================================================================
_KEEP_ALIVE: list = []


class TestPluginTab(_Quiet):
    def _panel(self, instances):
        from PyQt6.QtWidgets import QApplication

        from src.ui.plugin_settings_tab import _CategoryPanel, _ensure_categories

        app = QApplication.instance() or QApplication(["alima-tests"])
        _KEEP_ALIVE.append(app)
        if not isinstance(app, QApplication):
            self.skipTest("a bare QCoreApplication exists; widgets cannot be built")
        _ensure_categories()
        panel = _CategoryPanel("search_provider")
        panel.load(instances)
        return panel

    def test_catalog_has_the_switch_and_an_untouched_form_writes_no_key(self):
        from src.core.search.factory import SearchProviderCategory

        keys = [f.key for f in SearchProviderCategory().type_meta("catalog").config_fields]
        self.assertIn("dk_enabled", keys)
        panel = self._panel([_inst("catalog", token="t"), _inst("sru", preset="dnb")])
        for row in (0, 1, 0):
            panel.list.setCurrentRow(row)
        for out in panel.collect():
            self.assertNotIn("dk_enabled", out.settings, out.instance_id)

    def test_checking_the_switch_stores_it(self):
        # The form shows the effective state of an unstored switch (Nov 2026):
        # with finc as DK source Libero is not one, so the box starts empty.
        # - Claude Generated
        panel = self._panel([_inst("catalog", token="t"),
                             _inst("finc", base_url="https://f", dk_enabled=True)])
        panel.list.setCurrentRow(0)
        self.assertFalse(panel._field_widgets["dk_enabled"].isChecked())
        panel._field_widgets["dk_enabled"].setChecked(True)
        self.assertIs(panel.collect()[0].settings["dk_enabled"], True)

    def test_unchecking_the_effective_switch_stores_false(self):
        panel = self._panel([_inst("catalog", token="t")])
        panel.list.setCurrentRow(0)
        self.assertTrue(panel._field_widgets["dk_enabled"].isChecked())
        panel._field_widgets["dk_enabled"].setChecked(False)
        self.assertIs(panel.collect()[0].settings["dk_enabled"], False)

    def test_chain_line(self):
        from src.ui.plugin_settings_tab import dk_source_text

        text = dk_source_text([
            _inst("catalog", label="Katalog (Libero)", dk_enabled=True),
            _inst("finc", label="finc (VuFind)", base_url="https://f", dk_enabled=True,
                  cache_only=True)])
        self.assertEqual(text, "DK-Quellen im Moment: Katalog (Libero) [catalog]; dann "
                               "finc (VuFind) [finc], nur Cache (Rückfall je Schlagwort)")
        self.assertEqual(dk_source_text([_inst("catalog", label="Katalog (Libero)")]),
                         "DK-Quelle im Moment: Katalog (Libero) [catalog]")
        self.assertEqual(dk_source_text([_inst("catalog", dk_enabled=False)]),
                         "DK-Quelle im Moment: keine (bei keiner aktiven Instanz ist der "
                         "DK-Schalter an)")


if __name__ == "__main__":
    unittest.main()
