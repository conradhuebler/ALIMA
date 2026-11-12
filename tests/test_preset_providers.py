"""Presets carry LLM provider access: blocks ``providers`` and ``default_models`` - Claude Generated.

Pinned here (Nov 12, 2026):

1. Reading: ``api_key`` is dropped everywhere in a preset except directly in
   an entry of the top-level list ``providers``; ``database.password`` keeps its
   own rule.
2. Plan: every provider entry is new, replaced or unchanged, with "Schlüssel
   wird gesetzt"; a rejected entry is a hint and the rest stays applicable;
   default models overwrite unless opted out (``build_plan``/``apply_plan``,
   CLI ``--no-default-models``, check box of the preview).
3. Apply: same name (case-insensitive) is replaced at its place, a configured
   key stays when the package has none; a second application is empty.
4. Revert: previous entries, keys and defaults come back, also to an empty
   provider list; a provider changed since stays.
5. Export: every enabled provider without key; a key only by name
   (``--with-provider-key``), then 0600 and a warning in ``_comment``.
6. The key appears in no plan text, log record, CLI output or exception text.
7. Wizard pre-fill from ``providers``/``default_models``; the old block ``llm``
   is ignored with a hint.

All data is invented; HOME and the config directory are temporary; no network.
"""

from __future__ import annotations

import inspect
import io
import json
import logging
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils import preset_apply as pa
from src.utils.config_models import PROVIDER_TYPES, UnifiedProvider
from tests.test_institution_presets import _Home, _KEEP, _need_widgets

KEY = "sk-TESTKEY-0123456789abcdef"      # invented; must never show up in output
OTHER_KEY = "sk-OTHERKEY-fedcba9876543210"
OLD_KEY = "sk-OLDKEY-1111111111111111"


def _no_key(case: unittest.TestCase, text: str, where: str) -> None:
    """Assert without echoing ``text``: a failing assertion must not print a key."""
    for key in (KEY, OTHER_KEY, OLD_KEY):
        case.assertFalse(key in text, f"Schlüssel in {where}")


def provider_entry(**kw):
    entry = {"name": "ub-ollama", "provider_type": "ollama", "enabled": True,
             "preferred_model": "m-pipe", "host": "llm.ub.example", "port": 11434,
             "use_ssl": False, "connection_type": "native_client"}
    entry.update(kw)
    return entry


class _ProviderHome(_Home):
    def write_preset(self, data, name="pkg") -> Path:
        folder = self.tmp / name
        folder.mkdir(parents=True, exist_ok=True)
        base = {"format": 2, "preset": {"id": "llm-test", "version": "1", "label": "LLM-Test"}}
        base.update(data)
        (folder / pa.PRESET_FILENAME).write_text(json.dumps(base, indent=2), encoding="utf-8")
        return folder

    def set_providers(self, providers, **defaults):
        cfg = self.config()
        cfg.unified_config.providers = list(providers)
        for key, value in defaults.items():
            setattr(cfg.unified_config, key, value)
        self.assertTrue(self.cm.save_config(cfg, preserve_unified=False))

    def providers(self, config=None):
        cfg = config or self.config()
        return [(p.name, p.provider_type, p.base_url, p.host, p.preferred_model, p.enabled,
                 p.api_key) for p in cfg.unified_config.providers]

    def defaults(self, config=None):
        u = (config or self.config()).unified_config
        return {scope: (getattr(u, pf), getattr(u, mf))
                for scope, (pf, mf) in pa.DEFAULT_MODEL_FIELDS.items()}

    def disk_unified(self):
        return json.loads(Path(self.cm.config_file).read_text(encoding="utf-8"))["unified_config"]


def existing_providers():
    return [
        UnifiedProvider(name="lokal", provider_type="ollama", host="localhost", port=11434,
                        base_url="http://localhost:11434", preferred_model="m-lokal"),
        UnifiedProvider(name="UB-Ollama", provider_type="ollama", host="alt.ub.example",
                        port=11434, preferred_model="m-alt", api_key=OLD_KEY,
                        description="alt"),
        UnifiedProvider(name="chat", provider_type="openai_compatible",
                        base_url="https://chat.example/v1", preferred_model="c1",
                        api_key=OTHER_KEY),
    ]


# ===========================================================================
# 1. Reading: one exception to "api_key is dropped"
# ===========================================================================
class TestReadingKeys(_ProviderHome):
    def test_only_providers_api_key_is_kept(self):
        folder = self.write_preset({
            "api_key": "TOP",
            "llm": {"api_key": "LLM"},
            "catalog": {"token": "TOK"},
            "database": {"type": "mariadb", "host": "db.example", "password": "DB-PW"},
            "plugins": [{"instance": "swb", "settings": {"api_key": "PLUG", "x_token": "T"}}],
            "providers": [provider_entry(api_key=KEY, token="PT", gemini_api_key="G",
                                         extra={"api_key": "NESTED"})],
            "default_models": {"general": {"provider": "ub-ollama", "api_key": "DM"}},
        })
        preset = pa.load_preset(folder)
        self.assertEqual(preset.data["providers"][0]["api_key"], KEY)
        self.assertEqual(preset.data["database"]["password"], "DB-PW")
        self.assertEqual(sorted(preset.ignored_keys), sorted([
            "api_key", "llm.api_key", "catalog.token", "plugins[1].settings.api_key",
            "plugins[1].settings.x_token", "providers[1].token", "providers[1].gemini_api_key",
            "providers[1].extra.api_key", "default_models.general.api_key"]))
        self.assertNotIn("providers[1].api_key", preset.ignored_keys)
        for key in ("token", "gemini_api_key"):
            self.assertNotIn(key, preset.data["providers"][0])
        self.assertNotIn("api_key", preset.data["providers"][0]["extra"])

    def test_api_key_in_a_nested_providers_list_is_dropped(self):
        folder = self.write_preset({"x": {"providers": [{"api_key": "NESTED"}]}})
        preset = pa.load_preset(folder)
        self.assertEqual(preset.ignored_keys, ["x.providers[1].api_key"])


# ===========================================================================
# 2. Plan
# ===========================================================================
class TestProviderPlan(_ProviderHome):
    def test_new_provider_with_key(self):
        before = self.disk()
        plan = self.plan(self.write_preset({"providers": [provider_entry(api_key=KEY)]}))
        self.assertEqual(plan.errors, [])
        self.assertEqual([(p.name, p.status, p.key) for p in plan.providers],
                         [("ub-ollama", pa.PROVIDER_NEW, pa.KEY_SET)])
        text = "\n".join(plan.lines())
        self.assertIn("neu: ub-ollama (ollama, http://llm.ub.example:11434); "
                      "Schlüssel wird gesetzt", text)
        self.assertTrue(plan.has_provider_keys)
        _no_key(self, text, "Vorschau")
        _no_key(self, json.dumps(plan.as_dict()), "Plan.as_dict")
        _no_key(self, repr(plan), "repr(Plan)")
        self.assertEqual(self.disk(), before)

    def test_replaced_and_unchanged_name_case_insensitive(self):
        self.set_providers(existing_providers())
        same = {"name": "lokal", "provider_type": "ollama", "base_url": "http://localhost:11434",
                "preferred_model": "m-lokal", "host": "localhost"}
        plan = self.plan(self.write_preset({"providers": [
            same, provider_entry(name="ub-ollama"),
            {"name": "chat", "provider_type": "openai_compatible",
             "base_url": "https://chat.example/v1", "preferred_model": "c1"}]}))
        status = {p.name: (p.status, p.key) for p in plan.providers}
        self.assertEqual(status, {"lokal": (pa.PROVIDER_UNCHANGED, pa.KEY_NONE),
                                  "ub-ollama": (pa.PROVIDER_REPLACED, pa.KEY_KEPT),
                                  "chat": (pa.PROVIDER_UNCHANGED, pa.KEY_KEPT)})
        text = "\n".join(plan.lines())
        self.assertIn("ersetzt: ub-ollama", text)
        self.assertIn("host: alt.ub.example → llm.ub.example", text)
        self.assertIn("vorhandener Schlüssel bleibt", text)
        self.assertIn("unverändert: lokal", text)
        _no_key(self, text, "Vorschau")

    def test_rejected_entries_are_hints_and_the_rest_applies(self):
        bad = [
            {"name": "x", "provider_type": "llamafile"},
            provider_entry(name="a/b"),
            provider_entry(name="b;rm -rf"),
            {"name": "f", "provider_type": "openai_compatible", "base_url": "file:///etc/passwd"},
            {"name": "u", "provider_type": "openai_compatible",
             "base_url": "https://user:geheim@llm.example/v1"},
            provider_entry(name="h", host="http://llm.example/x"),
            provider_entry(name="p", port=70000),
            provider_entry(name="m", preferred_model="m; rm"),
            provider_entry(name="k", api_key="zwei teile"),
            {"provider_type": "ollama"},
            "kein Objekt",
        ]
        folder = self.write_preset({"institution": {"isil": "ZZ-P"},
                                    "providers": bad + [provider_entry(), provider_entry(
                                        name="UB-OLLAMA")]})
        plan = self.plan(folder)
        self.assertEqual(plan.errors, [])
        statuses = [p.status for p in plan.providers]
        self.assertEqual(statuses, [pa.PROVIDER_REJECTED] * len(bad)
                         + [pa.PROVIDER_NEW, pa.PROVIDER_REJECTED])  # duplicate name
        text = "\n".join(plan.lines())
        for reason in ("Typ 'llamafile' ist unbekannt", "'name' ist ungültig",
                       "nur eine http- oder https-Adresse", "Zugangsdaten",
                       "'host': nur der Rechnername",
                       "'port'", "ungültiger Modellname", "'api_key' enthält Leer",
                       "'name' fehlt", "ein Objekt erwartet", "doppelt"):
            self.assertIn(reason, text)
        self.assertNotIn("geheim", text)  # the password in the URL is not echoed
        pa.apply_plan(plan, self.cm)
        cfg = self.config()
        self.assertEqual([p.name for p in cfg.unified_config.providers], ["ub-ollama"])
        self.assertEqual(cfg.system_config.institution_isil, "ZZ-P")

    def test_ollama_fields_on_another_type_are_skipped(self):
        plan = self.plan(self.write_preset({"providers": [
            {"name": "c", "provider_type": "openai_compatible",
             "base_url": "https://c.example/v1", "host": "h", "id": "x"}]}))
        self.assertTrue(any("'host' gilt nur für den Typ ollama" in w for w in plan.warnings))
        self.assertTrue(any("unbekannter Schlüssel 'id'" in w for w in plan.warnings))
        pa.apply_plan(plan, self.cm)
        provider = self.config().unified_config.providers[0]
        self.assertEqual((provider.host, provider.base_url), ("", "https://c.example/v1"))

    def test_every_provider_type_is_accepted(self):
        entries = [{"name": f"p-{t}", "provider_type": t} for t in PROVIDER_TYPES]
        plan = self.plan(self.write_preset({"providers": entries}))
        self.assertEqual([p.status for p in plan.providers], [pa.PROVIDER_NEW] * len(PROVIDER_TYPES))

    def test_provider_types_match_the_llm_service_dispatch(self):
        from src.llm.llm_service import LlmService

        source = inspect.getsource(LlmService._init_unified_provider_configs)
        for ptype in PROVIDER_TYPES:
            self.assertIn(f'provider.provider_type == "{ptype}"', source)


# ===========================================================================
# 3. Default models
# ===========================================================================
class TestDefaultModels(_ProviderHome):
    PRESET = {
        "providers": [provider_entry(), {"name": "chat", "provider_type": "openai_compatible",
                                         "base_url": "https://chat.example/v1",
                                         "preferred_model": "c-neu"}],
        "default_models": {"general": {"provider": "CHAT"},
                           "pipeline": {"provider": "ub-ollama", "model": "m-pipe"},
                           "vision": {"provider": "ub-ollama", "model": "m-vis"},
                           "decision": {"provider": "ub-ollama", "model": "d1"}},
    }

    def setUp(self):
        super().setUp()
        self.set_providers(existing_providers(), preferred_provider="lokal",
                           preferred_model="m-lokal", pipeline_default_provider="lokal",
                           pipeline_default_model="m-lokal")

    def test_overwrite_by_default_with_before_and_after(self):
        plan = self.plan(self.write_preset(self.PRESET))
        text = "\n".join(plan.lines())
        self.assertIn("Standardmodelle (4, werden überschrieben; abwählbar):", text)
        self.assertIn("Standardmodell Pipeline: lokal / m-lokal → ub-ollama / m-pipe", text)
        self.assertIn("Standardmodell Allgemein (Agentik, Chat): lokal / m-lokal → chat / c-neu",
                      text)
        pa.apply_plan(plan, self.cm)
        self.assertEqual(self.defaults(), {"general": ("chat", "c-neu"),
                                           "pipeline": ("ub-ollama", "m-pipe"),
                                           "vision": ("ub-ollama", "m-vis"),
                                           "decision": ("ub-ollama", "d1")})

    def test_opt_out_in_build_plan(self):
        before = self.defaults()
        plan = pa.build_plan(pa.load_preset(self.write_preset(self.PRESET)), self.cm,
                             default_models=False)
        self.assertIn("Standardmodelle (4, abgewählt: bleiben, wie sie sind):",
                      "\n".join(plan.lines()))
        entry = pa.apply_plan(plan, self.cm)
        self.assertEqual(self.defaults(), before)
        self.assertIn("ub-ollama", [p.name for p in self.config().unified_config.providers])
        self.assertFalse(any(c["path"][0] == pa.DEFAULT_MODEL_PATH for c in entry["changes"]))

    def test_opt_out_in_apply_plan(self):
        before = self.defaults()
        plan = self.plan(self.write_preset(self.PRESET))
        pa.apply_plan(plan, self.cm, default_models=False)
        self.assertEqual(self.defaults(), before)
        self.assertFalse(any(s["path"][0] == pa.DEFAULT_MODEL_PATH for s in plan.steps()))

    def test_only_default_models_opted_out_is_an_empty_plan(self):
        preset = {"default_models": {"pipeline": {"provider": "chat", "model": "c1"}}}
        plan = pa.build_plan(pa.load_preset(self.write_preset(preset)), self.cm,
                             default_models=False)
        self.assertTrue(plan.is_empty)
        self.assertIsNone(pa.apply_plan(plan, self.cm))

    def test_unknown_provider_scope_and_decision_without_model_are_hints(self):
        plan = self.plan(self.write_preset({"default_models": {
            "general": {"provider": "gibt-es-nicht"}, "chat": {"provider": "lokal"},
            "decision": {"provider": "lokal"}}}))
        self.assertEqual(plan.changes, [])
        text = "\n".join(plan.warnings)
        self.assertIn("Provider 'gibt-es-nicht' gibt es nach dem Preset nicht", text)
        self.assertIn("unbekannter Bereich 'chat'", text)
        self.assertIn("Entscheidung: 'model' fehlt", text)


# ===========================================================================
# 4. Apply, second application, revert
# ===========================================================================
class TestApplyAndRevert(_ProviderHome):
    PRESET = {
        "providers": [provider_entry(name="ub-ollama", api_key=KEY),
                      {"name": "chat", "provider_type": "openai_compatible",
                       "base_url": "https://chat-neu.example/v1", "preferred_model": "c2"},
                      provider_entry(name="neu", host="neu.example", preferred_model="n1")],
        "default_models": {"general": {"provider": "neu", "model": "n1"}},
    }

    def test_apply_on_a_fresh_configuration(self):
        self.assertEqual(self.config().unified_config.providers, [])
        plan = self.plan(self.write_preset(self.PRESET))
        pa.apply_plan(plan, self.cm)
        from src.utils.config_manager import ConfigManager

        ConfigManager.reset()
        self.cm = ConfigManager()
        self.assertEqual(self.providers(), [
            ("ub-ollama", "ollama", "", "llm.ub.example", "m-pipe", True, KEY),
            ("chat", "openai_compatible", "https://chat-neu.example/v1", "", "c2", True, ""),
            ("neu", "ollama", "", "neu.example", "n1", True, "")])
        self.assertEqual(self.defaults()["general"], ("neu", "n1"))
        self.assertEqual(self.config().unified_config.resolve_default_provider_model(), ("neu", "n1"))

    def test_apply_on_an_existing_configuration_replaces_in_place(self):
        self.set_providers(existing_providers())
        ids = {p.name: p.id for p in self.config().unified_config.providers}
        pa.apply_plan(self.plan(self.write_preset(self.PRESET)), self.cm)
        cfg = self.config()
        self.assertEqual(self.providers(cfg), [
            ("lokal", "ollama", "http://localhost:11434", "localhost", "m-lokal", True, ""),
            ("ub-ollama", "ollama", "", "llm.ub.example", "m-pipe", True, KEY),
            ("chat", "openai_compatible", "https://chat-neu.example/v1", "", "c2", True,
             OTHER_KEY),  # no key in the package: the configured one stays
            ("neu", "ollama", "", "neu.example", "n1", True, "")])
        by_name = {p.name: p for p in cfg.unified_config.providers}
        self.assertEqual(by_name["chat"].id, ids["chat"])
        self.assertEqual(by_name["ub-ollama"].id, ids["UB-Ollama"])
        self.assertEqual(by_name["ub-ollama"].description, "")  # replaced, not merged

    def test_second_application_is_empty_and_writes_nothing(self):
        self.set_providers(existing_providers())
        folder = self.write_preset(self.PRESET)
        pa.apply_plan(self.plan(folder), self.cm)
        after = self.disk()
        again = self.plan(folder)
        self.assertTrue(again.is_empty, again.lines())
        self.assertEqual({p.status for p in again.providers}, {pa.PROVIDER_UNCHANGED})
        self.assertEqual(next(p.key for p in again.providers if p.name == "ub-ollama"),
                         pa.KEY_SAME)
        self.assertIsNone(pa.apply_plan(again, self.cm))
        self.assertEqual(self.disk(), after)

    def _state(self):
        cfg = self.config()
        return ([(p.name, p.provider_type, p.base_url, p.host, p.port, p.use_ssl,
                  p.connection_type, p.preferred_model, p.description, p.enabled, p.api_key,
                  p.id) for p in cfg.unified_config.providers], self.defaults(cfg))

    def test_revert_restores_entries_keys_and_defaults(self):
        self.set_providers(existing_providers(), preferred_provider="lokal",
                           preferred_model="m-lokal")
        before = self._state()
        pa.apply_plan(self.plan(self.write_preset(self.PRESET)), self.cm)
        self.assertNotEqual(self._state(), before)
        result = pa.revert_last(self.cm)
        self.assertEqual(self._state(), before)
        self.assertEqual(result.kept, [])
        _no_key(self, "\n".join(result.lines()), "Rückgängig-Zeilen")

    def test_revert_on_a_fresh_configuration_reaches_the_file(self):
        pa.apply_plan(self.plan(self.write_preset(self.PRESET)), self.cm)
        pa.revert_last(self.cm)
        disk = self.disk_unified()
        self.assertEqual(disk["providers"], [])
        self.assertEqual((disk["preferred_provider"], disk["preferred_model"]), ("", ""))

    def test_a_provider_changed_since_stays(self):
        pa.apply_plan(self.plan(self.write_preset(self.PRESET)), self.cm)
        cfg = self.config()
        by_name = {p.name: p for p in cfg.unified_config.providers}
        by_name["neu"].api_key = "eigener"   # the user entered a key afterwards
        by_name["chat"].base_url = "https://anders.example/v1"
        by_name["ub-ollama"].api_key = ""    # ... or removed the one of the package
        self.cm.save_config(cfg)
        result = pa.revert_last(self.cm)
        names = [p.name for p in self.config().unified_config.providers]
        self.assertEqual(names, ["ub-ollama", "chat", "neu"])
        for name in ("neu", "ub-ollama"):
            self.assertTrue(any(f"Provider {name}: seit dem Anwenden geändert, der Provider bleibt"
                                == line for line in result.kept), result.kept)

    def test_revert_of_a_gemini_provider_does_not_come_back_on_load(self):
        folder = self.write_preset({"providers": [
            {"name": "gemini", "provider_type": "gemini", "preferred_model": "g1",
             "api_key": KEY}]})
        pa.apply_plan(self.plan(folder), self.cm)
        self.assertEqual(self.disk_unified()["gemini_api_key"], KEY)  # legacy mirror
        pa.revert_last(self.cm)
        from src.utils.config_manager import ConfigManager

        ConfigManager.reset()
        self.cm = ConfigManager()
        self.assertEqual(self.config().unified_config.providers, [])
        self.assertEqual(self.disk_unified()["gemini_api_key"], "")


# ===========================================================================
# 5. Export
# ===========================================================================
class TestProviderExport(_ProviderHome):
    def setUp(self):
        super().setUp()
        providers = existing_providers()
        providers.append(UnifiedProvider(name="aus", provider_type="anthropic", enabled=False,
                                         api_key="sk-AUS"))
        self.set_providers(providers, preferred_provider="chat", preferred_model="c1",
                           vision_default_provider="aus", vision_default_model="x",
                           pipeline_default_provider="UB-Ollama",
                           pipeline_default_model="m-alt")

    def export(self, name="pkg", **kw):
        from src.utils.preset_export import export_preset

        return export_preset(self.cm, self.tmp / "out" / name, preset_id="x", **kw)

    def files_text(self, result) -> bytes:
        return b"".join(f.read_bytes() for f in result.path.rglob("*") if f.is_file())

    def test_enabled_providers_without_key_by_default(self):
        result = self.export()
        data = json.loads(result.preset_file.read_text(encoding="utf-8"))
        self.assertEqual([p["name"] for p in data["providers"]], ["lokal", "UB-Ollama", "chat"])
        self.assertEqual(data["providers"][1], {
            "name": "UB-Ollama", "provider_type": "ollama", "enabled": True,
            "preferred_model": "m-alt", "description": "alt", "host": "alt.ub.example",
            "port": 11434, "use_ssl": False, "connection_type": "native_client"})
        self.assertEqual(data["default_models"], {
            "general": {"provider": "chat", "model": "c1"},
            "pipeline": {"provider": "UB-Ollama", "model": "m-alt"}})
        self.assertTrue(any("Standardmodell vision" in w and "nicht im Paket" in w
                            for w in result.warnings))
        blob = self.files_text(result)
        for key in (OLD_KEY, OTHER_KEY, "sk-AUS"):
            self.assertFalse(key.encode() in blob, "Schlüssel im Paket")
        self.assertNotIn("llm", data)
        self.assertIn("Enthält keine API-Schlüssel", data["_comment"])
        self.assertNotEqual(result.preset_file.stat().st_mode & 0o077, 0)
        self.assertIn("Schlüssel im Klartext enthalten für: keinen", result.lines())

    @unittest.skipIf(sys.platform.startswith("win"), "POSIX permissions")
    def test_named_key_only_owner_only_file_and_comment(self):
        result = self.export(provider_keys=["ub-ollama"])
        data = json.loads(result.preset_file.read_text(encoding="utf-8"))
        keys = {p["name"]: p.get("api_key") for p in data["providers"]}
        self.assertEqual(keys, {"lokal": None, "UB-Ollama": OLD_KEY, "chat": None})
        self.assertEqual(result.preset_file.stat().st_mode & 0o777, 0o600)
        self.assertIn("ACHTUNG: enthält API-Schlüssel im Klartext (Provider UB-Ollama)",
                      data["_comment"])
        self.assertEqual(result.provider_keys, ["UB-Ollama"])
        self.assertIn("Schlüssel im Klartext enthalten für: UB-Ollama", result.lines())
        self.assertTrue(any("Klartext" in w for w in result.warnings))
        _no_key(self, "\n".join(result.lines()), "Export-Zusammenfassung")

    def test_unknown_or_switched_off_name_raises_and_writes_nothing(self):
        for names in (["gibt-es-nicht"], ["aus"]):
            with self.assertRaises(pa.PresetError):
                self.export(provider_keys=names)
            self.assertFalse((self.tmp / "out" / "pkg").exists())

    def test_named_provider_without_key_is_a_warning(self):
        result = self.export(provider_keys=["lokal"])
        self.assertEqual(result.provider_keys, [])
        self.assertTrue(any("lokal: kein Schlüssel" in w for w in result.warnings))

    def test_roundtrip_with_and_without_key(self):
        from src.utils.config_manager import ConfigManager

        with_key = self.export("mit", provider_keys=["UB-Ollama", "chat"])
        without = self.export("ohne")
        source = self.config()
        expected = [(p.name, p.provider_type, p.base_url, p.host, p.preferred_model, p.enabled,
                     p.api_key) for p in source.unified_config.providers if p.enabled]
        ConfigManager.reset()
        cm_b = ConfigManager()
        cm_b.config_file = self.tmp / "b" / ".config" / "alima" / "config.json"
        cm_b.config_file.parent.mkdir(parents=True)
        cm_b.save_config(cm_b.load_config())
        plan = pa.build_plan(pa.load_preset(with_key.path), cm_b)
        self.assertEqual(plan.errors, [])
        pa.apply_plan(plan, cm_b)
        target = cm_b.load_config(force_reload=True)
        self.assertEqual(self.providers(target), expected)
        self.assertEqual(self.defaults(target)["general"], ("chat", "c1"))
        self.assertEqual(self.defaults(target)["pipeline"], ("UB-Ollama", "m-alt"))
        # the package without keys on the same installation: nothing to change,
        # the keys stay
        again = pa.build_plan(pa.load_preset(without.path), cm_b)
        self.assertTrue(again.is_empty, again.lines())
        self.assertEqual({p.key for p in again.providers}, {pa.KEY_KEPT, pa.KEY_NONE})


# ===========================================================================
# 6. The key in no output
# ===========================================================================
class TestKeyNeverShown(_ProviderHome):
    @pytest.fixture(autouse=True)
    def _caplog(self, caplog):
        self.caplog = caplog

    def run_cli(self, *argv, answer="j"):
        from src.cli.commands import preset_cmd
        from src.cli.main import create_argument_parser

        args = create_argument_parser().parse_args(["preset", *argv])
        out = io.StringIO()
        with redirect_stdout(out), patch("builtins.input", return_value=answer):
            rc = preset_cmd.handle_preset(args, logging.getLogger("test"))
        return rc, out.getvalue()

    def test_plan_log_cli_and_exceptions(self):
        self.caplog.set_level(logging.DEBUG)
        self.set_providers(existing_providers())
        folder = self.write_preset({"providers": [provider_entry(name="UB-Ollama", api_key=KEY)],
                                    "default_models": {"general": {"provider": "ub-ollama"}}})
        outputs = []
        rc, out = self.run_cli("apply", str(folder), "--dry-run")
        outputs.append(out)
        self.assertIn("Schlüssel wird gesetzt", out)
        stale = self.plan(folder)
        outputs += stale.lines()
        rc, out = self.run_cli("apply", str(folder), "--yes")
        self.assertEqual(rc, 0)
        outputs.append(out)
        self.assertIn("Klartext in der config.json", out)
        with self.assertRaises(pa.PresetError) as ctx:   # a plan on a changed config
            pa.apply_plan(stale, self.cm)
        outputs.append(str(ctx.exception))
        rc, out = self.run_cli("show")
        outputs.append(out)
        rc, out = self.run_cli("export", str(self.tmp / "exp"), "--with-provider-key",
                               "UB-Ollama", "--with-provider-key", "chat")
        self.assertEqual(rc, 0)
        outputs.append(out)
        self.assertIn("API-Schlüssel im Klartext (UB-Ollama, chat)", out)
        rc, out = self.run_cli("revert", "--yes")
        self.assertEqual(rc, 0)
        outputs.append(out)
        outputs.append(json.dumps(stale.as_dict()))
        records = [r.getMessage() for r in self.caplog.records]
        records += [self.caplog.text]
        self.assertTrue(records)
        _no_key(self, "\n".join(outputs), "CLI-Ausgabe/Vorschau/Ausnahme")
        _no_key(self, "\n".join(records), "Log")
        # the export did put the keys into the package (they were asked for)
        self.assertIn(KEY, (self.tmp / "exp" / pa.PRESET_FILENAME).read_text(encoding="utf-8"))


class TestProviderCli(_ProviderHome):
    run_cli = TestKeyNeverShown.run_cli

    def test_no_default_models_switch(self):
        self.set_providers(existing_providers(), preferred_provider="lokal",
                           preferred_model="m-lokal")
        folder = self.write_preset({"providers": [provider_entry(name="neu")],
                                    "default_models": {"general": {"provider": "neu"}}})
        rc, out = self.run_cli("apply", str(folder), "--yes", "--no-default-models")
        self.assertEqual(rc, 0)
        self.assertIn("abgewählt: bleiben, wie sie sind", out)
        cfg = self.config()
        self.assertEqual(self.defaults(cfg)["general"], ("lokal", "m-lokal"))
        self.assertIn("neu", [p.name for p in cfg.unified_config.providers])
        pa.revert_last(self.cm)
        rc, out = self.run_cli("apply", str(folder), "--yes")
        self.assertEqual(self.defaults()["general"], ("neu", "m-pipe"))

    def test_with_provider_key_is_repeatable(self):
        self.set_providers(existing_providers())
        rc, out = self.run_cli("export", str(self.tmp / "exp"), "--with-provider-key",
                               "UB-Ollama", "--with-provider-key", "chat")
        self.assertEqual(rc, 0)
        data = json.loads((self.tmp / "exp" / pa.PRESET_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual({p["name"]: p.get("api_key") for p in data["providers"]},
                         {"lokal": None, "UB-Ollama": OLD_KEY, "chat": OTHER_KEY})
        rc, out = self.run_cli("export", str(self.tmp / "exp2"), "--with-provider-key", "nix")
        self.assertEqual(rc, 1)
        self.assertIn("Unbekannte(r) Provider: nix", out)
        _no_key(self, out, "CLI-Ausgabe")


# ===========================================================================
# 7. Old package, wizard pre-fill, setup wizard
# ===========================================================================
OLD_LLM_PRESET = {
    "institution_name": "Alte UB",
    "llm": {"provider_type": "openai_compatible", "provider_name": "Uni-LLM",
            "base_url": "https://llm.alte-ub.example/v1",
            "task_models": {"pipeline": "m-pipe", "vision": "m-vis"}},
}


class TestOldPackageAndWizard(_ProviderHome):
    def test_old_package_with_llm_gives_one_hint_and_no_change(self):
        before = self.disk()
        plan = self.plan(self.write_preset(OLD_LLM_PRESET))
        self.assertEqual((plan.errors, plan.changes, plan.providers), ([], [], []))
        hints = [w for w in plan.warnings if "llm" in w]
        self.assertEqual(hints, ["Veralteter Block 'llm' wird ignoriert; Provider stehen unter "
                                 "'providers', Standardmodelle unter 'default_models'"])
        self.assertFalse(any("Unbekannter Schlüssel" in w for w in plan.warnings))
        self.assertIsNone(pa.apply_plan(plan, self.cm))
        self.assertEqual(self.disk(), before)

    def test_prefill_from_providers_and_default_models(self):
        from src.utils.preset_loader import PresetLoader

        data = {"providers": [
            {"name": "aus", "provider_type": "gemini", "enabled": False},
            provider_entry(name="erst", host="erst.example", use_ssl=True, port=8443,
                           api_key=KEY),
            {"name": "chat", "provider_type": "openai_compatible",
             "base_url": "https://chat.example/v1", "preferred_model": "c1"}],
            "default_models": {"pipeline": {"provider": "ERST", "model": "m-p"},
                               "vision": {"provider": "chat", "model": "m-v"}}}
        p = PresetLoader._parse(data)
        self.assertEqual((p.llm_provider_type, p.llm_provider_name, p.llm_base_url),
                         ("ollama", "erst", "https://erst.example:8443"))
        self.assertEqual((p.pipeline_model(), p.vision_model()), ("m-p", "m-pipe"))
        _no_key(self, repr(p), "Vorbelegung")
        data["default_models"]["general"] = {"provider": "chat", "model": "c2"}
        p = PresetLoader._parse(data)
        self.assertEqual((p.llm_provider_type, p.llm_provider_name, p.llm_base_url),
                         ("openai_compatible", "chat", "https://chat.example/v1"))
        self.assertEqual((p.llm_default_model, p.vision_model()), ("c2", "m-v"))
        self.assertTrue(p.has_llm())

    def test_gui_wizard_pages_prefill_from_providers(self):
        _need_widgets(self)
        from src.ui.first_start_wizard import LLMSetupPage
        from src.utils.preset_loader import PresetLoader

        ollama = PresetLoader._parse({"providers": [provider_entry(
            host="llm.ub.example", port=11500, api_key=KEY)]})
        page = LLMSetupPage(presets=ollama)
        _KEEP.append(page)
        self.assertTrue(page.ollama_radio.isChecked())
        self.assertEqual((page.ollama_host_input.text(), page.ollama_port_input.value()),
                         ("llm.ub.example", 11500))
        self.assertEqual(page.provider_name, "ub-ollama")
        self.assertEqual(page.api_key_input.text(), "")
        chat = PresetLoader._parse({"providers": [{
            "name": "chat", "provider_type": "openai_compatible",
            "base_url": "https://chat.example/v1"}]})
        page = LLMSetupPage(presets=chat)
        _KEEP.append(page)
        self.assertTrue(page.openai_radio.isChecked())
        self.assertEqual(page.base_url_input.text(), "https://chat.example/v1")

    def test_apply_after_setup_leaves_providers_and_defaults_alone(self):
        self.set_providers(existing_providers()[:1], preferred_provider="lokal",
                           preferred_model="m-lokal")
        folder = self.write_preset({
            "institution": {"isil": "ZZ-W"},
            "providers": [provider_entry(api_key=KEY)],
            "default_models": {"general": {"provider": "ub-ollama"}}})
        lines = pa.apply_after_setup(folder, self.cm)
        self.assertIn("angewendet", lines[0])
        cfg = self.config()
        self.assertEqual(cfg.system_config.institution_isil, "ZZ-W")
        self.assertEqual([p.name for p in cfg.unified_config.providers], ["lokal"])
        self.assertEqual(self.defaults(cfg)["general"], ("lokal", "m-lokal"))
        _no_key(self, "\n".join(lines), "Assistent")
        summary = pa.describe_preset(pa.load_preset(folder))
        self.assertTrue(any(line.startswith("Provider im Preset (der Assistent legt sie nicht an")
                            and line.endswith(": ub-ollama") for line in summary))


# ===========================================================================
# 8. GUI: preview check box, export check boxes, settings dialog follows
# ===========================================================================
class TestProviderUi(_ProviderHome):
    def setUp(self):
        super().setUp()
        _need_widgets(self)
        import socket

        def no_network(*_a, **_k):
            raise OSError("Netz im Test gesperrt")

        for target in ("getaddrinfo", "create_connection"):
            guard = patch.object(socket, target, side_effect=no_network)
            guard.start()
            self.addCleanup(guard.stop)

    def test_preview_check_box_for_default_models(self):
        from src.ui.plugin_settings_tab import _PresetPlanDialog

        self.set_providers(existing_providers())
        only_defaults = self.plan(self.write_preset(
            {"default_models": {"pipeline": {"provider": "chat", "model": "c1"}}}))
        dialog = _PresetPlanDialog(only_defaults)
        _KEEP.append(dialog)
        self.assertTrue(dialog.default_models_cb.isVisibleTo(dialog))
        self.assertTrue(dialog.default_models_cb.isChecked())
        self.assertTrue(dialog.apply_button.isEnabled())
        dialog.default_models_cb.setChecked(False)
        self.assertFalse(only_defaults.apply_default_models)
        self.assertIn("abgewählt", dialog.text.toPlainText())
        self.assertFalse(dialog.apply_button.isEnabled())
        no_defaults = _PresetPlanDialog(self.plan(self.write_preset(
            {"providers": [provider_entry(api_key=KEY)]}, "p2")))
        _KEEP.append(no_defaults)
        self.assertFalse(no_defaults.default_models_cb.isVisibleTo(no_defaults))
        self.assertIn("im Klartext in die Konfigurationsdatei", no_defaults.note.text())
        _no_key(self, no_defaults.text.toPlainText(), "Vorschau-Dialog")

    def test_export_dialog_offers_a_key_box_per_provider_with_key(self):
        from src.ui.plugin_settings_tab import _ExportPresetDialog

        dialog = _ExportPresetDialog([("search_provider", "swb", "SWB", True)], providers=[
            ("lokal", "ollama", False), ("UB-Ollama", "ollama", True), ("chat", "openai_compatible", True)])
        _KEEP.append(dialog)
        self.assertEqual(list(dialog.provider_key_boxes), ["UB-Ollama", "chat"])
        self.assertEqual(dialog.selected_provider_keys(), [])
        dialog.provider_key_boxes["chat"].setChecked(True)
        self.assertEqual(dialog.selected_provider_keys(), ["chat"])

    def test_settings_dialog_follows_providers_and_defaults(self):
        from src.ui.comprehensive_settings_dialog import ComprehensiveSettingsDialog
        from src.utils.config_manager import ProviderDetectionService

        # no provider is probed or asked for models (no network)
        for name in ("get_available_providers", "get_available_models"):
            stub = patch.object(ProviderDetectionService, name, return_value=[])
            stub.start()
            self.addCleanup(stub.stop)
        stub = patch.object(ProviderDetectionService, "is_provider_reachable", return_value=False)
        stub.start()
        self.addCleanup(stub.stop)
        self.set_providers(existing_providers(), preferred_provider="lokal",
                           preferred_model="m-lokal")
        dialog = ComprehensiveSettingsDialog()
        _KEEP.append(dialog)
        plan = self.plan(self.write_preset({
            "providers": [provider_entry(name="neu", api_key=KEY)],
            "default_models": {"general": {"provider": "neu", "model": "m-neu"}}}))
        pa.apply_plan(plan, self.cm)
        dialog.plugin_settings_tab._after_preset(plan.steps(), "x")
        tab = dialog.unified_provider_tab
        self.assertIn("neu", [p.name for p in dialog.config_to_edit.unified_config.providers])
        self.assertIn("neu", [p.name for p in tab._providers_at_open])
        cfg = dialog._get_config_from_ui()
        self.assertEqual((cfg.unified_config.preferred_provider,
                          cfg.unified_config.preferred_model), ("neu", "m-neu"))
        # Cancel undoes only the dialog's own provider edits, not the preset
        self.assertFalse(tab.restore_providers_at_open())
        self.assertIn("neu", [p.name for p in self.config().unified_config.providers])


if __name__ == "__main__":
    unittest.main()
