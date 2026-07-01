"""Tests for the generic plugin framework - Claude Generated.

Covers the category-agnostic core: the ``ConfigField`` schema, the AST security
scanner + trust-on-first-use hashing, manifest parsing/validation, the category
registry, and the two-tier directory loader (declarative + code + approval gate).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.core.plugins import (
    BOOL,
    INT,
    SECRET,
    ConfigField,
    PluginCategory,
    PluginTypeMeta,
    availability_ok,
    coerce_settings,
    defaults,
    get_category,
    parse_manifest,
    register_category,
)
from src.core.plugins.category import PLUGIN_CATEGORY_REGISTRY
from src.core.plugins.manifest import ManifestError
from src.core.plugins import loader as loader_mod
from src.core.plugins import security


class ConfigFieldTest(unittest.TestCase):
    def test_secret_kind_sets_secret_and_defaults(self):
        f = ConfigField(key="token", label="Token", kind=SECRET)
        self.assertTrue(f.secret)
        self.assertEqual(f.default, "")

    def test_choice_requires_choices(self):
        with self.assertRaises(ValueError):
            ConfigField(key="x", label="x", kind="choice")

    def test_coerce_types_and_fallback(self):
        i = ConfigField(key="n", label="n", kind=INT, default=20)
        self.assertEqual(i.coerce("5"), 5)
        self.assertEqual(i.coerce("nan"), 20)  # bad → default
        self.assertEqual(i.coerce(""), 20)
        b = ConfigField(key="b", label="b", kind=BOOL)
        self.assertTrue(b.coerce("true"))
        self.assertFalse(b.coerce("no"))

    def test_defaults_and_coerce_settings_preserve_unknown(self):
        fields = [ConfigField(key="n", label="n", kind=INT, default=1)]
        self.assertEqual(defaults(fields), {"n": 1})
        out = coerce_settings(fields, {"n": "3", "extra": "keep"})
        self.assertEqual(out, {"n": 3, "extra": "keep"})

    def test_availability_gate(self):
        fields = [ConfigField(key="url", label="url", kind="url", gates_availability=True)]
        self.assertTrue(availability_ok(fields, {"url": "http://x"}))
        self.assertFalse(availability_ok(fields, {"url": ""}))


class SecurityScanTest(unittest.TestCase):
    def test_flags_dangerous_constructs(self):
        src = "import os\nimport requests\nos.system('x')\ny=eval('1')\nopen('f','w')\n"
        findings = security.scan_source(src)
        msgs = [f.message for f in findings]
        self.assertTrue(any("os.system" in m for m in msgs))
        self.assertTrue(any("eval" in m for m in msgs))
        self.assertTrue(any("writing" in m for m in msgs))
        self.assertEqual(security.max_severity(findings), "high")

    def test_clean_source_no_findings(self):
        self.assertEqual(security.scan_source("x = 1 + 2\ndef f():\n    return x\n"), [])

    def test_hash_dir_changes_on_edit(self):
        d = Path(tempfile.mkdtemp())
        (d / "a.py").write_text("x = 1")
        h1 = security.hash_dir(d)
        self.assertEqual(h1, security.hash_dir(d))  # stable
        (d / "a.py").write_text("x = 2")
        self.assertNotEqual(h1, security.hash_dir(d))


class ManifestTest(unittest.TestCase):
    def test_declarative_ok(self):
        m = parse_manifest({
            "plugin": {"id": "finc-x", "label": "X", "category": "search_provider", "type": "declarative", "kind": "finc"},
            "settings": {"base_url": "http://y"},
        })
        self.assertTrue(m.is_declarative)
        self.assertEqual(m.kind, "finc")
        self.assertEqual(m.settings["base_url"], "http://y")

    def test_code_requires_entry(self):
        with self.assertRaises(ManifestError):
            parse_manifest({"plugin": {"id": "x", "category": "c", "type": "code", "label": "X"}})

    def test_bad_id_and_type_and_version(self):
        for bad in (
            {"plugin": {"id": "BAD ID", "category": "c", "type": "declarative", "kind": "k"}},
            {"plugin": {"id": "x", "category": "c", "type": "weird"}},
            {"plugin": {"id": "x", "category": "c", "type": "declarative", "kind": "k", "api_version": "99"}},
        ):
            with self.assertRaises(ManifestError):
                parse_manifest(bad)


class _FakeCategory(PluginCategory):
    name = "test_fake_cat"

    def __init__(self):
        self.registered = []

    def list_types(self):
        return ["finc"]

    def type_meta(self, type_id):
        return PluginTypeMeta(
            type_id, type_id.title(), self.name,
            [ConfigField(key="base_url", label="url", kind="url")],
        )

    def build(self, instance):
        return ("built", instance.provider_id, dict(instance.settings))

    def register_code_type(self, cls):
        self.registered.append(cls)
        return getattr(cls, "id", "code_type")


class CategoryAndLoaderTest(unittest.TestCase):
    def setUp(self):
        loader_mod._reset_for_tests()
        self.cat = _FakeCategory()
        register_category(self.cat)

    def tearDown(self):
        PLUGIN_CATEGORY_REGISTRY.pop(self.cat.name, None)
        loader_mod._reset_for_tests()

    def _plugin_dir(self):
        return Path(tempfile.mkdtemp())

    def test_declarative_discovery_builds_instance(self):
        root = self._plugin_dir()
        d = root / "myfinc"
        d.mkdir()
        (d / "plugin.toml").write_text(
            '[plugin]\nid="myfinc"\nlabel="My finc"\ncategory="test_fake_cat"\n'
            'type="declarative"\nkind="finc"\n[settings]\nbase_url="http://z"\n'
        )
        result = loader_mod.discover(root)
        self.assertEqual([p.status for p in result.plugins], ["loaded"])
        self.assertEqual(len(result.instances), 1)
        self.assertEqual(result.instances[0].settings["base_url"], "http://z")

    def test_unknown_kind_errors(self):
        root = self._plugin_dir()
        d = root / "bad"
        d.mkdir()
        (d / "plugin.toml").write_text(
            '[plugin]\nid="bad"\ncategory="test_fake_cat"\ntype="declarative"\nkind="nope"\n'
        )
        result = loader_mod.discover(root)
        self.assertEqual(result.plugins[0].status, "error")
        self.assertEqual(result.instances, [])

    def test_code_plugin_blocked_when_disabled(self):
        root = self._plugin_dir()
        d = root / "code"
        d.mkdir()
        (d / "plugin.toml").write_text(
            '[plugin]\nid="code"\ncategory="test_fake_cat"\ntype="code"\nlabel="C"\n'
            '[entry]\nmodule="p.py"\nclass="P"\n'
        )
        (d / "p.py").write_text("class P:\n    id='p'\n")
        result = loader_mod.discover(root, enable_code_plugins=False)
        self.assertEqual(result.plugins[0].status, "blocked")

    def test_code_plugin_approval_gate_and_hash_pin(self):
        root = self._plugin_dir()
        d = root / "code"
        d.mkdir()
        (d / "plugin.toml").write_text(
            '[plugin]\nid="code"\ncategory="test_fake_cat"\ntype="code"\nlabel="C"\n'
            '[entry]\nmodule="p.py"\nclass="P"\n'
        )
        (d / "p.py").write_text("import os\nclass P:\n    id='p'\n    def go(self):\n        os.system('x')\n")

        # Deny → not loaded, nothing approved
        approved = {}
        result = loader_mod.discover(root, approved_plugins=approved, approve_cb=lambda m, f, h: False, enable_code_plugins=True)
        self.assertEqual(result.plugins[0].status, "denied")
        self.assertEqual(approved, {})
        self.assertTrue(any(x.severity == "high" for x in result.plugins[0].findings))

        # Approve → loaded, registered, hash pinned
        seen = {}
        result = loader_mod.discover(
            root, approved_plugins=approved,
            approve_cb=lambda m, f, h: seen.setdefault("h", h) or True,
            enable_code_plugins=True,
        )
        self.assertEqual(result.plugins[0].status, "loaded")
        self.assertEqual(len(self.cat.registered), 1)
        self.assertEqual(approved["code"], seen["h"])

    def test_missing_dir_returns_empty(self):
        result = loader_mod.discover(Path(tempfile.mkdtemp()) / "nope")
        self.assertEqual(result.plugins, [])
        self.assertEqual(result.instances, [])


if __name__ == "__main__":
    unittest.main()
