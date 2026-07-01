"""Static security checks for Tier-2 *code* plugins - Claude Generated.

Two mechanisms, both **deterrents, not guarantees**:

* :func:`scan_source` / :func:`scan_dir` — an AST walk that flags risky
  constructs (process/network/filesystem access, dynamic code execution). It
  raises the bar and catches naive or accidental cases; it does **not** prove a
  plugin is safe. Obfuscated code can evade it.
* :func:`hash_dir` — a SHA-256 over the plugin's Python/manifest files for
  trust-on-first-use: once an operator approves a plugin, a changed hash forces
  re-approval (tamper detection).

The honest limit (documented for the approval dialog): an approved plugin is
imported in-process with full privileges. This module supports informed consent,
not isolation. See ``docs/plugin_system.md``.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

# Modules whose mere import is worth flagging, grouped by severity.
_HIGH_IMPORTS: Set[str] = {
    "subprocess", "socket", "ctypes", "multiprocessing", "pty", "fcntl",
}
_MEDIUM_IMPORTS: Set[str] = {
    "os", "shutil", "pickle", "marshal", "importlib", "sys", "tempfile",
}
_LOW_IMPORTS: Set[str] = {
    "requests", "urllib", "http", "httpx", "ftplib", "smtplib", "telnetlib",
}

# Bare builtin calls that execute code / touch the environment.
_HIGH_CALLS: Set[str] = {"eval", "exec", "compile", "__import__"}

# ``module.attr`` call chains worth flagging (matched on the trailing attr with a
# known dangerous root name).
_HIGH_ATTR_CALLS = {
    ("os", "system"), ("os", "popen"), ("os", "execv"), ("os", "execve"),
    ("os", "spawnl"), ("os", "fork"), ("os", "kill"), ("os", "remove"),
    ("os", "unlink"), ("os", "rmdir"), ("os", "removedirs"), ("os", "rename"),
    ("subprocess", "run"), ("subprocess", "call"), ("subprocess", "Popen"),
    ("subprocess", "check_output"), ("subprocess", "check_call"),
    ("shutil", "rmtree"), ("shutil", "move"),
    ("socket", "socket"),
    ("importlib", "import_module"),
}


@dataclass
class ScanFinding:
    """One flagged construct. ``severity`` ∈ {high, medium, low}."""

    severity: str
    message: str
    lineno: int

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.severity}] line {self.lineno}: {self.message}"


class _RiskVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: List[ScanFinding] = []

    def _add(self, severity: str, message: str, node: ast.AST) -> None:
        self.findings.append(
            ScanFinding(severity, message, getattr(node, "lineno", 0))
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            self._flag_import(root, node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            self._flag_import(root, node)
        self.generic_visit(node)

    def _flag_import(self, root: str, node: ast.AST) -> None:
        if root in _HIGH_IMPORTS:
            self._add("high", f"imports '{root}' (process/network/native access)", node)
        elif root in _MEDIUM_IMPORTS:
            self._add("medium", f"imports '{root}' (filesystem/dynamic-code access)", node)
        elif root in _LOW_IMPORTS:
            self._add("low", f"imports '{root}' (network access)", node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in _HIGH_CALLS:
                self._add("high", f"calls '{func.id}()' (dynamic code execution)", node)
            elif func.id == "open" and self._open_is_write(node):
                self._add("medium", "opens a file for writing/appending", node)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            pair = (func.value.id, func.attr)
            if pair in _HIGH_ATTR_CALLS:
                self._add("high", f"calls '{pair[0]}.{pair[1]}()'", node)
        self.generic_visit(node)

    @staticmethod
    def _open_is_write(node: ast.Call) -> bool:
        mode = None
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            mode = node.args[1].value
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = kw.value.value
        return isinstance(mode, str) and any(c in mode for c in ("w", "a", "x", "+"))


def scan_source(source: str, *, filename: str = "<plugin>") -> List[ScanFinding]:
    """AST-scan one source string. A syntax error is itself a high finding."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:  # pragma: no cover - defensive
        return [ScanFinding("high", f"does not parse: {exc}", getattr(exc, "lineno", 0) or 0)]
    visitor = _RiskVisitor()
    visitor.visit(tree)
    # Stable ordering: severity (high first) then line number.
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(visitor.findings, key=lambda f: (order.get(f.severity, 3), f.lineno))


def scan_dir(plugin_dir: Path) -> List[ScanFinding]:
    """Scan every ``*.py`` under ``plugin_dir`` and merge findings."""
    plugin_dir = Path(plugin_dir)
    findings: List[ScanFinding] = []
    for py in sorted(plugin_dir.rglob("*.py")):
        findings.extend(
            scan_source(py.read_text(encoding="utf-8", errors="replace"), filename=str(py))
        )
    return findings


def _plugin_files(plugin_dir: Path) -> List[Path]:
    """Files that participate in the trust hash: code + manifests."""
    plugin_dir = Path(plugin_dir)
    out: List[Path] = []
    for pattern in ("*.py", "*.toml", "*.yaml", "*.yml", "*.json"):
        out.extend(plugin_dir.rglob(pattern))
    return sorted(set(out))


def hash_dir(plugin_dir: Path) -> str:
    """SHA-256 over the plugin's files (path + bytes), for trust-on-first-use.

    Paths are included relative to ``plugin_dir`` so a rename is a change too.
    Deterministic across runs (sorted file order).
    """
    plugin_dir = Path(plugin_dir)
    h = hashlib.sha256()
    for f in _plugin_files(plugin_dir):
        rel = f.relative_to(plugin_dir).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def max_severity(findings: List[ScanFinding]) -> str:
    """Return the highest severity present, or ``"none"`` when clean."""
    order = {"high": 0, "medium": 1, "low": 2}
    if not findings:
        return "none"
    return min((f.severity for f in findings), key=lambda s: order.get(s, 3))
