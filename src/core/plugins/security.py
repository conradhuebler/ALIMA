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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

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

# ``requests.<verb>()`` calls that must carry an explicit ``timeout=`` kwarg
# (project HTTP convention — a hung endpoint must not hang the pipeline).
_HTTP_VERB_ATTRS: Set[str] = {"get", "post", "put", "patch", "delete", "head", "request"}

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
            elif (
                func.value.id == "requests"
                and func.attr in _HTTP_VERB_ATTRS
                and not any(kw.arg == "timeout" for kw in node.keywords)
            ):
                self._add(
                    "medium",
                    f"calls 'requests.{func.attr}()' without an explicit timeout",
                    node,
                )
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


def iter_plugin_files(plugin_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Regular files + symlinks under ``plugin_dir``; symlinks are never followed.

    Skips ``__pycache__`` and hidden directories plus ``*.pyc`` noise. Symlinks
    (file or directory) are returned separately: they are excluded from hashing
    and scanning — a symlink can point outside the plugin dir, so callers flag
    them instead of trusting their content. - Claude Generated
    """
    plugin_dir = Path(plugin_dir)
    files: List[Path] = []
    symlinks: List[Path] = []
    for root, dirnames, filenames in os.walk(plugin_dir, followlinks=False):
        rootp = Path(root)
        kept_dirs = []
        for d in dirnames:
            dp = rootp / d
            if dp.is_symlink():
                symlinks.append(dp)
            elif d == "__pycache__" or d.startswith("."):
                continue
            else:
                kept_dirs.append(d)
        dirnames[:] = kept_dirs
        for f in filenames:
            fp = rootp / f
            if fp.is_symlink():
                symlinks.append(fp)
            elif not f.endswith(".pyc"):
                files.append(fp)
    return sorted(files), sorted(symlinks)


def scan_dir(plugin_dir: Path) -> List[ScanFinding]:
    """Scan every ``*.py`` under ``plugin_dir`` and merge findings.

    Any symlink in the plugin dir is itself a high finding: it is excluded from
    the trust hash and may point outside the directory the operator approves.
    """
    plugin_dir = Path(plugin_dir)
    files, symlinks = iter_plugin_files(plugin_dir)
    findings: List[ScanFinding] = []
    for link in symlinks:
        rel = link.relative_to(plugin_dir).as_posix()
        findings.append(
            ScanFinding(
                "high",
                f"contains symlink '{rel}' — excluded from the trust hash, may point outside the plugin dir",
                0,
            )
        )
    for py in files:
        if py.suffix != ".py":
            continue
        findings.extend(
            scan_source(py.read_text(encoding="utf-8", errors="replace"), filename=str(py))
        )
    return findings


def hash_dir(plugin_dir: Path) -> str:
    """SHA-256 over the plugin's files (path + bytes), for trust-on-first-use.

    Covers **all** regular files (code, manifests, data — a plugin's data files
    are part of what the operator approves); symlinks are excluded (flagged by
    :func:`scan_dir` instead). Paths are included relative to ``plugin_dir`` so
    a rename is a change too. Deterministic across runs (sorted file order).
    """
    plugin_dir = Path(plugin_dir)
    files, _symlinks = iter_plugin_files(plugin_dir)
    h = hashlib.sha256()
    for f in files:
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
