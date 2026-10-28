#!/usr/bin/env python3
"""Find Qt style-sheet text that Qt drops silently - Claude Generated.

Qt's QSS parser does not complain about most malformed input: a declaration it
cannot read, or a line such as ``#padding: 12px 8px;`` inside a rule, makes it
discard the rule (or parts of it) without a warning. The effect is a widget
that looks "as if the rule were not there", which is how a minimum width in the
same rule stayed inactive for months and then suddenly applied once the line
was cleaned up.

The checker reads every Python file under ``src/`` with ``ast`` and validates

* arguments of ``.setStyleSheet(...)`` that are literal strings or f-strings,
* every other string constant that looks like a rule set (braces around
  ``property: value`` pairs under a Qt class, ``#id`` or ``*`` selector).

f-string placeholders and ``{name}``/``{{ }}`` template fields are replaced by a
neutral value first, so colours computed at run time do not matter.

Errors (the rule or declaration is not valid QSS):
  * a declaration without ``property: value``
  * ``#``/``//`` comment lines (only ``/* */`` comments exist in QSS)
  * a missing ``;`` between two declarations
  * unbalanced braces, parentheses or quotes
Warnings (likely mistakes, but Qt may accept them):
  * a property name Qt does not know (Qt itself prints "Unknown property")

Usage: ``python scripts/lint_qss.py [paths...]``; exit status 1 on errors.
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

REPO = Path(__file__).resolve().parent.parent

# Properties from the Qt Style Sheets reference (plus the generic prefixes below).
KNOWN_PROPERTIES = frozenset("""
accent-color alignment alternate-background-color background background-attachment background-clip
background-color background-image background-origin background-position background-repeat
border border-bottom border-bottom-color border-bottom-left-radius border-bottom-right-radius
border-bottom-style border-bottom-width border-color border-image border-left border-left-color
border-left-style border-left-width border-radius border-right border-right-color
border-right-style border-right-width border-style border-top border-top-color
border-top-left-radius border-top-right-radius border-top-style border-top-width border-width
bottom button-layout color dialogbuttonbox-buttons-have-icons font font-family font-size
font-style font-variant font-weight gridline-color height icon icon-size image image-position
left letter-spacing line-height lineedit-password-character lineedit-password-mask-delay margin
margin-bottom margin-left margin-right margin-top max-height max-width messagebox-text-interaction-flags
min-height min-width opacity outline outline-color outline-offset outline-radius
outline-bottom-left-radius outline-bottom-right-radius outline-top-left-radius
outline-top-right-radius outline-style outline-width padding padding-bottom padding-left
padding-right padding-top paint-alternating-row-colors-for-empty-area position right
selection-background-color selection-color show-decoration-selected spacing subcontrol-origin
subcontrol-position text-align text-decoration text-decoration-color titlebar-show-tooltips-on-buttons
top widget-animation-duration width
""".split())
KNOWN_PREFIXES = ("qproperty-", "-qt-", "-webkit-")

DECLARATION = re.compile(r"^-?[A-Za-z][A-Za-z0-9-]*\s*:\s*\S")
# A string that is style-sheet text without being passed to setStyleSheet right
# there: a rule whose selector starts with a Qt class name ("QFrame {", "QTabBar::tab:selected {").
RULE_LIKE = re.compile(r"(?m)^[ \t]*Q[A-Z][A-Za-z0-9]*[\w:#.\[\]=\"' \t>,-]*\{[^{}]*:[^{}]*\}", re.S)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    level: str  # "error" | "warning"
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.level}: {self.message}"


# ---------------------------------------------------------------- QSS text ---

def _strip_comments(text: str) -> str:
    """Remove ``/* */`` comments but keep newlines so line numbers survive."""
    return re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)


def _balanced(value: str) -> Optional[str]:
    depth = 0
    quote = ""
    for ch in value:
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return "unmatched ')'"
    if quote:
        return f"unclosed quote {quote}"
    if depth:
        return "unclosed '('"
    return None


def _check_declarations(body: str, first_line: int, report) -> None:
    """Validate one rule body (or a bare declaration list)."""
    # split on ';' that is not inside parentheses or quotes
    chunks, buf, depth, quote, line_of = [], [], 0, "", first_line
    start_line = first_line
    for ch in body:
        if ch == "\n":
            line_of += 1
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == ";" and depth <= 0:
            chunks.append(("".join(buf), start_line))
            buf, start_line = [], line_of
            continue
        buf.append(ch)
    chunks.append(("".join(buf), start_line))

    for chunk, line in chunks:
        raw = chunk.strip()
        if not raw:
            continue
        # report on the first non-blank line of the chunk
        line += len(chunk) - len(chunk.lstrip("\n")) if chunk.startswith("\n") else 0
        if raw.startswith("#") or raw.startswith("//"):
            report(line, "error", f"comment-like line in a rule, Qt drops it and may "
                                  f"drop the rule: {raw.splitlines()[0][:60]!r}")
            continue
        first = raw.splitlines()[0].strip()
        if not DECLARATION.match(raw):
            report(line, "error", f"not a 'property: value' declaration: {first[:60]!r}")
            continue
        prop, _, value = raw.partition(":")
        prop = prop.strip().lower()
        flat = value
        while re.search(r"\([^()]*\)", flat):  # qlineargradient(x1:0,\n stop:0 ...) spans lines
            flat = re.sub(r"\([^()]*\)", "<>", flat)
        if re.search(r"\n\s*-?[A-Za-z][A-Za-z0-9-]*\s*:\s*\S", flat):
            report(line, "error", f"missing ';' before the next declaration after {prop!r}")
        problem = _balanced(value)
        if problem:
            report(line, "error", f"{prop!r}: {problem}")
        if prop not in KNOWN_PROPERTIES and not prop.startswith(KNOWN_PREFIXES):
            report(line, "warning", f"property {prop!r} is not a Qt style-sheet property")


def lint_qss_text(text: str, *, as_declarations: bool = False, first_line: int = 1,
                  path: str = "<qss>") -> List[Finding]:
    """Validate style-sheet ``text``; ``as_declarations`` for a bare ``a: b; c: d`` list."""
    findings: List[Finding] = []

    def report(line, level, message):
        findings.append(Finding(path, line, level, message))

    text = _strip_comments(text)
    if as_declarations or "{" not in text and "}" not in text:
        _check_declarations(text, first_line, report)
        return findings

    depth, i, line = 0, 0, first_line
    sel_start, body_start, body_line = 0, 0, first_line
    selector_line = first_line
    while i < len(text):
        ch = text[i]
        if ch == "\n":
            line += 1
        elif ch == "{":
            if depth == 0:
                selector = text[sel_start:i].strip()
                if not selector:
                    report(line, "error", "rule without a selector")
                elif ";" in selector or "}" in selector:
                    report(selector_line, "error",
                           f"stray ';' or '}}' in selector: {selector[:50]!r}")
                body_start, body_line = i + 1, line
            else:
                report(line, "error", "nested '{' inside a rule")
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                report(line, "error", "unmatched '}'")
                depth = 0
            elif depth == 0:
                _check_declarations(text[body_start:i], body_line, report)
                sel_start = i + 1
                selector_line = line
        i += 1
    if depth:
        report(line, "error", "unclosed '{'")
    elif text[sel_start:].strip():
        report(line, "error", f"text after the last rule: {text[sel_start:].strip()[:50]!r}")
    return findings


# ----------------------------------------------------------- source files ---

_TEMPLATE_FIELD = re.compile(r"\{[A-Za-z_][\w\.\[\]'\"]*\}")


def _literal_text(node: ast.AST) -> Optional[str]:
    """String value of a literal / f-string node with placeholders neutralised."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value
        # a plain string used as a .format() template: '{{' '}}' are braces
        text = text.replace("{{", "{").replace("}}", "}")
        return _TEMPLATE_FIELD.sub("0", text)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            else:
                parts.append("0")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal_text(node.left), _literal_text(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _calls_set_stylesheet(tree: ast.AST) -> Iterator[ast.Call]:
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setStyleSheet" and node.args):
            yield node


def lint_source(source: str, path: str = "<source>") -> List[Finding]:
    """Findings for every style-sheet-like string in Python ``source``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    findings: List[Finding] = []
    seen: set = set()
    # docstrings are prose (often with JSON/brace examples), never style sheets
    for owner in ast.walk(tree):
        if isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = owner.body
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], "value", None), ast.Constant):
                seen.add(id(body[0].value))

    def add(text: str, node: ast.AST, as_declarations: bool) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        findings.extend(lint_qss_text(
            text, as_declarations=as_declarations, first_line=node.lineno, path=path))

    for call in _calls_set_stylesheet(tree):
        arg = call.args[0]
        text = _literal_text(arg)
        if text is not None and text.strip():
            add(text, arg, as_declarations="{" not in text)
            for sub in ast.walk(arg):
                seen.add(id(sub))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Constant, ast.JoinedStr)) and id(node) not in seen:
            text = _literal_text(node)
            if text and RULE_LIKE.search(text):
                add(text, node, as_declarations=False)
                if isinstance(node, ast.JoinedStr):
                    for sub in ast.walk(node):
                        seen.add(id(sub))
    return findings


def iter_python_files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*.py") if "__pycache__" not in q.parts)
        elif p.suffix == ".py":
            yield p


def lint_paths(paths: Iterable[Path]) -> List[Finding]:
    out: List[Finding] = []
    for file in iter_python_files(paths):
        rel = file.relative_to(REPO) if file.is_absolute() and REPO in file.parents else file
        out.extend(lint_source(file.read_text(encoding="utf-8", errors="replace"), str(rel)))
    return out


def main(argv: List[str]) -> int:
    paths = [Path(a) for a in argv] or [REPO / "src"]
    findings = lint_paths(paths)
    for f in findings:
        print(f)
    errors = sum(1 for f in findings if f.level == "error")
    warnings = len(findings) - errors
    print(f"{errors} errors, {warnings} warnings in {sum(1 for _ in iter_python_files(paths))} files")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
