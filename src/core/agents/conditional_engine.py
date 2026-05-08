"""Minimal expression evaluator for workflow step conditions.

Supports ``when:`` blocks in workflow YAML::

    when: "${extra.intent} != ''"
    when: "len(${gnd_entries}) > 0"
    when: "${extra.flag} == true"

Claude Generated
"""

import logging
import operator
import re
from typing import Any, Callable, Dict

from .context_path import resolve_path

logger = logging.getLogger(__name__)


# Comparison operators
_OPS: Dict[str, Callable[[Any, Any], bool]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}

# Supported "functions"
_FUNCS: Dict[str, Callable[[Any], Any]] = {
    "len": len,
    "bool": bool,
    "str": str,
    "int": int,
}


class ConditionalEngine:
    """Evaluate ``when:`` expressions against a SharedContext."""

    @staticmethod
    def evaluate(expression: str, context: Any) -> bool:
        """Evaluate an expression string.  Returns False on error."""
        if not expression or not expression.strip():
            return True  # No condition = always run

        try:
            return ConditionalEngine._eval(expression.strip(), context)
        except Exception as exc:
            logger.warning(f"ConditionalEngine: expression '{expression}' failed: {exc} — defaulting to False")
            return False

    @staticmethod
    def _eval(expr: str, context: Any) -> bool:
        # Handle ``and`` / ``or`` (simple left-to-right)
        if " and " in expr.lower():
            parts = re.split(r'\s+and\s+', expr, flags=re.IGNORECASE)
            return all(ConditionalEngine._eval_single(p.strip(), context) for p in parts)
        if " or " in expr.lower():
            parts = re.split(r'\s+or\s+', expr, flags=re.IGNORECASE)
            return any(ConditionalEngine._eval_single(p.strip(), context) for p in parts)
        return ConditionalEngine._eval_single(expr, context)

    @staticmethod
    def _eval_single(expr: str, context: Any) -> bool:
        expr = expr.strip()

        # Handle ``not ...`` prefix
        if expr.lower().startswith("not "):
            return not ConditionalEngine._eval_single(expr[4:].strip(), context)

        # Handle ``in`` / ``not in``
        match = re.match(r"(.+?)\s+in\s+(.+)", expr, re.IGNORECASE)
        if match:
            left_raw = match.group(1).strip()
            right_raw = match.group(2).strip()
            left = ConditionalEngine._resolve_value(left_raw, context)
            right = ConditionalEngine._resolve_value(right_raw, context)
            return left in right

        match = re.match(r"(.+?)\s+not\s+in\s+(.+)", expr, re.IGNORECASE)
        if match:
            left_raw = match.group(1).strip()
            right_raw = match.group(2).strip()
            left = ConditionalEngine._resolve_value(left_raw, context)
            right = ConditionalEngine._resolve_value(right_raw, context)
            return left not in right

        # Handle comparison operators
        for op_symbol, op_fn in sorted(_OPS.items(), key=lambda x: -len(x[0])):
            if op_symbol in expr:
                parts = expr.split(op_symbol, 1)
                if len(parts) == 2:
                    left = ConditionalEngine._resolve_value(parts[0].strip(), context)
                    right = ConditionalEngine._resolve_value(parts[1].strip(), context)
                    return op_fn(left, right)

        # Bare value — truthiness
        val = ConditionalEngine._resolve_value(expr, context)
        return bool(val)

    @staticmethod
    def _resolve_value(raw: str, context: Any) -> Any:
        raw = raw.strip()

        # String literal
        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
            return raw[1:-1]

        # Number literal
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            pass

        # Boolean literal
        if raw.lower() == "true":
            return True
        if raw.lower() == "false":
            return False
        if raw.lower() == "none" or raw.lower() == "null":
            return None

        # Function call: len(${path}) or len(path)
        func_match = re.match(r"(\w+)\s*\(\s*(.+)\s*\)", raw)
        if func_match:
            func_name = func_match.group(1).lower()
            arg_raw = func_match.group(2).strip()
            arg_val = ConditionalEngine._resolve_value(arg_raw, context)
            if func_name in _FUNCS:
                try:
                    return _FUNCS[func_name](arg_val)
                except Exception:
                    return None
            logger.warning(f"ConditionalEngine: unknown function '{func_name}'")
            return None

        # ${path} placeholder
        if raw.startswith("${") and raw.endswith("}"):
            path = raw[2:-1]
            try:
                val = resolve_path(path, context)
                return val
            except Exception:
                return None

        # Bare path (no ${} wrapper) — try direct attribute access
        try:
            val = resolve_path(raw, context)
            return val
        except Exception:
            pass

        # Unknown — return as-is
        return raw
