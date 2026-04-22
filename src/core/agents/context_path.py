"""Context-Path Resolver for Workflow v4 - Claude Generated.

Resolves placeholder strings like ``${steps.extraction.keywords}`` against a
SharedContext (plus per-step results and the generic ``extra`` dict).

Supported roots:
    ${abstract}                      → SharedContext.abstract
    ${initial_keywords}              → SharedContext.initial_keywords
    ${working_title}                 → SharedContext.working_title
    ${extracted_keywords}            → SharedContext.extracted_keywords
    ${gnd_entries}                   → SharedContext.gnd_entries
    ${selected_keywords}             → SharedContext.selected_keywords
    ${dk_classifications}            → SharedContext.dk_classifications
    ${missing_concepts}              → SharedContext.missing_concepts
    ${steps.<step_id>[.field[...]]}  → SharedContext.step_results[<id>][...]
    ${extra.<key>[...]}              → SharedContext.extra[<key>][...]
    ${input.<key>}                   → same as ${extra.input.<key>}; see note

Dotted access traverses dicts; integer segments index into lists.

Unknown paths raise ``KeyError`` — callers should catch and fall back.
"""

from __future__ import annotations

import re
from typing import Any

PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")

# Roots that map to a typed attribute on SharedContext.
_CONTEXT_ATTRS = {
    "abstract",
    "initial_keywords",
    "working_title",
    "extracted_keywords",
    "gnd_entries",
    "selected_keywords",
    "keyword_chains",
    "missing_concepts",
    "dk_classifications",
    "rvk_classifications",
    "dk_search_results",
    "provider",
    "model",
    "temperature",
    "max_tokens",
    "input_type",
    "source_value",
}


def _walk(obj: Any, segments: list[str]) -> Any:
    """Walk into ``obj`` following dotted segments.

    Lists accept integer segments (e.g. ``steps.search.gnd_entries.0.title``).
    Raises KeyError for any missing intermediate.
    """
    cur = obj
    for seg in segments:
        if cur is None:
            raise KeyError(f"path segment '{seg}' hit None")
        if isinstance(cur, list):
            try:
                idx = int(seg)
            except ValueError as e:
                raise KeyError(f"list requires integer index, got '{seg}'") from e
            if idx < 0 or idx >= len(cur):
                raise KeyError(f"list index {idx} out of range")
            cur = cur[idx]
            continue
        if isinstance(cur, dict):
            if seg not in cur:
                raise KeyError(f"dict has no key '{seg}'")
            cur = cur[seg]
            continue
        # Object: try attribute access
        if hasattr(cur, seg):
            cur = getattr(cur, seg)
            continue
        raise KeyError(f"cannot resolve '{seg}' on {type(cur).__name__}")
    return cur


def resolve_path(path: str, context: Any) -> Any:
    """Resolve a single path expression (without the ``${...}`` wrapper).

    Args:
        path: Dotted path, e.g. ``steps.extraction.keywords`` or ``abstract``.
        context: SharedContext instance.

    Returns:
        The resolved value (any type — str, list, dict, ...).

    Raises:
        KeyError: If the path cannot be resolved.
    """
    if not path:
        raise KeyError("empty path")

    parts = path.split(".")
    root = parts[0]
    rest = parts[1:]

    if root == "steps":
        if not rest:
            return context.step_results
        step_id, *inner = rest
        if step_id not in context.step_results:
            raise KeyError(f"unknown step '{step_id}'")
        return _walk(context.step_results[step_id], inner)

    if root == "extra":
        extra = getattr(context, "extra", None)
        if extra is None:
            raise KeyError("context has no 'extra' dict")
        if not rest:
            return extra
        return _walk(extra, rest)

    if root == "input":
        # alias: treat as extra.input.<...> OR top-level when no nested key
        extra = getattr(context, "extra", {}) or {}
        input_blob = extra.get("input", {})
        if not rest:
            return input_blob
        return _walk(input_blob, rest)

    if root in _CONTEXT_ATTRS:
        if not rest:
            return getattr(context, root)
        return _walk(getattr(context, root), rest)

    raise KeyError(f"unknown path root '{root}'")


def resolve_string(template: str, context: Any) -> str:
    """Replace all ``${...}`` placeholders in a string.

    Non-string values are serialised with ``str()`` — for lists/dicts that
    is typically ``repr``-like.  Use ``resolve_value`` when the caller
    wants the native type.

    Missing paths are replaced with an empty string and a warning is NOT
    raised — for robust prompt rendering.  Use ``resolve_value`` for
    strict lookups.

    Args:
        template: String with ``${path}`` placeholders.
        context: SharedContext instance.

    Returns:
        String with placeholders replaced.
    """
    def _sub(match: re.Match) -> str:
        path = match.group(1).strip()
        try:
            value = resolve_path(path, context)
        except KeyError:
            return ""
        if value is None:
            return ""
        if isinstance(value, (list, dict)):
            import json
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    return PLACEHOLDER_RE.sub(_sub, template)


def resolve_value(expression: str, context: Any) -> Any:
    """Resolve an expression that may be either a bare ``${path}`` or a template.

    - If the whole expression is a single ``${path}``, returns the raw value.
    - Otherwise resolves as a string (embedded placeholders).

    This lets workflow YAML use ``inputs: { keywords: "${steps.x.y}" }`` to
    pass the original list through, instead of stringified JSON.

    Args:
        expression: Either "${path}" or any string with placeholders.
        context: SharedContext instance.

    Returns:
        Native value for bare-placeholder expressions; string otherwise.
        Returns the expression as-is if it has no placeholders at all.
    """
    if not isinstance(expression, str):
        return expression
    stripped = expression.strip()
    m = PLACEHOLDER_RE.fullmatch(stripped)
    if m:
        try:
            return resolve_path(m.group(1).strip(), context)
        except KeyError:
            return None
    if "${" in expression:
        return resolve_string(expression, context)
    return expression


def resolve_mapping(mapping: dict, context: Any) -> dict:
    """Resolve all values in a dict of {name: "${path}" or template}.

    Args:
        mapping: Input mapping from workflow YAML.
        context: SharedContext instance.

    Returns:
        New dict with values replaced.
    """
    if not mapping:
        return {}
    return {k: resolve_value(v, context) for k, v in mapping.items()}
