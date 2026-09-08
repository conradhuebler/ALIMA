"""Salvage a model's JSON answer, and read a payload it kept outside the JSON.

Claude Generated.

Both helpers exist because of one observed failure. The reflection gate answers
in JSON (``status``/``action``/``reason``). When a personal rule additionally
asks it to produce a multi-line block — a catalogue entry, say — the model puts
that block into a JSON string with **raw** newlines. That is invalid JSON, and
the parser then returns nothing at all: not just the block is lost, the verdict
is too, and the run ends silently on the default ``action: "finish"``.

So:

* ``extract_tagged_block`` reads a payload the model was told to put *after* the
  JSON, in ``<name>…</name>``. Multi-line text outside JSON needs no escaping,
  which is the whole point.
* ``repair_json_newlines`` is the safety net for a model that ignores that and
  writes the block into the JSON anyway: it escapes the control characters that
  appear inside string literals, so at least the verdict survives.
* ``extract_json_object`` is the extraction both gates use, salvage included.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Control characters that are legal in a JSON document but not inside a string
#: literal. A model emitting a formatted block hits the first two constantly.
_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def repair_json_newlines(text: str) -> str:
    """Escape raw control characters that sit inside JSON string literals.

    Walks the text tracking whether it is inside a string, so newlines that
    format the document itself are left alone. Anything it cannot make sense of
    is returned unchanged — this is a salvage attempt, never a parser.
    """
    if not text:
        return text
    out = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            elif ch in _ESCAPES:
                out.append(_ESCAPES[ch])
                continue
        elif ch == '"':
            in_string = True
        out.append(ch)
    return "".join(out)


def extract_tagged_block(content: str, name: str) -> str:
    """Return the text inside ``<name>…</name>``, or ``""``.

    The closing tag may be missing when the model ran out of budget mid-block;
    in that case everything after the opening tag is taken, which is better than
    dropping an otherwise complete answer.
    """
    if not content or not name:
        return ""
    open_re = re.compile(rf"<{re.escape(name)}\s*>", re.IGNORECASE)
    m = open_re.search(content)
    if not m:
        return ""
    rest = content[m.end():]
    close = re.search(rf"</{re.escape(name)}\s*>", rest, re.IGNORECASE)
    body = rest[: close.start()] if close else rest
    return _strip_code_fence(body.strip())


def _strip_code_fence(text: str) -> str:
    """Drop a surrounding ``` fence — models like to wrap a block in one."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_object(content: str) -> Dict[str, Any]:
    """Best-effort extraction of one JSON object from a model answer.

    Order: fenced ``json`` block, then the last balanced ``{...}``, then the
    salvage pass of ``repair_json_newlines``. Returns ``{}`` when nothing
    parses.

    One implementation for the two gates that share the failure: ``MetaAgent``
    and ``ReflectionStep`` carried byte-identical copies, and the salvage step
    below had to be patched into both. - Claude Generated
    """
    if not content:
        return {}

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    for m in reversed(list(re.finditer(r"\{[^{}]*\}", content, re.DOTALL))):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and obj:
                return obj
        except json.JSONDecodeError:
            continue

    # Last resort: a model that wrote a formatted block into a JSON string left
    # raw newlines in it. Without this the whole verdict is lost — status,
    # action and reason with it — and the run ends on the default "finish" as if
    # nothing had happened.
    repaired = repair_json_newlines(content)
    if repaired != content:
        for pattern in (r"```(?:json)?\s*(\{.*?\})\s*```", r"(\{.*\})"):
            m = re.search(pattern, repaired, re.DOTALL)
            if not m:
                continue
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj:
                logger.warning("JSON answer had raw newlines inside a string — salvaged")
                return obj
    return {}
