"""Mini-WP A.1: harden prompts.json — drop <|begin_of_thought|>/<|begin_of_solution|>
markers from user + system prompts.

Strategy per string:
1. Drop balanced ``<|begin_of_thought|>…<|end_of_thought|>`` blocks (DOTALL,
   non-greedy). When the two markers do NOT properly enclose a *protocol
   block* but instead appear as inline prose references (e.g.
   "Reasoning in `<|begin_of_thought|>`, Ergebnis in `<|begin_of_solution|>`"),
   the non-greedy span would over-match. We detect that case and strip the
   inline marker tokens individually.
2. Drop solution wrappers: ``<|begin_of_solution|>`` and ``<|end_of_solution|>``
   markers (keep the JSON payload between them).
3. Strip stray surrounding ```` ```json …``` ```` fences left over after the
   solution-wrapper removal, replacing them with their JSON body.
4. Append a strict instruction at the end of user prompts.

Idempotent: re-running produces no changes once markers are gone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROMPT_PATH = Path("prompts.json")

THOUGHT_BLOCK_RE = re.compile(
    r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>\s*", re.DOTALL
)
SOLUTION_WRAPPER_RE = re.compile(
    r"<\|begin_of_solution\|>\s*(.*?)\s*<\|end_of_solution\|>", re.DOTALL
)
INLINE_MARKER_RE = re.compile(
    r"`?<\|(?:begin|end)_of_(?:thought|solution)\|>`?"
)
FENCE_JSON_BLOCK_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", re.DOTALL
)

PLAIN_JSON_NOTE = (
    "\n\n**Output**: Genau ein valides JSON-Objekt. Keine Markdown-Fences, "
    "keine speziellen Marker-Tokens, keine Erläuterung außerhalb des JSON. "
    "Direkt mit `{` beginnen."
)


def _safe_strip_thought_block(text: str) -> str:
    """Strip thought blocks only when begin/end markers properly bracket content.

    A "proper bracket" means the number of opening and closing markers match
    AND each begin is followed by exactly one end before the next begin.
    """
    begins = text.count("<|begin_of_thought|>")
    ends = text.count("<|end_of_thought|>")
    if begins == 0 or ends == 0:
        return text
    if begins != ends:
        # Unbalanced — leave alone (inline-token cleanup will handle).
        return text
    # Iteratively remove the *innermost* well-formed block.
    out = text
    while True:
        m = re.search(
            r"<\|begin_of_thought\|>[^<]*?<\|end_of_thought\|>",
            out,
            re.DOTALL,
        )
        if not m:
            # No nested cases remain; try a broader pass that allows other
            # marker tokens inside but stops at the *first* end.
            m = re.search(
                r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>",
                out,
                re.DOTALL,
            )
            if not m:
                break
            # Sanity: rejected if the matched span itself contains another
            # `<|begin_of_thought|>` — that means non-greedy over-matched.
            if "<|begin_of_thought|>" in m.group(0)[len("<|begin_of_thought|>"):]:
                break
        out = out[: m.start()] + out[m.end():]
    return out


def harden(text: str, is_system: bool = False) -> str:
    if not isinstance(text, str):
        return text
    if "begin_of_thought" not in text and "begin_of_solution" not in text:
        return text

    out = _safe_strip_thought_block(text)
    out = SOLUTION_WRAPPER_RE.sub(lambda m: m.group(1), out)
    # Strip any remaining inline marker tokens (and their surrounding backticks).
    out = INLINE_MARKER_RE.sub("", out)

    # Unwrap any standalone fenced JSON blocks (keep body).
    out = FENCE_JSON_BLOCK_RE.sub(lambda m: m.group(1), out)

    # Tidy whitespace.
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = out.strip()

    if is_system:
        # Ensure system prompt mentions plain-JSON expectation once.
        if "JSON" not in out:
            out += " Antworte als plain JSON-Objekt ohne Fences oder Marker."
    else:
        out += PLAIN_JSON_NOTE
    return out


def main() -> None:
    data = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    changed = 0
    for task, body in data.items():
        if not isinstance(body, dict):
            continue
        prompts = body.get("prompts", [])
        if not isinstance(prompts, list):
            continue
        for variant in prompts:
            if not isinstance(variant, list):
                continue
            for idx in (0, 1):
                if idx >= len(variant) or not isinstance(variant[idx], str):
                    continue
                new = harden(variant[idx], is_system=(idx == 1))
                if new != variant[idx]:
                    variant[idx] = new
                    changed += 1

    PROMPT_PATH.write_text(
        json.dumps(data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"hardened {changed} prompt strings")


if __name__ == "__main__":
    main()
