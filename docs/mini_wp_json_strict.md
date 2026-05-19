# Mini-WP: JSON-Output-Strictness (Prompt + Parser)

**Trigger**: P-γ Live-Smoke (Cadmium_Cogito.json → `classification` cascade)
zeigte `selected_keywords=[]` aus `selection_chunks`-Step. Root-Cause-Analyse:
LLM-Output wrapped JSON in `<|begin_of_thought|>…<|begin_of_solution|>`-Tags,
Parser-Bug in `_extract_json` zog nested-Inner-Object statt Top-Level → leere
Output-Felder.

**Status**: ADD. Out-of-Scope für P-γ.

## Symptome (vom Live-Run)

* `selection_chunks chunked: 1540 items × 5 chunks` ran → `selected_keywords: []`.
* `selection`-LLM-Output enthält `<|begin_of_thought|>…<|end_of_thought|>` +
  `<|begin_of_solution|>{json}<|end_of_solution|>` Fence-Marker.
* Downstream-Steps (selection → dk_collect → classification) leer.

## Doppelpfad-Fix

### Track A — Prompt-Hardening

**Ziel**: LLM darf NICHT mehr in `<|...|>`-Marker oder Markdown-Fences wrappen.

**Dateien**:
- `prompts.json` — alle Tasks (initialisation, keywords, dk_classification,
  rvk_classification, etc.) mit `<|begin_of_thought|>`/`<|begin_of_solution|>`
  Instruktionen ([grep](#grep) zeigt aktuell ≥6 Stellen).
- `workflows/alima_classic.yaml:235-345` — Inline-Prompt für `classification`-Step.
- `workflows/alima.yaml:266+, 432+` — analog.

**Änderungen**:
1. Entferne Instruktion „Reasoning in `<|begin_of_thought|>`".
2. Ersetze Output-Format-Block durch:
   > "**Output**: Ein einziges valides JSON-Objekt. Keine Markdown-Fences,
   > keine `<|...|>`-Marker, keine Erläuterung. Direkt mit `{` beginnen."
3. Lasse Few-Shot-Beispiele *plain JSON* statt fence-wrapped.

**Risiko**: Modelle, die per Default Thought-Tokens emittieren (DeepSeek-R1,
Qwen-QwQ), umgehen den Tag-Verbot via internem Reasoning. Akzeptabel —
Reasoning bleibt im Hidden-Thinking-State, nicht im Response-Body.

**Tests**: `tests/test_prompts_json_format.py` (neu) — pro Task die finale
Prompt-Rendering enthält weder `<|begin_` noch ```` ```json ````.

### Track B — Parser-Robustheit

**Datei**: `src/core/agents/steps/llm_agent_step.py:395-428` (`_extract_json`).

**Aktueller Code**:
```python
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)

def _extract_json(content: str) -> Dict[str, Any]:
    if not content:
        return {}
    m = _JSON_BLOCK_RE.search(content)        # ← non-greedy → nested-inner
    if m:
        try:
            obj = json.loads(m.group(1))
            …
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", content, re.DOTALL))):
                                                # ← [^{}] → keine Nested
        …
    return {}
```

**Bugs**:
1. **Non-greedy in Fence**: `\{.*?\}` matched kleinstes `{…}` → bei nested
   JSON wie `{"keywords": [{"k": "v"}]}` zieht `{"k": "v"}` statt outer.
2. **Fallback-Regex** `\{[^{}]*\}` schließt nested explizit aus → schlägt
   bei realistischem Output fehl.
3. **Kein Marker-Strip**: `<|begin_of_solution|>{json}<|end_of_solution|>`
   wird nicht gestripped vor Parse-Versuch → Fence-Match scheitert
   (kein ``` ``` `).

**Neue Implementierung** (Skizze):
```python
_THOUGHT_RE = re.compile(
    r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", re.DOTALL
)
_SOL_RE = re.compile(
    r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL
)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

def _balanced_object(text: str) -> Optional[str]:
    """Find first balanced top-level {…} object via brace-counting."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False; continue
        if c == "\\" and in_str:
            esc = True; continue
        if c == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

def _extract_json(content: str) -> Dict[str, Any]:
    if not content:
        return {}
    # 1. Strip thought blocks entirely.
    content = _THOUGHT_RE.sub("", content)
    # 2. Prefer explicit <|begin_of_solution|> payload.
    m = _SOL_RE.search(content)
    if m:
        content = m.group(1)
    # 3. Unwrap markdown fences.
    m = _FENCE_RE.search(content)
    if m:
        content = m.group(1)
    # 4. Balanced-brace extraction.
    chunk = _balanced_object(content)
    if chunk:
        try:
            obj = json.loads(chunk)
            return obj if isinstance(obj, dict) else {"items": obj}
        except json.JSONDecodeError:
            pass
    return {}
```

**Tests**: `tests/test_extract_json.py` (neu, 5–7 Cases):
* Plain JSON object → direkter Parse.
* JSON in `<|begin_of_solution|>{…}<|end_of_solution|>` → unwrap + parse.
* JSON in ```` ```json {…} ``` ```` → unwrap + parse.
* Combined: Thought + Solution + Fence verschachtelt.
* Nested object mit inner-array `{"keys":[{"k":"v"}]}` → outer komplett.
* Pre-Fix-Regression-Case (Cadmium-Output) → keywords ≥ 1.
* Garbage / kein JSON → `{}`.

## Deliverables

| ID | Was | Dateien | Aufwand |
|---|---|---|---|
| **A.1** | prompts.json Hardening | `prompts.json` (≥6 Tasks) | 1 PT |
| **A.2** | YAML-Inline-Prompts Hardening | `workflows/alima_classic.yaml`, `workflows/alima.yaml` | 0.5 PT |
| **A.3** | Prompt-Format-Test | `tests/test_prompts_json_format.py` | 0.5 PT |
| **B.1** | `_extract_json` rewrite | `src/core/agents/steps/llm_agent_step.py` | 1 PT |
| **B.2** | Unit-Tests Parser | `tests/test_extract_json.py` | 0.5 PT |
| **B.3** | Cadmium-Regression-Smoke | manuell via SingleStepDialog | 0.5 PT |

**Total**: 4 PT.

## Reihenfolge

B vor A — Parser-Fix deckt auch Modelle ab, die trotz Prompt-Verbot Tags
emittieren. A reduziert Surface, B fixt Root-Cause.

## Smoke-Test

Nach B.1: Same Cadmium-Cascade aus P-γ-Live-Smoke wiederholen →
`selection_chunks.selected_keywords` muss ≥ 5 Einträge mit `gnd_id` haben.
Bei A.1+A.2: LLM-Output sollte direkt mit `{` beginnen (visuell via
SingleStepDialog Stream-Panel).

## Roll-Back-Tag

`mini-wp-json-strict-pre` vor Start.

## Grep

```bash
grep -c "begin_of_thought\|begin_of_solution" prompts.json
# Erwartet: ≥ 6 Treffer

grep -c "begin_of_thought\|begin_of_solution" workflows/alima*.yaml
# Erwartet: ≥ 4 Treffer
```

## Out-of-Scope

* XML-Output-Modus (legacy, `PromptConfigData.output_format == "xml"`) bleibt
  unverändert — separater Pfad in `processing_utils.py`.
* Per-Modell-Output-Schema-Validation (eigene WP, low priority).
