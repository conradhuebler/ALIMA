# Provider/Modell-Portabilität (WP11)

**Status**: Faktenbasis-Dokument für T2-Decision **„Capability-Schema
fix"** und T3-Decision **„Test-Matrix-Scope"**. Output von WP11 aus
[`wp_detailed_plans.md`](wp_detailed_plans.md).

**Methode**: Code-Inspektion (`LlmService`, `model_capabilities`,
`prompts.json`, `AgentLoop`, `prompt_service`) + Cross-Read 2025-
Provider-Strategy-Docs.

**Querverweise**:
- [`audit_tab_inventory.md`](audit_tab_inventory.md) (WP1)
- [`research_classic_vs_agentic.md`](research_classic_vs_agentic.md)
  (WP2) — Seed-Lücke verifiziert
- [`workflow_output_schemas.md`](../workflow_output_schemas.md) (WP3) —
  Slot-Vokabular, prompts.json-Status
- [`frontend_tier_model.md`](frontend_tier_model.md) (WP9)
- [`wp_detailed_plans.md`](wp_detailed_plans.md) WP11 — Soll-Definition
- **2025-Vor-Analyse** (Status revidiert):
  [`provider_strategy_summary.md`](provider_strategy_summary.md),
  [`provider_strategy_analysis.md`](provider_strategy_analysis.md),
  [`provider_strategy_technical_spec.md`](provider_strategy_technical_spec.md),
  [`provider_strategy_migration_guide.md`](provider_strategy_migration_guide.md)
  — Code-Inventar gültig, Empfehlungen revidiert (siehe Sektion 12).

## 1. Executive Summary

- 4 Provider-Typen, **3 aktiv** im Operator-Setup (Anthropic config-leer).
- Multi-Provider-Strategie über **prompts.json multi-variant** +
  **neue Capability-YAML** `config/model_capabilities.yaml`.
- **Selector bleibt 3-Tier** (exakt → strip-version → default). Kein
  Pattern-Matching → Modellnamen müssen explizit in `models:`-Liste
  pro Variante stehen.
- **4 Familien minimal**: `thinking`, `instruct-open`, `openai-chat`,
  `default`.
- **Seed-Lücke in 7+1 Stellen** blockiert agentic-Reproduzierbarkeit
  (Forschungspfad WP2 Sektion 3). Retrofit-Spec liefert ~5h Aufwand.
- **Test-Matrix**: 3 Provider × 3 Workflows × 4 Checks = 36 Tests
  (24 CI + 12 Manual wegen GWDG-Uni-Login).
- **2025-Docs**: Code-Inventar bleibt; Empfehlungs-Teil revidiert.
  Banner in den 4 alten Docs nach WP11-Approval.

## 2. Status-Quo-Inventar Provider-Abstraktion

| Aspekt | Status | Code-Pointer |
|---|---|---|
| 4 Provider-Generators | ✓ | `llm_service.py` (gemini, anthropic, openai_compatible, ollama-Konfigs) |
| `generate_response()` Dispatch | ✓ | `llm_service.py:788` |
| `generate_with_tools()` Dispatch | ✓ (aber kein seed) | `llm_service.py:2428` |
| 4 Tool-Use Native Handler | ✓ | `llm_service.py:2501, 2584, 2678, 2772` |
| Seed in `generate_response` | ✓ 3/4 Provider (Anthropic-API kennt keinen seed) | OpenAI Z. 1474, Ollama HTTP 1599, Ollama Native 1717, Gemini 1850 |
| **Seed in `generate_with_tools`** | **✗ Param fehlt** | `llm_service.py:2428` |
| JSON-Mode Native | nur OpenAI (`response_format`) | `llm_service.py:1479` |
| Capability-Patterns (Chunking) | 15+ Regex | `src/utils/model_capabilities.py` |
| Multi-Variant Prompts | 2/10 Tasks (`dk_list`, `image_text_extraction`) | `prompts.json` |
| Prompt-Selector | 3-Tier (exakt → strip-version → default) | `src/llm/prompt_service.py:77, 140` |
| Repetition-Detector | global Config | `src/utils/repetition_detector.py:34-59` |
| Thinking-Markup Strip (`<|begin_of_thought|>`) | aktiv | `processing_utils.py:157, 193`, `json_response_parser.py:38` |
| AgentLoop Tool-Use | unified Schema, repeat_threshold=3 | `agent_loop.py:33, 76, 104` |

## 3. Provider-Capability-Schema (T2-Deliverable)

### Format

YAML in neuer Datei `config/model_capabilities.yaml`. 2-Ebenen:
`providers.<provider_type>.<model_pattern>.<flags>`. Operator-pflegbar
ohne Code-Deploy. Code-Patterns in `model_capabilities.py` bleiben
als Fallback wenn YAML fehlt.

### Flags (10 pro Modell-Eintrag)

| Flag | Typ | Bedeutung |
|---|---|---|
| `json_mode` | bool | unterstützt `response_format={type: json_object}` |
| `tool_use` | enum: `none` / `text_fallback` / `native` / `parallel_native` | Tool-Calling-Fähigkeit |
| `vision` | bool | Multimodal-Image-Input |
| `max_context_tokens` | int | Kontext-Limit |
| `seed_support` | bool | API-Param `seed` (Anthropic = false) |
| `streaming` | bool | Token-Stream |
| `thinking_tokens` | bool | erzeugt `<|begin_of_thought|>`-Markup |
| `parallel_tool_calls` | bool | mehrere Tools pro Turn |
| `system_prompt` | bool | dedizierter System-Slot |
| `family` | enum: `thinking` / `instruct-open` / `openai-chat` / `default` | Familien-Bucket (Doku, nicht Selector-Input) |

### Schema-Skizze

```yaml
providers:
  openai_compatible:
    gpt-4*:
      json_mode: true
      tool_use: parallel_native
      vision: true
      max_context_tokens: 128000
      seed_support: true
      streaming: true
      thinking_tokens: false
      parallel_tool_calls: true
      system_prompt: true
      family: openai-chat

    qwen2.5*:
      json_mode: false
      tool_use: native
      vision: false
      max_context_tokens: 32000
      seed_support: true
      streaming: true
      thinking_tokens: true
      parallel_tool_calls: false
      system_prompt: true
      family: thinking

    gemma-3*:
      json_mode: false
      tool_use: native
      vision: false
      max_context_tokens: 8000
      seed_support: true
      streaming: true
      thinking_tokens: false
      parallel_tool_calls: false
      system_prompt: true
      family: instruct-open

  ollama:
    cogito:*:
      family: thinking
      thinking_tokens: true
      seed_support: true
      tool_use: native
      max_context_tokens: 32000

    llama3.1:*:
      family: instruct-open
      thinking_tokens: false
      seed_support: true
      tool_use: native
      max_context_tokens: 128000

  gemini:
    gemini-2.5*:
      json_mode: false
      tool_use: native
      vision: true
      seed_support: true
      family: default
```

### Population Operator-Setup

- 3 Provider × ~5 Modelle = ~15 explizite Einträge
- 4-6 Familien-Wildcards (`qwen*`, `llama*`, `gemma*`, `gpt-4*`,
  `deepseek*`, `cogito:*`)
- **Aufwand Initial-Population**: 2-3h (Provider-Docs lesen)

### Code-Integration

- Erweitere `model_capabilities.py` um YAML-Loader (Read + Lookup)
- Bestehende Regex-Patterns = Fallback wenn YAML-Eintrag fehlt
- Konsumenten: `LlmService` (Tool-Use-Fallback-Entscheidung), `Pipeline-
  StepExecutor` (Chunking-Threshold), Seed-Retrofit-Sonderfall
  (`seed_support`-Check)

## 4. Familien-Map (4 Familien minimal)

| Familie | Member-Beispiele | Charakteristik | Anthropic? |
|---|---|---|---|
| `thinking` | qwen2.5+, deepseek-r1, cogito:*, magistral | erzeugt `<|begin_of_thought|>`, longer-reasoning, JSON-Output toleriert Pre-Text | nein |
| `instruct-open` | llama3.x, gemma-3, mistral-7b, mixtral-8x7b | knappe Instructions, kein thinking, direkter JSON-Output | nein |
| `openai-chat` | gpt-4*, gpt-5*, gpt-3.5* | function-calling first-class, `response_format` support | nein |
| `default` | Fallback | generisch ohne Familien-Optimierung | ja |

**Anthropic-Sonderfall**: kein api_key in config.json → Familie
existiert nicht initial. Bei späterer Aktivierung neue Familie `claude`
(XML-Tag-Präferenz) hinzufügen.

## 5. Prompt-Varianten-Strategie

### Heute
2/10 Tasks Multi-Variant. 8 Tasks haben nur `["default"]`.

### WP11-Ziel
Varianten **nur wo Mehrwert**, nicht 30+ Pflicht-Varianten.

Priorisierung:

| Priorität | Task × Familien-Split | Neue Varianten |
|---|---|---|
| 1 | `keywords` × {thinking, instruct-open} | +2 |
| 2 | `dk_classification` × {thinking, instruct-open} | +2 |
| 3 | `initialisation` × {thinking, instruct-open} | +2 |
| 4 | `keywords` + `dk_classification` × `openai-chat` (function-calling-Hint) | +2 |
| 5 | `dk_list` heute 2 Varianten, ggf. `instruct-open` ergänzen | +1 |

**Realistic: ~9 neue Varianten** auf bestehende ~2 → ~11 total.
Pflege-Aufwand: 30-60 min pro Variante (Tuning + Vergleichs-Run gegen
Default).

### Konsequenz aus 3-Tier-Selector

Jede Variante muss alle zugehörigen Modellnamen **explizit** in
`models:` listen. Familie ist semantisches Konzept in prompts.json,
**nicht** Selector-Input.

Beispiel `keywords`-Task (vereinfacht):
```json
{
  "task": "keywords",
  "prompts": [
    ["thinking-prompt...", "system-thinking...", 0.5, 0.9,
     ["qwen2.5:14b", "qwen2.5:32b", "cogito:14b", "cogito:32b",
      "deepseek-v3.1:671b"], 0],
    ["instruct-prompt...", "system-instruct...", 0.5, 0.9,
     ["llama3.1:8b", "llama3.1:70b", "gemma-3-27b-it",
      "mistral-7b-instruct"], 0],
    ["openai-chat-prompt...", "system-openai...", 0.5, 0.9,
     ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], 0],
    ["default-prompt...", "system-default...", 0.5, 0.9,
     ["default"], 0]
  ]
}
```

**Pflegeaufwand bei neuer Modell-Version**: prompts.json Modellnamen-
Liste in passender Variante ergänzen — kein Code-Change. Operator-
pflegbar.

## 6. Selector-Status (3-Tier bleibt)

Operator-Entscheidung: kein Ausbau.

1. **Exakt**: `model in self.models_by_task[task]` (`prompt_service.py:77`)
2. **Strip-Version**: `cogito:14b` → `cogito` lookup
   (`prompt_service.py:140`)
3. **Default**: `["default"]` Fallback

**Konsequenz für WP11**: Capability-YAML `family:`-Flag dient nur als
Dokumentation/Test-Matrix-Bucket, nicht als Selector-Input. Pattern-
Matching-Erweiterung ist nicht WP11-Scope. Falls künftig benötigt:
separate WP.

**Risiko**: bei neuem Modell-Namen ohne Eintrag in prompts.json oder
ohne `default`-Variante → kein Hit, Run schlägt fehl. **Mitigation**:
jeder Task MUSS `default`-Variante haben (heute 10/10 erfüllt).

## 7. Tool-Use Audit (Befund-Tabelle)

| Provider-Typ | Tool-Use-Pfad | Status | Befund |
|---|---|---|---|
| `openai_compatible` (OpenAI direkt) | native (`llm_service.py:2584`) | ✓ getestet | `{type: function, function: {...}}` |
| `anthropic` | native (Z. 2678) | unbenutzt | `input_schema`-Format, Operator skip |
| `gemini` | native (Z. 2772) | ✓ getestet | `function_declarations`-Format |
| `ollama` | native (Z. 2501) | ✓ getestet | ab qwen2.5/llama3.1 |
| GWDG (via `openai_compatible`) | native-Pfad | **UNGETESTET** | → Test-Matrix-Pflicht F1 |
| Text-Fallback | Z. 2868 | ✓ | JSON-String-Parsing für Legacy |

**Tool-Schema-Abstraktion**: ✓ generic via `tool_registry.get_tool_schemas()`
(`agent_loop.py:76`), Provider-spezifische Konvertierung in
LlmService-Handlern.

**JSON-Mode-Lücke**: nur OpenAI setzt `response_format`. Ollama/Gemini/
Anthropic verlassen sich auf Prompt-Hint + Tolerant-Parser
([`json_response_parser.py`](../src/core/json_response_parser.py)).
**Akzeptabel heute**, kein Bau-Anlass.

## 8. Seed-Retrofit-Spec (7 + 1 Stellen)

**Status**: agentic nicht reproduzierbar (WP2 Sektion 3). WP11 liefert
Spec, **Code-Umsetzung gehört zu WP10**.

| # | Datei:Symbol | Änderung | Aufwand |
|---|---|---|---|
| 1 | `LlmService.generate_with_tools()` Signatur (`llm_service.py:2428`) | `seed: Optional[int] = None`-Param | 15 min |
| 2 | `_generate_ollama_native_with_tools()` (`llm_service.py:2501`) | `seed` in `options`-Payload | 30 min |
| 3 | `_generate_openai_with_tools()` (`llm_service.py:2584`) | `seed` in `params` | 30 min |
| 4 | `_generate_anthropic_with_tools()` (`llm_service.py:2678`) | Capability-Check `seed_support: false` → warn + skip; sonst temperature=0 | 30 min |
| 5 | `_generate_gemini_with_tools()` (`llm_service.py:2772`) | `seed` in `generation_config` | 30 min |
| 6 | `BaseSharedContext` + `LLMAgentStep._llm_params` + `AgentLoop.run()` | `seed`-Feld + Propagation (`agent_loop.py:104` benötigt neuen Param) | 1h |
| 7 | `shared_context.py:300/321/339` hartcodiert `seed=None` in `LlmKeywordAnalysis` | aus Context lesen | 30 min |
| (8) | `workflows/*.yaml` Schema: `seed:`-Feld in `llm:`-Block | YAML-Spec + Validator-Update | 30 min |

**Total**: ~4h Code + 1h Tests = **~5h Aufwand**.

**Anthropic-Sonderfall**: API kennt keinen `seed`. Capability-Flag
`seed_support: false` → Service warnt + fällt auf `temperature=0`
zurück (heute Praxis).

## 9. Test-Matrix (T3-Deliverable)

YAML-Spec im Doc. Implementation = WP10.

```yaml
test_matrix:
  providers:
    - name: openai-api
      ci_eligible: true
      env_key: OPENAI_API_KEY

    - name: ollama-local
      ci_eligible: true
      service: localhost:11434

    - name: gwdg
      ci_eligible: false   # Uni-Login, Manual-Run
      env_key: GWDG_API_KEY

  workflows:
    - alima_classic.yaml
    - catalog_search.yaml
    - title_list_search.yaml

  checks:
    - smoke              # end-to-end läuft, exit 0
    - reproducibility    # gleicher seed → gleiches Output
                         # (skip wenn !seed_support)
    - json_robustness    # json_response_parser akzeptiert Output
    - tool_use           # mind. 1 Tool-Call pro Workflow erfolgreich
```

**Matrix-Größe**: 3 × 3 × 4 = **36 Tests**.
- 24 CI-Tests (OpenAI + Ollama, 2 × 3 × 4)
- 12 Manual-Tests (GWDG, 1 × 3 × 4)

**CI-Eignung**: OpenAI (API-Key in Secrets) + Ollama-local (Service in
CI-Image) ja. GWDG-Uni-Login nein → Manual-Run-Doku im Test-Repo.

**Auswerter-Skript**: TODO WP10. WP11 liefert nur YAML-Spec.

## 10. Per-Step-Provider-Mix-UI (Vorschlag)

**Heute**: `--provider`/`--model` CLI + `global_override_combo` GUI.
Per-Step nur im YAML.

**WP11-Vorschlag** (Implementation = WP10):

| Option | Beschreibung | Aufwand |
|---|---|---|
| A: UI-Combo pro Step | `PipelineConfigDialog` Erweiterung um Per-Step-Combo | ~4h GUI |
| B: Workflow-Variante | `alima_premium.yaml` / `alima_cheap.yaml` mit hardcoded providers | 30 min Copy+Tune (pro Variante) |
| **A+B kombiniert (Empfehlung)** | UI überschreibt Workflow-Default | ~4h |

**Use-Case**: „cheap-extraction + premium-classification" — Operator
kombiniert Modelle pro Step. Operator-Frage F5 entscheidet.

## 11. Chat-Provider-Default (Querverweis WP7/WP8)

Vorschlag: neues Config-Feld
```json
{
  "chat": {
    "default_provider": "openai_compatible",
    "default_model": "gpt-4o-mini"
  }
}
```

**Begründung**: Chat = Multi-Turn = viele Calls → klein+schnell wegen
Kosten/Latenz. Kandidaten:
- **Lokal**: Ollama `llama3.1:8b` (kein Cloud-Cost, ggf. langsamer)
- **Cloud-mini**: `gpt-4o-mini` (schnell, kostenpflichtig)

Operator-Frage F4 entscheidet. WP7 (Chat-Tools) + WP8 (Chat-UI)
konsumieren diesen Default.

## 12. Verhältnis zu 2025-Provider-Strategy-Docs

`provider_strategy_*.md` (4 Docs, 951 Z., Stand 2025-09):

| Doc | Code-Inventar (gültig) | Empfehlungen (revidiert) |
|---|---|---|
| `summary.md` (114 Z.) | Status-Übersicht | Family-Recognition entfernen → **revidiert**: WP11 baut sie aus (über prompts.json + Capability-YAML, nicht über alten `_match_model_family`-Code) |
| `analysis.md` (294 Z.) | 4-Tier-Selector-Analyse | 2-Tier-Vereinfachung → **partiell valide**: heute 3-Tier (`prompt_service.py:77+140`), bleibt so (Operator-Entscheidung WP11) |
| `technical_spec.md` (291 Z.) | Code-Methoden 2025 | Simplification-Spec → **veraltet**: ersetzt durch Capability-YAML-Pfad (Sektion 3) |
| `migration_guide.md` (272 Z.) | User-Migration 2025 | Fuzzy-Matching entfernt-Hinweis → **veraltet**: `_strip_version_tag` existiert (April 2026 Compromise) |

**Action nach WP11-Approval**:
- Banner als erste Sektion in jedem der 4 alten Docs:
  > **Status**: Stand 2025-09. Code-Inventar gültig.
  > **Empfehlungen revidiert** durch [`provider_portability_design.md`](provider_portability_design.md)
  > (2026-05, WP11). Multi-Provider-Strategie reaktiviert.

## 13. Operator-Fragen (max 6)

### F1: GWDG Tool-Use — je benutzt oder nur Text-Calls?
**Empfehlung**: Operator-Aussage. Falls nie genutzt → Test-Matrix-
Tool-Use-Check für GWDG geringe Priorität. Falls ja → Pflicht-Test.

### F2: Welche Workflows haben Priorität für Reproducibility (Seed-Retrofit Reihenfolge)?
**Empfehlung**: `alima_classic.yaml` zuerst (Forschungspfad, WP2),
dann andere Workflows nach Bedarf. WP10-Migration-Reihenfolge.

### F3: Anthropic ganz raus (kein api_key) oder als Stub-Familie halten?
**Empfehlung**: Skip in WP11. Falls Operator später aktiviert →
eigene Familie `claude`. Heute kein Aufwand.

### F4: Chat-Default — lokal (Ollama) oder Cloud (gpt-4o-mini)?
**Empfehlung**: Ollama `llama3.1:8b` für Privacy + 0-Cost. Cloud-
Fallback wenn lokale Latenz inakzeptabel. Konfigurierbar bleibt
beides.

### F5: Per-Step-UI — A (Dialog) oder B (Workflow-Variante) oder A+B?
**Empfehlung**: A+B. UI für Ad-hoc, Workflow-Variante für gespeicherte
Presets. Implementation Reihenfolge: erst B (30 min), dann A nach
Bedarf.

### F6: Capability-YAML — Operator-pflegbar oder Code-Embed mit YAML als Override?
**Empfehlung**: Operator-pflegbar in `config/model_capabilities.yaml`,
Code-Patterns in `model_capabilities.py` als Fallback. Single-Source-
of-Truth pro Modell ist YAML.

## 14. Decision-Points T2 + T3

### T2: Capability-Schema fixiert
WP11 liefert:
- YAML-Format `config/model_capabilities.yaml` mit 10 Flags pro Modell
  (Sektion 3)
- 4 Familien-Map (Sektion 4)
- Population-Skizze für Operator-Setup (3 Provider × ~5 Modelle)
- Code-Integration via erweitertem `model_capabilities.py`-Loader,
  Fallback auf bestehende Regex-Patterns

### T3: Test-Matrix-Scope fixiert
WP11 liefert:
- 3 Provider × 3 Workflows × 4 Checks = 36 Tests
- 24 CI-Tests + 12 Manual-Tests (GWDG)
- Implementation = WP10 (Auswerter-Skript)

## 15. Querverweise / Folge-WPs

| Folge-WP | Konsumiert aus WP11 |
|---|---|
| **WP4** (Renderer-Registry) | Tool-Use-Audit Sektion 7 — Renderer ggf. provider-abhängig bei Tool-Output-Format |
| **WP7** (Chat-Tools) | Sektion 11 Chat-Provider-Default + Capability-Schema für Provider-Filter |
| **WP8** (Chat-UI) | Sektion 11 Default-Auflösung |
| **WP10** (Migration) | Sektion 8 Seed-Retrofit-Spec + Sektion 9 Test-Matrix + Sektion 10 Per-Step-UI |
| **WP3** (Output-Schemas) | rückwärts: WP11-Test-Matrix-Check `json_robustness` testet Schema-Robustheit pro Provider |
| **WP2** (Classic↔Agentic) | WP11-Seed-Retrofit löst WP2-Reproduzierbarkeits-Blocker für agentic |

## 16. Risiken + offene Validierungen

- **N×M-Varianten-Pflege**: ~11 Varianten realistisch, 30 wäre zu viel.
  Operator-Disziplin nötig.
- **Provider-API-Drift**: historisch (Anthropic-Format-Migration,
  OpenAI-Tools v1→v2). Capability-YAML muss Operator nachhalten.
- **GWDG-Uni-Login-CI-Lücke**: 12/36 Tests manual.
- **Capability-YAML Drift mit Code-Defaults**: Validator-Skript nötig
  (TODO WP10).
- **Seed-Retrofit 7+1-Stellen + Anthropic-Sonderfall**: ~5h Aufwand,
  gehört zu WP10.
- **Selector bleibt 3-Tier**: bei neuem Modell-Namen ohne Eintrag in
  prompts.json oder ohne `default`-Variante → kein Hit. Mitigation:
  jeder Task MUSS `default` haben (heute 10/10 erfüllt).
- **Tool-Use auf GWDG ungetestet**: WP11 dokumentiert, Test-Matrix
  klärt.

## 17. Status

✅ WP11 abgeschlossen. Pendend: Operator-Antworten F1-F6, dann T2
(Capability-Schema) + T3 (Test-Matrix) final.

**Damit alle 5 T0/T1-Audit-WPs abgeschlossen**: WP1, WP2, WP3, WP9, WP11.
Nächste Phase: T2-Decisions (Capability-Schema, EventBus, Renderer-
Plugin) → WP4 + WP6 startbar.
