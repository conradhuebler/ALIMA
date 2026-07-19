# Workflow-Output-Schema-Inventar (WP3)

**Status**: Faktenbasis-Dokument für T1-Entscheidung **„Render-Slot-
Vokabular fixiert"**. Output von WP3 aus
[`wp_detailed_plans.md`](legacy/wp_detailed_plans.md).

**Methode**: YAML-Inspektion aller 6 aktiven Workflows + Code-
Inspektion von SharedContext, register-fns, MCP-Tool-Registry und
JSONPath-Resolver.

**Querverweise**:
- [`audit_tab_inventory.md`](legacy/audit_tab_inventory.md) — Strukturvorbild.
- [`research_classic_vs_agentic.md`](legacy/research_classic_vs_agentic.md)
  — WP2-Output-Feld-Diff.
- [`wp_detailed_plans.md`](legacy/wp_detailed_plans.md) WP3 — Soll-Definition.
- [`agentic_workflow.md`](agentic_workflow.md),
  [`workflow_yaml_spec.md`](workflow_yaml_spec.md) — Workflow-Details.

## 1. Executive Summary

- **7 Workflows audit**: 6 aktive YAML + classic-Pipeline (Referenz).
- **30 SharedContext typed-fields** (10 Base, 20 ALIMA).
- **9 deterministic register-fns** + **17 MCP-Tools** (8 Knowledge,
  5 Library, 4 Pipeline-Result).
- **11 Render-Slot-Cluster** identifiziert. Namespace: `slot:xxx`.
- **Offen**: `catalog_hits` hat zwei inkompatible Schemata
  (`catalog_search` ≠ `title_list_search`). Operator-Frage F1.
- Schema-Versionierung heute: nur Workflow-Top-`version:`. Empfehlung
  für Code-side via register-fn-Metadata (Sektion 7).

## 2. Render-Slot-Vokabular (Master)

Slot-Namespace: `slot:<snake_case>` mit Doppelpunkt (kollisionsfrei mit
JSONPath `${...}` und Step-IDs). Erweiterung: neue Workflows dürfen
`slot:custom_*` deklarieren.

| Slot | Schema (Pseudo-TS) | Beispiel-Payload (gekürzt) | Frontend-Mapping (Qt / HTML / CLI) | Auftritt in Workflows |
|---|---|---|---|---|
| `slot:keyword_list` | `string[]` | `["Cadmium", "Ökotoxikologie"]` | KeywordChip-List / `<span class="chip">` / Comma-CSV | alima, alima_classic, classic |
| `slot:keyword_chains` | `{chain: string[], reason: string}[]` | `[{chain: ["Cadmium","Ökotoxikologie"], reason: "..."}]` | Tree-View / Nested-`<ul>` / Indent | alima, alima_classic |
| `slot:gnd_pool` | `{gnd_id: string, title: string, synonyms: string[], ddc?: string[], dk?: string[], count?: int}[]` | s. unten | Detail-Card-List / `<dl>` / TSV | alima, alima_classic, catalog_search, synonym_expansion (validate) |
| `slot:synonym_set` | `string[]` mit optional Relation | `["Cadmiumverbindungen", "Cd-Salze"]` | Inline-Chips / `<span>` / Comma-CSV | synonym_expansion |
| `slot:text_blob` | `string` | `"Pipeline analysiert Cadmium-Exposition..."` | QTextBrowser / `<p>` / Stdout | alima, alima_classic, title_list_search |
| `slot:dk_table` | `{keyword: string, dk: string, titles: string[], count: int, type: string}[]` | s. unten | QTableView / `<table>` / ASCII-Table | alima, alima_classic, classic |
| `slot:classification_list` | `{code: string, type: "DK"\|"RVK"}[]` | `[{code: "57.62", type: "DK"}]` | Code-Chips / `<code>` / Plain | alima, alima_classic, classic |
| `slot:title_list` | `{title: string, authors?: string[], year?: int, isbn?: string, dk_codes?: string[], rvk_codes?: string[]}[]` | s. unten | Biblio-Card / `<article>` / Bibliographic-CSV | title_list_search |
| `slot:duplicate_table` | `{input_title: string, status: "duplicate"\|"likely_duplicate"\|"different_edition"\|"new"\|"no_match", matches: any[], reasoning: string}[]` | s. unten | StatusColumn-Table / `<table>` / TSV mit Status-Spalte | title_list_search |
| `slot:metadata_record` | `Dict[gnd_id, {title, description, synonyms, ddcs}]` ODER `{title, gnd_id, description, synonyms, ddcs}` | `{"4007249-3": {title: "Cadmium", ...}}` | Key-Value-Form / `<dl>` / Key=Value Lines | batch_metadata, synonym_expansion (seed_entry) |
| `slot:raw_json` | `any` | `{tool_calls: 12}` | JSON-Viewer / `<pre>` / pretty-printed | Fallback aller Workflows |

**Beispiel-Payloads (ausführlich)**:

`slot:gnd_pool`:
```json
[{"gnd_id":"4007249-3","title":"Cadmium","synonyms":["Cd","Cadmium-Salze"],"ddc":["546.4"],"dk":["57.62"],"count":42}]
```

`slot:dk_table`:
```json
[{"keyword":"Cadmium","dk":"57.62","titles":["Schwermetalle in Böden","..."],"count":7,"type":"DK"}]
```

`slot:title_list`:
```json
[{"title":"Cadmium in soils","authors":["Müller, K."],"year":2024,"isbn":"978-3-...","dk_codes":["57.62"]}]
```

`slot:duplicate_table`:
```json
[{"input_title":"Cadmium in soils","status":"likely_duplicate","matches":[{"rsn":"123","title":"..."}],"reasoning":"ISBN match"}]
```

## 3. SharedContext-Feld-Master-Tabelle

Alle 30 typed-Felder ([`shared_context.py:108-133`](../src/core/agents/shared_context.py),
[`base_shared_context.py`](../src/core/agents/base_shared_context.py))
plus relevante `extra`-Felder, gemappt auf Slots.

### 3.1 Typed-Fields

| Feld | Quelle | Typ | Slot | Geschrieben in |
|---|---|---|---|---|
| `step_results` | Base | `Dict[str, Any]` | (intern) | alle |
| `quality_scores` | Base | `Dict[str, float]` | (intern) | reflection-aktive |
| `provider` | Base | `str` | (config) | alle |
| `model` | Base | `str` | (config) | alle |
| `temperature` | Base | `float` | (config) | alle |
| `max_tokens` | Base | `int` | (config) | alle |
| `verbose` | Base | `bool` | (config) | alle |
| `execution_history` | Base | `List[Dict]` | `slot:raw_json` | alle |
| `prompt_service` | Base | `Any` | (runtime) | alle |
| `extra` | Base | `Dict[str, Any]` | (siehe 3.2) | alle |
| `abstract` | ALIMA | `str` | `slot:text_blob` | alima, alima_classic |
| `initial_keywords` | ALIMA | `List[str]` | `slot:keyword_list` | alima, alima_classic |
| `input_type` | ALIMA | `str` | (meta) | alle |
| `source_value` | ALIMA | `Optional[str]` | (meta) | alle |
| `tool_result_cache` | ALIMA | `ToolResultCache` | (runtime) | alle |
| `conversation_memory` | ALIMA | `List[Dict]` | `slot:raw_json` | reflection-aktive |
| `working_title` | ALIMA | `str` | `slot:text_blob` | alima, alima_classic |
| `extracted_keywords` | ALIMA | `List[str]` | `slot:keyword_list` | alima, alima_classic (Step extraction) |
| `gnd_entries` | ALIMA | `List[Dict]` | `slot:gnd_pool` | alima, alima_classic, catalog_search (via extra), synonym_expansion (via extra) |
| `gnd_entries_per_keyword` | ALIMA | `Dict[str, List[str]]` | `slot:metadata_record` | alima, alima_classic |
| `selected_keywords` | ALIMA | `List[Dict]` | `slot:keyword_list` (flat) | alima, alima_classic (Step selection_chunks) |
| `keyword_chains` | ALIMA | `List[Dict]` | `slot:keyword_chains` | alima, alima_classic (Step selection) |
| `missing_concepts` | ALIMA | `List[str]` | `slot:keyword_list` | alima, alima_classic (Step selection) |
| `missing_concepts_searched` | ALIMA | `List[str]` | `slot:keyword_list` | reflection-driven |
| `dk_classifications` | ALIMA | `List[Dict]` | `slot:classification_list` | alima, alima_classic (Step classification) |
| `rvk_classifications` | ALIMA | `List[Dict]` | `slot:classification_list` | classification-Step optional |
| `dk_search_results` | ALIMA | `List[Dict]` | `slot:dk_table` | alima, alima_classic (Steps dk_collect, dk_postprocess) |
| `dk_catalog_stats` | ALIMA | `Dict[str, Any]` | `slot:raw_json` | classic-only (dk_statistics) |
| `quality_report` | ALIMA | `Dict[str, Any]` | `slot:raw_json` | reflection-aktive |
| `max_missing_reruns` | ALIMA | `int` | (config) | reflection-aktive |

### 3.2 Extra-Felder (Workflow-spezifisch)

| Extra-Pfad | Typ | Slot | Workflow |
|---|---|---|---|
| `extra.final_keywords` | `List[Dict: keyword, gnd_id]` | `slot:keyword_list` (flat) | alima, alima_classic |
| `extra.dk_entries` | `List[Dict]` | `slot:dk_table` | alima, alima_classic |
| `extra.dk_prompt_text` | `str` | `slot:text_blob` | alima, alima_classic |
| `extra.catalog_hits` (GND-enriched) | `List[Dict: title, gnd_ids, ddc_codes, dk_codes]` | `slot:gnd_pool` ⚠️ siehe Sektion 6 | catalog_search |
| `extra.catalog_hits` (Biblio-only) | `List[Dict: rsn, title, authors, dk_codes, rvk_codes, subjects]` | `slot:title_list` ⚠️ siehe Sektion 6 | title_list_search |
| `extra.catalog_tool_calls` | `int` | `slot:raw_json` | catalog_search, title_list_search |
| `extra.ranked_hits` | `List[Dict: title, gnd_id, score, reason]` | `slot:gnd_pool` | catalog_search (rank disabled) |
| `extra.seed_gnd_id` | `str` | `slot:raw_json` | synonym_expansion |
| `extra.seed_entry` | `Dict` | `slot:metadata_record` | synonym_expansion |
| `extra.seed_alternatives` | `List[Dict]` | `slot:gnd_pool` | synonym_expansion |
| `extra.seed_synonyms` | `List[str]` | `slot:synonym_set` | synonym_expansion |
| `extra.seed_ddcs` | `List[str]` | `slot:raw_json` | synonym_expansion |
| `extra.expansion_candidates` | `List[Dict: term, relation, reason]` | `slot:synonym_set` | synonym_expansion |
| `extra.validated_entries` | `List[Dict]` | `slot:gnd_pool` | synonym_expansion |
| `extra.metadata` | `Dict[gnd_id → Dict]` | `slot:metadata_record` | batch_metadata |
| `extra.missing_ids` | `List[str]` | `slot:raw_json` | batch_metadata |
| `extra.fetch_tool_calls` | `int` | `slot:raw_json` | batch_metadata |
| `extra.titles` | `List[Dict: title, authors, isbn, publisher, year]` | `slot:title_list` | title_list_search |
| `extra.duplicate_analysis` | `List[Dict: input_title, status, matches]` | `slot:duplicate_table` | title_list_search |
| `extra.duplicate_summary` | `str` | `slot:text_blob` | title_list_search |

## 4. Pro-Workflow-Schemas (Mini-Tabellen)

### 4.1 `alima_classic.yaml` (v4.0, 7 Steps)

7-Step Pipeline: extraction → search → selection_chunks → selection →
dk_collect → classification → dk_postprocess.

| Step-ID | Type | YAML Output-Mapping | SharedContext-Pfad | Slot | Quelle |
|---|---|---|---|---|---|
| `extraction` | llm_agent | `extracted_keywords ← response.keywords`, `working_title ← response.title` | `extracted_keywords`, `working_title` | `slot:keyword_list`, `slot:text_blob` | typed |
| `search` | deterministic (`gnd_batch_search`) | `gnd_entries ← result.entries` | `gnd_entries` | `slot:gnd_pool` | typed |
| `selection_chunks` | llm_agent (chunking) | `selected_keywords ← response.keywords` | `selected_keywords` | `slot:keyword_list` (flat) | typed |
| `selection` | llm_agent | `keyword_chains ← response.keyword_chains`, `missing_concepts ← response.missing_concepts`, `extra.final_keywords ← response.final_keywords` | `keyword_chains`, `missing_concepts`, `extra.final_keywords` | `slot:keyword_chains`, `slot:keyword_list` | typed + extra |
| `dk_collect` | deterministic (`dk_search_agentic`) | `extra.dk_entries ← result.dk_entries`, `extra.dk_prompt_text ← result.formatted_prompt`, `dk_search_results ← result.dk_search_results` | `extra.dk_entries`, `extra.dk_prompt_text`, `dk_search_results` | `slot:dk_table`, `slot:text_blob` | typed + extra |
| `classification` | llm_agent | `dk_classifications ← response.classifications`, `analyse ← response.analyse` | `dk_classifications` + `step_results.classification.analyse` | `slot:classification_list`, `slot:text_blob` | typed + step_results |
| `dk_postprocess` | deterministic (`build_dk_search_results`) | `dk_search_results ← result.results` | `dk_search_results` (überschreibt) | `slot:dk_table` | typed |

Besonderheit: `selection_chunks` aktiviert Chunking (siehe `chunking:`-
Block in YAML); LLM-Agent läuft N×.

### 4.2 `alima.yaml` (v5.0, 7 Steps, self-contained)

Step-Struktur identisch mit `alima_classic.yaml`. Unterschied: alle
System-/User-Prompts inline statt Delegation an `prompts.json`. Output-
Schema identisch. **Output-Tabelle wie 4.1 — nicht wiederholt.**

### 4.3 `catalog_search.yaml` (v4.0, 2 Steps)

| Step-ID | Type | YAML Output-Mapping | SharedContext-Pfad | Slot | Quelle |
|---|---|---|---|---|---|
| `search` | deterministic (`catalog_multi_search`) | `extra.catalog_hits ← result.hits`, `extra.catalog_tool_calls ← result.tool_calls` | `extra.catalog_hits`, `extra.catalog_tool_calls` | `slot:gnd_pool` ⚠️, `slot:raw_json` | extra |
| `rank` | llm_agent (**disabled** by default) | `extra.ranked_hits ← response.ranked` | `extra.ranked_hits` | `slot:gnd_pool` | extra |

Besonderheit: `rank` ist in YAML mit `enabled: false` deaktiviert.
Output `catalog_hits` ist **GND-enriched** (gnd_ids, ddc_codes,
dk_codes) — siehe Konflikt Sektion 6.

### 4.4 `synonym_expansion.yaml` (v4.0, 4 Steps)

| Step-ID | Type | YAML Output-Mapping | SharedContext-Pfad | Slot | Quelle |
|---|---|---|---|---|---|
| `lookup` | deterministic (`gnd_entry_lookup`) | `extra.seed_gnd_id ← result.gnd_id`, `extra.seed_entry ← result.entry`, `extra.seed_alternatives ← result.alternatives` | s. links | `slot:raw_json`, `slot:metadata_record`, `slot:gnd_pool` | extra |
| `extract_related` | deterministic (`extract_gnd_related`) | `extra.seed_synonyms ← result.synonyms`, `extra.seed_ddcs ← result.ddcs` | s. links | `slot:synonym_set`, `slot:raw_json` | extra |
| `expand` | llm_agent | `extra.expansion_candidates ← response.candidates` | `extra.expansion_candidates` | `slot:synonym_set` | extra |
| `validate` | deterministic (`gnd_batch_search`) | `extra.validated_entries ← result.entries`, `extra.validated_tool_calls ← result.tool_calls` | s. links | `slot:gnd_pool`, `slot:raw_json` | extra |

Hinweis: Wort „seed" hier = Ausgangs-Keyword, nicht LLM-Seed.

### 4.5 `batch_metadata.yaml` (v4.0, 1 Step)

| Step-ID | Type | YAML Output-Mapping | SharedContext-Pfad | Slot | Quelle |
|---|---|---|---|---|---|
| `fetch` | deterministic (`gnd_batch_metadata`) | `extra.metadata ← result.entries`, `extra.missing_ids ← result.missing`, `extra.fetch_tool_calls ← result.tool_calls` | s. links | `slot:metadata_record`, `slot:raw_json` | extra |

### 4.6 `title_list_search.yaml` (v4.3, 3 Steps)

| Step-ID | Type | YAML Output-Mapping | SharedContext-Pfad | Slot | Quelle |
|---|---|---|---|---|---|
| `extract_titles` | llm_agent | `extra.titles ← response.titles` | `extra.titles` | `slot:title_list` | extra |
| `search` | deterministic (`catalog_title_search`) | `extra.catalog_hits ← result.hits`, `extra.catalog_tool_calls ← result.tool_calls` | s. links | `slot:title_list` ⚠️, `slot:raw_json` | extra |
| `analyze_duplicates` | llm_agent | `extra.duplicate_analysis ← response.analysis`, `extra.duplicate_summary ← response.summary` | s. links | `slot:duplicate_table`, `slot:text_blob` | extra |

Besonderheit: `extra.catalog_hits` ist **Biblio-only** (rsn, authors,
dk_codes, rvk_codes, subjects, mab_subjects). Konflikt mit 4.3 — siehe
Sektion 6.

### 4.7 classic-Pipeline (Referenz aus WP2)

Liefert Felder als direkter Schreiber in `KeywordAnalysisState` (kein
`SharedContext`). Slot-Zuordnung identisch mit alima_classic.yaml.
**Zusätzlich classic-only**:

| Feld | Slot | Quelle |
|---|---|---|
| `refinement_iterations` | `slot:raw_json` (Vorschlag eigener `slot:iteration_log` in WP4) | classic-only |
| `repetition_meta` | `slot:raw_json` (Vorschlag eigener `slot:repetition_warning`) | classic-only |
| `dk_statistics` (= `dk_catalog_stats` befüllt) | `slot:raw_json` (Vorschlag eigener `slot:dk_statistics_panel`) | classic-only |

Vollständige Diff-Tabelle in [`research_classic_vs_agentic.md`](legacy/research_classic_vs_agentic.md)
Sektion 1.

## 5. Tool-Inventar-Matrizen

### 5.1 Workflow × MCP-Tool-Familie

Familien aus [`tool_registry.py:436+`](../src/mcp/tool_registry.py):
- `knowledge-read` (7): `search_gnd`, `get_gnd_entry`, `get_gnd_batch`,
  `get_search_cache`, `get_dk_cache`, `get_classification`,
  `get_db_stats`
- `knowledge-write` (1): `store_search_result`
- `library-search` (5): `search_lobid`, `search_swb`, `search_catalog`,
  `search_catalog_titles`, `resolve_doi`
- `pipeline-result-read` (4): `list_pipeline_results`,
  `get_pipeline_result`, `get_pipeline_keywords`,
  `get_pipeline_abstract`

Zellen-Wert: `R` = Read, `W` = Write, `R/W` = beides, `-` = nicht
genutzt, `*` = indirekt via register-fn statt direkter MCP-Tool-Call.

| Workflow | knowledge-read | knowledge-write | library-search | pipeline-result-read |
|---|---|---|---|---|
| alima_classic | R* | W* | R* | - |
| alima | R* | W* | R* | - |
| catalog_search | R* | - | R* | - |
| synonym_expansion | R* | - | R* | - |
| batch_metadata | R* | - | R* | - |
| title_list_search | R* | - | R* | - |

Befund: alle Workflows nutzen Library-Search und Knowledge-Read
**indirekt** über register-fns; **kein** Workflow heute hat
direktes MCP-Tool-Aufrufen im LLM-Agent-Tool-Block (alle `tools: []`
verifiziert in alima_classic.yaml). `pipeline-result-read` ist
Chat-only (WP7-Domäne), nicht in Workflows.

### 5.2 Workflow × deterministic-fn

| Workflow | gnd_batch_search | dk_data_collect | dk_search_agentic | build_dk_search_results | catalog_multi_search | catalog_title_search | gnd_entry_lookup | extract_gnd_related | gnd_batch_metadata |
|---|---|---|---|---|---|---|---|---|---|
| alima_classic | ✓ | - | ✓ | ✓ | - | - | - | - | - |
| alima | ✓ | - | ✓ | ✓ | - | - | - | - | - |
| catalog_search | - | - | - | - | ✓ | - | - | - | - |
| synonym_expansion | ✓ | - | - | - | - | - | ✓ | ✓ | - |
| batch_metadata | - | - | - | - | - | - | - | - | ✓ |
| title_list_search | - | - | - | - | - | ✓ | - | - | - |

`dk_data_collect` ist registriert ([`deterministic_functions.py:280`](../src/core/agents/deterministic_functions.py))
aber in den 6 Workflows nicht referenziert — verfügbar für Custom-
Workflows.

## 6. Catalog_hits-Konflikt (Optionen, Operator entscheidet)

`extra.catalog_hits` wird in zwei Workflows geschrieben — mit
inkompatiblen Schemata.

### Schema A: GND-enriched (`catalog_search.yaml`)

```typescript
{
  title: string,
  gnd_ids: string[],
  gnd_id?: string,
  description?: string,
  synonyms?: string[],
  ddc_codes?: string[],
  dk_codes?: string[],
  count?: int,
  sources?: string[]      // ["swb", "lobid", "catalog"]
}[]
```

### Schema B: Biblio-only (`title_list_search.yaml`)

```typescript
{
  query: string,          // Original-Titel aus Wunschliste
  rsn: string,            // RSN-ID
  title: string,
  authors: string[],
  year?: int,
  dk_codes?: string[],
  rvk_codes?: string[],
  ddc_codes?: string[],
  subjects?: string[],
  mab_subjects?: string[]
}[]
```

### Lösungsoptionen

| Option | Beschreibung | Vorteil | Nachteil |
|---|---|---|---|
| **A: Cluster-Split** | `catalog_search` → `slot:gnd_pool`, `title_list_search` → `slot:title_list` | Renderer simpel pro Slot, klare Trennung GND vs. Biblio | Workflow muss bei Slot-Resolution wissen welcher Cluster; zwei Slots für „Katalog-Treffer" |
| **B: Einheitsschema** | Ein `slot:catalog_hits` mit allen Feldern optional (Union A∪B) | Konzeptuelle Konsistenz, ein Renderer-Slot | Renderer braucht Switch-Logik nach Heuristik (`if 'gnd_ids' in row`); Optional-Inflation; schwer Tool-Filter pro Variante |

WP3 trifft KEINE Empfehlung. T1-blockierend. Siehe Operator-Frage F1.

## 7. Schema-Versionierung (Code-side via register-fn-Metadata)

**Heutiger Stand**: nur Workflow-Top-Level `version: "X.Y"`
([`workflow_loader.py`](../src/core/agents/workflow_loader.py)). Keine
pro-Step `outputs_schema:`.

**Vorschlag**: register-fn-Decorator-Erweiterung, YAML bleibt frei von
Annotationen.

### Decorator-Skizze

```python
# src/core/agents/registry.py — Erweiterung
def register_tool_fn(name: str, returns_slot: str = "slot:raw_json"):
    def decorator(fn):
        TOOL_FN_REGISTRY[name] = fn
        TOOL_FN_SLOT_HINT[name] = returns_slot   # NEU
        return fn
    return decorator
```

### Vorgeschlagenes Slot-Mapping für die 9 register-fns

| register-fn | Empfohlener `returns_slot` |
|---|---|
| `gnd_batch_search` | `slot:gnd_pool` |
| `dk_data_collect` | `slot:dk_table` |
| `dk_search_agentic` | `slot:dk_table` |
| `build_dk_search_results` | `slot:dk_table` |
| `catalog_multi_search` | `slot:gnd_pool` (Schema A) |
| `catalog_title_search` | `slot:title_list` (Schema B) |
| `gnd_entry_lookup` | `slot:metadata_record` |
| `extract_gnd_related` | `slot:synonym_set` |
| `gnd_batch_metadata` | `slot:metadata_record` |

LLM-Agent-Steps deklarieren NICHT explizit. WP4-Renderer-Registry
mappt auf:
1. register-fn-Slot-Hint (für deterministic-Step-Outputs)
2. Feldname-Pattern (`*_keywords` → `slot:keyword_list`,
   `*_chains` → `slot:keyword_chains`)
3. Fallback `slot:raw_json`

### Begründungen

Gegen **YAML-Annotation**: YAML-Inflation (jeder Step braucht
Annotation), LLM-Agent-Response-Felder lassen sich nicht universell
vorab annotieren.

Gegen **Auto-Inferenz aus Feldnamen**: bricht bei `catalog_hits`-
Doppelschema, das nicht der einzige extra-Pfad mit nicht-eindeutigem
Schema ist.

**Aufwand-Schätzung**: ~1 Tag in WP4 (Decorator-Erweiterung + WP4-
Registry-Mapping + Tests). Implementation NICHT in WP3.

## 8. Decision-Point T1

T1 = „Render-Slot-Vokabular fixiert" aus
[`agentic_ui_workpackages.md`](legacy/agentic_ui_workpackages.md)
Sektion 2.

Konkretes WP3-Deliverable für T1:
- **11 Slots namentlich fixiert** (Sektion 2).
- **Slot-Namespace `slot:<snake_case>`** festgelegt.
- **catalog_hits offen** → F1 als T1-Blocker.
- **Schema-Versionierungs-Vorschlag** (Sektion 7): Code-side via
  register-fn-Metadata, Implementation in WP4.

## 9. Operator-Fragen

Pro Frage Default-Empfehlung; Operator bestätigt oder revidiert.

### F1: `catalog_hits` — Option A (Cluster-Split) oder B (Einheitsschema)?
**Empfehlung**: persönliche Operator-Entscheidung. Audit-seitig leichte
Tendenz zu **A**, weil Renderer-Komplexität nicht in WP4 verschoben
werden sollte. **T1-blockierend** — ohne Antwort kann WP4 nicht
gestartet werden.

### F2: 11 Cluster ausreichend oder fehlen welche?
**Empfehlung**: behalte 11 + erlaube Erweiterung über `slot:custom_*`.
Kandidaten für eigene Slots in Zukunft (heute → `slot:raw_json`):
`slot:iteration_log`, `slot:repetition_warning`,
`slot:dk_statistics_panel`, `slot:reflection_panel`,
`slot:tool_call_trace`, `slot:meta_agent_log`. WP4 entscheidet pro
Renderer-Aufwand.

### F3: Slot-Namespace `slot:<snake_case>` akzeptiert?
**Empfehlung**: ja. Kollisionsfrei mit JSONPath. Alternative
`render.xxx` sieht nach Code-Pfad aus, `plain_name` kollidiert mit
Feldnamen.

### F4: Code-side-Schema via register-fn-Metadata akzeptiert für WP4?
**Empfehlung**: ja. YAML bleibt schlank; LLM-Agent-Output-Slots werden
via Feldname-Pattern + Fallback geschätzt; deterministic-fn-Outputs
sind explizit. Alternative (YAML-Annotation) wurde aus
Inflation-Gründen verworfen.

### F5: `slot:raw_json`-Fallback erlaubt oder soll jedes Feld typisiert sein?
**Empfehlung**: Fallback erlaubt. Long-Tail-Felder
(`fetch_tool_calls`, `execution_history`, `seed_ddcs`) brauchen keinen
Custom-Renderer. WP4 baut nur Renderer für die 10 typed-Slots.

### F6: `alima.yaml` (v5 inline) vs. `alima_classic.yaml` (v4 prompts.json) — Zukunftsformat?
**Empfehlung**: Operator entscheidet. v5 inline ist self-contained
(keine prompts.json-Abhängigkeit), v4 nutzt Multi-Variant-Prompts
(WP11-Pfad). Beeinflusst WP3-Folge-Audits, nicht T1-Deliverable.
WP3-Doc zeigt v4 als Referenz.

## 10. Querverweise / Folge-WPs

| Folge-WP | Konsumiert aus WP3 |
|---|---|
| **WP4** (Renderer-Registry) | Sektion 2 (Slot-Vokabular) + Sektion 7 (Code-side-Versionierung). 1:1-Mapping slot → Renderer-Klasse. |
| **WP7** (Chat-Tools workflow-aware) | Sektion 4 (Pro-Workflow-Schemas) + Sektion 5 (Tool-Inventar) für Workflow-Filter und tool-allowlist pro Workflow. |
| **WP11** (Provider-Portabilität) | Sektion 2 als Output-Schema-Quelle für Provider-Capability-Hints (welche Slots erwartet ein Workflow). |
| **WP10** (Migration) | Sektion 4 = Input für Tab-zu-Workflow-Mapping. |

## 11. Risiken + offene Validierungen

- **Cluster-Vollständigkeit nicht audit-beweisbar** — neue Workflows
  könnten 12. Slot brauchen. Mitigation: `slot:custom_*`-Konvention.
- **catalog_hits-Empfehlung nicht WP3** — Operator entscheidet via F1.
- **Reflection-Steps fehlen in 6 YAMLs**
  ([`reflection_step.py:72`](../src/core/agents/steps/reflection_step.py)).
  Audit zitiert Code-Pfad; sobald reflection-Step in YAML genutzt
  wird, müssen `quality_report`-Renderer evaluiert werden.
- **alima.yaml v5 vs alima_classic.yaml v4** — Doc nimmt v4 als
  Referenz. v5 hat dasselbe Step-Schema, Zukunftsformat klärt F6.
- **Code-side-Schema setzt voraus**, dass register-fn-Decorator-
  Erweiterung in WP4 möglich. Falls nicht → YAML-Fallback nötig.

## 12. Anhang

### 12.1 Glossar

- **Slot** — Render-Slot-Cluster, Namespace `slot:<snake_case>`.
  Mapping-Ziel von SharedContext-Feld zu Renderer-Klasse (WP4).
- **Step-Type** — Klassen-Tag in YAML (`llm_agent`, `deterministic`,
  `reflection`), Registry in
  [`registry.py:30`](../src/core/agents/registry.py).
- **register-fn** — Python-Callable mit `@register_tool_fn(name)`-
  Decorator, ausführbar in `DeterministicStep`.
- **JSONPath (Resolver)** — Dotted-Pfad-Syntax in YAML
  (`${steps.X.field}`, `${extra.key}`, `response.keywords`), aufgelöst
  in [`context_path.py:29-50`](../src/core/agents/context_path.py)
  und [`base_step.py:138-189`](../src/core/agents/steps/base_step.py).
- **Workflow-Version** — Top-Level `version: "X.Y"`, persistiert in
  `WorkflowDef.version`
  ([`workflow_loader.py:40`](../src/core/agents/workflow_loader.py)).

### 12.2 Code-Pfade (vollständig)

- Workflows: `workflows/alima_classic.yaml`, `workflows/alima.yaml`,
  `workflows/catalog_search.yaml`, `workflows/synonym_expansion.yaml`,
  `workflows/batch_metadata.yaml`, `workflows/title_list_search.yaml`
- SharedContext: `src/core/agents/shared_context.py:108-133`,
  `src/core/agents/base_shared_context.py`
- Step-Registry: `src/core/agents/registry.py` (decorator-Defs Z. 30,
  67), `src/core/agents/steps/llm_agent_step.py:64`,
  `src/core/agents/steps/deterministic_step.py:39`,
  `src/core/agents/steps/reflection_step.py:72`
- Deterministic fns: `src/core/agents/deterministic_functions.py:119,
  280, 376, 515, 576, 691, 788, 852, 880`
- JSONPath: `src/core/agents/context_path.py`,
  `src/core/agents/steps/base_step.py:138-189`
- MCP-Tools: `src/mcp/tool_registry.py:436+`

### 12.3 Re-Validation-Checklist

WP3-Audit aktualisieren wenn:
- Neuer Workflow in `workflows/*.yaml` hinzukommt → Sektion 4
  ergänzen, Slot-Zuordnung prüfen
- Neue `@register_tool_fn` in `deterministic_functions.py` →
  Sektion 5.2 + Sektion 7 Tabelle ergänzen
- Neuer MCP-Tool in `tool_registry.py:436+` → Sektion 5.1
  Familien-Mapping prüfen
- Neue Slots in WP4 entstehen → Sektion 2 aktualisieren

## Status

✅ WP3 abgeschlossen. Pendend: Operator-Antworten F1–F6, dann
T1-Render-Slot-Vokabular final fixiert.
