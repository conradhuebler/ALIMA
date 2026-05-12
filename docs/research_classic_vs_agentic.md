# Classic ↔ Agentic Pipeline-Vergleich (WP2)

**Status**: Faktenbasis-Dokument für T1-Entscheidung "classic-als-YAML
konsolidieren vs. dual-pfad dauerhaft". Output von WP2 aus
[`wp_detailed_plans.md`](wp_detailed_plans.md).

**Methode**: Code-Inspektion (statisch) + Prompt-Längen-basierte
Token-Untergrenze. **Echte Test-Runs sind offene Aufgabe**
(Sektion 5), Ausführung gehört zu WP11.

**Querverweise**:
- [`audit_findings.md`](audit_findings.md) — Finding 8 (Seed-Lücke
  agentic) ist zentrale Referenz.
- [`audit_tab_inventory.md`](audit_tab_inventory.md) — Strukturvorbild.
- [`pipeline_classic_flow.md`](pipeline_classic_flow.md),
  [`agentic_workflow.md`](agentic_workflow.md) — Pfad-Details.
- [`agentic_ui_workpackages.md`](agentic_ui_workpackages.md) Sektion 0
  — Operator-Vorgabe `alima_classic.yaml` = Forschungspfad/Fallback.
- [`wp_detailed_plans.md`](wp_detailed_plans.md) WP2 — Soll-Definition.

## 1. Output-Feld-Diff-Tabelle

`KeywordAnalysisState`-Felder vergleichen. Classic-Quelle = direkte
Befüllung in `pipeline_utils.py`. Agentic-Quelle = entweder
`SharedContext`-Typed-Field oder `SharedContext.extra`, konvertiert
über `SharedContext.to_keyword_analysis_state()`
([`shared_context.py:168`](../src/core/agents/shared_context.py)).

| Feld | Classic-Quelle | Agentic-Quelle | Diff | Renderer-Implikation (→WP4) |
|---|---|---|---|---|
| `original_abstract` | Parameter | `SharedContext.abstract` | = | Standard |
| `initial_keywords` | `execute_initial_keyword_extraction()` | `SharedContext.extracted_keywords` (alima_classic.yaml Step 1) | = | Standard |
| `working_title` | `execute_initial_keyword_extraction()` line 398 | `SharedContext.working_title` (Step 1 output) | = | Standard |
| `initial_gnd_classes` | `execute_initial_keyword_extraction()` | `SharedContext.gnd_entries` (Step 2) | = (Format ggf. ≠, Mapping in to_keyword_analysis_state) | Standard |
| `search_results` | `execute_gnd_search()` | `SharedContext.gnd_entries_per_keyword` | = | Standard |
| `final_llm_analysis` | `LlmKeywordAnalysis`-Objekt aus `execute_final_keyword_analysis()` line 1180 | merged aus `selected_keywords` + `keyword_chains` + `extra.final_keywords` (shared_context.py:293-342) | ≠ Struktur, gleiche Felder | Standard |
| `dk_search_results` | `execute_dk_search()` | `SharedContext.dk_search_results` (Step `dk_collect`/`dk_postprocess`) | = | Standard |
| `dk_search_results_flattened` | `execute_dk_search()` | aus DK-Conversion | = | Standard |
| `dk_statistics` | `_calculate_dk_statistics()` line 4361 | — | **∅ classic-only** | Neuer Renderer `dk_statistics_panel` (WP4) |
| `dk_classifications` | `execute_dk_classification()` line 1868 | `SharedContext.dk_classifications` (Step `classification`) | = | Standard |
| `dk_llm_analysis` | `execute_dk_classification()` | aus DK-Conversion | = | Standard |
| `rvk_provenance` | `execute_dk_classification()` | optional via RVK-Tools | = (oder ∅ agentic je Workflow) | Standard |
| `refinement_iterations` | `execute_iterative_keyword_refinement()` line 712 | — | **∅ classic-only** | Neuer Renderer `iteration_log` |
| `repetition_meta` | via `src/utils/repetition_detector.py` | — | **∅ classic-only** | Renderer `repetition_warning` |
| `chunks` | — | `LLMAgentStep` chunking-Block (z.B. alima_classic.yaml:120) | **∅ agentic-only** | Neuer Renderer `chunk_table` |
| `meta_agent_decisions` | — | `MetaAgent._plan_next_step()` line 205 (nur wenn `meta_agent.enabled: true`) | **∅ agentic-only** | Renderer `meta_agent_log` |
| `reflection_log` | — | `ReflectionStep` via MetaAgent line 415 | **∅ agentic-only** | Renderer `reflection_panel` |
| `tool_calls` | — | `AgentLoop` (agent_loop.py:18) Multi-Turn | **∅ agentic-only** | Renderer `tool_call_trace` |
| `seed_used` | gesetzt aus `kwargs["seed"]` (3 Stellen) | **hartcodiert auf `None`** in `to_keyword_analysis_state` (shared_context.py:300, 321, 339) | **≠ kritisch** | siehe Sektion 3 |

**Befund**: 6 Felder sind classic-only, 4 Felder sind agentic-only,
`seed_used` divergiert kritisch (Seed wird in agentic-Output **immer
als `None` serialisiert**, unabhängig davon ob Aufruf einen Seed
übergibt — verifiziert in shared_context.py:300/321/339).

## 2. Feature-Matrix

| Classic-only | Code-Pointer | Funktion |
|---|---|---|
| `iterative_keyword_refinement` | `pipeline_utils.py:712-986` | Konvergenz-Loop für fehlende GND-Konzepte |
| `repetition_detection` | `src/utils/repetition_detector.py` | Char-/N-Gram-Loop-Erkennung in LLM-Output |
| `_calculate_dk_statistics` | `pipeline_utils.py:4361` | Dedup-Metriken, Frequenz-Verteilung, Coverage |
| Seed-Threading (3 LLM-Steps) | `pipeline_utils.py:495, 1464, 1660` | Reproduzierbarkeit via `kwargs["seed"]` |
| GND-Verification (eigener Schritt) | `pipeline_utils.py:5047-5204` | DB-Fallback + GND-Pool-Match |
| Catalog-Subject-Validation | `pipeline_utils.py:988-1084` | SWB-Enrichment für Katalog-Subjects |

| Agentic-only | Code-Pointer | Funktion |
|---|---|---|
| Tool-Use via `AgentLoop` | `src/core/agent_loop.py:18` | Multi-Turn LLM↔Tool, diminishing-returns-Detection |
| MetaAgent-Replanning | `src/core/agents/meta_agent.py:205` | PLAN→EXECUTE→OBSERVE→REFLECT-Loop, **default `enabled: false`** in alima_classic.yaml/alima.yaml |
| Reflection-Step | `meta_agent.py:415-478` | Qualitäts-Reflexion über Step-Outputs |
| Chunking als Step-Option | `LLMAgentStep` (registry.py:36), alima_classic.yaml:120-129 | Per-Step chunking-Block in YAML |
| Workflow-YAML-Komposition | `workflows/*.yaml` (6 Workflows + legacy) | Freie Step-Verkettung ohne Python-Code |
| `SharedContext.extra` (unbegrenzt) | `base_shared_context.py:36` | Beliebige Felder pro Workflow |

**Hinweis MetaAgent**: in den Standard-Workflows ist `meta_agent.enabled:
false` gesetzt (alima_classic.yaml, alima.yaml). Default-agentic-Lauf ist
also **sequenziell deterministisch** wie classic. MetaAgent-Variabilität
nur opt-in. Das schwächt das übliche „agentic ist nicht-deterministisch"-
Argument — die echte Lücke ist Seed (Sektion 3), nicht Step-Reihenfolge.

## 3. Reproduzierbarkeits-Analyse (binär)

**Befund**: classic = reproduzierbar (mit Seed+Modell-Pin). Agentic =
**nicht reproduzierbar**.

### Seed-Audit

| Pfad | Seed-Stellen | Verifikation |
|---|---|---|
| Classic | 3 (`pipeline_utils.py:495, 1464, 1660`) | `kwargs.get("seed", 0)` an `LlmKeywordAnalysis` weitergereicht; default `0` |
| Agentic SharedContext | 0 | kein `seed`-Feld in `BaseSharedContext` / `SharedContext` |
| Agentic LLMAgentStep | 0 | `registry.py:36`, kein seed-Param in LLM-Call-Pfad |
| Agentic Workflows | 0 | kein `seed:`-Block in `alima_classic.yaml`, `alima.yaml`, `catalog_search.yaml`, `synonym_expansion.yaml`, `batch_metadata.yaml`, `title_list_search.yaml` (verifiziert via grep) |
| Agentic→Classic-Conversion | hartcodiert `None` | `shared_context.py:300, 321, 339` setzen `seed=None` beim Bau jedes `LlmKeywordAnalysis`-Objekts |

### MetaAgent-Variabilität

`MetaAgent._plan_next_step()` (meta_agent.py:205) wählt nächsten Step
dynamisch via LLM-Plan. **In Standard-Workflows default deaktiviert**
(`meta_agent.enabled: false`). Wenn aktiviert → Step-Reihenfolge
zusätzlich nicht-deterministisch.

### Gesamt-Aussage

- **Classic**: deterministisch bei `seed != 0` + Modell-Pin + identischer
  GND-Pool-Cache. Reproduzierbarkeit erreichbar.
- **Agentic (default)**: Step-Reihenfolge deterministisch (MetaAgent
  off), aber **LLM-Calls ohne Seed** → Output-Varianz pro Run.
  Reproduzierbarkeit **nicht** erreichbar.
- **Agentic (MetaAgent on)**: zusätzlich Step-Reihenfolge variabel.
  Reproduzierbarkeit ausgeschlossen.

### Retrofit-Skizze (NICHT Teil WP2)

Aufwand ~2h: `seed`-Feld in `BaseSharedContext`, `LLMAgentStep`
liest Config-Seed + reicht an `LlmService` weiter,
`to_keyword_analysis_state()` schreibt tatsächlichen Seed statt
hartcodiert `None`. Voraussetzung: MetaAgent off oder Plan-LLM
ebenfalls geseedet. **Gehört zu WP11** (Provider-Portabilität +
Seed-Pflicht-Forschungspfad).

## 4. Token-Cost-Schätzung (statisch, Untergrenze)

Berechnet aus `prompts.json` (Char-Count / 4 ≈ Tokens). Pro Standard-
Lauf mit 1 Abstract (~500 tok), 1 GND-Pool (~2000 tok), Output
~500 tok.

### Pro Task (System+Prompt-Anteil aus prompts.json)

| Task | Varianten | ~Tokens Prompt | Models-Marker |
|---|---|---|---|
| `extract_initial_keywords` | 1 | ~96 | `["default"]` |
| `initialisation` | 1 | ~724 | `["default"]` |
| `keywords` | 1 | ~1134 | `["default"]` |
| `keywords_chunked` | 1 | ~710 | `["default"]` |
| `dk_classification` | 1 | ~1281 | `["default"]` |
| `dk_list` | **2** | ~409 (avg) | `[gemini-1.5-flash, gemini-2.0-flash, DeepSeek-V3, cogito:14b, Meta-Llama-3-70B]`, `[default]` |
| `rvk_scoring` | 1 | ~267 | `["default"]` |
| `rvk_anchor_selection` | 1 | ~229 | `["default"]` |
| `image_text_extraction` | **4** | ~89 (avg) | multimodal (Gemini, GPT-4o, Claude, Cogito) |
| `rephrase` | 1 | ~82 | `["default"]` |

**Befund**: Nur `dk_list` und `image_text_extraction` haben echte
Modell-Familien-Varianten. Alle anderen 8 Tasks haben `["default"]` —
keine Provider-spezifischen Prompts. **Relevant für WP11.**

### Pro Standard-Lauf

Classic-Pfad (5 Steps, ohne Iter-Loop):
- Step 1 (`initialisation`): ~724 + 500 (Abstract) + 200 (Output) ≈ **1400 tok**
- Step 3 (`keywords`): ~1134 + 500 + 2000 (GND-Pool) + 500 (Output) ≈ **4100 tok**
- Step 4 (`dk_classification`): ~1281 + 1000 (DK-List-Input) + 500 ≈ **2800 tok**
- **Classic-Total**: ~8300 tok minimum (kein Iter-Refinement)
- Mit Iter-Loop (default max 2): +50% bis +100% → ~12000–16000 tok

Agentic-Pfad (alima_classic.yaml, 7 Steps, MetaAgent off):
- Step `extraction`: ähnlich `initialisation` ≈ 1400 tok
- Step `selection_chunks` (Chunking): N × ~1000 tok (N = Chunks)
- Step `selection`: ~3500 tok
- Step `classification`: ~2800 tok
- Plus Tool-Call-Overhead (AgentLoop:104): pro Step mit Tools 1-3 extra
  Roundtrips à ~500 tok (Tool-Result-Echo)
- **Agentic-Total**: ~10000–14000 tok minimum, plus Chunk-Multiplikation

**Gesamt-Schätzung**: agentic ist **+20% bis +40%** teurer pro Lauf,
Hauptursache Tool-Result-Echos und ggf. Chunking. Echte Messung in
Sektion 5.

## 5. Reproducible-Test-Spec (offen, für WP11)

Fixture-Layout, NICHT angelegt — Spezifikation für späteren Run.

### Fixture-Verzeichnis

`tests/fixtures/wp2_inputs/` (anzulegen durch Operator nach F5):
- `cadmium_short.json` — kurzer chem. Abstract (~300 Wörter)
- `cadmium_long.json` — selber Inhalt + Volltext-Anhang (Chunking-Trigger)
- `multilingual.json` — DE/EN-Mischtext
- `edge_no_topic.json` — Negativfall (Text ohne klares Thema)

### Modell-Matrix

| Provider-Familie | Modell | Verfügbarkeit |
|---|---|---|
| `ollama` (lokal) | `cogito:32b` | stabil |
| `openai_compatible` (cloud) | `gpt-4o-mini` | stabil bei API-Key |
| ~~GWDG~~ | — | **ausgeschlossen** (Instabilität, F6) |

### Run-Protokoll

`3 Runs × 2 Pfade × 2 Modelle × 4 Inputs = 48 Calls`.

Ablage: `tests/fixtures/wp2_runs/<ISO-timestamp>/<input>_<path>_<model>_run<N>.json`

Loader: `PipelineJsonManager.load_analysis_state()` für classic-Output;
agentic erst via `SharedContext.to_keyword_analysis_state()` → JSON.

### Metriken

- **m1**: Token-Total pro Run (aus `LlmKeywordAnalysis.filled_prompt` +
  Provider-Response-Usage falls verfügbar).
- **m2**: Jaccard-Overlap GND-Keywords zwischen Runs gleichen Pfads +
  Modells (Run1 ↔ Run2 ↔ Run3). Erwartung: classic mit Seed = 1.0,
  agentic < 1.0.
- **m3**: Feld-Belegungs-Vollständigkeit (welche Felder NULL/leer).
- **m4**: Wall-Clock-Time pro Step.

### Visualisierung

`ComparisonTab` (src/ui/comparison_tab.py) lädt 2 Result-JSONs und
zeigt Side-by-Side mit Overlap-Bar — geeignet für m2-Manual-Check.

### Auswerter-Skript

TODO WP11. Nicht Teil WP2.

## 6. Use-Case-Empfehlung (2 Personas)

| Persona | Empfohlener Pfad | Begründung |
|---|---|---|
| **P1: Routine-Bibliothekarin** (Standard-Erschließung) | **Classic** | Deterministisch, `dk_statistics` als Library-Confidence-Signal, `iterative_keyword_refinement` reduziert false-negatives |
| **P2: Forschungs-Run** (Reproduzierbarkeit erforderlich) | **Classic** | Nur classic hat Seed-Threading. Agentic-Pfad scheidet aus ohne WP11-Seed-Retrofit |

**Anmerkung**: Agentic bleibt sinnvoll für **Exploration** (neue
Domänen, freie Tool-Use durch LLM, Custom-Workflows). Das ist aber
keine eigene Persona, sondern eine Workflow-Wahl, die WP3
(Output-Schema-Inventar) und WP10 (Migration) behandeln.

## 7. Decision-Point T1: dual-pfad vs. classic-als-YAML

**Empfehlung: Option A (dual-pfad dauerhaft).**

| Option | Vorteil | Nachteil |
|---|---|---|
| **A: dual-pfad bleibt** | classic bleibt reproduzierbar; kein Migrations-Risiko; `alima_classic.yaml` = stabilisierte Spiegelung; Operator-Vorgabe (`agentic_ui_workpackages.md` Sektion 0) konsistent | zwei Codepfade pflegen; classic-only-Features (`iter_refinement`, `dk_statistics`, `repetition_detection`) bleiben agentic-unverfügbar |
| **B: classic-als-YAML konsolidieren** | ein Pfad; einheitliche Renderer (WP4 simpler); Wartung halbiert | `iter_refinement` + `repetition_detection` + `dk_statistics` als YAML-Steps nachbauen (~3–5 Tage Aufwand); Seed-Retrofit + MetaAgent-Determinismus-Garantie Voraussetzung; Forschungspfad-Risiko |

**Begründungs-Kern**: classic erreicht Reproduzierbarkeit nur als
eigener Pfad (Sektion 3). Bis WP11-Seed-Retrofit + MetaAgent-Off-
Garantie agentic-seitig umgesetzt sind, kann agentic den
Forschungspfad nicht ersetzen. Wartungsersparnis < Agentic-
Equivalence-Risiko.

**Klausel**: bei Option A bleibt `pipeline_utils.py` **kanonisch** für
Forschungspfad. `alima_classic.yaml` = Fallback-Spiegelung, **NICHT
Ersatz**. WP10-Migration darf classic nicht brechen. **T1-Re-Eval in
6 Monaten** nach Abschluss WP11 (Seed-Retrofit) und WP3 (Schema-
Stabilisierung).

## 8. Operator-Fragen

Pro Frage Default-Empfehlung; Operator bestätigt oder revidiert.

### F1: Wie oft wird der Forschungspfad faktisch genutzt?
**Empfehlung**: persönliche Einschätzung Operator. Falls Forschungspfad
in der Praxis nie ausgeführt wird → Option B (Konsolidierung) wäre
revisionsfähig. Falls regelmäßig → Option A bestätigt.

### F2: Ist „reproduzierbar" für Forschungspfad bit-exakt oder reicht inhaltlich-ähnlich (>80% Jaccard)?
**Empfehlung**: bit-exakt für klassische Forschungs-Veröffentlichung.
Inhaltlich-ähnlich reicht nur für Vergleichsstudien mit
mehrfach-Aggregation. Beeinflusst WP11-Seed-Retrofit-Scope.

### F3: Soll `iterative_keyword_refinement` langfristig auch in agentic verfügbar werden?
**Empfehlung**: ja, als optionaler `iterative_refinement`-Step in
YAML, aber kein WP2-Zwang. Falls Operator nein sagt → classic-only-
Feature bleibt, dual-pfad zwingend.

### F4: `dk_statistics` — Pflicht-Output für alle Pipelines oder classic-Spezifikum?
**Empfehlung**: Pflicht-Output. Statistik ist Library-Confidence-
Indikator, sollte beide Pfade liefern. Bei Option A: agentic-Pendant
als deterministischer Post-Step in `alima_classic.yaml`.

### F5: Welche Inputs sollen in `tests/fixtures/wp2_inputs/`?
**Empfehlung**: Operator liefert 3–4 echte Test-Abstracts aus
ALIMA-Routine (Bibliotheks-Datensätze). Synthetische Inputs nur als
Ergänzung. Mindestens 1× DE, 1× Mischtext, 1× Negativfall.

### F6: GWDG-Ausschluss aus Test-Matrix akzeptabel?
**Empfehlung**: ja, ausschließen. GWDG-Verfügbarkeit hängt von
Uni-Login + Instanz-Status ab. Test-Matrix soll CI-fähig sein. GWDG
als manuelle Stichprobe nach Bedarf, nicht als Pflicht-Lane.

## 9. Risiken + offene Validierungen

Was dieses Dokument **nicht** beantwortet ohne Sektion-5-Test-Runs:
- Echte Token-Cost-Differenz pro Provider (nur Untergrenze geliefert).
- Echte Wall-Clock-Time-Differenz.
- Output-Qualitäts-Vergleich (benötigt Evaluator + Gold-Standard-
  Annotation).
- Quantitative Agentic-Output-Varianz (ohne Seed-Retrofit nicht
  belastbar messbar).
- Provider-Verhaltens-Unterschiede (Ollama-Determinismus, GWDG-
  Drift) — nicht in WP2-Scope.

Alle Punkte als „TODO post-WP11" markiert. Ausführung über
Sektion-5-Spec.

## 10. Querverweise / Folge-WPs

| Folge-WP | Konsumiert aus WP2 |
|---|---|
| **WP3** (Output-Schema-Inventar) | Sektion 1 Diff-Tabelle = Input-Vorbedingung für Render-Slot-Vokabular |
| **WP4** (Renderer-Registry) | classic-only-Felder (`dk_statistics`, `refinement_iterations`, `repetition_meta`) + agentic-only-Felder (`reflection_log`, `tool_calls`, `meta_agent_decisions`, `chunks`) brauchen neue Renderer |
| **WP10** (Migration) | T1-Empfehlung Option A = dual-pfad-Constraint; classic darf nicht brechen |
| **WP11** (Provider-Portabilität) | Sektion 5 Test-Spec hier ausführen; Seed-Retrofit umsetzen; Multi-Variant-Prompts für 8/10 Tasks die heute nur `["default"]` haben |

## Status

✅ WP2 abgeschlossen. Pendend: Operator-Antworten F1–F6, dann
T1-Entscheidung final.
