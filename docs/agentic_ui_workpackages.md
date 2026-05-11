# Agentic + UI — 11 Arbeitspakete

> **Update**: WP11 (Provider-Portabilität) hinzugefügt nach Operator-Hinweis
> "Pipeline/Agent getestet, optimiert für eines [Modell], muss aber mit
> verschiedenen funktionieren". Audit-Findings in
> [`audit_findings.md`](audit_findings.md).

**Status**: Planungsdokument. **Keine Implementation.** Output dieses
Dokuments = klare WP-Liste mit Scope, Abhängigkeiten, Decision-Points,
Fälligkeiten. Implementation startet erst nach Freigabe pro WP.

## 0. Direktiven (vom Operator gesetzt)

- `alima_classic.yaml` ist **Forschungspfad / Fallback**. Bleibt
  vollwertig erhalten. Agentic ist Default, aber nicht alleinig.
- Probleme finden vor Lösen. Diskussion vor Code.
- K1 (Single-Step), K2 (Renderer), K6 (Tab-Audit) → in Plan integriert.
- K3 (Multi-Frontend) → muss schlau gelöst werden.
- K5 (Webapp) → bewusst späteres Todo.
- K7 (Chat) → Chat ist integraler Teil, kein Add-on.
- Selbstkritik (S1-S6 aus `agentic_chat_plan.md`) muss in WPs
  eingearbeitet sein.

## 1. WP-Übersicht (Dependency-DAG)

```
              ┌─────────────────────────────────────────────┐
              ▼                                             │
    [WP1: Tab-Audit]──┐                                     │
                      │                                     │
    [WP2: Classic↔Agentic-Diff]      [WP9: Tier+Frontend-Strategie]
                      │                          │
    [WP3: Output-Schema-Inventar]    [WP11: Provider-Portabilität]
                      │                          │
                      ▼                          │
              [WP4: Renderer-Registry-Skizze]   │
                      │                          │
                      ▼                          │
              [WP5: Single-Step-Modell]         │
                                                 │
              [WP6: State-Sync-Fundament]       │
                      │                          │
                      ▼                          │
              [WP7: Chat-Tools workflow-aware]  │
                      │                          │
                      ▼                          │
              [WP8: Chat-UI-Konzept]            │
                      │                          │
                      └────────────┬─────────────┘
                                   ▼
                       [WP10: Migrations- + Decision-Timeline]
```

Legende:
- Pfeil = "braucht Output von" (Decision oder Spec).
- Pakete ohne Pfeil zueinander sind parallelisierbar.

## 2. Decision-Calendar (Reihenfolge der Entscheidungen)

| Slot | Entscheidung fällig | Liefert | Blockiert |
|------|---------------------|---------|-----------|
| **T0** | WP1, WP2, WP3, WP9, WP11 starten parallel | Inventar + Schemas + Tiers + Provider-Capability-Skizze | Alles weitere |
| **T1** | "Bleibt classic dual-pfad oder als Workflow-YAML?" (aus WP2) | Klärt Code-Pfad-Konsolidierung | WP4-Scope, WP10-Migration |
| **T1** | "Welche Render-Slots gibt es?" (aus WP3) | Output-Type-Vokabular | WP4 |
| **T1** | "Tier-1/2/3 Festlegung" (aus WP9) | Welches Frontend kriegt was | WP4-Webapp-Echo, WP8-Webapp-Mirror |
| **T2** | "Renderer-Plugin-Architektur ja/nein?" (aus WP4) | Approach Option C vs B | WP5, WP8 |
| **T2** | "EventBus oder Signal-Spider?" (aus WP6) | State-Sync-Pattern | WP7-Schreib-Tools, WP8 |
| **T3** | "Generische vs spezialisierte Chat-Tools" (aus WP7) | Tool-Strategie | WP8-Tool-Call-UI |
| **T3** | "Path B Multi-Turn direkt?" (aus WP7) | Chat-Worker-API | WP8 |
| **T4** | "Migration-Reihenfolge der Tabs" (aus WP10) | Roadmap | Implementation-Start |

## 3. Arbeitspakete

### WP1 — Tab-Inventar-Audit (K6)
**Ziel**: Faktenbasis welche Tabs existieren, was sie tun, wo sie
überlappen, welche entfernt/gemerged werden können.

**Scope**:
- Tabelle pro Tab: Name, Datei, Zeilen, Funktion (eine Zeile),
  Pipeline-Step-Bezug, überlappt mit (Liste), Eigene Worker-Klassen,
  Single-Source-of-Truth-Verletzungen.
- Identische Bewertung für Webapp (Endpoints) + CLI (Subcommands).
- Welche Tabs lesen `KeywordAnalysisState` direkt vs über Manager.
- Welche Tabs greifen direkt in `prompts.json` / `LlmService`.

**Input**: nur Code-Inspektion. Optional Telemetrie wenn vorhanden.

**Output**: `docs/audit_tab_inventory.md` mit Tabelle + Überlapp-Matrix
+ Empfehlung pro Tab (keep/merge/delete/replace).

**Offene Fragen** (im WP klärbar):
- Telemetrie über Tab-Nutzung verfügbar? Sonst User-Survey?
- ComparisonTab vs ReviewTab — funktional gleich oder verschieden?
- AbstractTab `set_task()` deckt welche Workflow-Steps ab?

**Dependency**: keine. Startet T0.

**Blockiert**: WP10 (Audit ist Input für Migrationsplan).

**Parallel mit**: WP2, WP3, WP9.

**Fälligkeit**: vor WP10.

---

### WP2 — Classic ↔ Agentic Pipeline-Vergleich (Forschungspfad)
**Ziel**: Verstehen was klassisch besser/schlechter macht. Forschungspfad
braucht messbare Differenz, sonst kein Argument für duale Pflege.

**Scope**:
- Output-Felder von `pipeline_utils.py`-Pfad vs `WorkflowExecutor`-Pfad
  vergleichen.
- Welche Felder sind exklusiv classic (z.B. iterative refinement,
  repetition detection, dk_statistics)? Welche exklusiv agentic
  (`SharedContext.extra`, per_chunk responses)?
- Welche Bereiche, in denen classic robuster ist (deterministisch,
  reproduzierbar), welche agentic robuster (flexibel, MetaAgent-loop)?
- Reproduzierbarkeit: classic seedbar, agentic ebenfalls?
- Token-Cost classic vs agentic für gleiches Eingabe.

**Output**: `docs/research_classic_vs_agentic.md` — Tabelle, Empfehlung
pro Use-Case welcher Pfad.

**Offene Fragen**:
- Gibt es Benchmarks/Referenzdatensätze um beide zu messen?
- Soll classic als `alima_classic.yaml`-Workflow neu codiert werden,
  oder bleibt eigener Code-Pfad? **Decision-Point T1.**
- Wenn dual-pfad bleibt: wie verhindern wir dass Features nur in einem
  landen (Bug-Fix-Driften)?

**Dependency**: keine. Startet T0.

**Blockiert**: WP4 (entscheidet ob classic als Workflow gerendert wird),
WP10 (Migration darf classic nicht brechen).

**Parallel mit**: WP1, WP3, WP9.

**Fälligkeit**: T1 (Entscheidung classic-Konsolidierung).

---

### WP3 — Output-Schema-Inventar pro Workflow (K2 + S6)
**Ziel**: Pro existierendem Workflow + classic-Pipeline systematisch
auflisten welche Felder welcher Typ produziert. Vorbedingung für
Renderer-System UND für workflow-aware Chat-Tools.

**Scope**:
- Pro Workflow (`alima.yaml`, `alima_classic.yaml`, `catalog_search`,
  `synonym_expansion`, `batch_metadata`, `title_list_search`):
  - Welche Felder schreibt jeder Step in `SharedContext` (typed +
    `extra`)?
  - Welcher Datentyp pro Feld? (List[Dict], Str, Int, …)
  - Welcher Renderer-Slot würde inhaltlich passen (z.B.
    `keyword_chains`, `dk_table`, `duplicate_table`, `gnd_pool`,
    `classification_list`, `raw_json_fallback`)?
- Identifizieren von Überlappungen — z.B. mehrere Workflows produzieren
  `final_keywords` mit gleichem Schema.

**Output**: `docs/workflow_output_schemas.md` — Schema pro Workflow,
Render-Slot-Vokabular, Tool-Inventar-Vorschlag.

**Offene Fragen**:
- Soll Output-Schema in YAML deklariert werden (`output_schema:` pro
  Step) oder aus Code/Tool-Function gelesen?
- Welche generischen Renderer (`raw_json`, `key_value_table`) reichen
  für Long-Tail-Outputs, wo Custom-Renderer Overkill?
- Versionierung: was wenn Workflow später neue Felder hinzufügt?

**Dependency**: keine. Startet T0.

**Blockiert**: WP4 (Render-Slot-Liste = Input), WP7 (Tool-Inventar
braucht Schema).

**Parallel mit**: WP1, WP2, WP9.

**Fälligkeit**: T1 (Render-Slot-Vokabular fixiert).

---

### WP4 — Renderer-Registry-Skizze (K2 + Option C)
**Ziel**: Architektur für pluggable Output-Renderer ohne Code-Build.
Beweisen ob Option C trägt; Code-Sketch + Beispiel.

**Scope**:
- API-Sketch: `@register_renderer("dk_table")` decorator. Renderer-
  Klasse mit `render(output_data, context) → QWidget|html|cli_text`
  pro Frontend-Backend.
- Wie referenziert YAML einen Renderer? `output_schema: dk_table`
  pro Step? Oder Default basierend auf Step-Typ?
- Fallback-Renderer für unbekannte Outputs (`raw_json` im UI, JSON-
  Echo in CLI).
- Frontend-Splittung: Renderer-Implementation pro Frontend
  (Qt-Widget vs HTML-Template vs CLI-Tabelle) — gemeinsamer Vertrag?
- Backwards-compat: existierende Tabs werden zu Renderern, oder
  rufen Renderer intern auf?

**Output**: `docs/renderer_registry_design.md` — API-Skizze, Beispiel
für 2-3 Renderer (`dk_table`, `keyword_chains`, `duplicate_table`),
Frontend-Mapping-Skizze.

**Offene Fragen**:
- Renderer-Registry als gleiche Mechanik wie `STEP_REGISTRY`/
  `TOOL_FN_REGISTRY` (Konsistenz) oder eigene?
- Sollen Renderer wiederverwendbar sein zwischen GUI und Webapp
  (z.B. via gemeinsame HTML-Engine) oder explizit pro Frontend?
- Wann wechselt System auf Renderer (Big Bang) oder schrittweise
  Migration pro Tab?

**Dependency**: WP3 (Render-Slot-Vokabular).

**Blockiert**: WP5 (Single-Step braucht Renderer für Output-Anzeige),
WP8 (Chat-UI rendert Tool-Outputs ggf. mit Renderern).

**Parallel mit**: WP6, WP9 (wenn diese T0/T1 schon laufen).

**Fälligkeit**: T2 (Render-Architektur-Entscheidung).

---

### WP5 — Single-Step-Execution-Modell (K1)
**Ziel**: Wie wird ein einzelner Step eines Workflows isoliert ausgeführt?
Klärt Beziehung Single-Step-Tabs ↔ Workflow-Steps.

**Scope**:
- Heute: `WorkflowExecutor.run(..., only_step=<id>)` existiert + CLI
  `--only-step`. Was fehlt UI-seitig?
- Wie generiert UI das Eingabe-Form pro Step? Aus `inputs:`-Block der
  YAML? Welche Felder sind „User-fillable" vs „aus vorigem Step"?
- Warm-Start: existierender SharedContext (JSON) laden + nur einen
  Step neu ausführen.
- Output-Übernahme: Single-Step-Ergebnis in nachfolgenden Schritt
  übernehmen ohne kompletten Re-Run.
- Auto-generierte vs hardcoded Tabs: ist es realistisch Tabs aus
  YAML zu generieren? Was wenn 7 Steps × 5 Workflows = 35 Tabs?
  Lösung: Tab pro Step **des aktiven** Workflows.

**Output**: `docs/single_step_model.md` — Modell, UI-Pattern-Sketch,
Warm-Start-Sequence.

**Offene Fragen**:
- Wie verhält sich `chunking:`-Step bei Single-Run (will man Chunks
  einzeln laufen lassen?).
- Was wenn `inputs:` nicht serialisierbar ist (z.B. `${steps.X.Y}`
  noch nicht resolved)?
- AbstractTab heute = generischer LLM-Call mit Task-Wechsel.
  Migration: Tab fällt weg, ersetzt durch Step-spezifische Tabs?
  Oder bleibt als „Generic LLM Tool"?

**Dependency**: WP4 (Renderer für Output), WP3 (Input-Schema pro Step).

**Blockiert**: WP10 (Migration braucht klares Ziel-Modell).

**Parallel mit**: WP6, WP7, WP8 (Chat-Pfad), wenn WP4 fertig.

**Fälligkeit**: T2 nach WP4.

---

### WP6 — State-Sync + Mutations-Fundament (S4)
**Ziel**: Bevor irgendein Tool / Chat-Schreibvorgang erlaubt wird,
muss klar sein wie State-Mutationen propagieren. Sonst inkonsistente
UI.

**Scope**:
- Single-Source-of-Truth-Modell: ist `KeywordAnalysisState`
  Master? Oder `SharedContext`? Beide? Wie wird beim agentic-Lauf
  letzteres in ersteres überführt?
- EventBus oder Qt-Signals? Pro Tab subscribed an State-Änderungen.
- Mutation-API: `state.apply_keyword_replacement(old, new)` →
  emittiert `state_changed(diff)`. Tabs re-renderieren betroffene
  Sektionen.
- Undo/Redo-Stack? Für Chat-Mutationen sinnvoll.
- Persistenz: jeder State-Diff in JSON-Audit-Log?

**Output**: `docs/state_sync_design.md` — Pattern-Sketch,
Mutation-API-Skizze, Diff-Strategy.

**Offene Fragen**:
- Konflikt zwischen Pipeline-Run (großer Bulk-Write) und Chat-
  Mutation (kleiner Diff)?
- Wie verhält sich State-Sync in Webapp (separate Sessions, kein
  Qt-Signal-Bus)? **Hängt mit WP9 zusammen.**
- Race-Condition: User klickt Tab-Action während Worker mutiert.
  Locking-Pattern?

**Dependency**: keine direkte technische, aber nutzt WP3-Schema.

**Blockiert**: WP7-Schreib-Tools, WP8-Mutations-Dialoge.

**Parallel mit**: WP4, WP5.

**Fälligkeit**: T2 (vor allen Schreib-Operationen aus Chat).

---

### WP7 — Chat-Tools workflow-aware (S6 + S5 + S3)
**Ziel**: Chat-Tool-Set so designen, dass es nicht nur ALIMA-
Pipeline-Felder kennt. Plus: Cache-Mutations-Frage und Provider-Wahl.

**Scope**:
- Generische Tools über `SharedContext.extra`-Introspection:
  `list_extras()`, `get_extra(path)`. Plus per-Workflow registrierbare
  Spezial-Tools (z.B. `get_dk_titles_for_code` nur wenn `dk_search_results`
  in Schema).
- Tool-Discovery: bei Chat-Init wird aus aktivem Workflow + Output-
  Schema (WP3) die Tool-Liste generiert.
- Read-only-Modus: ein Setting "kein Cache-Write während Chat".
  `search_gnd` läuft dann gegen Cache-Read-Only oder API-Direct
  ohne Persistenz.
- Provider-Strategie: Chat-Provider unabhängig vom Pipeline-Provider
  (eigenes Combo, eigener Default). Vorschlag: günstig + schnell,
  weil Chat = viele Turns. Doc-Begründung pro Use-Case.
- Tool-Set per Workflow vorrendern (Cache) oder dynamisch?

**Output**: `docs/chat_tools_design.md` — Tool-Klassen-System,
Workflow-Discovery-Mechanik, Provider-Defaults, Read-Only-Modus.

**Offene Fragen**:
- Wie merken wir uns welche Tools für `title_list_search`
  spezialisiert sind, ohne im YAML Tool-Liste zu deklarieren?
  Oder doch: `chat_tools:`-Block in Workflow-YAML?
- Schreib-Tools: `propose_keyword_replacement` ist Standard-ALIMA.
  Hat `title_list_search` eigene Mutationen (z.B. „markiere Titel
  als manuell gewählt")?
- Multi-Turn: Path B (echte messages) direkt — wie viel `AgentLoop`-
  Erweiterung nötig?

**Dependency**: WP3 (Output-Schemas), WP6 (State-Sync für
Schreib-Tools).

**Blockiert**: WP8 (Chat-UI).

**Parallel mit**: WP5, wenn WP3+WP6 fertig.

**Fälligkeit**: T3.

---

### WP8 — Chat-UI-Konzept (S1 + S2 + K7)
**Ziel**: Chat als Tier-1-Feature in GUI, nicht nur Dock. Mutations-
Dialoge integriert. Tool-Call-Visualisierung. Path B Multi-Turn.

**Scope**:
- ChatWidget-Redesign: ist Dock OK oder eigener Tab? Beides? Pro
  Anwendungsfall (M1 Standard vs M2 Single-Step vs M3 Custom)?
- Tool-Call-Rendering: kollabierbare Blöcke in History? Status-Zeile?
  Hidden by default mit Toggle?
- Mutations-Modal: Konsens-Dialog ("Replace X with Y? [Yes / No /
  Edit]"). Wo angezeigt — modal über App oder in-chat?
- Re-Run-aus-Chat: wenn Pipeline-Step neu läuft, Output-Stream im
  Chat ODER im Pipeline-Stream-Widget?
- Multi-Turn Path B: AgentLoop nimmt `messages`-Liste statt einem
  user_prompt. AgentLoop-API-Erweiterung dokumentieren.
- Webapp-Echo: hier nur Konzeptklarheit, Webapp-Bau später (K5).
  Aber API muss agnostisch genug sein (z.B. Tool-Call-Stream als
  WS-Event für Webapp).

**Output**: `docs/chat_ui_design.md` — UI-Pattern (Wireframe-Text),
Modal-Flow, AgentLoop-API-Erweiterung.

**Offene Fragen**:
- Chat-Sessions persistieren? Pro Pipeline-Run einer? Cross-Run-
  History?
- Wie vermeiden wir, dass User in Chat etwas mutiert während
  Pipeline-Stream läuft (Race)?
- Re-Run-Blocker: während `propose_step_rerun` ausgeführt wird,
  Chat-Eingabe gesperrt?

**Dependency**: WP6 (State-Sync), WP7 (Tools).

**Blockiert**: WP10.

**Parallel mit**: nichts mehr in dieser Reihe.

**Fälligkeit**: T3 nach WP7.

---

### WP9 — Multi-Frontend-Tier-Modell (K3 + K5)
**Ziel**: Pro Feature definieren in welchem Frontend es ankommen muss.
Reduziert Maintenance-Last × 3. Webapp-Strategie expliziert
(auch wenn als „später").

**Scope**:
- Feature-Liste aus Anforderungskatalog (M1-M6) × Frontend-Matrix.
- Tier-Definition (Tier-1 = überall, Tier-2 = GUI+Webapp, Tier-3 =
  nur GUI). Begründung pro Tier.
- Webapp-Strategie: Lite-Variante (bewusst eingeschränkt) ODER
  catch-up-Plan (welche Features in welcher Reihenfolge nachziehen)?
- CLI-Strategie: was bleibt scriptable (= Tier-1)? Chat-CLI
  realistisch?
- Geteilte Schicht: JSON-Schemas/OpenAPI für Workflow-Inputs/Outputs
  generieren (siehe „Techniken" im Anforderungskatalog).

**Output**: `docs/frontend_tier_model.md` — Tier-Matrix, Webapp-
Strategie-Optionen, gemeinsame Schemas.

**Offene Fragen**:
- Wer nutzt Webapp (Anzahl, Profile)? Beeinflusst Strategie.
- Soll CLI Workflow-spezifische Outputs auch rendern (z.B.
  duplicate-tabelle als ASCII-Tabelle) oder nur JSON-Echo?
- Auto-generierte OpenAPI ist nice-to-have oder Voraussetzung für
  Webapp-Catch-Up?

**Dependency**: keine. Startet T0.

**Blockiert**: WP4 (Renderer-Frontend-Mapping), WP8 (Chat-Webapp-Echo).

**Parallel mit**: WP1, WP2, WP3.

**Fälligkeit**: T1 (Tier-Festlegung).

---

### WP11 — Provider/Modell-Portabilität (NEU)

**Vor-Analyse von 2025**: `provider_strategy_*.md`-Set existiert
(4 Docs, 951 Z.). Empfahl Vereinfachung der Fallback-Hierarchie
(4-Tier → 2-Tier) + Entfernung von Model-Family-Recognition. WP11
**baut Recognition aus**, nicht ab — neue Anforderung Multi-Provider
braucht Familie-Pattern. Vereinfachung Fallback bleibt valide.
Status-Marker in `provider_strategy_summary.md`.

**Ziel**: Pipeline + Agent funktionieren reproducible auf >1
Provider/Modell-Familie. Heute getunet für ein Setup (Audit:
`<|begin_of_thought|>`-Markup = DeepSeek/Qwen, prompts.json hat nur
`models: [default]`).

**Scope**:
- **Provider-Capability-Profil** definieren: pro Provider Eigenschaften
  (json_mode, tool_use_native, vision, max_context, seed_support,
  thinking_tokens, streaming, parallel_calls). Heute teilweise in
  `model_capabilities.py` für Chunking, aber nicht systematisch.
- **Prompt-Varianten pro Modell-Familie**: prompts.json-Schema
  unterstützt `models:`-Liste schon → dafür echte Varianten anlegen
  (z.B. "anthropic-claude-style", "openai-gpt-style", "ollama-qwen-style",
  "ollama-gemma-style"). Selektor matched Modellname → Familie → Variante
  → Default.
- **Tool-Use Abstraction**: `AgentLoop` + `LlmService.generate_with_tools`
  prüfen wie sauber sie über Provider abstrahieren. Audit, nicht Bau.
- **Seed-Pflicht für Forschungspfad**: agentic fehlt seed-Pfad
  (Audit-Finding 8). Entweder nachrüsten oder dokumentieren „classic
  für reproducible Runs".
- **Test-Matrix**: definieren welche Provider/Modell-Kombos getestet
  werden müssen pro Workflow. Mind. 1 lokal (Ollama), 1 cloud (OpenAI/
  Anthropic), evtl. 1 Gemini (eigene Eigenheiten).
- **Per-Step-Provider-Mix in UI**: heute nur global-Override + YAML-edit.
  UI-Pattern für „cheap-extraction + premium-classification"?
- **Chat-Provider-Frage** (S3) hier mitlösen: Chat-Provider ist
  unabhängig vom Pipeline-Provider, eigener Default.
- **Output-Format-Robustheit**: prompts müssen JSON-Output erzwingen
  ohne dass JSON-mode/function-calling Pflicht ist. Text-Parser-
  Toleranz testen pro Provider.

**Output**: `docs/provider_portability_design.md` —
Provider-Capability-Schema, Prompt-Varianten-Strategie, Test-Matrix,
Per-Step-Provider-Mix-Pattern.

**Offene Fragen**:
- Wie groß ist Aufwand für Prompt-Varianten? Pro Workflow × Pro Modell-
  Familie = N×M Prompts.
- Auto-Erkennung Modell-Familie oder explizite User-Wahl?
- Wer ist verantwortlich für Test gegen neue Provider?
- Soll ALIMA „Modelle ranken" (für gegebenen Workflow welches Modell
  empfohlen)?
- Repetition-Detection-Schwellen pro Modell konfigurierbar machen?

**Dependency**: WP3 (Workflow-Output-Schemas — wir wollen wissen ob
Prompts unterschiedlich versagen pro Provider/Schema), Audit-Finding 1
(prompts.json vs yaml entscheiden).

**Blockiert**: WP10 (Migration darf nicht ein Provider-Setup
hardcoden), WP7 (Chat-Tools — Provider-Default für Chat).

**Parallel mit**: WP4, WP5, WP6 (verschiedene Dimensionen).

**Fälligkeit**: T2 (Capability-Schema), T3 (Test-Matrix).

---

### WP10 — Migrations- und Decision-Timeline
**Ziel**: Synthese aller WPs zu konkretem Migrationsplan. Pro Tab/
Endpoint/Command Entscheidung „bleibt / mergen / ersetzen / löschen"
mit Reihenfolge + Roll-Back-Punkten.

**Scope**:
- Tabelle aller Tabs/Endpoints/Commands × Entscheidung × Phase ×
  abhängige WP-Outputs.
- Forschungspfad-Schutz: classic-Pipeline darf nicht in Migration
  brechen. Welche Tests/Smoke-Checks?
- Roll-Back-Pfade: wenn neue Tab-Architektur scheitert, wie zurück?
- Phasen-Definition (z.B. P-Alpha = Renderer-Skelett, P-Beta = erste
  3 Tabs migriert, P-Gamma = Single-Step-Auto-Tabs, P-Delta = Chat-
  Schreib-Tools, P-Epsilon = Webapp-Catch-Up).
- Risiko-Matrix pro Phase.

**Output**: `docs/migration_roadmap.md` — Phasen, Tab-Entscheidungen,
Roll-Back, Risiken, Smoke-Tests.

**Offene Fragen**:
- Big-Bang-Migration eines Tabs (mit Replace) vs Side-by-Side
  (alt + neu parallel) — pro Tab entscheiden.
- Wie kommunizieren wir Änderung an User (Release-Notes,
  In-App-Banner)?

**Dependency**: ALLE anderen WPs.

**Blockiert**: Implementation-Start.

**Parallel mit**: nichts.

**Fälligkeit**: T4 (vor jeder Tab-Implementation-Änderung).

---

## 4. Selbstkritik (Schwächen S1-S6) — wo abgefedert

| Selbstkritik | Adressiert in |
|---|---|
| S1 Phase-4 zu kurz veranschlagt | WP8 explizit dimensioniert, Webapp-Echo getrennt. |
| S2 Multi-turn Path A ist Tech-Debt | WP7 + WP8: Path B als MVP, nicht A. |
| S3 Chat-Provider unklar | WP7 explizit Provider-Strategie. |
| S4 Schreib-Tools vor State-Sync | WP6 ist Voraussetzung für WP7-Schreib-Tools — DAG erzwingt Reihenfolge. |
| S5 Cache-Mutation während Chat | WP7 Read-Only-Modus-Frage. |
| S6 Tools auf ALIMA hardcoded | WP7 workflow-aware Tools, generisch via Extra-Introspection. |
| S3 Chat-Provider-Wahl unklar | WP11 Provider-Portabilität (Chat-Provider-Default als Spezialfall). |

## 5. Globale offene Fragen (über WP-Grenzen hinweg)

Diese müssen wir explizit irgendwann durchsprechen — sie tauchen in
mehreren WPs auf:

1. **Telemetrie**: Tab-Nutzung gemessen? Wenn nein, alles Migrations-
   risiko basiert auf Bauchgefühl.
2. **Forschungspfad-Definition**: was bedeutet "klassisch ist
   Forschung" konkret — wer publiziert was, mit welcher Pipeline?
   Beeinflusst WP2 (Vergleichbarkeitskriterien) + WP10 (Schutz-Tests).
3. **Reproduzierbarkeit**: agentic ist tendenziell weniger
   reproduzierbar (MetaAgent-Loop, Tool-Reihenfolge). Forschung
   verlangt Determinismus → Default-classic für reproduzierbare Runs?
4. **User-Persona**: Bibliothekarin (manuell, einzelne Titel) vs
   Erwerbungsabteilung (Batch) vs Forscherin (Reproduzierbarkeit) —
   unterschiedliche UX. Pro Persona eigener Default-Workflow?
5. **Long-term Vision**: Chat als universelles Frontend (statt Tabs)
   — soll die Roadmap dorthin laufen oder Tabs bleiben Hauptweg?
6. **Provider-Empfehlung pro Workflow**: ALIMA „kennt" Empfehlung für
   alima.yaml = Modell X, für catalog_search = Modell Y? (WP11)
7. **Modell-Familien-Matrix**: welche Familien werden offiziell
   supported (Anthropic Claude, OpenAI GPT, Google Gemini, Ollama
   {Qwen, Llama, Gemma, DeepSeek})? Wer pflegt Test-Setups? (WP11)

## 6. Vorgeschlagene Reihenfolge nächster Schritte

1. WP-Liste reviewen, ggf. WPs splitten/mergen.
2. WP1, WP2, WP3, WP9 starten (parallel, verschiedene Zeit-Slots
   möglich).
3. Decision-Calendar T1: Entscheidungen aus WP1-3+9 zusammenbringen.
4. WP4 + WP6 (parallel) startbar.
5. WP5 + WP7 nach WP4 + WP6 zusammenbringen.
6. WP8.
7. WP10 als Synthese.

## Related

- [`agentic_workflow.md`](agentic_workflow.md)
- [`workflow_yaml_spec.md`](workflow_yaml_spec.md)
- [`agentic_chat_plan.md`](agentic_chat_plan.md)
- [`ui_requirements_catalog.md`](ui_requirements_catalog.md)
