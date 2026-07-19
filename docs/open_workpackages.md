# Offene Workpackages — Register (July 19, 2026)

> Ausführungsreife Kurz-Specs für alles, was in der ALIMA-Umstrukturierung
> code-seitig noch aussteht. Ergebnis des Struktur-Audits + Chat-UX-WPs
> (July 19). Konvention: pro WP Ziel / Ansatz / Dateien / Verifikation /
> Größe. Erledigte WPs wandern in `AIChangelog.md`.
> Klick-Tests macht der Operator on the fly beim Benutzen — sie sind kein
> Gate; Brüche werden gemeldet.

Empfohlene Reihenfolge: **D1 → D2** (die Daten-Achse ist der letzte inhaltlich
unfertige Teil der Umstrukturierung); K1–K5 als unabhängige Lückenfüller;
T-Reihe anlassbezogen; V1 ist neues Terrain.

---

## D — Daten-Achse (der eigentliche Rest der Umstrukturierung)

### WP-D1 · BibRecord P0 — Datenformen vereinheitlichen
**Status:** analysiert + verifiziert (July 10), Entscheidung getroffen: machen.
**Problem:** Das Plugin-System hat die *Verrohrung* vereinheitlicht, nicht die
*Daten*: Round-Trip-Rename-Shims (`gnd_search_core.py:116-128` ↔
`aggregate.py:188-193`), duale DOI-Shapes (Capitalized vs snake_case), `ddc`
mit 3 Werttypen, Klassifikation in 4 Kodierungen.
**Ansatz:** P0 = F-1-Collapse übers bestehende `ResultItem` (kein neuer Typ als
Erstschritt). Vorab drei Spec-Entscheidungen fixieren, die der Draft offen
lässt: (a) `authors`-Typ (Liste strukturiert vs. String), (b) URL-Rollen
(web/opac/api getrennt statt ein Feld), (c) `count`-Konvention (Pool-`count`=1
vs. `display_count` ist seit `038738e` etabliert — als Kontrakt festschreiben).
**Doc:** [`wp_records_as_first_class.md`](wp_records_as_first_class.md)
(Findings + Decision Point). **Größe:** groß (mehrere Sessions, phasenweise).
**Verifikation:** Suite + Vergleichslauf klassisch↔agentisch auf demselben Input.

### WP-D2 · Notation-Generalisierung — (system, notation)-Paare
**Status:** Naht existiert (Mixin-Extraktion `32670d5`), Arbeit unbegonnen.
**Ziel:** weg von DK-zentrisch; DK/DDC/RVK als gleichwertige
`(system, notation)`-Paare (Operator-Richtung, siehe Memory
`general_notation_direction`). Umfasst: DDC-Harvest, `dk_*`-Renames, Logik in
`_pipeline_dk_steps.py`/`_pipeline_rvk_scoring.py` generalisieren.
**Ansatz:** auf den frisch extrahierten Mixins arbeiten (nicht mehr im
5000-Zeilen-Executor); Datenform mit WP-D1 abstimmen — die Paare sind ein
`ResultItem`/`BibRecord`-Feld. Deshalb **nach oder mit D1**, nicht davor.
**Größe:** groß. **Verifikation:** Suite + DK- und RVK-Pipeline-Läufe.

---

## K — Konsolidierungen (je ~1 Session, unabhängig)

### WP-K1 · BusRenderBridge — die drei Bus-Konsumenten vereinigen
**Problem:** Drei StateBus→Renderer-Konsumenten mit fast identischen
Handler-Körpern, die im Gleichschritt editiert werden müssen (Präzedenz: der
§9.3-Fehlertext-Edit musste 2× angewendet werden): GUI `BusEventMixin`
(`src/ui/_chat_panel_bus.py`), Webapp `_SessionBusSubscriber`
(`src/webapp/render_bridge.py`), `UnifiedMessageRenderer.subscribe`
(Mini-Logs: `image_analysis_tab.py:363`, `analysis_review_tab.py:231`).
Ownership-Notizen stehen in allen drei Dateien.
**Ansatz:** Qt-freie geteilte `BusRenderBridge`-Klasse (Handler nehmen den
Renderer); Webapp nutzt sie direkt, GUI-Mixin wird dünne Subklasse
(Elapsed-Meta, Accumulator-Wiring), Renderer-`subscribe` bleibt der
Tool-only-Embedder-Pfad. ⚠️ `tests/test_state_bus.py` prüft Quelltext-Strings
der Subscribe-Wiring in `pipeline_chat_panel.py` — Wiring dort verbatim lassen;
`tool_events=False`-Guard (Chat) erhalten.
**Verifikation:** Suite (Bus-Tests nur in Full-Suite-Ordnung werten) +
Diff-Nachweis, dass Handler-Bodies verbatim wandern.

### WP-K2 · Lobid-Label-Lookup aus gnd_local — den 25-MB-Dump abschaffen
**Problem (F-10 Mittelfrist):** `LobidSuggester.transform` braucht den
JSON-LD-Dump nur als GND-ID→Label-Tabelle (`providers/lobid/suggester.py`,
`gnd_subjects`, seit `3e4d0cd` lazy + persistenter Pfad) — ein zweiter
GND-Authority-Bestand neben `gnd_local.db`.
**Ansatz:** Vorab prüfen, welche Label-Felder der Dump liefert, die
`gnd_local.db` fehlen (Synonyme/Alternativformen?). Dann Lookup-Funktion
gegen den `LocalGndStore` (Fallback: nackte GND-ID wie heute bei
Dump-Misses), Download-Pfad + GUI-Download-Aktion (`_main_window_data.py`)
entfernen. **Verifikation:** `tests/test_lobid_transform.py`-Golden bleibt
byte-identisch für abgedeckte IDs; Live-lobid-Suche.

### WP-K3 · DOI-SystemConfig-Mirror abbauen (P8)
**Problem:** Der letzte Derived-Mirror: `derive_input_mirrors` spiegelt die
DOI-Plugin-Instanzen bei jedem Save in `SystemConfig`-Felder (P7 hat bewusst
nur die Such-Mirrors entfernt).
**Ansatz:** exakt die P7-Methodik (`docs/wp_plugin_convergence.md`):
Leser-Inventar (Achtung: P7 fand tote Leser-Cluster — AST-prüfen), Leser auf
`factory.primary_settings(cfg, "doi_*")` umstellen, `derive_input_mirrors` +
Mirror-Felder löschen, Legacy-JSON bleibt einmalige Migrations-Eingabe.
**Dateien:** `config_models.py`, `config_manager.py`, `plugin_migration.py`,
DOI-Leser (`doi_resolver`-Aufrufer, Webapp `_get_doi_config`).
**Verifikation:** Roundtrip-Tests wie `test_plugin_config_roundtrip.py`.

### WP-K4 · Tool-Data-Passthrough: swb/catalog
**Status:** 🚧 letzter offener Teil des WPs
([`wp_tool_data_passthrough.md`](wp_tool_data_passthrough.md)); DOI/finc/lobid ✅.
**Problem:** `search_swb`/`search_catalog` (+ `catalog_titles`) reduzieren
per-Record; Agenten sehen nur den Pipeline-Ausschnitt.
**Ansatz:** Pool-View `{count,gndid,ddc,dk}` unverändert lassen (Ranking-
Landmine!), volles `record` *zusätzlich* durchreichen (Muster: lobid
`transform_agent_view`). **Verifikation:** Transform-Golden-Tests bleiben
byte-identisch; neuer agent_view-Test je Quelle.

### WP-K5 · SearchTab (`find_keywords.py`) — refresh + Zweck
**Problem (Operator-Eintrag `src/ui/CLAUDE.md` #0):** Quellen-Checkboxen
werden einmalig in `init_ui()` gebaut → Plugin-Enable/Disable greift erst
nach Neustart; `_refresh_plugin_tools` (`_main_window_settings.py:61`)
aktualisiert nur die ToolRegistry.
**Ansatz:** Minimalfix zuerst: `SearchTab.refresh_sources()` (Checkboxen aus
`_gnd_source_ids()` neu bauen, Auswahl erhalten), in
`_refresh_plugin_tools` einhängen. Die größere Überarbeitung erst nach
Zweck-Entscheidung des Operators (Pipeline ist Haupteinstieg — Memory
`tab_usage_reality`). **Größe:** Minimalfix klein.

---

## T — Anlassbezogen / geparkt (decide-on-touch)

- **T1 · Gemini-Streaming-with-Tools + LlmService-Split.** Gemini ist der
  letzte Provider ohne Tool-Streaming (`_generate_gemini_with_tools`
  completes-then-delivers). Im selben Zug den ~2900-Zeilen-`LlmService` in
  Per-Provider-Backends splitten (F-5-Mixin-Methodik) — nicht vorher, nicht
  getrennt.
- **T2 · Webapp-Session-Persistenz.** Sessions/`render_buffer`/`chat_history`
  sind in-memory (`session_state.py`), weg bei Restart; Verlauf existiert nur
  als Plan ([`webapp_session_history.md`](webapp_session_history.md)). Eher
  Feature als Umstrukturierung.
- **T3 · Secrets/Keyring.** API-Keys liegen im Klartext in
  `~/.config/alima/config.json` (bewusst dokumentiert in
  `src/utils/CLAUDE.md`; Env-Override existiert). Eigenes, plattformabhängiges
  Paket.
- **T4 · i18n-Ausbau.** Settings-Widget für `UIConfig.ui_language` +
  Nicht-Chat-Oberflächen in den Katalog (Konvention: Memory
  `i18n-convention`; de+en immer synchron, Parity-Test vorhanden).
- **Shims (F-11-Politik)** und **F-7/F-8** (Worker-Cancellation,
  ProviderModelSelector): nur bei Berührung —
  [`cleanup_findings.md`](cleanup_findings.md).

---

## V — Vision (neues Terrain, keine Umstrukturierung)

### WP-V1 · Agentic Hauptagent (`main_agent:`-Block)
Meta-Orchestrator im Workflow-YAML, der Sub-Workflows als Tools aufruft —
größter unbegonnener Architektur-Vision-Punkt (CLAUDE.md Future #5). Baut auf
`WorkflowExecutor` + `ToolRegistry` auf; braucht eine eigene Design-Runde
(Tool-Schema für Workflows, Kontext-Übergabe via `SharedContext`,
Abbruch-Semantik). Erst nach der Daten-Achse sinnvoll.
