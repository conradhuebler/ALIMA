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

### WP-D1 · BibRecord — Datenformen vereinheitlichen
**Status: P0 ✅ DONE + VERIFIZIERT, `to_bibrecord()` + F-2 ✅ DONE**
(July 19: 5 Commits ab `6991d57`; July 20: 6 Commits `2e2b647`…`87b6eb4`,
Suite 1357). EIN kanonisches GND-Pool-Vokabular `{count, gnd_ids,
classifications: {system: codes}, display_count?}` von Suggester-Vertrag v2 bis
persistierter KAS-Form; System-Keys **GROSS** (`classification_systems` als
alleiniger Owner); `BibRecord` + `to_bibrecord()` für finc/catalog/sru/k10plus;
DOI-Record-Keys klein. Verifikation ist ein deterministischer Headless-Test
(beide Pfade, eine Fixture, echter Produktionscode) statt eines Live-Laufs.
Details: `AIChangelog.md` (July 20) + [`wp_records_as_first_class.md`](wp_records_as_first_class.md).

**P1 erster Schnitt ✅ DONE (August 3, 2026):** `BibRecord.to_analysis_text()`
ersetzt die zwei duplizierten Batch-Formatter; neue Input-Sources `isbn`/`ppn`
(SRU-Lookup → `to_bibrecord` → Analyse-Text) sind der erste echte
`to_bibrecord()`-Konsument; die Landmine ist am Dispatcher entschärft
(`execute_input_extraction` löst zweistufig auf: exakte Registry-ID, dann der
vorher nie aufgerufene `can_handle`-Vertrag — Aliase `doi`/`url` erreichen
damit den einen Dispatcher). Nebenbefund behoben: `_from_sru` stringifizierte
`gnd_subjects`-Dicts. Details: `AIChangelog.md` (August 3).

**P2 ✅ DONE (August 3, 2026):** Record-Klassifikationen als Priors in
`dk_classification` — markierter Autoritäts-Block im Prompt (informiert, nie
überschreibt), RVK-Priors in `allowed_standard_rvk_map` mit Quelle
`input_record` (`_source_rank` 4); Kanäle: KAS-Feld
`input_record_classifications` (persistiert), Batch-Metadaten →
`execute_complete_pipeline`, `record_sink`-Seitenkanal im
Input-Source-Vertrag; klassischer Input-Step akzeptiert `isbn`/`ppn`.
Details: `AIChangelog.md` (August 3).

**P3 + P4 ✅ DONE (August 3, 2026):** P3 — `BibRecord.gnd_subjects` →
`merge_record_gnd_subjects` injiziert GND-verknüpfte Record-Schlagwörter als
verifizierte Pool-Kandidaten (`input_record`-Bucket, `count=1`); Kanäle wie
P2. P4 — `k10plus_resolve`-Tool: DOI ↔ PPN ↔ ISBN via K10plus, jeder Kandidat
client-seitig verifiziert (der `pica.doi`-Index tokenisiert!); nackte
10-stellige Nummern = PPN. Nebenbefund: PPN-Lookup suchte im Schlagwort-Index
und traf nie — jetzt `pica.ppn`, live bewiesen. Details: `AIChangelog.md`.

**DOI-Anreicherung ✅ DONE (August 4):** `crosswalk_doi_record` füttert die
P2/P3-Kanäle bei jedem DOI-Lauf (klassischer Input-Step = GUI+CLI, Batch,
`record_sink`); Gate = k10plus-Lookup-Plugin, ISBN-Fallback aus dem
DOI-Suffix; live bewiesen. Details: `AIChangelog.md` (August 4).

**Offen in D1 (Rest = Adoption, keine neuen Pfade):** GUI/CLI senden
`isbn`/`ppn` noch nicht als Eingabetyp; **Webapp** normalisiert DOI→Text vor
dem Start und verliert die Identität (bekommt die Anreicherung noch nicht);
Agentik erhält keine Record-Priors; `execute_notation_classification` bricht
ohne Katalog-Kandidaten weiter ab, auch mit Priors. Die drei
`ResultItem`-Nähte sind unberührt.

### WP-D2 · Notation-Generalisierung — ✅ ABGESCHLOSSEN (August 4, 2026)

> Datenform (D1-P0) + Ernte (July 20–21) + Title-Record-Keys + Sweep der
> internen `dk_*`-Namen (Schnitt 1+2, Aug 4) — Details unten und in
> `AIChangelog.md`. Strukturell offen bleibt nur die swb-Grenze (keine
> Notationen in der Antwort). Historie des WPs:
> ✅ **Ernte erledigt (July 20–21, 4 Commits `2f34e8c`…`8670d6c`, Suite 1405).**
> Der Pool trug in der Praxis **gar keine** Klassifikationen (0 von 5128 / 2364 /
> 1134 in drei echten Läufen), weil lobid und swb hart `{}` schrieben. lobid
> liefert sie die ganze Zeit mit — auf den `member`-Records. Jetzt geerntet, als
> **Ko-Vorkommens-Heuristik** mit `origin`/`count` (P0-Revision, s.
> [`wp_records_as_first_class.md`](wp_records_as_first_class.md)); BK ergänzt,
> bibliotheksspezifische Systematiken werden verworfen.
> **Abdeckung 6 %**, nicht die zunächst geschätzten 61 % — der Pool kommt aus dem
> Aggregation-Facet über die ganze Treffermenge (~100 Subjects), die
> Klassifikationen nur aus den ausgelieferten Records (15, jetzt 30 per
> `page_size`). Strukturelle Grenze der lobid-API.
>
> **Offen dazu:** swb liefert weiterhin nichts (dort ist keine Notation in der
> Antwort); `initial_gnd_classes` meint klassisch und agentisch Verschiedenes
> (klassisch = GND-Systematik-Klassen aus dem LLM-`<class>`-Tag, agentisch =
> geerntete DDC) — dieselbe Namenskollision wie F-1, bisher unentschieden.

**Status:** **Datenform ✅** (D1-P0, July 19) · **Ernte ✅** (July 20–21) ·
**Title-Record-Keys ✅ (August 4):** `search_titles` emittiert das kanonische
Dict, `dk_codes`/`rvk_codes`/`ddc_codes` sind weg; die drei kopierten
per-System-Schleifen in `tool_providers` sind eine generalisierte Funktion —
dabei fiel die DK-Frequenz-Asymmetrie (DDC-Kandidaten starben an Threshold > 1).
Details: `AIChangelog.md` (August 4).

**Rest (Entscheidungen, keine Mechanik):**
- **`dk_*`-Renames (WS2): Sweep entschieden** (Operator, Aug 4) — interne
  Python-Namen auf notation-agnostisch; **NICHT** angefasst werden die
  persistierten/Protokoll-Vokabulare: KAS-Feldnamen (Aliasse existieren),
  Config-Felder (`dk_frequency_threshold`), Step-/Task-IDs (`dk_search`,
  `dk_classification`), Tool-Namen/-Parameter, Row-Vokabel `"dk"`.
- ✅ **`initial_gnd_classes`-Kollision entschieden** (Operator, Aug 4):
  belassen + dokumentiert (data_models) — das Feld ist ein unscharfer
  thematischer Hinweis, kein exaktes Codefeld.
- swb liefert strukturell keine Notationen (keine in der Antwort).

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
**Ansatz:** kanonische Pool-View `{count, gnd_ids, classifications}` unverändert
lassen (Ranking-Landmine!), volles `record` *zusätzlich* durchreichen (Muster:
lobid `transform_agent_view`). **Verifikation:** Transform-Golden-Tests bleiben
byte-identisch; neuer agent_view-Test je Quelle.

### WP-K5 · SearchTab (`find_keywords.py`) — ✅ DONE (August 4, 2026)
Zweck-Entscheidung getroffen (Operator: „auf modernen Stand bringen") und
komplette Überarbeitung ausgeführt: `refresh_sources()` (live aus dem
Plugin-System, in `_refresh_plugin_tools` eingehängt), Suche asynchron
(`GndSearchWorker` — vorher blockte `search_gnd_keywords` den Main-Thread),
Häufigkeit = `display_count` (der „zeigt 1"-Bug lebte hier weiter),
Klassifikations-Spalte + lobid-Link, Pipeline-Mapping-Ansicht tatsächlich im
Layout, tote Signale/Methoden/Legacy-Zweige raus (1512→977 Z.).
Details: `AIChangelog.md` (August 4).

### WP-K6 · Tagzeilen-Export — drei Erzeuger, drei Ergebnisse
**Status:** 🚧 offen (gefunden bei der README-Prüfung, September 3, 2026).
**Problem:** Drei Stellen erzeugen 5550/6700-Zeilen und weichen voneinander ab:
`cli/formatters/protocol_formatters.py:395` setzt `DK` vor jede Notation,
`ui/analysis_review_tab.py:598` übernimmt das System aus dem Code,
`utils/exporters.py:136` verteilt Schlagwortketten auf 5550–5559 und ist der
einzige, der Ketten überhaupt kennt. Auf einem Ergebnis vom Januar 2026 liefert
die CLI dadurch `6700 DK RVK DK 02`. Keiner der drei prüft die Notation vor der
Ausgabe.
**Ansatz:** `generate_k10plus_lines` als einzige Quelle, CLI und GUI rufen sie
über die kanonische Export-Payload. Vorher steht die fachliche Frage: welche
Kategorien die UB Freiberg erfassen will (5550–5559 für Ketten, 6700 je System)
— das entscheidet die Signatur.
**Verifikation:** Golden-Test über zwei Ergebnisdateien, eine mit reinen
DK-Codes und eine mit System-Präfix; beide Pfade müssen dieselben Zeilen liefern.

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
