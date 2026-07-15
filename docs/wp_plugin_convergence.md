# WP: Plugin-Konvergenz — Orchestrierung, Built-in-Entkopplung, Lookup-Vertrag

> **Status:** P1 ✅ CODE-COMPLETE (July 15), P2–P5 📋 GEPLANT + entschieden,
> P6/P7 Backlog. Findings aus dem Drei-Agenten-Audit der Plugin-Umstellung
> (July 14). Die Quick-Wins des Audits (toter k10plus-Wrapper, Doc-Drift,
> Lookup-URL-Warnung + -Memoization, GUI-Bypässe) sind separat umgesetzt, siehe
> `AIChangelog.md` July 14.
>
> ⚠️ **Zeilenrefs unten sind der Stand vom 14. Juli und teils gedriftet** — beim
> Ausführen neu greppen. Verifiziert am 15. Juli: `_handle_rvk_lookup` ist `:1120`
> (nicht 1103), Lookup-Cache-Gate `:1556-1599` (nicht 1552-1578),
> `_lookup_instances` `:1497-1515` (nicht 1480-1498).

## Korrekturen am Befund (July 15, bei der P1-Ausführung verifiziert)

Vier Aussagen dieses Docs haben der Prüfung nicht standgehalten:

1. **Der Aggregate-Default war nie ein Provenienz-Risiko.** Die „Vergleichslauf
   nötig"-Warnung zu P1 ging davon aus, dass ein abgeleiteter Default die
   agentische Provenienz ändert. Tut er nicht: `gnd_local` ist zwar
   `GND_KEYWORDS`-fähig und aktiviert (der Default liefert **vier** Ids), hat aber
   keinen `.suggester` → `_source_transform` → `None` → der *bestehende* Filter
   verwirft ihn. Byte-identisch, strukturell und nicht per Config-Glück. Als Test
   festgenagelt (`test_aggregate.py::AggregateDefaultSourcesTest`), kein manueller
   Lauf nötig.
2. **P1s Nutzenversprechen stimmte nur halb.** „`poc_*`-E2E kann dann auch
   Aggregation/agent_view abdecken" — `agent_view` ja, Aggregation nein: der
   agentische Einstieg `gnd_batch_search` übergibt `sources` aus der hartkodierten
   `source_tools`-Map (`deterministic_functions.py:68-71`), unter dem POC sind diese
   Ids deaktiviert → leerer Pool. Das ist **P6**, nicht P1.
3. **P3(b) ist ein kwarg, kein Umbau.** Der Raw-Cache-Bypass von `rvk_lookup` ist
   `cache_manager=None` (`tool_registry.py:1151`); die RVK-Aufrufe darunter sind seit
   Phase D plugin-geroutet und `cached_call`-gewrappt, sie bekommen nur `km=None`.
   Kein Werteform-Risiko — aber der Fix *aktiviert* einen schlafenden Konflikt, siehe
   P3 unten.
4. **P5 hat zwei Stellen, nicht eine.** Neben `resolve.py:40-42` synthetisiert
   `_lookup_instances` (`if insts:`) bei „alle deaktiviert" sämtliche Lookup-Typen als
   enabled zurück. Eine Entscheidung, zwei Fixes.

## Operator-Entscheidungen (July 15)

| Punkt | Entscheid |
|---|---|
| **P5 Disable-Semantik** (war vertagt) | **Search-Parität** — Disable gated beide Pfade; Pipeline-Verhaltensänderung, Vergleichslauf nötig |
| **P3(a) RvkMarcIndex** | offen gelassen — P3 beschränkt sich auf (b) |
| **P4 `catalog_type`** | **sru bekommt eigenen `dk_enabled`** (symmetrisch zu finc); `catalog_type` wird vestigial (Migration nötig); löst Debt D-5 mit |

## Befund in einem Satz

Die *Konstruktions*-Schicht ist echt vereinheitlicht (Registry + `factory.build_provider`
+ `build_lookup` + einmaliges Trust-Modell in `src/core/plugins/security.py`); die
*Orchestrierung* darüber existiert dreifach (service.py / MCP-Handler-Generatoren /
`deterministic_functions.py`), und an genau den Stellen, wo Built-in-Quell*namen* noch
Sonderbehandlung bekommen, bleibt das Plugin-Versprechen („externes Plugin = gleichwertig")
uneingelöst.

## Priorisierte Punkte

### P1 — WP2-Entkopplung von Built-in-Namen ✅ CODE-COMPLETE (July 15)
Alle vier Built-in-Namen-Kopplungen aufgelöst, jede über einen **bereits vorhandenen**
Deklarationskanal (Leitregel: die Deklaration muss dort liegen, wo `deploy_poc.py` sie
beim Kopieren mitnimmt — also im Provider-Dir, nicht in einer zentralen Map):

- `_SOURCE_PARAM_KEYS` → Klassenattribut **`raw_cache_param_keys`** (Base-Default
  `("search_type",)`; nur swb + finc weichen ab — die Map war zu 60% redundant) +
  Registry-Accessor `raw_cache_param_keys(source)`. *Nicht* `ProviderToolSpec`:
  `catalog_titles` ist ein source label, keine provider id, und der primäre Writer
  keyt auf `self.id`. *Nicht* aus `default_opts` ableitbar: catalog deklariert keine.
- `_agent_view_deriver` → `getattr(underlying_suggester(p), "transform_agent_view")`,
  wörtlich gespiegelt von `_source_transform`. Ein Bool-Flag hätte nicht gesagt,
  *welche* Funktion.
- Aggregate-Default → `enabled_gnd_provider_ids(config=self._alima_config())` mit
  explizitem `is not None`-Check (`[]` = alle deaktiviert ≠ `None` = Config unlesbar).
- `hand_wired` gelöscht; canonical → `_make_search_handler` für **alle** Typen (der
  Pfad ist factory-gestützt, antwortet also aus der eigenen Klasse des Plugins). Das
  eine `"finc"`-Literal bleibt bewusst *in* `_make_search_handler` stehen → P2 ist
  eine Löschung. `raise ValueError` bei unbekannter `result_shape` entfernt (war
  unerreichbar, wurde durch den Edit erreichbar → hätte die *ganze* Tool-Liste
  gesprengt).

**Mit erledigt:** `_attach_agent_view` baute den Cache-Key von Hand
(`{"search_type": …}`) statt über `raw_cache_params_for` — passte nur für lobid
zufällig; `find_keywords` kollabierte `None`/`[]` und erfand an zwei weiteren Stellen
`"lobid"`, wenn keine Quelle aktiv war.

**Nutzen (ehrlich):** kopierte/externe Provider bekommen `agent_view`, eigene
Cache-Keys und Default-Provenienz. Die *agentische* Aggregation bleibt POC-untauglich,
bis P6 die `source_tools`-Map auflöst (siehe Korrektur 2 oben).

### P2 — finc-Konvergenz (mittel)
Der primäre `search_finc`-Handler baut weiter aus `CatalogConfig`, nicht aus der
Instanz-Config; finc hat damit zwei lebende Konstruktionspfade:
- `_init_suggesters` → `self._finc = FincSuggester(get_catalog_config()…)` — `tool_registry.py:139-197`
- Dispatch `result_shape=="finc"` → `_handle_search_finc` — `tool_registry.py:1640-1641`
  (nicht-primäre finc-Instanzen laufen dagegen über die Factory, `_make_instance_handler`)
- `FincProvider` re-implementiert `SuggesterBackedProvider` per Hand inkl. eigenem
  `_store_finc_raw` — `providers/finc/provider.py:177-239`

**Richtung:** Availability-/`web_url`-Logik in den `FincProvider` ziehen, Primär-Handler
auf `_provider_for` umstellen, `_init_suggesters._finc` löschen; `FincProvider` erbt
`SuggesterBackedProvider`. **Konvergenztest ergänzen** — der finc-Primärpfad ist heute
der einzige ohne Factory-Guard (`test_finc_client.py:952` testet nur das Schema).

### P3 — RVK-Konsolidierung (mittel, Entscheidungspunkt)
Drei lebende RVK-Pfade:
1. `rvk_api`-Lookup-Plugin (Phase D; Pipeline + Agent-Tools `rvk_search`/`rvk_validate`)
2. `_handle_rvk_lookup` — `tool_registry.py:1103` — handgeschrieben, fährt eine *eigene*
   Katalogsuche (`execute_dk_search`) + RVK-Validierung, umgeht den Raw-Cache
3. `RvkMarcIndex` direkt instanziiert — `pipeline_utils.py:2440` (MarcXML-GND-Dump)

**Entscheidung:** (a) MarcIndex als zweites Lookup-Plugin (`rvk_marc`) vs. bewusst
dokumentierte Core-Ausnahme; (b) `_handle_rvk_lookup` intern auf das Plugin umstellen
(Werteform beibehalten — es ist Pipeline-Anker-Maschinerie).

### P4 — DK-Extractor-Resolver generalisieren (mittel)
Nur der Custom-Plugin-Zweig ist capability-getrieben; die Built-ins sind hand-verdrahtet:
- if-elif `get_provider("finc"/"sru"/"catalog")` — `factory.py` (`resolve_dk_extractor`)
- ~15-kwargs-`CatalogConfig`-Wand am Call-Site — `pipeline_utils.py:4584-4604`
  (inkl. fragiler `'catalog_config' in dir()`-Guards)
- Label-Map über konkrete Client-Klassennamen — `pipeline_utils.py:4606-4610`

**Richtung:** Built-ins deklarieren ihre DK-Extractor-Settings wie der Custom-Zweig
(Instanz-Settings statt `CatalogConfig`-kwargs); Label vom Plugin (`label`-Attribut).

### P5 — Lookup-Vertrag härten (klein–mittel)
Die Lookup-Familie ist der schwächere Vertrags-Zwilling der Search-Familie:
- Kein Protocol/Basisklasse/Capabilities (Search: `SearchProvider`-Protocol +
  `SearchCapability` + typisierte `ProviderResult`) — Fehlkonformität fällt erst zur
  Laufzeit auf
- Raw-Cache-Gate doppelt: Tool-Handler `tool_registry.py:1552-1578` ↔
  `lookups/cache.py` (`cached_call`, Docstring gibt das „mirrors" selbst zu)
- Instanz-Resolution doppelt: `resolve_lookup_instance` (`lookups/resolve.py:24-42`)
  ↔ `_lookup_instances` (`tool_registry.py:1480-1498`)

**Offener Entscheidungspunkt (Operator, July 14 vertagt): Disable-Semantik.**
`resolve_lookup_instance` liefert für eine *deaktivierte* Instanz einen synthetischen
enabled-Default („Pipeline-Anker bricht nie", `resolve.py:40-42`): Operator deaktiviert
z.B. `rvk_api` im Plugin-Tab → Agent-Tool verschwindet, aber die klassische
Pipeline-RVK-Validierung läuft (mit Default-Settings!) weiter. Das ist das Gegenteil
der Search-Familie (`factory.enabled_gnd_provider_ids` gated beide Pfade). Bis zur
Entscheidung: kein Integrationstest, der die Divergenz festschreibt — wer
`resolve_lookup_instance` „fixt", ändert still das Pipeline-Verhalten.

### P6 — Orchestrierungs-Dreifachheit (groß, Backlog)
Drei parallele Build+Search+Merge+Raw-Write-Implementierungen über derselben Factory:
1. `service.py` (`search_gnd_keywords` / `_search_from_raw`) — Klassik + GUI
2. MCP-Handler-Generatoren (`_make_gnd_keywords_handler` etc., `tool_registry.py:1646-1709`)
   inkl. zweitem Raw-Dual-Write (`_store_suggester_raw` ↔ `provider_base._store_raw_responses`)
   und zweitem Aggregate-Reader (`_handle_aggregate_gnd_results` ↔ `service._search_from_raw`)
3. Agentische `deterministic_functions.py` mit hartkodierten source→tool-Maps
   (`{"swb":"search_swb","lobid":"search_lobid"}` Z. 68; `catalog_multi_search` Z. 955;
   bare `search_lobid` Z. 1511)

**Daten-Achse nicht hier duplizieren:** Rename-Shims (`gndid→gnd_ids→gndid`),
`determinancy`-Typo, 4 Klassifikations-Kodierungen + Webapp-Normalizer
(`webapp/result_serialization.py:58-144`), count-Familie (8+ Namen) sind im
**`BibRecord`-WP** erfasst → [`wp_records_as_first_class.md`](wp_records_as_first_class.md)
+ Counter-Bug [`wp_gnd_counter_divergence.md`](wp_gnd_counter_divergence.md).

### P7 — Config-Mirror-Abbau (Backlog)
- Drei parallele Synthesizer/Mirror-Ableiter in `plugin_migration.py`
  (`synthesize_search_instances:80` / `…_input_…:149` / `…_lookup_…:214` + je ein
  Mirror-/Seeding-Pendant) — eine vierte Kategorie kostet wieder beides
- Legacy-Mirrors `CatalogConfig`/`SearchProviderConfig` (~298 Reader) — bereits als
  D-5/D-8 in [`cleanup_findings.md`](cleanup_findings.md) erfasst, dort weiterführen

## Empfohlene Reihenfolge + Risiken

| Schritt | Warum zuerst | Hauptrisiko |
|---|---|---|
| P1 | Löst das sichtbarste Plugin-Versprechen ein; klein genug für eine Session | Aggregate-Default-Änderung berührt agentische Provenienz — Vergleichslauf nötig |
| P2 | Einziger Provider ohne Factory-Guard; danach gilt „alle Primaries über Factory" | finc ist institutionsspezifisch — Live-Verifikation nur mit erreichbarem finc |
| P3 | Braucht Operator-Entscheid (MarcIndex), sonst mechanisch | `rvk_lookup` ist Pipeline-Anker: Werteform darf sich nicht ändern |
| P4 | Baut auf P2 auf (finc-DK-Settings wandern in die Instanz) | DK-Suche ist klassik-kritisch; `test_dk_extractor_resolver.py` erweitern |
| P5 | Disable-Entscheid zuerst, dann mechanisch | Verhaltensänderung Pipeline-RVK je nach Entscheid |
| P6/P7 | Groß; nach P1–P5 ist die Restfläche klar umrissen | — |

## Was ausdrücklich GUT ist (nicht anfassen)

- Trust-Modell (AST-Scan, Hash-Pinning, Approval-Ledger) existiert genau **einmal**
  (`src/core/plugins/security.py` + `loader.py`), alle drei Familien + Bundles nutzen es.
- `raw_cache_params_for` als Single-Source für WP2-Cache-Keys (Writer + alle Reader).
- Secrets nur runtime-only via Env-Override in den Build-Pfaden.
- `CachingProvider` behandelt die F-4-count-Landmine an genau einer Stelle.
- DNB + k10plus sind seit Phase D (+ Quick-Wins July 14) echt single-path.
