# WP: Plugin-Konvergenz — Orchestrierung, Built-in-Entkopplung, Lookup-Vertrag

> **Status:** P1–P5 + P6a + **P7 ✅ CODE-COMPLETE** (July 17, s. P7-Block unten),
> **P6f geschlossen** (Fehlbefund, s.u.). Offen: Operator-Klick-Tests.
> Findings aus dem Drei-Agenten-Audit der
> Plugin-Umstellung (July 14). Die Quick-Wins des Audits (toter k10plus-Wrapper,
> Doc-Drift, Lookup-URL-Warnung + -Memoization, GUI-Bypässe) sind separat umgesetzt,
> siehe `AIChangelog.md` July 14.
>
> ⚠️ **Zweite Korrekturrunde July 16** (Code-Verifikation vor der Ausführung): vier
> **Live-Bugs** gefunden (RVK-Cache-Shape-Kollision → RVK-Codes verschwinden lautlos;
> `gnd_batch_search` schluckt `agg["error"]` → falsche „echte Nulltreffer";
> `_store_suggester_raw` ignoriert die Cache-Einstellung; GND-Counter) und **vier
> weitere Doc-Aussagen widerlegt** (P6-Dreifachheit, P7s „~298", „kein Test pinnt die
> Divergenz", „je ein Mirror-Pendant"). Jede Korrektur steht bei ihrem Punkt.
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
   P3 unten. ⚠️ **Teil-Widerruf (July 16): der Konflikt schläft nicht, er ist aktiv** —
   und trifft zu 100 %, nicht probabilistisch. Siehe die Korrektur unten.
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
**Entschieden (July 15):** (a) offen gelassen — `RvkMarcIndex` hat genau eine Call-Site
(`_inject_rvk_api_fallback`, `pipeline_utils.py:2884`); P3 beschränkt sich auf (b).

#### ⚠️ Vorbedingung: die Cache-Shape-Kollision ist aktiv, nicht schlafend (July 16)

Am Code verifiziert. `rvk_validate`s einziger Parameter *ist* sein Cache-Key → `cache_params`
ist auf beiden Seiten `{}` → **beide Writer treffen zu 100 % dieselbe Zeile**
`("rvk_validate", code, {})`:

| Writer | gecachter Wert |
|---|---|
| Pipeline `_validate_code` (`pipeline_utils.py:2690-2693`) | **innerer** Dict — hat `status` |
| Tool-Handler (`tool_registry.py:1550-1593`) | **äußerer** Dict `{"notation","result"}` — kein `status` |

`cache.py:64-65` erklärt das Teilen sogar zur Absicht („`source` should match the plugin tool
name so the pipeline and the agent share cache entries") — aber die Formen widersprechen sich.
Folge: `pipeline_utils.py:2730-2733` liest `.get("status") == "standard"` → bei einer
tool-geschriebenen Zeile `None` → **der RVK-Code fällt lautlos aus `standard_validated_codes`.**
Plausibel, falsch, ohne Fehlermeldung. Klassik (echter `km`) und Agent kollidieren **heute
schon** cross-session; P3(b) fügt einen dritten Writer im *selben Lauf* hinzu und macht es
zuverlässig. `rvk_search` hat dieselbe Asymmetrie (`["results"]` innen vs. voller Dict außen),
kollidiert aber nur bei `max_results=6`.

**Der Shape-Fix gehört daher vor P3** (eigener Commit C2, ~2 Edits: Unwrap außerhalb
`cached_call` ziehen). Danach ist P3(b) wieder das, was dieses Doc verspricht: ein kwarg.

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

**Entschieden (July 15): Search-Parität** — Disable gated beide Pfade.
`resolve_lookup_instance` liefert für eine *deaktivierte* Instanz einen synthetischen
enabled-Default („Pipeline-Anker bricht nie", `resolve.py:40-42`): Operator deaktiviert
z.B. `rvk_api` im Plugin-Tab → Agent-Tool verschwindet, aber die klassische
Pipeline-RVK-Validierung läuft (mit Default-Settings!) weiter. Das ist das Gegenteil
der Search-Familie (`factory.enabled_gnd_provider_ids` gated beide Pfade).

⚠️ **Korrektur (July 16):** die frühere Warnung „kein Integrationstest schreibt die
Divergenz fest — wer `resolve_lookup_instance` fixt, ändert still das Pipeline-Verhalten"
ist **falsch**. `tests/test_lookup_plugins.py:446-456`
(`test_resolve_falls_back_when_instance_disabled`) pinnt sie direkt und wird laut rot.
Er ist zu **invertieren, nicht zu löschen** — er ist das natürliche Zuhause der neuen
Semantik. Im Geiste stimmte die Warnung nur für den *Pipeline-Folgeeffekt*: den deckt
kein Test ab. Ebenso ungedeckt: der Synthesize-All-Fallback in `_lookup_instances`
(`if insts:`) — kein Test baut eine Config mit allen Lookups deaktiviert.

### P6 — Source→Tool-Maps (P6a; die „Dreifachheit" war ein Fehlbefund)

**Korrektur (July 16, am Code verifiziert):** die drei sind **Schichten, keine Kopien** —
und die Vereinheitlichung (ehemals P6f) ist **geschlossen** (Operator-Entscheid):
1. `service.py` (279 LoC) orchestriert **Provider** für einen Multi-Source-Merge
2. MCP-Handler orchestrieren **einen Provider je Tool** für einen LLM-JSON-Vertrag
3. `deterministic_functions.gnd_batch_search` (235 LoC) orchestriert **Tools** — bewusst:
   nur so gibt es `CachingToolRegistry`-Dedup, `tool_calls`-Accounting und
   Workflow-Config-Injektion. Es fasst `build_provider` nie an.

Ein Collapse auf eine Implementierung (~1870 LoC) würde den agentischen Pfad von der
Tool-Registry entkoppeln und genau diese drei Eigenschaften verlieren — das ist eine
Architekturänderung, keine Dedup.

Zwei Teilbefunde hielten der Prüfung nicht stand:
- **„zweiter Aggregate-Reader" — falsch.** `_handle_aggregate_gnd_results` (65 LoC) ist ein
  *reiner Cache-Read*, der den Pool liefert; `service._search_from_raw` (44 LoC) fährt eine
  *Live-Suche* und liefert nested+errors. Beide rufen schon dasselbe `aggregate_gnd_results`.
  Echt geteilt: ~5 Zeilen.
- **„zweiter Raw-Dual-Write" — wahr, aber ~20 LoC** — und das Interessante ist ein *Bug*,
  keine Dublette: `_store_suggester_raw` hat kein `cache_pref_enabled`-Gate (anders als
  `provider_base._store_raw_responses`, der Input- und der Lookup-Pfad). → C3.

Die schärfere Dublette nennt dieses Doc gar nicht: `_make_instance_handler` (63 LoC) ↔
`_make_gnd_keywords_handler` (64 LoC), inkl. eines zweiten `_serialize_provider_gnd`.
Optional als P6d nach P6a.

**P6a (offen, der einzige Punkt, der das Risiko wert ist):** hartkodierte source→tool-Maps
in `deterministic_functions.py` — Z. 68-71 (`gnd_batch_search`), Z. 955-958
(`catalog_multi_search`), Z. 1511 (bare `search_lobid`, **ohne** Escape-Hatch) sowie
Z. 1101/1253 (`search_catalog_titles`). Der `source_tool_map`-Ausweg existiert und wird von
**null** Workflows genutzt. Unter dem POC sind die gemappten Ids deaktiviert → leerer Pool.

**Daten-Achse nicht hier duplizieren:** Rename-Shims (`gndid→gnd_ids→gndid`),
`determinancy`-Typo, 4 Klassifikations-Kodierungen + Webapp-Normalizer
(`webapp/result_serialization.py:58-144`), count-Familie (8+ Namen) sind im
**`BibRecord`-WP** erfasst → [`wp_records_as_first_class.md`](wp_records_as_first_class.md)
+ Counter-Bug [`wp_gnd_counter_divergence.md`](wp_gnd_counter_divergence.md).

### P7 — Config-Mirror-Abbau ✅ ABGESCHLOSSEN (July 17)

> **Ergebnis:** `CatalogConfig` + `SearchProviderConfig` + `CatalogType` sind gelöscht.
> `AlimaConfig.plugins` ist die einzige Wahrheit; gelesen über `factory.primary_settings`,
> geschrieben über `set_primary_settings`. Die Legacy-JSON-Sektionen sind einmalige
> Migrations-Eingabe und verschwinden beim nächsten Save. Der DOI-`SystemConfig`-Mirror
> bleibt (nicht P7). Commits `8596d97` (A) · `fb2bf75` (B) · `354a39d` (C) · D.
>
> **Drei Planannahmen haben der Ausführung nicht standgehalten:**
> 1. **Der größte Cluster war tot, nicht migrationsbedürftig.** `execute_dk_search`s fünf
>    `catalog_*`-Parameter waren im 510-Zeilen-Rumpf **nirgends** referenziert (AST-geprüft)
>    → 16 der 29 Lesungen sind freie Löschungen. Dazu: `PipelineStepConfig` hat keinen
>    `__getattr__`-Proxy, `getattr(step_config,'catalog_token','')` lieferte **immer** `''` —
>    die „Step-Config schlägt globale Config"-Vorrangkette hat nie funktioniert.
> 2. **„`synthesize_search_instances` nimmt ein plain dict" war eine Falle, kein Fakt.**
>    `getattr(cc, attr, None)` lieferte für ungesetzte Felder die *Dataclass-Defaults*
>    (`catalog_type='libero_soap'`, `strict_gnd…=True`, `finc_default_limit=20`). Ein naives
>    `dict.get` hätte daraus `None` gemacht und via `cls(**settings)` an die
>    Provider-Konstruktoren gereicht → stille Fehlkonfiguration beim Upgrade. Absente Keys
>    werden **weggelassen**; Guard: `MigrationTest.test_absent_legacy_keys_are_omitted_not_none`.
> 3. **Der Mirror hatte einen Live-Bug.** `catalog_web_record_url` ist Ziel *zweier*
>    Mappings (catalog + finc); `derive_search_mirrors` iterierte in Dict-Ordnung → finc
>    gewann. Katalog-URL gesetzt + finc-Feld leer ⇒ Mirror `''` ⇒ **keine OPAC-Links**.
>    Am alten Code demonstriert, jetzt explizite Präzedenz katalog-vor-finc.
>
> **Vorarbeit `9ac148a`:** die Suite las die echte `~/.config/alima/config.json` — grün oder
> rot je nach letztem GUI-Klick (7 Fehler bei rvk_api aus + catalog an, 13 ohne Config).
> Zehn Tests hermetisch gemacht; verifiziert über drei Config-Zustände.

<details>
<summary>Ursprüngliche Analyse (July 16, vor der Ausführung)</summary>

**Korrektur (July 16, nachgezählt):** die Zahl **„~298 Reader" ist eine 6×-Überschätzung**
und war der Grund, P7 für unmachbar zu halten. Echt sind es **53 Attributlesungen in 10
Dateien**. Die 302 String-Treffer zählen u.a. `resolve_dk_extractor`s 15 kwargs (0 Reads),
`marcxml_client`s Konstruktor-Parameter (0 Reads), `ConfigField`-Key-Deklarationen (das
*Ersatzsystem*) und ~84 Wizard-**Schreibzugriffe**. Größter Cluster: `execute_dk_search`
mit 18 — **die frisst P4**. Danach bleiben ~35. `SearchProviderConfig`: 28 Refs, davon 16
in Tests → **5 Produktions-Sites**.

**Ebenfalls falsch: „je ein Mirror-/Seeding-Pendant".** `lookup` hat **keinen**
Mirror-Ableiter und **keinen** Reverse-Sync — by design (`plugin_migration.py:215-222`: es
gibt keine Legacy-Config-Sektion zu spiegeln, die Liste kommt direkt aus der
`LOOKUP_REGISTRY`). Lookup ist damit der *Beweis, dass das Muster billig skaliert*, nicht
ein Beleg für Triplizität. Echte Dedup zwischen den drei Synthesizern: ~16 Zeilen (drei
verschiedene Datenquellen, drei verschiedene Enable-Semantiken). Die einzige echte
Dublette der Datei nennt dieses Doc nicht: `synthesize_lookup_instances` ↔
`synthesize_missing_lookup_instances` (~24 geteilte LoC).

**Free deletes (verifiziert):** `sync_instances_from_mirrors` (`plugin_migration.py:300-327`)
ist **toter Code** — Definition + ein Test + ein staler Docstring, null Produktions-Caller;
seine Begründung („damit die Legacy-Catalog/System-Tabs weiterlaufen") ist obsolet, die Tabs
sind entfernt. Dazu `plugin_migration.py:14-16` (Docstring behauptet, `catalog_type` und die
Web-URLs seien ungemappt — `SEARCH_FIELD_MAP:29-33` mappt alle vier) und
`sru_database`/`sru_schema` (tot; die einzigen echt ungemappten Felder).

**End-State:** `CatalogConfig` löschen (17 der 19 Felder haben ein Instanz-Zuhause, 2 sind
tot) + `SearchProviderConfig`. Blocker: die Load-Migration muss überleben →
`synthesize_search_instances` nimmt ein plain `dict`; `test_plugin_config_roundtrip.py:65-77`
(assertet `asdict(cfg2.catalog_config) == cat_before`) **ist** D-8s Vertrag und muss auf
Instanzen umgeschrieben werden. Gated auf P2 (finc) + P4.

</details>

## Empfohlene Reihenfolge + Risiken

**Aktualisiert July 16** (Plan: `~/.claude/plans/counter-bug-dann-p2-immutable-wadler.md`).
Vier verifizierte Live-Bugs werden **vorab** als eigene Commits gefixt, damit die
Refactorings darunter verhaltenserhaltend bleiben und gegen ein dichteres Testnetz laufen:
`C1` GND-Counter · `C2` RVK-Cache-Shape · `C3` Agentik-Gates (`agg["error"]`-Schlucker +
fehlendes `_store_suggester_raw`-Gate).

Reihenfolge: `C0 Docs → C1 → C2 → C3 → P2.1 → P2.2 → P4 → P3 → P5 → P6a → P7`

| Schritt | Warum dort | Hauptrisiko |
|---|---|---|
| P1 ✅ | Löst das sichtbarste Plugin-Versprechen ein | erledigt (`5360e95`) |
| P2 | Einziger Provider ohne Factory-Guard; danach gilt „alle Primaries über Factory". **Blockiert P4 + P7.** | finc ist institutionsspezifisch — Live-Verifikation nur mit erreichbarem finc |
| P4 | Direkt nach P2: *ein* Subsystem (finc/catalog/DK-Konstruktion), teilt sich **einen** finc-Vergleichslauf. Frisst 18 der 53 P7-Reader. | DK-Suche ist klassik-kritisch; Migrations-Hook `sru.dk_enabled` — falsch abgeleitet = stiller DK-Verlust beim Upgrade |
| P3 | Unabhängig + mechanisch. **Braucht C2 zwingend vorher.** | `rvk_lookup` ist Pipeline-Anker: Werteform darf sich nicht ändern; der bestehende Test mockt den Executor komplett und ist kein Guard |
| P5 | Nach P3, weil P5 die *gewollte* Verhaltensänderung ist — allein in seinen Vergleichslauf | 120-Kandidaten-Fan-out schneidet bei deaktiviertem `rvk_api` kurz |
| P6a | Der einzige P6-Punkt, der das Risiko wert ist | `sources` wird Filter- statt Literal-Id-Semantik → speist `rank_pool.source_count` |
| P7 | Gated auf P2 + P4; danach ~35 Reader | Config-Round-Trip |

## Was ausdrücklich GUT ist (nicht anfassen)

- Trust-Modell (AST-Scan, Hash-Pinning, Approval-Ledger) existiert genau **einmal**
  (`src/core/plugins/security.py` + `loader.py`), alle drei Familien + Bundles nutzen es.
- `raw_cache_params_for` als Single-Source für WP2-Cache-Keys (Writer + alle Reader).
- Secrets nur runtime-only via Env-Override in den Build-Pfaden.
- `CachingProvider` behandelt die F-4-count-Landmine an genau einer Stelle.
- DNB + k10plus sind seit Phase D (+ Quick-Wins July 14) echt single-path.
