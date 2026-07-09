# CLAUDE — webindex (Webseiten-RAG-Lookup)

## [Preserved Section — Permanent Documentation]

Website/URL-Keyword-Index als **Lookup-Plugin** (`category = "lookup"`), selbst-
registriert via `@register_lookup`. Retrieval-Quelle für den ALIMA-Website-Chatbot.
Drei Concerns, sauber getrennt:

- **`store.py`** — eigene SQLite-DB (`webindex.db`), `LocalGndStore`-Muster
  (frisches `DatabaseConfig(sqlite)` + eigener `DatabaseManager`/`connection_name`,
  Pfad via `resolve_webindex_path`: `db_path`-Setting gewinnt, sonst Sibling der
  Haupt-DB). Tabellen: `pages` (URL+Cache-Text), `keywords` (zentraler Katalog),
  `page_keywords` (Inverted-Index, die "Synchronisation": `set_page_keywords`
  ersetzt die Links einer URL vollständig). Retrieval `get_pages_for_keywords`:
  Ranking = SUM(weight), Tie-Break = Anzahl unterschiedlicher match-Keywords.
  Meta/Heading-Keywords weight 1.0, LLM-Keywords weight 1.5.
- **`indexer.py`** — Operator-Crawler (`crawl_site`), nicht agent-sichtbar. BFS
  unter `base_url` (gleicher Host + Pfad-Prefix), `fetch_func` injizierbar (Tests
  netzfrei). Pro Seite: `fetch_guarded_response` → BeautifulSoup Main-Content
  (`_extract_main_text`, reusing `scrape_url`-Heuristik) oder PDF
  (`pdf_extractor.extract_text`) → Keywords aus Meta/Überschriften
  (`_extract_meta_keywords`, deterministisch) + LLM-Ergänzung
  (`_extract_llm_keywords`, JSON-Liste, fehler-toleranter Parser, nie fatal). Seed
  (depth 0) wird immer gecrawlt; `include_re`/`exclude_re` scopen nur Kinder.
  `dry_run` entdeckt URLs ohne zu schreiben.
- **`provider.py`** — `WebIndexLookup` mit 3 Tools (`search_webindex` /
  `fetch_page` / `list_webindex_keywords`). `fetch_page`: Cache-Hit → Text;
  Cache-Miss + `fetch_on_miss` → `_live_fetch` (SSRF-guarded, füllt Cache). Tools
  auto-generiert via `_generated_lookup_tools`; **keine** manuellen
  `tool_registry`-Edits nötig.

### Wiederverwendete Bausteine
- `fetch_guarded_response` (`src/utils/input_sources/url_fetch.py`) — SSRF-Guard.
- `pdf_extractor.extract_text` (`src/utils/pdf_extractor.py`) — PDF-Fallback.
- `DatabaseManager`/`SQLDialect` (`src/core/`) — Thread-safe, Dialekt-Helfer.
- `@register_lookup`/`LookupToolSpec` (`../registry.py`).

### Chatbot-Anschluss
Tools im `ToolRegistry` → `AgentLoop` (GUI-Chat, `alima agent`, `POST /agent/run`)
können sie aufrufen. Reproduzierbar: `workflows/website_rag.yaml` (ein `llm_agent`
mit `tools: [search_webindex, fetch_page, list_webindex_keywords]`).

## [Variable Section — Current State]

- ✅ **CODE-COMPLETE (July 9)**: Store + Indexer + Lookup-Plugin + CLI
  (`alima webindex crawl/stats/list-keywords/search`) + Antwort-Workflow
  `website_rag.yaml` + Keyword-Workflow `webindex_keywords.yaml` + Tests
  (netzfrei, Suite 1231 grün).
- ✅ **GUI-Indizierung (July 9)**: „Seite indizieren …"-Button auf der
  webindex-Instanz-Form via `_TYPE_ACTIONS`-Registry (`plugin_settings_tab.py`)
  + `WebIndexCrawlWorker(StoppableWorker)` (`src/ui/webindex_crawl.py`, Qt-Thread,
  Fortschritts-Log + Abbrechen). Crawl-Parameter + `llm_provider`/`llm_model` sind
  Config-Felder der Instanz (GUI-editierbar, CLI-Flags überschreiben).
- ✅ **Keyword-Extraktion als Workflow**: Indexer ruft einen injizierten
  `keyword_extractor`-Callable auf (workflow-agnostisch); `keywords.py` baut ihn
  per `webindex_keywords.yaml` (tool-less `llm_agent`, JSON-Output
  `extra.keywords: "response.keywords"`, `response_text`-Fallback). Standprompt
  lebt im Workflow, **nicht** in prompts.json (veraltet).
- ✅ **Model-Auflösung** (`keywords.resolve_crawl_model`): CLI-Flags → Instanz
  `llm_provider`/`llm_model` → globaler agentic Default
  (`UnifiedProviderConfig.resolve_default_provider_model`). Leer → Meta-only.
- ✅ **Relative-URL-Fix**: `fetch_page` löst relative Pfade (`/ub/ueber-uns`) vor
  Cache-Lookup + Live-Fetch gegen `base_url` auf (Index speichert absolute URLs).
- **Offen:** Operator-E2E gegen echte Biblio-URL (GUI-Button + Chat-Test) + Sign-off.
- **Bewusst nicht eingebaut**: Embedding-Retrieval (Operator wollte Keyword-Match);
  Re-Index-Scheduling/Cron (Voll-Re-Crawl, idempotent); inkrementelles Diff.