# Plugin: webindex — Webseiten-Index (RAG-Chatbot)

Built-in **Lookup-Plugin**: eine eigene SQLite-DB (`webindex.db`) über alle
Webseiten/URLs einer konfigurierten Haupt-URL plus eine **zentral synchronisierte
Keyword-Tabelle**. Dient als Retrieval-Quelle für den ALIMA-Chatbot bei
Webseiteninhalten (z.B. einer Bibliotheksseite). Registriert sich selbst via
`@register_lookup` beim Import des `lookups`-Packages.

## Was es tut

| | |
|---|---|
| Kategorie | `lookup` (reines Agent-Tool, freies Dict-Resultat) |
| Eingabe | Natürlichsprachliche Frage / URL |
| Ausgabe | Gerankte Treffer-URLs + Snippet (search), Seitentext (fetch) |
| MCP-Tools | `search_webindex`, `fetch_page`, `list_webindex_keywords` |
| DB | eigene `webindex.db` (LocalGndStore-Muster), Tabellen `pages` / `keywords` / `page_keywords` |

## Ablauf

1. **Indexieren** (Operator, CLI): `alima webindex crawl --base-url <url> …` crawlt
   alle Kinder der Basis-URL (BFS, gleicher Host + Pfad-Prefix), extrahiert pro
   Seite den Haupttext (BeautifulSoup) + Keywords (Meta/Überschriften + optionale
   LLM-Ergänzung) und befüllt die DB.
2. **Retrieval** (Agent): `search_webindex <frage>` tokenisiert die Frage und matcht
   gegen die zentrale `page_keywords`-Tabelle → gerankte Treffer-URLs + Snippets.
3. **Kontext holen**: `fetch_page <url>` liefert den gecachten Seitentext (Cache-Miss
   → Live-Fetch + Cache-Füllung, SSRF-guarded via `fetch_guarded_response`).
4. **Antworten**: Der Chat-Agent (`AgentLoop`: GUI-Chat, `alima agent`,
   `POST /agent/run`) beantwortet die Frage aus den geholten Texten. Alternativ
   Workflow `workflows/website_rag.yaml`.

## Konfiguration (Plugin-Instanz, Plugins-Tab)

| Feld | Typ | Default | Bedeutung |
|---|---|---|---|
| `base_url` | URL | "" | Haupt-URL der indexierten Website; auch Anker für relative `fetch_page`-URLs |
| `db_path` | TEXT | "" | Leer → Sibling der Haupt-DB (`~/.config/alima/webindex.db`) |
| `max_results` | INT | 10 | Max. Treffer pro `search_webindex` |
| `snippet_chars` | INT | 400 | Snippet-Länge |
| `fetch_on_miss` | BOOL | true | Bei Cache-Miss live fetchen |
| `fetch_timeout` | INT | 20 | Live-Fetch-Timeout (s) |
| `llm_provider` | TEXT | "" | Crawl-LLM Provider; leer → globaler ALIMA-Default (agentic) |
| `llm_model` | TEXT | "" | Crawl-LLM Modell; leer → zum Provider passender Default |
| `max_depth` | INT | 2 | Crawl-BFS-Tiefe (0 = nur Startseite) |
| `max_pages` | INT | 50 | Max. Seiten pro Crawl |
| `include_re` / `exclude_re` | TEXT | "" | Regex-Filter für Kind-URLs (leer = alle / keiner) |
| `min_chars` | INT | 50 | Seiten mit weniger Text werden nicht indiziert |
| `max_keywords` | INT | 15 | Max. LLM-Keywords pro Seite |

## Indizieren

**GUI**: webindex-Instanz im Plugins-Tab (Kategorie 🔖 Lookups) auswählen →
Basis-URL + ggf. Crawl-LLM setzen → **„Seite indizieren …"**-Button. Läuft im
Hintergrund (Fortschritts-Log + Abbrechen); Crawl-Parameter kommen aus den
Instanz-Feldern.

**CLI**: `alima webindex crawl --base-url <url> [--instance ID] [--max-depth N]
[--max-pages N] [--include RE --exclude RE] [--provider P --model M] [--dry-run]`.
Flags überschreiben die Instanz-Settings; ohne `--provider`/`--model` greift der
globale Default, ohne jeden Provider nur Meta-/Überschriften-Keywords.

Keywords: Meta-Tags + Überschriften (deterministisch, weight 1.0) + LLM-Ergänzung
(weight 1.5) via `workflows/webindex_keywords.yaml` (der editierbare Standprompt;
prompts.json ist veraltet).

## Grenzen (bewusst)

- **Keyword-Retrieval, keine Embeddings**: Treffer hängen am exakten
  Keyword-Match (normalisiert). Multi-Word-Heading-Keywords werden über die
  Phrasen-Token-Zugabe getroffen, aber semantische Nähe bleibt unberücksichtigt.
- **Voll-Re-Crawl**: jeder `crawl`-Aufruf indexiert neu (idempotent via
  `upsert_page` + `set_page_keywords`); kein inkrementelles Diff.
- **Operator-getriggert**: kein Cron/Scheduling; Crawlen ist ein manueller Akt.
- **Live-Fetch**: nur innerhalb der `base_url`-Familie empfohlen (SSRF-Guard via
  `fetch_guarded` + `SystemConfig.url_fetch_allowlist`).

## Beim Forken als externes Code-Plugin

`id` in `plugin.toml` **und** das `id`-Klassenattribut in `provider.py` müssen
übereinstimmen und eindeutig sein. Siehe [`docs/plugin_authoring.md`](../../../../docs/plugin_authoring.md).