# Plugin: catalog — Lokaler Bibliothekskatalog (Libero/SOAP)

Self-contained Suchprovider-Plugin; Blaupause für **Token-authentifizierte,
Multi-Capability-Plugins** (SOAP/HTTP-Katalog mit drei Gesichtern). Struktur,
Modi und Konventionen wie in [`../lobid/README.md`](../lobid/README.md).

## Was es tut

| | |
|---|---|
| Capabilities | `gnd_keywords`, `title_records`, `classification` (DK) |
| Eingabe | Suchbegriffe / Titelanfragen (je nach Capability) |
| Ausgabe | GND-Schlagworte, Titel-Records (rsn, web_url, …) oder DK-Notationen |
| MCP-Tools | `search_catalog`, `search_catalog_titles` |
| Konfiguration | `token` (SECRET, gated), SOAP-/OPAC-URLs, `dk_enabled` (DK-Quelle), `timeout_s`, `cache_only` |

## Spezifisch für diese Blaupause

- **Mehrere Capabilities**: `search()` verzweigt auf die angefragte Capability
  (`_require` + pro-Capability-Zweig in `provider.py`) — Muster für Plugins,
  die mehr als GND-Keywords liefern.
- **Secret-Konfig**: `token` ist `ConfigField(kind=SECRET, gates_availability=True)`
  → ohne Token ist die Instanz inaktiv; GUI maskiert; Laufzeit-Override via
  `ALIMA_PLUGIN_CATALOG_TOKEN`.
- **Shared Transport-Client**: `suggester.py` (ehem. BiblioSuggester) nutzt den
  geteilten `src/utils/clients/biblio_client.py` (SOAP + Web-Fallback), der auch
  von der DK-Suche der Pipeline direkt konsumiert wird. Beim Kopieren als
  eigenes Plugin: entweder ALIMAs Client absolut importieren oder einen eigenen
  Client als weitere Datei ins Plugin-Verzeichnis legen.
- Beim Kopieren umbenennen (müssen übereinstimmen): `id` in `plugin.toml`
  **und** das `id`-Klassenattribut in `provider.py`.

## Schalter „Nur Cache (kein Netz)“ (`cache_only`)

Keine SOAP- oder OPAC-Anfrage. Schlagworte und Titel kommen aus
`search_response_cache`, DK/RVK der Pipeline aus `catalog_dk_cache`, ohne
Altersgrenze und ohne Rückschreiben; Begriffe ohne Eintrag bleiben ohne Treffer
und werden je Lauf in einer Zeile gezählt. Grenzen: Es gibt nur, was frühere
Läufe mit Netz abgelegt haben. Schlagwort-Rohantworten legt die klassische Suche
ab (Pipeline, Suchtab), nicht die Agent-Werkzeuge; Titel-Rohantworten entstehen
nur über das Werkzeug einer zweiten catalog-Instanz. Ein gespeicherter
Fehlschlag (`error`/`timeout`) zählt als fehlender Eintrag. `alima test-catalog`
baut seinen Client selbst und fragt weiter live ab.

## DK-Schalter (`dk_enabled`)

„Libero für DK-Suche nutzen“: an = Libero ist DK-Quelle. Der DK-Schritt fragt
alle Instanzen mit eingeschaltetem DK-Schalter in der Reihenfolge der
Plugin-Liste; ein Schlagwort, das eine Quelle nicht beantwortet (gesperrt, im
Zustand „Nur Cache“ ohne Eintrag, Einrichtung nicht gesetzt, Verbindungsfehler,
Extraktor nicht startbar), geht an die nächste. „Keine Treffer“ ist eine Antwort.
Ist der Schlüssel nicht gespeichert (Konfigurationen von vor dem Schalter, neue
Installationen), ist Libero DK-Quelle, solange keine andere Instanz den
DK-Schalter an hat, und nur die erste solche catalog-Instanz; das Formular zeigt
dann „aus“ und schreibt den Schlüssel erst, wenn er geändert wird. Ein
gespeicherter Wert gilt wörtlich: aus bei allen Quellen heißt „Keine DK-Quelle“.
Welche Quellen gelten, steht im Plugin-Tab über der Liste. Details:
[`docs/plugin_system.md`](../../../../../docs/plugin_system.md).
