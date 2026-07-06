# POC: ALIMA komplett auf eigenen Plugins

Beweis, dass das Plugin-System vollständig ist: **alle 6 Built-in-Suchprovider**
werden als kopierbare, operator-eigene Plugins nachgebaut, die Built-ins werden
abgeschaltet — und beide Frontends (klassisch + agentisch) laufen weiter, nur noch
über die eigenen Plugins.

## Ein-Datei-Helfer

`deploy_poc.py` macht drei Dinge und kann alles zurückrollen:

| Befehl | Wirkung |
| --- | --- |
| `python examples/plugins_poc/deploy_poc.py` | `poc_*`-Plugins nach `~/.config/alima/plugins/` erzeugen, headless freigeben, **Built-ins deaktivieren** |
| `python examples/plugins_poc/deploy_poc.py --revert` | Built-ins wieder aktivieren, `poc_*`-Instanzen/Freigaben/Verzeichnisse entfernen |
| `python examples/plugins_poc/deploy_poc.py --keep-builtins` | `poc_*` daneben installieren (Built-ins bleiben aktiv) |
| `python examples/plugins_poc/deploy_poc.py --generate-into DIR` | nur kopieren+umbenennen (keine Config-Änderung) |

Es wird **nur die `id`** umbenannt (`lobid` → `poc_lobid`, in `plugin.toml` *und*
Provider-Klasse). Tool-Namen (`search_lobid` …), `source_label` und Klassenname
bleiben — deshalb greifen klassische Pipeline, deterministische Agentik-Tools und
Raw-Cache-Provenienz unverändert.

## Warum das ohne großen Umbau geht

- **Agentisch/MCP** ist bereits plugin-nativ: Tools werden aus den *aktivierten
  Instanzen* generiert; ein deaktivierter Built-in verschwindet, das gleichnamige
  `poc_*`-Plugin übernimmt das Tool über den generischen Factory-Handler.
- **Klassisch** brauchte einen kleinen Fix: `execute_gnd_search`
  (`src/utils/pipeline_utils.py`) fällt bei leerer Schnittmenge auf die aktive
  Provider-Menge zurück, statt leer zu suchen.

## Operator-Verifikation (GUI-Sign-off)

1. `python examples/plugins_poc/deploy_poc.py`
2. GUI → **Plugins-Tab**: 6 `poc_*` als „loaded/approved", die 6 Built-ins auf
   *deaktiviert*.
3. Eine **klassische** Pipeline und einen **agentischen** Lauf starten — die GND-
   Schlagworte kommen aus den `poc_*`-Quellen (Netz-Provider brauchen erreichbare
   Endpunkte; `poc_gnd_local` beweist den Offline-Pfad ohne Netz).
4. Zurückrollen: `python examples/plugins_poc/deploy_poc.py --revert`

## Automatischer Nachweis

`tests/test_all_external_plugins_poc.py` erzeugt alle 6 `poc_*`, lädt sie über den
echten Code-Plugin-Pfad, deaktiviert die Built-ins und prüft: alle laden
(inkl. catalog/finc/sru), die kanonischen Tools sind an `poc_*` gebunden, der
klassische Fallback routet auf `poc_*`, und `poc_gnd_local` liefert offline.

## Ehrliche Grenzen

Siehe [`docs/plugin_authoring.md` §10](../../docs/plugin_authoring.md): Built-in-
*Klassen* bleiben registriert (nur Instanzen/Tools werden abgeschaltet); die
WP2-Raw-Cache-Aggregation und der separate GUI-Tab „Find Keywords" verdrahten
`lobid`/`swb` weiterhin per Name.
