# Institutional Bundles — ALIMA für Einrichtungen ausrollen

Ein **Bundle** bündelt die ALIMA-Einstellung einer Einrichtung (Plugins/Endpunkte +
ein beratendes Config-Profil), damit sie mit einem Befehl auf viele Arbeitsplätze
kommt. Modul: [`src/utils/bundle.py`](../src/utils/bundle.py); CLI:
[`src/cli/commands/bundle_cmd.py`](../src/cli/commands/bundle_cmd.py); Beispiel:
[`examples/bundles/demo_institution/`](../examples/bundles/demo_institution/).

## Format

```
<bundle_id>/
  bundle.toml     # [bundle] id/version/label/institution/alima_min_version
                  # [secrets] required = [{plugin=…, key=…, hint=…}]   (Deklaration!)
  plugins/        # Tier-1 declarative (Endpunkte) und/oder Tier-2 code Plugin-Dirs
  profile.json    # beratendes Overlay — nur Whitelist-Keys
  approvals.json  # {plugin_id: sha256}, von `bundle build` erzeugt
```

## CLI

```bash
# Admin: aus der eigenen laufenden Einstellung ein Bundle erzeugen (Secrets raus!)
alima bundle export ub-freiberg -o ub-freiberg-2026.1.zip --label "UB Freiberg" --institution "TU BAF"
alima bundle export mine -o mine.zip --plugin catalog --plugin mycode   # nur diese Instanzen

# Admin: ein handgeschriebenes Bundle bauen (approvals.json + optional Zip)
alima bundle build examples/bundles/demo_institution -o demo-2026.1.zip

# Arbeitsplatz
alima bundle install demo-2026.1.zip      # dir oder .zip
alima bundle list
alima bundle remove demo_institution
```

`bundle` ist setup-exempt (läuft ohne First-Run-Wizard).

## Export — laufende Einstellung als Bundle erfassen

`alima bundle export` (bzw. GUI-Knopf) macht aus der aktuellen Maschine ein
verteilbares Bundle:

- Jede **aktivierte** Suchquellen-Instanz wird zu (a) einer Kopie ihres
  installierten Plugin-Dirs *oder* (b) einem synthetisierten **declarative**
  `plugin.toml` (`kind` = Provider-Typ). Synthetisierte Instanzen bekommen eine
  eindeutige id `<bundle_id>_<instanz>`, damit sie auf dem Ziel nicht mit der
  dortigen Built-in-Instanz kollidieren (sonst gingen die Settings beim Dedup
  verloren). Built-in-Instanzen **ohne** eigene Settings (lobid/swb) werden
  übersprungen.
- **Secrets werden entfernt** (SECRET-`config_fields`) und stattdessen unter
  `[secrets]` deklariert — Tokens/Keys verlassen die Maschine nie.
- Das Profil erfasst deaktivierte Provider-Typen (aus dem Instanz-Status) + die
  vom Default abweichenden Whitelist-`system_config`-Keys.
- **Auswahl:** `--plugin <instanz-id>` (mehrfach) exportiert nur diese Instanzen
  statt aller aktivierten; im GUI-Export-Dialog per Checkliste.
- **Code-Plugins:** enthält der Export ein Code-Plugin (kopiertes Dir), setzt das
  Profil `system_config.enable_code_plugins=true`, damit es auf dem Ziel auch nach
  App-Neustart lädt (sonst würde es nur beim Install einmalig geladen).

Exportiert wird der **zuletzt gespeicherte** Config-Stand (im GUI vorher
speichern). `export` schreibt `approvals.json` mit, das Ergebnis ist sofort
installierbar.

## GUI

Der Plugin-Tab (`PluginSettingsTab`) hat unten eine Gruppe **📦 Bundles**:
installierte Bundles auflisten, **Installieren…** (Datei-Dialog → Bestätigung →
Report), **Entfernen**, **Exportieren…** (id/Label + Speicherort). Die Buttons
rufen dieselben Qt-freien `bundle.py`-Funktionen wie die CLI.

## Design-Entscheidungen (mit Operator festgelegt)

- **Nativer Installer** — `alima bundle …` treibt die Qt-freien Funktionen in
  `bundle.py`; eine GUI-Aktion kann dieselben Funktionen später wiederverwenden.
- **Beratend (advisory), nicht managed** — das Profil *setzt Startwerte*; Nutzer
  dürfen danach alles ändern. Keine gesperrten Keys. `remove` stellt genau die vom
  Bundle gesetzten Keys wieder her (im Ledger `installed_bundles` protokolliert),
  lässt sonstige Nutzeränderungen in Ruhe.
- **Per-User-Secrets** — ein Bundle trägt **nie** Tokens/Keys. Es *deklariert*
  benötigte Secrets; der Installer meldet sie. Werte kommen per GUI-Plugin-Tab oder
  `ALIMA_PLUGIN_<INSTANZ_ID>_<KEY>` (Runtime-Override, nie persistiert). Plugins mit
  gatendem Secret bleiben „unavailable", bis gefüllt — kein Bundle-Code nötig.

## profile.json — die Schutzgrenze

Overlay nur über eine **Whitelist**; alles andere lässt `install` fehlschlagen
(fail-closed). Das verhindert, dass ein Bundle die LLM-Zugangsdaten der Nutzer
überschreibt.

| erlaubt | Bedeutung |
| --- | --- |
| `search_provider_config.providers` | Provider-**Typ** an/aus. Wirkt über die zugehörigen **Instanzen** (autoritativ), da `providers` ein aus den Instanzen abgeleiteter Spiegel ist. |
| `system_config.*` (Teilmenge) | `url_fetch_allowlist`, `url_fetch_max_bytes`, `enable_code_plugins`, `enable_response_cache`, `enable_dk_splitting`, `dk_split_threshold` |

**Verboten** (Installer bricht ab): `unified_config` (LLM-Provider + API-Keys),
`database_config`, `plugins`, `approved_plugins` sowie jeder Key mit
`api_key`/`token`/`secret`/`password`.

## Vertrauen & Integrität

`install` gibt die Bundle-Plugins **headless** frei (Institutions-Vertrauen: der
Operator führt `install` auf ein vertrautes Bundle aus). Approval = SHA-256 über die
tatsächlich kopierten Dateien (`security.hash_dir`), verglichen mit `approvals.json`
des Admins — Abweichungen werden als Warnung gemeldet (Transport-/Manipulations-
Signal). Der AST-Scan läuft weiter; hohe Findings erscheinen im Install-Report.

## Grenzen (bewusst, MVP)

- Kein Lock/Precedence (advisory), kein Bundle-Signing, kein Auto-Update/Pull, keine
  GUI-Installer-Aktion, keine geteilten Secret-Env-Pipelines — bei Bedarf später.
- Ist `alima_min_version` gesetzt, warnt der Installer nur, wenn die laufende Version
  bestimmbar ist (derzeit keine Versionskonstante im Repo).
- Grüne Tests / ein sauberer Lauf zeigen die geprüften Pfade — nicht die Korrektheit
  jedes Providers über alle Eingaben.

Tests: [`tests/test_bundle.py`](../tests/test_bundle.py) (build/install/list/remove,
Whitelist-Rejection, Integritäts-Mismatch, Ledger-Serialisierung).
