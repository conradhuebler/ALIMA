# Demo-Bundle — ALIMA für Institutionen ausrollen

Eine **Bundle** bündelt Plugins (Endpunkte) + ein beratendes Config-Profil, damit
eine Einrichtung ihre ALIMA-Einstellung mit einem Befehl auf viele Arbeitsplätze
bringt. Dieses Beispiel zeigt die Struktur.

```
demo_institution/
  bundle.toml     # id/version + benötigte Per-User-Secrets (Deklaration, kein Wert!)
  plugins/        # Tier-1-Endpunkte (hier: Katalog + SRU) — kein Code, kein Token
  profile.json    # Overlay: nur Whitelist-Keys (Provider an/aus, System-Toggles)
  approvals.json  # von `alima bundle build` erzeugt (SHA-256 je Plugin)
```

## Admin: bauen

```bash
alima bundle build examples/bundles/demo_institution -o demo_institution-2026.1.zip
```
Schreibt `approvals.json` (Hash je Plugin) und packt das Zip.

## Arbeitsplatz: installieren

```bash
alima bundle install demo_institution-2026.1.zip
alima bundle list
alima bundle remove demo_institution      # sauber zurückrollen
```

`install` kopiert die Plugins, gibt sie headless frei (Institutions-Vertrauen),
legt das Profil **beratend** darüber (Nutzer kann danach alles ändern) und meldet
die **Per-User-Secrets**, die noch fehlen — z. B. den persönlichen Katalog-Token
via GUI-Plugin-Tab oder `ALIMA_PLUGIN_UB_CATALOG_TOKEN`.

## Grenzen (bewusst)

- **Advisory**, nicht managed: keine gesperrten Keys. `remove` stellt die vom
  Bundle gesetzten Keys wieder her, lässt eigene Änderungen sonst in Ruhe.
- **Keine Secrets im Bundle.** `profile.json` darf `unified_config` (LLM-Keys),
  `database_config` und alles Secret-förmige nicht enthalten — der Installer
  bricht sonst ab. Das schützt die eigenen Zugangsdaten der Nutzer.
- Endpunkte hier sind Platzhalter (`*.example`) — echte URLs eintragen.

Vollständige Doku: [`docs/institutional_bundles.md`](../../../docs/institutional_bundles.md).
