# Refactoring-Plan und Analyse: ALIMA Code-Struktur

## 1. Architektonisches Ziel: Eine einheitliche und flexible UI-Architektur

Das übergeordnete Ziel ist die Vereinheitlichung der Anwendungslogik. Die aktuelle Architektur mit parallelen, veralteten UI-Logiken wird durch ein zentrales Service-Modell ersetzt.

**Neue Architektur-Prinzipien:**
1.  **Zentrale Services:** Die Logik für jeden Analyseschritt (z.B. Suche, Keyword-Extraktion) wird in wiederverwendbaren Klassen (`PipelineStepExecutor`, `SearchCLI`, etc.) in `src/utils/` gekapselt.
2.  **`PipelineTab` als "Mission Control":** Dient zur Ausführung des vollautomatischen, mehrstufigen Workflows. Er orchestriert die Aufrufe an die zentralen Services.
3.  **Spezialisierte "Hybrid-Tabs":** Die anderen UI-Tabs (`find_keywords`, `ubsearch`, etc.) werden so refaktorisiert, dass sie zwei Rollen erfüllen:
    *   **Viewer:** Sie können die detaillierten Ergebnisse eines Pipeline-Schritts zur Analyse und interaktiven Darstellung empfangen.
    *   **Manuelle Werkzeuge:** Sie behalten ihre UI-Steuerelemente (z.B. "Suche starten"), um gezielt **einzelne Analyseschritte** auszuführen. Dabei rufen sie dieselben zentralen Services auf, die auch die Pipeline verwendet.

---

## 2. Konkrete Refactoring-Aufgaben

### 2.1. UI-Tabs an die neue Architektur anbinden

*   **Ziel:** Die Backend-Logik der Tabs durch Aufrufe an die zentralen Services ersetzen und die duale Funktionalität (Viewer + Manuelles Werkzeug) implementieren.

*   **Refactoring-Plan:**
    *   [📝 **TODO**] **`find_keywords.py` (SearchTab) refaktorieren:**
        *   **IST:** Verwendet einen eigenen `SearchWorker`.
        *   **SOLL:** Der `SearchWorker` wird entfernt. Die manuelle Suche wird über die zentrale `SearchCLI`-Klasse abgewickelt. Der Tab erhält zusätzlich eine Methode `display_results(data)`, um Ergebnisse aus dem `PipelineTab` anzuzeigen.

    *   [📝 **TODO**] **`crossref_tab.py` refaktorieren:** 
        *   **IST:** Verwendet einen eigenen `CrossrefWorker`.
        *   **SOLL:** Der `CrossrefWorker` wird entfernt. Die manuelle DOI-Abfrage soll die zentrale Funktion `src.utils.doi_resolver.resolve_input_to_text` verwenden.

    *   [📝 **TODO**] **`abstract_tab.py` refaktorieren/ersetzen:** 
        *   **IST:** Implementiert einen kompletten, veralteten Analyse-Workflow.
        *   **SOLL:** Soll zu einem "Manuellen Analyse-Tab" umgebaut werden. Anstatt eines eigenen `AnalysisWorker` ruft er für eine Einzelanalyse gezielt Methoden des `PipelineStepExecutor` auf (z.B. `execute_final_keyword_analysis`).

    *   [📝 **TODO**] **`image_analysis_tab.py` refaktorieren:** 
        *   **IST:** Enthält die einzige funktionierende Logik zur Bild-zu-Text-Analyse.
        *   **SOLL:** Die Worker-Logik muss in den zentralen `TextExtractionWorker` (`unified_input_widget.py`) portiert werden. Der Tab ruft dann nur noch diesen zentralen Worker auf.

    *   [📝 **TODO**] **`ubsearch_tab.py` refaktorieren:** 
        *   **IST:** Implementiert eine fest verdrahtete Suche für einen spezifischen Katalog.
        *   **SOLL:** Der `UBSearchWorker` wird entfernt. Die manuelle Suche soll den konfigurierbaren `catalog_suggester` über die zentralen Services aufrufen.

### 2.2. Bereinigung von Altlasten

*   [✅ **Erledigt**] Veralteten `run`-Befehl aus `alima_cli.py` entfernt.
*   [✅ **Erledigt**] Experimentelle Skripte ins `laboratory`-Verzeichnis verschoben.
*   [✅ **Erledigt**] Veraltete Test-Dateien und `alima_cli_old.py` gelöscht.
*   [✅ **Erledigt**] Veraltete `CacheManager`-Klassen gelöscht und Referenzen korrigiert.
*   [✅ **Erledigt**] Veralteten `swbfetcher.py` als Duplikat von `swb_suggester.py` gelöscht.

### 2.3. Zukünftige Erweiterungen

*   **CLI-Pipeline für Einzelschritte erweitern:** Den `pipeline`-Befehl so erweitern, dass er die Ausführung von spezifischen Schritten erlaubt.