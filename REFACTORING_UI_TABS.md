# Refactoring-Plan: Vereinheitlichung der GUI-Tabs mit dem PipelineManager

**Datum:** 22. September 2025

**Autor:** Gemini

**Status:** Phase 1 & 2 initial umgesetzt, Feinschliff und Cleanup ausstehend.

## 1. Zielsetzung

Dieses Dokument beschreibt den Plan zur Refaktorierung der bestehenden, einzelnen GUI-Tabs (`AbstractTab`, `SearchTab`, etc.). Das Ziel ist es, ihre veraltete, eigenständige Logik zur Analyse und Suche zu entfernen und sie stattdessen an den neuen, zentralen `PipelineManager` anzubinden.

Die Tabs sollen dabei nicht entfernt, sondern als mächtige **"Experten-Werkzeuge"** oder **"Spielwiesen"** erhalten bleiben. Sie ermöglichen es dem Benutzer, gezielt einzelne Schritte der Pipeline mit benutzerdefinierten Parametern auszuführen, zu testen und zu wiederholen.

## 2. Architektonisches Kernprinzip

1.  **Single Source of Truth:** Jede Form von LLM-Analyse oder GND-Suche wird **ausschließlich** über eine Instanz des `PipelineManager` ausgeführt. Direkte Aufrufe an den `AlimaManager` oder den `SearchCLI` aus den UI-Tabs werden eliminiert.
2.  **Klare Rollenverteilung:**
    *   **UI-Tabs:** Sind nur für die *Konfiguration* und *Darstellung* zuständig. Sie sammeln die vom Benutzer gewünschten Parameter und zeigen die Ergebnisse an.
    *   **`PipelineManager`:** Ist allein für die *Ausführung, Orchestrierung und Zustandsverwaltung* der Analyse-Schritte verantwortlich.
3.  **Keine redundante Logik:** Eigene Worker-Threads und komplexe Signal-Verkettungen zwischen den Tabs zur Datenweitergabe werden durch den `PipelineManager` und sein Zustands-Objekt (`KeywordAnalysisState`) ersetzt.

---

## 3. Phase 1: Refactoring des `AbstractTab` (Abgeschlossen)

Der `AbstractTab` wurde erfolgreich umgestellt. Er nutzt nun den `PipelineManager` für die Ausführung von Einzel-Schritten.

---

## 4. Phase 2: Refactoring des `SearchTab` (Teilweise abgeschlossen)

Der `SearchTab` wurde initial umgestellt, um den `PipelineManager` zu nutzen. Folgender Feinschliff ist noch notwendig.

**Zieldatei:** `src/ui/find_keywords.py`

### 4.1. Nächste Schritte: Feinschliff und Cleanup

1.  **Alten Fallback-Pfad entfernen:**
    *   **Was:** In der `perform_search`-Methode existiert aktuell noch ein `else`-Block, der die alte Logik mit einem direkten `SearchCLI`-Aufruf als Fallback enthält.
    *   **Warum:** Dieser Code-Pfad untergräbt die Vereinheitlichung. Es darf nur noch einen einzigen Weg geben, eine Suche auszuführen: über den `PipelineManager`.
    *   **Aktion:** Der gesamte `else`-Block muss entfernt werden. Wenn der `PipelineManager` nicht verfügbar ist, sollte stattdessen eine Fehlermeldung angezeigt oder ein Fehler geloggt werden.

2.  **Nachverarbeitungs-Logik zentralisieren:**
    *   **Was:** Der `SearchTab` enthält die Methode `finalise_catalog_search`. Diese führt eine zweite, separate Suche durch, um Ergebnisse aus dem lokalen Katalog mit externen GND-IDs anzureichern. Diese Logik gehört nicht in die UI-Schicht.
    *   **Warum:** Diese Geschäftslogik muss zentralisiert werden, damit alle Pipeline-Ausführungen (GUI-Pipeline, CLI, manuelle Tab-Suche) davon profitieren und sich konsistent verhalten.
    *   **Aktion:** Die Logik aus `finalise_catalog_search` muss aus dem `SearchTab` entfernt und in den `PipelineStepExecutor` verschoben werden, spezifisch in die Methode `execute_gnd_search`. Diese Methode sollte dann die vollständigen und fertig validierten Ergebnisse zurückliefern.

---

## 5. Phase 3: Generalisierung (Optional, empfohlen)

*   **`PipelineWorker` zentralisieren:** Der `PipelineWorker` aus `pipeline_tab.py` sollte in eine zentralere Datei verschoben werden (z.B. `src/ui/workers.py`), damit er von allen Tabs ohne zyklische Importe genutzt werden kann.
*   **`PipelineManager` erweitern:** Es könnte sinnvoll sein, dem `PipelineManager` eine neue Methode `execute_single_step(self, step_id: str, config: PipelineConfig)` hinzuzufügen, um die Ad-hoc-Ausführung einzelner Schritte noch sauberer zu kapseln.

---

## 6. Phase 4: Finale Aufräumarbeiten (Code-Hygiene)

Nach Abschluss der Refactorings sind diverse Code-Teile in der gesamten Anwendung obsolet geworden. Diese sollten entfernt werden, um die Codebasis sauber und wartbar zu halten.

### 6.1. `src/ui/main_window.py`

*   **ZU LÖSCHEN: Überflüssige Signal-Slot-Verbindungen**
    *   **Grund:** Die Datenübergabe zwischen den Tabs (z.B. vom `AbstractTab` zum `SearchTab`) wird nun vom `PipelineManager` gesteuert. Die alten, direkten Verbindungen sind "Spaghetti-Code" und müssen entfernt werden.
    *   **Betroffener Code:** Nahezu alle `.connect()`-Aufrufe im `init_ui`-Teil, die die Tabs (`abstract_tab`, `search_tab`, `analyse_keywords`, `ub_search_tab`) miteinander verbinden. Beispiele:
        ```python
        # OBSOLET: Datenfluss wird jetzt vom PipelineManager gesteuert
        self.search_tab.keywords_found.connect(self.analyse_keywords.set_keywords)
        self.abstract_tab.abstract_changed.connect(self.analyse_keywords.set_abstract)
        self.analyse_keywords.final_list.connect(self.update_gnd_keywords)
        self.abstract_tab.final_list.connect(self.search_tab.update_search_field)
        # ... und viele weitere ...
        ```

### 6.2. `src/ui/find_keywords.py` (SearchTab)

*   **ZU LÖSCHEN:**
    *   **Methoden zur Prompt-Generierung:** `generate_ddc_prompt`, `generate_gnd_prompt`, `generate_initial_prompt`. Die Erstellung von Prompts und die Filterung von Ergebnissen ist Aufgabe des `keywords`-Schritts der Pipeline, nicht des `SearchTab`.
    *   **Zugehörige UI-Elemente:** Die Buttons und Checkboxen für die DDC- und GND-Filterung.
    *   **Methode `determine_relation`:** Diese Helfermethode kann entweder gelöscht oder in ein allgemeines Utility-Modul verschoben werden, falls sie an anderer Stelle noch benötigt wird.

### 6.3. `src/core/pipeline_manager.py`

*   **ZU LÖSCHEN:**
    *   **Methode `_get_smart_mode_provider_model`:** Diese Methode ist nach der Einführung von `_resolve_smart_mode_for_step` vollständig obsolet und kann entfernt werden.

### 6.4. `src/utils/pipeline_utils.py`

*   **ZU LÖSCHEN:**
    *   **Funktion `execute_complete_pipeline`:** Diese Funktion ist das Kernstück der veralteten CLI-Implementierung. Nach dem CLI-Refactoring wird sie nicht mehr verwendet und muss entfernt werden, um die Architektur endgültig zu vereinheitlichen.

## 7. Zusammenfassung der Vorteile

*   **Einheitliche Code-Basis:** Alle Analyse- und Suchvorgänge nutzen dieselbe Engine.
*   **Wartbarkeit:** Fehlerbehebungen und Erweiterungen am `PipelineManager` wirken sich sofort auf alle Teile der Anwendung (GUI-Pipeline, GUI-Tabs, CLI) aus.
*   **Stabilität:** Inkonsistentes Verhalten zwischen den Tabs und der Haupt-Pipeline wird eliminiert.
*   **Funktionserhalt:** Die wertvolle "Spielwiesen"-Funktionalität der einzelnen Tabs bleibt für Power-User und Entwickler erhalten und wird technisch auf eine solide Basis gestellt.
