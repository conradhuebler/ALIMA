# Refactoring-Plan: Vereinheitlichung der GUI-Tabs mit dem PipelineManager

**Datum:** 22. September 2025

**Autor:** Gemini

## 1. Zielsetzung

Dieses Dokument beschreibt den Plan zur Refaktorierung der bestehenden, einzelnen GUI-Tabs (`AbstractTab`, `SearchTab`, etc.). Das Ziel ist es, ihre veraltete, eigenständige Logik zur Analyse und Suche zu entfernen und sie stattdessen an den neuen, zentralen `PipelineManager` anzubinden.

Die Tabs sollen dabei nicht entfernt, sondern als mächtige **"Experten-Werkzeuge"** oder **"Spielwiesen"** erhalten bleiben. Sie ermöglichen es dem Benutzer, gezielt einzelne Schritte der Pipeline mit benutzerdefinierten Parametern auszuführen, zu testen und zu wiederholen.

## 2. Architektonisches Kernprinzip

1.  **Single Source of Truth:** Jede Form von LLM-Analyse oder GND-Suche wird **ausschließlich** über eine Instanz des `PipelineManager` ausgeführt. Direkte Aufrufe an den `AlimaManager` oder den `SearchCLI` aus den UI-Tabs werden eliminiert.
2.  **Klare Rollenverteilung:**
    *   **UI-Tabs:** Sind nur für die *Konfiguration* und *Darstellung* zuständig. Sie sammeln die vom Benutzer gewünschten Parameter (Provider, Modell, Prompt, Suchbegriffe etc.) und zeigen die Ergebnisse an.
    *   **`PipelineManager`:** Ist allein für die *Ausführung, Orchestrierung und Zustandsverwaltung* der Analyse-Schritte verantwortlich.
3.  **Keine redundante Logik:** Eigene Worker-Threads und komplexe Signal-Verkettungen zwischen den Tabs zur Datenweitergabe werden durch den `PipelineManager` und sein Zustands-Objekt (`KeywordAnalysisState`) ersetzt.

---

## 3. Phase 1: Refactoring des `AbstractTab`

Der `AbstractTab` dient als primäres Beispiel für das Refactoring-Muster.

**Zieldatei:** `src/ui/abstract_tab.py`

### 3.1. Zu entfernender Code

*   **`AnalysisWorker(QThread)` Klasse:** Diese Klasse wird vollständig entfernt. Ihre Funktionalität wird durch einen neuen, generischen Worker (siehe Phase 3) oder eine Ad-hoc-Implementierung ersetzt, die den `PipelineManager` nutzt.

### 3.2. Anpassung der `start_analysis` Methode

Dies ist die zentrale Änderung. Die Methode wird komplett neu implementiert.

**Alter Ablauf:**
1.  Parameter aus der UI sammeln.
2.  Einen `AnalysisWorker` instanziieren.
3.  Den Worker starten, der direkt `alima_manager.analyze_abstract()` aufruft.

**Neuer Ablauf:**
1.  **Parameter aus UI sammeln:** Dieser Teil bleibt konzeptionell gleich. Alle Einstellungen (Provider, Modell, Temperatur, Prompt-Texte, Chunking-Optionen etc.) werden aus den UI-Elementen ausgelesen.

2.  **Ad-hoc `PipelineConfig` erstellen:** Eine temporäre Pipeline-Konfiguration wird zur Laufzeit erstellt, die nur für diesen einen manuellen Lauf gilt.

    ```python
    # In start_analysis()
    from src.core.pipeline_manager import PipelineConfig
    from src.utils.unified_provider_config import PipelineStepConfig, PipelineMode

    # 1. Eine leere PipelineConfig erstellen
    adhoc_config = PipelineConfig()
    adhoc_config.auto_advance = False # Wichtig: Wir wollen nur einen Schritt ausführen

    # 2. Eine Step-Konfiguration für den gewählten Task erstellen
    # Annahme: self.task enthält den Task-Namen, z.B. "initialisation"
    step_config = PipelineStepConfig(
        step_id=self.task,
        mode=PipelineMode.EXPERT, # Manueller Lauf ist immer "Expert"
        provider=self.provider_combo.currentText(),
        model=self.model_combo.currentText(),
        task=self.task,
        temperature=self.temp_spinbox.value(),
        top_p=self.p_value_spinbox.value(),
        # ... weitere Parameter wie custom prompts ...
    )
    
    # 3. Die Ad-hoc-Konfiguration im PipelineConfig-Objekt speichern
    adhoc_config.step_configs_v2[self.task] = step_config
    ```

3.  **Pipeline in einem Worker ausführen:** Um die UI nicht zu blockieren, wird die Ausführung weiterhin in einem `QThread` stattfinden. Wir verwenden dafür den bereits in `pipeline_tab.py` existierenden `PipelineWorker`.

    ```python
    # In start_analysis()
    from .pipeline_tab import PipelineWorker # Ggf. an einen zentraleren Ort verschieben

    # Alten Worker-Code entfernen
    # self.analysis_worker = AnalysisWorker(...)

    # Neuen, PipelineManager-basierten Worker verwenden
    # Wichtig: Wir brauchen eine PipelineManager-Instanz
    # Diese sollte idealerweise im __init__ des Tabs erstellt werden.
    
    self.pipeline_worker = PipelineWorker(
        pipeline_manager=self.pipeline_manager, # Annahme: self.pipeline_manager existiert
        input_text=abstract_text,
        input_type="text"
    )
    
    # WICHTIG: Dem Worker die Ad-hoc-Konfiguration übergeben
    # Dies erfordert eine kleine Anpassung am PipelineWorker oder wir setzen die Config direkt
    self.pipeline_worker.pipeline_manager.set_config(adhoc_config)

    # Callbacks verbinden, um das Ergebnis zu erhalten
    self.pipeline_worker.step_completed.connect(self.on_analysis_completed)
    self.pipeline_worker.step_error.connect(self.on_analysis_error)
    self.pipeline_worker.stream_token.connect(self._update_results_text)

    self.pipeline_worker.start()
    ```

### 3.3. Anpassung der Callback-Methoden

*   **`on_analysis_completed(self, step: PipelineStep)`:** Diese Methode empfängt nun ein `PipelineStep`-Objekt. Der anzuzeigende Text muss aus `step.output_data` extrahiert werden.
*   **`on_analysis_error(self, step: PipelineStep, error_message: str)`:** Empfängt ebenfalls das `step`-Objekt und die Fehlermeldung.

---

## 4. Phase 2: Refactoring des `SearchTab`

Die Änderungen hier sind analog zu Phase 1.

**Zieldatei:** `src/ui/find_keywords.py`

1.  **Logik entfernen:** Die Methode `perform_search` wird in ihrer jetzigen Form entfernt. Der direkte Aufruf von `SearchCLI` entfällt.
2.  **UI reduzieren (Optional):** Die Checkboxen für Suggester könnten entfernt und stattdessen die Konfiguration aus dem `PipelineConfigDialog` genutzt werden. Alternativ bleiben sie als "Spielwiese" erhalten.
3.  **"Suche starten"-Button neu verkabeln:**
    a.  Liest Suchbegriffe und Suggester-Auswahl aus der UI.
    b.  Erstellt eine Ad-hoc-`PipelineConfig` nur für den `search`-Schritt.
    c.  Nutzt den `PipelineWorker` und den `PipelineManager`, um *nur* den `search`-Schritt auszuführen.
    d.  Das Ergebnis (ein Dictionary) wird an die bestehende Methode `process_results` übergeben, die für die Anzeige in der Tabelle zuständig ist.

---

## 5. Phase 3: Aufräumen und Generalisieren

*   **`PipelineWorker` zentralisieren:** Der `PipelineWorker` aus `pipeline_tab.py` sollte in eine zentralere Datei verschoben werden (z.B. `src/ui/workers.py`), damit er von allen Tabs ohne zyklische Importe genutzt werden kann.
*   **`PipelineManager` erweitern:** Es könnte sinnvoll sein, dem `PipelineManager` eine neue Methode `execute_single_step(self, step_id: str, config: PipelineConfig)` hinzuzufügen, um die Ad-hoc-Ausführung einzelner Schritte noch sauberer zu kapseln.

## 6. Zusammenfassung der Vorteile

*   **Einheitliche Code-Basis:** Alle Analyse- und Suchvorgänge nutzen dieselbe Engine.
*   **Wartbarkeit:** Fehlerbehebungen und Erweiterungen am `PipelineManager` wirken sich sofort auf alle Teile der Anwendung (GUI-Pipeline, GUI-Tabs, CLI) aus.
*   **Stabilität:** Inkonsistentes Verhalten zwischen den Tabs und der Haupt-Pipeline wird eliminiert.
*   **Funktionserhalt:** Die wertvolle "Spielwiesen"-Funktionalität der einzelnen Tabs bleibt für Power-User und Entwickler erhalten und wird technisch auf eine solide Basis gestellt.
