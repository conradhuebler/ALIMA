# Implementierungsplan: Generalisierte Task-Präferenzen

Dieses Dokument beschreibt den Plan zur Implementierung einer generalisierten Konfiguration für Task-spezifische Modell-Präferenzen. Der Fokus der Umsetzung liegt zunächst auf der Bilderkennung (`image_text_extraction`), die Architektur ist jedoch für alle Tasks ausgelegt.

## Architektonische Ziele

1.  **Entkopplung**: Die Konfiguration, *welches Modell für welchen Task* verwendet wird, wird komplett von der `prompts.json` (die das *Was* und den Prompt-Text definiert) getrennt.
2.  **Zentralisierung**: Alle Modell-Präferenzen werden zentral in der Hauptkonfiguration der Anwendung verwaltet und sind über eine UI zugänglich.
3.  **Generalisierung**: Die Struktur wird so aufgebaut, dass sie für jeden denkbaren Task (z.B. `initialisation`, `keywords`) erweitert werden kann.
4.  **Pipeline-Kompatibilität**: Die bestehende Pipeline (5 Schritte: Input → Initialisation → Search → Keywords → Classification) bleibt vollständig funktionsfähig und unverändert.
5.  **Chunked-Task-Hierarchie**: Tasks können Chunked-Subtasks haben (z.B. `keywords` → `keywords_chunked`), die spezielle Model-Präferenzen für große Texte nutzen.
6.  **Minimalinvasivität**: Die Umsetzung erfolgt zunächst nur für den Vision-Task, um bestehende, funktionierende Teile der Pipeline nicht unnötig zu refaktorisieren.

---

## Finaler Umsetzungsplan

### Schritt 1: Datenmodell anpassen

*   **Was?**
    Die bestehende `TaskPreference`-Klasse wird so erweitert, dass sie eine priorisierte Liste von spezifischen Provider/Modell-Paaren speichern kann, anstatt nur einer Liste von Provider-Namen.

*   **Wie?**
    Das Feld `preferred_providers: List[str]` wird durch `model_priority: List[Dict[str, str]]` ersetzt.

    **Vorher:**
    ```python
    @dataclass
    class TaskPreference:
        task_type: TaskType
        preferred_providers: List[str] = field(default_factory=list)
        # ...
    ```

    **Nachher:**
    ```python
    @dataclass
    class TaskPreference:
        task_type: TaskType
        model_priority: List[Dict[str, str]] = field(default_factory=list) # z.B. [{"provider_name": "p1", "model_name": "m1"}, ...]
        # NEU: Chunked-Subtask Support für große Texte
        chunked_model_priority: Optional[List[Dict[str, str]]] = None
        performance_preference: str = "balanced"
        allow_fallback: bool = True
    ```
    
    **Task-Hierarchie Beispiel:**
    ```python
    "keywords": TaskPreference(
        task_type=TaskType.TEXT_ANALYSIS,
        model_priority=[{"provider_name": "ollama", "model_name": "cogito:14b"}],
        chunked_model_priority=[{"provider_name": "ollama", "model_name": "cogito:32b"}]  # Größeres Modell für Chunking
    ),
    "image_text_extraction": TaskPreference(
        task_type=TaskType.VISION,
        model_priority=[{"provider_name": "gemini", "model_name": "gemini-2.0-flash"}]
        # Vision-Tasks haben normalerweise kein Chunking
    )
    ```
    
    Bestehende Methoden wie `_setup_default_task_preferences` müssen entsprechend angepasst werden.

*   **Wo?**
    `src/utils/unified_provider_config.py`

### Schritt 2: UI zur Verwaltung der Task-Präferenzen erstellen

*   **Was?**
    Eine neue UI im zentralen Einstellungsdialog, um die Modell-Prioritäten für jeden Task zu verwalten.

*   **Wie?**
    1.  Ein neuer Tab "Task-Präferenzen" wird im `ComprehensiveSettingsDialog` hinzugefügt.
    2.  **Pipeline-Sektion**: Zeigt Pipeline-Tasks (`initialisation`, `keywords`, `classification`) mit spezieller Kennzeichnung.
    3.  **Vision-Sektion**: Zeigt Vision-Tasks (`image_text_extraction`) getrennt an.
    4.  **Chunked-UI**: Checkbox/Toggle für "Spezielle Modelle für große Texte (Chunked)" pro anwendbarem Task.
    5.  Wählt der Benutzer einen Task aus, erscheint auf der rechten Seite eine umsortierbare Liste (`QListWidget` mit Drag & Drop). Diese Liste zeigt die `model_priority` für den ausgewählten Task an (z.B. "1. openai: gpt-4o", "2. ollama: llava").
    6.  Buttons für "Hinzufügen", "Entfernen", "Hoch" und "Runter" ermöglichen die Verwaltung dieser Liste.
    7.  Der "Hinzufügen"-Dialog lässt den Benutzer einen der zentral konfigurierten Provider und anschließend eines von dessen Modellen auswählen.
    8.  **Chunked-Modelle**: Separate Liste für `chunked_model_priority` wenn Chunked-Option aktiviert ist.

*   **Wo?**
    `src/ui/comprehensive_settings_dialog.py`

### Schritt 3: Generalisierte Ausführungslogik im Backend

*   **Was?**
    Eine zentrale Methode im Backend, die einen beliebigen Task mit der neu konfigurierten Modell-Priorität und Fallback-Logik ausführen kann.

*   **Wie?**
    1.  Eine neue, zentrale Methode `alima_manager.execute_task(task_name: str, context: dict)` wird erstellt.
    2.  Diese Methode liest die `task_preferences` aus der `UnifiedProviderConfig`.
    3.  **Chunked-Task-Logik**: Wenn `task_name` endet mit `_chunked`, wird `chunked_model_priority` des Basis-Tasks verwendet (Fallback zu Standard `model_priority`).
    4.  Sie holt das Prompt-Template für den Task vom `PromptService`.
    5.  Sie iteriert durch die `model_priority`-Liste und ruft für jeden Eintrag den `LlmService` auf, bis ein Aufruf erfolgreich ist.
    6.  Bei einem Fehler wird das nächste Modell in der Liste versucht. Schlagen alle fehl, wird ein Fehler zurückgegeben.
    7.  **Pipeline-Kompatibilität**: Bestehende Pipeline-Methoden (`keywords`, `initialisation`) können weiterhin direkte Methoden verwenden.

*   **Wo?**
    `src/core/alima_manager.py`

### Schritt 4: Anbindung des Vision-Tasks an das Input-Widget

*   **Was?**
    Der `image_text_extraction`-Task wird an das `UnifiedInputWidget` der Pipeline angebunden. Bestehende Pipeline-Aufrufe bleiben unberührt.

*   **Wie?**
    1.  Die Logik im `UnifiedInputWidget` (in `_extract_image_with_llm` Methode) wird erweitert.
    2.  Wenn eine Bild-Datei als Input erkannt wird, ruft das Widget die neue, generalisierte Methode auf: `alima_manager.execute_task(task_name='image_text_extraction', context={'image_data': ...})`.
    3.  Der von der Methode zurückgegebene Text wird in das Haupteingabefeld der Pipeline eingefügt.
    4.  **Pipeline-Garantie**: Die restliche Pipeline wird wie gewohnt mit dem extrahierten Text als Input fortgesetzt - alle bestehenden Pipeline-Methoden bleiben funktionsfähig.
    5.  **Fallback-Sicherheit**: Bei Fehlern in der neuen Task-Ausführung wird auf die bestehende Implementierung zurückgefallen.

*   **Wo?**
    `src/ui/unified_input_widget.py`

---

## Analyse zusätzlicher UI-Aspekte

### Obsolete UI-Elemente

Durch die Einführung des neuen, generalisierten "Task-Präferenzen"-Tabs wird die ältere, hartcodierte Logik zur Steuerung von Providern überflüssig. Konkret werden folgende UI-Elemente aus der (derzeit ungenutzten) Methode `_create_provider_preferences_tab` obsolet:

*   **`vision_provider_combo`**: Ersetzt durch die allgemeine Konfiguration für den `vision`-Task.
*   **`text_provider_combo`**: Ersetzt durch die allgemeine Konfiguration für Text-Tasks.
*   **`classification_provider_combo`**: Ersetzt durch die allgemeine Konfiguration für Klassifikations-Tasks.
*   **`preferred_provider_combo`** und **`priority_list`**: Die globale Provider-Priorisierung wird durch die spezifischere und mächtigere Pro-Task-Priorisierung abgelöst.

### Potenziale zur UI-Verbesserung

Die Analyse des `ComprehensiveSettingsDialog` zeigt, dass die UI bereits an vielen Stellen korrekte Steuerelemente wie `QComboBox` verwendet. Ein "sinnloses" UI-Element im Sinne einer freien Texteingabe, wo eine Auswahl hingehört, wurde nicht gefunden.

Das größte Verbesserungspotenzial liegt in der **Struktur und Zentralisierung**, was der hier beschriebene Plan direkt adressiert: Die verstreute und teilweise hartcodierte Konfiguration von Provider-Präferenzen wird in dem neuen "Task-Präferenzen"-Tab an einem einzigen, logischen Ort zusammengefasst und für alle denkbaren Tasks vereinheitlicht.