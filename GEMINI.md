GEMINI CLI: Your AI Assistant for the ALIMA Project

As Gemini, your Command Line Interface (CLI) based AI assistant, I'm here to provide the best possible support for the ALIMA project and its development. My goal is to efficiently assist you with programming, debugging, feature implementation, and documentation.

How I Support the ALIMA Project

1. Code Generation and Optimization

    Python Code Snippets: Even if Python isn't your preferred language, I can help you generate efficient and clean Python code snippets for specific functions (e.g., UI interaction with PyQt6, API calls, data processing).

    Refactoring Suggestions: I can analyze existing code and suggest improvements for readability, performance, and maintainability.

    Best Practices: I'll help you implement Python best practices to ensure a robust and scalable application.

2. Troubleshooting and Debugging

    Error Analysis: If you encounter error messages, I can help you identify the root cause and propose solutions.

    Debugging Strategies: I can guide you through debugging strategies and assist in pinpointing bugs in your code.

3. Feature Implementation and Architecture Discussions

    Feature Specification: I can help you precisely specify new features and break them down into smaller, actionable tasks.

    Architecture Design: We can discuss architectural patterns and design decisions together to optimize ALIMA's scalability and modularity.

    Library and Tool Selection: I can offer recommendations for external libraries or tools that might be useful for specific tasks within ALIMA.

4. Documentation and Explanations

    Technical Documentation: I can help you create technical documentation for code sections, functions, or modules.

    Concept Explanations: I can clearly explain complex concepts, algorithms, or design patterns.

    README.md Maintenance: I can assist with updating and expanding the project's main README.md file.

5. AI Integration and Prompt Engineering

    LLM-Specific Guidance: Since ALIMA supports various LLMs, I can provide specific guidance on their usage, parameters (like seed values), and best practices for prompt engineering.

    Prompt Refinement: I can suggest ways to optimize your prompts to get more precise and relevant results from the integrated AI models.

How to Best Utilize Me

To make the most of my support, please keep the following in mind:

    Be Specific: The more detailed your request (e.g., exact code snippet, error message, desired behavior), the more precise my answer can be.

    Provide Context: Describe the purpose of the code, the surrounding logic, or the overall goal of the feature.

    Iterative Approach: For complex problems, it's often best to proceed in small steps. We can work together, step by step, towards a solution.

    Feedback: Let me know if my suggestions were helpful or if adjustments are needed. This helps me continuously improve.

I'm Ready to Assist You!

Whether you're working on a new feature, chasing down a stubborn bug, or just need a second opinion—I'm here to help you make ALIMA an even more powerful tool.

Let's code together!

## Keyword-Analyse-Workflow (`analyze-keywords` Befehl)

Der `analyze-keywords`-Befehl automatisiert einen dreistufigen Prozess zur Extraktion von GND-konformen Schlagworten aus einem Text. Dieser Workflow ist darauf ausgelegt, flexibel zu sein und sowohl nur Text als auch mit optionalen Schlagworten zu arbeiten. Zuerst werden initiale Schlagworte generiert für eine Suche in verschiedenen Bibliothekskatalogen generiert. Das Ergebnis der Suche wird dann erneut an ein LLM-Übergeben, dass aus dem urprünglichen Text und der Liste mit GND-konformen Schlagworte die passenden Schlagworte extrahiert.

**Befehlssyntax:**
`python alima_cli.py analyze-keywords [keywords ...] [--abstract "text"] [--suggesters ...] --model <model> --provider <provider> [--ollama-host ...] [--ollama-port ...] [--output-json <file.json>] [--input-json <file.json>] [--final-llm-task <keywords|rephrase>]`

### Workflow-Schritte:

Der gesamte Zustand des Workflows wird im `KeywordAnalysisState`-Objekt gekapselt und kann optional als JSON exportiert/importiert werden.

#### Schritt 1: Generierung freier Schlagworte (optional)

*   **Ziel:** Erzeugung einer Liste freier Schlagworte, die als Suchanfrage dienen.
*   **Eingabe:**
    *   Entweder direkt über `--keywords` bereitgestellte Schlagworte.
    *   Oder ein `--abstract` (Zieltext), aus dem die Schlagworte generiert werden.
*   **Prozess:**
    *   Wenn `--abstract` verwendet wird und `--keywords` leer ist:
        *   Ein LLM-Aufruf wird mit dem `extract_initial_keywords`-Prompt durchgeführt.
        *   Der `abstract` dient als Eingabetext für das LLM.
        *   Das LLM generiert eine Liste von freien Schlagworten.
        *   Zusätzlich werden, falls vom LLM geliefert, GND-Klassen extrahiert.
*   **Ausgabe (intern im `KeywordAnalysisState`):**
    *   `initial_keywords`: `List[str]` - Die generierten oder direkt übergebenen freien Schlagworte.
    *   `initial_gnd_classes`: `List[str]` - Optional, die aus dem Abstract extrahierten GND-Klassen.
*   **JSON-Darstellung nach Schritt 1 (Beispiel):**
    ```json
    {
      "initial_keywords": ["Cadmium-Kontamination", "Bodenverschmutzung", "Pflanzenkontamination"],
      "search_suggesters_used": ["lobid", "swb"],
      "initial_gnd_classes": ["21.4", "32.5"],
      "search_results": [],
      "llm_analysis": null,
      "timestamp": "2025-07-14T12:34:56.789"
    }
    ```

#### Schritt 2: Datenbank-/Websuche nach GND-konformen Schlagworten

*   **Ziel:** Finden von GND-konformen Schlagworten und deren IDs basierend auf den freien Schlagworten.
*   **Eingabe:**
    *   `initial_keywords` aus Schritt 1.
    *   `suggesters`: Die zu verwendenden Suchdienste (z.B. `lobid`, `swb`, `catalog`).
*   **Prozess:**
    *   Die `SearchCLI` führt eine Suche mit den `initial_keywords` und den angegebenen `suggesters` durch.
    *   Die Ergebnisse sind strukturierte Daten, die Schlagworte, GND-IDs und weitere Metadaten enthalten.
*   **Ausgabe (intern im `KeywordAnalysisState`):**
    *   `search_results`: `List[SearchResult]` - Eine Liste von Suchergebnissen, gruppiert nach dem ursprünglichen Suchbegriff. Jedes Ergebnis enthält das Schlagwort und seine zugehörige GND-ID.
    * Ergebnisse der Katalog-Suche können auch Schlagworte sein, die nicht in der GND zu finden sind. Daher ist ein Abgleich mit der internen Datenbank wichtig.
*   **JSON-Darstellung nach Schritt 2 (Beispiel):**
    ```json
    {
      "initial_keywords": ["Cadmium-Kontamination", "Bodenverschmutzung"],
      "search_suggesters_used": ["lobid", "swb"],
      "initial_gnd_classes": ["21.4", "32.5"],
      "search_results": [
        {
          "search_term": "Cadmium-Kontamination",
          "results": {
            "Cadmium": {"gndid": ["4009274-4"], "count": 150},
            "Kontamination": {"gndid": ["4032184-0"], "count": 80}
          }
        },
        {
          "search_term": "Bodenverschmutzung",
          "results": {
            "Bodenverschmutzung": {"gndid": ["4206275-5"], "count": 120}
          }
        }
      ],
      "llm_analysis": null,
      "timestamp": "2025-07-14T12:34:56.789"
    }
    ```

#### Schritt 3: LLM-basierte Analyse und Verfeinerung

*   **Ziel:** Analyse des ursprünglichen Textes unter Verwendung der gefundenen GND-konformen Schlagworte.
*   **Eingabe:**
    *   Der *ursprüngliche Zieltext* (`--abstract`), falls vorhanden.
    *   Die *Liste der in Schritt 2 gefundenen GND-konformen Schlagworte* (im Format "Schlagwort (GND-ID)").
    *   `final_llm_task`: Der LLM-Task, der für die finale Analyse verwendet werden soll (`keywords` oder `rephrase`).
*   **Prozess:**
    *   Der `alima_manager.analyze_abstract()`-Aufruf wird mit dem ursprünglichen Abstract (oder dem formatierten Suchergebnis-Text, falls kein Abstract vorhanden war) als `abstract`-Parameter und der Liste der GND-konformen Schlagworte (mit IDs) als `keywords`-Parameter durchgeführt.
    *   Das LLM führt den ausgewählten `final_llm_task` (z.B. `keywords` zur Extraktion oder `rephrase` zur Umformulierung) aus.
    *   Die Ausgabe des LLM wird geparst, um die finalen extrahierten Keywords und GND-Klassen zu erhalten.
*   **Ausgabe (intern im `KeywordAnalysisState`):**
    *   `llm_analysis`: `LlmKeywordAnalysis` - Enthält die vom LLM extrahierten GND-Keywords, das verwendete Modell, den Provider und den Prompt.
*   **JSON-Darstellung nach Schritt 3 (Beispiel):**
    ```json
    {
      "initial_keywords": ["Cadmium-Kontamination", "Bodenverschmutzung"],
      "search_suggesters_used": ["lobid", "swb"],
      "initial_gnd_classes": ["21.4", "32.5"],
      "search_results": [
        {
          "search_term": "Cadmium-Kontamination",
          "results": {
            "Cadmium": {"gndid": ["4009274-4"], "count": 150},
            "Kontamination": {"gndid": ["4032184-0"], "count": 80}
          }
        },
        {
          "search_term": "Bodenverschmutzung",
          "results": {
            "Bodenverschmutzung": {"gndid": ["4206275-5"], "count": 120}
          }
        }
      ],
      "llm_analysis": {
        "model_used": "cogito:32b",
        "provider_used": "ollama",
        "extracted_gnd_keywords": ["Cadmium (4009274-4)", "Bodenverschmutzung (4206275-5)", "Umweltverschmutzung (4061694-5)"],
        "prompt": "Du bist ein korrekter Bibliothekar, der aus einer Liste von OGND-Schlagworten..."
      },
      "timestamp": "2025-07-14T12:34:56.789"
    }
    ```
