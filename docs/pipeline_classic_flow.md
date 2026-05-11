# Flussdiagramm: Komplette ALIMA-Keyword-Pipeline

```mermaid
graph TD
    subgraph "ALIMA 'analyze-keywords' Pipeline"
        A[Start] --> B{Eingabe: Abstract und/oder Keywords?};
        
        B -- Nur Abstract --> C[Schritt 1: LLM extrahiert initiale Keywords];
        B -- Nur Keywords --> D[Schritt 1: Verwende direkt die Nutzer-Keywords];
        B -- Beides --> D;

        C --> E[initial_keywords];
        D --> E;

        E --> F[Schritt 2: Suche mit initialen Keywords in Suggestern <br/>(LOBID, SWB, Katalog)];
        F --> G[Ergebnis: Liste mit GND-konformen Keywords];
        
        G --> H[Schritt 3: LLM analysiert den ursprünglichen Text <br/>mit der GND-Keyword-Liste als Kontext];
        H --> I[Ergebnis: Finale, extrahierte GND-Keywords];
        
        I --> J[Ende: Ausgabe als JSON];
    end
```
