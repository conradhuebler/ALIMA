# Flussdiagramm: DK-Retrieval-Prozess

```mermaid
graph TD
    subgraph "DK-Retrieval-Prozess"
        A[Start: extrahiere DKs für Keywords] --> B{Cache-Prüfung};
        B -- Treffer --> C[Lade DKs aus Cache];
        C --> D[Formatiere Ergebnisse];
        D --> E[Ende: Gib Ergebnisse zurück];

        B -- Kein Treffer --> F[Schleife: Für jedes Keyword];
        F --> G[Live-Suche im Katalog];
        G --> H{Ergebnisse erhalten?};
        H -- Nein --> F;
        H -- Ja --> I[Verarbeite Ergebnisse];
        I --> J[Schleife: Für jeden Medientreffer];
        J --> K[Hole Titel-Details (MAB-Daten)];
        K --> L[Extrahiere Klassifikations-Liste];
        L --> M[Filtere reine DKs aus der Liste];
        M --> J;
        
        I --> N[Sammle alle gefundenen DKs];
        N --> O[Speichere neue DKs im Cache];
        O --> E;
    end
```
