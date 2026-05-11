# Flussdiagramm: DK-Retrieval-Prozess (Technische Details)

```mermaid
graph TD
    subgraph "DK-Retrieval-Prozess (Technische Details)"
        A[Start: extract_dk_classifications_for_keywords(keywords)] --> B{Cache-Prüfung in <br> UnifiedKnowledgeManager};
        B -- Treffer --> C[Lade DK-Objekte aus DB];
        C --> D[Ende: Gib gecachte Ergebnisse zurück];

        B -- Kein Treffer --> E[Schleife: Für jedes 'keyword' in der Eingabeliste];
        E --> F["<b>API Call 1 (Suche)</b><br/>client.search(keyword)"];
        F --> G["POST an 'SEARCH_URL'<br/>Body: SOAP-XML mit <lib:term>keyword</lib:term>"];
        G --> H[Antwort: XML mit Liste von Medien (Titel, RSN, etc.)];
        
        H --> I["Schleife: Für jedes Medium in der Antwort"];
        I --> J["<b>API Call 2 (Details)</b><br/>client.get_title_details(rsn)"];
        J --> K["POST an 'DETAILS_URL'<br/>Body: SOAP-XML mit <lib:RSN>rsn</lib:RSN>"];
        K --> L[Antwort: XML mit vollen MAB-Daten des Titels];
        L --> M["Interne Logik:<br/>extract_decimal_classifications(mab_daten)"];
        M --> N["Regex-Filterung der Klassifikations-Strings<br/>(z.B. extrahiere '543.42' aus 'DK 543.42')"];
        N --> O[Sammle gefilterte DK-Nummern];
        O --> I;

        E --> P[Sammle alle DK-Nummern von allen Keywords];
        P --> Q[Speichere neue DK-Ergebnisse<br/>in UnifiedKnowledgeManager];
        Q --> R[Ende: Gib neue Ergebnisse zurück];
    end
```
