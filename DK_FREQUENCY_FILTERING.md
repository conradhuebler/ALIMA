# DK Classification Frequency Filtering - Implementation Documentation

## Overview - Claude Generated

Die DK-Klassifikations-Pipeline wurde erweitert, um große Ergebnissätze durch Häufigkeits-basierte Filterung zu optimieren. Dies reduziert die Prompt-Größe für LLMs und fokussiert auf die relevantesten Klassifikationen.

## Problem

Bei der DK-Klassifikationssuche können sehr große Ergebnismengen entstehen, die:
- LLM-Prompts überladen und ineffizient machen
- Viele schwach relevante Klassifikationen enthalten
- Unnötige Kosten und Latenz verursachen

## Lösung

**Frequenz-Threshold-Filter**: Nur DK-Klassifikationen mit ≥N Vorkommen im Katalog werden an das LLM weitergegeben.

### Implementation Details

#### 1. Pipeline Parameter

```python
def execute_dk_classification(
    ...
    dk_frequency_threshold: int = 10,  # Mindest-Häufigkeit
    ...
):
```

#### 2. Filterlogik

```python
# Filter results by frequency threshold
filtered_results = []
for result in dk_search_results:
    if "count" in result:
        count = result.get("count", 0)
        if count >= dk_frequency_threshold:
            filtered_results.append(result)
```

#### 3. CLI-Unterstützung

```bash
python3 src/alima_cli.py pipeline \
  --dk-frequency-threshold 15 \
  --input-text "Your text here" \
  # ... andere Parameter
```

#### 4. GUI-Unterstützung

Der Parameter ist über die **Pipeline-Konfiguration** in der ALIMA GUI verfügbar:

**Zugriff:**
1. ALIMA GUI öffnen
2. Pipeline-Tab auswählen
3. "Konfiguration" Button klicken
4. "📚 DK-Klassifikation" Tab öffnen
5. **"Häufigkeits-Schwellenwert"** einstellen

**UI-Features:**
- **SpinBox-Control**: 1-100 Vorkommen
- **Standard-Wert**: 10 Vorkommen
- **Tooltip**: Detaillierte Erklärung der Funktionalität
- **Live-Speicherung**: Wird automatisch in Pipeline-Konfiguration gespeichert

**Screenshot-Beschreibung:**
```
┌─ DK Klassifikation ──────────────────┐
│ Häufigkeits-Schwellenwert: [10] Vorkommen │
│                                      │
│ 💡 Tooltip:                          │
│ "Mindest-Häufigkeit für DK-Klassifikationen. │
│  Nur Klassifikationen mit ≥ N Vorkommen im   │
│  Katalog werden an das LLM weitergegeben."   │
└──────────────────────────────────────┘
```

**Pipeline-Konfiguration (JSON):**
```json
{
  "step_configs": {
    "dk_classification": {
      "enabled": true,
      "dk_frequency_threshold": 15
    }
  }
}
```

## Konfiguration

### Standard-Werte
- **Default**: 10 Vorkommen
- **Empfohlene Bereiche**: 
  - Konservativ: 5-8 (mehr Ergebnisse)
  - Standard: 10-15 (ausgewogen)  
  - Aggressiv: 20+ (nur häufigste Klassifikationen)

### Anpassung

**CLI:**
```bash
--dk-frequency-threshold 15
```

**Pipeline-Konfiguration:**
```json
{
  "dk_classification": {
    "dk_frequency_threshold": 15
  }
}
```

## Auswirkungen

### Vorher
```
[DK_CLASSIFICATION] Starte DK-Klassifikation mit 847 Katalog-Einträgen
```

### Nachher 
```
[DK_CLASSIFICATION] Filtere DK-Ergebnisse: 23 Einträge mit ≥10 Vorkommen, 824 mit niedrigerer Häufigkeit ausgeschlossen
```

## Benefits

1. **Kleinere Prompts**: Reduzierte Anzahl von Klassifikationen im LLM-Prompt
2. **Höhere Relevanz**: Fokus auf häufig auftretende, relevante Klassifikationen
3. **Bessere Performance**: Schnellere LLM-Verarbeitung und niedrigere Kosten
4. **Konfigurierbarkeit**: Anpassbarer Threshold je nach Anwendungsfall

## Future Enhancement (TODO)

**Chunking-Strategie**: Bei sehr großen gefilterten Ergebnissätzen könnte eine Chunking-Strategie implementiert werden:

```python
# TODO: Implement chunking for large filtered result sets
def chunk_dk_results(filtered_results, chunk_size=50):
    """Split filtered results into manageable chunks for LLM processing"""
    pass
```

## Testing

Beispiel-Test für verschiedene Threshold-Werte:

```python
# Test data: 6 results with counts [25, 15, 5, 12, 3, 20]
# Threshold 5:  5 results (excludes count=3)
# Threshold 10: 4 results (excludes count=5,3) 
# Threshold 15: 3 results (excludes count=12,5,3)
```

## Integration

Die Implementierung ist vollständig in die bestehende Pipeline integriert:

- ✅ `pipeline_utils.py`: Kern-Implementierung
- ✅ `pipeline_manager.py`: Parameter-Weiterleitung  
- ✅ `alima_cli.py`: CLI-Unterstützung
- ✅ Dokumentation und Logging

**Status**: Production Ready ✅