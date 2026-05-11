# ALIMA Pipeline Documentation

## Overview

ALIMA provides a unified 5-step analysis pipeline for library science keyword extraction and GND (Gemeinsame Normdatei) classification. The pipeline is available through both CLI and GUI interfaces, sharing identical core logic for consistent results.

## Pipeline Architecture

### Step Flow
```
1. INPUT → 2. INITIALISATION → 3. SEARCH → 4. KEYWORDS → 5. CLASSIFICATION
```

### Step Details

**1. Input (`input`)**
- Accepts text from various sources (clipboard, files, DOI, images)
- Validates and prepares text for analysis
- No LLM required

**2. Initialisation (`initialisation`)**
- Extracts free keywords from input text using LLM
- Uses "initialisation" task from prompts.json
- **LLM Required**: Yes
- **Output**: Initial keyword list and GND classes

**3. Search (`search`)**
- Searches GND/SWB/LOBID databases for keyword matches
- Uses configurable suggesters (lobid, swb)
- No LLM required
- **Output**: GND search results with IDs and metadata

**4. Keywords (`keywords`)**
- Performs "Verbale Erschließung" (final keyword analysis)
- Uses "keywords" or "rephrase" task from prompts.json
- **LLM Required**: Yes
- **Output**: Final GND-compliant keywords

**5. Classification (`classification`)**
- Optional DDC/DK/RVK classification assignment
- **LLM Required**: Yes (when enabled)
- Default: Disabled

## CLI Usage

### New Pipeline Command

#### Basic Usage
```bash
python alima_cli.py pipeline \
  --input-text "Your analysis text here" \
  --initial-model "cogito:14b" \
  --final-model "cogito:32b"
```

#### Full Options
```bash
python alima_cli.py pipeline \
  --input-text "Künstliche Intelligenz revolutioniert die Datenverarbeitung..." \
  --initial-model "cogito:14b" \
  --final-model "cogito:32b" \
  --provider "ollama" \
  --ollama-host "http://139.20.140.163" \
  --ollama-port 11434 \
  --suggesters "lobid" "swb" \
  --output-json "pipeline_results.json"
```

#### Resume from Saved State
```bash
python alima_cli.py pipeline \
  --resume-from "pipeline_results.json" \
  --output-json "continued_results.json"
```

#### List Available Models
```bash
python alima_cli.py list-models \
  --ollama-host "http://139.20.140.163" \
  --ollama-port 11434
```

### Pipeline Parameters

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--input-text` | Yes | Text to analyze | - |
| `--initial-model` | Yes | Model for initial keyword extraction | - |
| `--final-model` | Yes | Model for final keyword analysis | - |
| `--provider` | No | LLM provider (ollama, gemini, etc.) | "ollama" |
| `--ollama-host` | No | Ollama server host | "http://localhost" |
| `--ollama-port` | No | Ollama server port | 11434 |
| `--suggesters` | No | Search providers | ["lobid", "swb"] |
| `--output-json` | No | Save results to JSON file | - |
| `--resume-from` | No | Resume from saved JSON state | - |

## GUI Usage

### Pipeline Tab

1. **Input Step**: Use UnifiedInputWidget to provide text (clipboard, files, DOI, images)
2. **Auto-Pipeline Button**: Click "🚀 Auto-Pipeline" for complete workflow
3. **Real-time Feedback**: Watch live streaming in the PipelineStreamWidget
4. **Step Navigation**: Click on pipeline step tabs to view results
5. **Configuration**: Use "⚙️ Config" button to adjust pipeline settings

### GUI Features

- **Visual Step Indicators**: 
  - ▷ Pending (gray)
  - ▶ Running (blue)
  - ✓ Completed (green)
  - ✗ Error (red)

- **Live Streaming**: Real-time LLM token display
- **Progress Tracking**: Step timing and duration display
- **Result Display**: Each step shows output in dedicated text areas
- **Auto-Advancement**: Pipeline automatically progresses through steps

## Configuration

### Pipeline Configuration (GUI)

The pipeline can be configured through the GUI configuration dialog:

```python
PipelineConfig(
    auto_advance=True,              # Automatic step progression
    stop_on_error=True,             # Halt on first error
    save_intermediate_results=True, # Cache step outputs
    
    step_configs={
        "initialisation": {
            "enabled": True,
            "provider": "ollama",
            "model": "cogito:14b",
            "temperature": 0.7,
            "top_p": 0.1,
            "task": "initialisation"
        },
        "keywords": {
            "enabled": True,
            "provider": "ollama", 
            "model": "cogito:32b",
            "temperature": 0.7,
            "top_p": 0.1,
            "task": "keywords"
        },
        "classification": {
            "enabled": False,  # Optional step
            "provider": "gemini",
            "model": "gemini-1.5-flash"
        }
    },
    
    search_suggesters=["lobid", "swb"]
)
```

### Prompts Configuration

Pipeline tasks are defined in `prompts.json`:

```json
{
    "initialisation": {
        "prompts": [
            [
                "Du bist ein korrekter Bibliothekar...",
                "System prompt for reasoning",
                "0.7",    // temperature
                "0.1",    // top_p
                ["cogito:14b", "cogito:32b"],  // preferred models
                "0"       // seed
            ]
        ]
    },
    "keywords": {
        "prompts": [
            // Similar structure for final keyword analysis
        ]
    }
}
```

## JSON State Management

### Save Pipeline State
Both CLI and GUI can save complete pipeline state to JSON:

```json
{
    "original_abstract": "Input text...",
    "initial_keywords": ["keyword1", "keyword2"],
    "initial_gnd_classes": ["class1", "class2"],
    "search_results": [
        {
            "search_term": "keyword1",
            "results": {
                "matched_term": {
                    "gndid": ["123456789"],
                    "count": 5
                }
            }
        }
    ],
    "initial_llm_call_details": {
        "task_name": "initialisation",
        "model_used": "cogito:14b",
        "provider_used": "ollama",
        "response_full_text": "LLM response...",
        "extracted_gnd_keywords": ["keyword1", "keyword2"],
        "extracted_gnd_classes": ["class1", "class2"]
    },
    "final_llm_analysis": {
        "task_name": "keywords",
        "model_used": "cogito:32b",
        "provider_used": "ollama",
        "response_full_text": "Final LLM response...",
        "extracted_gnd_keywords": ["final_keyword1", "final_keyword2"],
        "extracted_gnd_classes": []
    }
}
```

### Resume from State
```bash
# CLI resume
python alima_cli.py pipeline --resume-from "saved_state.json"

# GUI resume (feature ready for implementation)
# Load JSON file through GUI dialog
```

## Architecture

### Shared Logic (`src/utils/pipeline_utils.py`)

**PipelineStepExecutor**: Core class handling all pipeline steps
- `execute_initial_keyword_extraction()`: Initial LLM analysis
- `execute_gnd_search()`: Search database queries  
- `execute_final_keyword_analysis()`: Final LLM verification

**PipelineJsonManager**: JSON serialization utilities
- `save_analysis_state()`: Save KeywordAnalysisState to JSON
- `load_analysis_state()`: Load from JSON file
- `convert_sets_to_lists()`: JSON compatibility conversion

**PipelineResultFormatter**: Display formatting
- `format_search_results_for_display()`: GUI display format
- `format_keywords_for_prompt()`: LLM prompt format
- `get_gnd_compliant_keywords()`: Extract GND-compliant results

### Interface Integration

**CLI (`alima_cli.py`)**:
- New `pipeline` command using `execute_complete_pipeline()`
- Legacy `analyze-keywords` command for compatibility
- JSON save/resume functionality

**GUI (`src/ui/pipeline_tab.py` + `src/core/pipeline_manager.py`)**:
- PipelineManager refactored to use PipelineStepExecutor
- Real-time streaming integration
- Visual progress tracking

## Error Handling

### Common Issues

**Model Not Found**:
```bash
# Check available models
python alima_cli.py list-models --ollama-host "http://server" --ollama-port 11434
```

**Connection Issues**:
```bash
# Test with correct host/port
python alima_cli.py pipeline --ollama-host "http://139.20.140.163" --ollama-port 11434
```

**Missing prompts.json**:
- Ensure `prompts.json` exists in project root
- Check task definitions for "initialisation" and "keywords"

**Parameter Conflicts**:
- Pipeline automatically filters incompatible parameters
- Only AlimaManager-compatible parameters are passed through

## Performance Considerations

- **Initial Model**: Lighter model (cogito:14b) for keyword extraction
- **Final Model**: Heavier model (cogito:32b) for detailed analysis
- **Caching**: Search results cached in SQLite database
- **Streaming**: Real-time token display for user feedback
- **Threading**: GUI pipeline runs in background thread

## Testing

### Verified Functionality
- ✅ CLI pipeline command working with custom Ollama host/port
- ✅ GUI auto-pipeline button functional
- ✅ Real-time streaming display in both interfaces
- ✅ Final GND keywords correctly displayed  
- ✅ JSON save/resume capabilities
- ✅ Parameter filtering and compatibility resolved
- ✅ Both interfaces produce identical results

### Test Commands
```bash
# Quick test
python alima_cli.py pipeline \
  --input-text "Machine learning algorithms" \
  --initial-model "cogito:14b" \
  --final-model "cogito:32b" \
  --ollama-host "http://139.20.140.163" \
  --ollama-port 11434

# Full test with save
python alima_cli.py pipeline \
  --input-text "Künstliche Intelligenz und Deep Learning revolutionieren die automatische Textverarbeitung" \
  --initial-model "cogito:14b" \
  --final-model "cogito:32b" \
  --provider "ollama" \
  --ollama-host "http://139.20.140.163" \
  --ollama-port 11434 \
  --suggesters "lobid" "swb" \
  --output-json "test_results.json"
```

## Future Enhancements

### Ready for Implementation
- **Batch Processing**: Multiple inputs through single pipeline
- **Pipeline Templates**: Save/load different configurations
- **Advanced Export**: Results in PDF, Excel, CSV formats
- **GUI JSON Resume**: Load saved states through file dialog
- **Pipeline Monitoring**: Detailed metrics and performance tracking

### Architecture Extensions
- **Plugin System**: Custom suggester implementations
- **Webhook Integration**: External system notifications
- **REST API**: HTTP endpoints for pipeline execution
- **Cloud Integration**: Remote processing capabilities