# ALIMA CLI How-To Guide

This guide explains how to use the ALIMA Command Line Interface (CLI) for abstract analysis, including saving and loading task states.

## 1. Running an Analysis Task

To run an analysis task and optionally save its state to a JSON file, use the `run` command:

```bash
python3 alima_cli.py run <task_name> --abstract "<your_abstract_text>" --model <model_name> [OPTIONS]
```

**Arguments:**

*   `<task_name>`: The analysis task to perform (e.g., `abstract`, `keywords`).
*   `--abstract "<your_abstract_text>"`: The abstract or text to analyze. Enclose in double quotes.
*   `--model <model_name>`: The LLM model to use for the analysis (e.g., `cogito:8b`, `gemini-1.5-flash`).

**Options:**

*   `--keywords "<optional_keywords>"`: Optional keywords to include in the analysis. Enclose in double quotes.
*   `--provider <provider_name>`: The LLM provider to use (e.g., `ollama`, `gemini`). Default is `ollama`.
*   `--ollama-host <host_url>`: Ollama host URL. Default is `http://localhost`.
*   `--ollama-port <port_number>`: Ollama port. Default is `11434`.
*   `--use-chunking-abstract`: Enable chunking for the abstract (flag).
*   `--abstract-chunk-size <size>`: Chunk size for the abstract (integer). Default is `100`.
*   `--use-chunking-keywords`: Enable chunking for keywords (flag).
*   `--keyword-chunk-size <size>`: Chunk size for keywords (integer). Default is `500`.
*   `--output-json <file_path>`: Path to save the `TaskState` JSON output. If provided, the full state of the analysis will be saved to this file.

**Example:**

```bash
python3 alima_cli.py run abstract \
    --abstract "This book covers cadmium contamination of soil and plants..." \
    --model cogito:8b \
    --provider ollama \
    --ollama-host http://139.20.140.163 \
    --ollama-port 11434 \
    --use-chunking-abstract \
    --abstract-chunk-size 10 \
    --output-json my_analysis_result.json
```

## 2. Loading a Saved Analysis State

To load and display a previously saved analysis state from a JSON file, use the `load-state` command:

```bash
python3 alima_cli.py load-state <input_file_path>
```

**Arguments:**

*   `<input_file_path>`: Path to the `TaskState` JSON input file.

**Example:**

```bash
python3 alima_cli.py load-state my_analysis_result.json
```

This will print the `full_text`, `matched_keywords`, and `gnd_systematic` from the loaded JSON file.

## 3. Key Information Extracted and Saved

When `--output-json` is used, the following information is saved in the JSON file:

*   `abstract_data`: The original abstract and keywords provided.
*   `analysis_result`: Contains the `full_text` response from the LLM, `matched_keywords` (extracted from LLM output), and `gnd_systematic` (extracted from LLM output).
*   `prompt_config`: The configuration of the prompt used for the analysis.
*   `status`: The status of the task (e.g., `completed`, `failed`).
*   `task_name`: The name of the task performed.
*   `model_used`: The LLM model used.
*   `provider_used`: The LLM provider used.
*   `use_chunking_abstract`, `abstract_chunk_size`, `use_chunking_keywords`, `keyword_chunk_size`: Chunking settings used.

This allows for reproducible analysis and the ability to resume or further process results programmatically.