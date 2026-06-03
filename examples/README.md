# Pipeline Comparison Examples

## DOI: 10.1002/cmtd.202200006

**Title:** SupraFit — An Open Source Qt Based Fitting Application to Determine Stability Constants from Titration Experiments

## Files

| File | Description |
|---|---|
| `doi_input.json` | Input data: abstract, DOI, title |
| `classic_result.json` | Full classical pipeline output |
| `agentic_result.json` | Full agentic (v4 workflow) output |
| `compare_pipeline_results.py` | Side-by-side comparison script |

## Usage

```bash
python examples/compare_pipeline_results.py
```

## Key Differences Observed

| Aspect | Classical | Agentic |
|---|---|---|
| Initial keywords | 20 | 19 |
| Final keywords | 19 | **65** |
| DK codes | 5-digit precision | 3-digit precision |
| Common DK codes | — | only `543.08` |

Run the script for the full diff.
