# Protocol Formatting Functions for ALIMA CLI
# Claude Generated - Extracted from alima_cli.py
"""
Functions for displaying pipeline results in various formats:
    - detailed: Full formatted display with sections
    - compact: CSV format for scripting
    - k10plus: K10+/WinIBW catalog export format
"""

import json
import os
import sys
import csv
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any

from src.core.data_models import KeywordAnalysisState


# K10+/WinIBW export format tags
K10PLUS_KEYWORD_TAG = "5550"
K10PLUS_CLASSIFICATION_TAG = "6700"


def display_protocol(json_file: str, steps: List[str]):
    """Display pipeline results from a JSON protocol file.

    Args:
        json_file: Path to JSON protocol file
        steps: List of steps to display (or ["all"] for all steps)
    """
    # Check if file exists
    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}")
        return

    try:
        # Load JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)

        # Convert to KeywordAnalysisState object
        state = KeywordAnalysisState(
            original_abstract=state_dict.get('original_abstract'),
            initial_keywords=state_dict.get('initial_keywords', []),
            search_suggesters_used=state_dict.get('search_suggesters_used', []),
            initial_gnd_classes=state_dict.get('initial_gnd_classes', []),
            initial_llm_call_details=state_dict.get('initial_llm_call_details'),
            final_llm_analysis=state_dict.get('final_llm_analysis'),
            dk_search_results=state_dict.get('dk_search_results', []),
            dk_search_results_flattened=state_dict.get('dk_search_results_flattened', []),
            dk_statistics=state_dict.get('dk_statistics'),
            dk_classifications=state_dict.get('dk_classifications', []),
            timestamp=state_dict.get('timestamp'),
        )

        # Handle search_results - reconstruct SearchResult objects if needed
        search_results = []
        for sr in state_dict.get('search_results', []):
            if isinstance(sr, dict):
                search_results.append(sr)
            else:
                search_results.append(sr)
        state.search_results = search_results

        # Determine which steps to display
        display_steps = steps if steps != ["all"] else [
            "input", "initialisation", "search", "keywords", "dk_search", "dk_classification"
        ]

        # Header
        print("\n" + "="*70)
        print(f"ALIMA Pipeline Protocol")
        print(f"File: {json_file}")
        print(f"Timestamp: {state.timestamp}")
        print("="*70)

        # Display each requested step
        for step in display_steps:
            if step == "input":
                format_step_input(state)
            elif step == "initialisation":
                format_step_initialisation(state)
            elif step == "search":
                format_step_search(state)
            elif step == "keywords":
                format_step_keywords(state)
            elif step == "dk_search":
                format_step_dk_search(state)
            elif step == "dk_classification":
                format_step_dk_classification(state)

        print("\n" + "="*70 + "\n")

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
    except Exception as e:
        print(f"Error reading protocol: {e}")
        import traceback
        traceback.print_exc()


def format_step_input(state: KeywordAnalysisState):
    """Format and display input step results."""
    print("\n[STEP: INPUT]")
    print("-" * 70)

    if state.original_abstract:
        abstract_preview = state.original_abstract[:500]
        if len(state.original_abstract) > 500:
            abstract_preview += f" ... ({len(state.original_abstract) - 500} more chars)"
        print(f"Input Text ({len(state.original_abstract)} chars):")
        print(f"  {abstract_preview}")
    else:
        print("  (No input text)")


def format_step_initialisation(state: KeywordAnalysisState):
    """Format and display initialisation step results."""
    print("\n[STEP: INITIALISATION]")
    print("-" * 70)

    if state.initial_llm_call_details:
        details = state.initial_llm_call_details
        print(f"Model Used: {details.get('model_used', 'unknown')}")
        print(f"Provider: {details.get('provider_used', 'unknown')}")
        print()

    print(f"Free Keywords ({len(state.initial_keywords)} found):")
    if state.initial_keywords:
        for kw in state.initial_keywords:
            print(f"  * {kw}")
    else:
        print("  (No keywords extracted)")

    if state.initial_gnd_classes:
        print(f"\nInitial GND Classes ({len(state.initial_gnd_classes)} found):")
        for gnd_class in state.initial_gnd_classes:
            print(f"  * {gnd_class}")


def format_step_search(state: KeywordAnalysisState):
    """Format and display search step results."""
    print("\n[STEP: SEARCH]")
    print("-" * 70)

    if state.search_suggesters_used:
        print(f"Suggesters Used: {', '.join(state.search_suggesters_used)}")

    if not state.search_results:
        print("  (No search results)")
        return

    print(f"\nSearch Results ({len(state.search_results)} search terms):")

    for result in state.search_results:
        if isinstance(result, dict):
            search_term = result.get('search_term', 'unknown')
            results = result.get('results', {})

            print(f"\n  Search Term: {search_term}")
            print(f"  -" * 20)

            if results:
                for keyword, data in results.items():
                    gndids = data.get('gndid', [])
                    count = data.get('count', 0)

                    if isinstance(gndids, list):
                        gndid_str = ", ".join(gndids)
                    else:
                        gndid_str = str(gndids)

                    print(f"    -> {keyword}")
                    print(f"      GND ID: {gndid_str}")
                    print(f"      Hits: {count}")
            else:
                print("    (No results for this term)")


def format_step_keywords(state: KeywordAnalysisState):
    """Format and display keywords step results."""
    print("\n[STEP: KEYWORDS]")
    print("-" * 70)

    if not state.final_llm_analysis:
        print("  (No final LLM analysis performed)")
        return

    analysis = state.final_llm_analysis

    if isinstance(analysis, dict):
        model = analysis.get('model_used', 'unknown')
        provider = analysis.get('provider_used', 'unknown')
        keywords = analysis.get('extracted_gnd_keywords', [])
    else:
        model = getattr(analysis, 'model_used', 'unknown')
        provider = getattr(analysis, 'provider_used', 'unknown')
        keywords = getattr(analysis, 'extracted_gnd_keywords', [])

    print(f"Model Used: {model}")
    print(f"Provider: {provider}")
    print(f"\nFinal GND Keywords ({len(keywords)} found):")

    if keywords:
        for kw in keywords:
            # Parse keyword and GND ID
            if isinstance(kw, str) and '(' in kw and ')' in kw:
                parts = kw.rsplit('(', 1)
                term = parts[0].strip()
                gndid = parts[1].rstrip(')')
                print(f"  [OK] {term}")
                print(f"    GND ID: {gndid}")
            else:
                print(f"  [OK] {kw}")
    else:
        print("  (No keywords extracted)")


def format_step_dk_search(state: KeywordAnalysisState):
    """Format and display DK search step results with titles."""
    print("\n[STEP: DK_SEARCH]")
    print("-" * 70)

    if not state.dk_search_results:
        print("  (No DK search results)")
        return

    print(f"DK Search Results ({len(state.dk_search_results)} classifications found):")

    # Display deduplication statistics if available
    if state.dk_statistics:
        format_dk_statistics(state.dk_statistics)
        print("-" * 70)

    for result in state.dk_search_results:
        if isinstance(result, dict):
            # Support both old and new data formats
            dk_code = result.get('dk', result.get('code', 'unknown'))
            titles = result.get('titles', [])
            total_count = result.get('count', len(titles))
            keywords = result.get('keywords', [])

            # Display DK code with count
            print(f"\n  DK {dk_code}")
            if keywords:
                print(f"     Keywords: {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''}")
            print(f"     Katalogisiert in {total_count} Titel{'n' if total_count != 1 else ''}")

            # Display sample titles (first 5)
            if titles:
                sample_size = min(5, len(titles))
                print(f"     Sample Titel ({sample_size}/{len(titles)}):")
                for i, title in enumerate(titles[:sample_size], 1):
                    # Truncate long titles to 70 characters
                    short_title = (title[:67] + "...") if len(title) > 70 else title
                    print(f"       {i}. {short_title}")

                if len(titles) > 5:
                    print(f"       ... und {len(titles) - 5} weitere Titel")


def format_dk_statistics(stats: Dict[str, Any]):
    """Format and display DK deduplication statistics.

    Args:
        stats: Statistics dictionary from _calculate_dk_statistics()
    """
    if not stats:
        return

    print("\nDK Deduplication Statistics")
    print("-" * 70)

    # Deduplication Metrics
    dedup = stats.get("deduplication_stats", {})
    if dedup:
        print(f"\n  Deduplication Metrics:")
        print(f"    Original classifications: {dedup.get('original_count', 0)}")
        print(f"    Duplicates removed:       {dedup.get('duplicates_removed', 0)}")
        print(f"    Deduplication rate:       {dedup.get('deduplication_rate', '0%')}")
        print(f"    Token savings (est.):     ~{dedup.get('estimated_token_savings', 0)} tokens")

    # Summary Stats
    total_class = stats.get("total_classifications", 0)
    total_kw = stats.get("total_keywords_searched", 0)
    print(f"\n  Search Summary:")
    print(f"    Keywords searched:        {total_kw}")
    print(f"    Unique classifications:   {total_class}")

    # Top 10 Most Frequent
    most_frequent = stats.get("most_frequent", [])
    if most_frequent:
        print(f"\n  Top 10 Most Frequent Classifications:")
        print(f"  {'Rank':<6} {'DK Code':<15} {'Type':<6} {'Count':<8} {'Keywords':<30} {'Titles'}")
        print(f"  {'-'*6} {'-'*15} {'-'*6} {'-'*8} {'-'*30} {'-'*6}")

        for idx, item in enumerate(most_frequent, 1):
            dk_code = item.get('dk', 'unknown')
            dk_type = item.get('type', 'DK')
            count = item.get('count', 0)
            keywords = item.get('keywords', [])
            unique_titles = item.get('unique_titles', 0)

            # Truncate keywords to 3
            kw_display = ', '.join(keywords[:3])
            if len(keywords) > 3:
                kw_display += f' (+{len(keywords)-3})'
            kw_display = (kw_display[:27] + '...') if len(kw_display) > 30 else kw_display

            print(f"  {idx:<6} {dk_code:<15} {dk_type:<6} {count:<8} {kw_display:<30} {unique_titles}")

    # Keyword Coverage Summary
    coverage = stats.get("keyword_coverage", {})
    if coverage:
        print(f"\n  Keyword Coverage: {len(coverage)} keywords matched to classifications")

    print()


def format_step_dk_classification(state: KeywordAnalysisState):
    """Format and display DK classification step results."""
    print("\n[STEP: DK_CLASSIFICATION]")
    print("-" * 70)

    if not state.dk_classifications:
        print("  (No DK classifications assigned)")
        return

    print(f"DK Classifications ({len(state.dk_classifications)} assigned):")

    for classification in state.dk_classifications:
        if isinstance(classification, str):
            print(f"  * {classification}")
        elif isinstance(classification, dict):
            code = classification.get('code', 'unknown')
            label = classification.get('label', '')
            print(f"  * {code}" + (f" - {label}" if label else ""))


def display_protocol_compact(json_file: str, steps: List[str]):
    """Display pipeline results in compact CSV format.

    Output format: filename,step,data
    One line per step with pipe-separated values
    """
    # Check if file exists
    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}", file=sys.stderr)
        return

    try:
        # Load JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)

        # Convert to KeywordAnalysisState object
        state = KeywordAnalysisState(
            original_abstract=state_dict.get('original_abstract'),
            initial_keywords=state_dict.get('initial_keywords', []),
            search_suggesters_used=state_dict.get('search_suggesters_used', []),
            initial_gnd_classes=state_dict.get('initial_gnd_classes', []),
            initial_llm_call_details=state_dict.get('initial_llm_call_details'),
            final_llm_analysis=state_dict.get('final_llm_analysis'),
            dk_search_results=state_dict.get('dk_search_results', []),
            dk_search_results_flattened=state_dict.get('dk_search_results_flattened', []),
            dk_statistics=state_dict.get('dk_statistics'),
            dk_classifications=state_dict.get('dk_classifications', []),
            timestamp=state_dict.get('timestamp'),
        )

        # Handle search_results
        search_results = []
        for sr in state_dict.get('search_results', []):
            if isinstance(sr, dict):
                search_results.append(sr)
        state.search_results = search_results

        # Get filename for output
        filename = Path(json_file).name

        # Determine which steps to display
        display_steps = steps if steps != ["all"] else [
            "input", "initialisation", "search", "keywords", "dk_search", "dk_classification"
        ]

        # Display each requested step
        for step in display_steps:
            csv_line = format_step_compact_csv(step, state, filename)
            if csv_line:
                print(csv_line)

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading protocol: {e}", file=sys.stderr)


def display_protocol_k10plus(json_file: str):
    """Display protocol in K10+/WinIBW catalog export format.

    Format:
        5550 Schlagwort
        6700 DK CODE
    """
    # Check if file exists
    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}", file=sys.stderr)
        return

    try:
        # Load JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)

        # Convert to KeywordAnalysisState object
        state = KeywordAnalysisState(
            original_abstract=state_dict.get('original_abstract'),
            initial_keywords=state_dict.get('initial_keywords', []),
            search_suggesters_used=state_dict.get('search_suggesters_used', []),
            initial_gnd_classes=state_dict.get('initial_gnd_classes', []),
            initial_llm_call_details=state_dict.get('initial_llm_call_details'),
            final_llm_analysis=state_dict.get('final_llm_analysis'),
            dk_search_results=state_dict.get('dk_search_results', []),
            dk_search_results_flattened=state_dict.get('dk_search_results_flattened', []),
            dk_statistics=state_dict.get('dk_statistics'),
            dk_classifications=state_dict.get('dk_classifications', []),
            timestamp=state_dict.get('timestamp'),
        )

        # Generate K10+ export lines
        lines = []

        # GND Keywords (5550 - Schlagworte)
        if state.final_llm_analysis:
            analysis = state.final_llm_analysis
            if isinstance(analysis, dict):
                keywords = analysis.get('extracted_gnd_keywords', [])
            else:
                keywords = getattr(analysis, 'extracted_gnd_keywords', [])

            for keyword in keywords:
                # Remove GND-ID from "Keyword (GNDID)" format
                if "(" in keyword and ")" in keyword:
                    term = keyword.split("(")[0].strip()
                else:
                    term = keyword.strip()

                lines.append(f"{K10PLUS_KEYWORD_TAG} {term}")

        # DK Classifications (6700 - Systematiken)
        for dk in state.dk_classifications:
            # Remove label if present ("628.5 - Label" -> "628.5")
            dk_code = dk.split(" - ")[0].strip() if " - " in dk else dk.strip()
            # Remove "DK " prefix if present
            dk_clean = dk_code.replace("DK ", "").strip()
            lines.append(f"{K10PLUS_CLASSIFICATION_TAG} DK {dk_clean}")

        # Print all lines
        for line in lines:
            print(line)

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading protocol: {e}", file=sys.stderr)


def format_step_compact_csv(step: str, state: KeywordAnalysisState, filename: str) -> str:
    """Generate CSV line for a pipeline step.

    Format: filename,step,data (pipe-separated values)
    """
    if step == "input":
        # Input text (truncated to first 100 chars)
        text_preview = state.original_abstract[:100].replace('\n', ' ').replace('"', '\'') if state.original_abstract else ""
        data = text_preview

    elif step == "initialisation":
        # Free keywords
        data = "|".join(state.initial_keywords) if state.initial_keywords else ""

    elif step == "search":
        # Search results: keyword:hits|keyword:hits
        result_parts = []
        for result in state.search_results:
            if isinstance(result, dict):
                for keyword, info in result.get('results', {}).items():
                    count = info.get('count', 0)
                    result_parts.append(f"{keyword}:{count}")
        data = "|".join(result_parts) if result_parts else ""

    elif step == "keywords":
        # Final GND keywords with IDs
        if state.final_llm_analysis:
            analysis = state.final_llm_analysis
            if isinstance(analysis, dict):
                keywords = analysis.get('extracted_gnd_keywords', [])
            else:
                keywords = getattr(analysis, 'extracted_gnd_keywords', [])
            data = "|".join(keywords) if keywords else ""
        else:
            data = ""

    elif step == "dk_search":
        # DK search: dk_code:count:sample_titles or keyword:hits (legacy format)
        result_parts = []
        for result in state.dk_search_results:
            if isinstance(result, dict):
                # Support both formats
                dk_code = result.get('dk', result.get('keyword', ''))
                titles = result.get('titles', [])
                total_count = result.get('count', result.get('hits', 0))

                if dk_code:
                    # New format with titles
                    if titles:
                        sample_titles = "|".join(titles[:3])  # First 3 titles
                        result_parts.append(f"{dk_code}:{total_count}:{sample_titles}")
                    else:
                        result_parts.append(f"{dk_code}:{total_count}")
        data = "||".join(result_parts) if result_parts else ""  # Use || to separate DK results

    elif step == "dk_classification":
        # DK codes
        dk_parts = []
        for classification in state.dk_classifications:
            if isinstance(classification, str):
                # Extract just the code part (e.g., "628.5" from "628.5 - Label")
                code = classification.split(' - ')[0].strip() if ' - ' in classification else classification
                dk_parts.append(code)
            elif isinstance(classification, dict):
                code = classification.get('code', '')
                if code:
                    dk_parts.append(code)
        data = "|".join(dk_parts) if dk_parts else ""
    else:
        return ""

    # Create CSV line with proper quoting for data with commas/quotes
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow([filename, step, data])
    return output.getvalue().rstrip('\r\n')
