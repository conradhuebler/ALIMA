#!/usr/bin/env python3
"""
Compare classical vs agentic pipeline results for DOI 10.1002/cmtd.202200006.

Usage:
    python examples/compare_pipeline_results.py

Output:
    Prints side-by-side comparison of initial keywords, final keywords,
    DK classifications, timing, and coverage metrics.
"""

import json
from pathlib import Path


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    examples_dir = Path(__file__).parent
    classic = load_json(examples_dir / "classic_result.json")
    agentic = load_json(examples_dir / "agentic_result.json")
    input_data = load_json(examples_dir / "doi_input.json")

    print("=" * 70)
    print("Pipeline Comparison: Classical vs Agentic")
    print("=" * 70)
    print(f"DOI : 10.1002/cmtd.202200006")
    print(f"Title: {input_data.get('title', 'N/A')}")
    print()

    # --- EXTRACTION / INITIAL KEYWORDS ---
    c_init = classic["results"]["initial_keywords"]
    a_extraction = next(
        (s for s in agentic["steps"] if s["step_id"] == "extraction"), {}
    )
    a_init = (
        a_extraction.get("data", {}).get("response", {}).get("keywords", [])
    )

    print("--- Initial / Extracted Keywords ---")
    print(f"Classical : {len(c_init)}")
    print(f"Agentic   : {len(a_init)}")
    c_set = {k.lower() for k in c_init}
    a_set = {k.lower() for k in a_init}
    print(f"Common    : {len(c_set & a_set)}")
    print(f"Only classical : {sorted(c_set - a_set)}")
    print(f"Only agentic   : {sorted(a_set - c_set)}")
    print()

    # --- FINAL KEYWORDS ---
    c_final = [k.split(" (GND-ID:")[0].strip() for k in classic["results"]["final_keywords"]]
    a_sel_chunks = next(
        (s for s in agentic["steps"] if s["step_id"] == "selection_chunks"), {}
    )
    a_final = []
    for kw in (
        a_sel_chunks.get("data", {}).get("response", {}).get("keywords", [])
    ):
        if isinstance(kw, dict):
            a_final.append(kw.get("keyword", ""))
        else:
            a_final.append(str(kw))

    print("--- Final Keywords ---")
    print(f"Classical : {len(c_final)}")
    print(f"Agentic   : {len(a_final)}")
    c_fset = {k.lower() for k in c_final}
    a_fset = {k.lower() for k in a_final}
    print(f"Common    : {len(c_fset & a_fset)}")
    print(f"Only classical : {sorted(c_fset - a_fset)}")
    print(f"Only agentic   : {sorted(a_fset - c_fset)}")
    print()

    # --- DK CLASSIFICATIONS ---
    c_dk = [d["display"] for d in classic["results"]["classifications"]]
    a_cl = next(
        (s for s in agentic["steps"] if s["step_id"] == "classification"), {}
    )
    a_dk_raw = (
        a_cl.get("data", {}).get("response", {}).get("classifications", [])
    )
    a_dk = [f"{cl.get('type', 'DK')} {cl.get('code', '')}" for cl in a_dk_raw]

    print("--- DK Classifications ---")
    print(f"Classical : {len(c_dk)}")
    for d in c_dk:
        print(f"  {d}")
    print(f"Agentic   : {len(a_dk)}")
    for d in a_dk:
        print(f"  {d}")

    c_dk_codes = {d.replace("DK ", "") for d in c_dk}
    a_dk_codes = {cl.get("code", "") for cl in a_dk_raw}
    print(f"Common codes : {sorted(c_dk_codes & a_dk_codes)}")
    print(f"Only classical: {sorted(c_dk_codes - a_dk_codes)}")
    print(f"Only agentic  : {sorted(a_dk_codes - c_dk_codes)}")
    print()

    # --- TIMING ---
    print("--- Timing ---")
    print(f"Classical : ~10 min (full pipeline)")
    print(
        f"Agentic   : {agentic.get('duration_seconds', 0) / 60:.1f} min (reported)"
    )
    print()

    # --- COVERAGE ---
    abstract = input_data.get("abstract", "").lower()
    c_in_abstract = sum(1 for k in c_final if k.lower() in abstract)
    a_in_abstract = sum(1 for k in a_final if k.lower() in abstract)

    print("--- Keyword Coverage (in abstract text) ---")
    print(f"Classical keywords found in abstract: {c_in_abstract}/{len(c_final)}")
    print(f"Agentic keywords found in abstract  : {a_in_abstract}/{len(a_final)}")
    print()

    # --- WORKING TITLE ---
    print("--- Working Title ---")
    print(f"Classical : {classic['results'].get('working_title', 'N/A')}")
    print(
        f"Agentic   : {a_extraction.get('data', {}).get('response', {}).get('title', 'N/A')}"
    )
    print()

    print("=" * 70)
    print("Comparison complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
