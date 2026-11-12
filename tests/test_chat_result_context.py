"""The chat gets the finished result: context summary, bridge and section tool. Claude Generated"""
import json
import types

from src.core.agents.shared_context import SharedContext
from src.core.data_models import KeywordAnalysisState, LlmKeywordAnalysis
from src.utils.result_summary import (
    chat_context_for_state, result_sections, result_summary_text,
)


def _state():
    final = LlmKeywordAnalysis(
        task_name="final", model_used="m", provider_used="p", prompt_template="", filled_prompt="",
        temperature=0.5, seed=None, response_full_text="",
        extracted_gnd_keywords=["Polymere (GND-ID: 1-1)", "Konformation (GND-ID: 2-2)"],
        keyword_chains=[{"chain": ["Polymere", "Konformation"], "reason": "Konformationsübergang des Polymers"}],
    )
    st = KeywordAnalysisState(original_abstract="Abstract " * 100, initial_keywords=[],
                              search_suggesters_used=[], working_title="PDMAEMA", final_llm_analysis=final)
    st.dk_classifications = ["DK 541.64", "DK 547"]
    st.classification_entries = [
        {"system": "DK", "code": "541.64", "display": "DK 541.64", "rank": "core"},
        {"system": "DK", "code": "547", "display": "DK 547", "rank": "additional"},
    ]
    st.core_keywords = ["Polymere"]
    st.shelf = {"placement": {"notation": "541.2/.8", "label": "Chemische Bindung", "location": "OG",
                              "selected_reason": "Kernnotation 541.64 fällt in den Bereich"}}
    st.geo_sachdb = {"terms": [], "notations": [
        {"system": "DK", "notation": "541.64", "status": "local",
         "local": [{"label": "Chemie-chem. Eigenschaften von Hochpolymere", "ppn": "1292236744"}]}],
        "not_found": ["Kraftfeld"]}
    st.source_usage = [{"source": "swb", "text": "SWB (BSZ) · 19 Anfragen"}]
    return st


def test_sections_carry_what_the_old_bridge_dropped():
    sec = result_sections(_state())
    assert "Konformationsübergang des Polymers" in sec["ketten"]          # chain reason
    assert "Kernnotation 541.64" in sec["aufstellung"]                    # shelf reason
    assert "SWB (BSZ)" in sec["quellen"]
    assert "Hochpolymere" in sec["klassifikation"]                        # designation from the result
    assert "DK 547 [additional]: (keine Bezeichnung im Ergebnis)" in sec["klassifikation"]


def test_summary_is_lean_and_names_the_other_sections():
    text = result_summary_text(result_sections(_state()))
    assert "Konformationsübergang" in text and "Kernnotation 541.64" in text
    assert "Weitere Abschnitte (Werkzeug get_result_section)" in text
    assert "5550" not in text                                             # WinIBW stays behind the tool


def test_context_has_title_abstract_start_and_result():
    ctx = chat_context_for_state(_state())
    assert ctx.startswith("Titel: PDMAEMA")
    assert "Ergebnis der Erschließung" in ctx
    assert ctx.count("Abstract ") < 100                                   # abstract cut at 500 characters


def test_bridge_restores_sections_and_dk_with_rank():
    ctx = SharedContext.from_keyword_analysis_state(_state())
    assert "aufstellung" in ctx.extra["result_sections"]
    assert ctx.dk_classifications[0]["rank"] == "core"                    # was: {"code": ...} only


def test_tool_returns_a_section_and_names_the_available_ones():
    from src.ui.chat_tools.generic import GetResultSectionTool

    session = types.SimpleNamespace(last_shared_context=SharedContext.from_keyword_analysis_state(_state()))
    tool = GetResultSectionTool()
    assert tool.available_for(session)
    assert "Kernnotation" in json.loads(tool.execute(session, section="aufstellung"))["text"]
    miss = json.loads(tool.execute(session, section="bericht"))
    assert "aufstellung" in miss["available_sections"]
    assert not tool.available_for(types.SimpleNamespace(last_shared_context=None))


def test_system_prompt_binds_the_answer_to_the_result():
    from src.core.chat_prompts import build_system_prompt

    for compact in (False, True):
        prompt = build_system_prompt(mode="verschlagwortung", compact=compact)
        assert "AUS DIESEM ERGEBNIS" in prompt and "rate sie nie aus der Nummer" in prompt
