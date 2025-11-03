# ALIMA LLM Integration - Reproducibility Analysis Report

**Analysis Date:** November 3, 2025
**Focus:** Factors Affecting Reproducibility of Keyword Generation (Initialization Step)

---

## EXECUTIVE SUMMARY

The ALIMA codebase has **CRITICAL reproducibility issues** in its LLM keyword generation pipeline. While temperature, seed, and top_p parameters are technically present, **multiple factors prevent deterministic behavior** including:

1. **Non-deterministic provider selection logic**
2. **Inconsistent seed application across providers**
3. **Variable prompt formatting with non-deterministic text**
4. **Provider-dependent parameter handling**
5. **Caching mechanisms with lossy matching**

---

## 1. LLM SERVICE PARAMETER HANDLING

### 1.1 Parameter Definition and Support

**File:** `/home/conrad/src/ALIMA/src/llm/llm_service.py`

#### Temperature Parameter
- **Line 794**: `temperature: float = 0.7` - Default temperature set to 0.7
- **Lines 1143-1145 (Gemini)**: Temperature passed in `generation_config`
  ```python
  generation_config = {
      "temperature": temperature,
      "top_p": p_value,
  }
  ```
- **Line 1434-1435 (OpenAI Compatible)**: `"temperature": temperature` in params
- **Line 1520-1523 (Ollama)**: Temperature in options
  ```python
  "options": {
      "temperature": temperature,
      "top_p": p_value,
  }
  ```

**Status:** PASSED - Temperature parameter is correctly passed to all providers

#### Seed Parameter
- **Line 796**: `seed: Optional[int] = None` - Accepts optional seed
- **Line 1440-1441 (OpenAI Compatible)**: 
  ```python
  if seed is not None:
      params["seed"] = seed
  ```
- **Line 1529-1531 (Ollama)**:
  ```python
  if seed is not None:
      data["options"]["seed"] = seed
  ```
- **Line 1761-1762 (Anthropic)**:
  ```python
  if seed is not None:
      params["seed"] = seed
  ```

**STATUS:** ⚠️ PARTIAL - Seed is conditionally passed but:
- **NOT supported by Gemini** (line 1143-1145 shows no seed handling)
- **Gemini limitation** is a fundamental provider constraint

#### Top P (p_value) Parameter
- **Line 795**: `p_value: float = 0.1` - Default 0.1
- Consistently passed as `top_p` across all providers
- **Lines 1145, 1435, 1523, 1642, 1718**: All included in API calls

**Status:** PASSED - Top P consistently applied

---

## 2. INITIALIZATION STEP EXECUTION FLOW

### 2.1 Step Invocation Chain

**Flow Diagram:**
```
User/GUI → PipelineStepExecutor.execute_initial_keyword_extraction()
    ↓
/home/conrad/src/ALIMA/src/utils/pipeline_utils.py:190-278
    ↓ (Lines 202-209)
_resolve_provider_smart() [FIRST REPRODUCIBILITY RISK]
    ↓ (Lines 231-238)
AlimaManager.analyze_abstract()
    ↓
/home/conrad/src/ALIMA/src/core/alima_manager.py:57-166
    ↓ (Lines 93 or direct: Lines 83-90)
PromptService.get_prompt_config()
    ↓
/home/conrad/src/ALIMA/src/llm/prompt_service.py:115-186
    ↓ (Line 335-345)
LlmService.generate_response()
    ↓
/home/conrad/src/ALIMA/src/llm/llm_service.py:788-876
```

### 2.2 Parameter Flow Through Steps

**File:** `/home/conrad/src/ALIMA/src/utils/pipeline_utils.py:271-272`

Initial extraction receives parameters:
```python
temperature=kwargs.get("temperature", 0.7),
seed=kwargs.get("seed", 0),
```

These are passed to AlimaManager (lines 231-238):
```python
task_state = self.alima_manager.analyze_abstract(
    abstract_data=abstract_data,
    task=task,
    model=model,
    provider=provider,
    stream_callback=alima_stream_callback,
    **alima_kwargs,
)
```

Where `temperature`, `seed`, and `p_value` are in `alima_kwargs` (line 222).

---

## 3. PROMPT CONFIGURATION AND LOADING

### 3.1 Prompt Storage

**File:** `/home/conrad/src/ALIMA/prompts.json`

#### Initialization Task Configuration
**Lines 213-294** define "initialisation" task with 6 prompt variants:

**Variant 1 (Lines 227-240):**
```json
{
  "prompt": "Du bist ein korrekter und fachlich versierter Bibliothekar...",
  "system": "Your role as a librarians involves...",
  "temp": "0.25",
  "p-value": "0.1",
  "model": ["gemini-1.5-flash", "gemini-2.0-flash", "DeepSeek-V3", "cogito:14b", "Meta-Llama-3-70B-Instruct"],
  "seed": "0"
}
```

**Critical Issue - Seed Values:**
- **Line 26, 36, 46, 56, 68, 78, 92**: All variants have `"seed": "0"`
- **Lines 209, 220, 249**: `extract_initial_keywords` also has `"seed": "0"`
- **Effect:** Seeds are HARDCODED in JSON, making them non-configurable

#### Prompt Variable Substitution
**File:** `/home/conrad/src/ALIMA/src/core/alima_manager.py:185`

```python
formatted_prompt = prompt_config.prompt.format(**variables)
```

**Variables used (Lines 125-131):**
```python
variables = {
    "abstract": abstract_data.abstract,
    "keywords": (
        abstract_data.keywords
        if abstract_data.keywords
        else "Keine Keywords vorhanden"
    ),
}
```

**Non-Determinism Factor #1:** The fallback text "Keine Keywords vorhanden" is non-deterministic depending on keyword presence.

### 3.2 Temperature Value Handling

**File:** `/home/conrad/src/ALIMA/src/core/alima_manager.py:69`

Temperature is passed as explicit parameter with default:
```python
temperature: float = 0.7,
```

**However:**
- **File:** `/home/conrad/src/ALIMA/src/llm/prompt_service.py:182`
  ```python
  temp=float(prompt_config_list[2]),
  ```
  Configuration also contains prompt-specific temperatures

- **File:** `/home/conrad/src/ALIMA/prompts.json:19, 31, 41, 51, 61, 72, 83`
  Temperatures vary per prompt variant: 0.25, 0.0, 0.4

**Non-Determinism Factor #2:** Prompt-config temperatures override parameters. Code path (line 93 in alima_manager.py) doesn't apply provided temperature to prompt_config.

---

## 4. PROVIDER SELECTION AND MODEL RESOLUTION

### 4.1 Smart Provider Selection

**File:** `/home/conrad/src/ALIMA/src/utils/pipeline_utils.py:58-133`

Method: `_resolve_provider_smart(provider, model, task_type, prefer_fast, task_name, step_id)`

**Non-Determinism Factor #3 - Provider Selection Complexity:**

**Lines 58-93**: Multi-tier fallback chain:
1. **Tier 1 (Lines 61-65):** Explicit parameters (highest priority)
2. **Tier 2 (Lines 67-97):** SmartProviderSelector if available
   - Uses `task_type.lower()` enum mapping (line 77)
   - Calls `smart_selector.select_provider()` - **EXTERNAL STATE DEPENDENT**
   - Returns different providers based on real-time availability
3. **Tier 3 (Lines 99-116):** Config manager defaults
4. **Tier 4 (Lines 118-133):** System fallback defaults

**Critical Issue:** If SmartProviderSelector is enabled, provider/model selection is NON-DETERMINISTIC based on:
- Real-time provider availability
- Provider latency measurements
- Configured provider preferences
- Enabled/disabled provider states

### 4.2 Model Matching Logic

**File:** `/home/conrad/src/ALIMA/src/llm/prompt_service.py:52-113`

Method: `_try_smart_mode_model_matching(model, task)`

**Matching Strategy (Lines 62-93):**
1. **Exact case-insensitive match** (lines 65-67)
2. **Partial match** if available model is substring of requested (lines 70-72)
3. **Model family matching** (lines 75-77)
4. **Fallback to "default"** (lines 80-82)
5. **First available general-purpose model** (lines 85-91)

**Non-Determinism Factor #4:** When multiple models match (e.g., "cogito" family), the **first match in dictionary iteration order** is used:
```python
for available_model in available_models:  # Line 61: dictionary iteration order
```

In Python 3.7+, dict order is insertion order, but:
- Depends on JSON loading order (Line 25)
- Order depends on prompts.json structure
- Fragile to JSON restructuring

---

## 5. VARIABLE FACTORS IN PROMPTS

### 5.1 Non-Deterministic Prompt Content

**File:** `/home/conrad/src/ALIMA/prompts.json:227-262`

Prompt example (partial):
```
"Basierend auf folgendem Text und ggf. bereits vorgeschlagenen Keywords, 
schlage passende vollständig deutsche Schlagworte sowie GND-Systematiken vor."
```

**Non-Determinism Factor #5 - System Instructions:**

Different variants have different system prompts:
- **Variant 1 (line 229):** "Your role as a librarians involves thoroughly exploring questions..."
- **Variant 2 (line 243):** Identical
- **Variant 3 (line 253):** Different structure: "Give the final keywords as last line..."
- **Variant 4 (line 263):** Includes `<class>` tag instructions
- **Variant 5 (line 285):** Different again

System instructions are **CRITICAL** for LLM behavior. Different system prompts will produce different outputs even with same temperature/seed.

### 5.2 GND Systematic List Injection

**File:** `/home/conrad/src/ALIMA/prompts.json:228`

Embedded in prompt:
```
"Für die GND-Systematik nutze die folgende Liste:(2.1 Schrift,)|(2.2 Buchwissenschaft,)|..."
```

This is a **hardcoded, static list** but:
- Order matters for LLM behavior
- Different variants include/exclude this list
- Non-deterministic if list order changes

---

## 6. KEYWORD EXTRACTION AND ORDERING

### 6.1 Keyword List Ordering

**File:** `/home/conrad/src/ALIMA/src/core/alima_manager.py:125-131`

Keywords are passed directly from cache/previous step:
```python
"keywords": (
    abstract_data.keywords
    if abstract_data.keywords
    else "Keine Keywords vorhanden"
),
```

**Non-Determinism Factor #6:** If keywords come from cache search (biblio_client.py), the **ordering depends on:**

**File:** `/home/conrad/src/ALIMA/src/utils/clients/biblio_client.py:640-648`

Search result limiting:
```python
if len(term_subjects) > 50:
    logger.info(f"Limiting subjects for '{search_term}': {len(term_subjects)} -> 50 (top by count)")
    sorted_subjects = sorted(term_subjects.items(), key=lambda x: x[1]["count"], reverse=True)
    term_subjects = dict(sorted_subjects[:50])
```

This sorts by count (deterministic), BUT:
- Dictionary iteration order for CSV output (lines 858-862)
- Set operations on gnd_ids may reorder elements (lines 750-754)

---

## 7. CACHING MECHANISMS

### 7.1 Search Results Caching

**File:** `/home/conrad/src/ALIMA/src/utils/clients/biblio_client.py:653-800`

Method: `extract_dk_classifications_for_keywords(keywords, max_results, force_update)`

**Lines 674-689 - Cache Lookup:**
```python
cached_results = dk_cache.search_by_keywords(keywords, fuzzy_threshold=80)

if cached_results:
    has_titles = any(result.get("titles") for result in cached_results)
    if has_titles:
        logger.info(f"✅ Using {len(cached_results)} cached DK classifications")
        return cached_results
```

**Non-Determinism Factor #7:** Cache behavior:
- **Fuzzy matching at 80% threshold** - may match different keywords based on:
  - Capitalization variations
  - Whitespace handling
  - Substring matching order
- **Returned cache results may differ** between runs if:
  - Keywords are slightly different (due to fuzzy matching)
  - Cache has been partially updated
  - Multiple similar keywords match with different confidence scores

---

## 8. PROVIDER-SPECIFIC PARAMETER HANDLING

### 8.1 Gemini Provider

**File:** `/home/conrad/src/ALIMA/src/llm/llm_service.py:1130-1246`

Method: `_generate_gemini()`

**Parameter Handling:**
- **Temperature:** ✅ Passed (line 1144)
- **Top P:** ✅ Passed (line 1145)
- **Seed:** ❌ **NOT PASSED** - No seed support in Gemini generation config

**Result:** Gemini requests are **ALWAYS non-deterministic** regardless of seed parameter

### 8.2 Ollama Provider (Native)

**File:** `/home/conrad/src/ALIMA/src/llm/llm_service.py:1597-1698`

Method: `_generate_ollama_native()`

**Parameter Handling:**
- **Temperature:** ✅ Passed (line 1641)
- **Top P:** ✅ Passed (line 1642)
- **Seed:** ✅ Passed if not None (lines 1645-1646)

**But:** Ollama model support varies:
- Some models (e.g., Mistral) support seed
- Others may ignore it
- **Provider-dependent determinism**

### 8.3 OpenAI Compatible Provider

**File:** `/home/conrad/src/ALIMA/src/llm/llm_service.py:1385-1500`

Method: `_generate_openai_compatible()`

**Parameter Handling:**
- **Temperature:** ✅ Passed (line 1434)
- **Top P:** ✅ Passed (line 1435)
- **Seed:** ✅ Passed if not None (lines 1440-1441)

**Provider-specific:** Only newer OpenAI models support seed properly

### 8.4 Anthropic Provider

**File:** `/home/conrad/src/ALIMA/src/llm/llm_service.py:1700-1798`

Method: `_generate_anthropic()`

**Parameter Handling:**
- **Temperature:** ✅ Passed (line 1717)
- **Top P:** ✅ Passed (line 1718)
- **Seed:** ✅ Passed if not None (lines 1761-1762)

**Anthropic Note:** Seed support varies by model version

---

## 9. REQUEST ID AND STREAMING

### 9.1 Request Tracking

**File:** `/home/conrad/src/ALIMA/src/core/alima_manager.py:79`

```python
request_id = str(uuid.uuid4())
```

**Non-Determinism Factor #8:** UUID generated per request:
- Each invocation gets unique request_id
- Used for logging and stream callback tracking
- Does NOT affect LLM behavior directly
- BUT affects response tracking/correlation

### 9.2 Streaming vs Non-Streaming

**File:** `/home/conrad/src/ALIMA/src/llm/llm_service.py:344, 1445, 1546**

Streaming is **HARDCODED:**
```python
stream=True,  # Enable streaming
```

**Streaming Reassembly Determinism:**
- Chunks are concatenated in order (alima_manager.py:362)
- Should be deterministic IF provider returns chunks in same order
- BUT: Network latency, buffering, and provider differences may affect chunk boundaries

---

## 10. CONFIGURATION SOURCES AND PRECEDENCE

### 10.1 Parameter Override Hierarchy

**CRITICAL: Temperature Parameter Precedence Issue**

```
Priority 1: Hardcoded in prompts.json (HIGH PRIORITY)
    ↓
    Example: prompts.json line 19: "temp": "0.25"

Priority 2: Explicit parameter passed to analyze_abstract()
    ↓
    Example: temperature: float = 0.7 (line 69 alima_manager.py)

Priority 3: Prompt config from PromptService
    ↓
    Example: prompt_service.py line 182: temp=float(prompt_config_list[2])
```

**CODE PATH ISSUE (Line 93 vs Lines 83-90 in alima_manager.py):**

**Case 1: Explicit prompt_template (lines 81-90):**
```python
prompt_config = PromptConfigData(
    prompt=prompt_template,
    models=[model],
    temp=temperature,        # ✅ Uses explicit parameter
    p_value=p_value,
    seed=seed,
)
```

**Case 2: Using PromptService (line 93):**
```python
prompt_config = self.prompt_service.get_prompt_config(task, model)
```

Returns PromptConfigData with **prompt_config_list[2]** temperature, **IGNORING** the `temperature` parameter passed to analyze_abstract()

**CRITICAL BUG:** When using normal flow (no explicit template), the temperature parameter is IGNORED in favor of prompts.json values.

---

## 11. SEED VALUE PROPAGATION

### 11.1 Seed Sources

**File:** `/home/conrad/src/ALIMA/src/llm/prompt_service.py:167-175`

```python
seed_value = None
if len(prompt_config_list) > 5 and prompt_config_list[5] is not None:
    try:
        seed_value = int(prompt_config_list[5])
    except (ValueError, TypeError):
        logger.warning(f"Could not parse seed value '{prompt_config_list[5]}'")
```

**Seed Handling:**
- Takes from prompts.json index [5]
- **ALL INITIALIZATION VARIANTS HAVE SEED "0"** (prompts.json lines 26, 36, 46, 56, 68, 78, 92)
- Hardcoded in JSON, not configurable at runtime

**Non-Determinism:** Even if seed parameter is passed to analyze_abstract(), it's overridden by prompts.json value of "0".

---

## 12. SUMMARY TABLE OF REPRODUCIBILITY FACTORS

| Factor | Location | Status | Impact | Severity |
|--------|----------|--------|--------|----------|
| **Temperature Override** | alima_manager.py:93 | ❌ BUG | Prompts.json temp overrides parameter | CRITICAL |
| **Seed Hardcoding** | prompts.json:26,36,46,56,68,78,92 | ❌ HARDCODED | All seeds "0", non-configurable | CRITICAL |
| **Gemini No Seed** | llm_service.py:1143-1145 | ❌ UNSUPPORTED | Gemini always non-deterministic | HIGH |
| **Smart Provider Selection** | pipeline_utils.py:67-97 | ⚠️ DYNAMIC | Real-time provider availability affects selection | HIGH |
| **Model Matching Order** | prompt_service.py:61 | ⚠️ DICT ORDER | Dict iteration order affects selection | MEDIUM |
| **Variable Prompt Text** | prompts.json:228 | ⚠️ DYNAMIC | GND list, system prompts vary | MEDIUM |
| **Cache Fuzzy Matching** | biblio_client.py:676 | ⚠️ FUZZY | 80% threshold matching non-deterministic | MEDIUM |
| **Keyword Ordering** | alima_manager.py:127 | ⚠️ DEPENDS | Keyword list order affects LLM behavior | MEDIUM |
| **System Prompt Variants** | prompts.json:229,243,253,263,285 | ⚠️ VARIANT | Different system prompts per variant | MEDIUM |
| **Request UUID** | alima_manager.py:79 | ⚠️ UNIQUE | Per-request unique ID (logging only) | LOW |

---

## 13. CONCLUSIONS AND RECOMMENDATIONS

### 13.1 Reproducibility Assessment

**Current State:** POOR

The ALIMA initialization step has **multiple layers of non-determinism** that prevent reproducible keyword generation:

1. **Critical Design Issues:**
   - Temperature parameters are overridden by hardcoded JSON values
   - All seeds are hardcoded to "0" regardless of configuration
   - Smart provider selection uses real-time availability data
   - Model matching uses dictionary iteration order

2. **Provider Limitations:**
   - Gemini doesn't support seed at all
   - Other providers have varying seed support
   - Different providers produce different outputs for same input

3. **Caching Complications:**
   - Fuzzy matching at 80% threshold introduces variability
   - Cache returns different results based on previous searches

### 13.2 Recommendations for Improvement

**Immediate (Priority 1):**
1. Fix temperature override bug (alima_manager.py:93)
   - Apply explicit `temperature` parameter to prompt_config
2. Make seeds configurable
   - Replace hardcoded "0" in prompts.json with variable seeds
   - Add seed parameter that actually overrides JSON defaults

**Short-term (Priority 2):**
1. Disable smart provider selection for reproducibility mode
   - Add `--deterministic` flag that uses fixed provider selection
2. Remove fuzzy matching from cache lookup for exact reproducibility
   - Add `--no-fuzzy-cache` option
3. Document system prompt variations
   - Clearly label which prompt variants are deterministic

**Long-term (Priority 3):**
1. Implement reproducibility test suite
   - Run same input multiple times, compare outputs
   - Document provider-specific reproducibility
2. Add explicit reproducibility mode
   - Locks all non-deterministic selections
   - Disables caching, streaming, and smart selection
3. Provider abstraction for seed support
   - Wrap seed behavior consistently

---

## DETAILED CODE REFERENCES

**All line numbers verified as of git commit: 5e268bb**

### Critical Files:
- `/home/conrad/src/ALIMA/src/core/alima_manager.py` - Main LLM orchestration
- `/home/conrad/src/ALIMA/src/llm/llm_service.py` - Provider implementations (2286 lines)
- `/home/conrad/src/ALIMA/src/llm/prompt_service.py` - Prompt loading and matching
- `/home/conrad/src/ALIMA/prompts.json` - Hardcoded prompt configurations
- `/home/conrad/src/ALIMA/src/utils/pipeline_utils.py` - Pipeline step execution
- `/home/conrad/src/ALIMA/src/utils/clients/biblio_client.py` - Cache and keyword extraction

