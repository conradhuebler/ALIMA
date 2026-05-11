# ALIMA Provider Strategy Technical Specification

## Overview

This document specifies the technical changes required to simplify ALIMA's provider strategy while maintaining full functionality. The changes focus on removing obsolete complexity and streamlining the prompt configuration system.

## Current Implementation Analysis

### PromptService Class (src/llm/prompt_service.py)

#### Current Methods to be Removed

1. **`_try_smart_mode_model_matching`**
   ```python
   def _try_smart_mode_model_matching(self, model: str, task: str) -> tuple[str, list] | None:
       """Try to match SmartProviderSelector models with available prompt configs"""
       # Contains complex fuzzy matching logic with 5 strategies:
       # 1. Exact case-insensitive match
       # 2. Partial matching (e.g., "cogito:14b" contains "cogito")
       # 3. Model family matching via _match_model_family()
       # 4. 'default' fallback
       # 5. General purpose model selection
   ```

2. **`_match_model_family`**
   ```python
   def _match_model_family(self, model_lower: str, available_lower: str) -> bool:
       """Match model families like cogito, gemini, claude etc"""
       # Contains dictionary of model family mappings
       # Complex logic to match similar model types
   ```

#### Current `get_prompt_config` Method (Complex)

```python
def get_prompt_config(self, task: str, model: str) -> Optional[PromptConfigData]:
    """
    Get the prompt configuration for a specific task and model with intelligent fallback.

    Fallback hierarchy:
    1. Exact model match
    2. Fuzzy/family match via _try_smart_mode_model_matching()
    3. 'default' prompt if available
    4. First available prompt as safety net
    """
    # 4-tier fallback system with complex logic
    # Approximately 40 lines of complex conditional logic
```

## Simplified Implementation

### Updated PromptService Class

#### Removed Methods
- `_try_smart_mode_model_matching`
- `_match_model_family`

#### Simplified `get_prompt_config` Method

```python
def get_prompt_config(self, task: str, model: str) -> Optional[PromptConfigData]:
    """
    Get the prompt configuration for a specific task and model with simplified fallback.

    Fallback hierarchy:
    1. Exact model match
    2. 'default' prompt if available

    Returns a PromptConfigData object or None if no matching prompt is found.
    """
    if task not in self.models_by_task:
        self.logger.warning(f"Task '{task}' not found in prompt configurations.")
        return None

    prompt_config_list = None

    # TIER 1: Try exact model match
    if model in self.models_by_task[task]:
        self.logger.info(f"✅ Exact model match: '{model}' for task '{task}'")
        prompt_config_list = self.models_by_task[task][model]

    # TIER 2: Try 'default' fallback
    elif "default" in self.models_by_task[task]:
        self.logger.info(f"⚙️ Using 'default' prompt for model '{model}' in task '{task}'")
        prompt_config_list = self.models_by_task[task]["default"]

    # If still no config found, return None
    if not prompt_config_list:
        self.logger.error(f"❌ No prompt configuration available for task '{task}' (model: '{model}')")
        return None

    # Parse seed value and return PromptConfigData (unchanged logic)
    seed_value = None
    if len(prompt_config_list) > 5 and prompt_config_list[5] is not None:
        try:
            seed_value = int(prompt_config_list[5])
        except (ValueError, TypeError):
            self.logger.warning(
                f"Could not parse seed value '{prompt_config_list[5]}'. Using None."
            )

    return PromptConfigData(
        prompt=prompt_config_list[0],
        system=prompt_config_list[1],
        temp=float(prompt_config_list[2]),
        p_value=float(prompt_config_list[3]),
        models=[model],  # Use actual requested model
        seed=seed_value,
    )
```

## Configuration Impact

### No Changes Required
- `prompts.json` structure remains unchanged
- All existing prompt configurations continue to work
- Public API remains the same
- All integration points (CLI, GUI, WebApp) unaffected

### Breaking Changes (Acceptable per User Requirements)
- Fuzzy matching no longer available
- Complex fallback hierarchy removed
- Legacy configuration fields ignored
- Some edge case behaviors may change

## Files to be Modified

### Primary File
**`src/llm/prompt_service.py`**
- Remove `_try_smart_mode_model_matching` method (~35 lines)
- Remove `_match_model_family` method (~20 lines)
- Replace `get_prompt_config` method with simplified version (~25 lines)
- Update docstring to reflect new behavior
- Total: Remove ~55 lines, add ~25 lines

### Secondary Files (Potential Cleanup)
**`src/utils/config_models.py`** *(Optional)*
- Remove references to fuzzy matching in comments
- Clean up deprecated configuration field comments
- No functional changes required

## Integration Points Verification

### Core Components
- ✅ `src/core/alima_manager.py` - Uses `prompt_service.get_prompt_config()`
- ✅ `src/utils/pipeline_utils.py` - Uses prompt service for specialized tasks
- ✅ `src/ui/unified_input_widget.py` - Uses prompt service for OCR
- ✅ `src/ui/pipeline_config_dialog.py` - Uses prompt service for configuration

### External Interfaces
- ✅ CLI (`src/alima_cli.py`) - Uses prompt service through AlimaManager
- ✅ WebApp (`src/webapp/app.py`) - Uses prompt service directly
- ✅ Tests (`tests/test_cli.py`) - Mocks prompt service appropriately

## Testing Requirements

### Unit Tests
1. **Exact Match Test**: Verify that exact model names return correct prompts
2. **Default Fallback Test**: Confirm "default" prompts used when no exact match
3. **Missing Task Test**: Ensure None returned for non-existent tasks
4. **Missing Prompt Test**: Verify error handling when no prompts available
5. **Seed Value Parsing**: Test integer seed value parsing and error handling

### Integration Tests
1. **CLI Pipeline Command**: Test `alima_cli.py pipeline` functionality
2. **GUI Pipeline Execution**: Verify GUI-based analysis works correctly
3. **WebApp Analysis**: Confirm web-based analysis functions properly
4. **Batch Processing**: Ensure batch operations continue to work
5. **Prompt Editor**: Test that UI prompt editing still functions

### Edge Case Tests
1. **Empty Model Name**: Test behavior with empty string model names
2. **Case Sensitivity**: Verify exact matching is case-sensitive
3. **Special Characters**: Test model names with special characters
4. **Numeric Models**: Confirm numeric model names work correctly

## Performance Impact

### Expected Improvements
- **Reduced CPU Usage**: Eliminate string similarity calculations
- **Faster Lookups**: Single exact match instead of 4-tier fallback
- **Lower Memory**: Remove complex matching algorithm data structures
- **Simpler Debugging**: Clearer execution path for troubleshooting

### Benchmark Scenarios
1. **Prompt Lookup Time**: Measure time for `get_prompt_config()` calls
2. **Startup Performance**: Verify no performance regression in application startup
3. **Batch Processing**: Test performance with high-volume prompt lookups

## Rollback Plan

### If Issues Arise
1. **Immediate Fix**: Restore original `prompt_service.py` file
2. **Temporary Workaround**: Disable new simplified logic with feature flag
3. **User Communication**: Document breaking changes and migration steps
4. **Monitoring**: Implement logging to track prompt lookup failures

### Compatibility Fallback
```python
# Optional feature flag for rollback capability
USE_SIMPLIFIED_PROMPT_LOOKUP = True

def get_prompt_config(self, task: str, model: str) -> Optional[PromptConfigData]:
    if USE_SIMPLIFIED_PROMPT_LOOKUP:
        return self._get_prompt_config_simplified(task, model)
    else:
        return self._get_prompt_config_original(task, model)
```

## Success Criteria

### Functional Requirements
- ✅ All existing prompt configurations continue to work
- ✅ Exact model matching functions correctly
- ✅ Default fallback mechanism operates as expected
- ✅ Error handling provides clear, actionable messages
- ✅ All integration points (CLI, GUI, WebApp) function normally

### Performance Requirements
- ✅ Prompt lookup performance improves by 50%+
- ✅ Memory usage decreases due to removed algorithms
- ✅ Startup time remains unchanged or improves
- ✅ No performance regressions in any subsystem

### Maintainability Requirements
- ✅ Code complexity reduced by 60%+
- ✅ Documentation simplified and clarified
- ✅ Testing coverage maintained or improved
- ✅ Debugging ease significantly enhanced

## Implementation Timeline

### Phase 1: Core Implementation (1 day)
1. Modify `PromptService` class in `src/llm/prompt_service.py`
2. Remove obsolete methods
3. Implement simplified `get_prompt_config` method
4. Update documentation and comments

### Phase 2: Testing (2 days)
1. Execute unit tests for prompt service
2. Run integration tests for all interfaces
3. Perform edge case and performance testing
4. Validate backward compatibility where intended

### Phase 3: Documentation (0.5 days)
1. Update inline documentation
2. Create migration guide for users
3. Update README and configuration documentation
4. Prepare release notes

## Risk Assessment

### Low Risk Items
- ✅ Backward compatibility not required per user specification
- ✅ Existing `prompts.json` structure unchanged
- ✅ Public API remains identical
- ✅ Core functionality preserved

### Medium Risk Items
- ⚠️ Users with fuzzy model names may need configuration updates
- ⚠️ Some edge case behaviors may differ from previous implementation
- ⚠️ Performance testing required to validate improvements

### Mitigation Strategies
- Clear communication about breaking changes
- Comprehensive testing before deployment
- Easy rollback capability
- Updated documentation and examples

## Approval Requirements

### Code Review Checklist
- [ ] Remove `_try_smart_mode_model_matching` method
- [ ] Remove `_match_model_family` method
- [ ] Implement simplified `get_prompt_config` method
- [ ] Update all documentation and comments
- [ ] Pass all unit tests
- [ ] Pass all integration tests
- [ ] Performance benchmarks show improvement
- [ ] No regressions in existing functionality

### Deployment Checklist
- [ ] Backup current `prompt_service.py`
- [ ] Deploy updated implementation
- [ ] Execute smoke tests
- [ ] Monitor for issues
- [ ] Update documentation
- [ ] Communicate changes to users

## Conclusion

This technical specification provides a clear roadmap for simplifying ALIMA's provider strategy while maintaining full functionality. The changes focus on removing obsolete complexity and streamlining the core prompt configuration system. With careful implementation and testing, these changes will result in a cleaner, faster, and more maintainable system that better serves ALIMA's users.