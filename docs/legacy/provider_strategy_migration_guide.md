# ALIMA Provider Strategy Migration Guide

## Overview

This guide explains the changes to ALIMA's provider strategy and how to migrate existing configurations to work with the simplified system. These changes focus on removing obsolete complexity while maintaining core functionality.

## What's Changing

### Removed Features
1. **Fuzzy Model Matching**: No longer automatically matches similar model names
2. **Complex Fallback Hierarchy**: 4-tier fallback reduced to 2-tier (exact match → default)
3. **Model Family Recognition**: Automatic grouping of similar models removed
4. **Legacy Configuration Support**: Old configuration fields no longer processed

### Maintained Features
1. **Exact Model Matching**: Direct string matching still works
2. **Default Fallback**: "default" prompt configurations continue to work
3. **Existing `prompts.json` Structure**: File format unchanged
4. **Public API**: All external interfaces remain identical

## Impact Assessment

### For Most Users
- ✅ **No Action Required**: Existing configurations will continue to work
- ✅ **Better Performance**: Faster prompt lookups without complex matching
- ✅ **Clearer Behavior**: Predictable prompt selection process

### For Advanced Users
- ⚠️ **Model Name Updates**: May need exact model names instead of fuzzy matches
- ⚠️ **Configuration Review**: Verify that all models have exact matches or defaults
- ⚠️ **Edge Case Changes**: Some rare behaviors may differ

## Migration Steps

### Step 1: Review Current Configuration

Check your `prompts.json` file for models that might rely on fuzzy matching:

```json
{
  "keywords": {
    "prompts": [
      [
        "prompt_content...",
        "system_content...",
        "0.7",
        "0.9",
        ["cogito:14b", "cogito:latest"]  // These models need exact matching
      ],
      [
        "default_prompt_content...",
        "default_system_content...",
        "0.7",
        "0.9",
        ["default"]  // Default fallback - this will continue to work
      ]
    ]
  }
}
```

### Step 2: Update Model Names (If Needed)

If you were relying on fuzzy matching, update to use exact model names:

**Before (might have worked with fuzzy matching):**
```json
{
  "model": "cogito"
}
```

**After (requires exact match):**
```json
{
  "model": "cogito:14b"
}
```

### Step 3: Ensure Default Prompts Exist

Verify that each task has a "default" prompt configuration:

```json
{
  "keywords": {
    "prompts": [
      // Your specific model prompts
      [
        "specific_prompt...",
        "specific_system...",
        "0.7",
        "0.9",
        ["cogito:14b", "gemma:7b"]
      ],
      // Default fallback (essential)
      [
        "default_prompt...",
        "default_system...",
        "0.7",
        "0.9",
        ["default"]
      ]
    ]
  }
}
```

### Step 4: Test Configuration

Run a sample analysis to verify the configuration works:

```bash
# Test CLI functionality
python3 src/alima_cli.py pipeline --input-text "Test abstract text"

# Test GUI functionality
python3 src/alima_gui.py
```

## Common Migration Scenarios

### Scenario 1: Generic Model References

**Issue**: Previously used generic model names like "gpt" or "claude"
**Solution**: Update to specific model versions like "gpt-4" or "claude-3-opus"

**Before:**
```json
["gpt", "claude"]
```

**After:**
```json
["gpt-4", "claude-3-opus-20240229"]
```

### Scenario 2: Family-Based Selection

**Issue**: Relied on model family recognition (e.g., "cogito" matching "cogito:14b")
**Solution**: Use exact model names or ensure "default" fallback exists

**Before:**
```json
["cogito"]  // Would match "cogito:14b", "cogito:32b", etc.
```

**After:**
```json
["cogito:14b"]  // Exact match
// OR
["default"]     // Fallback for any model
```

### Scenario 3: Mixed Provider Environments

**Issue**: Used automatic provider selection based on model capabilities
**Solution**: Configure explicit model priorities in `config.json`

**Before (implicit):**
```json
// Relying on fuzzy matching to find best available model
{
  "model": "best-available"
}
```

**After (explicit):**
```json
// In config.json task_preferences
"task_preferences": {
  "keywords": {
    "model_priority": [
      {"provider_name": "localhost", "model_name": "cogito:32b"},
      {"provider_name": "gemini", "model_name": "gemini-1.5-pro"}
    ]
  }
}
```

## Troubleshooting

### Error: "No prompt configuration available"

**Cause**: No exact model match and no "default" prompt found
**Solution**:
1. Verify model name spelling and case sensitivity
2. Add a "default" prompt configuration for the task
3. Use exact model names as they appear in your provider configuration

### Error: Unexpected prompt being used

**Cause**: Model name matches exactly but wasn't intended
**Solution**:
1. Review model names in `prompts.json`
2. Use more specific model identifiers
3. Ensure unique model names across providers

### Performance seems slower

**Cause**: This should not happen - simplified logic should be faster
**Solution**:
1. Verify you're using the updated version
2. Check for network or provider issues
3. Report as a bug if performance is genuinely worse

## Testing Your Migration

### Quick Verification Test
```bash
# Run a simple pipeline to verify prompt lookup works
python3 src/alima_cli.py pipeline --input-text "Migration test abstract"
```

### Comprehensive Test
1. **CLI Testing**:
   ```bash
   python3 src/alima_cli.py pipeline --input-text "Test"
   python3 src/alima_cli.py batch --batch-file test_sources.txt
   ```

2. **GUI Testing**:
   - Open ALIMA GUI
   - Run pipeline analysis
   - Verify results appear correctly

3. **WebApp Testing**:
   ```bash
   python3 src/webapp/app.py
   # Test via web interface
   ```

## Rollback Options

### If Issues Occur
1. **Immediate Fix**: Restore the original `src/llm/prompt_service.py` file
2. **Temporary Workaround**: Use a feature flag to switch between implementations
3. **Contact Support**: Report issues to development team

### Temporary Feature Flag Implementation
```python
# Add to config.json to temporarily revert to old behavior
{
  "system_config": {
    "use_simplified_prompt_lookup": false
  }
}
```

## Benefits of Migration

### Immediate Benefits
- ✅ **Faster Performance**: Eliminates complex matching algorithms
- ✅ **Clearer Debugging**: Exact matches are easier to troubleshoot
- ✅ **Better Predictability**: Know exactly which prompt will be used

### Long-term Benefits
- ✅ **Easier Maintenance**: Less complex code to maintain
- ✅ **Fewer Bugs**: Simpler logic means fewer edge cases
- ✅ **Better Documentation**: Clearer behavior is easier to document

## Questions and Support

For questions about this migration or issues encountered:

1. **Check Documentation**: Review `docs/provider_strategy_analysis.md`
2. **Review Code**: Examine `src/llm/prompt_service.py` for implementation details
3. **File Issues**: Report bugs or request help via GitHub issues
4. **Community Support**: Contact development team for assistance

## Conclusion

This migration simplifies ALIMA's provider strategy while maintaining core functionality. Most users will experience improved performance and clearer behavior without any configuration changes. Advanced users may need to update model names for exact matching, but the overall system becomes more predictable and easier to maintain.