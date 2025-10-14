"""
Pipeline Default Configuration Values
Claude Generated - Central definition of pipeline configuration defaults
"""

# DK Pipeline Search Configuration
DEFAULT_DK_MAX_RESULTS = 40
"""Maximum number of search results to retrieve per keyword from catalog (default: 20)"""

DEFAULT_DK_FREQUENCY_THRESHOLD = 1
"""Minimum occurrence count for DK classifications to be included in LLM analysis (default: 10).
Only classifications that appear >= this many times in the catalog will be passed to the LLM."""
