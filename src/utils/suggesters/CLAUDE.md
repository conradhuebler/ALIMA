# Suggesters - Search Provider Integration System

## [Preserved Section - Permanent Documentation]

### Suggester Architecture
The `suggesters/` directory implements a plugin-like system for integrating different search providers and keyword suggestion sources:

**Core Components:**
- `BaseSuggester`: Abstract base class defining the suggester interface
- `MetaSuggester`: Orchestrates multiple suggesters and aggregates results
- `LobidSuggester`: German National Library (DNB) API integration
- `SWBSuggester`: Southwest German Library Network integration
- `CatalogSuggester`: Local library catalog integration via SOAP/HTTP
- `CatalogFallbackSuggester`: Fallback implementation for catalog access

### Suggester Pattern Implementation
**Plugin Architecture:**
- Suggesters inherit from `BaseSuggester` (Qt) but are now **wrapped** by the
  capability-based provider plugin system in `src/core/search/` (`@register_provider`).
- `SuggesterType` enum **retired** (June 29) — `MetaSuggester` enumerates the provider
  registry by id ("lobid"/"swb"/"catalog"; "all" = the three legacy GND sources).
- Mapping-first caching moved out of `MetaSuggester` into the reusable
  `CachingProvider` wrapper (`src/core/search/caching.py`); it also restores the F-4
  display-only `display_count`. See [`docs/search_provider_plugins.md`](../../../docs/search_provider_plugins.md).
- Configurable suggester selection (single provider or combined results); per-provider
  enable/disable via `SearchProviderConfig` + GUI selector.

**Key Design Features:**
- **Caching Strategy**: Each suggester implements local and remote caching
- **Error Handling**: Graceful degradation when providers are unavailable
- **Rate Limiting**: Built-in throttling to respect API limits
- **Result Aggregation**: Intelligent merging and deduplication of results
- **Async Operations**: Non-blocking requests with proper threading support

### Provider-Specific Details

**LobidSuggester:**
- Integrates with Lobid.org REST API
- Supports GND (Gemeinsame Normdatei) authority data
- Implements subject heading and classification lookup
- Provides DDC (Dewey Decimal Classification) support

**SWBSuggester:**
- Connects to Southwest German Library Network
- SOAP-based web service integration
- Bibliographic metadata enrichment
- Regional library catalog access

**CatalogSuggester:**
- Generic interface for local library catalogs
- Configurable SOAP/HTTP endpoints
- Token-based authentication support
- Customizable search and detail retrieval

### Data Models and Processing
- **Result Normalization**: Standardized output format across all providers
- **Metadata Extraction**: Consistent field mapping from diverse APIs
- **Quality Scoring**: Result ranking based on relevance and completeness
- **Cache Management**: Intelligent caching with TTL and invalidation strategies

## [Variable Section - Short-term Information]

### Recent Enhancements (Claude Generated)
1. **Enhanced Error Handling**: Improved exception management across all suggesters
2. **Caching Optimization**: Better cache hit rates and reduced API calls
3. **Result Processing**: Enhanced result normalization and deduplication
4. **Threading Safety**: Improved thread-safe operations for GUI integration
5. **✅ ADDED — FincSuggester (June 2026)**: VuFind-JSON /api/v1/search backend
   - `src/utils/suggesters/finc_suggester.py` — BaseSuggester wrapper around FincClient
   - Returns **records** (id, title, authors, subjects, formats, languages, series, web_url) per term, NOT aggregated GND keywords
   - `search_type` mapping: kw/freetext→AllFields, title→Title, subject→Subject, author→Author
   - Lazy-init in `tool_registry._init_suggesters` (gated by `finc_base_url`)
   - MCP tool `search_finc` registered in LIBRARY_TOOLS

### ✅ finc integration (resolved June 2026)
- **finc is a LOCAL catalog backend** (alongside Libero); SWB/Lobid stay EXTERNAL. Integrated opt-in, gated by `CatalogConfig`:
  - **Keyword step**: `finc_subject_harvest` (`finc_harvest_enabled`) — finc Subject search per keyword → record subjects reconciled against the LOCAL GND cache → GND-validated entries merged into the pool. `gnd_batch_search` stays `["swb","lobid"]`.
  - **DK step**: `PipelineStepExecutor.execute_dk_search` (`finc_dk_enabled`) — per-title `udk_raw_de105`/`rvk_facet` via `FincCatalogClient` (BiblioClient-compatible), Libero/SRU fallback.
  - `catalog_multi_search` unchanged (swb/lobid/catalog).
- `search_finc` MCP tool supports `search_type` (subject/title/author/kw) + `facets` (DK/RVK distribution). finc default `''` → all features off until configured.

### Current Configuration
- **Default Providers**: Lobid, SWB, and local catalog when available
- **Cache TTL**: 24 hours for most results, 1 hour for real-time data
- **Rate Limits**: Respectful API usage with configurable delays
- **Fallback Strategy**: Automatic failover to available providers

### Development Notes
- All new suggester functions marked as "Claude Generated"
- Comprehensive logging implemented for debugging and monitoring
- Type hints maintained throughout for better IDE support

## [Instructions Block - Operator-Defined Tasks]

### Future Tasks
1. **ADD - Additional Providers**: Integrate more library systems (OCLC, local catalogs)
2. **ADD - Advanced Caching**: Implement redis/memcached for distributed caching
3. **ADD - Result Analytics**: Add scoring and relevance ranking improvements
4. **ADD - Provider Health Monitoring**: Implement automatic provider health checks

### Vision
- Create a comprehensive ecosystem of library and academic search providers
- Establish ALIMA as the go-to tool for bibliographic metadata aggregation
- Implement intelligent result fusion from multiple authoritative sources
- Provide real-time provider status and automatic failover capabilities
- Support for emerging library standards and protocols (GraphQL, JSON-LD, etc.)