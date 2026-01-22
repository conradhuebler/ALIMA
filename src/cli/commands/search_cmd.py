# Search Command Handlers for ALIMA CLI
# Claude Generated - Extracted from alima_cli.py
"""
Handlers for search-related commands:
    - search: Search for keywords using various suggesters
    - test-catalog: Test catalog search functionality
"""

import logging
from src.core.search_cli import SearchCLI
from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.utils.suggesters.meta_suggester import SuggesterType
from src.utils.logging_utils import print_result
from src.utils.config_manager import ConfigManager


def handle_search(args, logger: logging.Logger):
    """Handle 'search' command - Search for keywords using suggesters.

    Args:
        args: Parsed command-line arguments with:
            - search_terms: List of search terms
            - suggesters: List of suggester types to use
        logger: Logger instance
    """
    cache_manager = UnifiedKnowledgeManager()
    search_cli = SearchCLI(cache_manager)

    suggester_types = []
    for suggester in args.suggesters:
        try:
            suggester_types.append(SuggesterType[suggester.upper()])
        except KeyError:
            logger.warning(f"Unknown suggester: {suggester}")

    if not suggester_types:
        logger.error("No valid suggesters specified.")
        return

    results = search_cli.search(args.search_terms, suggester_types)

    for search_term, term_results in results.items():
        print_result(f"--- Results for: {search_term} ---")
        if cache_manager.gnd_keyword_exists(search_term):
            print_result("  (Results found in cache)")
        else:
            print_result("  (Results not found in cache)")

        for keyword, data in term_results.items():
            print_result(f"  - {keyword}:")
            print_result(f"    GND IDs: {data.get('gndid')}")
            print_result(f"    Count: {data.get('count')}")


def handle_test_catalog(args, logger: logging.Logger):
    """Handle 'test-catalog' command - Test catalog search functionality.

    Args:
        args: Parsed command-line arguments with:
            - search_terms: Search terms to test with
            - max_results: Maximum results per term
            - debug: Enable debug logging
            - catalog_token: Override catalog token
            - catalog_search_url: Override search URL
            - catalog_details_url: Override details URL
        logger: Logger instance
    """
    from src.utils.clients.biblio_client import BiblioClient

    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("biblio_extractor").setLevel(logging.DEBUG)

    print(f"🔍 Testing catalog search for terms: {', '.join(args.search_terms)}")
    print(f"📊 Max results per term: {args.max_results}")
    print("-" * 60)

    try:
        # Get catalog configuration
        config_manager = ConfigManager()
        catalog_config = config_manager.get_catalog_config()

        catalog_token = args.catalog_token or getattr(catalog_config, "catalog_token", "")
        catalog_search_url = args.catalog_search_url or getattr(catalog_config, "catalog_search_url", "")
        catalog_details_url = args.catalog_details_url or getattr(catalog_config, "catalog_details_url", "")

        if not catalog_token:
            logger.error("❌ No catalog token found. Configure in settings or use --catalog-token TOKEN")
            return

        if not catalog_search_url:
            logger.error("❌ No catalog search URL found. Configure in settings or use --catalog-search-url URL")
            return

        if not catalog_details_url:
            logger.error("❌ No catalog details URL found. Configure in settings or use --catalog-details-url URL")
            return

        print(f"🔑 Using catalog token: {catalog_token[:10]}..." if len(catalog_token) > 10 else catalog_token)
        if catalog_search_url:
            print(f"🌐 Search URL: {catalog_search_url}")
        if catalog_details_url:
            print(f"🌐 Details URL: {catalog_details_url}")
        print()

        # Initialize BiblioClient
        extractor = BiblioClient(
            token=catalog_token,
            debug=args.debug
        )

        if catalog_search_url:
            extractor.SEARCH_URL = catalog_search_url
        if catalog_details_url:
            extractor.DETAILS_URL = catalog_details_url

        # Test search_subjects method
        print("🚀 Starting catalog subject search...")
        results = extractor.search_subjects(
            search_terms=args.search_terms,
            max_results=args.max_results
        )

        print("=" * 60)
        print("📋 SEARCH RESULTS SUMMARY")
        print("=" * 60)

        total_subjects = 0
        for search_term, term_results in results.items():
            subject_count = len(term_results)
            total_subjects += subject_count

            print(f"\n🔸 Search term: '{search_term}'")
            print(f"   Found subjects: {subject_count}")

            if subject_count > 0:
                print("   📝 Subjects found:")
                for i, (subject, data) in enumerate(term_results.items(), 1):
                    print(f"      {i}. {subject}")
                    print(f"         Count: {data.get('count', 0)}")
                    dk_count = len(data.get('dk', set()))
                    if dk_count > 0:
                        print(f"         DK classifications: {dk_count}")
            else:
                print("   ❌ No subjects found")

        print(f"\n🎯 TOTAL: {total_subjects} subjects found across {len(args.search_terms)} search terms")

        if total_subjects == 0:
            print("\n⚠️  TROUBLESHOOTING:")
            print("   1. Check if catalog token is valid")
            print("   2. Verify catalog URLs are correct")
            print("   3. Try different search terms")
            print("   4. Run with --debug flag for detailed logs")

    except Exception as e:
        logger.error(f"❌ Catalog test failed: {str(e)}")
        if args.debug:
            raise
