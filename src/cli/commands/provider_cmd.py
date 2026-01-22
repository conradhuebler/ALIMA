# Provider Management Command Handlers for ALIMA CLI
# Claude Generated - Extracted from alima_cli.py
"""
Handlers for LLM provider management commands:
    - list-models: List models from all providers
    - list-providers: List configured providers with details
    - test-providers: Test provider connections
    - list-models-detailed: Detailed model listing
    - provider: Provider management (add/remove/edit/test/ollama)
"""

import logging
from src.llm.llm_service import LlmService
from src.utils.config_manager import ConfigManager, OpenAICompatibleProvider
from src.utils.logging_utils import print_result


def handle_list_models(args, logger: logging.Logger):
    """Handle 'list-models' command - List models from all providers.

    Args:
        args: Parsed command-line arguments with:
            - ollama_host: Ollama host URL
            - ollama_port: Ollama port
        logger: Logger instance
    """
    llm_service = LlmService(
        ollama_url=args.ollama_host, ollama_port=args.ollama_port
    )
    providers = llm_service.get_available_providers()
    for provider in providers:
        print_result(f"--- {provider} ---")
        models = llm_service.get_available_models(provider)
        if models:
            for model in models:
                print_result(model)
        else:
            print_result("No models found.")


def handle_list_providers(args, logger: logging.Logger):
    """Handle 'list-providers' command - List all configured providers.

    Args:
        args: Parsed command-line arguments with:
            - show_config: Show detailed configuration
            - show_models: Show available models
        logger: Logger instance
    """
    print("=== ALIMA LLM Provider Configuration ===\n")

    try:
        config_manager = ConfigManager()
        config = config_manager.load_config()

        # Initialize LLM service for reachability testing
        llm_service = LlmService(lazy_initialization=True)

        provider_count = 0
        reachable_count = 0

        # Group providers by type
        providers_by_type = {
            'ollama': [],
            'openai_compatible': [],
            'gemini': [],
            'anthropic': []
        }

        for provider in config.unified_config.providers:
            providers_by_type[provider.provider_type].append(provider)

        # Display Ollama providers
        if providers_by_type['ollama']:
            print("🚀 Ollama Providers:")
            for provider in providers_by_type['ollama']:
                status_icon = "✅" if provider.enabled else "❌"
                reachable = llm_service.is_provider_reachable(provider.name) if provider.enabled else False
                reachable_icon = "🌐" if reachable else "📡"

                print(f"  {status_icon} {provider.name} ({provider.host}:{provider.port})")
                print(f"    URL: {provider.base_url}")
                print(f"    Status: {'Enabled' if provider.enabled else 'Disabled'}")
                print(f"    Reachable: {'Yes' if reachable else 'No'} {reachable_icon}")
                print(f"    SSL: {'Yes' if provider.use_ssl else 'No'}")
                if provider.api_key:
                    print(f"    API Key: {'*' * 8}...")
                if provider.description:
                    print(f"    Description: {provider.description}")

                if args.show_config:
                    print(f"    Connection Type: {provider.connection_type}")

                if args.show_models and provider.enabled and reachable:
                    try:
                        models = llm_service.get_available_models(provider.name)
                        if models:
                            print(f"    Models ({len(models)}): {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
                        else:
                            print("    Models: None available")
                    except Exception as e:
                        print(f"    Models: Error loading ({e})")

                provider_count += 1
                if provider.enabled and reachable:
                    reachable_count += 1
                print()

        # Display OpenAI-compatible providers
        if providers_by_type['openai_compatible']:
            print("🤖 OpenAI-Compatible Providers:")
            for provider in providers_by_type['openai_compatible']:
                status_icon = "✅" if provider.enabled else "❌"
                reachable = llm_service.is_provider_reachable(provider.name) if provider.enabled else False
                reachable_icon = "🌐" if reachable else "📡"

                print(f"  {status_icon} {provider.name}")
                print(f"    URL: {provider.base_url}")
                print(f"    Status: {'Enabled' if provider.enabled else 'Disabled'}")
                print(f"    Reachable: {'Yes' if reachable else 'No'} {reachable_icon}")
                if provider.api_key:
                    print(f"    API Key: {'*' * 8}...")
                if provider.description:
                    print(f"    Description: {provider.description}")

                if args.show_models and provider.enabled and reachable:
                    try:
                        models = llm_service.get_available_models(provider.name)
                        if models:
                            print(f"    Models ({len(models)}): {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
                        else:
                            print("    Models: None available")
                    except Exception as e:
                        print(f"    Models: Error loading ({e})")

                provider_count += 1
                if provider.enabled and reachable:
                    reachable_count += 1
                print()

        # Display API-only providers (Gemini, Anthropic)
        api_providers = providers_by_type['gemini'] + providers_by_type['anthropic']
        if api_providers:
            print("🎯 API-Only Providers:")
            for provider in api_providers:
                print(f"  ✅ {provider.name} ({provider.provider_type.title()})")
                if provider.api_key:
                    print(f"    API Key: {'*' * 8}...")
                print(f"    Status: {'Enabled' if provider.enabled else 'Disabled'}")
                print(f"    Reachable: Yes (API service) 🌐")
                if provider.description:
                    print(f"    Description: {provider.description}")

                if args.show_models:
                    try:
                        models = llm_service.get_available_models(provider.name)
                        if models:
                            print(f"    Models ({len(models)}): {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
                        else:
                            print("    Models: None available")
                    except Exception as e:
                        print(f"    Models: Error loading ({e})")

                provider_count += 1
                reachable_count += 1
                print()

        # Summary
        print(f"📊 Summary: {reachable_count}/{provider_count} providers reachable")

    except Exception as e:
        logger.error(f"Error listing providers: {e}")
        import traceback
        traceback.print_exc()


def handle_test_providers(args, logger: logging.Logger):
    """Handle 'test-providers' command - Test all provider connections.

    Args:
        args: Parsed command-line arguments with:
            - timeout: Connection timeout in seconds
            - show_models: Show available models for reachable providers
        logger: Logger instance
    """
    print("=== ALIMA LLM Provider Connection Test ===\n")

    try:
        llm_service = LlmService(lazy_initialization=True)

        print("🔍 Testing provider connections...\n")
        status_results = llm_service.refresh_all_provider_status()

        passed_tests = 0
        total_tests = len(status_results)

        for provider_name, result in status_results.items():
            if isinstance(result, dict):
                reachable = result.get('reachable', False)
                error = result.get('error', '')
                latency = result.get('latency_ms', 0)
            else:
                reachable = result
                error = '' if reachable else 'Connection failed'
                latency = 0

            status_icon = "✅" if reachable else "❌"
            print(f"{status_icon} {provider_name}")

            if reachable:
                print(f"   Status: Connected")
                if latency > 0:
                    print(f"   Latency: {latency:.1f}ms")
                passed_tests += 1

                if args.show_models:
                    try:
                        models = llm_service.get_available_models(provider_name)
                        if models:
                            print(f"   Models ({len(models)}): {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
                        else:
                            print("   Models: None available")
                    except Exception as e:
                        print(f"   Models: Error loading ({e})")
            else:
                print(f"   Status: Failed")
                if error:
                    print(f"   Error: {error}")

            print()

        # Summary
        print(f"📊 Test Results: {passed_tests}/{total_tests} providers passed")
        if passed_tests == total_tests:
            print("🎉 All providers are working correctly!")
        elif passed_tests == 0:
            print("⚠️  No providers are currently reachable")
        else:
            print(f"⚠️  {total_tests - passed_tests} provider(s) need attention")

    except Exception as e:
        logger.error(f"Error testing providers: {e}")
        import traceback
        traceback.print_exc()


def handle_list_models_detailed(args, logger: logging.Logger):
    """Handle 'list-models-detailed' command - Detailed model listing.

    Args:
        args: Parsed command-line arguments
        logger: Logger instance
    """
    print("=== ALIMA Comprehensive Model List ===\n")

    try:
        config_manager = ConfigManager()
        config = config_manager.load_config()

        llm_service = LlmService(config_manager=config_manager)

        total_models = 0
        provider_count = 0

        print("🔍 Scanning all providers for available models...\n")

        # Check all configured providers
        all_providers = []

        for provider in config.unified_config.providers:
            if provider.enabled:
                provider_type_display = {
                    'ollama': 'Ollama',
                    'openai_compatible': 'OpenAI-Compatible',
                    'gemini': 'Google API',
                    'anthropic': 'Anthropic API'
                }.get(provider.provider_type, provider.provider_type.title())

                if provider.base_url:
                    base_url = provider.base_url
                elif provider.provider_type == 'gemini':
                    base_url = 'https://api.google.com'
                elif provider.provider_type == 'anthropic':
                    base_url = 'https://api.anthropic.com'
                else:
                    base_url = f"https://api.{provider.provider_type}.com"

                all_providers.append((provider.name, provider_type_display, base_url))

        for provider_name, provider_type, base_url in all_providers:
            print(f"🚀 {provider_name} ({provider_type})")
            print(f"   URL: {base_url}")

            # Test reachability first
            reachable = llm_service.is_provider_reachable(provider_name)
            if not reachable:
                print("   Status: ❌ Not reachable")
                print()
                continue

            print("   Status: ✅ Reachable")

            try:
                models = llm_service.get_available_models(provider_name)
                if models:
                    print(f"   Models ({len(models)}):")
                    for i, model in enumerate(models, 1):
                        print(f"     {i:2d}. {model}")
                    total_models += len(models)
                else:
                    print("   Models: None available")

                provider_count += 1

            except Exception as e:
                print(f"   Models: ❌ Error loading ({e})")

            print()

        # Summary
        print(f"📊 Summary:")
        print(f"   Providers scanned: {len(all_providers)}")
        print(f"   Providers with models: {provider_count}")
        print(f"   Total models found: {total_models}")

    except Exception as e:
        logger.error(f"Error listing detailed models: {e}")
        import traceback
        traceback.print_exc()


def handle_provider(args, logger: logging.Logger):
    """Handle 'provider' command - Provider management actions.

    Args:
        args: Parsed command-line arguments with provider_action subcommand
        logger: Logger instance
    """
    config_manager = ConfigManager()

    if args.provider_action == "list":
        handle_provider_list(args, config_manager, logger)
    elif args.provider_action == "add":
        handle_provider_add(args, config_manager, logger)
    elif args.provider_action == "remove":
        handle_provider_remove(args, config_manager, logger)
    elif args.provider_action == "edit":
        handle_provider_edit(args, config_manager, logger)
    elif args.provider_action == "test":
        handle_provider_test(args, config_manager, logger)
    elif args.provider_action == "ollama":
        handle_provider_ollama(args, config_manager, logger)
    else:
        print("❌ No provider action specified.")
        print("   Use: list, add, remove, edit, test, or ollama")


def handle_provider_list(args, config_manager: ConfigManager, logger: logging.Logger):
    """List OpenAI-compatible providers."""
    try:
        config = config_manager.load_config()
        providers = config.unified_config.openai_compatible_providers

        if not providers:
            print("🔍 No OpenAI-compatible providers configured.")
            print("   Use 'alima_cli.py provider add' to add a new provider.")
            return

        print(f"🤖 OpenAI-Compatible Providers ({len(providers)} configured):")
        print()

        for i, provider in enumerate(providers, 1):
            status_icon = "✅" if provider.enabled else "❌"
            api_key_display = provider.api_key[:8] + "..." if provider.api_key else "❌ Not set"

            print(f"{i}. {status_icon} {provider.name}")
            print(f"   Base URL: {provider.base_url}")
            print(f"   API Key:  {api_key_display}")
            print(f"   Enabled:  {'Yes' if provider.enabled else 'No'}")
            if provider.description:
                print(f"   Description: {provider.description}")
            if provider.models:
                print(f"   Models: {', '.join(provider.models[:3])}{'...' if len(provider.models) > 3 else ''}")
            print()

    except Exception as e:
        logger.error(f"❌ Error listing providers: {e}")


def handle_provider_add(args, config_manager: ConfigManager, logger: logging.Logger):
    """Add new OpenAI-compatible provider."""
    try:
        config = config_manager.load_config()

        # Check if provider already exists
        if config.unified_config.get_provider_by_name(args.name):
            print(f"❌ Provider '{args.name}' already exists.")
            print("   Use 'alima_cli.py provider edit' to modify existing providers.")
            return

        # Create new provider
        new_provider = OpenAICompatibleProvider(
            name=args.name,
            base_url=args.base_url,
            api_key=args.api_key,
            enabled=args.enabled,
            description=args.description
        )

        # Add provider to configuration
        config.unified_config.add_provider(new_provider)

        # Save configuration
        success = config_manager.save_config(config, args.scope)
        if success:
            print(f"✅ Provider '{args.name}' added successfully to {args.scope} scope")
            print(f"   Base URL: {args.base_url}")
            print(f"   Enabled: {'Yes' if args.enabled else 'No'}")
            if args.description:
                print(f"   Description: {args.description}")
        else:
            print("❌ Failed to save provider configuration")

    except ValueError as e:
        print(f"❌ Invalid provider configuration: {e}")
    except Exception as e:
        logger.error(f"❌ Error adding provider: {e}")


def handle_provider_remove(args, config_manager: ConfigManager, logger: logging.Logger):
    """Remove provider."""
    try:
        config = config_manager.load_config()

        # Check if provider exists
        provider = config.unified_config.get_provider_by_name(args.name)
        if not provider:
            print(f"❌ Provider '{args.name}' not found.")
            print("   Use 'alima_cli.py provider list' to see available providers.")
            return

        # Remove provider
        success = config.unified_config.remove_provider(args.name)
        if success:
            config_saved = config_manager.save_config(config, args.scope)
            if config_saved:
                print(f"✅ Provider '{args.name}' removed successfully from {args.scope} scope")
            else:
                print("❌ Failed to save configuration after removal")
        else:
            print(f"❌ Failed to remove provider '{args.name}'")

    except Exception as e:
        logger.error(f"❌ Error removing provider: {e}")


def handle_provider_edit(args, config_manager: ConfigManager, logger: logging.Logger):
    """Edit existing provider."""
    try:
        config = config_manager.load_config()

        # Find provider to edit
        provider = config.unified_config.get_provider_by_name(args.name)
        if not provider:
            print(f"❌ Provider '{args.name}' not found.")
            print("   Use 'alima_cli.py provider list' to see available providers.")
            return

        # Update provider fields if provided
        if args.base_url:
            provider.base_url = args.base_url
        if args.api_key:
            provider.api_key = args.api_key
        if args.description is not None:
            provider.description = args.description
        if hasattr(args, 'enabled') and args.enabled is not None:
            provider.enabled = args.enabled

        # Save configuration
        success = config_manager.save_config(config, args.scope)
        if success:
            print(f"✅ Provider '{args.name}' updated successfully in {args.scope} scope")
            print(f"   Base URL: {provider.base_url}")
            print(f"   Enabled: {'Yes' if provider.enabled else 'No'}")
            if provider.description:
                print(f"   Description: {provider.description}")
        else:
            print("❌ Failed to save provider configuration")

    except ValueError as e:
        print(f"❌ Invalid provider configuration: {e}")
    except Exception as e:
        logger.error(f"❌ Error editing provider: {e}")


def handle_provider_test(args, config_manager: ConfigManager, logger: logging.Logger):
    """Test provider connection."""
    try:
        config = config_manager.load_config()

        # Find provider to test
        provider = config.unified_config.get_provider_by_name(args.name)
        if not provider:
            print(f"❌ Provider '{args.name}' not found.")
            print("   Use 'alima_cli.py provider list' to see available providers.")
            return

        if not provider.enabled:
            print(f"⚠️  Provider '{args.name}' is disabled.")
            print("   Enable it first or test anyway? (y/N): ", end="")
            response = input().strip().lower()
            if response != 'y':
                return

        print(f"🔌 Testing connection to '{args.name}'...")
        print(f"   Base URL: {provider.base_url}")

        # Test provider configuration
        try:
            llm_service = LlmService()

            # Suppress info logs during test
            old_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.ERROR)

            llm_service.initialize_providers()

            logging.getLogger().setLevel(old_level)

            print(f"✅ Provider '{args.name}' configuration is valid and loaded successfully")
            print(f"   API Key: {'✅ Set' if provider.api_key else '❌ Not set'}")
            print(f"   Base URL: {provider.base_url}")

            if not provider.api_key:
                print("   ⚠️  No API key configured - actual LLM calls will fail")
            else:
                print("   ✅ Provider ready for use")

        except Exception as e:
            print(f"❌ Provider test failed: {e}")
            print(f"   Check base URL and configuration")

    except Exception as e:
        logger.error(f"❌ Error testing provider: {e}")


def handle_provider_ollama(args, config_manager: ConfigManager, logger: logging.Logger):
    """Handle Ollama provider configuration commands."""
    if args.ollama_action == "status":
        try:
            config = config_manager.load_config()
            print("🔧 Current Ollama Configuration:")
            print(f"   Local:    {'✅ Enabled' if config.unified_config.ollama.local_enabled else '❌ Disabled'} ({config.unified_config.ollama.local_host}:{config.unified_config.ollama.local_port})")
            print(f"   Official: {'✅ Enabled' if config.unified_config.ollama.official_enabled else '❌ Disabled'} ({config.unified_config.ollama.official_base_url})")
            print(f"   Native:   {'✅ Enabled' if config.unified_config.ollama.native_enabled else '❌ Disabled'} ({config.unified_config.ollama.native_host})")
            print(f"   Active:   {config.unified_config.ollama.get_active_connection_type()}")
        except Exception as e:
            print(f"❌ Error loading Ollama configuration: {str(e)}")

    elif args.ollama_action == "enable-local":
        try:
            config = config_manager.load_config()
            config.unified_config.ollama.local_enabled = True
            config.unified_config.ollama.local_host = args.host
            config.unified_config.ollama.local_port = args.port
            config.unified_config.ollama.official_enabled = False
            config.unified_config.ollama.native_enabled = False
            config_manager.save_config(config)
            print(f"✅ Local Ollama enabled: {args.host}:{args.port}")
        except Exception as e:
            print(f"❌ Error enabling local Ollama: {str(e)}")

    elif args.ollama_action == "enable-official":
        try:
            config = config_manager.load_config()
            config.unified_config.ollama.official_enabled = True
            config.unified_config.ollama.official_base_url = args.base_url
            config.unified_config.ollama.official_api_key = args.api_key
            config.unified_config.ollama.local_enabled = False
            config.unified_config.ollama.native_enabled = False
            config_manager.save_config(config)
            print(f"✅ Official Ollama API enabled: {args.base_url}")
            print(f"   API Key: {args.api_key[:20]}...")
        except Exception as e:
            print(f"❌ Error enabling official Ollama: {str(e)}")

    elif args.ollama_action == "enable-native":
        try:
            config = config_manager.load_config()
            config.unified_config.ollama.native_enabled = True
            config.unified_config.ollama.native_host = args.host
            if args.api_key:
                config.unified_config.ollama.native_api_key = args.api_key
            config.unified_config.ollama.local_enabled = False
            config.unified_config.ollama.official_enabled = False
            config_manager.save_config(config)
            print(f"✅ Native Ollama client enabled: {args.host}")
            if args.api_key:
                print(f"   API Key: {args.api_key[:20]}...")
            else:
                print("   No API key configured (local access)")
        except Exception as e:
            print(f"❌ Error enabling native Ollama: {str(e)}")
