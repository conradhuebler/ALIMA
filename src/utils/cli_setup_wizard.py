#!/usr/bin/env python3
"""
ALIMA CLI First-Start Setup Wizard
Interactive terminal-based setup for new users
Claude Generated
"""

import sys
import logging
from typing import Optional, List

from .config_manager import ConfigManager
from .setup_utils import (
    OllamaConnectionValidator, APIKeyValidator, GNDDatabaseDownloader,
    ConfigurationBuilder, PromptValidator, SetupResult
)


logger = logging.getLogger(__name__)


class CLISetupWizard:
    """Interactive CLI setup wizard - Claude Generated"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.provider_type = None
        self.provider_name = None
        self.base_url = None
        self.api_key = None
        self.available_models = []

    def run(self) -> bool:
        """Run the setup wizard - Claude Generated

        Returns:
            bool: True if setup completed successfully
        """
        try:
            self._print_welcome()
            self._setup_llm_provider()
            self._setup_gnd_database()
            self._print_summary()

            # Create and save configuration
            config = ConfigurationBuilder.create_initial_config(
                provider_type=self.provider_type,
                provider_name=self.provider_name,
                base_url=self.base_url,
                api_key=self.api_key,
                models=self.available_models
            )
            config.system_config.first_run_completed = True

            self.config_manager.save_config(config)
            logger.info("CLI setup wizard completed successfully")
            print("\n✅ Configuration saved! You can now run ALIMA.\n")
            return True

        except KeyboardInterrupt:
            print("\n\n❌ Setup cancelled by user")
            return False
        except Exception as e:
            logger.error(f"Setup error: {str(e)}")
            print(f"\n❌ Setup error: {str(e)}")
            return False

    def _print_welcome(self):
        """Print welcome message - Claude Generated"""
        print("\n" + "=" * 60)
        print("🚀 Welcome to ALIMA Setup Wizard")
        print("=" * 60)
        print("\nALIMA (Automatic Library Indexing and Metadata Analysis)")
        print("helps you analyze library materials and extract metadata with AI.")
        print("\nThis wizard will guide you through:")
        print("  1. Setting up an LLM provider (local or cloud)")
        print("  2. Optionally downloading GND authority data")
        print("  3. Reviewing your configuration")
        print("\n💡 You can always change these settings later in the settings menu.")
        print("=" * 60 + "\n")

    def _setup_llm_provider(self):
        """Interactive LLM provider setup - Claude Generated"""
        print("\n📌 Step 1: LLM Provider Configuration\n")
        print("Choose your LLM provider:\n")
        print("  1) Ollama (Local Server) - Recommended")
        print("  2) OpenAI-Compatible API")
        print("  3) Google Gemini API")
        print("  4) Anthropic Claude API")

        while True:
            choice = input("\nSelect provider (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                break
            print("❌ Invalid selection. Please choose 1-4.")

        if choice == '1':
            self._setup_ollama()
        elif choice == '2':
            self._setup_openai_compatible()
        elif choice == '3':
            self._setup_gemini()
        elif choice == '4':
            self._setup_anthropic()

    def _setup_ollama(self):
        """Setup Ollama provider - Claude Generated"""
        print("\n🔹 Ollama Configuration\n")

        host = input("Ollama host (default: localhost): ").strip() or "localhost"
        port_str = input("Ollama port (default: 11434): ").strip() or "11434"

        try:
            port = int(port_str)
        except ValueError:
            port = 11434
            print(f"Invalid port, using default: {port}")

        print("\n🧪 Testing connection...\n")
        result = OllamaConnectionValidator.test_native(host, port)

        if result.success:
            print(f"✅ {result.message}\n")
            print(f"Available models: {', '.join(result.data[:5])}")
            if len(result.data) > 5:
                print(f"   ... and {len(result.data) - 5} more")

            self.provider_type = "ollama"
            self.provider_name = f"Ollama ({host}:{port})"
            self.base_url = f"http://{host}:{port}"
            self.available_models = result.data
        else:
            print(f"❌ {result.message}\n")
            retry = input("Try different settings? (y/n): ").lower()
            if retry == 'y':
                self._setup_ollama()
            else:
                raise Exception("Ollama setup failed")

    def _setup_openai_compatible(self):
        """Setup OpenAI-compatible provider - Claude Generated"""
        print("\n🔷 OpenAI-Compatible API Configuration\n")

        base_url = input("API Base URL (e.g., http://localhost:8000/v1): ").strip()
        if not base_url:
            print("❌ Base URL is required")
            return self._setup_openai_compatible()

        api_key = input("API Key (or leave blank if not required): ").strip()

        print("\n🧪 Testing connection...\n")
        result = OllamaConnectionValidator.test_openai_compatible(base_url, api_key)

        if result.success:
            print(f"✅ {result.message}\n")
            print(f"Available models: {', '.join(result.data[:5])}")
            if len(result.data) > 5:
                print(f"   ... and {len(result.data) - 5} more")

            self.provider_type = "openai_compatible"
            self.provider_name = f"OpenAI-Compatible ({base_url})"
            self.base_url = base_url
            self.api_key = api_key
            self.available_models = result.data
        else:
            print(f"❌ {result.message}\n")
            retry = input("Try different settings? (y/n): ").lower()
            if retry == 'y':
                self._setup_openai_compatible()
            else:
                raise Exception("OpenAI-compatible setup failed")

    def _setup_gemini(self):
        """Setup Gemini provider - Claude Generated"""
        print("\n🟡 Google Gemini API Configuration\n")
        print("Get your API key from: https://aistudio.google.com/app/apikey\n")

        api_key = input("Paste your Gemini API key: ").strip()
        if not api_key:
            print("❌ API key is required")
            return self._setup_gemini()

        print("\n🧪 Testing API key...\n")
        result = APIKeyValidator.validate_gemini(api_key)

        if result.success:
            print(f"✅ {result.message}\n")
            self.provider_type = "gemini"
            self.provider_name = "Google Gemini"
            self.api_key = api_key
            self.available_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
        else:
            print(f"❌ {result.message}\n")
            retry = input("Try different API key? (y/n): ").lower()
            if retry == 'y':
                self._setup_gemini()
            else:
                raise Exception("Gemini setup failed")

    def _setup_anthropic(self):
        """Setup Anthropic provider - Claude Generated"""
        print("\n🔴 Anthropic Claude API Configuration\n")
        print("Get your API key from: https://console.anthropic.com/\n")

        api_key = input("Paste your Anthropic API key: ").strip()
        if not api_key:
            print("❌ API key is required")
            return self._setup_anthropic()

        print("\n🧪 Testing API key...\n")
        result = APIKeyValidator.validate_anthropic(api_key)

        if result.success:
            print(f"✅ {result.message}\n")
            self.provider_type = "anthropic"
            self.provider_name = "Anthropic Claude"
            self.api_key = api_key
            self.available_models = ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"]
        else:
            print(f"❌ {result.message}\n")
            retry = input("Try different API key? (y/n): ").lower()
            if retry == 'y':
                self._setup_anthropic()
            else:
                raise Exception("Anthropic setup failed")

    def _setup_gnd_database(self):
        """Interactive GND database setup - Claude Generated"""
        print("\n\n📌 Step 2: GND Authority Database (Optional)\n")
        print("The GND (Gemeinsame Normdatei) database contains German keywords.")
        print("Downloading improves keyword suggestions and search accuracy.")
        print("This is optional - you can use the Lobid API instead.\n")

        choice = input("Download GND database? (y/n): ").lower().strip()

        if choice == 'y':
            self._download_gnd_database()
        else:
            print("⏭️  Skipping GND database (will use Lobid API)")

    def _download_gnd_database(self):
        """Download GND database - Claude Generated"""
        print("\n🌐 Downloading GND database from DNB...")
        print("   This may take a few minutes...\n")

        def progress_callback(percent):
            # Print progress bar
            bar_length = 40
            filled = int(bar_length * percent // 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r📦 [{bar}] {percent}%", end='', flush=True)

        result = GNDDatabaseDownloader.download(progress_callback)

        if result.success:
            print(f"\n✅ GND database downloaded successfully")
            print(f"   Location: {result.data}\n")
        else:
            print(f"\n❌ Download failed: {result.message}\n")
            retry = input("Retry download? (y/n): ").lower()
            if retry == 'y':
                self._download_gnd_database()

    def _print_summary(self):
        """Print configuration summary - Claude Generated"""
        print("\n" + "=" * 60)
        print("✅ Configuration Summary")
        print("=" * 60)
        print(f"\nLLM Provider Configuration:")
        print(f"  Type: {self.provider_type}")
        print(f"  Name: {self.provider_name}")
        if self.base_url:
            print(f"  Base URL: {self.base_url}")
        if self.api_key:
            print(f"  API Key: {'*' * len(self.api_key[:-4]) + self.api_key[-4:]}")
        print(f"  Available Models: {len(self.available_models)}")

        print(f"\nDatabase:")
        print(f"  Type: SQLite")
        config = self.config_manager.load_config()
        print(f"  Path: {config.database_config.sqlite_path}")

        print("\n" + "=" * 60)
        print("✅ ALIMA is ready to use!")
        print("=" * 60 + "\n")


def run_cli_setup_wizard() -> bool:
    """Run the CLI setup wizard - Claude Generated

    Returns:
        bool: True if setup completed successfully
    """
    wizard = CLISetupWizard()
    return wizard.run()
