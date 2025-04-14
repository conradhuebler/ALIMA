from typing import Optional, Union, List, Dict, Any, Callable
import os
from pathlib import Path
import importlib
import logging
import base64
import json
import sys
import traceback


class LLMInterface:
    """
    A unified interface for interacting with various Large Language Models.
    
    This class provides a consistent API for different LLM providers like OpenAI,
    Anthropic, Google Gemini, and others. It handles initialization, configuration,
    and generation requests across all supported providers.
    """
    
    def __init__(self, 
                 providers: List[str] = None,
                 config_file: Path = Path.home() / '.llm_config.json',
                 api_keys: Dict[str, str] = None):
        """
        Initialize LLM interface with specified providers and API keys.
        
        Args:
            providers: List of provider names to initialize. If None, tries to initialize all supported providers.
            config_file: Path to configuration file for storing API keys and provider settings.
            api_keys: Dictionary of provider API keys {provider_name: api_key}.
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        
        # Dictionary to store provider clients
        self.clients = {}
        
        # Define provider configurations
        self._init_provider_configs()
        
        # Load existing config if available
        self.config = self._load_config()
        
        # Update config with new API keys if provided
        if api_keys:
            self.config.update(api_keys)
            self._save_config()
        
        # Initialize specified or all providers
        self.initialize_providers(providers)

    def _init_provider_configs(self):
        """Initialize the configuration for all supported providers."""
        self.supported_providers = {
            "gemini": {
                "module": "google.generativeai",
                "class": None,
                "api_key": "GEMINI_API_KEY",
                "initializer": self._init_gemini,
                "generator": self._generate_gemini,
            },
            "chatai": {
                "module": "openai",
                "class": "OpenAI",
                "api_key": "GWDG_API_KEY",
                "base_url": "https://chat-ai.academiccloud.de/v1",
                "initializer": self._init_openai_compatible,
                "generator": self._generate_openai_compatible,
                "params": {
                    "base_url": "https://chat-ai.academiccloud.de/v1"
                }
            },
            "openai": {
                "module": "openai",
                "class": "OpenAI",
                "api_key": "OPENAI_API_KEY",
                "initializer": self._init_openai_compatible,
                "generator": self._generate_openai_compatible,
            },
            "comet": {
                "module": "openai",
                "class": "OpenAI",
                "api_key": "COMET_API_KEY",
                "base_url": "https://api.cometapi.com/v1",
                "initializer": self._init_openai_compatible,
                "generator": self._generate_openai_compatible,
                "params": {
                    "base_url": "https://api.cometapi.com/v1"
                }
            },
            "anthropic": {
                "module": "anthropic",
                "class": "Anthropic",
                "api_key": "ANTHROPIC_API_KEY",
                "initializer": self._init_anthropic,
                "generator": self._generate_anthropic,
            },
            "ollama": {
                "module": "requests",
                "class": None,
                "api_key": None,
                "initializer": self._init_ollama,
                "generator": self._generate_ollama,
            },
            "github": {
                "module": "azure.ai.inference", 
                "class": "ChatCompletionsClient",
                "api_key": "GITHUB_TOKEN",
                "initializer": self._init_azure_inference,
                "generator": self._generate_azure_inference,
                "params": {
                    "endpoint": "github_endpoint", 
                    "default_model": "DeepSeek-V3",
                    "supported_models": [
                        "DeepSeek-V3",
                        "Meta-Llama-3-70B-Instruct",
                        "Mistral-small", 
                        "Mistral-large", 
                        "DeepSeek-R1",
                        "Llama-3.2-90B-Vision-Instruct",
                        "Phi-4-multimodal-instruct",
                        "Phi-4-mini-instruct",
                        "Phi-4",
                        "o3-mini"
                    ]
                }
            },
            "azure": {
                "module": "azure.ai.inference", 
                "class": "ChatCompletionsClient",
                "api_key": "AZURE_API_KEY",
                "initializer": self._init_azure_inference,
                "generator": self._generate_azure_inference,
                "params": {
                    "endpoint": "azure_endpoint",
                    "default_model": "gpt-4",
                    "supported_models": [
                        "gpt-4", 
                        "gpt-35-turbo", 
                        "gpt-4-vision", 
                        "gpt-4-turbo"
                    ]
                }
            }
        }

    def _load_config(self) -> Dict[str, str]:
        """
        Load configuration from file.
        
        Returns:
            Dict containing configuration values.
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}")
        return {}

    def _save_config(self):
        """Save configuration to file."""
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with pretty formatting
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            self.logger.error(f"Could not save config: {e}")

    def set_api_key(self, provider: str, api_key: str):
        """
        Set API key for provider and save to config.
        
        Args:
            provider: Provider name
            api_key: API key
        """
        if provider in self.supported_providers:
            self.config[provider] = api_key
            self._save_config()
            # Reinitialize provider with new key
            self._initialize_single_provider(provider)
        else:
            self.logger.warning(f"Unsupported provider: {provider}")

    def _initialize_single_provider(self, provider: str):
        """
        Initialize a single provider.
        
        Args:
            provider: The provider name to initialize.
        """
        provider_info = self.supported_providers[provider]
        
        try:
            # Try to import the required module
            module = importlib.import_module(provider_info["module"])
            
            # Handle API key if required
            api_key = None
            if provider_info["api_key"]:
                # Try config first, then environment
                api_key = self.config.get(provider) or os.getenv(provider_info["api_key"])
                if not api_key:
                    self.logger.warning(f"No API key found for {provider}")
                    return
            
            # Call the specific initializer for this provider
            if provider_info["initializer"]:
                provider_info["initializer"](provider, module, api_key, provider_info)
            else:
                self.logger.warning(f"No initializer defined for {provider}")
                
            self.logger.info(f"Successfully initialized {provider}")
            
        except ImportError as ie:
            self.logger.warning(f"Could not import {provider_info['module']} for {provider}: {str(ie)}")
        except Exception as e:
            self.logger.error(f"Error initializing {provider}: {str(e)}")
            self.logger.debug(traceback.format_exc())

    def initialize_providers(self, providers: List[str] = None):
        """
        Initialize specified providers or all supported ones.
        
        Args:
            providers: List of providers to initialize. If None, tries to initialize all providers.
        """
        if providers is None:
            providers = list(self.supported_providers.keys())
            
        for provider in providers:
            if provider.lower() not in self.supported_providers:
                self.logger.warning(f"Unsupported provider: {provider}")
                continue
                
            self._initialize_single_provider(provider.lower())

    def get_available_providers(self) -> List[str]:
        """
        Get list of successfully initialized providers.
        
        Returns:
            List of provider names.
        """
        return [name for name in self.clients.keys() 
                if not name.endswith("_modules")]  # Filter out module storage

    def get_available_models(self, provider: str) -> List[str]:
        """
        Get available models for specified provider.
        
        Args:
            provider: The provider name.
            
        Returns:
            List of model names.
        """
        if provider not in self.clients:
            return []
            
        try:
            if provider == "gemini":
                return [model.name.split('/')[-1] for model in self.clients[provider].list_models()]
                
            elif provider == "ollama":
                response = self.clients[provider].get("http://localhost:11434/api/tags")
                return [model["name"] for model in response.json()["models"]]
                
            elif provider in ["openai", "comet", "chatai"]:
                return [model.id for model in self.clients[provider].models.list()]

            elif provider == "anthropic":
                model_list = self.clients[provider].models.list()
                return [model.id for model in model_list]
                
            elif provider == "azure":
                return self.supported_providers[provider]["params"]["supported_models"]
                
            elif provider == "github":
                return self.supported_providers[provider]["params"]["supported_models"]
                
        except Exception as e:
            self.logger.error(f"Error getting models for {provider}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            
        return []

    def process_image(self, image_input: Union[str, bytes]) -> bytes:
        """
        Convert image input to bytes.
        
        Args:
            image_input: Path to image file or image bytes.
            
        Returns:
            Image content as bytes.
        """
        if isinstance(image_input, str):
            # If it's a path string, read the file
            with open(image_input, 'rb') as img_file:
                return img_file.read()
        # If it's already bytes, return as is
        return image_input

    def generate_response(self, 
                        provider: str,
                        model: str,
                        prompt: str,
                        temperature: float = 0.7,
                        seed: Optional[int] = None,
                        image: Optional[Union[str, bytes]] = None,
                        system: Optional[str] = "") -> str:
        """
        Generate response with specified parameters.
        
        Args:
            provider: LLM provider name
            model: Model name
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 1.0)
            seed: Random seed for reproducibility
            image: Optional image input (path or bytes)
            system: Optional system prompt
            
        Returns:
            Generated text response.
        """
        if provider not in self.clients:
            return f"Provider {provider} not initialized"
            
        try:
            # Fallback to provider's default system message if no system prompt is provided
            # but we have a default one configured
            if not system and self.config.get(f"{provider}_system_message"):
                system = self.config.get(f"{provider}_system_message")
            
            # Call the specific generator function for this provider
            provider_info = self.supported_providers[provider]
            if provider_info["generator"]:
                return provider_info["generator"](
                    model.strip(), prompt, temperature, seed, image, system
                )
            else:
                return f"No generation implementation for {provider}"
                
        except Exception as e:
            self.logger.error(f"Error generating response from {provider}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return f"Error: {str(e)}"
            
        return "Provider not supported"


    # Provider-specific initialization methods
    
    def _init_gemini(self, provider: str, module: Any, api_key: str, provider_info: Dict[str, Any]):
        """Initialize Gemini provider."""
        module.configure(api_key=api_key)
        self.clients[provider] = module

    def _init_openai_compatible(self, provider: str, module: Any, api_key: str, provider_info: Dict[str, Any]):
        """Initialize OpenAI-compatible providers (OpenAI, ChatAI, Comet)."""
        params = {"api_key": api_key}
        
        # Add base_url if specified
        if "base_url" in provider_info:
            params["base_url"] = provider_info["base_url"]
        elif provider_info.get("params", {}).get("base_url"):
            params["base_url"] = provider_info["params"]["base_url"]
            
        # Create client
        client_class = getattr(module, provider_info["class"])
        self.clients[provider] = client_class(**params)
        
        if "base_url" in params:
            self.logger.info(f"{provider} initialized with base URL: {params['base_url']}")

    def _init_anthropic(self, provider: str, module: Any, api_key: str, provider_info: Dict[str, Any]):
        """Initialize Anthropic provider."""
        client_class = getattr(module, provider_info["class"])
        self.clients[provider] = client_class(api_key=api_key)

    def _init_ollama(self, provider: str, module: Any, api_key: str, provider_info: Dict[str, Any]):
        """Initialize Ollama provider."""
        # Test if Ollama is running
        try:
            response = module.get("http://localhost:11434/api/tags")
            if response.ok:
                self.clients[provider] = module
        except:
            self.logger.warning("Ollama server not accessible")

    def _init_azure_inference(self, provider: str, module: Any, api_key: str, provider_info: Dict[str, Any]):
        """Initialize Azure Inference-based providers (Azure OpenAI, GitHub Copilot)."""
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential
        
        # Get endpoint from config or environment
        endpoint_key = provider_info["params"]["endpoint"]
        endpoint = self.config.get(endpoint_key) or os.getenv(endpoint_key.upper())
        
        if not endpoint and provider == "github":
            # Default endpoint for GitHub if not specified
            endpoint = "https://models.inference.ai.azure.com"
        
        if not endpoint:
            self.logger.warning(f"Missing endpoint configuration for {provider}")
            return
        
        # Create client
        self.clients[provider] = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
        
        # Store additional modules for message creation
        self.clients[f"{provider}_modules"] = importlib.import_module("azure.ai.inference.models")
        
        # Default model
        default_model = provider_info["params"]["default_model"]
        self.config[f"{provider}_default_model"] = self.config.get(f"{provider}_default_model", default_model)

    # Provider-specific generation methods
    
    def _generate_gemini(self, model: str, prompt: str, temperature: float, seed: Optional[int], 
                     image: Optional[Union[str, bytes]] = None, system: Optional[str] = "") -> str:
        """Generate response using Google Gemini."""
        try:
            generation_config = {
                "temperature": temperature,
            }
            
            if seed is not None:
                generation_config["seed"] = seed

            # Create model instance with system instruction if provided
            system_instruction = system if system else None
            model_instance = self.clients["gemini"].GenerativeModel(model, system_instruction=system_instruction)
            
            if image:
                img_bytes = self.process_image(image)
                response = model_instance.generate_content(
                    [prompt, {"mime_type": "image/jpeg", "data": img_bytes}],
                    generation_config=generation_config
                )
            else:
                response = model_instance.generate_content(
                    prompt,
                    generation_config=generation_config
                )

            response.resolve()
            return response.text
        except Exception as e:
            self.logger.error(f"Gemini error: {str(e)}")
            return f"Error with Gemini: {str(e)}"

    def _generate_openai_compatible(self, model: str, prompt: str, temperature: float, seed: Optional[int], 
                                image: Optional[Union[str, bytes]] = None, system: Optional[str] = "") -> str:
        """Generate response using OpenAI-compatible APIs (OpenAI, ChatAI, Comet)."""
        provider = [p for p, client in self.clients.items() 
                if isinstance(client, self.clients[p].__class__) and 
                p in ["openai", "chatai", "comet"]][0]
        
        try:
            # Create messages array
            messages = []
            
            # Add system message if provided
            if system:
                messages.append({"role": "system", "content": system})
            
            # Add user message with optional image
            if image:
                img_bytes = self.process_image(image)
                messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"}}
                    ]
                })
            else:
                messages.append({"role": "user", "content": prompt})
            
            # Set up parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            
            # Add seed if provided
            if seed is not None:
                params["seed"] = seed
                
            # Make API call
            response = self.clients[provider].chat.completions.create(**params)
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"{provider.capitalize()} error: {str(e)}")
            return f"Error with {provider.capitalize()}: {str(e)}"

    def _generate_ollama(self, model: str, prompt: str, temperature: float, seed: Optional[int], 
                        image: Optional[Union[str, bytes]] = None, system: Optional[str] = "") -> str:
        """Generate response using Ollama."""
        try:
            # Set up request data
            data = {
                "model": model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature
                }
            }
            
            # Add seed if provided
            if seed is not None:
                data["options"]["seed"] = seed
                
            # Add image if provided
            if image:
                img_bytes = self.process_image(image)
                data["images"] = [base64.b64encode(img_bytes).decode()]
            
            # Add system prompt if provided
            if system:
                data["system"] = system
                
            self.logger.info(f"Sending Ollama request with model: {model}")
            
            # Make API call with streaming
            response = self.clients["ollama"].post(
                "http://localhost:11434/api/generate", 
                json=data, 
                stream=True
            )
            
            # Process streaming response
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        chunk = json_response['response']
                        full_response += chunk
                        sys.stdout.write(chunk)
                        sys.stdout.flush()  # Ensure text is displayed immediately
            
            print()  # New line at the end of the stream
            return full_response
            
        except Exception as e:
            self.logger.error(f"Ollama error: {str(e)}")
            return f"Error with Ollama: {str(e)}"

    def _generate_anthropic(self, model: str, prompt: str, temperature: float, seed: Optional[int], 
                        image: Optional[Union[str, bytes]] = None, system: Optional[str] = "") -> str:
        """Generate response using Anthropic."""
        try:
            # Set up parameters
            params = {
                "model": model,
                "max_tokens": 1024,
                "temperature": temperature,
                "messages": []
            }
            
            # Add system message if provided
            if system:
                params["system"] = system
                
            # Add user message
            if image:
                img_bytes = self.process_image(image)
                
                # Check if Anthropic supports image in the current version
                try:
                    from anthropic import ImageContent, ContentBlock, TextContent
                    
                    # Create content blocks
                    content_blocks = [
                        TextContent(text=prompt, type="text"),
                        ImageContent(
                            source={
                                "type": "base64", 
                                "media_type": "image/jpeg", 
                                "data": base64.b64encode(img_bytes).decode()
                            },
                            type="image"
                        )
                    ]
                    
                    params["messages"].append({"role": "user", "content": content_blocks})
                except (ImportError, AttributeError):
                    # Fallback if the current Anthropic version doesn't support images
                    self.logger.warning("This version of Anthropic Python SDK might not support images. Sending text only.")
                    params["messages"].append({"role": "user", "content": prompt})
            else:
                params["messages"].append({"role": "user", "content": prompt})
            
            # Add seed if provided
            if seed is not None:
                params["seed"] = seed
                
            # Make API call
            message = self.clients["anthropic"].messages.create(**params)
            return message.content[0].text
            
        except Exception as e:
            self.logger.error(f"Anthropic error: {str(e)}")
            return f"Error with Anthropic: {str(e)}"

    def _generate_azure_inference(self, model: str, prompt: str, temperature: float, seed: Optional[int], 
                                image: Optional[Union[str, bytes]] = None, system: Optional[str] = "") -> str:
        """Generate response using Azure Inference-based providers (Azure OpenAI, GitHub Copilot)."""
        # Determine provider (azure or github)
        provider = [p for p in ["azure", "github"] if p in self.clients and f"{p}_modules" in self.clients][0]
        
        try:
            # Get modules for this provider
            modules = self.clients[f"{provider}_modules"]
            UserMessage = modules.UserMessage
            SystemMessage = modules.SystemMessage
            
            # Create messages array
            messages = []
            
            # Add system message if provided
            if system:
                messages.append(SystemMessage(system))
            
            # Vision model and image handling
            vision_models = ["gpt-4-vision", "phi-3-vision", "phi-4-multimodal-instruct", 
                            "llama-3.2-90b-vision-instruct"]
            supports_vision = any(vm.lower() in model.lower() for vm in vision_models)
            
            # Add user message with optional image
            if image and supports_vision:
                # For vision models with image
                img_bytes = self.process_image(image)
                encoded_image = base64.b64encode(img_bytes).decode("ascii")
                
                # Import necessary classes
                ImageContent = modules.ImageContent
                MultiModalContent = modules.MultiModalContent
                
                image_content = ImageContent(
                    data=encoded_image,
                    mime_type="image/jpeg"
                )
                
                multi_modal_content = MultiModalContent(
                    text=prompt, 
                    images=[image_content]
                )
                
                messages.append(UserMessage(content=multi_modal_content))
            else:
                if image and not supports_vision:
                    self.logger.warning(f"Model {model} does not support image processing.")
                messages.append(UserMessage(prompt))
            
            # Set up parameters
            params = {
                "messages": messages,
                "model": model,
                "max_tokens": 1024,
                "temperature": temperature
            }
            
            # Add seed if API supports it
            if seed is not None and hasattr(modules, "CompletionParams"):
                CompletionParams = modules.CompletionParams
                params["params"] = CompletionParams(seed=seed)
            
            # Make API call
            response = self.clients[provider].complete(**params)
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"{provider.capitalize()} error: {str(e)}")
            return f"Error with {provider.capitalize()}: {str(e)}"

    # Zusätzlich brauchen wir eine Methode, um Provider-spezifische System-Prompts zu konfigurieren
    def set_system_prompt(self, provider: str, system_prompt: str):
        """
        Set default system prompt for a specific provider.
        
        Args:
            provider: Provider name
            system_prompt: Default system prompt to use when none is provided
        """
        if provider in self.supported_providers:
            self.config[f"{provider}_system_message"] = system_prompt
            self._save_config()
            self.logger.info(f"Set default system prompt for {provider}")
        else:
            self.logger.warning(f"Unsupported provider: {provider}")