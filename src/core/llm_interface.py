from typing import Optional, Union, List, Dict
import os
from pathlib import Path
import importlib
import logging
import base64
import asyncio
import json

class LLMInterface:
    def __init__(self, 
                 providers: List[str] = None,
                 config_file: Path = Path.home() / '.llm_config.json',
                 api_keys: Dict[str, str] = None):
        """
        Initialize LLM interface with specified providers and API keys
        
        Args:
            providers: List of provider names to initialize
            config_file: Path to configuration file
            api_keys: Dictionary of provider API keys {provider_name: api_key}
        """
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        
        # Dictionary to store provider clients
        self.clients = {}
        
        # Dictionary of supported providers and their requirements
        self.supported_providers = {
            "gemini": {
                "module": "google.generativeai",
                "class": None,
                "api_key": "GEMINI_API_KEY"
            },
            "openai": {
                "module": "openai",
                "class": "OpenAI",
                "api_key": "OPENAI_API_KEY",
            },
            "comet": {
                "module": "openai",
                "class": "OpenAI",
                "api_key": "COMET_API_KEY",
                "base_url" : "https://api.cometapi.com/v1"
            },
            "anthropic": {
                "module": "anthropic",
                "class": "Anthropic",
                "api_key": "ANTHROPIC_API_KEY"
            },
            "ollama": {
                "module": "requests",
                "class": None,
                "api_key": None
            },
            # GitHub Copilot über Azure Inference
            "github": {
                "module": "azure.ai.inference", 
                "class": "ChatCompletionsClient",
                "api_key": "GITHUB_TOKEN"
            },
            # Azure OpenAI (auch über Azure Inference)
            "azure": {
                "module": "azure.ai.inference", 
                "class": "ChatCompletionsClient",
                "api_key": "AZURE_API_KEY"
            }
        }

        # Load existing config if available
        self.config = self.load_config()
        
        # Update config with new API keys if provided
        if api_keys:
            self.config.update(api_keys)
            self.save_config()
        
        # Initialize specified or all providers
        self.initialize_providers(providers)

    def load_config(self) -> Dict[str, str]:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}")
        return {}

    def save_config(self):
        """Save configuration to file"""
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
        Set API key for provider and save to config
        
        Args:
            provider: Provider name
            api_key: API key
        """
        if provider in self.supported_providers:
            self.config[provider] = api_key
            self.save_config()
            # Reinitialize provider with new key
            self._initialize_single_provider(provider)
        else:
            self.logger.warning(f"Unsupported provider: {provider}")

    def _initialize_single_provider(self, provider: str):
        """Initialize a single provider"""
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

            # Initialize client based on provider
            if provider == "gemini" and api_key:
                module.configure(api_key=api_key)
                self.clients[provider] = module
                
            elif provider == "comet" and api_key:
                self.clients[provider] = module.OpenAI(api_key=api_key, base_url="https://api.cometapi.com/v1")
                self.clients[provider] = module.ChatCompletion.create(api_key=api_key)
            
            elif provider == "ollama":
                # Test if Ollama is running
                import requests
                try:
                    response = requests.get("http://localhost:11434/api/tags")
                    if response.ok:
                        self.clients[provider] = requests
                except:
                    self.logger.warning("Ollama server not accessible")
                    return
            
            elif provider == "azure" and api_key:
                # Für Azure OpenAI via Azure Inference API
                from azure.ai.inference import ChatCompletionsClient
                from azure.core.credentials import AzureKeyCredential
                
                # Azure Endpoint
                azure_endpoint = self.config.get("azure_endpoint") or os.getenv("AZURE_ENDPOINT")
                if not azure_endpoint:
                    self.logger.warning("Missing Azure endpoint configuration")
                    return
                
                self.clients[provider] = ChatCompletionsClient(
                    endpoint=azure_endpoint,
                    credential=AzureKeyCredential(api_key)
                )
                
                # Store additional modules for message creation
                self.clients["azure_modules"] = importlib.import_module("azure.ai.inference.models")
                
                # Default Azure model
                self.config["azure_default_model"] = self.config.get("azure_default_model", "gpt-4")
            
            elif provider == "github" and api_key:
                # Import der benötigten Klassen für GitHub über Azure Inference
                from azure.ai.inference import ChatCompletionsClient
                from azure.core.credentials import AzureKeyCredential
                
                # Endpunkt für GitHub Copilot über Azure Inference
                github_endpoint = self.config.get("github_endpoint") or os.getenv("GITHUB_ENDPOINT") or "https://models.inference.ai.azure.com"
                
                self.clients[provider] = ChatCompletionsClient(
                    endpoint=github_endpoint,
                    credential=AzureKeyCredential(api_key)
                )
                
                # Store additional modules for message creation
                self.clients["github_modules"] = importlib.import_module("azure.ai.inference.models")
                
                # Default GitHub model
                self.config["github_default_model"] = self.config.get("github_default_model", "DeepSeek-V3")
                    
            elif provider_info["class"] and api_key:
                # Für Provider die Klasseninstanziierung benötigen (OpenAI, Anthropic)
                client_class = getattr(module, provider_info["class"])
                self.clients[provider] = client_class(api_key=api_key)
                
            self.logger.info(f"Successfully initialized {provider}")
            
        except ImportError as ie:
            self.logger.warning(f"Could not import {provider_info['module']} for {provider}: {str(ie)}")
        except Exception as e:
            self.logger.error(f"Error initializing {provider}: {str(e)}")

    def initialize_providers(self, providers: List[str] = None):
        """Initialize specified providers or all supported ones"""
        if providers is None:
            providers = list(self.supported_providers.keys())
            
        for provider in providers:
            if provider.lower() not in self.supported_providers:
                self.logger.warning(f"Unsupported provider: {provider}")
                continue
                
            self._initialize_single_provider(provider.lower())

    def get_available_providers(self) -> List[str]:
        """Get list of successfully initialized providers"""
        return list(self.clients.keys())

    def get_available_models(self, provider: str) -> List[str]:
        """Get available models for specified provider"""
        if provider not in self.clients:
            return []
            
        try:
            if provider == "gemini":
                return [model.name.split('/')[-1] for model in self.clients[provider].list_models()]               
            elif provider == "ollama":
                response = self.clients[provider].get("http://localhost:11434/api/tags")
                return [model["name"] for model in response.json()["models"]]
                
            elif provider == "openai" or provider == "comet":
                return [model.id for model in self.clients[provider].models.list()]

            elif provider == "anthropic":
                final_list = []
                list = self.clients[provider].models.list()
                for model in list:
                    final_list.append(model.id)
                return final_list
                
            elif provider == "azure":
                # Azure OpenAI Modelle über Azure Inference
                return ["gpt-4", "gpt-35-turbo", "gpt-4-vision", "gpt-4-turbo"]
                
            elif provider == "github":
                # GitHub Copilot Modelle über Azure Inference
                return ["DeepSeek-V3", 
                        "Meta-Llama-3-70B-Instruct", 
                        "Mistral-small", 
                        "Mistral-large", 
                        "DeepSeek-R1", 
              #          "o1", 
                        "Llama-3.2-90B-Vision-Instruct",
                        "Phi-4-multimodal-instruct",
                        "Phi-4-mini-instruct",
                        "Phi-4",
                        "o3-mini"]
                
        except Exception as e:
            self.logger.error(f"Error getting models for {provider}: {str(e)}")
            return []
            
        return []

    def process_image(self, image_input: Union[str, bytes]) -> bytes:
        """Convert image input to bytes"""
        if isinstance(image_input, str):
            with open(image_input, 'rb') as img_file:
                return img_file.read()
        return image_input

    def generate_response(self, 
                        provider: str,
                        model: str,
                        prompt: str,
                        temperature: float = 0.7,
                        seed: Optional[int] = None,
                        image: Optional[Union[str, bytes]] = None) -> str:
        """
        Generate response with specified parameters
        
        Args:
            provider: LLM provider name
            model: Model name
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 1.0)
            seed: Random seed for reproducibility
            image: Optional image input
        """
        if provider not in self.clients:
            return f"Provider {provider} not initialized"
            
        try:
            if provider == "gemini":
                return self._generate_gemini(model.strip(), prompt, temperature, seed, image)
            elif provider == "ollama": 
                return self._generate_ollama(model.strip(), prompt, temperature, seed, image)
            elif provider == "openai":
                return self._generate_openai(model.strip(), prompt, temperature, seed, image)
            elif provider == "anthropic":
                return self._generate_anthropic(model.strip(), prompt, temperature, seed, image)
            elif provider == "azure":
                return self._generate_azure(model.strip(), prompt, temperature, seed, image)
            elif provider == "github":
                return self._generate_github(model.strip(), prompt, temperature, seed, image)
                
        except Exception as e:
            self.logger.error(f"Error generating response from {provider}: {str(e)}")
            return f"Error: {str(e)}"
            
        return "Provider not supported"

    def _generate_gemini(self, model: str, prompt: str, temperature: float, seed: Optional[int], image: Optional[Union[str, bytes]] = None) -> str:
        try:
            generation_config = {
                "temperature": temperature,
            }
            if seed is not None:
                generation_config["seed"] = seed
                
            if image:
                model_instance = self.clients["gemini"].GenerativeModel(model)
                img_bytes = self.process_image(image)
                response = model_instance.generate_content(
                    [prompt, {"mime_type": "image/jpeg", "data": img_bytes}],
                    generation_config=generation_config
                )
            else:
                model_instance = self.clients["gemini"].GenerativeModel(model)
                response = model_instance.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            response.resolve()
            return response.text
        except Exception as e:
            self.logger.error(f"Gemini error: {str(e)}")
            return f"Error with Gemini: {str(e)}"

    def _generate_ollama(self, model: str, prompt: str, temperature: float, seed: Optional[int], image: Optional[Union[str, bytes]] = None) -> str:
        try:
            data = {
                "model": model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature
                }
            }
            if seed is not None:
                data["options"]["seed"] = seed
                
            if image:
                img_bytes = self.process_image(image)
                data["images"] = [base64.b64encode(img_bytes).decode()]
            
            response = self.clients["ollama"].post("http://localhost:11434/api/generate", json=data, stream=True)
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        full_response += json_response['response']
                        
            return full_response
        except Exception as e:
            self.logger.error(f"Ollama error: {str(e)}")
            return f"Error with Ollama: {str(e)}"

    def _generate_openai(self, model: str, prompt: str, temperature: float, seed: Optional[int], image: Optional[Union[str, bytes]] = None) -> str:
        try:
            params = {
                "model": model,
                "temperature": temperature,
            }
            if seed is not None:
                params["seed"] = seed
                
            if image:
                img_bytes = self.process_image(image)
                params["messages"] = [
                    {"role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"}}
                    ]}
                ]
            else:
                params["messages"] = [{"role": "user", "content": prompt}]
                
            response = self.clients["openai"].chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI error: {str(e)}")
            return f"Error with OpenAI: {str(e)}"

    def _generate_anthropic(self, model: str, prompt: str, temperature: float, seed: Optional[int], image: Optional[Union[str, bytes]] = None) -> str:
        try:
            params = {
                "model": model,
                "max_tokens": 1024,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
            if seed is not None:
                params["seed"] = seed
                
            message = self.clients["anthropic"].messages.create(**params)
            return message.content[0].text
        except Exception as e:
            self.logger.error(f"Anthropic error: {str(e)}")
            return f"Error with Anthropic: {str(e)}"

    def _generate_azure(self, model: str, prompt: str, temperature: float, seed: Optional[int], image: Optional[Union[str, bytes]] = None) -> str:
        """Generate response from Azure OpenAI via Azure Inference API"""
        try:
            # Zugriff auf die Klassen für Nachrichten
            UserMessage = self.clients["azure_modules"].UserMessage
            SystemMessage = self.clients["azure_modules"].SystemMessage
            
            messages = []
            
            # Optionale System-Nachricht für Kontext
            system_message = self.config.get("azure_system_message", "")
            if system_message:
                messages.append(SystemMessage(system_message))
            
            # Benutzereingabe hinzufügen
            if image and "vision" in model.lower():
                # Für Vision-Modelle mit Bild
                img_bytes = self.process_image(image)
                encoded_image = base64.b64encode(img_bytes).decode("ascii")
                
                # Import MultiModalContent und ImageContent
                ImageContent = self.clients["azure_modules"].ImageContent
                MultiModalContent = self.clients["azure_modules"].MultiModalContent
                
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
                messages.append(UserMessage(prompt))
            
            # Parameter für die Anfrage
            params = {
                "messages": messages,
                "model": model,
                "max_tokens": 1024,
                "temperature": temperature
            }
            
            # Seed hinzufügen, falls angegeben
            if seed is not None and hasattr(self.clients["azure_modules"], "CompletionParams"):
                CompletionParams = self.clients["azure_modules"].CompletionParams
                params["params"] = CompletionParams(seed=seed)
            
            response = self.clients["azure"].complete(**params)
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Azure OpenAI error: {str(e)}")
            return f"Error with Azure OpenAI: {str(e)}"

    def _generate_github(self, model: str, prompt: str, temperature: float, seed: Optional[int], image: Optional[Union[str, bytes]] = None) -> str:
        """Generate response from GitHub Copilot via Azure Inference"""
        try:
            # Zugriff auf die Klassen für Nachrichten
            UserMessage = self.clients["github_modules"].UserMessage
            SystemMessage = self.clients["github_modules"].SystemMessage
            
            messages = []
            
            # Optionale System-Nachricht für Kontext
            system_message = self.config.get("github_system_message", "")
            if system_message:
                messages.append(SystemMessage(system_message))
            
            # Füge Bild hinzu, falls vorhanden und Phi-3-Vision
            if image and model.lower() == "phi-3-vision":
                # Import MultiModalContent und ImageContent
                ImageContent = self.clients["github_modules"].ImageContent
                MultiModalContent = self.clients["github_modules"].MultiModalContent
                
                img_bytes = self.process_image(image)
                encoded_image = base64.b64encode(img_bytes).decode("ascii")
                
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
                if image:
                    return f"Model {model} does not support image processing. Use Phi-3-Vision instead."
                messages.append(UserMessage(prompt))
            
            # Parameter für die Anfrage
            params = {
                "messages": messages,
                "model": model,
                "max_tokens": 1000,
                "temperature": temperature
            }
            
            # Seed hinzufügen, falls angegeben
            if seed is not None and hasattr(self.clients["github_modules"], "CompletionParams"):
                CompletionParams = self.clients["github_modules"].CompletionParams
                params["params"] = CompletionParams(seed=seed)
            
            response = self.clients["github"].complete(**params)
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"GitHub Copilot error: {str(e)}")
            return f"Error with GitHub Copilot: {str(e)}"
