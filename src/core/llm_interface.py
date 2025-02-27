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
                "api_key": "OPENAI_API_KEY"
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
                    
            elif provider_info["class"] and api_key:
                # For providers that need class instantiation (OpenAI, Anthropic)
                client_class = getattr(module, provider_info["class"])
                self.clients[provider] = client_class(api_key=api_key)
                
            self.logger.info(f"Successfully initialized {provider}")
            
        except ImportError:
            self.logger.warning(f"Could not import {provider_info['module']} for {provider}")
        except Exception as e:
            self.logger.error(f"Error initializing {provider}: {str(e)}")

    # ... Rest der Klasse bleibt unverändert ...

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
                
            elif provider == "openai":
                return [model.id for model in self.clients[provider].models.list()]
                
            elif provider == "anthropic":
                final_list = []
                list = self.clients[provider].models.list()
                for model in list:
                    final_list.append(model.id)
                return final_list
                
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
                
        except Exception as e:
            self.logger.error(f"Error generating response from {provider}: {str(e)}")
            return f"Error: {str(e)}"
            
        return "Provider not supported"

    # Provider-spezifische Methoden anpassen:

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
