import json
import os
import logging
from typing import Optional, List, Dict

from ..core.data_models import PromptConfigData


class PromptService:
    def __init__(self, config_path: str, logger: logging.Logger = None):
        """Initialize the PromptService with the configuration file path"""
        self.config_path = config_path
        self.logger = logger or logging.getLogger(__name__)
        self.config = self.load_config(config_path)
        self.tasks = self.config.keys()
        self.models_by_task = self._build_model_index()

    def load_config(self, config_path: str) -> Dict:
        """Load the configuration file"""
        if not os.path.exists(config_path):
            self.logger.error(f"Config file not found at {config_path}")
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_model_index(self) -> Dict:
        """Build an index of available models for each task"""
        models_by_task = {}
        for task in self.tasks:
            if "prompts" in self.config[task]:
                models_by_task[task] = {}
                for prompt_config in self.config[task]["prompts"]:
                    models = prompt_config[4]  # Models are at index 4
                    for model in models:
                        models_by_task[task][model] = prompt_config
        return models_by_task

    def get_available_tasks(self) -> List[str]:
        """Return a list of available tasks"""
        return list(self.tasks)

    def get_prompts_for_task(self, task_name: str) -> list:
        """
        Return the list of all prompt configurations for a given task.
        Each prompt configuration is a list: [prompt, system, temp, p-value, models, seed]
        """
        if task_name in self.config and "prompts" in self.config[task_name]:
            return self.config[task_name]["prompts"]
        return []

    def _try_smart_mode_model_matching(self, model: str, task: str) -> tuple[str, list] | None:
        """Try to match SmartProviderSelector models with available prompt configs - Claude Generated"""
        if not model or task not in self.models_by_task:
            return None

        available_models = list(self.models_by_task[task].keys())
        model_lower = model.lower()

        # Try different matching strategies for common SmartProviderSelector models
        for available_model in available_models:
            available_lower = available_model.lower()

            # 1. Exact case-insensitive match
            if available_lower == model_lower:
                self.logger.debug(f"Smart Mode: Exact match '{model}' to '{available_model}' for task '{task}'")
                return available_model, self.models_by_task[task][available_model]

            # 2. Model name contains available model name (e.g., "cogito:14b" contains "cogito")
            if len(available_model) > 3 and available_lower in model_lower:
                self.logger.debug(f"Smart Mode: Partial match '{model}' to '{available_model}' for task '{task}'")
                return available_model, self.models_by_task[task][available_model]

            # 3. Common model family matching
            if self._match_model_family(model_lower, available_lower):
                self.logger.debug(f"Smart Mode: Family match '{model}' to '{available_model}' for task '{task}'")
                return available_model, self.models_by_task[task][available_model]

        # 4. Enhanced fallback: for unknown models, use 'default' if available - Claude Generated
        if "default" in available_models:
            self.logger.debug(f"Smart Mode: Using 'default' config for unknown model '{model}' in task '{task}'")
            return "default", self.models_by_task[task]["default"]

        # 5. Last resort: use first available general purpose model - Claude Generated
        if available_models:
            # Prefer general purpose models over specialized ones
            general_models = [m for m in available_models if any(x in m.lower() for x in ['gemini', 'gpt', 'claude', 'meta', 'llama', 'deepseek'])]
            if general_models:
                fallback_model = general_models[0]
                self.logger.debug(f"Smart Mode: Using general purpose fallback '{fallback_model}' for '{model}' in task '{task}'")
                return fallback_model, self.models_by_task[task][fallback_model]

        return None

    def _match_model_family(self, model_lower: str, available_lower: str) -> bool:
        """Match model families like cogito, gemini, claude etc - Claude Generated"""
        model_families = {
            'cogito': ['cogito', 'llama', 'mistral'],
            'gemini': ['gemini', 'palm', 'google'],
            'claude': ['claude', 'anthropic'],
            'gpt': ['gpt', 'openai'],
            'llama': ['llama', 'cogito', 'mistral'],
            'mistral': ['mistral', 'cogito', 'llama'],
            'qwq': ['qwq', 'cogito', 'llama'],  # Common Ollama model
        }

        # Find model family for requested model
        for family, variants in model_families.items():
            if any(variant in model_lower for variant in variants):
                # Check if available model is from the same family
                if any(variant in available_lower for variant in variants):
                    return True
        return False

    def get_prompt_config(self, task: str, model: str) -> Optional[PromptConfigData]:
        """
        Get the prompt configuration for a specific task and model with intelligent fallback - Claude Generated

        Fallback hierarchy:
        1. Exact model match
        2. Fuzzy/family match via _try_smart_mode_model_matching()
        3. 'default' prompt if available
        4. First available prompt as safety net

        Returns a PromptConfigData object or None if task doesn't exist.
        """
        if task not in self.models_by_task:
            self.logger.warning(f"Task '{task}' not found in prompt configurations.")
            return None

        prompt_config_list = None
        matched_model = None

        # TIER 1: Try exact model match
        if model in self.models_by_task[task]:
            self.logger.info(f"✅ Exact model match: '{model}' for task '{task}'")
            prompt_config_list = self.models_by_task[task][model]
            matched_model = model

        # TIER 2: Try fuzzy/family matching - Claude Generated
        if not prompt_config_list:
            smart_match_result = self._try_smart_mode_model_matching(model, task)
            if smart_match_result:
                matched_model, prompt_config_list = smart_match_result
                self.logger.info(f"🔍 Fuzzy match: '{model}' → '{matched_model}' for task '{task}'")

        # TIER 3: Try 'default' fallback - Claude Generated
        if not prompt_config_list and "default" in self.models_by_task[task]:
            self.logger.info(f"⚙️ Using 'default' prompt for unknown model '{model}' in task '{task}'")
            prompt_config_list = self.models_by_task[task]["default"]
            matched_model = "default"

        # TIER 4: Use first available prompt as last resort - Claude Generated
        if not prompt_config_list and self.models_by_task[task]:
            available_models = list(self.models_by_task[task].keys())
            if available_models:
                first_model = available_models[0]
                self.logger.warning(f"⚠️ No 'default' found, using first available prompt '{first_model}' for model '{model}' in task '{task}'")
                prompt_config_list = self.models_by_task[task][first_model]
                matched_model = first_model

        # If still no config found, return None
        if not prompt_config_list:
            self.logger.error(f"❌ No prompt configuration available for task '{task}' (model: '{model}')")
            return None

        # Parse seed value
        seed_value = None
        if len(prompt_config_list) > 5 and prompt_config_list[5] is not None:
            try:
                seed_value = int(prompt_config_list[5])
            except (ValueError, TypeError):
                self.logger.warning(
                    f"Could not parse seed value '{prompt_config_list[5]}'. Using None."
                )

        # Return PromptConfigData with ACTUAL requested model (not matched_model)
        # This ensures LlmService uses the model the user selected
        return PromptConfigData(
            prompt=prompt_config_list[0],
            system=prompt_config_list[1],
            temp=float(prompt_config_list[2]),
            p_value=float(prompt_config_list[3]),
            models=[model],  # Use actual requested model, not matched_model!
            seed=seed_value,
        )

    def get_combination_prompt(self) -> Optional[str]:
        """
        Returns a specific prompt for combining chunk results.
        This could be loaded from a dedicated section in prompts.json.
        For now, it returns a default string.
        """
        # In a real scenario, you would load this from your prompts.json
        # For example: self.config.get("combination_prompts", {}).get("default_combination")
        return "Combine the following text chunks into a single, coherent response:\n\n{chunks}"
