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



    def get_prompt_config(self, task: str, model: str) -> Optional[PromptConfigData]:
        """
        Get the prompt configuration for a specific task and model with simplified fallback.

        Fallback hierarchy:
        1. Exact model match
        2. 'default' prompt if available

        Returns a PromptConfigData object or None if no matching prompt is found.
        """
        if task not in self.models_by_task:
            self.logger.warning(f"Task '{task}' not found in prompt configurations.")
            return None

        prompt_config_list = None

        # TIER 1: Try exact model match
        if model in self.models_by_task[task]:
            self.logger.info(f"✅ Exact model match: '{model}' for task '{task}'")
            prompt_config_list = self.models_by_task[task][model]

        # TIER 1.5: Try fuzzy model match (strip version tags) - Claude Generated
        elif ':' in model:
            fuzzy_model = self._strip_version_tag(model)
            if fuzzy_model != model and fuzzy_model in self.models_by_task[task]:
                self.logger.info(f"🔍 Fuzzy model match: '{model}' → '{fuzzy_model}' for task '{task}'")
                prompt_config_list = self.models_by_task[task][fuzzy_model]

        # TIER 2: Try 'default' fallback
        if not prompt_config_list and "default" in self.models_by_task[task]:
            self.logger.info(f"⚙️ Using 'default' prompt for model '{model}' in task '{task}'")
            prompt_config_list = self.models_by_task[task]["default"]

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

        # Return PromptConfigData with ACTUAL requested model
        # This ensures LlmService uses the model the user selected
        return PromptConfigData(
            prompt=prompt_config_list[0],
            system=prompt_config_list[1],
            temp=float(prompt_config_list[2]),
            p_value=float(prompt_config_list[3]),
            models=[model],  # Use actual requested model
            seed=seed_value,
        )

    def _strip_version_tag(self, model: str) -> str:
        """
        Strip version tags from model names for fuzzy matching - Claude Generated

        Examples:
            cogito:14b → cogito
            ministral:3-14b → ministral
            qwen2.5:32b-instruct → qwen2.5
            default → default (unchanged)

        Returns:
            Base model name without version tag
        """
        # Don't strip special keywords
        if model in ["default"]:
            return model

        # Split on ':' and take first part
        base_name = model.split(':')[0]
        return base_name

    def get_combination_prompt(self) -> Optional[str]:
        """
        Returns a specific prompt for combining chunk results.
        This could be loaded from a dedicated section in prompts.json.
        For now, it returns a default string.
        """
        # In a real scenario, you would load this from your prompts.json
        # For example: self.config.get("combination_prompts", {}).get("default_combination")
        return "Combine the following text chunks into a single, coherent response:\n\n{chunks}"
