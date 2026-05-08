"""YAML Prompt Service — loads prompt definitions from YAML files with named fields.

Replaces the positional-array format in prompts.json with a self-documenting
YAML structure.  Provides the same interface as PromptService so both can
 coexist or be merged.

Claude Generated
"""

import logging
import os
from typing import Dict, List, Optional

import yaml

from ..core.data_models import PromptConfigData

logger = logging.getLogger(__name__)


class YamlPromptService:
    """Load prompt definitions from YAML files.

    Each task is a mapping with named fields::

        initialisation:
          system: "Du bist ALIMA..."
          prompt: "Analysiere: {abstract}"
          temperature: 0.5
          top_p: 0.9
          models: [default]

    Multiple prompt variants per task are supported via ``prompts:`` list.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: Dict[str, Dict] = {}
        self.models_by_task: Dict[str, Dict] = {}
        if os.path.exists(config_path):
            self.config = self._load(config_path)
            self.models_by_task = self._build_model_index()
        else:
            logger.warning(f"YAML prompt file not found: {config_path}")

    def _load(self, path: str) -> Dict[str, Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected top-level mapping in {path}")
        return data

    def _build_model_index(self) -> Dict[str, Dict]:
        index: Dict[str, Dict] = {}
        for task, task_cfg in self.config.items():
            index[task] = {}
            prompts = task_cfg.get("prompts", [])
            # If prompts is a dict (single variant without list wrapper), wrap it
            if isinstance(prompts, dict):
                prompts = [prompts]
            for entry in prompts:
                if not isinstance(entry, dict):
                    continue
                models = entry.get("models", ["default"])
                for model in models:
                    index[task][model] = entry
        return index

    def get_prompt_config(self, task: str, model: str) -> Optional[PromptConfigData]:
        """Get prompt config with fallback hierarchy:

        1. Exact model match
        2. Fuzzy match (strip version tag)
        3. ``default`` fallback
        """
        if task not in self.models_by_task:
            return None

        entry = None

        # Tier 1: exact match
        if model in self.models_by_task[task]:
            logger.debug(f"YAML exact match: {model} for {task}")
            entry = self.models_by_task[task][model]

        # Tier 1.5: fuzzy match
        if entry is None and ":" in model:
            fuzzy = model.split(":")[0]
            if fuzzy in self.models_by_task[task]:
                logger.debug(f"YAML fuzzy match: {model} -> {fuzzy} for {task}")
                entry = self.models_by_task[task][fuzzy]

        # Tier 2: default fallback
        if entry is None and "default" in self.models_by_task[task]:
            logger.debug(f"YAML default fallback for {task}")
            entry = self.models_by_task[task]["default"]

        if entry is None:
            return None

        # Parse seed
        seed = entry.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except (ValueError, TypeError):
                seed = None

        return PromptConfigData(
            prompt=entry.get("prompt", ""),
            system=entry.get("system", ""),
            temp=float(entry.get("temperature", 0.5)),
            p_value=float(entry.get("top_p", 0.9)),
            models=[model],
            seed=seed,
            output_format=entry.get("output_format", "json"),
        )

    def merge_into(self, other_service) -> None:
        """Merge this YAML config into another PromptService instance.

        YAML entries override JSON entries for the same task+model.
        """
        for task, task_cfg in self.config.items():
            if task not in other_service.config:
                other_service.config[task] = {
                    "fields": task_cfg.get("fields", []),
                    "required": task_cfg.get("required", []),
                    "output_format": task_cfg.get("output_format", "json"),
                    "prompts": [],
                }
            prompts = task_cfg.get("prompts", [])
            if isinstance(prompts, dict):
                prompts = [prompts]
            # Convert named fields back to positional arrays for compatibility
            for entry in prompts:
                positional = [
                    entry.get("prompt", ""),
                    entry.get("system", ""),
                    str(entry.get("temperature", 0.5)),
                    str(entry.get("top_p", 0.9)),
                    entry.get("models", ["default"]),
                    entry.get("seed"),
                ]
                other_service.config[task]["prompts"].append(positional)

        # Rebuild index
        other_service.models_by_task = other_service._build_model_index()
        logger.info(f"Merged {len(self.config)} YAML prompt tasks into PromptService")
