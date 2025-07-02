import json
import os


class PromptManager:
    def __init__(self, config_path):
        """Initialize the PromptManager with the configuration file path"""
        self.config = self.load_config(config_path)
        self.tasks = self.config.keys()
        self.models_by_task = self._build_model_index()

    def load_config(self, config_path):
        """Load the configuration file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_model_index(self):
        """Build an index of available models for each task"""
        models_by_task = {}
        for task in self.tasks:
            if "prompts" in self.config[task]:
                models_by_task[task] = {}
                for prompt_config in self.config[task]["prompts"]:
                    models = prompt_config[
                        4
                    ]  # Models are at index 4 in each prompt config
                    for model in models:
                        models_by_task[task][model] = prompt_config
        return models_by_task

    def get_available_tasks(self):
        """Return a list of available tasks"""
        return list(self.tasks)

    def get_available_models(self, task):
        """Return a list of available models for a given task"""
        if task not in self.models_by_task:
            return []
        return list(self.models_by_task[task].keys())

    def get_required_fields(self, task):
        """Return the list of required fields for a task"""
        if task not in self.config or "required" not in self.config[task]:
            return []
        return self.config[task]["required"]

    def get_prompt_config(self, task, model):
        """
        Get the prompt configuration for a specific task and model

        Returns:
            dict: A dictionary containing 'prompt', 'system', 'temp', 'p-value'
        """
        if task not in self.models_by_task:
            return {"prompt": "", "system": "", "temp": 0, "p-value": 0}

        # Try to find the exact model
        if model in self.models_by_task[task]:
            print("Found Model")
            prompt_config = self.models_by_task[task][model]
        # If not found, try to use the 'default' model if it exists
        elif "default" in self.models_by_task[task]:
            print("Using Default Model")
            prompt_config = self.models_by_task[task]["default"]
        else:
            print("Using First Model")
            # If no default and no exact match, use the first available config
            first_model = next(iter(self.models_by_task[task]))
            prompt_config = self.models_by_task[task][first_model]

        return {
            "prompt": prompt_config[0],
            "system": prompt_config[1],
            "temp": float(prompt_config[2]),
            "p-value": float(prompt_config[3]),
        }

    def format_prompt(self, task, model, **kwargs):
        """
        Format the prompt template with provided variables

        Args:
            task (str): The task name
            model (str): The model name
            **kwargs: Variables to insert into the prompt template

        Returns:
            dict: A dictionary with formatted 'prompt', 'system', 'temp', 'p-value'
        """
        config = self.get_prompt_config(task, model)

        # Check if all required fields are provided
        required_fields = self.get_required_fields(task)
        missing_fields = [field for field in required_fields if field not in kwargs]
        if missing_fields:
            raise ValueError(
                f"Missing required fields for {task}: {', '.join(missing_fields)}"
            )

        # Format the prompt with provided variables
        formatted_prompt = config["prompt"]
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            formatted_prompt = formatted_prompt.replace(placeholder, str(value))

        # Return the formatted config
        return {
            "prompt": formatted_prompt,
            "system": config["system"],
            "temp": config["temp"],
            "p-value": config["p-value"],
        }

    def update_prompt_config(self, task: str, old_model: str, new_prompt_data: dict):
        """
        Update an existing prompt configuration.
        new_prompt_data should contain 'prompt', 'system', 'temp', 'p-value', 'models', 'seed'.
        """
        if task not in self.config:
            raise ValueError(f"Task '{task}' not found.")

        prompts_list = self.config[task]["prompts"]
        found = False
        for i, prompt_config in enumerate(prompts_list):
            if old_model in prompt_config[4]:  # Check if old_model is in the models list
                # Update the existing entry
                prompts_list[i][0] = new_prompt_data["prompt"]
                prompts_list[i][1] = new_prompt_data["system"]
                prompts_list[i][2] = str(new_prompt_data["temp"])
                prompts_list[i][3] = str(new_prompt_data["p-value"])
                prompts_list[i][4] = new_prompt_data["models"]
                prompts_list[i][5] = str(new_prompt_data["seed"])
                found = True
                break
        
        if not found:
            raise ValueError(f"Model '{old_model}' not found for task '{task}'.")
        
        self._build_model_index() # Rebuild index after update

    def add_prompt_config(self, task: str, new_prompt_data: dict):
        """
        Add a new prompt configuration.
        new_prompt_data should contain 'prompt', 'system', 'temp', 'p-value', 'models', 'seed'.
        """
        if task not in self.config:
            self.config[task] = {"fields": ["prompt", "system", "temp", "p-value", "model", "seed"], "required": [], "prompts": []}
        
        # Ensure models is a list
        models = new_prompt_data.get("models", [])
        if not isinstance(models, list):
            models = [models]

        new_entry = [
            new_prompt_data["prompt"],
            new_prompt_data["system"],
            str(new_prompt_data["temp"]),
            str(new_prompt_data["p-value"]),
            models,
            str(new_prompt_data["seed"])
        ]
        self.config[task]["prompts"].append(new_entry)
        self._build_model_index() # Rebuild index after adding

    def delete_prompt_config(self, task: str, model: str):
        """
        Delete a prompt configuration by task and model name.
        If the model is the only one in its prompt entry, the entire entry is removed.
        Otherwise, only the model is removed from the list.
        """
        if task not in self.config:
            raise ValueError(f"Task '{task}' not found.")

        prompts_list = self.config[task]["prompts"]
        new_prompts_list = []
        deleted = False
        for prompt_config in prompts_list:
            models = prompt_config[4]
            if model in models:
                if len(models) == 1:
                    # Remove the entire prompt entry if it's the only model
                    deleted = True
                else:
                    # Remove only the model from the list
                    prompt_config[4].remove(model)
                    new_prompts_list.append(prompt_config)
                    deleted = True
            else:
                new_prompts_list.append(prompt_config)
        
        if not deleted:
            raise ValueError(f"Model '{model}' not found for task '{task}'.")
        
        self.config[task]["prompts"] = new_prompts_list
        self._build_model_index() # Rebuild index after deletion

    def save_config(self):
        """Save the current configuration back to the file"""
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)


# Example usage
if __name__ == "__main__":
    # Initialize the PromptManager with the config file
    manager = PromptManager("../../prompts.json")

    # Get available tasks
    print("Available tasks:", manager.get_available_tasks())

    # Get available models for a task
    task = "abstract"
    print(f"Available models for '{task}':", manager.get_available_models(task))

    # Get required fields for a task
    print(f"Required fields for '{task}':", manager.get_required_fields(task))

    # Get prompt configuration for a specific model
    model = "llama3:70b-instruct-q2_K"
    config = manager.get_prompt_config(task, model)
    print(
        f"Prompt config for {model}:",
        {
            "temp": config["temp"],
            "p-value": config["p-value"],
            "prompt": config["prompt"],  # Showing just the beginning for brevity
            "system": config["system"],
        },
    )

    # Format a prompt with variables
    formatted_config = manager.format_prompt(
        task,
        model,
        abstract="This is a sample abstract for testing purposes.",
        keywords="Metadata, Information Retrieval, Library Science",
    )
    print("Formatted prompt beginning:", formatted_config["prompt"][:100] + "...")
