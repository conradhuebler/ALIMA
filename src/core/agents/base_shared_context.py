"""BaseSharedContext — generic shared state for workflow execution.

Provides only framework-level fields.  ALIMA-specific fields live in the
subclass ``SharedContext`` so non-ALIMA workflows aren't forced to use
``extra`` for everything.

Claude Generated
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BaseSharedContext:
    """Generic execution context for workflow steps."""

    # Step results
    step_results: Dict[str, Any] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)

    # LLM configuration
    provider: str = ""
    model: str = ""
    temperature: float = 0.5
    max_tokens: int = 4096
    # Operator budget for every LLM step, outranking the workflow YAML's
    # per-step value. None = the YAML decides (and ``max_tokens`` above is the
    # fallback for steps that name no budget). - Claude Generated
    max_tokens_override: Optional[int] = None
    seed: Optional[int] = None
    # None = leave the provider default; True/False = explicit thinking control.
    think: Optional[bool] = None
    verbose: bool = False

    # Execution tracking
    execution_history: List[Dict] = field(default_factory=list)

    # Prompt service reference
    prompt_service: Any = None

    # Generic escape hatch
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_step_result(self, step_name: str) -> Optional[Dict]:
        return self.step_results.get(step_name)

    def set_step_result(self, step_name: str, result: Dict, quality: float = None) -> None:
        self.step_results[step_name] = result
        if quality is not None:
            self.quality_scores[step_name] = quality

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_results": self.step_results,
            "quality_scores": self.quality_scores,
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_tokens_override": self.max_tokens_override,
            "seed": self.seed,
            "think": self.think,
            "execution_history": self.execution_history,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseSharedContext":
        ctx = cls(
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            temperature=data.get("temperature", 0.5),
            max_tokens=data.get("max_tokens", 4096),
            max_tokens_override=data.get("max_tokens_override"),
            seed=data.get("seed"),
            think=data.get("think"),
        )
        ctx.step_results = data.get("step_results", {})
        ctx.quality_scores = data.get("quality_scores", {})
        ctx.execution_history = data.get("execution_history", [])
        ctx.extra = data.get("extra", {})
        return ctx

    def get_summary(self) -> Dict[str, Any]:
        return {
            "steps_completed": list(self.step_results.keys()),
            "provider": f"{self.provider}/{self.model}",
            "execution_history_count": len(self.execution_history),
        }
