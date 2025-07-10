from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class AbstractData:
    abstract: str
    keywords: str = ""

@dataclass
class KeywordResult:
    keyword: str
    gnd_id: str

@dataclass
class AnalysisResult:
    full_text: str
    matched_keywords: Dict[str, str] = field(default_factory=dict)
    gnd_systematic: Optional[str] = None
    # Add other relevant fields as needed

@dataclass
class PromptConfigData:
    prompt: str
    system: str
    temp: float
    p_value: float
    models: List[str]
    seed: Optional[int]

# Add other data models as needed for different parts of the application
