"""Vorlagen für OpenAI-kompatible Anbieter - Claude Generated.

Nur Daten für die Eingabe (Anbieter-Dialog, Einrichtungsassistent): Name,
Basis-URL, ob ein Schlüssel nötig ist. Die Adressen stammen aus der
Dokumentation der Anbieter und werden von ALIMA **nicht** automatisch gegen die
Dienste geprüft; „Verbindung testen“ im Dialog prüft sie mit dem eigenen
Schlüssel. Die Laufzeit liest ausschließlich die gespeicherte Konfiguration,
diese Liste wirkt nur beim Ausfüllen des Formulars.

Bewusst nicht enthalten: die GWDG und weitere deutsche Hochschul-Dienste
(institutionsspezifisch, dafür gibt es Institutions-Presets, siehe
``preset_loader``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class OpenAIPreset:
    id: str  # Vorschlag für den Anbieternamen
    label: str
    base_url: str
    key_required: bool = True
    note: str = ""


OPENAI_COMPATIBLE_PRESETS: Tuple[OpenAIPreset, ...] = (
    OpenAIPreset("openai", "OpenAI (ChatGPT-Modelle)", "https://api.openai.com/v1"),
    OpenAIPreset("mistral", "Mistral AI", "https://api.mistral.ai/v1"),
    OpenAIPreset("nvidia", "NVIDIA NIM (build.nvidia.com)",
                 "https://integrate.api.nvidia.com/v1"),
    OpenAIPreset("openrouter", "OpenRouter (viele Anbieter, ein Schlüssel)",
                 "https://openrouter.ai/api/v1"),
    OpenAIPreset("groq", "Groq", "https://api.groq.com/openai/v1"),
    OpenAIPreset("together", "Together AI", "https://api.together.xyz/v1"),
    OpenAIPreset("deepseek", "DeepSeek", "https://api.deepseek.com/v1"),
    OpenAIPreset("xai", "xAI (Grok)", "https://api.x.ai/v1"),
    OpenAIPreset("lmstudio", "LM Studio (lokal)", "http://localhost:1234/v1",
                 key_required=False),
    OpenAIPreset("vllm", "vLLM (lokal)", "http://localhost:8000/v1",
                 key_required=False),
    OpenAIPreset("llamacpp", "llama.cpp-Server (lokal)", "http://localhost:8080/v1",
                 key_required=False),
    OpenAIPreset("ollama_v1", "Ollama über /v1 (lokal)", "http://localhost:11434/v1",
                 key_required=False,
                 note="Ollama lässt sich auch nativ einrichten (Typ „ollama“); "
                      "das ist der bevorzugte Weg."),
)

DISCLAIMER = (
    "Adressen aus der Dokumentation der Anbieter, nicht automatisch geprüft. "
    "„Verbindung testen“ prüft sie mit Ihrem Schlüssel."
)


def _norm(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def preset_for_url(url: str) -> Optional[OpenAIPreset]:
    """Die Vorlage, deren Basis-URL genau dieser URL entspricht (sonst None)."""
    target = _norm(url)
    if not target:
        return None
    for p in OPENAI_COMPATIBLE_PRESETS:
        if _norm(p.base_url) == target:
            return p
    return None
