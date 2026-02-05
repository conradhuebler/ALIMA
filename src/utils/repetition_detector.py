"""
Repetition Detector - Detect LLM repetition loops and provide recovery suggestions
Claude Generated

This module provides detection of repetitive patterns in LLM output streams:
1. Character patterns (e.g., "!!!..." or "aaaa...")
2. N-gram repetitions (repeated word sequences)
3. Window similarity (similar text blocks via Jaccard similarity)

Usage:
    from repetition_detector import RepetitionDetector

    detector = RepetitionDetector()
    for chunk in llm_stream:
        result = detector.add_chunk(chunk)
        if result and result.is_repetitive:
            # Handle repetition - abort, warn user, suggest parameters
            suggestions = get_parameter_variation_suggestions(
                temp=0.7, top_p=0.9, rep_penalty=1.0,
                detection_type=result.detection_type
            )
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class RepetitionDetectorConfig:
    """Configuration for repetition detection - Claude Generated

    Tuned to avoid false positives: requires substantial repetition before triggering.
    """

    # N-gram detection - tuned for real repetition loops, not normal text
    ngram_size: int = 6              # Words per N-gram (longer = more specific)
    ngram_threshold: int = 8         # Max repetitions before flagging (higher = more lenient)

    # Window similarity detection
    window_size: int = 300           # Characters per window (larger context)
    similarity_threshold: float = 0.90  # Jaccard similarity threshold (higher = stricter)
    min_windows: int = 4             # Minimum windows to compare

    # Character pattern detection
    char_repeat_threshold: int = 80  # Consecutive identical chars (higher = more lenient)

    # Processing control
    check_interval: int = 200        # Check every N characters (less frequent)
    min_text_length: int = 1000      # Start checking after N characters (more context first)

    # Behavior
    enabled: bool = True
    auto_abort: bool = True          # Automatically abort on detection


@dataclass
class RepetitionResult:
    """Result of repetition check - Claude Generated"""

    is_repetitive: bool
    detection_type: str = ""         # 'char_pattern', 'ngram', 'window_similarity'
    details: str = ""                # Human-readable description
    confidence: float = 0.0          # 0.0-1.0 confidence score
    repeated_content: str = ""       # The repeated pattern/content
    position: int = 0                # Position in text where detected

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization - Claude Generated"""
        return {
            "is_repetitive": self.is_repetitive,
            "detection_type": self.detection_type,
            "details": self.details,
            "confidence": self.confidence,
            "repeated_content": self.repeated_content[:100],  # Truncate for logging
            "position": self.position,
        }


class RepetitionDetector:
    """
    Detect repetitive patterns in streaming LLM output - Claude Generated

    Implements three detection methods in order of computational cost:
    1. Character patterns (fastest) - regex for consecutive identical chars
    2. N-gram counting (medium) - track repeated word sequences
    3. Window similarity (slowest) - Jaccard similarity between text windows

    Usage:
        detector = RepetitionDetector()
        detector.reset()  # Start new detection session

        for chunk in llm_stream:
            result = detector.add_chunk(chunk)
            if result and result.is_repetitive:
                # Take action (abort, warn, etc.)
                break
    """

    def __init__(self, config: Optional[RepetitionDetectorConfig] = None):
        self.config = config or RepetitionDetectorConfig()
        self._accumulated_text = ""
        self._last_check_position = 0
        self._ngram_counter: Counter = Counter()
        self._word_buffer: List[str] = []

    def reset(self):
        """Reset detector state for new generation - Claude Generated"""
        self._accumulated_text = ""
        self._last_check_position = 0
        self._ngram_counter.clear()
        self._word_buffer.clear()

    def add_chunk(self, chunk: str) -> Optional[RepetitionResult]:
        """
        Add text chunk and check for repetition - Claude Generated

        Args:
            chunk: New text chunk from LLM stream

        Returns:
            RepetitionResult if repetition detected, None otherwise
        """
        if not self.config.enabled or not chunk:
            return None

        self._accumulated_text += chunk
        text_len = len(self._accumulated_text)

        # Don't check until minimum text length reached
        if text_len < self.config.min_text_length:
            return None

        # Only check at specified intervals
        if text_len - self._last_check_position < self.config.check_interval:
            return None

        self._last_check_position = text_len

        # Run checks in order of computational cost (fast to slow)

        # 1. Character pattern check (fastest)
        result = self._check_char_patterns()
        if result and result.is_repetitive:
            logger.warning(f"Repetition detected (char_pattern): {result.details}")
            return result

        # 2. N-gram check (medium)
        result = self._check_ngrams(chunk)
        if result and result.is_repetitive:
            logger.warning(f"Repetition detected (ngram): {result.details}")
            return result

        # 3. Window similarity check (slowest)
        result = self._check_window_similarity()
        if result and result.is_repetitive:
            logger.warning(f"Repetition detected (window_similarity): {result.details}")
            return result

        return None

    def _check_char_patterns(self) -> Optional[RepetitionResult]:
        """
        Check for repeated character patterns like "!!!..." or "aaaa..." - Claude Generated

        Uses regex to find sequences of identical characters exceeding threshold.
        """
        threshold = self.config.char_repeat_threshold

        # Pattern: any character repeated threshold+ times
        pattern = rf'(.)\1{{{threshold},}}'

        match = re.search(pattern, self._accumulated_text)
        if match:
            repeated_char = match.group(1)
            repeat_count = len(match.group(0))

            return RepetitionResult(
                is_repetitive=True,
                detection_type="char_pattern",
                details=f"Character '{repr(repeated_char)}' repeated {repeat_count} times",
                confidence=min(1.0, repeat_count / (threshold * 2)),
                repeated_content=match.group(0)[:50],
                position=match.start(),
            )

        return None

    def _check_ngrams(self, new_chunk: str) -> Optional[RepetitionResult]:
        """
        Check for repeated N-gram patterns (word sequences) - Claude Generated

        Tracks N-grams across the entire text and flags when any N-gram
        appears more than ngram_threshold times.
        """
        # Tokenize new chunk into words
        words = re.findall(r'\b\w+\b', new_chunk.lower())
        self._word_buffer.extend(words)

        # Keep word buffer manageable (last 1000 words)
        if len(self._word_buffer) > 1000:
            self._word_buffer = self._word_buffer[-1000:]

        # Build N-grams from buffer
        n = self.config.ngram_size
        if len(self._word_buffer) < n:
            return None

        # Count new N-grams
        for i in range(len(self._word_buffer) - n + 1):
            ngram = tuple(self._word_buffer[i:i + n])
            self._ngram_counter[ngram] += 1

        # Check for repeated N-grams exceeding threshold
        threshold = self.config.ngram_threshold
        for ngram, count in self._ngram_counter.most_common(5):
            if count >= threshold:
                ngram_text = ' '.join(ngram)

                return RepetitionResult(
                    is_repetitive=True,
                    detection_type="ngram",
                    details=f"Phrase '{ngram_text}' repeated {count} times (threshold: {threshold})",
                    confidence=min(1.0, count / (threshold * 2)),
                    repeated_content=ngram_text,
                    position=self._accumulated_text.rfind(ngram_text),
                )

        return None

    def _check_window_similarity(self) -> Optional[RepetitionResult]:
        """
        Check for similar text windows using Jaccard similarity - Claude Generated

        Divides text into windows and compares adjacent windows for similarity.
        High similarity between windows indicates repetitive content.
        """
        window_size = self.config.window_size
        min_windows = self.config.min_windows

        text = self._accumulated_text
        if len(text) < window_size * min_windows:
            return None

        # Create windows
        windows: List[Set[str]] = []
        for i in range(0, len(text) - window_size + 1, window_size // 2):  # 50% overlap
            window_text = text[i:i + window_size]
            # Convert to word set for Jaccard
            word_set = set(re.findall(r'\b\w+\b', window_text.lower()))
            if word_set:  # Skip empty windows
                windows.append(word_set)

        if len(windows) < min_windows:
            return None

        # Compare consecutive windows
        high_similarity_count = 0
        max_similarity = 0.0
        similar_position = 0

        for i in range(len(windows) - 1):
            similarity = self._jaccard_similarity(windows[i], windows[i + 1])
            if similarity > max_similarity:
                max_similarity = similarity
                similar_position = i * (window_size // 2)

            if similarity >= self.config.similarity_threshold:
                high_similarity_count += 1

        # Flag if multiple consecutive windows are highly similar
        if high_similarity_count >= min_windows - 1:
            return RepetitionResult(
                is_repetitive=True,
                detection_type="window_similarity",
                details=f"{high_similarity_count} window pairs with >{self.config.similarity_threshold:.0%} similarity",
                confidence=max_similarity,
                repeated_content=text[similar_position:similar_position + window_size][:100],
                position=similar_position,
            )

        return None

    @staticmethod
    def _jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets - Claude Generated"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


def get_parameter_variation_suggestions(
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    detection_type: str = ""
) -> List[Dict[str, Any]]:
    """
    Generate parameter variation suggestions to recover from repetition - Claude Generated

    Based on the type of repetition detected, suggests specific parameter
    adjustments that may help resolve the issue.

    Args:
        temperature: Current temperature setting
        top_p: Current top_p setting
        repetition_penalty: Current repetition penalty
        detection_type: Type of repetition detected ('char_pattern', 'ngram', 'window_similarity')

    Returns:
        List of suggestion dicts with 'label', 'description', and 'params' keys
    """
    suggestions = []

    # Always suggest increasing temperature if low
    if temperature < 1.0:
        new_temp = min(1.5, temperature + 0.3)
        suggestions.append({
            "label": "Temperatur erhöhen",
            "description": f"Erhöht Kreativität/Variation ({temperature:.1f} → {new_temp:.1f})",
            "params": {"temperature": new_temp},
            "priority": 1,
        })

    # Suggest repetition penalty increase
    if repetition_penalty < 1.5:
        new_penalty = min(2.0, repetition_penalty + 0.3)
        suggestions.append({
            "label": "Repetition Penalty erhöhen",
            "description": f"Bestraft wiederholte Token ({repetition_penalty:.1f} → {new_penalty:.1f})",
            "params": {"repetition_penalty": new_penalty},
            "priority": 2 if detection_type == "ngram" else 3,
        })

    # Suggest top_p adjustment
    if top_p > 0.5:
        new_top_p = max(0.3, top_p - 0.2)
        suggestions.append({
            "label": "Top-P reduzieren",
            "description": f"Fokussiert auf wahrscheinlichere Token ({top_p:.1f} → {new_top_p:.1f})",
            "params": {"top_p": new_top_p},
            "priority": 3,
        })
    elif top_p < 0.8:
        new_top_p = min(0.95, top_p + 0.15)
        suggestions.append({
            "label": "Top-P erhöhen",
            "description": f"Erweitert Token-Auswahl ({top_p:.1f} → {new_top_p:.1f})",
            "params": {"top_p": new_top_p},
            "priority": 3,
        })

    # Combined suggestion for severe repetition
    if detection_type in ("char_pattern", "window_similarity"):
        suggestions.append({
            "label": "Aggressive Anpassung",
            "description": "Temperatur + Penalty kombinierterhöhen",
            "params": {
                "temperature": min(1.5, temperature + 0.4),
                "repetition_penalty": min(2.0, repetition_penalty + 0.5),
            },
            "priority": 1 if detection_type == "char_pattern" else 2,
        })

    # Sort by priority
    suggestions.sort(key=lambda x: x.get("priority", 99))

    return suggestions


def format_repetition_warning(result: RepetitionResult) -> str:
    """
    Format repetition result for user display - Claude Generated

    Args:
        result: RepetitionResult from detector

    Returns:
        Human-readable warning message
    """
    type_labels = {
        "char_pattern": "Zeichenwiederholung",
        "ngram": "Phrasenwiederholung",
        "window_similarity": "Textblock-Wiederholung",
    }

    type_label = type_labels.get(result.detection_type, result.detection_type)

    warning = f"⚠️ Wiederholung erkannt: {type_label}\n"
    warning += f"   {result.details}\n"

    if result.repeated_content:
        preview = result.repeated_content[:80]
        if len(result.repeated_content) > 80:
            preview += "..."
        warning += f"   Muster: \"{preview}\"\n"

    warning += f"   Konfidenz: {result.confidence:.0%}"

    return warning
