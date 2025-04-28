import re
from typing import List, Set, Dict, Tuple, Optional
import unicodedata
from dataclasses import dataclass
from collections import Counter
import logging
from pathlib import Path
import json


@dataclass
class ProcessingResult:
    """Ergebnis der Textverarbeitung"""

    cleaned_text: str
    keywords: List[str]
    stats: Dict[str, int]
    language: str


class TextProcessor:
    """Klasse für die Verarbeitung und Analyse von Texten"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._stopwords = self._load_stopwords()
        self._special_chars = set('.,;:!?()[]{}«»""\'"')
        self._language_patterns = self._load_language_patterns()

    def _load_stopwords(self) -> Dict[str, Set[str]]:
        """Lädt Stoppwörter für verschiedene Sprachen"""
        try:
            stopwords_path = Path(__file__).parent / "data" / "stopwords.json"
            with open(stopwords_path, "r", encoding="utf-8") as f:
                return {lang: set(words) for lang, words in json.load(f).items()}
        except Exception as e:
            self.logger.warning(f"Konnte Stoppwörter nicht laden: {e}")
            return {
                "de": set(["der", "die", "das", "und", "in", "ist", "von", "für"]),
                "en": set(["the", "and", "is", "in", "of", "to", "a", "for"]),
            }

    def _load_language_patterns(self) -> Dict[str, List[str]]:
        """Lädt Spracherkennungsmuster"""
        return {
            "de": ["der", "die", "das", "und", "ist", "von", "für"],
            "en": ["the", "and", "is", "of", "to", "for"],
            "fr": ["le", "la", "les", "et", "est", "pour"],
        }

    def process_text(self, text: str, min_word_length: int = 3) -> ProcessingResult:
        """
        Verarbeitet einen Text vollständig.

        Args:
            text: Zu verarbeitender Text
            min_word_length: Minimale Wortlänge für Keywords

        Returns:
            ProcessingResult mit verarbeitetem Text und Analysen
        """
        # Grundlegende Reinigung
        cleaned_text = self.clean_text(text)

        # Spracherkennung
        language = self.detect_language(cleaned_text)

        # Keyword-Extraktion
        keywords = self.extract_keywords(
            cleaned_text, language=language, min_length=min_word_length
        )

        # Statistiken erstellen
        stats = self.generate_stats(cleaned_text, language)

        return ProcessingResult(
            cleaned_text=cleaned_text, keywords=keywords, stats=stats, language=language
        )

    def clean_text(self, text: str) -> str:
        """
        Reinigt einen Text von unerwünschten Zeichen und normalisiert ihn.

        Args:
            text: Zu reinigender Text

        Returns:
            Gereinigter Text
        """
        if not text:
            return ""

        # Normalisiere Unicode-Zeichen
        text = unicodedata.normalize("NFKC", text)

        # Entferne mehrfache Leerzeichen und Zeilenumbrüche
        text = re.sub(r"\s+", " ", text)

        # Ersetze spezielle Anführungszeichen
        text = re.sub(r'[""″\'\'′]', '"', text)

        # Behandle Bindestriche und Gedankenstriche
        text = re.sub(r"[-‐‑‒–—―]", "-", text)

        # Entferne URLs
        text = re.sub(r"http[s]?://\S+", "", text)

        # Normalisiere Satzzeichen
        for char in self._special_chars:
            text = text.replace(char, f" {char} ")

        return text.strip()

    def extract_keywords(
        self,
        text: str,
        language: str = "de",
        min_length: int = 3,
        max_keywords: int = 20,
    ) -> List[str]:
        """
        Extrahiert Keywords aus einem Text.

        Args:
            text: Zu analysierender Text
            language: Sprache des Textes
            min_length: Minimale Wortlänge
            max_keywords: Maximale Anzahl der Keywords

        Returns:
            Liste der gefundenen Keywords
        """
        # Hole Stoppwörter für die Sprache
        stopwords = self._stopwords.get(language, set())

        # Tokenisierung
        words = text.lower().split()

        # Filtere Wörter
        filtered_words = [
            word
            for word in words
            if len(word) >= min_length
            and word not in stopwords
            and not any(c in self._special_chars for c in word)
            and not word.isdigit()
        ]

        # Zähle Häufigkeiten
        word_counts = Counter(filtered_words)

        # Wähle die häufigsten Wörter
        return [word for word, _ in word_counts.most_common(max_keywords)]

    def detect_language(self, text: str) -> str:
        """
        Erkennt die Sprache eines Textes.

        Args:
            text: Zu analysierender Text

        Returns:
            Erkannte Sprache (ISO-Code)
        """
        text_lower = text.lower()
        word_counts = {
            lang: sum(1 for pattern in patterns if f" {pattern} " in f" {text_lower} ")
            for lang, patterns in self._language_patterns.items()
        }

        # Wähle die Sprache mit den meisten Übereinstimmungen
        if word_counts:
            return max(word_counts.items(), key=lambda x: x[1])[0]
        return "de"  # Fallback auf Deutsch

    def generate_stats(self, text: str, language: str) -> Dict[str, int]:
        """
        Generiert Statistiken für einen Text.

        Args:
            text: Zu analysierender Text
            language: Sprache des Textes

        Returns:
            Dictionary mit Statistiken
        """
        words = text.split()
        sentences = re.split(r"[.!?]+", text)

        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "average_word_length": (
                sum(len(word) for word in words) / len(words) if words else 0
            ),
            "average_sentence_length": len(words) / len(sentences) if sentences else 0,
        }

    def find_compound_terms(self, text: str, max_length: int = 3) -> List[str]:
        """
        Findet zusammengesetzte Begriffe im Text.

        Args:
            text: Zu analysierender Text
            max_length: Maximale Anzahl der Wörter in einem Begriff

        Returns:
            Liste gefundener zusammengesetzter Begriffe
        """
        words = text.split()
        compounds = []

        for i in range(len(words)):
            for j in range(2, max_length + 1):
                if i + j <= len(words):
                    compound = " ".join(words[i : i + j])
                    # Prüfe ob der Begriff sinnvoll ist (z.B. keine Stoppwörter am Anfang/Ende)
                    if self._is_valid_compound(compound):
                        compounds.append(compound)

        return compounds

    def _is_valid_compound(self, compound: str) -> bool:
        """
        Prüft, ob ein zusammengesetzter Begriff valid ist.

        Args:
            compound: Zu prüfender Begriff

        Returns:
            True wenn valid, sonst False
        """
        words = compound.lower().split()

        # Prüfe erste und letzte Wörter
        if words[0] in self._stopwords.get("de", set()) or words[
            -1
        ] in self._stopwords.get("de", set()):
            return False

        # Prüfe auf Sonderzeichen
        if any(char in compound for char in self._special_chars):
            return False

        return True

    def normalize_term(self, term: str) -> str:
        """
        Normalisiert einen Suchbegriff.

        Args:
            term: Zu normalisierender Begriff

        Returns:
            Normalisierter Begriff
        """
        # Grundlegende Reinigung
        term = self.clean_text(term)

        # Entferne Klammern und deren Inhalt
        term = re.sub(r"\([^)]*\)", "", term)

        # Normalisiere Bindestriche
        term = re.sub(r"\s*-\s*", "-", term)

        # Entferne übrige Sonderzeichen
        term = "".join(c for c in term if c not in self._special_chars)

        return term.strip()

    def get_context(self, text: str, term: str, context_size: int = 50) -> List[str]:
        """
        Findet Kontexte für einen Begriff im Text.

        Args:
            text: Zu durchsuchender Text
            term: Gesuchter Begriff
            context_size: Anzahl der Zeichen vor/nach dem Begriff

        Returns:
            Liste von Kontexten
        """
        contexts = []
        term_lower = term.lower()
        text_lower = text.lower()

        start = 0
        while True:
            pos = text_lower.find(term_lower, start)
            if pos == -1:
                break

            context_start = max(0, pos - context_size)
            context_end = min(len(text), pos + len(term) + context_size)

            context = text[context_start:context_end]
            if context_start > 0:
                context = f"...{context}"
            if context_end < len(text):
                context = f"{context}..."

            contexts.append(context)
            start = pos + 1

        return contexts
