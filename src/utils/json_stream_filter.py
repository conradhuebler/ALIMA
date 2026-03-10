"""JSON Stream Filter - Entfernt JSON-Syntax aus dem Token-Stream-Display - Claude Generated

Verfolgt den Zustand eines eingehenden JSON-Streams und gibt nur die
menschenlesbaren Werte aus, ohne Klammern, Keys oder Satzzeichen.
"""
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)

# Key-Name → Header Mapping für lesbare Ausgabe
KEY_LABELS = {
    "analyse": "Analyse",
    "keywords": "Schlagworte",
    "keyword": "",
    "gnd_id": " (GND-ID: ",  # Special: inline prefix
    "gnd_classes": "GND-Systematik",
    "title": "Arbeitstitel",
    "keyword_chains": "Schlagwortketten",
    "chain": "",
    "reason": " - ",  # Special: inline prefix for chain reason
    "missing_concepts": "Fehlende Konzepte",
    "missing_superordinate_terms": "Fehlende Oberbegriffe",
    "classifications": "Klassifikationen",
    "code": "",
    "type": " (",  # Special: inline prefix for type
}

# Keys whose values get special inline suffixes
_INLINE_SUFFIX = {
    "gnd_id": ")",
    "type": ")",
    "reason": "",
}

# Keys whose values should be completely suppressed
_SUPPRESSED_KEYS = set()  # Currently none


class _State(Enum):
    OUTSIDE = auto()
    IN_OBJECT = auto()
    IN_KEY = auto()
    AFTER_KEY = auto()
    AFTER_COLON = auto()
    IN_STRING_VALUE = auto()
    IN_ARRAY = auto()
    IN_NUMBER = auto()
    IN_LITERAL = auto()


class JsonStreamFilter:
    """Filters JSON syntax from a token stream, outputting only human-readable values.

    Usage:
        filter = JsonStreamFilter(enabled=True)
        for token in stream:
            display_text = filter.feed(token)
            if display_text:
                show_to_user(display_text)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.reset()

    def reset(self):
        """Reset filter state for a new stream."""
        self._state = _State.OUTSIDE
        self._current_key = ""
        self._escape_next = False
        self._buffer = ""
        self._first_value_in_array = True
        # Context stack: each entry is ('object',) or ('array', first_value_flag)
        self._context_stack = []
        self._value_key = ""  # Key of the current string value being read

    def feed(self, token: str) -> str:
        """Process a token and return filtered display text."""
        if not self.enabled:
            return token

        output = []
        for char in token:
            result = self._process_char(char)
            if result:
                output.append(result)

        return "".join(output)

    def _push_context(self, ctx_type: str):
        self._context_stack.append(ctx_type)

    def _pop_context(self) -> str:
        if self._context_stack:
            return self._context_stack.pop()
        return "outside"

    def _current_context(self) -> str:
        if self._context_stack:
            return self._context_stack[-1]
        return "outside"

    def _return_to_context(self) -> _State:
        """Determine which state to return to based on context stack."""
        ctx = self._current_context()
        if ctx == "object":
            return _State.IN_OBJECT
        elif ctx == "array":
            return _State.IN_ARRAY
        return _State.OUTSIDE

    def _process_char(self, char: str) -> str:
        """Process a single character through the state machine."""

        # Handle escape sequences in strings
        if self._escape_next:
            self._escape_next = False
            if self._state == _State.IN_STRING_VALUE:
                if char == 'n':
                    return "\n"
                elif char == 't':
                    return "\t"
                elif char == '"':
                    return '"'
                elif char == '\\':
                    return "\\"
                return char
            if self._state == _State.IN_KEY:
                self._current_key += char
            return ""

        if char == '\\' and self._state in (_State.IN_KEY, _State.IN_STRING_VALUE):
            self._escape_next = True
            return ""

        # State machine
        if self._state == _State.OUTSIDE:
            if char == '{':
                self._state = _State.IN_OBJECT
                self._push_context("object")
                return ""
            return ""

        elif self._state == _State.IN_OBJECT:
            if char == '"':
                self._state = _State.IN_KEY
                self._current_key = ""
                return ""
            elif char == '}':
                self._pop_context()
                self._state = self._return_to_context()
                return ""
            elif char == ',':
                return ""
            return ""

        elif self._state == _State.IN_KEY:
            if char == '"':
                self._state = _State.AFTER_KEY
                return ""
            self._current_key += char
            return ""

        elif self._state == _State.AFTER_KEY:
            if char == ':':
                self._state = _State.AFTER_COLON
                return ""
            return ""

        elif self._state == _State.AFTER_COLON:
            if char == '"':
                self._state = _State.IN_STRING_VALUE
                self._value_key = self._current_key
                header = self._get_header_for_key(self._current_key)
                return header if header else ""
            elif char == '[':
                self._state = _State.IN_ARRAY
                self._push_context("array")
                self._first_value_in_array = True
                header = self._get_header_for_key(self._current_key)
                return header if header else ""
            elif char == '{':
                self._state = _State.IN_OBJECT
                self._push_context("object")
                return ""
            elif char in '0123456789-':
                self._state = _State.IN_NUMBER
                self._value_key = self._current_key
                self._buffer = char
                header = self._get_header_for_key(self._current_key)
                return header if header else ""
            elif char in 'tfn':
                self._state = _State.IN_LITERAL
                self._buffer = char
                return ""
            return ""

        elif self._state == _State.IN_STRING_VALUE:
            if char == '"':
                # End of string value - return to containing context
                suffix = _INLINE_SUFFIX.get(self._value_key, "")
                self._value_key = ""
                self._state = self._return_to_context()
                return suffix
            return char

        elif self._state == _State.IN_ARRAY:
            if char == '"':
                self._state = _State.IN_STRING_VALUE
                self._value_key = self._current_key
                prefix = ""
                if not self._first_value_in_array:
                    prefix = ", "
                self._first_value_in_array = False
                return prefix
            elif char == '{':
                self._state = _State.IN_OBJECT
                self._push_context("object")
                prefix = ""
                if not self._first_value_in_array:
                    prefix = "\n"
                self._first_value_in_array = False
                return prefix
            elif char == ']':
                self._pop_context()
                self._state = self._return_to_context()
                return "\n"
            elif char == ',':
                return ""
            elif char in '0123456789-':
                self._state = _State.IN_NUMBER
                self._value_key = self._current_key
                self._buffer = char
                prefix = ""
                if not self._first_value_in_array:
                    prefix = ", "
                self._first_value_in_array = False
                return prefix
            return ""

        elif self._state == _State.IN_NUMBER:
            if char in '0123456789.eE+-':
                self._buffer += char
                return ""
            result = self._buffer
            suffix = _INLINE_SUFFIX.get(self._value_key, "")
            self._buffer = ""
            self._value_key = ""
            if char == ',':
                self._state = self._return_to_context()
                return result + suffix
            elif char == '}':
                self._pop_context()
                self._state = self._return_to_context()
                return result + suffix
            elif char == ']':
                self._pop_context()
                self._state = self._return_to_context()
                return result + suffix + "\n"
            return result + suffix

        elif self._state == _State.IN_LITERAL:
            if char.isalpha():
                self._buffer += char
                return ""
            self._buffer = ""
            if char == ',':
                self._state = self._return_to_context()
            elif char == '}':
                self._pop_context()
                self._state = self._return_to_context()
            elif char == ']':
                self._pop_context()
                self._state = self._return_to_context()
            return ""

        return ""

    def _get_header_for_key(self, key: str) -> str:
        """Get a readable section header or inline prefix for a JSON key."""
        label = KEY_LABELS.get(key, key)
        if not label:
            return ""
        # Inline prefixes (like " (GND-ID: ") don't get newlines
        if key in _INLINE_SUFFIX:
            return label
        return f"\n{label}: "
