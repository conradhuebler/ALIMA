"""
Fixed CollapsibleTextEdit for Qt6 - Proper toggle functionality
"""

from PyQt6.QtWidgets import QTextEdit, QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import (QTextCursor, QTextDocument, QTextBlockFormat,
                         QTextCharFormat, QFont, QPainter, QColor,
                         QTextFrameFormat, QTextFrame, QMouseEvent)
import re

class CollapsibleTextEdit(QTextEdit):
    """
    QTextEdit subclass with properly working collapsible text sections for Qt6
    """

    def __init__(self, parent=None):
        super(CollapsibleTextEdit, self).__init__(parent)
        self.collapsible_sections = {}  # Store section data with stable IDs
        self.next_section_id = 0

        # Enable mouse tracking for click detection
        self.setMouseTracking(True)

        # Setup default styling
        self.setup_default_styles()

    def setup_default_styles(self):
        """Setup default styles for collapsible sections"""
        font = QFont("Consolas", 10)
        if not font.exactMatch():
            font = QFont("Courier New", 10)
        self.setFont(font)

        self.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                line-height: 1.3;
            }
        """)

    def add_collapsible_section(self, title, content, collapsed=True):
        """
        Add a collapsible section to the text edit

        Args:
            title: Title for the collapsible section
            content: Content to be collapsed/expanded
            collapsed: Initial state (True = collapsed)

        Returns:
            section_id: Unique identifier for this section
        """
        section_id = self.next_section_id
        self.next_section_id += 1

        # Store section data
        self.collapsible_sections[section_id] = {
            'title': title,
            'content': content,
            'collapsed': collapsed,
            'full_content': content
        }

        # Insert the section
        self._insert_section(section_id)

        return section_id

    def _insert_section(self, section_id):
        """Insert or update a section in the document"""
        if section_id not in self.collapsible_sections:
            return

        section = self.collapsible_sections[section_id]

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Add spacing if not at start
        if cursor.position() > 0:
            cursor.insertText("\n")

        # Create section markers
        start_marker = f"<<<SECTION_START_{section_id}>>>"
        end_marker = f"<<<SECTION_END_{section_id}>>>"

        # Insert start marker (invisible)
        marker_start = cursor.position()
        cursor.insertText(start_marker)

        # Format start marker as hidden
        cursor.setPosition(marker_start)
        cursor.setPosition(cursor.position(), QTextCursor.MoveMode.KeepAnchor)
        hidden_format = QTextCharFormat()
        hidden_format.setForeground(QColor(255, 255, 255, 0))  # Transparent
        hidden_format.setFontPointSize(1)
        cursor.setCharFormat(hidden_format)

        # Insert header
        toggle_char = "▶" if section['collapsed'] else "▼"
        header_text = f"{toggle_char} {section['title']}"

        header_start = cursor.position()
        cursor.insertText(header_text)
        header_end = cursor.position()

        # Format header as clickable
        cursor.setPosition(header_start)
        cursor.setPosition(header_end, QTextCursor.MoveMode.KeepAnchor)

        header_format = QTextCharFormat()
        header_format.setForeground(QColor("#0066cc"))
        header_format.setFontWeight(QFont.Weight.Bold)
        header_format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.SingleUnderline)
        header_format.setProperty(QTextCharFormat.Property.UserProperty, section_id)  # Store section ID
        cursor.setCharFormat(header_format)

        # Insert content
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText("\n")

        content_start = cursor.position()

        if section['collapsed']:
            placeholder_text = "[Click to expand content...]"
            cursor.insertText(placeholder_text)

            # Format placeholder
            cursor.setPosition(content_start)
            cursor.setPosition(cursor.position(), QTextCursor.MoveMode.KeepAnchor)

            placeholder_format = QTextCharFormat()
            placeholder_format.setForeground(QColor("#888888"))
            placeholder_format.setFontItalic(True)
            cursor.setCharFormat(placeholder_format)
        else:
            cursor.insertText(section['content'])

            # Format content
            cursor.setPosition(content_start)
            cursor.setPosition(cursor.position(), QTextCursor.MoveMode.KeepAnchor)

            content_format = QTextCharFormat()
            content_format.setBackground(QColor("#f8f8f8"))
            content_format.setFontFamily("Consolas")
            cursor.setCharFormat(content_format)

        # Insert end marker
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(f"\n{end_marker}\n")

        # Format end marker as hidden
        end_pos = cursor.position()
        cursor.setPosition(end_pos - len(end_marker) - 2)
        cursor.setPosition(end_pos, QTextCursor.MoveMode.KeepAnchor)
        cursor.setCharFormat(hidden_format)

        # Move cursor to end
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

    def toggle_section(self, section_id):
        """Toggle a collapsible section by completely regenerating it"""
        if section_id not in self.collapsible_sections:
            return

        # Toggle the state
        section = self.collapsible_sections[section_id]
        section['collapsed'] = not section['collapsed']

        # Find and remove the old section
        self._remove_section_from_document(section_id)

        # Re-insert the section with new state
        self._insert_section_at_end(section_id)

    def _remove_section_from_document(self, section_id):
        """Remove a section from the document"""
        start_marker = f"<<<SECTION_START_{section_id}>>>"
        end_marker = f"<<<SECTION_END_{section_id}>>>"

        document_text = self.toPlainText()

        start_index = document_text.find(start_marker)
        end_index = document_text.find(end_marker)

        if start_index != -1 and end_index != -1:
            # Remove the section text
            end_index += len(end_marker)

            cursor = self.textCursor()
            cursor.setPosition(start_index)
            cursor.setPosition(end_index, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()

    def _insert_section_at_end(self, section_id):
        """Insert section at the end of document"""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)
        self._insert_section(section_id)

    def mousePressEvent(self, event):
        """Handle mouse clicks for toggling sections"""
        if event.button() == Qt.MouseButton.LeftButton:
            cursor = self.cursorForPosition(event.pos())

            # Get the character format at click position
            char_format = cursor.charFormat()

            # Check if this character has a section ID property
            section_id = char_format.property(QTextCharFormat.Property.UserProperty)

            if section_id is not None and isinstance(section_id, int):
                self.toggle_section(section_id)
                event.accept()
                return

        super().mousePressEvent(event)

    def add_prompt_result_pair(self, prompt, result):
        """
        Convenience method to add a prompt-result pair

        Args:
            prompt: The prompt text
            result: The result text

        Returns:
            prompt_section_id: ID of the collapsible prompt section
        """
        # Add collapsible prompt section
        prompt_id = self.add_collapsible_section("💡 Current Prompt", prompt, collapsed=True)

        # Add result section (always visible)
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Add separator
        separator = "\n" + "="*20 + " 🤖 AI RESPONSE " + "="*20 + "\n"
        cursor.insertText(separator)

        # Add result with formatting
        result_start = cursor.position()
        cursor.insertText(result)
        result_end = cursor.position()

        # Format result
        cursor.setPosition(result_start)
        cursor.setPosition(result_end, QTextCursor.MoveMode.KeepAnchor)

        result_format = QTextCharFormat()
        result_format.setBackground(QColor("#f0f8ff"))
        result_format.setFontFamily("Consolas")
        cursor.setCharFormat(result_format)

        # Add closing separator
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText("\n" + "="*54 + "\n\n")

        self.setTextCursor(cursor)

        return prompt_id

    def clear_content(self):
        """Clear all content and reset section tracking"""
        self.clear()
        self.collapsible_sections.clear()
        self.next_section_id = 0

    def get_section_state(self, section_id):
        """Get the current state of a section"""
        if section_id in self.collapsible_sections:
            return self.collapsible_sections[section_id]['collapsed']
        return None

    def set_section_state(self, section_id, collapsed):
        """Set the state of a section"""
        if section_id in self.collapsible_sections:
            current_state = self.collapsible_sections[section_id]['collapsed']
            if current_state != collapsed:
                self.toggle_section(section_id)


# Demo application for testing
class CollapsibleTextDemo(QMainWindow):
    """Demo application for CollapsibleTextEdit"""

    def __init__(self):
        super(CollapsibleTextDemo, self).__init__()
        self.setWindowTitle("Qt6 Collapsible QTextEdit Demo - Fixed Toggle")
        self.setGeometry(100, 100, 1000, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Create the collapsible text edit
        self.text_edit = CollapsibleTextEdit()
        layout.addWidget(self.text_edit)

        # Add control buttons
        button_layout = QVBoxLayout()

        add_prompt_btn = QPushButton("➕ Add Prompt-Result Pair")
        add_prompt_btn.clicked.connect(self.add_demo_prompt_result)
        button_layout.addWidget(add_prompt_btn)

        add_section_btn = QPushButton("📁 Add Collapsible Section")
        add_section_btn.clicked.connect(self.add_demo_section)
        button_layout.addWidget(add_section_btn)

        add_debug_btn = QPushButton("🐛 Add Debug Info")
        add_debug_btn.clicked.connect(self.add_debug_info)
        button_layout.addWidget(add_debug_btn)

        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.clicked.connect(self.text_edit.clear_content)
        button_layout.addWidget(clear_btn)

        layout.addLayout(button_layout)

        # Add initial demo content
        self.add_initial_content()

    def add_initial_content(self):
        """Add some initial demo content"""
        prompt1 = """Create a Python function that implements the QuickSort algorithm.

Requirements:
- Use in-place sorting to minimize memory usage
- Include proper type hints for all parameters and return values
- Add comprehensive docstrings following Google style
- Handle edge cases like empty lists and single-element lists
- Include example usage in the docstring

Performance considerations:
- Implement proper pivot selection (median-of-three)
- Add tail recursion optimization where possible
- Consider switching to insertion sort for small subarrays"""

        result1 = """Here's an optimized QuickSort implementation with all requested features:

```python
def quicksort(arr: list[int], low: int = 0, high: int = None) -> None:
    \"\"\"
    Sort array in-place using QuickSort algorithm with optimizations.

    This implementation uses median-of-three pivot selection and switches
    to insertion sort for small subarrays to improve performance.

    Args:
        arr: List of integers to be sorted
        low: Starting index (default: 0)
        high: Ending index (default: len(arr) - 1)

    Returns:
        None: Sorts the array in-place

    Raises:
        TypeError: If arr is not a list of comparable elements

    Example:
        >>> numbers = [64, 34, 25, 12, 22, 11, 90]
        >>> quicksort(numbers)
        >>> print(numbers)
        [11, 12, 22, 25, 34, 64, 90]

    Time Complexity:
        - Best/Average: O(n log n)
        - Worst: O(n²) - rare with median-of-three pivot

    Space Complexity: O(log n) - recursion stack
    \"\"\"
    if high is None:
        high = len(arr) - 1

    # Base case and small array optimization
    if high - low < 10:  # Switch to insertion sort for small subarrays
        insertion_sort(arr, low, high)
        return

    if low < high:
        # Partition and get pivot index
        pivot_index = partition_median_of_three(arr, low, high)

        # Recursively sort elements before and after partition
        quicksort(arr, low, pivot_index - 1)
        quicksort(arr, pivot_index + 1, high)


def partition_median_of_three(arr: list[int], low: int, high: int) -> int:
    \"\"\"Partition with median-of-three pivot selection.\"\"\"
    # Choose median of first, middle, and last elements as pivot
    mid = (low + high) // 2

    # Sort the three elements and use middle as pivot
    if arr[mid] < arr[low]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[high] < arr[low]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[high] < arr[mid]:
        arr[mid], arr[high] = arr[high], arr[mid]

    # Place pivot at second-to-last position
    arr[mid], arr[high - 1] = arr[high - 1], arr[mid]

    pivot = arr[high - 1]
    i = low

    for j in range(low, high - 1):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    # Place pivot in correct position
    arr[i], arr[high - 1] = arr[high - 1], arr[i]
    return i


def insertion_sort(arr: list[int], low: int, high: int) -> None:
    \"\"\"Insertion sort for small subarrays.\"\"\"
    for i in range(low + 1, high + 1):
        key = arr[i]
        j = i - 1
        while j >= low and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# Example usage and testing:
if __name__ == "__main__":
    import random
    import time

    # Test with various array sizes
    test_arrays = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 2, 8, 1, 9],
        [1],  # Single element
        [],   # Empty array
        [3, 3, 3, 3],  # Duplicates
        list(range(1000, 0, -1))  # Reverse sorted - worst case
    ]

    for i, test_arr in enumerate(test_arrays):
        original = test_arr.copy()
        start_time = time.time()
        quicksort(test_arr)
        end_time = time.time()

        print(f"Test {i+1}: {original[:10]}{'...' if len(original) > 10 else ''}")
        print(f"Sorted: {test_arr[:10]}{'...' if len(test_arr) > 10 else ''}")
        print(f"Time: {(end_time - start_time)*1000:.2f}ms")
        print(f"Correct: {test_arr == sorted(original)}")
        print("-" * 50)
```

Key optimizations implemented:
✅ Median-of-three pivot selection reduces worst-case probability
✅ Insertion sort for small subarrays (< 10 elements) improves performance
✅ In-place sorting minimizes memory usage
✅ Comprehensive error handling and documentation
✅ Type hints for better code clarity"""

        self.text_edit.add_prompt_result_pair(prompt1, result1)

    def add_demo_prompt_result(self):
        """Add another demo prompt-result pair"""
        prompt = """Explain the differences between Python's list, tuple, set, and dictionary data structures.

Include:
- When to use each one
- Performance characteristics (time complexity for common operations)
- Memory usage considerations
- Practical examples showing appropriate use cases
- Common pitfalls and how to avoid them"""

        result = """Python Data Structures Comparison:

## 📋 LIST - Ordered, Mutable Sequence
```python
my_list = [1, 2, 3, 2]  # Allows duplicates
```
**Use when:** You need ordered, changeable data with possible duplicates
**Time Complexity:**
- Access by index: O(1)
- Append: O(1) amortized
- Insert/Delete: O(n)
- Search: O(n)

**Example Use Cases:**
- Shopping cart items
- Sequence of user actions
- Queue/stack implementation

## 🔒 TUPLE - Ordered, Immutable Sequence
```python
coordinates = (10, 20)  # Immutable
```
**Use when:** You need ordered data that never changes
**Benefits:**
- Hashable (can be dict keys)
- Slightly faster than lists
- Thread-safe
- Memory efficient

**Example Use Cases:**
- Coordinates (x, y)
- RGB color values
- Database records

## 🎯 SET - Unordered, Unique Elements
```python
unique_ids = {1, 2, 3}  # No duplicates
```
**Use when:** You need unique elements and fast membership testing
**Time Complexity:**
- Add/Remove/Contains: O(1) average
- Union/Intersection: O(min(len(s1), len(s2)))

**Example Use Cases:**
- Removing duplicates
- Mathematical set operations
- Checking membership quickly

## 📖 DICTIONARY - Key-Value Mapping
```python
user_data = {"name": "Alice", "age": 30}
```
**Use when:** You need fast lookups by key
**Time Complexity:**
- Get/Set/Delete: O(1) average
- Keys must be hashable

**Example Use Cases:**
- Caching/memoization
- Configuration settings
- Database-like lookups

## ⚠️ Common Pitfalls:

1. **Mutable Default Arguments:**
```python
# DON'T DO THIS:
def add_item(item, items=[]):  # Shared between calls!
    items.append(item)
    return items

# DO THIS:
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

2. **Dictionary Key Requirements:**
```python
# This will fail:
bad_dict = {[1, 2]: "value"}  # Lists aren't hashable

# This works:
good_dict = {(1, 2): "value"}  # Tuples are hashable
```

3. **Set vs List for Membership:**
```python
# Slow for large collections:
if item in large_list:  # O(n)

# Fast:
if item in large_set:   # O(1)
```

## 📊 Performance Summary:
| Operation | List | Tuple | Set | Dict |
|-----------|------|-------|-----|------|
| Access    | O(1) | O(1)  | N/A | O(1) |
| Search    | O(n) | O(n)  | O(1)| O(1) |
| Insert    | O(n) | N/A   | O(1)| O(1) |
| Delete    | O(n) | N/A   | O(1)| O(1) |

Choose the right data structure based on your specific use case!"""

        self.text_edit.add_prompt_result_pair(prompt, result)

    def add_demo_section(self):
        """Add a demo collapsible section"""
        title = "📋 Code Review Checklist"
        content = """Here's a comprehensive code review checklist:

## Functionality ✅
- [ ] Code works as intended
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] No obvious bugs

## Code Quality 📝
- [ ] Follows PEP 8 style guide
- [ ] Functions are single-purpose
- [ ] Variable names are descriptive
- [ ] No redundant code

## Documentation 📚
- [ ] Functions have docstrings
- [ ] Complex logic is commented
- [ ] Type hints are present
- [ ] README updated if needed

## Testing 🧪
- [ ] Unit tests exist
- [ ] Tests cover edge cases
- [ ] All tests pass
- [ ] Test names are descriptive

## Security 🔒
- [ ] No hardcoded secrets
- [ ] Input validation present
- [ ] SQL injection prevention
- [ ] XSS prevention (if web app)

## Performance ⚡
- [ ] No obvious bottlenecks
- [ ] Appropriate algorithms used
- [ ] Memory usage reasonable
- [ ] Database queries optimized"""

        self.text_edit.add_collapsible_section(title, content, collapsed=True)

    def add_debug_info(self):
        """Add debug information section"""
        title = "🐛 Debug Information"
        content = """Session Debug Info:
===================

Timestamp: 2024-01-15 14:30:22 UTC
Qt Version: 6.5.0
Python Version: 3.11.0
Platform: Windows 11

Memory Usage:
- Current: 156 MB
- Peak: 203 MB
- Available: 8.2 GB

Performance Metrics:
- Section Toggle Time: 12ms avg
- Document Render Time: 45ms
- Text Processing: 8ms
- Mouse Event Handling: <1ms

Active Sections: 3
Total Text Length: 2,847 characters
Line Count: 89

Widget State:
- Cursor Position: Line 45, Column 12
- Scroll Position: 25%
- Font: Consolas 10pt
- Background: #ffffff

Recent Actions:
1. Added prompt-result pair (14:28:15)
2. Toggled section 'Debug Info' (14:28:45)
3. Scrolled to bottom (14:29:02)
4. Added new section (14:29:12)

Last Error: None
Warnings: 0
Performance Issues: None detected"""

        self.text_edit.add_collapsible_section(title, content, collapsed=True)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    demo = CollapsibleTextDemo()
    demo.show()
    sys.exit(app.exec())
