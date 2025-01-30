---
title: "How can I create hyperlinks in a PyQt5 QPlainTextEdit?"
date: "2025-01-30"
id: "how-can-i-create-hyperlinks-in-a-pyqt5"
---
The core challenge in embedding hyperlinks within a PyQt5 `QPlainTextEdit` arises from the widget’s primary function: displaying and editing plain text. Unlike `QTextEdit`, which supports rich text formats including embedded HTML, `QPlainTextEdit` requires a different approach to achieve clickable hyperlinks. This involves strategically using QTextCursor, QTextCharFormat, and event handling to simulate hyperlink behavior. I've encountered this limitation frequently while developing internal documentation tools where plain text editors are preferred for their performance.

The primary methodology rests on three key steps: 1) identifying text sequences that should function as hyperlinks, 2) formatting these sequences to visually resemble hyperlinks, and 3) intercepting mouse clicks to trigger the desired action (typically opening a URL in a web browser).

Here's a breakdown of the implementation, incorporating my experience from several past projects:

**1. Identifying and Formatting Hyperlinks:**

The initial step involves scanning the text content for patterns that represent URLs or other link targets. Regular expressions offer a powerful way to accomplish this. Once a matching pattern is found, the text region is modified with a specific `QTextCharFormat`. This format is applied using a `QTextCursor`, which allows precise manipulation of the text’s characteristics.

Consider the following Python code snippet, which illustrates this process:

```python
from PyQt5.QtWidgets import QApplication, QPlainTextEdit
from PyQt5.QtGui import QTextCharFormat, QTextCursor, QColor, QFont
import re

class HyperlinkEditor(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hyperlink_format = QTextCharFormat()
        self.hyperlink_format.setForeground(QColor("blue"))
        self.hyperlink_format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.SingleUnderline)
        self.hyperlink_format.setFont(QFont("Arial", 10))
        self.url_regex = re.compile(r'(https?://\S+)')
        self.links = {} # Store link targets alongside their text spans
        self.textChanged.connect(self.format_hyperlinks)
        self.mouseReleaseEvent = self.open_link_on_click #Override default mouseReleaseEvent
    def format_hyperlinks(self):
        cursor = self.textCursor()
        text = self.toPlainText()
        cursor.setPosition(0) #Start at the beginning
        self.links = {} #Clear before reformatting

        for match in self.url_regex.finditer(text):
            start, end = match.span()
            url = match.group(0)

            cursor.setPosition(start, QTextCursor.MoveMode.MoveAnchor)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)

            cursor.mergeCharFormat(self.hyperlink_format)
            self.links[(start, end)] = url

    def open_link_on_click(self, event):
         cursor = self.textCursor()
         position = self.cursorForPosition(event.pos()).position()
         for (start, end), url in self.links.items():
            if start <= position <= end:
                #Open Link in external browser (implement the logic here)
                import webbrowser
                webbrowser.open_new(url)
                break
```

Here, the `HyperlinkEditor` class extends `QPlainTextEdit`. The `__init__` method initializes the text format used for hyperlinks and the regular expression used to find them. The `format_hyperlinks` method iterates through matches found by the regex, retrieves their starting and ending character positions, and formats them. Crucially, the found links are stored in the `links` dictionary. This dictionary maps text span ranges to their corresponding URL values.

**2. Intercepting Mouse Clicks:**

After formatting, the next critical step is capturing mouse clicks within the text area. To accomplish this, the `mouseReleaseEvent` needs to be overridden within the `HyperlinkEditor`.

The following code segment within the `HyperlinkEditor` class outlines how clicks are detected and acted upon:

```python
    def open_link_on_click(self, event):
         cursor = self.textCursor()
         position = self.cursorForPosition(event.pos()).position()
         for (start, end), url in self.links.items():
            if start <= position <= end:
                #Open Link in external browser (implement the logic here)
                import webbrowser
                webbrowser.open_new(url)
                break

```

This event handler gets the cursor position at the click's location using `cursorForPosition(event.pos())`, and then the current position using `.position()`. It iterates over the entries in the `links` dictionary, comparing the cursor's current position to the range of each hyperlink. If the clicked position falls within a hyperlink, the linked URL (stored in the `links` dictionary) is opened with `webbrowser.open_new(url)`.

**3. Handling Dynamic Content:**

When the content of a `QPlainTextEdit` is dynamically updated, it's important to re-apply the hyperlink formatting. This involves connecting the `textChanged` signal of the `QPlainTextEdit` to the `format_hyperlinks` method, as demonstrated in the class constructor:

```python
        self.textChanged.connect(self.format_hyperlinks)
```

This ensures that hyperlinks are properly identified and styled, even if the text is inserted or changed programmatically or via user input.

Here's a complete runnable example:
```python
from PyQt5.QtWidgets import QApplication, QPlainTextEdit, QVBoxLayout, QWidget
from PyQt5.QtGui import QTextCharFormat, QTextCursor, QColor, QFont
import re
import webbrowser


class HyperlinkEditor(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hyperlink_format = QTextCharFormat()
        self.hyperlink_format.setForeground(QColor("blue"))
        self.hyperlink_format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.SingleUnderline)
        self.hyperlink_format.setFont(QFont("Arial", 10))
        self.url_regex = re.compile(r'(https?://\S+)')
        self.links = {}  # Store link targets alongside their text spans
        self.textChanged.connect(self.format_hyperlinks)
        self.mouseReleaseEvent = self.open_link_on_click  # Override default mouseReleaseEvent

    def format_hyperlinks(self):
        cursor = self.textCursor()
        text = self.toPlainText()
        cursor.setPosition(0)  # Start at the beginning
        self.links = {}  # Clear before reformatting

        for match in self.url_regex.finditer(text):
            start, end = match.span()
            url = match.group(0)

            cursor.setPosition(start, QTextCursor.MoveMode.MoveAnchor)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)

            cursor.mergeCharFormat(self.hyperlink_format)
            self.links[(start, end)] = url

    def open_link_on_click(self, event):
        cursor = self.textCursor()
        position = self.cursorForPosition(event.pos()).position()
        for (start, end), url in self.links.items():
            if start <= position <= end:
                webbrowser.open_new(url)
                break


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hyperlink Example")
        self.layout = QVBoxLayout()
        self.editor = HyperlinkEditor()
        self.editor.setPlainText("This is a test.  Visit https://www.google.com or https://www.example.org. Also try no url")
        self.layout.addWidget(self.editor)
        self.setLayout(self.layout)


if __name__ == "__main__":
    app = QApplication([])
    window = Window()
    window.show()
    app.exec_()
```

This example creates a window containing a `HyperlinkEditor`. It populates the editor with some initial text containing hyperlinks which are automatically formatted, with the ability to click the link using the mouse event.

**Resource Recommendations**

For further exploration of related concepts, I recommend examining these resources:

*   The PyQt5 documentation, specifically the sections pertaining to `QPlainTextEdit`, `QTextCursor`, `QTextCharFormat`, and event handling within widgets.
*   Materials focusing on regular expressions in Python, as the effective use of regular expressions is paramount for flexible and accurate link detection.
*   Web browser integration mechanisms and API calls. This is important to understand to control how URLs are opened after being clicked in the editor.
*   Examples of `QTextEdit` implementations in pyQt as some of the concepts transfer to a `QPlainTextEdit`

By integrating these techniques and leveraging well-documented resources, I’ve consistently been able to create interactive plain-text editors with hyperlink functionality, despite the limitations of the `QPlainTextEdit` widget. This requires a deeper engagement with the widget’s underlying mechanisms compared to the more feature-rich `QTextEdit`.
