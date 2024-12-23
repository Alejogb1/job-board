---
title: "Can PyCharm define keymap shortcuts for external tools based on file type?"
date: "2024-12-23"
id: "can-pycharm-define-keymap-shortcuts-for-external-tools-based-on-file-type"
---

Let's tackle this. I've actually run into this specific need more than a few times, especially when working on projects with diverse technology stacks all housed within a single repository. Setting up distinct workflows for different file types, within a consistent development environment like PyCharm, is a significant efficiency booster. So, the core question, "can PyCharm define keymap shortcuts for external tools based on file type?"... yes, absolutely, with a little configuration finesse. It's not inherently obvious, but PyCharm's flexibility allows for this type of targeted shortcut customization.

The magic lies in combining the power of “External Tools” and “Keymaps,” along with a crucial understanding of how PyCharm interprets file types. Let's break this down into the constituent parts and I'll walk you through how it works in practice, drawing from some past experiences.

First, consider that ‘external tools’ in PyCharm aren’t limited to just things like linters or formatters. They can essentially be any executable or script that you might want to run from within the IDE. This includes build processes, custom preprocessors, anything really. Now, while you can directly assign keyboard shortcuts to these, the shortcuts will be global. That’s where the file type sensitivity comes in.

To achieve the behavior you're after, you won't be assigning shortcuts directly to the *external tool* definition itself. Instead, you'll assign shortcuts to what PyCharm refers to as a "Menu Item". And this menu item can be conditionally available based on file type.

Here’s the process I typically follow, with a practical, albeit fictional, example based on something I configured on a project involving a mix of Python and custom configuration files:

1.  **Define the External Tool:** Let's say you have a script named `config_validator.py` that needs to be executed only when editing files ending in `.config`. This `config_validator.py` script might, for example, check the validity of a configuration file's syntax. First, you would define this script in PyCharm’s external tools configuration (Settings/Preferences -> Tools -> External Tools). You might configure the `config_validator.py` tool as follows:
    *   **Name:** `Validate Config`
    *   **Description:** "Validates the current config file"
    *   **Program:** The full path to the Python executable. (like `/usr/bin/python3` or `C:\Program Files\Python311\python.exe`)
    *   **Arguments:** The full path to the `config_validator.py` script, followed by `$FilePath$` which PyCharm will replace at runtime with the path to the currently active file.
    *   **Working Directory:**  `$FileDir$` (where the active file is located)

    This alone, though, does not give us a shortcut bound to `.config` files.

2.  **Define Menu Actions and Shortcuts:** Now you need to create a menu action that leverages this external tool, and *this* is where we introduce file type restriction. You accomplish this via the Keymap setting.
    *   Go to Settings/Preferences -> Keymap
    *   Search for the external tool's name ("Validate Config" in our example)
    *   You will see the external tool listed under the "External Tools" menu node. This is not where we want to add the shortcut directly. Instead, we need to right-click on this entry and select "Add Keyboard Shortcut".
    *   When the dialog appears, press the key combination you desire (e.g., `Ctrl+Alt+V`).
    *   Here's the crucial part: After assigning the keyboard shortcut, select the shortcut entry and click on the "Add..." button *under* the 'Shortcuts' box.
    *   In the "Add Keyboard Shortcut" dialog, click on "Use Only in..." and select the dropdown.
    *   You are now presented with a list of file types. Find or create a new file type using the `+` sign. If you don't have a `.config` type, click +, and provide the name `.config` in the "File Pattern" box, leaving all other fields at their default and click ok.
    *   Then select `.config` from the Use Only in menu, and finally, press "Ok" to confirm. This means your keybinding is only active when a file matching the pattern `*.config` is open.

Now, `Ctrl+Alt+V` will execute `config_validator.py` only when editing a `.config` file. If you open, for example a Python file, the shortcut will be inactive.

Let’s make this concrete with some code snippets. While these aren’t *PyCharm configuration* files, they demonstrate the principle of defining and executing file-type-specific actions.

**Snippet 1: The `config_validator.py` script:**

```python
#!/usr/bin/env python3

import sys

def validate_config(file_path):
    """ A simplistic config validator"""
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith("#") and "=" not in line:
                    print(f"Error: invalid config line: {line.strip()}")
                    sys.exit(1)
        print("Config file is valid.")
        sys.exit(0)
    except FileNotFoundError:
       print(f"Error: File not found: {file_path}")
       sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: config_validator.py <config_file_path>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    validate_config(config_file_path)
```

This Python script is what is executed by the external tool. It’s a basic validator for a fictional config file, checking for lines that aren't comments or key-value pairs.

**Snippet 2: How to apply a custom formatter to Markdown files:**

Consider another example. We can similarly integrate a markdown formatter via an external tool.

First, set up an external tool in PyCharm settings. Let’s call this tool "Markdown Formatter". Configure it to execute a markdown formatting tool (e.g., `prettier` if you have it) by specifying the full path to the `prettier` executable or similar, and setting the arguments to `"$FilePath$"` and the working directory to `$FileDir$`.

Now, add the keyboard shortcut for this tool and make it file-type-specific for `*.md` files, following the steps we outlined earlier for the config validator tool.

**Snippet 3: Example using Prettier:**

(Note: This example assumes Prettier is installed and available in your system’s PATH)

External Tool Configuration:
*   **Name:** Markdown Formatter
*   **Program:** `prettier` (or the full path to prettier if not in path)
*   **Arguments:**  `--write "$FilePath$"`
*   **Working Directory:**  `$FileDir$`

Keymap entry for shortcut `Ctrl+Alt+F` with a file type restriction to `*.md`: This will run prettier on the open .md file.

Through these steps, I’ve successfully created shortcuts that are context-aware. They don’t clutter the general keymap, and they apply only where relevant, keeping my development workflow clean and efficient.

For a deeper dive into how PyCharm handles file types and external tool integration, I recommend consulting the official PyCharm documentation. Specifically, look for sections on “File Types” and “External Tools”. For more abstract discussions on configurable keybindings and command systems in general-purpose IDEs, you may want to investigate research papers on user interface customization and efficient interaction methods. While this information is not always presented explicitly in research papers focusing on IDEs, the principles apply in a broad sense across many programmable software interfaces.

My experience has shown that understanding PyCharm’s modular nature – the separation between external tool definitions, menu actions, and keymap configurations – is the key to achieving this type of sophisticated workflow customization. This isn't a hack; it’s an intended, flexible feature of the platform. By using this approach, I’ve not only automated many repetitive tasks but have also created a personalized development experience that significantly boosts productivity. It’s about tailoring the tool to the task, rather than the other way around.
