---
title: "Why does the `mkdir` command fail with '!' in Python?"
date: "2025-01-30"
id: "why-does-the-mkdir-command-fail-with-"
---
The failure of the `mkdir` command within Python when encountering a "!" character in the path stems from a fundamental difference in how shell commands and Python's operating system interfaces handle special characters, specifically within the context of file system interaction. Having spent a considerable portion of my career automating system tasks, I've repeatedly observed this specific pitfall. The crux of the issue is that the exclamation mark, `!`, possesses special meaning to the shell, often denoting history expansion or other forms of command substitution, whereas Python's native file system functions treat it as a literal character within a filename or directory name.

When a Python script attempts to execute `mkdir !test` (or any path containing `!`) via functions like `os.system` or subprocess.run, the underlying shell interprets the command before `mkdir` ever sees it. The shell's attempt at command substitution, finding no preceding command, often results in an error from the shell itself, or might interpret the `!` in an unexpected way that does not lead to the intended result. This erroneous interpretation is then reflected in Python, resulting in either a non-zero exit status, a file created in an undesired location or a permission related error.  Python's `os.mkdir` and other similar functions, conversely, work directly with the operating system’s API, bypassing the shell entirely, and hence will treat `!` as a valid part of the directory name. The disconnect lies in the layers of abstraction between the application layer (Python) and the operating system.

Now, let's examine why using `os.system` or the like is problematic and then delve into how to resolve this.

**First Problematic Example:**

```python
import os

directory_name = "!test_dir"

try:
  os.system(f"mkdir {directory_name}") # Attempt to create a directory with '!'
except Exception as e:
    print(f"Error during mkdir: {e}")
```

In this initial code snippet, we're using `os.system`. When the `os.system` command is called with `mkdir !test_dir`, the shell, not Python, interprets `mkdir !test_dir`. Typically in shells like bash the `!` character signals a history expansion. When there is nothing to expand or the syntax for expansion is incorrect the shell will fail. If history expansion were enabled, something unpredictable could happen, possibly creating a directory with a name that was part of an earlier command. The outcome, therefore, is unreliable and not what was intended, leading to the raised exception. This behavior depends heavily on the user's shell settings and history. In most cases this will fail and result in an error, depending on the shell.

**Second Problematic Example:**

```python
import subprocess

directory_name = "!another_dir"

try:
  subprocess.run(["mkdir", directory_name], check=True)
except subprocess.CalledProcessError as e:
  print(f"Error during subprocess run: {e}")
```

This example uses `subprocess.run`. While `subprocess` offers improvements over `os.system`, in this configuration, we are still vulnerable. By passing the command and arguments as a list, we attempt to pass `mkdir` and the string `!another_dir` to the subprocess, which uses the shell.  The shell will, as before, try to interpret the exclamation mark, resulting in a `CalledProcessError`. Despite avoiding the issues with string construction, the shell's interpretation causes a similar error as the previous example. Crucially, even when specifying a list,  `subprocess.run` still defaults to executing commands through the shell unless `shell=False` is explicitly set.

**Third Correct Example:**

```python
import os

directory_name = "!safe_dir"

try:
    os.mkdir(directory_name)
    print(f"Directory created: {directory_name}")
except OSError as e:
    print(f"Error using os.mkdir: {e}")


```

Here, we directly employ Python's `os.mkdir` function. This function directly communicates with the operating system kernel, bypassing the shell. Consequently, the `!` is treated as a valid character within a directory name. This approach is the preferred method when dealing with file system operations, providing reliability and security against shell-injection vulnerabilities. It demonstrates how Python's direct OS interaction mitigates the issues encountered when involving the shell. This version will correctly create the directory "!safe_dir".

To summarise, the key takeaway here is that when interacting with the shell, particularly when using user-supplied input or special characters in file paths, you should avoid `os.system` or `subprocess.run` without careful attention to the `shell` option (which is generally not recommended for these purposes) and instead use Python’s built in operating system functions.

For those seeking deeper understanding, I recommend exploring the following resources:

1.  **Python's official documentation:** The documentation for the `os` module and `subprocess` module provides complete specifications for all methods used here. The documentation will also provide clarity on the use of the `shell` option in `subprocess.run` and its security implications. Pay close attention to the methods that directly interact with the OS, like `os.mkdir`, `os.makedirs`, `os.rename` etc.
2.  **Operating System Concepts:** Familiarizing yourself with fundamental operating system concepts, specifically system calls, and how applications interact with the kernel will give context to these challenges. Understanding how the file system is structured and how permissions work helps to appreciate the intricacies of this kind of programming.
3.  **Shell Programming Texts:** While not directly related to Python, study of how shell's work, particularly command substitution, quoting and the purpose of special characters will help in avoiding these types of bugs in systems that mix shell and programming language execution. Focus on `bash` if your environment is Linux or macOS, and `cmd.exe` or PowerShell if your working in Windows.

By utilizing the appropriate Python methods that interact directly with the operating system, we avoid shell-related interpretation, allowing us to handle special characters in filenames and directory names reliably. Relying on native Python methods, like `os.mkdir` shown above, ensures that operations are predictable and safe.
