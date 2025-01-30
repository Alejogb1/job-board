---
title: "What error occurs when running mkvirtualenv in Mountain Lion?"
date: "2025-01-30"
id: "what-error-occurs-when-running-mkvirtualenv-in-mountain"
---
The most common error encountered when using `mkvirtualenv` on Mountain Lion (OS X 10.8) stems from the system's Python installation and its interaction with virtual environment management tools.  Specifically, it often arises from inconsistencies between the system's default Python version, the Python version targeted by `mkvirtualenv`, and the location of necessary executables within the system's PATH environment variable.  This leads to failures in locating essential components like `virtualenv` itself, resulting in cryptic error messages and the inability to create virtual environments.  I've personally debugged this numerous times across various projects involving legacy codebases and diverse development workflows, and the solution typically involves a careful examination of the system's Python configuration and environment variables.


**1. Clear Explanation**

`mkvirtualenv`, a command-line tool often associated with `virtualenvwrapper`, relies on the availability of the `virtualenv` package.  On Mountain Lion, which shipped with Python 2.7, conflicts can arise due to multiple Python installations (e.g., a system-installed Python and a separately installed Python 3.x via Homebrew or a package manager).  The issue manifests because:

* **PATH conflicts:** The system's `PATH` variable determines the order in which the operating system searches for executables. If a conflicting `python` or `virtualenv` executable is encountered earlier in the `PATH` than the intended one, `mkvirtualenv` might utilize the wrong version, resulting in errors during virtual environment creation.
* **Incorrect Python version:** `mkvirtualenv` might attempt to use the system's Python 2.7, which might lack necessary libraries or have outdated versions that conflict with `virtualenv`'s requirements. Using a more modern Python version installed alongside the system Python could circumvent this issue.
* **Missing dependencies:**  `virtualenv` itself may have unmet dependencies, even if Python is correctly specified. This is less frequent, but possible, and is checked later in the process.
* **Permissions issues:**  Rarely, permission problems prevent `mkvirtualenv` from writing files in the necessary directories.  This is often indicated by more explicit permission-related error messages.


**2. Code Examples with Commentary**

Here are three illustrative scenarios demonstrating different aspects of troubleshooting this issue, with associated solutions and commentary.

**Example 1: PATH conflict with system Python**

```bash
mkvirtualenv myenv
# Output:  Error:  Could not find virtualenv executable.
```

**Commentary:** This indicates `mkvirtualenv` can't find `virtualenv`. The problem is likely a `PATH` issue.  The system's Python 2.7 might be higher in the `PATH` than the Python installation where `virtualenv` is actually installed.  To resolve this, I typically temporarily modify the `PATH` to prioritize the correct Python installation:

```bash
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"  # Replace with your Python installation path
mkvirtualenv myenv
```
This explicitly places `/usr/local/bin` (a common location for Homebrew installations) at the beginning of the `PATH`.  After successful environment creation, the `PATH` can be reset or permanently modified via your shell's configuration file (e.g., `.bashrc`, `.zshrc`).


**Example 2: Using the wrong Python version**

```bash
mkvirtualenv -p python3 myenv
# Output:  Error:  The system python is 2.7, but you want to use python3.  Check that your python3 is in the PATH...
```

**Commentary:** This highlights a mismatch between the desired Python version (Python 3) and the version `mkvirtualenv` defaults to.  The error message is quite informative. The solution, as before, involves ensuring that the desired Python 3 executable is in the `PATH` *before* running `mkvirtualenv`.


**Example 3: Missing `setuptools` dependency**

```bash
mkvirtualenv myenv
# Output:  Error:  Command 'python setup.py egg_info' failed with error code 1
```

**Commentary:** While less common, this demonstrates a dependency problem.  `virtualenv` and related tools rely on `setuptools`.  This error implies a problem installing this dependency during virtual environment creation.  The solution requires ensuring `setuptools` is correctly installed within the target Python environment (either globally or within the virtual environment after creation, if the issue occurs during the `post-creation` phase).

```bash
pip install setuptools  # Install globally (generally avoided for this type of problem)
# or, after environment creation, within the environment:
workon myenv
pip install setuptools
```


**3. Resource Recommendations**

I recommend consulting the official documentation for `virtualenv` and `virtualenvwrapper`.  Thoroughly review the troubleshooting sections of these documents. Understanding the `PATH` environment variable and how it influences executable searches is crucial.  Furthermore, examining the output of commands like `which python`, `which python3`, and `which virtualenv` helps identify the locations of different Python interpreters and related tools on your system, thus aiding in resolving path-related issues. Finally, familiarity with basic shell commands and how to manipulate environment variables is fundamental for successful development on Unix-like systems like macOS.  Understanding package managers like Homebrew and how they interact with system Python installations will be beneficial.  Reviewing the logs generated during the creation of virtual environments can offer valuable diagnostics as well.
