---
title: "Why am I getting an `Azure ML ExecutableNotFound: failed to execute PosixPath('dot'), make sure the Graphviz executables are on your systems' PATH`?"
date: "2024-12-23"
id: "why-am-i-getting-an-azure-ml-executablenotfound-failed-to-execute-posixpathdot-make-sure-the-graphviz-executables-are-on-your-systems-path"
---

Okay, so you've encountered the `Azure ML ExecutableNotFound: failed to execute PosixPath('dot'), make sure the Graphviz executables are on your systems' PATH` error. I remember battling with that particular message myself, oh, probably three or four projects back. It's a classic case of a dependency hiccup, specifically concerning Graphviz, a vital tool for visualizing directed and undirected graphs. You’re seeing it during some Azure Machine Learning operation, likely one involving model interpretation or pipeline visualization. Let me break down what's going on and how we can tackle it.

Essentially, the Azure ML environment, or whatever process you’re triggering, is trying to use `dot`, which is the command-line executable within the Graphviz toolkit. It's needed to render those graph-based visualizations. The error plainly states that it can’t find `dot` on your system's path. The “system's path” in this context refers to the list of directories where your operating system looks for executable files. If `dot` isn't in one of these directories, or its location isn’t declared there, the system simply can’t find it, and hence, the `ExecutableNotFound` error.

Now, let’s talk about potential causes and solutions. More often than not, the issue stems from one of these scenarios:

1.  **Graphviz isn't installed:** The most obvious reason. You haven't installed the Graphviz software itself, so the `dot` command doesn’t exist on your system.

2.  **Graphviz is installed, but not in the path:** You *have* Graphviz, but it's tucked away in a directory that's not included in your system’s `PATH` environment variable.

3.  **Incorrect Path specification:** You might have added Graphviz to the `PATH`, but perhaps there was a typo in the directory path or a formatting issue during the configuration.

4.  **Environment Specific Issues:** If you are working in a containerized environment like Azure Machine Learning compute, the Graphviz installation and path specification might be isolated to the container and will differ from your local machine.

Let's address these one by one with practical steps and then look at some code snippets to solidify the solutions.

**Solution Steps:**

First, we need to confirm if Graphviz is indeed installed. If you are on Windows, you could typically look for it in "Program Files" or "Program Files(x86)". On MacOS you should have installed it by brew or other manager and on linux you could also check using package managers like `apt`.

If you find it, note the installation directory. For windows it often is under `C:\Program Files\Graphviz\bin\` or similar. On linux it is often under `/usr/bin` or `/usr/local/bin`.

Next, we'll verify if it's correctly included in your system's `PATH`. The method to modify the path differs based on your OS:

*   **Windows:** Search for "environment variables" in the Start menu, select "Edit the system environment variables." Click "Environment Variables..." Select "Path" in the system variables section, then "Edit...". Add the directory containing the `dot` executable. It is important to add the complete path. Click "OK" on all windows.
*   **Linux/macOS:** Open your terminal. You can inspect your path by executing the command `echo $PATH`. Modify your `.bashrc`, `.zshrc` (or the relevant shell configuration file) to add the Graphviz directory. For bash it would be `export PATH=$PATH:/path/to/graphviz/bin` where `/path/to/graphviz/bin` is the location of dot. Remember to `source ~/.bashrc` (or the correct file) after making the changes to make sure your current session is updated.

If you didn't have Graphviz installed, you can download it from the official website or install it via package manager (e.g., `apt install graphviz` or `brew install graphviz`). Make sure you install it correctly according to the provided instructions.

After modifying the path, close all your active shells/terminal and start a new one or restart your environment (like Jupyter Notebook or VSCode) for the changes to take effect.

Now let us look at some code snippets that illustrate some of the fixes, in the context of Azure ML.

**Code Snippets**

**Snippet 1: Checking if Graphviz is installed and available**

This snippet helps verify if `dot` is available on your path directly using python.

```python
import subprocess

def check_graphviz():
    try:
        result = subprocess.run(['dot', '-V'], capture_output=True, text=True, check=True)
        print(f"Graphviz version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("Error: 'dot' executable not found. Graphviz might not be installed or not in your PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error executing dot command: {e}")
        return False

if __name__ == "__main__":
    check_graphviz()
```

This script tries to execute `dot -V` which will print the version number if `dot` is on the path. If the `dot` command is not found, then a `FileNotFoundError` will be triggered and printed, confirming the issue. If it can be found and is available the version number will be printed to screen and will verify that Graphviz has been correctly installed.

**Snippet 2: Dynamically adding Graphviz to the PATH (within a session)**

This is a temporary fix to test the solution within your current notebook session, without the need to modify environment variables. This is not a persistent solution and should only be used for testing.

```python
import os
import subprocess

def add_graphviz_to_path(graphviz_install_path):
    """Adds the graphviz installation path to the current session's PATH."""
    if not os.path.exists(graphviz_install_path):
        raise ValueError(f"Graphviz installation path not found: {graphviz_install_path}")

    bin_path = os.path.join(graphviz_install_path, 'bin')
    if not os.path.exists(bin_path):
        raise ValueError(f"Graphviz 'bin' directory not found: {bin_path}")

    os.environ['PATH'] = f"{bin_path}{os.pathsep}{os.environ['PATH']}"

    try:
        result = subprocess.run(['dot', '-V'], capture_output=True, text=True, check=True)
        print(f"Graphviz is now available. Version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Error: Graphviz is not accessible, even after modification.")

if __name__ == '__main__':
    # Example: Replace with your actual graphviz installation directory
    graphviz_installation_directory = r"C:\Program Files\Graphviz"  # on windows
    # graphviz_installation_directory = "/usr/bin"  # on linux / MacOS
    try:
        add_graphviz_to_path(graphviz_installation_directory)
    except ValueError as e:
        print(f"Error: {e}")
```

This code will attempt to add the correct Graphviz directory to the environment variable `PATH` within the current session. If you are working in a notebook, this code will make the changes to the current python session only and thus this change will be lost after restarting the environment.

**Snippet 3: Installing graphviz from within the container (if needed)**

This code snippet is relevant if you are working within an Azure Machine Learning Compute Instance or Container, where you might need to manually install Graphviz.
```python
import subprocess

def install_graphviz_in_container():
    try:
        subprocess.run(["apt-get", "update", "-y"], check=True)
        subprocess.run(["apt-get", "install", "-y", "graphviz"], check=True)
        print("Graphviz has been installed in the container.")

        # Verify the installation
        result = subprocess.run(['dot', '-V'], capture_output=True, text=True, check=True)
        print(f"Graphviz is now available. Version: {result.stdout.strip()}")

    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install Graphviz in container: {e}")

if __name__ == '__main__':
    install_graphviz_in_container()
```

This will attempt to update the package list and install the `graphviz` package. This is specifically for containerized environments using `apt`, so adjust the command if you are working within a different environment.

**Resources**

For a more detailed understanding of environment variables, I recommend reading "Operating System Concepts" by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne. Their chapter on process management and environment variables is insightful. Also, the official Graphviz documentation ([https://graphviz.org/doc/](https://graphviz.org/doc/)) offers comprehensive information on installation and usage. Understanding the specifics of how operating systems handle executable paths is very helpful for this kind of troubleshooting.

In summary, the `Azure ML ExecutableNotFound` error is a common hiccup, primarily caused by Graphviz not being accessible. Following these systematic steps to install Graphviz and correctly add its directory to your system's path should clear up this issue. If it persists, checking within your execution environment (e.g., your docker image) should be the next step. Remember to always verify the installation by attempting to run the `dot -V` command to ensure that the system can see the executable.
