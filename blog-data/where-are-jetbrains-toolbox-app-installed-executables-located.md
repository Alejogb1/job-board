---
title: "Where are JetBrains Toolbox App installed executables located?"
date: "2024-12-23"
id: "where-are-jetbrains-toolbox-app-installed-executables-located"
---

, so executable locations with the JetBrains Toolbox App—that’s a question I’ve encountered more than once, especially when setting up custom IDE configurations or needing to interact directly with the underlying tools. I recall vividly a situation a few years back, during a project overhaul, where the CI/CD pipeline was misbehaving, and it eventually traced back to inconsistencies in the configured path variables for a particular IntelliJ plugin build. Resolving that required a deep dive into how Toolbox manages installations, and that experience solidified my understanding.

The core thing to grasp here is that the Toolbox app doesn't install executables into a single, static location like some other software installers do. Instead, it maintains them within a specific directory structure that is managed by the application itself. This design has several advantages, including allowing you to install multiple versions of the same IDE side by side and simplifies the management of updates.

Fundamentally, the location of the installed IDE executables depends primarily on your operating system. On macOS, they're typically situated within a directory akin to `~/Library/Application Support/JetBrains/Toolbox/apps/`. Within that location, you'll find subdirectories corresponding to each installed IDE, for example, `idea` for IntelliJ IDEA or `pycharm` for PyCharm. Inside these individual IDE folders, you'll find the actual executable files, often nestled inside a subfolder like `Contents/MacOS` on macOS.

On Windows, the pattern is very similar, but the root path will be different; usually, you will find them somewhere like `%localappdata%\JetBrains\Toolbox\apps`. Again, each IDE will have its folder, for example, `idea` or `pycharm`, and the executables are often within a `bin` subfolder. For Linux distributions, the base directory will usually be under `~/.local/share/JetBrains/Toolbox/apps`, but this can sometimes differ based on how you installed the Toolbox, perhaps using a snap or flatpak.

The exact name of the executable will vary slightly based on the IDE. For IntelliJ IDEA, on macOS, the main executable is typically named `idea`. On windows it's `idea64.exe` and on linux often it is `idea.sh` or `idea.desktop` depending on the distribution. Similarly, PyCharm’s executables would be `pycharm` on macOS or `pycharm64.exe` on Windows. For other JetBrains products such as CLion, WebStorm, etc, the pattern is generally consistent.

The Toolbox App also facilitates a command-line interface named `toolbox`. Although it can be used for launching installed applications and viewing installed versions, it doesn't directly expose the location of each executable with a dedicated command. I've found that the most effective approach to locate these is navigating directly to the mentioned file paths based on the OS.

Now, to make this practical, here are a few code snippets illustrating how you can determine these paths, with each tailored for a different shell or scripting language.

**Example 1: Bash Script for macOS/Linux**

```bash
#!/bin/bash

# Define the base path based on macOS convention (adjust for Linux if needed)
base_path="$HOME/Library/Application Support/JetBrains/Toolbox/apps/"

# Check if macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    base_path="$HOME/Library/Application Support/JetBrains/Toolbox/apps/"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    base_path="$HOME/.local/share/JetBrains/Toolbox/apps/"
fi


# Define the IDE name
ide_name="idea"  # change this to "pycharm", "clion", etc. as needed

# Construct the path to the executable (adapt the specific path based on os)
if [[ "$OSTYPE" == "darwin"* ]]; then
  executable_path="$base_path/$ide_name/*/Contents/MacOS/$ide_name"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  executable_path="$base_path/$ide_name/*/bin/$ide_name.sh"
fi

# Output the path or "not found"
if [[ -x "$executable_path" ]]; then
  echo "Executable path for $ide_name: $executable_path"
else
  echo "Executable for $ide_name not found"
fi
```

This script first defines the common base path based on the operating system. Then it constructs the IDE executable path based on an example IDE (in this case, IntelliJ IDEA) and checks if the executable exists.

**Example 2: PowerShell Script for Windows**

```powershell
# Define the base path
$basePath = "$env:LOCALAPPDATA\JetBrains\Toolbox\apps\"

# Define the IDE name
$ideName = "idea" #change to "pycharm", "clion" etc as needed

# construct the full executable path
$executablePath = Join-Path $basePath  $ideName  "*/bin/$ideName64.exe"


# Check if the executable exists
if (Test-Path $executablePath -PathType Leaf) {
    Write-Host "Executable path for $ideName: $executablePath"
} else {
    Write-Host "Executable for $ideName not found"
}
```

This script is similar to the bash one but adapted for PowerShell syntax and Windows paths. It uses environment variables and `Join-Path` for reliable path construction.

**Example 3: Python script**

```python
import os
import platform
import glob

def get_toolbox_ide_executable_path(ide_name):
    os_type = platform.system()

    if os_type == 'Darwin':  # macOS
        base_path = os.path.expanduser('~/Library/Application Support/JetBrains/Toolbox/apps/')
        executable_pattern = f'{base_path}/{ide_name}/*/Contents/MacOS/{ide_name}'
    elif os_type == 'Windows':
        base_path = os.path.join(os.getenv('LOCALAPPDATA'), 'JetBrains', 'Toolbox', 'apps')
        executable_pattern = os.path.join(base_path, f'{ide_name}', '*/bin', f'{ide_name}64.exe')
    elif os_type == 'Linux':
      base_path = os.path.expanduser('~/.local/share/JetBrains/Toolbox/apps/')
      executable_pattern = f'{base_path}/{ide_name}/*/bin/{ide_name}.sh'
    else:
        return None

    found_paths = glob.glob(executable_pattern)
    if found_paths:
        return found_paths[0]  # Return the first match if multiple version are present
    else:
        return None

# example usage:
ide = "idea"
executable_path = get_toolbox_ide_executable_path(ide)
if executable_path:
    print(f"Executable path for {ide}: {executable_path}")
else:
    print(f"Executable for {ide} not found.")

```

This Python script uses `os` and `platform` modules to dynamically get the OS type and construct the appropriate paths, using `glob` to locate the relevant files.

Regarding additional resources, I recommend delving into *Operating System Concepts* by Abraham Silberschatz and co-authors to deepen your understanding of file systems. If you're frequently using a particular OS, digging into its documentation directly (for example, Apple's developer documentation or Microsoft’s Windows documentation) can provide granular details on application installation conventions. For a comprehensive overview of command-line interactions, “The Linux Command Line” by William Shotts is invaluable and although focussed on Linux, the concepts generalise well to other systems. Finally, official documentation from JetBrains about their Toolbox App is also extremely helpful. This combination of resources has helped me resolve the trickiest pathing and configuration issues over the years, and I expect it will help others too.
